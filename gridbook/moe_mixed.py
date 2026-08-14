"""Serving method for v1 per-expert split-format MoE stacks.

The legacy uniform method remains in :mod:`gridbook.moe` and is never wrapped
when the declaration is absent.  This method owns only positively-declared
mixed layers: family-specific CB substacks, family-specific expert index maps,
and family-scoped MXFP4 delegated subgroups.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from types import SimpleNamespace
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from . import codec
from .cb_fill_guard import mark_filled, mark_unfilled
from .moe_gemv_select import (cb_fp8_gemv_v2_requested,
                              cb_gemv_choice)
from .per_expert_format import (
    ExpertFormatGroup,
    LayerFormatGroups,
    MXFP4_SOURCE,
)
from .source_passthrough import format_for as source_format_for


def _row_bytes(in_features: int, type_size: int) -> int:
    return (in_features // codec.SUPERBLOCK) * type_size


def _parameter_name(family: str, group: ExpertFormatGroup, suffix: str) -> str:
    discriminator = group.tensor_prefix.rsplit(".", 1)[-1]
    if not discriminator.startswith("format_group_"):
        raise ValueError(
            f"{group.tensor_prefix!r}: CB split-stack tensor prefix has no "
            "format_group_ discriminator"
        )
    return f"{family}_{discriminator}_{suffix}"


@dataclass
class _CBGroupRuntime:
    declaration: ExpertFormatGroup
    group_index: int
    method: Any
    lane: Any


@dataclass
class _SourceGroupRuntime:
    declaration: ExpertFormatGroup
    group_index: int
    method: Any
    layer_name: str


class _DelegatedLayer(torch.nn.Module):
    """The smallest RoutedExperts-shaped owner a native MoE method needs."""

    def __init__(self, parent: torch.nn.Module, moe_config: FusedMoEConfig):
        super().__init__()
        self.moe_config = moe_config
        self.activation = parent.activation
        self.apply_router_weight_on_input = False
        self.global_num_experts = moe_config.num_experts
        self.expert_map = None
        self.swiglu_limit = getattr(parent, "swiglu_limit", None)

    def _expert_routing_tables(self):
        return None


class PrismaQuantMixedMoEMethod(FusedMoEMethodBase):
    """One vLLM quant method coordinating all formats inside one MoE layer."""

    def __init__(self, quant_config, moe: FusedMoEConfig,
                 groups: LayerFormatGroups, prefix: str) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.groups = groups
        self.prefix = prefix
        self._cb_groups: dict[str, list[_CBGroupRuntime]] = {
            "w13": [], "w2": []
        }
        self._source_groups: dict[str, list[_SourceGroupRuntime]] = {
            "w13": [], "w2": []
        }

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        return None

    def _require_fp8_v2_dispatch_supported(self) -> None:
        """Refuse an explicit routed-FP8-v2 arm on mixed expert groups.

        The uniform method owns the qualified whole-row dispatch.  Mixed
        groups have separate family-local maps and currently call the
        inherited FP8 op directly; silently accepting the global opt-in here
        would produce a mixed/inherited benchmark mislabeled as FP8-v2.
        Parsing the selector even for FP4-only declarations also keeps invalid
        spellings process-global instead of depending on which layer loads
        first.
        """
        if not cb_fp8_gemv_v2_requested():
            return
        fp8_groups = [
            f"{family}:{group.format_wire_id}"
            for family in ("w13", "w2")
            for group in self.groups.groups(family)
            if not group.is_passthrough
            and self._scheme_for(group).get("grid") == "fp8"
        ]
        if fp8_groups:
            raise RuntimeError(
                f"{self.prefix}: PRISMAQUANT_CB_FP8_GEMV_V2=1 is not "
                "implemented for per-expert mixed FP8 groups "
                f"{fp8_groups}; refusing a silently inherited candidate arm"
            )

    def _scheme_for(self, group: ExpertFormatGroup) -> dict:
        target = self.quant_config._per_expert_serving_prefixes[
            group.tensor_prefix
        ]
        try:
            return self.quant_config.target_scheme[target]
        except KeyError:
            raise ValueError(
                f"{self.prefix}: CB group {target!r} has no resolved scheme"
            ) from None

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if num_experts != self.groups.num_experts:
            raise ValueError(
                f"{self.prefix}: create_weights num_experts={num_experts} != "
                f"declared {self.groups.num_experts}"
            )
        layer._cb_hidden = hidden_size
        layer._cb_inter = intermediate_size_per_partition
        layer._cb_E = num_experts
        attrs = dict(extra_weight_attrs)

        # Device-resident maps are useful to future fused dispatch too; the
        # Python tuple on the declaration remains the load-time oracle.
        for family in ("w13", "w2"):
            mapping = self.groups.index_maps[family]
            group_ids = torch.tensor([item[0] for item in mapping], dtype=torch.int32)
            positions = torch.tensor([item[1] for item in mapping], dtype=torch.int32)
            if hasattr(layer, "register_buffer"):
                layer.register_buffer(f"_{family}_format_group", group_ids,
                                      persistent=False)
                layer.register_buffer(f"_{family}_format_position", positions,
                                      persistent=False)
            else:  # deliberately tiny CPU fixtures
                setattr(layer, f"_{family}_format_group", group_ids)
                setattr(layer, f"_{family}_format_position", positions)

        static_families: set[str] = set()
        for family in ("w13", "w2"):
            in_features = (hidden_size if family == "w13"
                           else intermediate_size_per_partition)
            out_features = (2 * intermediate_size_per_partition
                            if family == "w13" else hidden_size)
            for group_index, group in enumerate(self.groups.groups(family)):
                if group.is_passthrough:
                    continue
                scheme = self._scheme_for(group)
                type_size = int(scheme["type_size"])
                count = len(group.expert_ids)
                qname = _parameter_name(family, group, "cb_qweight")
                qweight = torch.nn.Parameter(torch.empty(
                    count, out_features, _row_bytes(in_features, type_size),
                    dtype=torch.uint8), requires_grad=False)
                set_weight_attrs(qweight, {**attrs, "is_transposed": False})
                mark_unfilled(qweight)
                layer.register_parameter(qname, qweight)
                if scheme["grid"] == "fp8":
                    sname = _parameter_name(family, group, "weight_scale")
                    scale = torch.nn.Parameter(torch.empty(
                        count, out_features, dtype=torch.float32),
                        requires_grad=False)
                    set_weight_attrs(scale, dict(attrs))
                    layer.register_parameter(sname, scale)
                if scheme.get("activation_contract") is not None:
                    static_families.add(family)

        # One scalar parameter per stage, shared by every FP4 subgroup in that
        # stage.  The loader below compares every physical copy before reusing
        # it, making the K0.2 "one scale, many groups" claim executable.
        for family in sorted(static_families):
            scale = torch.nn.Parameter(
                torch.full((1,), float("nan"), dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(scale, dict(attrs))
            layer.register_parameter(f"{family}_input_global_scale", scale)

        # The producer partitions w13 and w2 independently.  Build one native
        # owner per declared source family so a CB w13 may feed a native w2
        # (and vice versa).  The native method still creates both parameter
        # families because that is its public contract; only the declared
        # family is loadable and launched.
        fmt = source_format_for(MXFP4_SOURCE, unit=self.prefix)
        delegates: dict[tuple[int, ...], tuple[Any, str, _DelegatedLayer]] = {}
        for family in ("w13", "w2"):
            for group_index, group in enumerate(self.groups.groups(family)):
                if not group.is_passthrough:
                    continue
                cached = delegates.get(group.expert_ids)
                if cached is None:
                    subgroup_size = len(group.expert_ids)
                    subconfig = replace(
                        self.moe,
                        num_experts=subgroup_size,
                        num_local_experts=subgroup_size,
                        num_logical_experts=subgroup_size,
                    )
                    delegated_layer = _DelegatedLayer(layer, subconfig)
                    method = self.quant_config._delegate_passthrough(
                        delegated_layer, f"{self.prefix}/{family}", fmt
                    )
                    method.create_weights(
                        delegated_layer, subgroup_size, hidden_size,
                        intermediate_size_per_partition, params_dtype,
                        **extra_weight_attrs,
                    )
                    layer_name = (
                        f"_gridbook_mxfp4_{family}_subgroup_"
                        f"{len(delegates)}"
                    )
                    layer.add_module(layer_name, delegated_layer)
                    delegates[group.expert_ids] = (
                        method, layer_name, delegated_layer
                    )
                else:
                    method, layer_name, delegated_layer = cached
                for suffix in ("weight", "weight_scale"):
                    param = getattr(delegated_layer, f"{family}_{suffix}")
                    setattr(param, "_gridbook_source_expert_ids",
                            group.expert_ids)
                self._source_groups[family].append(_SourceGroupRuntime(
                    group, group_index, method, layer_name
                ))

        self._install_loader(layer)

    @staticmethod
    def _name_matches(name: str, tensor_prefix: str, suffix: str) -> bool:
        full = tensor_prefix + suffix
        if name == full or name.endswith("." + full):
            return True
        # FusedMoE.load_weights often strips the module prefix before invoking
        # the quant method's instance hook.
        leaf = "gate_up_proj" if ".gate_up_proj" in tensor_prefix else "down_proj"
        relative = tensor_prefix[tensor_prefix.rfind(leaf):] + suffix
        return name == relative or name.endswith("." + relative)

    def _install_loader(self, layer: torch.nn.Module) -> None:
        original = getattr(layer, "load_weights", None)
        if original is None or getattr(layer, "_cb_mixed_load_wrapped", False):
            return
        prefix = self.prefix

        def load_weights(weights):
            deferred = []
            for name, weight in weights:
                handled = False
                for family in ("w13", "w2"):
                    for group_index, group in enumerate(self.groups.groups(family)):
                        if group.is_passthrough:
                            continue
                        for suffix, param_suffix in (
                            (".cb_qweight", "cb_qweight"),
                            (".weight_scale", "weight_scale"),
                        ):
                            if not self._name_matches(
                                name, group.tensor_prefix, suffix
                            ):
                                continue
                            pname = _parameter_name(family, group, param_suffix)
                            param = getattr(layer, pname)
                            if tuple(param.shape) != tuple(weight.shape):
                                raise ValueError(
                                    f"{prefix}.{name}: checkpoint shape "
                                    f"{tuple(weight.shape)} != parameter "
                                    f"{tuple(param.shape)}"
                                )
                            param.data.copy_(weight.to(param.dtype))
                            if param_suffix == "cb_qweight":
                                mark_filled(param)
                            handled = True
                            yield pname
                            break
                        if handled:
                            break
                        if (hasattr(layer, f"{family}_input_global_scale")
                                and self._name_matches(
                                    name, group.tensor_prefix,
                                    ".input_global_scale")):
                            param = getattr(layer, f"{family}_input_global_scale")
                            incoming = weight.to(param.dtype).reshape_as(param)
                            if torch.isnan(param.data).all():
                                param.data.copy_(incoming)
                            elif not torch.equal(param.data, incoming):
                                raise ValueError(
                                    f"{prefix}/{family}: per-layer "
                                    "input_global_scale differs across format "
                                    "subgroups"
                                )
                            handled = True
                            yield f"{family}_input_global_scale"
                            break
                    if handled:
                        break
                if not handled and self._load_delegated_slice(layer, name, weight):
                    handled = True
                    yield name
                if not handled:
                    deferred.append((name, weight))
            if deferred:
                yield from original(deferred)

        layer.load_weights = load_weights
        layer._cb_mixed_load_wrapped = True

    def _load_delegated_slice(self, layer, name: str, weight: torch.Tensor) -> bool:
        match = re.search(
            r"(?:^|[.]experts[.])(\d+)[.]"
            r"(w1|w3|w2|gate_proj|up_proj|down_proj)"
            r"[.](weight|scale)$", name
        )
        if match is None:
            return False
        expert_id = int(match.group(1))
        leaf, plane = match.group(2), match.group(3)
        if leaf in ("w1", "gate_proj", "w3", "up_proj"):
            family = "w13"
            shard = 0 if leaf in ("w1", "gate_proj") else 1
        else:
            family = "w2"
            shard = None
        runtime = next((
            item for item in self._source_groups[family]
            if expert_id in item.declaration.expert_ids
        ), None)
        if runtime is None:
            return False
        local = runtime.declaration.expert_ids.index(expert_id)
        delegated = getattr(layer, runtime.layer_name)
        target = getattr(delegated, f"{family}_{plane}")
        if family == "w13":
            rows = target.shape[1] // 2
            destination = target.data[local, shard * rows:(shard + 1) * rows]
        else:
            destination = target.data[local]
        if tuple(destination.shape) != tuple(weight.shape):
            raise ValueError(
                f"{self.prefix}.{name}: checkpoint shape {tuple(weight.shape)} "
                f"!= delegated slice {tuple(destination.shape)}"
            )
        incoming = (
            weight.contiguous().view(destination.dtype)
            if weight.element_size() == destination.element_size()
            else weight.to(destination.dtype)
        )
        destination.copy_(incoming)
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from .moe import PrismaQuantCBMoEMethod
        from .native_cutlass import require_native_moe_activation
        from .cuda_ext import (
            require_bf16_grouped_ext,
            require_ext,
            require_fp4_v2_expander,
        )
        from .native_cutlass import require_native_fp8_cutlass

        self._require_fp8_v2_dispatch_supported()
        codebooks = self.quant_config.get_codebooks()
        require_ext(f"{self.prefix} mixed routed CB decode/QDQ/expansion")
        require_bf16_grouped_ext(
            f"{self.prefix} mixed routed quality prefill"
        )
        saw_fp4 = False
        saw_fp8 = False
        for family in ("w13", "w2"):
            runtimes = self._cb_groups[family]
            for group_index, group in enumerate(self.groups.groups(family)):
                if group.is_passthrough:
                    continue
                scheme = self._scheme_for(group)
                method = PrismaQuantCBMoEMethod(
                    self.quant_config, self.moe, scheme,
                    f"{self.prefix}/{family}/{group.format_wire_id}",
                )
                saw_fp4 |= method.is_fp4
                saw_fp8 |= not method.is_fp4
                lane = SimpleNamespace(
                    activation=layer.activation,
                    _cb_hidden=layer._cb_hidden,
                    _cb_inter=layer._cb_inter,
                    _cb_E=len(group.expert_ids),
                )
                qparam = getattr(
                    layer, _parameter_name(family, group, "cb_qweight")
                )
                setattr(lane, f"{family}_cb_qweight", qparam)
                # Fill guard accepts a family-only lane when given the exact
                # parameter directly; report its declaration on failure.
                if not getattr(qparam, "_pq_cb_filled", False):
                    raise ValueError(
                        f"{self.prefix}/{family}/{group.format_wire_id}: "
                        "declared CB sub-stack was not loaded"
                    )
                if scheme["grid"] == "fp8":
                    setattr(lane, f"{family}_weight_scale", getattr(
                        layer, _parameter_name(
                            family, group, "weight_scale"
                        )
                    ))
                if "codebook_ref_by_role" in scheme:
                    # Per-expert format groups and per-role books are separate
                    # features; composing them needs the role split applied
                    # inside each group's lane. `config._moe_scheme_for_prefix`
                    # refuses the composition before a method is ever built, so
                    # this is the second line of defence, not the first.
                    raise ValueError(
                        f"{method.prefix}: per-expert format group {group!r} "
                        "declares per-role codebooks; that combination is not "
                        "implemented"
                    )
                refs = scheme["codebook_ref"]
                names = refs if isinstance(refs, list) else [refs]
                subs = [codebooks[name].to(qparam.device) for name in names]
                lane._cb_flat = codec.build_flat_codebook(
                    subs, method.prefix, scheme["grid"]
                )
                lane._cb_compose = (
                    codec.build_compose_table(method._sub_table).to(qparam.device)
                    if method.is_v2 else
                    torch.zeros(1, dtype=torch.float32, device=qparam.device)
                )
                if not method.is_fp4:
                    lane._cb_flat_fp8 = codec.flat_codebook_fp8(
                        lane._cb_flat, method.prefix
                    )
                in_features = (layer._cb_hidden if family == "w13"
                               else layer._cb_inter)
                use_v2 = False
                if method.is_fp4 and method.is_v2:
                    use_v2, _why = cb_gemv_choice(
                        method.k, method.n_sub, method.type_size,
                        in_features, qparam.device,
                    )
                setattr(lane, f"_cb_use_v2_{family}", use_v2)
                runtimes.append(_CBGroupRuntime(
                    group, group_index, method, lane
                ))

            static_groups = [
                runtime for runtime in runtimes
                if runtime.method.has_static_fp4_activation
            ]
            if static_groups:
                param = getattr(layer, f"{family}_input_global_scale", None)
                if param is None or bool(torch.isnan(param.data).any()):
                    raise ValueError(
                        f"{self.prefix}/{family}: contracted FP4 substacks "
                        "have no loaded per-layer input_global_scale"
                    )
                expected = []
                for runtime in static_groups:
                    expected.extend(
                        self.quant_config.activation_scales_for_targets([
                            self.quant_config._per_expert_serving_prefixes[
                                runtime.declaration.tensor_prefix
                            ]
                        ])
                    )
                if not expected or any(value != expected[0] for value in expected):
                    raise ValueError(
                        f"{self.prefix}/{family}: input_global_scale is not "
                        "one per-layer value shared by all format subgroups"
                    )
                if float(param.data.float().cpu().item()) != float(expected[0]):
                    raise ValueError(
                        f"{self.prefix}/{family}: loaded input_global_scale "
                        "does not match the attested per-layer value"
                    )

        if saw_fp4:
            require_fp4_v2_expander(
                f"{self.prefix} mixed routed FP4-v2 expansion",
                device=next(layer.parameters()).device,
            )
        if saw_fp8:
            require_native_fp8_cutlass(
                f"{self.prefix} mixed routed FP8 quality prefill"
            )

        processed_source_layers: set[str] = set()
        for family in ("w13", "w2"):
            for runtime in self._source_groups[family]:
                delegated = getattr(layer, runtime.layer_name)
                if runtime.layer_name not in processed_source_layers:
                    runtime.method.process_weights_after_loading(delegated)
                    processed_source_layers.add(runtime.layer_name)
                experts = getattr(runtime.method.moe_kernel, "fused_experts", None)
                if experts is None or experts.__class__.__name__ != "MarlinExperts":
                    got = type(experts).__name__ if experts is not None else None
                    raise RuntimeError(
                        f"{self.prefix}/{family}: scoped {MXFP4_SOURCE} "
                        f"requires the attested MarlinExperts route, got {got}"
                    )
        layer._cb_native_activation = require_native_moe_activation(
            layer.activation.value, f"{self.prefix} mixed routed activation"
        )
        from .ops import register_cb_layer
        layer._cb_layer_id = register_cb_layer(self, layer)

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts, shared_experts_input):
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input unsupported for mixed Gridbook MoE"
            )
        from .ops import cb_moe_forward
        return cb_moe_forward(
            x, topk_weights, topk_ids, layer._cb_layer_id
        )

    def _apply_inline(self, layer, x, topk_weights, topk_ids):
        T, topk = topk_ids.shape
        flat_global = topk_ids.reshape(-1).long()
        pair_tokens = torch.arange(
            T, device=x.device, dtype=torch.long
        ).repeat_interleave(topk)
        active_pair = torch.arange(flat_global.numel(), device=x.device)

        pair_global = flat_global.index_select(0, active_pair)
        pair_token = pair_tokens.index_select(0, active_pair)
        gate_up = x.new_empty((active_pair.numel(), 2 * layer._cb_inter))
        self._run_family(
            layer, "w13", x, pair_global, pair_token, gate_up,
            prefill=T > 16,
        )
        activated = x.new_empty((active_pair.numel(), layer._cb_inter))
        from .native_cutlass import native_moe_activation
        native_moe_activation(
            layer._cb_native_activation, activated, gate_up
        )
        pair_output = x.new_empty((active_pair.numel(), layer._cb_hidden))
        pair_rows = torch.arange(
            active_pair.numel(), device=x.device, dtype=torch.long
        )
        self._run_family(
            layer, "w2", activated, pair_global, pair_rows, pair_output,
            prefill=T > 16,
        )
        pair_weight = topk_weights.reshape(-1).index_select(
            0, active_pair
        ).to(pair_output.dtype)
        pair_output.mul_(pair_weight[:, None])
        output = x.new_zeros((T, layer._cb_hidden))
        output.index_add_(0, pair_token, pair_output.to(output.dtype))
        return output

    def _run_family(self, layer, family, inputs, global_ids, input_rows,
                    output, *, prefill: bool):
        mapping_group = getattr(layer, f"_{family}_format_group").long()
        mapping_position = getattr(layer, f"_{family}_format_position").long()
        pair_group = mapping_group.index_select(0, global_ids)
        pair_local = mapping_position.index_select(0, global_ids)
        for runtime in self._cb_groups[family]:
            selected = (pair_group == runtime.group_index).nonzero(
                as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            local_ids = pair_local.index_select(0, selected)
            rows = input_rows.index_select(0, selected)
            if prefill:
                result = self._run_cb_prefill(
                    runtime, family, inputs.index_select(0, rows), local_ids
                )
            else:
                result = self._run_cb_decode(
                    runtime, family, inputs, rows, local_ids
                )
            output.index_copy_(0, selected, result)
        for runtime in self._source_groups[family]:
            selected = (pair_group == runtime.group_index).nonzero(
                as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            local_ids = pair_local.index_select(0, selected)
            rows = input_rows.index_select(0, selected)
            result = self._run_source_stage(
                layer, runtime, family,
                inputs.index_select(0, rows).contiguous(), local_ids,
            )
            output.index_copy_(0, selected, result)

    @staticmethod
    def _run_source_stage(layer, runtime, family, inputs, local_ids):
        """Launch one family of the attested native Marlin MXFP4 method.

        vLLM exposes MXFP4 as a two-stage method, while the producer's wire
        contract partitions the two physical families independently.  The
        delegated instance still owns conversion, quant metadata, backend
        selection, and attestation; this adapter calls the same Marlin GEMM
        primitive used by that instance for only the declared family.
        """
        from vllm.model_executor.layers.fused_moe.experts import (
            marlin_moe as native_marlin,
        )

        delegated = getattr(layer, runtime.layer_name)
        experts = runtime.method.moe_kernel.fused_experts
        count = len(runtime.declaration.expert_ids)
        rows = inputs.shape[0]
        topk_ids = local_ids.reshape(-1, 1)
        topk_weights = torch.ones(
            (rows, 1), dtype=torch.float32, device=inputs.device
        )
        block_size = 64
        for candidate in (8, 16, 32, 48, 64):
            if rows / count / candidate < 0.9:
                block_size = candidate
                break
        input_dtype = experts.input_dtype
        if input_dtype is not None and input_dtype.itemsize == 1:
            block_size = max(block_size, 16)
        sorted_tokens, expert_ids, padded = native_marlin.moe_align_block_size(
            topk_ids, block_size, count, None, ignore_invalid_experts=True
        )

        activation_scale = None
        gemm_input = inputs
        if input_dtype == torch.int8:
            gemm_input, activation_scale = native_marlin.marlin_quant_input(
                inputs, input_dtype
            )
            global_input_scale = (
                experts.a1_gscale if family == "w13" else experts.a2_gscale
            )
            if global_input_scale is not None:
                activation_scale = activation_scale * global_input_scale
        elif input_dtype == torch.float8_e4m3fn:
            gemm_input, activation_scale = native_marlin.marlin_quant_input(
                inputs, input_dtype
            )

        if family == "w13":
            weight = delegated.w13_weight
            bias = experts.w1_bias
            scale = experts.w1_scale
            global_scale = experts.g1_alphas
            zeros = experts.w1_zp
            g_idx = experts.w13_g_idx
            sort_indices = experts.w13_g_idx_sort_indices
            output_features = 2 * layer._cb_inter
        else:
            weight = delegated.w2_weight
            bias = experts.w2_bias
            scale = experts.w2_scale
            global_scale = experts.g2_alphas
            zeros = experts.w2_zp
            g_idx = experts.w2_g_idx
            sort_indices = experts.w2_g_idx_sort_indices
            output_features = layer._cb_hidden
        if scale is None:
            raise RuntimeError(
                f"{runtime.declaration.tensor_prefix}/{family}: delegated "
                "Marlin method produced no weight scale"
            )
        result = torch.empty(
            (rows, output_features), dtype=inputs.dtype, device=inputs.device
        )
        return native_marlin.ops.moe_wna16_marlin_gemm(
            gemm_input,
            result,
            weight,
            bias,
            scale,
            activation_scale,
            global_scale,
            zeros,
            g_idx,
            sort_indices,
            native_marlin.marlin_make_workspace_new(inputs.device, 4),
            sorted_tokens,
            expert_ids,
            padded,
            topk_weights,
            moe_block_size=block_size,
            top_k=1,
            mul_topk_weights=False,
            b_q_type=native_marlin.ScalarType.from_id(experts.quant_type_id),
            size_m=rows,
            size_n=output_features,
            size_k=inputs.shape[1],
            is_k_full=experts.is_k_full,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    @staticmethod
    def _quantize(method, values):
        from . import ops
        return (ops.fp4_act_qdq(values) if method.is_fp4
                else ops.fp8_act_qdq(values.to(torch.bfloat16)))

    def _run_cb_decode(self, runtime, family, inputs, rows, local_ids):
        from . import ops
        method, lane = runtime.method, runtime.lane
        quantized = self._quantize(method, inputs)
        qweight = getattr(lane, f"{family}_cb_qweight").data
        if method.is_fp4:
            if getattr(lane, f"_cb_use_v2_{family}", False):
                return ops.cb_moe_gemv_v2(
                    quantized, qweight, lane._cb_flat, lane._cb_compose,
                    local_ids.to(torch.int32), rows.to(torch.int32),
                    method.k, method.type_size, 0, 0,
                )
            return ops.cb_moe_gemv_fp4_v2(
                quantized, qweight, lane._cb_flat, lane._cb_compose,
                local_ids.to(torch.int32), rows.to(torch.int32),
                method.k, method.n_sub, method.type_size,
            )
        return ops.cb_moe_gemv_fp8(
            quantized, qweight, lane._cb_flat_fp8,
            getattr(lane, f"{family}_weight_scale").data,
            local_ids.to(torch.int32), rows.to(torch.int32),
            method.k, method.n_sub, method.type_size,
        )

    def _run_cb_prefill(self, runtime, family, inputs, local_ids):
        from . import ops
        method, lane = runtime.method, runtime.lane
        quantized = self._quantize(method, inputs)
        order = torch.argsort(local_ids, stable=True)
        sorted_ids = local_ids.index_select(0, order)
        sorted_input = quantized.index_select(0, order).contiguous()
        counts = torch.zeros(
            lane._cb_E, dtype=torch.int32, device=inputs.device
        )
        counts.scatter_add_(0, sorted_ids, torch.ones_like(
            sorted_ids, dtype=torch.int32
        ))
        expert_ends = torch.cumsum(counts, 0, dtype=torch.int32).contiguous()
        weight = method._expand_native_bf16_slice(
            lane, family, 0, lane._cb_E
        )
        sorted_output = torch.empty(
            (inputs.shape[0], weight.shape[1]), dtype=torch.bfloat16,
            device=inputs.device,
        )
        ops.cb_bf16_grouped_mm_out(
            sorted_output, sorted_input, weight, expert_ends, 0
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return sorted_output.index_select(0, inverse)
