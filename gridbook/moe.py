"""``PrismaQuantCBMoEMethod`` — FusedMoE serving for stacked CB expert weights
(docs/lanes/nvfp4-cb/moe_cb_design.md §4, LAYOUT.md §3 stacked layout).

Each expert stack ships as ONE tensor per role: ``<q>.cb_qweight`` uint8
``(E, out, (in/256)·type_size)`` (+ fp8 ``<q>.weight_scale`` ``(E, out)``), where
``cb_qweight[e]`` is exactly the dense §1 superblock layout. All experts of a
stack share one format + one codebook (per-layer uniformity, union-find at
export; asserted here). Serving registers the w13/w2 stacks and dispatches only
to owned native CUDA/CUTLASS kernels. Decode uses grouped GEMV. Quality prefill
uses a gated FP8 collective, an explicitly contracted native NVFP4 collective,
or exact activation QDQ plus bounded BF16 weight expansion feeding an owned
grouped CUTLASS GEMM.

  w13 = fused gate_up_proj : (E, 2·inter, hidden)  -> cb_qweight (E, 2·inter, bytes)
  w2  = down_proj          : (E, hidden, inter)    -> cb_qweight (E, hidden, bytes)

The loader attests every reachable native extension before model construction
finishes, so no JIT build or implementation selection occurs in first forward
or CUDA-graph capture.
"""
from __future__ import annotations

import os
import sys

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from . import codec
from .cb_fill_guard import (
    assert_cb_experts_filled,
    mark_filled,
    mark_unfilled,
)
from .moe_gemv_select import cb_gemv_choice
from .moe_routing import (
    cb_cached_row_offsets,
    cb_grouped_pad_routing,
)
from .native_cutlass import (native_fp4_quant, native_fp8_quant,
                             native_moe_activation,
                             require_native_fp4_quant,
                             require_native_fp8_cutlass,
                             require_native_moe_activation)
from .nvfp4_activation_contract import (
    CONTRACT_KEY as _NVFP4_ACTIVATION_CONTRACT_KEY,
    reciprocal_vector as _nvfp4_reciprocal_vector,
    require_identical_loaded_scales,
    rowwise_range_multiplier as _nvfp4_rowwise_range_multiplier,
)
from .ops import fp4_act_qdq_or_codec


def _row_bytes(in_features: int, type_size: int) -> int:
    return (in_features // codec.SUPERBLOCK) * type_size


_FUSED_FP4_MOE_STATE: list[str] = []
_FUSED_FP4_MOE_STATIC_MODES = frozenset(("1", "128", "256"))
_FUSED_FP4_MOE_STATIC_LSQ_MODES = frozenset(
    ("static_lsq", "static_lsq128", "static_lsq256")
)
_FUSED_FP4_MOE_ROWWISE_MODES = frozenset(
    ("rowwise", "rowwise128", "rowwise256")
)
_FUSED_FP4_MOE_MODES = (
    _FUSED_FP4_MOE_STATIC_MODES
    | _FUSED_FP4_MOE_STATIC_LSQ_MODES
    | _FUSED_FP4_MOE_ROWWISE_MODES
)
_FUSED_FP4_MOE_ALLOWED_MODES = frozenset(("",)) | _FUSED_FP4_MOE_MODES


def _requested_fused_fp4_moe_mode() -> str:
    """Validated, process-stable fused-FP4 MoE opt-in selector.

    A typo must not turn an intended enablement A/B into an unlabelled baseline
    run, and changing the activation contract between forwards is never a
    supported way to compare the paths.
    """
    current = os.environ.get("PRISMAQUANT_CB_FUSED_FP4_MOE", "").strip()
    if current not in _FUSED_FP4_MOE_ALLOWED_MODES:
        raise ValueError(
            "invalid PRISMAQUANT_CB_FUSED_FP4_MOE="
            f"{current!r}; expected '', '1', '128', '256', 'static_lsq', "
            "'static_lsq128', 'static_lsq256', 'rowwise', 'rowwise128', "
            "or 'rowwise256'"
        )
    if not _FUSED_FP4_MOE_STATE:
        _FUSED_FP4_MOE_STATE.append(current)
    elif current != _FUSED_FP4_MOE_STATE[0]:
        raise RuntimeError(
            "PRISMAQUANT_CB_FUSED_FP4_MOE changed after Gridbook dispatch was "
            "fixed; restart the process instead of mixing activation contracts"
        )
    return _FUSED_FP4_MOE_STATE[0]


def _native_scaled_fp4_quant_available() -> bool:
    """Whether the pinned native static-NVFP4 ABI can be attested."""
    try:
        require_native_fp4_quant("routed static FP4 activation")
        return True
    except RuntimeError:
        return False


# Backward-compatible private name for the integrated follow-on patch and its
# focused tests.  The implementation lives in codec so dense, MoE and
# top-level-loader codebook construction all enforce the same contract.
_assert_cast_lossless = codec.assert_cast_lossless
# Compatibility seam for focused CPU tests; implementation is the shared,
# directly registered native ABI in ``native_cutlass``.
_native_scaled_fp4_quant = native_fp4_quant


class PrismaQuantCBMoEMethod(FusedMoEMethodBase):
    """CB decode for RoutedExperts (FusedMoE) — one uniform CB format per layer."""

    # Process-wide fail-loud state for native grouped decode.
    _DECODE_ENGAGED_LOGGED = False       # one-time decode-path engagement line
    _DECODE_DISABLED_LOGGED = False      # deduplicate a host-wide ext failure

    def __init__(self, quant_config, moe: FusedMoEConfig, scheme: dict,
                 prefix: str) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.scheme = scheme
        self.prefix = prefix
        self.is_fp4 = scheme["grid"] == "fp4"
        self.k = int(scheme["k"])
        self.n_sub = int(scheme["n_sub"])
        self.type_size = int(scheme["type_size"])
        self.has_static_fp4_activation = (
            self.is_fp4
            and scheme.get("activation_contract")
            == _NVFP4_ACTIVATION_CONTRACT_KEY
        )
        sc = scheme.get("scale_coding")
        if isinstance(sc, dict):
            self.is_v2 = sc.get("kind") == codec.SCALE_CODING_TWO_TIER
            self._sub_table = sc.get("table") or codec.TWO_TIER_SUB_TABLE
        else:
            self.is_v2 = (sc == codec.SCALE_CODING_TWO_TIER)
            self._sub_table = codec.TWO_TIER_SUB_TABLE if self.is_v2 else None
        if self.is_fp4 and not self.is_v2:
            # fp4-v1 expert transient is a follow-up (no compose-during-expand);
            # export MoE experts as fp8-CB or fp4 two-tier v2.
            raise NotImplementedError(
                f"{prefix}: fp4 MoE experts require two-tier v2 scale coding "
                "(fp4-v1 expert transient not yet implemented)")
        if self.is_fp4 and (self.n_sub not in (1, 2)
                            or self.type_size != 4 * self.k + 9):
            raise NotImplementedError(
                f"{prefix}: native FP4 MoE requires the v2 serialized layout "
                "(n_sub in {1,2}, type_size=4*k+9)")

    # -- weight buffers (stacked experts) ------------------------------------
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        del params_dtype
        E = num_experts
        inter = intermediate_size_per_partition
        layer._cb_hidden = hidden_size
        layer._cb_inter = inter
        # extra_weight_attrs already carries the weight_loader; ONE
        # set_weight_attrs per param (a second call trips vLLM's
        # "Overwriting existing tensor attribute" assert — 35B first serve).
        attrs = dict(extra_weight_attrs)
        # w13 = gate_up: out=2*inter, in=hidden.  w2 = down: out=hidden, in=inter.
        w13 = torch.nn.Parameter(torch.empty(
            E, 2 * inter, _row_bytes(hidden_size, self.type_size),
            dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(w13, {**attrs, "is_transposed": False})
        # Fill sentinel (cb_fill_guard): False until a fill path copies
        # checkpoint bytes in; checked in process_weights_after_loading so a
        # missing per-arch loader opt-in fails loudly instead of serving
        # uninitialised memory. Set AFTER set_weight_attrs — that helper asserts
        # on overwriting an existing attribute.
        mark_unfilled(w13)
        layer.register_parameter("w13_cb_qweight", w13)

        w2 = torch.nn.Parameter(torch.empty(
            E, hidden_size, _row_bytes(inter, self.type_size),
            dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(w2, {**attrs, "is_transposed": False})
        mark_unfilled(w2)
        layer.register_parameter("w2_cb_qweight", w2)

        if not self.is_fp4:                       # fp8: per-(expert, out) scale
            w13s = torch.nn.Parameter(
                torch.empty(E, 2 * inter, dtype=torch.float32),
                requires_grad=False)
            set_weight_attrs(w13s, dict(attrs))
            layer.register_parameter("w13_weight_scale", w13s)
            w2s = torch.nn.Parameter(
                torch.empty(E, hidden_size, dtype=torch.float32),
                requires_grad=False)
            set_weight_attrs(w2s, dict(attrs))
            layer.register_parameter("w2_weight_scale", w2s)

        if getattr(self, "has_static_fp4_activation", False):
            for which in ("w13", "w2"):
                scale = torch.nn.Parameter(
                    torch.full((1,), float("nan"), dtype=torch.float32),
                    requires_grad=False,
                )
                set_weight_attrs(scale, dict(attrs))
                layer.register_parameter(
                    f"{which}_input_global_scale", scale
                )

        # Instance-level load hook (GGUF-plugin pattern, zero core patches):
        # vLLM's RoutedExperts.load_weights maps checkpoint names by
        # substring-replacing the projection name, which (a) derives DOTTED
        # attribute names for our `<proj>.cb_qweight` suffix (getattr fails)
        # and (b) applies a bf16-orientation transpose heuristic that would
        # corrupt byte tensors (last dim = row_bytes, never hidden). CB
        # tensors therefore load DIRECTLY into our stacked params; every
        # other tensor delegates to the original loader untouched.
        orig_load = getattr(layer, "load_weights", None)
        if not getattr(layer, "_cb_load_wrapped", False) and orig_load is not None:
            prefix = self.prefix
            cb_map = {
                "gate_up_proj.cb_qweight": "w13_cb_qweight",
                "down_proj.cb_qweight": "w2_cb_qweight",
                "gate_up_proj.weight_scale": "w13_weight_scale",
                "down_proj.weight_scale": "w2_weight_scale",
                "gate_up_proj.input_global_scale": (
                    "w13_input_global_scale"
                ),
                "down_proj.input_global_scale": "w2_input_global_scale",
            }

            def _cb_load_weights(weights):
                deferred = []
                for name, w in weights:
                    pname = cb_map.get(name)
                    if pname is not None and hasattr(layer, pname):
                        p = getattr(layer, pname)
                        if tuple(p.shape) != tuple(w.shape):
                            raise ValueError(
                                f"{prefix}.{name}: checkpoint shape "
                                f"{tuple(w.shape)} != param {tuple(p.shape)}"
                                f" — stacked (E, out, bytes) contract violated")
                        p.data.copy_(w.to(p.dtype))
                        if pname.endswith("cb_qweight"):
                            mark_filled(p)      # fill path 1 of 2 (per-layer)
                        yield pname
                    else:
                        deferred.append((name, w))
                if deferred:
                    yield from orig_load(deferred)

            layer.load_weights = _cb_load_weights
            layer._cb_load_wrapped = True

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig | None:
        return None

    # -- per-stack codebook / compose + uniformity assert --------------------
    def process_weights_after_loading(self, layer: torch.nn.Module):
        # The one place a never-installed per-arch loader is detectable: vLLM
        # calls this for every CB MoE layer on every load path, and neither fill
        # path ran if the stacks are still unfilled. No env bypass (R10/D3).
        assert_cb_experts_filled(layer, self.prefix)
        dev = layer.w13_cb_qweight.device
        E = layer.w13_cb_qweight.shape[0]
        codebooks = self.quant_config.get_codebooks()
        if getattr(self, "has_static_fp4_activation", False):
            stage_targets = self.quant_config.moe_activation_stage_targets(
                self.prefix
            )
            for which in ("w13", "w2"):
                param_name = f"{which}_input_global_scale"
                if not hasattr(layer, param_name):
                    raise ValueError(
                        f"{self.prefix}: contracted FP4-CB MoE has no "
                        f"{param_name} parameter"
                    )
                targets = stage_targets[which]
                if not targets:
                    raise ValueError(
                        f"{self.prefix}: contracted FP4-CB MoE has no "
                        f"physical activation target for {which}"
                    )
                expected = self.quant_config.activation_scales_for_targets(
                    targets
                )
                scale = require_identical_loaded_scales(
                    getattr(layer, param_name).data,
                    prefix=f"{self.prefix}.{which}",
                    expected=expected,
                )
                setattr(layer, f"_cb_fp4_input_global_scale_{which}", scale)
                # The shared static-LSQ quantizer takes the already-attested
                # F32 by value. Cache it once at load so MoE prefill never
                # performs a device-to-host scalar read in steady state.
                setattr(
                    layer,
                    f"_cb_fp4_input_global_scale_{which}_f32",
                    float(expected[0]),
                )
        ref = self.scheme["codebook_ref"]
        names = ref if isinstance(ref, list) else [ref]
        subs = [codebooks[n].to(dev) for n in names]
        layer._cb_flat = codec.build_flat_codebook(
            subs, self.prefix, "fp4" if self.is_fp4 else "fp8")
        layer._cb_row0 = torch.zeros(1, dtype=torch.int32, device=dev)
        if self.is_v2:
            layer._cb_compose = codec.build_compose_table(
                self._sub_table).to(dev)
        else:
            layer._cb_compose = torch.zeros(1, dtype=torch.float32, device=dev)
            # Materialize and validate every representation while the model is
            # loading.  A disabled grouped-decode gate must not defer this to
            # the first stock-prefill forward (which may be under capture).
            self._stock_cb_flat_fp8(layer)
        # Per-layer uniformity: one format for all experts (union-find at
        # export). The stacked buffer is single-format by construction; assert
        # the byte width matches the scheme so a mis-exported stack fails loudly.
        exp_w13 = _row_bytes(layer._cb_hidden, self.type_size)
        exp_w2 = _row_bytes(layer._cb_inter, self.type_size)
        assert layer.w13_cb_qweight.shape[2] == exp_w13, (
            f"{self.prefix}: w13 byte width {layer.w13_cb_qweight.shape[2]} != "
            f"{exp_w13} (type_size/uniformity mismatch)")
        assert layer.w2_cb_qweight.shape[2] == exp_w2
        # Which grouped GEMV each stack runs. Decided HERE, at load, on python
        # ints only — so `_apply_grouped_decode` gains no host work, the
        # call-site branch is a trace-time constant (cudagraph-safe by
        # construction, no host sync reachable from a captured region), and the
        # operator gets a startup diagnostic instead of a first-forward
        # surprise. Same reasoning as the byte-width asserts above.
        # The two stacks are decided separately: they sit on different K
        # (hidden vs inter) and the occupancy predicate is a function of K, so
        # w13 and w2 of ONE layer can legitimately land on different kernels.
        for which, in_f in (("w13", layer._cb_hidden), ("w2", layer._cb_inter)):
            if self.is_fp4 and self.is_v2:
                use_v2, why = cb_gemv_choice(
                    self.k, self.n_sub, self.type_size, in_f, dev)
            else:
                # fp8-CB v1 has no v2 kernel. Short-circuit BEFORE the probe so
                # an fp8-only serve never pays the v2 JIT build.
                use_v2, why = False, "not fp4-CB two-tier v2"
            setattr(layer, f"_cb_use_v2_{which}", use_v2)
            print(f"[prismaquant-cb] cb_gemv_kernel {self.prefix}.{which} "
                  f"k={self.k} n_sub={self.n_sub} type_size={self.type_size} "
                  f"K={in_f} -> {'v2' if use_v2 else 'inherited'} ({why})",
                  flush=True)
        layer._cb_E = E

        # Attest every native extension reachable from production dispatch at
        # model load. JIT compilation must not occur inside first prefill or a
        # CUDA-graph capture, and no missing kernel may select a stock/Triton
        # substitute. FP8 uses the main extension for decode/QDQ/expansion;
        # FP4 also needs the exact v2 expander. Both use the grouped CUTLASS
        # quality bridge whenever the fused shape gate misses.
        from .cuda_ext import (NativeKernelUnavailableError,
                               require_bf16_grouped_ext, require_ext,
                               require_fp4_v2_expander)
        if layer._cb_hidden % 8 or layer._cb_inter % 8:
            raise NativeKernelUnavailableError(
                f"{self.prefix}: native BF16 grouped quality prefill requires "
                "hidden and intermediate dimensions divisible by 8, got "
                f"hidden={layer._cb_hidden}, intermediate={layer._cb_inter}")
        require_ext(f"{self.prefix} routed CB decode/QDQ/expansion")
        if self.is_fp4:
            require_fp4_v2_expander(
                f"{self.prefix} routed FP4-v2 expansion", device=dev)
        else:
            require_native_fp8_cutlass(
                f"{self.prefix} routed FP8 quality prefill")
        require_bf16_grouped_ext(
            f"{self.prefix} routed quality prefill")
        if not self._cuda_moe_ok(layer):
            raise NativeKernelUnavailableError(
                f"{self.prefix}: routed decode layout has no native grouped "
                "CUDA kernel")
        layer._cb_native_activation = require_native_moe_activation(
            layer.activation.value, f"{self.prefix} routed activation")
        # Resolve optional fused CUTLASS eligibility (and any JIT build) now;
        # both False and True are cached before the first request.
        if not self.is_fp4:
            self._gf2_ok(layer)
        else:
            mode = _requested_fused_fp4_moe_mode()
            layer._cb_fused_fp4_moe_mode = mode
            if mode:
                rowwise = mode in _FUSED_FP4_MOE_ROWWISE_MODES
                static_lsq = mode in _FUSED_FP4_MOE_STATIC_LSQ_MODES
                fused_ok = self._gf4_ok(
                    layer, rowwise=rowwise, static_lsq=static_lsq
                )
                if not fused_ok:
                    cache_attr = (
                        "_cb_gf4_rowwise_ok_reason" if rowwise else
                        "_cb_gf4_static_lsq_ok_reason" if static_lsq else
                        "_cb_gf4_static_ok_reason"
                    )
                    reason = getattr(layer, cache_attr, "unknown constraint")
                    from .cuda_ext import NativeKernelUnavailableError
                    raise NativeKernelUnavailableError(
                        f"{self.prefix}: requested native fused FP4 MoE mode "
                        f"{mode!r} is unavailable ({reason}); changing to the "
                        "exact BF16 bridge would violate the explicit "
                        "activation contract")
            elif self.n_sub != 2:
                from .cuda_ext import NativeKernelUnavailableError
                raise NativeKernelUnavailableError(
                    f"{self.prefix}: signed FP4-CB experts have no exact "
                    "native BF16 prefill bridge; select a supported native "
                    "fused FP4 MoE activation mode")
        from .ops import register_cb_layer
        layer._cb_layer_id = register_cb_layer(self, layer)

    def apply(self, layer: RoutedExperts, x: torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts, shared_experts_input) -> torch.Tensor:
        # Shared expert (e.g. hy_v3): apply() correctly returns ROUTED-ONLY.
        # vLLM's MoERunner._apply_quant_method runs the SharedExperts wrapper
        # SEPARATELY (`_maybe_apply_shared_experts` -> `SharedExperts._layer(
        # shared_experts_input)`), producing shared_output, and _maybe_combine
        # adds it to our routed output — verified against the vLLM v0.23 runner
        # (moe_runner.py). The wrapper's `_layer` IS the shared_mlp module,
        # whose projections are native CB Linears through config prefix aliases,
        # so the shared contribution is computed and included; a routed-only
        # return here is the contract, NOT a dropped component. (The `_unpack` path also
        # accepts a `(shared, routed)` tuple for methods that fuse the shared
        # expert into their kernel — we don't, so single-tensor is correct.)
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input unsupported for CB MoE")
        # M-branch hoist (see ops.py/linear.py): the token-count branch AND
        # the prefill path's host routing/dispatch logic lives inside ONE
        # opaque custom op, so
        # compile/capture at decode sizes record the grouped decode kernels
        # and dynamo never traces the loop. Opaque dispatch is mandatory:
        # exposing routing/pointwise ATen code to Inductor could generate
        # Triton, which is outside Gridbook's native-only contract.
        lid = getattr(layer, "_cb_layer_id", None)
        if lid is None:
            from .cuda_ext import NativeKernelUnavailableError
            raise NativeKernelUnavailableError(
                f"{self.prefix}: CB MoE was not registered during "
                "process_weights_after_loading")
        from .ops import cb_moe_forward
        return cb_moe_forward(x, topk_weights, topk_ids, lid)

    def _apply_inline(self, layer, x, topk_weights, topk_ids):
        act = getattr(layer, "_cb_native_activation", layer.activation.value)
        num_tokens = x.shape[0]

        if num_tokens <= 16:
            if not self._cuda_moe_ok(layer):
                from .cuda_ext import NativeKernelUnavailableError
                raise NativeKernelUnavailableError(
                    f"{self.prefix}: routed decode requires Gridbook's native "
                    "grouped CUDA GEMV extension")
            return self._apply_grouped_decode(
                layer, x, topk_weights, topk_ids, act)

        if self.is_fp4:
            mode = getattr(layer, "_cb_fused_fp4_moe_mode", None)
            if mode is None:
                # Production fixes this at model load. This branch supports
                # deliberately minimal test layers without changing serving.
                mode = _requested_fused_fp4_moe_mode()
            if mode in _FUSED_FP4_MOE_MODES:
                out = self._apply_prefill_grouped_fused_fp4(
                    layer, x, topk_weights, topk_ids, act,
                    tile_m=256 if mode.endswith("256") else 128,
                    rowwise=mode in _FUSED_FP4_MOE_ROWWISE_MODES,
                    static_lsq=mode in _FUSED_FP4_MOE_STATIC_LSQ_MODES)
                if out is not None:
                    return out
                from .cuda_ext import NativeKernelUnavailableError
                raise NativeKernelUnavailableError(
                    f"{self.prefix}: requested native fused FP4 MoE mode "
                    f"{mode!r} became unavailable after model load")
            return self._apply_prefill_native_bf16(
                layer, x, topk_weights, topk_ids, act)

        # FP8-CB keeps its quality-gated decode-in-prologue CUTLASS path where
        # eligible. Every miss uses exact QDQ and bounded native BF16 expansion
        # feeding Gridbook's owned grouped CUTLASS bridge.
        out = self._apply_prefill_grouped_fused_v2(
                layer, x, topk_weights, topk_ids, act)
        if out is None:
            out = self._apply_prefill_grouped_fused(
                layer, x, topk_weights, topk_ids, act)
        if out is not None:
            return out
        return self._apply_prefill_native_bf16(
            layer, x, topk_weights, topk_ids, act)

    # -- prefill: fp4 grouped FUSED (block-scaled decode-in-prologue) --------
    def _gf4_ok(self, layer, *, rowwise: bool = False,
                static_lsq: bool = False) -> bool:
        """Eligibility for the fp4 grouped fused prefill (cached per layer).
        Mirrors cb_fused_fp4_moe_grouped's TORCH_CHECKs and records a diagnostic
        before serving starts. An explicit activation selector fails closed on
        a miss; unset product FP4 uses the exact native BF16 bridge. Static,
        static-LSQ, and rowwise symbol families are isolated from each other.
        """
        if rowwise and static_lsq:
            return False
        cache_attr = (
            "_cb_gf4_rowwise_ok" if rowwise else
            "_cb_gf4_static_lsq_ok" if static_lsq else
            "_cb_gf4_static_ok"
        )
        reason_attr = f"{cache_attr}_reason"
        ok = getattr(layer, cache_attr, None)
        if ok is not None:
            return ok
        reason = None
        if not self.is_fp4:
            reason = "format is not FP4-CB"
        elif not self.is_v2:
            reason = "format is not two-tier layout v2"
        elif not rowwise and not getattr(
            self, "has_static_fp4_activation", False
        ):
            reason = "artifact has no attested static activation contract"
        elif not rowwise and any(
            getattr(layer, f"_cb_fp4_input_global_scale_{which}", None)
            is None
            for which in ("w13", "w2")
        ):
            reason = "artifact has no loaded stage activation scales"
        elif not 12 <= self.k <= 24:
            reason = f"k={self.k} is outside the fused range [12, 24]"
        elif self.n_sub not in (1, 2):
            reason = f"n_sub={self.n_sub} is not 1 or 2"
        elif self.type_size != 4 * self.k + 9:
            reason = "serialized row type_size is not FP4-CB layout v2"
        elif layer._cb_hidden % codec.SUPERBLOCK != 0:
            reason = "hidden size is not superblock aligned"
        elif layer._cb_inter % codec.SUPERBLOCK != 0:
            reason = "intermediate size is not superblock aligned"
        if reason is None and static_lsq:
            have_host_scales = all(
                getattr(
                    layer,
                    f"_cb_fp4_input_global_scale_{which}_f32",
                    None,
                ) is not None
                for which in ("w13", "w2")
            )
            if not have_host_scales:
                reason = "attested stage scales have no cached host values"
        if (reason is None
                and codec.fp4_value_lut_nbytes(self.k, self.n_sub)
                > codec.FP4_FUSED_LUT_MAX_BYTES):
            reason = "decoded codebook exceeds the fused LUT capacity"
        if (reason is None and not rowwise and not static_lsq
                and not _native_scaled_fp4_quant_available()):
            reason = "direct native scaled_fp4_quant ABI is unavailable"
        if reason is None:
            from .cuda_ext import get_fused_fp4_ext
            fext = get_fused_fp4_ext()
            if fext is None:
                reason = "fused FP4 extension is unavailable"
            elif not hasattr(fext, "cb_fused_fp4_moe_grouped"):
                reason = "grouped fused FP4 symbol is unavailable"
            elif rowwise and not hasattr(fext, "cb_nvfp4_quantize_rows"):
                reason = "rowwise native FP4 quantizer is unavailable"
            elif static_lsq and not hasattr(
                fext, "cb_nvfp4_quantize_static_lsq"
            ):
                reason = "static-LSQ native FP4 quantizer is unavailable"
        ok = reason is None
        setattr(layer, cache_attr, bool(ok))
        setattr(layer, reason_attr, reason)
        if not ok:
            family = "rowwise" if rowwise else (
                "static_lsq" if static_lsq else "static"
            )
            print(
                f"[prismaquant-cb] fused_fp4_moe {self.prefix} "
                f"mode={family} -> unavailable ({reason})",
                flush=True,
            )
        return bool(ok)

    def _fp4_quant(self, layer, x2: torch.Tensor, which: str, *,
                   rowwise: bool = False, static_lsq: bool = False,
                   fext=None):
        """Native NVFP4 activation quant (packed e2m1 + swizzled ue4m3 SFA +
        the per-row fp32 residual for the EVT). Static mode uses the producer-
        calibrated, stage-specific scalar. Static-LSQ keeps the same scalar and
        packed payload while fitting the existing per-row residual. Rowwise
        mode derives each scalar independently at runtime and consumes no
        artifact metadata."""
        if rowwise and static_lsq:
            raise ValueError("rowwise and static_lsq are mutually exclusive")
        if rowwise:
            if fext is None:
                from .cuda_ext import get_fused_fp4_ext
                fext = get_fused_fp4_ext()
            return fext.cb_nvfp4_quantize_rows(
                x2.contiguous(), _nvfp4_rowwise_range_multiplier()
            )
        if static_lsq:
            if fext is None:
                from .cuda_ext import get_fused_fp4_ext
                fext = get_fused_fp4_ext()
            gs = getattr(
                layer, f"_cb_fp4_input_global_scale_{which}_f32"
            )
            return fext.cb_nvfp4_quantize_static_lsq(x2.contiguous(), gs)

        gs = getattr(layer, f"_cb_fp4_input_global_scale_{which}")
        aq, sf = _native_scaled_fp4_quant(x2, gs)
        recip = _nvfp4_reciprocal_vector(
            layer, which=which, scale=gs, rows=x2.shape[0]
        )
        return aq, sf.view(torch.uint8).reshape(-1), recip

    def _apply_prefill_grouped_fused_fp4(self, layer, x, topk_weights,
                                         topk_ids, act, *, tile_m=128,
                                         rowwise: bool = False,
                                         static_lsq: bool = False):
        """fp4 counterpart of ``_apply_prefill_grouped_fused_v2``: identical
        padded routing / gather / throwaway-row combine; the projection GEMMs
        are ONE ``cb_fused_fp4_moe_grouped`` each (native NVF4 block-scaled
        MMA, packed rows decoded in the producer/prologue, weight scales
        applied in-MMA via SFB). Activations are quantized AFTER the gather
        (the swizzled SFA plane is row-block-interleaved and cannot be
        gathered), which is bit-equivalent per row at a fixed global scale.
        Returns None on any constraint miss."""
        if not self._gf4_ok(
            layer, rowwise=rowwise, static_lsq=static_lsq
        ):
            return None
        if x.dtype not in (torch.bfloat16, torch.float16):
            return None
        if tile_m not in (128, 256):
            return None
        from .cuda_ext import get_fused_fp4_ext
        fext = get_fused_fp4_ext()

        E = layer._cb_E
        T = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden
        inter = layer._cb_inter
        N1 = 2 * inter
        d = N1 // 2
        kb = self.k

        luts = getattr(layer, "_cb_fp4_gf", None)
        if luts is None:
            lut = codec.build_fp4_value_lut(
                layer._cb_flat, kb, self.n_sub).to(dev)
            compose = codec.build_compose_u8(self._sub_table).to(dev)
            luts = (lut, compose,
                    torch.ones(N1, dtype=torch.float32, device=dev),
                    torch.ones(Kh, dtype=torch.float32, device=dev))
            layer._cb_fp4_gf = luts
        lut, compose, ones_n1, ones_kh = luts
        w13 = layer.w13_cb_qweight.data
        w2 = layer.w2_cb_qweight.data

        # Stable expert grouping plus tile padding remains entirely on device;
        # padding rows gather the appended zero row and scatter into throwaway
        # destination T.
        pair_expert = topk_ids.reshape(-1).to(torch.long)
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)
        order = torch.argsort(pair_expert, stable=True)
        ptok_sorted = pair_token[order]
        pw_sorted = topk_weights.reshape(-1)[order].to(torch.float32)

        expert_ids, row_src, is_pad, n_blocks = cb_grouped_pad_routing(
            topk_ids, E, tile_m)
        if os.environ.get("PRISMAQUANT_CB_GROUPED_TRIM", "1") == "1":
            nb = int(n_blocks.item())
            expert_ids = expert_ids[:nb].contiguous()
            row_src = row_src[:nb * tile_m]
            is_pad = is_pad[:nb * tile_m]

        rows = ptok_sorted.index_select(0, row_src)
        dest = torch.where(is_pad, torch.full_like(rows, T), rows)
        x1 = torch.cat([x, x.new_zeros((1, Kh))])
        a_pad = x1.index_select(0, dest).contiguous()
        padded_rows = a_pad.shape[0]

        aq1, sfa1, recip1 = self._fp4_quant(
            layer, a_pad, "w13", rowwise=rowwise,
            static_lsq=static_lsq, fext=fext)
        del a_pad
        gate_up = fext.cb_fused_fp4_moe_grouped(
            aq1, sfa1, w13, lut, compose,
            recip1, ones_n1, expert_ids,
            N1, Kh, kb, self.n_sub, self.type_size, True, tile_m)
        del aq1, sfa1
        activated = torch.empty(
            (padded_rows, d), dtype=gate_up.dtype, device=dev)
        native_moe_activation(act, activated, gate_up)
        del gate_up

        aq2, sfa2, recip2 = self._fp4_quant(
            layer, activated, "w2", rowwise=rowwise,
            static_lsq=static_lsq, fext=fext)
        del activated
        y = fext.cb_fused_fp4_moe_grouped(
            aq2, sfa2, w2, lut, compose,
            recip2, ones_kh, expert_ids,
            Kh, inter, kb, self.n_sub, self.type_size, True, tile_m)
        del aq2, sfa2

        pw_pad = pw_sorted.index_select(0, row_src)
        y = y * pw_pad[:, None].to(y.dtype)
        out = torch.zeros((T + 1, Kh), dtype=x.dtype, device=dev)
        out.index_add_(0, dest, y.to(out.dtype))
        return out[:T]

    def _native_bf16_chunk(self, layer) -> int:
        """Fixed expert chunk for the bounded BF16 CUTLASS bridge.

        The larger w13 transient is the sizing authority. A user override is
        retained for measurement, while the default keeps each decoded chunk
        under one GiB. All values are model/config integers, so no device read
        or routing-dependent Python control flow enters the hot path.
        """
        E = int(layer._cb_E)
        override = os.environ.get("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")
        if override is not None and override.strip():
            try:
                value = int(override)
            except ValueError as exc:
                raise ValueError(
                    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK must be positive") \
                    from exc
            if value <= 0:
                raise ValueError(
                    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK must be positive")
            return min(E, value)
        raw_budget = os.environ.get("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES")
        try:
            budget = int(raw_budget) if raw_budget else (1 << 30)
        except ValueError as exc:
            raise ValueError(
                "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES must be positive") from exc
        if budget <= 0:
            raise ValueError(
                "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES must be positive")
        per_expert = (2 * int(layer._cb_inter)
                      * int(layer._cb_hidden) * 2)  # w13 BF16 bytes
        return max(1, min(E, budget // max(1, per_expert)))

    def _expand_native_bf16_slice(self, layer, which: str,
                                  c0: int, c1: int) -> torch.Tensor:
        """Expand one contiguous expert slice to exact BF16 natively."""
        from . import ops as pq_ops

        packed = getattr(layer, f"{which}_cb_qweight")[c0:c1].contiguous()
        n_e = c1 - c0
        out_f = int(packed.shape[1])
        in_f = (int(layer._cb_hidden) if which == "w13"
                else int(layer._cb_inter))
        rows = n_e * out_f
        raw = packed.reshape(rows, -1)
        if self.is_fp4:
            if not (self.is_v2 and self.n_sub == 2
                    and self.type_size == 4 * self.k + 9):
                from .cuda_ext import NativeKernelUnavailableError
                raise NativeKernelUnavailableError(
                    f"{self.prefix}: native quality prefill supports only "
                    "FP4-CB-v2 product experts (n_sub=2, type_size=4k+9)")
            weight = pq_ops.cb_expand_fp4_v2(
                raw.view(-1), layer._cb_flat, layer._cb_compose,
                0, rows, in_f, self.k, self.type_size)
        else:
            row0 = cb_cached_row_offsets(layer, rows, packed.device)
            # FP8 codeword loads use an aligned 8-byte window that may cross
            # the logical end of a row. Preserve the main expander's required
            # per-row read slack; the stacked checkpoint plane itself is
            # tightly packed and does not provide it for the final row.
            raw = codec.pad_qweight(raw)
            value = pq_ops.cb_expand_fp8(
                raw, self._stock_cb_flat_fp8(layer), row0,
                rows, in_f, self.k, self.n_sub, self.type_size)
            scale = getattr(layer, f"{which}_weight_scale")[c0:c1] \
                .reshape(rows).to(torch.float32)
            weight = (value.float() * scale[:, None]).to(torch.bfloat16)
        return weight.view(n_e, out_f, in_f)

    def _apply_prefill_native_bf16(self, layer, x, topk_weights,
                                   topk_ids, act):
        """Quality-preserving routed prefill using owned native kernels.

        Weight decode is bit-exact FP4-v2/FP8-CB -> BF16; activation QDQ is the
        same exact native QDQ used by decode, both before FC1 and between FC1
        and FC2. Matrix multiplication is one device-scheduled CUTLASS grouped
        launch per expert chunk. Routing stays device-resident and zero-token
        experts remain zero-M CUTLASS problems (no host compaction or sync).
        """
        from . import ops as pq_ops

        if x.dtype is not torch.bfloat16:
            raise TypeError("native CB MoE prefill requires BF16 activations")
        E = int(layer._cb_E)
        T = int(x.shape[0])
        top_k = int(topk_ids.shape[-1])
        hidden = int(layer._cb_hidden)
        inter = int(layer._cb_inter)
        pair_expert = topk_ids.reshape(-1).to(torch.int64)
        order = torch.argsort(pair_expert, stable=True)
        pair_token = torch.arange(
            T, dtype=torch.int64, device=x.device).repeat_interleave(top_k)
        rows = pair_token.index_select(0, order)
        counts = torch.bincount(pair_expert, minlength=E)
        expert_ends = torch.cumsum(
            counts, 0, dtype=torch.int32).contiguous()

        xq = (pq_ops.fp4_act_qdq(x) if self.is_fp4
              else pq_ops.fp8_act_qdq(x))
        x_sorted = xq.index_select(0, rows).contiguous()
        pair_count = int(x_sorted.shape[0])
        chunk = self._native_bf16_chunk(layer)

        gate_up = torch.empty(
            (pair_count, 2 * inter), dtype=torch.bfloat16, device=x.device)
        for c0 in range(0, E, chunk):
            c1 = min(E, c0 + chunk)
            weight = self._expand_native_bf16_slice(
                layer, "w13", c0, c1)
            pq_ops.cb_bf16_grouped_mm_out(
                gate_up, x_sorted, weight, expert_ends, c0)
            del weight

        activated = torch.empty(
            (pair_count, inter), dtype=torch.bfloat16, device=x.device)
        native_moe_activation(act, activated, gate_up)
        del gate_up
        aq = (pq_ops.fp4_act_qdq(activated) if self.is_fp4
              else pq_ops.fp8_act_qdq(activated))
        del activated

        pair_output = torch.empty(
            (pair_count, hidden), dtype=torch.bfloat16, device=x.device)
        for c0 in range(0, E, chunk):
            c1 = min(E, c0 + chunk)
            weight = self._expand_native_bf16_slice(
                layer, "w2", c0, c1)
            pq_ops.cb_bf16_grouped_mm_out(
                pair_output, aq, weight, expert_ends, c0)
            del weight

        pair_weight = topk_weights.reshape(-1).index_select(0, order) \
            .to(pair_output.dtype)
        pair_output.mul_(pair_weight[:, None])
        output = torch.zeros((T, hidden), dtype=x.dtype, device=x.device)
        output.index_add_(0, rows, pair_output.to(output.dtype))
        return output


    # -- prefill: grouped FUSED (decode-in-prologue, round 1) ---------------
    def _gf_ok(self, layer) -> bool:
        """Eligibility for the grouped-fused prefill (cached per layer; the
        answer never changes after load). Mirrors the kernel's own
        TORCH_CHECKs so a miss is a silent fall-through, never a crash:

          * fp8-CB only (the fused kernel's LUT is the 4-sub-table e4m3
            codebook; fp4 two-tier composes a scale the prologue can't);
          * k in {28,32,36,40,44,48} — the KBits template dispatch, uniform
            per layer by the export union-find;
          * both projection K's are multiples of 256 (SUPERBLOCK, and the
            kernel's K%256 check);
          * packed row stride == row_bytes == (K/256)*4*k, which satisfies
            BOTH `stride(0) % 16 == 0` (4*k in {144,160,176,192}) and
            `stride(0) >= (K/256)*4*k_bits` with equality (UNPADDED rows).
        """
        ok = getattr(layer, "_cb_gf_ok", None)
        if ok is not None:
            return ok
        ok = (not self.is_fp4
              and self.n_sub == 4
              and self.k in (28, 32, 36, 40, 44, 48)
              and self.type_size == 4 * self.k
              and layer._cb_hidden % codec.SUPERBLOCK == 0
              and layer._cb_inter % codec.SUPERBLOCK == 0
              and hasattr(layer, "w13_weight_scale"))
        if ok:
            from .cuda_ext import get_fused_ext
            fext = get_fused_ext()
            ok = fext is not None and hasattr(fext, "cb_fused_prefill_mm_scaled")
        if ok:
            # 3-D stacked buffers: [e] must give a 2-D CONTIGUOUS [out, bytes]
            # view (stride(1)==1, stride(0)==row_bytes) — true for a slice of a
            # contiguous 3-D tensor, asserted rather than assumed.
            for which, in_f in (("w13", layer._cb_hidden),
                                ("w2", layer._cb_inter)):
                qw = getattr(layer, f"{which}_cb_qweight")
                rb = _row_bytes(in_f, self.type_size)
                # stride(0) matters only to R2, which hands the WHOLE stack to
                # the kernel and TORCH_CHECKs the expert stride; checking it
                # here keeps an exotic layout a silent fall-through rather than
                # a crash inside the extension.
                ok = ok and (qw.dim() == 3 and qw.stride(2) == 1
                             and qw.stride(1) == rb and qw.shape[2] == rb
                             and qw.stride(0) == qw.shape[1] * rb
                             and rb % 16 == 0)
        layer._cb_gf_ok = bool(ok)
        return layer._cb_gf_ok

    def _apply_prefill_grouped_fused(self, layer, x, topk_weights, topk_ids,
                                     act):
        """ROUND 1 of the native FP8-CB grouped fused prefill.

        Returns ``None`` on any constraint miss so the caller falls through to
        the owned native BF16 quality bridge. Eligibility is resolved by the
        fixed native dispatch; the retired ``PRISMAQUANT_CB_PREFILL`` selector
        is not part of the serving contract.

        WHAT IT REMOVES. The native quality bridge expands each expert chunk's
        packed CB rows into a bounded BF16 tile and runs Gridbook's owned
        grouped CUTLASS GEMM over it. ``cb_fused_prefill_mm_scaled`` instead
        decodes the packed rows INSIDE the CUTLASS prologue, so the transient
        tile never exists.

        ROUTING / SYNC DESIGN. Per (token, expert) PAIR p = t*top_k + j:
        ``pair_expert = topk_ids.reshape(-1)``. A STABLE argsort by expert makes
        each expert's rows a contiguous, token-major segment (the loop path's
        ``torch.where(topk_ids == e)`` order, so per-segment GEMMs bit-match the
        loop and only the combine reassociates). Segment boundaries are
        ``cumsum(bincount(pair_expert, minlength=E))`` — an [E+1] tensor built
        entirely on device. ONE ``.tolist()`` fetches those E+1 offsets; that is
        the path's ONLY device->host sync per layer (the 'batched' path already
        tolerates two, the loop one). Python then slices the sorted row-index
        tensor per expert with no further reads, and zero-row experts are
        skipped on the fetched offsets alone. NOT capture-safe (host-read
        control flow) — same class as 'batched'; prefill is eager
        (FULL_DECODE_ONLY), so that costs nothing today.

        NUMERICS. Activations: the registered native per-token FP8 quantizer,
        bit-equivalent to ``codec.fp8_dynamic_act_qdq``, runs on the
        module input AND the intermediate, gathered per expert (QDQ is a
        per-row op, so gather-after-quant == quant-after-gather bit-exactly).
        Weights: the fused prologue's decode is bit-exact vs the expander.
        Scales: the kernel's fp32 EVT epilogue applies per-token a_scales and
        per-channel b_scales and rounds ONCE to bf16 — the same rounding order
        as cutlass_scaled_mm and as the stock W8A8 kernel. Only the GEMM
        accumulation and the cross-expert combine reassociate
        (REASSOCIATION-CLASS, the suite's 2e-2 contract).

        MIN-M. ``check_fused_inputs`` imposes NO minimum M: M is a runtime GEMM
        extent and CUTLASS covers a short expert with one partial 128-row tile.
        No padding is needed (and none is done).
        """
        if not self._gf_ok(layer):
            return None
        from .cuda_ext import get_fused_ext
        fext = get_fused_ext()

        E = layer._cb_E
        T = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden                        # w13 in / w2 out
        inter = layer._cb_inter
        N1 = 2 * inter                               # w13 out (gate_up)
        d = N1 // 2
        lut = self._stock_cb_flat_fp8(layer)
        kb = self.k

        out = torch.zeros_like(x)

        # ---- routing (all device-side up to the single sync) ---------------
        pair_expert = topk_ids.reshape(-1).to(torch.long)          # [P]
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)                              # [P]
        order = torch.argsort(pair_expert, stable=True)
        ptok_sorted = pair_token[order]                            # [P]
        pw_sorted = topk_weights.reshape(-1)[order]                # [P]
        counts = torch.bincount(pair_expert, minlength=E)          # [E]
        bounds_t = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        bounds = bounds_t.tolist()                                 # THE one sync

        # ---- input activation QDQ, ONCE over all token rows ----------------
        a1, a1s = native_fp8_quant(x)
        a1s = a1s.reshape(-1).to(torch.float32)

        w13q = layer.w13_cb_qweight.data
        w2q = layer.w2_cb_qweight.data
        w13s = layer.w13_weight_scale
        w2s = layer.w2_weight_scale

        for e in range(E):
            p0, p1 = bounds[e], bounds[e + 1]
            if p1 == p0:                              # zero-row expert: skip
                continue
            rows = ptok_sorted[p0:p1]
            ae = a1.index_select(0, rows).contiguous()             # [m_e, Kh]
            ase = a1s.index_select(0, rows).contiguous()           # [m_e]

            # stage 1: gate_up = ae @ W13[e]^T (decode in prologue)
            gate_up = fext.cb_fused_prefill_mm_scaled(
                ae, w13q[e], lut, ase,
                w13s[e].reshape(-1).to(torch.float32).contiguous(),
                N1, Kh, kb)                                        # [m_e, N1]
            del ae, ase
            a = torch.empty((gate_up.shape[0], d), dtype=gate_up.dtype,
                            device=dev)
            native_moe_activation(act, a, gate_up)                # silu(g)*u
            del gate_up

            # Intermediate QDQ uses the same native per-token quantizer.
            a2, a2s = native_fp8_quant(a)
            del a

            # stage 2: y = a2 @ W2[e]^T
            y = fext.cb_fused_prefill_mm_scaled(
                a2.contiguous(), w2q[e], lut,
                a2s.reshape(-1).to(torch.float32).contiguous(),
                w2s[e].reshape(-1).to(torch.float32).contiguous(),
                Kh, inter, kb)                                     # [m_e, Kh]
            del a2, a2s
            y = y * pw_sorted[p0:p1, None].to(y.dtype)             # router w
            out.index_add_(0, rows, y.to(out.dtype))
            del y
        return out

    # -- prefill: grouped FUSED round 2 (one launch per stage) --------------
    def _gf2_ok(self, layer) -> bool:
        """Eligibility for ROUND 2. Everything R1 requires (``_gf_ok`` — fp8-CB,
        the KBits rung, K%256, unpadded 16B-aligned rows) PLUS the grouped
        binding itself, which ships independently of ``cb_fused_prefill_mm_scaled``:
        an older extension build must fall back to R1, not crash."""
        ok = getattr(layer, "_cb_gf2_ok", None)
        if ok is not None:
            return ok
        ok = self._gf_ok(layer)
        if ok:
            from .cuda_ext import get_fused_ext
            fext = get_fused_ext()
            ok = (fext is not None
                  and hasattr(fext, "cb_fused_moe_grouped")
                  and hasattr(fext, "cb_fused_moe_tile_m"))
        layer._cb_gf2_ok = bool(ok)
        return layer._cb_gf2_ok

    def _gf2_tile_sizes(self, layer) -> list[int]:
        """The TileM values the grouped kernel is actually COMPILED for on this
        build, ascending. Always enumerated from the extension — a (TileM,
        k_bits) pair can be uncompilable (shared-memory limit), so a hardcoded
        list would hand the kernel a tile it cannot serve.

        Precedence: the per-rung query (the only one that knows this layer's
        k_bits), then the general one, then the back-compat single default. An
        extension build predating the newer symbols therefore degrades to
        exactly the one tile it has always supported. Cached per layer (the
        answer is a property of the build + the layer's rung)."""
        sizes = getattr(layer, "_cb_gf2_tiles", None)
        if sizes is not None:
            return sizes
        from .cuda_ext import get_fused_ext
        fext = get_fused_ext()
        sizes = []
        if fext is not None:
            for getter, args in (
                    ("cb_fused_moe_tile_sizes_for_kbits", (self.k,)),
                    ("cb_fused_moe_tile_sizes", ()),
                    ("cb_fused_moe_tile_m", ())):
                fn = getattr(fext, getter, None)
                if fn is None:
                    continue
                try:
                    got = fn(*args)
                except Exception:  # noqa: BLE001 — treat as "not offered"
                    continue
                got = [got] if isinstance(got, int) else list(got)
                sizes = sorted({int(s) for s in got if int(s) > 0})
                if sizes:
                    break
        layer._cb_gf2_tiles = sizes
        return sizes

    def _grouped_call(self, fext, args, tile_m: int):
        """``cb_fused_moe_grouped`` with an explicit TileM, tolerating an
        extension build whose binding predates the ``tile_m`` parameter: that
        build only ever has ONE tile, so dropping the argument is correct when
        it equals the default and a fall-through (``None``) otherwise."""
        if getattr(self, "_cb_grouped_tile_arg", True):
            try:
                return fext.cb_fused_moe_grouped(*args, tile_m)
            except TypeError:
                self._cb_grouped_tile_arg = False
        if tile_m != int(fext.cb_fused_moe_tile_m()):
            return None
        return fext.cb_fused_moe_grouped(*args)

    def _apply_prefill_grouped_fused_v2(self, layer, x, topk_weights, topk_ids,
                                        act, *, tile_m=None):
        """ROUND 2 of the native FP8-CB grouped fused prefill.

        The fixed native dispatch tries this grouped binding when eligible.
        Any constraint miss falls back to R1 and then to the owned native BF16
        quality bridge; no retired runtime prefill selector is consulted.

        WHAT IT REMOVES. R1 already decodes CB rows inside the CUTLASS prologue,
        so the only structural cost it left is the HOST LOOP: 2*E dense kernel
        launches per layer (~5-10 ms/layer at E=256, comparable to the GEMM work
        itself). R2 issues ONE ``cb_fused_moe_grouped`` per projection stage over
        the whole routed collective; the kernel picks each M-tile's B operand
        from a per-tile expert id.

        PADDED ROUTING. The kernel's tile is uniform, so each expert's rows must
        start on a TileM boundary — hence a PADDED gather rather than R1's exact
        segments. ``cb_grouped_pad_routing`` (see its docstring for the
        capacity-bound proof and the no-host-read construction) returns per-tile
        expert ids and, per padded row, the index into the STABLE-argsorted pair
        array. Stable sort keeps each expert's rows token-major, so a padded
        segment's GEMM sees the same row order as ``_apply_prefill_loop``'s
        ``torch.where(topk_ids == e)`` and only the combine reassociates.
        ``expert_ids`` is built once and REUSED across both stages: the row
        layout is a property of the routing, not of the projection.

        PADDING IS INERT BY CONSTRUCTION, not by cancellation. Padding rows
        gather activation zeros (scale 1.0 — any finite value works since the
        row is zero, but 1.0 avoids a degenerate 0*0 scale in the intermediate
        per-token QDQ), and the combine scatters them into a THROWAWAY row T of
        a ``[T+1, hidden]`` accumulator that is then sliced off. That is
        stronger than zeroing their router weight: it holds even if a padding
        row's output were non-finite, so the guarantee does not depend on the
        kernel's behaviour on an all-zero tile.

        TRIM. ``PRISMAQUANT_CB_GROUPED_TRIM`` (default "1") spends ONE
        ``.item()`` on the real block total and slices the padded collective
        down to it, so no wasted tile is ever launched. R1 already spends one
        sync per layer, so the default is strictly no worse. Setting it to "0"
        keeps the full static capacity — up to E wasted tiles per stage, but the
        path then contains NO host read of device data at all, which is the
        prerequisite for future CUDA-graph capture of prefill (today prefill is
        eager/FULL_DECODE_ONLY, so the trim is the better default).

        NUMERICS. Identical contract to R1: the registered native per-token
        FP8 quantizer runs on the module input AND
        the intermediate (a per-row op, so gather-after-quant ==
        quant-after-gather bit-exactly), bit-exact prologue weight decode, and
        the same fp32 EVT ``bf16_rn(b_scale * (a_scale * acc))`` rounding order.

        TILE M. ``tile_m`` selects among the tile sizes the kernel was compiled
        for (``_gf2_tile_sizes``); ``None`` means the kernel's own default. It is
        a PERFORMANCE knob with a real tradeoff — a larger tile amortises the
        per-tile B decode over more rows but wastes more padded rows on short
        experts — which is why 'auto' measures it rather than fixing it. Nothing
        else in this method is tile-specific: the capacity bound ``P//tile_m + E``
        is proved for arbitrary tile_m (see ``cb_grouped_pad_routing``), and the
        combine is indexed by ``dest``/``row_src``, whose length is
        ``cap_blocks*tile_m`` by construction — so both generalise unchanged.
        """
        if not self._gf2_ok(layer):
            return None
        from .cuda_ext import get_fused_ext
        fext = get_fused_ext()

        E = layer._cb_E
        T = x.shape[0]
        top_k = topk_ids.shape[-1]
        dev = x.device
        Kh = layer._cb_hidden                        # w13 in / w2 out
        inter = layer._cb_inter
        N1 = 2 * inter                               # w13 out (gate_up)
        d = N1 // 2
        lut = self._stock_cb_flat_fp8(layer)
        kb = self.k
        # Never hardcode a tile: the kernel owns which TileM it compiled.
        tile_m = int(fext.cb_fused_moe_tile_m() if tile_m is None else tile_m)
        if tile_m <= 0:
            return None

        # ---- routing (device-side; static shapes) --------------------------
        pair_expert = topk_ids.reshape(-1).to(torch.long)          # [P]
        pair_token = torch.arange(T, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)                              # [P]
        order = torch.argsort(pair_expert, stable=True)            # STABLE
        ptok_sorted = pair_token[order]
        pw_sorted = topk_weights.reshape(-1)[order].to(torch.float32)

        expert_ids, row_src, is_pad, n_blocks = cb_grouped_pad_routing(
            topk_ids, E, tile_m)

        if os.environ.get("PRISMAQUANT_CB_GROUPED_TRIM", "1") == "1":
            # Kept intentionally: a host-read-free replacement must either
            # launch the full static-capacity tail (a performance change) or
            # teach the grouped kernel to consume n_blocks on device. A tensor
            # slice bound still performs this same host conversion implicitly.
            nb = int(n_blocks.item())                # THE one sync (optional)
            expert_ids = expert_ids[:nb].contiguous()
            row_src = row_src[:nb * tile_m]
            is_pad = is_pad[:nb * tile_m]

        # ---- input activation QDQ, ONCE over all token rows ----------------
        a1, a1s = native_fp8_quant(x)
        a1s = a1s.reshape(-1).to(torch.float32)

        # ONE index vector does both the gather and the scatter: real rows map
        # to their token, padding rows to the appended zero row / throwaway
        # output row T. Gathering the zeros (rather than masking an fp8 tensor
        # after the fact) keeps every op on this path a plain index_select.
        rows = ptok_sorted.index_select(0, row_src)                 # [Mp]
        dest = torch.where(is_pad, torch.full_like(rows, T), rows)
        a1x = torch.cat([a1, a1.new_zeros((1, Kh))])
        # Padding scale 1.0 (see docstring): the row is zero either way, but a
        # zero scale would make the intermediate QDQ's per-row amax degenerate.
        a1sx = torch.cat([a1s, a1s.new_ones(1)])
        a_pad = a1x.index_select(0, dest).contiguous()
        as_pad = a1sx.index_select(0, dest).contiguous()

        w13s = layer.w13_weight_scale.reshape(E, N1).to(
            torch.float32).contiguous()
        w2s = layer.w2_weight_scale.reshape(E, Kh).to(
            torch.float32).contiguous()

        # stage 1: gate_up = a_pad @ W13[expert_of_tile]^T
        gate_up = self._grouped_call(
            fext, (a_pad, layer.w13_cb_qweight.data, lut, as_pad, w13s,
                   expert_ids, N1, Kh, kb), tile_m)                # [Mp, N1]
        del a_pad, as_pad
        if gate_up is None:                          # binding has no TileM knob
            return None
        a = torch.empty((gate_up.shape[0], d), dtype=gate_up.dtype, device=dev)
        native_moe_activation(act, a, gate_up)                    # silu(g)*u
        del gate_up

        a2, a2s = native_fp8_quant(a)
        del a

        # stage 2: y = a2 @ W2[expert_of_tile]^T — SAME expert_ids, the row
        # layout is a property of the routing, not of the projection.
        y = self._grouped_call(
            fext, (a2.contiguous(), layer.w2_cb_qweight.data, lut,
                   a2s.reshape(-1).to(torch.float32).contiguous(), w2s,
                   expert_ids, Kh, inter, kb), tile_m)             # [Mp, Kh]
        del a2, a2s
        if y is None:
            return None

        # Padding rows carry pair 0's router weight; harmless, because they are
        # scattered into throwaway row T, which is sliced off.
        pw_pad = pw_sorted.index_select(0, row_src)
        y = y * pw_pad[:, None].to(y.dtype)
        out = torch.zeros((T + 1, Kh), dtype=x.dtype, device=dev)
        out.index_add_(0, dest, y.to(out.dtype))
        return out[:T]


    def _stock_cb_flat_fp8(self, layer) -> torch.Tensor:
        """The per-layer codebook re-encoded to E4M3 bytes for the fp8-direct
        expand. Cached on the layer (also built by the CUDA-decode gate); built
        once, before capture. "Every CB value is on the e4m3 grid — lossless"
        used to be a comment here; it is now a check (see
        ``_assert_cast_lossless``), because learned tables must satisfy the
        same serving-grid contract as lattice tables.
        (Was a ``@staticmethod``; every call site already went through ``self``,
        and the check wants the layer prefix for its message.)"""
        cb = getattr(layer, "_cb_flat_fp8", None)
        if cb is None:
            cb = codec.flat_codebook_fp8(layer._cb_flat, self.prefix)
            layer._cb_flat_fp8 = cb
        return cb

    # -- grouped CUDA decode path -------------------------------------------
    def _cuda_moe_ok(self, layer) -> bool:
        ok = getattr(layer, "_cb_moe_cuda_ok", None)
        if ok is None:
            # Grouped CUDA GEMV support: fp8-CB v1 (n_sub=4, per-(expert,out)
            # fp32 scale) and fp4-CB two-tier v2 (n_sub=2, scale composed
            # in-kernel from the packed 9-byte section). fp4-v1 (bare e4m3
            # plane) has no grouped path yet.
            fmt_ok = ((self.n_sub == 4 and not self.is_fp4)
                      or (self.n_sub in (1, 2) and self.is_fp4
                          and self.is_v2))
            ok = fmt_ok
            if ok:
                from .cuda_ext import get_ext
                ok = get_ext() is not None
                if not ok:
                    cls = PrismaQuantCBMoEMethod
                    if not cls._DECODE_DISABLED_LOGGED:
                        print("[prismaquant-cb] ERROR: grouped CUDA decode "
                              "extension unavailable; native-only routed "
                              "execution will fail closed.",
                              file=sys.stderr, flush=True)
                        cls._DECODE_DISABLED_LOGGED = True
            if ok and not self.is_fp4:
                # Normally materialized during process_weights_after_loading;
                # retain this defensive check for callers that construct a
                # layer through an alternate/test load path.
                self._stock_cb_flat_fp8(layer)
            layer._cb_moe_cuda_ok = ok
            if ok and not PrismaQuantCBMoEMethod._DECODE_ENGAGED_LOGGED:
                # Positive counterpart to the warning above: one line per
                # process proving WHICH decode path a run actually engaged.
                print("[prismaquant-cb] decode=grouped-cuda", flush=True)
                PrismaQuantCBMoEMethod._DECODE_ENGAGED_LOGGED = True
        return ok

    def _apply_grouped_decode(self, layer, x, topk_weights, topk_ids, act):
        """One grouped GEMV launch per projection over all routed
        (token, expert) pairs, numerics-matched to the per-expert loop:
        per-token activation QDQ on the module input AND the intermediate
        (fp8 dynamic for fp8-CB v1, fp4 group-16 RTN for fp4-CB v2), weights
        bf16(val*scale), fp32-accum GEMVs, per-add bf16 combine in the loop's
        expert-ascending order."""
        from . import ops as pq_ops
        T = x.shape[0]
        topk = topk_ids.shape[-1]
        # Pairs sorted (token-major, expert-ascending) — the loop's
        # index_add_ order per token. All GPU, no host syncs.
        ids_sorted, order = torch.sort(topk_ids.to(torch.int32), dim=-1)
        w_sorted = torch.gather(topk_weights, -1, order.to(torch.int64))
        pair_expert = ids_sorted.reshape(-1).contiguous()
        pair_xrow = (torch.arange(T, device=x.device, dtype=torch.int32)
                     .repeat_interleave(topk))
        # Match the loop's bf16-rounded router weight before the multiply.
        pair_w = (w_sorted.reshape(-1).to(torch.bfloat16)
                  .to(torch.float32).contiguous())
        tok_start = (torch.arange(T + 1, device=x.device, dtype=torch.int32)
                     * topk)
        pair_self = torch.arange(pair_expert.numel(), device=x.device,
                                 dtype=torch.int32)

        if self.is_fp4:                                  # fp4-CB two-tier v2
            # Activation QDQ (fp4 group-16 RTN) stays OUTSIDE the kernel,
            # bit-identical to the loop's codec.fp4_group16_act_qdq; the kernel
            # composes the two-tier weight scale in-register from the packed
            # 9-byte section + the resident (256,16) compose table.
            xq = fp4_act_qdq_or_codec(x)
            # CB-GEMV-v2 dispatch. `_cb_use_v2_w13` is a python bool fixed at
            # load (process_weights_after_loading), so this `if` resolves at
            # trace time and both arms are pure device work — nothing here
            # reads the env, syncs, or branches on a tensor. Layers built by a
            # fixture that never ran process_weights_after_loading default to
            # the inherited kernel, i.e. to today's behaviour.
            # v2 takes (…, k, type_size, rpb, dict_mode) — no n_sub (it is
            # product-mode only) — with rpb=0 / dict_mode=0 selecting the
            # kernel's measured auto policies. Like the inherited extension,
            # the C++ launcher reads PRISMAQUANT_CB_DECODE_CONTRACT on every
            # call, so mixed dispatch cannot cache two different contracts.
            if getattr(layer, "_cb_use_v2_w13", False):
                gate_up = pq_ops.cb_moe_gemv_v2(
                    xq, layer.w13_cb_qweight.data, layer._cb_flat,
                    layer._cb_compose, pair_expert, pair_xrow,
                    self.k, self.type_size, 0, 0)
            else:
                gate_up = pq_ops.cb_moe_gemv_fp4_v2(
                    xq, layer.w13_cb_qweight.data, layer._cb_flat,
                    layer._cb_compose, pair_expert, pair_xrow,
                    self.k, self.n_sub, self.type_size)  # (P, 2*inter)
        else:                                            # fp8-CB v1
            xq = pq_ops.fp8_act_qdq(x.to(torch.bfloat16))
            gate_up = pq_ops.cb_moe_gemv_fp8(
                xq, layer.w13_cb_qweight.data, layer._cb_flat_fp8,
                layer.w13_weight_scale.data, pair_expert, pair_xrow,
                self.k, self.n_sub, self.type_size)      # (P, 2*inter)

        d = gate_up.shape[-1] // 2
        a = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                        device=gate_up.device)
        native_moe_activation(act, a, gate_up)

        if self.is_fp4:
            aq = fp4_act_qdq_or_codec(a)
            # v2 dispatch, w2 stack — decided independently of w13 (different
            # K, and the occupancy predicate is a function of K).
            if getattr(layer, "_cb_use_v2_w2", False):
                y_down = pq_ops.cb_moe_gemv_v2(
                    aq, layer.w2_cb_qweight.data, layer._cb_flat,
                    layer._cb_compose, pair_expert, pair_self,
                    self.k, self.type_size, 0, 0)
            else:
                y_down = pq_ops.cb_moe_gemv_fp4_v2(
                    aq, layer.w2_cb_qweight.data, layer._cb_flat,
                    layer._cb_compose, pair_expert, pair_self,
                    self.k, self.n_sub, self.type_size)  # (P, hidden)
        else:
            aq = pq_ops.fp8_act_qdq(a)
            y_down = pq_ops.cb_moe_gemv_fp8(
                aq, layer.w2_cb_qweight.data, layer._cb_flat_fp8,
                layer.w2_weight_scale.data, pair_expert, pair_self,
                self.k, self.n_sub, self.type_size)      # (P, hidden)
        return pq_ops.cb_moe_combine(y_down, pair_w, tok_start, T)
