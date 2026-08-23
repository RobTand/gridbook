"""Raw-resident W8A16 serving for DeepSeek block-128 source FP8.

The source-passthrough declaration is a weight-storage contract, not an
activation-quantization request.  This method therefore keeps exactly the
checkpoint's E4M3 ``[N,K]`` value plane and UE8M0 block-128 scale plane
resident while accepting BF16 activations unchanged.

Execution is native and fail-closed:

* decode (``M <= 8``) streams both raw planes through Gridbook's CUDA GEMV;
* larger inputs expand one caller-scoped BF16 tile and immediately consume it
  through Gridbook's owned CUTLASS grouped-BF16 bridge;
* the entire size dispatch is an opaque custom op, so Dynamo never substitutes
  Triton or bakes the prefill arm into captured decode.

There is no persistent BF16 expansion, activation QDQ, ``F.linear``,
``torch.bmm``, or CPU serving fallback.  Direct per-32 MXFP8 is intentionally
not accepted here; it remains the separate W8A8 ``Mxfp8DenseLinearMethod``.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import torch

from .mxfp8 import DS_BLOCK

__all__ = [
    "WIRE_FP8_BLOCK128",
    "build_fp8_source_w8a16_method",
]


WIRE_FP8_BLOCK128 = "fp8_e4m3_ue8m0_block128"

_READY_ATTR = "_gridbook_fp8_source_w8a16_ready"
_SHARD_ATTR = "_fp8_source_shard_plan"
_READY_ABI = 1
_DECODE_MAX_M = 8
_GEMM_ALIGNMENT = 8
_DSV4_BMM_GROUPS = 8
_DSV4_BMM_ROWS = 1024
_DSV4_BMM_K = 4096
_DSV4_BMM_QUALIFIED_SHARD_DEGREE = 1

#: Both parallel axes must land on a whole source block, because vLLM narrows
#: the UE8M0 plane with the SAME shard arithmetic it uses for the value plane
#: (``BlockQuantScaleParameter`` is ``ModelWeightParameter``'s narrowing over
#: the block grid, with ``adjust_block_scale_shard`` converting element counts
#: to block counts by CEIL division).  The narrow start is therefore
#: ``rank * ceil(local / 128)``, which indexes the full plane's block grid
#: correctly if and only if the local extent is a whole multiple of 128 —
#: otherwise ranks silently read overlapping or shifted blocks with no error.
_TP_SHARD_ALIGNMENT = 128

if _TP_SHARD_ALIGNMENT != DS_BLOCK:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "source-FP8 W8A16 shard alignment must be the source block size")

def _qualified_bmm_geometries():
    """Grouped-BMM geometries a kernel qualification stands behind.

    Each entry is ``(groups, rows_per_group, K, shard_degree)``.  Sharding a
    grouped plane column-wise divides the kernel's group count, so a sharded
    geometry is a NEW qualification -- measure it, then add its tuple here --
    rather than something a shard law can grant.  Built from the constants at
    call time so a test may pin a small geometry by patching them.
    """

    return frozenset({
        (_DSV4_BMM_GROUPS, _DSV4_BMM_ROWS, _DSV4_BMM_K,
         _DSV4_BMM_QUALIFIED_SHARD_DEGREE),
    })


class ShardAlignmentError(ValueError):
    """A tensor-parallel shard this lane refuses to construct.

    Raised at weight CONSTRUCTION, before vLLM copies a single byte: a
    misaligned block-scale narrow is silently wrong rather than loud, so the
    refusal has to precede the load.  Callers may catch it as ``ValueError``.
    """


class ShardPlan(NamedTuple):
    """The structural shard degrees of one Linear, per parallel axis.

    Derived from ``create_weights``' own arguments, never from
    ``layer.tp_size``: vLLM's ``LinearBase`` stamps the world size onto EVERY
    layer, including ``ReplicatedLinear`` and ``disable_tp=True`` merged
    planes, so ``tp_size`` cannot distinguish a sharded plane from a
    replicated one.  DeepSeek-V4 has 43 replicated fused ``wqa_wkv`` planes and
    21 replicated indexer ``wq_b`` planes that report ``tp_size = N``.
    """

    row_degree: int
    col_degree: int

    @property
    def degree(self) -> int:
        return max(self.row_degree, self.col_degree)


def _axis_degree(full, local, *, axis: str, prefix: str) -> int:
    """Whole-number shard degree along one axis, or a structured refusal."""

    full = int(full)
    local = int(local)
    where = f" at {prefix!r}" if prefix else ""
    if full <= 0 or local <= 0:
        raise ShardAlignmentError(
            f"source-FP8 W8A16{where} needs a positive {axis} extent; got "
            f"full={full}, per-rank={local}")
    if full % local != 0:
        raise ShardAlignmentError(
            f"source-FP8 W8A16{where} refuses an uneven {axis} partition: "
            f"the full extent {full} is not a whole multiple of the per-rank "
            f"extent {local}. Gridbook serves this lane only where every rank "
            "holds the same shape; serve this model at a tensor-parallel size "
            "that divides the layer evenly.")
    return full // local


def _resolve_shard_plan(*, input_size, input_size_per_partition,
                        output_size, output_partition_sizes,
                        prefix: str) -> ShardPlan:
    return ShardPlan(
        row_degree=_axis_degree(
            input_size, input_size_per_partition,
            axis="input (row-parallel)", prefix=prefix),
        col_degree=_axis_degree(
            output_size, sum(int(w) for w in output_partition_sizes),
            axis="output (column-parallel)", prefix=prefix),
    )


def _require_shard_alignment(plan: ShardPlan, *, input_size_per_partition: int,
                             output_partition_sizes: Sequence[int],
                             prefix: str) -> None:
    """Refuse any shard whose block-scale narrowing would be misindexed.

    The laws, one per axis, plus the merged-plane law that makes the per-role
    block offsets tile the local scale plane exactly.
    """

    where = f" at {prefix!r}" if prefix else ""
    block = _TP_SHARD_ALIGNMENT
    if plan.row_degree > 1 and plan.col_degree > 1:
        raise ShardAlignmentError(
            f"source-FP8 W8A16{where} refuses a plane sharded on BOTH axes "
            f"(row degree {plan.row_degree}, column degree "
            f"{plan.col_degree}); this lane is qualified for one-dimensional "
            "tensor parallelism only.")
    if plan.row_degree > 1:
        local_k = int(input_size_per_partition)
        if local_k % block != 0:
            raise ShardAlignmentError(
                f"source-FP8 W8A16{where} refuses a row-parallel shard of "
                f"degree {plan.row_degree}: the per-rank K extent {local_k} is "
                f"not a multiple of the {block}-element source block, so "
                "vLLM's block-scale narrow would read shifted scale columns. "
                f"Serve at a tensor-parallel size whose per-rank K is a "
                f"multiple of {block}.")
    if plan.col_degree > 1:
        widths = [int(w) for w in output_partition_sizes]
        bad = [w for w in widths if w % block != 0]
        if bad:
            raise ShardAlignmentError(
                f"source-FP8 W8A16{where} refuses a column-parallel shard of "
                f"degree {plan.col_degree}: per-rank output width(s) "
                f"{bad} are not multiples of the {block}-element source "
                "block, so vLLM's block-scale narrow would read shifted "
                "scale rows. On a merged plane EVERY fused role's per-rank "
                f"width must be a multiple of {block}. Serve at a "
                "tensor-parallel size that keeps every role aligned.")
        # The merged-plane offsets are converted to block units role by role,
        # so exact tiling of the local scale plane is the thing to verify.
        rows = sum(w // block for w in widths)
        local_n = sum(widths)
        if rows != -(-local_n // block):
            raise ShardAlignmentError(  # pragma: no cover - implied by above
                f"source-FP8 W8A16{where} refuses a merged column shard whose "
                f"per-role block rows {rows} do not tile the local scale "
                f"plane ({-(-local_n // block)} rows)")


def _checked_source_loader(weight_loader, *, expected_dtype: torch.dtype,
                           plane: str):
    """Reject a mis-typed checkpoint tensor before vLLM can cast-copy it.

    vLLM's default, fused, and merged loaders all receive the destination
    parameter followed by ``loaded_weight``; fused loaders may add a positional
    shard id, while merged loaders may pass it by keyword.  Keep that call
    shape opaque and delegate it unchanged after checking the source tensor.
    """

    if weight_loader is None:
        return None

    def checked_loader(*args, **kwargs):
        if len(args) >= 2:
            loaded_weight = args[1]
        elif "loaded_weight" in kwargs:
            loaded_weight = kwargs["loaded_weight"]
        else:
            raise TypeError(
                f"source-FP8 W8A16 {plane} loader requires the checkpoint "
                "tensor as loaded_weight")
        if not isinstance(loaded_weight, torch.Tensor):
            raise TypeError(
                f"source-FP8 W8A16 {plane} checkpoint value must be a "
                f"torch.Tensor before vLLM loader delegation, got "
                f"{type(loaded_weight).__name__}")
        if loaded_weight.dtype != expected_dtype:
            raise TypeError(
                f"source-FP8 W8A16 {plane} checkpoint tensor must be "
                f"exactly {expected_dtype} before vLLM loader delegation, "
                f"got {loaded_weight.dtype}")
        return weight_loader(*args, **kwargs)

    return checked_loader


def _require_source_cuda(tensor: torch.Tensor) -> None:
    """Fail closed off CUDA; kept as the narrow CPU policy-test seam."""

    if tensor.device.type != "cuda":
        raise RuntimeError(
            "source-FP8 W8A16 model loading requires CUDA; Gridbook has no "
            "CPU or interpreted-kernel fallback")


def _build_method_class():
    """Build the vLLM-bound class lazily so policy imports stay lightweight."""

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    class Fp8SourceW8A16LinearMethod(LinearMethodBase):
        """Raw block128 FP8 weights with unquantized BF16 activations."""

        def __init__(self) -> None:
            self._wire = WIRE_FP8_BLOCK128

        def create_weights(self, layer, input_size_per_partition,
                           output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            del params_dtype
            out_size = int(sum(output_partition_sizes))
            in_size = int(input_size_per_partition)
            if out_size <= 0 or in_size <= 0:
                raise ValueError(
                    "source-FP8 W8A16 needs positive N and K, got "
                    f"N={out_size}, K={in_size}")

            # Tensor-parallel legality FIRST: a misaligned block-scale narrow
            # is silent, so it must be refused before any parameter exists for
            # vLLM's loader to copy into.
            prefix = str(getattr(layer, "prefix", "") or "")
            plan = _resolve_shard_plan(
                input_size=input_size,
                input_size_per_partition=in_size,
                output_size=output_size,
                output_partition_sizes=output_partition_sizes,
                prefix=prefix,
            )
            _require_shard_alignment(
                plan,
                input_size_per_partition=in_size,
                output_partition_sizes=output_partition_sizes,
                prefix=prefix,
            )
            setattr(layer, _SHARD_ATTR, plan)

            weight_loader = extra_weight_attrs.get("weight_loader")
            value_loader = _checked_source_loader(
                weight_loader,
                expected_dtype=torch.float8_e4m3fn,
                plane="value-plane",
            )
            scale_loader = _checked_source_loader(
                weight_loader,
                expected_dtype=torch.float8_e8m0fnu,
                plane="scale-plane",
            )
            weight = ModelWeightParameter(
                data=torch.empty(out_size, in_size,
                                 dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=value_loader,
            )
            layer.register_parameter("weight", weight)

            # vLLM's fused-shard loader needs the source block geometry to
            # place each physical role's scale rows at the correct offset.
            layer.weight_block_size = [DS_BLOCK, DS_BLOCK]
            scale = BlockQuantScaleParameter(
                data=torch.empty(
                    (out_size + DS_BLOCK - 1) // DS_BLOCK,
                    (in_size + DS_BLOCK - 1) // DS_BLOCK,
                    dtype=torch.float8_e8m0fnu,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=scale_loader,
            )
            # DeepSeek's checkpoint spelling.  The loader copies the UE8M0
            # bytes verbatim; "inv" is a name, not a numeric transformation.
            layer.register_parameter("weight_scale_inv", scale)

        def process_weights_after_loading(self, layer) -> None:
            if getattr(layer, _READY_ATTR, None) is not None:
                raise RuntimeError(
                    "source-FP8 W8A16 weights were finalized more than once")

            q = layer.weight.data
            scales = layer.weight_scale_inv.data
            if q.dtype != torch.float8_e4m3fn or q.ndim != 2:
                raise TypeError(
                    "source-FP8 W8A16 needs a 2-D float8_e4m3fn value "
                    f"plane, got dtype={q.dtype}, shape={tuple(q.shape)}")
            if scales.dtype != torch.float8_e8m0fnu:
                raise TypeError(
                    "source-FP8 W8A16 needs a float8_e8m0fnu scale plane, "
                    f"got {scales.dtype}")
            if scales.ndim != 2 or scales.device != q.device:
                raise ValueError(
                    "source-FP8 W8A16 value and scale planes must be 2-D "
                    f"on one device; got {q.device} and {scales.device}")
            _require_source_cuda(q)

            n, k = int(q.shape[0]), int(q.shape[1])
            expected_scales = (
                (n + DS_BLOCK - 1) // DS_BLOCK,
                (k + DS_BLOCK - 1) // DS_BLOCK,
            )
            if tuple(scales.shape) != expected_scales:
                raise ValueError(
                    "source-FP8 W8A16 block128 scale shape mismatch: got "
                    f"{tuple(scales.shape)}, expected {expected_scales} for "
                    f"weight {(n, k)}")
            if k % _GEMM_ALIGNMENT != 0:
                raise ValueError(
                    "source-FP8 W8A16 needs K divisible by the native BF16 "
                    f"bridge alignment {_GEMM_ALIGNMENT}, got {k}")

            is_bmm = bool(getattr(layer, "is_bmm", False))
            groups = 1
            rows = n
            tp_size = int(getattr(layer, "tp_size", 1))
            plan = getattr(layer, _SHARD_ATTR, None)
            if plan is None:
                # Nothing structural to read: this layer's parameters were not
                # built by THIS method.  Refuse above one rank rather than
                # guess a shard degree from tp_size, which is stamped on
                # replicated layers too.
                if tp_size != 1:
                    raise ShardAlignmentError(
                        "source-FP8 W8A16 has no shard plan for this layer "
                        "(its weights were not constructed by this lane's "
                        f"create_weights) and the worker reports TP={tp_size}; "
                        "refusing rather than assuming a shard degree")
                plan = ShardPlan(1, 1)
            shard_degree = plan.degree
            if is_bmm:
                groups = int(getattr(layer, "bmm_batch_size", 0))
                if groups <= 0 or n % groups != 0:
                    raise ValueError(
                        "source-FP8 W8A16 BMM needs a positive group count "
                        f"dividing N; got groups={groups}, N={n}")
                rows = n // groups
                geometry = (groups, rows, k, shard_degree)
                if geometry not in _qualified_bmm_geometries():
                    raise ValueError(
                        "source-FP8 W8A16 BMM is qualified only for grouped "
                        f"geometry G={_DSV4_BMM_GROUPS}, N={_DSV4_BMM_ROWS}, "
                        f"K={_DSV4_BMM_K} at shard degree "
                        f"{_DSV4_BMM_QUALIFIED_SHARD_DEGREE}; got "
                        f"G={groups}, N={rows}, K={k}, shard degree="
                        f"{shard_degree} (TP={tp_size}). Column-sharding a "
                        "grouped plane divides the kernel's group count, "
                        "which is a NEW kernel qualification rather than a "
                        "shard law; qualify the sharded geometry and add it "
                        "to this lane's qualified table.")
            elif shard_degree > 1:
                # The construction-time laws already refused every misaligned
                # shard; re-assert them against the RESIDENT shape so a loader
                # that produced an unexpected local extent cannot slip past.
                block = _TP_SHARD_ALIGNMENT
                if plan.col_degree > 1 and n % block != 0:
                    raise ShardAlignmentError(
                        "source-FP8 W8A16 column shard finalized with "
                        f"N={n}, which is not a multiple of {block}")
                if plan.row_degree > 1 and k % block != 0:
                    raise ShardAlignmentError(
                        "source-FP8 W8A16 row shard finalized with "
                        f"K={k}, which is not a multiple of {block}")
            if rows % _GEMM_ALIGNMENT != 0:
                raise ValueError(
                    "source-FP8 W8A16 needs per-group N divisible by the "
                    f"native BF16 bridge alignment {_GEMM_ALIGNMENT}, got "
                    f"{rows}")

            # Resolve BOTH possible size arms now, outside first forward and
            # graph capture.  Neither helper is allowed a torch/Triton fallback.
            from .cuda_ext import (
                require_bf16_grouped_ext,
                require_fp8_source_w8a16_ext,
            )

            require_fp8_source_w8a16_ext(
                "source-FP8 W8A16 model loading", device=q.device)
            require_bf16_grouped_ext(
                "source-FP8 W8A16 transient prefill")

            # Contiguity may replace a non-contiguous loader view, but never
            # changes the wire representation or creates a second resident
            # plane.  Both raw parameters deliberately remain registered.
            layer.weight.data = q.contiguous()
            layer.weight_scale_inv.data = scales.contiguous()
            q = layer.weight.data
            scales = layer.weight_scale_inv.data
            if q.element_size() != 1 or scales.element_size() != 1:
                raise RuntimeError(
                    "source-FP8 W8A16 resident planes must remain byte-wide")

            layer._fp8_source_N = n
            layer._fp8_source_K = k
            layer._fp8_source_groups = groups
            layer._fp8_source_rows = rows
            layer._fp8_source_shard_degree = shard_degree
            layer._fp8_source_resident_bytes = q.numel() + scales.numel()

            from .ops import register_cb_layer

            layer._fp8_source_layer_id = register_cb_layer(self, layer)

            if is_bmm:
                from .dsv4_woa import (
                    DSV4_FP8_SOURCE_W8A16_BMM_ABI,
                    DSV4_FP8_SOURCE_W8A16_BMM_ATTR,
                    install_dsv4_woa_adapter,
                )

                setattr(layer, DSV4_FP8_SOURCE_W8A16_BMM_ATTR,
                        DSV4_FP8_SOURCE_W8A16_BMM_ABI)
                install_dsv4_woa_adapter()

            setattr(layer, _READY_ATTR, _READY_ABI)

        def apply(self, layer, x: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            if bias is not None:
                raise ValueError(
                    "source-FP8 W8A16 does not serve biased linears")
            if getattr(layer, _READY_ATTR, None) != _READY_ABI:
                raise RuntimeError(
                    "source-FP8 W8A16 weight was not finalized at model load")
            layer_id = getattr(layer, "_fp8_source_layer_id", None)
            if layer_id is None:
                raise RuntimeError(
                    "source-FP8 W8A16 dispatch layer was not registered")
            from .ops import fp8_source_linear_forward

            return fp8_source_linear_forward(x, int(layer_id))

        def _apply_inline(self, layer, x: torch.Tensor) -> torch.Tensor:
            """Run inside the opaque whole-dispatch custom op."""

            if x.dtype != torch.bfloat16:
                raise TypeError(
                    "source-FP8 W8A16 preserves BF16 activations and refuses "
                    f"dtype {x.dtype}")
            q = layer.weight
            scales = layer.weight_scale_inv
            if x.device != q.device or scales.device != q.device:
                raise RuntimeError(
                    "source-FP8 W8A16 activations and raw planes must share "
                    "one CUDA device")
            if not q.is_contiguous() or not scales.is_contiguous():
                raise RuntimeError(
                    "source-FP8 W8A16 resident planes lost contiguity")

            n = int(layer._fp8_source_N)
            k = int(layer._fp8_source_K)
            groups = int(layer._fp8_source_groups)
            rows = int(layer._fp8_source_rows)
            is_bmm = bool(getattr(layer, "is_bmm", False))

            if is_bmm:
                if x.ndim < 2 or int(x.shape[-2]) != groups:
                    raise ValueError(
                        "source-FP8 W8A16 BMM expected [..., groups, K] with "
                        f"groups={groups}, got {tuple(x.shape)}")
                if int(x.shape[-1]) != k:
                    raise ValueError(
                        f"source-FP8 W8A16 BMM input K={x.shape[-1]} does "
                        f"not match weight K={k}")
                outer = tuple(x.shape[:-2])
                x3 = x.reshape(-1, groups, k).contiguous()
            else:
                if x.ndim < 1 or int(x.shape[-1]) != k:
                    raise ValueError(
                        "source-FP8 W8A16 dense input must end in "
                        f"K={k}, got {tuple(x.shape)}")
                outer = tuple(x.shape[:-1])
                x3 = x.reshape(-1, 1, k).contiguous()

            m = int(x3.shape[0])
            if m <= 0:
                raise ValueError("source-FP8 W8A16 does not serve empty inputs")

            from .ops import (
                cb_bf16_grouped_mm,
                fp8_source_expand_bf16,
                fp8_source_gemv,
            )
            from .nvfp4_activation_contract import emit_route

            route = {
                "kind": "dense",
                "shape": (
                    f"M{m}:G{groups}:N{rows}:K{k}"
                    if is_bmm else f"M{m}:N{n}:K{k}"
                ),
                "contract": "bf16_preserved",
                "tile_m": 0,
            }

            if m <= _DECODE_MAX_M:
                emit_route(
                    layer,
                    policy="source_fp8_w8a16_decode",
                    symbol="fp8_source_gemv",
                    state="error",
                    reason="launch did not return",
                    **route,
                )
                flat = fp8_source_gemv(x3, q, scales, groups)
                if is_bmm:
                    result = flat.reshape(*outer, groups, rows)
                else:
                    result = flat.reshape(*outer, n)
                emit_route(
                    layer,
                    policy="source_fp8_w8a16_decode",
                    symbol="fp8_source_gemv",
                    state="served",
                    reason=None,
                    **route,
                )
                return result

            # One bounded caller-scoped transient, consumed immediately by
            # Gridbook's owned CUTLASS bridge and never attached to the layer.
            emit_route(
                layer,
                policy="source_fp8_w8a16_prefill",
                symbol="fp8_source_expand_bf16+cb_bf16_grouped_mm",
                state="error",
                reason="launch chain did not return",
                **route,
            )
            expanded = fp8_source_expand_bf16(q, scales)
            if not is_bmm:
                expert_ends = torch.full(
                    (1,), m, dtype=torch.int32, device=x.device)
                result = cb_bf16_grouped_mm(
                    x3[:, 0, :], expanded.view(1, n, k), expert_ends, 0)
                del expanded
                result = result.reshape(*outer, n)
                emit_route(
                    layer,
                    policy="source_fp8_w8a16_prefill",
                    symbol="fp8_source_expand_bf16+cb_bf16_grouped_mm",
                    state="served",
                    reason=None,
                    **route,
                )
                return result

            # The bridge's row contract is expert/group-major.  Flatten A in
            # that order, run G equal contiguous segments, then restore the
            # caller's [..., G, N_per_group] layout.
            a_group_major = x3.permute(1, 0, 2).contiguous().view(
                groups * m, k)
            weights = expanded.view(groups, rows, k)
            expert_ends = torch.arange(
                1, groups + 1, dtype=torch.int32, device=x.device) * m
            grouped = cb_bf16_grouped_mm(
                a_group_major, weights, expert_ends, 0)
            del expanded
            restored = grouped.view(groups, m, rows).permute(
                1, 0, 2).contiguous()
            result = restored.reshape(*outer, groups, rows)
            emit_route(
                layer,
                policy="source_fp8_w8a16_prefill",
                symbol="fp8_source_expand_bf16+cb_bf16_grouped_mm",
                state="served",
                reason=None,
                **route,
            )
            return result

    return Fp8SourceW8A16LinearMethod


def build_fp8_source_w8a16_method(wire_id: str):
    """Construct the block128-only source-FP8 W8A16 serving method."""

    if wire_id != WIRE_FP8_BLOCK128:
        raise ValueError(
            "Fp8SourceW8A16LinearMethod accepts only "
            f"{WIRE_FP8_BLOCK128!r}, got {wire_id!r}; direct g32 MXFP8 stays "
            "on Mxfp8DenseLinearMethod (W8A8)")
    return _build_method_class()()
