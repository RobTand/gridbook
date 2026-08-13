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

from typing import Optional

import torch

from .mxfp8 import DS_BLOCK

__all__ = [
    "WIRE_FP8_BLOCK128",
    "Fp8SourceW8A16LinearMethod",
    "build_fp8_source_w8a16_method",
]


WIRE_FP8_BLOCK128 = "fp8_e4m3_ue8m0_block128"

_READY_ATTR = "_gridbook_fp8_source_w8a16_ready"
_READY_ABI = 1
_DECODE_MAX_M = 8
_GEMM_ALIGNMENT = 8


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
            del input_size, output_size, params_dtype
            out_size = int(sum(output_partition_sizes))
            in_size = int(input_size_per_partition)
            if out_size <= 0 or in_size <= 0:
                raise ValueError(
                    "source-FP8 W8A16 needs positive N and K, got "
                    f"N={out_size}, K={in_size}")
            weight_loader = extra_weight_attrs.get("weight_loader")
            weight = ModelWeightParameter(
                data=torch.empty(out_size, in_size,
                                 dtype=torch.float8_e4m3fn),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
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
                weight_loader=weight_loader,
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
            if scales.dtype not in (torch.float8_e8m0fnu, torch.uint8):
                raise TypeError(
                    "source-FP8 W8A16 needs float8_e8m0fnu/uint8 scales, "
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
            if is_bmm:
                groups = int(getattr(layer, "bmm_batch_size", 0))
                if groups <= 0 or n % groups != 0:
                    raise ValueError(
                        "source-FP8 W8A16 BMM needs a positive group count "
                        f"dividing N; got groups={groups}, N={n}")
                rows = n // groups
                if int(getattr(layer, "tp_size", 1)) != 1:
                    raise ValueError(
                        "source-FP8 W8A16 BMM is release-gated only for TP=1")
                if rows % DS_BLOCK != 0 or k % DS_BLOCK != 0:
                    raise ValueError(
                        "block128 source-FP8 W8A16 BMM needs per-group N and "
                        f"K divisible by {DS_BLOCK}; got N={rows}, K={k}")
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

            if m <= _DECODE_MAX_M:
                flat = fp8_source_gemv(x3, q, scales, groups)
                if is_bmm:
                    return flat.reshape(*outer, groups, rows)
                return flat.reshape(*outer, n)

            # One bounded caller-scoped transient, consumed immediately by
            # Gridbook's owned CUTLASS bridge and never attached to the layer.
            expanded = fp8_source_expand_bf16(q, scales)
            if not is_bmm:
                expert_ends = torch.full(
                    (1,), m, dtype=torch.int32, device=x.device)
                result = cb_bf16_grouped_mm(
                    x3[:, 0, :], expanded.view(1, n, k), expert_ends, 0)
                del expanded
                return result.reshape(*outer, n)

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
            return restored.reshape(*outer, groups, rows)

    return Fp8SourceW8A16LinearMethod


def build_fp8_source_w8a16_method(wire_id: str):
    """Construct the block128-only source-FP8 W8A16 serving method."""

    if wire_id != WIRE_FP8_BLOCK128:
        raise ValueError(
            "Fp8SourceW8A16LinearMethod accepts only "
            f"{WIRE_FP8_BLOCK128!r}, got {wire_id!r}; direct g32 MXFP8 stays "
            "on Mxfp8DenseLinearMethod (W8A8)")
    return _build_method_class()()


class Fp8SourceW8A16LinearMethod:  # pragma: no cover - public identity marker
    """Public identity of Gridbook's block128 source-FP8 W8A16 route."""
