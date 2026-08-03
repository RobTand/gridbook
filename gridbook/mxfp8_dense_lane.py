"""The MXFP8 dense serving lane: one linear method, two wire spellings.

Serves passthrough ``linear`` units whose weights are E4M3 with UE8M0
scales — either producer-emitted MXFP8 (``mxfp8_e4m3_e8m0_g32``, scales
row-major per 32) or DeepSeek block-128 FP8 (``fp8_e4m3_ue8m0_block128``,
scales per 128x128 tile, embedded exactly into MXFP8 by scale replication at
load; see ``gridbook/mxfp8.py`` for the one-line proof).

W8A8: activations are quantized dynamically to MXFP8 per 32-element group —
strictly FINER than the per-[1,128] grouping the checkpoint's own serving
path uses, so reference accuracy bounds this lane's.  The GEMM is the stock
sm120/sm121 block-scaled CollectiveBuilder collective (``kind::mxf8f6f4``)
via ``cuda_ext.get_mxfp8_dense_ext``.

OPT-IN (``GRIDBOOK_MXFP8_DENSE=1``): the lane is correctness-audited
(kernel-vs-oracle parity over the DSV4 body shapes, recorded in
``source_passthrough.py``); NATIVE-PARITY *served* evidence is still pending,
and Gridbook does not default-enable a lane on correctness evidence alone.
With the flag unset, a passthrough unit that needs this lane refuses at load
with the flag named — fail-closed, exactly like every other lane.

Activation-side cost note: quantization and SF-plane scatter run as torch ops
ahead of the kernel (correctness-first); fusing them is a later optimization
and changes no numerics contract.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from . import cuda_ext
from .lane_select import latched_bool, require_lane
from .mxfp8 import (
    DS_BLOCK,
    SFVEC,
    broadcast_block128_scales,
    fill_sf_plane,
    quantize_mxfp8,
)

__all__ = [
    "MXFP8_DENSE_FLAG",
    "mxfp8_dense_enabled",
    "build_mxfp8_dense_method",
    "WIRE_MXFP8_G32",
    "WIRE_FP8_BLOCK128",
]

MXFP8_DENSE_FLAG = "GRIDBOOK_MXFP8_DENSE"

#: The two wire spellings this lane serves (ids owned by source_passthrough).
WIRE_MXFP8_G32 = "mxfp8_e4m3_e8m0_g32"
WIRE_FP8_BLOCK128 = "fp8_e4m3_ue8m0_block128"


def mxfp8_dense_enabled() -> bool:
    """The latched OPT-IN flag for this lane."""
    return latched_bool(MXFP8_DENSE_FLAG, default=False,
                        meaning="the MXFP8 dense block-scaled lane")


def _require_lane_ext(device=None):
    """Load-time attestation: symbols, device capability, built-for target."""
    return require_lane(
        "MXFP8 dense passthrough serving",
        flag=MXFP8_DENSE_FLAG,
        lane="the MXFP8 block-scaled lane",
        source="mxfp8_dense_gemm.cu",
        alternative="refusing the passthrough unit (fail-closed)",
        get_ext=cuda_ext.get_mxfp8_dense_ext,
        symbols=cuda_ext._MXFP8_DENSE_SYMBOLS,
        buildable=cuda_ext.mxfp8_dense_buildable,
        device=device,
    )


class _SfOffsetCache:
    """Per-(rows, K, side) swizzled-plane offsets, computed once on host.

    The extension evaluates the mainloop's own CuTe layout, so the cache holds
    the only copy of the swizzle there is.  Keyed small: serving reuses a
    handful of M values and exactly one (N, K) per layer.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, bool, Any], torch.Tensor] = {}

    def get(self, ext, rows: int, k: int, *, is_b: bool,
            device) -> torch.Tensor:
        key = (rows, k, is_b, device)
        hit = self._cache.get(key)
        if hit is None:
            hit = ext.mxfp8_sf_offsets(rows, k, is_b).to(device)
            self._cache[key] = hit
        return hit


_OFFSETS = _SfOffsetCache()


def _quantize_activations(ext, x: torch.Tensor) -> tuple[torch.Tensor,
                                                         torch.Tensor]:
    """bf16 ``[M, K]`` -> (e4m3 ``[M, K]``, swizzled SF plane)."""
    m, k = x.shape
    a_q, a_sf = quantize_mxfp8(x)
    offs = _OFFSETS.get(ext, m, k, is_b=False, device=x.device)
    plane = fill_sf_plane(a_sf, offs, int(ext.mxfp8_sf_plane_numel(m, k)))
    return a_q, plane


def build_mxfp8_dense_method(wire_id: str):
    """Construct the vLLM linear method serving ``wire_id``.

    Lazy vLLM import: policy modules and CPU tests import this file without
    vLLM present.  The returned instance's CLASS NAME is the audited backend
    label in ``source_passthrough.FORMATS`` — renaming it is a registry edit.
    """
    if wire_id not in (WIRE_MXFP8_G32, WIRE_FP8_BLOCK128):
        raise ValueError(f"unknown MXFP8 dense wire id {wire_id!r}")

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        GroupQuantScaleParameter,
        ModelWeightParameter,
    )

    class Mxfp8DenseLinearMethod(LinearMethodBase):
        """W8A8 MXFP8 block-scaled linear (Gridbook-owned lane)."""

        def __init__(self, wire: str) -> None:
            self._wire = wire

        def create_weights(self, layer, input_size_per_partition,
                           output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            out_size = int(sum(output_partition_sizes))
            in_size = int(input_size_per_partition)
            if in_size % SFVEC != 0:
                raise ValueError(
                    f"MXFP8 dense lane needs K % {SFVEC} == 0, got {in_size}")
            weight_loader = extra_weight_attrs.get("weight_loader")
            weight = ModelWeightParameter(
                data=torch.empty(out_size, in_size,
                                 dtype=torch.float8_e4m3fn),
                input_dim=1, output_dim=0, weight_loader=weight_loader)
            layer.register_parameter("weight", weight)
            if self._wire == WIRE_FP8_BLOCK128:
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (out_size + DS_BLOCK - 1) // DS_BLOCK,
                        (in_size + DS_BLOCK - 1) // DS_BLOCK,
                        dtype=torch.float8_e8m0fnu),
                    input_dim=1, output_dim=0, weight_loader=weight_loader)
                # The checkpoint spelling: DeepSeek stores the reciprocal-form
                # name; bytes are copied verbatim by the producer.
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = GroupQuantScaleParameter(
                    data=torch.empty(out_size, in_size // SFVEC,
                                     dtype=torch.float8_e8m0fnu),
                    input_dim=1, output_dim=0, weight_loader=weight_loader)
                layer.register_parameter("weight_scale", scale)

        def process_weights_after_loading(self, layer) -> None:
            device = layer.weight.device
            ext = _require_lane_ext(device)
            w = layer.weight.data
            n, k = int(w.shape[0]), int(w.shape[1])
            if self._wire == WIRE_FP8_BLOCK128:
                s_block = layer.weight_scale_inv.data
                sf_rm = broadcast_block128_scales(s_block, n, k)
                del layer.weight_scale_inv
            else:
                sf_rm = layer.weight_scale.data
                del layer.weight_scale
            offs = _OFFSETS.get(ext, n, k, is_b=True, device=device)
            plane = fill_sf_plane(sf_rm.to(device), offs,
                                  int(ext.mxfp8_sf_plane_numel(n, k)))
            layer.register_buffer("weight_sf_plane", plane, persistent=False)
            layer.weight.data = w.contiguous()

        def apply(self, layer, x: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            ext = _require_lane_ext(x.device)
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            a_q, a_plane = _quantize_activations(ext, x2)
            y = ext.mxfp8_dense_mm(a_q, a_plane, layer.weight,
                                   layer.weight_sf_plane)
            if bias is not None:
                y = y + bias
            return y.reshape(*orig_shape[:-1], y.shape[-1])

    return Mxfp8DenseLinearMethod(wire_id)
