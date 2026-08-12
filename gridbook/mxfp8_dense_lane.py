"""The direct-g32 MXFP8 dense serving lane.

Serves producer-emitted MXFP8 ``linear`` units whose weights are E4M3 with
row-major UE8M0 scales per 32 K elements (``mxfp8_e4m3_e8m0_g32``).
DeepSeek block-128 source FP8 is deliberately rejected: that wire is W8A16
and belongs exclusively to ``Fp8SourceW8A16LinearMethod``.  Keeping the
rejection here as well as in the source-format registry makes it impossible
for a stale factory to silently restore dynamic activation quantization.

W8A8: activations are quantized dynamically to MXFP8 per 32-element group.
The GEMM is the stock sm120/sm121 block-scaled CollectiveBuilder collective
(``kind::mxf8f6f4``) via ``cuda_ext.get_mxfp8_dense_ext``.

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
    SFVEC,
    fill_sf_plane,
    quantize_mxfp8,
)

__all__ = [
    "MXFP8_DENSE_FLAG",
    "mxfp8_dense_enabled",
    "build_mxfp8_dense_method",
    "WIRE_MXFP8_G32",
]

MXFP8_DENSE_FLAG = "GRIDBOOK_MXFP8_DENSE"

#: The sole wire spelling this W8A8 lane serves.
WIRE_MXFP8_G32 = "mxfp8_e4m3_e8m0_g32"
# Kept private as an explicit hard-refusal sentinel for stale callers.
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


def _cudagraph_capture_sizes() -> tuple[int, ...]:
    """Decode row counts vLLM will capture graphs at, or ``()`` if unknown.

    Read the same way ``ops.warn_if_capture_sizes_exceed_the_decode_gates``
    reads it: vLLM imported inside the function so ``import gridbook`` stays
    vLLM-free, and every failure degrading to silence rather than breaking a
    load, because the shape of the compilation config has moved across
    releases. Returning ``()`` costs only the pre-warm below; it cannot make
    serving wrong.
    """
    try:
        from vllm.config import get_current_vllm_config

        compilation = get_current_vllm_config().compilation_config
        sizes = getattr(compilation, "cudagraph_capture_sizes", None) or ()
        return tuple(sorted({int(s) for s in sizes if int(s) > 0}))
    except Exception:  # noqa: BLE001 — advisory only; never break a load
        return ()


def _prewarm_activation_offsets(ext, k: int, device) -> None:
    """Populate the A-side offset cache for every graph capture size.

    WHY THIS EXISTS. ``_SfOffsetCache.get`` computes offsets on the host and
    moves them with an unpinned ``.to(device)``. That copy is illegal inside a
    CUDA graph capture ("Cannot copy between CPU and CUDA tensors during CUDA
    graph capture unless the CPU tensor is pinned"), so any key first seen
    *during* capture is a hard failure rather than a slow path.

    The B side never had this problem: ``process_weights_after_loading``
    resolves it from weight dimensions known at load. The A side is keyed by
    the runtime row count, which under ``FULL_DECODE_ONLY`` is first seen
    inside the capture region — one first-time miss per capture size.

    Doing it here is ``docs/KERNELS.md`` CUDA-graph safety rule 3 ("all
    device-side constants and per-device kernel setup happen once, at model
    load"), and mirrors ``cb_gemv_v2_prepare`` / ``cb_moe_persistent_b_prepare``
    / ``cb_fused_fp4v2_prepare``, which are called from this same hook for the
    same reason.

    Not wrapped in try/except: at load these are the identical calls the
    forward will make, so a failure here is a real defect and should surface
    at load rather than mid-capture.
    """
    for rows in _cudagraph_capture_sizes():
        _OFFSETS.get(ext, rows, k, is_b=False, device=device)


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
    if wire_id == WIRE_FP8_BLOCK128:
        raise ValueError(
            "block128 source FP8 is a W8A16 contract and cannot enter "
            "Mxfp8DenseLinearMethod (W8A8); use "
            "Fp8SourceW8A16LinearMethod")
    if wire_id != WIRE_MXFP8_G32:
        raise ValueError(f"unknown MXFP8 dense wire id {wire_id!r}")

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import (
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
            layer.weight_block_size = None
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
            # A-side (activation) offsets, for graph capture. Both call sites
            # that need them -- ``_quantize_activations`` and the BMM branch of
            # ``apply`` -- key on the runtime row count at this same ``k``
            # (``k == layer.weight.shape[1]`` for both), so one pre-warm here
            # covers both paths.
            _prewarm_activation_offsets(ext, k, device)
            sf_rm = layer.weight_scale.data
            del layer.weight_scale
            is_bmm = bool(getattr(layer, "is_bmm", False))
            if is_bmm:
                groups = int(getattr(layer, "bmm_batch_size", 0))
                if groups <= 0 or n % groups != 0:
                    raise ValueError(
                        f"MXFP8 BMM needs a positive batch size dividing N; "
                        f"got batch={groups}, N={n}")
                rows = n // groups
                if int(getattr(layer, "tp_size", 1)) != 1:
                    raise ValueError("MXFP8 BMM is audited only for TP=1")
                sf_groups = sf_rm.reshape(groups, rows, k // SFVEC).to(device)
                offs = _OFFSETS.get(
                    ext, rows, k, is_b=True, device=device)
                plane_numel = int(ext.mxfp8_sf_plane_numel(rows, k))
                planes = torch.stack([
                    fill_sf_plane(sf_groups[group], offs, plane_numel)
                    for group in range(groups)
                ])
                layer.register_buffer(
                    "weight_sf_planes", planes, persistent=False)
                from .dsv4_woa import (
                    DSV4_MXFP8_BMM_ABI,
                    DSV4_MXFP8_BMM_ATTR,
                    install_dsv4_woa_adapter,
                )
                setattr(layer, DSV4_MXFP8_BMM_ATTR, DSV4_MXFP8_BMM_ABI)
                install_dsv4_woa_adapter()
            else:
                offs = _OFFSETS.get(ext, n, k, is_b=True, device=device)
                plane = fill_sf_plane(
                    sf_rm.to(device), offs,
                    int(ext.mxfp8_sf_plane_numel(n, k)))
                layer.register_buffer(
                    "weight_sf_plane", plane, persistent=False)
            layer.weight.data = w.contiguous()

        def apply(self, layer, x: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            ext = _require_lane_ext(x.device)
            if bool(getattr(layer, "is_bmm", False)):
                groups = int(getattr(layer, "bmm_batch_size", 0))
                if x.ndim < 3 or int(x.shape[-2]) != groups:
                    raise ValueError(
                        f"MXFP8 BMM expected [..., {groups}, K], got "
                        f"{tuple(x.shape)}")
                k = int(layer.weight.shape[1])
                if int(x.shape[-1]) != k:
                    raise ValueError(
                        f"MXFP8 BMM input K={x.shape[-1]} does not match "
                        f"weight K={k}")
                n = int(layer.weight.shape[0])
                if groups <= 0 or n % groups != 0:
                    raise ValueError(
                        f"MXFP8 BMM has invalid batch={groups}, N={n}")
                rows = n // groups
                outer = tuple(x.shape[:-2])
                x3 = x.reshape(-1, groups, k)
                if x3.dtype != torch.bfloat16:
                    x3 = x3.to(torch.bfloat16)
                a_q, a_sf = quantize_mxfp8(x3)
                m = int(x3.shape[0])
                offs = _OFFSETS.get(
                    ext, m, k, is_b=False, device=x.device)
                plane_numel = int(ext.mxfp8_sf_plane_numel(m, k))
                weights = layer.weight.view(groups, rows, k)
                weight_planes = getattr(layer, "weight_sf_planes", None)
                if weight_planes is None or int(weight_planes.shape[0]) != groups:
                    raise RuntimeError(
                        "MXFP8 BMM weight scale planes were not finalized")
                outputs = []
                for group in range(groups):
                    a_plane = fill_sf_plane(
                        a_sf[:, group, :], offs, plane_numel)
                    # A size-one leading dimension makes PyTorch treat the
                    # group slice as contiguous even though its row stride is
                    # still ``groups * K``. The native ABI requires the exact
                    # compact ``(K, 1)`` stride at decode M=1, so ``contiguous``
                    # is insufficient here; clone materializes that layout.
                    a_group = a_q[:, group, :].clone(
                        memory_format=torch.contiguous_format)
                    y = ext.mxfp8_dense_mm(
                        a_group, a_plane,
                        weights[group], weight_planes[group])
                    outputs.append(y)
                result = torch.stack(outputs, dim=1)
                if bias is not None:
                    result = result + bias.view(groups, rows)
                return result.reshape(*outer, groups, rows)
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
