"""Native transient expansion wrappers for Gridbook codebook weights.

The public function signatures are retained for serving and loader callers,
but every decode is now performed by a packaged CUDA extension. Unsupported
format contracts fail explicitly; this module has no Triton implementation or
runtime fallback.

FP8-CB expands directly to an E4M3 ``[N, K]`` tile through
``cb_gemv.cu::cb_expand_fp8``. FP4-CB v2 product mode expands to a composed BF16
tile through ``cb_gemv_v2.cu::cb_expand_v2``. Both outputs remain bounded,
per-layer transients rather than model-resident expanded weights.
"""
from __future__ import annotations

import torch


def _validate_packed_rows(cb_qweight: torch.Tensor, N: int, K: int,
                          type_size: int) -> int:
    """Validate the common superblock layout and return raw bytes per row."""
    if N < 0:
        raise ValueError(f"N={N} must be non-negative")
    if K <= 0 or K % 256 != 0:
        raise ValueError(
            f"K={K} must be a positive multiple of the 256-weight superblock")
    if type_size <= 0:
        raise ValueError(f"type_size={type_size} must be positive")
    if cb_qweight.dtype is not torch.uint8:
        raise TypeError("packed CB weights must be uint8")
    if cb_qweight.dim() != 2 or cb_qweight.shape[0] != N:
        raise ValueError(
            f"packed CB weights must have shape [N, row_bytes], got "
            f"{tuple(cb_qweight.shape)} for N={N}")
    if cb_qweight.stride(1) != 1:
        raise ValueError("packed CB weight rows must be contiguous in bytes")
    row_bytes = (K // 256) * type_size
    if cb_qweight.shape[1] < row_bytes:
        raise ValueError(
            f"packed CB row has {cb_qweight.shape[1]} bytes, needs at least "
            f"{row_bytes} for K={K} and type_size={type_size}")
    return row_bytes


def _validate_row_offsets(cb_row_offset: torch.Tensor, N: int,
                          device: torch.device) -> None:
    if cb_row_offset.dtype is not torch.int32:
        raise TypeError("cb_row_offset must be int32")
    if cb_row_offset.dim() != 1 or cb_row_offset.numel() != N:
        raise ValueError(f"cb_row_offset must contain exactly N={N} elements")
    if cb_row_offset.device != device:
        raise ValueError("cb_row_offset and packed weights must share a device")


def expand_cb_to_fp8(
    cb_qweight_padded: torch.Tensor,
    cb_flat_fp8: torch.Tensor,
    cb_row_offset: torch.Tensor,
    N: int,
    K: int,
    k_bits: int,
    n_sub: int,
    type_size: int,
) -> torch.Tensor:
    """Expand FP8-CB values to a native E4M3 ``[N, K]`` transient.

    The native kernel currently implements the shipping four-subcodebook
    product layout. Other layouts are rejected rather than routed to an
    interpreted implementation.
    """
    _validate_packed_rows(cb_qweight_padded, N, K, type_size)
    _validate_row_offsets(cb_row_offset, N, cb_qweight_padded.device)
    if n_sub != 4:
        raise NotImplementedError(
            f"native cb_expand_fp8 supports n_sub=4, got {n_sub}")
    if k_bits <= 0 or type_size != 4 * k_bits or type_size > 192:
        raise NotImplementedError(
            "native cb_expand_fp8 requires k_bits>0, type_size=4*k_bits, "
            f"and type_size<=192; got k_bits={k_bits}, type_size={type_size}")
    if cb_flat_fp8.dtype is not torch.uint8:
        raise TypeError("expand_cb_to_fp8 wants an E4M3-byte uint8 codebook")
    if cb_flat_fp8.dim() != 1 or not cb_flat_fp8.is_contiguous():
        raise ValueError("the E4M3-byte codebook must be a contiguous vector")
    if cb_flat_fp8.device != cb_qweight_padded.device:
        raise ValueError("codebook and packed weights must share a device")

    # Custom-op indirection keeps the native pybind call opaque to Dynamo and
    # CUDA-graph tracing. The op itself uses cuda_ext.require_ext(), so a
    # missing build is a clear NativeKernelUnavailableError.
    from .ops import cb_expand_fp8
    return cb_expand_fp8(cb_qweight_padded, cb_flat_fp8, cb_row_offset,
                         N, K, k_bits, n_sub, type_size)


def expand_cb_to_value(
    cb_qweight_padded: torch.Tensor,
    cb_flat: torch.Tensor,
    cb_row_offset: torch.Tensor,
    N: int,
    K: int,
    k_bits: int,
    n_sub: int,
    type_size: int,
    is_fp4: bool,
) -> torch.Tensor:
    """Expand FP8-CB values to a BF16 ``[N, K]`` transient.

    FP8-CB codebooks are format-constrained to the E4M3 grid. Converting their
    BF16 storage to raw E4M3 bytes and invoking the native direct expander is
    therefore exact; the final E4M3-to-BF16 conversion is exact as well.
    """
    if is_fp4:
        raise NotImplementedError(
            "expand_cb_to_value has no native FP4 contract; use "
            "expand_fp4_v2_to_weight for FP4-v2 product codebooks")
    if cb_flat.dtype is not torch.bfloat16:
        raise TypeError("expand_cb_to_value wants a BF16 FP8-CB codebook")
    if cb_flat.dim() != 1 or not cb_flat.is_contiguous():
        raise ValueError("the BF16 codebook must be a contiguous vector")
    cb_flat_fp8 = cb_flat.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return expand_cb_to_fp8(
        cb_qweight_padded, cb_flat_fp8, cb_row_offset,
        N, K, k_bits, n_sub, type_size).to(torch.bfloat16)


def expand_fp4_v2_to_weight(
    cb_qweight_padded: torch.Tensor,
    cb_flat: torch.Tensor,
    cb_row_offset: torch.Tensor,
    compose: torch.Tensor,
    N: int,
    K: int,
    k_bits: int,
    n_sub: int,
    type_size: int,
) -> torch.Tensor:
    """Expand native FP4-v2 product mode to a composed BF16 transient.

    One invocation of ``cb_expand_v2`` implements one zero-based physical
    two-subcodebook product dictionary. The dense loader handles fused modules
    with multiple dictionaries by invoking this wrapper once per contiguous
    role segment and concatenating the native BF16 results. Passing a raw
    concatenation to one invocation, and signed ``n_sub=1`` encoding, remain
    unsupported and are rejected. A padded row input is compacted to the raw
    byte plane expected by the existing extension.
    """
    row_bytes = _validate_packed_rows(
        cb_qweight_padded, N, K, type_size)
    _validate_row_offsets(cb_row_offset, N, cb_qweight_padded.device)
    if not (0 < k_bits <= 24) or type_size != 4 * k_bits + 9:
        raise NotImplementedError(
            "native cb_expand_v2 requires k_bits in [1,24] and "
            f"type_size=4*k_bits+9; got {k_bits=} and {type_size=}")
    if n_sub != 2:
        raise NotImplementedError(
            "native cb_expand_v2 currently supports only the two-subcodebook "
            f"product layout, got n_sub={n_sub}")
    if cb_flat.dtype is not torch.bfloat16 or cb_flat.dim() != 1 \
            or not cb_flat.is_contiguous():
        raise TypeError("native cb_expand_v2 needs a contiguous BF16 codebook")
    expected_cb_elems = ((4 << ((k_bits + 1) // 2))
                         + (4 << (k_bits // 2)))
    if cb_flat.numel() != expected_cb_elems:
        raise NotImplementedError(
            "native cb_expand_v2 supports one global product codebook; "
            f"expected {expected_cb_elems} elements for k={k_bits}, got "
            f"{cb_flat.numel()} (concatenated per-role codebooks are not "
            "supported)")
    if compose.dtype is not torch.float32 or compose.numel() != 256 * 16 \
            or not compose.is_contiguous():
        raise TypeError(
            "native cb_expand_v2 needs a contiguous float32 compose table "
            "with 4096 elements")
    if cb_flat.device != cb_qweight_padded.device \
            or compose.device != cb_qweight_padded.device:
        raise ValueError(
            "packed weights, codebook, offsets, and compose table must share "
            "a device")

    # The selected one-block codebook is zero-based for this invocation. A
    # larger concatenated codebook is rejected above. Offset *values* are not
    # consumed by cb_expand_v2; the tensor remains in this wrapper's signature
    # to validate that the loader supplied exactly one routing entry per row.
    # This also lets the segmented dense caller reuse its resident offset slice
    # without allocating a zero vector or synchronizing CUDA back to the host.
    raw_rows = cb_qweight_padded[:, :row_bytes]
    qw_flat = raw_rows.contiguous().view(-1)

    from .ops import cb_expand_fp4_v2
    return cb_expand_fp4_v2(qw_flat, cb_flat, compose, 0, N, K,
                            k_bits, type_size)
