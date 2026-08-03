"""MXFP8 reference quantization and the DeepSeek block-128 embedding.

Torch-only reference semantics for the MXFP8 dense lane (E4M3 elements, one
UE8M0 — power-of-two, ``float8_e8m0fnu`` — scale per 32 contiguous K
elements), plus the exact embedding of DeepSeek's block-quantized FP8
convention into it.

The embedding is the reason this file exists and deserves its one-line proof:
DeepSeek stores one UE8M0 scale per 128x128 tile.  128 = 4 * 32, so a 32-wide
MX chunk never straddles a tile boundary, and

    SF_mx[n, c] = S_ds[n // 128, c // 4]

is total and exact: element bytes are untouched and the scale map is pure
replication of the same exponent byte.  Every dequantized product is
bit-identical between the two readings, for any K divisible by 32 (a ragged
final tile still contains only whole chunks).  The converse embedding does not
exist — MXFP8's finer scales cannot be represented in block-128 form — so the
containment is one-directional.

Quantization rule (activations and any producer-side use): the ceil rule,
``scale = 2 ** ceil(log2(amax / 448))`` per group, which guarantees
``amax / scale <= 448`` and therefore never saturates E4M3.  This matches the
UE8M0 semantics vLLM's own utilities implement; the floor-based OCP variant
saturates up to amax/scale < 512 and would not round-trip DeepSeek's own
tensors bit-exactly.

Everything here is reference-grade and torch-only: the CUDA extension owns the
fast path, and the tests hold the two to each other.  The swizzled SF plane is
filled by scattering with offsets computed by the extension from the SAME CuTe
layout the mainloop reads, so there is no Python respelling of the swizzle to
drift.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "E4M3_MAX",
    "SFVEC",
    "DS_BLOCK",
    "quantize_mxfp8",
    "dequant_mxfp8",
    "broadcast_block128_scales",
    "fill_sf_plane",
    "mxfp8_reference_mm",
]

#: Largest finite E4M3 magnitude.
E4M3_MAX = 448.0
#: MX scale granularity: one UE8M0 byte per this many contiguous K elements.
SFVEC = 32
#: DeepSeek block-quantization tile edge.
DS_BLOCK = 128

# float8_e8m0fnu encodes 2**(byte - 127); byte 0x00 is 2**-127, 0xFF is NaN.
_E8M0_BIAS = 127
_E8M0_MIN_EXP = -127
_E8M0_MAX_EXP = 127


def _require_k_multiple(k: int) -> None:
    if k % SFVEC != 0:
        raise ValueError(
            f"MXFP8 needs K divisible by {SFVEC} (one UE8M0 scale per "
            f"{SFVEC} elements); got K={k}")


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` to (e4m3 bytes ``[..., K]``, ue8m0 exponents
    ``[..., K // 32]`` as uint8).

    Ceil rule per 32-group: ``exp = ceil(log2(amax / 448))`` clamped to the
    E8M0 range, so no element saturates.  An all-zero group takes the smallest
    scale (byte 0) and quantizes to zeros exactly.
    """
    if x.dim() < 1:
        raise ValueError("quantize_mxfp8 needs at least one dimension")
    k = x.shape[-1]
    _require_k_multiple(k)
    xf = x.to(torch.float32)
    groups = xf.unflatten(-1, (k // SFVEC, SFVEC))
    amax = groups.abs().amax(dim=-1)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX))
    # amax == 0 -> log2 gives -inf -> clamp lands on the minimum exponent,
    # which is exactly the all-zero-group convention.
    exp = torch.clamp(exp, min=float(_E8M0_MIN_EXP), max=float(_E8M0_MAX_EXP))
    sf = (exp.to(torch.int16) + _E8M0_BIAS).to(torch.uint8)
    scale = torch.exp2(exp)
    q = (groups / scale.unsqueeze(-1)).flatten(-2)
    q = q.to(torch.float8_e4m3fn)
    return q, sf


def dequant_mxfp8(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """fp32 reconstruction of (e4m3 ``[..., K]``, ue8m0-as-uint8
    ``[..., K // 32]``)."""
    k = q.shape[-1]
    _require_k_multiple(k)
    if sf.shape != q.shape[:-1] + (k // SFVEC,):
        raise ValueError(
            f"scale shape {tuple(sf.shape)} does not match value shape "
            f"{tuple(q.shape)} at one scale per {SFVEC}")
    exp = sf.to(torch.int16) - _E8M0_BIAS
    scale = torch.exp2(exp.to(torch.float32))
    vals = q.to(torch.float32).unflatten(-1, (k // SFVEC, SFVEC))
    return (vals * scale.unsqueeze(-1)).flatten(-2)


def broadcast_block128_scales(
    s_block: torch.Tensor, rows: int, k: int
) -> torch.Tensor:
    """DeepSeek block-128 UE8M0 scales -> row-major per-32 MX scales, exactly.

    ``s_block`` is ``[ceil(rows / 128), ceil(k / 128)]`` (uint8 or
    float8_e8m0fnu storage); the result is ``[rows, k // 32]`` uint8 with

        out[n, c] = s_block[n // 128, c // 4]

    Pure replication — no arithmetic touches the exponent bytes, which is what
    makes the embedding bit-exact.  Ragged edges are handled by the index
    computation itself: a final partial tile still owns whole 32-chunks.
    """
    _require_k_multiple(k)
    expect = ((rows + DS_BLOCK - 1) // DS_BLOCK,
              (k + DS_BLOCK - 1) // DS_BLOCK)
    if tuple(s_block.shape) != expect:
        raise ValueError(
            f"block-scale shape {tuple(s_block.shape)} does not match "
            f"{expect} for rows={rows}, K={k}")
    src = s_block
    if src.dtype != torch.uint8:
        src = src.view(torch.uint8)
    row_idx = torch.arange(rows, device=src.device) // DS_BLOCK
    col_idx = (torch.arange(k // SFVEC, device=src.device) * SFVEC) // DS_BLOCK
    return src[row_idx][:, col_idx].contiguous()


def fill_sf_plane(
    sf_rowmajor: torch.Tensor,
    offsets: torch.Tensor,
    plane_numel: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter row-major per-32 scales into the swizzled SF plane.

    ``offsets`` comes from the extension's ``mxfp8_sf_offsets`` — the
    mainloop's own CuTe layout evaluated per (row, group) — so this function
    contains no layout knowledge at all.  Padding bytes stay zero
    (``2 ** -127`` under UE8M0), which multiplies padded garbage toward zero
    rather than amplifying it.
    """
    flat = sf_rowmajor
    if flat.dtype != torch.uint8:
        flat = flat.view(torch.uint8)
    flat = flat.reshape(-1)
    if offsets.numel() != flat.numel():
        raise ValueError(
            f"offsets numel {offsets.numel()} != scales numel {flat.numel()}")
    if out is None:
        out = torch.zeros(plane_numel, dtype=torch.uint8, device=flat.device)
    elif out.numel() != plane_numel or out.dtype != torch.uint8:
        raise ValueError("out must be a uint8 tensor of plane_numel elements")
    out[offsets.to(flat.device)] = flat
    return out


def mxfp8_reference_mm(
    a_q: torch.Tensor, a_sf: torch.Tensor,
    b_q: torch.Tensor, b_sf: torch.Tensor,
) -> torch.Tensor:
    """The oracle: fp32 dequant-then-matmul, bf16 rounding at the end.

    The kernel computes the same contraction with hardware SF application and
    fp32 accumulation; the parity tests compare against this at fp32-oracle
    tolerance, from THE SAME quantized operands, so the comparison isolates
    the GEMM rather than the quantizer.
    """
    a = dequant_mxfp8(a_q, a_sf)
    b = dequant_mxfp8(b_q, b_sf)
    return (a @ b.t()).to(torch.bfloat16)
