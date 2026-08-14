"""E2M1 nibble encode/decode and E8M0 per-32 scale handling for MXFP4-CB.

Reference semantics only (CPU torch).  Matches OCP MX v1.0 and Gridbook's
grid definition so cross-platform comparison is apples-to-apples.

E2M1 grid
---------
  Magnitudes: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0  (sorted ascending)
  16 codes: bit3 = sign (1 negative), bits[2:0] = magnitude index.
  Encoding order matches ml_dtypes / gridbook.codec.fp4_e2m1_codes:
    code = mag_idx | (sign<<3)

E8M0 (UE8M0) per-32
------------------
  Byte E encodes power-of-two scale  2^(E-127).
  Byte 0x00 => 2^-127, 0xFF is reserved NaN per OCP — encoder clamps to
  0xFE (2^127) for overflow and to 0x00 for zero groups.  Zero groups decode
  to the minimum scale (same rule as gridbook/mxfp8).

  Per-32 quant rule (same as MXFP8 reference, adapted for FP4 max=6):
    exp = ceil_log2( amax / 6 )   clamped to [-127, 127], zeros -> -127
  Implemented via frexp (exact integer arithmetic) to avoid the log2
  float32 rounding defect pinned in gridbook/mxfp8.py.
"""
from __future__ import annotations

import torch

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_GRID_MAX = 6.0
E8M0_BIAS = 127
E8M0_MIN = -127
E8M0_MAX = 127
MX_BLOCK = 32

# Sorted signed grid for nearest-neighbour search (debug / distance)
_SIGNED_E2M1_GRID: torch.Tensor | None = None


def _signed_grid(device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    global _SIGNED_E2M1_GRID
    # Build per call; tiny.
    vals = sorted({v for m in E2M1_MAGNITUDES for v in (m, -m)})
    return torch.tensor(vals, dtype=dtype, device=device or torch.device("cpu"))


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode floating tensor to E2M1 nibble codes (uint8 0..15).

    Round-to-nearest, ties to even on magnitude index (matches NVFP4
    nearest-even in gridbook.codec fallback).  Off-grid infinities raise.
    """
    if not torch.isfinite(x).all():
        raise ValueError("e2m1_encode: input contains non-finite values")
    xf = x.to(torch.float32)
    mag = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=xf.device)
    # Magnitude index via nearest with tie->even
    xmag = xf.abs()
    # Upper index via bucketize
    upper = torch.bucketize(xmag, mag).clamp_max(mag.numel() - 1)
    lower = (upper - 1).clamp_min(0)
    lo = mag[lower]
    hi = mag[upper]
    lower_dist = xmag - lo
    upper_dist = hi - xmag
    choose_upper = upper_dist < lower_dist
    ties = upper_dist == lower_dist
    # tie to even magnitude index
    choose_upper = choose_upper | (ties & ((upper & 1) == 0))
    mag_idx = torch.where(choose_upper, upper, lower)
    # Safety: verify choice is on-grid (for huge values >6, both bounds are 6)
    # Clamp to valid range already done.
    is_neg = xf < 0  # signbit for -0.0 -> false (we want +0 for zero)
    # But preserve sign of zero that maps to 0 magnitude?
    # OCP / NVFP4 preserve sign of zero on underflow; for normal encode,
    # -0.0 should still be +0 code (sign 0 for mag 0).  We force sign 0 when mag 0.
    is_neg = is_neg & (mag_idx != 0)
    codes = mag_idx.to(torch.uint8) | (is_neg.to(torch.uint8) << 3)
    return codes


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 nibble codes (uint8 0..15) to float32 values."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"e2m1_decode expects uint8, got {codes.dtype}")
    if bool((codes > 15).any()):
        raise ValueError("e2m1_decode: nibble code out of range 0..15")
    mag = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=codes.device)
    mag_idx = (codes & 0x7).to(torch.long)
    sign = (codes >> 3) & 0x1
    vals = mag[mag_idx]
    vals = torch.where(sign.bool(), -vals, vals)
    return vals


def e2m1_decode_bf16(codes: torch.Tensor) -> torch.Tensor:
    return e2m1_decode(codes).to(torch.bfloat16)


def e2m1_nibbles_to_packed(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] nibbles -> [..., K/2] packed bytes, even = low nibble."""
    if codes.dtype != torch.uint8:
        raise TypeError("e2m1_nibbles_to_packed expects uint8 nibbles")
    if codes.shape[-1] % 2 != 0:
        raise ValueError("nibbles dim must be even to pack")
    lo = codes[..., 0::2].to(torch.int16)
    hi = codes[..., 1::2].to(torch.int16)
    return (lo | (hi << 4)).to(torch.uint8)


def e2m1_packed_to_nibbles(packed: torch.Tensor, elems: int | None = None) -> torch.Tensor:
    """Unpack bytes -> nibbles (uint8 0..15). If elems is None, unpack fully."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed must be uint8")
    if elems is not None and packed.dim() == 1:
        flat = packed.reshape(-1)
        lo = flat & 0xF
        hi = (flat >> 4) & 0xF
        nibbles = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
        nibbles[0::2] = lo
        nibbles[1::2] = hi
        nibbles = nibbles[:elems]
        return nibbles.reshape(*packed.shape[:-1], elems)
    # Batched case: work per last dim
    lo = (packed & 0xF).to(torch.uint8)
    hi = ((packed >> 4) & 0xF).to(torch.uint8)
    out_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    out = torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    if elems is not None:
        out = out[..., :elems]
    return out


# ---------------------------------------------------------------------------
# E8M0 per-32
# ---------------------------------------------------------------------------

def e8m0_encode_amax(amax: torch.Tensor) -> torch.Tensor:
    """Per-group amax (float) -> UE8M0 byte (uint8) per group.

    Rule: smallest power-of-two scale s.t. amax/scale <= 6.  frexp-based,
    zeros -> byte 0 (2^-127).  Clamped to [0, 254] (0xFF reserved as NaN).
    """
    if not torch.isfinite(amax).all():
        raise ValueError("e8m0_encode_amax: amax non-finite")
    amax_f = amax.to(torch.float32)
    # frexp: amax = frac * 2^exp, frac in [0.5, 1)
    # We need scale = 2^s with amax/scale <=6 => s >= ceil(log2(amax/6))
    # Rewrite without log2: use frexp as in mxfp8.py.
    # amax/6: frac' = frac/6 *? Instead compute target = amax/6 and frexp it.
    # Simpler: compute desired exponent via frexp on amax, correct with threshold.
    # Follow mxfp8 pattern: we want minimal s with amax <= 6*2^s.
    # So 6*2^s >= amax  => 2^s >= amax/6.
    # Let amax/6 = f'*2^E' => s = E (+1 if f'>0.5? Actually need ceil).
    # Use torch.frexp on (amax / 6).
    # For zero amax, special-case to E8M0_MIN.
    scale_ge_zero = amax_f > 0
    # Avoid division by zero for zero amax: substitute dummy 1.0 then mask
    safe = torch.where(scale_ge_zero, amax_f / E2M1_GRID_MAX, torch.ones_like(amax_f))
    frac, exp = torch.frexp(safe)  # safe = frac * 2^exp, frac in [0.5,1)
    # ceil_log2(safe) = exp if frac==0.5 exactly? Actually for powers-of-two
    # frac==0.5 => safe is exact power of two => log2 = exp-1.
    # General: ceil(log2(safe)) = exp-1 when frac==0.5 else exp
    # But frexp with frac in [0.5,1): safe = frac*2^exp => log2(safe)=log2(frac)+exp
    # For frac==0.5, log2=-1+exp => exp-1 exact. Otherwise frac>0.5 => log2 in (exp-1, exp)
    # So ceil = exp-1 if frac==0.5 else exp.
    # We need integer s = ceil_log2(safe)
    is_pow2 = frac == 0.5
    ceil_log2 = torch.where(is_pow2, exp - 1, exp)
    # For zeros we already masked, ceil_log2 is dummy; will be overwritten.
    exp_i = torch.where(scale_ge_zero, ceil_log2, torch.full_like(exp, E8M0_MIN))
    exp_i = exp_i.clamp(E8M0_MIN, E8M0_MAX)
    byte = (exp_i.to(torch.int16) + E8M0_BIAS).to(torch.uint8)
    # Reserve 0xFF as NaN: clamp overflowed max to 0xFE per OCP.
    byte = torch.where(byte == 255, torch.full_like(byte, 254), byte)
    return byte


def e8m0_decode(byte: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte -> float32 power-of-two scale."""
    if byte.dtype != torch.uint8:
        byte = byte.to(torch.uint8)
    b = byte.to(torch.int16).to(torch.float32)
    # 0xFF is NaN per OCP — callers should not emit it; decode as NaN for visibility.
    is_nan = byte == 255
    scale = torch.exp2(b - E8M0_BIAS)
    scale = torch.where(is_nan, torch.full_like(scale, float("nan")), scale)
    return scale


def quantize_mxfp4_block(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct OCP MXFP4 quantize for comparison baseline.

    Args:
        x: [..., K] float, K % 32 == 0
    Returns:
        (q_packed, sf): q_packed is uint8 packed nibbles [..., K/2],
                        sf uint8 per-32 block [..., K//32]
        Dequant is e2m1_decode(unpack) * e8m0_decode(sf).repeat_interleave(32)
    """
    if x.dim() < 1:
        raise ValueError("quantize_mxfp4_block needs at least 1 dim")
    K = x.shape[-1]
    if K % MX_BLOCK != 0:
        raise ValueError(f"MXFP4 needs K % {MX_BLOCK} == 0, got K={K}")
    xf = x.to(torch.float32)
    groups = xf.unflatten(-1, (K // MX_BLOCK, MX_BLOCK))
    amax = groups.abs().amax(dim=-1)
    sf = e8m0_encode_amax(amax)
    scale = e8m0_decode(sf).unsqueeze(-1)  # [..., G, 1]
    # protect zero-scale groups: scale is 2^-127, division still finite
    q_nibbles = e2m1_encode((groups / scale).clamp(-E2M1_GRID_MAX, E2M1_GRID_MAX))
    flat_codes = q_nibbles.flatten(-2)  # [..., K]
    q_packed = e2m1_nibbles_to_packed(flat_codes)
    return q_packed, sf


def dequant_mxfp4_block(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 block-packed tensors to float32 [..., K]."""
    if q_packed.dtype != torch.uint8 or sf.dtype != torch.uint8:
        raise TypeError("MXFP4 tensors must be uint8")
    K2 = q_packed.shape[-1] * 2
    if K2 % MX_BLOCK != 0:
        raise ValueError(f"packed K={K2} not multiple of {MX_BLOCK}")
    if sf.shape[-1] != K2 // MX_BLOCK:
        raise ValueError(f"sf shape {tuple(sf.shape)} vs packed K={K2}")
    nibbles = e2m1_packed_to_nibbles(q_packed, K2)
    vals = e2m1_decode(nibbles)
    scale = e8m0_decode(sf)  # [..., G]
    vals = vals.unflatten(-1, (sf.shape[-1], MX_BLOCK))
    out = vals * scale.unsqueeze(-1)
    return out.flatten(-2).to(torch.float32)


def mxfp4_reference_mm(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Quantize two matrices and return dequant-MM + scales for inspection.

    Both a,b are [..., K] with same K.
    """
    a_q, a_sf = quantize_mxfp4_block(a)
    b_q, b_sf = quantize_mxfp4_block(b)
    a_dq = dequant_mxfp4_block(a_q, a_sf)
    b_dq = dequant_mxfp4_block(b_q, b_sf)
    # simple matmul over last dim
    out = (a_dq @ b_dq.transpose(-1, -2))
    return out, {"a_sf": a_sf, "b_sf": b_sf}

