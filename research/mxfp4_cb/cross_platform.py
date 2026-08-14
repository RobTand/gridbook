"""Cross-platform hypothesis test: canonical codebook -> NVFP4 vs MXFP4.

The stronger hypothesis: a single FP32/BF16 logical codebook can be
deterministically projected to both
  (A) the repository's current NVFP4 representation (E2M1 values + E4M3 /
      two-tier scale — here we model the v1 E4M3-direct 16-byte plane), and
  (B) the OCP-MXFP4 representation (E2M1 values + E8M0 per-32 power-of-two),

keeping the SAME packed indices, and still be coherent.

This module:
  * defines a canonical codebook (FP32, arbitrary — here synthetic normal),
  * projects it deterministically to two physical tables:
      physical_nvfp4 = snap canonical to E2M1 grid (BF16 exact) — same grid,
        but the *interpretation* at decode pairs it with E4M3 scales.
      physical_mx    = snap canonical to E2M1 grid (identical values, since
        element grids coincide) — paired with E8M0 scales.
    The projection itself is lossless between tables if we stop at values.
    Coherence must therefore be evaluated on *wire* reconstruction including
    scales: same indices + different scale planes => diverging weights.

  * exposes helpers to encode synthetic expert-like tensors under each scheme,
    then measures assignment stability (fraction of codewords where NVFP4-optimal
    and MXFP4-optimal encoders choose the same index) and reconstruction /
    output divergence (relative L2, max-abs) when reusing one scheme's indices
    under the other's scales.

If identical indices are not coherent (expected), we quantify the failure and
propose the minimum metadata to restore coherence: per-platform index streams
(and scale planes), or equivalently a per-superblock platform flag that selects
which stream to decode.

Code is CPU-only, deterministic, and the experiment is reproducible with a
fixed seed.
"""
from __future__ import annotations

import torch

from .format import Mxfp4CbFormat, SUPERBLOCK, E2M1_MAGNITUDES
from .e2m1 import e2m1_decode, e2m1_encode
from .codec import encode_mxfp4_cb, decode_mxfp4_cb, pack_indices, unpack_indices
from .e2m1 import e8m0_encode_amax, e8m0_decode

# NVFP4 scale handling for projection comparison:
#   group 16, E4M3 byte: we model the v1 direct E4M3 plane (16 bytes per SB).
#   NVFP4 grid max is still 6.0, scale = amax/6 snapped to E4M3.

E4M3_MAX = 448.0
FP4_GROUP = 16


def _to_e4m3(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn).to(torch.float32)


def nvfp4_scale_per16(weight_block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight block [256] -> (scales_f32 [16], sf_bytes [16] uint8)."""
    groups = weight_block.reshape(16, FP4_GROUP)
    amax = groups.abs().amax(dim=-1)
    # NVFP4 scale convention: amax/6 snapped to E4M3, zeros -> minimum?
    # gridbook uses plain amax/6 clamped to E4M3; we replicate.
    raw = (amax / 6.0).clamp_min(0)
    # For zeros keep smallest E4M3? In NVFP4 codec, zeros give scale 0? Actually
    # E4M3 scale zero would be 0, which multiplies values to 0.  We follow the
    # reference: amax==0 -> scale 0? But we choose smallest positive for parity.
    # Here we keep E4M3 byte quant: (raw.to_e4m3).  Zero stays 0.
    e4m3 = raw.to(torch.float8_e4m3fn)
    sf_bytes = e4m3.view(torch.uint8)
    scales = e4m3.to(torch.float32)
    # For comparison, treat byte 0 as scale 0 (so decode zero-multiplies).
    # To avoid NaN divergence, keep as 0.
    return scales, sf_bytes


def snap_to_e2m1_grid(t: torch.Tensor) -> torch.Tensor:
    """Deterministic projection: FP32 canonical -> E2M1 grid (BF16-exact)."""
    codes = e2m1_encode(t)
    vals = e2m1_decode(codes)
    return vals.to(torch.bfloat16).to(torch.float32)


def make_canonical_codebook(
    k: int, mode: str = "product", seed: int = 0
) -> list[torch.Tensor]:
    """Single FP32/BF16 logical product codebook (non-grid) for hypothesis.

    Returns list of sub-tables as float32, NOT yet on E2M1.
    Values drawn normal(0,1) so projection is non-trivial.
    """
    g = torch.Generator().manual_seed(seed)
    if mode != "product":
        raise ValueError("canonical generator only supports product mode for hypothesis")
    base, extra = divmod(k, 2)
    w0 = base + (1 if 0 < extra else 0)
    w1 = base + (1 if 1 < extra else 0)
    parts = []
    for w in (w0, w1):
        t = torch.randn((1 << w), 4, generator=g, dtype=torch.float32)
        parts.append(t)
    return parts


def project_to_physical(
    canonical_parts: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Deterministically project canonical FP32 parts to NVFP4 and MXFP4 physical tables.

    Both grids are E2M1, so values coincide; we return two copies to make the
    intent explicit and to allow future divergence (e.g., different rounding).
    """
    nv = [snap_to_e2m1_grid(p) for p in canonical_parts]
    mx = [snap_to_e2m1_grid(p) for p in canonical_parts]
    return nv, mx


def _gen_synthetic_weight(rows: int, K: int, seed: int = 42) -> torch.Tensor:
    """Expert-like synthetic weight: per-row scaled normal with outlier channels."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn((rows, K), generator=g, dtype=torch.float32) * 0.05
    # Add a few large outlier channels per row to stress scale quantizers differently.
    for r in range(rows):
        idx = torch.randperm(K, generator=g)[: K // 64]
        w[r, idx] *= 8.0
    return w.to(torch.bfloat16).to(torch.float32)


def encode_with_nvfp4_scales(
    weight: torch.Tensor, fmt_mx: Mxfp4CbFormat, phy: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode weight using MX codebook but NVFP4 per-16 E4M3 scales (mismatched).

    Returns (codes [rows, n_sb, 32], scales_e4m3 [rows, n_sb, 16]).
    This is the 'NVFP4-optimal assignment' oracle for stability measurement.
    """
    rows, K = weight.shape
    n_sb = K // SUPERBLOCK
    codes_all = torch.zeros((rows, n_sb, 32), dtype=torch.int64)
    scales_all = torch.zeros((rows, n_sb, 16), dtype=torch.float32)
    # Codebook lookup via brute force (same as codec but with per-16 scales)
    w0, w1 = fmt_mx.sub_widths
    cb0, cb1 = phy
    for r in range(rows):
        for sb in range(n_sb):
            block = weight[r, sb * SUPERBLOCK : (sb + 1) * SUPERBLOCK]
            scales, _ = nvfp4_scale_per16(block)
            scales_all[r, sb] = scales
            # expand per-16 scales to per-element then group to vectors
            scale_exp = scales.repeat_interleave(FP4_GROUP)  # [256]
            # avoid divide-by-zero: where scale==0, scaled vector is 0 and any code decodes 0 after scale
            safe_scale = torch.where(scales.repeat_interleave(FP4_GROUP) == 0, torch.ones_like(scale_exp), scale_exp)
            scaled_block = block / safe_scale
            # zero-scale groups' scaled values already 0 after divide by 1, fine
            # null out those groups: they were already near zero so distance stable
            for gi in range(16):
                if scales[gi].item() == 0:
                    scaled_block[gi * FP4_GROUP : (gi + 1) * FP4_GROUP] = 0
            scaled_block = scaled_block.clamp(-6, 6).reshape(32, 8)
            for ci in range(32):
                v0 = scaled_block[ci, :4]
                v1 = scaled_block[ci, 4:]
                d0 = ((cb0 - v0) ** 2).sum(dim=-1)
                d1 = ((cb1 - v1) ** 2).sum(dim=-1)
                i0 = int(d0.argmin().item())
                i1 = int(d1.argmin().item())
                codes_all[r, sb, ci] = i0 | (i1 << w0)
    return codes_all, scales_all


def encode_with_mx_scales(
    weight: torch.Tensor, fmt_mx: Mxfp4CbFormat, phy: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """MX-optimal assignment (per-32 E8M0). Returns (codes, sf_bytes)."""
    rows, K = weight.shape
    n_sb = K // SUPERBLOCK
    codes_all = torch.zeros((rows, n_sb, 32), dtype=torch.int64)
    sf_all = torch.zeros((rows, n_sb, 8), dtype=torch.uint8)
    w0 = fmt_mx.sub_widths[0]
    cb0, cb1 = phy
    for r in range(rows):
        for sb in range(n_sb):
            block = weight[r, sb * SUPERBLOCK : (sb + 1) * SUPERBLOCK]
            groups = block.reshape(8, 32)
            amax = groups.abs().amax(dim=-1)
            sf = e8m0_encode_amax(amax)
            sf_all[r, sb] = sf
            scales = e8m0_decode(sf)
            scaled = (groups / scales.unsqueeze(-1)).clamp(-6, 6).reshape(32, 8)
            for ci in range(32):
                v0 = scaled[ci, :4]
                v1 = scaled[ci, 4:]
                d0 = ((cb0 - v0) ** 2).sum(dim=-1)
                d1 = ((cb1 - v1) ** 2).sum(dim=-1)
                i0 = int(d0.argmin().item())
                i1 = int(d1.argmin().item())
                codes_all[r, sb, ci] = i0 | (i1 << w0)
    return codes_all, sf_all


def cross_platform_report(
    k: int = 16, rows: int = 4, K: int = 512, seed: int = 0
) -> dict:
    """Run the hypothesis experiment and return quantitative metrics.

    1. Build one canonical product codebook.
    2. Project to nv/mx physical tables (both E2M1, values identical).
    3. Encode a synthetic weight tensor with each scale scheme's optimal
       encoder (same physical codebook values, different scales).
    4. Measure:
       - assignment stability = fraction of code slots with equal codes
       - reconstruction divergence when reusing MX indices with NVFP4 scales
         (and vice versa)
       - direct weight-space relative L2 between the two optimal reconstructions
       - a mock output divergence: ||X @ W_nv - X @ W_mx|| / ||X @ W||
    """
    fmt = Mxfp4CbFormat(k=k, mode="product")
    canon = make_canonical_codebook(k, seed=seed)
    phy_nv, phy_mx = project_to_physical(canon)
    # physical values identical, but we keep two copies for clarity
    w = _gen_synthetic_weight(rows, K, seed=seed + 1)

    codes_nv, scales_nv = encode_with_nvfp4_scales(w, fmt, phy_nv)
    codes_mx, sf_mx = encode_with_mx_scales(w, fmt, phy_mx)

    total = rows * (K // SUPERBLOCK) * 32
    stable = int((codes_nv == codes_mx).sum().item())
    stability = stable / total

    # Reconstructions for each scheme's optimal path
    def recon_nv(codes, scales):
        out = torch.zeros_like(w)
        w0 = fmt.sub_widths[0]
        for r in range(rows):
            for sb in range(K // SUPERBLOCK):
                scale_exp = scales[r, sb].repeat_interleave(FP4_GROUP)  # [256]
                for ci in range(32):
                    code = int(codes[r, sb, ci].item())
                    i0 = code & ((1 << fmt.sub_widths[0]) - 1)
                    i1 = (code >> fmt.sub_widths[0]) & ((1 << fmt.sub_widths[1]) - 1)
                    vals = torch.cat([phy_nv[0][i0], phy_nv[1][i1]])  # [8]
                    base = sb * SUPERBLOCK + ci * 8
                    out[r, base : base + 8] = vals * scale_exp[ci * 8 : ci * 8 + 8]
        return out

    def recon_mx(codes, sf):
        out = torch.zeros_like(w)
        for r in range(rows):
            for sb in range(K // SUPERBLOCK):
                scales = e8m0_decode(sf[r, sb])  # [8]
                scale_exp = scales.repeat_interleave(32)  # [256]
                for ci in range(32):
                    code = int(codes[r, sb, ci].item())
                    i0 = code & ((1 << fmt.sub_widths[0]) - 1)
                    i1 = (code >> fmt.sub_widths[0]) & ((1 << fmt.sub_widths[1]) - 1)
                    vals = torch.cat([phy_mx[0][i0], phy_mx[1][i1]])
                    base = sb * SUPERBLOCK + ci * 8
                    out[r, base : base + 8] = vals * scale_exp[ci * 8 : ci * 8 + 8]
        return out

    w_nv_opt = recon_nv(codes_nv, scales_nv)
    w_mx_opt = recon_mx(codes_mx, sf_mx)

    # Cross: reuse MX codes with NVFP4 scales, and vice versa
    w_cross_nv = recon_nv(codes_mx, scales_nv)  # MX indices interpreted with NVFP4 scales
    w_cross_mx = recon_mx(codes_nv, sf_mx)  # NVFP4 indices interpreted with MX scales

    def rel_l2(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    # Weight-space divergences
    w_div_opt = rel_l2(w_nv_opt, w_mx_opt)
    w_cross_nv_div = rel_l2(w_cross_nv, w_nv_opt)
    w_cross_mx_div = rel_l2(w_cross_mx, w_mx_opt)

    # Output-space divergence: random activation
    g = torch.Generator().manual_seed(seed + 99)
    x = torch.randn((8, K), generator=g, dtype=torch.float32) * 0.1
    # Use float matmuls for comparison
    y_nv = x @ w_nv_opt.t()
    y_mx = x @ w_mx_opt.t()
    y_cross_nv = x @ w_cross_nv.t()
    out_div_opt = rel_l2(y_nv, y_mx)
    out_cross_nv = rel_l2(y_cross_nv, y_nv)

    # Codebook projection divergence: max abs between physical tables (should be 0)
    codebook_max_abs = max(
        float((phy_nv[i] - phy_mx[i]).abs().max().item()) for i in range(len(phy_nv))
    )

    return {
        "k": k,
        "rows": rows,
        "K": K,
        "total_codewords": total,
        "stable_assignments": stable,
        "stability_frac": stability,
        "instability_frac": 1 - stability,
        "codebook_max_abs_between_physical": codebook_max_abs,
        "weight_rel_l2_opt_vs_opt": w_div_opt,
        "weight_rel_l2_cross_nv": w_cross_nv_div,
        "weight_rel_l2_cross_mx": w_cross_mx_div,
        "output_rel_l2_opt": out_div_opt,
        "output_rel_l2_cross_nv": out_cross_nv,
        "explanation": (
            "stability<1.0 means identical indices not coherent: reusing one "
            "platform's indices under the other's scale grid diverges in "
            "weight and output space. Minimum metadata is per-platform index "
            "stream (+ scale plane) or a per-superblock selector + re-encode."
        ),
    }

