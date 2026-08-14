"""Direct MXFP4 baseline for comparison.

Encodes weight without a codebook (plain MXFP4 per-32 E2M1) and decodes.
Used in tests to report compression/quality promise: codebook at k/8+0.25
bpw vs direct MXFP4 at 4.25 bpw.
"""
from __future__ import annotations

import torch

from .e2m1 import quantize_mxfp4_block, dequant_mxfp4_block
from .format import SUPERBLOCK, MX_BLOCK


def baseline_mxfp4_encode_decode(weight: torch.Tensor) -> torch.Tensor:
    """Direct MXFP4 round-trip (no codebook) -> BF16 reconstruction."""
    if weight.dim() != 2:
        raise ValueError("weight must be 2-D")
    rows, K = weight.shape
    if K % SUPERBLOCK != 0:
        raise ValueError(f"K={K} not multiple of {SUPERBLOCK}")
    # operate per row as blocks of MX_BLOCK
    out = torch.zeros_like(weight, dtype=torch.bfloat16)
    for r in range(rows):
        q, sf = quantize_mxfp4_block(weight[r : r + 1, :])
        dq = dequant_mxfp4_block(q, sf)
        out[r] = dq[0].to(torch.bfloat16)
    return out


def reconstruction_metrics(orig: torch.Tensor, recon: torch.Tensor) -> dict:
    """SNR / relative error helpers (float32)."""
    o = orig.to(torch.float32)
    r = recon.to(torch.float32)
    err = r - o
    rel_l2 = float(err.norm() / o.norm().clamp_min(1e-12))
    max_abs = float(err.abs().max().item())
    mse = float((err ** 2).mean().item())
    # per-group breakdown optional
    return {"rel_l2": rel_l2, "max_abs": max_abs, "mse": mse, "orig_norm": float(o.norm().item())}
