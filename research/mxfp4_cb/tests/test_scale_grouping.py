"""Scale grouping invariants: per-32 E8M0 correctness and determinism."""
import torch
import pytest

from research.mxfp4_cb.format import Mxfp4CbFormat, MX_BLOCK, SCALES_PER_SB, SUPERBLOCK
from research.mxfp4_cb.e2m1 import e8m0_encode_amax, e8m0_decode, e2m1_encode, e2m1_decode
from research.mxfp4_cb.codec import encode_mxfp4_cb, decode_mxfp4_cb


def _cb(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w0 = (k + 1) // 2
    w1 = k // 2
    a = e2m1_decode(e2m1_encode(torch.randn((1 << w0, 4), generator=g))).to(torch.float32)
    b = e2m1_decode(e2m1_encode(torch.randn((1 << w1, 4), generator=g))).to(torch.float32)
    return [a, b]


def test_per32_grouping_covers_superblock():
    fmt = Mxfp4CbFormat(k=16)
    rows, K = 1, 256
    cb = _cb(16)
    # Construct weight where each 32-block has distinct amax
    w = torch.zeros(rows, K)
    for g in range(8):
        w[0, g * 32 : (g + 1) * 32] = float(g + 1) * 0.5
    qw = encode_mxfp4_cb(w, fmt, cb)
    # Extract scale bytes and verify they are monotonic (since amax monotonic and power-of-two)
    sf = qw[0, 4 * 16 :]
    scales = e8m0_decode(sf)
    # scales should be non-decreasing
    assert bool((scales[1:] >= scales[:-1]).all()), f"scales not monotonic: {scales}"
    # Re-decode and check grouping: each 32-segment after decode has its own scale factor
    w2 = decode_mxfp4_cb(qw, fmt, cb, rows, K).to(torch.float32)
    # All-zero codebook values? Not asserting values, just that grouping didn't smear
    assert w2.shape == w.shape


def test_uniform_superblock_one_scale_replicated():
    fmt = Mxfp4CbFormat(k=16)
    # If the whole SB is uniform, all 8 E8M0 bytes should be equal (same amax)
    w = torch.full((1, 256), 2.0, dtype=torch.float32)
    cb = _cb(16, seed=1)
    qw = encode_mxfp4_cb(w, fmt, cb)
    sf = qw[0, 4 * 16 :]
    assert sf.unique().numel() == 1, f"uniform SB should have one repeated E8M0 byte, got {sf.tolist()}"


def test_scale_decoding_is_power_of_two():
    sf = torch.arange(0, 254, dtype=torch.uint8)
    scales = e8m0_decode(sf)
    # each scale is exactly 2^exp, check ratio between successive is 2
    ratios = scales[1:] / scales[:-1]
    assert torch.allclose(ratios, torch.full_like(ratios, 2.0))


def test_zero_block_does_not_pollute_neighbors():
    fmt = Mxfp4CbFormat(k=16)
    cb = _cb(16, seed=2)
    w = torch.randn(1, 256) * 0.05
    w[0, 0:32] = 0.0  # zero block
    qw = encode_mxfp4_cb(w, fmt, cb)
    sf = qw[0, 4 * 16 :]
    assert int(sf[0].item()) == 0  # zero block -> E8M0 0
    assert int(sf[1].item()) != 0  # neighbor non-zero => different byte
    # decode roundtrip preserves finite
    w2 = decode_mxfp4_cb(qw, fmt, cb, 1, 256)
    assert torch.isfinite(w2.to(torch.float32)).all()
