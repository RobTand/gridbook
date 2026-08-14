"""Comparison baseline vs direct MXFP4 on synthetic expert-like tensors."""
import torch
import pytest

from research.mxfp4_cb.format import Mxfp4CbFormat
from research.mxfp4_cb.e2m1 import e2m1_encode, e2m1_decode
from research.mxfp4_cb.codec import encode_mxfp4_cb, decode_mxfp4_cb
from research.mxfp4_cb.baseline import baseline_mxfp4_encode_decode, reconstruction_metrics


def _product_cb(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w0 = (k + 1) // 2
    w1 = k // 2
    cb0 = e2m1_decode(e2m1_encode(torch.randn((1 << w0, 4), generator=g))).to(torch.float32)
    cb1 = e2m1_decode(e2m1_encode(torch.randn((1 << w1, 4), generator=g))).to(torch.float32)
    return [cb0, cb1]


@pytest.mark.parametrize("k", [12, 16, 20])
def test_codebook_vs_direct_mxfp4_quality(k):
    fmt = Mxfp4CbFormat(k=k)
    rows, K = 8, 512
    torch.manual_seed(k)
    # expert-like: heavy-tailed per-channel
    w = torch.randn(rows, K) * 0.05
    # inject outlier channels like MoE down_proj
    w[:, ::32] *= 6
    w_bf16 = w.to(torch.bfloat16).to(torch.float32)

    cbs = _product_cb(k, seed=k)
    qw = encode_mxfp4_cb(w_bf16, fmt, cbs)
    w_cb = decode_mxfp4_cb(qw, fmt, cbs, rows, K).to(torch.float32)
    w_mx = baseline_mxfp4_encode_decode(w_bf16).to(torch.float32)

    m_cb = reconstruction_metrics(w_bf16, w_cb)
    m_mx = reconstruction_metrics(w_bf16, w_mx)

    # Direct MXFP4 should be significantly more accurate (stores 4 bits/weight)
    # than a k<=20 codebook at ~1.5-2.75 bpw.  We don't assert a hard threshold,
    # we assert the relationship and that both are finite / not catastrophic.
    assert m_cb["rel_l2"] < 1.0, f"k={k} cb rel_l2 {m_cb}"
    assert m_mx["rel_l2"] < 1.0
    # At k=20, codebook should be closer to direct than at k=12
    # Just sanity: codebook error > direct error for low k (compression costs)
    # Allow equality at high k with generous tolerance.
    # For k=12, cb should be worse than direct
    if k == 12:
        assert m_cb["rel_l2"] > m_mx["rel_l2"], f"k=12 expected cb worse than direct: cb {m_cb['rel_l2']:.4f} vs mx {m_mx['rel_l2']:.4f}"

    # Output-space divergence check: random activation matmul
    x = torch.randn(16, K) * 0.1
    y_orig = x @ w_bf16.t()
    y_cb = x @ w_cb.t()
    y_mx = x @ w_mx.t()
    out_cb = float((y_cb - y_orig).norm() / y_orig.norm().clamp_min(1e-12))
    out_mx = float((y_mx - y_orig).norm() / y_orig.norm().clamp_min(1e-12))
    assert out_cb < 1.0 and out_mx < 1.0
    # Record for honest assessment (printed when -s)
    print(f"\nk={k} bpw={fmt.bpw:.3f}  cb rel_l2={m_cb['rel_l2']:.4f} mx rel_l2={m_mx['rel_l2']:.4f}  out_cb={out_cb:.4f} out_mx={out_mx:.4f}")


def test_compression_claim():
    """Byte accounting vs direct MXFP4: report numbers, assert strict saving."""
    for k in (12, 16, 20, 24):
        fmt = Mxfp4CbFormat(k=k)
        direct_bpw = 4.0 + 8 / 32  # 4.25
        assert fmt.bpw < direct_bpw, f"k={k} {fmt.bpw} not < {direct_bpw}"
        ratio = direct_bpw / fmt.bpw
        print(f"k={k}: type_size={fmt.type_size} bpw={fmt.bpw:.3f} saving {ratio:.2f}x vs direct MXFP4")
        assert ratio > 1.5 if k <= 16 else ratio > 1.2
