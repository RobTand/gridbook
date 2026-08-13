"""MXFP8 dense kernel correctness (GPU image only; skips without the ext).

The full audit (real DSV4-Flash body tensors, seven distinct shapes, worst
rel-Frobenius 5.9e-5) ran against the checkpoint on the GB10 and is recorded
in ``source_passthrough.py``; these tests keep the fast, checkpoint-free core
of it pinned in CI: SF-plane offsets from the mainloop's own layout, synthetic
kernel-vs-oracle parity, and the mathematical DeepSeek block-128 embedding.
That embedding remains reference coverage; serving block128 source weights is
owned by the separate W8A16 lane and does not enter this W8A8 kernel.
"""
import pytest

torch = pytest.importorskip("torch")

from gridbook import cuda_ext as ce  # noqa: E402
from gridbook.mxfp8 import (  # noqa: E402
    broadcast_block128_scales,
    dequant_mxfp8,
    fill_sf_plane,
    mxfp8_reference_mm,
    quantize_mxfp8,
)

if not torch.cuda.is_available():
    pytest.skip("MXFP8 dense kernel needs CUDA", allow_module_level=True)

ext = ce.get_mxfp8_dense_ext()
if ext is None:
    pytest.skip("MXFP8 dense extension unavailable on this device/toolchain",
                allow_module_level=True)


def _plane(sf, rows, k, is_b):
    offs = ext.mxfp8_sf_offsets(rows, k, is_b).cuda()
    return fill_sf_plane(sf.cuda(), offs,
                         int(ext.mxfp8_sf_plane_numel(rows, k)))


@pytest.mark.parametrize("rows,k", [(136, 576), (128, 128), (257, 1024)])
@pytest.mark.parametrize("is_b", [False, True])
def test_sf_offsets_bijective_within_padded_plane(rows, k, is_b):
    offs = ext.mxfp8_sf_offsets(rows, k, is_b)
    assert offs.numel() == rows * (k // 32)
    assert int(torch.unique(offs).numel()) == offs.numel()
    assert int(offs.max()) < int(ext.mxfp8_sf_plane_numel(rows, k))
    assert int(offs.min()) >= 0


@pytest.mark.parametrize("m,n,k", [(128, 128, 256), (1, 128, 512),
                                   (5, 96, 576), (200, 264, 1024)])
def test_kernel_matches_fp32_oracle(m, n, k):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    a_q, a_sf = quantize_mxfp8(a)
    b_q, b_sf = quantize_mxfp8(b)
    y = ext.mxfp8_dense_mm(a_q, _plane(a_sf, m, k, False),
                           b_q, _plane(b_sf, n, k, True))
    ref = mxfp8_reference_mm(a_q.cpu(), a_sf.cpu(), b_q.cpu(), b_sf.cpu())
    err = ((y.cpu().float() - ref.float()).norm()
           / ref.float().norm().clamp_min(1e-30)).item()
    assert err < 5e-3, f"kernel-vs-oracle rel_fro {err} at {(m, n, k)}"


def test_deepseek_block128_embedding_end_to_end():
    """Reference-only: block weights -> broadcast scales -> swizzled plane ->
    kernel, judged against the block-dequant oracle (NOT the broadcast one),
    so the mathematical embedding is verified without claiming serving
    ownership for this W8A8 lane."""
    torch.manual_seed(1)
    n, k, m = 136, 576, 64
    w_q = (torch.randn(n, k) * 64).to(torch.float8_e4m3fn)
    s_exp = torch.randint(112, 132, ((n + 127) // 128, (k + 127) // 128),
                          dtype=torch.uint8)
    scale = torch.exp2(s_exp.float() - 127.0)
    w_ref = w_q.float() * scale.repeat_interleave(128, 0)[:n]\
        .repeat_interleave(128, 1)[:, :k]
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a_q, a_sf = quantize_mxfp8(a)
    sf_rm = broadcast_block128_scales(s_exp, n, k)
    y = ext.mxfp8_dense_mm(a_q, _plane(a_sf, m, k, False),
                           w_q.cuda(), _plane(sf_rm, n, k, True))
    ref = (dequant_mxfp8(a_q.cpu(), a_sf.cpu()) @ w_ref.t()).to(torch.bfloat16)
    err = ((y.cpu().float() - ref.float()).norm()
           / ref.float().norm().clamp_min(1e-30)).item()
    assert err < 5e-3, f"embedding chain rel_fro {err}"


def test_kernel_refuses_misaligned_and_mismatched_inputs():
    a = torch.zeros(8, 64, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(16, 64, device="cuda", dtype=torch.float8_e4m3fn)
    sfa = torch.zeros(int(ext.mxfp8_sf_plane_numel(8, 64)),
                      device="cuda", dtype=torch.uint8)
    sfb = torch.zeros(int(ext.mxfp8_sf_plane_numel(16, 64)),
                      device="cuda", dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        ext.mxfp8_dense_mm(torch.zeros(8, 48, device="cuda",
                                       dtype=torch.float8_e4m3fn),
                           sfa, b, sfb)
    with pytest.raises(RuntimeError, match="padded swizzled plane"):
        ext.mxfp8_dense_mm(a, sfa[:-1], b, sfb)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        ext.mxfp8_dense_mm(a, sfa,
                           torch.zeros(12, 64, device="cuda",
                                       dtype=torch.float8_e4m3fn),
                           torch.zeros(int(ext.mxfp8_sf_plane_numel(12, 64)),
                                       device="cuda", dtype=torch.uint8))
