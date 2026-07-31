"""Two-tier v2 scale coding — plugin compose path (docs/lanes/nvfp4-cb/
two-tier-scale-spec.md §4). The kernel composes the E4M3 scale plane in-register
from the packed 9 bytes (1 E8M0 super + 8 sub nibble bytes) and must match
`nvfp4_cb_reconstruct` bit-exactly. Venv (triton + prismaquant, no vLLM); the
dispatch test is vLLM-guarded.

  PYTHONPATH=/home/rob/prismaquant:/home/rob/prismaquant/plugins/gridbook \
    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest \
    plugins/gridbook/tests/test_two_tier_v2.py -q
"""
import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
kernels = pytest.importorskip("gridbook.kernels")
fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")

cb_decode_linear = kernels.cb_decode_linear
DEV = "cuda"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable")


def test_compose_table_matches_reference():
    """codec.build_compose_table == nvfp4_cb_formats._two_tier_tables compose
    (the reference the kernel gathers), exact on every legal (E, c)."""
    _, compose_ref, legal = fmt._two_tier_tables("cpu")          # (256,16)
    mine = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).reshape(256, 16)
    assert torch.equal(mine[legal], compose_ref[legal])
    # our table constant must match the reference's.
    assert tuple(codec.TWO_TIER_SUB_TABLE) == tuple(fmt.TWO_TIER_SUB_TABLE)


@requires_cuda
@pytest.mark.parametrize("k", [13, 16, 17, 18, 20, 24])
@pytest.mark.parametrize("M", [1, 17])
def test_v2_decode_matches_reconstruct(k, M):
    """Pack a weight with two-tier v2, decode-GEMM via the kernel's in-kernel
    compose, and match nvfp4_cb_reconstruct @ x (bf16 accum)."""
    torch.manual_seed(k)
    rows, in_f = 128, 512                    # in % 256 == 0
    w = torch.randn(rows, in_f, device=DEV) * 0.05
    cb = fmt._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    packed, fields = fmt.nvfp4_cb_pack(
        w, k, grid="fp4", mode="product", codebook=cb,
        scale_coding=fmt.SCALE_CODING_TWO_TIER)
    ts = fmt.nvfp4_cb_type_size(k, "fp4", fmt.SCALE_CODING_TWO_TIER)
    assert packed.shape[1] == (in_f // 256) * ts and ts == 4 * k + 9

    w_ref = fmt.nvfp4_cb_reconstruct(fields, k, grid="fp4", mode="product",
                                     codebook=cb).to(torch.bfloat16)
    x = torch.randn(M, in_f, dtype=torch.bfloat16, device=DEV)
    y_ref = x.float() @ w_ref.float().t()

    cb_flat = codec.build_flat_codebook([t.to(DEV) for t in cb])
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV)
    row_off = torch.zeros(rows, dtype=torch.int32, device=DEV)
    qwp = codec.pad_qweight(packed)
    y = cb_decode_linear(x, qwp, cb_flat, row_off, torch.zeros(1, device=DEV),
                         compose, N=rows, K=in_f, k_bits=k, n_sub=2,
                         type_size=ts, is_fp4=True, is_v2=True)
    rel = (y.float() - y_ref).norm() / y_ref.norm().clamp_min(1e-6)
    assert rel <= 1e-2, f"k={k} M={M}: v2 decode rel err {rel:.4e}"


@requires_cuda
def test_v2_scale_extraction_bitexact():
    """Mirror the kernel's super/sub byte extraction in torch and confirm the
    composed scales equal the reference fields['scales'] exactly."""
    k, rows, in_f = 16, 64, 512
    torch.manual_seed(1)
    w = torch.randn(rows, in_f, device=DEV) * 0.05
    cb = fmt._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    packed, fields = fmt.nvfp4_cb_pack(
        w, k, grid="fp4", mode="product", codebook=cb,
        scale_coding=fmt.SCALE_CODING_TWO_TIER)
    ts = 4 * k + 9
    n_sb = in_f // 256
    blk = packed.reshape(rows, n_sb, ts)
    super_e = blk[:, :, 4 * k].to(torch.int64)                  # (rows, n_sb)
    sub = blk[:, :, 4 * k + 1:4 * k + 9].to(torch.int64)        # (rows, n_sb, 8)
    lo = sub & 0xF
    hi = (sub >> 4) & 0xF
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, n_sb, 16)
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV).reshape(
        256, 16)
    got = compose[super_e[..., None].expand_as(codes), codes].reshape(
        rows, n_sb * 16)
    assert torch.equal(got, fields["scales"].to(DEV).float())


@requires_cuda
@pytest.mark.parametrize("k", [13, 16, 24])
def test_v2_transient_weight_matches_reconstruct(k):
    """fp4-v2 transient prefill: the expanded [N,K] bf16 weight (value × composed
    v2 scale) matches nvfp4_cb_reconstruct, so F.linear == reconstruct @ x."""
    expand = pytest.importorskip("gridbook.expand")
    import torch.nn.functional as F
    torch.manual_seed(k)
    rows, in_f, M = 128, 512, 40
    w = torch.randn(rows, in_f, device=DEV) * 0.05
    cb = fmt._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    packed, fields = fmt.nvfp4_cb_pack(
        w, k, grid="fp4", mode="product", codebook=cb,
        scale_coding=fmt.SCALE_CODING_TWO_TIER)
    w_ref = fmt.nvfp4_cb_reconstruct(fields, k, grid="fp4", mode="product",
                                     codebook=cb).to(torch.bfloat16)
    cb_flat = codec.build_flat_codebook([t.to(DEV) for t in cb])
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV)
    qwp = codec.pad_qweight(packed)
    row_off = torch.zeros(rows, dtype=torch.int32, device=DEV)
    W = expand.expand_fp4_v2_to_weight(qwp, cb_flat, row_off, compose, rows,
                                       in_f, k, 2, 4 * k + 9)
    relw = (W.float() - w_ref.float()).norm() / w_ref.float().norm()
    assert relw <= 5e-3, f"k={k}: transient weight rel {relw:.4e}"
    x = torch.randn(M, in_f, dtype=torch.bfloat16, device=DEV)
    rel = (F.linear(x, W).float() - (x.float() @ w_ref.float().t())).norm() \
        / (x.float() @ w_ref.float().t()).norm()
    assert rel <= 1e-2


def test_v2_dispatch_flag():
    """A scheme with scale_coding.kind=='two_tier' -> is_v2; absence -> v1
    (both must serve; per-layer, so a mixed v1/v2 artifact dispatches each)."""
    pytest.importorskip("vllm")
    from gridbook.linear import PrismaQuantCBLinearMethod
    base = {"grid": "fp4", "mode": "product", "k": 16, "n_sub": 2}
    v1 = dict(base, type_size=80)
    v2 = dict(base, type_size=73, scale_coding={
        "kind": "two_tier", "sub_bits": 4, "super_bias": 127,
        "table": list(codec.TWO_TIER_SUB_TABLE)})
    assert PrismaQuantCBLinearMethod(None, v1, "x").is_v2 is False
    m2 = PrismaQuantCBLinearMethod(None, v2, "x")
    assert m2.is_v2 is True and m2._sub_table is not None
