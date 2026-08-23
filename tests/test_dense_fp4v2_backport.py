"""Bit-exactness gate for the dense fp4-v2 round-2 backport.

``PRISMAQUANT_CB_FP4V2_DENSE_R2=1`` selects a second instantiation of
``cb_gemv_fp4_v2_kernel`` (template flag ``R2BACKPORT``) that ports the grouped
MoE fp4-v2 kernel's round-2 load schedule onto the dense kernel: the
predicated third stage-word read, packed uint2 codebook gathers, and the
aligned-down u64 burst staging. The port changes only which loads are issued,
never the arithmetic, so the contract is BITWISE: every
(``PRISMAQUANT_CB_FP4V2_SCHED`` x flag) arm must produce identical bits on
identical inputs, and the legacy arm must still match the independent
pure-Torch decode within the file-wide 1-bf16-ULP gate.

The flag is parsed strictly host-side in the extension launcher (the
``lane_select.latched_bool`` contract: only '', '0', '1'; anything else
raises), and it is re-read per launch from the environment, so a test may
flip it between calls inside one process. The final arm ordering also covers
both decode-contract chains (per-weight round and raw-value) and the signed
(n_sub=1) layout, whose gather is shared with the grouped kernel.
"""
import pytest
import torch

codec = pytest.importorskip(
    "gridbook.codec",
    reason="gridbook plugin not importable")
from gridbook.cuda_ext import get_ext  # noqa: E402
from cb_torch_reference import cb_linear_reference  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

ext = get_ext()
if ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)",
                allow_module_level=True)

DEV = "cuda"
R2_FLAG = "PRISMAQUANT_CB_FP4V2_DENSE_R2"
SCHED_FLAG = "PRISMAQUANT_CB_FP4V2_SCHED"


def _assert_reference_close(y_cuda, y_reference, tag):
    """Same gate as test_cuda_gemv.py: at most one output-rounding bf16 ULP
    elementwise plus a norm backstop (only summation order may differ)."""
    a, b = y_cuda.float(), y_reference.float()
    d = (a - b).abs()
    tol = torch.maximum(a.abs(), b.abs()) * 2.0 ** -7 + 1e-5
    nbad = int((d > tol).sum())
    assert nbad == 0, (
        f"{tag}: {nbad} elements beyond 1 bf16 output ULP "
        f"(max delta {d.max():.3e} vs tol {tol.flatten()[d.argmax()]:.3e})")
    rel = d.norm() / b.norm().clamp_min(1e-6)
    assert rel <= 1e-3, f"{tag}: norm backstop rel {rel:.3e}"


def _encode_dense(pq, k, N, K, cb, seed):
    """(N, K) weight -> fp4 two-tier v2 on-disk bytes via the REAL encoder
    (reusing the stack encoder with E=1), so every (super, sub) scale pair is
    legal by construction."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(1, N, K, generator=g) * 0.02).to(DEV)
    fields = pq.nvfp4_cb_fields(w, k, grid="fp4", mode="product", codebook=cb,
                                scale_coding="two_tier", encode_tier="fast")
    b = pq.nvfp4_cb_assemble_bytes(fields, k, grid="fp4", mode="product")
    ts = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")
    n_sb = K // codec.SUPERBLOCK
    return b.reshape(1, N, n_sb * ts)[0].contiguous().to(DEV), ts


def _prep(pq, k, N, K, seed, cb=None):
    if cb is None:
        cb = pq._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    packed, ts = _encode_dense(pq, k, N, K, cb, seed)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    return dict(qwp=codec.pad_qweight(packed),
                cb_flat=codec.build_flat_codebook(subs),
                compose=codec.build_compose_table(
                    codec.TWO_TIER_SUB_TABLE).to(DEV),
                row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
                N=N, K=K, k=k, n_sub=2, ts=ts)


def _run(p, xq):
    return ext.cb_gemv_fp4_v2(xq, p["qwp"], p["cb_flat"], p["row_off"],
                              p["compose"], p["N"], p["K"], p["k"],
                              p["n_sub"], p["ts"])


def _run_both_r2_arms(monkeypatch, p, xq, sched):
    """Run the registered entry under both flag settings for one schedule and
    return {(sched, r2): y} with the env restored afterward."""
    if sched is None:
        monkeypatch.delenv(SCHED_FLAG, raising=False)
    else:
        monkeypatch.setenv(SCHED_FLAG, sched)
    outs = {}
    for r2 in ("0", "1"):
        monkeypatch.setenv(R2_FLAG, r2)
        outs[(sched, r2)] = _run(p, xq).contiguous()
    monkeypatch.delenv(R2_FLAG, raising=False)
    monkeypatch.delenv(SCHED_FLAG, raising=False)
    return outs


def _assert_arms_bit_equal(outs, tag):
    legacy = outs[(None, "0")].view(torch.uint16)
    for key, y in outs.items():
        assert torch.equal(y.view(torch.uint16), legacy), (
            f"{tag}: arm sched={key[0]} R2={key[1]} differs from the legacy "
            f"kernel at {(y.view(torch.uint16) != legacy).sum().item()} of "
            f"{legacy.numel()} elements")
    return legacy


def _assert_matches_torch(p, xq, y, tag):
    ref = cb_linear_reference(
        xq, p["qwp"], p["cb_flat"], p["row_off"], torch.zeros(1, device=DEV),
        p["compose"], N=p["N"], K=p["K"], k_bits=p["k"], n_sub=p["n_sub"],
        type_size=p["ts"], is_fp4=True, is_v2=True)
    _assert_reference_close(y, ref, tag)


@pytest.mark.parametrize("K", [512, 1024])
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("k", [13, 16, 20])
def test_r2_backport_bit_exact_product(k, M, K, monkeypatch):
    """Product rungs, both warp branches (K=512 -> 8 warps, K=1024 -> 4):
    all four schedule x flag arms are bit-identical, and the legacy arm keeps
    matching the independent Torch decode."""
    pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    p = _prep(pq, k, N=96, K=K, seed=1000 + k + K)
    torch.manual_seed(k + K + M)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    outs = {}
    for sched in (None, "db"):
        outs.update(_run_both_r2_arms(monkeypatch, p, xq, sched))
    legacy = _assert_arms_bit_equal(outs, f"dense fp4-v2 R2 k={k} K={K} M={M}")
    _assert_matches_torch(p, xq, outs[(None, "0")],
                          f"dense fp4-v2 R2 k={k} K={K} M={M} vs Torch")


def test_r2_backport_bit_exact_v2_contract(monkeypatch):
    """PRISMAQUANT_CB_DECODE_CONTRACT=v2 chain (raw codebook values, scale on
    the lane partial): the packed-gather branch the backport actually targets
    in serving."""
    pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    monkeypatch.setenv("PRISMAQUANT_CB_DECODE_CONTRACT", "v2")
    p = _prep(pq, k=16, N=96, K=512, seed=77)
    torch.manual_seed(7)
    x = torch.randn(2, 512, dtype=torch.bfloat16, device=DEV)
    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    outs = {}
    for sched in (None, "db"):
        outs.update(_run_both_r2_arms(monkeypatch, p, xq, sched))
    _assert_arms_bit_equal(outs, "dense fp4-v2 R2 contract v2")


def test_r2_backport_bit_exact_fused_row_offset(monkeypatch):
    """Two roles with DIFFERENT codebooks (the qkv/gate_up fusion mechanism):
    the R2 uint2 gather must add each row's nonzero codebook base. Role B is
    a scaled copy (still bf16-exact) so a base-ignoring bug cannot hide."""
    pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    k, K = 16, 512
    cb_a = pq._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    cb_b = tuple(t * 1.5 for t in cb_a)
    Na, Nb = 64, 32
    pa, ts = _encode_dense(pq, k, Na, K, cb_a, seed=11)
    pb, _ = _encode_dense(pq, k, Nb, K, cb_b, seed=22)
    flat_a = codec.build_flat_codebook(list(cb_a))
    flat_b = codec.build_flat_codebook(list(cb_b))
    assert flat_a.numel() % 4 == 0, (
        "role block sizes are whole 4-element sub-table concatenations; the "
        "uint2 gather alignment depends on it")
    p = dict(qwp=codec.pad_qweight(torch.cat([pa, pb], dim=0).contiguous()),
             cb_flat=torch.cat([flat_a, flat_b]).contiguous(),
             compose=codec.build_compose_table(
                 codec.TWO_TIER_SUB_TABLE).to(DEV),
             row_off=torch.cat([
                 torch.zeros(Na, dtype=torch.int32, device=DEV),
                 torch.full((Nb,), flat_a.numel(), dtype=torch.int32,
                            device=DEV)]),
             N=Na + Nb, K=K, k=k, n_sub=2, ts=ts)
    torch.manual_seed(3)
    x = torch.randn(4, K, dtype=torch.bfloat16, device=DEV)
    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    outs = {}
    for sched in (None, "db"):
        outs.update(_run_both_r2_arms(monkeypatch, p, xq, sched))
    legacy = _assert_arms_bit_equal(outs, "dense fp4-v2 R2 fused row offset")
    _assert_matches_torch(p, xq, outs[(None, "0")],
                          "dense fp4-v2 R2 fused row offset vs Torch")


def test_r2_flag_rejects_invalid_spelling(monkeypatch):
    """The flag carries the latched_bool parsing contract: a typo raises and
    names the accepted spellings instead of silently running the legacy arm."""
    pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    p = _prep(pq, k=16, N=32, K=512, seed=99)
    torch.manual_seed(9)
    x = torch.randn(1, 512, dtype=torch.bfloat16, device=DEV)
    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    monkeypatch.setenv(R2_FLAG, "true")
    with pytest.raises(RuntimeError, match=R2_FLAG):
        _run(p, xq)
    monkeypatch.delenv(R2_FLAG, raising=False)


def test_unpadded_weights_fail_loud(monkeypatch):
    """The R2 burst stage overruns a superblock's last byte by up to 7 bytes;
    on a row's final superblock that lands in the row's pad slack, so the
    launcher requires pad_qweight's >= 8-byte read slack for BOTH arms and
    raises naming it instead of silently reading out of bounds. Exact-size
    rows (stride == n_sb*type_size) must be rejected; the padded tensor from
    the same bytes must be accepted."""
    pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    k, K, N = 16, 512, 32
    cb = pq._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    packed, ts = _encode_dense(pq, k, N, K, cb, seed=321)
    assert packed.stride(0) == (K // codec.SUPERBLOCK) * ts, "exact-size rows"
    torch.manual_seed(11)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=DEV)
    xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    base = dict(cb_flat=codec.build_flat_codebook(list(cb)),
                compose=codec.build_compose_table(
                    codec.TWO_TIER_SUB_TABLE).to(DEV),
                row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
                N=N, K=K, k=k, n_sub=2, ts=ts)
    monkeypatch.setenv(R2_FLAG, "1")
    with pytest.raises(RuntimeError, match="pad_qweight"):
        ext.cb_gemv_fp4_v2(xq, packed.contiguous(), base["cb_flat"],
                           base["row_off"], base["compose"], N, K, k,
                           base["n_sub"], ts)
    padded = codec.pad_qweight(packed)
    y = ext.cb_gemv_fp4_v2(xq, padded, base["cb_flat"], base["row_off"],
                           base["compose"], N, K, k, base["n_sub"], ts)
    assert y.shape == torch.Size([1, N])
    monkeypatch.delenv(R2_FLAG, raising=False)
