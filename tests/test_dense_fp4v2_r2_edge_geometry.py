"""R2 bit-identity across EDGE geometry (the default-ON safety net).

``PRISMAQUANT_CB_FP4V2_DENSE_R2`` is ON by default since 0.8.13. The flip is
justified as PERF-ONLY, and that justification rests entirely on the two
instantiations being bit-identical: if they are, no shipped artifact's outputs,
KL or PPL can move and no artifact needs re-validation.

``test_dense_fp4v2_backport.py`` gates the claim on k in {13,16,20} x
K in {512,1024} x M in {1,8}. That leaves the surface R2 actually changed
under-tested: the aligned-down u64 burst staging has a LAST-SUPERBLOCK
fallback and a slot-bound guard, and ``type_size`` moves with every rung, so
the staging boundary lands differently for each k. This module sweeps that
boundary -- every shipped rung, both warp branches (n_sb % 8 vs % 4), odd and
tiny row counts, non-power-of-2 M (runtime ``m`` tail), and both schedules.

Origin: dq-runs/r2-kernel-2026-08-23 ran this as a 1728-config fuzz with zero
mismatches before the default was flipped. Kept here so a future edit to the
staging path cannot silently break the property the flip depends on.
"""
import itertools

import pytest
import torch

codec = pytest.importorskip("gridbook.codec", reason="gridbook plugin not importable")
from gridbook.cuda_ext import get_ext  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)
ext = get_ext()
if ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)", allow_module_level=True)

DEV = "cuda"
R2_FLAG = "PRISMAQUANT_CB_FP4V2_DENSE_R2"
SCHED_FLAG = "PRISMAQUANT_CB_FP4V2_SCHED"


def _prep(pq, k, N, K, seed):
    cb = pq._resolve_codebook(k, "fp4", "product", None, torch.device(DEV))
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(1, N, K, generator=g) * 0.02).to(DEV)
    fields = pq.nvfp4_cb_fields(w, k, grid="fp4", mode="product", codebook=cb,
                                scale_coding="two_tier", encode_tier="fast")
    b = pq.nvfp4_cb_assemble_bytes(fields, k, grid="fp4", mode="product")
    ts = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")
    n_sb = K // codec.SUPERBLOCK
    packed = b.reshape(1, N, n_sb * ts)[0].contiguous().to(DEV)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    return dict(qwp=codec.pad_qweight(packed),
                cb_flat=codec.build_flat_codebook(subs),
                compose=codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV),
                row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
                N=N, K=K, k=k, n_sub=2, ts=ts)


def _run(p, xq):
    return ext.cb_gemv_fp4_v2(xq, p["qwp"], p["cb_flat"], p["row_off"],
                              p["compose"], p["N"], p["K"], p["k"],
                              p["n_sub"], p["ts"])


# n_sb = K >> 8 -> 2, 4, 8, 12: covers the use4 branch (n_sb % 8 != 0 and
# n_sb % 4 == 0 and n_sb < 48) and the 8-warp branch on both sides.
@pytest.mark.parametrize("K", [512, 1024, 2048, 3072])
@pytest.mark.parametrize("k", [12, 13, 14, 15, 16, 17, 18, 19, 20])
def test_r2_bit_identical_across_edge_geometry(k, K, monkeypatch):
    """Every shipped rung x both warp branches x odd/tiny N x non-power-of-2 M
    x both schedules: the R2 and legacy instantiations agree bitwise."""
    pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    for N in (1, 17, 48, 96):
        p = _prep(pq, k, N, K, seed=7000 + k * 97 + K + N)
        for M, sched in itertools.product((1, 3, 5, 8, 15, 16), (None, "db")):
            torch.manual_seed(k * 31 + K + N + M)
            x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
            xq = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
            if sched is None:
                monkeypatch.delenv(SCHED_FLAG, raising=False)
            else:
                monkeypatch.setenv(SCHED_FLAG, sched)
            monkeypatch.setenv(R2_FLAG, "0")
            y_legacy = _run(p, xq).contiguous().clone()
            monkeypatch.setenv(R2_FLAG, "1")
            y_r2 = _run(p, xq).contiguous().clone()
            a, b = y_legacy.view(torch.uint16), y_r2.view(torch.uint16)
            assert torch.equal(a, b), (
                f"R2 differs from legacy at k={k} K={K} N={N} M={M} "
                f"sched={sched}: {(a != b).sum().item()} of {a.numel()} "
                f"elements -- the default-ON flip's perf-only justification "
                f"does not hold at this geometry")
    monkeypatch.delenv(R2_FLAG, raising=False)
    monkeypatch.delenv(SCHED_FLAG, raising=False)


def test_r2_default_is_off():
    """R2 must stay default OFF until the n_sb < WARPS regression is fixed.

    A 2026-08-23 flip to default-ON was reverted: R2 wins on large n_sb but
    LOSES at M=1 when n_sb < WARPS (+9.14% at k=12/K=768/n_sb=3 against a
    0.36% control spread) because the burst staging never amortizes over a
    single warp iteration. Guards against re-flipping without either a kernel
    early-exit for that regime or an explicit measured n_sb crossover.
    """
    import pathlib
    from gridbook.cuda_ext import csrc_dir
    # The source the loader actually compiles: the checkout's in-tree, the
    # wheel's when the tests are staged outside the checkout (release gate).
    src = pathlib.Path(csrc_dir()) / "cb_gemv.cu"
    text = src.read_text()
    assert 'pq_env_bool01("PRISMAQUANT_CB_FP4V2_DENSE_R2", false)' in text, (
        "PRISMAQUANT_CB_FP4V2_DENSE_R2 defaults to ON, but the documented "
        "measurement says it regresses at M=1 with n_sb < WARPS. If the "
        "kernel now early-exits that regime, update this test WITH the "
        "measurement that justifies it")
