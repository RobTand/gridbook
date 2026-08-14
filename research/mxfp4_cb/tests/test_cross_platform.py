"""Cross-platform hypothesis: stability and divergence.

Runs the experiment from research.mxfp4_cb.cross_platform and asserts the
strong hypothesis (same indices coherent) FAILS, quantifying by how much.
Also verifies deterministic projection.
"""
import torch
import pytest

from research.mxfp4_cb.cross_platform import (
    make_canonical_codebook,
    project_to_physical,
    cross_platform_report,
    _gen_synthetic_weight,
)
from research.mxfp4_cb.format import Mxfp4CbFormat


def test_projection_is_deterministic():
    canon = make_canonical_codebook(16, seed=7)
    a_nv, a_mx = project_to_physical(canon)
    b_nv, b_mx = project_to_physical(canon)
    for a, b in zip(a_nv, b_nv):
        assert torch.equal(a, b)
    for a, b in zip(a_mx, b_mx):
        assert torch.equal(a, b)
    # Canonical->physical snaps to E2M1, so values differ from canonical but are identical across platforms
    for c, p in zip(canon, a_nv):
        assert not torch.equal(c, p)  # projection does change values (unless canonical was already on grid)


def test_stability_and_divergence():
    rep = cross_platform_report(k=16, rows=4, K=512, seed=0)
    print("\nCross-platform report k=16:")
    for k, v in rep.items():
        if k != "explanation":
            print(f"  {k}: {v}")
    print(f"  explanation: {rep['explanation']}")

    # Hypothesis: identical indices coherent => stability ~1.0 and cross weight divergence ~0
    # Reality: scales differ (E4M3 16 groups vs E8M0 8 groups), so assignments diverge.
    # We assert the strong hypothesis FAILS quantitatively:
    assert rep["stability_frac"] < 0.95, (
        f"unexpected high stability {rep['stability_frac']:.3f} — hypothesis would be unexpectedly coherent"
    )
    assert rep["instability_frac"] > 0.02, "need at least 2% instability to demonstrate failure"
    # Cross-decoding with wrong scales diverges
    assert rep["weight_rel_l2_cross_nv"] > 0.1 or rep["weight_rel_l2_cross_mx"] > 0.1, (
        f"cross weight divergence too small: {rep}"
    )
    assert rep["output_rel_l2_cross_nv"] > 0.05, f"output cross divergence too small: {rep['output_rel_l2_cross_nv']}"
    # Optimal-vs-optimal also diverges (different scale grids)
    assert rep["weight_rel_l2_opt_vs_opt"] > 0.05


@pytest.mark.parametrize("k", [12, 20])
def test_stability_varies_with_k(k):
    rep = cross_platform_report(k=k, rows=2, K=256, seed=k)
    # At any k, some instability remains because scale grouping differs
    assert rep["instability_frac"] > 0.01


def test_synthetic_weight_determinism():
    a = _gen_synthetic_weight(2, 256, seed=123)
    b = _gen_synthetic_weight(2, 256, seed=123)
    assert torch.equal(a, b)
    c = _gen_synthetic_weight(2, 256, seed=124)
    assert not torch.equal(a, c)


def test_minimum_metadata_proposal():
    """The failure mode implies per-platform streams are needed.

    This test documents the proposal: 1 byte flag per superblock is insufficient
    — you need full re-encoded indices.  We verify that byte-count math holds.
    """
    for k in (16, 20):
        fmt = Mxfp4CbFormat(k=k)
        # Single shared stream would be type_size per SB.
        # Per-platform streams would be 2 * (4k + scale_bytes_per_platform)
        # For MXFP4 scale_bytes=8, NVFP4 v1 scale_bytes=16, so naive dual = 4k+8 + 4k+16
        # Minimal metadata if we must keep both: at least flag + second stream.
        shared = fmt.type_size
        # Demonstrate that storing only a 1-bit selector without second stream cannot rescue cross divergence
        # (assert cross divergence > threshold already above).  So minimum is:
        #   per-superblock: 4k bytes NVFP4 indices + 16 B scales  +  4k bytes MX indices + 8 B scales
        # or equivalently, re-encode at export time and ship per-target-platform artifact.
        # We just check arithmetic:
        dual = (4 * k + 16) + (4 * k + 8)
        assert dual > shared
        overhead_vs_shared = dual / shared - 1
        print(f"k={k}: shared {shared}B, dual {dual}B, overhead {overhead_vs_shared:.1%}")
