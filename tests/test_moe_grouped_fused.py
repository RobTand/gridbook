"""Correctness gate for the GROUPED-FUSED MoE PREFILL path
(``moe.PrismaQuantCBMoEMethod._apply_prefill_grouped_fused``, round 1 of the
MoE fused campaign; ``PRISMAQUANT_CB_PREFILL=grouped_fused``).

The grouped-fused path removes the stock path's HBM e4m3 expand round-trip by
decoding each expert's packed CB rows inside the CUTLASS prologue
(``cb_fused_prefill_mm_scaled``). Weights decode bit-exactly and the activation
QDQ is the stock path's own per-token fp8 dynamic, so the two differ only by
GEMM accumulation + cross-expert combine reassociation — the suite's
REASSOCIATION-CLASS 2e-2 contract (same bound as loop-vs-batched).

Run scopes:

Round 2 (``_apply_prefill_grouped_fused_v2``) replaces R1's host loop over
experts with ONE ``cb_fused_moe_grouped`` launch per projection stage over a
TileM-padded, expert-sorted row collective. Its parity reference is R1, not
stock: the two share the routing order and the QDQ, differing only in how the
padded/segment GEMMs and the combine reassociate.

Run scopes:

* ``-k routing`` (build venv, NO vLLM/CUDA needed): the sort/boundary property
  the one-sync R1 routing rests on, plus the R2 padded-routing construction
  (``gridbook.moe_routing`` is torch-only for exactly this reason).
* everything else (serving container: vLLM + CUDA + the fused extension):
    docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
      --entrypoint bash vllm-node-tf5-cu132-lfm:latest -c 'pip install -q pytest; \\
      PYTHONPATH=/repo:/repo/plugins/gridbook python3 -m pytest \\
      /repo/plugins/gridbook/tests/test_moe_grouped_fused.py -v'
"""
import pytest
import torch

pytest.importorskip("gridbook.codec")
# Torch-only by design (see gridbook/moe_routing.py) so the ROUND-2 padded
# routing — all of R2's index arithmetic — is exercised in the build venv.
from gridbook.moe_routing import cb_grouped_pad_routing  # noqa: E402

from test_moe_batched_prefill import (  # noqa: E402
    DEV,
    _REL,
    _build,
    _report,
    _require_stack,
    _routing,
    _silu_act,
)


# --------------------------------------------------------------------------- #
# Routing property (CPU ok): stable expert-sort + cumsum boundaries reproduce   #
# exactly the loop path's per-expert row selection, in the loop's order.        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_routing_boundaries_match_loop_selection(topk):
    torch.manual_seed(0)
    T, E = 37, 11
    topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(T)])
    pair_expert = topk_ids.reshape(-1).to(torch.long)
    pair_token = torch.arange(T).repeat_interleave(topk)
    order = torch.argsort(pair_expert, stable=True)
    ptok_sorted = pair_token[order]
    counts = torch.bincount(pair_expert, minlength=E)
    bounds = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).tolist()

    assert bounds[-1] == T * topk
    for e in range(E):
        p0, p1 = bounds[e], bounds[e + 1]
        tok_idx, _slot = torch.where(topk_ids == e)     # the loop's selection
        assert torch.equal(ptok_sorted[p0:p1], tok_idx), (
            f"expert {e}: segment != loop selection (order or bounds wrong)")


def test_zero_row_experts_are_skippable_without_extra_syncs():
    """Empty experts show up as p1 == p0 on the ALREADY-fetched boundaries, so
    skipping them costs no additional device read."""
    E, topk = 8, 1
    topk_ids = torch.full((16, topk), 3, dtype=torch.long)
    counts = torch.bincount(topk_ids.reshape(-1), minlength=E)
    bounds = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).tolist()
    hit = [e for e in range(E) if bounds[e + 1] > bounds[e]]
    assert hit == [3]


# --------------------------------------------------------------------------- #
# ROUND 2 padded routing (CPU ok) — the construction the grouped kernel's        #
# per-tile expert selection rests on.                                           #
# --------------------------------------------------------------------------- #
def _ref_segments(topk_ids, E):
    """The loop path's per-expert row selection, in the loop's order."""
    return [torch.where(topk_ids == e)[0] for e in range(E)]


def _check_routing(topk_ids, E, tile_m):
    T, topk = topk_ids.shape
    P = T * topk
    cap = P // tile_m + E                       # the static capacity bound
    eids, row_src, is_pad, n_blocks = cb_grouped_pad_routing(
        topk_ids, E, tile_m)
    assert eids.shape == (cap,) and eids.dtype == torch.int32
    assert row_src.shape == (cap * tile_m,) and is_pad.shape == (cap * tile_m,)

    nb = int(n_blocks)
    assert nb <= cap, f"capacity bound violated: {nb} > {cap}"

    pair_expert = topk_ids.reshape(-1).to(torch.long)
    pair_token = torch.arange(T).repeat_interleave(topk)
    order = torch.argsort(pair_expert, stable=True)
    ptok_sorted = pair_token[order]
    counts = torch.bincount(pair_expert, minlength=E)

    # Live blocks carry a real expert; capacity past the total is flagged -1
    # and every one of its rows is padding.
    assert torch.equal(eids[nb:], torch.full((cap - nb,), -1,
                                             dtype=torch.int32))
    assert bool(is_pad[nb * tile_m:].all())
    assert bool((eids[:nb] >= 0).all())

    seen = {e: [] for e in range(E)}
    for b in range(nb):
        e = int(eids[b])
        sl = slice(b * tile_m, (b + 1) * tile_m)
        pad_b = is_pad[sl]
        # Padding only ever occupies the TAIL of an expert's LAST block.
        assert not bool(pad_b[:-1].bitwise_and(~pad_b[1:]).any()), \
            "padding is not contiguous at the tail of the block"
        seen[e].extend(int(v) for v in row_src[sl][~pad_b])

    for e in range(E):
        assert len(seen[e]) == int(counts[e])
        # Each expert's rows appear in STABLE token order, i.e. exactly the
        # loop's torch.where(topk_ids == e) selection.
        assert torch.equal(ptok_sorted[torch.tensor(seen[e], dtype=torch.long)]
                           if seen[e] else torch.empty(0, dtype=torch.long),
                           _ref_segments(topk_ids, E)[e])
        # ... and start on a block boundary: blocks per expert is exact.
        assert sum(1 for b in range(nb) if int(eids[b]) == e) == \
            -(-int(counts[e]) // tile_m)
    return nb, cap


@pytest.mark.parametrize("tile_m", [4, 8, 128])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_padded_routing_matches_loop_selection(tile_m, topk):
    torch.manual_seed(0)
    T, E = 37, 11
    topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(T)])
    _check_routing(topk_ids.to(torch.int32), E, tile_m)


def test_padded_routing_zero_row_experts_consume_no_blocks():
    """E-1 empty experts must cost zero tiles — otherwise a 256-expert layer
    launches 256 tiles for a one-expert prefill."""
    E, tile_m = 8, 4
    topk_ids = torch.full((16, 1), 3, dtype=torch.int32)
    eids, _row_src, _pad, n_blocks = cb_grouped_pad_routing(topk_ids, E, tile_m)
    assert int(n_blocks) == 4
    assert torch.equal(eids[:4], torch.full((4,), 3, dtype=torch.int32))


@pytest.mark.parametrize("seed", list(range(24)))
def test_padded_routing_capacity_bound_randomized(seed):
    """The static bound cap = P//tile_m + E is what makes the whole path
    host-read-free at build time; fuzz it over ragged/skewed routings."""
    g = torch.Generator().manual_seed(seed)
    E = int(torch.randint(1, 17, (1,), generator=g))
    topk = int(torch.randint(1, min(E, 6) + 1, (1,), generator=g))
    T = int(torch.randint(1, 40, (1,), generator=g))
    tile_m = int([1, 2, 4, 8, 128][seed % 5])
    # Skewed (with replacement) routing: the worst case for ragged block counts.
    ids = torch.randint(0, E, (T, topk), generator=g, dtype=torch.int32)
    nb, cap = _check_routing(ids, E, tile_m)
    assert nb <= cap


# --------------------------------------------------------------------------- #
# GPU parity: grouped_fused vs stock (the path it replaces).                    #
# --------------------------------------------------------------------------- #
def _require_fused(m, layer):
    _require_stack()
    if not m._gf_ok(layer):
        pytest.skip("fused CB extension / rung constraints unmet")


@pytest.mark.parametrize("dist", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [2, 4])
def test_grouped_fused_vs_stock_parity(dist, topk):
    _require_stack()
    m, layer, d = _build("fp8", seed=1)
    _require_fused(m, layer)
    act = _silu_act()
    T = 48
    ti, tw = _routing(T, d["E"], topk, dist, seed=7)
    torch.manual_seed(2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_gf = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    assert o_gf is not None, "grouped_fused returned None despite _gf_ok"
    assert o_gf.shape == o_stock.shape == (T, d["hidden"])
    rel = _report(f"gf-vs-stock[{dist},topk={topk}]", o_stock, o_gf)
    assert rel <= _REL, f"{dist}/topk={topk}: rel {rel:.3e} > {_REL}"


@pytest.mark.parametrize("T", [1, 3, 17, 129])
def test_grouped_fused_small_and_partial_tile_m(T):
    """No minimum M: an expert with a handful of rows must run through one
    partial CUTLASS tile, not crash and not need padding."""
    _require_stack()
    m, layer, d = _build("fp8", seed=3)
    _require_fused(m, layer)
    act = _silu_act()
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=5)
    torch.manual_seed(4)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_gf = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    rel = _report(f"gf-vs-stock[M={T}]", o_stock, o_gf)
    assert rel <= _REL


def test_grouped_fused_all_tokens_one_expert():
    """Ragged extreme: E-1 zero-row experts + one full segment."""
    _require_stack()
    m, layer, d = _build("fp8", seed=6)
    _require_fused(m, layer)
    act = _silu_act()
    T = 40
    ti, tw = _routing(T, d["E"], 1, "one_expert", seed=1)
    torch.manual_seed(8)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_gf = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    rel = _report("gf-vs-stock[one_expert]", o_stock, o_gf)
    assert rel <= _REL


def test_fp4_falls_through():
    """fp4-CB is not eligible (the prologue can't compose a two-tier scale):
    _gf_ok must be False and the path must return None, not raise."""
    _require_stack()
    m, layer, d = _build("fp4v2")
    assert m._gf_ok(layer) is False
    act = _silu_act()
    ti, tw = _routing(8, d["E"], 2, "uniform", seed=0)
    x = torch.randn(8, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    assert m._apply_prefill_grouped_fused(layer, x, tw, ti, act) is None


# --------------------------------------------------------------------------- #
# ROUND 2 GPU parity: v2 (one launch per stage) vs R1 (the bisection reference). #
# --------------------------------------------------------------------------- #
def _require_fused2(m, layer):
    _require_stack()
    if not m._gf2_ok(layer):
        pytest.skip("grouped cb_fused_moe_grouped binding / rung constraints "
                    "unmet")


@pytest.mark.parametrize("dist", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [2, 4])
def test_grouped_fused_v2_vs_r1_parity(dist, topk):
    _require_stack()
    m, layer, d = _build("fp8", seed=1)
    _require_fused2(m, layer)
    act = _silu_act()
    T = 48
    ti, tw = _routing(T, d["E"], topk, dist, seed=7)
    torch.manual_seed(2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_r1 = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    o_v2 = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act)
    assert o_v2 is not None, "v2 returned None despite _gf2_ok"
    assert o_v2.shape == o_r1.shape == (T, d["hidden"])
    rel = _report(f"v2-vs-r1[{dist},topk={topk}]", o_r1, o_v2)
    assert rel <= _REL, f"{dist}/topk={topk}: rel {rel:.3e} > {_REL}"


@pytest.mark.parametrize("T", [1, 3, 17, 129])
def test_grouped_fused_v2_small_m(T):
    """Sub-TileM M: the padded layout must still produce exactly one tile per
    short expert, with the pad rows inert."""
    _require_stack()
    m, layer, d = _build("fp8", seed=3)
    _require_fused2(m, layer)
    act = _silu_act()
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=5)
    torch.manual_seed(4)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_r1 = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    o_v2 = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act)
    rel = _report(f"v2-vs-r1[M={T}]", o_r1, o_v2)
    assert rel <= _REL


def test_grouped_fused_v2_all_tokens_one_expert():
    """Ragged extreme: E-1 zero-row experts (zero tiles) + one full segment."""
    _require_stack()
    m, layer, d = _build("fp8", seed=6)
    _require_fused2(m, layer)
    act = _silu_act()
    T = 40
    ti, tw = _routing(T, d["E"], 1, "one_expert", seed=1)
    torch.manual_seed(8)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_r1 = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    o_v2 = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act)
    rel = _report("v2-vs-r1[one_expert]", o_r1, o_v2)
    assert rel <= _REL


def test_grouped_fused_v2_trim_is_bit_identical(monkeypatch):
    """TRIM=0 launches up to E extra all-padding tiles. If padding is truly
    inert the two outputs are BIT-identical — not merely close."""
    _require_stack()
    m, layer, d = _build("fp8", seed=11)
    _require_fused2(m, layer)
    act = _silu_act()
    T = 33
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=3)
    torch.manual_seed(12)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "1")
    o_trim = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act)
    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "0")
    o_full = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act)
    assert torch.equal(o_trim, o_full), "padding tiles are not inert"


def test_grouped_fused_v2_falls_back_to_r1_without_binding(monkeypatch):
    """An extension build predating cb_fused_moe_grouped must fall back to R1,
    never crash — the gate is the only thing standing between them."""
    _require_stack()
    m, layer, d = _build("fp8", seed=13)
    _require_fused(m, layer)
    layer._cb_gf2_ok = False                      # simulate the missing binding
    act = _silu_act()
    ti0, tw0 = _routing(8, d["E"], 2, "uniform", seed=0)
    x0 = torch.randn(8, d["hidden"], dtype=torch.bfloat16, device=DEV)
    assert m._apply_prefill_grouped_fused_v2(layer, x0, tw0, ti0, act) is None

    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "grouped_fused")
    seen = {}
    orig = m._apply_prefill_grouped_fused
    m._apply_prefill_grouped_fused = lambda *a, **kw: (
        seen.setdefault("hit", True), orig(*a, **kw))[1]
    T = 32
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o = m._apply_inline(layer, x, tw, ti)
    assert seen.get("hit"), "dispatch did not fall back to R1"
    assert o.shape == (T, d["hidden"])


def test_mode_dispatch_prefers_v2(monkeypatch):
    """With the binding present, 'grouped_fused' means ROUND 2."""
    _require_stack()
    m, layer, d = _build("fp8", seed=14)
    _require_fused2(m, layer)
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "grouped_fused")
    seen = {}
    orig = m._apply_prefill_grouped_fused_v2
    m._apply_prefill_grouped_fused_v2 = lambda *a, **kw: (
        seen.setdefault("hit", True), orig(*a, **kw))[1]
    T = 32
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o = m._apply_inline(layer, x, tw, ti)
    assert seen.get("hit"), "mode dispatch did not reach v2"
    assert o.shape == (T, d["hidden"])


def test_mode_dispatch_selects_grouped_fused(monkeypatch):
    """PRISMAQUANT_CB_PREFILL=grouped_fused_r1 pins the ROUND-1 path (the
    bisection reference) regardless of the grouped binding's presence."""
    _require_stack()
    m, layer, d = _build("fp8", seed=9)
    _require_fused(m, layer)
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "grouped_fused_r1")
    seen = {}
    orig = m._apply_prefill_grouped_fused

    def _spy(*a, **kw):
        seen["hit"] = True
        return orig(*a, **kw)

    m._apply_prefill_grouped_fused = _spy
    T = 32
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o = m._apply_inline(layer, x, tw, ti)
    assert seen.get("hit"), "mode dispatch did not reach grouped_fused"
    assert o.shape == (T, d["hidden"])
