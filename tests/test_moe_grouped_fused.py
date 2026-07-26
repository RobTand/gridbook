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

Round 3 adds ``PRISMAQUANT_CB_PREFILL=auto``: MEASURED per-layer selection
between stock and grouped_fused at each COMPILED TileM, because the grouped path
wins on small-expert MoEs and loses on large-expert ones. Its policy lives in
``gridbook.moe_autotune`` (torch-only) and is tested here on CPU with an injected
timer — no vLLM, no GPU, no real kernel.

Run scopes:

* ``-k "routing or auto"`` (build venv, NO vLLM/CUDA needed): the sort/boundary
  property the one-sync R1 routing rests on, the R2 padded-routing construction
  at every TileM (``gridbook.moe_routing`` is torch-only for exactly this
  reason), and the auto-selection policy.
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
from gridbook.moe_autotune import (  # noqa: E402
    STOCK,
    cb_autotune_prefill,
    cb_prefill_auto,
    resolve_candidate,
)

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


@pytest.mark.parametrize("tile_m", [4, 8, 128, 256])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_padded_routing_matches_loop_selection(tile_m, topk):
    torch.manual_seed(0)
    T, E = 37, 11
    topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(T)])
    _check_routing(topk_ids.to(torch.int32), E, tile_m)


@pytest.mark.parametrize("tile_m,exp_blocks", [(4, 4), (8, 2), (128, 1),
                                               (256, 1)])
def test_padded_routing_zero_row_experts_consume_no_blocks(tile_m, exp_blocks):
    """E-1 empty experts must cost zero tiles — otherwise a 256-expert layer
    launches 256 tiles for a one-expert prefill. Holds at every TileM the
    'auto' path may pick, including one where the whole segment is a single
    mostly-padded tile (tile_m=256 over 16 rows)."""
    E = 8
    topk_ids = torch.full((16, 1), 3, dtype=torch.int32)
    eids, _row_src, _pad, n_blocks = cb_grouped_pad_routing(topk_ids, E, tile_m)
    assert int(n_blocks) == exp_blocks
    assert torch.equal(eids[:exp_blocks],
                       torch.full((exp_blocks,), 3, dtype=torch.int32))


@pytest.mark.parametrize("tile_m", [128, 256])
def test_padded_routing_ragged_at_serving_tiles(tile_m):
    """The TileM values the kernel actually compiles, on a ragged 8k-ish
    prefill: capacity, block starts, pad flags and stable order all hold
    unchanged — the tile size is a free parameter of the construction, which is
    what lets 'auto' sweep it."""
    g = torch.Generator().manual_seed(5)
    E, T, topk = 64, 1024, 4
    ids = torch.randint(0, E, (T, topk), generator=g, dtype=torch.int32)
    ids[:, 0] = 7                       # a hot expert -> many full tiles
    _check_routing(ids, E, tile_m)


@pytest.mark.parametrize("seed", list(range(24)))
def test_padded_routing_capacity_bound_randomized(seed):
    """The static bound cap = P//tile_m + E is what makes the whole path
    host-read-free at build time; fuzz it over ragged/skewed routings."""
    g = torch.Generator().manual_seed(seed)
    E = int(torch.randint(1, 17, (1,), generator=g))
    topk = int(torch.randint(1, min(E, 6) + 1, (1,), generator=g))
    T = int(torch.randint(1, 40, (1,), generator=g))
    tile_m = int([1, 2, 4, 8, 128, 256][seed % 6])
    # Skewed (with replacement) routing: the worst case for ragged block counts.
    ids = torch.randint(0, E, (T, topk), generator=g, dtype=torch.int32)
    nb, cap = _check_routing(ids, E, tile_m)
    assert nb <= cap


# --------------------------------------------------------------------------- #
# ROUND 3 auto-selection policy (CPU ok) — the timer is injected, so this is     #
# the real dispatch/caching/determinism logic with no vLLM, GPU or kernel.       #
# --------------------------------------------------------------------------- #
class _FakeLayer:
    """Stands in for the FusedMoE layer: the policy only ever touches
    ``_cb_prefill_choice`` on it (per-LAYER caching, not global)."""


def _auto_fixture(ms):
    """Candidates named by ``ms`` (name -> fake milliseconds) with tensor
    outputs tagged by name, plus a spy timer counting how often it is called."""
    tags = {name: torch.full((2,), float(i))
            for i, name in enumerate(sorted(ms))}
    calls = {"timed": 0, "ran": []}

    def build():
        out = []
        for name in ms:                    # deliberately NOT sorted: tie-break
            def fn(n=name):                # must not depend on this order
                calls["ran"].append(n)
                return tags[n]
            out.append((name, fn))
        return out

    def timer(fn):
        calls["timed"] += 1
        out = fn()
        # The candidate's identity is its output tag; map back to its ms.
        name = next(n for n, t in tags.items() if torch.equal(t, out))
        return out, ms[name]

    return tags, build, timer, calls


def _run_auto(layer, build, timer, calls, num_tokens, **kw):
    return cb_prefill_auto(
        layer, num_tokens, build,
        lambda: (calls["ran"].append("stock:direct"), build()[0][1]())[1],
        timer=timer, **kw)


def test_auto_caches_the_measured_winner_on_the_layer():
    """The argmin is cached on the LAYER and later calls dispatch straight to it
    with NO further timing — the tuning cost is once per layer per process."""
    ms = {STOCK: 9.0, "grouped_fused:tile_m=128": 4.0,
          "grouped_fused:tile_m=256": 7.0}
    tags, build, timer, calls = _auto_fixture(ms)
    layer = _FakeLayer()

    out = _run_auto(layer, build, timer, calls, 4096)
    assert layer._cb_prefill_choice == "grouped_fused:tile_m=128"
    assert calls["timed"] == 3, "every candidate must be timed exactly once"
    # DETERMINISM: the tuning call returns the STOCK output, not the winner's.
    assert torch.equal(out, tags[STOCK])

    calls["timed"] = 0
    calls["ran"].clear()
    out2 = _run_auto(layer, build, timer, calls, 4096)
    assert calls["timed"] == 0, "re-timed a layer that already has a winner"
    assert calls["ran"] == ["grouped_fused:tile_m=128"]
    assert torch.equal(out2, tags["grouped_fused:tile_m=128"])


def test_auto_below_min_m_uses_stock_and_caches_nothing():
    """A short prefill is not the steady-state shape, so it must not tune AND
    must not cache — the first qualifying call still gets to measure."""
    ms = {STOCK: 9.0, "grouped_fused:tile_m=128": 4.0}
    tags, build, timer, calls = _auto_fixture(ms)
    layer = _FakeLayer()
    out = _run_auto(layer, build, timer, calls, 512, min_m=1024)
    assert calls["timed"] == 0
    assert getattr(layer, "_cb_prefill_choice", None) is None
    assert torch.equal(out, tags[STOCK])

    out = _run_auto(layer, build, timer, calls, 1024, min_m=1024)
    assert calls["timed"] == 2 and layer._cb_prefill_choice == \
        "grouped_fused:tile_m=128"
    assert torch.equal(out, tags[STOCK])


def test_auto_stock_can_win():
    """The large-expert regression case: stock must be selectable, or 'auto'
    would just be grouped_fused with extra steps."""
    ms = {STOCK: 3.0, "grouped_fused:tile_m=128": 8.0}
    tags, build, timer, calls = _auto_fixture(ms)
    layer = _FakeLayer()
    _run_auto(layer, build, timer, calls, 2048)
    assert layer._cb_prefill_choice == STOCK
    calls["ran"].clear()
    assert torch.equal(_run_auto(layer, build, timer, calls, 2048), tags[STOCK])
    assert calls["ran"] == [STOCK]


def test_auto_force_pins_without_timing():
    """PRISMAQUANT_CB_PREFILL_AUTO_FORCE bisection: pin a path, measure nothing.
    A bare family name resolves to that family's first (smallest-TileM)
    candidate."""
    ms = {STOCK: 1.0, "grouped_fused:tile_m=128": 9.0,
          "grouped_fused:tile_m=256": 9.5}
    tags, build, timer, calls = _auto_fixture(ms)
    layer = _FakeLayer()
    out = _run_auto(layer, build, timer, calls, 4096, forced="grouped_fused")
    assert calls["timed"] == 0
    assert layer._cb_prefill_choice == "grouped_fused"
    assert torch.equal(out, tags["grouped_fused:tile_m=128"])

    layer2 = _FakeLayer()
    out2 = _run_auto(layer2, build, timer, calls, 4096,
                     forced="grouped_fused:tile_m=256")
    assert torch.equal(out2, tags["grouped_fused:tile_m=256"])


def test_auto_disqualified_candidates_are_skipped():
    """A (TileM, k_bits) pair the build could not compile raises or returns
    None; it must be dropped, not crash the serve, and stock must still win the
    argmin over the survivors."""
    calls = {"n": 0}

    def build():
        return [(STOCK, lambda: torch.zeros(2)),
                ("grouped_fused:tile_m=128", lambda: None),
                ("grouped_fused:tile_m=256",
                 lambda: (_ for _ in ()).throw(RuntimeError("no smem")))]

    def timer(fn):
        calls["n"] += 1
        try:
            out = fn()
        except Exception:
            return None
        return None if out is None else (out, 1.0)

    best, timings, kept = cb_autotune_prefill(build(), timer=timer)
    assert best == STOCK and list(timings) == [STOCK] and kept is not None
    assert calls["n"] == 3


def test_auto_all_candidates_disqualified_falls_back_to_stock():
    """If nothing can be timed the policy must still serve the request."""
    layer = _FakeLayer()
    out = cb_prefill_auto(
        layer, 4096, lambda: [(STOCK, lambda: None)],
        lambda: torch.ones(2), timer=lambda fn: None)
    assert torch.equal(out, torch.ones(2))
    assert getattr(layer, "_cb_prefill_choice", None) is None


def test_auto_stale_choice_falls_back_to_stock():
    """A cached/forced name that resolves to nothing (or whose path became
    unavailable) serves from the default rather than failing the request."""
    layer = _FakeLayer()
    layer._cb_prefill_choice = "grouped_fused:tile_m=512"
    out = cb_prefill_auto(
        layer, 4096, lambda: [(STOCK, lambda: torch.zeros(2))],
        lambda: torch.ones(2))
    assert torch.equal(out, torch.ones(2))


def test_auto_tie_breaks_on_name_not_iteration_order():
    """Equal timings must resolve to the same winner regardless of the order the
    candidates happened to be enumerated in — otherwise the served path is a
    race outcome."""
    cands = [("grouped_fused:tile_m=256", lambda: torch.zeros(1)),
             (STOCK, lambda: torch.zeros(1)),
             ("grouped_fused:tile_m=128", lambda: torch.zeros(1))]
    timer = lambda fn: (fn(), 5.0)                             # noqa: E731
    best_a, _, _ = cb_autotune_prefill(cands, timer=timer)
    best_b, _, _ = cb_autotune_prefill(list(reversed(cands)), timer=timer)
    assert best_a == best_b == "grouped_fused:tile_m=128"


def test_resolve_candidate_precedence():
    cands = [(STOCK, "s"), ("grouped_fused:tile_m=128", "g128"),
             ("grouped_fused:tile_m=256", "g256")]
    assert resolve_candidate(STOCK, cands) == "s"
    assert resolve_candidate("grouped_fused:tile_m=256", cands) == "g256"
    assert resolve_candidate("grouped_fused", cands) == "g128"
    assert resolve_candidate("nonsense", cands) is None


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


# --------------------------------------------------------------------------- #
# ROUND 3 GPU: every COMPILED TileM must be a correct candidate.                 #
# --------------------------------------------------------------------------- #
def _tile_sizes(m, layer):
    tiles = m._gf2_tile_sizes(layer)
    if not tiles:
        pytest.skip("extension exposes no grouped TileM")
    return tiles


def test_grouped_fused_v2_parity_at_every_compiled_tile_m():
    """'auto' may pick any TileM the build compiled, so each one must hit the
    same R1 reassociation bound — a tile is a performance knob, never a
    numerics knob."""
    _require_stack()
    m, layer, d = _build("fp8", seed=1)
    _require_fused2(m, layer)
    act = _silu_act()
    T = 48
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=7)
    torch.manual_seed(2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_r1 = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    for tm in _tile_sizes(m, layer):
        o = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act, tile_m=tm)
        assert o is not None, f"tile_m={tm} advertised but returned None"
        rel = _report(f"v2[tile_m={tm}]-vs-r1", o_r1, o)
        assert rel <= _REL, f"tile_m={tm}: rel {rel:.3e} > {_REL}"


@pytest.mark.parametrize("dist,T", [("one_expert", 40), ("subset", 17),
                                    ("uniform", 3)])
def test_grouped_fused_v2_ragged_at_tile_m_256(dist, T):
    """Ragged / zero-row cases at the LARGE tile, where padding dominates: a
    handful of rows must still be one mostly-padded, inert tile."""
    _require_stack()
    m, layer, d = _build("fp8", seed=6)
    _require_fused2(m, layer)
    if 256 not in m._gf2_tile_sizes(layer):
        pytest.skip("tile_m=256 not compiled in this extension build")
    act = _silu_act()
    topk = 1 if dist == "one_expert" else 2
    ti, tw = _routing(T, d["E"], topk, dist, seed=1)
    torch.manual_seed(8)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_r1 = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    o = m._apply_prefill_grouped_fused_v2(layer, x, tw, ti, act, tile_m=256)
    rel = _report(f"v2[tile_m=256,{dist},M={T}]-vs-r1", o_r1, o)
    assert rel <= _REL


def test_auto_mode_end_to_end(monkeypatch):
    """PRISMAQUANT_CB_PREFILL=auto on a real layer: it tunes once, caches a
    choice, and the tuning call's output matches the stock path bit-for-bit."""
    _require_stack()
    m, layer, d = _build("fp8", seed=17)
    _require_fused(m, layer)
    act = _silu_act()
    T = 64
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "auto")
    monkeypatch.setenv("PRISMAQUANT_CB_AUTOTUNE_MIN_M", str(T))
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=2)
    torch.manual_seed(5)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_auto = m._apply_inline(layer, x, tw, ti)
    assert getattr(layer, "_cb_prefill_choice", None) is not None
    assert torch.equal(o_auto, o_stock), "tuning call is not deterministic"
    o_next = m._apply_inline(layer, x, tw, ti)
    assert o_next.shape == (T, d["hidden"])


def test_auto_mode_below_min_m_is_stock(monkeypatch):
    """Under the threshold: no tuning, no cached choice, stock output."""
    _require_stack()
    m, layer, d = _build("fp8", seed=19)
    _require_fused(m, layer)
    act = _silu_act()
    T = 32
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "auto")
    monkeypatch.setenv("PRISMAQUANT_CB_AUTOTUNE_MIN_M", "1024")
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=2)
    torch.manual_seed(6)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o = m._apply_inline(layer, x, tw, ti)
    assert getattr(layer, "_cb_prefill_choice", None) is None
    assert torch.equal(o, m._apply_prefill_stock(layer, x, tw, ti, act))


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
