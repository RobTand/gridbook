"""Correctness gate for the ROUND-4 L2-pipeline MoE PREFILL path
(``moe.PrismaQuantCBMoEMethod._apply_prefill_l2_pipeline``,
``PRISMAQUANT_CB_PREFILL=l2_pipeline``).

R4 decodes each expert group into one half of a rotating, L2-PINNED scratch
arena instead of a fresh HBM tile, and runs ``cutlass_scaled_mm`` per expert over
it. The decoded bytes are the stock path's bytes (same expander kernel), the
activation QDQ is the stock path's own per-token fp8 dynamic, and the scales are
applied in the CUTLASS epilogue in the promoted rounding order — so R4 differs
from stock only by GEMM accumulation + cross-expert combine reassociation, the
suite's REASSOCIATION-CLASS 2e-2 contract.

The test that matters most here is the FORCED-TINY-WINDOW rotation test: with
one expert per scratch fill, consecutive units alternate buffers, and a missing
write-after-read event would let the decode of unit u+1 overwrite the tile unit
u-1 is still reading. That is a silent WRONG ANSWER, not a crash, so it is
asserted as a numerical parity failure against stock with weights that differ
strongly between experts.

Run scopes:

* ``-k "l2_plan or l2_cap or l2_group or l2_live"`` (build venv, NO vLLM/CUDA):
  the whole grouping/fall-through arithmetic, which lives in ``gridbook.moe_l2``
  torch-free for exactly this reason.
* everything else (serving container: vLLM + CUDA + the extension carrying
  ``cb_expand_fp8_into``)::

    docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
      --entrypoint bash vllm-node-tf5-cu132-lfm:latest -c 'pip install -q pytest; \\
      PYTHONPATH=/repo:/repo/plugins/gridbook python3 -m pytest \\
      /repo/plugins/gridbook/tests/test_moe_l2_pipeline.py -v'
"""
import pytest
import torch

pytest.importorskip("gridbook.codec")
# Torch-free by design (see gridbook/moe_l2.py) so R4's arithmetic — the window
# cap, the group size, the fall-through decision — runs in the build venv.
from gridbook.moe_l2 import (  # noqa: E402
    CB_L2_DEFAULT_MIN_M,
    L2_PIPELINE,
    cb_l2_cap_bytes,
    cb_l2_group_size,
    cb_l2_live_groups,
    cb_l2_min_m,
    cb_l2_pin_action,
    cb_l2_plan,
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

_MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# CPU — window cap derivation.                                                  #
# --------------------------------------------------------------------------- #
def test_l2_cap_halves_the_device_persisting_capacity():
    """The pinned window covers BOTH halves of the arena, so one half may claim
    at most half of what the device reports."""
    assert cb_l2_cap_bytes(64 * _MB) == 32 * _MB


def test_l2_cap_clamped_by_driver_window_and_env():
    # The driver's max window extent is an independent ceiling.
    assert cb_l2_cap_bytes(64 * _MB, max_window_bytes=16 * _MB) == 8 * _MB
    # The env override is the SAME quantity (a per-half cap) and intersects.
    assert cb_l2_cap_bytes(64 * _MB, env_mb="4") == 4 * _MB
    assert cb_l2_cap_bytes(8 * _MB, env_mb="64") == 4 * _MB


def test_l2_cap_zero_when_nothing_usable_is_reported():
    assert cb_l2_cap_bytes(0) == 0
    assert cb_l2_cap_bytes(None) == 0
    assert cb_l2_cap_bytes(64 * _MB, env_mb="0") == 0
    assert cb_l2_cap_bytes(64 * _MB, env_mb="not-a-number") == 0


# --------------------------------------------------------------------------- #
# CPU — group size / plan (the ceil(cap/expert_bytes) arithmetic).              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("expert_bytes,cap,expect", [
    (1 * _MB, 8 * _MB, 8),        # small experts: fill the window
    (3 * _MB, 8 * _MB, 2),        # floor, never a partial tile
    (8 * _MB, 8 * _MB, 1),        # exactly one fits
    (9 * _MB, 8 * _MB, 0),        # LARGER than the cap -> fall through
    (1 * _MB, 0, 0),              # no window reported -> fall through
])
def test_l2_group_size(expert_bytes, cap, expect):
    assert cb_l2_group_size(expert_bytes, cap) == expect


def test_l2_plan_groups_partition_the_experts():
    plan = cb_l2_plan(10, 1 * _MB, 4 * _MB)
    assert plan.group_size == 4
    assert plan.groups == [(0, 4), (4, 8), (8, 10)]     # last group is short
    covered = [e for e0, e1 in plan.groups for e in range(e0, e1)]
    assert covered == list(range(10))
    # The arena holds a PAIR of full-size halves, even though the last group is
    # short: the buffer is sized on the plan, not on a particular group.
    assert plan.buffer_bytes == 4 * _MB
    assert plan.arena_bytes == 8 * _MB


def test_l2_plan_group_never_exceeds_expert_count():
    plan = cb_l2_plan(3, 1 * _MB, 64 * _MB)
    assert plan.group_size == 3 and plan.groups == [(0, 3)]


def test_l2_plan_none_when_largest_tile_exceeds_cap():
    """The fall-through decision: no rotation of this pair could keep a tile
    resident, so the caller must run stock."""
    assert cb_l2_plan(8, 9 * _MB, 8 * _MB) is None
    assert cb_l2_plan(8, 1 * _MB, 0) is None


def test_l2_plan_force_group_overrides():
    plan = cb_l2_plan(8, 64 * _MB, 1 * _MB, force_group=1)
    assert plan.group_size == 1 and len(plan.groups) == 8


# --------------------------------------------------------------------------- #
# CPU — live-group selection off the already-fetched routing boundaries.        #
# --------------------------------------------------------------------------- #
def test_l2_live_groups_drops_empty_groups():
    E, topk = 8, 1
    topk_ids = torch.full((16, topk), 3, dtype=torch.long)
    counts = torch.bincount(topk_ids.reshape(-1), minlength=E)
    bounds = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).tolist()
    plan = cb_l2_plan(E, 1, 2)                      # group_size 2 -> 4 groups
    live = cb_l2_live_groups(plan.groups, bounds)
    assert live == [(2, 4, 0, 16)]                  # only the group holding e=3


def test_l2_live_groups_cover_every_routed_pair():
    torch.manual_seed(0)
    E, T, topk = 9, 30, 2
    topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(T)])
    counts = torch.bincount(topk_ids.reshape(-1), minlength=E)
    bounds = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).tolist()
    plan = cb_l2_plan(E, 1, 4)
    live = cb_l2_live_groups(plan.groups, bounds)
    assert sum(p1 - p0 for _e0, _e1, p0, p1 in live) == T * topk


# --------------------------------------------------------------------------- #
# CPU — the tiny-M floor and the pin-window lifecycle.                          #
# --------------------------------------------------------------------------- #
def test_l2_min_m_default_and_override():
    assert cb_l2_min_m(None) == CB_L2_DEFAULT_MIN_M
    assert cb_l2_min_m("") == CB_L2_DEFAULT_MIN_M
    assert cb_l2_min_m("512") == 512
    assert cb_l2_min_m("0") == 0            # explicit "no floor" is honoured
    # A typo must NOT silently re-open the regime the floor exists to close.
    assert cb_l2_min_m("lots") == CB_L2_DEFAULT_MIN_M
    assert cb_l2_min_m("-4") == CB_L2_DEFAULT_MIN_M


def test_l2_pipeline_returns_none_below_the_m_floor(monkeypatch):
    """The floor is checked BEFORE any CUDA/vLLM touch, so it is a CPU test:
    the live wedge came from PRISMAQUANT_CB_PREFILL=l2_pipeline driving a
    17-row prefill into a per-expert pipeline."""
    # gridbook.moe imports vLLM at module scope; skip like the GPU cases when
    # the serving stack is absent (the floor arithmetic itself is covered by
    # test_l2_min_m_default_and_override, which is genuinely dependency-free).
    try:
        from gridbook.moe import PrismaQuantCBMoEMethod
    except Exception:  # noqa: BLE001
        pytest.skip("gridbook.moe needs the vLLM serving stack")

    monkeypatch.delenv("PRISMAQUANT_CB_L2_MIN_M", raising=False)

    class _X:                     # only .shape[0] is reached before the floor
        shape = (17,)

    assert PrismaQuantCBMoEMethod._apply_prefill_l2_pipeline(
        object(), object(), _X(), None, None, None) is None


def test_l2_pin_action_is_a_steady_state():
    """The device-wide reservation/reset pair must fire only on a key CHANGE —
    per-forward pinning was ~2 implicit device syncs per layer per forward."""
    assert cb_l2_pin_action(None, ("a", 1)) == (False, True)   # first pin
    assert cb_l2_pin_action(("a", 1), ("a", 1)) == (False, False)  # steady
    assert cb_l2_pin_action(("a", 1), ("b", 1)) == (True, True)    # re-aim
    assert cb_l2_pin_action(("a", 1), None) == (True, False)       # released
    assert cb_l2_pin_action(None, None) == (False, False)


# --------------------------------------------------------------------------- #
# GPU — parity, rotation, fall-through.                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_m_floor(monkeypatch):
    """The GPU cases run deliberately tiny shapes (T=8..64) to keep the race
    window dense; the production floor would fall them all through. Disable it
    explicitly — the floor itself is covered by its own CPU tests above."""
    monkeypatch.setenv("PRISMAQUANT_CB_L2_MIN_M", "0")


@pytest.fixture(params=["default_stream", "side_stream"])
def stream_ctx(request):
    """Run each GPU case on BOTH the default stream and a NON-DEFAULT one.

    The live-serve hang was invisible to this suite precisely because every
    case ran on the default stream, while vLLM serves on its own. Any path that
    caches a stream handle, or launches cross-stream work, now has to survive
    the serving stream context here.
    """
    import contextlib

    if request.param == "default_stream":
        return contextlib.nullcontext()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return torch.cuda.stream(torch.cuda.Stream())


def _require_l2():
    """Skip unless the whole R4 stack is present: vLLM + CUDA (via
    ``_require_stack``, which probes a real fused_moe submodule so a STUBBED
    vllm skips like an absent one) plus an extension build carrying the
    into-buffer expander, which ships independently of everything else."""
    _require_stack()
    from gridbook import ops as pq_ops
    if not pq_ops.cb_expand_fp8_into_available():
        pytest.skip("extension build lacks cb_expand_fp8_into (R4)")


def _fresh(m, layer):
    """Drop the per-layer R4 caches so an env change actually takes effect
    (eligibility and the plan are cached on the layer by design)."""
    for attr in ("_cb_l2_ok", "_cb_l2_plan", "_cb_l2_scratch", "_cb_l2_row0",
                 "_cb_l2_pinned_key", "_cb_l2_pinned_streams"):
        if hasattr(layer, attr):
            delattr(layer, attr)


@pytest.mark.parametrize("dist", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [1, 2])
def test_l2_pipeline_matches_stock(dist, topk, stream_ctx):
    _require_l2()
    m, layer, d = _build("fp8", seed=3)
    act = _silu_act()
    T = 64
    ti, tw = _routing(T, d["E"], topk, dist, seed=11)
    torch.manual_seed(5)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    _fresh(m, layer)
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_l2 = m._apply_prefill_l2_pipeline(layer, x, tw, ti, act)
    if o_l2 is None:
        pytest.skip("no usable L2 window on this device (path falls to stock)")
    rel = _report(f"l2-parity[{dist},topk={topk}]", o_stock, o_l2)
    assert rel <= _REL, f"{dist}/topk={topk}: l2 rel {rel:.3e} > {_REL}"


def test_l2_pipeline_ragged_and_zero_row_experts(stream_ctx):
    """Every pair routes to ONE expert: all other experts are zero-row, so most
    groups are dropped and one group carries every row."""
    _require_l2()
    m, layer, d = _build("fp8", seed=4)
    act = _silu_act()
    T = 40
    ti, tw = _routing(T, d["E"], 1, "one_expert", seed=3)
    torch.manual_seed(6)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    _fresh(m, layer)
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_l2 = m._apply_prefill_l2_pipeline(layer, x, tw, ti, act)
    if o_l2 is None:
        pytest.skip("no usable L2 window on this device")
    rel = _report("l2-parity[zero-row]", o_stock, o_l2)
    assert rel <= _REL


def test_l2_scratch_rotation_under_forced_group_of_one(monkeypatch, stream_ctx):
    """THE race test. ``PRISMAQUANT_CB_L2_GROUP=1`` forces one expert per
    scratch fill, so the unit sequence is w13(e0), w2(e0), w13(e1), w2(e1), ...
    alternating buffers on EVERY unit — the maximum rate of reuse the event
    structure has to serialise.

    A missing write-after-read event lets the decode of unit u+1 land in the
    buffer unit u-1 is still reading, i.e. one expert's GEMM runs against a
    DIFFERENT expert's weights. The fixture's experts are independent random
    draws, so such a swap moves the output by order-1 relative error — far above
    the 2e-2 reassociation bound — and shows up as a deterministic assertion
    failure, not a flaky tolerance miss. Every expert is routed to (uniform
    top-k over all E) so no unit is skipped and the alternation is dense.
    """
    _require_l2()
    m, layer, d = _build("fp8", seed=7)
    act = _silu_act()
    E = d["E"]
    T = 8 * E
    ti, tw = _routing(T, E, E, "uniform", seed=13)      # topk == E: all hit
    torch.manual_seed(8)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    _fresh(m, layer)
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)

    monkeypatch.setenv("PRISMAQUANT_CB_L2_GROUP", "1")
    _fresh(m, layer)
    o_l2 = m._apply_prefill_l2_pipeline(layer, x, tw, ti, act)
    assert o_l2 is not None, "forced group of 1 must not fall through"
    plan = m._l2_plan(layer)
    assert plan.group_size == 1 and len(plan.groups) == E
    rel = _report("l2-rotation[group=1]", o_stock, o_l2)
    assert rel <= _REL, (
        f"rotation rel {rel:.3e} > {_REL}: a scratch buffer was overwritten "
        "before its GEMM consumed it (missing write-after-read event)")


def test_l2_falls_through_to_stock_when_window_cap_exceeded(monkeypatch, stream_ctx):
    """A cap smaller than one expert tile must DISABLE the path (return None)
    and the dispatcher must then serve the stock answer — bit-identical, since
    it IS the stock path."""
    _require_l2()
    m, layer, d = _build("fp8", seed=9)
    act = _silu_act()
    T = 32
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=17)
    torch.manual_seed(10)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    _fresh(m, layer)
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)

    # 1/1024 MB = 1 KiB per half: smaller than any real expert tile.
    monkeypatch.setenv("PRISMAQUANT_CB_L2_WINDOW_MB", str(1.0 / 1024))
    _fresh(m, layer)
    assert m._l2_plan(layer) is None
    assert m._apply_prefill_l2_pipeline(layer, x, tw, ti, act) is None

    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", L2_PIPELINE)
    _fresh(m, layer)
    o_dispatch = m._apply_inline(layer, x, tw, ti)
    assert torch.equal(o_dispatch, o_stock), (
        "cap-exceeded fall-through must serve the stock path unchanged")


def test_l2_is_an_auto_candidate_when_eligible(stream_ctx):
    import os as _os
    _os.environ["PRISMAQUANT_CB_L2_AUTOTUNE"] = "1"
    try:
        """R4 is wired in as just another MEASURED candidate; no promotion decision
        is taken in code and 'stock' stays first (the tuner's kept output)."""
        _require_l2()
        m, layer, d = _build("fp8", seed=2)
        act = _silu_act()
        ti, tw = _routing(16, d["E"], 2, "uniform", seed=1)
        x = torch.randn(16, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
        _fresh(m, layer)
        names = [n for n, _fn in m._prefill_candidates(layer, x, tw, ti, act)]
        assert names[0] == "stock"
        if m._l2_plan(layer) is not None:
            assert L2_PIPELINE in names
    finally:
        _os.environ.pop("PRISMAQUANT_CB_L2_AUTOTUNE", None)
