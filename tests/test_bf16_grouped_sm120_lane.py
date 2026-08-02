"""Contract and wiring gates for the OPT-IN sm12x-native BF16 grouped lane.

The selector, attestation and tile-order cases below are CPU-only. The lane
itself is a GPU kernel (``tests/test_bf16_grouped_cutlass.py`` gates its
numerics); what is testable without a device is the part that decides WHETHER
it runs, and that part carries the whole opt-in promise:

* with ``PRISMAQUANT_CB_BF16_SM120`` unset, nothing about the dispatch changes
  and nothing probes or builds;
* with it set on a machine that cannot serve the lane, the model LOAD fails
  with an actionable message instead of quietly running the SM80 schedule —
  which would answer a different question than the operator asked;
* the selector is process-stable and rejects typos, so an intended A/B can
  never become an unlabelled baseline run.

The last section gates the WIRING: ``gridbook/moe.py``'s routed prefill through
its own dispatch with the flag on and off, so the two claims the lane makes at
the operator level are asserted where an operator actually meets them — that
stage one carries NO padded activation copy any more (the in-mainloop A-row
gather reads the compact tensor and is bit-identical to the copy it replaced),
and that the swizzle-group packed expert ORDER is bit-neutral and applied only
where the expert-chunk loop's expert-major assumption still holds. Those cases
need CUDA, the grouped-BF16 extension and the sm12x lane; each skips on its own
rather than skipping the file, exactly as sections A-C stay live on a CI host
with no GPU.
"""
from __future__ import annotations

import importlib.util
import inspect
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import bf16_grouped_lane as lane  # noqa: E402
from gridbook.cuda_ext import NativeKernelUnavailableError  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_flag(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_BF16_SM120", raising=False)
    lane._reset_for_tests()
    yield
    lane._reset_for_tests()


def test_flag_defaults_to_the_sm80_schedule(monkeypatch):
    assert lane.requested() is False
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "0")
    lane._reset_for_tests()
    assert lane.requested() is False


def test_flag_enables_the_lane(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    assert lane.requested() is True


@pytest.mark.parametrize("value", ["yes", "true", "128", "sm120"])
def test_typos_are_refused_rather_than_silently_ignored(monkeypatch, value):
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", value)
    with pytest.raises(ValueError, match="PRISMAQUANT_CB_BF16_SM120"):
        lane.requested()


def test_selector_cannot_change_mid_process(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    assert lane.requested() is True
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "0")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        lane.requested()


def test_require_lane_fails_closed_without_the_bindings(monkeypatch):
    """A module without the sm12x entry points is refused, not fallen back."""
    stub = types.SimpleNamespace(cb_bf16_grouped_mm=lambda *a, **k: None,
                                 cb_bf16_grouped_mm_out=lambda *a, **k: None)
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: stub)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("routed quality prefill")
    message = str(exc_info.value)
    assert "PRISMAQUANT_CB_BF16_SM120=1" in message
    assert "cb_bf16_grouped_mm_sm120" in message
    assert "compute capability 12.0/12.1" in message
    assert "does not substitute a different kernel" in message


def test_require_lane_fails_closed_without_the_extension(monkeypatch):
    """No grouped-BF16 module at all is the same fail-closed answer."""
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: None)
    with pytest.raises(NativeKernelUnavailableError,
                       match="cb_bf16_grouped_gemm.cu"):
        lane.require_lane("routed quality prefill")


def _complete_stub(*, omit=(), tile_m=128):
    """A stand-in carrying exactly what ``cuda_ext`` requires, minus ``omit``.

    Built FROM the loader's tuples rather than listed here. The lane used to
    keep its own six-name list while the loader enforced seven — the missing
    one was ``cb_bf16_grouped_sm120_tile_sizes`` — under a comment asserting
    "the two lists now agree", and a hand-written stub in this file is what let
    that pass. A stub derived from the source of truth cannot.
    """
    from gridbook import cuda_ext

    symbols = {name: (lambda *a, **k: None)
               for name in (cuda_ext._BF16_GROUPED_SYMBOLS
                            + cuda_ext._BF16_GROUPED_SM120_SYMBOLS)}
    symbols["cb_bf16_grouped_sm120_tile_m"] = lambda: tile_m
    symbols["cb_bf16_grouped_sm120_tile_sizes"] = lambda: [tile_m]
    # The forward path reads the swizzle group on every routed prefill (the
    # packed expert order's group size), so a module without a usable config
    # query is not COMPLETE: accepting one would let `require_lane` call a
    # build good that then AttributeErrors at first forward, which is exactly
    # what attesting at load exists to prevent.
    symbols["cb_bf16_grouped_sm120_config"] = (
        lambda: [64, 128, 64, 3, 0, 0, 128, 1, 8, 64, 1])
    for name in omit:
        del symbols[name]
    return types.SimpleNamespace(**symbols)


def test_require_lane_accepts_a_complete_module(monkeypatch):
    stub = _complete_stub()
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: stub)
    assert lane.require_lane("routed quality prefill") is stub
    assert lane.tile_m(stub) == 128
    assert lane.swizzle_group(stub) == 8


@pytest.mark.parametrize("missing", [
    "cb_bf16_grouped_mm_sm120_gather",
    "cb_bf16_grouped_mm_sm120_gather_out",
    "cb_bf16_grouped_sm120_config",
    "cb_bf16_grouped_sm120_tile_sizes",
    "cb_bf16_grouped_sm120_tile_m",
    "cb_bf16_grouped_mm_sm120",
    "cb_bf16_grouped_mm_sm120_out",
])
def test_require_lane_fails_closed_on_any_missing_lane_binding(monkeypatch,
                                                               missing):
    """EVERY sm120 binding the loader requires is required here too.

    Two of these are the reason this is parametrized rather than spelled out.
    ``cb_bf16_grouped_mm_sm120_gather*`` absent means serving a partial lane
    that silently reintroduces the padded activation copy. And
    ``cb_bf16_grouped_sm120_tile_sizes`` was in the loader's tuple but NOT in
    the lane's local list until 2026-08-02, so a module missing it passed this
    gate — the drift a shared tuple makes impossible.
    """
    stub = _complete_stub(omit=(missing,))
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: stub)
    with pytest.raises(NativeKernelUnavailableError, match=missing):
        lane.require_lane("routed quality prefill")


def test_the_lane_checks_exactly_the_loaders_symbol_contract():
    """No local restatement of the required set can exist to drift."""
    import inspect

    from gridbook import cuda_ext

    source = inspect.getsource(lane.require_lane)
    assert "_BF16_GROUPED_SM120_SYMBOLS" in source
    for name in cuda_ext._BF16_GROUPED_SM120_SYMBOLS:
        assert f'"{name}"' not in source, (
            f"{name} is restated inside require_lane; spell the contract "
            f"against cuda_ext's tuple so the two cannot disagree")


def test_dense_helper_pads_to_one_tile_and_slices_back(monkeypatch):
    """Without the gather mode (an old stub), M=100 becomes one padded tile."""
    seen = {}

    def fake_op(a, weights, expert_ids, tile_m):
        seen["a"] = a
        seen["expert_ids"] = expert_ids
        seen["tile_m"] = tile_m
        assert weights.shape[0] == 1, "dense is E=1"
        return torch.arange(a.shape[0] * weights.shape[1],
                            dtype=torch.bfloat16).reshape(
                                a.shape[0], weights.shape[1])

    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120", fake_op)
    ext = types.SimpleNamespace(cb_bf16_grouped_sm120_tile_m=lambda: 128)
    a = torch.randn(100, 64, dtype=torch.bfloat16)
    w = torch.randn(32, 64, dtype=torch.bfloat16)

    y = lane.dense_mm(ext, a, w)

    assert y.shape == (100, 32)
    assert seen["tile_m"] == 128
    assert seen["a"].shape == (128, 64)
    assert torch.equal(seen["a"][:100], a)
    assert not seen["a"][100:].any(), "padding rows must be zero"
    assert seen["expert_ids"].tolist() == [0]
    assert seen["expert_ids"].dtype is torch.int32


def test_dense_helper_leaves_an_exact_multiple_alone(monkeypatch):
    def fake_op(a, weights, expert_ids, tile_m):
        assert a.shape[0] == 256
        assert expert_ids.tolist() == [0, 0]
        return torch.zeros(a.shape[0], weights.shape[1],
                           dtype=torch.bfloat16)

    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120", fake_op)
    ext = types.SimpleNamespace(cb_bf16_grouped_sm120_tile_m=lambda: 128)
    y = lane.dense_mm(ext, torch.randn(256, 64, dtype=torch.bfloat16),
                      torch.randn(8, 64, dtype=torch.bfloat16))
    assert y.shape == (256, 8)


def test_dense_helper_prefers_the_gather_mode_and_copies_nothing(monkeypatch):
    """With the gather entry point present, no padded copy is built at all:
    the compact activation goes straight in with row_src = arange(Mp), whose
    ids past M read zeros inside the kernel."""
    seen = {}

    def fake_gather(a, row_src, weights, expert_ids, tile_m):
        seen["a"] = a
        seen["row_src"] = row_src
        seen["expert_ids"] = expert_ids
        seen["tile_m"] = tile_m
        return torch.zeros(row_src.shape[0], weights.shape[1],
                           dtype=torch.bfloat16)

    def forbidden(*args, **kwargs):
        raise AssertionError("the padded-copy op must not be called")

    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120_gather",
                        fake_gather)
    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120", forbidden)
    ext = types.SimpleNamespace(
        cb_bf16_grouped_sm120_tile_m=lambda: 128,
        cb_bf16_grouped_mm_sm120_gather=lambda *a, **k: None,
    )
    a = torch.randn(100, 64, dtype=torch.bfloat16)
    y = lane.dense_mm(ext, a, torch.randn(32, 64, dtype=torch.bfloat16))

    assert y.shape == (100, 32)
    assert seen["a"] is not None and seen["a"].shape == (100, 64)
    assert seen["row_src"].dtype is torch.int32
    assert seen["row_src"].tolist() == list(range(128))
    assert seen["expert_ids"].tolist() == [0]
    assert seen["tile_m"] == 128


# ---------------------------------------------------------------------------
# The swizzle-group-aligned expert ORDER (tile-order policy).
# ---------------------------------------------------------------------------


def test_pack_expert_blocks_fills_groups_exactly_when_possible():
    """{3,3,2}-block experts tile a group of 8 with no straddle at all."""
    counts = [129, 130, 65, 129, 130, 66, 129, 129]  # blocks: 3,3,2,3,3,2,3,3
    order, touched, minimum = lane.pack_expert_blocks(counts, 64, 8)
    assert sorted(order) == list(range(8))
    assert touched == minimum, "an exactly packable histogram must align"


def test_pack_expert_blocks_is_deterministic_and_skips_empty_experts():
    counts = [0, 100, 0, 500, 64, 0, 1, 320]
    first = lane.pack_expert_blocks(counts, 64, 8)
    second = lane.pack_expert_blocks(counts, 64, 8)
    assert first == second
    order = first[0]
    assert sorted(order) == [1, 3, 4, 6, 7]
    assert 0 not in order and 2 not in order and 5 not in order


def test_pack_expert_blocks_handles_experts_larger_than_a_group():
    counts = [1200, 30, 30]  # 19 blocks + 1 + 1
    order, touched, minimum = lane.pack_expert_blocks(counts, 64, 8)
    assert sorted(order) == [0, 1, 2]
    assert minimum <= touched <= minimum + 2


def test_pack_expert_blocks_group_one_never_straddles():
    order, touched, minimum = lane.pack_expert_blocks([70, 3, 900], 64, 1)
    assert touched == minimum


# ===========================================================================
# THE WIRED LANE — ``gridbook/moe.py``'s routed prefill, through its own
# dispatch, with the selector on and off.
#
# Everything above answers "would the lane be selected?". This answers "and
# when it is, does the operator still compute the same thing?" — for the two
# changes that landed on 2026-08-02:
#
#   1. stage one (w13/gate_up) uses the collective's IN-MAINLOOP A-row gather,
#      so the ``[Mp, K]`` padded activation copy and its appended zero row are
#      gone; ``dest`` itself is the row-source vector. Stage two (w2) stays in
#      padded mode on purpose — its input IS the padded intermediate;
#   2. when one expert chunk covers every expert, the block order is
#      SWIZZLE-GROUP PACKED, which must permute whole expert segments only.
#
# The harness (``_build`` / ``_silu_act`` / ``_report`` / ``_REL``) is the
# minimal copy of tests/test_moe_grouped_fused.py's, duplicated rather than
# imported for the reason tests/test_moe_persistent_b_lane.py states about its
# own borrowed helpers: that file stays free to change without silently
# changing what this one asserts.
# ===========================================================================
DEV = "cuda"

# The suite's documented reassociation envelope. NOT a new number: it is
# tests/test_moe_grouped_fused.py's ``_REL``, on that file's justification —
# a narrow envelope for two lanes that quantize the same values and round ONCE
# to bf16 but accumulate them in different tensor-core orders (GB10/CUDA 13.0
# measured 1.906-2.039e-2 there, 2026-08-01, torch 2.11+cu130). It is also the
# bound the sibling's own sm12x-lane-vs-bridge cases already use, so flipping
# the selector is held to exactly the same contract as swapping the lane object
# by hand. A regression bound, not a claim of bit equivalence — the bit claims
# below are stated as ``torch.equal`` where they hold.
_REL = 2.1e-2

# ``gridbook.moe`` imports vLLM at module scope; the wiring cases are skipped
# rather than skipping this whole file, because everything above is what a
# vLLM-free CI host can still gate.
_HAS_VLLM = importlib.util.find_spec("vllm") is not None
requires_vllm = pytest.mark.skipif(
    not _HAS_VLLM,
    reason="vLLM not importable here; gridbook.moe imports it at module scope")


def _require_stack():
    # Probe a real vLLM submodule: another compatibility test may have
    # installed lightweight ``vllm`` stubs into sys.modules.
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the wired-lane forward gates")


def _sm120_lane():
    """The extension whose sm12x lane this build carries, else skip."""
    _require_stack()
    from gridbook.cuda_ext import get_bf16_grouped_ext

    ext = get_bf16_grouped_ext()
    if ext is None:
        pytest.skip("owned grouped-BF16 CUTLASS extension unavailable")
    if not hasattr(ext, "cb_bf16_grouped_mm_sm120_gather_out"):
        pytest.skip("this build carries no sm12x lane (needs cc 12.0/12.1)")
    return ext


def _build(*, experts=8, hidden=512, inter=768, seed=0, k=44):
    """A synthetic FP8-CB MoE layer, without invoking vLLM weight loading."""
    _require_stack()
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    codec = pytest.importorskip("gridbook.codec")
    from gridbook.moe import PrismaQuantCBMoEMethod

    n_sub = 4
    type_size = fmt.nvfp4_cb_type_size(k, "fp8")
    codebook = fmt._resolve_codebook(
        k, "fp8", "product", None, torch.device(DEV))

    torch.manual_seed(seed)
    w13 = torch.randn(experts, 2 * inter, hidden, device=DEV) * 0.05
    w2 = torch.randn(experts, hidden, inter, device=DEV) * 0.05
    p13, f13 = fmt.nvfp4_cb_pack(
        w13, k, grid="fp8", mode="product", codebook=codebook)
    p2, f2 = fmt.nvfp4_cb_pack(
        w2, k, grid="fp8", mode="product", codebook=codebook)

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.scheme = {"grid": "fp8", "mode": "product", "k": k,
                     "n_sub": n_sub, "type_size": type_size}
    method.prefix = "test.sm120_lane"
    method.is_fp4 = False
    method.is_v2 = False
    method.k = k
    method.n_sub = n_sub
    method.type_size = type_size
    method._sub_table = None

    layer = types.SimpleNamespace(
        _cb_E=experts,
        _cb_hidden=hidden,
        _cb_inter=inter,
        w13_cb_qweight=p13.reshape(experts, 2 * inter, -1).contiguous(),
        w2_cb_qweight=p2.reshape(experts, hidden, -1).contiguous(),
        w13_weight_scale=f13["scales"].reshape(
            experts, 2 * inter).to(DEV).float(),
        w2_weight_scale=f2["scales"].reshape(experts, hidden).to(DEV).float(),
        _cb_flat=codec.build_flat_codebook([t.to(DEV) for t in codebook]),
        _cb_compose=torch.zeros(1, device=DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer, {"E": experts, "hidden": hidden, "inter": inter}


def _silu_act():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    try:
        return MoEActivation.from_str("silu")
    except Exception:  # noqa: BLE001 - enum spelling differs across vLLM
        return MoEActivation.SILU


def _routing(tokens, experts, topk, seed, device=DEV):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=generator)[:topk]
                       for _ in range(tokens)]).to(torch.int32)
    weights = (torch.rand(tokens, topk, generator=generator) + 0.1).float()
    return ids.to(device), weights.to(device)


def _skewed_counts(tile_m):
    """Per-expert routed-row counts with a deliberately RAGGED block histogram.

    At the compiled ``tile_m`` the padded block counts are
    ``{2, 5, 1, 0, 3, 1, 3, 1}``, which first-fit-decreasing packing reorders
    to experts ``[1, 4, 6, 0, 2, 5, 7]`` — i.e. the packed order is NOT the
    expert-major one, so the bit-neutrality claim below is not vacuous. One
    expert is empty (a zero-M problem) and one is a hair under a whole tile.
    """
    return (tile_m + 6, 4 * tile_m + 4, tile_m - 24, 0,
            2 * tile_m + 2, 5, 2 * tile_m + 62, tile_m)


def _routing_counts(counts, seed, device=DEV):
    """``top_k=1`` routing realizing an EXACT per-expert histogram.

    ``top_k=1`` is load-bearing wherever a case below asserts ``torch.equal``
    end to end: every token then receives exactly ONE routed contribution, so
    the closing ``index_add_`` is a pure scatter into a zeroed tensor and the
    row ORDER of the padded layout cannot reassociate it. (At ``top_k>1`` the
    combine sums several rows per token with CUDA atomics, which reorders.)
    """
    ids = torch.cat([torch.full((int(c),), e, dtype=torch.int32)
                     for e, c in enumerate(counts) if c])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = ids[torch.randperm(ids.numel(), generator=generator)].reshape(-1, 1)
    weights = (torch.rand(ids.shape[0], 1, generator=generator) + 0.1).float()
    return ids.to(device), weights.to(device)


def _report(tag, reference, candidate):
    rel = ((candidate.float() - reference.float()).norm()
           / reference.float().norm().clamp_min(1e-6)).item()
    maxabs = (candidate.float() - reference.float()).abs().max().item()
    print(f"{tag}: rel={rel:.3e} maxabs={maxabs:.3e}")
    return rel


def _resolve_lane_at_load(layer, device):
    """Exactly what ``process_weights_after_loading`` does for this lane.

    Two lines, calling ``gridbook.moe``'s OWN bound names — so a case below
    that flips the environment variable reaches the operator by the same route
    a model load does, and cannot drift from it silently: the case immediately
    after this helper asserts those names are this module's functions and that
    the loader still resolves the attribute through them.
    """
    from gridbook import moe

    layer._cb_bf16_sm120 = None
    if moe.bf16_sm120_requested():
        layer._cb_bf16_sm120 = moe.bf16_sm120_require_lane(
            "test routed quality prefill", device=device)
    return layer._cb_bf16_sm120


def _spy_padded_route(monkeypatch):
    """Record every ``_padded_route`` call the prefill makes and its result."""
    from gridbook import moe

    real = moe._padded_route
    seen = []

    def spy(*args, **kwargs):
        route = real(*args, **kwargs)
        seen.append(types.SimpleNamespace(
            pack_group=kwargs.get("pack_group"), route=route))
        return route

    monkeypatch.setattr(moe, "_padded_route", spy)
    return seen


@requires_vllm
def test_moe_load_resolves_this_lane_through_this_selector():
    """The wiring under test is this module's functions, not a second copy.

    This is what licenses ``_resolve_lane_at_load`` above to stand in for the
    loader in the forward cases: same predicate, same attestation, same
    attribute.
    """
    from gridbook import moe

    assert moe.bf16_sm120_requested is lane.requested
    assert moe.bf16_sm120_require_lane is lane.require_lane
    assert moe.bf16_sm120_tile_m is lane.tile_m
    assert moe.bf16_sm120_swizzle_group is lane.swizzle_group

    load = inspect.getsource(
        moe.PrismaQuantCBMoEMethod.process_weights_after_loading)
    assert "layer._cb_bf16_sm120 = None" in load
    assert "if bf16_sm120_requested():" in load
    assert "bf16_sm120_require_lane(" in load
    # ...and the hot path reads that attribute, never the environment.
    hot = inspect.getsource(
        moe.PrismaQuantCBMoEMethod._apply_prefill_native_bf16_sm120)
    code = "\n".join(line.split("#", 1)[0] for line in hot.splitlines())
    assert "os.environ" not in code
    assert "PRISMAQUANT_CB_BF16_SM120" not in code


@requires_vllm
@pytest.mark.parametrize("tokens,topk", [(48, 2), (17, 4), (129, 1)],
                         ids=["M48-topk2", "M17-topk4", "M129-topk1"])
def test_the_flag_moves_the_routed_operator_onto_the_lane(monkeypatch, tokens,
                                                          topk):
    """Flag off vs flag on, through ``_apply_prefill_native_bf16`` itself.

    Not the raw op: the whole routed operator — routing, chunked weight
    expansion, activation QDQ before and between the projections, the padded
    layout, the throwaway-row combine — resolved the way a model load resolves
    it. The arms must land inside the suite's documented reassociation band,
    which is the entire numerical claim the opt-in makes.

    ``lane.requested()`` is a PROCESS-STABLE latch that raises if the variable
    moves after dispatch was fixed, so each arm clears it first: that is the
    supported way to simulate two serving processes, and the reset is exactly
    what the file's autouse fixture already does between cases.
    """
    ext = _sm120_lane()
    method, layer, dims = _build(seed=2)
    act = _silu_act()
    ids, weights = _routing(tokens, dims["E"], topk, seed=11)
    torch.manual_seed(3)
    x = torch.randn(tokens, dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    monkeypatch.delenv("PRISMAQUANT_CB_BF16_SM120", raising=False)
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is None
    reference = method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext
    candidate = method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    assert candidate.shape == reference.shape == (tokens, dims["hidden"])
    assert torch.isfinite(candidate).all()
    assert _report(f"sm120-flag-on-vs-off[M={tokens},topk={topk}]",
                   reference, candidate) <= _REL


@requires_vllm
@pytest.mark.parametrize("tokens,topk", [(48, 2), (129, 1)],
                         ids=["M48-topk2", "M129-topk1"])
def test_stage_one_gather_is_bit_identical_to_the_padded_copy(monkeypatch,
                                                              tokens, topk):
    """The in-mainloop gather == the ``[Mp, K]`` copy it replaced, to the BIT.

    ``tests/test_bf16_grouped_cutlass.py`` gates that equality on synthetic
    layouts. This gates it on the REAL one, inside the real dispatch: the
    operator's own ``dest`` vector, its own routing histogram, its own chunk
    slicing and its own expanded weights. The gate is installed as a wrapper
    around the op the prefill calls, so for every launch stage one issues it
    materializes exactly the copy the lane deleted — ``cat([xq, zeros(1,K)])``
    indexed by the row-source vector, ids outside ``[0, T)`` naming the zero
    row — runs the padded-mode op on it, and refuses to return unless the two
    outputs are equal. Nothing here is a tolerance.
    """
    ext = _sm120_lane()
    method, layer, dims = _build(seed=23)
    act = _silu_act()
    ids, weights = _routing(tokens, dims["E"], topk, seed=41)
    torch.manual_seed(24)
    x = torch.randn(tokens, dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext

    from gridbook import ops as pq_ops

    real_gather = pq_ops.cb_bf16_grouped_mm_sm120_gather_out
    launches = []

    def gated_gather(out, a, row_src, w, expert_ids, tile_m):
        real_gather(out, a, row_src, w, expert_ids, tile_m)
        source_rows = int(a.shape[0])
        index = row_src.long()
        index = torch.where((index >= 0) & (index < source_rows), index,
                            torch.full_like(index, source_rows))
        a_pad = torch.cat([a, a.new_zeros((1, a.shape[1]))]) \
            .index_select(0, index).contiguous()
        padded = torch.empty_like(out)
        pq_ops.cb_bf16_grouped_mm_sm120_out(padded, a_pad, w, expert_ids,
                                            tile_m)
        assert torch.equal(out, padded), (
            "the in-mainloop A-row gather loaded different bytes than the "
            "padded activation copy it replaced")
        launches.append((tuple(a.shape), tuple(out.shape)))

    monkeypatch.setattr(pq_ops, "cb_bf16_grouped_mm_sm120_gather_out",
                        gated_gather)
    y = method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    assert launches, "stage one never reached the gather lane"
    assert y.shape == (tokens, dims["hidden"])
    assert torch.isfinite(y).all()


@requires_vllm
def test_stage_one_takes_the_compact_activation_and_stage_two_stays_padded(
        monkeypatch):
    """The padded activation copy is GONE from stage one — and only stage one.

    Shapes are the whole proof. Stage one's A operand has exactly ``T`` rows
    (the compact QDQ'd activation) while its output has the padded row count,
    which is only possible in gather mode; stage two's A operand is the padded
    intermediate itself, so it has nothing compact to gather from and
    deliberately keeps the padded entry point.
    """
    from gridbook import ops as pq_ops

    ext = _sm120_lane()
    tile = lane.tile_m(ext)
    counts = _skewed_counts(tile)
    method, layer, dims = _build(seed=25)
    act = _silu_act()
    ids, weights = _routing_counts(counts, seed=43)
    tokens = int(ids.shape[0])
    n_rows = sum(-(-int(c) // tile) for c in counts) * tile
    torch.manual_seed(26)
    x = torch.randn(tokens, dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    def watch(name, record):
        real = getattr(pq_ops, name)

        def wrapper(*args):
            # args are (out, a, ...) for both entry points.
            record.append((tuple(args[1].shape), tuple(args[0].shape)))
            return real(*args)

        monkeypatch.setattr(pq_ops, name, wrapper)

    gather_calls, padded_calls = [], []
    watch("cb_bf16_grouped_mm_sm120_gather_out", gather_calls)
    watch("cb_bf16_grouped_mm_sm120_out", padded_calls)

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext
    method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    # One chunk covers every expert at this size, so one launch per stage.
    assert gather_calls == [((tokens, dims["hidden"]),
                             (n_rows, 2 * dims["inter"]))]
    assert padded_calls == [((n_rows, dims["inter"]),
                             (n_rows, dims["hidden"]))]
    assert n_rows > tokens, "the case must actually carry padding rows"


@requires_vllm
def test_packed_expert_order_is_bit_neutral_end_to_end(monkeypatch):
    """The packed block order moves rows, never operands — ``torch.equal``.

    Both arms are the real lane on the real operator; the only difference is
    the swizzle group the order is packed against, and a group of 1 has no
    boundary to align to, which is precisely how the kernel's own small-grid
    regime turns the policy off. The case first proves the order really did
    move (otherwise the equality would be trivially true), that it moved WHOLE
    expert segments (same multiset of tile owners), and that the unpacked arm
    is the expert-major order the chunk loop assumes.
    """
    from gridbook import moe

    ext = _sm120_lane()
    counts = _skewed_counts(lane.tile_m(ext))
    method, layer, dims = _build(seed=27)
    act = _silu_act()
    ids, weights = _routing_counts(counts, seed=45)   # top_k=1: pure scatter
    torch.manual_seed(28)
    x = torch.randn(int(ids.shape[0]), dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext

    seen = _spy_padded_route(monkeypatch)
    packed = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    monkeypatch.setattr(moe, "bf16_sm120_swizzle_group", lambda _ext: 1)
    plain = method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    assert seen[0].pack_group == lane.swizzle_group(ext)
    assert seen[1].pack_group == 1
    packed_ids = seen[0].route.expert_ids.tolist()
    plain_ids = seen[1].route.expert_ids.tolist()
    assert packed_ids != plain_ids, "the packed order did not move at all"
    assert sorted(packed_ids) == sorted(plain_ids)   # whole segments only
    assert plain_ids == sorted(plain_ids)            # unpacked = expert-major
    assert torch.equal(packed, plain), (
        "the swizzle-group packed expert order changed the result")


@requires_vllm
def test_expert_chunking_turns_the_packed_order_off(monkeypatch):
    """Packing is offered ONLY when one chunk covers every expert.

    The chunk loop slices blocks as ``block_off[c0]..block_off[c1]`` and so
    assumes expert-major contiguity; a permuted order would hand a chunk
    another expert's tiles. With ``PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK=1`` the
    lane must pass no ``pack_group`` at all, and the resulting tile owners must
    still be non-decreasing.
    """
    ext = _sm120_lane()
    counts = _skewed_counts(lane.tile_m(ext))
    method, layer, dims = _build(seed=29)
    act = _silu_act()
    ids, weights = _routing_counts(counts, seed=47)
    torch.manual_seed(30)
    x = torch.randn(int(ids.shape[0]), dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5

    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    lane._reset_for_tests()
    assert _resolve_lane_at_load(layer, x.device) is ext
    seen = _spy_padded_route(monkeypatch)

    assert method._native_bf16_chunk(layer) >= dims["E"]
    method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    assert seen[-1].pack_group == lane.swizzle_group(ext)
    assert seen[-1].route.expert_ids.tolist() != sorted(
        seen[-1].route.expert_ids.tolist())

    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", "1")
    assert method._native_bf16_chunk(layer) == 1 < dims["E"]
    chunked = method._apply_prefill_native_bf16(layer, x, weights, ids, act)

    assert seen[-1].pack_group is None, (
        "a chunk narrower than E must not receive a permuted block order")
    owners = seen[-1].route.expert_ids.tolist()
    assert owners == sorted(owners), "the chunk loop's expert-major assumption"
    assert chunked.shape == x.shape
    assert torch.isfinite(chunked).all()


# --- ``_padded_route``'s ``pack_group`` on its own (no device, no ext) ------
# Pure host math over the routing histogram, so these run wherever
# ``gridbook.moe`` imports. tile_m 64 / group 8 are what the compiled lane
# reports; the properties asserted hold for any pair.


def _real_pairs(route, tile_m):
    """``{(tile owner, gathered source row)}`` over the non-padding rows."""
    owners = route.expert_ids.repeat_interleave(tile_m)
    keep = ~route.is_pad
    return set(zip(owners[keep].tolist(), route.row_src[keep].tolist()))


@requires_vllm
def test_padded_route_pack_group_permutes_whole_expert_segments():
    """The packed layout is the unpacked one with its SEGMENTS reordered.

    Same tile owners as a multiset, same (owner, source row) pairs, same host
    block offsets, same router weights — only the order of whole expert
    segments differs, and it differs deterministically. That is the entire
    reason the permutation can be called bit-neutral for the GEMM.
    """
    from gridbook.moe import _padded_route

    tile_m, group = 64, 8
    counts = _skewed_counts(tile_m)
    ids, weights = _routing_counts(counts, seed=51, device="cpu")
    experts = len(counts)
    kw = dict(trim=True, block_offsets=True)

    plain = _padded_route(ids, weights, experts, tile_m, **kw)
    packed = _padded_route(ids, weights, experts, tile_m,
                           pack_group=group, **kw)
    again = _padded_route(ids, weights, experts, tile_m,
                          pack_group=group, **kw)

    assert torch.equal(packed.expert_ids, again.expert_ids)
    assert torch.equal(packed.row_src, again.row_src)
    assert torch.equal(packed.dest, again.dest)

    assert not torch.equal(packed.expert_ids, plain.expert_ids)
    assert (sorted(packed.expert_ids.tolist())
            == sorted(plain.expert_ids.tolist()))
    assert _real_pairs(packed, tile_m) == _real_pairs(plain, tile_m)
    assert packed.block_offsets == plain.block_offsets
    assert torch.equal(packed.pw_sorted, plain.pw_sorted)
    assert int(packed.is_pad.sum()) == int(plain.is_pad.sum())

    # Whole BLOCKS moved, so each tile still holds its real rows first: a
    # padding row may never precede a real one inside a tile.
    flags = packed.is_pad.reshape(-1, tile_m).int()
    assert bool((flags[:, 1:] >= flags[:, :-1]).all())
    # ...and padding rows still name the throwaway destination row T.
    assert bool((packed.dest[packed.is_pad] == int(ids.shape[0])).all())


@requires_vllm
def test_padded_route_pack_group_is_inert_without_the_block_offsets():
    """``pack_group`` needs the host block read it derives the order from.

    The docstring says it is ignored otherwise; assert that rather than let a
    caller find out by getting a silently different layout.
    """
    from gridbook.moe import _padded_route

    tile_m = 64
    ids, weights = _routing_counts(_skewed_counts(tile_m), seed=53,
                                   device="cpu")
    experts = len(_skewed_counts(tile_m))

    plain = _padded_route(ids, weights, experts, tile_m, trim=True)
    asked = _padded_route(ids, weights, experts, tile_m, trim=True,
                          pack_group=8)
    assert torch.equal(plain.expert_ids, asked.expert_ids)
    assert torch.equal(plain.row_src, asked.row_src)
    assert plain.block_offsets is None and asked.block_offsets is None
