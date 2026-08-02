"""Correctness gates for the active FP8-CB fused CUTLASS MoE kernel.

``cb_fused_moe_grouped`` decodes CB weights in the CUTLASS GEMM prologue and
serves the whole routed collective with one padded grouped launch per
projection. It is a production candidate only for quality-green FP8-CB shapes;
every miss uses Gridbook's exact-QDQ, exact-weight-expansion, grouped-BF16
CUTLASS bridge, which is therefore the ONLY oracle here — there is no stock,
loop, batched, L2, or runtime-autoselector one. (The per-expert host-loop
"round 1" that used to sit between them was retired on 2026-08-01: the grouped
launch supersedes it, and its own gate could no longer differ. Its cases are
gone; the quality assertions it carried now point at the bridge.)

The routing tests are CPU-only. Forward tests require the serving vLLM/CUDA
stack plus Gridbook's fused and grouped-BF16 extensions.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
from gridbook.moe_routing import cb_grouped_pad_routing  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_process_stable_moe_selectors():
    """Give every selector test the equivalent of a fresh serving process.

    The fused-FP4 selector deliberately rejects changing an execution contract
    after its first read. Several tests below select different contracts, so
    retaining that cache across pytest cases makes the module depend on
    collection/execution order instead of testing one process contract at a
    time. Preserve any state owned by an outer test suite, clear the selector
    for this case, and restore it exactly afterward.
    """

    moe = sys.modules.get("gridbook.moe")
    fused_before = list(moe._FUSED_FP4_MOE_STATE) if moe is not None else None
    if moe is not None:
        moe._FUSED_FP4_MOE_STATE.clear()
    try:
        yield
    finally:
        active = sys.modules.get("gridbook.moe")
        if active is moe and moe is not None:
            moe._FUSED_FP4_MOE_STATE[:] = fused_before
        elif moe is None and active is not None:
            # The test imported Gridbook's MoE module after fixture setup.
            # Its selector cache belongs to that simulated process only.
            active._FUSED_FP4_MOE_STATE.clear()


# --------------------------------------------------------------------------- #
# Routing property (CPU ok): stable expert-sort + cumsum boundaries reproduce   #
# exactly the loop path's per-expert row selection, in the loop's order.        #
# --------------------------------------------------------------------------- #
DEV = "cuda"
# GB10/CUDA 13.0, grouped-fused vs the native BF16 bridge: 1.906–2.039e-2 over
# the four fixed routing cases (measured 2026-08-01, torch 2.11+cu130). Moving
# the oracle off the retired per-expert round did not move this bound: that
# round measured 1.906–2.040e-2 against the same bridge, and the two rounds
# differed from each other by only 2.9–3.9e-4.
# Keep a narrow 2.1e-2 reassociation envelope: the fused path and native BF16
# bridge quantize the same values, but accumulate them in different tensor-core
# types/orders. This is a regression bound, not a claim of bit equivalence.
_REL = 2.1e-2


def _require_stack():
    # Probe a real vLLM submodule because another compatibility test may have
    # installed lightweight ``vllm`` stubs into sys.modules.
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for grouped-fused forward tests")


def _build(*, experts=8, hidden=512, inter=768, seed=0, k=44):
    """Build a synthetic FP8-CB MoE layer without invoking vLLM loading.

    ``k`` is a parameter because TileM=256 is smem-feasible only at k28/k32
    (csrc/cb_fused_gemm.cu's measured table). With the rung hardcoded at 44,
    every 256 gate below skipped unconditionally — so "quality at every
    compiled tile" had no 256 execution behind it at all.
    """
    _require_stack()
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    from gridbook.moe import PrismaQuantCBMoEMethod

    n_sub = 4
    type_size = fmt.nvfp4_cb_type_size(k, "fp8")
    codebook = fmt._resolve_codebook(
        k, "fp8", "product", None, torch.device(DEV))

    torch.manual_seed(seed)
    w13 = torch.randn(
        experts, 2 * inter, hidden, device=DEV) * 0.05
    w2 = torch.randn(experts, hidden, inter, device=DEV) * 0.05
    p13, f13 = fmt.nvfp4_cb_pack(
        w13, k, grid="fp8", mode="product", codebook=codebook)
    p2, f2 = fmt.nvfp4_cb_pack(
        w2, k, grid="fp8", mode="product", codebook=codebook)

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.scheme = {
        "grid": "fp8",
        "mode": "product",
        "k": k,
        "n_sub": n_sub,
        "type_size": type_size,
    }
    method.prefix = "test.grouped_fused"
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
        w13_cb_qweight=p13.reshape(
            experts, 2 * inter, -1).contiguous(),
        w2_cb_qweight=p2.reshape(experts, hidden, -1).contiguous(),
        w13_weight_scale=f13["scales"].reshape(
            experts, 2 * inter).to(DEV).float(),
        w2_weight_scale=f2["scales"].reshape(
            experts, hidden).to(DEV).float(),
        _cb_flat=codec.build_flat_codebook([t.to(DEV) for t in codebook]),
        _cb_compose=torch.zeros(1, device=DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer, {
        "E": experts,
        "hidden": hidden,
        "inter": inter,
    }


def _silu_act():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    try:
        return MoEActivation.from_str("silu")
    except Exception:  # noqa: BLE001 - enum spelling differs across vLLM
        return MoEActivation.SILU


def _routing(tokens, experts, topk, distribution, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if distribution == "uniform":
        ids = torch.stack([
            torch.randperm(experts, generator=generator)[:topk]
            for _ in range(tokens)
        ])
    elif distribution == "subset":
        pool = max(topk, experts // 2)
        ids = torch.stack([
            torch.randperm(pool, generator=generator)[:topk]
            for _ in range(tokens)
        ])
    elif distribution == "one_expert":
        assert topk == 1
        ids = torch.full((tokens, 1), 3, dtype=torch.long)
    else:
        raise ValueError(distribution)
    weights = torch.rand(tokens, topk, generator=generator) + 0.1
    return ids.to(torch.int32).to(DEV), weights.float().to(DEV)


def _report(tag, reference, candidate):
    rel = ((candidate.float() - reference.float()).norm()
           / reference.float().norm().clamp_min(1e-6)).item()
    exact = (candidate == reference).float().mean().item()
    maxabs = (candidate.float() - reference.float()).abs().max().item()
    print(f"{tag}: rel={rel:.3e} exact={exact:.3f} maxabs={maxabs:.3e}")
    return rel


# ---------------------------------------------------------------------------
# Padded routing construction (CPU)
# ---------------------------------------------------------------------------
def _reference_segments(topk_ids, experts):
    return [torch.where(topk_ids == expert)[0] for expert in range(experts)]


def _check_routing(topk_ids, experts, tile_m):
    tokens, topk = topk_ids.shape
    pairs = tokens * topk
    capacity = pairs // tile_m + experts
    expert_ids, row_src, is_pad, n_blocks = cb_grouped_pad_routing(
        topk_ids, experts, tile_m)
    assert expert_ids.shape == (capacity,)
    assert row_src.shape == (capacity * tile_m,)
    assert is_pad.shape == (capacity * tile_m,)

    live_blocks = int(n_blocks)
    assert live_blocks <= capacity
    pair_expert = topk_ids.reshape(-1).to(torch.long)
    pair_token = torch.arange(tokens).repeat_interleave(topk)
    order = torch.argsort(pair_expert, stable=True)
    sorted_tokens = pair_token[order]
    counts = torch.bincount(pair_expert, minlength=experts)

    assert torch.equal(
        expert_ids[live_blocks:],
        torch.full((capacity - live_blocks,), -1, dtype=torch.int32),
    )
    assert bool(is_pad[live_blocks * tile_m:].all())
    seen = {expert: [] for expert in range(experts)}
    for block in range(live_blocks):
        expert = int(expert_ids[block])
        sl = slice(block * tile_m, (block + 1) * tile_m)
        block_pad = is_pad[sl]
        assert not bool(
            block_pad[:-1].bitwise_and(~block_pad[1:]).any())
        seen[expert].extend(int(value) for value in row_src[sl][~block_pad])

    references = _reference_segments(topk_ids, experts)
    for expert in range(experts):
        assert len(seen[expert]) == int(counts[expert])
        actual = (sorted_tokens[torch.tensor(seen[expert], dtype=torch.long)]
                  if seen[expert] else torch.empty(0, dtype=torch.long))
        assert torch.equal(actual, references[expert])
        assert sum(
            int(expert_ids[block]) == expert
            for block in range(live_blocks)
        ) == -(-int(counts[expert]) // tile_m)
    return live_blocks, capacity


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_routing_boundaries_match_stable_reference(topk):
    torch.manual_seed(0)
    tokens, experts = 37, 11
    ids = torch.stack([
        torch.randperm(experts)[:topk] for _ in range(tokens)
    ]).to(torch.int32)
    _check_routing(ids, experts, 8)


@pytest.mark.parametrize("tile_m", [4, 8, 128, 256])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_padded_routing_matches_stable_reference(tile_m, topk):
    torch.manual_seed(0)
    tokens, experts = 37, 11
    ids = torch.stack([
        torch.randperm(experts)[:topk] for _ in range(tokens)
    ]).to(torch.int32)
    _check_routing(ids, experts, tile_m)


@pytest.mark.parametrize(
    "tile_m,expected_blocks", [(4, 4), (8, 2), (128, 1), (256, 1)])
def test_zero_row_experts_consume_no_blocks(tile_m, expected_blocks):
    ids = torch.full((16, 1), 3, dtype=torch.int32)
    expert_ids, _rows, _pads, blocks = cb_grouped_pad_routing(ids, 8, tile_m)
    assert int(blocks) == expected_blocks
    assert torch.equal(
        expert_ids[:expected_blocks],
        torch.full((expected_blocks,), 3, dtype=torch.int32),
    )


@pytest.mark.parametrize("seed", list(range(24)))
def test_padded_routing_capacity_bound_randomized(seed):
    generator = torch.Generator().manual_seed(seed)
    experts = int(torch.randint(1, 17, (1,), generator=generator))
    topk = int(torch.randint(
        1, min(experts, 6) + 1, (1,), generator=generator))
    tokens = int(torch.randint(1, 40, (1,), generator=generator))
    tile_m = [1, 2, 4, 8, 128, 256][seed % 6]
    ids = torch.randint(
        0, experts, (tokens, topk), generator=generator, dtype=torch.int32)
    live, capacity = _check_routing(ids, experts, tile_m)
    assert live <= capacity


# ---------------------------------------------------------------------------
# CUDA quality and dispatch gates
# ---------------------------------------------------------------------------
def _require_grouped_fused(method, layer):
    _require_stack()
    if not method._gf2_ok(layer):
        pytest.skip("FP8-CB grouped fused CUTLASS prefill unavailable")
    from gridbook.cuda_ext import get_bf16_grouped_ext
    if get_bf16_grouped_ext() is None:
        pytest.skip("owned grouped-BF16 CUTLASS reference unavailable")


@pytest.mark.parametrize("distribution", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [2, 4])
def test_grouped_fused_matches_native_quality_bridge(distribution, topk):
    method, layer, dims = _build(seed=1)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    tokens = 48
    ids, weights = _routing(
        tokens, dims["E"], topk, distribution, seed=7)
    torch.manual_seed(2)
    x = torch.randn(
        tokens, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    reference = method._apply_prefill_native_bf16(
        layer, x, weights, ids, act)
    candidate = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act)
    assert candidate is not None
    rel = _report(
        f"grouped-vs-native[{distribution},topk={topk}]", reference, candidate)
    assert rel <= _REL


@pytest.mark.parametrize("tokens", [17, 33, 129])
def test_grouped_fused_partial_tiles_match_native_quality_bridge(tokens):
    method, layer, dims = _build(seed=3)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    ids, weights = _routing(
        tokens, dims["E"], 2, "uniform", seed=5)
    torch.manual_seed(4)
    x = torch.randn(
        tokens, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    reference = method._apply_prefill_native_bf16(
        layer, x, weights, ids, act)
    candidate = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act)
    assert candidate is not None
    assert _report(
        f"grouped-vs-native[M={tokens}]", reference, candidate) <= _REL


def test_padding_trim_is_bit_identical(monkeypatch):
    method, layer, dims = _build(seed=11)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    ids, weights = _routing(33, dims["E"], 2, "uniform", seed=3)
    torch.manual_seed(12)
    x = torch.randn(
        33, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "1")
    trimmed = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act)
    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "0")
    full = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act)
    assert torch.equal(trimmed, full)


def test_native_dispatch_prefers_the_grouped_fused_kernel():
    method, layer, dims = _build(seed=14)
    _require_grouped_fused(method, layer)
    seen = {}
    original = method._apply_prefill_grouped_fused_v2

    def spy(*args, **kwargs):
        seen["hit"] = True
        return original(*args, **kwargs)

    method._apply_prefill_grouped_fused_v2 = spy
    ids, weights = _routing(32, dims["E"], 2, "uniform", seed=2)
    x = torch.randn(
        32, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    out = method._apply_inline(layer, x, weights, ids)
    assert seen.get("hit")
    assert out.shape == (32, dims["hidden"])


def test_native_dispatch_falls_from_grouped_fused_to_native_bridge():
    """An ineligible layer goes straight to the owned BF16 bridge.

    This is the whole fallback cascade now that the per-expert host loop is
    retired: exactly one fused arm, then the exact native route. Nothing may
    sit between them.
    """
    method, layer, dims = _build(seed=13)
    _require_stack()
    from gridbook.cuda_ext import get_bf16_grouped_ext
    if get_bf16_grouped_ext() is None:
        pytest.skip("owned grouped-BF16 CUTLASS reference unavailable")
    layer._cb_gf2_ok = False
    seen = {}
    bridge = method._apply_prefill_native_bf16
    fused = method._apply_prefill_grouped_fused_v2

    def bridge_spy(*args, **kwargs):
        seen["bridge"] = True
        return bridge(*args, **kwargs)

    def fused_spy(*args, **kwargs):
        seen["fused"] = fused(*args, **kwargs)
        return seen["fused"]

    method._apply_prefill_native_bf16 = bridge_spy
    method._apply_prefill_grouped_fused_v2 = fused_spy
    ids, weights = _routing(32, dims["E"], 2, "uniform", seed=2)
    x = torch.randn(
        32, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    out = method._apply_inline(layer, x, weights, ids)
    assert "fused" in seen and seen["fused"] is None  # the gate declined
    assert seen.get("bridge")                         # the bridge served it
    assert out.shape == (32, dims["hidden"])


def _tile_sizes(method, layer):
    sizes = method._gf2_tile_sizes(layer)
    if not sizes:
        pytest.skip("extension exposes no grouped TileM")
    return sizes


def test_quality_at_every_compiled_tile():
    method, layer, dims = _build(seed=1)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    ids, weights = _routing(48, dims["E"], 2, "uniform", seed=7)
    torch.manual_seed(2)
    x = torch.randn(
        48, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    reference = method._apply_prefill_native_bf16(
        layer, x, weights, ids, act)
    for tile_m in _tile_sizes(method, layer):
        candidate = method._apply_prefill_grouped_fused_v2(
            layer, x, weights, ids, act, tile_m=tile_m)
        assert candidate is not None
        assert _report(
            f"grouped[tile={tile_m}]-vs-native", reference, candidate) <= _REL


@pytest.mark.parametrize(
    "distribution,tokens", [("one_expert", 40), ("subset", 17),
                             ("uniform", 33)])
def test_ragged_routing_at_tile_256(distribution, tokens):
    method, layer, dims = _build(seed=6)
    _require_grouped_fused(method, layer)
    if 256 not in method._gf2_tile_sizes(layer):
        pytest.skip("tile_m=256 not compiled")
    act = _silu_act()
    topk = 1 if distribution == "one_expert" else 2
    ids, weights = _routing(
        tokens, dims["E"], topk, distribution, seed=1)
    torch.manual_seed(8)
    x = torch.randn(
        tokens, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    reference = method._apply_prefill_native_bf16(
        layer, x, weights, ids, act)
    candidate = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act, tile_m=256)
    assert candidate is not None
    assert _report(
        f"grouped[tile=256,{distribution},M={tokens}]-vs-native",
        reference, candidate) <= _REL


# ---------------------------------------------------------------------------
# The OPT-IN sm12x-native bridge lane, end to end through the routed operator
# ---------------------------------------------------------------------------
def _sm120_bridge_lane():
    """The grouped-BF16 extension when it carries the sm12x lane, else skip."""
    _require_stack()
    from gridbook.cuda_ext import get_bf16_grouped_ext

    ext = get_bf16_grouped_ext()
    if ext is None:
        pytest.skip("owned grouped-BF16 CUTLASS extension unavailable")
    if not hasattr(ext, "cb_bf16_grouped_mm_sm120"):
        pytest.skip("this build carries no sm12x lane (needs cc 12.0/12.1)")
    return ext


@pytest.mark.parametrize("distribution,tokens,topk", [
    ("uniform", 48, 2),
    ("subset", 17, 4),
    ("one_expert", 129, 1),
])
def test_sm120_bridge_lane_matches_the_default_bridge(distribution, tokens,
                                                      topk):
    """Both bridge lanes, one routed operator, end to end.

    This is the integration counterpart to the kernel gates in
    ``test_bf16_grouped_cutlass.py``: it runs the WHOLE quality bridge —
    routing, weight expansion over expert CHUNKS, activation QDQ before and
    between the projections, the router combine — once on the default
    exact-segment SM80 lane and once on the padded sm12x lane, and requires the
    two to agree to the suite's reassociation contract. It is the only test
    that exercises the padded gather, the per-expert block offsets each chunked
    launch slices with, and the throwaway-row scatter together.
    """
    ext = _sm120_bridge_lane()
    method, layer, dims = _build(seed=2)
    act = _silu_act()
    ids, weights = _routing(tokens, dims["E"], topk, distribution, seed=11)
    torch.manual_seed(3)
    x = torch.randn(
        tokens, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    layer._cb_bf16_sm120 = None
    reference = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    layer._cb_bf16_sm120 = ext
    try:
        candidate = method._apply_prefill_native_bf16(
            layer, x, weights, ids, act)
    finally:
        layer._cb_bf16_sm120 = None

    assert candidate.shape == reference.shape
    assert torch.isfinite(candidate).all()
    assert _report(
        f"sm120-lane-vs-sm80-bridge[{distribution},M={tokens},topk={topk}]",
        reference, candidate) <= _REL


def test_sm120_bridge_lane_survives_a_single_expert_chunk():
    """One expert per chunk: every launch takes its own block sub-range.

    ``PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK=1`` makes the chunk loop issue E
    launches, each over the tiles of exactly one expert — the configuration
    where a wrong block offset or a missing ``expert_ids - c0`` rebase would
    multiply rows by another expert's weights instead of failing loudly.
    """
    ext = _sm120_bridge_lane()
    method, layer, dims = _build(seed=4)
    act = _silu_act()
    ids, weights = _routing(40, dims["E"], 2, "uniform", seed=13)
    torch.manual_seed(5)
    x = torch.randn(
        40, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    layer._cb_bf16_sm120 = None
    reference = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    previous = os.environ.get("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")
    os.environ["PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK"] = "1"
    layer._cb_bf16_sm120 = ext
    try:
        candidate = method._apply_prefill_native_bf16(
            layer, x, weights, ids, act)
    finally:
        layer._cb_bf16_sm120 = None
        if previous is None:
            os.environ.pop("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", None)
        else:
            os.environ["PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK"] = previous

    assert _report("sm120-lane[chunk=1]-vs-sm80-bridge",
                   reference, candidate) <= _REL


# ===========================================================================
# K0.4 — the grouped TileM SELECTOR.
#
# CPU-only, no vLLM, no GPU: the selector is pure integer arithmetic over
# host-known shapes, which is exactly why it lives in ``moe_routing`` and why
# it is testable here next to the routing construction.
# ===========================================================================
from gridbook.moe_routing import (  # noqa: E402
    GROUPED_TILE_M_BASE,
    GROUPED_TILE_M_WIDE,
    GROUPED_WIDE_TILE_MIN_ROWS_PER_EXPERT as _RHO_MIN,
    cb_grouped_tile_m,
)

_SEL = dict(hidden=4096, inter=2048, tile_n=64, compiled=(128, 256),
            sm_count=48, k_bits=28)


def _sel(**kw):
    args = dict(_SEL)
    args.update(kw)
    return cb_grouped_tile_m(**args)


@pytest.mark.parametrize("tokens,top_k,experts,expect,why", [
    # rho = P/E must EXCEED the threshold; the boundary itself stays narrow.
    (_RHO_MIN * 128 // 8, 8, 128, GROUPED_TILE_M_BASE, "rho == threshold"),
    ((_RHO_MIN + 1) * 128 // 8, 8, 128, GROUPED_TILE_M_WIDE, "rho > threshold"),
    (16, 1, 8, GROUPED_TILE_M_BASE, "P < 256 shape guard"),
    (4, 1, 4, GROUPED_TILE_M_BASE, "tiny routing"),
    (1 << 16, 8, 1, GROUPED_TILE_M_WIDE, "one expert takes everything"),
])
def test_selector_boundaries(tokens, top_k, experts, expect, why):
    assert _sel(tokens=tokens, top_k=top_k, experts=experts) == expect, why


def test_selector_fails_closed_without_device_metadata():
    """A failed SM probe must pick the incumbent tile, exactly like the dense
    selector does — an unknown device is not a reason to widen."""
    assert _sel(tokens=1 << 16, top_k=8, experts=1, sm_count=0) == \
        GROUPED_TILE_M_BASE


def test_selector_never_leaves_the_compiled_set():
    """The failure this prevents is a hard abort: cb_fused_moe_grouped
    TORCH_CHECKs `moe_tile_supported(tile_m, k_bits)`, so proposing an
    uncompiled tile aborts the request rather than degrading it."""
    for compiled in ((128,), (256,), (128, 256)):
        for tokens in (1, 64, 4096, 1 << 16):
            for experts in (1, 8, 256):
                got = _sel(tokens=tokens, top_k=4, experts=experts,
                           compiled=compiled)
                assert got in compiled
    assert _sel(tokens=1 << 16, top_k=8, experts=1, compiled=()) == 0


def test_selector_refuses_the_zero_margin_rung():
    """TileM=256/k32 lands on EXACTLY the 101,376 B smem ceiling. It is
    compiled but launch-unverified, so the SELECTOR must not choose it; an
    explicit operator override still can."""
    wide = dict(tokens=1 << 16, top_k=8, experts=1)
    assert _sel(k_bits=28, **wide) == GROUPED_TILE_M_WIDE
    assert _sel(k_bits=32, **wide) == GROUPED_TILE_M_BASE


def _exact_cost(counts, tile, x):
    """B(t) * (d + t*m) with d/m = x, in units of m."""
    blocks = sum((c + tile - 1) // tile for c in counts if c > 0)
    return blocks * (x + tile)


@pytest.mark.parametrize("name", ["uniform", "skewed", "tiny", "adversarial"])
def test_selector_verdict_is_correct_for_every_histogram(name):
    """THE property, not a restatement of the rule.

    The selector sees only (P, E). Whenever it says 256, the EXACT cost model
    evaluated on the real histogram must agree — for every histogram summing to
    that P. ``adversarial`` is the worst case the derivation is built around
    (c_e = 128 mod 256 maximises the padding penalty of widening while buying
    no tile-count reduction at all).
    """
    x = 85                       # pessimistic end of the dense-inverted range
    experts = 8
    for scale in (1, 4, 16, 64, 256):
        if name == "uniform":
            counts = [128 * scale] * experts
        elif name == "skewed":
            counts = [128 * scale * experts - (experts - 1)] + \
                     [1] * (experts - 1)
        elif name == "tiny":
            counts = [1] * min(experts, scale)
        else:
            counts = [128 + 256 * (scale - 1)] * experts
        pairs = sum(counts)
        if pairs < 1:
            continue
        got = cb_grouped_tile_m(
            tokens=pairs, top_k=1, experts=experts, hidden=4096, inter=2048,
            tile_n=64, compiled=(128, 256), sm_count=48, k_bits=28)
        if got == GROUPED_TILE_M_WIDE:
            assert _exact_cost(counts, 256, x) < _exact_cost(counts, 128, x), (
                f"{name}: selector widened but the exact model prefers 128 "
                f"(counts={counts})")


def test_selector_reads_no_tensor():
    """Graph safety, checked by construction rather than by audit.

    The persistent-B lesson is that "no .item() in my code" is not a proof — a
    sync hides inside innocuous ATen (bincount sizes its CUDA output from
    .max().item()). The selector's defence is stronger and mechanically
    checkable: its signature admits no tensor at all, so there is nothing that
    COULD sync.
    """
    import inspect

    sig = inspect.signature(cb_grouped_tile_m)
    assert all(p.kind is p.KEYWORD_ONLY for p in sig.parameters.values())
    # Every argument the live call site passes is a python int (or a sequence
    # of ints); a tensor argument would raise here rather than sync silently.
    for value in (_SEL["hidden"], _SEL["inter"], _SEL["tile_n"],
                  _SEL["sm_count"], _SEL["k_bits"]):
        assert isinstance(value, int)
    assert isinstance(_sel(tokens=64, top_k=2, experts=4), int)


def test_routing_counts_avoid_the_bincount_host_sync():
    """The padded routing must count with scatter_add_, not bincount.

    ATen's CUDA ``bincount`` sizes its output from ``.max().item()`` and so
    host-syncs — the persistent-B lane proved it breaks capture, with a
    negative control. ``cb_grouped_pad_routing``'s docstring claimed "NO HOST
    READS" while calling it anyway, which made the graph-safety story for every
    padded grouped lane false. Pin the repair here so it cannot regress.
    """
    import gridbook.moe_routing as mr

    import inspect

    # The CALL, not the word: the module documents at length why it does not
    # use bincount, so the prose legitimately names it.
    assert "torch.bincount(" not in inspect.getsource(mr)
    counts = mr._expert_counts(torch.tensor([0, 2, 2, 5, 5, 5]), 6)
    assert counts.tolist() == [1, 0, 2, 0, 0, 3]
    assert counts.dtype == torch.int64


# ===========================================================================
# K0.4 — dispatch TELEMETRY and TILE EQUALITY (forward tests).
# ===========================================================================
def test_grouped_fused_emits_the_full_dispatch_record():
    """Every K0.4 field, populated, on a served routed call."""
    from gridbook.nvfp4_activation_contract import (ROUTE_CONTRACTS,
                                                    ROUTE_FIELDS, read_route)

    method, layer, dims = _build(seed=31)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    ids, weights = _routing(48, dims["E"], 2, "uniform", seed=3)
    torch.manual_seed(4)
    x = torch.randn(48, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    out = method._apply_prefill_grouped_fused_v2(layer, x, weights, ids, act)
    assert out is not None

    record = read_route(layer)
    assert record is not None and set(record) == set(ROUTE_FIELDS)
    assert record["kind"] == "moe"
    assert record["state"] == "served" and record["reason"] is None
    assert record["symbol"] == "cb_fused_moe_grouped"
    assert record["contract"] in ROUTE_CONTRACTS
    assert record["tile_m"] in method._gf2_tile_sizes(layer)
    assert record["shape"] == (f"T48:P96:E{dims['E']}:H{dims['hidden']}"
                               f":I{dims['inter']}:topk2")
    # Selector provenance: enough to re-derive the verdict from a JSON report
    # with no GPU present.
    assert record["tile_rho"] == 96 // dims["E"]
    assert record["tile_candidate_ctas"] > 0
    assert record["tile_compiled"] == ",".join(
        str(t) for t in method._gf2_tile_sizes(layer))


def test_grouped_fused_records_the_exact_fallback_reason():
    """A declined gate must say WHY, and the bridge that serves it must
    overwrite the record — the last-write-wins semantics the probe relies on."""
    from gridbook.nvfp4_activation_contract import read_route

    method, layer, dims = _build(seed=32)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    ids, weights = _routing(32, dims["E"], 2, "uniform", seed=5)
    torch.manual_seed(6)
    x = torch.randn(32, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    layer._cb_gf2_ok = False
    layer._cb_gf2_ok_reason = "sentinel: gate declined for a stated reason"
    assert method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act) is None
    record = read_route(layer)
    assert record["state"] == "fallback"
    assert record["reason"] == "sentinel: gate declined for a stated reason"
    assert record["symbol"] == ""


def test_gf2_gate_reason_names_the_failing_clause():
    """The FP8 grouped gate recorded a bare bool before K0.4, so every routed
    fp8 fallback looked alike in a dispatch report."""
    method, layer, _ = _build(seed=33)
    _require_stack()
    layer._cb_hidden = layer._cb_hidden + 1          # break ONE clause
    layer.__dict__.pop("_cb_gf2_ok", None)
    assert method._gf2_ok(layer) is False
    assert "superblock aligned" in layer._cb_gf2_ok_reason


def _pair_order(y, ids, experts, tile_m):
    """Stage output in stable-argsorted PAIR order — tile-independent."""
    _eids, row_src, is_pad, _n = cb_grouped_pad_routing(ids, experts, tile_m)
    row_src = row_src[:y.shape[0]]
    is_pad = is_pad[:y.shape[0]]
    keep = ~is_pad
    order = torch.argsort(row_src[keep], stable=True)
    return y[keep][order]


@pytest.mark.parametrize("distribution,tokens,topk",
                         [("uniform", 300, 2), ("subset", 257, 4),
                          ("one_expert", 400, 1)])
def test_both_compiled_tiles_are_bit_identical(distribution, tokens, topk):
    """The selector must be a PURE PERFORMANCE choice.

    The two tiles differ only in how rows are partitioned across CTAs: each
    output row's K-reduction is the same ordered sequence of TileK=128 chunks
    under the same TiledMma, the padding rows are inert by construction, and
    the stable argsort fixes each expert's row order independently of tile_m.
    So a bit difference here is a real defect with exactly two possible causes
    — per-row accumulation depending on TileM (which would break the
    `is_same_v<MoeTile<128>, TileF>` identity the whole ladder rests on), or
    the routing's row->pair mapping changing with the tile.

    Compared PRE-COMBINE, in pair order: the final combine is an index_add_
    into a bf16 accumulator whose index length changes with the tile, and
    atomic ordering there is not architecturally guaranteed.
    """
    method, layer, dims = _build(seed=34, k=28)      # 256 needs k in {28, 32}
    _require_grouped_fused(method, layer)
    if 256 not in method._gf2_tile_sizes(layer):
        pytest.skip("tile_m=256 not compiled for this rung")
    act = _silu_act()
    ids, weights = _routing(tokens, dims["E"], topk, distribution, seed=9)
    torch.manual_seed(10)
    x = torch.randn(
        tokens, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    captured: dict[int, list] = {}
    original = method._grouped_call

    def spy(fext, args, tile_m):
        out = original(fext, args, tile_m)
        captured.setdefault(tile_m, []).append(out)
        return out

    method._grouped_call = spy
    try:
        out128 = method._apply_prefill_grouped_fused_v2(
            layer, x, weights, ids, act, tile_m=128)
        out256 = method._apply_prefill_grouped_fused_v2(
            layer, x, weights, ids, act, tile_m=256)
    finally:
        method._grouped_call = original
    assert out128 is not None and out256 is not None
    assert len(captured[128]) == 2 and len(captured[256]) == 2

    for stage in (0, 1):
        a = _pair_order(captured[128][stage], ids, dims["E"], 128)
        b = _pair_order(captured[256][stage], ids, dims["E"], 256)
        assert a.shape == b.shape
        assert torch.equal(a.view(torch.uint16), b.view(torch.uint16)), (
            f"stage {stage} differs between TileM 128 and 256")
