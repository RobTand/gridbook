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
# GB10/CUDA 13.0 measured 2.015–2.040e-2 on the four fixed routing cases.
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


def _build(*, experts=8, hidden=512, inter=768, seed=0):
    """Build a synthetic FP8-CB MoE layer without invoking vLLM loading."""
    _require_stack()
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    from gridbook.moe import PrismaQuantCBMoEMethod

    k, n_sub = 44, 4
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
