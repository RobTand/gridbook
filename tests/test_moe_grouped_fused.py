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
# GB10/CUDA 13.0, torch 2.11+cu130. TWO comparison classes live in this file
# and they do NOT share an envelope.
#
# _REL — SAME-REPRESENTATION. Both sides are BF16-operand lanes over identical
# expanded weights and identical QDQ'd activations (the sm12x bridge lane vs
# the default sm80 one). Only the FP32 reduction order differs, so this really
# is reassociation, and 2.1e-2 is loose for it. Unchanged.
#
# _REL_FUSED — CROSS-REPRESENTATION, and NOT reassociation. The grouped fused
# lane hands the tensor cores FP8 operands with FP32 EVT scales; the bridge
# hands them BF16 operands it obtained by rounding ``code * scale`` to BF16.
# The two lanes therefore do not multiply the same numbers (measured
# 2026-08-02, uniform/M=33/k28):
#   * their per-token E4M3 activation quantizers — vLLM's
#     ``dynamic_per_token_scaled_fp8_quant`` for the fused lane, Gridbook's
#     ``fp8_act_qdq`` for the bridge — put 0.82% of elements on DIFFERENT
#     codes. Both derive scale = amax/448 (agreeing to 1.2e-7); they break
#     bin-boundary ties differently, and each flip is a full FP8 ULP;
#   * the bridge's BF16 dequant of ``code * scale`` costs another 1.5e-3;
#   * and the pipeline then RE-quantizes the intermediate to E4M3, whose 3-bit
#     mantissa turns that ~2.3e-3 input perturbation into a few percent of
#     flipped intermediate codes, each again worth a full ULP.
# So this disagreement is AMPLIFICATION-dominated: it moves with routing luck
# and does not shrink as either kernel improves.
#
# 2.1e-2 was fitted to FOUR hand-picked routing cases (1.906–2.039e-2, measured
# 2026-08-01) — a four-sample fit to a distribution that reaches further. Over
# 224 configurations (both compiled rungs x 7 build seeds x 8 routing shapes x
# 2 routing seeds, measured 2026-08-02) the disagreement runs 1.566e-2 to
# 2.316e-2, with 23/224 above 2.1e-2 — and the DEFAULT k=44 rung breaches it
# more often (13/112) than k28 does (10/112) and owns the maximum. Every new
# quality case added to this file therefore had a ~1-in-10 chance of failing on
# arrival; ``test_ragged_routing_at_tile_256[uniform-33]`` (2.174e-2) drew one.
# 2.5e-2 clears the measured maximum by 8%.
#
# Loosening it costs nothing that was being gated, because the number it
# loosens was never the sharp claim: a fused-vs-bridge tolerance cannot say
# WHICH lane moved. That claim is pinned separately and far more tightly by
# ``test_fused_lane_is_no_less_exact_than_the_bridge`` (both lanes against the
# exact FP32 computation) and, on the tile dimension, at bit equality by
# ``test_both_compiled_tiles_are_bit_identical``.
_REL = 2.1e-2
_REL_FUSED = 2.5e-2


def _require_stack():
    # Probe a real vLLM submodule because another compatibility test may have
    # installed lightweight ``vllm`` stubs into sys.modules.
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for grouped-fused forward tests")


def _build(*, experts=8, hidden=512, inter=768, seed=0, k=44, codebook=None):
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
    # `codebook=None` keeps the lattice book every existing caller gets. A
    # caller may pass its own sub-tables to build a layer whose weights are
    # PACKED against that book, which is what a per-role artifact needs: three
    # roles encoded against three different books, not one book grafted onto
    # bytes encoded for another.
    codebook = fmt._resolve_codebook(
        k, "fp8", "product", codebook, torch.device(DEV))

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
    assert rel <= _REL_FUSED


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
        f"grouped-vs-native[M={tokens}]", reference, candidate) <= _REL_FUSED


def test_padding_trim_is_bit_identical(monkeypatch):
    method, layer, dims = _build(seed=11)
    _require_grouped_fused(method, layer)
    act = _silu_act()
    ids, weights = _routing(33, dims["E"], 2, "uniform", seed=3)
    torch.manual_seed(12)
    x = torch.randn(
        33, dims["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    # The trim flag is process-stable since 2026-08-02, so flipping it mid-run
    # raises rather than taking effect. An operator switches arms by
    # restarting; clearing that one latch is this test's stand-in for the
    # restart, and running both arms in one process is what makes the
    # bit-identity claim checkable at all.
    from gridbook import lane_select

    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "1")
    lane_select.reset_for_tests("PRISMAQUANT_CB_GROUPED_TRIM")
    trimmed = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act)
    monkeypatch.setenv("PRISMAQUANT_CB_GROUPED_TRIM", "0")
    lane_select.reset_for_tests("PRISMAQUANT_CB_GROUPED_TRIM")
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


@pytest.mark.parametrize("k", [28, 44])
def test_quality_at_every_compiled_tile(k):
    """k is parametrized because TileM=256 is smem-feasible only at k28/k32.
    At the previously hardcoded k44 this test had exactly one compiled tile, so
    "every compiled tile" was a claim with one arm behind it."""
    method, layer, dims = _build(seed=1, k=k)
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
            f"grouped[tile={tile_m}]-vs-native",
            reference, candidate) <= _REL_FUSED


@pytest.mark.parametrize(
    "distribution,tokens", [("one_expert", 40), ("subset", 17),
                             ("uniform", 33)])
def test_ragged_routing_at_tile_256(distribution, tokens):
    # k28, not the fixture default k44: TileM=256 is smem-infeasible above k32,
    # so at k44 this test skipped unconditionally and the 256 path had no live
    # coverage at all.
    method, layer, dims = _build(seed=6, k=28)
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
        reference, candidate) <= _REL_FUSED


# ---------------------------------------------------------------------------
# The claim `_REL_FUSED` is too loose to make: WHICH lane is wrong?
# ---------------------------------------------------------------------------
def _fp8_qdq_f32(a):
    """Per-token E4M3 quantize-dequantize as MATH, in FP32.

    Deliberately neither lane's kernel: privileging one lane's quantizer here
    would hand that lane a head start on the very comparison below.
    """
    amax = a.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    code = (a.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return code.float() * scale


def _exact_weight_f32(method, layer, which, experts):
    """The dequantized weight as ``fp8 grid value * fp32 per-row scale``.

    This is exactly the product the FUSED lane forms (FP8 operand, FP32 EVT
    scale) and that the BRIDGE additionally rounds to BF16 before its GEMM —
    so it is the unrounded value both approximate, not either one's operand.
    Mirrors ``PrismaQuantCBMoEMethod._expand_native_bf16_slice`` up to that
    final ``.to(torch.bfloat16)``.
    """
    from gridbook import ops as pq_ops
    from gridbook.moe_routing import cb_cached_row_offsets

    packed = getattr(layer, f"{which}_cb_qweight")[:experts].contiguous()
    out_f = int(packed.shape[1])
    in_f = int(layer._cb_hidden if which == "w13" else layer._cb_inter)
    rows = experts * out_f
    raw = codec.pad_qweight(packed.reshape(rows, -1))
    row0 = cb_cached_row_offsets(layer, rows, packed.device)
    value = pq_ops.cb_expand_fp8(
        raw, method._stock_cb_flat_fp8(layer), row0, rows, in_f,
        method.k, method.n_sub, method.type_size)
    scale = getattr(layer, f"{which}_weight_scale")[:experts] \
        .reshape(rows).to(torch.float32)
    return (value.float() * scale[:, None]).view(experts, out_f, in_f)


def _exact_fp32_moe(method, layer, dims, x, topk_weights, topk_ids):
    """The routed collective in FP32, carrying ONLY the two mandated E4M3
    activation quantizations — the computation both lanes approximate.

    One (token, expert) pair at a time, so no cross-pair accumulation order is
    baked in either. Small-M only: it is a Python loop over routed pairs.
    """
    experts, hidden = dims["E"], dims["hidden"]
    inter = dims["inter"]
    tokens, topk = topk_ids.shape
    w13 = _exact_weight_f32(method, layer, "w13", experts)
    w2 = _exact_weight_f32(method, layer, "w2", experts)
    xq = _fp8_qdq_f32(x)

    pair_expert = topk_ids.reshape(-1).to(torch.long)
    pair_token = torch.arange(
        tokens, device=topk_ids.device).repeat_interleave(topk)
    pair_weight = topk_weights.reshape(-1).float()

    out = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for p in range(int(pair_expert.numel())):
        e, t = int(pair_expert[p]), int(pair_token[p])
        gate_up = xq[t] @ w13[e].t()
        activated = torch.nn.functional.silu(gate_up[:inter]) \
            * gate_up[inter:]
        out[t] += pair_weight[p] * (_fp8_qdq_f32(activated) @ w2[e].t())
    return out


@pytest.mark.parametrize(
    "distribution,tokens", [("one_expert", 40), ("subset", 17),
                            ("uniform", 33)])
def test_fused_lane_is_no_less_exact_than_the_bridge(distribution, tokens):
    """Measure BOTH lanes against the exact FP32 computation.

    A fused-vs-bridge tolerance cannot say which lane moved, which is why
    ``_REL_FUSED`` had to be widened to cover routing luck rather than tuned to
    catch a defect. This is the gate that catches the defect: a real numerics
    fault in the fused lane — a mis-indexed padded row, a wrong per-tile expert
    id, a dropped or mis-broadcast EVT scale — drives the fused lane AWAY from
    exact while the bridge stays put, and that is a signed, per-lane signal a
    symmetric disagreement bound cannot produce.

    Same configuration as ``test_ragged_routing_at_tile_256`` above, and the
    same TileM=256, so the cell that test failed on is the cell this one
    attests. Measured 2026-08-02 over these three cases plus three more: the
    bridge sits 1.899–2.082e-2 from exact and the fused lane 1.853–2.025e-2,
    ratio 0.89–1.05x. The two lanes are equally accurate, and the ~2e-2 they
    disagree by is the norm of two independent error vectors of that size —
    not a defect in either. 1.25x leaves 19% headroom over the measured worst.
    """
    method, layer, dims = _build(seed=6, k=28)
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

    exact = _exact_fp32_moe(method, layer, dims, x, weights, ids)
    bridge = method._apply_prefill_native_bf16(layer, x, weights, ids, act)
    fused = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act, tile_m=256)
    assert fused is not None

    tag = f"[{distribution},M={tokens}]"
    err_bridge = _report(f"bridge-vs-exact{tag}", exact, bridge)
    err_fused = _report(f"fused[tile=256]-vs-exact{tag}", exact, fused)

    # Each lane is independently a quality-green approximation...
    assert err_bridge <= _REL_FUSED
    assert err_fused <= _REL_FUSED
    # ...and the fused lane is not the one carrying the error.
    assert err_fused <= 1.25 * err_bridge, (
        f"the fused lane is {err_fused / err_bridge:.3f}x farther from the "
        f"exact FP32 computation than the bridge is; measured range is "
        f"0.89-1.05x, so this is a fused-lane numerics regression, not the "
        f"reassociation the disagreement bound tolerates")


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
    # Same restart stand-in as the trim A/B above: the chunk knob is latched
    # process-stable because it gates the packed expert ORDER, and this case
    # deliberately runs both settings in one process to compare them.
    from gridbook import lane_select

    previous = os.environ.get("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")
    os.environ["PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK"] = "1"
    lane_select.reset_for_tests("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")
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
        lane_select.reset_for_tests("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")

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


def _gridbook_source(name: str) -> str:
    """One packaged module's TEXT, without importing it.

    ``gridbook.moe`` imports vLLM at module scope, so a source-level ratchet
    that used ``inspect.getsource`` would silently not run on a vLLM-free host
    — which is most of CI, and exactly where a ratchet has to hold.
    """
    import os

    import gridbook.moe_routing as anchor

    path = os.path.join(os.path.dirname(anchor.__file__), name)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_routing_counts_avoid_the_bincount_host_sync():
    """No serving module may count routed pairs with bincount.

    ATen's CUDA ``bincount`` sizes its output from ``.max().item()`` and so
    host-syncs — the persistent-B lane proved it breaks capture, with a
    negative control. ``cb_grouped_pad_routing``'s docstring claimed "NO HOST
    READS" while calling it anyway, which made the graph-safety story for every
    padded grouped lane false.

    The ratchet scanned ``moe_routing`` ONLY, which is how the 2026-08-02 sweep
    could land while ``moe.py``'s DEFAULT routed prefill — the most-served path
    in the file — still called ``torch.bincount`` on every request. Both
    modules are scanned now.
    """
    import gridbook.moe_routing as mr

    # The CALL, not the word: both modules document at length why they do not
    # use bincount, so the prose legitimately names it.
    for name in ("moe_routing.py", "moe.py"):
        assert "torch.bincount(" not in _gridbook_source(name), (
            f"{name} counts routed pairs with bincount, which host-syncs and "
            f"cannot be captured")
    counts = mr._expert_counts(torch.tensor([0, 2, 2, 5, 5, 5]), 6)
    assert counts.tolist() == [1, 0, 2, 0, 0, 3]
    assert counts.dtype == torch.int64


def test_every_routed_count_goes_through_the_one_shared_helper():
    """One implementation, not three copies that can each drift.

    ``moe.py`` had a hand-written ``scatter_add_`` in the persistent-B lane and
    a ``bincount`` in the default lane; the helper's own docstring records the
    sweep that was supposed to unify them. Spelling the ratchet against the
    HELPER rather than against the absence of one call means a fourth lane
    cannot reintroduce the sync under a different spelling.
    """
    source = _gridbook_source("moe.py")
    assert source.count("_expert_counts(") >= 2
    assert ".scatter_add_(" not in source, (
        "a routed-count scatter_add_ was open-coded again instead of calling "
        "moe_routing._expert_counts")


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
                          ("one_expert", 400, 1),
                          # The RAGGED arm: at M=33/topk=2 an expert holds ~8
                          # rows, so a TileM=256 block is ~97% padding where a
                          # TileM=128 block is ~94%. This is the regime
                          # test_ragged_routing_at_tile_256 was added for, and
                          # the regime its tolerance gate is least able to
                          # speak about — a padded-row indexing fault that
                          # perturbs a handful of the 66 live rows moves a
                          # Frobenius ratio by less than routing luck does, but
                          # breaks equality here outright.
                          ("uniform", 33, 2), ("subset", 17, 2),
                          ("one_expert", 40, 1)])
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


# --------------------------------------------------------------------------- #
# Destination slice (0.8.3): `out` / `n_offset` let two GEMMs write column      #
# halves of one buffer, which is how a per-role gate/up stack avoids a concat.  #
# Only the D leading dimension changes, so a bit difference is a real defect.   #
# --------------------------------------------------------------------------- #

def _stage1_args(method, layer, tokens=96, topk=4, seed=41):
    """The exact stage-1 argument tuple the grouped lane would issue."""
    act = _silu_act()
    ids, weights = _routing(tokens, layer._cb_E, topk, "uniform", seed)
    torch.manual_seed(seed)
    x = torch.randn(tokens, layer._cb_hidden,
                    dtype=torch.bfloat16, device=DEV) * 0.5

    captured = []
    original = method._grouped_call

    def spy(fext, args, tile_m, **kwargs):
        captured.append((args, tile_m))
        return original(fext, args, tile_m, **kwargs)

    method._grouped_call = spy
    try:
        assert method._apply_prefill_grouped_fused_v2(
            layer, x, weights, ids, act, tile_m=128) is not None
    finally:
        method._grouped_call = original
    return captured[0]


def test_destination_slice_is_bit_identical_to_a_fresh_buffer():
    """Writing columns [n, n+N) of a wide buffer must equal writing [0, N)."""
    from gridbook.cuda_ext import get_fused_ext

    method, layer, dims = _build(seed=52, k=28)
    _require_grouped_fused(method, layer)
    fext = get_fused_ext()
    args, tile_m = _stage1_args(method, layer)
    n = int(args[6])

    reference = fext.cb_fused_moe_grouped(*args, tile_m)
    mp = reference.shape[0]

    for offset in (0, n):
        wide = torch.full((mp, 2 * n), float("nan"),
                          dtype=torch.bfloat16, device=DEV)
        returned = fext.cb_fused_moe_grouped(*args, tile_m, wide, offset)
        assert returned.data_ptr() == wide.data_ptr(), (
            "the destination form must return the caller's buffer")
        written = wide[:, offset:offset + n]
        assert torch.equal(written.reshape(-1).view(torch.uint16),
                           reference.reshape(-1).view(torch.uint16)), (
            f"column slice at n_offset={offset} differs from a fresh buffer")
        # The other half must be untouched — a stride bug shows up here first.
        other = wide[:, n - offset:2 * n - offset]
        assert torch.isnan(other.float()).all(), (
            f"writing at n_offset={offset} spilled into the other half")


def test_destination_slice_rejects_a_slice_that_does_not_fit():
    from gridbook.cuda_ext import get_fused_ext

    method, layer, dims = _build(seed=53, k=28)
    _require_grouped_fused(method, layer)
    fext = get_fused_ext()
    args, tile_m = _stage1_args(method, layer)
    n = int(args[6])
    reference = fext.cb_fused_moe_grouped(*args, tile_m)
    mp = reference.shape[0]

    wide = torch.zeros((mp, 2 * n), dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="does not fit"):
        fext.cb_fused_moe_grouped(*args, tile_m, wide, n + 1)
    with pytest.raises(RuntimeError, match="one row per padded A row"):
        fext.cb_fused_moe_grouped(
            *args, tile_m,
            torch.zeros((mp + 128, 2 * n), dtype=torch.bfloat16, device=DEV), 0)
    with pytest.raises(RuntimeError, match="row-major bf16"):
        fext.cb_fused_moe_grouped(
            *args, tile_m,
            torch.zeros((mp, 2 * n), dtype=torch.float32, device=DEV), 0)
