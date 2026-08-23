"""Two-half exactness for CB MoE expert stacks under expert parallelism.

There is one GPU in this box and no second rank, so this file simulates the
EP split IN-PROCESS: build a whole-E layer, gather each rank's experts out of
it exactly as the loaders do, attach that rank's ``expert_map``, and run the
real ``_apply_inline`` on every path a serve would take. What that proves is
the NUMERICS of the split — that a rank's stacks are byte-identical slices,
that zero-weighted remote pairs contribute nothing through the actual kernels,
and that the per-rank partials sum to the whole-layer answer. It does NOT
prove a distributed run: no engine, no collectives, no second device, and the
final all-reduce is simulated by adding the partials here.

Three claims, in the order they build on each other:

1. **The bytes.** A rank's expert stack is ``index_select`` of the whole
   stack, byte for byte — the ``sharded_dims: none`` row of the contract's
   ``expert_parallel`` table. Nothing inside an expert is cut, so per-expert
   numerics cannot drift with the world size.
2. **The inert pair.** A remote pair is aliased to a local expert at weight
   exactly ``0.0``. If that is truly inert, the output cannot depend on WHICH
   local expert it was aliased to — so three different alias targets must give
   bit-identical output through the real kernels. This is the load-bearing
   test: it is what makes the shape-static design safe, and unlike a tolerance
   check it cannot pass by rounding.
3. **The partition.** Every (token, pair) is live on exactly one rank at its
   own router weight, so the per-rank partials sum to the whole-layer output.
   This one crosses a kernel-configuration boundary (a rank's ``E_local``
   changes the grouped tile choice) and adds BF16 partials the way vLLM's
   all-reduce does, so it is checked at tolerance, not bitwise.

Run: ``python -m pytest tests/test_moe_ep_exactness.py -q`` on a CUDA box.
"""
from __future__ import annotations

import types

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")

from gridbook.moe_ep import (  # noqa: E402
    local_expert_gather_index,
    remap_local_expert_ids,
)

DEV = "cuda"

# Tolerance for the cross-rank sum only (claim 3). The suite's grouped-MoE
# comparators use 2.1e-2 relative against an exact-FP32 oracle; this compares
# two runs of the SAME kernels differing only in grouping and final-add order,
# so it is held far tighter. Claims 1 and 2 are bitwise and use no tolerance.
_SUM_REL = 2e-3

_E_GLOBAL = 16
_HIDDEN = 512
_INTER = 768
_TOPK = 4


def _require_stack():
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for expert-parallel forward tests")


# --------------------------------------------------------------------------- #
# Fixtures: one whole-E layer per format, sliced per rank exactly as the       #
# loaders slice it.                                                            #
# --------------------------------------------------------------------------- #

_EXPERT_MAJOR = ("w13_cb_qweight", "w2_cb_qweight",
                 "w13_weight_scale", "w2_weight_scale")


def _build_fp8(*, experts=_E_GLOBAL, hidden=_HIDDEN, inter=_INTER, k=44,
               seed=0):
    """A synthetic FP8-CB expert stack (per-role fp32 scales)."""
    _require_stack()
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    from gridbook.moe import PrismaQuantCBMoEMethod

    n_sub = 4
    type_size = fmt.nvfp4_cb_type_size(k, "fp8")
    codebook = fmt._resolve_codebook(k, "fp8", "product", None,
                                     torch.device(DEV))
    torch.manual_seed(seed)
    w13 = torch.randn(experts, 2 * inter, hidden, device=DEV) * 0.05
    w2 = torch.randn(experts, hidden, inter, device=DEV) * 0.05
    p13, f13 = fmt.nvfp4_cb_pack(w13, k, grid="fp8", mode="product",
                                 codebook=codebook)
    p2, f2 = fmt.nvfp4_cb_pack(w2, k, grid="fp8", mode="product",
                               codebook=codebook)

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.scheme = {"grid": "fp8", "mode": "product", "k": k,
                     "n_sub": n_sub, "type_size": type_size}
    method.prefix = "test.ep_fp8"
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
        w13_weight_scale=f13["scales"].reshape(experts, 2 * inter).to(
            DEV).float(),
        w2_weight_scale=f2["scales"].reshape(experts, hidden).to(DEV).float(),
        _cb_flat=codec.build_flat_codebook([t.to(DEV) for t in codebook]),
        _cb_compose=torch.zeros(1, device=DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer


def _build_fp4(*, experts=_E_GLOBAL, hidden=_HIDDEN, inter=_INTER, k=15,
               seed=0):
    """A synthetic NVFP4-CB v2 expert stack (two-tier scale coding)."""
    _require_stack()
    fmt = pytest.importorskip("prismaquant.nvfp4_cb_formats")
    from gridbook.moe import PrismaQuantCBMoEMethod

    n_sub = 2
    coding = fmt.SCALE_CODING_TWO_TIER
    type_size = fmt.nvfp4_cb_type_size(k, "fp4", coding)
    assert type_size == 4 * k + 9, "two-tier v2 type size moved"
    codebook = fmt._resolve_codebook(k, "fp4", "product", None,
                                     torch.device(DEV))
    torch.manual_seed(seed)
    w13 = torch.randn(experts, 2 * inter, hidden, device=DEV) * 0.05
    w2 = torch.randn(experts, hidden, inter, device=DEV) * 0.05
    p13, _ = fmt.nvfp4_cb_pack(w13, k, grid="fp4", mode="product",
                               codebook=codebook, scale_coding=coding)
    p2, _ = fmt.nvfp4_cb_pack(w2, k, grid="fp4", mode="product",
                              codebook=codebook, scale_coding=coding)

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.scheme = {"grid": "fp4", "mode": "product", "k": k,
                     "n_sub": n_sub, "type_size": type_size}
    method.prefix = "test.ep_fp4"
    method.is_fp4 = True
    method.is_v2 = True
    method.k = k
    method.n_sub = n_sub
    method.type_size = type_size
    method.has_static_fp4_activation = False
    method._sub_table = codec.TWO_TIER_SUB_TABLE

    cb_flat = codec.build_flat_codebook([t.to(DEV) for t in codebook])
    layer = types.SimpleNamespace(
        _cb_E=experts,
        _cb_hidden=hidden,
        _cb_inter=inter,
        w13_cb_qweight=p13.reshape(experts, 2 * inter, -1).contiguous(),
        w2_cb_qweight=p2.reshape(experts, hidden, -1).contiguous(),
        _cb_flat=cb_flat,
        _cb_compose=codec.build_compose_u8(codec.TWO_TIER_SUB_TABLE).to(DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer


BUILDERS = {"fp8": _build_fp8, "fp4": _build_fp4}


def _linear_map(e_global, ep_size, rank):
    per = e_global // ep_size
    em = torch.full((e_global,), -1, dtype=torch.int32)
    em[rank * per:(rank + 1) * per] = torch.arange(per, dtype=torch.int32)
    return em


def _round_robin_map(e_global, ep_size, rank):
    em = torch.full((e_global,), -1, dtype=torch.int32)
    owned = torch.arange(rank, e_global, ep_size)
    em[owned] = torch.arange(owned.numel(), dtype=torch.int32)
    return em


MAPS = {"contiguous": _linear_map, "interleaved": _round_robin_map}


def _rank_layer(layer, em, e_local):
    """The layer this rank would have built, gathered exactly as the loaders do.

    ``moe_ep.gather_expert_major`` is the shipping rule; this restates
    ``index_select`` on the same index so the test proves byte identity of the
    result rather than trusting the helper it is testing.
    """
    idx = local_expert_gather_index(em, e_local, surface="test", prefix="test")
    fields = dict(vars(layer))
    for name in _EXPERT_MAJOR:
        tensor = fields.get(name)
        if tensor is not None:
            fields[name] = tensor.index_select(
                0, idx.to(tensor.device)).contiguous()
    fields["_cb_E"] = e_local
    fields["_cb_ep_map"] = em.to(DEV)
    return types.SimpleNamespace(**fields), idx


def _routing(tokens, e_global, topk, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([
        torch.randperm(e_global, generator=g)[:topk] for _ in range(tokens)
    ]).to(torch.int32).to(DEV)
    w = torch.rand(tokens, topk, generator=g, dtype=torch.float32).to(DEV)
    return ids, w / w.sum(dim=-1, keepdim=True)


def _bits(t):
    return t.view(torch.uint16) if t.dtype is torch.bfloat16 else t


def _x(tokens, hidden, seed):
    torch.manual_seed(seed)
    return torch.randn(tokens, hidden, dtype=torch.bfloat16, device=DEV)


# --------------------------------------------------------------------------- #
# 1. The bytes: a rank's stack is a slice, not a re-encode.                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", list(BUILDERS))
@pytest.mark.parametrize("placement", list(MAPS))
def test_rank_stacks_are_byte_identical_slices_of_the_whole_stack(
        fmt, placement):
    method, full = BUILDERS[fmt]()
    del method
    ep = 4
    for rank in range(ep):
        em = MAPS[placement](_E_GLOBAL, ep, rank)
        sub, idx = _rank_layer(full, em, _E_GLOBAL // ep)
        for name in _EXPERT_MAJOR:
            whole = getattr(full, name, None)
            if whole is None:
                continue
            part = getattr(sub, name)
            assert part.shape[0] == _E_GLOBAL // ep
            for slot, gid in enumerate(idx.tolist()):
                assert torch.equal(part[slot], whole[gid]), (
                    f"{fmt}/{placement}/rank{rank}: {name} slot {slot} is not "
                    f"byte-identical to whole-stack expert {gid}")


# --------------------------------------------------------------------------- #
# 2. The inert pair: output must not depend on the alias target.                #
# --------------------------------------------------------------------------- #


def _alias_to(em, ids, w, e_local, rule):
    """Alias remote pairs by an alternative rule, at weight exactly 0.0."""
    local = em.index_select(0, ids.reshape(-1).long()).view(ids.shape).long()
    is_local = local >= 0
    if rule == "zero":
        target = torch.zeros_like(local)
    elif rule == "last":
        target = torch.full_like(local, e_local - 1)
    elif rule == "largest_local":
        sentinel = torch.full_like(local, -1)
        target = torch.where(is_local, local, sentinel).amax(
            dim=-1, keepdim=True).expand_as(local)
        target = torch.where(target < 0, torch.zeros_like(target), target)
    else:                                       # pragma: no cover
        raise AssertionError(rule)
    out = torch.where(is_local, local, target)
    return out.to(ids.dtype), torch.where(is_local, w, torch.zeros_like(w))


@pytest.mark.parametrize("fmt", list(BUILDERS))
@pytest.mark.parametrize("placement", list(MAPS))
@pytest.mark.parametrize("tokens", [1, 4, 16, 17, 64, 129])
def test_the_alias_target_cannot_change_the_output(fmt, placement, tokens):
    """The load-bearing bitwise claim, through the real kernels.

    Decode (T <= 16) and every prefill lane above it are covered by the token
    sweep: 1/4/16 take the grouped GEMV, 17/64/129 take the prefill path for
    this format, and 129 crosses the padded-route tile boundary.
    """
    method, full = BUILDERS[fmt]()
    ep, e_local = 4, _E_GLOBAL // 4
    em = MAPS[placement](_E_GLOBAL, ep, 1)
    sub, _ = _rank_layer(full, em, e_local)
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=tokens)
    x = _x(tokens, _HIDDEN, seed=tokens)

    # The shipping rule, applied inside _apply_inline by the layer's map.
    shipped = method._apply_inline(sub, x, w, ids)

    # Three alternative inert paddings, applied by hand on a layer with NO
    # map so _apply_inline consumes the local ids as given.
    plain = types.SimpleNamespace(**{k: v for k, v in vars(sub).items()
                                     if k != "_cb_ep_map"})
    emd = em.to(DEV)
    for rule in ("zero", "last", "largest_local"):
        alt_ids, alt_w = _alias_to(emd, ids, w, e_local, rule)
        got = method._apply_inline(plain, x, alt_w, alt_ids)
        assert torch.equal(_bits(shipped), _bits(got)), (
            f"{fmt}/{placement}/T={tokens}: output changed when the "
            f"zero-weight alias target changed to {rule!r} — the remote pair "
            "is NOT inert")


@pytest.mark.parametrize("fmt", list(BUILDERS))
def test_a_token_with_no_local_pair_contributes_exactly_zero(fmt):
    """Its whole row is zero-weighted; the other ranks supply its output."""
    method, full = BUILDERS[fmt]()
    e_local = 4
    em = torch.full((_E_GLOBAL,), -1, dtype=torch.int32)
    em[:e_local] = torch.arange(e_local, dtype=torch.int32)
    sub, _ = _rank_layer(full, em, e_local)

    # Token 0 routes entirely off-rank; token 1 routes entirely on-rank.
    ids = torch.tensor([[8, 9, 10, 11], [0, 1, 2, 3]],
                       dtype=torch.int32, device=DEV)
    w = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
                     device=DEV)
    x = _x(2, _HIDDEN, seed=7)
    out = method._apply_inline(sub, x, w, ids)
    assert torch.equal(out[0], torch.zeros_like(out[0])), (
        "a token with no local expert must contribute an exact zero")
    assert not torch.equal(out[1], torch.zeros_like(out[1]))


@pytest.mark.parametrize("fmt", list(BUILDERS))
@pytest.mark.parametrize("tokens", [4, 64])
def test_an_all_local_rank_is_bit_identical_to_no_expert_parallelism(
        fmt, tokens):
    """Regression guard: the remap must be a no-op when nothing is remote."""
    method, full = BUILDERS[fmt]()
    identity = torch.arange(_E_GLOBAL, dtype=torch.int32)
    with_map = types.SimpleNamespace(**vars(full))
    with_map._cb_ep_map = identity.to(DEV)
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=11)
    x = _x(tokens, _HIDDEN, seed=11)
    assert torch.equal(_bits(method._apply_inline(full, x, w, ids)),
                       _bits(method._apply_inline(with_map, x, w, ids)))


# --------------------------------------------------------------------------- #
# 3. The partition: per-rank partials sum to the whole-layer output.            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", list(BUILDERS))
@pytest.mark.parametrize("placement", list(MAPS))
@pytest.mark.parametrize("tokens", [4, 64])
@pytest.mark.parametrize("ep", [2, 4])
def test_rank_partials_sum_to_the_whole_layer_output(fmt, placement, tokens,
                                                     ep):
    """Simulates vLLM's final all-reduce by adding the partials here.

    Checked at tolerance, not bitwise, and deliberately so: a rank's E_local
    changes the grouped tile selection (``moe_routing.cb_grouped_tile_m``
    picks on pairs-per-expert), and the partials are added in BF16 exactly as
    ``tensor_model_parallel_all_reduce`` would.
    """
    method, full = BUILDERS[fmt]()
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=ep * 100 + tokens)
    x = _x(tokens, _HIDDEN, seed=ep * 100 + tokens)
    whole = method._apply_inline(full, x, w, ids).float()

    total = torch.zeros_like(whole)
    for rank in range(ep):
        em = MAPS[placement](_E_GLOBAL, ep, rank)
        sub, _ = _rank_layer(full, em, _E_GLOBAL // ep)
        total += method._apply_inline(sub, x, w, ids).float()

    denom = whole.abs().max().clamp_min(1e-6)
    rel = (total - whole).abs().max() / denom
    assert rel < _SUM_REL, (
        f"{fmt}/{placement}/EP={ep}/T={tokens}: rank partials sum to a "
        f"different answer (max rel {float(rel):.3e})")


@pytest.mark.parametrize("placement", list(MAPS))
def test_every_pair_is_live_on_exactly_one_rank(placement):
    """The routing-level statement behind the sum, checked without kernels."""
    _require_stack()
    ids, w = _routing(64, _E_GLOBAL, _TOPK, seed=2)
    live = torch.zeros_like(w)
    for rank in range(4):
        em = MAPS[placement](_E_GLOBAL, 4, rank).to(DEV)
        _, wl = remap_local_expert_ids(em, ids, w, _E_GLOBAL // 4)
        live += wl
    assert torch.equal(live, w)


# --------------------------------------------------------------------------- #
# 4. Capture: the remap survives a real CUDA graph.                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", list(BUILDERS))
def test_the_remapped_decode_captures_and_replays(fmt):
    """Decode is the captured regime; a host read here would fail capture.

    Replay with new routing values in the same static buffers must also
    produce the same answer as an eager call, which is what proves the graph
    baked no data-dependent decision from the capture-time routing.
    """
    method, full = BUILDERS[fmt]()
    e_local = _E_GLOBAL // 2
    em = _round_robin_map(_E_GLOBAL, 2, 0)
    sub, _ = _rank_layer(full, em, e_local)

    tokens = 8
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=21)
    x = _x(tokens, _HIDDEN, seed=21)
    static_x = x.clone()
    static_ids = ids.clone()
    static_w = w.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            method._apply_inline(sub, static_x, static_w, static_ids)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = method._apply_inline(sub, static_x, static_w, static_ids)

    ids2, w2 = _routing(tokens, _E_GLOBAL, _TOPK, seed=22)
    x2 = _x(tokens, _HIDDEN, seed=22)
    static_x.copy_(x2)
    static_ids.copy_(ids2)
    static_w.copy_(w2)
    graph.replay()
    torch.cuda.synchronize()
    replayed = static_out.clone()

    eager = method._apply_inline(sub, x2, w2, ids2)
    assert torch.equal(_bits(replayed), _bits(eager)), (
        f"{fmt}: replayed graph disagrees with eager on new routing — the "
        "capture baked a decision the remap should have kept dynamic")
