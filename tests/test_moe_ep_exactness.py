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
   all-reduce does, so it is checked against a bound derived from BF16
   rounding rather than bitwise.

Two measured properties of the SHIPPING kernels shape how those claims are
asserted here. Both were established by running the same call repeatedly (see
``test_decode_repeats_bitwise_and_prefill_does_not``), not assumed:

* **Decode (``T <= MOE_PREFILL_M_THRESHOLD``) is run-to-run bitwise
  repeatable.** Its router combine is the native ``cb_moe_combine``. So every
  bitwise claim at decode is asserted directly against the shipping path.
* **Prefill is NOT.** Prefill lanes combine pairs with ``Tensor.index_add_``
  (``moe.py`` :1380, :1602, :1735, :1872, :2298), which ATen documents as
  nondeterministic on CUDA: it accumulates with atomics, so the summation
  order changes between two identical calls. Measured here at 260/8704
  (T=17) to 2920/32768 (T=64) elements differing between two runs of one
  unchanged call. That is a pre-existing property of gridbook's prefill
  combine and has nothing to do with expert parallelism — but it means a
  bitwise A/B at prefill cannot pass, whatever the code does. So the prefill
  arms of claim 2 run under ``torch.use_deterministic_algorithms``, which
  swaps ATen's ``index_add_`` for its deterministic implementation and leaves
  every gridbook kernel and the remap itself untouched. Verified to restore
  bit-repeatability on all six prefill (format, T) cells.

Run: ``python -m pytest tests/test_moe_ep_exactness.py -q`` on a CUDA box.
"""
from __future__ import annotations

import contextlib
import types

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")

from gridbook.moe import MOE_PREFILL_M_THRESHOLD, _CB_ROLES  # noqa: E402
from gridbook.moe_ep import (  # noqa: E402
    local_expert_gather_index,
    remap_local_expert_ids,
)

DEV = "cuda"

# BF16 has an 8-bit significand, so consecutive representable values differ by
# a relative 2**-8. Every tolerance in this file is built from that constant
# and a measured magnitude; none is picked.
_BF16_EPS = 2.0 ** -8

_E_GLOBAL = 16
_HIDDEN = 512
_INTER = 768
_TOPK = 4

_DECODE_TOKENS = (1, 4, 16)
_PREFILL_TOKENS = (17, 64, 129)


def _require_stack():
    pytest.importorskip("vllm")
    pytest.importorskip("vllm.model_executor.layers.fused_moe.config")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for expert-parallel forward tests")


@contextlib.contextmanager
def _ordered_reduction():
    """ATen's deterministic ``index_add_``, for the prefill lanes only.

    This changes the ATen reduction gridbook's prefill lanes call; it does not
    change any gridbook kernel, the packed bytes, or the remap under test. It
    is used only where the shipping combine's atomics would otherwise mask the
    signal — see the module docstring.
    """
    was = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn)


def _reduction_for(tokens):
    """Decode is repeatable as shipped; prefill needs the ordered reduction."""
    if tokens <= MOE_PREFILL_M_THRESHOLD:
        return contextlib.nullcontext()
    return _ordered_reduction()


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
        # fp8-CB is v1: no two-tier compose section, and `process_weights_-
        # after_loading` parks a 1-element dummy here (moe.py:685).
        _cb_compose=torch.zeros(1, dtype=torch.float32, device=DEV),
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
        # The v2 decode GEMV composes the two-tier weight scale in-register
        # from the packed 9-byte section and THIS table, and validates it as
        # `(256*16,)` fp32 contiguous. `build_compose_u8` is the E4M3-byte
        # table a different lane wants (moe.py:1331) — not this one.
        _cb_compose=codec.build_compose_table(
            codec.TWO_TIER_SUB_TABLE).to(DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer


def _split_by_role(method, layer):
    """Turn a fused FP8-CB layer into the per-role form, as production does.

    DSv4's shipped artifact splits 11 of its 43 routed-expert layers into
    per-role ``gate_proj``/``up_proj`` stacks (32 stay fused), and those take a
    different decode branch: ``_apply_grouped_decode`` reads
    ``_cb_w13_{role}_qweight`` and ``_role_flat_fp8`` instead of
    ``w13_cb_qweight``/``_cb_flat_fp8`` (moe.py:2520). This calls the shipping
    ``_split_w13_by_role`` so the halves are produced by the code under test
    rather than by a restatement of it.

    All three roles share one book here: this fixture exercises the per-role
    DISPATCH, which is what expert parallelism has to keep intact. Per-role
    codebook SELECTION is orthogonal to the expert split.

    ``_cb_flat`` is deleted because production leaves it unset in per-role
    mode on purpose (moe.py:665-671) — its absence is what makes a lane that
    was never taught the split fail loudly instead of decoding with another
    role's book. A fixture that kept it would let an untaught lane pass here
    and fail in production.
    """
    flat = layer._cb_flat
    layer._cb_role_split = True
    layer._cb_flat_by_role = {role: flat for role in _CB_ROLES}
    del layer._cb_flat
    method._split_w13_by_role(layer)
    for role in _CB_ROLES:
        method._role_flat_fp8(layer, role)
    return layer


# name -> (whole-stack builder, post-gather step or None). The post step runs
# AFTER the per-rank gather, exactly as production orders it: the loaders
# gather at weight load, `process_weights_after_loading` splits afterwards.
# The order is load-bearing — `_split_w13_by_role` RELEASES the fused stack
# (`w13_cb_qweight.data = new_empty((0,))`, moe.py:2419), so a gather that ran
# after it would find an empty tensor.
BUILDERS = {
    "fp8": (_build_fp8, None),
    "fp4": (_build_fp4, None),
    "fp8_role_split": (_build_fp8, _split_by_role),
}


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


def _identity_map(e_global):
    return torch.arange(e_global, dtype=torch.int32)


MAPS = {"contiguous": _linear_map, "interleaved": _round_robin_map}


def _gather(layer, em, e_local):
    """The rank's expert-major tensors, gathered exactly as the loaders do.

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
    return types.SimpleNamespace(**fields), idx


def _materialize(fmt, method, raw, em, e_local, *, with_map):
    """One rank's live layer: gather, then the format's post-gather step."""
    layer, idx = _gather(raw, em, e_local)
    if with_map:
        layer._cb_ep_map = em.to(DEV)
    post = BUILDERS[fmt][1]
    if post is not None:
        post(method, layer)
    return layer, idx


def _whole(fmt):
    """The no-expert-parallelism reference, built through the same machinery.

    Gathered with an identity map so its tensors are fresh copies: the
    per-role step mutates ``.data`` in place, and the reference must not share
    storage with the rank layers it is compared against.
    """
    method, raw = BUILDERS[fmt][0]()
    layer, _ = _materialize(fmt, method, raw, _identity_map(_E_GLOBAL),
                            _E_GLOBAL, with_map=False)
    return method, raw, layer


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
# 0. The two kernel properties every assertion below is calibrated against.     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", list(BUILDERS))
def test_decode_repeats_bitwise_and_prefill_does_not(fmt):
    """Measures what the rest of the file assumes, so it cannot rot silently.

    Decode's combine is the native ``cb_moe_combine``; prefill's is ATen
    ``index_add_``, which accumulates with atomics on CUDA. If a future change
    makes prefill repeatable as shipped, this test fails and the prefill arms
    below can drop ``_ordered_reduction``.
    """
    method, _, whole = _whole(fmt)
    for tokens in _DECODE_TOKENS:
        ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=tokens)
        x = _x(tokens, _HIDDEN, seed=tokens)
        first = method._apply_inline(whole, x, w, ids)
        again = method._apply_inline(whole, x, w, ids)
        assert torch.equal(_bits(first), _bits(again)), (
            f"{fmt}/T={tokens}: decode is no longer run-to-run bitwise "
            "repeatable — the captured regime must be")

    unrepeatable = 0
    for tokens in _PREFILL_TOKENS:
        ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=tokens)
        x = _x(tokens, _HIDDEN, seed=tokens)
        first = method._apply_inline(whole, x, w, ids)
        again = method._apply_inline(whole, x, w, ids)
        unrepeatable += int(not torch.equal(_bits(first), _bits(again)))
        with _ordered_reduction():
            a = method._apply_inline(whole, x, w, ids)
            b = method._apply_inline(whole, x, w, ids)
        assert torch.equal(_bits(a), _bits(b)), (
            f"{fmt}/T={tokens}: the ordered reduction did not make prefill "
            "repeatable, so the prefill bitwise arms below prove nothing")
    assert unrepeatable > 0, (
        f"{fmt}: prefill is now bitwise repeatable as shipped — good news, "
        "and the prefill arms no longer need _ordered_reduction")


# --------------------------------------------------------------------------- #
# 1. The bytes: a rank's stack is a slice, not a re-encode.                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", list(BUILDERS))
@pytest.mark.parametrize("placement", list(MAPS))
def test_rank_stacks_are_byte_identical_slices_of_the_whole_stack(
        fmt, placement):
    method, raw, _ = _whole(fmt)
    ep = 4
    for rank in range(ep):
        em = MAPS[placement](_E_GLOBAL, ep, rank)
        sub, idx = _gather(raw, em, _E_GLOBAL // ep)
        for name in _EXPERT_MAJOR:
            whole = getattr(raw, name, None)
            if whole is None:
                continue
            part = getattr(sub, name)
            assert part.shape[0] == _E_GLOBAL // ep
            for slot, gid in enumerate(idx.tolist()):
                assert torch.equal(part[slot], whole[gid]), (
                    f"{fmt}/{placement}/rank{rank}: {name} slot {slot} is not "
                    f"byte-identical to whole-stack expert {gid}")


@pytest.mark.parametrize("placement", list(MAPS))
def test_the_per_role_halves_are_slices_of_the_gathered_fused_stack(placement):
    """The role split must cut columns, and the EP gather must cut experts.

    Those are different axes, and doing them in the production order (gather
    then split) has to give the same halves as splitting the whole stack and
    then gathering. If it did not, one of the two cuts would be reading the
    other's stride.
    """
    method, raw, whole_split = _whole("fp8_role_split")
    ep, e_local = 4, _E_GLOBAL // 4
    for rank in range(ep):
        em = MAPS[placement](_E_GLOBAL, ep, rank)
        sub, idx = _materialize("fp8_role_split", method, raw, em, e_local,
                                with_map=True)
        for role in ("gate", "up"):
            part = getattr(sub, f"_cb_w13_{role}_qweight")
            ref = getattr(whole_split, f"_cb_w13_{role}_qweight")
            for slot, gid in enumerate(idx.tolist()):
                assert torch.equal(part[slot], ref[gid]), (
                    f"{placement}/rank{rank}: per-role {role} slot {slot} is "
                    f"not the whole stack's expert {gid}")
        assert sub.w13_cb_qweight.numel() == 0, (
            "the fused stack must be released once the halves exist")


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
@pytest.mark.parametrize("tokens", _DECODE_TOKENS + _PREFILL_TOKENS)
def test_the_alias_target_cannot_change_the_output(fmt, placement, tokens):
    """The load-bearing bitwise claim, through the real kernels.

    Decode (T <= 16) and every prefill lane above it are covered by the token
    sweep: 1/4/16 take the grouped GEMV, 17/64/129 take the prefill path for
    this format, and 129 crosses the padded-route tile boundary. The prefill
    arms run under the ordered reduction for the reason in the module
    docstring; the decode arms run exactly as shipped.
    """
    method, raw, _ = _whole(fmt)
    ep, e_local = 4, _E_GLOBAL // 4
    em = MAPS[placement](_E_GLOBAL, ep, 1)
    sub, _ = _materialize(fmt, method, raw, em, e_local, with_map=True)
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=tokens)
    x = _x(tokens, _HIDDEN, seed=tokens)

    # A layer with NO map, so _apply_inline consumes the local ids as given.
    plain = types.SimpleNamespace(**{k: v for k, v in vars(sub).items()
                                     if k != "_cb_ep_map"})
    emd = em.to(DEV)
    with _reduction_for(tokens):
        # The shipping rule, applied inside _apply_inline by the layer's map.
        shipped = method._apply_inline(sub, x, w, ids)
        for rule in ("zero", "last", "largest_local"):
            alt_ids, alt_w = _alias_to(emd, ids, w, e_local, rule)
            got = method._apply_inline(plain, x, alt_w, alt_ids)
            assert torch.equal(_bits(shipped), _bits(got)), (
                f"{fmt}/{placement}/T={tokens}: output changed when the "
                f"zero-weight alias target changed to {rule!r} — the remote "
                "pair is NOT inert")


@pytest.mark.parametrize("fmt", list(BUILDERS))
def test_a_token_with_no_local_pair_contributes_exactly_zero(fmt):
    """Its whole row is zero-weighted; the other ranks supply its output."""
    method, raw, _ = _whole(fmt)
    e_local = 4
    em = torch.full((_E_GLOBAL,), -1, dtype=torch.int32)
    em[:e_local] = torch.arange(e_local, dtype=torch.int32)
    sub, _ = _materialize(fmt, method, raw, em, e_local, with_map=True)

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
    method, _, whole = _whole(fmt)
    with_map = types.SimpleNamespace(**vars(whole))
    with_map._cb_ep_map = _identity_map(_E_GLOBAL).to(DEV)
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=11)
    x = _x(tokens, _HIDDEN, seed=11)
    with _reduction_for(tokens):
        assert torch.equal(_bits(method._apply_inline(whole, x, w, ids)),
                           _bits(method._apply_inline(with_map, x, w, ids)))


# --------------------------------------------------------------------------- #
# 3. The partition: per-rank partials sum to the whole-layer output.            #
# --------------------------------------------------------------------------- #


def _pair_magnitude(method, layer, x, w, ids):
    """Sum over pairs of |that pair's contribution|, measured not modelled.

    Running with one pair's router weight live and the rest at exactly 0.0
    isolates ``fl(w_j * y_j)``: the pair set, the grouping and the per-row
    activation quantization are all unchanged, and the other terms add an
    exact zero. This is the magnitude the BF16 accumulation error is
    proportional to, and it is the only quantity the bound below needs.
    """
    total = None
    for j in range(w.shape[1]):
        one = torch.zeros_like(w)
        one[:, j] = w[:, j]
        contribution = method._apply_inline(layer, x, one, ids).float().abs()
        total = contribution if total is None else total + contribution
    return total


@pytest.mark.parametrize("fmt", list(BUILDERS))
@pytest.mark.parametrize("placement", list(MAPS))
@pytest.mark.parametrize("tokens", [4, 16, 64])
@pytest.mark.parametrize("ep", [2, 4])
def test_rank_partials_sum_to_the_whole_layer_output(fmt, placement, tokens,
                                                     ep):
    """Simulates vLLM's final all-reduce by adding the partials here.

    Not bitwise, and the bound is derived rather than picked. Both sides sum
    the same TOPK per-token contributions; they differ only in the order and
    in how many intermediate BF16 roundings each takes. Rounding a BF16
    accumulator once costs at most half a step, i.e. ``2**-9`` relative, so
    accumulating K terms costs at most ``K * 2**-9 * sum|contribution|`` on
    each side, and the two sides together at most ``K * 2**-8 * S`` where S is
    the measured per-pair magnitude. Nothing here assumes an accumulation
    width or a kernel schedule.

    A rank's ``E_local`` also changes the grouped tile selection
    (``moe_routing.cb_grouped_tile_m`` picks on pairs-per-expert), so this is
    the one claim that crosses a kernel-configuration boundary.
    """
    method, raw, whole_layer = _whole(fmt)
    ids, w = _routing(tokens, _E_GLOBAL, _TOPK, seed=ep * 100 + tokens)
    x = _x(tokens, _HIDDEN, seed=ep * 100 + tokens)

    with _reduction_for(tokens):
        whole = method._apply_inline(whole_layer, x, w, ids).float()
        magnitude = _pair_magnitude(method, whole_layer, x, w, ids)
        total = torch.zeros_like(whole)
        for rank in range(ep):
            em = MAPS[placement](_E_GLOBAL, ep, rank)
            sub, _ = _materialize(fmt, method, raw, em, _E_GLOBAL // ep,
                                  with_map=True)
            total += method._apply_inline(sub, x, w, ids).float()

    bound = _TOPK * _BF16_EPS * magnitude
    excess = ((total - whole).abs() - bound).max()
    assert excess <= 0, (
        f"{fmt}/{placement}/EP={ep}/T={tokens}: rank partials sum to an "
        f"answer that BF16 accumulation cannot explain (worst element exceeds "
        f"its bound by {float(excess):.3e}; max |diff| "
        f"{float((total - whole).abs().max()):.3e})")


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
    method, raw, _ = _whole(fmt)
    e_local = _E_GLOBAL // 2
    em = _round_robin_map(_E_GLOBAL, 2, 0)
    sub, _ = _materialize(fmt, method, raw, em, e_local, with_map=True)

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
