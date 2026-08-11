"""Per-role codebooks for routed MoE stacks (0.8.3).

A learned codebook is fit per (layer, projection), so gate, up and down want
three different books at the same rung. Through 0.8.2 the runtime bound ONE
book per layer and the resolver compared a format signature that deliberately
excluded ``codebook_ref`` — so a per-role artifact would have loaded without
complaint and decoded every stack with whichever role sorted first
(``down_proj``). These tests pin the refusal, the resolution, and the property
that matters most: a uniform artifact must resolve exactly as it did before.
"""

from __future__ import annotations

import pytest
import torch

from gridbook.config import PrismaQuantConfig


# fp8-CB v1 product layout — the only shape 0.8.3 splits per role.
_SCHEME = {"grid": "fp8", "mode": "product", "k": 28, "n_sub": 4,
           "type_size": 112, "group_size": 0, "vec_dim": 8,
           "codebook_group": "routed_experts", "codebook_source": "learned"}

_PREFIX = "model.layers.1.mlp.experts"


def _scheme(*refs):
    return {**_SCHEME, "codebook_ref": list(refs)}


def _config(groups):
    """One config_group per distinct book, as the exporter emits them."""
    return {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            f"group_{index}": {"format": "FP8_CB_K28",
                               "targets": list(targets),
                               "scheme": scheme}
            for index, (targets, scheme) in enumerate(groups)
        },
        "ignore": ["lm_head"],
    }


def _resolved(groups):
    cfg = PrismaQuantConfig.from_config(_config(groups))
    cfg._ensure_resolved()
    return cfg


def _leaf(name):
    return f"{_PREFIX}.{name}"


# ---------------------------------------------------------------------------
# A. The uniform artifact must not notice any of this
# ---------------------------------------------------------------------------

def test_uniform_stack_resolves_to_the_untouched_scheme_object():
    """Every artifact shipped to date names one book for the whole stack.

    The assertion is identity, not equality: the resolver must hand back the
    very dict 0.8.2 handed back, so no downstream consumer can observe a new
    key, a dropped ``codebook_ref``, or a re-ordered mapping.
    """
    shared = _scheme("cb.routed.K28.sub0", "cb.routed.K28.sub1")
    cfg = _resolved([([_leaf("gate_up_proj"), _leaf("down_proj")], shared)])

    resolved = cfg._moe_scheme_for_prefix(_PREFIX)
    assert resolved is cfg.target_scheme[_leaf("down_proj")]
    assert "codebook_ref_by_role" not in resolved
    assert resolved["codebook_ref"] == ["cb.routed.K28.sub0",
                                        "cb.routed.K28.sub1"]


def test_uniform_stack_written_as_three_unfused_targets_is_still_uniform():
    """Unfused role spellings with ONE book are a uniform stack, not a split."""
    shared = _scheme("cb.routed.K28.sub0")
    cfg = _resolved([([_leaf("gate_proj"), _leaf("up_proj"),
                      _leaf("down_proj")], shared)])

    resolved = cfg._moe_scheme_for_prefix(_PREFIX)
    assert "codebook_ref_by_role" not in resolved
    assert resolved["codebook_ref"] == ["cb.routed.K28.sub0"]


# ---------------------------------------------------------------------------
# B. Per-role resolution
# ---------------------------------------------------------------------------

def test_three_books_resolve_to_three_roles():
    cfg = _resolved([
        ([_leaf("gate_proj")], _scheme("cb.l1.gate.K28.sub0")),
        ([_leaf("up_proj")], _scheme("cb.l1.up.K28.sub0")),
        ([_leaf("down_proj")], _scheme("cb.l1.down.K28.sub0")),
    ])

    resolved = cfg._moe_scheme_for_prefix(_PREFIX)
    assert resolved["codebook_ref_by_role"] == {
        "gate": ["cb.l1.gate.K28.sub0"],
        "up": ["cb.l1.up.K28.sub0"],
        "down": ["cb.l1.down.K28.sub0"],
    }
    # The singular key is DROPPED, not carried through: it would otherwise
    # hold down_proj's book and let an untaught consumer decode all three
    # roles with it — the exact fail-open this resolver closes.
    assert "codebook_ref" not in resolved


def test_fused_w13_target_claims_both_halves_with_one_book():
    """A `gate_up_proj` target speaks for gate AND up; only `down` differs."""
    cfg = _resolved([
        ([_leaf("gate_up_proj")], _scheme("cb.l1.w13.K28.sub0")),
        ([_leaf("down_proj")], _scheme("cb.l1.down.K28.sub0")),
    ])

    roles = cfg._moe_scheme_for_prefix(_PREFIX)["codebook_ref_by_role"]
    assert roles["gate"] == roles["up"] == ["cb.l1.w13.K28.sub0"]
    assert roles["down"] == ["cb.l1.down.K28.sub0"]


# ---------------------------------------------------------------------------
# C. Fail closed
# ---------------------------------------------------------------------------

def test_a_role_claimed_twice_with_different_books_is_refused():
    """`gate_up_proj` and `gate_proj` both claim `gate`. One role, one book."""
    cfg = _resolved([
        ([_leaf("gate_up_proj")], _scheme("cb.l1.w13.K28.sub0")),
        ([_leaf("gate_proj")], _scheme("cb.l1.gate.K28.sub0")),
        ([_leaf("down_proj")], _scheme("cb.l1.down.K28.sub0")),
    ])

    with pytest.raises(ValueError, match="claim the 'gate' codebook role"):
        cfg._moe_scheme_for_prefix(_PREFIX)


def test_a_missing_role_is_refused_rather_than_borrowed():
    """No `up` book. Decoding those rows with gate's book would be silent."""
    cfg = _resolved([
        ([_leaf("gate_proj")], _scheme("cb.l1.gate.K28.sub0")),
        ([_leaf("down_proj")], _scheme("cb.l1.down.K28.sub0")),
    ])

    with pytest.raises(ValueError, match=r"names no book for \['up'\]"):
        cfg._moe_scheme_for_prefix(_PREFIX)


def test_mixed_decode_contract_still_raises_before_the_role_check():
    """The pre-existing signature guard keeps its wording and its precedence."""
    cfg = _resolved([
        ([_leaf("gate_up_proj")], _scheme("cb.l1.w13.K28.sub0")),
        ([_leaf("down_proj")], {**_scheme("cb.l1.down.K30.sub0"), "k": 30}),
    ])

    with pytest.raises(ValueError, match="mixed CB decode/activation"):
        cfg._moe_scheme_for_prefix(_PREFIX)


@pytest.mark.parametrize("leaf", ["gate_proj", "gate_up_proj", "down_proj"])
def test_role_crossed_with_a_per_expert_format_group_is_refused(leaf):
    """Both features exist; composing them is an explicit 0.8.3 non-feature.

    It has to be refused BY NAME: `_moe_target_keys` matches on the final
    component, so a `…gate_proj.format_group_0` target would otherwise be
    skipped in silence and the stack resolved from whatever else matched.
    """
    cfg = _resolved([
        ([_leaf("gate_up_proj"), _leaf("down_proj")],
         _scheme("cb.routed.K28.sub0")),
        ([f"{_leaf(leaf)}.format_group_0"], _scheme("cb.l1.grp0.K28.sub0")),
    ])

    with pytest.raises(ValueError, match="per-expert format group"):
        cfg._moe_scheme_for_prefix(_PREFIX)


def test_a_neighbouring_stacks_format_group_does_not_trip_the_guard():
    """The refusal is scoped to THIS prefix, like every other lookup here."""
    cfg = _resolved([
        ([_leaf("gate_up_proj"), _leaf("down_proj")],
         _scheme("cb.routed.K28.sub0")),
        (["model.layers.2.mlp.experts.gate_proj.format_group_0"],
         _scheme("cb.l2.grp0.K28.sub0")),
    ])

    assert cfg._moe_scheme_for_prefix(_PREFIX) is not None


# ---------------------------------------------------------------------------
# D. The load-time split (CPU tensors; no kernels involved)
# ---------------------------------------------------------------------------

E, INTER, HIDDEN, ROW_BYTES = 3, 8, 16, 32


def _method(**attrs):
    """A method object with no ``__init__`` side effects; stays CPU-only."""
    from gridbook.moe import PrismaQuantCBMoEMethod

    method = object.__new__(PrismaQuantCBMoEMethod)
    method.prefix = _PREFIX
    for name, value in attrs.items():
        setattr(method, name, value)
    return method


def _layer():
    layer = torch.nn.Module()
    layer._cb_inter, layer._cb_hidden = INTER, HIDDEN
    # Distinct per (expert, row, byte) so a mis-sliced copy cannot pass.
    layer.w13_cb_qweight = torch.nn.Parameter(
        torch.arange(E * 2 * INTER * ROW_BYTES, dtype=torch.uint8)
        .remainder(251).reshape(E, 2 * INTER, ROW_BYTES),
        requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.arange(E * 2 * INTER, dtype=torch.float32).reshape(E, 2 * INTER),
        requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.arange(E * HIDDEN, dtype=torch.float32).reshape(E, HIDDEN),
        requires_grad=False)
    return layer


def test_split_preserves_the_gate_then_up_row_contract():
    """gate is rows [0, inter), up is [inter, 2*inter).

    That is the layout `native_moe_activation` assumes when it reads the fused
    buffer as silu(first half) * (second half); if the split disagreed, the
    two halves would swap and the model would still produce plausible output.
    """
    layer, method = _layer(), _method()
    fused = layer.w13_cb_qweight.data.clone()
    fused_scale = layer.w13_weight_scale.data.clone()

    method._split_w13_by_role(layer)

    assert torch.equal(layer._cb_w13_gate_qweight, fused[:, :INTER, :])
    assert torch.equal(layer._cb_w13_up_qweight, fused[:, INTER:, :])
    assert torch.equal(layer._cb_w13_gate_scale, fused_scale[:, :INTER])
    assert torch.equal(layer._cb_w13_up_scale, fused_scale[:, INTER:])


def test_split_halves_are_contiguous_and_fp32():
    """The kernels validate `packed.size(1) == N` on a stacked-expert buffer,
    and a column view of the fused stack keeps the fused expert stride — so
    the halves must be real contiguous copies, not views."""
    layer, method = _layer(), _method()
    method._split_w13_by_role(layer)

    for role in ("gate", "up"):
        packed = getattr(layer, f"_cb_w13_{role}_qweight")
        assert packed.is_contiguous() and packed.shape == (E, INTER, ROW_BYTES)
        scale = getattr(layer, f"_cb_w13_{role}_scale")
        assert scale.is_contiguous() and scale.dtype is torch.float32
    assert layer._cb_w2_scale.shape == (E, HIDDEN)
    assert layer._cb_w2_scale.dtype is torch.float32


def test_split_releases_the_fused_stack():
    """Keeping both would double w13 residency for the whole serve — ~24 GB
    across DSv4's 43 layers, on a box whose entire budget is 128 GB."""
    layer, method = _layer(), _method()
    method._split_w13_by_role(layer)

    assert layer.w13_cb_qweight.data.numel() == 0
    assert layer._cb_w13_fused_shape == (E, 2 * INTER, ROW_BYTES)


# ---------------------------------------------------------------------------
# E. Formats the split is not implemented for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_fp4,is_v2,n_sub,match", [
    (True, True, 2, "FP8-CB v1 only"),
    (True, False, 4, "FP8-CB v1 only"),
    (False, True, 4, "FP8-CB v1 only"),
    (False, False, 1, "n_sub=4"),
])
def test_unsupported_layouts_refuse_at_load(is_fp4, is_v2, n_sub, match):
    """fp4 carries a second per-layer table and a v2 in-kernel scale section.
    Splitting those is the same idea at a different set of call sites;
    shipping it untested would be worse than refusing it."""
    method = _method(is_fp4=is_fp4, is_v2=is_v2, n_sub=n_sub)

    with pytest.raises(ValueError, match=match):
        method._require_per_role_supported()


def test_supported_layout_is_accepted():
    _method(is_fp4=False, is_v2=False, n_sub=4)._require_per_role_supported()


def test_untaught_lane_refuses_on_a_split_stack():
    layer, method = _layer(), _method()
    layer._cb_role_split = True

    with pytest.raises(ValueError, match="no per-role codebook support"):
        method._require_per_role_lane(layer, "persistent_b")


def test_untaught_lane_is_a_no_op_on_a_uniform_stack():
    """Uniform serving pays one attribute read for the whole guard."""
    _method()._require_per_role_lane(_layer(), "persistent_b")


# ---------------------------------------------------------------------------
# (F) Device equivalence: identical books through the split are bit-exact
# ---------------------------------------------------------------------------
#
# The strongest per-role evidence obtainable before a per-role ARTIFACT exists.
# Bind all three roles to the SAME codebook and the split path must reproduce
# the fused path bit for bit -- same weights, same routing, same books, only
# the plumbing differs -- so the test needs no new ground truth to call a
# difference a defect.
#
# It covers the two GATES as well as the three lanes, which is where the
# split's real hazards live. `_cuda_moe_ok` materialises a LUT, and the stock
# materialiser reads `_cb_flat`, which a per-role layer does not have: that is
# an AttributeError inside the gate, before the per-role decode branch runs.
# `_gf2_ok` validates `w13_cb_qweight`'s 3-D stride, which the split RELEASES:
# a False there is not a crash but a silent demotion to the BF16 bridge at
# ~663 ms/layer against ~29 ms on the fused lane, a 22x prefill cliff with no
# error and no log. Both gates are asserted True, not merely non-raising.


def _identical_book_pair(k: int = 28, seed: int = 101):
    """A uniform layer and a per-role twin whose three roles share one book.

    Two `_build` calls at one seed are identical by construction (it seeds
    torch itself); asserted rather than assumed, because the whole comparison
    is void if the stacks differ.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for per-role equivalence")
    from test_moe_grouped_fused import _build

    ref_method, ref_layer, dims = _build(seed=seed, k=k)
    method, layer, _ = _build(seed=seed, k=k)
    assert torch.equal(ref_layer.w13_cb_qweight, layer.w13_cb_qweight)
    assert torch.equal(ref_layer.w2_cb_qweight, layer.w2_cb_qweight)

    method._require_per_role_supported()
    flat = layer._cb_flat
    layer._cb_flat_by_role = {role: flat for role in ("gate", "up", "down")}
    # A real per-role load never builds `_cb_flat` -- only the by-role books.
    # Deleting it is what lets this fixture CATCH a lane that reaches for the
    # stock book; leaving it would make every such bug pass silently here and
    # fail only on the first real artifact.
    del layer._cb_flat
    layer._cb_role_split = True
    method._split_w13_by_role(layer)
    assert layer.w13_cb_qweight.numel() == 0, "fused w13 stack must be released"
    return (ref_method, ref_layer), (method, layer), dims


def _bits(t):
    return t.contiguous().view(torch.uint16)


def _inputs(dims, tokens, topk, seed):
    from test_moe_grouped_fused import DEV, _routing, _silu_act
    ids, weights = _routing(tokens, dims["E"], topk, "uniform", seed=seed)
    torch.manual_seed(seed + 8)
    x = torch.randn(tokens, dims["hidden"],
                    dtype=torch.bfloat16, device=DEV) * 0.5
    return x, weights, ids, _silu_act()


def test_decode_gate_admits_a_split_layer():
    """`_cuda_moe_ok` must build the role books, not the absent stock book."""
    _, (method, layer), _ = _identical_book_pair()
    assert method._cuda_moe_ok(layer) is True
    assert not hasattr(layer, "_cb_flat_fp8"), \
        "per-role layer must never materialise the stock per-layer LUT"
    assert set(layer._cb_flat_fp8_by_role) == {"gate", "up", "down"}


def test_prefill_gate_admits_a_split_layer():
    """A per-role layer must stay ON the fused lane, not fall to the bridge."""
    (ref_method, ref_layer), (method, layer), _ = _identical_book_pair()
    if not ref_method._gf2_ok(ref_layer):
        pytest.skip("FP8-CB grouped fused CUTLASS prefill unavailable")
    assert method._gf2_ok(layer) is True, (
        "per-role layer was demoted off the fused prefill lane: "
        f"{layer._cb_gf2_ok_reason}")


@pytest.mark.parametrize("tokens,topk", [(1, 6), (13, 2)])
def test_decode_is_bit_identical_with_identical_books(tokens, topk):
    (ref_method, ref_layer), (method, layer), dims = _identical_book_pair()
    assert method._cuda_moe_ok(layer) and ref_method._cuda_moe_ok(ref_layer)
    x, weights, ids, act = _inputs(dims, tokens, topk, seed=13)
    reference = ref_method._apply_grouped_decode(
        ref_layer, x, weights, ids, act)
    candidate = method._apply_grouped_decode(layer, x, weights, ids, act)
    assert torch.equal(_bits(reference), _bits(candidate))


def test_grouped_prefill_is_bit_identical_with_identical_books():
    (ref_method, ref_layer), (method, layer), dims = _identical_book_pair()
    if not ref_method._gf2_ok(ref_layer):
        pytest.skip("FP8-CB grouped fused CUTLASS prefill unavailable")
    assert method._gf2_ok(layer), layer._cb_gf2_ok_reason
    x, weights, ids, act = _inputs(dims, 48, 2, seed=17)
    reference = ref_method._apply_prefill_grouped_fused_v2(
        ref_layer, x, weights, ids, act)
    candidate = method._apply_prefill_grouped_fused_v2(
        layer, x, weights, ids, act)
    assert reference is not None and candidate is not None
    assert torch.equal(_bits(reference), _bits(candidate))


def test_bridge_expand_is_bit_identical_with_identical_books():
    """The quality bridge expands w13 per role and must rebuild the fused row
    order exactly (gate rows then up rows, per expert)."""
    from gridbook.cuda_ext import get_bf16_grouped_ext
    (ref_method, ref_layer), (method, layer), dims = _identical_book_pair()
    if get_bf16_grouped_ext() is None:
        pytest.skip("owned grouped-BF16 CUTLASS reference unavailable")
    x, weights, ids, act = _inputs(dims, 33, 2, seed=23)
    reference = ref_method._apply_prefill_native_bf16(
        ref_layer, x, weights, ids, act)
    candidate = method._apply_prefill_native_bf16(
        layer, x, weights, ids, act)
    assert torch.equal(_bits(reference), _bits(candidate))
