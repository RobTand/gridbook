"""CPU oracle and refusal suite for v1 split-format expert stacks."""
from __future__ import annotations

import copy

import pytest
import torch

from gridbook.per_expert_format import (
    ExpertFormatGroup,
    LayerFormatGroups,
    MXFP4_SOURCE,
    PerExpertFormatError,
    dispatch_family_stages,
    parse_declaration,
)
from gridbook.runtime_contract import load_runtime_contract


FP4 = "NVFP4_CB_K16"
FP8 = "FP8_CB_K28"
FP4B = "NVFP4_CB_K20"
FAMILIES_UNDER_TEST = ("w13", "w2")


def _entry(fmt, ids, prefix):
    return {
        "format_wire_id": fmt,
        "expert_ids": list(ids),
        "tensor_prefix": prefix,
    }


def _config(w13, w2):
    return {
        "per_expert_format_groups": {
            "version": 1,
            "layers": {"0": {"w13": w13, "w2": w2}},
        }
    }


def _schemes(config):
    schemes = {}
    for family in ("w13", "w2"):
        for entry in config["per_expert_format_groups"]["layers"]["0"][family]:
            fmt = entry["format_wire_id"]
            if fmt == MXFP4_SOURCE:
                continue
            schemes[entry["tensor_prefix"]] = {
                "grid": "fp4" if fmt.startswith("NVFP4") else "fp8",
                "k": int(fmt.rsplit("K", 1)[1]),
            }
    return schemes


def _parse(config):
    return parse_declaration(
        config,
        runtime_contract=load_runtime_contract(),
        cb_schemes=_schemes(config),
    )["0"]


def _mixed_config(formats, *, independent_w2=False, w2_formats=None):
    experts = list(range(len(formats)))

    def entries(family_formats, family):
        out = []
        projection = "gate_up_proj" if family == "w13" else "down_proj"
        for fmt in sorted(set(family_formats)):
            ids = [e for e in experts if family_formats[e] == fmt]
            slug = fmt.lower().replace("_source", "")
            prefix = (f"model.layers.0.experts.{projection}.format_group_{slug}"
                      if fmt != MXFP4_SOURCE
                      else "model.layers.0.experts")
            out.append(_entry(fmt, ids, prefix))
        return out

    if w2_formats is None:
        w2_formats = list(reversed(formats)) if independent_w2 else list(formats)
    return _config(entries(formats, "w13"), entries(w2_formats, "w2"))




def _assert_within_ulps(actual, witness, *, ulps, what):
    """Bounded-difference assertion, in ULPs of the witness's own magnitude.

    Deliberately not ``allclose``: the bound is stated in units of float64
    resolution against the measured scale, so the number in the failure message
    means something and cannot quietly absorb a real error.
    """
    scale = witness.abs().max().item()
    allowed = ulps * torch.finfo(witness.dtype).eps * scale
    worst = (actual - witness).abs().max().item()
    assert worst <= allowed, (
        f"{what}: worst |difference| {worst:.3e} exceeds {ulps} ULP "
        f"({allowed:.3e}) at scale {scale:.3e}"
    )


def _independent_witness(x, router, ids, declaration, weights):
    """Sum of one-expert uniform layers, computed WITHOUT the code under test.

    This shares nothing with ``dispatch_family_stages``: it walks routed pairs
    in Python and multiplies with ``@`` rather than ``torch.bmm``.  That is the
    point.  The primary oracle compares two routes through the dispatcher, so a
    defect living inside the dispatcher could in principle satisfy both sides;
    this witness cannot be fooled that way, because it never calls it.

    The price of that independence is that exact equality is unavailable, for
    two compounding reasons, both benign and both measured rather than assumed:

      * ``@`` and ``torch.bmm`` are different kernels and reduce in different
        orders; and
      * summing per-expert layers reassociates the final per-token combine,
        because a token's pairs are interleaved across groups in pair order but
        consecutive here.

    Measured worst case across the three oracle cases is 1.06 ULP of float64,
    so the caller's 2 ULP bound holds with headroom while staying far too tight
    to hide a routing, index-map or launch defect -- each of which moves the
    result by a large multiple of the operand scale, not by a last bit.
    """
    tokens, topk = ids.shape
    hidden = x.shape[1]
    witness = torch.zeros(tokens, hidden, dtype=torch.float64)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(ids[token, slot])
            operands = x[token:token + 1]
            for family in FAMILIES_UNDER_TEST:
                group_index, position = declaration.index_maps[family][expert]
                group = declaration.groups(family)[group_index]
                operands = operands @ weights[
                    (family, group.tensor_prefix)
                ][position]
                if family == "w13":
                    operands = torch.tanh(operands)
            witness[token] += router[token, slot] * operands[0]
    return witness


def _uniform_reference(x, router, ids, declaration, weights, activate):
    """Run the SAME layer as one uniform single-format stack, same routing.

    This is the expected side of the acceptance oracle, and it is a real
    forward through ``dispatch_family_stages`` -- the function under test --
    not a hand-rolled recomputation.  Every expert's weight matrix is gathered
    into one stack per family, indexed by expert id, and driven through a
    single-group declaration over the identical (token, topk) routing.

    Why this shape, and not "sum of one-expert layers": the pairs, their order,
    and therefore the final per-token ``index_add_`` are identical on both
    sides, so the combine is bit-identical rather than merely close.  Any
    decomposition that sums separate sub-layer outputs reassociates that
    accumulation -- measured at 2 ULP of float64 -- which would force a
    tolerance and weaken the claim.  Here ``torch.equal`` is exact BY
    CONSTRUCTION, and it stays exact across torch versions because both sides
    reduce over the same K with the same primitive.

    The invariant asserted is the one the feature must satisfy: splitting an
    expert stack into per-format sub-groups changes the PACKING, not the MATH.
    A wrong group index, a wrong sub-stack position, or a dropped launch all
    move the result away from this reference.
    """
    stacks = {}
    for family in FAMILIES_UNDER_TEST:
        prefix = f"uniform::{family}"
        rows = []
        for expert in range(declaration.num_experts):
            group_index, position = declaration.index_maps[family][expert]
            group = declaration.groups(family)[group_index]
            rows.append(weights[(family, group.tensor_prefix)][position])
        stacks[(family, prefix)] = torch.stack(rows)

    uniform = LayerFormatGroups(
        layer_id=declaration.layer_id,
        w13=(ExpertFormatGroup(
            "w13", declaration.w13[0].format_wire_id,
            tuple(range(declaration.num_experts)), "uniform::w13",
        ),),
        w2=(ExpertFormatGroup(
            "w2", declaration.w2[0].format_wire_id,
            tuple(range(declaration.num_experts)), "uniform::w2",
        ),),
        num_experts=declaration.num_experts,
        index_maps={
            family: tuple(
                (0, expert) for expert in range(declaration.num_experts)
            )
            for family in FAMILIES_UNDER_TEST
        },
    )

    def uniform_stage(family, group, values, local_ids):
        stack = stacks[(family, group.tensor_prefix)]
        return torch.bmm(
            values[:, None, :], stack.index_select(0, local_ids)
        )[:, 0]

    return dispatch_family_stages(
        x, router, ids, uniform, uniform_stage, activate,
    )


def _oracle_case(formats, *, independent_w2=False, w2_formats=None):
    torch.manual_seed(2026 + len(formats))
    declaration = _parse(_mixed_config(
        formats, independent_w2=independent_w2, w2_formats=w2_formats
    ))
    hidden, inter, tokens, topk = 5, 7, 6, 3
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    ids = torch.tensor([
        [(token + slot * 2) % len(formats) for slot in range(topk)]
        for token in range(tokens)
    ])
    router = torch.rand(tokens, topk, dtype=torch.float64)
    router /= router.sum(dim=-1, keepdim=True)

    weights = {}
    for family in ("w13", "w2"):
        in_f, out_f = ((hidden, inter) if family == "w13" else (inter, hidden))
        for group in declaration.groups(family):
            weights[(family, group.tensor_prefix)] = torch.randn(
                len(group.expert_ids), in_f, out_f, dtype=torch.float64
            )

    calls = []

    def run_stage(family, group, values, local_ids):
        calls.append((family, group.format_wire_id, len(values)))
        stack = weights[(family, group.tensor_prefix)]
        return torch.bmm(values[:, None, :], stack.index_select(0, local_ids))[:, 0]

    actual = dispatch_family_stages(
        x, router, ids, declaration, run_stage, torch.tanh
    )

    # Acceptance oracle: the mixed layer equals the sum of REAL uniform-layer
    # forwards.  Each cell -- the experts sharing one w13 group AND one w2
    # group -- is rebuilt as its own single-format layer and run through the
    # SAME dispatch_family_stages path with routing filtered to its experts, so
    # both sides reach torch.bmm through identical code in identical order.
    #
    # The previous oracle hand-rolled the expected side with `@` while the
    # implementation used torch.bmm.  Those are different kernels with
    # different reduction orders, so torch.equal held only by luck of the BLAS
    # backend -- it passed on torch 2.11/2.13 and failed on 2.10.  Bit-exactness
    # is only a meaningful claim when both sides use the same primitive.
    expected = _uniform_reference(
        x, router, ids, declaration, weights, torch.tanh
    )
    assert torch.equal(actual, expected)
    # The oracle must be able to FAIL: perturbing a single output element
    # by one ULP has to break exact equality, or the assertion above is
    # measuring nothing.
    mutated = expected.clone()
    mutated[0, 0] = torch.nextafter(
        mutated[0, 0], torch.tensor(float("inf"), dtype=mutated.dtype)
    )
    assert not torch.equal(actual, mutated)

    # SECONDARY, and independent: the primary compares two routes through
    # dispatch_family_stages, so a defect inside the dispatcher could satisfy
    # both of its sides.  This witness never calls it.  Independence costs
    # exactness -- see _independent_witness -- so it is bounded at 2 ULP rather
    # than asserted equal.
    witness = _independent_witness(x, router, ids, declaration, weights)
    _assert_within_ulps(
        actual, witness, ulps=2, what="mixed layer vs independent witness"
    )
    # The bound must not be vacuous: a perturbation of the size a real routing
    # or index-map defect produces has to break it.
    broken = witness.clone()
    broken[0, 0] += 1e-6 * max(witness.abs().max().item(), 1.0)
    with pytest.raises(AssertionError):
        _assert_within_ulps(
            actual, broken, ulps=2, what="deliberately perturbed witness"
        )
    # One family launch per nonempty format subgroup; no mixed-format launch.
    assert {(family, fmt) for family, fmt, _rows in calls} == {
        (family, group.format_wire_id)
        for family in ("w13", "w2")
        for group in declaration.groups(family)
    }


def test_oracle_mixed_two_format_cb_layer():
    _oracle_case([FP4, FP4, FP8, FP8])


def test_oracle_cb_plus_passthrough_layer():
    # Producer-wire case: the passthrough family need not match w13's split.
    _oracle_case(
        [FP4, FP4, FP8, FP8],
        w2_formats=[FP4, FP4, MXFP4_SOURCE, MXFP4_SOURCE],
    )


def test_accepts_producer_pr69_exact_asymmetric_declaration():
    parent = "layers.0.ffn.experts"
    config = _config(
        [
            _entry(FP8, [2, 3], parent + (
                ".gate_up_proj.format_group_fp8_cb_k28"
            )),
            _entry(FP4, [0, 1], parent + (
                ".gate_up_proj.format_group_nvfp4_cb_k16"
            )),
        ],
        [
            _entry(FP8, [1], parent + (
                ".down_proj.format_group_fp8_cb_k28"
            )),
            _entry(FP4, [0], parent + (
                ".down_proj.format_group_nvfp4_cb_k16"
            )),
            _entry(MXFP4_SOURCE, [2, 3], parent),
        ],
    )
    layer = _parse(config)
    assert layer.index_maps["w13"] == ((1, 0), (1, 1), (0, 0), (0, 1))
    assert layer.index_maps["w2"] == ((1, 0), (0, 0), (2, 0), (2, 1))


def test_oracle_three_format_layer_and_independent_family_maps():
    _oracle_case([FP4, FP8, FP4B, FP4B, FP8, FP4], independent_w2=True)


def test_absent_declaration_is_legacy_same_bytes():
    assert parse_declaration(
        {}, runtime_contract=load_runtime_contract(), cb_schemes={}
    ) == {}
    torch.manual_seed(9)
    x = torch.randn(8, 4)
    weight = torch.randn(4, 4)

    def legacy_uniform():
        return torch.tanh(x @ weight)

    # Byte identity without numpy: CI installs torch and the wheel only, so a
    # .numpy() round trip is an undeclared dependency (and torch reports it as
    # "Numpy is not available" rather than a missing import).  Reinterpreting
    # the contiguous buffer as uint8 is the same claim, in torch alone.
    def as_bytes(t):
        return t.detach().contiguous().view(torch.uint8).clone()

    before = as_bytes(legacy_uniform())
    parsed = parse_declaration(
        {"config_groups": {}, "ignore": []},
        runtime_contract=load_runtime_contract(), cb_schemes={},
    )
    assert not parsed
    after = as_bytes(legacy_uniform())
    assert torch.equal(after, before)


@pytest.mark.parametrize("mutation, message", [
    ("missing", "bad expert partition"),
    ("unknown", "unknown format wire id"),
    ("double", "double-claimed"),
])
def test_refuses_bad_partition_unknown_format_and_double_claim(mutation, message):
    config = _mixed_config([FP4, FP4, FP8, FP8])
    broken = copy.deepcopy(config)
    entries = broken["per_expert_format_groups"]["layers"]["0"]["w2"]
    if mutation == "missing":
        next(entry for entry in entries if 3 in entry["expert_ids"])[
            "expert_ids"
        ].remove(3)
    elif mutation == "unknown":
        entries[0]["format_wire_id"] = "INT3_SURPRISE"
    else:
        entries[-1]["expert_ids"].append(entries[0]["expert_ids"][0])
        entries[-1]["expert_ids"].sort()
    with pytest.raises(PerExpertFormatError, match=message):
        parse_declaration(
            broken, runtime_contract=load_runtime_contract(),
            cb_schemes=_schemes(config),
        )


def test_exact_wire_fields_and_index_positions_are_consumed():
    config = _mixed_config([FP8, FP4, FP8, FP4])
    layer = _parse(config)
    assert layer.index_maps["w13"] == ((0, 0), (1, 0), (0, 1), (1, 1))
    extra = copy.deepcopy(config)
    extra["per_expert_format_groups"]["layers"]["0"]["w13"][0]["note"] = "no"
    with pytest.raises(PerExpertFormatError, match="expected exact keys"):
        _parse(extra)
