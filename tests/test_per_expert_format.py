"""CPU oracle and refusal suite for v1 split-format expert stacks."""
from __future__ import annotations

import copy

import pytest
import torch

from gridbook.per_expert_format import (
    MXFP4_SOURCE,
    PerExpertFormatError,
    dispatch_family_stages,
    parse_declaration,
)
from gridbook.runtime_contract import load_runtime_contract


FP4 = "NVFP4_CB_K16"
FP8 = "FP8_CB_K28"
FP4B = "NVFP4_CB_K20"


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

    # Acceptance oracle: sum of each routed expert evaluated as its own
    # one-expert uniform layer, with the same family-local stack slices.
    expected = torch.zeros(tokens, hidden, dtype=torch.float64)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(ids[token, slot])
            operands = x[token:token + 1]
            for family in ("w13", "w2"):
                group_index, position = declaration.index_maps[family][expert]
                group = declaration.groups(family)[group_index]
                operands = operands @ weights[(family, group.tensor_prefix)][position]
                if family == "w13":
                    operands = torch.tanh(operands)
            expected[token] += router[token, slot] * operands[0]
    assert torch.equal(actual, expected)
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

    before = legacy_uniform().numpy().tobytes()
    parsed = parse_declaration(
        {"config_groups": {}, "ignore": []},
        runtime_contract=load_runtime_contract(), cb_schemes={},
    )
    after = legacy_uniform().numpy().tobytes() if not parsed else b"mixed"
    assert after == before


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
