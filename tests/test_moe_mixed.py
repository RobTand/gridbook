"""CPU registration/loading checks for family-specific split stacks."""
from __future__ import annotations

from dataclasses import dataclass
import types

import pytest
import torch

from gridbook.per_expert_format import parse_declaration
from gridbook.runtime_contract import load_runtime_contract


def _declaration():
    config = {
        "per_expert_format_groups": {
            "version": 1,
            "layers": {"0": {
                "w13": [
                    {"format_wire_id": "NVFP4_CB_K16", "expert_ids": [0, 2],
                     "tensor_prefix": (
                         "model.layers.0.experts.gate_up_proj."
                         "format_group_nvfp4_cb_k16")},
                    {"format_wire_id": "FP8_CB_K28", "expert_ids": [1, 3],
                     "tensor_prefix": (
                         "model.layers.0.experts.gate_up_proj."
                         "format_group_fp8_cb_k28")},
                ],
                "w2": [
                    {"format_wire_id": "FP8_CB_K28", "expert_ids": [0, 1, 3],
                     "tensor_prefix": (
                         "model.layers.0.experts.down_proj."
                         "format_group_fp8_cb_k28")},
                    {"format_wire_id": "NVFP4_CB_K16", "expert_ids": [2],
                     "tensor_prefix": (
                         "model.layers.0.experts.down_proj."
                         "format_group_nvfp4_cb_k16")},
                ],
            }},
        }
    }
    schemes = {}
    for family in ("w13", "w2"):
        for entry in config["per_expert_format_groups"]["layers"]["0"][family]:
            fp4 = entry["format_wire_id"].startswith("NVFP4")
            schemes[entry["tensor_prefix"]] = {
                "grid": "fp4" if fp4 else "fp8",
                "mode": "product", "k": 16 if fp4 else 28,
                "n_sub": 2 if fp4 else 4,
                "type_size": 73 if fp4 else 112,
                "scale_coding": "two_tier" if fp4 else "v1",
                "codebook_ref": "unused",
            }
    groups = parse_declaration(
        config, runtime_contract=load_runtime_contract(), cb_schemes=schemes
    )["0"]
    return groups, schemes


def test_registers_per_family_buffers_and_index_maps():
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    method_cls = moe_mod.PrismaQuantMixedMoEMethod
    groups, schemes = _declaration()
    method = method_cls.__new__(method_cls)
    method.groups = groups
    method.prefix = "model.layers.0.experts"
    method.quant_config = types.SimpleNamespace(
        target_scheme=schemes,
        _per_expert_serving_prefixes={name: name for name in schemes},
    )
    method._source_groups = {"w13": [], "w2": []}
    layer = torch.nn.Module()
    layer.load_weights = lambda weights: (name for name, _weight in weights)

    method.create_weights(
        layer, 4, 256, 256, torch.bfloat16, weight_loader=None
    )
    assert layer.w13_format_group_nvfp4_cb_k16_cb_qweight.shape == (2, 512, 73)
    assert layer.w13_format_group_fp8_cb_k28_cb_qweight.shape == (2, 512, 112)
    assert layer.w2_format_group_fp8_cb_k28_cb_qweight.shape == (3, 256, 112)
    assert layer.w2_format_group_nvfp4_cb_k16_cb_qweight.shape == (1, 256, 73)
    assert layer.w13_format_group_fp8_cb_k28_weight_scale.shape == (2, 512)
    assert layer.w2_format_group_fp8_cb_k28_weight_scale.shape == (3, 256)
    assert layer._w13_format_group.tolist() == [0, 1, 0, 1]
    assert layer._w13_format_position.tolist() == [0, 0, 1, 1]
    assert layer._w2_format_group.tolist() == [0, 0, 1, 0]
    assert layer._w2_format_position.tolist() == [0, 1, 0, 2]


def test_explicit_fp8_v2_refuses_mixed_fp8_groups(monkeypatch):
    """A global candidate arm must not silently inherit inside mixed groups."""
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    from gridbook import moe_gemv_select

    groups, schemes = _declaration()
    method = moe_mod.PrismaQuantMixedMoEMethod.__new__(
        moe_mod.PrismaQuantMixedMoEMethod
    )
    method.groups = groups
    method.prefix = "model.layers.0.experts"
    method.quant_config = types.SimpleNamespace(
        target_scheme=schemes,
        _per_expert_serving_prefixes={name: name for name in schemes},
    )
    monkeypatch.setattr(moe_gemv_select, "_CB_FP8_GEMV_V2", None)
    monkeypatch.setenv("PRISMAQUANT_CB_FP8_GEMV_V2", "1")

    with pytest.raises(RuntimeError, match="silently inherited candidate arm"):
        method._require_fp8_v2_dispatch_supported()


def test_disabled_fp8_v2_leaves_mixed_groups_on_inherited(monkeypatch):
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    from gridbook import moe_gemv_select

    groups, schemes = _declaration()
    method = moe_mod.PrismaQuantMixedMoEMethod.__new__(
        moe_mod.PrismaQuantMixedMoEMethod
    )
    method.groups = groups
    method.prefix = "model.layers.0.experts"
    method.quant_config = types.SimpleNamespace(
        target_scheme=schemes,
        _per_expert_serving_prefixes={name: name for name in schemes},
    )
    monkeypatch.setattr(moe_gemv_select, "_CB_FP8_GEMV_V2", None)
    monkeypatch.setenv("PRISMAQUANT_CB_FP8_GEMV_V2", "0")

    method._require_fp8_v2_dispatch_supported()


def test_family_loader_copies_exact_substacks_without_legacy_transpose():
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    groups, schemes = _declaration()
    method = moe_mod.PrismaQuantMixedMoEMethod.__new__(
        moe_mod.PrismaQuantMixedMoEMethod
    )
    method.groups = groups
    method.prefix = "model.layers.0.experts"
    method.quant_config = types.SimpleNamespace(
        target_scheme=schemes,
        _per_expert_serving_prefixes={name: name for name in schemes},
    )
    method._source_groups = {"w13": [], "w2": []}
    layer = torch.nn.Module()
    delegated = []

    def original(weights):
        for name, _weight in weights:
            delegated.append(name)
            yield name

    layer.load_weights = original
    method.create_weights(
        layer, 4, 256, 256, torch.bfloat16, weight_loader=None
    )
    q = torch.arange(2 * 512 * 73, dtype=torch.uint8).reshape(2, 512, 73)
    names = list(layer.load_weights(iter([
        (groups.w13[0].tensor_prefix + ".cb_qweight", q),
        ("router.weight", torch.ones(1)),
    ])))
    assert torch.equal(layer.w13_format_group_nvfp4_cb_k16_cb_qweight, q)
    assert getattr(
        layer.w13_format_group_nvfp4_cb_k16_cb_qweight, "_pq_cb_filled"
    ) is True
    assert names == ["w13_format_group_nvfp4_cb_k16_cb_qweight", "router.weight"]
    assert delegated == ["router.weight"]


def test_static_input_scale_is_one_shared_value_per_family():
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    groups, schemes = _declaration()
    for group in groups.w13:
        schemes[group.tensor_prefix]["activation_contract"] = "k0.2"
    method = moe_mod.PrismaQuantMixedMoEMethod.__new__(
        moe_mod.PrismaQuantMixedMoEMethod
    )
    method.groups = groups
    method.prefix = "model.layers.0.experts"
    method.quant_config = types.SimpleNamespace(
        target_scheme=schemes,
        _per_expert_serving_prefixes={name: name for name in schemes},
    )
    method._source_groups = {"w13": [], "w2": []}
    layer = torch.nn.Module()
    layer.load_weights = lambda weights: (name for name, _weight in weights)
    method.create_weights(layer, 4, 256, 256, torch.bfloat16)

    one, two = groups.w13
    assert list(layer.load_weights(iter([
        (one.tensor_prefix + ".input_global_scale", torch.tensor([0.25])),
        (two.tensor_prefix + ".input_global_scale", torch.tensor([0.25])),
    ]))) == ["w13_input_global_scale", "w13_input_global_scale"]
    with pytest.raises(ValueError, match="differs across format subgroups"):
        list(layer.load_weights(iter([(
            two.tensor_prefix + ".input_global_scale", torch.tensor([0.5])
        )])))


def test_partial_passthrough_builds_only_declared_family_delegate():
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    prefix = "model.layers.0.experts"
    config = {
        "per_expert_format_groups": {
            "version": 1,
            "layers": {"0": {
                "w13": [{
                    "format_wire_id": "FP8_CB_K28",
                    "expert_ids": [0, 1, 2, 3],
                    "tensor_prefix": (
                        prefix + ".gate_up_proj.format_group_fp8_cb_k28"
                    ),
                }],
                "w2": [{
                    "format_wire_id": "NVFP4_CB_K16",
                    "expert_ids": [0, 1],
                    "tensor_prefix": (
                        prefix + ".down_proj.format_group_nvfp4_cb_k16"
                    ),
                }, {
                    "format_wire_id": "mxfp4_e2m1_ue8m0_g32",
                    "expert_ids": [2, 3],
                    "tensor_prefix": prefix,
                }],
            }},
        }
    }
    schemes = {
        config["per_expert_format_groups"]["layers"]["0"]["w13"][0][
            "tensor_prefix"
        ]: {
            "grid": "fp8", "mode": "product", "k": 28, "n_sub": 4,
            "type_size": 112, "scale_coding": "v1", "codebook_ref": "x",
        },
        config["per_expert_format_groups"]["layers"]["0"]["w2"][0][
            "tensor_prefix"
        ]: {
            "grid": "fp4", "mode": "product", "k": 16, "n_sub": 2,
            "type_size": 73, "scale_coding": "two_tier",
            "codebook_ref": "x",
        },
    }
    groups = parse_declaration(
        config, runtime_contract=load_runtime_contract(), cb_schemes=schemes
    )["0"]

    @dataclass(frozen=True)
    class FakeMoe:
        num_experts: int
        num_local_experts: int
        num_logical_experts: int

    class FakeNativeMethod:
        def create_weights(self, owner, count, hidden, inter, dtype, **attrs):
            del dtype, attrs
            for family, shape, scale_shape in (
                ("w13", (count, 2 * inter, hidden // 2),
                 (count, 2 * inter, hidden // 32)),
                ("w2", (count, hidden, inter // 2),
                 (count, hidden, inter // 32)),
            ):
                owner.register_parameter(
                    f"{family}_weight",
                    torch.nn.Parameter(torch.zeros(shape, dtype=torch.uint8),
                                       requires_grad=False),
                )
                owner.register_parameter(
                    f"{family}_weight_scale",
                    torch.nn.Parameter(torch.zeros(
                        scale_shape, dtype=torch.uint8
                    ), requires_grad=False),
                )

    delegates = []

    def delegate(_owner, unit, _format):
        delegates.append(unit)
        return FakeNativeMethod()

    quant = types.SimpleNamespace(
        target_scheme=schemes,
        _per_expert_serving_prefixes={
            group.tensor_prefix: group.tensor_prefix
            for family in ("w13", "w2")
            for group in groups.groups(family)
        },
        _delegate_passthrough=delegate,
    )
    method = moe_mod.PrismaQuantMixedMoEMethod(
        quant, FakeMoe(4, 4, 4), groups, prefix
    )
    layer = torch.nn.Module()
    layer.activation = types.SimpleNamespace(value="silu")
    layer.load_weights = lambda weights: (name for name, _weight in weights)
    method.create_weights(layer, 4, 256, 256, torch.bfloat16)

    assert delegates == [prefix + "/w2"]
    assert method._source_groups["w13"] == []
    assert len(method._source_groups["w2"]) == 1
    runtime = method._source_groups["w2"][0]
    owner = getattr(layer, runtime.layer_name)
    assert owner.w2_weight._gridbook_source_expert_ids == (2, 3)
    assert not hasattr(owner.w13_weight, "_gridbook_source_expert_ids")


def test_split_prefixes_follow_vllm_mapper_without_losing_wire_names():
    moe_mod = pytest.importorskip("gridbook.moe_mixed")
    from gridbook.config import PrismaQuantConfig

    groups, schemes = _declaration()
    raw_groups = {}
    for index, (target, scheme) in enumerate(schemes.items()):
        raw_groups[str(index)] = {
            "format": "NVFP4_CB_K16" if scheme["grid"] == "fp4"
                      else "FP8_CB_K28",
            "targets": [target],
            "scheme": scheme,
        }
    declaration = {
        "version": 1,
        "layers": {"0": {
            "w13": [{
                "format_wire_id": group.format_wire_id,
                "expert_ids": list(group.expert_ids),
                "tensor_prefix": group.tensor_prefix,
            } for group in groups.w13],
            "w2": [{
                "format_wire_id": group.format_wire_id,
                "expert_ids": list(group.expert_ids),
                "tensor_prefix": group.tensor_prefix,
            } for group in groups.w2],
        }},
    }
    config = PrismaQuantConfig.from_config({
        "quant_method": "gridbook", "format": "mixed-precision",
        "codebook_file": "unused", "config_groups": raw_groups,
        "ignore": [], "per_expert_format_groups": declaration,
    })
    config._ensure_resolved()

    class WrapperMapper:
        @staticmethod
        def _map(name):
            return "language_model." + name if name.startswith("model.") else name

        def apply_list(self, values):
            return [self._map(value) for value in values]

        def apply_dict(self, values):
            return {self._map(key): value for key, value in values.items()}

    physical = groups.w13[0].tensor_prefix
    config.apply_vllm_mapper(WrapperMapper())
    assert config._per_expert_serving_prefixes[physical] == (
        "language_model." + physical
    )
    resolved = config._mixed_moe_groups_for_prefix(
        "language_model.model.layers.0.experts"
    )
    assert resolved is not None
    assert resolved.w13[0].tensor_prefix == physical
    method = moe_mod.PrismaQuantMixedMoEMethod.__new__(
        moe_mod.PrismaQuantMixedMoEMethod
    )
    method.quant_config = config
    method.prefix = "language_model.model.layers.0.experts"
    assert method._scheme_for(resolved.w13[0])["k"] in (16, 28)
