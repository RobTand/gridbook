"""DSpark Gridbook loader contract: construction, registered, and wire names.

The production DSpark class constructs decoder layers with quantization
prefixes ``model.layers.43/44/45``, registers its three-element ``ModuleList``
as ``model.layers.0/1/2``, and reads physical checkpoint names ``mtp.0/1/2``.
These CPU fixtures preserve that split and the production
``_remap_dspark_name`` behavior.
"""
from __future__ import annotations

import importlib
import re
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from gridbook.cb_fill_guard import CB_FILLED_ATTR
from gridbook.mixed_linear import MixedFusedShard
from gridbook.moe_toplevel_loader import (
    _dspark_rename,
    _registered_mixed_source,
    _validate_dspark_target_bridge_model,
    install_toplevel_cb_expert_loader,
)
from gridbook.runtime_contract import load_runtime_contract


E, HID, INTER, BYTES = 2, 8, 4, 3


def _sharded_param(rows: int) -> torch.Tensor:
    param = torch.zeros(2 * rows, BYTES, dtype=torch.uint8)

    def weight_loader(destination, incoming, shard_id):
        start = int(shard_id) * rows
        destination.data[start:start + rows].copy_(incoming)

    param.weight_loader = weight_loader
    return param


def _mixed_carrier(*, source: str, group: str) -> torch.Tensor:
    param = torch.zeros(HID, BYTES, dtype=torch.uint8)
    param._gridbook_mixed_fused_source = source
    param._gridbook_mixed_fused_group = group
    param._gridbook_mixed_fused_plane = "cb_qweight"
    # The carrier ABI includes the narrowing law the real
    # MixedFusedLinearMethod.create_weights stamps on every plane; the router
    # refuses an unstamped carrier rather than assume a shard degree.  This
    # stub is an unsharded, output_dim-less plane, which is what that method
    # stamps for a bare tensor at tensor-parallel degree 1.
    param._gridbook_mixed_fused_shard = MixedFusedShard(
        col_degree=1, tp_rank=0, output_dim=None,
        packed_dim=None, packed_factor=None)
    return param


def _params(*, include_mixed: bool = False) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    for stage in range(3):
        routed = (
            f"model.layers.{stage}.ffn.experts.routed_experts"
        )
        params[routed + ".w13_cb_qweight"] = torch.zeros(
            E, 2 * INTER, BYTES, dtype=torch.uint8
        )
        params[routed + ".w2_cb_qweight"] = torch.zeros(
            E, HID, BYTES, dtype=torch.uint8
        )
        params[routed + ".w13_weight_scale"] = torch.zeros(
            E, 2 * INTER, dtype=torch.float32
        )
        params[routed + ".w2_weight_scale"] = torch.zeros(
            E, HID, dtype=torch.float32
        )
        params[routed + ".w13_input_global_scale"] = torch.full(
            (1,), torch.nan, dtype=torch.float32
        )
        params[routed + ".w2_input_global_scale"] = torch.full(
            (1,), torch.nan, dtype=torch.float32
        )

    # Stock DSpark owns ordinary direct and fused CB Linears. These are stage
    # zero only because that is enough to prove raw-name delegation.
    params["model.layers.0.attn.wq_b.cb_qweight"] = torch.zeros(
        HID, BYTES, dtype=torch.uint8
    )
    params["model.layers.0.attn.fused_wqa_wkv.cb_qweight"] = \
        _sharded_param(HID)
    params[
        "model.layers.0.ffn.shared_experts.gate_up_proj.cb_qweight"
    ] = _sharded_param(INTER)
    params[
        "model.layers.0.ffn.shared_experts.down_proj.cb_qweight"
    ] = torch.zeros(HID, BYTES, dtype=torch.uint8)

    # A heterogeneous fusion is owned by Gridbook's existing transaction
    # router. DSpark construction metadata stays at layer 44 while the actual
    # ModuleList parameter path is registered at layer 1.
    if include_mixed:
        registered_group = "model.layers.1.attn.fused_wqa_wkv"
        construction_group = "model.layers.44.attn.fused_wqa_wkv"
        for index, role in enumerate(("wq_a", "wkv")):
            name = (registered_group
                    + f"._gridbook_mixed_roles.{index}.cb_qweight")
            params[name] = _mixed_carrier(
                source=f"model.layers.44.attn.{role}",
                group=construction_group,
            )
    return params


def _make_dspark_cls(*, include_mixed: bool = False):
    class _FakeDSpark:
        def __init__(self):
            self._params = _params(include_mixed=include_mixed)
            self.delegated_raw: list[str] = []

        def named_parameters(self):
            return list(self._params.items())

        def _remap_dspark_name(self, name: str) -> str | None:
            """Faithful copy of production DSpark's namespace remapper."""
            match = re.match(r"mtp\.(\d+)\.(.*)", name)
            if match is None:
                return None
            stage = int(match.group(1))
            rest = match.group(2)
            if rest.startswith("confidence_head."):
                return None
            heads = (
                "norm.",
                "hc_head_fn",
                "hc_head_base",
                "hc_head_scale",
                "markov_head.",
            )
            if rest.startswith(("main_proj.", "main_norm.")) \
                    or rest.startswith(heads):
                return f"model.{rest}"
            return f"model.layers.{stage}.{rest}"

        def load_weights(self, weights):
            """Production-shaped stock handling for the exercised CB names."""
            loaded = set()
            stacked = (
                ("gate_up_proj", "w1", 0),
                ("gate_up_proj", "w3", 1),
                ("attn.fused_wqa_wkv", "attn.wq_a", 0),
                ("attn.fused_wqa_wkv", "attn.wkv", 1),
            )
            for raw_name, weight in weights:
                self.delegated_raw.append(raw_name)
                name = self._remap_dspark_name(raw_name)
                if name is None:
                    continue
                # Production DSpark rewrites source-checkpoint ``.scale`` to
                # the scale parameter spelling selected by the quant method.
                # Ordinary non-expert source FP8 uses ``weight_scale_inv``.
                if name.endswith(".scale"):
                    name = name.removesuffix(".scale") + \
                        ".weight_scale_inv"
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if ".experts." in name:
                    # Production's per-expert mapping does not claim a stacked
                    # Gridbook name. The wrapper must consume supported stacks;
                    # an unmatched one defers and is dropped here.
                    continue
                is_layer = name.startswith("model.layers.")
                for target, source, shard in stacked:
                    if not is_layer or source not in name:
                        continue
                    mapped = name.replace(source, target)
                    param = self._params[mapped]
                    param.weight_loader(param, weight, shard)
                    loaded.add(mapped)
                    break
                else:
                    param = self._params[name]
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is None:
                        param.data.copy_(weight.to(param.dtype))
                    else:
                        weight_loader(param, weight)
                    loaded.add(name)
            return loaded

    return _FakeDSpark


def test_dspark_uses_its_own_physical_to_registered_mapping():
    model = _make_dspark_cls()()
    rename = _dspark_rename(model)
    assert rename is not None
    assert rename("mtp.0.attn.wq_b.cb_qweight") == \
        "model.layers.0.attn.wq_b.cb_qweight"
    assert rename("mtp.2.markov_head.weight") == \
        "model.markov_head.weight"
    # The model returns None for confidence/non-MTP tensors. The interception
    # name stays physical so the original loader retains drop/ownership rules.
    assert rename("mtp.1.confidence_head.weight") == \
        "mtp.1.confidence_head.weight"
    assert rename("model.layers.43.attn.wq_b.cb_qweight") == \
        "model.layers.43.attn.wq_b.cb_qweight"
    assert _dspark_rename(object()) is None


def test_dspark_bridge_topology_matches_the_instantiated_model():
    class _Model:
        quant_config = SimpleNamespace(
            _dspark_target_bridge_topology=(43, 3)
        )
        config = SimpleNamespace(num_hidden_layers=43)
        model = SimpleNamespace(num_dspark_layers=3)

        @staticmethod
        def _remap_dspark_name(name):
            return name

    _validate_dspark_target_bridge_model(_Model())

    _Model.quant_config._dspark_target_bridge_topology = (42, 3)
    with pytest.raises(RuntimeError, match="does not match"):
        _validate_dspark_target_bridge_model(_Model())

    del _Model._remap_dspark_name
    with pytest.raises(RuntimeError, match="without callable"):
        _validate_dspark_target_bridge_model(_Model())


def test_all_dspark_stacked_expert_planes_load_and_fill():
    cls = _make_dspark_cls()
    install_toplevel_cb_expert_loader(cls)
    model = cls()
    checkpoint = []
    for stage in range(3):
        base = f"mtp.{stage}.ffn.experts"
        checkpoint.extend((
            (base + ".gate_up_proj.cb_qweight", torch.full(
                (E, 2 * INTER, BYTES), 10 + stage, dtype=torch.uint8
            )),
            (base + ".down_proj.cb_qweight", torch.full(
                (E, HID, BYTES), 20 + stage, dtype=torch.uint8
            )),
            (base + ".gate_up_proj.weight_scale", torch.full(
                (E, 2 * INTER), 30.0 + stage
            )),
            (base + ".down_proj.weight_scale", torch.full(
                (E, HID), 40.0 + stage
            )),
            (base + ".gate_up_proj.input_global_scale", torch.tensor(
                [50.0 + stage]
            )),
            (base + ".down_proj.input_global_scale", torch.tensor(
                [60.0 + stage]
            )),
        ))

    loaded = model.load_weights(iter(checkpoint))

    assert model.delegated_raw == []
    assert len(loaded) == 18
    for stage in range(3):
        routed = f"model.layers.{stage}.ffn.experts.routed_experts"
        assert torch.all(model._params[routed + ".w13_cb_qweight"]
                         == 10 + stage)
        assert torch.all(model._params[routed + ".w2_cb_qweight"]
                         == 20 + stage)
        assert torch.all(model._params[routed + ".w13_weight_scale"]
                         == 30 + stage)
        assert torch.all(model._params[routed + ".w2_weight_scale"]
                         == 40 + stage)
        assert torch.all(model._params[routed + ".w13_input_global_scale"]
                         == 50 + stage)
        assert torch.all(model._params[routed + ".w2_input_global_scale"]
                         == 60 + stage)
        for leaf in ("w13_cb_qweight", "w2_cb_qweight"):
            assert getattr(model._params[routed + "." + leaf],
                           CB_FILLED_ATTR, False) is True


def test_dspark_dense_direct_and_fused_cb_stay_stock_owned():
    cls = _make_dspark_cls()
    install_toplevel_cb_expert_loader(cls)
    model = cls()
    checkpoint = [
        ("mtp.0.attn.wq_b.cb_qweight",
         torch.full((HID, BYTES), 1, dtype=torch.uint8)),
        ("mtp.0.attn.wq_a.cb_qweight",
         torch.full((HID, BYTES), 2, dtype=torch.uint8)),
        ("mtp.0.attn.wkv.cb_qweight",
         torch.full((HID, BYTES), 3, dtype=torch.uint8)),
        ("mtp.0.ffn.shared_experts.w1.cb_qweight",
         torch.full((INTER, BYTES), 4, dtype=torch.uint8)),
        ("mtp.0.ffn.shared_experts.w3.cb_qweight",
         torch.full((INTER, BYTES), 5, dtype=torch.uint8)),
        ("mtp.0.ffn.shared_experts.w2.cb_qweight",
         torch.full((HID, BYTES), 6, dtype=torch.uint8)),
    ]

    loaded = model.load_weights(iter(checkpoint))

    # The wrapper resolves against registered names but delegates the original
    # physical names. This is the critical ownership boundary.
    assert model.delegated_raw == [name for name, _ in checkpoint]
    assert "model.layers.0.attn.wq_b.cb_qweight" in loaded
    assert torch.all(model._params[
        "model.layers.0.attn.wq_b.cb_qweight"] == 1)
    fused = model._params[
        "model.layers.0.attn.fused_wqa_wkv.cb_qweight"]
    assert torch.all(fused[:HID] == 2)
    assert torch.all(fused[HID:] == 3)
    shared = model._params[
        "model.layers.0.ffn.shared_experts.gate_up_proj.cb_qweight"]
    assert torch.all(shared[:INTER] == 4)
    assert torch.all(shared[INTER:] == 5)
    assert torch.all(model._params[
        "model.layers.0.ffn.shared_experts.down_proj.cb_qweight"] == 6)


@pytest.mark.parametrize(
    ("physical_base", "registered_base"),
    (
        ("mtp.0.main_proj", "model.main_proj"),
        ("mtp.2.attn.wo_a", "model.layers.2.attn.wo_a"),
    ),
)
def test_dspark_source_physical_planes_load_gridbook_w8a16_params(
        monkeypatch, physical_base, registered_base):
    """The wrapper delegates raw names; DSpark remaps into W8A16 params."""

    # Instantiate the real Gridbook method's parameter contract without
    # importing a GPU-bound vLLM installation into this CPU loader test.
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.linear", linear
    )
    parameter = types.ModuleType("vllm.model_executor.parameter")

    class _StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **_kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kwargs):
            del data
            for name, value in kwargs.items():
                setattr(self, name, value)

    parameter.ModelWeightParameter = _StubParameter
    parameter.BlockQuantScaleParameter = _StubParameter
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.parameter", parameter
    )

    from gridbook.fp8_source_w8a16 import (
        WIRE_FP8_BLOCK128,
        build_fp8_source_w8a16_method,
    )

    def copy_loader(destination, incoming, *_args, **_kwargs):
        destination.data.copy_(incoming)

    owner = torch.nn.Module()
    method = build_fp8_source_w8a16_method(WIRE_FP8_BLOCK128)
    method.create_weights(
        owner,
        input_size_per_partition=4,
        output_partition_sizes=[4],
        input_size=4,
        output_size=4,
        params_dtype=torch.bfloat16,
        weight_loader=copy_loader,
    )
    assert owner.weight.dtype == torch.float8_e4m3fn
    assert owner.weight_scale_inv.dtype == torch.float8_e8m0fnu

    cls = _make_dspark_cls()
    install_toplevel_cb_expert_loader(cls)
    model = cls()
    model._params[registered_base + ".weight"] = owner.weight
    model._params[
        registered_base + ".weight_scale_inv"
    ] = owner.weight_scale_inv

    source_weight = torch.empty_like(owner.weight)
    source_weight.view(torch.uint8).copy_(
        torch.arange(source_weight.numel(), dtype=torch.uint8).reshape(
            source_weight.shape
        )
    )
    source_scale = torch.empty_like(owner.weight_scale_inv)
    source_scale.view(torch.uint8).fill_(127)
    checkpoint = [
        (physical_base + ".weight", source_weight),
        (physical_base + ".scale", source_scale),
    ]

    loaded = model.load_weights(iter(checkpoint))

    assert model.delegated_raw == [name for name, _ in checkpoint]
    assert loaded == {
        registered_base + ".weight",
        registered_base + ".weight_scale_inv",
    }
    assert torch.equal(
        owner.weight.view(torch.uint8), source_weight.view(torch.uint8)
    )
    assert torch.equal(
        owner.weight_scale_inv.view(torch.uint8),
        source_scale.view(torch.uint8),
    )


def test_dspark_mixed_fusion_resolves_registered_names_without_delegation():
    cls = _make_dspark_cls(include_mixed=True)
    install_toplevel_cb_expert_loader(cls)
    model = cls()
    checkpoint = [
        ("mtp.1.attn.wq_a.cb_qweight",
         torch.full((HID, BYTES), 7, dtype=torch.uint8)),
        ("mtp.1.attn.wkv.cb_qweight",
         torch.full((HID, BYTES), 8, dtype=torch.uint8)),
    ]

    loaded = model.load_weights(iter(checkpoint))

    assert model.delegated_raw == []
    group = "model.layers.1.attn.fused_wqa_wkv"
    assert torch.all(model._params[
        group + "._gridbook_mixed_roles.0.cb_qweight"] == 7)
    assert torch.all(model._params[
        group + "._gridbook_mixed_roles.1.cb_qweight"] == 8)
    assert loaded == {
        group + "._gridbook_mixed_roles.0.cb_qweight",
        group + "._gridbook_mixed_roles.1.cb_qweight",
    }


def test_dspark_mixed_alias_is_structurally_44_to_1_and_fails_on_mutation():
    construction_group = "model.layers.44.attn.fused_wqa_wkv"
    construction_source = "model.layers.44.attn.wq_a"
    registered_param = (
        "model.layers.1.attn.fused_wqa_wkv."
        "_gridbook_mixed_roles.0.cb_qweight"
    )
    assert _registered_mixed_source(
        registered_param, construction_group, construction_source
    ) == "model.layers.1.attn.wq_a"

    # Mutating the construction source to another stage cannot silently route
    # through the layer-1 carrier: the group/source sibling proof rejects it.
    with pytest.raises(ValueError, match="not a sibling"):
        _registered_mixed_source(
            registered_param,
            construction_group,
            "model.layers.45.attn.wq_a",
        )
    # Mutating the registered fused leaf likewise invalidates the structural
    # join instead of inventing a layer-offset rule.
    with pytest.raises(ValueError, match="incompatible"):
        _registered_mixed_source(
            registered_param.replace("fused_wqa_wkv", "other_fusion"),
            construction_group,
            construction_source,
        )


def test_dspark_unmatched_names_defer_to_stock_filter_or_failure():
    cls = _make_dspark_cls()
    install_toplevel_cb_expert_loader(cls)
    model = cls()

    # Non-MTP and confidence names are stock-filtered; an unmatched stacked
    # expert is likewise deferred to DSpark's per-expert branch and dropped.
    names = [
        "model.layers.7.attn.wq_b.cb_qweight",
        "mtp.1.confidence_head.weight",
        "mtp.9.ffn.experts.gate_up_proj.cb_qweight",
    ]
    assert model.load_weights(iter((name, torch.zeros(1))
                                   for name in names)) == set()
    assert model.delegated_raw == names

    # An unmatched ordinary dense tensor follows production's direct
    # params_dict lookup and therefore fails closed in the stock loader.
    with pytest.raises(KeyError, match="model.layers.9.attn.wq_b"):
        model.load_weights(iter([
            ("mtp.9.attn.wq_b.cb_qweight",
             torch.zeros(HID, BYTES, dtype=torch.uint8)),
        ]))


def test_dspark_quant_config_resolves_construction_layers_not_registered_ones():
    pytest.importorskip("vllm")
    from gridbook.config import PrismaQuantConfig

    scheme = {
        "grid": "fp4",
        "mode": "product",
        "k": 15,
        "n_sub": 2,
        "type_size": 69,
        "group_size": 16,
        "vec_dim": 8,
        "scale_coding": {"kind": "two_tier"},
        "codebook_ref": ["cb0", "cb1"],
    }
    targets = []
    for layer in range(43, 46):
        targets.extend((
            f"model.layers.{layer}.attn.wq_b",
            f"model.layers.{layer}.ffn.experts.gate_up_proj",
            f"model.layers.{layer}.ffn.experts.down_proj",
        ))
    config = PrismaQuantConfig.from_config({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": {
            "dspark": {"scheme": scheme, "targets": targets},
        },
        "ignore": [],
    })
    config._ensure_resolved()

    for layer in range(43, 46):
        assert config._scheme_for_prefix(
            f"model.layers.{layer}.attn.wq_b"
        ) is not None
        assert config._moe_scheme_for_prefix(
            f"model.layers.{layer}.ffn.experts"
        ) is not None
    assert config._scheme_for_prefix("model.layers.0.attn.wq_b") is None
    assert config._moe_scheme_for_prefix(
        "model.layers.0.ffn.experts"
    ) is None


def test_real_vllm_dspark_entrypoint_is_registered_and_structural():
    pytest.importorskip("vllm")
    from gridbook.plugin import _install_on_module_classes

    module_path = "vllm.models.deepseek_v4.nvidia.dspark"
    assert module_path in load_runtime_contract()[
        "producer_profiles"
    ]["top_level_loader_modules"]
    cls = importlib.import_module(module_path).DSparkDeepseekV4ForCausalLM
    assert cls.__module__ == module_path
    assert callable(getattr(cls, "_remap_dspark_name", None))

    _install_on_module_classes(module_path)
    assert cls.__dict__.get("_pq_cb_wrapped") is True
    assert getattr(cls.load_weights, "_pq_cb_wrapper", False) is True
