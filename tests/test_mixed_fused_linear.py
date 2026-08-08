"""Heterogeneous dense fusion: resolver, carrier layout, load and apply.

These tests pin the seam vLLM exposes: one merged module, ordered checkpoint
siblings.  Gridbook must keep the siblings' existing methods independent and
only concatenate their outputs; a routed-expert format elsewhere in the same
decoder layer is unrelated.
"""
from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("vllm")

from vllm.model_executor.layers.linear import LinearBase  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
)

from gridbook.config import PrismaQuantConfig  # noqa: E402
from gridbook.linear import PrismaQuantCBLinearMethod  # noqa: E402
from gridbook.mixed_linear import MixedFusedLinearMethod  # noqa: E402
from gridbook.moe_toplevel_loader import (  # noqa: E402
    MIXED_FUSED_LOADER_ABI,
    _MixedFusedTransactions,
    install_toplevel_cb_expert_loader,
    mixed_fused_loader_active,
)
from gridbook.mxfp8_dense_lane import (  # noqa: E402
    WIRE_FP8_BLOCK128,
    build_mxfp8_dense_method,
)


FP8_BLOCK = "fp8_e4m3_ue8m0_block128"
MXFP4 = "mxfp4_e2m1_ue8m0_g32"


@pytest.fixture(autouse=True)
def _single_tp_parameter_context(monkeypatch):
    """vLLM parameter constructors normally run after TP initialization."""
    import vllm.model_executor.parameter as parameter
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size",
                        lambda: 1, raising=False)


def _scheme(grid: str, k: int) -> dict:
    if grid == "fp8":
        return {
            "grid": "fp8", "mode": "product", "k": k, "n_sub": 4,
            "type_size": 4 * k, "group_size": 0, "vec_dim": 8,
            "codebook_ref": [f"cb.fp8.k{k}.sub{i}" for i in range(4)],
        }
    return {
        "grid": "fp4", "mode": "product", "k": k, "n_sub": 2,
        "type_size": 4 * k + 9, "group_size": 16, "vec_dim": 8,
        "scale_coding": {"kind": "two_tier"},
        "codebook_ref": [f"cb.fp4.k{k}.sub{i}" for i in range(2)],
    }


K37 = _scheme("fp8", 37)
K38 = _scheme("fp8", 38)
K12 = _scheme("fp4", 12)
K12_CONTRACTED = {**K12, "activation_contract": "nvfp4_w4a4"}


def _mixed_config() -> PrismaQuantConfig:
    cfg = {
        "quant_method": "gridbook", "format": "mixed-precision",
        "layout_version": 2, "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "k37": {"format": "FP8_CB_K37", "scheme": K37,
                    "targets": [
                        "model.layers.14.ffn.shared_experts.w1",
                    ]},
            "k38": {"format": "FP8_CB_K38", "scheme": K38,
                    "targets": [
                        "model.layers.5.ffn.shared_experts.w3",
                        "model.layers.14.ffn.shared_experts.w3",
                        "model.layers.39.ffn.shared_experts.w1",
                        "model.layers.2.ffn.shared_experts.w1",
                        "model.layers.2.ffn.shared_experts.w3",
                        "model.layers.0.custom.right",
                    ]},
            "k12": {"format": "NVFP4_CB_K12", "scheme": K12,
                    "targets": [
                        "model.layers.39.ffn.shared_experts.w3",
                        "model.layers.0.attn.wq_a",
                        "model.layers.0.custom.left",
                        "model.layers.16.ffn.shared_experts.w1",
                        "model.layers.16.ffn.shared_experts.w3",
                    ]},
        },
        "ignore": ["lm_head"],
        "source_passthrough": {"version": 1, "units": {
            # Producer/live Transformers spelling; vLLM uses ffn+w1.
            "model.layers.5.mlp.shared_experts.gate_proj": FP8_BLOCK,
            # Producer/live Transformers spelling; vLLM uses attn.
            "model.layers.0.self_attn.wkv": FP8_BLOCK,
            # Homogeneous source fusion keeps the audited native merged path.
            "model.layers.1.mlp.shared_experts.gate_proj": FP8_BLOCK,
            "model.layers.1.mlp.shared_experts.up_proj": FP8_BLOCK,
            # Same decoder-layer routed stack is not a dense fusion member.
            "model.layers.39.mlp.experts": MXFP4,
        }},
    }
    result = PrismaQuantConfig.from_config(cfg)
    # Resolver-only fixture: no contracted FP4 activation payload is declared.
    result._ensure_resolved()
    return result


@pytest.mark.parametrize(("prefix", "expected"), [
    ("model.layers.5.ffn.shared_experts.gate_up_proj",
     (("source", None), ("cb", 38))),
    ("model.layers.14.ffn.shared_experts.gate_up_proj",
     (("cb", 37), ("cb", 38))),
    ("model.layers.39.ffn.shared_experts.gate_up_proj",
     (("cb", 38), ("cb", 12))),
    ("model.layers.0.attn.fused_wqa_wkv",
     (("cb", 12), ("source", None))),
])
def test_dsv4_mixed_fusions_resolve_every_role_in_vllm_order(prefix, expected):
    owners = _mixed_config().fused_role_owners(prefix)
    actual = tuple((owner.kind,
                    owner.payload.get("k") if owner.kind == "cb" else None)
                   for owner in owners)
    assert actual == expected
    # A heterogeneous fusion has no honest single scheme; importantly, this
    # introspection is no longer an export-union exception.
    assert _mixed_config()._scheme_for_prefix(prefix) is None


def test_packed_modules_mapping_order_is_authoritative():
    cfg = _mixed_config()
    cfg.packed_modules_mapping = {"merged": ["right", "left"]}
    owners = cfg.fused_role_owners("model.layers.0.custom.merged")
    assert [owner.target.rsplit(".", 1)[-1] for owner in owners] == [
        "right", "left"]
    assert [owner.payload["k"] for owner in owners] == [38, 12]


def test_partial_known_fusion_fails_instead_of_claiming_first_scheme():
    cfg = _mixed_config()
    prefix = "model.layers.14.ffn.shared_experts.gate_up_proj"
    del cfg.target_scheme["model.layers.14.ffn.shared_experts.w3"]
    assert cfg.fused_role_owners(prefix) == []
    assert cfg.incomplete_fused_roles(prefix) == [
        "model.layers.14.ffn.shared_experts.w3"]
    with pytest.raises(RuntimeError, match="partial explicit role ownership"):
        cfg.get_quant_method(LinearBase.__new__(LinearBase), prefix)


def test_homogeneous_source_fusion_and_routed_mxfp4_are_independent():
    cfg = _mixed_config()
    owners = cfg.fused_role_owners(
        "model.layers.1.ffn.shared_experts.gate_up_proj")
    assert [owner.payload.id for owner in owners] == [FP8_BLOCK, FP8_BLOCK]
    assert cfg._passthrough_format("model.layers.39.ffn.experts").id == MXFP4
    # The routed stack does not leak into the dense shared-expert owner set.
    shared = cfg.fused_role_owners(
        "model.layers.39.ffn.shared_experts.gate_up_proj")
    assert [owner.kind for owner in shared] == ["cb", "cb"]


def test_homogeneous_source_fusion_uses_native_merged_shard_loader():
    """w1/w3 weight + ``.scale`` fill one native block-FP8 fused param."""

    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.output_sizes = [128, 128]
    layer.output_size = 256
    layer.tp_size = 1
    layer.tp_rank = 0
    method = build_mxfp8_dense_method(WIRE_FP8_BLOCK128)
    method.create_weights(
        layer, 128, [128, 128], 128, 256, torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    assert layer.weight_block_size == [128, 128]
    for shard, value in ((0, 1), (1, 2)):
        weight = torch.full((128, 128), value,
                            dtype=torch.float8_e4m3fn)
        scale = torch.full((1, 1), value, dtype=torch.float8_e8m0fnu)
        layer.weight.weight_loader(layer.weight, weight, shard)
        layer.weight_scale_inv.weight_loader(
            layer.weight_scale_inv, scale, shard)
    assert torch.all(layer.weight[:128] == 1)
    assert torch.all(layer.weight[128:] == 2)
    assert torch.equal(
        layer.weight_scale_inv,
        torch.tensor([[1.0], [2.0]], dtype=torch.float8_e8m0fnu),
    )


def _parent() -> torch.nn.Module:
    return torch.nn.Module()


@pytest.mark.parametrize(("roles", "widths", "expected"), [
    (
        lambda cfg: [
            ("model.layers.5.ffn.shared_experts.w1",
             build_mxfp8_dense_method(WIRE_FP8_BLOCK128)),
            ("model.layers.5.ffn.shared_experts.w3",
             PrismaQuantCBLinearMethod(cfg, K38, "role.w3")),
        ],
        [2048, 2048],
        (((2048, 4096), (16, 32)), ((2048, 2432), (2048,))),
    ),
    (
        lambda cfg: [
            ("model.layers.14.ffn.shared_experts.w1",
             PrismaQuantCBLinearMethod(cfg, K37, "role.w1")),
            ("model.layers.14.ffn.shared_experts.w3",
             PrismaQuantCBLinearMethod(cfg, K38, "role.w3")),
        ],
        [2048, 2048],
        (((2048, 2368), (2048,)), ((2048, 2432), (2048,))),
    ),
    (
        lambda cfg: [
            ("model.layers.39.ffn.shared_experts.w1",
             PrismaQuantCBLinearMethod(cfg, K38, "role.w1")),
            ("model.layers.39.ffn.shared_experts.w3",
             PrismaQuantCBLinearMethod(cfg, K12_CONTRACTED, "role.w3")),
        ],
        [2048, 2048],
        (((2048, 2432), (2048,)), ((2048, 912), (1,))),
    ),
])
def test_each_mixed_class_keeps_role_native_parameter_shapes(roles, widths,
                                                              expected):
    cfg = types.SimpleNamespace()
    method = MixedFusedLinearMethod("model.layers.0.fused", roles(cfg))
    layer = _parent()
    method.create_weights(layer, 4096, widths, 4096, sum(widths),
                          torch.bfloat16)
    actual = []
    for carrier in layer._gridbook_mixed_roles:
        assert not hasattr(carrier, "quant_method")
        actual.append(tuple(tuple(param.shape) for param in
                            carrier.parameters(recurse=False)))
    assert tuple(actual) == expected


class _FakeRoleMethod:
    def __init__(self, value: float, planes: tuple[str, ...] = ("weight",)):
        self.value = value
        self.planes = planes
        self.processed = 0

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, *args, **kwargs):
        width = output_partition_sizes[0]
        for plane in self.planes:
            layer.register_parameter(
                plane,
                torch.nn.Parameter(torch.full(
                    (width, input_size_per_partition), -99.0),
                    requires_grad=False),
            )

    def process_weights_after_loading(self, layer):
        self.processed += 1

    def apply(self, layer, x, bias=None):
        width = next(layer.parameters()).shape[0]
        return torch.full((*x.shape[:-1], width), self.value,
                          dtype=x.dtype, device=x.device)


def test_get_quant_method_composes_only_semantically_different_roles():
    source = _FakeRoleMethod(3.0)

    def exercise(prefix, *, loader=True):
        cfg = _mixed_config()
        cfg._delegate_passthrough = types.MethodType(
            lambda self, layer, role_prefix, fmt: source, cfg)
        cfg._has_mixed_fused_loader = types.MethodType(
            lambda self: loader, cfg)
        return cfg.get_quant_method(LinearBase.__new__(LinearBase), prefix)

    for prefix in (
        "model.layers.5.ffn.shared_experts.gate_up_proj",   # source + CB
        "model.layers.14.ffn.shared_experts.gate_up_proj",  # K37 + K38
        "model.layers.39.ffn.shared_experts.gate_up_proj",  # FP8 + FP4
        "model.layers.0.attn.fused_wqa_wkv",                # generic DSV4 merge
    ):
        assert isinstance(exercise(prefix), MixedFusedLinearMethod)

    # Honest homogeneous paths retain the existing merged parameter/method.
    assert isinstance(exercise(
        "model.layers.2.ffn.shared_experts.gate_up_proj"),
        PrismaQuantCBLinearMethod)
    assert exercise(
        "model.layers.1.ffn.shared_experts.gate_up_proj") is source

    with pytest.raises(RuntimeError, match="mixed-fused loader ABI 1"):
        exercise("model.layers.14.ffn.shared_experts.gate_up_proj",
                 loader=False)


def test_same_scheme_gate_roles_keep_different_activation_scalars():
    """A shared scheme does not imply one physical activation contract.

    DeepSeek-V4 has same-rung gate/up pairs whose serialized static scalars
    differ. A single merged CB method rejects them as non-identical; separate
    carriers must validate each role against its own attested scalar instead.
    """

    cfg = _mixed_config()
    prefix = "model.layers.16.ffn.shared_experts.gate_up_proj"
    targets = [
        "model.layers.16.ffn.shared_experts.w1",
        "model.layers.16.ffn.shared_experts.w3",
    ]
    for target in targets:
        cfg.target_scheme[target] = K12_CONTRACTED
        cfg._target_physical_name[target] = target
    cfg._nvfp4_activation_contract = {"synthetic_test_contract": True}
    expected_scales = dict(zip(targets, (28.8451, 29.9707)))
    cfg._nvfp4_activation_scales = expected_scales
    cfg._has_mixed_fused_loader = types.MethodType(lambda self: True, cfg)

    assert cfg._scheme_for_prefix(prefix) is None
    method = cfg.get_quant_method(LinearBase.__new__(LinearBase), prefix)
    assert isinstance(method, MixedFusedLinearMethod)
    layer = _parent()
    method.create_weights(
        layer, 256, [8, 8], 256, 16, torch.bfloat16,
    )
    params = dict(layer.named_parameters())
    tx = _MixedFusedTransactions(params)
    for target, scalar in expected_scales.items():
        assert tx.stage(
            target + ".cb_qweight",
            torch.zeros(8, K12_CONTRACTED["type_size"], dtype=torch.uint8),
        ) == ()
        committed = tx.stage(
            target + ".input_global_scale",
            torch.tensor([scalar], dtype=torch.float32),
        )
    assert len(committed) == len(params)
    tx.finish()

    actual = []
    for target, carrier, child in zip(
            targets, layer._gridbook_mixed_roles,
            layer._gridbook_mixed_methods):
        child._finalize_static_activation_scale(carrier, [target])
        actual.append(carrier._cb_fp4_input_global_scale_f32)
    assert actual == pytest.approx([28.8451, 29.9707])
    assert actual[0] != actual[1]


def test_construction_gate_is_exact_inherited_and_reset_on_error():
    observations = []
    dispatches = []
    cfg = _mixed_config()
    mixed_prefix = "model.layers.14.ffn.shared_experts.gate_up_proj"
    with pytest.raises(RuntimeError, match="mixed-fused loader ABI 1"):
        cfg.get_quant_method(LinearBase.__new__(LinearBase), mixed_prefix)

    class Target:
        def __init__(self, *, fail=False):
            observations.append(mixed_fused_loader_active())
            dispatches.append(isinstance(cfg.get_quant_method(
                LinearBase.__new__(LinearBase), mixed_prefix
            ), MixedFusedLinearMethod))
            if fail:
                raise ValueError("boom")

        def named_parameters(self):
            return ()

        def load_weights(self, weights):
            return {name for name, _ in weights}

    install_toplevel_cb_expert_loader(Target)
    assert getattr(Target.load_weights,
                   "_gridbook_mixed_fused_loader_abi") \
        == MIXED_FUSED_LOADER_ABI
    assert not mixed_fused_loader_active()
    Target()
    assert observations == [True]
    assert dispatches == [True]
    assert not mixed_fused_loader_active()

    class Inherited(Target):
        pass

    Inherited()
    assert observations == [True, True]
    assert dispatches == [True, True]
    assert not mixed_fused_loader_active()

    with pytest.raises(ValueError, match="boom"):
        Target(fail=True)
    assert observations == [True, True, True]
    assert dispatches == [True, True, True]
    assert not mixed_fused_loader_active()

    class Override(Target):
        def load_weights(self, weights):
            return set()

    with pytest.raises(RuntimeError, match="effective load_weights"):
        Override()
    assert observations == [True, True, True]
    assert dispatches == [True, True, True]
    assert not mixed_fused_loader_active()


def test_transaction_commits_all_role_planes_together_and_apply_concats():
    left = _FakeRoleMethod(1.0, ("weight", "weight_scale"))
    right = _FakeRoleMethod(2.0, ("weight",))
    method = MixedFusedLinearMethod("model.layers.0.mod.merged", [
        ("model.layers.0.mod.left", left),
        ("model.layers.0.mod.right", right),
    ])
    layer = _parent()
    method.create_weights(layer, 4, [2, 3], 4, 5, torch.bfloat16)
    params = dict(layer.named_parameters())
    tx = _MixedFusedTransactions(params)

    assert tx.stage("model.layers.0.mod.left.weight",
                    torch.full((2, 4), 10.0)) == ()
    assert tx.stage("model.layers.0.mod.left.weight_scale",
                    torch.full((2, 4), 11.0)) == ()
    # Nothing mutates before the final plane validates.
    assert all(torch.all(param == -99) for param in params.values())
    committed = tx.stage("model.layers.0.mod.right.weight",
                         torch.full((3, 4), 20.0))
    assert set(committed) == set(params)
    tx.finish()
    method.process_weights_after_loading(layer)
    assert left.processed == right.processed == 1

    x = torch.zeros(2, 4)
    bias = torch.arange(5, dtype=x.dtype)
    expected = torch.tensor([[1, 1, 2, 2, 2]], dtype=x.dtype).repeat(2, 1)
    assert torch.equal(method.apply(layer, x), expected)
    assert torch.equal(method.apply(layer, x, bias), expected + bias)


def test_transaction_rejects_missing_plane_without_partial_copy():
    method = MixedFusedLinearMethod("model.layers.0.mod.merged", [
        ("model.layers.0.mod.left", _FakeRoleMethod(1.0)),
        ("model.layers.0.mod.right", _FakeRoleMethod(2.0)),
    ])
    layer = _parent()
    method.create_weights(layer, 4, [2, 3], 4, 5, torch.bfloat16)
    params = dict(layer.named_parameters())
    tx = _MixedFusedTransactions(params)
    tx.stage("model.layers.0.mod.left.weight", torch.ones(2, 4))
    with pytest.raises(RuntimeError, match="incomplete mixed fused"):
        tx.finish()
    assert all(torch.all(param == -99) for param in params.values())
    with pytest.raises(RuntimeError, match="missing checkpoint planes"):
        method.process_weights_after_loading(layer)


def test_transaction_rejects_unexpected_dtype_without_partial_copy():
    method = MixedFusedLinearMethod("model.layers.0.mod.merged", [
        ("model.layers.0.mod.left", _FakeRoleMethod(1.0)),
        ("model.layers.0.mod.right", _FakeRoleMethod(2.0)),
    ])
    layer = _parent()
    method.create_weights(layer, 4, [2, 3], 4, 5, torch.bfloat16)
    params = dict(layer.named_parameters())
    tx = _MixedFusedTransactions(params)
    tx.stage("model.layers.0.mod.left.weight",
             torch.ones(2, 4, dtype=torch.int32))
    with pytest.raises(ValueError, match="checkpoint dtype"):
        tx.stage("model.layers.0.mod.right.weight", torch.ones(3, 4))
    assert all(torch.all(param == -99) for param in params.values())


class _Mapper:
    def _map_name(self, name):
        if name.startswith("layers."):
            name = "model." + name
        if name.endswith(".scale"):
            name = name[:-len(".scale")] + ".weight_scale_inv"
        return name


def test_top_level_wrapper_routes_real_dsv4_wire_names_transactionally():
    method = MixedFusedLinearMethod(
        "model.layers.5.ffn.shared_experts.gate_up_proj", [
            ("model.layers.5.ffn.shared_experts.w1",
             _FakeRoleMethod(1.0, ("weight", "weight_scale_inv"))),
            ("model.layers.5.ffn.shared_experts.w3",
             _FakeRoleMethod(2.0, ("cb_qweight", "weight_scale"))),
        ])
    fused = _parent()
    method.create_weights(fused, 4, [2, 2], 4, 4, torch.bfloat16)

    class _FakeDSV4:
        def __init__(self):
            self.fused = fused
            self.hf_to_vllm_mapper = _Mapper()
            self.delegated = []

        def named_parameters(self):
            prefix = ("model.layers.5.ffn.shared_experts.gate_up_proj.")
            return [(prefix + name, param)
                    for name, param in self.fused.named_parameters()]

        def load_weights(self, weights):
            for name, _ in weights:
                self.delegated.append(name)
            return set()

    install_toplevel_cb_expert_loader(_FakeDSV4)
    model = _FakeDSV4()
    loaded = model.load_weights(iter([
        ("layers.5.ffn.shared_experts.w1.weight", torch.full((2, 4), 1.0)),
        ("layers.5.ffn.shared_experts.w1.scale", torch.full((2, 4), 2.0)),
        ("layers.5.ffn.shared_experts.w3.cb_qweight", torch.full((2, 4), 3.0)),
        ("layers.5.ffn.shared_experts.w3.weight_scale", torch.full((2, 4), 4.0)),
    ]))
    assert model.delegated == []
    assert loaded == {name for name, _ in model.named_parameters()}
    values = [float(param.flatten()[0]) for _, param in model.named_parameters()]
    assert values == [1.0, 2.0, 3.0, 4.0]
