"""Per-layer dispatch for a MIXED artifact: CB units, passthrough units, stock.

vLLM-only (``PrismaQuantConfig`` resolves against real vLLM layer classes), so
skip-guarded like ``tests/test_delegation.py`` — run in the gridbook:test
container:

  docker run --rm --gpus all -v /home/rob/gb-mxfp4-wt:/w --entrypoint bash \
    gridbook:test -c 'cd /w; PYTHONPATH=/w python3 -m pytest \
    tests/test_source_passthrough_dispatch.py -q'

No large weights are involved: layers are allocated with ``__new__`` so they
satisfy the ``isinstance`` dispatch without building parameters, and the vLLM
method construction itself is stubbed. What is pinned is which BRANCH of
``get_quant_method`` a declaration selects, and that every guard on the
passthrough branch fails closed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from vllm.model_executor.layers.fused_moe import RoutedExperts  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    LinearBase, UnquantizedLinearMethod,
)

import gridbook.config as config_mod  # noqa: E402
from gridbook.config import PrismaQuantConfig  # noqa: E402
from gridbook.delegated_preflight import DelegatedBackendError  # noqa: E402
from gridbook.linear import PrismaQuantCBLinearMethod  # noqa: E402
from gridbook.source_passthrough import (  # noqa: E402
    FORMATS, SCHEMA_KEY, SourcePassthroughError,
)


MXFP4 = "mxfp4_e2m1_ue8m0_g32"
CB_TARGET = "model.layers.0.mlp.down_proj"
PT_EXPERTS = "model.layers.7.mlp.experts"

_CB_SCHEME = {
    "grid": "fp8", "mode": "product", "k": 44, "n_sub": 4, "type_size": 176,
    "group_size": 0, "vec_dim": 8, "codebook_group": "down_proj",
    "codebook_source": "learned",
    "codebook_ref": ["cb_codebook.down_proj.FP8_CB_K44.sub0"],
}


def _artifact(passthrough=None, *, extra_groups=None):
    cfg = {
        "quant_method": "prismaquant", "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "group_cb": {"format": "FP8_CB_K44", "targets": [CB_TARGET],
                         "scheme": dict(_CB_SCHEME)},
        },
        "ignore": ["lm_head"],
    }
    if extra_groups:
        cfg["config_groups"].update(extra_groups)
    if passthrough is not None:
        cfg[SCHEMA_KEY] = passthrough
    return cfg


class _StubMethod:
    """Mirrors the two attributes vLLM's Mxfp4MoEMethod exposes post-selection."""

    def __init__(self, backend_label, experts_cls):
        self.mxfp4_backend = backend_label
        self.experts_cls = experts_cls


def _experts_cls(name, module):
    cls = type(name, (), {})
    cls.__module__ = module
    cls.__qualname__ = name
    return cls


_MARLIN = _experts_cls(
    "MarlinExperts", "vllm.model_executor.layers.fused_moe.experts.marlin_moe")
_DEEPGEMM = _experts_cls(
    "DeepGemmFP4Experts",
    "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe")
_MRO_BASE = _experts_cls(
    "TritonExperts", "vllm.model_executor.layers.fused_moe.experts.triton_moe")
_EMULATION = type("OCP_MXQuantizationEmulationExperts", (_MRO_BASE,), {})
_EMULATION.__module__ = (
    "vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe")


@pytest.fixture
def native_marlin(monkeypatch):
    """Pretend we are on the audited device and vLLM resolved the audited rung."""
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda *a, **k: (12, 1))
    method = _StubMethod("Mxfp4MoeBackend.MARLIN", _MARLIN)
    monkeypatch.setattr(config_mod, "_build_passthrough_method",
                        lambda fmt, layer, prefix: method)
    return method


def _linear():
    return LinearBase.__new__(LinearBase)


def _experts():
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = object()
    return layer


# --- legacy artifacts are untouched ------------------------------------------


def test_legacy_artifact_has_no_passthrough_and_dispatches_as_before():
    c = PrismaQuantConfig.from_config(_artifact())
    c._ensure_resolved()
    assert c._passthrough_units == {}
    assert isinstance(c.get_quant_method(_linear(), CB_TARGET),
                      PrismaQuantCBLinearMethod)
    assert isinstance(c.get_quant_method(_linear(), "lm_head"),
                      UnquantizedLinearMethod)


# --- declaration parsing at resolve time -------------------------------------


def test_declared_units_are_resolved_and_typed():
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    c._ensure_resolved()
    assert {k: v.id for k, v in c._passthrough_units.items()} == {
        PT_EXPERTS: MXFP4}


def test_unknown_declaration_value_refused_at_resolve():
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: "sorcery4"}}))
    with pytest.raises(SourcePassthroughError, match="unknown source-passthrough"):
        c._ensure_resolved()


def test_unknown_schema_version_refused_at_resolve():
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 99, "units": {PT_EXPERTS: MXFP4}}))
    with pytest.raises(SourcePassthroughError, match="version 99"):
        c._ensure_resolved()


def test_unit_declared_both_cb_and_passthrough_refused():
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {CB_TARGET: MXFP4}}))
    with pytest.raises(SourcePassthroughError, match="BOTH as a CB target"):
        c._ensure_resolved()


def test_passthrough_units_are_hidden_from_delegated_compressed_tensors():
    """A stock group must not be able to claim a unit Gridbook owns."""
    stock = {"group_fp8": {
        "format": "float-quantized",
        "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                    "symmetric": True, "dynamic": False},
        "input_activations": {"num_bits": 8, "type": "float",
                              "strategy": "token", "symmetric": True,
                              "dynamic": True},
        "targets": ["re:.*mlp.*"]}}
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}, extra_groups=stock))
    c._ensure_resolved()
    assert c.ct_config is not None
    assert PT_EXPERTS in c.ct_config.ignore
    assert CB_TARGET in c.ct_config.ignore


# --- the dispatch branch -----------------------------------------------------


def test_passthrough_moe_unit_gets_the_native_vllm_method(native_marlin):
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    assert c.get_quant_method(_experts(), PT_EXPERTS) is native_marlin


def test_cb_units_are_unaffected_by_a_passthrough_declaration(native_marlin):
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    assert isinstance(c.get_quant_method(_linear(), CB_TARGET),
                      PrismaQuantCBLinearMethod)


def test_undeclared_moe_unit_is_not_passed_through(native_marlin):
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    # No CB scheme, no declaration, no stock CT config -> nothing to serve.
    assert c.get_quant_method(_experts(), "model.layers.8.mlp.experts") is None


# --- fail-closed guards ------------------------------------------------------


def test_unaudited_device_is_a_load_time_refusal(monkeypatch):
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda *a, **k: (9, 0))
    monkeypatch.setattr(
        config_mod, "_build_passthrough_method",
        lambda *a, **k: pytest.fail("device gate must run before construction"))
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    with pytest.raises(SourcePassthroughError, match="sm_90"):
        c.get_quant_method(_experts(), PT_EXPERTS)


def test_unreadable_device_capability_is_a_refusal(monkeypatch):
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda *a, **k: None)
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    with pytest.raises(SourcePassthroughError, match="could not read"):
        c.get_quant_method(_experts(), PT_EXPERTS)


@pytest.mark.parametrize("backend,experts_cls,expected", [
    ("Mxfp4MoeBackend.DEEPGEMM_MXFP4", _DEEPGEMM, "Unknown SF transformation"),
    ("Mxfp4MoeBackend.EMULATION", _EMULATION, "Triton-backed"),
    ("Mxfp4MoeBackend.HUMMING",
     _experts_cls("HummingExperts",
                  "vllm.model_executor.layers.fused_moe.experts.humming_moe"),
     "not in Gridbook's audited set"),
])
def test_preflight_refuses_non_audited_backends(monkeypatch, backend,
                                                experts_cls, expected):
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda *a, **k: (12, 1))
    monkeypatch.setattr(
        config_mod, "_build_passthrough_method",
        lambda fmt, layer, prefix: _StubMethod(backend, experts_cls))
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {PT_EXPERTS: MXFP4}}))
    with pytest.raises(DelegatedBackendError, match=expected):
        c.get_quant_method(_experts(), PT_EXPERTS)


def test_passthrough_linear_unit_dispatches_through_the_same_guard(monkeypatch):
    """Format-general: the branch is not MoE-specific."""
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda *a, **k: (12, 1))
    method = _StubMethod("Mxfp4MoeBackend.MARLIN", _MARLIN)
    monkeypatch.setattr(config_mod, "_build_passthrough_method",
                        lambda fmt, layer, prefix: method)
    target = "model.layers.4.self_attn.q_proj"
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {target: MXFP4}}))
    assert c.get_quant_method(_linear(), target) is method


def test_fp8_block_broken_vllm_rung_still_named_and_refused(monkeypatch):
    """A measured-broken vLLM rung refuses BY NAME even now that the format's
    audited route is Gridbook's own MXFP8 lane: if dispatch ever resolves one
    of the broken classes, the operator gets the diagnosed symptom rather
    than a generic UNKNOWN."""
    monkeypatch.setattr(config_mod, "_live_device_capability",
                        lambda *a, **k: (12, 1))
    monkeypatch.setattr(
        config_mod, "_build_passthrough_method",
        lambda fmt, layer, prefix: _StubMethod("CutlassFp8BlockScaledMMKernel",
                                               _MARLIN))
    target = "model.layers.4.self_attn.q_proj"
    c = PrismaQuantConfig.from_config(_artifact(
        {"version": 1, "units": {target: "fp8_e4m3_ue8m0_block128"}}))
    with pytest.raises(DelegatedBackendError):
        c.get_quant_method(_linear(), target)


def test_registry_covers_the_ship_critical_format():
    assert MXFP4 in FORMATS
    assert FORMATS[MXFP4].audited_backends
