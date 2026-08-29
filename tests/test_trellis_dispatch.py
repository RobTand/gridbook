"""Config-level dispatch: a trellis target in a checkpoint reaches a lane.

WHAT THIS CLOSES.  Both trellis lanes existed as ``LinearMethod`` classes that
nothing constructed. ``get_quant_method`` had no arm for them, no config parsed
a trellis scheme, and their finalize hooks required a prepared wire that no
caller produced -- so "Gridbook can serve a trellis wire" was a statement about
three files that never met. These tests are the seam: a sidecar goes in, a lane
method comes out, and every way of getting it wrong is refused by name.

STILL NOT ESTABLISHED, and the file says so rather than letting a green run
imply it: vLLM is STUBBED here (the build venv has none), so this is the
CONTRACT between our config and vLLM's dispatch protocol, not evidence that
vLLM loads a trellis artifact. That owes a container run.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook.trellis import TCQ_E2M1_R256, TCQ_E4M3_R256   # noqa: E402
from gridbook.trellis_e2m1_lane import (                    # noqa: E402
    TRELLIS_E2M1_FLAG, TRELLIS_E2M1_MODE_ENV)
from gridbook.trellis_e4m3_lane import (                    # noqa: E402
    TRELLIS_E4M3_FLAG, TRELLIS_E4M3_MODE_ENV)


def _install_vllm_stubs():
    def module(name):
        value = types.ModuleType(name)
        sys.modules[name] = value
        return value

    module("vllm")
    module("vllm.model_executor")
    module("vllm.model_executor.layers")
    module("vllm.model_executor.layers.quantization")
    linear = module("vllm.model_executor.layers.linear")
    linear.LinearBase = type("LinearBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    parameter = module("vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kwargs):
            pass

    parameter.ModelWeightParameter = StubParameter
    parameter.BasevLLMParameter = StubParameter
    parameter.ChannelQuantScaleParameter = StubParameter
    parameter.PerTensorScaleParameter = StubParameter
    base = module("vllm.model_executor.layers.quantization.base_config")
    base.QuantizationConfig = type("QuantizationConfig", (), {})
    base.QuantizeMethodBase = object
    embedding = module("vllm.model_executor.layers.vocab_parallel_embedding")
    embedding.UnquantizedEmbeddingMethod = type("UEM", (), {})
    embedding.VocabParallelEmbedding = type("VPE", (), {})
    embedding.ParallelLMHead = type(
        "ParallelLMHead", (embedding.VocabParallelEmbedding,), {})
    fused = module("vllm.model_executor.layers.fused_moe")
    fused.RoutedExperts = type("RoutedExperts", (), {})


@pytest.fixture(scope="module", autouse=True)
def runtime_modules(isolated_gridbook_runtime_imports):
    del isolated_gridbook_runtime_imports
    _install_vllm_stubs()
    from gridbook.config import PrismaQuantConfig
    globals()["PrismaQuantConfig"] = PrismaQuantConfig


@pytest.fixture(autouse=True)
def _fresh_flags(monkeypatch):
    from gridbook import lane_select
    for flag in (TRELLIS_E2M1_FLAG, TRELLIS_E4M3_FLAG):
        lane_select.reset_for_tests(flag)
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.delenv(TRELLIS_E2M1_MODE_ENV, raising=False)
    monkeypatch.delenv(TRELLIS_E4M3_MODE_ENV, raising=False)
    yield
    for flag in (TRELLIS_E2M1_FLAG, TRELLIS_E4M3_FLAG):
        lane_select.reset_for_tests(flag)


TARGET = "model.layers.0.self_attn.o_proj"


def _scheme(family=TCQ_E4M3_R256, **over):
    base = {
        TCQ_E4M3_R256: {"family": TCQ_E4M3_R256, "body_rate_q256": 1152},
        TCQ_E2M1_R256: {"family": TCQ_E2M1_R256, "body_rate_q256": 512},
    }[family]
    return {**base, "rows": 128, "columns": 512, "wire_bytes": 65536, **over}


def _config(scheme=None, targets=(TARGET,)):
    return {
        "quant_method": "gridbook",
        "format": "mixed-precision",
        "config_groups": {
            "trellis": {
                "format": "TRELLIS",
                "targets": list(targets),
                "scheme": _scheme() if scheme is None else scheme,
            }
        },
        "ignore": [],
    }


def _resolved(cfg=None):
    config = PrismaQuantConfig.from_config(cfg or _config())
    config._ensure_resolved()
    return config


def _layer():
    from vllm.model_executor.layers.linear import LinearBase
    return object.__new__(LinearBase)


def _enable(monkeypatch, family=TCQ_E4M3_R256, mode="streamed"):
    from gridbook import lane_select
    flag, env = ((TRELLIS_E4M3_FLAG, TRELLIS_E4M3_MODE_ENV)
                 if family == TCQ_E4M3_R256
                 else (TRELLIS_E2M1_FLAG, TRELLIS_E2M1_MODE_ENV))
    lane_select.reset_for_tests(flag)
    monkeypatch.setenv(flag, "1")
    monkeypatch.setenv(env, mode)


# --- the seam itself -------------------------------------------------------

@pytest.mark.parametrize(
    "family,expected",
    [(TCQ_E4M3_R256, "TrellisE4M3LinearMethod"),
     (TCQ_E2M1_R256, "TrellisE2M1LinearMethod")],
)
def test_a_trellis_target_dispatches_to_its_lane(monkeypatch, family,
                                                 expected):
    """The whole point: a sidecar declaration reaches the right lane class.

    The class NAME is asserted because that is what a route probe reads off
    the constructed method -- the same string an ``emit_route`` histogram
    would carry.
    """
    _enable(monkeypatch, family)
    config = _resolved(_config(_scheme(family)))
    method = config.get_quant_method(_layer(), TARGET)
    assert type(method).__name__ == expected


def test_trellis_targets_are_kept_out_of_the_cb_map(monkeypatch):
    """Both vocabularies use the key "scheme"; only the discriminator differs.

    If a trellis scheme landed in ``target_scheme`` it would be claimed by CB
    fused-owner resolution, CB activation-contract validation and the CB
    device-capability gate -- three surfaces that assume the CB vocabulary and
    would each misread it.
    """
    config = _resolved()
    assert TARGET in config.target_trellis
    assert TARGET not in config.target_scheme
    assert TARGET not in config._cb_targets


def test_a_trellis_target_is_ignored_by_delegated_compressed_tensors(
        monkeypatch):
    """Ours means ours: stock compressed-tensors must not also claim the unit.

    A target both configs think they own is served by whichever arm runs
    first -- the silent-ownership failure CB targets are added to CT's ignore
    list to prevent. The delegated config is stubbed and its input dict
    captured, so this asserts the ignore list we HAND to CT rather than
    trusting a real CT to have honoured it.
    """
    stock = "model.layers.0.mlp.down_proj"
    cfg = _config()
    cfg["config_groups"]["stock_nvfp4"] = {
        "format": "nvfp4-pack-quantized",
        "targets": [stock],
        "weights": {"num_bits": 4, "type": "float",
                    "strategy": "tensor_group", "group_size": 16},
        "input_activations": {"num_bits": 4, "type": "float",
                              "strategy": "tensor_group", "group_size": 16,
                              "dynamic": "local"},
    }
    delegated: dict = {}
    pkg = "vllm.model_executor.layers.quantization.compressed_tensors"
    package = types.ModuleType(pkg)
    package.__path__ = []
    module = types.ModuleType(pkg + ".compressed_tensors")

    class _FakeCT:
        def __init__(self, raw):
            self.raw = raw
            self.ignore = list(raw.get("ignore", []))
            self.packed_modules_mapping = {}

        @classmethod
        def from_config(cls, raw):
            delegated["config"] = raw
            return cls(raw)

    module.CompressedTensorsConfig = _FakeCT
    utils = types.ModuleType(pkg + ".utils")
    utils.find_matched_target = (
        lambda prefix, layer, targets, fused: (
            prefix if prefix in targets else None))
    utils.should_ignore_layer = lambda *a, **k: False
    monkeypatch.setitem(sys.modules, pkg, package)
    monkeypatch.setitem(sys.modules, pkg + ".compressed_tensors", module)
    monkeypatch.setitem(sys.modules, pkg + ".utils", utils)

    config = _resolved(cfg)
    assert TARGET in delegated["config"]["ignore"]
    assert stock not in delegated["config"]["ignore"]
    assert TARGET in config.target_trellis


# --- refusals --------------------------------------------------------------

def test_dispatch_refuses_when_the_lane_flag_is_unset():
    """Opt-in is enforced AT DISPATCH, not only inside the factory.

    An artifact whose sidecar declares trellis units must fail loudly on a
    box that has not opted in -- silently falling through to bf16 would serve
    a different kernel and a different activation contract than the one the
    checkpoint describes.
    """
    config = _resolved()
    with pytest.raises(RuntimeError, match=TRELLIS_E4M3_FLAG):
        config.get_quant_method(_layer(), TARGET)


def test_dispatch_refuses_a_fused_module(monkeypatch):
    """Per-role wires cannot be concatenated, so say that rather than crash.

    Each wire carries its own alphabets, per-column rate schedule and row
    padding. vLLM fuses q/k/v and gate/up on nearly every architecture, so
    this refusal is common and its message has to name the supported form.
    """
    _enable(monkeypatch)
    fused_target = "model.layers.0.self_attn.qkv_proj"
    config = _resolved(_config(targets=[fused_target]))
    monkeypatch.setattr(
        type(config), "fused_role_owners",
        lambda self, prefix: [object(), object()], raising=False)
    with pytest.raises(ValueError, match="per-role trellis wires"):
        config.get_quant_method(_layer(), fused_target)


def test_dispatch_refuses_tensor_parallel_above_one(monkeypatch):
    """A blob has no splittable axis; a sharded artifact needs per-rank wires."""
    _enable(monkeypatch)
    config = _resolved()
    monkeypatch.setattr(type(config), "_tensor_parallel_world_size",
                        lambda self: 2, raising=False)
    with pytest.raises(ValueError, match="tensor-parallel size 1"):
        config.get_quant_method(_layer(), TARGET)


@pytest.mark.parametrize("bad,match", [
    ({"body_rate_q256": 4}, "outside the reader domain"),
    ({"columns": 500}, "multiple of 16"),
    ({"rows": 0}, "must be positive"),
])
def test_an_unserveable_scheme_is_refused_at_sidecar_parse(bad, match):
    """Refusal happens while the sidecar is parsed, before a parameter exists.

    Same fail-closed moment ``_validate_cb_format_scheme`` owns for CB: a
    geometry the reader cannot serve must not survive long enough to size a
    resident byte plane.
    """
    scheme = _scheme(TCQ_E2M1_R256, **bad)
    with pytest.raises(ValueError, match=match):
        _resolved(_config(scheme))


def test_a_scheme_missing_its_geometry_is_refused():
    """The sidecar must be gateable WITHOUT parsing the blob."""
    scheme = {k: v for k, v in _scheme().items() if k != "rows"}
    with pytest.raises(ValueError, match="missing"):
        _resolved(_config(scheme))
