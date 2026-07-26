"""Backward-compat: historical CHECKPOINT-namespace targets must resolve.

The shipped ``rdtand/Qwen3.6-27B-prismaquant-gridbook-5.5bit-vllm`` artifact
stores ``config_groups[*].targets`` as ``model.language_model.layers.N.…``
while newer artifacts (Laguna) store canonical ``model.layers.N.…``. After the
wrapper-class canonicalization work the resolver probes with canonical /
serving prefixes, so the old form resolved to None for every LM Linear.

Both sides are now normalised onto one canonical string: incoming prefixes via
``_canonical_prefix``, stored targets/ignore via ``_canonical_target`` applied
once at parse time. CPU-only; vLLM symbols are stubbed when unavailable.
"""
import sys
import types

import pytest

if "vllm" not in sys.modules:
    try:
        import vllm  # noqa: F401
    except Exception:
        torch = pytest.importorskip("torch")

        def _mod(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            return m

        _mod("vllm")
        _mod("vllm.model_executor")
        _mod("vllm.model_executor.layers")
        _mod("vllm.model_executor.layers.quantization")
        lin = _mod("vllm.model_executor.layers.linear")

        class LinearBase:  # minimal stand-ins: only isinstance/base use here
            pass

        class UnquantizedLinearMethod:
            pass

        lin.LinearBase = LinearBase
        lin.UnquantizedLinearMethod = UnquantizedLinearMethod
        bc = _mod("vllm.model_executor.layers.quantization.base_config")

        class QuantizationConfig:
            def __init__(self):
                pass

        bc.QuantizationConfig = QuantizationConfig
        bc.QuantizeMethodBase = object
        vpe = _mod("vllm.model_executor.layers.vocab_parallel_embedding")
        vpe.UnquantizedEmbeddingMethod = type("UEM", (), {})
        vpe.VocabParallelEmbedding = type("VPE", (), {})
        fm = _mod("vllm.model_executor.layers.fused_moe")
        fm.RoutedExperts = type("RoutedExperts", (), {})

from gridbook.config import (  # noqa: E402
    PrismaQuantConfig,
    _canonical_prefix,
    _canonical_target,
)

_SCHEME = {"grid": "fp8", "mode": "product", "k": 44, "n_sub": 4,
           "type_size": 176, "group_size": 0, "vec_dim": 8,
           "codebook_group": "mlp", "codebook_source": "learned"}

# The three namespace forms a serving prefix can arrive in for layer 0's
# fused gate_up module.
_FUSED_PROBES = [
    "language_model.model.layers.0.mlp.gate_up_proj",   # wrapper-class serving
    "model.layers.0.mlp.gate_up_proj",                  # canonical
    "model.language_model.layers.0.mlp.gate_up_proj",   # old checkpoint form
]


def _config():
    """Old-checkpoint-form CB group + ignore entry, plus one new-form group."""
    return {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "old_form": {
                "format": "FP8_CB_K44",
                "targets": [
                    "model.language_model.layers.0.mlp.gate_proj",
                    "model.language_model.layers.0.mlp.up_proj",
                    "model.language_model.layers.0.mlp.down_proj",
                    "visual.blocks.0.mlp.down_proj",
                ],
                "scheme": dict(_SCHEME)},
            "new_form": {
                "format": "FP8_CB_K44",
                "targets": ["model.layers.1.mlp.down_proj"],
                "scheme": dict(_SCHEME)},
        },
        "ignore": ["model.language_model.layers.0.mlp.foo", "lm_head"],
    }


def _resolved():
    cfg = PrismaQuantConfig.from_config(_config())
    cfg._ensure_resolved()
    return cfg


def test_canonical_target_rules():
    assert _canonical_target(
        "model.language_model.layers.3.mlp.gate_proj") == \
        "model.layers.3.mlp.gate_proj"
    assert _canonical_target(
        "language_model.model.layers.3.mlp.gate_proj") == \
        "model.layers.3.mlp.gate_proj"
    # untouched namespaces
    for name in ("visual.blocks.0.mlp.down_proj", "mtp.layers.0.mlp.up_proj",
                 "model.layers.0.mlp.up_proj", "lm_head"):
        assert _canonical_target(name) == name


def test_prefix_and_target_agree():
    """Requirement: a probe prefix P and a stored target T naming the same
    Linear must canonicalise to the same string."""
    same = {_canonical_prefix(p) for p in _FUSED_PROBES}
    assert same == {"model.layers.0.mlp.gate_up_proj"}
    assert _canonical_target("model.language_model.layers.0.mlp.gate_proj") == \
        _canonical_prefix("language_model.model.layers.0.mlp.gate_proj")


def test_old_form_targets_are_normalised_at_parse():
    cfg = _resolved()
    assert "model.layers.0.mlp.gate_proj" in cfg.target_scheme
    assert "model.layers.0.mlp.down_proj" in cfg.target_scheme
    assert "model.layers.1.mlp.down_proj" in cfg.target_scheme     # new form
    assert "visual.blocks.0.mlp.down_proj" in cfg.target_scheme    # untouched
    assert not any(t.startswith("model.language_model.")
                   for t in cfg.target_scheme)


@pytest.mark.parametrize("probe", _FUSED_PROBES)
def test_fused_gate_up_resolves_in_every_namespace(probe):
    """MergedColumnParallelLinear probes with the FUSED name; the resolver
    re-fuses it from the per-member gate_proj/up_proj targets."""
    assert _resolved()._scheme_for_prefix(probe) is not None


@pytest.mark.parametrize("suffix", ["mlp.down_proj"])
@pytest.mark.parametrize("pre", ["language_model.model.layers.0.",
                                 "model.layers.0.",
                                 "model.language_model.layers.0."])
def test_unfused_member_resolves_in_every_namespace(pre, suffix):
    assert _resolved()._scheme_for_prefix(pre + suffix) is not None


@pytest.mark.parametrize("pre", ["language_model.model.layers.0.",
                                 "model.layers.0.",
                                 "model.language_model.layers.0."])
def test_ignore_entry_honored_in_every_namespace(pre):
    cfg = _resolved()
    assert cfg._is_ignored(pre + "mlp.foo")
    assert not cfg._is_ignored(pre + "mlp.down_proj")


def test_new_form_layer_still_resolves():
    cfg = _resolved()
    for pre in ("model.layers.1.", "language_model.model.layers.1.",
                "model.language_model.layers.1."):
        assert cfg._scheme_for_prefix(pre + "mlp.down_proj") is not None
