"""Per-(layer, STACK) CB scheme resolution for FusedMoE expert stacks.

A graded checkpoint may put ``…experts.gate_up_proj`` on one rung and
``…experts.down_proj`` on another.  The two stacks are separately allocated
``(E, out, row_bytes)`` byte buffers whose last dimension is a function of the
stack's OWN ``type_size``, so a resolver that returns ONE scheme per layer
mis-sizes the other buffer (load-time shape error) — or, for two rungs that
happen to share a ``type_size``, decodes it against the wrong codebook with no
error at all.

These tests are CPU-only and cover the config resolver
(``_moe_stack_schemes_for_prefix``).  They deliberately re-assert the
namespace-canonicalisation property that ``test_target_namespace_compat.py``
owns for the single-scheme accessor: the per-stack resolver is a NEW entry
point, and the multimodal wrapper regression (35B CB serve boot) must not be
able to come back through it.

vLLM symbols are stubbed when vLLM is unavailable, exactly as the namespace
compat suite does — ``gridbook.config`` imports them at module scope.
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
        lin.LinearBase = type("LinearBase", (), {})
        lin.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
        lin.LinearMethodBase = type("LinearMethodBase", (), {})
        lin.register_weight_loader_v2_supported_method = lambda cls: cls
        par = _mod("vllm.model_executor.parameter")

        class _StubParam(torch.nn.Parameter):
            def __new__(cls, data, **kw):
                return super().__new__(cls, data, requires_grad=False)

            def __init__(self, data, **kw):
                pass

        par.ModelWeightParameter = _StubParam
        par.ChannelQuantScaleParameter = _StubParam
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

from gridbook.config import PrismaQuantConfig  # noqa: E402


def _scheme(k, type_size, codebook_ref):
    return {"grid": "fp8", "mode": "product", "k": k, "n_sub": 4,
            "type_size": type_size, "group_size": 0, "vec_dim": 8,
            "codebook_ref": [codebook_ref], "codebook_source": "learned"}


# k44 -> type_size 4*44 = 176; k36 -> 144.  Two rungs, two byte widths.
_RUNG_HI = _scheme(44, 176, "cb.k44")
_RUNG_LO = _scheme(36, 144, "cb.k36")

# The three namespace vintages a serving prefix can arrive in for layer 1's
# expert stack (identical to the compat suite's list — the per-stack resolver
# must answer the namespace question the same way the single-scheme one does).
_MOE_EXPERT_PREFIXES = [
    "language_model.model.layers.1.mlp.experts",   # wrapper-class serving form
    "model.layers.1.mlp.experts",                  # canonical
    "model.language_model.layers.1.mlp.experts",   # old checkpoint form
]


def _cfg(groups):
    """``groups`` = list of (targets, scheme)."""
    cfg = PrismaQuantConfig.from_config({
        "quant_method": "prismaquant", "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            f"g{i}": {"format": "FP8_CB", "targets": list(t),
                      "scheme": dict(s)}
            for i, (t, s) in enumerate(groups)
        },
        "ignore": ["lm_head"],
    })
    cfg._ensure_resolved()
    return cfg


def _graded(stored_ns="model.language_model."):
    """Layer 1's experts: gate_up on k44/ts176, down on k36/ts144."""
    base = stored_ns + "layers.1.mlp.experts."
    return _cfg([([base + "gate_up_proj"], _RUNG_HI),
                 ([base + "down_proj"], _RUNG_LO)])


def _uniform(stored_ns="model.language_model."):
    base = stored_ns + "layers.1.mlp.experts."
    return _cfg([([base + "gate_up_proj", base + "down_proj"], _RUNG_HI)])


# ---------------------------------------------------------------------------
# 1. The namespace property, on the NEW entry point.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stored_ns", ["model.language_model.", "model."])
@pytest.mark.parametrize("prefix", _MOE_EXPERT_PREFIXES)
def test_per_stack_resolves_across_namespaces(stored_ns, prefix):
    """Every (stored target namespace x serving prefix namespace) pair naming
    layer 1's expert stack must resolve BOTH stacks. A raw ``startswith`` here
    is the 35B CB serve-boot bug: the wrapper serving prefix
    ``language_model.model.layers.N.mlp.experts`` matches no canonicalised
    target, no CB MoE method is created, and the arch's expert mapping then
    AttributeErrors on ``experts.w2_weight.cb_qweight``."""
    stacks = _uniform(stored_ns)._moe_stack_schemes_for_prefix(prefix)
    assert stacks is not None, (stored_ns, prefix)
    assert set(stacks) == {"gate_up_proj", "down_proj"}
    assert all(s["k"] == _RUNG_HI["k"] for s in stacks.values())


def test_per_stack_does_not_overmatch():
    """A layer with no CB expert group still resolves to None (stock CT MoE
    path), and a dense sibling prefix must not pick the expert group up."""
    cfg = _uniform()
    assert cfg._moe_stack_schemes_for_prefix(
        "language_model.model.layers.2.mlp.experts") is None
    assert cfg._moe_stack_schemes_for_prefix(
        "language_model.model.layers.1.mlp.shared_expert") is None


@pytest.mark.parametrize("neighbour", ["experts2", "experts_backup"])
def test_per_stack_requires_a_dotted_prefix_boundary(neighbour):
    """A raw startswith(``...experts``) silently steals either neighbour's
    schemes.  Only an actual ``...experts.<projection>`` child may match."""
    base = f"model.layers.1.mlp.{neighbour}."
    cfg = _cfg([([base + "gate_up_proj", base + "down_proj"], _RUNG_HI)])
    assert cfg._moe_stack_schemes_for_prefix(
        "model.layers.1.mlp.experts") is None


# ---------------------------------------------------------------------------
# 2. The graded ladder: two stacks, two rungs, two byte widths.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", _MOE_EXPERT_PREFIXES)
def test_mixed_rung_layer_resolves_two_type_sizes(prefix):
    """THE regression this change exists for: the resolver must hand back BOTH
    rungs, not collapse the layer onto the first target it matched."""
    stacks = _graded()._moe_stack_schemes_for_prefix(prefix)
    assert stacks is not None, prefix
    assert stacks["gate_up_proj"]["type_size"] == 176
    assert stacks["down_proj"]["type_size"] == 144
    assert stacks["gate_up_proj"]["k"] == 44
    assert stacks["down_proj"]["k"] == 36
    assert (stacks["gate_up_proj"]["codebook_ref"]
            != stacks["down_proj"]["codebook_ref"])


def test_unfused_gate_and_up_fold_into_one_stack():
    """Unfused ``gate_proj``/``up_proj`` targets land in the ONE w13 buffer, so
    they fold into ``gate_up_proj`` — and, agreeing, resolve cleanly."""
    base = "model.layers.1.mlp.experts."
    cfg = _cfg([([base + "gate_proj", base + "up_proj"], _RUNG_HI),
                ([base + "down_proj"], _RUNG_LO)])
    stacks = cfg._moe_stack_schemes_for_prefix(base.rstrip("."))
    assert set(stacks) == {"gate_up_proj", "down_proj"}
    assert stacks["gate_up_proj"]["type_size"] == 176
    assert stacks["down_proj"]["type_size"] == 144


def test_disagreeing_members_of_one_stack_raise():
    """gate_proj and up_proj share the w13 buffer; two schemes for it is an
    exporter bug and must fail loudly rather than pick one."""
    base = "model.layers.1.mlp.experts."
    cfg = _cfg([([base + "gate_proj"], _RUNG_HI),
                ([base + "up_proj"], _RUNG_LO),
                ([base + "down_proj"], _RUNG_LO)])
    with pytest.raises(ValueError, match="gate_up_proj"):
        cfg._moe_stack_schemes_for_prefix(base.rstrip("."))


# ---------------------------------------------------------------------------
# 3. The single-scheme accessor keeps its shape for uniform layers.
# ---------------------------------------------------------------------------

def test_single_scheme_accessor_unchanged_for_uniform_layers():
    """``_moe_scheme_for_prefix`` still returns ONE subscriptable scheme — the
    shape ``test_target_namespace_compat.py`` asserts."""
    sch = _uniform()._moe_scheme_for_prefix(
        "language_model.model.layers.1.mlp.experts")
    assert sch is not None and sch["k"] == _RUNG_HI["k"]


def test_single_scheme_accessor_refuses_a_graded_layer():
    """There IS no single scheme for a graded layer; returning one is exactly
    the bug. Callers that need an answer use the per-stack entry point."""
    with pytest.raises(ValueError, match="different CB rungs"):
        _graded()._moe_scheme_for_prefix(
            "language_model.model.layers.1.mlp.experts")
