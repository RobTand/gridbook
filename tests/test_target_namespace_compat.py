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

        class LinearMethodBase:  # PrismaQuantCBLinearMethod's base
            pass

        lin.LinearBase = LinearBase
        lin.UnquantizedLinearMethod = UnquantizedLinearMethod
        lin.LinearMethodBase = LinearMethodBase
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
        ctu = _mod(
            "vllm.model_executor.layers.quantization.compressed_tensors.utils")

        def _stub_should_ignore(layer_name, ignore=(), fused_mapping=None):
            try:
                import regex as regex_engine
            except ImportError:
                # ``regex`` arrives with a real vLLM installation, but it is
                # intentionally not a Gridbook dependency.  Keep the CPU-only
                # test shim usable in the wheel's minimal dependency set.
                import re as regex_engine

            def _matches(name):
                return any(
                    regex_engine.match(entry[3:], name) is not None
                    if entry.startswith("re:") else entry == name
                    for entry in ignore)

            fused_mapping = fused_mapping or {}
            leaf = layer_name.rsplit(".", 1)[-1]
            if leaf in fused_mapping and layer_name not in ignore:
                stem = layer_name[:-len(leaf)]
                verdicts = [_matches(stem + shard)
                            for shard in fused_mapping[leaf]]
                if len(set(verdicts)) != 1:
                    raise ValueError("Found a different quantization schemes")
                return verdicts[0]
            return _matches(layer_name)

        ctu.should_ignore_layer = _stub_should_ignore
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


# ---------------------------------------------------------------------------
# 35B CB serve-boot regression: the MoE expert-stack lookup must canonicalise
# the serving prefix exactly like the Linear lookup does. Before the fix,
# `_moe_scheme_for_prefix` did a RAW `startswith`, so the wrapper-class serving
# prefix `language_model.model.layers.N.mlp.experts` matched no canonicalised
# target, no CB MoE method was created, no w13/w2_cb_qweight params existed, and
# the arch's own expert mapping then blew up with
# `'RoutedExperts' object has no attribute 'w2_weight.cb_qweight'`.
# ---------------------------------------------------------------------------

_MOE_EXPERT_PREFIXES = [
    "language_model.model.layers.1.mlp.experts",   # wrapper-class serving form
    "model.layers.1.mlp.experts",                  # canonical
    "model.language_model.layers.1.mlp.experts",   # old checkpoint form
]


def _moe_config(targets):
    return {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {
            "moe": {"format": "FP8_CB_K28", "targets": list(targets),
                    "scheme": dict(_SCHEME)},
        },
        "ignore": ["lm_head"],
    }


@pytest.mark.parametrize("stored_ns", ["model.language_model.", "model."])
@pytest.mark.parametrize("prefix", _MOE_EXPERT_PREFIXES)
def test_moe_scheme_resolves_across_namespaces(stored_ns, prefix):
    """Every (stored target namespace × serving prefix namespace) pair that
    names layer 1's expert stack must resolve to the CB scheme."""
    cfg = PrismaQuantConfig.from_config(_moe_config([
        stored_ns + "layers.1.mlp.experts.gate_up_proj",
        stored_ns + "layers.1.mlp.experts.down_proj",
    ]))
    cfg._ensure_resolved()
    sch = cfg._moe_scheme_for_prefix(prefix)
    assert sch is not None, (stored_ns, prefix)
    assert sch["k"] == _SCHEME["k"]


def test_moe_scheme_does_not_overmatch():
    """A layer with no CB expert group must still resolve to None (so the stock
    CT MoE path is used) — canonicalisation must not turn the prefix test into a
    substring free-for-all."""
    cfg = PrismaQuantConfig.from_config(_moe_config([
        "model.language_model.layers.1.mlp.experts.gate_up_proj",
    ]))
    cfg._ensure_resolved()
    assert cfg._moe_scheme_for_prefix(
        "language_model.model.layers.2.mlp.experts") is None
    # a dense MLP prefix must not pick up the expert group either
    assert cfg._moe_scheme_for_prefix(
        "language_model.model.layers.1.mlp.shared_expert") is None


@pytest.mark.parametrize("neighbour", ["experts2", "experts_backup"])
@pytest.mark.parametrize("prefix", _MOE_EXPERT_PREFIXES)
def test_moe_scheme_requires_a_dotted_prefix_boundary(neighbour, prefix):
    """A sibling whose name merely starts with ``experts`` is not that stack."""
    cfg = PrismaQuantConfig.from_config(_moe_config([
        f"model.layers.1.mlp.{neighbour}.gate_up_proj",
        f"model.layers.1.mlp.{neighbour}.down_proj",
    ]))
    cfg._ensure_resolved()

    assert cfg._moe_scheme_for_prefix(prefix) is None


# ---------------------------------------------------------------------------
# The FOURTH namespace vintage: post-``apply_vllm_mapper`` (issue #1,
# RobTand/gridbook, 2026-07-25).
#
# ``_ensure_resolved`` canonicalises the STORED targets, but
# ``apply_vllm_mapper`` then rewrites ``target_scheme`` keys INTO the vLLM
# mapper's namespace (``model.`` -> ``language_model.model.`` for a VL wrapper
# class). Dense fused re-fusion built its shard keys from the CANONICAL prefix
# only, so on a Qwen3.5/3.6-VL-class hybrid the GDN fused ``in_proj_qkvz``
# matched nothing, fell through to ``UnquantizedLinearMethod``, and the layer
# allocated BF16 instead of FP8-CB. ``_shard_roles`` had the identical bug.
# Both now go through ``PrismaQuantConfig.shard_target_keys``, which owns the
# "which namespace vintage?" question for every dense call site.
# ---------------------------------------------------------------------------


class _WrapperMapper:
    """The one rewrite that matters here, mimicking vLLM's ``WeightsMapper``:
    nest ``model.*`` under ``language_model.`` for a VL wrapper class."""

    @staticmethod
    def _apply(name):
        return "language_model." + name if name.startswith("model.") else name

    def apply_list(self, names):
        return [self._apply(n) for n in names]

    def apply_dict(self, mapping):
        return {self._apply(k): v for k, v in mapping.items()}


_GDN_BASE = "model.layers.45.linear_attn."
_GDN_ROLES = ("in_proj_qkv", "in_proj_z")
_GDN_SERVE = "language_model.model.layers.45.linear_attn.in_proj_qkvz"
_GDN_MAPPED_ROLES = {"language_model." + _GDN_BASE + r for r in _GDN_ROLES}


def _mapped_config(targets, ignore=("lm_head",)):
    """Resolve, then push the targets into the vLLM mapper namespace — exactly
    the state ``get_quant_method`` sees at serve time on a wrapper-class model."""
    cfg = PrismaQuantConfig.from_config({
        "quant_method": "prismaquant", "format": "fp8_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {"gdn": {"format": "FP8_CB_K44",
                                  "targets": list(targets),
                                  "scheme": dict(_SCHEME)}},
        "ignore": list(ignore),
    })
    cfg._ensure_resolved()
    cfg.apply_vllm_mapper(_WrapperMapper())
    return cfg


def _gdn_cfg():
    return _mapped_config([_GDN_BASE + r for r in _GDN_ROLES])


def test_mapper_moves_targets_out_of_the_canonical_namespace():
    """Precondition for the two tests below: after the mapper the keys are
    NEITHER the stored form NOR the canonical form."""
    assert set(_gdn_cfg().target_scheme) == _GDN_MAPPED_ROLES


def test_fused_gdn_resolves_in_the_mapper_namespace():
    """Bug 1: the fused GDN module must resolve to its members' CB scheme."""
    sch = _gdn_cfg()._scheme_for_prefix(_GDN_SERVE)
    assert sch is not None
    assert sch["k"] == _SCHEME["k"]


def test_gdn_shard_roles_resolve_in_the_mapper_namespace():
    """Bug 2: ``_shard_roles`` must return BOTH CB roles, in shard order, so
    ``process_weights_after_loading`` builds a full-length ``cb_row_offset``."""
    from gridbook.linear import PrismaQuantCBLinearMethod
    cfg = _gdn_cfg()
    method = PrismaQuantCBLinearMethod(cfg, dict(_SCHEME), _GDN_SERVE)
    assert method._shard_roles() == [
        "language_model." + _GDN_BASE + r for r in _GDN_ROLES]


@pytest.mark.parametrize("leaf,members", [
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("gate_up_proj", ("gate_proj", "up_proj")),
    ("in_proj_ba", ("in_proj_b", "in_proj_a")),
])
def test_every_fused_family_resolves_in_the_mapper_namespace(leaf, members):
    base = "model.layers.7.mod."
    cfg = _mapped_config([base + m for m in members])
    assert cfg._scheme_for_prefix("language_model." + base + leaf) is not None


def test_plain_linear_and_shard_roles_survive_the_mapper():
    """Already-working case: an unfused Linear resolves and reports itself as
    its own single role (the ``or [leaf]`` fallback ``_shard_roles`` relies on
    and ``_scheme_for_prefix`` deliberately does NOT have)."""
    from gridbook.linear import PrismaQuantCBLinearMethod
    base = "model.layers.3.mlp."
    cfg = _mapped_config([base + "down_proj"])
    serve = "language_model." + base + "down_proj"
    assert cfg._scheme_for_prefix(serve) is not None
    method = PrismaQuantCBLinearMethod(cfg, dict(_SCHEME), serve)
    assert method._shard_roles() == ["language_model." + base + "down_proj"]


def test_shard_keys_never_mix_namespaces():
    """If BOTH vintages are present, the resolver must take one base's hits
    whole — a mixed list would pair shards from two namespaces (and, on a real
    load, two different tensors)."""
    cfg = _gdn_cfg()
    # Add the canonical vintage of ONE role only.
    cfg.target_scheme[_GDN_BASE + "in_proj_qkv"] = dict(_SCHEME)
    keys = cfg.shard_target_keys(_GDN_SERVE)
    assert keys == ["language_model." + _GDN_BASE + r for r in _GDN_ROLES]


def test_no_scheme_when_nothing_matches():
    cfg = _gdn_cfg()
    assert cfg._scheme_for_prefix(
        "language_model.model.layers.46.linear_attn.in_proj_qkvz") is None
    assert cfg.shard_target_keys(
        "language_model.model.layers.46.mlp.down_proj",
        unfused_fallback=True) == []


def test_mixed_fused_formats_still_raise():
    """The export union-find guarantee is load-bearing; the guard that catches
    a violation must survive the namespace refactor."""
    cfg = _gdn_cfg()
    other = dict(_SCHEME)
    other["k"] = 28
    other["type_size"] = 112
    cfg.target_scheme["language_model." + _GDN_BASE + "in_proj_z"] = other
    with pytest.raises(ValueError, match="mixed CB decode"):
        cfg._scheme_for_prefix(_GDN_SERVE)


# ---------------------------------------------------------------------------
# ``ignore`` follows compressed-tensors exact/regex semantics, not substrings.
#
# ``_is_ignored`` used to be ``any(ig in base for base in
# _candidate_bases(prefix) for ig in self.ignore)`` — an unanchored substring
# test. Exporters collapse ``.mlp.router.gate`` to ``.mlp.gate``, so the ignored
# BF16 router entry ``model.layers.N.mlp.gate`` was a raw substring of the
# QUANTIZED ``…mlp.gate_proj`` / ``…mlp.gate_up_proj``: the shared-expert stack
# was forced to ``UnquantizedLinearMethod``, its NVFP4 tensors had nowhere to
# land, and the load died with ``KeyError:
# 'layers.1.mlp.shared_mlp.down_proj.input_global_scale'``.
#
# The same list is handed to a real ``CompressedTensorsConfig`` by
# ``_build_ct_config``, whose matcher is exact/``re:`` — so the readings must
# agree. These tests pin both the fix and the safety property (for non-``re:``
# entries the new predicate is a strict subset of the old one).
# ---------------------------------------------------------------------------

_ROUTER_NS = ["model.layers.1.",
              "language_model.model.layers.1.",
              "model.language_model.layers.1."]


def _ignore_config(ignore):
    """A resolved config whose only interesting content is *ignore*."""
    cfg = PrismaQuantConfig.from_config({
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb",
        "config_groups": {"cb": {"format": "FP8_CB_K44",
                                 "targets": ["model.layers.0.mlp.down_proj"],
                                 "scheme": dict(_SCHEME)}},
        "ignore": list(ignore),
    })
    cfg._ensure_resolved()
    return cfg


@pytest.mark.parametrize("pre", _ROUTER_NS)
def test_collapsed_router_does_not_ignore_the_gate_projections(pre):
    """The regression itself: an ignored ``…mlp.gate`` must not swallow
    ``…mlp.gate_proj`` / ``…mlp.gate_up_proj`` in ANY namespace vintage."""
    cfg = _ignore_config(["model.layers.1.mlp.gate"])
    assert cfg._is_ignored(pre + "mlp.gate")            # the router: still BF16
    assert not cfg._is_ignored(pre + "mlp.gate_proj")
    assert not cfg._is_ignored(pre + "mlp.gate_up_proj")


def test_literal_entry_is_exact_like_compressed_tensors():
    """A literal entry names one module; parents are not implicit globs."""
    cfg = _ignore_config(["model.layers.0.mlp"])
    assert cfg._is_ignored("model.layers.0.mlp")
    assert not cfg._is_ignored("model.layers.0.mlp.down_proj")
    assert not cfg._is_ignored("model.layers.0.mlp.shared_mlp.up_proj")
    assert not cfg._is_ignored("model.layers.01.mlp.down_proj")
    assert not cfg._is_ignored("model.layers.0.mlp_extra.down_proj")


def test_exact_entry_matches_only_itself():
    cfg = _ignore_config(["lm_head"])
    assert cfg._is_ignored("lm_head")
    assert not cfg._is_ignored("lm_head_proj")
    # An entry matches the module it names and that module's descendants — not
    # a same-suffix module elsewhere in the tree. (``lm_head`` reaches
    # ``_is_ignored`` only if the head is a LinearBase; the
    # VocabParallelEmbedding arm of get_quant_method never consults it.)
    assert not cfg._is_ignored("model.lm_head")


def test_regex_entry_uses_re_match_like_compressed_tensors():
    """``re:`` delegates to ``re.match`` — start-anchored, NOT ``fullmatch`` —
    because the delegated ``CompressedTensorsConfig`` reads the identical list
    with ``re.match``. A ``fullmatch`` here would make one list mean two
    different things in one process."""
    cfg = _ignore_config([r"re:model\.layers\.\d+\.mlp\.gate$"])
    assert cfg._is_ignored("model.layers.7.mlp.gate")
    assert not cfg._is_ignored("model.layers.7.mlp.gate_proj")
    # start-anchored, so a mid-string match does not count...
    assert not cfg._is_ignored("prefix.model.layers.7.mlp.gate")
    # ...but an unanchored tail does (this is re.match, not fullmatch).
    assert _ignore_config([r"re:model\.layers\.0"])._is_ignored(
        "model.layers.0.mlp.down_proj")


def test_regex_entry_uses_compressed_tensors_regex_engine():
    """``regex`` supports Unicode properties that stdlib ``re`` rejects."""
    pytest.importorskip("regex", reason="installed with the real vLLM stack")
    cfg = _ignore_config([r"re:model\.\p{L}+$"])
    assert cfg._is_ignored("model.layers")


def test_fused_regex_is_evaluated_on_unfused_shards():
    # A regex naming only the fused spelling does not ignore its checkpoint
    # shards.  This is compressed-tensors' fused contract.
    cfg = _ignore_config([r"re:.*gate_up_proj$"])
    assert not cfg._is_ignored("model.layers.0.mlp.gate_up_proj")

    both = _ignore_config([
        r"re:.*gate_proj$", r"re:.*up_proj$",
    ])
    assert both._is_ignored("model.layers.0.mlp.gate_up_proj")

    mixed = _ignore_config([r"re:.*gate_proj$"])
    with pytest.raises(ValueError, match="different quantization schemes"):
        mixed._is_ignored("model.layers.0.mlp.gate_up_proj")


def test_regex_entry_survives_the_vllm_mapper():
    """``apply_vllm_mapper`` rewrites ignore entries; a ``re:`` entry that the
    mapper leaves alone must still be honoured against the mapped prefix."""
    cfg = _mapped_config([_GDN_BASE + r for r in _GDN_ROLES],
                         ignore=("re:.*\\.mlp\\.gate$",))
    assert cfg._is_ignored("language_model.model.layers.9.mlp.gate")
    assert not cfg._is_ignored("language_model.model.layers.9.mlp.gate_proj")


def test_mapper_is_never_asked_to_rewrite_regex_syntax():
    cfg = _ignore_config([r"re:.*\.mlp\.gate$", "model.layers.0.mlp.foo"])

    class _LiteralOnlyMapper:
        def apply_list(self, names):
            assert all(not name.startswith("re:") for name in names)
            return ["mapped." + name for name in names]

        def apply_dict(self, values):
            return values

    cfg.apply_vllm_mapper(_LiteralOnlyMapper())
    assert r"re:.*\.mlp\.gate$" in cfg.ignore
    assert "mapped.model.layers.0.mlp.foo" in cfg.ignore


def test_get_quant_method_delegates_gate_projection_not_router(monkeypatch):
    """Exercise the public dispatch order, not only the private predicate."""
    import gridbook.config as config_mod

    class _Linear:
        pass

    class _CBMethod:
        def __init__(self, *args, **kwargs):
            pass

    fake_linear_module = types.ModuleType("gridbook.linear")
    fake_linear_module.PrismaQuantCBLinearMethod = _CBMethod
    monkeypatch.setitem(sys.modules, "gridbook.linear", fake_linear_module)
    monkeypatch.setattr(config_mod, "LinearBase", _Linear)

    delegated = object()
    ct = types.SimpleNamespace(
        packed_modules_mapping=None,
        get_quant_method=lambda layer, prefix: delegated,
    )
    cfg = _ignore_config(["model.layers.1.mlp.gate"])
    cfg.ct_config = ct

    router = cfg.get_quant_method(_Linear(), "model.layers.1.mlp.gate")
    assert isinstance(router, config_mod.UnquantizedLinearMethod)
    assert cfg.get_quant_method(
        _Linear(), "model.layers.1.mlp.gate_proj") is delegated


_SUBSET_ENTRIES = ["model.layers.1.mlp.gate", "model.layers.1.mlp", "lm_head",
                   "mlp.down_proj", "model.layers.1"]
_SUBSET_NAMES = ["model.layers.1.mlp.gate", "model.layers.1.mlp.gate_proj",
                 "model.layers.1.mlp.gate_up_proj", "model.layers.1.mlp",
                 "model.layers.1.mlp.shared_mlp.down_proj", "lm_head",
                 "model.layers.11.mlp.gate", "language_model.model.layers.1"]


@pytest.mark.parametrize("entry", _SUBSET_ENTRIES)
@pytest.mark.parametrize("name", _SUBSET_NAMES)
def test_non_regex_match_is_a_subset_of_the_old_substring_test(entry, name):
    """Safety property: the new predicate can only ever *un*-ignore a module,
    never newly ignore one. (Asserted, not argued.)"""
    cfg = _ignore_config([entry])
    new = cfg._is_ignored(name)
    old = entry in name
    assert not (new and not old), (entry, name)
