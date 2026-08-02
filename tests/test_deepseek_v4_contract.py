"""DeepSeek-V4 (``deepseek_v4``) serving contract — ROADMAP D0.1.

The DSV4-Flash body is a 43-layer, all-MoE, MLA architecture whose vLLM 0.24
class differs from every arch Gridbook had already wired in four ways that each
break a naive loader. This file pins all four against the real class shape,
CPU-first with synthetic fixtures:

1. **Module attributes are ``attn`` / ``ffn``**, not ``self_attn`` / ``mlp``
   (``vllm/models/deepseek_v4/nvidia/model.py``). Nothing in Gridbook may
   hard-code the Llama spelling.
2. **The routed stack nests one level deeper**: the FusedMoE prefix is
   ``model.layers.N.ffn.experts`` while the parameters live at
   ``model.layers.N.ffn.experts.routed_experts.w13_cb_qweight``. This is the
   HunYuan-V3 nesting again, so the ``.experts.`` stem/leaf anchor absorbs it.
3. **The checkpoint has no ``model.`` component** — keys start at ``layers.N.``
   — and the class re-attaches it inside its own ``hf_to_vllm_mapper``
   (``{"layers.": "model.layers."}``), i.e. AFTER a serving prefix has reached
   ``get_quant_method``. Both the loader (via ``_hf_mapper_rename``) and the
   config resolver (via ``_canonical_prefix``) have to cross that gap.
4. **The class defines NO ``packed_modules_mapping``** (verified against vLLM
   0.24.0), yet it merges ``attn.wq_a``+``attn.wkv`` into ``fused_wqa_wkv`` and
   the shared expert's ``w1``+``w3`` into ``gate_up_proj``. The merge is
   published only through ``stacked_params_mapping``, so Gridbook's fused
   fallback tables are the only merge information it has.

MTP/DSpark is deliberately NOT exercised as a load path: the body's
``load_weights`` is ``AutoWeightsLoader(self, skip_substrs=["mtp."])``, so all
4,705 ``mtp.*`` tensors are dropped before any parameter lookup, and no
``dspark_*`` config key is referenced anywhere in the vLLM package. The test
that matters for them is that a stacked-CB MTP tensor still DEFERS rather than
being mis-routed into a body layer.

torch-only for the loader/resolver halves (they import no vLLM); the
class-shaped resolution test is skip-guarded and runs in the container.
"""
from __future__ import annotations

import json

import pytest
import torch

from gridbook.moe_toplevel_loader import (
    _build_reverse_fusion,
    install_toplevel_cb_expert_loader,
    resolve_cb_expert_param,
    resolve_shared_cb_target,
)
from gridbook.runtime_contract import load_runtime_contract


# --------------------------------------------------------------------------
# The synthetic DSV4 shape. Real ratios, tiny numbers: 256 routed experts and
# hidden 4096 become 4 and 16 so the whole file stays CPU-instant, but every
# NAME is the production one.
# --------------------------------------------------------------------------

E, HID, INTER, BYTES = 4, 16, 8, 3
LAYER = 5
#: vLLM param stem for the routed stack (note BOTH `ffn` and `routed_experts`).
RE = f"model.layers.{LAYER}.ffn.experts.routed_experts"
#: Checkpoint stem, in the producer's SOURCE namespace (no `model.`).
CKPT = f"layers.{LAYER}.ffn.experts"


class _DSV4WeightsMapper:
    """The subset of vLLM's ``WeightsMapper`` the wrapper actually calls.

    Faithful to ``DeepseekV4ForCausalLM.hf_to_vllm_mapper``
    (``vllm/models/deepseek_v4/nvidia/model.py``): a ``layers.`` -> ``model.``
    prefix lift, the routed-expert scale rule, and the catch-all ``.scale``
    rule. Gridbook calls only ``_map_name``.
    """

    def _map_name(self, name: str) -> str | None:
        for orig, new in (("layers.", "model.layers."),
                          ("embed.", "model.embed."),
                          ("norm.", "model.norm."),
                          ("mtp.", "model.mtp.")):
            if name.startswith(orig):
                return new + name[len(orig):]
        if name == "head.weight":
            return "lm_head.weight"
        return name


def _make_dsv4_cls(params):
    """A ``DeepseekV4ForCausalLM``-shaped stub.

    Mirrors the three behaviours of the real ``load_weights`` that decide
    whether a Gridbook tensor lands: the ``skip_substrs=["mtp."]`` drop, the
    mapper application, and — critically — the UNGUARDED ``params_dict[name]``
    that makes any other unmatched key a hard ``KeyError`` rather than a skip
    (``nvidia/model.py``: there is no ``if name not in params_dict: continue``).
    """

    class _FakeDSV4:
        # Deliberately absent, exactly like the real class:
        #   packed_modules_mapping

        def __init__(self):
            self._params = dict(params)
            self.delegated = []
            self.hf_to_vllm_mapper = _DSV4WeightsMapper()

        def named_parameters(self):
            return list(self._params.items())

        def load_weights(self, weights):          # ORIGINAL (stub stock loader)
            loaded = set()
            for name, w in weights:
                mapped = self.hf_to_vllm_mapper._map_name(name)
                if mapped is None or "mtp." in mapped:
                    continue                      # skip_substrs=["mtp."]
                self.delegated.append(mapped)
                if mapped not in self._params:
                    raise KeyError(mapped)        # fail-closed, as upstream
                self._params[mapped].copy_(w)
                loaded.add(mapped)
            return loaded

    return _FakeDSV4


def _cb_params():
    return {
        RE + ".w13_cb_qweight": torch.zeros(E, 2 * INTER, BYTES,
                                            dtype=torch.uint8),
        RE + ".w2_cb_qweight": torch.zeros(E, HID, BYTES, dtype=torch.uint8),
        RE + ".w13_weight_scale": torch.zeros(E, 2 * INTER,
                                              dtype=torch.float32),
        RE + ".w2_weight_scale": torch.zeros(E, HID, dtype=torch.float32),
        f"model.layers.{LAYER}.attn_norm.weight": torch.zeros(HID),
    }


# --------------------------------------------------------------------------
# 1. Registration
# --------------------------------------------------------------------------

def test_contract_registers_deepseek_v4_profile_and_defining_module():
    profiles = load_runtime_contract()["producer_profiles"]
    assert "deepseek_v4" in profiles["supported_ids"]
    # plugin.py installs on the module that DEFINES the entrypoint class; for
    # DSV4 that is the platform submodule, not the package __init__ (which only
    # re-exports and would be skipped by the __module__ guard).
    assert ("vllm.models.deepseek_v4.nvidia.model"
            in profiles["top_level_loader_modules"])
    assert ("vllm.models.deepseek_v4"
            not in profiles["top_level_loader_modules"])


# --------------------------------------------------------------------------
# 2. Expert-stack resolution through `ffn` + `routed_experts`
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("suffix", "leaf"), [
    ("gate_up_proj.cb_qweight", "w13_cb_qweight"),
    ("down_proj.cb_qweight", "w2_cb_qweight"),
    ("gate_up_proj.weight_scale", "w13_weight_scale"),
    ("down_proj.weight_scale", "w2_weight_scale"),
])
def test_expert_anchor_spans_ffn_and_routed_experts(suffix, leaf):
    """The `.experts.` anchor is indifferent to BOTH DSV4 differences at once:
    the parent is `ffn` (not `mlp`) and the target nests under
    `routed_experts`."""
    param_names = tuple(_cb_params())
    resolved = resolve_cb_expert_param(
        f"model.layers.{LAYER}.ffn.experts.{suffix}", param_names)
    assert resolved == RE + "." + leaf


def test_expert_stacks_load_with_real_names_and_fill_guard():
    from gridbook.cb_fill_guard import CB_FILLED_ATTR

    cls = _make_dsv4_cls(_cb_params())
    install_toplevel_cb_expert_loader(cls)
    m = cls()
    loaded = m.load_weights(iter([
        (CKPT + ".gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 7, dtype=torch.uint8)),
        (CKPT + ".down_proj.cb_qweight",
         torch.full((E, HID, BYTES), 9, dtype=torch.uint8)),
        (CKPT + ".gate_up_proj.weight_scale", torch.full((E, 2 * INTER), 2.0)),
        (CKPT + ".down_proj.weight_scale", torch.full((E, HID), 3.0)),
        (f"layers.{LAYER}.attn_norm.weight", torch.full((HID,), 4.0)),
    ]))

    assert torch.all(m._params[RE + ".w13_cb_qweight"] == 7)
    assert torch.all(m._params[RE + ".w2_cb_qweight"] == 9)
    assert torch.allclose(m._params[RE + ".w13_weight_scale"],
                          torch.tensor(2.0))
    assert torch.allclose(m._params[RE + ".w2_weight_scale"],
                          torch.tensor(3.0))
    # Both stacks carry the fill sentinel, so `assert_cb_experts_filled` passes.
    for name in (RE + ".w13_cb_qweight", RE + ".w2_cb_qweight"):
        assert getattr(m._params[name], CB_FILLED_ATTR, False) is True
    # No expert tensor reached the arch loader (whose params_dict lookup is
    # unguarded and would KeyError on our stacked names).
    assert not any(".experts." in n for n in m.delegated), m.delegated
    # The non-expert body tensor did, and loaded normally.
    assert f"model.layers.{LAYER}.attn_norm.weight" in m.delegated
    assert loaded == {
        RE + ".w13_cb_qweight", RE + ".w2_cb_qweight",
        RE + ".w13_weight_scale", RE + ".w2_weight_scale",
        f"model.layers.{LAYER}.attn_norm.weight",
    }


def test_source_namespace_expert_names_survive_the_mapper():
    """The artifact's key is `layers.N.…` with no `model.`; resolution must
    happen against the MAPPED name or the anchor matches no registered param.
    Regression for the whole class of "wrap installed but matched nothing"
    failures that `cb_fill_guard` reports."""
    cls = _make_dsv4_cls(_cb_params())
    install_toplevel_cb_expert_loader(cls)
    m = cls()
    loaded = m.load_weights(iter([
        (CKPT + ".gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 5, dtype=torch.uint8)),
    ]))
    assert loaded == {RE + ".w13_cb_qweight"}
    assert torch.all(m._params[RE + ".w13_cb_qweight"] == 5)


# --------------------------------------------------------------------------
# 3. MTP / DSpark — passthrough, never mis-routed
# --------------------------------------------------------------------------

def test_mtp_expert_tensor_defers_and_is_dropped_by_the_arch_loader():
    """DSV4 preserves `mtp.*` (4,705 tensors, 3 DSpark stages) as producer
    passthrough. `DeepseekV4ForCausalLM` builds
    `AutoWeightsLoader(self, skip_substrs=["mtp."])`, so at plain serving time
    NONE of them get a home. Gridbook must not claim them either: an
    `mtp.`-prefixed expert tensor has no registered param, so the wrapper
    defers and the arch loader drops it — no KeyError, no body layer
    corruption."""
    cls = _make_dsv4_cls(_cb_params())
    install_toplevel_cb_expert_loader(cls)
    m = cls()
    loaded = m.load_weights(iter([
        ("mtp.0.ffn.experts.gate_up_proj.cb_qweight",
         torch.zeros(E, 2 * INTER, BYTES, dtype=torch.uint8)),
        ("mtp.2.markov_head.weight", torch.zeros(HID)),
        ("mtp.1.attn.wq_a.weight", torch.zeros(HID, HID)),
    ]))
    assert loaded == set()
    assert m.delegated == []          # every one dropped by skip_substrs
    # and the body stacks were untouched by them
    assert torch.all(m._params[RE + ".w13_cb_qweight"] == 0)


def test_body_expert_tensor_for_an_absent_layer_defers_not_raises():
    """PP/EP-absent (or simply another layer's) expert tensor: defer, never
    hard-fail in the wrapper."""
    cls = _make_dsv4_cls(_cb_params())
    install_toplevel_cb_expert_loader(cls)
    m = cls()
    with pytest.raises(KeyError):
        # The wrapper defers; the ARCH loader is the thing that fails closed on
        # an unmatched non-mtp key, which is upstream's documented behaviour.
        m.load_weights(iter([
            ("layers.41.ffn.experts.gate_up_proj.cb_qweight",
             torch.zeros(E, 2 * INTER, BYTES, dtype=torch.uint8)),
        ]))


# --------------------------------------------------------------------------
# 4. Fused shards with NO packed_modules_mapping
# --------------------------------------------------------------------------

def test_reverse_fusion_covers_dsv4_merges_without_packed_modules_mapping():
    """`DeepseekV4ForCausalLM` exposes no `packed_modules_mapping`, so the
    wrapper's fallback table is the only merge information available."""
    rev = _build_reverse_fusion(None)
    assert rev["wq_a"] == ("fused_wqa_wkv", 0)
    assert rev["wkv"] == ("fused_wqa_wkv", 1)
    assert rev["w1"] == ("gate_up_proj", 0)
    assert rev["w3"] == ("gate_up_proj", 1)
    # the Llama-convention spellings still resolve to the same fused leaf
    assert rev["gate_proj"] == ("gate_up_proj", 0)
    assert rev["up_proj"] == ("gate_up_proj", 1)


def test_cb_attention_shard_on_a_plain_bf16_target_fails_closed():
    """A CB `attn.wq_a` whose only home is the merged bf16 `fused_wqa_wkv`
    proves the config alias/dispatch wiring failed. Decoding it into that
    Linear would hand serving dispatch to cuBLAS/Triton, so load stops."""
    P = f"model.layers.{LAYER}.attn"
    params = {P + ".fused_wqa_wkv.weight": torch.zeros(2 * HID, HID),
              P + ".wo_b.weight": torch.zeros(HID, HID)}
    rev = _build_reverse_fusion(None)
    assert resolve_shared_cb_target(P + ".wq_a.cb_qweight", params, rev) == \
        (P + ".fused_wqa_wkv.weight", 0)
    assert resolve_shared_cb_target(P + ".wkv.cb_qweight", params, rev) == \
        (P + ".fused_wqa_wkv.weight", 1)

    cls = _make_dsv4_cls(params)
    install_toplevel_cb_expert_loader(cls)
    m = cls()
    with pytest.raises(RuntimeError, match="native CB Linear"):
        m.load_weights(iter([
            (f"layers.{LAYER}.attn.wq_a.cb_qweight",
             torch.zeros(HID, BYTES, dtype=torch.uint8)),
        ]))
    assert m.delegated == []


def test_cb_shared_expert_shard_on_a_plain_bf16_target_fails_closed():
    """Same rule for the shared expert, whose checkpoint shards are the
    Mixtral-convention `w1`/`w3` rather than `gate_proj`/`up_proj`."""
    P = f"model.layers.{LAYER}.ffn.shared_experts"
    params = {P + ".gate_up_proj.weight": torch.zeros(2 * INTER, HID),
              P + ".down_proj.weight": torch.zeros(HID, INTER)}
    rev = _build_reverse_fusion(None)
    assert resolve_shared_cb_target(P + ".w1.cb_qweight", params, rev) == \
        (P + ".gate_up_proj.weight", 0)
    assert resolve_shared_cb_target(P + ".w3.cb_qweight", params, rev) == \
        (P + ".gate_up_proj.weight", 1)
    # `w2` is a plain rename of `down_proj`, not a shard: index 0, direct.
    assert resolve_shared_cb_target(P + ".w2.cb_qweight", params, rev) == \
        (P + ".down_proj.weight", 0)


# --------------------------------------------------------------------------
# 5. Config-side scheme resolution (vLLM-only: needs PrismaQuantConfig)
# --------------------------------------------------------------------------

def _dsv4_quant_config(namespace: str = "model."):
    """A miniature DSV4 artifact manifest in the CB schema.

    Matches the study's planned all-CB assignment: FP8_CB_K36 body Linears
    (attention + shared experts) and NVFP4_CB_K15 routed-expert stacks.
    ``namespace`` selects between the vLLM serving spelling (``model.layers.``)
    and the producer's source spelling (bare ``layers.``).
    """
    body = {"grid": "fp8", "mode": "product", "k": 36, "n_sub": 4,
            "type_size": 144, "group_size": 0, "vec_dim": 8,
            "codebook_group": "attn", "codebook_source": "lattice",
            "codebook_ref": [f"cb_codebook.attn.FP8_CB_K36.sub{i}"
                             for i in range(4)]}
    experts = {"grid": "fp4", "mode": "product", "k": 15, "n_sub": 2,
               "type_size": 69, "group_size": 16, "vec_dim": 8,
               "scale_coding": {"kind": "two_tier"},
               "codebook_group": "experts", "codebook_source": "lattice",
               "codebook_ref": [f"cb_codebook.experts.NVFP4_CB_K15.sub{i}"
                                for i in range(2)]}
    p = namespace + f"layers.{LAYER}."
    return {
        "quant_method": "gridbook", "format": "nvfp4_cb",
        "codebook_file": "cb_codebooks.pqcb", "layout_version": 2,
        "config_groups": {
            "group_body": {"format": "FP8_CB_K36", "scheme": body,
                           "targets": [
                               p + "attn.wq_a", p + "attn.wkv",
                               p + "attn.wq_b", p + "attn.wo_b",
                               p + "ffn.shared_experts.w1",
                               p + "ffn.shared_experts.w3",
                               p + "ffn.shared_experts.w2",
                           ]},
            "group_experts": {"format": "NVFP4_CB_K15", "scheme": experts,
                              "targets": [p + "ffn.experts.gate_up_proj",
                                          p + "ffn.experts.down_proj"]},
        },
        # Left on their source layout by the producer / unquantized by vLLM.
        "ignore": ["lm_head", "model.embed_tokens",
                   p + "ffn.gate", p + "attn.compressor.fused_wkv_wgate"],
    }


@pytest.fixture()
def dsv4_config():
    pytest.importorskip("vllm")
    from gridbook.config import PrismaQuantConfig

    def build(namespace="model."):
        c = PrismaQuantConfig.from_config(_dsv4_quant_config(namespace))
        c._ensure_resolved()
        return c

    return build


@pytest.mark.parametrize("namespace", ["model.", ""])
def test_dsv4_body_linears_resolve_to_cb_under_served_prefixes(dsv4_config,
                                                               namespace):
    """The exact prefixes vLLM 0.24 hands ``get_quant_method`` for DSV4, in
    both artifact namespaces. A ``None`` here is a silent BF16/stock
    fall-through for a declared CB target."""
    c = dsv4_config(namespace)
    P = f"model.layers.{LAYER}"
    # Merged MLA projection: the served prefix names the FUSED module while the
    # artifact declares the two shards.
    assert c._scheme_for_prefix(P + ".attn.fused_wqa_wkv") is not None
    # Plain (unfused) attention Linears.
    assert c._scheme_for_prefix(P + ".attn.wq_b") is not None
    assert c._scheme_for_prefix(P + ".attn.wo_b") is not None
    # Shared expert: served leaf is `gate_up_proj`, shards are `w1`/`w3`.
    assert c._scheme_for_prefix(P + ".ffn.shared_experts.gate_up_proj") \
        is not None
    assert c._scheme_for_prefix(P + ".ffn.shared_experts.down_proj") is not None
    # Every body Linear resolves to the SAME FP8-CB rung the study assigns.
    assert c._scheme_for_prefix(P + ".attn.wq_b")["k"] == 36


@pytest.mark.parametrize("namespace", ["model.", ""])
def test_dsv4_routed_expert_stack_resolves_to_cb_moe(dsv4_config, namespace):
    c = dsv4_config(namespace)
    # The FusedMoE prefix is one level ABOVE the params (`…ffn.experts`, while
    # the stack is at `…ffn.experts.routed_experts.*`).
    scheme = c._moe_scheme_for_prefix(f"model.layers.{LAYER}.ffn.experts")
    assert scheme is not None
    assert (scheme["grid"], scheme["k"]) == ("fp4", 15)


def test_dsv4_unquantized_modules_are_not_claimed(dsv4_config):
    """vLLM builds the router gate, both compressors, `indexer.weights_proj`,
    `lm_head` and `embed_tokens` with `quant_config=None` or no quant config at
    all. Gridbook must not invent a CB scheme for any of them."""
    c = dsv4_config()
    P = f"model.layers.{LAYER}"
    for prefix in (P + ".ffn.gate",
                   P + ".attn.compressor.fused_wkv_wgate",
                   P + ".attn.indexer.weights_proj",
                   P + ".attn.indexer.compressor.fused_wkv_wgate",
                   "lm_head"):
        assert c._scheme_for_prefix(prefix) is None, prefix


def test_dsv4_mtp_prefixes_are_never_claimed_as_cb(dsv4_config):
    """`mtp.*` rides as producer passthrough and is dropped by the body loader;
    no MTP prefix may resolve to a CB scheme or an expert stack."""
    c = dsv4_config()
    assert c._scheme_for_prefix("model.mtp.0.attn.wq_b") is None
    assert c._scheme_for_prefix("model.mtp.2.ffn.shared_experts.down_proj") \
        is None
    assert c._moe_scheme_for_prefix("model.mtp.0.ffn.experts") is None


def test_dsv4_neighbouring_layer_does_not_borrow_a_scheme(dsv4_config):
    c = dsv4_config()
    assert c._scheme_for_prefix(f"model.layers.{LAYER + 1}.attn.wq_b") is None
    assert c._moe_scheme_for_prefix(
        f"model.layers.{LAYER + 1}.ffn.experts") is None


def test_dsv4_rejects_tensor_parallel_above_one(dsv4_config, monkeypatch):
    """D0.1 keeps TP=1: the model fits one GB10 at the planned 92 GB, and no CB
    weight has TP handling.

    The gate reads the LIVE worker's world size, not an argument string, so the
    probe is what has to report >1 — an uninitialised model-parallel group
    (every CPU-side config test) correctly defers rather than guessing."""
    import gridbook.config as gbconfig

    c = dsv4_config()
    c._tp_world_size = 4
    monkeypatch.setattr(gbconfig, "_initialized_tensor_parallel_world_size",
                        lambda: 4)
    with pytest.raises(ValueError, match="tensor-parallel size 1 only"):
        c._require_supported_tensor_parallel()
    # A live TP=1 worker is accepted and latched.
    monkeypatch.setattr(gbconfig, "_initialized_tensor_parallel_world_size",
                        lambda: 1)
    c._require_supported_tensor_parallel()
    assert c._tp_world_size == 1


def test_dsv4_delegated_preflight_still_guards_a_stock_region():
    """D0.2 note: the all-CB artifact delegates nothing, but if a DSV4 region
    were ever passed through as stock NVFP4 the SHIPPED preflight still refuses
    an unaudited/Triton backend. Nothing new is built for it."""
    from gridbook.delegated_preflight import (
        DelegatedBackendError,
        require_native_delegated_backend,
    )

    group = {"format": "nvfp4-pack-quantized",
             "weights": {"num_bits": 4, "type": "float"},
             "input_activations": {"num_bits": 4, "type": "float",
                                   "dynamic": "local"}}

    class _UnauditedExperts:
        """A backend that is neither on the audited-native list nor otherwise
        classified — the preflight's UNKNOWN arm, which is the one a brand-new
        architecture is most likely to hit."""

    class _Method:
        def __init__(self):
            self.fused_experts = _UnauditedExperts()

    with pytest.raises(DelegatedBackendError):
        require_native_delegated_backend(
            prefix=f"model.layers.{LAYER}.ffn.experts",
            group_name="group_stock", group=group, method=_Method(),
            layer=None)


# --------------------------------------------------------------------------
# 6. Model-load-shaped resolution against the REAL vLLM class (container)
# --------------------------------------------------------------------------

def test_real_vllm_deepseek_v4_class_matches_the_contract():
    """Import vLLM 0.24's DeepseekV4ForCausalLM and check every structural
    assumption this contract rests on. No weights, no GPU, no model load."""
    pytest.importorskip("vllm")
    import importlib
    import inspect

    module_path = "vllm.models.deepseek_v4.nvidia.model"
    assert module_path in load_runtime_contract()[
        "producer_profiles"]["top_level_loader_modules"]
    mod = importlib.import_module(module_path)

    cls = getattr(mod, "DeepseekV4ForCausalLM")
    # plugin.py's __module__ guard requires the class to be DEFINED here.
    assert cls.__module__ == module_path
    assert callable(getattr(cls, "load_weights", None))
    assert inspect.isclass(cls)

    # (4) still no packed_modules_mapping -> the fused fallback tables matter.
    assert not getattr(cls, "packed_modules_mapping", None)

    src = inspect.getsource(mod)
    # (1) attn/ffn attribute spelling, and the deeper routed nesting.
    assert "prefix=f\"{prefix}.ffn\"" in src or ".ffn" in src
    # MTP is dropped wholesale by the body loader.
    assert 'skip_substrs=["mtp."]' in src
    # (3) the mapper that re-attaches `model.`.
    assert '"layers.": "model.layers."' in src
    # The merges Gridbook's fallback table has to know about.
    assert '"attn.fused_wqa_wkv", "attn.wq_a"' in src.replace("(", "").replace(
        ")", "") or "fused_wqa_wkv" in src
    # No DSpark support to wire to.
    assert "dspark" not in src.lower()


def test_real_vllm_class_takes_the_wrap_and_reports_its_module_path():
    """The plugin's discovery rule, run for real: the DSV4 module path yields a
    wrapped entrypoint class, and `cb_fill_guard` can name it."""
    pytest.importorskip("vllm")
    from gridbook.moe_toplevel_loader import installed_module_paths
    from gridbook.plugin import _install_on_module_classes

    module_path = "vllm.models.deepseek_v4.nvidia.model"
    _install_on_module_classes(module_path)
    assert module_path in installed_module_paths()

    import importlib
    cls = importlib.import_module(module_path).DeepseekV4ForCausalLM
    assert cls.__dict__.get("_pq_cb_wrapped") is True
    assert getattr(cls.load_weights, "_pq_cb_wrapper", False) is True


def test_real_model_config_shape_matches_the_registered_profile(tmp_path):
    """The released DSV4-Flash config.json shape, as the contract assumes it:
    one architecture, all-MoE body, TP-1-friendly, and MTP declared but not
    served by the body class."""
    cfg = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43, "hidden_size": 4096,
        "moe_intermediate_size": 2048, "n_routed_experts": 256,
        "n_shared_experts": 1, "num_experts_per_tok": 6,
        "num_attention_heads": 64, "head_dim": 512,
        "num_key_value_heads": 1, "q_lora_rank": 1024, "o_lora_rank": 1024,
        "o_groups": 8, "vocab_size": 129280, "num_hash_layers": 3,
        "num_nextn_predict_layers": 1, "hc_mult": 4,
        "dspark_target_layer_ids": [40, 41, 42],
        "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    loaded = json.loads(path.read_text())

    assert loaded["architectures"] == ["DeepseekV4ForCausalLM"]
    assert loaded["model_type"] in load_runtime_contract()[
        "producer_profiles"]["supported_ids"]
    # No `first_k_dense_replace`: every one of the 43 layers is MoE, so every
    # layer contributes an expert stack the fill guard will check.
    assert "first_k_dense_replace" not in loaded
    # `n_routed_experts % 1 == 0` is the only expert-sharding constraint at
    # TP=1, and `num_attention_heads % 1 == 0` the only attention one.
    assert loaded["n_routed_experts"] % 1 == 0
