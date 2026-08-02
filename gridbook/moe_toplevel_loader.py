"""Top-level stacked-CB expert loader shim.

Some MoE architectures load their experts at the **top-level** ``*ForCausalLM``
(equivalently, the top ``*Model``) ``load_weights`` via an
``expert_params_mapping`` keyed on *per-expert* checkpoint names
(``…experts.{eid}.gate_proj.``), and explicitly ``continue`` past
``mlp.experts`` in their ``stacked_params_mapping`` loop. Such models **never**
call the per-layer ``FusedMoE.load_weights`` — so the instance-level CB load
hook that ``PrismaQuantCBMoEMethod.create_weights`` installs on the FusedMoE
module (``moe.py`` ``_cb_load_weights``) is *dead code* for them.

HunYuan V3 (``HYV3ForCausalLM``) is exactly this shape. Our exporter writes
**stacked** CB expert tensors — one tensor per role holding all experts:

    model.layers.N.mlp.experts.gate_up_proj.cb_qweight   uint8 (E, 2·inter, bytes)
    model.layers.N.mlp.experts.down_proj.cb_qweight      uint8 (E, hidden,  bytes)
    model.layers.N.mlp.experts.gate_up_proj.weight_scale f32   (E, 2·inter)   # fp8 rungs only
    model.layers.N.mlp.experts.down_proj.weight_scale    f32   (E, hidden)    # fp8 rungs only
    model.layers.N.mlp.experts.gate_up_proj.input_global_scale f32 (1,)       # contracted fp4
    model.layers.N.mlp.experts.down_proj.input_global_scale    f32 (1,)       # contracted fp4

These match neither the arch's ``stacked_params_mapping`` (experts skipped) nor
its ``expert_params_mapping`` (which matches per-expert ``experts.{eid}.…``
names, not our fused+stacked ``experts.gate_up_proj.…``), so the arch's loader
falls to its final ``params_dict[name]`` and ``KeyError``s.

``PrismaQuantCBMoEMethod.create_weights`` registers these stacked tensors on the
FusedMoE module (path ``…experts``) verbatim as params ``w13_cb_qweight`` /
``w2_cb_qweight`` (+ fp8 ``w13_weight_scale`` / ``w2_weight_scale``) — SAME
shapes as the checkpoint tensors. So loading is a plain ``copy_`` with no
per-expert split and no transpose.

This module installs a thin wrapper on the top-level ``load_weights`` that
intercepts exactly those stacked-CB expert tensors, copies each into its
registered fused param, and delegates *every other* tensor (dense CB,
router.gate, expert_bias, norms, embeddings, lm_head, attention) unchanged to
the original loader. One line per arch registers it
(``install_toplevel_cb_expert_loader(SomeForCausalLM)``). A future architecture
with the same top-level expert mapping can reuse the wrapper, but it is not
supported until its model class is explicitly registered and tested.

**DeepSeek-V4 (ROADMAP D0.1).** ``DeepseekV4ForCausalLM`` is this shape and is
now registered. Two properties of vLLM 0.24's class made it resolve here with
no new per-arch loader:

* its module attributes are ``attn`` / ``ffn`` (NOT ``self_attn`` / ``mlp``),
  and the routed stack nests one level deeper again as
  ``model.layers.N.ffn.experts.routed_experts.w13_cb_qweight``. The
  ``.experts.`` prefix anchor in ``resolve_cb_expert_param`` matches on the
  *stem before* ``.experts.`` and the target *leaf*, so it is indifferent to
  both — the same property that already absorbed HunYuan V3's nesting;
* the checkpoint carries no ``model.`` component (keys start at ``layers.N.``)
  and the class re-attaches it in its own ``hf_to_vllm_mapper``
  (``{"layers.": "model.layers."}``). ``_hf_mapper_rename`` below already
  applies the model's own mapper before resolution, so
  ``layers.N.ffn.experts.gate_up_proj.cb_qweight`` lands on the registered
  param without a Gridbook-side rewrite.

Its MTP/DSpark payload does NOT come through here: ``DeepseekV4ForCausalLM.
load_weights`` builds ``AutoWeightsLoader(self, skip_substrs=["mtp."])``, so all
``mtp.*`` tensors are dropped before any parameter lookup. See the
``deepseek_v4`` notes in ``docs/PLUGIN.md``.

**Shared-expert (``shared_mlp``) CB tensors.** ``config.py`` aliases the
architecture's collapsed parent prefix (and its MTP ``.mtp_block`` form) so a
shared expert is constructed as a native CB Linear. The wrapper still detects
the structural failure mode where the checkpoint carries ``cb_qweight`` but
vLLM constructed only a plain bf16 ``.weight``. That condition now FAILS AT
LOAD. The former compatibility behavior decoded CB into that plain Linear;
upstream unquantized dispatch could then select cuBLAS or a Triton override,
which violates Gridbook's native-only serving contract.

Prefix note: we wrap the OUTERMOST class (e.g. ``HYV3ForCausalLM``), whose
incoming weight names and ``named_parameters()`` BOTH carry the ``model.``
prefix (the raw safetensors stream). The KeyError in the bug report
(``layers.1.mlp.experts.down_proj.cb_qweight``, no ``model.``) originates one
level down in ``HYV3Model.load_weights``, because ``AutoWeightsLoader`` strips
the ``model.`` prefix before delegating to the child. By intercepting at the top
level we never let those tensors reach that child, and prefix handling stays
self-consistent: both the incoming name and the mapped param name carry
``model.``. The mapping is a pure suffix rewrite, so it is prefix-agnostic
regardless; the ``params_dict`` membership check keeps it robust if a future
vLLM changes prefix handling (an unmapped name simply defers to the original).

**Spec-layer (MTP drafter) support.** A speculative-decode drafter such as
vLLM's ``HYV3MTP`` is fed the WHOLE checkpoint stream but keeps only its spec
layer(s) — ``model.layers.{num_hidden_layers + i}.*`` — renaming each layer's
transformer-block tensors into a ``.mtp_block.`` sub-module nesting via the
model's own ``_rewrite_spec_layer_name`` (``enorm``/``hnorm``/``eh_proj``/
``final_layernorm`` stay at the layer level; ``embed_tokens``/``shared_head`` are
``"__skip__"``-ped, vLLM reusing the main model's embedding + lm_head). Our
stacked-CB expert tensors arrive WITHOUT the ``.mtp_block.`` infix (checkpoint
convention) while the registered params carry it (``…mtp_block.mlp.experts.
routed_experts.w13_cb_qweight``), so the ``.experts.`` anchor here misses unless
we apply the SAME rename to the incoming name first. When the wrapped class exposes
``_rewrite_spec_layer_name`` (+ a ``config`` with ``num_hidden_layers``) we build
that rename once and apply it before resolution; a ``"__skip__"`` is delegated
so the original drops it exactly as it would, and body tensors (index <
num_hidden_layers) pass through unchanged and defer. Classes without the method
(the body ``HYV3ForCausalLM``) get an identity rename — bit-identical to the
prior behaviour.
"""
from __future__ import annotations

import torch

from .cb_fill_guard import mark_filled

# vLLM module paths this process actually installed the wrap on — diagnostics
# for the serve-time fill assertion (cb_fill_guard), which prints them so an
# unwired arch's error says exactly which list it is missing from.
_INSTALLED_MODULE_PATHS: set[str] = set()


def installed_module_paths() -> set[str]:
    """The vLLM module paths whose classes carry the top-level CB wrap."""
    return set(_INSTALLED_MODULE_PATHS)


# Checkpoint expert-tensor suffix  ->  the registered FusedMoE param LEAF name.
# The leading ``.experts.`` anchor is load-bearing: it excludes ``shared_mlp``
# and dense MLP projections that share the ``gate_up_proj`` / ``down_proj`` leaf
# names (e.g. ``…mlp.shared_mlp.gate_proj.cb_qweight``,
# ``…layers.0.mlp.down_proj.cb_qweight``), which must go to the ORIGINAL loader.
_CB_EXPERT_SUFFIX_TO_LEAF: dict[str, str] = {
    ".experts.gate_up_proj.cb_qweight": "w13_cb_qweight",
    ".experts.down_proj.cb_qweight": "w2_cb_qweight",
    ".experts.gate_up_proj.weight_scale": "w13_weight_scale",
    ".experts.down_proj.weight_scale": "w2_weight_scale",
    ".experts.gate_up_proj.input_global_scale": "w13_input_global_scale",
    ".experts.down_proj.input_global_scale": "w2_input_global_scale",
}


def resolve_cb_expert_param(name: str,
                            param_names) -> str | None:
    """Resolve a stacked-CB expert checkpoint tensor name to the registered
    FusedMoE param name, by matching against the model's ACTUAL parameter names.

    The checkpoint writes ``…mlp.experts.<proj>.cb_qweight`` (flat), but vLLM
    may nest the routed FusedMoE one level deeper — HunYuan V3's SharedFusedMoE
    registers params at ``…mlp.experts.routed_experts.w13_cb_qweight`` (the
    ``routed_experts`` sub-module holds the routed stack; ``shared_experts`` is
    fused alongside). A fixed prefix rewrite therefore misses; instead we anchor
    on the layer's ``…mlp.experts.`` prefix and the target leaf (``w13_cb_qweight``
    etc.) and find the unique registered param in between. This is robust to
    that nesting and any future one.

    Returns the resolved param name, or ``None`` when *name* is not a stacked-CB
    expert tensor OR no such param exists on this rank (both defer to the
    original loader). Raises on an ambiguous (>1) match — that would mean the
    layer structure is not what we assume, and a silent wrong copy is worse.
    """
    for suffix, leaf in _CB_EXPERT_SUFFIX_TO_LEAF.items():
        if not name.endswith(suffix):
            continue
        want_prefix = name[: -len(suffix)] + ".experts."   # …mlp.experts.
        want_suffix = "." + leaf                           # .w13_cb_qweight
        matches = [k for k in param_names
                   if k.startswith(want_prefix) and k.endswith(want_suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"prismaquant CB expert '{name}': ambiguous target params "
                f"{matches} — layer structure differs from the "
                "one-FusedMoE-per-layer assumption")
        return None                                        # missing on rank
    return None


# Back-compat pure-suffix mapper (no params context); the loader uses the
# params-aware resolver above. Kept for the unit tests that assert the
# suffix/exclusion logic in isolation.
def map_cb_expert_name(name: str) -> str | None:
    for suffix, leaf in _CB_EXPERT_SUFFIX_TO_LEAF.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".experts." + leaf
    return None


# ---------------------------------------------------------------------------
# Shared-expert (shared_mlp) native-ownership assertion. A plain-BF16 target is
# detected only to fail closed; it is never populated from a CB checkpoint.
# ---------------------------------------------------------------------------

_CB_QWEIGHT_SUFFIX = ".cb_qweight"
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
_INPUT_GLOBAL_SCALE_SUFFIX = ".input_global_scale"

# Standard vLLM fusions, as a fallback when the model class exposes no
# packed_modules_mapping (only the leaves we route matter here).
#
# DeepSeek-V4 is exactly the "exposes none" case — `DeepseekV4ForCausalLM`
# defines no `packed_modules_mapping` at all (verified against vLLM 0.24.0), so
# this table is the ONLY source of merge information the orphan guard below
# has for it. Its two merges are `attn.fused_wqa_wkv` <- (`attn.wq_a`,
# `attn.wkv`) and the shared expert's `gate_up_proj` <- (`w1`, `w3`); the
# Mixtral-convention shard leaves coexist with the Llama-convention ones
# because the reverse map is keyed by SHARD leaf, which stays unambiguous.
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_wqa_wkv": ["wq_a", "wkv"],
}

# Extra shard spellings for an ALREADY-listed fused leaf, merged into the
# reverse map (which is keyed by shard leaf, so these never collide with the
# primary spelling above).
_FUSED_SHARD_ALIASES = {
    "w1": ("gate_up_proj", 0),
    "w3": ("gate_up_proj", 1),
}


def _build_reverse_fusion(packed_modules_mapping) -> dict[str, tuple[str, int]]:
    """``{shard_leaf: (fused_leaf, shard_index)}`` from a
    ``packed_modules_mapping`` (``{fused_leaf: [shard_leaf, ...]}``), unioned with
    the standard qkv/gate_up fallback. Routes an unfused checkpoint projection
    (``gate_proj``/``up_proj``) to vLLM's fused bf16 param (``gate_up_proj``) at
    the correct row block (gate=0, up=1)."""
    merged = dict(_FUSED_FALLBACK)
    merged.update(packed_modules_mapping or {})
    rev: dict[str, tuple[str, int]] = dict(_FUSED_SHARD_ALIASES)
    for fused_leaf, shards in merged.items():
        if isinstance(shards, (list, tuple)):
            for idx, shard_leaf in enumerate(shards):
                rev[shard_leaf] = (fused_leaf, idx)
    return rev


def resolve_shared_cb_target(name: str, params_dict,
                             reverse_fusion) -> tuple[str, int] | None:
    """Resolve a checkpoint CB tensor (packed weight or either scale) to
    the bf16 ``.weight`` param vLLM built for a shared-expert-style Linear, or
    ``None`` when it is NOT such a tensor (delegate to the original loader).

    Structural test (no hard-coded ``shared_mlp``): intercept ONLY when vLLM built
    no CB param for the module — neither the fused target's ``.cb_qweight`` nor the
    module's own ``.cb_qweight`` is registered — but a bf16 ``.weight`` target IS.
    Genuine dense-CB Linears and fused-attention shards (``q/k/v_proj`` ->
    ``qkv_proj``) have a registered ``.cb_qweight`` and therefore defer. A tensor
    whose target is entirely absent on this rank also defers.

    Returns ``(target_weight_param_name, shard_index)`` — ``shard_index`` orders
    the parts of a merged param (gate=0, up=1); 0 for a direct (unfused) Linear.
    """
    if name.endswith(_CB_QWEIGHT_SUFFIX):
        base = name[: -len(_CB_QWEIGHT_SUFFIX)]
        scalar_suffix = None
    elif name.endswith(_WEIGHT_SCALE_SUFFIX):
        base = name[: -len(_WEIGHT_SCALE_SUFFIX)]
        scalar_suffix = _WEIGHT_SCALE_SUFFIX
    elif name.endswith(_INPUT_GLOBAL_SCALE_SUFFIX):
        base = name[: -len(_INPUT_GLOBAL_SCALE_SUFFIX)]
        scalar_suffix = _INPUT_GLOBAL_SCALE_SUFFIX
    else:
        return None
    parent, _, leaf = base.rpartition(".")
    # Fused-sibling case: gate_proj/up_proj -> gate_up_proj, q/k/v -> qkv_proj.
    if parent and leaf in reverse_fusion:
        fused_leaf, shard = reverse_fusion[leaf]
        fbase = parent + "." + fused_leaf
        if fbase + _CB_QWEIGHT_SUFFIX in params_dict:
            return None                       # vLLM built a fused CB Linear here
        # A weight_scale whose OWN param exists is a stock-CT tensor (a joint
        # menu can put a shared-expert Linear on vanilla fp8: it has .weight AND
        # .weight_scale params) — never an orphaned CB scale. Defer.
        if (scalar_suffix is not None
                and fbase + scalar_suffix in params_dict):
            return None
        if fbase + ".weight" in params_dict:
            return fbase + ".weight", shard   # merged bf16 target (gate|up)
    # Direct (unfused) case: down_proj, or a non-fused projection.
    if base + _CB_QWEIGHT_SUFFIX in params_dict:
        return None                           # vLLM built a CB Linear here
    if scalar_suffix is not None and base + scalar_suffix in params_dict:
        return None                           # stock-CT scale with a real home
    if base + ".weight" in params_dict:
        return base + ".weight", 0
    return None                               # target absent on this rank


# ---------------------------------------------------------------------------
# Spec-layer (MTP drafter) name rewrite — see the module docstring.
# ---------------------------------------------------------------------------

def _spec_layer_of(name: str, spec_layers) -> int | None:
    """The spec-layer index owning *name* (``model.layers.{idx}.…``), or None.
    Mirrors vLLM ``get_spec_layer_idx_from_weight_name`` (a ``startswith``
    test against ``model.layers.{idx}.``)."""
    for s in spec_layers:
        if name.startswith(f"model.layers.{s}."):
            return s
    return None


def _spec_layer_rename(model):
    """A ``name -> resolution_name`` callable for a spec-layer (MTP) drafter, or
    ``None`` when *model* is not one.

    Detection is structural: the model must expose ``_rewrite_spec_layer_name``
    (the arch's own rewrite) and a ``config`` with ``num_hidden_layers``. The
    spec layers are ``range(num_hidden_layers, num_hidden_layers + n)`` with
    ``n = num_nextn_predict_layers`` — exactly how ``HYV3MTP`` derives them. The
    returned callable applies the model's rewrite to a spec-layer name (which may
    yield ``"__skip__"``) and returns any other name unchanged, so body tensors
    pass through untouched and defer to the original loader."""
    rewrite = getattr(model, "_rewrite_spec_layer_name", None)
    config = getattr(model, "config", None)
    if rewrite is None or config is None:
        return None
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        return None
    n_spec = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    if n_spec <= 0:
        return None
    spec_layers = tuple(range(int(n_layers), int(n_layers) + n_spec))

    def rename(name: str) -> str:
        s = _spec_layer_of(name, spec_layers)
        return rewrite(s, name) if s is not None else name

    return rename


def _hf_mapper_rename(model):
    """A ``checkpoint name -> vLLM module-tree name`` callable built from the
    model's OWN ``hf_to_vllm_mapper`` (a vLLM ``WeightsMapper``), or ``None``
    when the class has none.

    Needed for multimodal wrappers such as Qwen3.5-MoE
    (``Qwen3_5MoeForConditionalGeneration``), whose checkpoint carries
    ``model.language_model.layers.N.…`` while ``named_parameters()`` reads
    ``language_model.model.layers.N.…``. The original ``load_weights`` applies
    the mapper *inside* itself (``AutoWeightsLoader(..., mapper=...)``), i.e.
    AFTER our wrapper has seen the raw stream — so without this the
    ``…mlp.experts.`` prefix anchor matches no registered param and every
    stacked-CB expert tensor falls through to the arch loader (the bug).

    We reuse the model's own mapper rather than hard-coding a prefix rewrite, so
    any arch/prefix convention is handled by definition. A mapper that maps a
    name to ``None`` means "drop"; we return the name unchanged so the original
    loader performs the drop exactly as it would."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    map_name = getattr(mapper, "_map_name", None)
    if not callable(map_name):
        return None

    def rename(name: str) -> str:
        mapped = map_name(name)
        return name if mapped is None else mapped

    return rename


def _compose_renames(*renames):
    """Left-to-right composition of the non-``None`` renames, or ``None`` when
    all are absent. A ``"__skip__"`` short-circuits (it is a terminal marker)."""
    fns = [f for f in renames if f is not None]
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]

    def rename(name: str) -> str:
        for f in fns:
            if name == "__skip__":
                return name
            name = f(name)
        return name

    return rename


def install_toplevel_cb_expert_loader(model_cls: type) -> None:
    """Idempotently wrap ``model_cls.load_weights`` so stacked-CB expert tensors
    load directly into the registered FusedMoE params, and everything else
    delegates to the original loader.

    Safe to call repeatedly (guarded by a ``_pq_cb_wrapped`` class sentinel) and
    safe if the model has no CB experts at serve time (the wrapper only fires on
    matching names; all others pass straight through)."""
    # Sentinel checked in the class's OWN __dict__: a subclass that defines its
    # own ``load_weights`` must still be wrapped even though it inherits a
    # wrapped base's sentinel (Qwen3_5 has a whole ForCausalLMBase /
    # ForConditionalGeneration hierarchy). If the class merely INHERITS an
    # already-wrapped function there is nothing to do.
    if model_cls.__dict__.get("_pq_cb_wrapped", False):
        _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
        return
    orig_load_weights = model_cls.load_weights
    if getattr(orig_load_weights, "_pq_cb_wrapper", False):
        model_cls._pq_cb_wrapped = True
        _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
        return

    def load_weights(self, weights):  # noqa: ANN001, ANN202
        # named_parameters() here carries the same module-nesting prefix as the
        # incoming checkpoint names (both ``model.…`` at the top level), so the
        # suffix-rewritten target is a direct key.
        params_dict = dict(self.named_parameters())
        param_names = tuple(params_dict)
        reverse_fusion = _build_reverse_fusion(
            getattr(self, "packed_modules_mapping", None))
        # Spec-layer (MTP) drafters nest a spec layer's block tensors under
        # ``.mtp_block.``; ``rename`` maps an incoming checkpoint name to the
        # served param naming before resolution (identity for a body model).
        # Multimodal wrappers rename the checkpoint prefix
        # (``model.language_model.`` -> ``language_model.model.``) via their own
        # WeightsMapper, which the original loader applies only internally;
        # apply it FIRST so our anchors match the registered param names.
        rename = _compose_renames(_hf_mapper_rename(self),
                                  _spec_layer_rename(self))
        loaded: set[str] = set()
        def _passthrough():
            # A generator (not a materialized list): only one checkpoint tensor
            # is live at a time, preserving the streaming/mmap semantics the
            # original loader relies on for a 100 GB+ model. CB expert tensors
            # are copied inline as a side effect and recorded in ``loaded``;
            # every other tensor is yielded on to the original loader.
            for name, w in weights:
                # Resolve against the served param naming (spec-layer rename or
                # identity). A ``"__skip__"`` — the spec drafter reusing the main
                # model's embed/lm_head — is delegated so the original drops it.
                res_name = rename(name) if rename is not None else name
                if res_name == "__skip__":
                    yield name, w
                    continue
                mapped = resolve_cb_expert_param(res_name, param_names)
                if mapped is not None:
                    param = params_dict[mapped]
                    if tuple(param.shape) != tuple(w.shape):
                        raise ValueError(
                            f"prismaquant CB expert '{name}' -> '{mapped}': "
                            f"checkpoint shape {tuple(w.shape)} != param "
                            f"shape {tuple(param.shape)} — stacked "
                            "(E, out, bytes) contract violated")
                    param.data.copy_(w.to(param.dtype))
                    # fill path 2 of 2 (top-level): stamp the sentinel that
                    # process_weights_after_loading checks (cb_fill_guard).
                    if mapped.endswith("cb_qweight"):
                        mark_filled(param)
                    loaded.add(mapped)
                    continue
                # A CB tensor targeting a plain bf16 Linear proves that config
                # prefix resolution failed. The former compatibility path
                # decoded the weight into that Linear, whose upstream serving
                # dispatch could select cuBLAS or Triton. Fail during load so a
                # nominal Gridbook serve can never change kernel family.
                orphan = resolve_shared_cb_target(
                    res_name, params_dict, reverse_fusion)
                if orphan is not None:
                    raise RuntimeError(
                        f"prismaquant CB tensor '{name}' resolved to plain "
                        f"bf16 parameter '{orphan[0]}'. Gridbook requires a "
                        "native CB Linear for every quantized shared expert; "
                        "fix the architecture prefix alias/loader wiring. "
                        "Decode-to-bf16 serving fallback is forbidden.")
                # Not a stacked-CB expert tensor (or the target param is absent
                # on this rank: PP/EP-missing, or an MTP/spec layer the
                # original's own filter drops). Delegate to the original loader.
                yield name, w

        # Some loaders return the loaded-name set (HYV3ForCausalLM), others
        # return None (HYV3MTP's drafter loader) — tolerate both; our own
        # `loaded` set still reports the tensors WE placed.
        ret = orig_load_weights(self, _passthrough())
        if ret:
            loaded |= set(ret)
        return loaded

    load_weights._pq_cb_wrapper = True
    model_cls.load_weights = load_weights
    model_cls._pq_cb_wrapped = True
    _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
