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

The target body's MTP payload does NOT come through here:
``DeepseekV4ForCausalLM.load_weights`` builds
``AutoWeightsLoader(self, skip_substrs=["mtp."])``, so all ``mtp.*`` tensors
are dropped before any parameter lookup.  The separate DSpark draft class does
load that payload.  Its constructor and loader deliberately use different
namespaces: quantization dispatch sees construction prefixes
``model.layers.{num_hidden_layers + i}``, while registered parameters are
``model.layers.{i}`` and checkpoint tensors are ``mtp.{i}``.  For interception
only, the wrapper calls the draft model's own ``_remap_dspark_name`` before the
existing mixed-fused / expert resolvers.  Any tensor those Gridbook paths do
not own is delegated under its original ``mtp.*`` name so DSpark's stock loader
remains the sole owner of ordinary dense, head, and passthrough loading.  See
the ``deepseek_v4`` notes in ``docs/PLUGIN.md``.

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

from contextvars import ContextVar
from functools import wraps
import os
import re
from typing import Any
import torch

from .cb_fill_guard import mark_filled

# vLLM module paths this process actually installed the wrap on — diagnostics
# for the serve-time fill assertion (cb_fill_guard), which prints them so an
# unwired arch's error says exactly which list it is missing from.
_INSTALLED_MODULE_PATHS: set[str] = set()

# Versioned capability stamp carried by the effective ``load_weights`` method.
# Its paired __init__ gate verifies ``type(self)`` and authorizes config-side
# composite dispatch only while that exact instance constructs its layers.
MIXED_FUSED_LOADER_ABI = 1
_ACTIVE_MIXED_FUSED_LOADER_ABI: ContextVar[int | None] = ContextVar(
    "gridbook_active_mixed_fused_loader_abi", default=None
)
# vLLM's DSpark loader intentionally keeps the target ModelConfig on the
# current VllmConfig while it constructs the separate draft model.  That is
# correct for DSpark's target-layer metadata, but it is not the source of the
# draft checkpoint's Gridbook sidecars.  Bind the *explicit* speculative draft
# ModelConfig only around the exact DSpark construction lifetime; config.py
# reads this lazily and every non-DSpark class retains the historical global
# model_config path.
_ACTIVE_DSPARK_DRAFT_MODEL_CONFIG: ContextVar[Any | None] = ContextVar(
    "gridbook_active_dspark_draft_model_config", default=None
)
_DSPARK_DRAFT_MODULE = "vllm.models.deepseek_v4.nvidia.dspark"


def mixed_fused_loader_active() -> bool:
    """Whether the model currently constructing has the loader ABI."""

    return _ACTIVE_MIXED_FUSED_LOADER_ABI.get() == MIXED_FUSED_LOADER_ABI


def active_dspark_draft_model_config() -> Any | None:
    """Return the draft ModelConfig during structural DSpark construction.

    The value is deliberately unavailable before and after ``__init__``.  A
    Gridbook config that resolves a pointer sidecar during that construction
    can therefore select the draft source without making the mere presence of
    ``speculative_config`` affect target/body model loading.
    """

    return _ACTIVE_DSPARK_DRAFT_MODEL_CONFIG.get()


def _is_dspark_draft_construction(model: Any) -> bool:
    """Whether this exact registered model class is the DSpark draft."""

    cls = type(model)
    return (
        cls.__module__ == _DSPARK_DRAFT_MODULE
        and callable(getattr(cls, "_remap_dspark_name", None))
    )


def _require_dspark_draft_model_config() -> Any:
    """Resolve the sole sidecar authority for a DSpark draft, fail closed."""

    try:
        from vllm.config import get_current_vllm_config

        current = get_current_vllm_config()
    except Exception as exc:  # noqa: BLE001 - convert context absence clearly
        raise RuntimeError(
            "DSpark Gridbook construction has no current vLLM config from "
            "which to resolve speculative_config.draft_model_config"
        ) from exc

    speculative_config = getattr(current, "speculative_config", None)
    draft_config = getattr(speculative_config, "draft_model_config", None)
    if draft_config is None:
        raise RuntimeError(
            "DSpark Gridbook construction requires vLLM "
            "speculative_config.draft_model_config"
        )
    try:
        draft_source = os.fspath(draft_config.model)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "DSpark speculative_config.draft_model_config.model is not a "
            "filesystem path or Hub ID"
        ) from exc
    if not isinstance(draft_source, str) or not draft_source:
        raise RuntimeError(
            "DSpark speculative_config.draft_model_config.model is not a "
            "nonempty string path or Hub ID"
        )
    return draft_config


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

_SPLIT_CB_EXPERT_RE = re.compile(
    r"^(?P<parent>.*[.]experts)[.]"
    r"(?P<projection>gate_up_proj|down_proj)[.]"
    r"(?P<group>format_group_[a-z0-9_]+)[.]"
    r"(?P<plane>cb_qweight|weight_scale|input_global_scale)$"
)


def _split_cb_expert_leaf(name: str) -> tuple[str, str] | None:
    """Return ``(expert parent, registered leaf)`` for a v1 sub-stack."""

    match = _SPLIT_CB_EXPERT_RE.match(name)
    if match is None:
        return None
    family = "w13" if match.group("projection") == "gate_up_proj" else "w2"
    plane = match.group("plane")
    leaf = (f"{family}_input_global_scale" if plane == "input_global_scale"
            else f"{family}_{match.group('group')}_{plane}")
    return match.group("parent"), leaf


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
    split = _split_cb_expert_leaf(name)
    if split is not None:
        parent, leaf = split
        want_prefix = parent + "."
        want_suffix = "." + leaf
        matches = [key for key in param_names
                   if key.startswith(want_prefix) and key.endswith(want_suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"prismaquant split CB expert {name!r}: ambiguous target "
                f"params {matches}"
            )
        return None
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
    split = _split_cb_expert_leaf(name)
    if split is not None:
        parent, leaf = split
        return parent + "." + leaf
    for suffix, leaf in _CB_EXPERT_SUFFIX_TO_LEAF.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".experts." + leaf
    return None


_SOURCE_SPLIT_RE = re.compile(
    r"^(?P<parent>(?:.*[.])?experts)[.](?P<expert>\d+)[.]"
    r"(?P<leaf>w1|w3|w2|gate_proj|up_proj|down_proj)[.]"
    r"(?P<plane>weight|scale)$"
)


def load_source_split_expert(name: str, weight: torch.Tensor,
                             params_dict) -> str | None:
    """Load one producer-verbatim MXFP4 slice into a scoped native subgroup.

    Parameters owned by the mixed method carry their ordered global expert-id
    tuple. The mapped ``...experts`` parent anchors the exact decoder layer,
    then that tuple supplies global->local and gate/up shard mapping. Both are
    required: many layers legitimately contain the same expert id and leaf.
    """

    match = _SOURCE_SPLIT_RE.search(name)
    if match is None:
        return None
    expert_id = int(match.group("expert"))
    parent = match.group("parent")
    leaf, plane = match.group("leaf"), match.group("plane")
    family = "w13" if leaf in ("w1", "w3", "gate_proj", "up_proj") else "w2"
    param_leaf = f"{family}_{'weight_scale' if plane == 'scale' else 'weight'}"
    candidates = []
    for param_name, param in params_dict.items():
        expert_ids = getattr(param, "_gridbook_source_expert_ids", None)
        if (expert_ids is not None and expert_id in expert_ids
                and param_name.startswith(parent + ".")
                and param_name.endswith("." + param_leaf)):
            candidates.append((param_name, param, tuple(expert_ids)))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"source split expert {name!r}: ambiguous delegated params "
            f"{[item[0] for item in candidates]}"
        )
    param_name, param, expert_ids = candidates[0]
    local = expert_ids.index(expert_id)
    if family == "w13":
        shard = 0 if leaf in ("w1", "gate_proj") else 1
        rows = param.shape[1] // 2
        destination = param.data[local, shard * rows:(shard + 1) * rows]
    else:
        destination = param.data[local]
    if tuple(destination.shape) != tuple(weight.shape):
        raise ValueError(
            f"source split expert {name!r} -> {param_name!r}: checkpoint "
            f"shape {tuple(weight.shape)} != delegated slice "
            f"{tuple(destination.shape)}"
        )
    incoming = (
        weight.contiguous().view(destination.dtype)
        if weight.element_size() == destination.element_size()
        else weight.to(destination.dtype)
    )
    destination.copy_(incoming)
    return param_name


# ---------------------------------------------------------------------------
# Shared-expert (shared_mlp) native-ownership assertion. A plain-BF16 target is
# detected only to fail closed; it is never populated from a CB checkpoint.
# ---------------------------------------------------------------------------

_CB_QWEIGHT_SUFFIX = ".cb_qweight"
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
_INPUT_GLOBAL_SCALE_SUFFIX = ".input_global_scale"

# Independently encoded dense fusions register one private carrier per
# checkpoint sibling (``mixed_linear.py``).  The architecture's ordinary loader
# cannot name those nested parameters, so this wrapper routes their planes by
# metadata.  Keep the strings local: this module deliberately remains
# torch-only and must not import the vLLM-bound mixed method just to inspect a
# parameter.
_MIXED_GROUP_ATTR = "_gridbook_mixed_fused_group"
_MIXED_SOURCE_ATTR = "_gridbook_mixed_fused_source"
_MIXED_PLANE_ATTR = "_gridbook_mixed_fused_plane"
_MIXED_FILLED_ATTR = "_gridbook_mixed_fused_filled"
_MIXED_CARRIER_MARKER = "._gridbook_mixed_roles."


def _registered_mixed_source(param_name: str, group: str,
                             source: str) -> str | None:
    """Derive a role's registered source name from its actual parameter path.

    DSpark constructs a fused module under ``model.layers.43/44/45`` but its
    three-element ModuleList registers the same modules under layers ``0/1/2``.
    Mixed-carrier metadata truthfully retains the construction prefix, while
    ``named_parameters()`` is authoritative for the registered prefix used by
    the loader.  The carrier marker is the structural join between those two
    facts.  No layer offset is inferred here.

    A standalone carrier fixture can have no owning-module prefix; in that
    case there is no registered alias to derive and the construction route is
    retained unchanged.  Any partially matching or internally inconsistent
    full path fails closed.
    """
    if _MIXED_CARRIER_MARKER not in param_name:
        return None
    registered_group, marker, _ = param_name.partition(_MIXED_CARRIER_MARKER)
    if not marker or not registered_group:
        raise ValueError(
            f"mixed fused parameter {param_name!r} has malformed carrier path"
        )
    if registered_group == group:
        return source

    construction_parent, separator, construction_leaf = group.rpartition(".")
    registered_parent, registered_separator, registered_leaf = \
        registered_group.rpartition(".")
    if (not separator or not registered_separator
            or construction_leaf != registered_leaf):
        raise ValueError(
            f"mixed fused parameter {param_name!r} is registered under "
            f"{registered_group!r}, incompatible with construction group "
            f"{group!r}"
        )
    construction_stem = construction_parent + "."
    if not source.startswith(construction_stem):
        raise ValueError(
            f"mixed fused source {source!r} is not a sibling of construction "
            f"group {group!r}"
        )
    return registered_parent + "." + source[len(construction_stem):]


def _mixed_plane_candidates(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Split a mapped checkpoint name into role prefix + possible planes.

    DeepSeek's real mapper rewrites source ``.scale`` to
    ``.weight_scale_inv``.  Keeping ``.scale`` as an accepted wire spelling
    makes the transaction resolver robust to loaders that call the wrapper
    before applying that final regex (and keeps the CPU model stub faithful).
    It is still resolved only against explicitly registered carrier metadata.
    """

    for suffix in (".input_global_scale", ".weight_scale_inv",
                   ".weight_scale", ".cb_qweight", ".weight"):
        if name.endswith(suffix):
            return name[:-len(suffix)], (suffix[1:],)
    if name.endswith(".scale"):
        return name[:-len(".scale")], ("weight_scale_inv", "weight_scale")
    return None


class _MixedFusedTransactions:
    """Stage and atomically commit every plane of one fused module.

    A fused role may need differently-shaped packed bytes and scale metadata
    from its sibling.  Copying a tensor as soon as it appears leaves a
    half-populated module if another plane is absent or malformed.  This router
    retains the streaming tensor references until *all* registered planes of a
    fused module are present, validates every destination first, then performs
    the copies as one load transaction.  ``finish`` and the mixed method's
    independent post-load fill gate both reject an incomplete stream.
    """

    def __init__(self, params_dict) -> None:
        self._params = params_dict
        self._routes: dict[tuple[str, str], str] = {}
        self._expected: dict[str, set[str]] = {}
        for param_name, param in params_dict.items():
            group = getattr(param, _MIXED_GROUP_ATTR, None)
            source = getattr(param, _MIXED_SOURCE_ATTR, None)
            plane = getattr(param, _MIXED_PLANE_ATTR, None)
            if group is None and source is None and plane is None:
                continue
            if not all(isinstance(value, str) and value
                       for value in (group, source, plane)):
                raise ValueError(
                    f"mixed fused parameter {param_name!r} has incomplete "
                    "routing metadata")
            registered_source = _registered_mixed_source(
                param_name, group, source
            )
            route_sources = {source}
            if registered_source is not None:
                route_sources.add(registered_source)
            for route_source in route_sources:
                route = (route_source, plane)
                previous = self._routes.setdefault(route, param_name)
                if previous != param_name:
                    raise ValueError(
                        f"mixed fused route {route!r} is ambiguous between "
                        f"{previous!r} and {param_name!r}")
            self._expected.setdefault(group, set()).add(param_name)
        self._pending: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
        self._committed: set[str] = set()

    def stage(self, name: str, weight: torch.Tensor
              ) -> tuple[str, ...] | None:
        """Stage one plane; return committed params, ``()`` or ``None``.

        ``None`` means the tensor is not owned by a mixed fused carrier and the
        caller must delegate it.  An empty tuple means it was consumed but its
        module transaction is still waiting for other planes.
        """

        split = _mixed_plane_candidates(name)
        if split is None:
            return None
        source, planes = split
        matches = [self._routes[(source, plane)] for plane in planes
                   if (source, plane) in self._routes]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(
                f"mixed fused tensor {name!r} matches multiple carrier "
                f"parameters {matches}")
        param_name = matches[0]
        param = self._params[param_name]
        group = getattr(param, _MIXED_GROUP_ATTR)
        if group in self._committed:
            raise ValueError(
                f"mixed fused tensor {name!r} arrived after module {group!r} "
                "was already committed")
        pending = self._pending.setdefault(group, {})
        if param_name in pending:
            raise ValueError(
                f"mixed fused parameter {param_name!r} received duplicate "
                f"checkpoint planes {pending[param_name][0]!r} and {name!r}")
        if tuple(param.shape) != tuple(weight.shape):
            raise ValueError(
                f"mixed fused tensor {name!r} -> {param_name!r}: checkpoint "
                f"shape {tuple(weight.shape)} != parameter shape "
                f"{tuple(param.shape)}")
        pending[param_name] = (name, weight)

        expected = self._expected[group]
        if set(pending) != expected:
            return ()

        # Prepare and validate every source before mutating any destination.
        prepared: dict[str, torch.Tensor] = {}
        for target in sorted(expected):
            destination = self._params[target]
            incoming = pending[target][1]
            if incoming.dtype == destination.dtype:
                converted = incoming
            elif ({incoming.dtype, destination.dtype}
                  == {torch.float8_e8m0fnu, torch.uint8}):
                # vLLM stores some native MX scale parameters as uint8 while
                # safetensors names their wire dtype F8_E8M0.  Those are the
                # same exponent bytes and must be reinterpreted, not converted.
                converted = incoming.contiguous().view(destination.dtype)
            else:
                raise ValueError(
                    f"mixed fused tensor {pending[target][0]!r} -> "
                    f"{target!r}: checkpoint dtype {incoming.dtype} != "
                    f"parameter dtype {destination.dtype}; only the audited "
                    "F8_E8M0/uint8 raw-scale spelling may be reinterpreted"
                )
            if tuple(converted.shape) != tuple(destination.shape):
                raise ValueError(
                    f"mixed fused tensor {pending[target][0]!r} changed shape "
                    f"during dtype preparation: {tuple(converted.shape)} != "
                    f"{tuple(destination.shape)}")
            prepared[target] = converted
        for target in sorted(expected):
            destination = self._params[target]
            destination.data.copy_(prepared[target])
            setattr(destination, _MIXED_FILLED_ATTR, True)
        self._committed.add(group)
        del self._pending[group]
        return tuple(sorted(expected))

    def finish(self) -> None:
        missing_groups = sorted(set(self._expected) - self._committed)
        if not missing_groups:
            return
        detail = []
        for group in missing_groups:
            present = set(self._pending.get(group, {}))
            missing = sorted(self._expected[group] - present)
            detail.append(f"{group}: missing {missing}")
        raise RuntimeError(
            "incomplete mixed fused checkpoint transactions; " +
            "; ".join(detail))

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

# Extra checkpoint spellings, merged into the reverse map (which is keyed by
# SHARD leaf, so these never collide with the primary spellings above). `w1`/`w3`
# are DeepSeek-V4's shared-expert shards of the merged `gate_up_proj`; `w2` is
# not a shard at all but the plain rename of that expert's `down_proj`, entered
# at index 0 exactly as a direct (unfused) Linear would be. Without it an
# orphaned CB `w2` would defer and surface as the arch loader's bare KeyError
# instead of this module's "needs a native CB Linear" diagnosis.
_FUSED_SHARD_ALIASES = {
    "w1": ("gate_up_proj", 0),
    "w3": ("gate_up_proj", 1),
    "w2": ("down_proj", 0),
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


def _dspark_rename(model):
    """Return DSpark's physical-to-registered name mapping, when available.

    The DSpark checkpoint uses ``mtp.{stage}.*`` while the live draft module
    registers those parameters below ``model.layers.{stage}.*`` (with a few
    model-level heads).  The model's own ``_remap_dspark_name`` is the
    authoritative mapping and is intentionally reused instead of duplicated
    here.  ``None`` from that method means the stock loader owns or drops the
    tensor; preserving the original name makes the Gridbook wrapper inert for
    those cases and lets delegation retain DSpark's exact semantics.

    Detection is structural rather than tied to a vLLM version or class name:
    a callable ``_remap_dspark_name`` is the complete capability contract.
    """
    remap = getattr(model, "_remap_dspark_name", None)
    if not callable(remap):
        return None

    def rename(name: str) -> str:
        if not name.startswith("mtp."):
            return name
        mapped = remap(name)
        return name if mapped is None else mapped

    return rename


def _validate_dspark_target_bridge_model(model) -> None:
    """Match a declared bridge topology to the instantiated DSpark model.

    Config validation proves the map is internally consistent; this load-time
    gate proves its producer-stamped ``L``/``n`` are the topology the selected
    runtime actually constructed.  It runs before the checkpoint generator is
    consumed, so a mismatch cannot leave a partially populated draft.
    """
    quant_config = getattr(model, "quant_config", None)
    expected = getattr(
        quant_config, "_dspark_target_bridge_topology", None
    )
    if expected is None:
        return
    if not callable(getattr(model, "_remap_dspark_name", None)):
        raise RuntimeError(
            "dspark_target_bridge was declared for a model without callable "
            "_remap_dspark_name"
        )
    config = getattr(model, "config", None)
    inner = getattr(model, "model", None)
    live_hidden = getattr(config, "num_hidden_layers", None)
    live_mtp = getattr(inner, "num_dspark_layers", None)
    if (isinstance(live_hidden, bool) or not isinstance(live_hidden, int)
            or isinstance(live_mtp, bool) or not isinstance(live_mtp, int)):
        raise RuntimeError(
            "dspark_target_bridge requires instantiated model topology "
            "config.num_hidden_layers and model.num_dspark_layers"
        )
    live = (live_hidden, live_mtp)
    if tuple(expected) != live:
        raise RuntimeError(
            f"dspark_target_bridge topology {tuple(expected)} does not match "
            f"the instantiated DSpark model topology {live}"
        )


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


def _install_mixed_fused_construction_gate(model_cls: type) -> None:
    """Authorize composite dispatch only while this exact instance builds.

    The effective loader is checked through ``type(self)`` on every
    construction. Thus a subclass that simply inherits the wrapped loader and
    initializer remains authorized, while a subclass that overrides
    ``load_weights`` without installing Gridbook's wrapper is rejected before
    any nested carrier can be registered. ContextVar tokens make nested model
    construction and concurrent target/draft builds restore their prior state.
    """

    orig_init = model_cls.__init__
    if (getattr(orig_init, "_gridbook_mixed_fused_init_abi", None)
            == MIXED_FUSED_LOADER_ABI):
        return

    @wraps(orig_init)
    def init(self, *args, **kwargs):  # noqa: ANN001, ANN202
        effective_loader = getattr(type(self), "load_weights", None)
        loader_abi = getattr(
            effective_loader, "_gridbook_mixed_fused_loader_abi", None)
        if loader_abi != MIXED_FUSED_LOADER_ABI:
            raise RuntimeError(
                f"{type(self).__module__}.{type(self).__name__} inherits a "
                "Gridbook mixed-fused construction gate but its effective "
                f"load_weights method has ABI {loader_abi!r}, expected "
                f"{MIXED_FUSED_LOADER_ABI}; install the top-level loader "
                "wrapper on this overriding class"
            )
        token = _ACTIVE_MIXED_FUSED_LOADER_ABI.set(loader_abi)
        dspark_token = None
        try:
            # A callable remapper is DSpark's structural construction
            # capability.  No target/body class exposes it, so speculative
            # configuration alone can never redirect an ordinary Gridbook
            # checkpoint to the draft's sidecars.
            if _is_dspark_draft_construction(self):
                draft_config = _require_dspark_draft_model_config()
                dspark_token = _ACTIVE_DSPARK_DRAFT_MODEL_CONFIG.set(
                    draft_config
                )
            return orig_init(self, *args, **kwargs)
        finally:
            if dspark_token is not None:
                _ACTIVE_DSPARK_DRAFT_MODEL_CONFIG.reset(dspark_token)
            _ACTIVE_MIXED_FUSED_LOADER_ABI.reset(token)

    init._gridbook_mixed_fused_init_abi = MIXED_FUSED_LOADER_ABI
    model_cls.__init__ = init


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
        _install_mixed_fused_construction_gate(model_cls)
        _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
        return
    orig_load_weights = model_cls.load_weights
    if getattr(orig_load_weights, "_pq_cb_wrapper", False):
        model_cls._pq_cb_wrapped = True
        _install_mixed_fused_construction_gate(model_cls)
        _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
        return

    def load_weights(self, weights):  # noqa: ANN001, ANN202
        _validate_dspark_target_bridge_model(self)
        # named_parameters() here carries the same module-nesting prefix as the
        # incoming checkpoint names (both ``model.…`` at the top level), so the
        # suffix-rewritten target is a direct key.
        params_dict = dict(self.named_parameters())
        param_names = tuple(params_dict)
        reverse_fusion = _build_reverse_fusion(
            getattr(self, "packed_modules_mapping", None))
        # Spec-layer (MTP) drafters nest a spec layer's block tensors under
        # ``.mtp_block.``; DSpark instead maps physical ``mtp.i.*`` tensors to
        # registered ``model.layers.i.*`` params. ``rename`` reuses each
        # model's own mapping before Gridbook resolution (identity for a body
        # model). Delegation below still yields the original checkpoint name.
        # Multimodal wrappers rename the checkpoint prefix
        # (``model.language_model.`` -> ``language_model.model.``) via their own
        # WeightsMapper, which the original loader applies only internally;
        # apply it FIRST so our anchors match the registered param names.
        rename = _compose_renames(_hf_mapper_rename(self),
                                  _spec_layer_rename(self),
                                  _dspark_rename(self))
        mixed_transactions = _MixedFusedTransactions(params_dict)
        loaded: set[str] = set()
        def _passthrough():
            # A generator (not a materialized list) preserves the original
            # streaming/mmap path. Ordinary tensors remain one-at-a-time. A
            # composite fused module retains only its bounded set of role
            # planes until the complete module validates and commits; CB expert
            # tensors are copied inline. Every other tensor is yielded onward.
            for name, w in weights:
                # Resolve against the served param naming (spec-layer rename or
                # identity). A ``"__skip__"`` — the spec drafter reusing the main
                # model's embed/lm_head — is delegated so the original drops it.
                res_name = rename(name) if rename is not None else name
                if res_name == "__skip__":
                    yield name, w
                    continue
                mixed_mapped = mixed_transactions.stage(res_name, w)
                if mixed_mapped is not None:
                    loaded.update(mixed_mapped)
                    continue
                source_mapped = load_source_split_expert(
                    res_name, w, params_dict
                )
                if source_mapped is not None:
                    loaded.add(source_mapped)
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
                    incoming = w.to(param.dtype)
                    if mapped.endswith("_input_global_scale") \
                            and not torch.isnan(param.data).all():
                        if not torch.equal(param.data, incoming.reshape_as(param)):
                            raise ValueError(
                                f"prismaquant CB expert '{name}' -> "
                                f"'{mapped}': per-layer input_global_scale "
                                "differs across format subgroups"
                            )
                    else:
                        param.data.copy_(incoming)
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
        mixed_transactions.finish()
        if ret:
            loaded |= set(ret)
        return loaded

    load_weights._pq_cb_wrapper = True
    load_weights._gridbook_mixed_fused_loader_abi = MIXED_FUSED_LOADER_ABI
    model_cls.load_weights = load_weights
    model_cls._pq_cb_wrapped = True
    _install_mixed_fused_construction_gate(model_cls)
    _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
