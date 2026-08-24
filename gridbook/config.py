"""``PrismaQuantConfig`` — the vLLM quantization config for the NVFP4-CB /
FP8-CB out-of-tree lane (docs/SPEC.md §5 for the config vocabulary this parses,
§6 for the registry keys and per-module dispatch it implements).

vLLM auto-detects the canonical ``quant_method == "gridbook"`` and accepts the
legacy ``"prismaquant"`` alias declared by the packaged runtime contract. The exporter writes
``config.json['quantization_config']`` as a *pointer* (``config_file`` ->
``quant_config.json`` + ``codebook_file`` -> ``cb_codebooks.pqcb``); the full
``config_groups`` / ``ignore`` live in ``quant_config.json``. We resolve that
sidecar **lazily** (via ``get_current_vllm_config()``, the same handle
``get_codebooks`` uses) since ``from_config`` runs before the model dir is
plumbed. Inlined configs (``config_groups`` already present) are also accepted.

**Mixed-container dispatch (docs/SPEC.md §6).** A config group with a
``"scheme"`` key is a CB group (our nvfp4_cb/fp8_cb vocabulary) -> our
``PrismaQuantCBLinearMethod``. A group WITHOUT it uses the exact stock
compressed-tensors vocabulary -> a real ``CompressedTensorsConfig`` we construct
and delegate to (``CompressedTensorsW4A4Nvfp4`` for NVFP4 groups, the fp8 scheme
for FP8_DYNAMIC). ``ignore`` -> ``UnquantizedLinearMethod``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from .runtime_contract import load_runtime_contract
from .delegated_preflight import (
    require_native_delegated_backend,
    require_native_passthrough_backend,
)
from .source_passthrough import (
    SourcePassthroughError,
    SourceFormat as _SourceFormat,
    build_delegated_method as _build_passthrough_method,
    parse_declaration as _parse_passthrough_declaration,
    require_audited_device as _require_passthrough_device,
)
from .fp8_source_w8a16 import (
    # The ONE passthrough lane whose own weight construction enforces
    # structural tensor-parallel shard laws; see _delegate_passthrough.
    WIRE_FP8_BLOCK128 as _WIRE_FP8_BLOCK128_TP,
)
from .embedding import (
    EmbeddingFormat as _EmbeddingFormat,
    parse_declaration as _parse_embedding_declaration,
)
from .lane_select import device_capability as _live_device_capability
from .nvfp4_activation_contract import (
    CONTRACT_KEY as _NVFP4_ACTIVATION_CONTRACT_KEY,
    TENSOR_SUFFIX as _NVFP4_ACTIVATION_TENSOR_SUFFIX,
    parse_contract as _parse_nvfp4_activation_contract,
    validate_payload as _validate_nvfp4_activation_payload,
)
from .per_expert_format import (
    LayerFormatGroups as _LayerFormatGroups,
    layer_id_for_prefix as _per_expert_layer_id_for_prefix,
    parse_declaration as _parse_per_expert_format_groups,
)
try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
except Exception:  # pragma: no cover - older vLLM
    RoutedExperts = None

_MOE_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")
# Which logical codebook roles a physical target leaf speaks for. A fused
# ``gate_up_proj`` target claims BOTH halves of the w13 stack with one book;
# the unfused spellings claim one role each. ``down_proj`` is always a
# one-role stack — which is why w2 never splits, it only picks a book.
_MOE_LEAF_ROLES = {
    "gate_up_proj": ("gate", "up"),
    "gate_proj": ("gate",),
    "up_proj": ("up",),
    "down_proj": ("down",),
}
_MOE_ROLES = ("gate", "up", "down")


def _codebook_ref_key(scheme: dict) -> tuple[str, ...]:
    """A scheme's codebook identity as a hashable tuple.

    ``codebook_ref`` is a str for a single-sub book and a list of sub-book
    names for product mode; normalising both to a tuple lets refs be compared
    and used as dict keys without caring which spelling the exporter used.
    """
    ref = scheme.get("codebook_ref")
    if ref is None:
        return ()
    return tuple(ref) if isinstance(ref, (list, tuple)) else (ref,)
# vLLM resolves a RoutedExperts stack's declared group through the *unfused*
# per-expert projection names (``CompressedTensorsMoEMethod.get_moe_method``
# builds ``<prefix>.0.{gate,up,down}_proj``). Gridbook's D0.2 preflight has to
# read the same declaration vLLM read, so it probes the same spellings — plus
# the fused/unsuffixed forms an exporter may legitimately have written.
_MOE_DECLARATION_SUFFIXES = (
    ".0.gate_proj", ".0.up_proj", ".0.down_proj",
    ".gate_up_proj", ".gate_proj", ".up_proj", ".down_proj",
    "",
)
_RUNTIME_CONTRACT = load_runtime_contract()
_QUANT_METHOD_CANONICAL = _RUNTIME_CONTRACT["quant_method"]["canonical"]
_QUANT_METHOD_ACCEPTED = frozenset(
    _RUNTIME_CONTRACT["quant_method"]["accepted"]
)

# vLLM fuses these siblings into one module; packed_modules_mapping is populated
# by dispatch time, but we keep the standard mapping as a fallback.
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    # DeepSeek-V4 MLA: the q-LoRA down-projection and the joint KV projection
    # are ONE MergedColumnParallelLinear (vllm/models/deepseek_v4/attention.py,
    # `fused_wqa_wkv`, [q_lora_rank, head_dim] = [1024, 512]). The checkpoint
    # keeps them apart as `attn.wq_a` / `attn.wkv`, and the class publishes the
    # merge only through its `stacked_params_mapping` — it defines NO
    # `packed_modules_mapping` — so without this entry a CB `wq_a` target
    # resolves to nothing under the served `…attn.fused_wqa_wkv` prefix and the
    # layer silently falls through to BF16 / stock dispatch.
    "fused_wqa_wkv": ["wq_a", "wkv"],
}

# A served leaf can correspond to MORE THAN ONE checkpoint spelling across
# architectures, in two ways this table covers together:
#
#   * a different FUSION — DeepSeek-V4's shared expert fuses the
#     Mixtral-convention `w1`/`w3` into the same `gate_up_proj` leaf that
#     Llama-class models fuse `gate_proj`/`up_proj` into, so one
#     `_FUSED_FALLBACK` value cannot express both;
#   * a plain 1:1 RENAME — the same shared expert's un-fused down projection is
#     `w2` in the checkpoint and `down_proj` in the module tree. That is not a
#     fusion at all, so it never reached the fused table, and before this it
#     resolved to nothing: the exact-key lookup missed and `down_proj` had no
#     declared shard spelling, so a declared CB target fell silently through to
#     BF16/stock dispatch. A one-element spelling expresses it exactly.
#
# Resolution tries the primary spelling first, then each alternate, and the
# first spelling with any hit wins WHOLE (never mixed) — the same rule
# `shard_target_keys` already applies across namespace vintages. An alternate is
# only ever consulted after the primary misses, so an architecture using the
# canonical spelling is unaffected.
_ALTERNATE_SHARD_SPELLINGS = {
    "gate_up_proj": (["w1", "w3"],),
    "gate_proj": (["w1"],),
    "up_proj": (["w3"],),
    "down_proj": (["w2"],),
}


@dataclass(frozen=True)
class _FusedRoleOwner:
    """One physical source role of a vLLM ``MergedColumnParallelLinear``.

    ``target`` is kept in the serving/checkpoint shard spelling (rather than
    the fused module spelling) so the mixed loader can route the corresponding
    tensor planes without guessing.  ``kind`` is either ``"cb"`` or
    ``"source"``; ``payload`` is the existing scheme / ``SourceFormat`` object
    consumed by that lane's ordinary linear method.
    """

    target: str
    kind: str
    payload: Any


def _source_passthrough_aliases(name: str) -> tuple[str, ...]:
    """Additional live-module spellings for a source-passthrough unit.

    DeepSeek-V4 has two live namespaces: Transformers exposes
    ``self_attn``/``mlp`` and gate/up/down shared-expert leaves, while vLLM
    0.24 constructs ``attn``/``ffn`` and consumes the checkpoint's
    ``w1``/``w3``/``w2`` leaves.  The producer declaration is intentionally in
    the former (the profile's live namespace), so serving must try that exact
    structural alias after the vLLM spelling.  An alias only wins when it is an
    explicitly declared unit; it cannot claim an unrelated model by pattern.
    """

    aliases: list[str] = []
    if ".attn." in name:
        aliases.append(name.replace(".attn.", ".self_attn.", 1))
    if ".ffn." in name:
        live = name.replace(".ffn.", ".mlp.", 1)
        replacements = {
            ".shared_experts.w1": ".shared_experts.gate_proj",
            ".shared_experts.w3": ".shared_experts.up_proj",
            ".shared_experts.w2": ".shared_experts.down_proj",
        }
        for source, target in replacements.items():
            if source in live:
                live = live.replace(source, target, 1)
                break
        aliases.append(live)
    return tuple(dict.fromkeys(aliases))


def _initialized_tensor_parallel_world_size() -> int | None:
    """Return vLLM's TP size once model parallelism exists.

    Config/resolver unit tests intentionally run without a distributed vLLM
    process.  Production calls ``get_quant_method`` while constructing the
    worker model, after vLLM initializes its model-parallel groups; that is the
    first point where the documented TP=1 restriction can be enforced against
    the real serving state instead of an argument string.
    """

    try:
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            model_parallel_is_initialized,
        )
    except (ImportError, AttributeError):  # pragma: no cover - minimal stubs
        return None
    if not model_parallel_is_initialized():
        return None
    return int(get_tensor_model_parallel_world_size())


def _canonical_prefix(prefix: str) -> str:
    """vLLM serving prefix -> canonical target namespace. Some model classes
    wrap the LM (`language_model.model.layers.*` on Qwen3.5-class VL) while
    targets are canonical `model.layers.*`; strip the wrapper when the next
    component is `model.` (measured via PRISMAQUANT_DEBUG_PREFIXES,
    2026-07-22 — every LM layer resolved no-scheme without this)."""
    if prefix.startswith("language_model.model."):
        return prefix[len("language_model."):]
    if prefix.startswith("language_model."):
        return "model." + prefix[len("language_model."):]
    # Pre-fix multimodal CHECKPOINT namespace (shipped 27B gridbook artifact):
    # ``model.language_model.layers.*`` denotes the same Linear as canonical
    # ``model.layers.*``. Normalising it here (as well as in
    # ``_canonical_target``) keeps probe-side and target-side on one string.
    if prefix.startswith("model.language_model."):
        return "model." + prefix[len("model.language_model."):]
    # Producer SOURCE namespace (DeepSeek-V4 class): the released checkpoint's
    # keys start at `layers.N.` with no `model.` component at all, and the vLLM
    # class re-attaches it inside its own `hf_to_vllm_mapper`
    # (`{"layers.": "model.layers."}`) — i.e. AFTER a serving prefix has already
    # been handed to get_quant_method. An artifact that stored its CB targets in
    # that source spelling therefore has to be lifted here, or every DSV4 body
    # Linear resolves no-scheme. `_candidate_bases` still tries the string as
    # given first, so an architecture that genuinely owns a top-level `layers.`
    # module keeps its exact match.
    if prefix.startswith("layers."):
        return "model." + prefix
    return prefix


def _candidate_bases(name: str) -> list[str]:
    """Every namespace vintage *name* can legitimately be matched against,
    **most specific first** (the string as given, then its canonical form).

    THE one place that answers "which namespace am I in?". A stored target /
    serving prefix reaches us in one of three vintages — the old multimodal
    CHECKPOINT form (``model.language_model.*``), the canonical form
    (``model.*``), and the vLLM wrapper-class SERVING form
    (``language_model.model.*``) — and ``apply_vllm_mapper`` can move the
    stored keys into a *fourth*, the mapper's own namespace, AFTER
    ``_ensure_resolved`` canonicalised them. Anything that matches a prefix
    against ``target_scheme`` / ``ignore`` must therefore try both sides, and
    must do so HERE: the dense fused path grew its own single-namespace copy of
    this logic and silently mis-resolved for it (issue #1). A future fifth
    vintage should mean editing this function and nothing else.
    """
    canonical = _canonical_prefix(name)
    return [name] if canonical == name else [name, canonical]


def _canonical_target(name: str) -> str:
    """Stored ``config_groups[*].targets`` / ``ignore`` entry -> canonical
    target namespace, so historical checkpoint-namespace artifacts resolve
    against the canonicalised serving prefixes ``_canonical_prefix`` produces.

    Rewrites (prefix-anchored only):
      ``model.language_model.`` -> ``model.``          (old multimodal ckpt)
      ``language_model.model.`` -> ``model.``          (serving wrapper form)
      ``language_model.<rest>`` -> ``model.<rest>``
    Everything else (``visual.*``, ``mtp.*``, plain ``model.layers.*``,
    bare leaf names) passes through untouched."""
    return _canonical_prefix(name)


_DSPARK_TARGET_BRIDGE_SCHEMA = "gridbook.dspark-target-bridge.v1"
_DSPARK_CANONICAL_INDEX = r"(?:0|[1-9]\d*)"
_DSPARK_CANONICAL_TAIL = (
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"(?:[.](?:[A-Za-z_][A-Za-z0-9_]*|0|[1-9]\d*))*"
)
_DSPARK_CONSTRUCTION_TARGET_RE = re.compile(
    rf"^model[.]layers[.](?P<layer>{_DSPARK_CANONICAL_INDEX})[.]"
    rf"(?P<rest>{_DSPARK_CANONICAL_TAIL})$"
)
_DSPARK_PHYSICAL_TARGET_RE = re.compile(
    rf"^mtp[.](?P<stage>{_DSPARK_CANONICAL_INDEX})[.]"
    rf"(?P<rest>{_DSPARK_CANONICAL_TAIL})$"
)


def _parse_dspark_target_bridge(config: dict, contract: dict | None
                                ) -> dict[str, str]:
    """Validate the explicit DSpark construction -> physical target map.

    Quantization dispatch constructs DSpark's layers under
    ``model.layers.{num_hidden_layers + stage}``, while activation-contract
    scalars are serialized under the checkpoint's ``mtp.{stage}`` namespace.
    The bridge is producer metadata, not a runtime guess: its topology and
    every same-tail mapping are checked here, and its physical values must be
    exactly the digest-bound activation contract target set.
    """
    raw = config.get("dspark_target_bridge")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("dspark_target_bridge must be an object")
    expected_fields = {
        "schema", "num_hidden_layers", "n_mtp_layers",
        "construction_to_physical",
    }
    missing = sorted(expected_fields - set(raw))
    extra = sorted(set(raw) - expected_fields)
    if missing or extra:
        raise ValueError(
            "dspark_target_bridge fields mismatch: "
            f"missing={missing}, unknown={extra}"
        )
    if raw["schema"] != _DSPARK_TARGET_BRIDGE_SCHEMA:
        raise ValueError(
            f"dspark_target_bridge.schema={raw['schema']!r}; expected "
            f"{_DSPARK_TARGET_BRIDGE_SCHEMA!r}"
        )
    num_hidden = raw["num_hidden_layers"]
    n_mtp = raw["n_mtp_layers"]
    if (isinstance(num_hidden, bool) or not isinstance(num_hidden, int)
            or num_hidden <= 0):
        raise ValueError(
            "dspark_target_bridge.num_hidden_layers must be a positive integer"
        )
    if (isinstance(n_mtp, bool) or not isinstance(n_mtp, int) or n_mtp <= 0):
        raise ValueError(
            "dspark_target_bridge.n_mtp_layers must be a positive integer"
        )
    if contract is None:
        raise ValueError(
            "dspark_target_bridge requires the nvfp4_w4a4 execution contract"
        )
    mapping = raw["construction_to_physical"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "dspark_target_bridge.construction_to_physical must be a "
            "non-empty object"
        )

    result: dict[str, str] = {}
    for construction, physical in mapping.items():
        if (not isinstance(construction, str) or not construction
                or not isinstance(physical, str) or not physical):
            raise ValueError(
                "dspark_target_bridge mappings require non-empty string keys "
                "and values"
            )
        canonical = _canonical_target(construction)
        if canonical != construction:
            raise ValueError(
                f"dspark construction target {construction!r} is not in the "
                f"canonical namespace; use {canonical!r}"
            )
        construction_match = _DSPARK_CONSTRUCTION_TARGET_RE.fullmatch(
            construction
        )
        physical_match = _DSPARK_PHYSICAL_TARGET_RE.fullmatch(physical)
        if construction_match is None or physical_match is None:
            raise ValueError(
                f"invalid DSpark target bridge {construction!r} -> "
                f"{physical!r}; expected model.layers.N.<tail> -> "
                "mtp.S.<tail>"
            )
        stage = int(physical_match.group("stage"))
        layer = int(construction_match.group("layer"))
        if stage >= n_mtp:
            raise ValueError(
                f"DSpark physical target {physical!r} has stage {stage}, "
                f"outside n_mtp_layers={n_mtp}"
            )
        if layer != num_hidden + stage:
            raise ValueError(
                f"DSpark construction target {construction!r} has layer "
                f"{layer}; expected num_hidden_layers + stage = "
                f"{num_hidden + stage}"
            )
        if construction_match.group("rest") != physical_match.group("rest"):
            raise ValueError(
                f"DSpark target bridge {construction!r} -> {physical!r} "
                "changes the target tail"
            )
        result[construction] = physical

    if len(set(result.values())) != len(result):
        raise ValueError(
            "dspark_target_bridge physical targets must be one-to-one"
        )
    contract_targets = set(contract["target_names"])
    if set(result.values()) != contract_targets:
        raise ValueError(
            "dspark_target_bridge physical targets must exactly equal the "
            "nvfp4_w4a4 execution contract target_names; "
            f"bridge_only={sorted(set(result.values()) - contract_targets)}, "
            f"contract_only={sorted(contract_targets - set(result.values()))}"
        )
    return result


def _sidecar_revision(model_config: Any, model_dir: str) -> str | None:
    """Return one immutable revision for every sidecar of a Hub model.

    ``hf_config._commit_hash`` is the revision Transformers actually resolved
    while vLLM prepared the model, so it is authoritative over a requested
    tag or branch in ``model_config.revision``.  If Transformers did not expose
    it, only a full 40-hex commit in ``model_config.revision`` is an immutable
    fallback.  Local directories need no Hub revision and retain their ordinary
    path-join behavior.
    """

    if os.path.isdir(model_dir):
        return None

    def immutable_commit(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if (
            len(value) == 40
            and all(char in "0123456789abcdefABCDEF" for char in value)
        ):
            return value.lower()
        return None

    hf_config = getattr(model_config, "hf_config", None)
    resolved_commit = getattr(hf_config, "_commit_hash", None)
    immutable_resolved_commit = immutable_commit(resolved_commit)
    if immutable_resolved_commit is not None:
        return immutable_resolved_commit
    requested_revision = getattr(model_config, "revision", None)
    immutable_requested_revision = immutable_commit(requested_revision)
    if immutable_requested_revision is not None:
        return immutable_requested_revision
    raise RuntimeError(
        f"Hub model {model_dir!r} has no immutable revision for Gridbook "
        "sidecars; neither vLLM model_config.hf_config._commit_hash nor "
        "model_config.revision is a full 40-hex commit SHA"
    )


def _resolve_model_file(
    model_dir: str, fname: str, *, revision: str | None = None
) -> str:
    """Local path for a sidecar file next to the model. When the model was
    given as a Hub repo id (``vllm serve rdtand/...``) rather than a local
    directory, fetch the sidecar from the Hub — vLLM's own loader handles the
    weights that way, but OUR sidecars (quant_config.json, the .pqcb codebook
    blob) were opened with a plain path join, which broke every serve-by-id
    until 2026-07-22."""
    if os.path.isdir(model_dir):
        return os.path.join(model_dir, fname)
    if not isinstance(revision, str) or not revision.strip():
        raise RuntimeError(
            f"refusing to fetch Gridbook sidecar {fname!r} for unpinned Hub "
            f"model {model_dir!r}"
        )
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=model_dir, filename=fname, revision=revision.strip()
    )


class PrismaQuantConfig(QuantizationConfig):
    """Per-layer dispatch: CB decode / stock-CT delegation / unquantized."""

    def __init__(self, raw_config: dict) -> None:
        super().__init__()
        self._raw_config = dict(raw_config or {})
        self.codebook_file = self._raw_config.get("codebook_file",
                                                  "cb_codebooks.pqcb")
        # Resolved lazily (the sidecar quant_config.json needs the model dir).
        self._resolved = False
        self._full_config: dict = {}
        self.config_groups: dict = {}
        self.ignore: list[str] = []
        self.target_scheme: dict[str, dict] = {}    # CB module -> scheme dict
        self._cb_targets: set[str] = set()
        self._embedding_units: dict[str, "_EmbeddingFormat"] = {}
        self.ct_config = None                        # stock CompressedTensorsConfig
        self._codebooks: dict[str, torch.Tensor] | None = None
        self._tp_world_size: int | None = None
        # Cached as one pair so pointer quant_config.json and its declared
        # codebook can never be fetched from different revisions if a mutable
        # requested tag moves between the two lazy reads.
        self._sidecar_source: tuple[str, str | None] | None = None
        # Producer-owned static W4A4 contract.  The record is validated while
        # resolving config; the exact physical scalar payload is verified once
        # against safetensors, then shared by dense and MoE loaders.
        self._nvfp4_activation_contract: dict | None = None
        self._nvfp4_activation_scales: dict[str, float] | None = None
        self._target_physical_name: dict[str, str] = {}
        # Producer-declared DSpark construction topology, rechecked against
        # the instantiated draft model by the top-level loader before copying
        # any tensor. None is the permanent non-DSpark/legacy state.
        self._dspark_target_bridge_topology: tuple[int, int] | None = None
        # Delegated (non-CB) target -> the stock group that declares it. The
        # D0.2 preflight needs the *declaration*, and the declaration lives in
        # the config group, not in whatever tensors happen to be on disk.
        self._stock_group_by_target: dict[str, str] = {}
        # Canonical unit prefix -> the SOURCE format it passes through
        # verbatim.  Empty for every artifact published before the schema
        # existed, which is exactly the legacy all-CB meaning.
        self._passthrough_units: dict[str, "_SourceFormat"] = {}
        # Layer-id -> the producer's v1 family-specific expert partitions.
        # Empty is the permanent legacy path: no different method, buffer,
        # loader wrapper, or dispatch branch is installed for old artifacts.
        self._per_expert_format_groups: dict[str, "_LayerFormatGroups"] = {}
        # Producer-physical declaration prefix -> current serving namespace.
        # Values follow vLLM's mapper; keys remain checkpoint wire identities.
        self._per_expert_serving_prefixes: dict[str, str] = {}

    def _get_sidecar_source(self) -> tuple[str, str | None]:
        if self._sidecar_source is None:
            from vllm.config import get_current_vllm_config
            from .moe_toplevel_loader import (
                active_dspark_draft_model_config,
            )

            # vLLM retains the target model_config while constructing a
            # separate DSpark draft.  Only the DSpark class's exact
            # construction context can override this source; the existence of
            # speculative_config by itself changes nothing for body/legacy
            # configs.
            model_config = active_dspark_draft_model_config()
            if model_config is None:
                model_config = get_current_vllm_config().model_config
            try:
                model_dir = os.fspath(model_config.model)
            except TypeError as exc:
                raise RuntimeError(
                    "vLLM model_config.model is not a filesystem path or Hub ID"
                ) from exc
            if not isinstance(model_dir, str) or not model_dir:
                raise RuntimeError(
                    "vLLM model_config.model is not a nonempty string path or "
                    "Hub ID"
                )
            self._sidecar_source = (
                model_dir,
                _sidecar_revision(model_config, model_dir),
            )
        return self._sidecar_source

    def _tensor_parallel_world_size(self) -> int | None:
        """Live vLLM TP size; ``None`` until model parallelism initializes.

        Latched only once it resolves to 1 (the overwhelmingly common serve),
        so repeated ``get_quant_method`` calls do not re-enter vLLM's
        distributed module. A resolved value above one is never latched,
        which keeps the retry behaviour of the previous TP=1-only gate.
        """
        if self._tp_world_size == 1:
            return 1
        world_size = _initialized_tensor_parallel_world_size()
        if world_size == 1:
            self._tp_world_size = 1
        return world_size

    def _require_ep_moe_serving(self, surface: str, prefix: str,
                                layer) -> str:
        """Admit stacked CB expert stacks above one rank ONLY under EP.

        Expert parallelism is the one multi-rank MoE mode Gridbook serves. A CB
        expert stack is a byte tensor whose last dimension is
        ``(in/256)·type_size`` — superblock bytes, not input columns — so
        vLLM's tensor-parallel intermediate split would cut a superblock in
        half and there is no partial-superblock decode. Expert parallelism
        instead gives each rank a disjoint subset of WHOLE experts: whole
        stacks, whole superblocks, and the identical per-expert numerics the
        same artifact serves at world size 1.

        Everything else above one rank refuses here, at method construction,
        before a single buffer is sized — including the expert-parallel
        topologies whose premise Gridbook has not established. The premise is
        that this rank computes its own experts and vLLM's stock final
        all-reduce sums the per-rank partials
        (``fused_moe/runner/moe_runner.py`` ``_maybe_reduce_final_output``:
        fires when ``ep_size > 1``, and is skipped under sequence parallelism
        or ``skip_final_all_reduce``). Data-, pipeline-context- and
        sequence-parallel EP also switch vLLM to all2all dispatch/combine
        kernels, which expect a MoE method that exchanges tokens; Gridbook has
        none.

        Returns the admitted mode, for the load-time announcement.
        """
        world_size = self._tensor_parallel_world_size()
        if world_size is None or world_size == 1:
            return "single_rank"
        where = f" at {prefix!r}" if prefix else ""
        cfg = getattr(layer, "moe_config", None)
        par = getattr(cfg, "moe_parallel_config", None)

        def _refuse(reason: str, fix: str) -> None:
            raise ValueError(
                f"Gridbook serves {surface}{where} above one rank only under "
                f"expert parallelism; the live vLLM worker reports "
                f"TP={world_size} and {reason}. {fix} A CB expert stack is a "
                "byte tensor whose last dimension is superblock bytes rather "
                "than input columns, so tensor-parallel sharding would cut a "
                "superblock; expert parallelism keeps whole experts per rank "
                "and is the supported multi-rank MoE mode."
            )

        if par is None:
            _refuse("its MoE parallel configuration is unreadable",
                    "Serve with -tp N --enable-expert-parallel.")
        if not bool(getattr(par, "use_ep", False)):
            _refuse("expert parallelism is off for this MoE layer",
                    "Add --enable-expert-parallel to the serve command; "
                    "-tp N alone tensor-parallelizes the expert stacks.")
        moe_tp = int(getattr(par, "tp_size", 1))
        if moe_tp != 1:
            _refuse(f"the MoE layer is still tensor-parallel "
                    f"(moe tp_size={moe_tp})",
                    "Expert parallelism must own the whole MoE axis.")
        if bool(getattr(par, "use_all2all_kernels", False)):
            _refuse(
                f"this is an all2all expert-parallel topology "
                f"(dp_size={int(getattr(par, 'dp_size', 1))}, "
                f"pcp_size={int(getattr(par, 'pcp_size', 1))}, "
                f"sp_size={int(getattr(par, 'sp_size', 1))})",
                "Gridbook's CB MoE method computes this rank's own experts "
                "and relies on vLLM's stock final all-reduce; it implements "
                "no dispatch/combine token exchange. Serve with "
                "data-parallel, pipeline-context-parallel and "
                "sequence-parallel sizes of 1.")
        if bool(getattr(par, "enable_eplb", False)):
            _refuse("expert-load-balancing (EPLB) is enabled",
                    "Gridbook holds whole CB expert stacks resident and "
                    "cannot follow a live re-placement; serve without EPLB.")
        if bool(getattr(cfg, "skip_final_all_reduce", False)):
            _refuse("skip_final_all_reduce is set",
                    "Per-rank expert partials would never be summed. Gridbook "
                    "does not reduce its own output.")
        return f"expert_parallel(ep_size={int(getattr(par, 'ep_size', world_size))})"

    def _require_tp1_serving(self, surface: str,
                             prefix: str | None = None,
                             note: str | None = None) -> None:
        """Fail closed for every serving surface that is still TP=1-only.

        Dense CB Linears are the ONE admitted surface above a single rank:
        their shard laws (whole-superblock K windows, kernel-aligned N rows,
        replicated sidecar tables, rank-local role geometry) are enforced
        structurally at weight construction and load finalization. Every
        other surface names ITSELF here — at method construction, before any
        parameter exists — instead of failing later against a generic shape
        mismatch or, worse, serving unattested sharding.
        """
        world_size = self._tensor_parallel_world_size()
        if world_size is None or world_size == 1:
            return
        where = f" at {prefix!r}" if prefix else ""
        raise ValueError(
            f"Gridbook serves {surface}{where} at tensor-parallel size 1 "
            f"only; the live vLLM worker reports TP={world_size}. Dense CB "
            "Linears are the only supported tensor-parallel surface."
            + (f" {note}" if note else "")
        )

    def _has_mixed_fused_loader(self) -> bool:
        """Whether the class constructing this layer has Gridbook's loader ABI.

        A composite method owns nested per-role parameters which ordinary
        ``AutoWeightsLoader`` lookup cannot name.  The top-level Gridbook
        wrapper is therefore a hard ownership precondition, not merely an MoE
        optimization. The installer sets a thread/task-local context only
        around the exact model instance's ``__init__``, after verifying its
        effective ``load_weights`` method. This is intentionally not derived
        from vLLM's global model_config: speculative draft construction keeps
        the target config current and would otherwise inherit target authority.
        """

        try:
            from .moe_toplevel_loader import mixed_fused_loader_active
            return mixed_fused_loader_active()
        except Exception:  # noqa: BLE001 — absence means loader unavailable
            return False

    def _fused_owners_share_single_method(
        self, owners: list[_FusedRoleOwner]
    ) -> bool:
        """Whether the legacy merged method preserves every role contract."""

        kinds = {owner.kind for owner in owners}
        if kinds == {"source"}:
            return all(owner.payload.id == owners[0].payload.id
                       for owner in owners[1:])
        if kinds != {"cb"}:
            return False
        # Codebook refs may differ by role: PrismaQuantCBLinearMethod already
        # concatenates those tables and row offsets. These fields instead
        # describe the physical packed representation / activation ABI shared
        # by the one resident parameter set.
        keys = ("grid", "mode", "k", "n_sub", "type_size", "group_size",
                "vec_dim", "scale_coding", "activation_contract")
        first = owners[0].payload
        if any(any(owner.payload.get(key) != first.get(key) for key in keys)
               for owner in owners[1:]):
            return False
        if first.get("activation_contract") is None:
            return True
        # Matching scheme dictionaries do not prove matching physical static
        # scalars. The legacy fused method owns one activation contract and
        # intentionally rejects non-identical role scalars, so decide that
        # before selecting it rather than failing late in post-load finalize.
        scales = self.activation_scales_for_targets(
            [owner.target for owner in owners])
        return bool(scales) and all(scale == scales[0]
                                    for scale in scales[1:])

    @staticmethod
    def _validate_cb_format_scheme(
        scheme: dict, target: str, runtime_contract: dict,
    ) -> None:
        """Fail closed on a CB scheme outside the packaged reader ABI.

        ``formats[].rungs`` is the reader domain.  It is intentionally broader
        than ``producer_rungs`` for FP8 so legacy irregular K28..K48 artifacts
        keep loading; NVFP4 has no such legacy artifacts and both lists stop at
        K25.  Dense and routed constructors used to trust ``k``, ``n_sub``
        and ``type_size`` independently; a typo could therefore size a resident
        byte plane before the CUDA binding finally rejected it.  Resolve the
        physical fields against the one packaged contract while the sidecar is
        already being parsed.  For FP4, scale coding selects the v1 (16-byte)
        or v2/two-tier (9-byte) scale plane.
        """

        grid = scheme.get("grid")
        family_by_grid = {"fp4": "NVFP4_CB_K", "fp8": "FP8_CB_K"}
        if grid not in family_by_grid:
            raise ValueError(
                f"CB target {target!r}: grid must be 'fp4' or 'fp8', got "
                f"{grid!r}"
            )
        family = family_by_grid[grid]
        formats = [
            item for item in runtime_contract.get("formats", ())
            if item.get("family") == family
        ]
        if len(formats) != 1:
            raise ValueError(
                f"CB target {target!r}: packaged runtime contract must carry "
                f"exactly one {family} format row"
            )
        fmt = formats[0]
        if scheme.get("mode") != fmt["mode"]:
            raise ValueError(
                f"CB target {target!r}: {family} mode must be "
                f"{fmt['mode']!r}"
            )
        for field in ("k", "n_sub", "type_size"):
            value = scheme.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"CB target {target!r}: {family} {field} must be an integer"
                )
        k = scheme["k"]
        if k not in fmt["rungs"]:
            raise ValueError(
                f"CB target {target!r}: {family}{k} is outside the packaged "
                f"reader domain {fmt['rungs']}"
            )
        if scheme["n_sub"] != fmt["n_sub"]:
            raise ValueError(
                f"CB target {target!r}: {family}{k} requires "
                f"n_sub={fmt['n_sub']}, got {scheme['n_sub']}"
            )

        scale_coding = scheme.get("scale_coding")
        if isinstance(scale_coding, dict):
            scale_coding = scale_coding.get("kind")
            if not isinstance(scale_coding, str):
                raise ValueError(
                    f"CB target {target!r}: scale_coding.kind must be a "
                    "string"
                )
        elif scale_coding is None:
            scale_coding = "v1"
        if not isinstance(scale_coding, str):
            raise ValueError(
                f"CB target {target!r}: scale_coding must be a string, an "
                "object with a string 'kind', or absent"
            )
        if grid == "fp8":
            if scale_coding != "v1":
                raise ValueError(
                    f"CB target {target!r}: FP8_CB_K{k} requires v1 scale "
                    f"coding, got {scale_coding!r}"
                )
            expected_type_size = 4 * k
            rule = "4*k"
        else:
            scale_bytes = {"v1": 16, "two_tier": 9}.get(scale_coding)
            if scale_bytes is None:
                raise ValueError(
                    f"CB target {target!r}: NVFP4_CB_K{k} scale coding must "
                    f"be 'v1' or 'two_tier', got {scale_coding!r}"
                )
            expected_type_size = 4 * k + scale_bytes
            rule = f"4*k+{scale_bytes}"
        if scheme["type_size"] != expected_type_size:
            raise ValueError(
                f"CB target {target!r}: {family}{k} with "
                f"scale_coding={scale_coding!r} requires type_size={rule}="
                f"{expected_type_size}, got {scheme['type_size']}"
            )

    @staticmethod
    def _require_cb_device_capability(scheme: dict, prefix: str) -> None:
        """Reject a CB artifact before its first architecture-illegal path.

        FP8-CB's shipping large-M path uses vLLM's native FP8 quantizer and
        CUTLASS scaled GEMM, whose hardware floor is sm_89.  The main decode
        extension itself can compile for sm_80, so a global capability floor
        would otherwise let an A100 load successfully and fail only when the
        first prompt crosses the 16-token decode boundary.  NVFP4-CB is a
        Blackwell activation contract: an exact BF16 expansion primitive on
        Ada does not make the format supported there, and AQUA must never see
        an FP4 candidate on RTX 40.  Enforce both activation-family floors at
        this one load-time choke point.
        """

        if not torch.cuda.is_available():
            return
        grid = scheme.get("grid")
        if grid == "fp8":
            floor = (8, 9)
            family = "FP8-CB"
        elif grid == "fp4":
            floor = (10, 0)
            family = "NVFP4-CB"
        else:
            return
        capability = tuple(torch.cuda.get_device_capability())
        if capability < floor:
            raise ValueError(
                f"{family} target {prefix!r} requires compute capability "
                f"sm_{floor[0]}{floor[1]}+ for its activation contract; "
                f"the current device reports sm_{capability[0]}{capability[1]}"
            )

    @staticmethod
    def _validate_cb_activation_scheme(
        scheme: dict, target: str, contract: dict | None,
        *, physical_target: str | None = None,
    ) -> None:
        """Validate a custom scheme's top-level activation-contract link."""

        reference = scheme.get("activation_contract")
        grid = scheme.get("grid")
        if reference is not None and grid != "fp4":
            raise ValueError(
                f"CB target {target!r}: activation_contract is fp4-only"
            )
        if reference is not None and reference != _NVFP4_ACTIVATION_CONTRACT_KEY:
            raise ValueError(
                f"CB target {target!r}: unsupported activation_contract "
                f"{reference!r}"
            )
        if reference is not None and contract is None:
            raise ValueError(
                f"CB target {target!r}: references "
                f"{_NVFP4_ACTIVATION_CONTRACT_KEY!r}, but the top-level "
                "execution contract is absent"
            )
        contract_target = physical_target or target
        if (reference is not None and contract is not None
                and contract_target not in contract["target_names"]):
            raise ValueError(
                f"CB target {target!r}: physical activation target "
                f"{contract_target!r} is absent "
                "from execution_contracts.nvfp4_w4a4.target_names"
            )
        if contract is not None and grid == "fp4" and reference is None:
            raise ValueError(
                f"CB target {target!r}: contracted artifacts require every "
                "custom FP4-CB scheme to declare activation_contract="
                f"{_NVFP4_ACTIVATION_CONTRACT_KEY!r}"
            )

    def _activation_safetensor_files(self) -> list[str]:
        """Resolve the minimum safetensors set that can contain scale tensors."""

        model_dir, revision = self._get_sidecar_source()
        suffix = "." + _NVFP4_ACTIVATION_TENSOR_SUFFIX
        if os.path.isdir(model_dir):
            files = sorted(str(path) for path in Path(model_dir).glob(
                "*.safetensors"))
            if not files:
                raise ValueError(
                    f"contracted Gridbook artifact {model_dir!r} contains no "
                    "safetensors files"
                )
            return files

        # An index lets a Hub load fetch only shards containing the tiny
        # scalars instead of pulling unrelated weight shards early.
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.utils import EntryNotFoundError
        except ImportError:  # pragma: no cover - older huggingface-hub
            EntryNotFoundError = FileNotFoundError
        try:
            index_path = hf_hub_download(
                repo_id=model_dir,
                filename="model.safetensors.index.json",
                revision=revision,
            )
        except EntryNotFoundError:
            return [hf_hub_download(
                repo_id=model_dir,
                filename="model.safetensors",
                revision=revision,
            )]
        with open(index_path) as fh:
            index = json.load(fh)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(
                "model.safetensors.index.json has no object weight_map"
            )
        shard_names = sorted({
            filename for name, filename in weight_map.items()
            if str(name).endswith(suffix)
        })
        if not shard_names:
            raise ValueError(
                "contracted Gridbook artifact index lists no "
                f"*{suffix} tensors"
            )
        return [hf_hub_download(
            repo_id=model_dir, filename=filename, revision=revision
        ) for filename in shard_names]

    def _read_nvfp4_activation_scales(self) -> dict[str, torch.Tensor]:
        """Read every physical ``*.input_global_scale`` scalar from storage."""

        from safetensors import safe_open

        suffix = "." + _NVFP4_ACTIVATION_TENSOR_SUFFIX
        found: dict[str, torch.Tensor] = {}
        sources: dict[str, str] = {}
        for filename in self._activation_safetensor_files():
            with safe_open(filename, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if not name.endswith(suffix):
                        continue
                    target = name[: -len(suffix)]
                    if target in found:
                        raise ValueError(
                            f"duplicate {name!r} in {sources[target]!r} and "
                            f"{filename!r}"
                        )
                    found[target] = reader.get_tensor(name)
                    sources[target] = filename
        return found

    def _ensure_nvfp4_activation_payload(self) -> None:
        if self._nvfp4_activation_contract is None:
            return
        if self._nvfp4_activation_scales is None:
            raw = self._read_nvfp4_activation_scales()
            self._nvfp4_activation_scales = _validate_nvfp4_activation_payload(
                self._nvfp4_activation_contract, raw
            )

    def activation_scales_for_targets(self, targets: list[str]) -> list[float]:
        """Return attested F32 values for resolved custom-CB target keys."""

        self._ensure_resolved()
        if self._nvfp4_activation_contract is None:
            return []
        self._ensure_nvfp4_activation_payload()
        assert self._nvfp4_activation_scales is not None
        values = []
        for target in targets:
            physical = self._target_physical_name.get(target)
            if physical is None:
                raise ValueError(
                    f"CB target {target!r} declares the NVFP4 activation "
                    "contract but has no physical tensor identity"
                )
            try:
                values.append(self._nvfp4_activation_scales[physical])
            except KeyError as exc:
                raise ValueError(
                    f"contracted CB target {target!r} expects physical scalar "
                    f"{physical}.{_NVFP4_ACTIVATION_TENSOR_SUFFIX}, but it is "
                    "absent from the attested payload"
                ) from exc
        return values

    # -- lazy resolution of the (possibly pointer) quant config --------------
    def _ensure_resolved(self) -> None:
        # Pointer configs necessarily bind their model source below when they
        # open quant_config.json.  An inline config can otherwise finish
        # resolution without touching either sidecar, then first ask for its
        # codebook during post-load finalization after the DSpark constructor's
        # draft-authority ContextVar has been restored.  Pin that authority
        # while it is available, even when this config was already resolved;
        # ordinary target configs still see no DSpark context and retain the
        # historical current-vLLM-config path.
        if self._sidecar_source is None:
            from .moe_toplevel_loader import active_dspark_draft_model_config

            if active_dspark_draft_model_config() is not None:
                self._get_sidecar_source()
        if self._resolved:
            return
        cfg = self._raw_config
        if "config_groups" not in cfg:
            cfg_file = cfg.get("config_file", "quant_config.json")
            model_dir, revision = self._get_sidecar_source()
            with open(_resolve_model_file(
                    model_dir, cfg_file, revision=revision)) as fh:
                cfg = json.load(fh)
            self.codebook_file = cfg.get("codebook_file", self.codebook_file)
        self._nvfp4_activation_contract = _parse_nvfp4_activation_contract(cfg)
        runtime_contract = _RUNTIME_CONTRACT
        dspark_target_bridge = _parse_dspark_target_bridge(
            cfg, self._nvfp4_activation_contract
        )
        if dspark_target_bridge:
            bridge_record = cfg["dspark_target_bridge"]
            self._dspark_target_bridge_topology = (
                int(bridge_record["num_hidden_layers"]),
                int(bridge_record["n_mtp_layers"]),
            )

        # Preserve the producer's physical spelling before resolver namespace
        # canonicalization. Digest membership is over these exact names. The
        # DSpark bridge covers the complete activation contract (including a
        # delegated stock-NVFP4 target); custom CB targets add the legacy
        # identity mapping when no bridge is present.
        physical_by_canonical: dict[str, str] = dict(dspark_target_bridge)
        declared_targets: set[str] = set()
        for group in cfg["config_groups"].values():
            declared_targets.update(
                _canonical_target(str(target))
                for target in group.get("targets", [])
            )
            scheme = group.get("scheme")
            if scheme is None:
                continue
            if not isinstance(scheme, dict):
                raise ValueError("CB config group scheme must be an object")
            scheme_target = next(iter(group.get("targets", ())), "<no target>")
            self._validate_cb_format_scheme(
                scheme, str(scheme_target), runtime_contract)
            for raw_target in group.get("targets", []):
                target = str(raw_target)
                canonical = _canonical_target(target)
                physical = dspark_target_bridge.get(canonical, target)
                self._validate_cb_activation_scheme(
                    scheme,
                    target,
                    self._nvfp4_activation_contract,
                    physical_target=physical,
                )
                if scheme.get("activation_contract") is None:
                    continue
                previous = physical_by_canonical.setdefault(
                    canonical, physical
                )
                if previous != physical:
                    raise ValueError(
                        f"contracted CB runtime target {canonical!r} maps to "
                        f"conflicting physical targets {previous!r} and "
                        f"{physical!r}"
                    )
        if dspark_target_bridge:
            bridge_keys = set(dspark_target_bridge)
            undeclared = bridge_keys - declared_targets
            if undeclared:
                raise ValueError(
                    "dspark_target_bridge construction targets must be "
                    "declared by config_groups; "
                    f"undeclared={sorted(undeclared)}"
                )
        # Normalise stored namespaces ONCE, here, so all downstream resolution
        # (ours and the delegated CT config's) sees canonical target names.
        cfg = dict(cfg)
        cfg["config_groups"] = {
            name: {**g, "targets": [_canonical_target(t)
                                    for t in g.get("targets", [])]}
            for name, g in cfg["config_groups"].items()
        }
        cfg["ignore"] = [_canonical_target(i) for i in cfg.get("ignore", [])]
        self._full_config = cfg
        self.config_groups = cfg["config_groups"]
        self.ignore = list(cfg["ignore"])
        self._target_physical_name = physical_by_canonical
        stock_groups: dict = {}
        for name, g in self.config_groups.items():
            if "scheme" in g:                        # CB group (our vocabulary)
                for t in g["targets"]:
                    self.target_scheme[t] = g["scheme"]
                    self._cb_targets.add(t)
            elif (str(g.get("format", "")).strip().lower()
                  == "source-passthrough"
                  or (isinstance(g.get("weights"), dict)
                      and bool(g["weights"].get("source_passthrough")))):
                # Producer metadata for source-format units is not compressed-
                # tensors vocabulary.  Gridbook owns their dispatch through
                # the versioned ``source_passthrough`` declaration below, and
                # _build_ct_config adds those units to CT's ignore list.  In
                # particular, fields such as ``element_dtype`` and
                # ``source_passthrough`` are intentionally rejected by CT's
                # Pydantic schema and must not be reinterpreted as a stock
                # quantization group.
                continue
            else:                                    # stock CT vocabulary
                stock_groups[name] = g
                for t in g.get("targets", []):
                    self._stock_group_by_target[str(t)] = name
        # SOURCE-format passthrough units.  Parsed after ``_cb_targets`` is
        # populated so a unit claimed by both vocabularies is caught here
        # rather than resolved by whichever branch of ``get_quant_method``
        # happens to run first.
        self._passthrough_units = _parse_passthrough_declaration(
            cfg,
            canonicalize=_canonical_target,
            cb_targets=frozenset(self._cb_targets),
        )
        # Quantized embedding units.  Parsed alongside the passthrough units
        # and against the same ``_cb_targets`` set, so a unit claimed by two
        # vocabularies is caught here rather than by whichever branch of
        # ``get_quant_method`` happens to run first.
        self._embedding_units = _parse_embedding_declaration(
            cfg,
            canonicalize=_canonical_target,
            cb_targets=frozenset(self._cb_targets),
        )
        self._per_expert_format_groups = _parse_per_expert_format_groups(
            cfg,
            runtime_contract=_RUNTIME_CONTRACT,
            cb_schemes=self.target_scheme,
            canonicalize=_canonical_target,
        )
        self._per_expert_serving_prefixes = {
            group.tensor_prefix: _canonical_target(group.tensor_prefix)
            for layer_groups in self._per_expert_format_groups.values()
            for family in ("w13", "w2")
            for group in layer_groups.groups(family)
        }
        self._alias_collapsed_shared_prefixes()
        self.ct_config = (self._build_ct_config(stock_groups)
                          if stock_groups else None)
        # Validate the complete payload now, including delegated stock NVFP4
        # targets that Gridbook's custom methods will never otherwise see.
        self._ensure_nvfp4_activation_payload()
        self._resolved = True

    def _alias_collapsed_shared_prefixes(self) -> None:
        """HunYuan-V3-style shared-expert dispatch aliases. HYV3MoEFused builds
        its shared MLP with ``prefix=f"{prefix}"`` — the ``.shared_mlp`` segment
        never reaches ``get_quant_method``, which instead sees the PARENT-prefix
        names ``…mlp.gate_up_proj`` / ``…mlp.down_proj``. Module paths (params,
        checkpoint tensors) DO keep ``.shared_mlp.``, so only the dispatch key
        collapses. MTP wraps the same block under ``.mtp_block.`` before making
        that parent-prefix call, so it needs both nested and collapsed MTP
        aliases as well.

        Alias every ``….shared_mlp.<leaf>`` CB target and ignore entry to all
        valid construction-time forms so the CB method owns the shared expert
        natively. A missing alias is fatal in the top-level loader; decoding CB
        into a plain bf16 Linear is forbidden because its upstream dispatch can
        select cuBLAS or Triton. Collision-safe: ``setdefault`` keeps any real
        key authoritative, and aliases for module trees that do not exist match
        nothing. Runs before the delegated-CT build so its ignore list covers
        the aliases too."""

        def aliases(name: str) -> set[str]:
            if ".shared_mlp." not in name:
                return set()
            out = {name.replace(".shared_mlp.", ".")}
            if ".mlp.shared_mlp." in name:
                nested = name.replace(
                    ".mlp.shared_mlp.", ".mtp_block.mlp.shared_mlp.")
                out.add(nested)
                out.add(nested.replace(".shared_mlp.", "."))
            return out

        for target in [k for k in self.target_scheme if ".shared_mlp." in k]:
            for alias in aliases(target):
                self.target_scheme.setdefault(alias,
                                              self.target_scheme[target])
                if target in self._target_physical_name:
                    self._target_physical_name.setdefault(
                        alias, self._target_physical_name[target]
                    )
                self._cb_targets.add(alias)
        for ignored in list(self.ignore):
            self.ignore.extend(sorted(aliases(ignored)))

    def _build_ct_config(self, stock_groups: dict):
        """A stock CompressedTensorsConfig over the non-CB groups. They are
        already CT vocabulary; we re-key quant_method, add our CB modules to
        CT's ignore (so CT never owns them), and give it a valid top-level
        format (our container's is a CB marker; stock groups carry per-group
        formats that CT reads under "mixed-precision")."""
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
            CompressedTensorsConfig,
        )
        ct_dict = dict(self._full_config)
        # Gridbook-only construction/physical namespace metadata is consumed
        # above and must not leak into compressed-tensors' closed vocabulary.
        ct_dict.pop("dspark_target_bridge", None)
        ct_dict["quant_method"] = "compressed-tensors"
        ct_dict["config_groups"] = dict(stock_groups)
        # Passthrough units join CB targets in CT's ignore: both are owned by
        # Gridbook's own dispatch, and a stock group that also happens to name
        # one must not be able to claim it.
        # Quantized embeddings join them for the same reason, and for one more:
        # compressed-tensors' embedding path accepts weight-only INT schemes
        # and RAISES for FP8/NVFP4, so a stock group naming the embedding would
        # not merely mis-own it -- it would refuse to load the artifact at all.
        ct_dict["ignore"] = (list(self.ignore) + sorted(self._cb_targets)
                             + sorted(self._passthrough_units)
                             + sorted(self._embedding_units))
        ct_dict.pop("codebook_file", None)
        ct_dict.pop("provenance", None)
        # Gridbook has already attested this producer-owned container record;
        # compressed-tensors does not define the field in its own schema.
        ct_dict.pop("execution_contracts", None)
        ct_dict.pop("per_expert_format_groups", None)
        raw_fmt = str(self._full_config.get("format", ""))
        if raw_fmt in ("", "nvfp4_cb", "fp8_cb", "cb", "mixed-precision"):
            ct_dict["format"] = "mixed-precision"
        return CompressedTensorsConfig.from_config(ct_dict)

    def __repr__(self) -> str:
        return (f"PrismaQuantConfig(resolved={self._resolved}, "
                f"cb_targets={len(self.target_scheme)}, "
                f"stock_ct={'yes' if self.ct_config is not None else 'no'})")

    @classmethod
    def get_name(cls):
        return _QUANT_METHOD_CANONICAL

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Shipping CUDA decode and grouped-MoE bindings require BF16 inputs.
        # Advertising FP16 lets vLLM accept a model dtype that later fails at
        # the native boundary (or changes dtype at a fallback/crossover).
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PrismaQuantConfig":
        # Defer parsing: a pointer config resolves quant_config.json lazily.
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, **kwargs):
        # "gridbook" is the registry key going forward; "prismaquant" is the
        # legacy key older local artifacts carry — both dispatch here.
        if user_quant in _QUANT_METHOD_ACCEPTED:
            return _QUANT_METHOD_CANONICAL
        if hf_quant_cfg is not None and \
                hf_quant_cfg.get("quant_method") in _QUANT_METHOD_ACCEPTED:
            return _QUANT_METHOD_CANONICAL
        return None

    # -- codebook sidecar (loaded once, shared across all layers) ------------
    def get_codebooks(self) -> dict[str, torch.Tensor]:
        if self._codebooks is None:
            from .cb_digest import load_codebooks

            # Resolve the full quant config before opening the sidecar: the
            # expected per-table hashes live outside the .pqcb in
            # quant_config.json.  A digest carried only by the sidecar would
            # attest to the wrong file just as readily as the right one.
            self._ensure_resolved()
            provenance = self._full_config.get("provenance")
            if provenance is None:
                expected_sha256 = None       # legacy, intentionally optional
            elif not isinstance(provenance, dict):
                raise ValueError(
                    "provenance must be an object when it is present")
            elif ("codebook_sha256" in provenance
                  and provenance["codebook_sha256"] is None):
                raise ValueError(
                    "provenance.codebook_sha256 must be an object when "
                    "declared; omit the field for a legacy artifact")
            else:
                expected_sha256 = provenance.get("codebook_sha256")
            model_dir, revision = self._get_sidecar_source()
            # This is the single choke point used by linear.py, moe.py, and
            # moe_toplevel_loader.py, and it is memoized after verification.
            self._codebooks = load_codebooks(
                _resolve_model_file(
                    model_dir, self.codebook_file, revision=revision),
                expected_sha256=expected_sha256)
        return self._codebooks

    # -- per-prefix scheme resolution (handles vLLM fused qkv/gate_up) -------
    def _is_ignored(self, prefix: str) -> bool:
        """Read ``ignore`` exactly as delegated compressed-tensors does.

        In particular, fused modules are checked through their unfused shard
        names and regexes use compressed-tensors' ``regex`` engine.  Keeping a
        local near-copy caused the same list to mean different things on the CB
        and stock sides of one mixed artifact.
        """
        # Lazy so codec/config tests that provide the repository's minimal
        # vLLM stubs can still import this module without having to recreate
        # compressed-tensors' entire package tree. Production always has the
        # helper because compressed-tensors is a supported vLLM dependency.
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (  # noqa: E501
            should_ignore_layer,
        )
        fused = dict(_FUSED_FALLBACK)
        fused.update(getattr(self, "packed_modules_mapping", {}) or {})
        return any(should_ignore_layer(base, self.ignore, fused)
                   for base in _candidate_bases(prefix))

    def shard_target_keys(self, prefix: str, *,
                          unfused_fallback: bool = False) -> list[str]:
        """``target_scheme`` keys naming the CB shards of (possibly fused)
        *prefix*, in shard order — ``[]`` if none resolve.

        THE single owner of fused-shard resolution: ``_scheme_for_prefix``
        (which format does this module decode as?) and
        ``PrismaQuantCBLinearMethod._shard_roles`` (which per-role codebooks
        does it concatenate?) must agree module-for-module, and before issue #1
        they were two hand-rolled copies that had already drifted — the copies
        built their shard keys from the CANONICAL prefix only, so once
        ``apply_vllm_mapper`` moved the stored keys into the mapper's namespace
        a fused GDN ``in_proj_qkvz`` resolved to nothing (silent BF16
        fall-through) and every dense ``_shard_roles`` returned ``[]`` (a
        load-time width assert). Namespace choice is delegated wholesale to
        ``_candidate_bases``.

        Bases are tried in order and the FIRST base with any hit wins **whole**:
        hits are never mixed across bases, because two vintages of one key can
        name two different on-disk tensors and pairing shards across them would
        silently fuse the wrong weights.

        ``unfused_fallback`` reproduces ``_shard_roles``' extra ``or [leaf]``
        rung — a plain Linear is its own single role. ``_scheme_for_prefix``
        deliberately omits it (it has already tried the exact keys itself, and
        a bare-leaf retry there would only re-ask the same question).
        """
        pmm = getattr(self, "packed_modules_mapping", {}) or {}
        for base in _candidate_bases(prefix):
            leaf = base.split(".")[-1]
            primary = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf)
            spellings = [primary] if primary is not None else []
            spellings.extend(_ALTERNATE_SHARD_SPELLINGS.get(leaf, ()))
            if primary is None and unfused_fallback:
                # ``_shard_roles``' "a plain Linear is its own single role"
                # rung. It stays LAST so an alternate spelling that actually
                # matches wins, and it is still withheld from a leaf with a
                # declared fusion, exactly as before.
                spellings.append([leaf])
            if not spellings:
                continue
            stem = base[: -len(leaf)]
            for shard_leaves in spellings:
                hits = [stem + sl for sl in shard_leaves
                        if stem + sl in self.target_scheme]
                if hits:
                    return hits
        return []

    def _scheme_for_prefix(self, prefix: str) -> dict | None:
        for base in _candidate_bases(prefix):
            if base in self.target_scheme:
                return self.target_scheme[base]
        owners = self.fused_role_owners(prefix)
        if owners:
            # A heterogeneous fused module has no truthful single scheme.
            # ``get_quant_method`` consumes the complete owner list before it
            # reaches this helper and installs the composite method.  Returning
            # None here keeps config/introspection callers from mistaking the
            # first role's format for the whole merged module.
            if any(owner.kind != "cb" for owner in owners):
                return None
            if not self._fused_owners_share_single_method(owners):
                return None
            return owners[0].payload
        schemes = [self.target_scheme[k]
                   for k in self.shard_target_keys(prefix)]
        if not schemes:
            return None
        fmt_keys = ("grid", "mode", "k", "n_sub", "type_size",
                    "activation_contract")
        sig = {kk: schemes[0].get(kk) for kk in fmt_keys}
        for s in schemes[1:]:
            if {kk: s.get(kk) for kk in fmt_keys} != sig:
                return None
        return schemes[0]

    def fused_role_owners(self, prefix: str) -> list[_FusedRoleOwner]:
        """Complete ordered owners for a fused Linear, or ``[]``.

        vLLM exposes one module for q/k/v-, gate/up-, and architecture-specific
        merges, but each source sibling remains a distinct checkpoint tensor.
        Gridbook formats are per-Linear decisions, so the siblings may use
        different CB rungs or mix CB with a declared source-native lane.  This
        resolver preserves the order published by ``packed_modules_mapping``
        (falling back to Gridbook's audited table) and only returns a spelling
        when *every* role has an explicit owner. Partial ownership is rejected
        separately rather than letting the first role claim the whole merge.
        """

        pmm = getattr(self, "packed_modules_mapping", {}) or {}
        for base in _candidate_bases(prefix):
            leaf = base.rsplit(".", 1)[-1]
            primary = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf)
            spellings = [primary] if primary is not None else []
            spellings.extend(_ALTERNATE_SHARD_SPELLINGS.get(leaf, ()))
            if not spellings:
                continue
            stem = base[: -len(leaf)]
            for shard_leaves in spellings:
                if not isinstance(shard_leaves, (list, tuple)) \
                        or len(shard_leaves) < 2:
                    continue
                owners: list[_FusedRoleOwner] = []
                for shard_leaf in shard_leaves:
                    target = stem + shard_leaf
                    scheme = self.target_scheme.get(target)
                    source_format = self._passthrough_format(target)
                    if scheme is not None and source_format is not None:
                        raise ValueError(
                            f"fused role {target!r} is claimed by both CB and "
                            "source-passthrough dispatch")
                    if scheme is not None:
                        owners.append(_FusedRoleOwner(target, "cb", scheme))
                    elif source_format is not None:
                        owners.append(_FusedRoleOwner(
                            target, "source", source_format))
                    else:
                        break
                if len(owners) == len(shard_leaves):
                    return owners
        return []

    def incomplete_fused_roles(self, prefix: str) -> list[str]:
        """Missing role targets when a known fusion is only partly owned.

        A single CB hit used to fall through ``_scheme_for_prefix`` and claim
        the whole merged module, leaving its unrepresented sibling to be
        interpreted through the first role's format. Complete alternate
        spellings (notably DeepSeek w1/w3 after gate/up misses) still win; only
        after every spelling has been checked do we report the best partial
        match. An exact fused CB/source declaration is a deliberate whole-
        module representation and therefore not partial.
        """

        for base in _candidate_bases(prefix):
            if (base in self.target_scheme
                    or self._passthrough_format(base) is not None):
                return []
        pmm = getattr(self, "packed_modules_mapping", {}) or {}
        best_owned = -1
        best_missing: list[str] = []
        for base in _candidate_bases(prefix):
            leaf = base.rsplit(".", 1)[-1]
            primary = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf)
            spellings = [primary] if primary is not None else []
            spellings.extend(_ALTERNATE_SHARD_SPELLINGS.get(leaf, ()))
            stem = base[: -len(leaf)]
            for shard_leaves in spellings:
                if not isinstance(shard_leaves, (list, tuple)) \
                        or len(shard_leaves) < 2:
                    continue
                owned = []
                missing = []
                for shard_leaf in shard_leaves:
                    target = stem + shard_leaf
                    if (target in self.target_scheme
                            or self._passthrough_format(target) is not None):
                        owned.append(target)
                    else:
                        missing.append(target)
                if not missing:
                    return []
                if owned and len(owned) > best_owned:
                    best_owned = len(owned)
                    best_missing = missing
        return best_missing

    def _stock_group_for_prefix(
        self, layer: torch.nn.Module, prefix: str, *, moe: bool = False
    ) -> tuple[str | None, dict | None]:
        """The delegated config group that declares *prefix*, or ``(None, None)``.

        Resolution is delegated wholesale to compressed-tensors'
        ``find_matched_target`` — the same helper vLLM itself uses — so a
        regex target, a fused shard, or a module-class target means the same
        thing on both sides of one mixed artifact. ``_is_ignored`` made the
        opposite choice once (a local near-copy) and the two lists drifted;
        this does not repeat that.
        """

        if not self._stock_group_by_target:
            return None, None
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (  # noqa: E501
            find_matched_target,
        )
        fused = dict(_FUSED_FALLBACK)
        fused.update(getattr(self, "packed_modules_mapping", {}) or {})
        targets = list(self._stock_group_by_target)
        suffixes = _MOE_DECLARATION_SUFFIXES if moe else ("",)
        for base in _candidate_bases(prefix):
            for suffix in suffixes:
                try:
                    matched = find_matched_target(base + suffix, layer,
                                                  targets, fused)
                except ValueError:
                    # Older compressed-tensors raised for "no match" where the
                    # audited builds return None. Either way it means this
                    # prefix is not a delegated target; the unconditional
                    # Triton rule still runs on whatever vLLM resolved.
                    continue
                if matched is None:
                    continue
                group_name = self._stock_group_by_target[matched]
                return group_name, self.config_groups.get(group_name)
        return None, None

    def _passthrough_format(self, prefix: str) -> "_SourceFormat | None":
        """The declared source format for *prefix*, or ``None``.

        Matched through ``_candidate_bases`` for the same reason every other
        lookup here is: the declaration is stored in the canonical namespace
        and the serving prefix may arrive in any of the vintages that helper
        enumerates.
        """

        if not self._passthrough_units:
            return None
        for base in _candidate_bases(prefix):
            for candidate in (base, *_source_passthrough_aliases(base)):
                fmt = self._passthrough_units.get(candidate)
                if fmt is not None:
                    return fmt
        return None

    def _embedding_format(self, prefix: str) -> "_EmbeddingFormat | None":
        """The declared quantized-embedding format for *prefix*, or ``None``.

        Matched through ``_candidate_bases`` for the same reason every other
        lookup here is: the declaration is stored in the canonical namespace
        and the serving prefix may arrive in any of the vintages that helper
        enumerates.
        """

        if not self._embedding_units:
            return None
        for base in _candidate_bases(prefix):
            fmt = self._embedding_units.get(base)
            if fmt is not None:
                return fmt
        return None

    def _delegate_passthrough(
        self, layer: torch.nn.Module, prefix: str, fmt: "_SourceFormat"
    ) -> "QuantizeMethodBase | None":
        """Hand a passthrough unit to vLLM's native method, fail-closed.

        Two gates, both at model load and in this order:

        1. **Device attestation.**  Which backend a format resolves to is a
           property of the (format, device) pair — vLLM's MXFP4 oracle admits
           the whole sm12x family for a rung that only works on part of it — so
           an unaudited device is refused before any vLLM object is built.
        2. **Backend preflight.**  Constructing the method is what makes the
           resolved backend a fact rather than a prediction (the MXFP4 method
           runs its oracle in ``__init__`` and stores the winner on
           ``self.mxfp4_backend`` / ``self.experts_cls``), so the check happens
           the moment the constructor returns — still model-load time, still
           before any weights are processed.
        3. **Tensor parallel.**  A passthrough lane may be served above one
           rank only when the lane itself enforces structural shard laws at
           weight construction.  The source-FP8 W8A16 lane does
           (``fp8_source_w8a16._require_shard_alignment``: whole-128 local
           extents on the sharded axis, per-role alignment on merged planes,
           and its grouped-BMM arm admitted only at a MEASURED shard degree,
           since column-sharding a grouped plane divides the kernel's group
           count).  Every other passthrough format has no sharded audit at
           all, so it refuses above one HERE rather than discovering it after
           weights are copied.
        """

        if fmt.id != _WIRE_FP8_BLOCK128_TP:
            self._require_tp1_serving(
                f"the source-passthrough unit format {fmt.id!r}", prefix)
        _require_passthrough_device(
            fmt, prefix=prefix, capability=_live_device_capability())
        method = _build_passthrough_method(fmt, layer, prefix)
        require_native_passthrough_backend(
            prefix=prefix, source_format=fmt, method=method, layer=layer)
        return method

    def _delegate(self, layer: torch.nn.Module, prefix: str, *,
                  moe: bool = False) -> "QuantizeMethodBase | None":
        """Hand *prefix* to the stock compressed-tensors config, fail-closed.

        THE delegation choke point (ROADMAP D0.2). vLLM resolves a delegated
        group through its own backend ladder, and that ladder can silently
        rewrite a declared W4A4 into weight-only W4A16 (Marlin) or land on a
        Triton-backed backend (``emulation``). Both are decided *inside* the
        call below — ``CompressedTensorsW4A4Nvfp4MoEMethod.__init__`` calls
        ``select_nvfp4_moe_backend`` and stores the winner, and the dense path
        attaches its resolved scheme/kernel to the layer — so the moment the
        call returns is the earliest point at which the resolved backend is a
        fact rather than a prediction, and it is still model-load time.
        Checking here (rather than in ``moe.py``/``linear.py``, which never see
        a delegated layer) also keeps every delegated layer class on one rule.

        The stock compressed-tensors ladder IS tensor-parallel-capable in
        vLLM, but Gridbook's audited delegation contract (backend preflight,
        MXFP8 lane, NVFP4/FP8-DYNAMIC menu) was measured at TP=1 only, so a
        delegated group refuses above one at this same choke point instead of
        serving an unaudited sharding combination.
        """

        self._require_tp1_serving(
            "delegated stock compressed-tensors groups"
            + (" (MoE)" if moe else ""), prefix)
        method = self.ct_config.get_quant_method(layer, prefix)
        group_name, group = self._stock_group_for_prefix(layer, prefix, moe=moe)
        require_native_delegated_backend(
            prefix=prefix, group_name=group_name, group=group,
            method=method, layer=layer,
        )
        return method

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> "QuantizeMethodBase | None":
        # Tensor-parallel policy is resolved PER ARM below, not as a blanket
        # pre-gate: dense CB Linears admit TP>1 under their structured
        # shard-alignment gates (``linear.ShardGroupAlignmentError``), while
        # every other surface calls ``_require_tp1_serving`` with its own
        # name at method construction. The delegated/passthrough choke
        # points carry their own gates so every caller shares one rule.
        self._ensure_resolved()
        from .linear import PrismaQuantCBLinearMethod

        # Keep the delegated CT config's fused-module mapping in lockstep.
        if self.ct_config is not None:
            self.ct_config.packed_modules_mapping = getattr(
                self, "packed_modules_mapping", {}) or {}

        if isinstance(layer, LinearBase):
            # 1) A complete vLLM fusion is resolved role-by-role before the
            #    single-scheme path. The legacy merged method remains truthful
            #    when physical representation and activation metadata match.
            #    Otherwise a loader-ABI-gated composite preserves each role's
            #    method and concatenates only the outputs.
            owners = self.fused_role_owners(prefix)
            if owners:
                if self._fused_owners_share_single_method(owners):
                    if owners[0].kind == "cb":
                        # DENSE CB, single legacy merged method: the admitted
                        # TP>1 surface. Shard legality is enforced per layer
                        # by PrismaQuantCBLinearMethod.create_weights.
                        scheme = owners[0].payload
                        self._require_cb_device_capability(scheme, prefix)
                        return PrismaQuantCBLinearMethod(self, scheme, prefix)
                    return self._delegate_passthrough(
                        layer, prefix, owners[0].payload)

                # No blanket TP gate on this surface. The composite owns no
                # law of its own beyond "a merged projection is column
                # parallel": MixedFusedLinearMethod.create_weights derives the
                # column degree from vLLM's constructor arguments and builds
                # every carrier at that ROLE's whole-tensor output size, so
                # each role's existing shard law — CB's group/quantum gate,
                # the source lane's block gate, or whatever refusal
                # _delegate_passthrough raises for an unqualified format —
                # decides legality per role, before a parameter exists. The
                # top-level mixed router then narrows each whole checkpoint
                # plane to this rank.
                if not self._has_mixed_fused_loader():
                    roles = ", ".join(
                        f"{owner.target} ({owner.kind})" for owner in owners)
                    raise RuntimeError(
                        f"fused module {prefix!r} has independently encoded "
                        f"roles [{roles}], but the selected model class's "
                        "effective load_weights method does not expose "
                        "Gridbook mixed-fused loader ABI 1; add and audit the "
                        "top-level loader wrapper for that exact class before "
                        "serving this valid mixed-format fusion"
                    )

                from .mixed_linear import MixedFusedLinearMethod
                role_methods = []
                for owner in owners:
                    if owner.kind == "cb":
                        self._require_cb_device_capability(
                            owner.payload, owner.target)
                        method = PrismaQuantCBLinearMethod(
                            self, owner.payload, owner.target)
                    else:
                        method = self._delegate_passthrough(
                            layer, owner.target, owner.payload)
                    role_methods.append((owner.target, method))
                return MixedFusedLinearMethod(prefix, role_methods)

            missing_roles = self.incomplete_fused_roles(prefix)
            if missing_roles:
                raise RuntimeError(
                    f"fused module {prefix!r} has only partial explicit role "
                    f"ownership; missing {missing_roles}. Every physical role "
                    "must declare a supported CB or source-native owner before "
                    "the merged module can be constructed"
                )

            # 1b) Ordinary CB target (has a "scheme") — ours (precise and
            #     ahead of the ignore test).
            scheme = self._scheme_for_prefix(prefix)
            if os.environ.get("PRISMAQUANT_DEBUG_PREFIXES") == "1":
                import sys
                print(f"[pq-prefix] {prefix} -> "
                      f"{'CB' if scheme is not None else 'no-scheme'}",
                      file=sys.stderr, flush=True)
            if scheme is not None:
                # DENSE CB target: the admitted TP>1 surface (see the fused
                # arm above). Unsupported CB LAYOUTS keep refusing at model
                # load through the same format gates as TP=1.
                self._require_cb_device_capability(scheme, prefix)
                return PrismaQuantCBLinearMethod(self, scheme, prefix)
            # 1c) SOURCE-format passthrough -> vLLM's own native method for
            #     that format, guarded by device attestation + backend
            #     preflight. Ahead of the ignore test for the same reason CB
            #     is: an explicit per-unit declaration outranks a pattern.
            fmt = self._passthrough_format(prefix)
            if fmt is not None:
                return self._delegate_passthrough(layer, prefix, fmt)
            # 2) explicitly-ignored -> BF16 passthrough. This is vLLM-native
            #    surface (stock UnquantizedLinearMethod with engine-owned
            #    partitioning), not a Gridbook kernel contract, so no TP gate
            #    belongs here; every shipped artifact carries ignored Linears
            #    (e.g. the sub-quantum GDN scaler projections).
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            # 3) stock NVFP4 / FP8_DYNAMIC -> compressed-tensors delegation
            #    (canonical prefix — CT targets are serving-namespace names).
            if self.ct_config is not None:
                return self._delegate(layer, _canonical_prefix(prefix))
            return UnquantizedLinearMethod()

        if isinstance(layer, VocabParallelEmbedding):
            # ParallelLMHead SUBCLASSES VocabParallelEmbedding but is an output
            # projection, and compressed-tensors already serves it quantized
            # through its own LINEAR method. Only a true lookup table may be
            # claimed here; the head keeps falling through to delegation.
            embed_fmt = (None if isinstance(layer, ParallelLMHead)
                         else self._embedding_format(prefix))
            if embed_fmt is not None:
                # Gridbook-owned packed-embedding kernel contract, attested
                # only with whole-vocab replicated weights.
                self._require_tp1_serving("quantized embedding units", prefix)
                from .embedding import GridbookNVFP4EmbeddingMethod
                return GridbookNVFP4EmbeddingMethod(embed_fmt, prefix)
            if self.ct_config is not None:
                method = self._delegate(layer, prefix)
                if method is not None:
                    return method
            return UnquantizedEmbeddingMethod()

        # FusedMoE expert stacks (RoutedExperts): a CB expert group -> our MoE
        # method; else delegate to the stock CT MoE path.
        if RoutedExperts is not None and isinstance(layer, RoutedExperts):
            mixed_groups = self._mixed_moe_groups_for_prefix(prefix)
            scheme = self._moe_scheme_for_prefix(prefix)
            fmt = self._passthrough_format(prefix)
            # Unlike dense siblings whose outputs vLLM merely concatenates,
            # one RoutedExperts object owns a single physical expert stack.
            # A source-native parent and CB children would make two methods
            # claim the same resident weights.  Refuse that overlap before
            # dispatch order can silently pick one representation.
            if fmt is not None and (mixed_groups is not None
                                    or scheme is not None):
                cb_kind = ("per-expert CB groups" if mixed_groups is not None
                           else "a CB expert group")
                raise SourcePassthroughError(
                    f"MoE stack {prefix!r} is declared both as source-native "
                    f"{fmt.id!r} and as {cb_kind}; NVFP4-CB/FP8-CB and "
                    "source MXFP4 may coexist in separate modules of one "
                    "decoder layer, but cannot own the same routed-expert "
                    "stack"
                )
            if mixed_groups is not None:
                if int(layer.moe_config.num_experts) != mixed_groups.num_experts:
                    raise ValueError(
                        f"MoE stack {prefix} has "
                        f"num_experts={layer.moe_config.num_experts}, but "
                        "per_expert_format_groups declares a partition of "
                        f"{mixed_groups.num_experts} experts"
                    )
                # Single-format CB expert stacks now serve above one rank
                # under expert parallelism (see _require_ep_moe_serving). A
                # MIXED stack does not, and refuses here — before any buffer
                # is sized with per-rank geometry the loaders cannot fill.
                self._require_tp1_serving(
                    "CB MoE expert stacks (per-expert format groups)", prefix,
                    note=(
                        "Expert parallelism (-tp N --enable-expert-parallel) "
                        "serves single-format CB expert stacks above one rank, "
                        "but not per-expert format groups: the format "
                        "partition is declared over GLOBAL expert ids and each "
                        "per-format sub-stack is sized from that global "
                        "partition, so a rank owning an arbitrary subset can "
                        "neither size nor fill them, and the resident formats "
                        "would differ per rank. Export this layer as one CB "
                        "format to serve it expert-parallel."
                    ))
                from .moe_mixed import PrismaQuantMixedMoEMethod
                return PrismaQuantMixedMoEMethod(
                    self, layer.moe_config, mixed_groups, prefix
                )
            if scheme is not None:
                self._require_cb_device_capability(scheme, prefix)
                # Above one rank this surface admits ONLY expert parallelism.
                # Tensor parallelism still refuses: vLLM's FusedMoE allocates
                # intermediate-SHARDED buffers at moe tp_size > 1, while a CB
                # stack's last dimension is superblock bytes, so the stacked
                # whole-tensor loaders could not fill them. Expert parallelism
                # shards the EXPERT axis instead, which is the axis a stack is
                # already indexed on: the loaders gather this rank's experts
                # (moe_ep.gather_expert_major) and the forward relabels router
                # ids inside the custom op (moe_ep.remap_local_expert_ids).
                ep_mode = self._require_ep_moe_serving(
                    "CB MoE expert stacks (stacked whole-tensor loader)",
                    prefix, layer)
                if ep_mode != "single_rank":
                    # Announced, not merely permitted: which parallel mode
                    # admitted a layer is the fact an operator needs when a
                    # two-node serve behaves unlike the single-rank one.
                    print(f"[prismaquant-cb] moe_admission {prefix} -> "
                          f"{ep_mode}; this rank holds "
                          f"{int(layer.moe_config.num_local_experts)} of "
                          f"{int(layer.moe_config.num_experts)} experts",
                          flush=True)
                from .moe import PrismaQuantCBMoEMethod
                return PrismaQuantCBMoEMethod(
                    self, layer.moe_config, scheme, prefix)
            if fmt is not None:
                return self._delegate_passthrough(layer, prefix, fmt)
            if self.ct_config is not None:
                return self._delegate(layer, prefix, moe=True)
            return None
        return None

    def _mixed_moe_groups_for_prefix(
        self, prefix: str
    ) -> "_LayerFormatGroups | None":
        """Resolve v1 by layer id *and* exact expert-unit namespace."""

        layer_id = _per_expert_layer_id_for_prefix(prefix)
        if layer_id is None:
            return None
        groups = self._per_expert_format_groups.get(layer_id)
        if groups is None:
            return None
        bases = _candidate_bases(prefix)
        matched = []
        for family in ("w13", "w2"):
            for group in groups.groups(family):
                serving = self._per_expert_serving_prefixes[
                    group.tensor_prefix
                ]
                variants = _candidate_bases(serving)
                matched.append(any(
                    variant == base or variant.startswith(base.rstrip(".") + ".")
                    for variant in variants for base in bases
                ))
        if all(matched):
            return groups
        if any(matched):
            raise ValueError(
                f"MoE stack {prefix}: per_expert_format_groups layer "
                f"{layer_id} mixes tensor prefixes from different expert units"
            )
        return None

    def _moe_scheme_for_prefix(self, prefix: str) -> dict | None:
        """A CB expert stack (targets like ``…experts.gate_up_proj`` /
        ``…experts.down_proj``) under this FusedMoE prefix — return its scheme
        (uniform per layer, so any matching target's scheme is the layer's)."""
        # Canonicalise BOTH sides, exactly as ``_scheme_for_prefix`` does for
        # Linears. Without this the multimodal wrapper breaks experts ONLY:
        # vLLM hands us the serving prefix ``language_model.model.layers.N.mlp.
        # experts`` while the checkpoint-namespace targets read
        # ``model.language_model.layers.N.mlp.experts.gate_up_proj``, so a raw
        # ``startswith`` misses, no CB MoE method is created, no
        # ``w13_cb_qweight``/``w2_cb_qweight`` params exist, and the arch's own
        # expert mapping then derives ``experts.w2_weight.cb_qweight`` and
        # AttributeErrors (35B CB serve boot). Dense Linears were unaffected
        # because their lookup already canonicalised — that asymmetry WAS the bug.
        #
        # Structurally different from the dense lookup (the TARGET is longer
        # than the prefix here, so this is a ``startswith``, not a key lookup),
        # but the namespace question is the same one — so it comes from the same
        # ``_candidate_bases``, on BOTH sides. Cross-vintage matches are safe
        # here: ``_canonical_prefix`` only rewrites the ``language_model``
        # wrapper, i.e. it renames the SAME module; it can never move a match to
        # a different layer index or leaf.
        matches = self._moe_target_keys(prefix)
        if not matches:
            return None
        self._reject_unsupported_moe_target_shapes(prefix)
        schemes = [self.target_scheme[name] for name in matches]
        fmt_keys = ("grid", "mode", "k", "n_sub", "type_size",
                    "activation_contract")
        signature = {key: schemes[0].get(key) for key in fmt_keys}
        for scheme in schemes[1:]:
            if {key: scheme.get(key) for key in fmt_keys} != signature:
                raise ValueError(
                    f"MoE stack {prefix} maps to mixed CB decode/activation "
                    "contracts — export union-find should prevent this"
                )
        return self._resolve_moe_codebook_roles(prefix, matches, schemes)

    def _resolve_moe_codebook_roles(
        self, prefix: str, matches: list[str], schemes: list[dict],
    ) -> dict:
        """Bind a codebook to each logical role of one routed expert stack.

        Through v0.8.2 this returned ``schemes[0]`` outright, and the format
        signature checked just above deliberately does NOT include
        ``codebook_ref``. That combination **failed open**: an artifact naming
        a different book per projection loaded without complaint and decoded
        every stack with whichever scheme sorted first — ``down_proj``, since
        ``_moe_target_keys`` returns sorted names. Silent numerical
        corruption, not a refusal. No exporter we ship could produce such an
        artifact (PrismaQuant self-gates its per-role emission on a ``0.8.3``
        runtime), which is why it was never observed; the guard is here
        because "no producer does this today" is not a loading contract.

        Uniform artifacts — every routed target naming one book, which is
        every artifact shipped to date — resolve to ``schemes[0]`` exactly as
        before, so the whole per-role path stays dark and the decode is
        byte-identical. Only a genuinely per-role artifact takes the split.
        """
        refs = {name: _codebook_ref_key(self.target_scheme[name])
                for name in matches}
        if len(set(refs.values())) == 1:
            return schemes[0]                       # v0.8.2 path, untouched

        by_role: dict[str, tuple[str, ...]] = {}
        claimed_by: dict[str, str] = {}
        for name in matches:
            for role in _MOE_LEAF_ROLES[name.rsplit(".", 1)[-1]]:
                if by_role.setdefault(role, refs[name]) != refs[name]:
                    raise ValueError(
                        f"MoE stack {prefix}: targets {claimed_by[role]!r} and "
                        f"{name!r} both claim the {role!r} codebook role with "
                        "different codebook_ref — a role has exactly one book"
                    )
                claimed_by.setdefault(role, name)
        missing = [role for role in _MOE_ROLES if not by_role.get(role)]
        if missing:
            raise ValueError(
                f"MoE stack {prefix} declares per-role codebooks but names no "
                f"book for {missing} (targets: {matches}). Refusing rather "
                "than decoding those rows with another role's codebook"
            )
        resolved = {key: value for key, value in schemes[0].items()
                    if key != "codebook_ref"}
        # `codebook_ref` is dropped, not carried through. It would otherwise
        # hold whichever role sorted first, and any consumer still reading the
        # singular key would build that one book and decode all three roles
        # with it — the exact fail-open this resolver exists to close. Absent,
        # such a consumer raises instead.
        resolved["codebook_ref_by_role"] = {role: list(by_role[role])
                                            for role in _MOE_ROLES}
        return resolved

    def _reject_unsupported_moe_target_shapes(self, prefix: str) -> None:
        """Refuse routed CB targets this runtime cannot honour.

        Per-role books and per-expert format groups are independent axes the
        exporter's naming can compose (a role qname keeps its optional
        ``.format_group_N`` discriminator), which would need the per-role
        split applied inside each of ``moe_mixed``'s per-group lanes. No
        allocation produces it — all three ``ALLOC-2p53`` variants assign one
        rung per routed layer across all 43 layers, since union-find promotes
        a routed stack to a single format — so 0.8.3 does not implement the
        composition. It must still refuse it by name: ``_moe_target_keys``
        matches on the final component, so a ``…gate_proj.format_group_0``
        target would otherwise be skipped in silence and the stack resolved
        from whatever else matched.
        """
        bases = _candidate_bases(prefix)
        for name in self.target_scheme:
            parts = name.split(".")
            if len(parts) < 2 or not parts[-1].startswith("format_group_"):
                continue
            if parts[-2] not in _MOE_LEAF_ROLES:
                continue
            if any(v.startswith(b.rstrip(".") + ".")
                   for v in _candidate_bases(name) for b in bases):
                raise ValueError(
                    f"MoE stack {prefix}: target {name!r} composes a logical "
                    "codebook role with a per-expert format group. This "
                    "runtime implements per-role books and per-expert format "
                    "groups separately, not together"
                )

    def _moe_target_keys(self, prefix: str) -> list[str]:
        """Resolved CB projection target keys below one RoutedExperts prefix."""

        bases = _candidate_bases(prefix)
        matches = []
        for name in self.target_scheme:
            if name.split(".")[-1] not in _MOE_LEAVES:
                continue
            variants = _candidate_bases(name)
            # A target must be a dotted child of this exact expert prefix.
            # Raw ``startswith`` also accepts neighbouring module names such
            # as ``experts2`` and ``experts_backup``, silently assigning their
            # scheme to the live ``experts`` stack.
            if any(v.startswith(b.rstrip(".") + ".")
                   for v in variants for b in bases):
                matches.append(name)
        return sorted(set(matches))

    def moe_activation_stage_targets(self, prefix: str) -> dict[str, list[str]]:
        """Contracted physical roles feeding the w13 and w2 expert stages."""

        self._ensure_resolved()
        matches = self._moe_target_keys(prefix)
        by_leaf: dict[str, list[str]] = {}
        for name in matches:
            by_leaf.setdefault(name.rsplit(".", 1)[-1], []).append(name)
        w13 = by_leaf.get("gate_up_proj") or (
            by_leaf.get("gate_proj", []) + by_leaf.get("up_proj", [])
        )
        w2 = by_leaf.get("down_proj", [])
        return {"w13": sorted(w13), "w2": sorted(w2)}

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        self._ensure_resolved()
        # vLLM hands us the UNSTACKED mapper (get_unstacked_mapper()), so the
        # q_proj->qkv_proj fusion is NOT rewritten (per-role leaf names survive
        # for _scheme_for_prefix to re-fuse) — but genuine renames/prefixes ARE
        # applied. For hybrid/VLM checkpoints that means the module-nesting
        # prefix (e.g. Qwen3-VL: ``model.language_model.`` -> ``language_model.
        # model.``) must be applied to the CB target keys too: _scheme_for_prefix
        # matches serve-time prefixes EXACTLY (the ignore test additionally
        # accepts a parent-module entry and ``re:`` patterns — see
        # ``_ignore_entry_matches`` — but neither path is a substring search),
        # so an un-remapped key silently falls through to unquantized and the
        # cb_qweight load then fails ("no parameter named …cb_qweight"). Mirror
        # exactly what the delegated stock-CT config does for its own targets.
        # vLLM's compressed-tensors mapper deliberately leaves regex entries
        # untouched: a name mapper cannot safely rewrite regex syntax.  Mirror
        # that rule before handing the same list to our own ignore check.
        regex_ignores = [name for name in self.ignore
                         if name.startswith("re:")]
        literal_ignores = [name for name in self.ignore
                           if not name.startswith("re:")]
        self.ignore = (hf_to_vllm_mapper.apply_list(literal_ignores)
                       + regex_ignores)
        self.target_scheme = hf_to_vllm_mapper.apply_dict(self.target_scheme)
        self._target_physical_name = hf_to_vllm_mapper.apply_dict(
            self._target_physical_name
        )
        self._cb_targets = set(
            hf_to_vllm_mapper.apply_list(sorted(self._cb_targets)))
        if self._per_expert_serving_prefixes:
            physical = list(self._per_expert_serving_prefixes)
            served = hf_to_vllm_mapper.apply_list([
                self._per_expert_serving_prefixes[name] for name in physical
            ])
            self._per_expert_serving_prefixes = dict(zip(physical, served))
        # The delegated-group index is matched against serving prefixes exactly
        # like the CT config's own targets, so it moves into the mapper
        # namespace with them. Regex entries are left alone for the same reason
        # compressed-tensors leaves them alone: a name mapper cannot safely
        # rewrite regex syntax. A stale index would not mis-serve — the D0.2
        # preflight would just lose the declaration and fall back to its
        # unconditional Triton rule — but it would silently weaken the check.
        literal_targets = {name: group
                           for name, group in self._stock_group_by_target.items()
                           if not name.startswith("re:")}
        regex_targets = {name: group
                         for name, group in self._stock_group_by_target.items()
                         if name.startswith("re:")}
        self._stock_group_by_target = {
            **hf_to_vllm_mapper.apply_dict(literal_targets), **regex_targets}
        if self.ct_config is not None:
            self.ct_config.apply_vllm_mapper(hf_to_vllm_mapper)
