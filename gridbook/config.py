"""``PrismaQuantConfig`` — the vLLM quantization config for the NVFP4-CB /
FP8-CB out-of-tree lane (docs/lanes/nvfp4-cb/serving-kernel.md §2, LAYOUT.md §4).

vLLM auto-detects us from ``quant_method == "prismaquant"``. The exporter writes
``config.json['quantization_config']`` as a *pointer* (``config_file`` ->
``quant_config.json`` + ``codebook_file`` -> ``cb_codebooks.pqcb``); the full
``config_groups`` / ``ignore`` live in ``quant_config.json``. We resolve that
sidecar **lazily** (via ``get_current_vllm_config()``, the same handle
``get_codebooks`` uses) since ``from_config`` runs before the model dir is
plumbed. Inlined configs (``config_groups`` already present) are also accepted.

**Mixed-container dispatch (serving-kernel.md §2).** A config group with a
``"scheme"`` key is a CB group (our nvfp4_cb/fp8_cb vocabulary) -> our
``PrismaQuantCBLinearMethod``. A group WITHOUT it uses the exact stock
compressed-tensors vocabulary -> a real ``CompressedTensorsConfig`` we construct
and delegate to (``CompressedTensorsW4A4Nvfp4`` for NVFP4 groups, the fp8 scheme
for FP8_DYNAMIC). ``ignore`` -> ``UnquantizedLinearMethod``.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
except Exception:  # pragma: no cover - older vLLM
    RoutedExperts = None

_MOE_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")

# vLLM fuses these siblings into one module; packed_modules_mapping is populated
# by dispatch time, but we keep the standard mapping as a fallback.
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


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


def _resolve_model_file(model_dir: str, fname: str) -> str:
    """Local path for a sidecar file next to the model. When the model was
    given as a Hub repo id (``vllm serve rdtand/...``) rather than a local
    directory, fetch the sidecar from the Hub — vLLM's own loader handles the
    weights that way, but OUR sidecars (quant_config.json, the .pqcb codebook
    blob) were opened with a plain path join, which broke every serve-by-id
    until 2026-07-22."""
    if os.path.isdir(model_dir):
        return os.path.join(model_dir, fname)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=model_dir, filename=fname)


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
        self.ct_config = None                        # stock CompressedTensorsConfig
        self._codebooks: dict[str, torch.Tensor] | None = None

    # -- lazy resolution of the (possibly pointer) quant config --------------
    def _ensure_resolved(self) -> None:
        if self._resolved:
            return
        cfg = self._raw_config
        if "config_groups" not in cfg:
            cfg_file = cfg.get("config_file", "quant_config.json")
            from vllm.config import get_current_vllm_config
            model_dir = get_current_vllm_config().model_config.model
            with open(_resolve_model_file(model_dir, cfg_file)) as fh:
                cfg = json.load(fh)
            self.codebook_file = cfg.get("codebook_file", self.codebook_file)
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
        stock_groups: dict = {}
        for name, g in self.config_groups.items():
            if "scheme" in g:                        # CB group (our vocabulary)
                for t in g["targets"]:
                    self.target_scheme[t] = g["scheme"]
                    self._cb_targets.add(t)
            else:                                    # stock CT vocabulary
                stock_groups[name] = g
        self._alias_collapsed_shared_prefixes()
        self.ct_config = (self._build_ct_config(stock_groups)
                          if stock_groups else None)
        self._resolved = True

    def _alias_collapsed_shared_prefixes(self) -> None:
        """HunYuan-V3-style shared-expert dispatch collapse. HYV3MoEFused builds
        its shared MLP with ``prefix=f"{prefix}"`` — the ``.shared_mlp`` segment
        never reaches ``get_quant_method``, which instead sees the PARENT-prefix
        names ``…mlp.gate_up_proj`` / ``…mlp.down_proj``. Module paths (params,
        checkpoint tensors) DO keep ``.shared_mlp.``, so only the dispatch key
        collapses. Alias every ``….shared_mlp.<leaf>`` CB target and ignore
        entry to its collapsed form so the CB method owns the shared expert
        natively (packed decode in-kernel) instead of vLLM building plain bf16
        Linears that the loader must fill by decode-at-load. Collision-safe: a
        layer is either dense-MLP (real ``…mlp.<leaf>`` keys, no shared_mlp) or
        MoE-with-shared (no real collapsed keys), and ``setdefault`` keeps any
        real key authoritative. Archs that thread correct shared prefixes are
        unaffected (their aliases match no module prefix). Runs before the
        delegated-CT build so CT's ignore covers the aliases too."""
        for t in [k for k in self.target_scheme if ".shared_mlp." in k]:
            alias = t.replace(".shared_mlp.", ".")
            self.target_scheme.setdefault(alias, self.target_scheme[t])
            self._cb_targets.add(alias)
        self.ignore.extend(ig.replace(".shared_mlp.", ".")
                           for ig in list(self.ignore) if ".shared_mlp." in ig)

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
        ct_dict["quant_method"] = "compressed-tensors"
        ct_dict["config_groups"] = dict(stock_groups)
        ct_dict["ignore"] = list(self.ignore) + sorted(self._cb_targets)
        ct_dict.pop("codebook_file", None)
        ct_dict.pop("provenance", None)
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
        return "gridbook"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

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
        if user_quant in ("gridbook", "prismaquant"):
            return "gridbook"
        if hf_quant_cfg is not None and \
                hf_quant_cfg.get("quant_method") in ("gridbook", "prismaquant"):
            return "gridbook"
        return None

    # -- codebook sidecar (loaded once, shared across all layers) ------------
    def get_codebooks(self) -> dict[str, torch.Tensor]:
        if self._codebooks is None:
            from vllm.config import get_current_vllm_config

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
            model_dir = get_current_vllm_config().model_config.model
            # This is the single choke point used by linear.py, moe.py, and
            # moe_toplevel_loader.py, and it is memoized after verification.
            self._codebooks = load_codebooks(
                _resolve_model_file(model_dir, self.codebook_file),
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
            shard_leaves = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf)
            if shard_leaves is None:
                if not unfused_fallback:
                    continue
                shard_leaves = [leaf]
            stem = base[: -len(leaf)]
            hits = [stem + sl for sl in shard_leaves
                    if stem + sl in self.target_scheme]
            if hits:
                return hits
        return []

    def _scheme_for_prefix(self, prefix: str) -> dict | None:
        for base in _candidate_bases(prefix):
            if base in self.target_scheme:
                return self.target_scheme[base]
        schemes = [self.target_scheme[k]
                   for k in self.shard_target_keys(prefix)]
        if not schemes:
            return None
        fmt_keys = ("grid", "mode", "k", "n_sub", "type_size")
        sig = {kk: schemes[0][kk] for kk in fmt_keys}
        for s in schemes[1:]:
            if {kk: s[kk] for kk in fmt_keys} != sig:
                raise ValueError(
                    f"fused module {prefix} maps to mixed CB decode "
                    "formats — export union-find should prevent this")
        return schemes[0]

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> "QuantizeMethodBase | None":
        self._ensure_resolved()
        from .linear import PrismaQuantCBLinearMethod

        # Keep the delegated CT config's fused-module mapping in lockstep.
        if self.ct_config is not None:
            self.ct_config.packed_modules_mapping = getattr(
                self, "packed_modules_mapping", {}) or {}

        if isinstance(layer, LinearBase):
            # 1) CB target (has a "scheme") — ours (precise, fused-aware; ahead
            #    of the ignore test).
            scheme = self._scheme_for_prefix(prefix)
            if os.environ.get("PRISMAQUANT_DEBUG_PREFIXES") == "1":
                import sys
                print(f"[pq-prefix] {prefix} -> "
                      f"{'CB' if scheme is not None else 'no-scheme'}",
                      file=sys.stderr, flush=True)
            if scheme is not None:
                return PrismaQuantCBLinearMethod(self, scheme, prefix)
            # 2) explicitly-ignored -> BF16 passthrough.
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            # 3) stock NVFP4 / FP8_DYNAMIC -> compressed-tensors delegation
            #    (canonical prefix — CT targets are serving-namespace names).
            if self.ct_config is not None:
                return self.ct_config.get_quant_method(
                    layer, _canonical_prefix(prefix))
            return UnquantizedLinearMethod()

        if isinstance(layer, VocabParallelEmbedding):
            if self.ct_config is not None:
                method = self.ct_config.get_quant_method(layer, prefix)
                if method is not None:
                    return method
            return UnquantizedEmbeddingMethod()

        # FusedMoE expert stacks (RoutedExperts): a CB expert group -> our MoE
        # method; else delegate to the stock CT MoE path.
        if RoutedExperts is not None and isinstance(layer, RoutedExperts):
            scheme = self._moe_scheme_for_prefix(prefix)
            if scheme is not None:
                from .moe import PrismaQuantCBMoEMethod
                return PrismaQuantCBMoEMethod(
                    self, layer.moe_config, scheme, prefix)
            if self.ct_config is not None:
                return self.ct_config.get_quant_method(layer, prefix)
            return None
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
        bases = _candidate_bases(prefix)
        for name, sch in self.target_scheme.items():
            if name.split(".")[-1] not in _MOE_LEAVES:
                continue
            variants = _candidate_bases(name)
            # A target must be a dotted child of this exact expert prefix.
            # Raw ``startswith`` also accepts neighbouring module names such
            # as ``experts2`` and ``experts_backup``, silently assigning their
            # scheme to the live ``experts`` stack.
            if any(v.startswith(b.rstrip(".") + ".")
                   for v in variants for b in bases):
                return sch
        return None

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
        self._cb_targets = set(
            hf_to_vllm_mapper.apply_list(sorted(self._cb_targets)))
        if self.ct_config is not None:
            self.ct_config.apply_vllm_mapper(hf_to_vllm_mapper)
