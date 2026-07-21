"""``PrismaQuantConfig`` — the vLLM quantization config for the NVFP4-CB /
FP8-CB out-of-tree lane (docs/nvfp4-cb-plan/serving-kernel.md §2, LAYOUT.md §4).

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
}


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
            with open(os.path.join(model_dir, cfg_file)) as fh:
                cfg = json.load(fh)
            self.codebook_file = cfg.get("codebook_file", self.codebook_file)
        self._full_config = cfg
        self.config_groups = cfg["config_groups"]
        self.ignore = list(cfg.get("ignore", []))
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
            from safetensors.torch import load_file
            from vllm.config import get_current_vllm_config
            model_dir = get_current_vllm_config().model_config.model
            self._codebooks = load_file(os.path.join(model_dir,
                                                     self.codebook_file))
        return self._codebooks

    # -- per-prefix scheme resolution (handles vLLM fused qkv/gate_up) -------
    def _is_ignored(self, prefix: str) -> bool:
        return any(ig in prefix for ig in self.ignore)

    def _scheme_for_prefix(self, prefix: str) -> dict | None:
        if prefix in self.target_scheme:
            return self.target_scheme[prefix]
        leaf = prefix.split(".")[-1]
        pmm = getattr(self, "packed_modules_mapping", {}) or {}
        shard_leaves = pmm.get(leaf) or _FUSED_FALLBACK.get(leaf)
        if shard_leaves:
            schemes = []
            for shard_leaf in shard_leaves:
                sp = prefix[: -len(leaf)] + shard_leaf
                if sp in self.target_scheme:
                    schemes.append(self.target_scheme[sp])
            if schemes:
                fmt_keys = ("grid", "mode", "k", "n_sub", "type_size")
                sig = {kk: schemes[0][kk] for kk in fmt_keys}
                for s in schemes[1:]:
                    if {kk: s[kk] for kk in fmt_keys} != sig:
                        raise ValueError(
                            f"fused module {prefix} maps to mixed CB decode "
                            "formats — export union-find should prevent this")
                return schemes[0]
        return None

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
            #    of the substring ignore test).
            scheme = self._scheme_for_prefix(prefix)
            if scheme is not None:
                return PrismaQuantCBLinearMethod(self, scheme, prefix)
            # 2) explicitly-ignored -> BF16 passthrough.
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            # 3) stock NVFP4 / FP8_DYNAMIC -> compressed-tensors delegation.
            if self.ct_config is not None:
                return self.ct_config.get_quant_method(layer, prefix)
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
        for name, sch in self.target_scheme.items():
            if name.startswith(prefix) and name.split(".")[-1] in _MOE_LEAVES:
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
        # matches serve-time prefixes EXACTLY (unlike the substring ignore test),
        # so an un-remapped key silently falls through to unquantized and the
        # cb_qweight load then fails ("no parameter named …cb_qweight"). Mirror
        # exactly what the delegated stock-CT config does for its own targets.
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
        self.target_scheme = hf_to_vllm_mapper.apply_dict(self.target_scheme)
        self._cb_targets = set(
            hf_to_vllm_mapper.apply_list(sorted(self._cb_targets)))
        if self.ct_config is not None:
            self.ct_config.apply_vllm_mapper(hf_to_vllm_mapper)
