"""``PrismaQuantConfig`` — the vLLM quantization config for the NVFP4-CB /
FP8-CB out-of-tree lane (docs/lanes/nvfp4-cb/serving-kernel.md §2, LAYOUT.md §4).

vLLM auto-detects the canonical ``quant_method == "gridbook"`` and accepts the
legacy ``"prismaquant"`` alias declared by the packaged runtime contract. The exporter writes
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
from pathlib import Path
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
from .runtime_contract import load_runtime_contract
from .delegated_preflight import require_native_delegated_backend
from .nvfp4_activation_contract import (
    CONTRACT_KEY as _NVFP4_ACTIVATION_CONTRACT_KEY,
    TENSOR_SUFFIX as _NVFP4_ACTIVATION_TENSOR_SUFFIX,
    parse_contract as _parse_nvfp4_activation_contract,
    validate_payload as _validate_nvfp4_activation_payload,
)
try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
except Exception:  # pragma: no cover - older vLLM
    RoutedExperts = None

_MOE_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")
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
}


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
        # Delegated (non-CB) target -> the stock group that declares it. The
        # D0.2 preflight needs the *declaration*, and the declaration lives in
        # the config group, not in whatever tensors happen to be on disk.
        self._stock_group_by_target: dict[str, str] = {}

    def _get_sidecar_source(self) -> tuple[str, str | None]:
        if self._sidecar_source is None:
            from vllm.config import get_current_vllm_config

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

    def _require_supported_tensor_parallel(self) -> None:
        if self._tp_world_size == 1:
            return
        world_size = _initialized_tensor_parallel_world_size()
        if world_size is None:
            return
        if world_size != 1:
            raise ValueError(
                "Gridbook currently supports tensor-parallel size 1 only; "
                f"the live vLLM worker reports TP={world_size}"
            )
        self._tp_world_size = world_size

    @staticmethod
    def _require_cb_device_capability(scheme: dict, prefix: str) -> None:
        """Reject an FP8-CB artifact before its first illegal prefill.

        FP8-CB's shipping large-M path uses vLLM's native FP8 quantizer and
        CUTLASS scaled GEMM, whose hardware floor is sm_89.  The main decode
        extension itself can compile for sm_80, so a global capability floor
        would otherwise let an A100 load successfully and fail only when the
        first prompt crosses the 16-token decode boundary.  FP4-CB retains the
        broader BF16 transient fallback and is not rejected here.
        """

        if scheme.get("grid") != "fp8" or not torch.cuda.is_available():
            return
        capability = tuple(torch.cuda.get_device_capability())
        if capability < (8, 9):
            raise ValueError(
                f"FP8-CB target {prefix!r} requires compute capability sm_89+ "
                "for its shipping prefill path; "
                f"the current device reports sm_{capability[0]}{capability[1]}"
            )

    @staticmethod
    def _validate_cb_activation_scheme(
        scheme: dict, target: str, contract: dict | None
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
        if (reference is not None and contract is not None
                and target not in contract["target_names"]):
            raise ValueError(
                f"CB target {target!r}: activation_contract scalar is absent "
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

        # Preserve the producer's physical spelling before resolver namespace
        # canonicalization.  Digest membership is over these exact names.
        physical_by_canonical: dict[str, str] = {}
        for group in cfg["config_groups"].values():
            scheme = group.get("scheme")
            if scheme is None:
                continue
            if not isinstance(scheme, dict):
                raise ValueError("CB config group scheme must be an object")
            for raw_target in group.get("targets", []):
                target = str(raw_target)
                self._validate_cb_activation_scheme(
                    scheme, target, self._nvfp4_activation_contract
                )
                if scheme.get("activation_contract") is None:
                    continue
                canonical = _canonical_target(target)
                previous = physical_by_canonical.setdefault(canonical, target)
                if previous != target:
                    raise ValueError(
                        f"contracted CB targets {previous!r} and {target!r} "
                        f"collapse to runtime namespace {canonical!r}"
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
            else:                                    # stock CT vocabulary
                stock_groups[name] = g
                for t in g.get("targets", []):
                    self._stock_group_by_target[str(t)] = name
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
        ct_dict["quant_method"] = "compressed-tensors"
        ct_dict["config_groups"] = dict(stock_groups)
        ct_dict["ignore"] = list(self.ignore) + sorted(self._cb_targets)
        ct_dict.pop("codebook_file", None)
        ct_dict.pop("provenance", None)
        # Gridbook has already attested this producer-owned container record;
        # compressed-tensors does not define the field in its own schema.
        ct_dict.pop("execution_contracts", None)
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
        fmt_keys = ("grid", "mode", "k", "n_sub", "type_size",
                    "activation_contract")
        sig = {kk: schemes[0].get(kk) for kk in fmt_keys}
        for s in schemes[1:]:
            if {kk: s.get(kk) for kk in fmt_keys} != sig:
                raise ValueError(
                    f"fused module {prefix} maps to mixed CB decode "
                    "formats — export union-find should prevent this")
        return schemes[0]

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
                matched = find_matched_target(base + suffix, layer, targets,
                                              fused)
                if matched is None:
                    continue
                group_name = self._stock_group_by_target[matched]
                return group_name, self.config_groups.get(group_name)
        return None, None

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
        """

        method = self.ct_config.get_quant_method(layer, prefix)
        group_name, group = self._stock_group_for_prefix(layer, prefix, moe=moe)
        require_native_delegated_backend(
            prefix=prefix, group_name=group_name, group=group,
            method=method, layer=layer,
        )
        return method

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> "QuantizeMethodBase | None":
        self._require_supported_tensor_parallel()
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
                self._require_cb_device_capability(scheme, prefix)
                return PrismaQuantCBLinearMethod(self, scheme, prefix)
            # 2) explicitly-ignored -> BF16 passthrough.
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            # 3) stock NVFP4 / FP8_DYNAMIC -> compressed-tensors delegation
            #    (canonical prefix — CT targets are serving-namespace names).
            if self.ct_config is not None:
                return self._delegate(layer, _canonical_prefix(prefix))
            return UnquantizedLinearMethod()

        if isinstance(layer, VocabParallelEmbedding):
            if self.ct_config is not None:
                method = self._delegate(layer, prefix)
                if method is not None:
                    return method
            return UnquantizedEmbeddingMethod()

        # FusedMoE expert stacks (RoutedExperts): a CB expert group -> our MoE
        # method; else delegate to the stock CT MoE path.
        if RoutedExperts is not None and isinstance(layer, RoutedExperts):
            scheme = self._moe_scheme_for_prefix(prefix)
            if scheme is not None:
                self._require_cb_device_capability(scheme, prefix)
                from .moe import PrismaQuantCBMoEMethod
                return PrismaQuantCBMoEMethod(
                    self, layer.moe_config, scheme, prefix)
            if self.ct_config is not None:
                return self._delegate(layer, prefix, moe=True)
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
        matches = self._moe_target_keys(prefix)
        if not matches:
            return None
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
        return schemes[0]

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
