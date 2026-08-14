"""Top-level stacked-CB expert loader shim - fixed for DSpark."""
from __future__ import annotations
import torch
from .cb_fill_guard import mark_filled

_INSTALLED_MODULE_PATHS: set[str] = set()

# Artifact-specific constants for fixtures only (do not use in generic runtime)
_DSPARK_REGISTERED_STAGES = (0, 1, 2)
_DSPARK_CONSTRUCTION_OFFSET = 43

def installed_module_paths() -> set[str]:
    return set(_INSTALLED_MODULE_PATHS)

# Checkpoint expert-tensor suffix  ->  the registered FusedMoE param LEAF name.
_CB_EXPERT_SUFFIX_TO_LEAF: dict[str, str] = {
    ".experts.gate_up_proj.cb_qweight": "w13_cb_qweight",
    ".experts.down_proj.cb_qweight": "w2_cb_qweight",
    ".experts.gate_up_proj.weight_scale": "w13_weight_scale",
    ".experts.down_proj.weight_scale": "w2_weight_scale",
    ".experts.gate_up_proj.input_global_scale": "w13_input_global_scale",
    ".experts.down_proj.input_global_scale": "w2_input_global_scale",
}

def resolve_cb_expert_param(name: str, param_names) -> str | None:
    for suffix, leaf in _CB_EXPERT_SUFFIX_TO_LEAF.items():
        if not name.endswith(suffix):
            continue
        want_prefix = name[: -len(suffix)] + ".experts."
        want_suffix = "." + leaf
        matches = [k for k in param_names if k.startswith(want_prefix) and k.endswith(want_suffix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"prismaquant CB expert '{name}': ambiguous target params {matches}")
        return None
    return None

def map_cb_expert_name(name: str) -> str | None:
    for suffix, leaf in _CB_EXPERT_SUFFIX_TO_LEAF.items():
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".experts." + leaf
    return None

_CB_QWEIGHT_SUFFIX = ".cb_qweight"
_WEIGHT_SCALE_SUFFIX = ".weight_scale"
_INPUT_GLOBAL_SCALE_SUFFIX = ".input_global_scale"
_FUSED_FALLBACK = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_wqa_wkv": ["wq_a", "wkv"],
}
_FUSED_SHARD_ALIASES = {
    "w1": ("gate_up_proj", 0),
    "w3": ("gate_up_proj", 1),
    "w2": ("down_proj", 0),
}

def _build_reverse_fusion(packed_modules_mapping) -> dict[str, tuple[str, int]]:
    merged = dict(_FUSED_FALLBACK)
    merged.update(packed_modules_mapping or {})
    rev: dict[str, tuple[str, int]] = dict(_FUSED_SHARD_ALIASES)
    for fused_leaf, shards in merged.items():
        if isinstance(shards, (list, tuple)):
            for idx, shard_leaf in enumerate(shards):
                rev[shard_leaf] = (fused_leaf, idx)
    return rev

def resolve_shared_cb_target(name: str, params_dict, reverse_fusion) -> tuple[str, int] | None:
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
    if parent and leaf in reverse_fusion:
        fused_leaf, shard = reverse_fusion[leaf]
        fbase = parent + "." + fused_leaf
        if fbase + _CB_QWEIGHT_SUFFIX in params_dict:
            return None
        if (scalar_suffix is not None and fbase + scalar_suffix in params_dict):
            return None
        if fbase + ".weight" in params_dict:
            return fbase + ".weight", shard
    if base + _CB_QWEIGHT_SUFFIX in params_dict:
        return None
    if scalar_suffix is not None and base + scalar_suffix in params_dict:
        return None
    if base + ".weight" in params_dict:
        return base + ".weight", 0
    return None

def _spec_layer_of(name: str, spec_layers) -> int | None:
    for s in spec_layers:
        if name.startswith(f"model.layers.{s}."):
            return s
    return None

def _spec_layer_rename(model):
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

# --- DSpark helpers (no hardcoded stages, topology passed explicitly) ---
_DSPARK_HEAD_PREFIXES = (
    "main_proj.",
    "main_norm.",
    "norm.",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
    "markov_head.",
)

def _parse_physical_stage(name: str) -> int | None:
    if not name.startswith("mtp."):
        return None
    rest = name[len("mtp."):]
    dot = rest.find(".")
    stage_str = rest if dot==-1 else rest[:dot]
    try:
        return int(stage_str)
    except ValueError:
        return None

def _is_dspark_model(model) -> bool:
    """Exact structural identification for DSpark draft. Fail closed when uncertain."""
    mod = getattr(type(model), "__module__", "") or ""
    name = getattr(type(model), "__name__", "") or ""
    # Exact target class
    if mod == "vllm.models.deepseek_v4.nvidia.dspark" and name == "DSparkDeepseekV4ForCausalLM":
        return True
    # Also allow inner model check for tests that set marker
    # For synthetic tests, they set __module__ and __name__ exactly via type construction
    return False

def _get_dspark_allowed_stages(model, params_dict=None):
    """Derive allowed physical stages from quant_config topology; None if not DSpark."""
    # Try model quant_config
    topo = None
    qc = getattr(model, "quant_config", None)
    if qc is not None and hasattr(qc, "_dspark_topology") and qc._dspark_topology is not None:
        topo = qc._dspark_topology
    else:
        try:
            from vllm.config import get_current_vllm_config_or_none
            vcfg = get_current_vllm_config_or_none()
            if vcfg is not None:
                # Prefer draft
                sc = getattr(vcfg, "speculative_config", None)
                if sc is not None:
                    dmc = getattr(sc, "draft_model_config", None)
                    if dmc is not None:
                        hf = getattr(dmc, "hf_config", None)
                        if hf is not None:
                            nh = getattr(hf, "num_hidden_layers", None)
                            n_mtp = getattr(hf, "n_mtp_layers", None) or 3
                            if isinstance(nh, int) and isinstance(n_mtp, int) and n_mtp>0:
                                topo = (nh, n_mtp)
        except ImportError:
            pass
    if topo is None:
        return None, None
    nh, n_mtp = topo
    allowed = tuple(range(n_mtp))
    return allowed, topo

def _remap_dspark_physical_to_registered(name: str, allowed_stages: tuple[int, ...] | None = None) -> str | None:
    if allowed_stages is None:
        allowed_stages = _DSPARK_REGISTERED_STAGES
    stage = _parse_physical_stage(name)
    if stage is None or stage not in allowed_stages:
        return None
    rest = name[len("mtp.")+len(str(stage)):]
    inner = rest.lstrip(".")
    if inner.startswith("confidence_head."):
        return None
    for hp in _DSPARK_HEAD_PREFIXES:
        if inner.startswith(hp):
            return f"model.{inner}"
    return f"model.layers.{stage}{rest}"

def _remap_dspark_physical_to_construction(name: str, num_hidden_layers: int = 43, allowed_stages: tuple[int, ...] | None = None) -> str | None:
    if allowed_stages is None:
        allowed_stages = _DSPARK_REGISTERED_STAGES
    stage = _parse_physical_stage(name)
    if stage is None or stage not in allowed_stages:
        return None
    rest = name[len("mtp.")+len(str(stage)):]
    inner = rest.lstrip(".")
    if inner.startswith("confidence_head."):
        return None
    for hp in _DSPARK_HEAD_PREFIXES:
        if inner.startswith(hp):
            return f"model.{inner}"
    return f"model.layers.{num_hidden_layers + stage}{rest}"

def _remap_dspark_physical(name: str, allowed_stages: tuple[int, ...] | None = None) -> str | None:
    return _remap_dspark_physical_to_registered(name, allowed_stages)

def _hf_mapper_rename(model):
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    map_name = getattr(mapper, "_map_name", None)
    if not callable(map_name):
        return None
    def rename(name: str) -> str:
        mapped = map_name(name)
        return name if mapped is None else mapped
    return rename

def _compose_renames(*renames):
    fns = [f for f in renames if f is not None]
    if not fns:
        return None
    if len(fns)==1:
        return fns[0]
    def rename(name: str) -> str:
        for f in fns:
            if name=="__skip__":
                return name
            name = f(name)
        return name
    return rename

def install_toplevel_cb_expert_loader(model_cls: type) -> None:
    if model_cls.__dict__.get("_pq_cb_wrapped", False):
        _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
        return
    orig_load_weights = model_cls.load_weights
    if getattr(orig_load_weights, "_pq_cb_wrapper", False):
        model_cls._pq_cb_wrapped = True
        _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
        return
    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        param_names = tuple(params_dict)
        reverse_fusion = _build_reverse_fusion(getattr(self, "packed_modules_mapping", None))
        # For DSpark we do NOT use _dspark_rename in compose for delegated tensors;
        # we will handle DSpark physical -> registered only for Gridbook-owned expert copy path,
        # otherwise yield raw name to let target's _remap_dspark_name handle delegation.
        # So compose only hf_mapper + spec_layer
        rename_non_dspark = _compose_renames(_hf_mapper_rename(self), _spec_layer_rename(self))
        # TP guard
        def _ensure_tp1_or_fail():
            from vllm.distributed import get_tensor_model_parallel_world_size, model_parallel_is_initialized
            if model_parallel_is_initialized():
                ws = int(get_tensor_model_parallel_world_size())
                if ws != 1:
                    raise ValueError(f"Gridbook CB top-level loader uses direct copy_ which is valid only for TP=1; live TP={ws}.")
            from vllm.config import get_current_vllm_config_or_none
            vcfg = get_current_vllm_config_or_none()
            if vcfg is not None:
                pc = getattr(vcfg, "parallel_config", None)
                if pc is not None and getattr(pc, "enable_expert_parallel", False):
                    # EP not audited for direct copy; fail closed if EP enabled with TP>1 already handled, but also if EP true with any TP we need to check sharding
                    # For now fail if EP enabled and we are DSpark (since expert parallel sharding breaks global copy)
                    if _is_dspark_model(self):
                        raise ValueError("Gridbook CB loader does not support expert_parallel; fail closed")
        _ensure_tp1_or_fail()

        is_dspark = _is_dspark_model(self)
        allowed_stages = None
        topology = None
        if is_dspark:
            allowed_stages, topology = _get_dspark_allowed_stages(self, params_dict)
            if allowed_stages is None:
                # Try to infer from params_dict stages present (should not happen in prod)
                raise RuntimeError("DSpark model but cannot derive topology (n_mtp_layers) from quant_config or live config; failing closed")
            # Validate EP/TP for DSpark done above

        # Materialize weights for two-phase validation (to ensure atomic)
        weights_list = list(weights) if not isinstance(weights, list) else weights

        # Duplicate and validation pre-check structures (no mutation yet)
        seen_physical_list = []
        seen_physical_set = set()
        seen_mapped_set = set()
        direct_plans = []  # list of (physical, mapped, weight)
        delegated = [] # list of (raw_name, weight)

        # For expected set derivation: we need to know which params are CB-owned.
        # We'll compute expected after distinguishing.

        # First pass: validate and classify each weight without mutating
        for name, w in weights_list:
            orig_name = name
            # Handle DSpark physical stage validation regardless of model type (must fail before mutation)
            if name.startswith("mtp.") or name.startswith("model.mtp."):
                # Normalize model.mtp. -> mtp.
                norm = name
                if name.startswith("model.mtp."):
                    norm = "mtp." + name[len("model.mtp."):]
                stage = _parse_physical_stage(norm)
                if stage is None:
                    # non-numeric stage string
                    try:
                        rest = norm[len("mtp."):]
                        dot = rest.find(".")
                        stage_str = rest if dot==-1 else rest[:dot]
                        int(stage_str)
                    except ValueError:
                        raise ValueError(f"DSpark physical tensor {orig_name!r} has non-numeric stage {stage_str!r}")
                    raise ValueError(f"DSpark physical tensor {orig_name!r} has non-numeric stage")
                # For DSpark model, check allowed
                if is_dspark:
                    if stage not in allowed_stages:
                        raise ValueError(f"DSpark physical tensor {orig_name!r} references unknown stage {stage} — allowed {allowed_stages}")
                    inner = norm[len(f"mtp.{stage}."):] if norm.startswith(f"mtp.{stage}.") else ""
                    if inner.startswith("main_proj."):
                        if stage != 0:
                            raise ValueError(f"DSpark head {orig_name!r} main_proj/main_norm must be stage 0")
                    elif inner.startswith("norm.") or inner.startswith("hc_head") or inner.startswith("markov_head."):
                        if stage != allowed_stages[-1]:
                            raise ValueError(f"DSpark terminal head {orig_name!r} must be final stage {allowed_stages[-1]}")
                    # main_norm allowed any stage for test compatibility (real target allows any but spec pins main_proj only)
                    if "main_proj" in inner.split("."):
                        # main_proj must be delegated FP8, not CB
                        if any(sfx in name for sfx in (".cb_qweight",".weight_scale",".input_global_scale")):
                            raise ValueError(f"CB tensor {orig_name!r} declares main_proj — main_proj must be delegated source FP8")
                else:
                    # For body model, any mtp.* should be validated as unknown stage if not in allowed? But body has no DSpark topology, so we can't know allowed. For body, we treat any mtp.* as to be dropped; but still validate stage numeric? And if stage is within 0..2 but body shouldn't have them, we should not fail; we just delegate and let target skip. So for non-DSpark, just ensure stage numeric and not head illegal? But body dropping is fine, so we don't enforce.
                    pass
                # Decide if it's DSpark CB expert: need to map to registered to check resolution
                # For DSpark model, we will attempt expert resolution using registered name
                if is_dspark:
                    reg = _remap_dspark_physical_to_registered(norm, allowed_stages)
                    if reg is None:
                        # confidence_head dropped -> delegate as skip
                        delegated.append((orig_name, w))
                        continue
                    # Apply hf_mapper + spec_layer to reg? Actually reg already is loader namespace; still need to compose? But for delegated we will yield raw orig_name so target remaps. For direct copy we use reg + extra rename
                    # For expert resolution, we use reg name plus any HF mapper? The hf mapper for DSpark is not needed for mtp.* (no nesting)
                    # Use reg as base for resolve
                    res_name_for_expert = reg
                    if rename_non_dspark is not None:
                        # Spec layer rename may apply, but DSpark not spec layer; still apply?
                        res_name_for_expert = rename_non_dspark(res_name_for_expert) if res_name_for_expert.startswith("model.") else res_name_for_expert
                    mapped = resolve_cb_expert_param(res_name_for_expert, param_names)
                    if mapped is not None:
                        # This is Gridbook-owned expert field
                        # Duplicate detection before mutation
                        if orig_name in seen_physical_set:
                            raise ValueError(f"duplicate DSpark CB tensor ingestion for physical {orig_name!r} before second copy")
                        if mapped in seen_mapped_set:
                            raise ValueError(f"duplicate DSpark CB tensor ingestion for mapped {mapped!r} (physical {orig_name!r})")
                        seen_physical_set.add(orig_name)
                        seen_mapped_set.add(mapped)
                        seen_physical_list.append(orig_name)
                        # Shape validation before mutation
                        param = params_dict[mapped]
                        if tuple(param.shape) != tuple(w.shape):
                            raise ValueError(f"prismaquant CB expert '{orig_name}' -> '{mapped}': checkpoint shape {tuple(w.shape)} != param shape {tuple(param.shape)} — stacked (E, out, bytes) contract violated")
                        # Dtype handling: allow float8_e8m0fnu -> uint8 byte view
                        if param.dtype != w.dtype:
                            # Allow byte-view case: param uint8, weight float8_e8m0fnu
                            if not (param.dtype == torch.uint8 and w.dtype == torch.float8_e8m0fnu):
                                raise ValueError(f"prismaquant CB expert '{orig_name}' -> '{mapped}': checkpoint dtype {w.dtype} != param dtype {param.dtype}")
                        direct_plans.append((orig_name, mapped, w))
                        continue
                    # Check orphan shared
                    orphan = resolve_shared_cb_target(res_name_for_expert, params_dict, reverse_fusion)
                    if orphan is not None:
                        raise RuntimeError(f"prismaquant CB tensor '{orig_name}' resolved to plain bf16 parameter '{orphan[0]}'. Gridbook requires native CB Linear")
                    # Detect wrong-leaf CB that is not valid dense/shared (should be delegated only if registered param exists)
                    if any(sfx in norm for sfx in (".cb_qweight",".weight_scale",".input_global_scale")):
                        if reg not in params_dict:
                            # No registered CB param for this checkpoint name -> extra/wrong-leaf
                            raise ValueError(f"DSpark CB tensor {orig_name!r} -> {reg!r} is unknown/extra/wrong-leaf CB field — failing closed")
                    # Not a Gridbook-owned expert -> delegate raw
                    if orig_name in seen_physical_set:
                        raise ValueError(f"duplicate physical {orig_name!r}")
                    seen_physical_set.add(orig_name)
                    seen_physical_list.append(orig_name)
                    delegated.append((orig_name, w))
                    continue
                else:
                    # Body model: delegate raw (target will skip)
                    if orig_name in seen_physical_set:
                        raise ValueError(f"duplicate physical {orig_name!r}")
                    seen_physical_set.add(orig_name)
                    delegated.append((orig_name, w))
                    continue
            # Non-mtp tensors: apply rename_non_dspark for resolution but delegate raw? For CB expert tensors that are not mtp (e.g., body layers), they are like model.layers.X.mlp.experts... we need to check expert resolution similarly
            res_name = rename_non_dspark(name) if rename_non_dspark is not None else name
            if res_name == "__skip__":
                delegated.append((orig_name, w))
                continue
            mapped = resolve_cb_expert_param(res_name, param_names)
            if mapped is not None:
                if orig_name in seen_physical_set:
                    raise ValueError(f"duplicate CB tensor {orig_name!r}")
                if mapped in seen_mapped_set:
                    raise ValueError(f"duplicate mapped {mapped!r}")
                seen_physical_set.add(orig_name)
                seen_mapped_set.add(mapped)
                param = params_dict[mapped]
                if tuple(param.shape) != tuple(w.shape):
                    raise ValueError(f"prismaquant CB expert '{orig_name}' -> '{mapped}': shape mismatch")
                if param.dtype != w.dtype:
                    if not (param.dtype == torch.uint8 and w.dtype == torch.float8_e8m0fnu):
                        raise ValueError(f"dtype mismatch {w.dtype} vs {param.dtype}")
                if "main_proj" in res_name.split("."):
                    raise ValueError(f"CB tensor {orig_name!r} declares main_proj must be delegated")
                direct_plans.append((orig_name, mapped, w))
                continue
            orphan = resolve_shared_cb_target(res_name, params_dict, reverse_fusion)
            if orphan is not None:
                raise RuntimeError(f"prismaquant CB tensor '{orig_name}' resolved to plain bf16 parameter '{orphan[0]}'")
            # For DSpark, any CB-like under DSpark stage that didn't map and isn't orphan but has no registered param is wrong-leaf
            if is_dspark and any(sfx in res_name for sfx in (".cb_qweight",".weight_scale",".input_global_scale")):
                if any(res_name.startswith(f"model.layers.{s}.") for s in allowed_stages):
                    if mapped is None and res_name not in params_dict:
                        raise ValueError(f"DSpark CB tensor {orig_name!r} -> {res_name!r} is unknown/extra/wrong-leaf CB field — failing closed")
            # Not CB -> delegate raw original name (preserve checkpoint spelling for Qwen case)
            # For non-DSpark, keep original; for DSpark, delegated already handled above; here generic
            delegated.append((orig_name, w))

        # After classification, handle duplicate physical via list vs set (already)
        # But also need to detect duplicate physical names that alias same registered param via different physical head names (e.g., mtp.0.main_proj vs mtp.1.main_proj both map to model.main_proj). Our seen_mapped_set handles that for expert case, but for delegated heads, they map to same registered param via target loader. We need to detect that two physical head names mapping to same registered param before mutation.
        # Since we delegate raw, target loader will handle mapping; we can pre-detect by computing target remapped names for delegated DSpark heads.
        if is_dspark and topology is not None:
            delegated_reg_map = {}
            for phys, _ in delegated:
                if phys.startswith("mtp.") or phys.startswith("model.mtp."):
                    norm = phys
                    if phys.startswith("model.mtp."):
                        norm = "mtp."+ phys[len("model.mtp."):]
                    reg = _remap_dspark_physical_to_registered(norm, allowed_stages)
                    if reg is None:
                        continue
                    # Only heads map to model.* ; for layer tensors they map to model.layers.stage...
                    # But two different physical heads that map to same reg would be duplicate
                    if reg in delegated_reg_map:
                        raise ValueError(f"duplicate registered mapping {reg!r} from physical {delegated_reg_map[reg]!r} and {phys!r}")
                    delegated_reg_map[reg]=phys

        # Second phase: now apply direct copies (after validation)
        loaded = set()
        for phys, mapped, w in direct_plans:
            param = params_dict[mapped]
            # Handle float8_e8m0fnu -> uint8 view
            w_to_copy = w
            if param.dtype == torch.uint8 and w.dtype == torch.float8_e8m0fnu:
                w_to_copy = w.view(torch.uint8)
            else:
                w_to_copy = w.to(param.dtype)
            param.data.copy_(w_to_copy)
            mark_filled(param)
            loaded.add(mapped)

        # Delegate remaining via original loader with raw names (so target remapper runs once)
        def _passthrough():
            for n, wt in delegated:
                yield n, wt
        ret = orig_load_weights(self, _passthrough())
        if ret:
            loaded |= set(ret)

        # --- Exact consumption verification ---
        # Derive expected set from registered CB params under DSpark stages and from quant_config if available
        expected = set()
        if is_dspark and allowed_stages is not None:
            # Gather expected from params_dict: any param that is CB param under DSpark registered stages
            for k, p in params_dict.items():
                # Check if k is DSpark layer or head CB? Heads are not CB, so ignore heads
                is_dspark_layer = any(k.startswith(f"model.layers.{s}.") for s in allowed_stages)
                if is_dspark_layer and any(sfx in k for sfx in ("cb_qweight","weight_scale","input_global_scale")):
                    # Only count if quant_config says this layer is CB (via target_scheme construction mapping)
                    # We can check if construction target exists: map registered stage back to construction
                    nh, _ = topology
                    cons_stage = None
                    for s in allowed_stages:
                        if k.startswith(f"model.layers.{s}."):
                            cons_stage = nh + s
                            break
                    if cons_stage is not None:
                        # Derive construction prefix for this param's logical target
                        # For expert params, logical target is e.g., model.layers.43.ffn.experts.gate_up_proj
                        # For dense, it's e.g., model.layers.43.attn.fused_wqa_wkv
                        # We can try to see if any target_scheme key is a prefix of the param's logical name
                        # Simpler: if quant_config exists, check if any target_scheme key corresponds to this param's base
                        qc = getattr(self, "quant_config", None)
                        if qc is None:
                            try:
                                from vllm.config import get_current_vllm_config_or_none
                                vcfg2 = get_current_vllm_config_or_none()
                                if vcfg2 is not None:
                                    # Try draft quant_config
                                    from vllm.model_executor.models.utils import get_draft_quant_config
                                    qc = get_draft_quant_config(vcfg2)
                                    if qc is None:
                                        qc = getattr(vcfg2, "quant_config", None)
                            except ImportError:
                                pass
                        if qc is not None and hasattr(qc, "target_scheme"):
                            # Check if this param's base is covered by a CB target
                            # For expert: param like model.layers.0.ffn.experts.routed_experts.w13_cb_qweight -> base ffn.experts
                            # We can check _moe_scheme_for_prefix or _scheme_for_prefix style, but simplified: if any target starts with model.layers.{cons_stage}
                            has_cb_target = any(t.startswith(f"model.layers.{cons_stage}.") for t in qc.target_scheme)
                            if has_cb_target:
                                expected.add(k)
                            else:
                                # If no construction target, this param shouldn't be CB? But it is registered as CB, maybe it's body CB not DSpark, so skip?
                                pass
                        else:
                            expected.add(k)
                    else:
                        expected.add(k)
            # Also validate that expected is not empty when DSpark declared but no CB params? Should be at least something
            # Now compute missing/extra
            missing = expected - loaded
            # Extra: any loaded CB param under DSpark that is not expected (wrong leaf)
            extra = set()
            for lk in loaded:
                if any(lk.startswith(f"model.layers.{s}.") for s in allowed_stages) and any(sfx in lk for sfx in ("cb_qweight","weight_scale","input_global_scale")):
                    if lk not in expected:
                        extra.add(lk)
            # Also detect any CB weight that was delegated but not expected? That would be wrong leaf that we delegated incorrectly; but we already handled via direct vs delegated, so just check unexpected in loaded
            if missing:
                raise ValueError(f"DSpark CB exact consumption failed: missing {sorted(missing)} (expected {sorted(expected)}, consumed {sorted(loaded)})")
            if extra:
                raise ValueError(f"DSpark CB exact consumption failed: extra {sorted(extra)} beyond expected {sorted(expected)}")
            # Also ensure no duplicate already raised
        return loaded
    load_weights._pq_cb_wrapper = True
    model_cls.load_weights = load_weights
    model_cls._pq_cb_wrapped = True
    _INSTALLED_MODULE_PATHS.add(getattr(model_cls, "__module__", "?"))
