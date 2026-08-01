"""vLLM registration hook.

We register the quantization config; the shared codebooks are read from the
`codebook_file` sidecar by the config itself (via `get_current_vllm_config()`),
and the packed custom op is registered on import of `.ops`.

**Model-loader wrap (one class of arch).** The GGUF plugin needed no model-loader
monkeypatches, and neither do we for models that route expert weights through the
per-layer `FusedMoE.load_weights` (Qwen3.5-MoE / 35B-A3B: our instance-level CB
hook in `moe.py` handles them). But some MoE archs — HunYuan V3 (`HYV3ForCausalLM`),
and DSv4-class — load experts at the **top-level** model via an
`expert_params_mapping` keyed on per-expert checkpoint names, and explicitly
`continue` past `mlp.experts` in their stacked-params loop; they NEVER call the
per-layer FusedMoE loader. Our exporter writes *stacked* CB expert tensors
(`…experts.gate_up_proj.cb_qweight`, all experts in one tensor), which match
neither that arch's stacked- nor its per-expert expert-mapping, so its
`load_weights` KeyErrors. For exactly this class we install a thin top-level
`load_weights` wrapper (`moe_toplevel_loader`) that copies our stacked-CB expert
tensors into the registered fused params and delegates everything else unchanged.
No vLLM-core files are patched — only the model class's own `load_weights`, and
only for archs we opt in below.
"""
from __future__ import annotations

import os

from vllm.model_executor.layers.quantization import register_quantization_config

from . import ops  # noqa: F401  (registers Gridbook's native custom ops)
from .config import PrismaQuantConfig
from .moe_toplevel_loader import install_toplevel_cb_expert_loader
from .runtime_contract import load_runtime_contract


_TOPLEVEL_CLASS_SUFFIXES = ("ForCausalLM", "ForCausalLMBase",
                            "ForConditionalGeneration", "MTP")


def _install_on_module_classes(module_path: str) -> None:
    """Install the stacked-CB wrap on every top-level model class a vLLM arch
    module DEFINES (``__module__`` guard: never on classes it merely imports)
    whose name looks like an entrypoint and which exposes ``load_weights``.

    Version-robust by construction: a missing module, or renamed classes within
    it, degrade to a no-op instead of an ImportError. The wrap itself is inert
    for non-CB checkpoints (it only fires on ``…experts.<proj>.cb_qweight``
    names), so over-installing on a sibling class is harmless."""
    import importlib
    import inspect

    try:
        mod = importlib.import_module(module_path)
    except Exception:  # noqa: BLE001 — arch absent from this vLLM build
        return
    for name, obj in vars(mod).items():
        if not inspect.isclass(obj):
            continue
        if getattr(obj, "__module__", None) != module_path:
            continue
        if not name.endswith(_TOPLEVEL_CLASS_SUFFIXES):
            continue
        if not callable(getattr(obj, "load_weights", None)):
            continue
        install_toplevel_cb_expert_loader(obj)


# ---------------------------------------------------------------------------
# The per-arch CB opt-in, as DATA (architecture re-vet R10 / debt D3).
#
# One entry per vLLM MODULE PATH whose model classes load MoE experts at the
# **top level** — mapping per-expert or fused expert names in the model's own
# ``load_weights`` and never delegating to the per-layer ``FusedMoE.load_weights``
# (archs that DO delegate need no entry: the instance-level hook in ``moe.py``
# already covers them, see this module's docstring).
#
# Module paths, not class imports: ``_install_on_module_classes`` discovers the
# entrypoint classes a module DEFINES, so it is robust to the class renames that
# happen across vLLM versions (Qwen3.5 alone ships ForCausalLM / MoeForCausalLM /
# (Moe)ForConditionalGeneration variants), a missing module degrades to a no-op
# instead of an ImportError, and over-installing on a sibling class is harmless
# (the wrap is inert for non-CB checkpoints — see ``:44-46``).
#
# A MISSING entry does not crash: it serves uninitialised expert memory. That is
# caught at serve time by ``cb_fill_guard.assert_cb_experts_filled``
# (``moe.py:process_weights_after_loading``), which names the module path to add.
_RUNTIME_CONTRACT = load_runtime_contract()
_CB_TOPLEVEL_MODULE_PATHS: tuple[str, ...] = tuple(
    _RUNTIME_CONTRACT["producer_profiles"]["top_level_loader_modules"]
)
_QUANTIZATION_METHODS: tuple[str, ...] = tuple(
    _RUNTIME_CONTRACT["quant_method"]["accepted"]
)


def _install_toplevel_cb_expert_loaders() -> None:
    """Install the stacked-CB top-level expert-loader wrap on every registered
    module path. Absence of a module (or of matching classes within it) in this
    vLLM build is a no-op."""
    for module_path in _CB_TOPLEVEL_MODULE_PATHS:
        _install_on_module_classes(module_path)


def register() -> None:
    # Residency-matched A/B support: force-build+load the fused ext even when
    # its dispatch env is off, so both arms of a served logprob comparison
    # carry identical CUDA-extension residency (the session-arithmetic-drift
    # mechanism otherwise confounds the gate).
    if os.environ.get("PRISMAQUANT_PRELOAD_FUSED") == "1":
        from .cuda_ext import preload_fused_extensions
        preload_fused_extensions()
    for quant_method in _QUANTIZATION_METHODS:
        try:
            register_quantization_config(quant_method)(PrismaQuantConfig)
        except ValueError:
            # Already registered (idempotent across repeated plugin loads).
            pass
    _install_toplevel_cb_expert_loaders()
