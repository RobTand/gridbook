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

from . import ops  # noqa: F401  (registers the prismaquant::cb_gemm custom op)
from .config import PrismaQuantConfig
from .moe_toplevel_loader import install_toplevel_cb_expert_loader


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
# An in-code tuple is deliberately the whole registry: gridbook ships into the
# vLLM container and cannot import ``prismaquant.model_profiles``, so it needs
# its own list, and a tuple is the smallest thing that is data. If third parties
# ever need to extend it without patching the package, promote it to a JSON
# sidecar read at ``register()`` time — the consumer below does not care.
#
# A MISSING entry does not crash: it serves uninitialised expert memory. That is
# caught at serve time by ``cb_fill_guard.assert_cb_experts_filled``
# (``moe.py:process_weights_after_loading``), which names the module path to add.
_CB_TOPLEVEL_MODULE_PATHS: tuple[str, ...] = (
    # HunYuan V3 (Hy3) — 80-layer MoE, experts loaded top-level.
    "vllm.model_executor.models.hy_v3",
    # Hy3 MTP drafter (spec decode) — the same top-level stacked-CB experts,
    # nested one level under ``.mtp_block.`` (its ``_rewrite_spec_layer_name``
    # renames ``model.layers.{N}.X`` -> ``…mtp_block.X``). The wrap's spec-layer
    # rename (moe_toplevel_loader) applies that offset before resolving, so a
    # CB-quantized MTP module loads exactly like the body.
    "vllm.model_executor.models.hy_v3_mtp",
    # poolside Laguna (S/XS 2.x) — 48-layer 256-expert MoE; vLLM's class maps
    # experts per-expert (``experts.{eid}.gate_proj``) at the top level, so the
    # stacked-CB tensors need the same wrap as Hy3. Its DFlash drafter is a
    # separate vLLM module; add that path when a CB-quantized drafter ships.
    "vllm.model_executor.models.laguna",
    # Qwen3.5-MoE (35B-A3B and the multimodal wrapper). Its ``Qwen3_5Model.
    # load_weights`` matches our stacked ``experts.gate_up_proj.cb_qweight`` with
    # the arch's *fused* expert mapping (``experts.gate_up_proj`` is a substring)
    # and derives a DOTTED param name (``…experts.w2_weight.cb_qweight``) — in
    # newer vLLM that lands inside ``RoutedExperts.load_weights`` as
    # ``getattr(self, "w2_weight.cb_qweight")`` and AttributeErrors. It never
    # reaches the instance-level CB hook, so the top-level wrap is the fix.
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.models.qwen3_5_mtp",
    # DSv4-class: an entry the moment its vLLM module path exists in the target
    # build — check ``vllm.model_executor.models.deepseek_v4`` (the class is
    # top-level-expert-mapping like Hy3) and uncomment:
    # "vllm.model_executor.models.deepseek_v4",
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
        try:
            from .cuda_ext import get_fused_ext
            get_fused_ext()
        except Exception:  # noqa: BLE001
            pass
    try:
        register_quantization_config("gridbook")(PrismaQuantConfig)
    except ValueError:
        pass
    try:
        # Legacy key: artifacts exported before the gridbook rename carry
        # quant_method="prismaquant"; register the same config under it.
        register_quantization_config("prismaquant")(PrismaQuantConfig)
    except ValueError:
        # Already registered (idempotent across repeated plugin loads).
        pass
    _install_toplevel_cb_expert_loaders()
