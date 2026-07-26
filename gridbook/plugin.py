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


def _install_toplevel_cb_expert_loaders() -> None:
    """Install the stacked-CB top-level expert-loader wrap on every MoE arch
    whose vLLM loader maps experts at the top level (never delegating to the
    per-layer FusedMoE loader). Each arch is a guarded one-liner; absence of the
    class in this vLLM build is a no-op."""
    # HunYuan V3 (Hy3) — 80-layer MoE, experts loaded top-level.
    try:
        from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM
    except ImportError:
        pass
    else:
        install_toplevel_cb_expert_loader(HYV3ForCausalLM)

    # HunYuan V3 MTP drafter (spec decode) — the same top-level stacked-CB
    # experts, nested one level under ``.mtp_block.`` (its ``_rewrite_spec_
    # layer_name`` renames ``model.layers.{N}.X`` -> ``…mtp_block.X``). The
    # wrap's spec-layer rename (moe_toplevel_loader) applies that offset before
    # resolving, so a CB-quantized MTP module loads exactly like the body.
    try:
        from vllm.model_executor.models.hy_v3_mtp import HYV3MTP
    except ImportError:
        pass
    else:
        install_toplevel_cb_expert_loader(HYV3MTP)

    # poolside Laguna (S/XS 2.x) — 48-layer 256-expert MoE; vLLM's class maps
    # experts per-expert (`experts.{eid}.gate_proj`) at the top level, so the
    # stacked-CB tensors (`experts.gate_up_proj.cb_qweight`) need the same
    # wrap as Hy3. The DFlash drafter is a separate vLLM class; add it here
    # when a CB-quantized drafter ships.
    try:
        from vllm.model_executor.models.laguna import LagunaForCausalLM
    except ImportError:
        pass
    else:
        install_toplevel_cb_expert_loader(LagunaForCausalLM)

    # Qwen3.5-MoE (35B-A3B and the multimodal ForConditionalGeneration wrapper).
    # Its ``Qwen3_5Model.load_weights`` matches our stacked
    # ``experts.gate_up_proj.cb_qweight`` with the arch's *fused* expert mapping
    # (``experts.gate_up_proj`` is a substring) and derives a DOTTED param name
    # (``…experts.w2_weight.cb_qweight``) — in newer vLLM this lands inside
    # ``RoutedExperts.load_weights`` as ``getattr(self, "w2_weight.cb_qweight")``
    # and AttributeErrors. It never reaches the instance-level CB hook that
    # ``moe.py`` installs on the FusedMoE, so the top-level wrap is the fix.
    # The class names differ across vLLM versions (ForCausalLM / MoeForCausalLM /
    # (Moe)ForConditionalGeneration), so discover them from the module rather
    # than pinning names.
    _install_on_module_classes("vllm.model_executor.models.qwen3_5")
    _install_on_module_classes("vllm.model_executor.models.qwen3_5_mtp")

    # DSv4-class and any future top-level-expert-mapping MoE arch: add a guarded
    # import + install_toplevel_cb_expert_loader(<cls>) line here.


def register() -> None:
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
