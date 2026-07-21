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
