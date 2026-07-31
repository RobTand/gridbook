"""Serve-time proof that every registered CB expert stack was actually FILLED.

The per-arch top-level loader install (``plugin.py``) is opt-in per vLLM module
path. Its failure mode when a line is missing is not a crash but **coherent
garbage generation**: the arch's own ``load_weights`` never matches our stacked
``…experts.<proj>.cb_qweight`` tensors, the registered ``w13/w2_cb_qweight``
buffers keep their ``torch.empty`` contents, and the FusedMoE happily serves
uninitialised memory (commit ``9a79963``: Laguna, 93% of params).

The detection therefore **cannot live in the loader wrapper** — the wrapper is
exactly what was never installed. It anchors instead on
``process_weights_after_loading``, which vLLM calls for every CB MoE layer on
every load path:

  * ``moe.PrismaQuantCBMoEMethod.create_weights`` marks each stacked CB param
    ``_pq_cb_filled = False``;
  * BOTH fill paths mark it ``True`` — the instance-level hook in ``moe.py`` and
    ``moe_toplevel_loader.load_weights`` at its ``loaded.add`` site;
  * ``assert_cb_experts_filled`` raises otherwise, naming the model class and the
    vLLM module path that has to be added to ``plugin._CB_TOPLEVEL_MODULE_PATHS``.

Zero extra memory, arch-independent, fires once per layer, and there is no env
bypass by design. Scoped to the params the local rank actually registered (an
EP/PP-absent or zero-expert shard proves nothing and is skipped). A dummy-weight
load (``--load-format dummy``) is not a supported CB path and will trip this.

Import-light on purpose (no torch, no vLLM at module scope) so the guard is
unit-testable in any venv.
"""
from __future__ import annotations

#: Sentinel attribute stamped on every stacked CB expert parameter.
CB_FILLED_ATTR = "_pq_cb_filled"

#: The stacked CB expert params a CB MoE layer registers (moe.py create_weights).
CB_EXPERT_PARAM_NAMES = ("w13_cb_qweight", "w2_cb_qweight")


def mark_unfilled(param) -> None:
    """Stamp a freshly created CB param as not-yet-filled (``create_weights``)."""
    setattr(param, CB_FILLED_ATTR, False)


def mark_filled(param) -> None:
    """Record that a fill path copied checkpoint bytes into ``param``.

    Called from both fill paths. Unconditional: marking a param the guard does
    not inspect (a weight_scale) is harmless, and a param object that was never
    stamped by ``create_weights`` still ends up truthfully marked.
    """
    try:
        setattr(param, CB_FILLED_ATTR, True)
    except AttributeError:  # pragma: no cover — exotic param types
        pass


def unfilled_cb_params(layer) -> list[str]:
    """Names of the stacked CB params this rank registered that were never filled.

    A param that is absent (EP/PP shard without this stack) or empty (zero
    experts on this rank) proves nothing and is skipped. A registered, non-empty
    param carrying no sentinel at all counts as UNFILLED — ``create_weights``
    always stamps one, so its absence means the buffer did not come from us.
    """
    missing: list[str] = []
    for name in CB_EXPERT_PARAM_NAMES:
        param = getattr(layer, name, None)
        if param is None:
            continue
        numel = getattr(param, "numel", None)
        if callable(numel) and numel() == 0:
            continue
        if not getattr(param, CB_FILLED_ATTR, False):
            missing.append(name)
    return missing


def _model_class_hint() -> str:
    """Best-effort ``ClassName (vllm.module.path)`` for the model being loaded.

    Only ever called on the failure path, so the registry probe's cost and its
    version fragility are both irrelevant; every step degrades to a placeholder.
    """
    try:
        from vllm.config import get_current_vllm_config

        archs = list(get_current_vllm_config().model_config.hf_config.architectures)
    except Exception:  # noqa: BLE001 — diagnostics only
        return "<unknown model class>"
    described = []
    for arch in archs:
        module = "<vllm module path unknown>"
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            cls, _ = ModelRegistry.resolve_model_cls(arch)
            module = getattr(cls, "__module__", module)
        except Exception:  # noqa: BLE001 — diagnostics only
            pass
        described.append(f"{arch} ({module})")
    return ", ".join(described) or "<unknown model class>"


def _installed_paths_hint() -> str:
    try:
        from .moe_toplevel_loader import installed_module_paths

        paths = sorted(installed_module_paths())
    except Exception:  # noqa: BLE001 — diagnostics only
        return "<unavailable>"
    return ", ".join(paths) if paths else "<none>"


def assert_cb_experts_filled(layer, prefix: str) -> None:
    """Raise unless every stacked CB expert param this rank registered was filled."""
    missing = unfilled_cb_params(layer)
    if not missing:
        return
    raise RuntimeError(
        f"{prefix}: CB expert stack(s) {missing} were REGISTERED but never "
        f"FILLED from the checkpoint — the model class would serve uninitialised "
        f"memory (coherent garbage, not a crash).\n"
        f"  model class: {_model_class_hint()}\n"
        f"  gridbook top-level CB loaders installed on: {_installed_paths_hint()}\n"
        f"If the model class's vLLM module path is not in that list, this arch "
        f"maps MoE experts at the TOP LEVEL and needs its module path added to "
        f"_CB_TOPLEVEL_MODULE_PATHS in gridbook/plugin.py. If it IS in the list, "
        f"the wrap installed but matched no stacked "
        f"'…experts.<proj>.cb_qweight' tensor — check the checkpoint's expert "
        f"tensor names against moe_toplevel_loader.resolve_cb_expert_param."
    )
