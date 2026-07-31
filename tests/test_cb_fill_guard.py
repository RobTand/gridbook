"""Serve-time CB-fill assertion (cb_fill_guard.py) — R10 / debt D3.

The regression this pins is the #1 CB-lane support trap: an MoE arch whose vLLM
loader maps experts at the TOP LEVEL, with no entry in
``plugin._CB_TOPLEVEL_MODULE_PATHS``. Nothing crashes — the stacked CB expert
params keep their ``torch.empty`` contents and the FusedMoE serves uninitialised
memory (commit ``9a79963``: Laguna, 93% of params). The guard turns that into a
hard failure at ``process_weights_after_loading``.

Two arms, same synthetic top-level model class:
  * loader deliberately NOT installed -> stacks unfilled -> the guard raises;
  * ``install_toplevel_cb_expert_loader`` installed -> stacks filled -> passes.

torch-only (no vLLM): both ``cb_fill_guard`` and ``moe_toplevel_loader`` import
nothing from vLLM, so this runs in any venv with torch.

  PYTHONPATH=/home/rob/prismaquant/plugins/gridbook CUDA_VISIBLE_DEVICES= \\
    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \\
    -m pytest plugins/gridbook/tests/test_cb_fill_guard.py -q
"""
import pytest
import torch

from gridbook.cb_fill_guard import (
    CB_FILLED_ATTR,
    assert_cb_experts_filled,
    mark_filled,
    mark_unfilled,
    unfilled_cb_params,
)
from gridbook.moe_toplevel_loader import (
    install_toplevel_cb_expert_loader,
    installed_module_paths,
)

E, HID, INTER, BYTES = 4, 16, 8, 3


class _CBLayer(torch.nn.Module):
    """Stand-in for the FusedMoE layer as ``moe.create_weights`` leaves it."""

    def __init__(self, prefix="model.layers.1.mlp.experts"):
        super().__init__()
        self.prefix = prefix
        w13 = torch.nn.Parameter(
            torch.empty(E, 2 * INTER, BYTES, dtype=torch.uint8),
            requires_grad=False)
        w2 = torch.nn.Parameter(
            torch.empty(E, HID, BYTES, dtype=torch.uint8), requires_grad=False)
        # exactly what create_weights does, in order
        mark_unfilled(w13)
        mark_unfilled(w2)
        self.register_parameter("w13_cb_qweight", w13)
        self.register_parameter("w2_cb_qweight", w2)


def _make_model_cls():
    """A fresh synthetic top-level-expert-mapping model class per test.

    Fresh because ``install_toplevel_cb_expert_loader`` stamps a class-level
    sentinel; a shared class would leak the install across the two arms.
    """

    class _FakeTopLevelCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # nested exactly like a real arch so named_parameters() carries the
            # ``.experts.`` anchor the resolver keys on.
            mlp = torch.nn.Module()
            mlp.experts = _CBLayer()
            layer = torch.nn.Module()
            layer.mlp = mlp
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList([layer])
            self.model = model
            self.other = torch.nn.Parameter(torch.zeros(4))
            self.delegated = []

        @property
        def experts(self):
            return self.model.layers[0].mlp.experts

        def load_weights(self, weights):
            # The stock top-level loader for this arch class: it maps experts
            # itself and never reaches our stacked CB names, so it silently
            # drops them (its own mapping loop `continue`s past mlp.experts).
            loaded = set()
            params = dict(self.named_parameters())
            for name, w in weights:
                self.delegated.append(name)
                if name in params:
                    params[name].data.copy_(w)
                    loaded.add(name)
            return loaded

    return _FakeTopLevelCausalLM


def _checkpoint():
    P = "model.layers.0.mlp.experts."
    return [
        (P + "gate_up_proj.cb_qweight",
         torch.full((E, 2 * INTER, BYTES), 7, dtype=torch.uint8)),
        (P + "down_proj.cb_qweight",
         torch.full((E, HID, BYTES), 9, dtype=torch.uint8)),
        ("other", torch.ones(4)),
    ]


def test_unwired_arch_raises_naming_the_module_path():
    """Loader NOT installed: the stacks are never filled -> hard failure."""
    cls = _make_model_cls()
    m = cls()
    m.load_weights(iter(_checkpoint()))
    # the stock loader saw (and dropped) the CB expert tensors: silent today
    assert any("cb_qweight" in n for n in m.delegated)
    assert unfilled_cb_params(m.experts) == ["w13_cb_qweight", "w2_cb_qweight"]

    with pytest.raises(RuntimeError) as exc:
        assert_cb_experts_filled(m.experts, m.experts.prefix)
    msg = str(exc.value)
    assert "w13_cb_qweight" in msg and "w2_cb_qweight" in msg
    assert m.experts.prefix in msg
    # the message must point at the registry line to add
    assert "_CB_TOPLEVEL_MODULE_PATHS" in msg
    assert "gridbook/plugin.py" in msg


def test_wired_arch_passes_and_records_the_module_path():
    """Loader installed (what plugin.py's module-path tuple does): fills, passes."""
    cls = _make_model_cls()
    install_toplevel_cb_expert_loader(cls)
    assert cls.__module__ in installed_module_paths()

    m = cls()
    m.load_weights(iter(_checkpoint()))
    assert torch.all(m.experts.w13_cb_qweight == 7)
    assert torch.all(m.experts.w2_cb_qweight == 9)
    assert unfilled_cb_params(m.experts) == []
    assert_cb_experts_filled(m.experts, m.experts.prefix)   # no raise


def test_per_layer_fill_path_also_satisfies_the_guard():
    """Fill path 1 of 2: the instance-level hook in moe.py marks the same
    sentinel (asserted here at the guard's contract level, since importing
    gridbook.moe needs vLLM)."""
    layer = _CBLayer()
    layer.w13_cb_qweight.data.copy_(torch.full_like(layer.w13_cb_qweight, 7))
    mark_filled(layer.w13_cb_qweight)
    assert unfilled_cb_params(layer) == ["w2_cb_qweight"]
    mark_filled(layer.w2_cb_qweight)
    assert_cb_experts_filled(layer, layer.prefix)


def test_partial_fill_still_raises():
    """One stack filled, one not (a half-matching name map) must NOT pass."""
    layer = _CBLayer()
    mark_filled(layer.w13_cb_qweight)
    with pytest.raises(RuntimeError, match="w2_cb_qweight"):
        assert_cb_experts_filled(layer, layer.prefix)


def test_scoped_to_params_the_local_rank_registered():
    """EP/PP caveat: an absent stack, or an empty (zero-expert) one, proves
    nothing and must be skipped rather than reported unfilled."""

    class _PartialRank(torch.nn.Module):        # only w13 registered
        def __init__(self):
            super().__init__()
            w13 = torch.nn.Parameter(
                torch.empty(E, 2 * INTER, BYTES, dtype=torch.uint8),
                requires_grad=False)
            mark_unfilled(w13)
            self.register_parameter("w13_cb_qweight", w13)

    layer = _PartialRank()
    assert unfilled_cb_params(layer) == ["w13_cb_qweight"]
    mark_filled(layer.w13_cb_qweight)
    assert_cb_experts_filled(layer, "rank-local")

    class _ZeroExpertRank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, shape in (("w13_cb_qweight", (0, 2 * INTER, BYTES)),
                                ("w2_cb_qweight", (0, HID, BYTES))):
                p = torch.nn.Parameter(torch.empty(*shape, dtype=torch.uint8),
                                       requires_grad=False)
                mark_unfilled(p)
                self.register_parameter(name, p)

    assert unfilled_cb_params(_ZeroExpertRank()) == []


def test_registry_is_data_not_a_try_except_chain():
    """R10 (i): the per-arch opt-in is a module-path tuple.

    Read with ``ast`` rather than imported — ``gridbook.plugin`` needs vLLM,
    which this venv does not have, and the point of the check is the literal.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "gridbook" / "plugin.py"
    tree = ast.parse(src.read_text())
    paths = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", "") == "_CB_TOPLEVEL_MODULE_PATHS"):
            paths = ast.literal_eval(node.value)
    assert isinstance(paths, tuple) and all(isinstance(p, str) for p in paths)
    for arch in ("hy_v3", "hy_v3_mtp", "laguna", "qwen3_5", "qwen3_5_mtp"):
        assert f"vllm.model_executor.models.{arch}" in paths, arch
    # no arch-specific class imports left in the install path
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any((m or "").startswith("vllm.model_executor.models")
                   for m in imported)


def test_untracked_param_counts_as_unfilled():
    """A registered, non-empty stack carrying no sentinel did not come from our
    create_weights — treat it as unfilled rather than as a bypass."""
    layer = _CBLayer()
    for name in ("w13_cb_qweight", "w2_cb_qweight"):
        delattr(getattr(layer, name), CB_FILLED_ATTR)
    assert unfilled_cb_params(layer) == ["w13_cb_qweight", "w2_cb_qweight"]
