"""Lifecycle guarantees for the custom-op layer indirection registry."""
from __future__ import annotations

import gc
import weakref

import pytest
import torch

from gridbook import ops


class _Layer:
    _cb_N = 5


class _Method:
    def _apply_inline(self, layer, x, *args):
        del args
        return torch.zeros((*x.shape[:-1], layer._cb_N), dtype=x.dtype)


class _MoEMethod:
    def _apply_inline(self, layer, x, topk_weights, topk_ids):
        del layer, topk_weights, topk_ids
        return torch.full_like(x, 2)


def test_registry_does_not_retain_unloaded_model():
    method = _Method()
    layer = _Layer()
    method_ref = weakref.ref(method)
    layer_ref = weakref.ref(layer)
    layer_id = ops.register_cb_layer(method, layer)

    del method, layer
    gc.collect()

    assert method_ref() is None
    assert layer_ref() is None
    assert layer_id not in ops._LAYER_REGISTRY
    with pytest.raises(RuntimeError, match="stale or unknown"):
        ops._lookup_cb_layer(layer_id)


def test_layer_ownership_keeps_method_live_until_model_unload():
    method = _Method()
    layer = _Layer()
    # This mirrors vLLM's LinearBase.quant_method ownership.
    layer.quant_method = method
    method_ref = weakref.ref(method)
    layer_ref = weakref.ref(layer)
    layer_id = ops.register_cb_layer(method, layer)

    del method
    gc.collect()
    assert method_ref() is not None
    assert layer_id in ops._LAYER_REGISTRY

    del layer
    gc.collect()
    assert method_ref() is None
    assert layer_ref() is None
    assert layer_id not in ops._LAYER_REGISTRY


def test_expired_id_is_never_reused_for_new_layer():
    first_method = _Method()
    first_layer = _Layer()
    first_id = ops.register_cb_layer(first_method, first_layer)
    del first_method, first_layer
    gc.collect()

    second_method = _Method()
    second_layer = _Layer()
    second_id = ops.register_cb_layer(second_method, second_layer)

    assert second_id > first_id
    with pytest.raises(RuntimeError, match="stale or unknown"):
        ops._lookup_cb_layer(first_id)
    assert ops._lookup_cb_layer(second_id) == (second_method, second_layer)


def test_live_linear_dispatch_keeps_output_contract():
    method = _Method()
    layer = _Layer()
    layer_id = ops.register_cb_layer(method, layer)

    out = ops.cb_linear_forward(torch.ones((2, 3)), layer_id)

    assert out.shape == (2, layer._cb_N)
    assert torch.equal(out, torch.zeros_like(out))


def test_live_moe_dispatch_keeps_output_contract():
    method = _MoEMethod()
    layer = _Layer()
    layer_id = ops.register_cb_layer(method, layer)
    x = torch.ones((3, 7))

    out = ops.cb_moe_forward(
        x,
        torch.ones((3, 2)),
        torch.zeros((3, 2), dtype=torch.int64),
        layer_id,
    )

    assert out.shape == x.shape
    assert torch.equal(out, torch.full_like(x, 2))


def test_linear_custom_op_fake_and_schema_contract():
    method = _Method()
    layer = _Layer()
    layer_id = ops.register_cb_layer(method, layer)
    x = torch.ones((2, 3))

    torch.library.opcheck(ops.cb_linear_forward, (x, layer_id))

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake_out = ops.cb_linear_forward(mode.from_tensor(x), layer_id)
    assert fake_out.shape == (2, layer._cb_N)


def test_linear_custom_op_compiles_with_static_layer_id():
    method = _Method()
    layer = _Layer()
    layer_id = ops.register_cb_layer(method, layer)

    def forward(x):
        return ops.cb_linear_forward(x, layer_id)

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    out = compiled(torch.ones((2, 3)))

    assert out.shape == (2, layer._cb_N)
    assert torch.equal(out, torch.zeros_like(out))


# --- the cudagraph_unsafe tag: retired, and kept retired --------------------


def test_no_prismaquant_op_claims_cudagraph_unsafe():
    """No Gridbook op may carry ``torch.Tag.cudagraph_unsafe``.

    The tag's only consumer is inductor's ``should_partition``, reachable only
    under ``use_inductor_graph_partition=True`` with PIECEWISE cudagraphs —
    off at every vLLM optimization level, and ignored entirely by FULL
    capture. Since the M-branch hoist the kernel ops below are never graph
    NODES anyway, so their tags were metadata nothing read.

    Tagging the two whole-dispatch ops instead is the change that WOULD take
    effect, and it recreates the 2026-07-21 corruption at worse granularity:
    every CB layer an eager partition boundary. See gridbook/ops.py's header
    for the root-cause record this test protects.
    """
    namespace = torch.ops.prismaquant
    checked = []
    for name in dir(namespace):
        if name.startswith("_"):
            continue
        packet = getattr(namespace, name)
        if not hasattr(packet, "overloads"):
            continue          # a plain attribute of the namespace, not an op
        for overload in packet.overloads():
            op = getattr(packet, overload)
            checked.append(f"{name}.{overload}")
            assert torch.Tag.cudagraph_unsafe not in op.tags, (
                f"prismaquant::{name}.{overload} is tagged cudagraph_unsafe. "
                f"See gridbook/ops.py's capture-safety header: no Gridbook op "
                f"carries this tag, and tagging the whole-dispatch ops "
                f"recreates the 2026-07-21 output corruption.")
    assert checked, "no prismaquant ops were registered; the scan proved nothing"


def test_whole_dispatch_op_is_opaque_to_the_compiler():
    """Dynamo must see ONE prismaquant node and never trace the M-branch.

    This is the property that makes the tag irrelevant: the kernel ops run
    inside this op's eager implementation, so they are not graph nodes and
    cannot be partition boundaries. If the hoist is ever reverted, the graph
    grows the traced ATen body and this fails loudly rather than silently
    reintroducing a per-kernel partition surface.
    """
    traced = []
    ran = []

    class _CountingMethod:
        def _apply_inline(self, layer, x, *args):
            del args
            ran.append(x.shape)
            return torch.zeros((*x.shape[:-1], layer._cb_N), dtype=x.dtype)

    method = _CountingMethod()
    layer = _Layer()
    layer_id = ops.register_cb_layer(method, layer)

    def backend(gm, example_inputs):
        del example_inputs
        traced.append(gm)
        return gm.forward

    def forward(x):
        return ops.cb_linear_forward(x, layer_id)

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    out = compiled(torch.ones((2, 3)))

    assert out.shape == (2, layer._cb_N)
    assert len(traced) == 1
    nodes = [n for n in traced[0].graph.nodes
             if n.op == "call_function"
             and "prismaquant" in str(n.target)]
    assert len(nodes) == 1, (
        f"expected exactly one opaque prismaquant node, traced {nodes}; the "
        f"M-branch hoist is what keeps the kernel ops out of the graph")
    # And the body ran at EXECUTION, not while tracing.
    assert ran == [torch.Size([2, 3])]


def test_capture_size_gate_constants_match_the_dispatch_boundaries():
    """The advisory's numbers must be the ones dispatch actually uses.

    ``ops.py`` holds its own copies so this check does not pull the vLLM-bound
    modules into its import graph; that is only safe if a ratchet pins them
    together, or the warning would quote boundaries nothing enforces.
    """
    pytest.importorskip("vllm")
    from gridbook.linear import CUDA_GEMV_M_MAX
    import inspect

    from gridbook import moe

    assert ops._DENSE_DECODE_MAX == CUDA_GEMV_M_MAX
    source = inspect.getsource(moe.PrismaQuantCBMoEMethod._apply_inline)
    assert f"num_tokens <= {ops._MOE_DECODE_MAX}" in source


def test_capture_size_advisory_is_silent_without_a_vllm_config(capsys):
    ops._CAPTURE_GATE_WARNED.clear()
    try:
        ops.warn_if_capture_sizes_exceed_the_decode_gates()
    finally:
        ops._CAPTURE_GATE_WARNED.clear()
    assert "cudagraph_capture_sizes" not in capsys.readouterr().err


def test_capture_size_advisory_names_the_offending_sizes(monkeypatch, capsys):
    """A capture size above the decode gates records the prefill arm."""
    import sys
    import types as _types

    config = _types.ModuleType("vllm.config")
    config.get_current_vllm_config = lambda: _types.SimpleNamespace(
        compilation_config=_types.SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 8, 32, 256]))
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setitem(sys.modules, "vllm",
                        sys.modules.get("vllm") or _types.ModuleType("vllm"))

    ops._CAPTURE_GATE_WARNED.clear()
    try:
        ops.warn_if_capture_sizes_exceed_the_decode_gates()
        error = capsys.readouterr().err
        # Warned once per process, not once per layer.
        ops.warn_if_capture_sizes_exceed_the_decode_gates()
        assert capsys.readouterr().err == ""
    finally:
        ops._CAPTURE_GATE_WARNED.clear()
    assert "[32, 256]" in error
    assert "FULL_DECODE_ONLY" in error
    assert "8" in error and "16" in error
