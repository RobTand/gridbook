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
