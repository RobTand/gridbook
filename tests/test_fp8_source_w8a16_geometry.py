"""CPU-only admission gates for source-FP8 W8A16 grouped BMM geometry."""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _method(monkeypatch):
    _module(monkeypatch, "vllm")
    _module(monkeypatch, "vllm.model_executor")
    _module(monkeypatch, "vllm.model_executor.layers")
    linear = _module(monkeypatch, "vllm.model_executor.layers.linear")
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    parameter = _module(monkeypatch, "vllm.model_executor.parameter")
    parameter.BlockQuantScaleParameter = type(
        "BlockQuantScaleParameter", (), {})
    parameter.ModelWeightParameter = type("ModelWeightParameter", (), {})

    from gridbook import fp8_source_w8a16 as lane

    return lane, lane.build_fp8_source_w8a16_method(
        lane.WIRE_FP8_BLOCK128)


def _layer(*, groups, rows, k, tp_size, is_bmm=True):
    layer = torch.nn.Module()
    total_rows = groups * rows
    q = torch.empty(
        total_rows, k, dtype=torch.float8_e4m3fn, device="meta")
    scales = torch.empty(
        (total_rows + 127) // 128,
        (k + 127) // 128,
        dtype=torch.float8_e8m0fnu,
        device="meta",
    )
    layer.register_parameter(
        "weight", torch.nn.Parameter(q, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scales, requires_grad=False))
    layer.tp_size = tp_size
    if is_bmm:
        layer.is_bmm = True
        layer.bmm_batch_size = groups
    return layer


def _patch_load_edges(monkeypatch, lane):
    from gridbook import cuda_ext, dsv4_woa, ops

    calls = []
    monkeypatch.setattr(lane, "_require_source_cuda", lambda tensor: None)
    monkeypatch.setattr(
        cuda_ext,
        "require_fp8_source_w8a16_ext",
        lambda operation="", device=None: calls.append("source_ext"),
    )
    monkeypatch.setattr(
        cuda_ext,
        "require_bf16_grouped_ext",
        lambda operation="": calls.append("grouped_ext"),
    )
    monkeypatch.setattr(
        ops, "register_cb_layer", lambda method, layer: 17)
    monkeypatch.setattr(
        dsv4_woa,
        "install_dsv4_woa_adapter",
        lambda: calls.append("adapter"),
    )
    return dsv4_woa, calls


@pytest.mark.parametrize(
    ("groups", "rows", "k", "tp_size", "got"),
    (
        (7, 1024, 4096, 1, "G=7, N=1024, K=4096, TP=1"),
        (8, 896, 4096, 1, "G=8, N=896, K=4096, TP=1"),
        (8, 1024, 3968, 1, "G=8, N=1024, K=3968, TP=1"),
        (8, 1024, 4096, 2, "G=8, N=1024, K=4096, TP=2"),
    ),
)
def test_grouped_geometry_near_miss_refuses_before_native_resolution_or_marker(
        monkeypatch, groups, rows, k, tp_size, got):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=groups, rows=rows, k=k, tp_size=tp_size)

    with pytest.raises(ValueError, match="qualified only") as exc:
        method.process_weights_after_loading(layer)

    message = str(exc.value)
    assert "G=8, N=1024, K=4096, TP=1" in message
    assert got in message
    assert calls == []
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)
    assert not hasattr(layer, lane._READY_ATTR)


def test_exact_grouped_geometry_resolves_both_arms_then_installs_marker(
        monkeypatch):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(groups=8, rows=1024, k=4096, tp_size=1)

    method.process_weights_after_loading(layer)

    assert calls == ["source_ext", "grouped_ext", "adapter"]
    assert layer._fp8_source_groups == 8
    assert layer._fp8_source_rows == 1024
    assert layer._fp8_source_K == 4096
    assert getattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR) == \
        dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ABI


def test_dense_geometry_remains_separate_from_grouped_dsv4_gate(monkeypatch):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=1, rows=136, k=256, tp_size=1, is_bmm=False)

    method.process_weights_after_loading(layer)

    assert calls == ["source_ext", "grouped_ext"]
    assert layer._fp8_source_groups == 1
    assert layer._fp8_source_rows == 136
    assert layer._fp8_source_K == 256
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)


def test_dense_tp_near_miss_refuses_before_native_resolution(monkeypatch):
    lane, method = _method(monkeypatch)
    dsv4_woa, calls = _patch_load_edges(monkeypatch, lane)
    layer = _layer(
        groups=1, rows=136, k=256, tp_size=2, is_bmm=False)

    with pytest.raises(ValueError, match="dense serving.*TP=1.*TP=2"):
        method.process_weights_after_loading(layer)

    assert calls == []
    assert not hasattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR)
    assert not hasattr(layer, lane._READY_ATTR)
