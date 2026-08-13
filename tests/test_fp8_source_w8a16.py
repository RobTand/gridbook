"""CPU contract tests for raw-resident block128 source-FP8 W8A16."""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_vllm_method_stubs(monkeypatch):
    _module(monkeypatch, "vllm")
    _module(monkeypatch, "vllm.model_executor")
    _module(monkeypatch, "vllm.model_executor.layers")
    linear = _module(monkeypatch, "vllm.model_executor.layers.linear")
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    parameter = _module(monkeypatch, "vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    parameter.BlockQuantScaleParameter = StubParameter
    parameter.ModelWeightParameter = StubParameter


def _dequant_block128(q, scales):
    scale_bytes = scales.view(torch.uint8)
    exponents = scale_bytes.to(torch.int16) - 127
    sf = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32), exponents)
    sf = sf.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return (q.float() * sf[:q.shape[0], :q.shape[1]]).to(torch.bfloat16)


def _patch_native_contract(monkeypatch, lane, dsv4_woa):
    from gridbook import cuda_ext, ops

    monkeypatch.setattr(lane, "_require_source_cuda", lambda tensor: None)
    monkeypatch.setattr(
        cuda_ext, "require_fp8_source_w8a16_ext",
        lambda operation="", device=None: object())
    monkeypatch.setattr(
        cuda_ext, "require_bf16_grouped_ext",
        lambda operation="": object())
    monkeypatch.setattr(dsv4_woa, "install_dsv4_woa_adapter", lambda: None)

    calls = []

    def gemv(x, q, scales, groups):
        calls.append(("gemv", int(x.shape[0]), groups))
        w = _dequant_block128(q, scales)
        rows = q.shape[0] // groups
        return torch.stack([
            x[:, group].float() @
            w[group * rows:(group + 1) * rows].float().t()
            for group in range(groups)
        ], dim=1).to(torch.bfloat16).reshape(x.shape[0], q.shape[0])

    def expand(q, scales):
        calls.append(("expand",))
        return _dequant_block128(q, scales)

    def grouped_mm(a, weights, expert_ends, expert_start=0):
        calls.append(("grouped", tuple(int(v) for v in expert_ends.tolist())))
        out = torch.empty(a.shape[0], weights.shape[1], dtype=torch.bfloat16)
        start = 0 if expert_start == 0 else int(expert_ends[expert_start - 1])
        for local, weight in enumerate(weights):
            end = int(expert_ends[expert_start + local])
            out[start:end] = (a[start:end].float() @ weight.float().t()).to(
                torch.bfloat16)
            start = end
        return out

    monkeypatch.setattr(ops, "fp8_source_gemv", gemv)
    monkeypatch.setattr(ops, "fp8_source_expand_bf16", expand)
    monkeypatch.setattr(ops, "cb_bf16_grouped_mm", grouped_mm)
    # Exercise the implementation behind the opaque boundary without asking a
    # CPU-only torch build to dispatch a CUDA custom op.
    monkeypatch.setattr(
        ops, "fp8_source_linear_forward",
        lambda x, layer_id: ops._lookup_cb_layer(layer_id)[0]._apply_inline(
            ops._lookup_cb_layer(layer_id)[1], x))
    return calls


def _built_layer(monkeypatch, *, groups=1):
    _install_vllm_method_stubs(monkeypatch)
    from gridbook import dsv4_woa
    from gridbook import fp8_source_w8a16 as lane

    calls = _patch_native_contract(monkeypatch, lane, dsv4_woa)
    rows, k = 128, 128
    layer = torch.nn.Module()
    layer.tp_size = 1
    method = lane.build_fp8_source_w8a16_method(lane.WIRE_FP8_BLOCK128)
    method.create_weights(
        layer, k, [groups * rows], k, groups * rows, torch.bfloat16)
    if groups > 1:
        layer.is_bmm = True
        layer.bmm_batch_size = groups
        # Keep this small tensor focused on group-ordering math. The immutable
        # release geometry and every near miss are pinned independently in
        # test_fp8_source_w8a16_geometry.py.
        monkeypatch.setattr(lane, "_DSV4_BMM_GROUPS", groups)
        monkeypatch.setattr(lane, "_DSV4_BMM_ROWS", rows)
        monkeypatch.setattr(lane, "_DSV4_BMM_K", k)
    torch.manual_seed(9)
    layer.weight.data.copy_(
        (torch.randn(groups * rows, k) * 0.25).to(torch.float8_e4m3fn))
    scale_bytes = torch.tensor(
        [[127], [128]][:groups], dtype=torch.uint8)
    layer.weight_scale_inv.data.view(torch.uint8).copy_(scale_bytes)
    q_before = layer.weight.detach().view(torch.uint8).clone()
    s_before = layer.weight_scale_inv.detach().view(torch.uint8).clone()
    method.process_weights_after_loading(layer)
    return lane, dsv4_woa, method, layer, calls, q_before, s_before


def test_method_accepts_block128_only_before_vllm_import(monkeypatch):
    from gridbook import fp8_source_w8a16 as lane

    with pytest.raises(ValueError, match="accepts only.*direct g32 MXFP8"):
        lane.build_fp8_source_w8a16_method("mxfp8_e4m3_e8m0_g32")


def test_process_keeps_exact_raw_planes_resident(monkeypatch):
    lane, _, method, layer, calls, q_before, s_before = _built_layer(monkeypatch)
    assert type(method).__name__ == "Fp8SourceW8A16LinearMethod"
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype == torch.float8_e8m0fnu
    assert torch.equal(layer.weight.detach().view(torch.uint8), q_before)
    assert torch.equal(layer.weight_scale_inv.detach().view(torch.uint8),
                       s_before)
    assert layer._fp8_source_resident_bytes == q_before.numel() + s_before.numel()
    assert not any(name.startswith("weight_sf") for name, _ in
                   layer.named_buffers())
    assert calls == []


def test_dense_decode_and_prefill_choose_native_arms(monkeypatch):
    _, _, method, layer, calls, _, _ = _built_layer(monkeypatch)
    w = _dequant_block128(layer.weight, layer.weight_scale_inv)

    decode_x = torch.randn(2, 128, dtype=torch.bfloat16)
    decode = method.apply(layer, decode_x)
    expected_decode = (decode_x.float() @ w.float().t()).to(torch.bfloat16)
    assert torch.equal(decode, expected_decode)
    assert calls == [("gemv", 2, 1)]

    calls.clear()
    prefill_x = torch.randn(9, 128, dtype=torch.bfloat16)
    prefill = method.apply(layer, prefill_x)
    expected_prefill = (prefill_x.float() @ w.float().t()).to(torch.bfloat16)
    assert torch.equal(prefill, expected_prefill)
    assert calls == [("expand",), ("grouped", (9,))]


def test_bmm_decode_and_prefill_preserve_group_mapping(monkeypatch):
    _, dsv4_woa, method, layer, calls, _, _ = _built_layer(
        monkeypatch, groups=2)
    assert getattr(layer, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR) == \
        dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ABI
    w = _dequant_block128(
        layer.weight, layer.weight_scale_inv).view(2, 128, 128)

    def oracle(x):
        return torch.stack([
            (x[:, group].float() @ w[group].float().t()).to(torch.bfloat16)
            for group in range(2)
        ], dim=1)

    decode_x = torch.randn(1, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(method.apply(layer, decode_x), oracle(decode_x))
    assert calls == [("gemv", 1, 2)]

    calls.clear()
    prefill_x = torch.randn(9, 2, 128, dtype=torch.bfloat16)
    assert torch.equal(method.apply(layer, prefill_x), oracle(prefill_x))
    assert calls == [("expand",), ("grouped", (9, 18))]


def test_apply_refuses_activation_cast_and_bias(monkeypatch):
    _, _, method, layer, _, _, _ = _built_layer(monkeypatch)
    with pytest.raises(TypeError, match="preserves BF16"):
        method.apply(layer, torch.zeros(1, 128, dtype=torch.float32))
    with pytest.raises(ValueError, match="does not serve biased"):
        method.apply(
            layer, torch.zeros(1, 128, dtype=torch.bfloat16),
        torch.zeros(128, dtype=torch.bfloat16))
