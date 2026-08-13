"""CPU checks for source-dtype guards on raw FP8 checkpoint loaders."""
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


def _invoke(pattern, loader, param, loaded_weight):
    if pattern == "default":
        return loader(param, loaded_weight)
    if pattern == "fused-positional":
        return loader(param, loaded_weight, "q")
    if pattern == "merged-keyword":
        return loader(
            param=param,
            loaded_weight=loaded_weight,
            loaded_shard_id=1,
        )
    raise AssertionError(f"unknown test loader pattern {pattern}")


@pytest.mark.parametrize(
    ("plane", "expected_dtype", "wrong_dtype", "parameter_name"),
    [
        ("value-plane", torch.float8_e4m3fn, torch.bfloat16, "weight"),
        ("scale-plane", torch.float8_e8m0fnu, torch.uint8,
         "weight_scale_inv"),
    ],
)
@pytest.mark.parametrize(
    "pattern", ["default", "fused-positional", "merged-keyword"])
def test_source_loader_rejects_before_delegation_and_preserves_valid_bytes(
        monkeypatch, plane, expected_dtype, wrong_dtype, parameter_name,
        pattern):
    _install_vllm_method_stubs(monkeypatch)
    from gridbook import fp8_source_w8a16 as lane

    calls = []
    delegated = object()

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        if args:
            param, loaded_weight = args[:2]
        else:
            param = kwargs["param"]
            loaded_weight = kwargs["loaded_weight"]
        param.data.copy_(loaded_weight)
        return delegated

    method = lane.build_fp8_source_w8a16_method(lane.WIRE_FP8_BLOCK128)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=4,
        output_partition_sizes=[4],
        input_size=4,
        output_size=4,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    param = getattr(layer, parameter_name)

    wrong = torch.zeros(param.shape, dtype=wrong_dtype)
    before = param.detach().view(torch.uint8).clone()
    with pytest.raises(
            TypeError,
            match=rf"{plane} checkpoint tensor must be exactly "
                  rf"{expected_dtype} before vLLM loader delegation"):
        _invoke(pattern, param.weight_loader, param, wrong)
    assert calls == []
    assert torch.equal(param.detach().view(torch.uint8), before)

    source = torch.empty(param.shape, dtype=expected_dtype)
    source.view(torch.uint8).copy_(
        torch.arange(source.numel(), dtype=torch.uint8).reshape(source.shape))
    result = _invoke(pattern, param.weight_loader, param, source)

    assert result is delegated
    assert len(calls) == 1
    args, kwargs = calls[0]
    if pattern == "default":
        assert len(args) == 2 and args[0] is param and args[1] is source
        assert kwargs == {}
    elif pattern == "fused-positional":
        assert len(args) == 3 and args[0] is param and args[1] is source
        assert args[2] == "q" and kwargs == {}
    else:
        assert args == ()
        assert kwargs["param"] is param
        assert kwargs["loaded_weight"] is source
        assert kwargs["loaded_shard_id"] == 1
    assert torch.equal(
        param.detach().view(torch.uint8), source.detach().view(torch.uint8))
