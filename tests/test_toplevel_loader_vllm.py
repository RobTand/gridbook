"""Live vLLM contract for the legacy fused native-NVFP4 compatibility path.

The torch-only suite proves routing and rejection behavior. This file verifies
that vLLM 0.24's actual CompressedTensorsW4A4Nvfp4MoEMethod registers the shapes,
dtypes, and bound weight-loader metadata that the fail-closed shim inspects.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

pytest.importorskip("vllm")
if not torch.cuda.is_available():
    pytest.skip("live NVFP4 backend selection needs CUDA", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.compressed_tensors.\
    compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )

from gridbook.moe_toplevel_loader import (
    _NATIVE_NVFP4_EXPERT_SUFFIX_TO_LEAF,
    install_toplevel_cb_expert_loader,
)


def _source_stack(prefix: str, experts: int, hidden: int, intermediate: int):
    def u8(shape, offset):
        return ((torch.arange(prod(shape)).reshape(shape)
                 + offset) % 256).to(torch.uint8)

    def fp8(shape, offset):
        values = torch.tensor([0.5, 1.0, 2.0, 4.0])
        idx = ((torch.arange(prod(shape)).reshape(shape)
                + offset) % values.numel())
        return values[idx].to(torch.float8_e4m3fn)

    return {
        prefix + "gate_up_proj.weight_packed": u8(
            (experts, 2 * intermediate, hidden // 2), 1),
        prefix + "down_proj.weight_packed": u8(
            (experts, hidden, intermediate // 2), 17),
        prefix + "gate_up_proj.weight_scale": fp8(
            (experts, 2 * intermediate, hidden // 16), 0),
        prefix + "down_proj.weight_scale": fp8(
            (experts, hidden, intermediate // 16), 1),
        prefix + "gate_up_proj.weight_global_scale": (
            torch.arange(experts * 2, dtype=torch.float32).reshape(experts, 2)
            + 11),
        prefix + "down_proj.weight_global_scale": (
            torch.arange(experts, dtype=torch.float32) + 21),
        prefix + "gate_up_proj.input_global_scale": (
            torch.arange(experts * 2, dtype=torch.float32).reshape(experts, 2)
            + 31),
        prefix + "down_proj.input_global_scale": (
            torch.arange(experts, dtype=torch.float32) + 41),
    }


def test_actual_vllm_nvfp4_method_registration_and_whole_stack_copy():
    experts, hidden, intermediate = 2, 32, 16
    parallel = FusedMoEParallelConfig.make_no_parallel()
    config = FusedMoEConfig(
        num_experts=experts,
        experts_per_token=1,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=experts,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    method = CompressedTensorsW4A4Nvfp4MoEMethod(
        config, layer_name="model.layers.0.mlp.experts", use_a16=False)

    class _Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = config
            self.quant_method = method
            self.hidden_size = hidden
            self.intermediate_size_per_partition = intermediate
            self.weight_loader_calls = 0

        def weight_loader(self, *args, **kwargs):
            self.weight_loader_calls += 1
            raise AssertionError("raw whole-stack copy must not call this")

    owner = _Owner()
    method.create_weights(
        layer=owner,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
        weight_loader=owner.weight_loader,
    )

    class _Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            layer = self.model.layers[0]
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.Module()
            layer.mlp.experts.routed_experts = owner
            self.delegated = []

        def load_weights(self, weights):
            self.delegated.extend(name for name, _ in weights)
            return set()

    install_toplevel_cb_expert_loader(_Top)
    model = _Top()
    checkpoint_prefix = "model.layers.0.mlp.experts."
    param_prefix = "model.layers.0.mlp.experts.routed_experts."
    source = _source_stack(
        checkpoint_prefix, experts, hidden, intermediate)

    loaded = model.load_weights(iter(source.items()))

    assert model.delegated == []
    assert owner.weight_loader_calls == 0
    assert loaded == {
        param_prefix + leaf
        for leaf in _NATIVE_NVFP4_EXPERT_SUFFIX_TO_LEAF.values()
    }
    params = dict(model.named_parameters())
    for name, expected in source.items():
        suffix = next(
            suffix for suffix in _NATIVE_NVFP4_EXPERT_SUFFIX_TO_LEAF
            if name.endswith(suffix))
        param = params[
            param_prefix + _NATIVE_NVFP4_EXPERT_SUFFIX_TO_LEAF[suffix]]
        assert param.weight_loader.__self__ is owner
        assert torch.equal(param, expected), suffix
