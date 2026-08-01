"""CPU integration gates for config, dense load, and fused eligibility."""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from gridbook.nvfp4_activation_contract import (
    CONTRACT_KEY,
    CONTRACT_SCHEMA,
    EXECUTION_CONTRACT,
    GROUP_SIZE,
    LEGACY_POLICY,
    TENSOR_SUFFIX,
    VALUE_DTYPE,
    target_values_sha256,
)


def _install_vllm_stubs():
    def module(name):
        value = types.ModuleType(name)
        sys.modules[name] = value
        return value

    module("vllm")
    module("vllm.model_executor")
    module("vllm.model_executor.layers")
    module("vllm.model_executor.layers.quantization")
    linear = module("vllm.model_executor.layers.linear")
    linear.LinearBase = type("LinearBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    parameter = module("vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kwargs):
            pass

    parameter.ModelWeightParameter = StubParameter
    parameter.ChannelQuantScaleParameter = StubParameter
    parameter.PerTensorScaleParameter = StubParameter
    base = module("vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        def __init__(self):
            pass

    base.QuantizationConfig = QuantizationConfig
    base.QuantizeMethodBase = object
    embedding = module(
        "vllm.model_executor.layers.vocab_parallel_embedding"
    )
    embedding.UnquantizedEmbeddingMethod = type("UEM", (), {})
    embedding.VocabParallelEmbedding = type("VPE", (), {})
    fused = module("vllm.model_executor.layers.fused_moe")
    fused.RoutedExperts = type("RoutedExperts", (), {})


@pytest.fixture(scope="module", autouse=True)
def runtime_modules(isolated_gridbook_runtime_imports):
    del isolated_gridbook_runtime_imports
    try:
        from vllm.model_executor.parameter import PerTensorScaleParameter  # noqa:F401
    except Exception:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                sys.modules.pop(name, None)
        _install_vllm_stubs()
    from gridbook.config import PrismaQuantConfig
    from gridbook.linear import PrismaQuantCBLinearMethod

    globals()["PrismaQuantConfig"] = PrismaQuantConfig
    globals()["PrismaQuantCBLinearMethod"] = PrismaQuantCBLinearMethod


def _scheme(*, contract=True, grid="fp4"):
    scheme = {
        "grid": grid,
        "mode": "product",
        "k": 16 if grid == "fp4" else 44,
        "n_sub": 2 if grid == "fp4" else 4,
        "type_size": 73 if grid == "fp4" else 176,
        "group_size": 16 if grid == "fp4" else 0,
        "vec_dim": 8,
        "codebook_ref": ["cb.a", "cb.b"] if grid == "fp4" else "cb.fp8",
        "scale_coding": "two_tier" if grid == "fp4" else None,
    }
    if contract:
        scheme["activation_contract"] = CONTRACT_KEY
    return scheme


def _contract(scales):
    return {
        "schema": CONTRACT_SCHEMA,
        "contract": EXECUTION_CONTRACT,
        "group_size": GROUP_SIZE,
        "tensor_suffix": TENSOR_SUFFIX,
        "value_dtype": VALUE_DTYPE,
        "input_global_scale_policy": LEGACY_POLICY,
        "target_count": len(scales),
        "target_names": sorted(scales),
        "target_values_sha256": target_values_sha256(
            scales, policy=LEGACY_POLICY
        ),
    }


def _config(scales, *, scheme=None, targets=None, contract=True):
    targets = targets or list(scales)
    result = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": {
            "cb": {
                "format": "NVFP4_CB_K16",
                "targets": targets,
                "scheme": _scheme(contract=contract) if scheme is None else scheme,
            }
        },
        "ignore": [],
    }
    if contract:
        result["execution_contracts"] = {CONTRACT_KEY: _contract(scales)}
    return result


def _resolved(tmp_path, scales, *, cfg=None):
    tensors = {
        f"{name}.{TENSOR_SUFFIX}": torch.tensor([value], dtype=torch.float32)
        for name, value in scales.items()
    }
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")
    config = PrismaQuantConfig.from_config(cfg or _config(scales))
    config._sidecar_source = (str(tmp_path), None)
    config._ensure_resolved()
    return config


def test_config_attests_complete_physical_payload(tmp_path):
    scales = {"model.layers.0.mlp.down_proj": 3.25}
    cfg = _resolved(tmp_path, scales)
    assert cfg._nvfp4_activation_scales == scales
    target = next(iter(scales))
    assert cfg.activation_scales_for_targets([target]) == [3.25]

    class Mapper:
        @staticmethod
        def apply_dict(values):
            return {"served." + key: value for key, value in values.items()}

        @staticmethod
        def apply_list(values):
            return ["served." + value for value in values]

    cfg.apply_vllm_mapper(Mapper())
    assert cfg.activation_scales_for_targets(["served." + target]) == [3.25]


def test_moe_stage_resolution_keeps_gate_up_and_down_scales_distinct(tmp_path):
    prefix = "model.layers.0.mlp.experts"
    gate_up = prefix + ".gate_up_proj"
    down = prefix + ".down_proj"
    scales = {gate_up: 2.5, down: 1.25}
    cfg = _resolved(tmp_path, scales)
    assert cfg._moe_scheme_for_prefix(prefix)["activation_contract"] == CONTRACT_KEY
    stages = cfg.moe_activation_stage_targets(prefix)
    assert stages == {"w13": [gate_up], "w2": [down]}
    assert cfg.activation_scales_for_targets(stages["w13"]) == [2.5]
    assert cfg.activation_scales_for_targets(stages["w2"]) == [1.25]


def test_config_rejects_digest_mismatch_and_missing_scalar(tmp_path):
    target = "model.layers.0.mlp.down_proj"
    cfg = _config({target: 3.25})
    # Payload is valid F32 but does not match the digest-bound value.
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _resolved(tmp_path, {target: 4.0}, cfg=cfg)

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    safetensors_torch.save_file(
        {"unrelated": torch.ones(1)}, missing_dir / "model.safetensors"
    )
    config = PrismaQuantConfig.from_config(cfg)
    config._sidecar_source = (str(missing_dir), None)
    with pytest.raises(ValueError, match="target_count"):
        config._ensure_resolved()


def test_scheme_contract_links_fail_closed_before_weight_load(tmp_path):
    target = "model.layers.0.mlp.down_proj"
    tagged_without_top = _config({target: 3.25}, contract=False)
    tagged_without_top["config_groups"]["cb"]["scheme"] = _scheme(contract=True)
    with pytest.raises(ValueError, match="top-level execution contract"):
        _resolved(tmp_path, {target: 3.25}, cfg=tagged_without_top)

    top_without_tag = _config({target: 3.25})
    top_without_tag["config_groups"]["cb"]["scheme"].pop(
        "activation_contract"
    )
    with pytest.raises(ValueError, match="every custom FP4-CB"):
        _resolved(tmp_path, {target: 3.25}, cfg=top_without_tag)

    fp8_tagged = _config({target: 3.25}, scheme=_scheme(grid="fp8"))
    with pytest.raises(ValueError, match="fp4-only"):
        _resolved(tmp_path, {target: 3.25}, cfg=fp8_tagged)

    mismatched_physical_name = _config({target: 3.25})
    mismatched_physical_name["config_groups"]["cb"]["targets"] = [
        "model.layers.0.mlp.up_proj"
    ]
    with pytest.raises(ValueError, match="target_names"):
        _resolved(tmp_path, {target: 3.25}, cfg=mismatched_physical_name)


def test_legacy_fp4_config_does_not_read_or_register_scale(monkeypatch):
    target = "model.layers.0.mlp.down_proj"
    config = PrismaQuantConfig.from_config(
        _config({target: 3.25}, contract=False)
    )
    monkeypatch.setattr(
        config, "_read_nvfp4_activation_scales",
        lambda: pytest.fail("legacy config must not scan activation scalars"),
    )
    config._ensure_resolved()
    method = PrismaQuantCBLinearMethod(
        config, config.target_scheme[target], target
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer, 256, [8], 256, 8, torch.bfloat16, weight_loader=None
    )
    assert not hasattr(layer, "input_global_scale")
    assert method._fused_fp4_ok(layer, 256) is False


def test_dense_contracted_scale_load_is_fail_closed_and_merged_exact(tmp_path):
    targets = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    scales = {target: 2.5 for target in targets}
    config = _resolved(tmp_path, scales)
    method = PrismaQuantCBLinearMethod(
        config, config._scheme_for_prefix("model.layers.0.self_attn.qkv_proj"),
        "model.layers.0.self_attn.qkv_proj",
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer, 256, [8, 8, 8], 256, 24, torch.bfloat16,
        weight_loader=None,
    )
    with pytest.raises(ValueError, match="finite and > 0"):
        method._finalize_static_activation_scale(layer, targets)
    layer.input_global_scale.data.fill_(2.5)
    method._finalize_static_activation_scale(layer, targets)
    assert layer._cb_fp4_input_global_scale.ndim == 0
    assert torch.equal(layer._cb_fp4_input_global_scale, torch.tensor(2.5))
    layer._cb_N = 24
    layer._cb_row_offset = torch.zeros(24, dtype=torch.int32)
    assert method._fused_fp4_ok(layer, 256) is False  # no loaded kernel state

    layer.input_global_scale.data[1] = 2.0
    with pytest.raises(ValueError, match="non-identical"):
        method._finalize_static_activation_scale(layer, targets)


def test_static_scale_makes_native_quantization_chunk_invariant(monkeypatch):
    """The same row gets the same quantizer input regardless of batch peers."""

    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    method.is_v2 = True
    method._sub_table = [1.0] * 16
    method._fused_fp4_ok = lambda layer, K, *, rowwise=False: not rowwise
    layer = types.SimpleNamespace(
        _cb_fp4_input_global_scale=torch.tensor([3.0]),
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )
    calls = []

    vops = types.ModuleType("vllm._custom_ops")

    def quantize(x, scale):
        calls.append((x.clone(), scale.clone()))
        return x * scale, torch.zeros(x.shape[0], 1)

    vops.scaled_fp4_quant = quantize
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", vops)
    monkeypatch.setattr(sys.modules["vllm"], "_custom_ops", vops,
                        raising=False)
    from gridbook import cuda_ext

    class Ext:
        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(aq, *args):
            return aq[:, :8]

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    row = torch.arange(256, dtype=torch.float32).reshape(1, 256)
    method._try_fused_fp4(layer, row, 8, 256, 1)
    method._try_fused_fp4(
        layer, torch.cat([row, torch.full_like(row, 1e6)]), 8, 256, 2
    )
    assert torch.equal(calls[0][0][0] * calls[0][1],
                       calls[1][0][0] * calls[1][1])
    assert calls[0][1].item() == calls[1][1].item() == 3.0


def test_dense_rowwise_quantizer_outputs_feed_the_existing_fused_gemm(
    monkeypatch,
):
    from gridbook import nvfp4_activation_contract as activation_contract

    # A phase screen must touch only the rowwise activation family.  The
    # weight-side E4M3 ceiling remains fixed in codec.build_compose_u8.
    monkeypatch.setattr(
        activation_contract, "ROWWISE_RANGE_MULTIPLIER", 256.0
    )
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    method.is_v2 = True
    method._sub_table = [1.0] * 16
    method._fused_fp4_ok = (
        lambda layer, K, *, rowwise=False: rowwise
    )
    layer = types.SimpleNamespace(
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )
    quant_calls = []
    gemm_calls = []

    vops = types.ModuleType("vllm._custom_ops")
    vops.scaled_fp4_quant = lambda *_args: pytest.fail(
        "rowwise mode must not enter the static vLLM quantizer"
    )
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", vops)
    monkeypatch.setattr(sys.modules["vllm"], "_custom_ops", vops,
                        raising=False)
    from gridbook import cuda_ext

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_rows(x, multiplier):
            # A deterministic row-local stand-in lets this CPU test prove that
            # no batch reduction is added around the extension call.
            packed = x[:, :128].to(torch.float32).round().to(torch.uint8)
            sfa = x[:, :16].to(torch.float32).round().to(torch.uint8)
            scales = x.float().abs().amax(dim=1) / (multiplier * 6.0)
            quant_calls.append((packed.clone(), sfa.clone(), scales.clone(),
                                multiplier))
            return packed, sfa, scales

        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(
            aq, sfa, packed_weight, lut, compose, a_scales, b_scales,
            n, k, k_bits, n_sub, type_size, is_v2, lut_tile_ids,
        ):
            gemm_calls.append((aq.clone(), sfa.clone(), a_scales.clone(),
                               packed_weight, lut_tile_ids))
            return torch.zeros(aq.shape[0], n, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    target = torch.arange(256, dtype=torch.bfloat16).reshape(1, 256)
    peers = torch.full((2, 256), 1.0e4, dtype=torch.bfloat16)
    one = method._try_fused_fp4(
        layer, target, 8, 256, 1, rowwise=True
    )
    batch = method._try_fused_fp4(
        layer, torch.cat((target, peers)), 8, 256, 3, rowwise=True
    )

    assert one.shape == (1, 8)
    assert batch.shape == (3, 8)
    assert [call[3] for call in quant_calls] == [256.0, 256.0]
    assert torch.equal(quant_calls[0][0][0], quant_calls[1][0][0])
    assert torch.equal(quant_calls[0][1][0], quant_calls[1][1][0])
    assert torch.equal(quant_calls[0][2][0], quant_calls[1][2][0])
    # Packed data, flattened SFA, and the returned residual scales flow
    # directly into the sole existing GEMM implementation.
    assert torch.equal(gemm_calls[0][0], quant_calls[0][0])
    assert torch.equal(gemm_calls[0][1], quant_calls[0][1].reshape(-1))
    assert torch.equal(gemm_calls[0][2], quant_calls[0][2])


def test_rowwise_phase_override_cannot_change_weight_compose_bytes(
    monkeypatch,
):
    from gridbook import codec
    from gridbook import nvfp4_activation_contract as activation_contract

    expected = codec.build_compose_u8().clone()
    assert codec.FP8_ELEMENT_MAX == 448.0
    monkeypatch.setattr(
        activation_contract, "ROWWISE_RANGE_MULTIPLIER", 256.0
    )

    assert activation_contract.rowwise_range_multiplier() == 256.0
    assert codec.FP8_ELEMENT_MAX == 448.0
    assert torch.equal(codec.build_compose_u8(), expected)
