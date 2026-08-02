"""CPU integration gates for config, dense load, and fused eligibility."""
from __future__ import annotations

import json
import struct
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

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
    # This CPU contract needs only vLLM's class surface. Importing the real
    # package registers process-global Torch opaque types that cannot be undone
    # when the isolation fixture restores sys.modules, so always use stubs.
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
    _write_f32_scalars(
        tmp_path / "model.safetensors",
        {f"{name}.{TENSOR_SUFFIX}": value for name, value in scales.items()},
    )
    config = PrismaQuantConfig.from_config(cfg or _config(scales))
    config._sidecar_source = (str(tmp_path), None)
    config._ensure_resolved()
    return config


def _write_f32_scalars(path, values):
    """Write a minimal safetensors fixture without NumPy or writer APIs."""
    header = {}
    payload = []
    offset = 0
    for name, value in sorted(values.items()):
        data = struct.pack("<f", float(value))
        header[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + len(data)],
        }
        payload.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


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
    _write_f32_scalars(missing_dir / "model.safetensors", {"unrelated": 1.0})
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
    assert layer._cb_fp4_input_global_scale_f32 == 2.5
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

    def quantize(x, scale):
        calls.append((x.clone(), scale.clone()))
        return x * scale, torch.zeros(x.shape[0], 1)

    from gridbook import linear as cb_linear
    monkeypatch.setattr(cb_linear, "native_fp4_quant", quantize)
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


def test_dense_static_lsq_uses_attested_g_and_existing_fused_gemm(monkeypatch):
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    method.is_v2 = True
    method._sub_table = [1.0] * 16
    method._fused_fp4_ok = (
        lambda layer, K, *, rowwise=False, static_lsq=False: static_lsq
    )
    layer = types.SimpleNamespace(
        _cb_fp4_input_global_scale=torch.tensor(3.0),
        _cb_fp4_input_global_scale_f32=3.0,
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )

    from gridbook import linear as cb_linear
    monkeypatch.setattr(
        cb_linear,
        "native_fp4_quant",
        lambda *_args: pytest.fail(
            "static_lsq must use the shared Gridbook activation primitive"
        ),
    )
    from gridbook import cuda_ext

    calls = []

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_static_lsq(x, global_scale):
            packed = x[:, :128].to(torch.float32).round().to(torch.uint8)
            sfa = torch.zeros(128 * 16, dtype=torch.uint8)
            residual = torch.arange(
                1, x.shape[0] + 1, dtype=torch.float32
            )
            calls.append(("quant", x.clone(), global_scale, residual.clone()))
            return packed, sfa, residual

        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(
            aq, sfa, packed_weight, lut, compose, a_scales, b_scales,
            n, k, k_bits, n_sub, type_size, is_v2, lut_tile_ids, tile_m,
        ):
            calls.append((
                "gemm", aq, sfa, a_scales.clone(), packed_weight, tile_m,
            ))
            return torch.zeros(aq.shape[0], n, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    x = torch.arange(2 * 256, dtype=torch.bfloat16).reshape(2, 256)
    out = method._try_fused_fp4(
        layer, x, 8, 256, 2, static_lsq=True
    )

    assert out.shape == (2, 8)
    assert calls[0][0] == "quant"
    assert calls[0][2] == 3.0
    assert calls[1][0] == "gemm"
    assert torch.equal(calls[1][3], calls[0][3])
    assert calls[1][4] is layer._cb_qw_padded
    assert calls[1][5] == 128


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

    from gridbook import linear as cb_linear
    monkeypatch.setattr(
        cb_linear,
        "native_fp4_quant",
        lambda *_args: pytest.fail(
            "rowwise mode must not enter the static quantizer"
        ),
    )
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
            n, k, k_bits, n_sub, type_size, is_v2, lut_tile_ids, tile_m,
        ):
            gemm_calls.append((aq.clone(), sfa.clone(), a_scales.clone(),
                               packed_weight, lut_tile_ids, tile_m))
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
    assert gemm_calls[0][5] == gemm_calls[1][5] == 128


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


def _rowwise_dense_method():
    """A dense FP4 method whose rowwise fused mode is ATTESTED, as at load."""
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    method.is_v2 = True
    method.is_fp4 = True
    method.prefix = "model.layers.0.mlp.gate_up_proj"
    method._sub_table = [1.0] * 16
    method._fused_fp4_ok = (
        lambda layer, K, *, rowwise=False, static_lsq=False: True
    )
    return method


@pytest.mark.parametrize("mode,dtype", [
    ("rowwise", torch.float32),
    ("static_lsq", torch.float32),
    ("rowwise", torch.float64),
])
def test_dense_fused_fp4_raises_instead_of_serving_a_different_contract(
    monkeypatch, mode, dtype,
):
    """A call-time miss must fail, exactly as the MoE twin already does.

    The mode is attested at model load, so ``_try_fused_fp4`` returning None
    means the lane declined THIS call — here through the rowwise/static-LSQ
    quantizers' half-precision guard. Falling through would have served the
    exact BF16 quality route, whose activation bucket is the fp32-emulated
    group QDQ rather than the format's native ue4m3 scale factors: a different
    served activation contract than the operator selected, chosen silently.
    """
    from gridbook.cuda_ext import NativeKernelUnavailableError

    method = _rowwise_dense_method()
    layer = types.SimpleNamespace(
        _cb_N=8, _cb_K=256, _cb_fused_fp4_mode=mode,
        _cb_fp4_input_global_scale=torch.tensor([3.0]),
        _cb_fp4_input_global_scale_f32=3.0,
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )
    from gridbook import cuda_ext

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext",
                        lambda: types.SimpleNamespace())
    # Anything downstream of the raise would be a fall-through, so make the
    # exact BF16 route loudly unreachable rather than merely unlikely.
    monkeypatch.setattr(
        method, "_require_fp4_v2_product",
        lambda *_a: pytest.fail(
            "fell through to the emulated-QDQ quality route"))

    x = torch.zeros(32, 256, dtype=dtype)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        method._apply_inline(layer, x)
    message = str(exc_info.value)
    assert "became unavailable after model load" in message
    assert repr(mode) in message
    assert str(dtype) in message


def test_dense_fused_fp4_still_serves_the_contract_it_attested(monkeypatch):
    """The negative control: half-precision activations take the lane."""
    method = _rowwise_dense_method()
    layer = types.SimpleNamespace(
        _cb_N=8, _cb_K=256, _cb_fused_fp4_mode="rowwise",
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )
    from gridbook import cuda_ext

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_rows(x, multiplier):
            return (x[:, :128].to(torch.uint8),
                    torch.zeros(x.shape[0] * 16, dtype=torch.uint8),
                    torch.ones(x.shape[0], dtype=torch.float32))

        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(aq, *args, **kwargs):
            return torch.zeros(aq.shape[0], 8, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    monkeypatch.setattr(
        method, "_require_fp4_v2_product",
        lambda *_a: pytest.fail("the attested lane must serve this call"))

    y = method._apply_inline(
        layer, torch.zeros(32, 256, dtype=torch.bfloat16))
    assert y.shape == (32, 8)


# --- K0.4 telemetry: every dense route records what it served --------------


def test_dense_fused_fp4_records_the_served_route(monkeypatch):
    """The fused NVFP4 dense lane had only its three TileM integers.

    A report could see WHICH tile ran but not whether the fused route ran at
    all — and the alternative serves a different activation contract, which is
    the one distinction the record exists to make.
    """
    from gridbook.nvfp4_activation_contract import (ROUTE_CONTRACTS,
                                                    ROUTE_FIELDS, read_route)

    method = _rowwise_dense_method()
    layer = types.SimpleNamespace(
        _cb_N=8, _cb_K=256, _cb_fused_fp4_mode="rowwise",
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )
    from gridbook import cuda_ext

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_rows(x, multiplier):
            return (x[:, :128].to(torch.uint8),
                    torch.zeros(x.shape[0] * 16, dtype=torch.uint8),
                    torch.ones(x.shape[0], dtype=torch.float32))

        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(aq, *args, **kwargs):
            # The two-phase write must already name the symbol when the launch
            # is entered, so a raise mid-launch stays attributable.
            record = read_route(layer)
            assert record["state"] == "error"
            assert record["symbol"] == "cb_fused_fp4_prefill_mm_scaled"
            return torch.zeros(aq.shape[0], 8, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    method._try_fused_fp4(
        layer, torch.zeros(32, 256, dtype=torch.bfloat16), 8, 256, 32,
        rowwise=True)

    record = read_route(layer)
    assert set(record) == set(ROUTE_FIELDS)
    assert record["kind"] == "dense"
    assert record["state"] == "served" and record["reason"] is None
    assert record["policy"] == "rowwise"
    assert record["contract"] == "nvfp4_rowwise"
    assert record["contract"] in ROUTE_CONTRACTS
    assert record["symbol"] == "cb_fused_fp4_prefill_mm_scaled"
    assert record["shape"] == "M32:N8:K256"
    assert record["tile_m"] in (128, 256)
    assert record["tile_candidate_ctas"] > 0
    assert record["tile_compiled"] == "256,128"


@pytest.mark.parametrize("mode,flags,contract", [
    ("rowwise", {"rowwise": True}, "nvfp4_rowwise"),
    ("static_lsq", {"static_lsq": True}, "nvfp4_static_lsq"),
])
def test_dense_fused_fp4_records_the_exact_fallback_reason(
    monkeypatch, mode, flags, contract,
):
    """A declined call must say WHY, in the vocabulary the report validates."""
    from gridbook.nvfp4_activation_contract import read_route

    method = _rowwise_dense_method()
    layer = types.SimpleNamespace(
        _cb_N=8, _cb_K=256, _cb_fused_fp4_mode=mode,
        _cb_fp4_input_global_scale_f32=3.0,
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(8),
        _cb_qw_padded=torch.ones(8, 73, dtype=torch.uint8),
    )
    from gridbook import cuda_ext

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext",
                        lambda: types.SimpleNamespace())
    assert method._try_fused_fp4(
        layer, torch.zeros(32, 256, dtype=torch.float32), 8, 256, 32,
        **flags) is None

    record = read_route(layer)
    assert record["state"] == "fallback"
    assert record["symbol"] == ""
    assert record["policy"] == mode
    assert record["contract"] == contract
    assert "torch.float32" in record["reason"]
    assert "half precision" in record["reason"]


def test_the_eligibility_gate_declining_is_also_recorded(monkeypatch):
    from gridbook.nvfp4_activation_contract import read_route

    method = _rowwise_dense_method()
    method._fused_fp4_ok = (
        lambda layer, K, *, rowwise=False, static_lsq=False: False)
    layer = types.SimpleNamespace(_cb_N=8, _cb_K=256,
                                  _cb_fused_fp4_mode="rowwise")
    assert method._try_fused_fp4(
        layer, torch.zeros(32, 256, dtype=torch.bfloat16), 8, 256, 32,
        rowwise=True) is None
    record = read_route(layer)
    assert record["state"] == "fallback"
    assert record["reason"] == "fused fp4 dense eligibility gate declined"


# --- dense dispatch truth table (fp4v2 mid-M, sm12x, precedence) ------------


def _quality_dense_method():
    """A dense FP4-v2 method with no fused-NVFP4 mode selected."""
    method = PrismaQuantCBLinearMethod.__new__(PrismaQuantCBLinearMethod)
    method.k = 16
    method.n_sub = 2
    method.type_size = 73
    method.is_v2 = True
    method.is_fp4 = True
    method.prefix = "model.layers.0.mlp.gate_up_proj"
    method._sub_table = [1.0] * 16
    method._require_fp4_v2_product = lambda *_a: None
    method._expand_fp4_quality_weight = (
        lambda layer, N, K: torch.zeros(N, K, dtype=torch.bfloat16))
    return method


def _quality_dense_layer(**overrides):
    layer = types.SimpleNamespace(
        _cb_N=2048, _cb_K=4096, _cb_fused_fp4_mode="",
        _cb_flat=torch.zeros(1, dtype=torch.bfloat16),
        _cb_fp4v2_midm=None, _cb_bf16_sm120=None,
        _cb_qw_padded=torch.ones(2048, 73, dtype=torch.uint8),
        _cb_compose=torch.zeros(4096), _cb_row_offset=torch.zeros(2048),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _dense_route(monkeypatch, method, layer, M=64):
    """Run one dense prefill with every native leaf replaced by a marker."""
    from gridbook import linear as cb_linear

    taken = []
    monkeypatch.setattr(cb_linear, "fp4_act_qdq_or_codec", lambda x: x)
    monkeypatch.setattr(
        cb_linear, "fp4v2_midm_fused_mm",
        lambda *a, **k: taken.append("fp4v2_midm")
        or torch.zeros(M, layer._cb_N, dtype=torch.bfloat16))
    monkeypatch.setattr(
        cb_linear, "bf16_sm120_dense_mm",
        lambda *a, **k: taken.append("sm120")
        or torch.zeros(M, layer._cb_N, dtype=torch.bfloat16))
    monkeypatch.setattr(
        cb_linear, "cb_bf16_grouped_mm",
        lambda *a, **k: taken.append("bridge")
        or torch.zeros(M, layer._cb_N, dtype=torch.bfloat16))
    method._apply_inline(
        layer, torch.zeros(M, layer._cb_K, dtype=torch.bfloat16))
    return taken


def _midm_ext(*, max_m=128, kbits=(12, 16, 20, 24)):
    return types.SimpleNamespace(
        cb_fused_fp4v2_max_m=lambda: max_m,
        cb_fused_fp4v2_kbits=lambda: list(kbits))


def test_dense_default_takes_the_expand_plus_bridge_route(monkeypatch):
    method, layer = _quality_dense_method(), _quality_dense_layer()
    assert _dense_route(monkeypatch, method, layer) == ["bridge"]


def test_dense_fp4v2_midm_lane_serves_when_resolved_and_eligible(monkeypatch):
    """Zero coverage existed for this call-dispatch decision."""
    method = _quality_dense_method()
    layer = _quality_dense_layer(
        _cb_fp4v2_midm=_midm_ext(),
        _cb_flat=torch.zeros(4 * (1 << 8) + 4 * (1 << 8),
                             dtype=torch.bfloat16))
    assert _dense_route(monkeypatch, method, layer) == ["fp4v2_midm"]


@pytest.mark.parametrize("M,why", [
    (8, "at or below the decode-GEMV boundary"),
    (256, "above the HARD mid-M ceiling"),
])
def test_dense_fp4v2_midm_falls_through_outside_the_mid_m_band(
    monkeypatch, M, why,
):
    """The M band is the ONE documented call-time fall-through.

    Every other condition became a load-time gate; this one is a property of
    the REQUEST and is what makes this a mid-M lane.
    """
    del why
    method = _quality_dense_method()
    layer = _quality_dense_layer(
        _cb_fp4v2_midm=_midm_ext(),
        _cb_flat=torch.zeros(4 * (1 << 8) + 4 * (1 << 8),
                             dtype=torch.bfloat16))
    method._require_fp4_cuda_gemv = lambda *_a: None
    from gridbook import linear as cb_linear

    monkeypatch.setattr(
        cb_linear, "cb_gemv_fp4_v2",
        lambda *a, **k: torch.zeros(M, layer._cb_N, dtype=torch.bfloat16))
    taken = _dense_route(monkeypatch, method, layer, M=M)
    assert "fp4v2_midm" not in taken


def test_dense_sm120_lane_serves_the_e1_gemm_when_resolved(monkeypatch):
    """Dense E=1 through the sm12x lane had no dispatch coverage."""
    method = _quality_dense_method()
    layer = _quality_dense_layer(_cb_bf16_sm120=object())
    assert _dense_route(monkeypatch, method, layer) == ["sm120"]


@pytest.mark.parametrize("mode", ["1", "rowwise", "static_lsq"])
def test_dense_precedence_fused_fp4_outranks_midm_and_sm120(monkeypatch, mode):
    """PIN THE DOCUMENTED CHAIN: FUSED_FP4 > MIDM > SM120 > bridge.

    The rule is what each flag CHANGES: PRISMAQUANT_CB_FUSED_FP4 changes the
    served ACTIVATION CONTRACT, so it outranks the two lanes below it, which
    only move the GEMM schedule behind the contract the artifact declares.
    Every losing lane is still resolved and attested at load.
    """
    method = _quality_dense_method()
    method._fused_fp4_ok = (
        lambda layer, K, *, rowwise=False, static_lsq=False: True)
    layer = _quality_dense_layer(
        _cb_fused_fp4_mode=mode,
        _cb_fp4v2_midm=_midm_ext(), _cb_bf16_sm120=object(),
        _cb_flat=torch.zeros(4 * (1 << 8) + 4 * (1 << 8),
                             dtype=torch.bfloat16),
        _cb_fp4_input_global_scale=torch.tensor([3.0]),
        _cb_fp4_input_global_scale_f32=3.0,
        _cb_fp4_lut=torch.ones(1),
        _cb_fp4_compose_u8=torch.ones(1, dtype=torch.uint8),
        _cb_fp4_ones=torch.ones(2048))
    from gridbook import cuda_ext, linear as cb_linear

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_rows(x, multiplier):
            return (x[:, :128].to(torch.uint8),
                    torch.zeros(x.shape[0] * 16, dtype=torch.uint8),
                    torch.ones(x.shape[0], dtype=torch.float32))

        cb_nvfp4_quantize_static_lsq = cb_nvfp4_quantize_rows

        @staticmethod
        def cb_fused_fp4_prefill_mm_scaled(aq, *a, **k):
            return torch.zeros(aq.shape[0], 2048, dtype=torch.bfloat16)

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: Ext())
    monkeypatch.setattr(cb_linear, "native_fp4_quant",
                        lambda x, gs: (x[:, :128].to(torch.uint8),
                                       torch.zeros(x.shape[0], 1)))
    monkeypatch.setattr(
        cb_linear, "_nvfp4_reciprocal_vector",
        lambda layer, **k: torch.ones(64, dtype=torch.float32))
    assert _dense_route(monkeypatch, method, layer) == []


def test_dense_precedence_midm_outranks_sm120(monkeypatch):
    """With both schedule lanes resolved, the fused mid-M lane wins."""
    method = _quality_dense_method()
    layer = _quality_dense_layer(
        _cb_fp4v2_midm=_midm_ext(), _cb_bf16_sm120=object(),
        _cb_flat=torch.zeros(4 * (1 << 8) + 4 * (1 << 8),
                             dtype=torch.bfloat16))
    assert _dense_route(monkeypatch, method, layer) == ["fp4v2_midm"]
