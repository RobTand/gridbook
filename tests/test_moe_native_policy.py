"""CPU policy gates for the native-only routed-expert serving contract.

The FP4 activation modes are native CUTLASS lanes, not a resurrection of the
removed stock/loop/Triton prefill selectors.  Their process-stable selector,
artifact attestation, and stage-specific activation operands remain part of
the serving contract and are covered here alongside the no-fallback policy.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
from gridbook.cuda_ext import NativeKernelUnavailableError  # noqa: E402


def _install_vllm_stubs():
    """Install only the native MoE class surface imported by Gridbook."""

    def module(name):
        value = types.ModuleType(name)
        sys.modules[name] = value
        return value

    module("vllm")
    module("vllm.model_executor")
    utils = module("vllm.model_executor.utils")
    utils.set_weight_attrs = lambda param, attrs: [
        setattr(param, name, value) for name, value in attrs.items()
    ]
    module("vllm.model_executor.layers")
    linear = module("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    fused = module("vllm.model_executor.layers.fused_moe")
    fused.RoutedExperts = type("RoutedExperts", (), {})
    config = module("vllm.model_executor.layers.fused_moe.config")
    config.FusedMoEConfig = type("FusedMoEConfig", (), {})
    config.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
    base = module(
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
    )
    base.FusedMoEMethodBase = type(
        "FusedMoEMethodBase",
        (),
        {"__init__": lambda self, moe_config=None: None},
    )
    parameter = module("vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **_kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **_kwargs):
            del data

    parameter.ModelWeightParameter = StubParameter
    parameter.ChannelQuantScaleParameter = StubParameter
    parameter.PerTensorScaleParameter = StubParameter


@pytest.fixture(scope="module", autouse=True)
def _runtime_modules(isolated_gridbook_runtime_imports):
    """Run CPU policy gates with a private stubbed vLLM graph.

    Importing a real vLLM inside the isolation fixture registers process-global
    Torch opaque types that cannot be unregistered when ``sys.modules`` is
    restored. A later test file would then fail while re-importing vLLM. These
    tests need only the class surface, so never initialize the real runtime.
    """
    del isolated_gridbook_runtime_imports
    _install_vllm_stubs()

    from gridbook import moe as moe_module

    globals()["moe"] = moe_module
    yield


def _method(*, is_fp4=True, is_v2=True, n_sub=2, k=15):
    method = moe.PrismaQuantCBMoEMethod.__new__(moe.PrismaQuantCBMoEMethod)
    method.prefix = "model.layers.0.mlp.experts"
    method.scheme = {
        "grid": "fp4" if is_fp4 else "fp8",
        "k": k,
        "n_sub": n_sub,
        "type_size": 4 * k + 9 if is_fp4 and is_v2 else 4 * k,
    }
    method.is_fp4 = is_fp4
    method.has_static_fp4_activation = is_fp4
    method.is_v2 = is_v2
    method.n_sub = n_sub
    method.k = k
    method.type_size = 4 * k + 9 if is_fp4 and is_v2 else 4 * k
    method._sub_table = None
    return method


def _layer(*, experts=4, hidden=512, inter=256):
    return types.SimpleNamespace(
        _cb_E=experts,
        _cb_hidden=hidden,
        _cb_inter=inter,
        _cb_fp4_input_global_scale_w13=torch.tensor([2.5]),
        _cb_fp4_input_global_scale_w2=torch.tensor([1.25]),
        activation=types.SimpleNamespace(value="silu"),
    )


@pytest.fixture(autouse=True)
def _reset_fused_fp4_mode():
    moe._FUSED_FP4_MOE_STATE.clear()
    yield
    moe._FUSED_FP4_MOE_STATE.clear()


def test_obsolete_prefill_selectors_are_not_part_of_the_runtime():
    from gridbook import cuda_ext, ops

    assert not hasattr(moe, "_requested_prefill_mode")
    assert hasattr(moe, "_requested_fused_fp4_moe_mode")
    for name in (
        "_apply_prefill_stock",
        "_apply_prefill_loop",
        "_apply_prefill_batched",
        "_apply_prefill_auto",
        "_apply_prefill_l2_pipeline",
        "_grouped_gemm",
    ):
        assert not hasattr(moe.PrismaQuantCBMoEMethod, name)
    assert not hasattr(cuda_ext, "get_persistent_ext")
    assert not hasattr(ops, "cb_prefill_persistent_tc")


@pytest.mark.parametrize(
    "value",
    [
        "", "1", "128", "256",
        "static_lsq", "static_lsq128", "static_lsq256",
        "rowwise", "rowwise128", "rowwise256",
    ],
)
def test_native_fp4_activation_selector_accepts_only_named_modes(
    monkeypatch, value
):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", value)
    assert moe._requested_fused_fp4_moe_mode() == value


@pytest.mark.parametrize(
    "value", ["yes", "MIDM", "129", "rowwise_128", "static_lsq_128"]
)
def test_native_fp4_activation_selector_rejects_unknown_modes(
    monkeypatch, value
):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", value)
    with pytest.raises(ValueError, match="invalid PRISMAQUANT_CB_FUSED_FP4_MOE"):
        moe._requested_fused_fp4_moe_mode()


def test_native_fp4_activation_selector_cannot_change_mid_process(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", "128")
    assert moe._requested_fused_fp4_moe_mode() == "128"
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", "256")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        moe._requested_fused_fp4_moe_mode()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", (False, False, 128)),
        ("256", (False, False, 256)),
        ("static_lsq", (False, True, 128)),
        ("static_lsq256", (False, True, 256)),
        ("rowwise", (True, False, 128)),
        ("rowwise256", (True, False, 256)),
    ],
)
def test_native_fp4_activation_selector_plumbs_family_and_tile(
    monkeypatch, value, expected
):
    method = _method(k=16)
    layer = _layer()
    observed = []

    def fused(_layer, _x, _weights, _ids, _act, **kwargs):
        observed.append((
            kwargs["rowwise"], kwargs["static_lsq"], kwargs["tile_m"]
        ))
        return "native-fp4"

    method._apply_prefill_grouped_fused_fp4 = fused
    method._apply_prefill_native_bf16 = (
        lambda *_args, **_kwargs: pytest.fail(
            "an eligible explicit native FP4 mode must reach its CUTLASS path"
        )
    )
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", value)
    x = torch.zeros(32, layer._cb_hidden)
    ids = torch.zeros(32, 2, dtype=torch.int32)
    weights = torch.ones(32, 2)
    assert method._apply_inline(layer, x, weights, ids) == "native-fp4"
    assert observed == [expected]


def test_unset_fp4_mode_uses_owned_native_bf16_bridge(monkeypatch):
    method = _method(k=16)
    layer = _layer()
    calls = []
    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4_MOE", raising=False)
    method._apply_prefill_grouped_fused_fp4 = (
        lambda *_args, **_kwargs: pytest.fail(
            "unset FP4 mode must not enter the W4A4 path"
        )
    )
    method._apply_prefill_native_bf16 = (
        lambda *_args, **_kwargs: calls.append("native-bf16") or "native-bf16"
    )
    x = torch.zeros(32, layer._cb_hidden)
    ids = torch.zeros(32, 2, dtype=torch.int32)
    weights = torch.ones(32, 2)
    assert method._apply_inline(layer, x, weights, ids) == "native-bf16"
    assert calls == ["native-bf16"]


def test_requested_native_fp4_mode_fails_closed_if_kernel_becomes_unavailable(
    monkeypatch,
):
    method = _method(k=16)
    layer = _layer()
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", "1")
    method._apply_prefill_grouped_fused_fp4 = lambda *_args, **_kwargs: None
    method._apply_prefill_native_bf16 = (
        lambda *_args, **_kwargs: pytest.fail(
            "an explicit activation contract must not silently change"
        )
    )
    x = torch.zeros(32, layer._cb_hidden)
    ids = torch.zeros(32, 2, dtype=torch.int32)
    weights = torch.ones(32, 2)
    with pytest.raises(NativeKernelUnavailableError, match="became unavailable"):
        method._apply_inline(layer, x, weights, ids)


@pytest.mark.parametrize(
    "k,n_sub,expected",
    [(24, 2, True), (20, 1, True), (21, 1, False)],
    ids=["product-k24-16k", "signed-s20-16k", "signed-s21-over-16k"],
)
def test_native_fp4_moe_eligibility_enforces_lut_smem_limit(
    monkeypatch, k, n_sub, expected
):
    from gridbook import cuda_ext

    method = _method(k=k, n_sub=n_sub)
    layer = _layer()

    class FusedExt:
        cb_fused_fp4_moe_grouped = object()

    monkeypatch.setattr(moe, "_native_scaled_fp4_quant_available", lambda: True)
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: FusedExt())
    assert method._gf4_ok(layer) is expected


def test_legacy_fp4_artifact_is_eligible_only_for_rowwise_native_quant(
    monkeypatch,
):
    from gridbook import cuda_ext

    method = _method(k=16)
    method.has_static_fp4_activation = False
    layer = _layer()
    del layer._cb_fp4_input_global_scale_w13
    del layer._cb_fp4_input_global_scale_w2

    class FusedExt:
        cb_fused_fp4_moe_grouped = object()
        cb_nvfp4_quantize_rows = object()

    monkeypatch.setattr(moe, "_native_scaled_fp4_quant_available", lambda: True)
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: FusedExt())
    assert method._gf4_ok(layer) is False
    assert method._gf4_ok(layer, rowwise=True) is True
    assert method._gf4_ok(layer, static_lsq=True) is False


def test_native_fp4_moe_quantizer_families_are_symbol_isolated(monkeypatch):
    from gridbook import cuda_ext

    method = _method(k=16)
    layer = _layer()

    class StaticOnlyExt:
        cb_fused_fp4_moe_grouped = object()

    monkeypatch.setattr(moe, "_native_scaled_fp4_quant_available", lambda: True)
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: StaticOnlyExt())
    assert method._gf4_ok(layer) is True
    assert method._gf4_ok(layer, rowwise=True) is False
    assert method._gf4_ok(layer, static_lsq=True) is False


def test_static_lsq_moe_requires_attested_host_scales_and_native_symbol(
    monkeypatch,
):
    from gridbook import cuda_ext

    method = _method(k=16)
    layer = _layer()
    layer._cb_fp4_input_global_scale_w13_f32 = 2.5
    layer._cb_fp4_input_global_scale_w2_f32 = 1.25

    class LsqExt:
        cb_fused_fp4_moe_grouped = object()
        cb_nvfp4_quantize_static_lsq = object()

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: LsqExt())
    assert method._gf4_ok(layer, static_lsq=True) is True
    assert method._gf4_ok(layer, rowwise=True, static_lsq=True) is False

    unstamped = _layer()
    assert method._gf4_ok(unstamped, static_lsq=True) is False

    legacy = _method(k=16)
    legacy.has_static_fp4_activation = False
    legacy_layer = _layer()
    legacy_layer._cb_fp4_input_global_scale_w13_f32 = 2.5
    legacy_layer._cb_fp4_input_global_scale_w2_f32 = 1.25
    assert legacy._gf4_ok(legacy_layer, static_lsq=True) is False
    assert legacy_layer._cb_gf4_static_lsq_ok_reason == (
        "artifact has no attested static activation contract"
    )

    no_symbol = _layer()
    no_symbol._cb_fp4_input_global_scale_w13_f32 = 2.5
    no_symbol._cb_fp4_input_global_scale_w2_f32 = 1.25

    class GroupedOnlyExt:
        cb_fused_fp4_moe_grouped = object()

    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: GroupedOnlyExt())
    assert method._gf4_ok(no_symbol, static_lsq=True) is False


def test_static_native_quant_uses_distinct_attested_stage_scales(monkeypatch):
    method = _method(k=16)
    layer = _layer()
    observed = []

    def quantize(x, scale):
        observed.append(scale.item())
        return x.to(torch.uint8), torch.zeros(x.shape[0], 1)

    monkeypatch.setattr(moe, "_native_scaled_fp4_quant", quantize)
    rows = torch.ones(3, 16)
    _aq1, _sf1, reciprocal1 = method._fp4_quant(layer, rows, "w13")
    _aq2, _sf2, reciprocal2 = method._fp4_quant(layer, rows, "w2")
    assert observed == [2.5, 1.25]
    assert torch.equal(reciprocal1, torch.full((3,), 0.4))
    assert torch.equal(reciprocal2, torch.full((3,), 0.8))


def test_rowwise_moe_quant_uses_returned_per_row_native_operands(monkeypatch):
    from gridbook import nvfp4_activation_contract as activation_contract

    monkeypatch.setattr(activation_contract, "ROWWISE_RANGE_MULTIPLIER", 256.0)
    monkeypatch.setattr(
        moe,
        "_native_scaled_fp4_quant",
        lambda *_args: pytest.fail(
            "rowwise quantization must not read static stage scalars"
        ),
    )
    method = _method(k=16)
    method.has_static_fp4_activation = False
    calls = []

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_rows(x, multiplier):
            packed = x[:, :128].to(torch.float32).round().to(torch.uint8)
            sfa = x[:, :16].to(torch.float32).round().to(torch.uint8)
            scales = x.float().abs().amax(dim=1) / (multiplier * 6.0)
            calls.append((packed.clone(), sfa.clone(), scales.clone(), multiplier))
            return packed, sfa, scales

    target = torch.arange(256, dtype=torch.bfloat16).reshape(1, 256)
    one = method._fp4_quant(
        types.SimpleNamespace(), target, "w13", rowwise=True, fext=Ext()
    )
    batch = method._fp4_quant(
        types.SimpleNamespace(),
        torch.cat((target, torch.full_like(target, 1.0e4))),
        "w2",
        rowwise=True,
        fext=Ext(),
    )
    assert [call[3] for call in calls] == [256.0, 256.0]
    assert torch.equal(one[0], calls[0][0])
    assert torch.equal(one[1], calls[0][1])
    assert torch.equal(one[2], calls[0][2])
    assert torch.equal(one[0][0], batch[0][0])
    assert torch.equal(one[1][0], batch[1][0])
    assert torch.equal(one[2][0], batch[2][0])


def test_static_lsq_moe_quant_uses_attested_stage_host_scales(monkeypatch):
    method = _method(k=16)
    layer = types.SimpleNamespace(
        _cb_fp4_input_global_scale_w13_f32=2.5,
        _cb_fp4_input_global_scale_w2_f32=1.25,
    )
    calls = []

    class Ext:
        @staticmethod
        def cb_nvfp4_quantize_static_lsq(x, scale):
            calls.append((x, scale))
            return (
                x,
                torch.zeros(1, dtype=torch.uint8),
                torch.full((x.shape[0],), scale, dtype=torch.float32),
            )

    monkeypatch.setattr(
        moe,
        "_native_scaled_fp4_quant",
        lambda *_args: pytest.fail(
            "static-LSQ must use Gridbook's owned LSQ quantizer"
        ),
    )
    rows = torch.arange(512, dtype=torch.bfloat16).reshape(2, 256)
    w13 = method._fp4_quant(
        layer, rows, "w13", static_lsq=True, fext=Ext()
    )
    w2 = method._fp4_quant(
        layer,
        torch.flip(rows, dims=(1,)).contiguous(),
        "w2",
        static_lsq=True,
        fext=Ext(),
    )
    assert [scale for _x, scale in calls] == [2.5, 1.25]
    assert calls[0][0].is_contiguous() and calls[1][0].is_contiguous()
    assert torch.equal(w13[2], torch.full((2,), 2.5))
    assert torch.equal(w2[2], torch.full((2,), 1.25))


def test_moe_post_load_caches_distinct_attested_stage_scales(monkeypatch):
    from gridbook import cuda_ext, ops

    method = _method(k=16)
    method.scheme["codebook_ref"] = "cb"

    class QuantConfig:
        @staticmethod
        def get_codebooks():
            return {"cb": torch.zeros(1)}

        @staticmethod
        def moe_activation_stage_targets(prefix):
            assert prefix == method.prefix
            return {"w13": ["gate_up"], "w2": ["down"]}

        @staticmethod
        def activation_scales_for_targets(targets):
            return [2.5 if targets == ["gate_up"] else 1.25]

    method.quant_config = QuantConfig()
    method._cuda_moe_ok = lambda _layer: True
    layer = _layer(experts=1, hidden=256, inter=256)
    layer.w13_input_global_scale = torch.nn.Parameter(torch.tensor([2.5]))
    layer.w2_input_global_scale = torch.nn.Parameter(torch.tensor([1.25]))
    layer.w13_cb_qweight = torch.nn.Parameter(
        torch.empty(1, 512, 73, dtype=torch.uint8), requires_grad=False
    )
    layer.w2_cb_qweight = torch.nn.Parameter(
        torch.empty(1, 256, 73, dtype=torch.uint8), requires_grad=False
    )

    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4_MOE", raising=False)
    monkeypatch.setattr(moe, "assert_cb_experts_filled", lambda *_args: None)
    monkeypatch.setattr(
        moe.codec, "build_flat_codebook", lambda *_args: torch.zeros(1)
    )
    monkeypatch.setattr(
        moe.codec, "build_compose_table", lambda *_args: torch.zeros(1)
    )
    monkeypatch.setattr(moe, "cb_gemv_choice", lambda *_args: (True, "test"))
    monkeypatch.setattr(moe, "require_native_moe_activation", lambda *_args: "silu")
    monkeypatch.setattr(cuda_ext, "require_ext", lambda *_args: None)
    monkeypatch.setattr(cuda_ext, "require_ext_v2", lambda *_args: None)
    monkeypatch.setattr(
        cuda_ext, "require_fp4_v2_expander",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cuda_ext, "require_bf16_grouped_ext", lambda *_args: None)
    monkeypatch.setattr(ops, "register_cb_layer", lambda *_args: 7)

    method.process_weights_after_loading(layer)
    assert layer._cb_fp4_input_global_scale_w13.item() == 2.5
    assert layer._cb_fp4_input_global_scale_w2.item() == 1.25
    assert layer._cb_fp4_input_global_scale_w13_f32 == 2.5
    assert layer._cb_fp4_input_global_scale_w2_f32 == 1.25
    assert layer._cb_fused_fp4_moe_mode == ""
    assert layer._cb_layer_id == 7


def test_moe_weight_creation_and_layer_loader_own_static_stage_scalars():
    method = _method(k=16)
    layer = torch.nn.Module()

    def original_load(weights):
        for name, _value in weights:
            yield "delegated:" + name

    layer.load_weights = original_load
    method.create_weights(
        layer,
        num_experts=4,
        hidden_size=512,
        intermediate_size_per_partition=256,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    assert torch.isnan(layer.w13_input_global_scale).all()
    assert torch.isnan(layer.w2_input_global_scale).all()
    loaded = list(layer.load_weights(iter([
        ("gate_up_proj.input_global_scale", torch.tensor([2.5])),
        ("down_proj.input_global_scale", torch.tensor([1.25])),
    ])))
    assert loaded == ["w13_input_global_scale", "w2_input_global_scale"]
    assert layer.w13_input_global_scale.item() == 2.5
    assert layer.w2_input_global_scale.item() == 1.25


def test_dense_persistent_experiment_is_not_serving_reachable():
    from gridbook.linear import PrismaQuantCBLinearMethod

    assert not hasattr(PrismaQuantCBLinearMethod, "_ptc_ok")


def test_native_bf16_chunk_is_bounded_by_the_larger_w13_tile(monkeypatch):
    method = _method()
    layer = types.SimpleNamespace(_cb_E=256, _cb_hidden=4096,
                                  _cb_inter=2048)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", str(1 << 30))
    # One DSV4 w13 expert is 4096 x 4096 BF16 = 32 MiB.
    assert method._native_bf16_chunk(layer) == 32


@pytest.mark.parametrize("name", [
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK",
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES",
])
@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_native_bf16_chunk_rejects_invalid_overrides(monkeypatch, name, value):
    method = _method()
    layer = types.SimpleNamespace(_cb_E=8, _cb_hidden=256, _cb_inter=256)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="must be positive"):
        method._native_bf16_chunk(layer)


def test_signed_fp4_experts_fail_before_any_expansion_kernel():
    method = _method(n_sub=1)
    layer = types.SimpleNamespace(
        _cb_hidden=256,
        _cb_inter=256,
        w13_cb_qweight=torch.zeros(1, 512, method.type_size,
                                   dtype=torch.uint8),
        _cb_flat=torch.zeros(1, dtype=torch.bfloat16),
        _cb_compose=torch.zeros(4096, dtype=torch.float32),
    )
    with pytest.raises(NativeKernelUnavailableError,
                       match="FP4-CB-v2 product experts"):
        method._expand_native_bf16_slice(layer, "w13", 0, 1)


def test_unsupported_activation_fails_before_external_dispatch(monkeypatch):
    from gridbook import native_cutlass

    monkeypatch.setattr(native_cutlass, "_import_native_ops",
                        lambda _context: None)
    with pytest.raises(NativeKernelUnavailableError,
                       match="no direct native Gridbook operator"):
        native_cutlass.require_native_moe_activation(
            "swiglustep", "test routed activation")
