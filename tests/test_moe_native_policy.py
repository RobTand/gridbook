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


@pytest.mark.parametrize("use_v2", [False, True])
def test_routed_fp8_decode_routes_gate_up_and_down_through_load_fixed_choice(
        monkeypatch, use_v2):
    """Exercise the real per-role call chain, not just selector arithmetic.

    Role identity is part of the assertion: choosing the right symbol while
    accidentally reusing one role's LUT or packed rows would still return
    plausible tensors and silently corrupt the model.
    """
    from gridbook import ops

    method = _method(is_fp4=False, is_v2=False, n_sub=4, k=28)
    layer = types.SimpleNamespace(
        _cb_role_split=True,
        _cb_use_fp8_v2_w13=use_v2,
        _cb_use_fp8_v2_w2=use_v2,
        _cb_w13_gate_qweight=torch.full((2, 3, 1), 11,
                                        dtype=torch.uint8),
        _cb_w13_up_qweight=torch.full((2, 3, 1), 22,
                                      dtype=torch.uint8),
        _cb_w13_gate_scale=torch.full((2, 3), 1.25),
        _cb_w13_up_scale=torch.full((2, 3), 2.5),
        w2_cb_qweight=torch.nn.Parameter(
            torch.full((2, 4, 1), 33, dtype=torch.uint8),
            requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(
            torch.full((2, 4), 3.75), requires_grad=False),
        _cb_flat_fp8_by_role={
            "gate": torch.full((8,), 41, dtype=torch.uint8),
            "up": torch.full((8,), 42, dtype=torch.uint8),
            "down": torch.full((8,), 43, dtype=torch.uint8),
        },
    )
    calls = []

    monkeypatch.setattr(ops, "fp8_act_qdq",
                        lambda value: value.to(torch.bfloat16))

    def gemv(route, *args):
        calls.append((route, args))
        return torch.zeros((args[4].numel(), args[1].shape[1]),
                           dtype=torch.bfloat16)

    monkeypatch.setattr(
        ops, "cb_moe_gemv_fp8",
        lambda *args: gemv("inherited", *args))
    monkeypatch.setattr(
        ops, "cb_moe_gemv_fp8_v2",
        lambda *args: gemv("whole-row-v2", *args))

    def activate(_act, output, gate_up):
        assert gate_up.shape == (4, 6)
        output.zero_()

    def combine(y, pair_w, tok_start, tokens):
        assert y.shape == (4, 4)
        assert pair_w.shape == (4,)
        assert torch.equal(tok_start, torch.tensor([0, 2, 4],
                                                   dtype=torch.int32))
        return torch.zeros((tokens, y.shape[1]), dtype=y.dtype)

    monkeypatch.setattr(moe, "native_moe_activation", activate)
    monkeypatch.setattr(ops, "cb_moe_combine", combine)

    x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    topk_ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    result = method._apply_grouped_decode(
        layer, x, topk_weights, topk_ids, object())

    assert result.shape == (2, 4)
    expected_route = "whole-row-v2" if use_v2 else "inherited"
    assert [route for route, _ in calls] == [expected_route] * 3

    gate, up, down = (args for _, args in calls)
    for args, role in zip((gate, up, down), ("gate", "up", "down")):
        assert args[2] is layer._cb_flat_fp8_by_role[role]
        assert args[6:] == (28, 4, 112)
        assert torch.equal(args[4], torch.tensor([0, 1, 0, 1],
                                                 dtype=torch.int32))
    assert gate[1] is layer._cb_w13_gate_qweight
    assert gate[3] is layer._cb_w13_gate_scale
    assert up[1] is layer._cb_w13_up_qweight
    assert up[3] is layer._cb_w13_up_scale
    assert down[1].data_ptr() == layer.w2_cb_qweight.data_ptr()
    assert down[3].data_ptr() == layer.w2_weight_scale.data_ptr()
    assert torch.equal(gate[5], torch.tensor([0, 0, 1, 1],
                                             dtype=torch.int32))
    assert torch.equal(up[5], gate[5])
    assert torch.equal(down[5], torch.arange(4, dtype=torch.int32))


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


# --- OPT-IN sm12x-native BF16 bridge lane (audit §3 P1) --------------------


def test_bridge_keeps_the_sm80_schedule_when_the_lane_is_unresolved():
    """Flag off is the DEFAULT and must not reach the padded lane at all.

    The layer attribute is what ``process_weights_after_loading`` sets; with
    the flag unset it stays None and the bridge runs exactly the code it ran
    before the lane existed.
    """
    method = _method()
    layer = _layer()
    layer._cb_bf16_sm120 = None
    method._apply_prefill_native_bf16_sm120 = (
        lambda *_a, **_k: pytest.fail(
            "the padded sm12x lane must not run with the flag unset"))
    x = torch.zeros(4, layer._cb_hidden, dtype=torch.bfloat16)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    weights = torch.ones(4, 2)
    # The default arm proceeds into the exact-segment path and stops at the
    # first native op, which is the proof that it took that arm.
    with pytest.raises(Exception) as exc_info:
        method._apply_prefill_native_bf16(layer, x, weights, ids, "silu")
    assert "sm120" not in str(exc_info.value)


def test_bridge_takes_the_padded_lane_once_it_is_resolved():
    method = _method()
    layer = _layer()
    layer._cb_bf16_sm120 = types.SimpleNamespace(
        cb_bf16_grouped_sm120_tile_m=lambda: 128)
    seen = []
    method._apply_prefill_native_bf16_sm120 = (
        lambda *args, **kwargs: seen.append(args) or "sm120")
    x = torch.zeros(4, layer._cb_hidden, dtype=torch.bfloat16)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    weights = torch.ones(4, 2)
    assert method._apply_prefill_native_bf16(
        layer, x, weights, ids, "silu") == "sm120"
    assert len(seen) == 1


def test_padded_route_layout_is_shared_and_block_offsets_agree():
    """One helper builds the layout for all three grouped lanes.

    The block offsets the BF16 bridge slices its chunked launches with must
    name the same tiles ``expert_ids`` does, or a chunk would multiply rows by
    another chunk's weights.
    """
    tile_m, E = 128, 6
    torch.manual_seed(11)
    ids = torch.randint(0, E, (300, 2), dtype=torch.int32)
    w = torch.rand(300, 2)
    route = moe._padded_route(ids, w, E, tile_m, trim=True,
                              block_offsets=True)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=E)
    blocks = ((counts + tile_m - 1) // tile_m).tolist()

    assert route.block_offsets[0] == 0
    assert route.block_offsets[E] == sum(blocks)
    assert int(route.expert_ids.numel()) == sum(blocks)
    for e in range(E):
        b0, b1 = route.block_offsets[e], route.block_offsets[e + 1]
        assert b1 - b0 == blocks[e]
        assert route.expert_ids[b0:b1].tolist() == [e] * blocks[e]
    assert route.dest.numel() == sum(blocks) * tile_m
    # Padding rows point at the throwaway destination T.
    assert int(route.dest.max()) <= ids.shape[0]


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
@pytest.mark.parametrize("value", ["0", "-1", "not-an-int", "1.5", " "])
def test_native_bf16_chunk_rejects_invalid_overrides(monkeypatch, name, value):
    """A bad chunk knob raises, and the message names the flag.

    These gate the swizzle-group packed expert ORDER on the sm12x lane, so a
    silently ignored value would change the FP32 reduction order without
    saying so — which is why they are parsed strictly rather than coerced.
    """
    method = _method()
    layer = types.SimpleNamespace(_cb_E=8, _cb_hidden=256, _cb_inter=256)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    monkeypatch.setenv(name, value)
    if value.strip() == "":
        # An all-whitespace value is "unset", not a typo: the default applies.
        assert method._native_bf16_chunk(layer) == 8
        return
    with pytest.raises(ValueError, match=name):
        method._native_bf16_chunk(layer)


@pytest.mark.parametrize("name", [
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK",
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES",
])
def test_native_bf16_chunk_knobs_are_process_stable(monkeypatch, name):
    """Changing a chunk knob mid-process raises instead of taking effect.

    The chunk gates ``pack_expert_blocks`` on the sm12x lane (``chunk >= E``),
    so a value that changed between two forwards of one run would silently
    change the packed expert order — and therefore the FP32 reduction order —
    inside a single measurement. Both knobs were read from the environment on
    EVERY call until 2026-08-02.
    """
    method = _method()
    layer = types.SimpleNamespace(_cb_E=8, _cb_hidden=256, _cb_inter=256)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", raising=False)
    monkeypatch.setenv(name, "4" if name.endswith("CHUNK") else str(1 << 20))
    first = method._native_bf16_chunk(layer)

    monkeypatch.setenv(name, "2" if name.endswith("CHUNK") else str(1 << 30))
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        method._native_bf16_chunk(layer)
    # And the latched value is still what the first call resolved.
    monkeypatch.setenv(name, "4" if name.endswith("CHUNK") else str(1 << 20))
    assert method._native_bf16_chunk(layer) == first


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


# --- Routed FP4 lane PRECEDENCE (fused NVFP4 > persistent-B > sm12x) -------


def _reset_lane_latches():
    from gridbook import bf16_grouped_lane, moe_persistent_b_lane

    moe_persistent_b_lane._reset_for_tests()
    bf16_grouped_lane._reset_for_tests()
    moe._FUSED_FP4_MOE_STATE.clear()


@pytest.fixture
def lane_latches():
    _reset_lane_latches()
    yield
    _reset_lane_latches()


def _loadable_layer():
    layer = _layer(experts=1, hidden=256, inter=256)
    layer.w13_input_global_scale = torch.nn.Parameter(torch.tensor([2.5]))
    layer.w2_input_global_scale = torch.nn.Parameter(torch.tensor([1.25]))
    layer.w13_cb_qweight = torch.nn.Parameter(
        torch.empty(1, 512, 73, dtype=torch.uint8), requires_grad=False)
    layer.w2_cb_qweight = torch.nn.Parameter(
        torch.empty(1, 256, 73, dtype=torch.uint8), requires_grad=False)
    return layer


def _stub_load(monkeypatch, method):
    """Neutralize every native attestation so the LOAD's own logic is visible."""
    from gridbook import cuda_ext, ops

    class QuantConfig:
        @staticmethod
        def get_codebooks():
            return {"cb": torch.zeros(1)}

        @staticmethod
        def moe_activation_stage_targets(prefix):
            del prefix
            return {"w13": ["gate_up"], "w2": ["down"]}

        @staticmethod
        def activation_scales_for_targets(targets):
            return [2.5 if targets == ["gate_up"] else 1.25]

    method.scheme["codebook_ref"] = "cb"
    method.quant_config = QuantConfig()
    method._cuda_moe_ok = lambda _layer: True
    monkeypatch.setattr(moe, "assert_cb_experts_filled", lambda *_a: None)
    monkeypatch.setattr(moe.codec, "build_flat_codebook",
                        lambda *_a: torch.zeros(1))
    monkeypatch.setattr(moe.codec, "build_compose_table",
                        lambda *_a: torch.zeros(1))
    monkeypatch.setattr(moe, "cb_gemv_choice", lambda *_a: (True, "test"))
    monkeypatch.setattr(moe, "require_native_moe_activation",
                        lambda *_a: "silu")
    monkeypatch.setattr(cuda_ext, "require_ext", lambda *_a: None)
    monkeypatch.setattr(cuda_ext, "require_ext_v2", lambda *_a: None)
    monkeypatch.setattr(cuda_ext, "require_fp4_v2_expander",
                        lambda *_a, **_k: None)
    monkeypatch.setattr(cuda_ext, "require_bf16_grouped_ext", lambda *_a: None)
    monkeypatch.setattr(ops, "register_cb_layer", lambda *_a: 7)
    # The persistent-B lane is ATTESTED even when it will be overridden, so a
    # broken explicit selection still fails the load. Stand in for the module.
    ext = types.SimpleNamespace(cb_moe_persistent_b_configs=lambda: [[64, 64]])
    monkeypatch.setattr(moe, "persistent_b_require_lane",
                        lambda *_a, **_k: ext)
    return ext


def test_load_print_names_the_route_that_will_actually_serve(
    monkeypatch, capsys, lane_latches,
):
    """Both flags set: the log must name the fused mode, not persistent-B.

    Until 2026-08-02 this printed "-> persistent-B decode-in-mainloop" and
    then served the fused NVFP4 MoE kernel for every request. A dispatch log
    that names the wrong kernel is worse than none: it is the artifact an A/B
    is read from.
    """
    monkeypatch.setenv("PRISMAQUANT_CB_MOE_PERSISTENT_B", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_FUSED_FP4_MOE", "rowwise")
    method = _method(k=16)
    ext = _stub_load(monkeypatch, method)
    layer = _loadable_layer()
    method._gf4_ok = lambda *_a, **_k: True

    method.process_weights_after_loading(layer)

    out = capsys.readouterr().out
    assert "-> fused NVFP4 MoE 'rowwise'" in out
    assert "PRECEDENCE" in out
    assert "PRISMAQUANT_CB_MOE_PERSISTENT_B" in out
    assert "will NOT serve this layer" in out
    assert "-> persistent-B decode-in-mainloop" not in out
    # Attested regardless, so an unserveable explicit selection still fails
    # the LOAD rather than being silently ignored.
    assert layer._cb_moe_persistent_b is ext
    assert layer._cb_fused_fp4_moe_mode == "rowwise"


def test_load_print_is_unchanged_when_persistent_b_is_the_only_flag(
    monkeypatch, capsys, lane_latches,
):
    monkeypatch.setenv("PRISMAQUANT_CB_MOE_PERSISTENT_B", "1")
    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4_MOE", raising=False)
    method = _method(k=16)
    _stub_load(monkeypatch, method)
    layer = _loadable_layer()

    method.process_weights_after_loading(layer)

    out = capsys.readouterr().out
    assert "-> persistent-B decode-in-mainloop" in out
    assert "no expanded [E,N,K] transient" in out
    assert "PRECEDENCE" not in out


def test_dispatch_matches_the_precedence_the_load_announced(monkeypatch):
    """The coded order IS fused > persistent-B; pin it as deliberate."""
    Method = moe.PrismaQuantCBMoEMethod
    taken = []
    monkeypatch.setattr(
        Method, "_apply_prefill_grouped_fused_fp4",
        lambda self, *a, **k: taken.append("fused-fp4") or "fused")
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16",
        lambda self, *a, **k: taken.append("bridge-family") or "bridge")

    method = _method(k=16)
    method._cuda_moe_ok = lambda _layer: True
    layer = types.SimpleNamespace(
        _cb_fused_fp4_moe_mode="rowwise",
        _cb_moe_persistent_b=object(),
        _cb_bf16_sm120=object(),
        _cb_native_activation="silu",
        # `_apply_inline`'s getattr default is evaluated eagerly, so the
        # module attribute must exist even when the cached value is present.
        activation=types.SimpleNamespace(value="silu"))
    x = torch.zeros(64, 256, dtype=torch.bfloat16)
    ids = torch.zeros(64, 2, dtype=torch.int32)

    assert method._apply_inline(layer, x, torch.ones(64, 2), ids) == "fused"
    assert taken == ["fused-fp4"]


def test_persistent_b_serves_once_the_contract_flag_is_absent(monkeypatch):
    """The negative control for the row above: no fused mode, lane serves."""
    Method = moe.PrismaQuantCBMoEMethod
    taken = []
    monkeypatch.setattr(
        Method, "_apply_prefill_grouped_fused_fp4",
        lambda self, *a, **k: taken.append("fused-fp4") or "fused")
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16_persistent_b",
        lambda self, *a, **k: taken.append("persistent-b") or "pb")
    monkeypatch.setattr(
        Method, "_apply_prefill_native_bf16_sm120",
        lambda self, *a, **k: taken.append("sm120") or "sm120")

    method = _method(k=16)
    method._cuda_moe_ok = lambda _layer: True
    layer = types.SimpleNamespace(
        _cb_fused_fp4_moe_mode="",
        _cb_moe_persistent_b=object(),
        _cb_bf16_sm120=object(),
        _cb_native_activation="silu",
        # `_apply_inline`'s getattr default is evaluated eagerly, so the
        # module attribute must exist even when the cached value is present.
        activation=types.SimpleNamespace(value="silu"))
    x = torch.zeros(64, 256, dtype=torch.bfloat16)
    ids = torch.zeros(64, 2, dtype=torch.int32)

    assert method._apply_inline(layer, x, torch.ones(64, 2), ids) == "pb"
    assert taken == ["persistent-b"]


# --- K0.4 telemetry: no route may serve without recording itself -----------


def test_fused_fp4_moe_records_the_exact_fallback_reason():
    """The fused NVFP4 MoE lane had no telemetry at all.

    A served fused routed prefill was indistinguishable in a dispatch report
    from one that took the bridge — and those two run DIFFERENT activation
    contracts, which is precisely what the report exists to tell apart.
    """
    from gridbook.nvfp4_activation_contract import (ROUTE_CONTRACTS,
                                                    ROUTE_FIELDS, read_route)

    method = _method(k=16)
    layer = _layer(experts=4, hidden=256, inter=256)
    layer._cb_gf4_rowwise_ok_reason = "sentinel: gate declined, stated reason"
    method._gf4_ok = lambda *_a, **_k: False
    ids = torch.zeros(32, 2, dtype=torch.int32)
    x = torch.zeros(32, 256, dtype=torch.bfloat16)

    assert method._apply_prefill_grouped_fused_fp4(
        layer, x, torch.ones(32, 2), ids, "silu", rowwise=True) is None

    record = read_route(layer)
    assert set(record) == set(ROUTE_FIELDS)
    assert record["kind"] == "moe"
    assert record["state"] == "fallback" and record["symbol"] == ""
    assert record["reason"] == "sentinel: gate declined, stated reason"
    assert record["policy"] == "rowwise"
    assert record["contract"] == "nvfp4_rowwise"
    assert record["contract"] in ROUTE_CONTRACTS
    assert record["shape"] == "T32:P64:E4:H256:I256:topk2"
    assert record["tile_m"] == 128


def test_fused_fp4_moe_records_a_non_half_activation_as_the_reason():
    from gridbook.nvfp4_activation_contract import read_route

    method = _method(k=16)
    layer = _layer(experts=4, hidden=256, inter=256)
    method._gf4_ok = lambda *_a, **_k: True
    ids = torch.zeros(32, 2, dtype=torch.int32)

    assert method._apply_prefill_grouped_fused_fp4(
        layer, torch.zeros(32, 256, dtype=torch.float32), torch.ones(32, 2),
        ids, "silu", static_lsq=True, tile_m=256) is None

    record = read_route(layer)
    assert record["state"] == "fallback"
    assert record["contract"] == "nvfp4_static_lsq"
    assert record["tile_m"] == 256
    assert "torch.float32" in record["reason"]


def test_every_serving_route_writes_a_dispatch_record():
    """A new lane must not be able to ship untelemetered.

    K0.4's claim is that the requested policy, the kernel symbol, the tile, the
    shape, the activation contract and the fallback state are attestable on
    every route. That was true of the FP8 lanes only: the fused NVFP4 lanes
    (dense and routed) and all three quality routes had no record, so the
    claim was false for most of the dispatch surface.
    """
    import inspect

    for owner, names in (
        (moe.PrismaQuantCBMoEMethod, (
            "_apply_prefill_grouped_fused_fp4",
            "_apply_prefill_grouped_fused_v2",
            "_apply_prefill_native_bf16",
            "_apply_prefill_native_bf16_sm120",
            "_apply_prefill_native_bf16_persistent_b",
        )),
    ):
        for name in names:
            source = inspect.getsource(getattr(owner, name))
            assert "emit_route(" in source, (
                f"{owner.__name__}.{name} serves a request without recording "
                f"the route; K0.4's attestability claim covers every lane")

    from gridbook.linear import PrismaQuantCBLinearMethod

    for name in ("_try_fused_fp4", "_apply_inline"):
        source = inspect.getsource(getattr(PrismaQuantCBLinearMethod, name))
        assert "emit_route(" in source, (
            f"PrismaQuantCBLinearMethod.{name} serves a request without "
            f"recording the route")
