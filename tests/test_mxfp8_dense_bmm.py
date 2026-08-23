"""Direct-g32 W8A8 BMM ownership in the Gridbook MXFP8 dense lane."""
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
    parameter.GroupQuantScaleParameter = StubParameter
    parameter.ModelWeightParameter = StubParameter


class _IdentitySfExtension:
    @staticmethod
    def mxfp8_sf_offsets(rows, k, is_b):
        del is_b
        return torch.arange(rows * (k // 32), dtype=torch.int64)

    @staticmethod
    def mxfp8_sf_plane_numel(rows, k):
        return rows * (k // 32)

    @staticmethod
    def mxfp8_dense_mm(a_q, a_plane, b_q, b_plane):
        from gridbook.mxfp8 import mxfp8_reference_mm

        assert a_q.stride() == (a_q.shape[1], 1)
        assert b_q.stride() == (b_q.shape[1], 1)
        a_sf = a_plane.reshape(a_q.shape[0], a_q.shape[1] // 32)
        b_sf = b_plane.reshape(b_q.shape[0], b_q.shape[1] // 32)
        return mxfp8_reference_mm(a_q, a_sf, b_q, b_sf)


def test_direct_g32_bmm_keeps_group_scales_and_outputs_isolated(
    monkeypatch, isolated_gridbook_runtime_imports
):
    del isolated_gridbook_runtime_imports
    _install_vllm_method_stubs(monkeypatch)

    from gridbook import dsv4_woa
    from gridbook import mxfp8_dense_lane as lane
    from gridbook.mxfp8 import dequant_mxfp8, quantize_mxfp8

    extension = _IdentitySfExtension()
    lane._OFFSETS._cache.clear()
    monkeypatch.setattr(lane, "_require_lane_ext", lambda device=None: extension)
    monkeypatch.setattr(dsv4_woa, "install_dsv4_woa_adapter", lambda: None)

    groups, rows, k = 2, 128, 128
    layer = torch.nn.Module()
    layer.tp_size = 1
    method = lane.build_mxfp8_dense_method(lane.WIRE_MXFP8_G32)
    method.create_weights(
        layer, k, [groups * rows], k, groups * rows, torch.bfloat16)
    # Matches vLLM: these BMM attributes appear after ColumnParallelLinear's
    # constructor/create_weights, but before process_weights_after_loading.
    layer.is_bmm = True
    layer.bmm_batch_size = groups

    torch.manual_seed(13)
    layer.weight.data.copy_(
        (torch.randn(groups * rows, k) * 12).to(torch.float8_e4m3fn))
    scale_bytes = torch.cat([
        torch.full((rows, k // 32), 124, dtype=torch.uint8),
        torch.full((rows, k // 32), 130, dtype=torch.uint8),
    ])
    layer.weight_scale.data.view(torch.uint8).copy_(scale_bytes)
    weight_before = layer.weight.detach().clone()

    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "weight_scale")
    assert tuple(layer.weight_sf_planes.shape) == (groups, rows * (k // 32))
    assert getattr(layer, dsv4_woa.DSV4_MXFP8_BMM_ATTR) == \
        dsv4_woa.DSV4_MXFP8_BMM_ABI

    x = torch.randn(3, groups, k, dtype=torch.bfloat16)
    result = method.apply(layer, x)
    assert tuple(result.shape) == (3, groups, rows)

    a_q, a_sf = quantize_mxfp8(x)
    full_sf = scale_bytes.reshape(groups, rows, k // 32)
    weights = weight_before.reshape(groups, rows, k)
    expected = torch.stack([
        (dequant_mxfp8(a_q[:, group], a_sf[:, group]) @
         dequant_mxfp8(weights[group], full_sf[group]).t()).to(torch.bfloat16)
        for group in range(groups)
    ], dim=1)
    assert torch.equal(result, expected)

    # Serving-critical decode shape. A plain ``contiguous()`` on the group
    # slice incorrectly preserves stride ``(groups * K, 1)`` when M == 1.
    one = method.apply(layer, x[:1])
    assert torch.equal(one, expected[:1])


def test_bmm_apply_refuses_unfinalized_planes(
    monkeypatch, isolated_gridbook_runtime_imports
):
    del isolated_gridbook_runtime_imports
    _install_vllm_method_stubs(monkeypatch)
    from gridbook import mxfp8_dense_lane as lane

    extension = _IdentitySfExtension()
    monkeypatch.setattr(lane, "_require_lane_ext", lambda device=None: extension)
    method = lane.build_mxfp8_dense_method(lane.WIRE_MXFP8_G32)
    layer = types.SimpleNamespace(
        is_bmm=True,
        bmm_batch_size=2,
        weight=torch.zeros(256, 128, dtype=torch.float8_e4m3fn),
    )
    with pytest.raises(RuntimeError, match="scale planes were not finalized"):
        method.apply(layer, torch.zeros(1, 2, 128, dtype=torch.bfloat16))


def test_contract_tp_row_matches_the_bmm_refusal_at_tp2(
    monkeypatch, isolated_gridbook_runtime_imports
):
    """The contract's MXFP8 BMM row restates THIS refusal.

    The packaged contract says the ``mxfp8_e4m3_e8m0_g32`` BMM arm serves at
    tensor-parallel size 1 only; driving the lane's own finalization hook at
    TP=2 must raise before any scale plane is built. If the gate and the row
    ever disagree, this fails.
    """
    import json
    from importlib.resources import files as resource_files

    del isolated_gridbook_runtime_imports
    _install_vllm_method_stubs(monkeypatch)

    contract = json.loads(resource_files("gridbook").joinpath(
        "runtime_contract.json").read_text(encoding="utf-8"))
    row = next(unit for unit in contract["tensor_parallel"]["units"]
               if unit["unit"] == "mxfp8_e4m3_e8m0_g32")
    assert row["max_world_size"] == 1
    bmm = next(arm for arm in row["arms"] if arm["arm"] == "bmm")
    dense = next(arm for arm in row["arms"] if arm["arm"] == "dense")
    assert bmm["max_world_size"] == 1
    assert dense["max_world_size"] == 1

    from gridbook import dsv4_woa
    from gridbook import mxfp8_dense_lane as lane

    extension = _IdentitySfExtension()
    lane._OFFSETS._cache.clear()
    monkeypatch.setattr(lane, "_require_lane_ext", lambda device=None: extension)
    monkeypatch.setattr(dsv4_woa, "install_dsv4_woa_adapter", lambda: None)

    groups, rows, k = 2, 128, 128
    layer = torch.nn.Module()
    layer.tp_size = 2
    method = lane.build_mxfp8_dense_method(lane.WIRE_MXFP8_G32)
    method.create_weights(
        layer, k, [groups * rows], k, groups * rows, torch.bfloat16)
    layer.is_bmm = True
    layer.bmm_batch_size = groups

    with pytest.raises(ValueError, match=r"audited only for TP=1"):
        method.process_weights_after_loading(layer)

    # The refusal precedes finalization: no scale plane was built or
    # registered. (The lane drops its staging ``weight_scale`` parameter
    # before the gate, so its absence is expected here.)
    assert not hasattr(layer, "weight_sf_planes")
