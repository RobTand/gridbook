"""Narrow DeepSeek-V4 output-projection adapter for Gridbook-owned BMM."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import dsv4_woa  # noqa: E402


class _Rotary:
    def __init__(self) -> None:
        self.calls = []

    def forward_native(self, positions, query, key=None, *, inverse=False):
        self.calls.append((positions.clone(), key, inverse))
        return query * 2, key


class _WoA(torch.nn.Module):
    def __init__(self, groups: int, rows: int, k: int) -> None:
        super().__init__()
        setattr(
            self,
            dsv4_woa.DSV4_MXFP8_BMM_ATTR,
            dsv4_woa.DSV4_MXFP8_BMM_ABI,
        )
        values = torch.arange(groups * rows * k, dtype=torch.float32)
        self.register_buffer("weights", values.reshape(groups, rows, k) / 100)

    def forward(self, x):
        return torch.einsum("tgk,grk->tgr", x.float(), self.weights)


class _WoB(torch.nn.Module):
    def forward(self, x):
        return x + 7


def _attention(groups=2, rows=3, heads=4, head_dim=2):
    return types.SimpleNamespace(
        n_local_groups=groups,
        o_lora_rank=rows,
        rotary_emb=_Rotary(),
        wo_a=_WoA(groups, rows, heads * head_dim // groups),
        wo_b=_WoB(),
    )


def test_adapter_runs_inverse_rope_group_bmm_then_wo_b():
    attention = _attention()
    o = torch.arange(2 * 4 * 2, dtype=torch.bfloat16).reshape(2, 4, 2)
    positions = torch.tensor([3, 9])
    result = dsv4_woa.dsv4_mxfp8_o_proj(attention, o, positions)

    grouped = (o * 2).reshape(2, 2, 4)
    z = torch.einsum(
        "tgk,grk->tgr", grouped.float(), attention.wo_a.weights)
    assert torch.equal(result, z.flatten(1) + 7)
    assert len(attention.rotary_emb.calls) == 1
    called_positions, called_key, inverse = attention.rotary_emb.calls[0]
    assert torch.equal(called_positions, positions)
    assert called_key is None
    assert inverse is True


def test_adapter_refuses_missing_or_stale_method_protocol():
    attention = _attention()
    setattr(attention.wo_a, dsv4_woa.DSV4_MXFP8_BMM_ATTR, 99)
    with pytest.raises(RuntimeError, match="does not own a supported"):
        dsv4_woa.dsv4_mxfp8_o_proj(
            attention, torch.zeros(1, 4, 2), torch.zeros(1, dtype=torch.long))


def test_adapter_accepts_source_fp8_w8a16_bmm_marker():
    attention = _attention()
    delattr(attention.wo_a, dsv4_woa.DSV4_MXFP8_BMM_ATTR)
    setattr(attention.wo_a, dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ATTR,
            dsv4_woa.DSV4_FP8_SOURCE_W8A16_BMM_ABI)
    o = torch.arange(2 * 4 * 2, dtype=torch.bfloat16).reshape(2, 4, 2)
    result = dsv4_woa.dsv4_gridbook_o_proj(
        attention, o, torch.tensor([3, 9]))
    grouped = (o * 2).reshape(2, 2, 4)
    expected = torch.einsum(
        "tgk,grk->tgr", grouped.float(), attention.wo_a.weights)
    assert torch.equal(result, expected.flatten(1) + 7)


def test_class_wrapper_leaves_stock_wo_a_byte_for_byte():
    sentinel = object()

    class Attention:
        def _o_proj(self, o, positions):
            return sentinel

    assert dsv4_woa._wrap_attention_class(Attention) is True
    assert dsv4_woa._wrap_attention_class(Attention) is False
    instance = Attention()
    instance.wo_a = types.SimpleNamespace()
    assert instance._o_proj(None, None) is sentinel


def test_installer_wraps_only_loaded_audited_class(monkeypatch):
    sentinel = object()

    class Attention:
        def _o_proj(self, o, positions):
            return sentinel

    module = types.ModuleType(
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse")
    module.DeepseekV4FlashInferSM120Attention = Attention
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    monkeypatch.setattr(dsv4_woa, "_LOGGED", False)

    dsv4_woa.install_dsv4_woa_adapter()
    instance = Attention()
    instance.wo_a = types.SimpleNamespace()
    assert instance._o_proj(None, None) is sentinel
    # Idempotence is part of the process-wide model-class ABI.
    dsv4_woa.install_dsv4_woa_adapter()
