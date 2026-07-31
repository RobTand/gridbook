"""Runtime contracts for graded FusedMoE stacks (requires a live vLLM).

The resolver-only suite is deliberately vLLM-free.  This file exercises the
actual ``PrismaQuantCBMoEMethod`` imported against the installed serving API:
allocation/fill/post-load state, per-stack decode arguments, default-on fused
FP4 arguments, and durable auto-tune telemetry.
"""
from __future__ import annotations

import json
import types

import pytest
import torch

moe_mod = pytest.importorskip("gridbook.moe")
from gridbook import codec  # noqa: E402
from gridbook.cb_fill_guard import mark_filled  # noqa: E402

PrismaQuantCBMoEMethod = moe_mod.PrismaQuantCBMoEMethod


def _scheme(grid, k, n_sub, type_size, refs, **extra):
    out = {
        "grid": grid,
        "mode": "product",
        "k": k,
        "n_sub": n_sub,
        "type_size": type_size,
        "codebook_ref": list(refs),
    }
    if grid == "fp4":
        out["scale_coding"] = codec.SCALE_CODING_TWO_TIER
    out.update(extra)
    return out


def _method(w13, w2):
    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.prefix = "model.layers.0.mlp.experts"
    method.stack_scheme = {"w13": w13, "w2": w2}
    method.scheme = w13
    method._stack_formats_match = (w13 == w2)
    method.is_fp4 = w13["grid"] == "fp4"
    method.is_v2 = (w13.get("scale_coding")
                    == codec.SCALE_CODING_TWO_TIER)
    method._sub_table = (codec.TWO_TIER_SUB_TABLE if method.is_v2 else None)
    method.k = {"w13": int(w13["k"]), "w2": int(w2["k"])}
    method.n_sub = {"w13": int(w13["n_sub"]),
                    "w2": int(w2["n_sub"])}
    method.type_size = {"w13": int(w13["type_size"]),
                        "w2": int(w2["type_size"])}
    return method


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deferred = []

    def load_weights(self, weights):
        for name, _weight in weights:
            self.deferred.append(name)
            yield name


def _fp8_case():
    refs13 = tuple(f"cb32.{i}" for i in range(4))
    refs2 = tuple(f"cb28.{i}" for i in range(4))
    w13 = _scheme("fp8", 32, 4, 128, refs13)
    w2 = _scheme("fp8", 28, 4, 112, refs2)
    method = _method(w13, w2)
    codebooks = {}
    # Product n_sub=4: each sub-vector has width 2.  K32 -> 2^8 entries;
    # K28 -> 2^7.  Distinct values also make the two resident LUTs observable.
    for ref in refs13:
        codebooks[ref] = torch.zeros(1 << 8, 2)
    for ref in refs2:
        codebooks[ref] = torch.ones(1 << 7, 2)
    method.quant_config = types.SimpleNamespace(
        get_codebooks=lambda: codebooks)
    layer = _Layer()
    method.create_weights(
        layer, num_experts=2, hidden_size=256,
        intermediate_size_per_partition=256, params_dtype=torch.bfloat16,
        weight_loader=None)
    return method, layer


def _fill_instance_stacks(layer):
    loaded = list(layer.load_weights(iter([
        ("gate_up_proj.cb_qweight",
         torch.full_like(layer.w13_cb_qweight, 7)),
        ("down_proj.cb_qweight", torch.full_like(layer.w2_cb_qweight, 9)),
    ])))
    assert loaded == ["w13_cb_qweight", "w2_cb_qweight"]


def test_graded_create_fill_and_postload_remove_legacy_aliases():
    method, layer = _fp8_case()
    assert layer.w13_cb_qweight.shape == (2, 512, 128)
    assert layer.w2_cb_qweight.shape == (2, 256, 112)
    _fill_instance_stacks(layer)

    # A reused layer carrying old uniform state must not retain it.
    layer._cb_flat = torch.tensor([-1.0])
    layer._cb_compose = torch.tensor([-1.0])
    layer._cb_flat_fp8 = torch.tensor([255], dtype=torch.uint8)
    stale_specific = torch.tensor([254], dtype=torch.uint8)
    layer._cb_flat_fp8_w13 = stale_specific
    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "_cb_flat")
    assert not hasattr(layer, "_cb_compose")
    assert not hasattr(layer, "_cb_flat_fp8")
    assert layer._cb_stack_uniform is False
    assert layer._cb_flat_w13.numel() == 4 * (1 << 8) * 2
    assert layer._cb_flat_w2.numel() == 4 * (1 << 7) * 2
    assert layer._cb_flat_fp8_w13.data_ptr() != \
        layer._cb_flat_fp8_w2.data_ptr()
    assert layer._cb_flat_fp8_w13 is not stale_specific
    assert torch.all(layer.w13_cb_qweight == 7)
    assert torch.all(layer.w2_cb_qweight == 9)


def test_alias_requires_the_complete_scheme_signature():
    refs = tuple(f"cb.{i}" for i in range(4))
    w13 = _scheme("fp8", 28, 4, 112, refs, future_format_revision=1)
    w2 = _scheme("fp8", 28, 4, 112, refs, future_format_revision=2)
    method = _method(w13, w2)
    method.quant_config = types.SimpleNamespace(get_codebooks=lambda: {
        ref: torch.zeros(1 << 7, 2) for ref in refs
    })
    layer = _Layer()
    method.create_weights(layer, 1, 256, 256, torch.bfloat16,
                          weight_loader=None)
    _fill_instance_stacks(layer)
    method.process_weights_after_loading(layer)
    # Same current rung + same tensors is still not proof that a future format
    # field is interchangeable.  Per-stack tensors may dedupe; aliases may not.
    assert layer._cb_flat_w13 is layer._cb_flat_w2
    assert layer._cb_stack_uniform is False
    assert not hasattr(layer, "_cb_flat")


def test_grouped_decode_passes_each_stack_its_own_lut_and_format(monkeypatch):
    method, layer = _fp8_case()
    _fill_instance_stacks(layer)
    method.process_weights_after_loading(layer)
    calls = []

    def gemv(xq, qw, lut, scale, pair_expert, pair_xrow,
             k_bits, n_sub, type_size):
        calls.append((lut.data_ptr(), k_bits, n_sub, type_size, qw.shape[1]))
        return torch.zeros((pair_expert.numel(), qw.shape[1]), dtype=xq.dtype)

    from gridbook import ops
    monkeypatch.setattr(ops, "fp8_act_qdq", lambda x: x)
    monkeypatch.setattr(ops, "cb_moe_gemv_fp8", gemv)
    monkeypatch.setattr(
        ops, "cb_moe_combine",
        lambda y, pair_w, tok_start, tokens: torch.zeros(
            (tokens, y.shape[1]), dtype=y.dtype))
    monkeypatch.setattr(
        moe_mod, "apply_moe_activation",
        lambda _act, out, gate_up: out.zero_())

    out = method._apply_grouped_decode(
        layer, torch.zeros(2, 256, dtype=torch.bfloat16),
        torch.ones(2, 1), torch.tensor([[0], [1]]), object())
    assert out.shape == (2, 256)
    assert [(k, ns, ts) for _ptr, k, ns, ts, _n in calls] == [
        (32, 4, 128), (28, 4, 112)]
    assert calls[0][0] == layer._cb_flat_fp8_w13.data_ptr()
    assert calls[1][0] == layer._cb_flat_fp8_w2.data_ptr()


def test_default_fused_fp4_launches_with_each_stack_format(monkeypatch):
    w13 = _scheme("fp4", 12, 2, 57, ("a", "b"))
    w2 = _scheme("fp4", 16, 2, 73, ("c", "d"))
    method = _method(w13, w2)
    layer = types.SimpleNamespace(
        _cb_E=2, _cb_hidden=256, _cb_inter=256,
        w13_cb_qweight=torch.zeros(2, 512, 57, dtype=torch.uint8),
        w2_cb_qweight=torch.zeros(2, 256, 73, dtype=torch.uint8),
    )
    lut13 = torch.tensor([13], dtype=torch.uint8)
    lut2 = torch.tensor([2], dtype=torch.uint8)
    compose13 = torch.tensor([31], dtype=torch.uint8)
    compose2 = torch.tensor([21], dtype=torch.uint8)
    layer._cb_fp4_gf_w13 = (lut13, compose13)
    layer._cb_fp4_gf_w2 = (lut2, compose2)
    calls = []

    class _Ext:
        @staticmethod
        def cb_fused_fp4_moe_grouped(*args):
            calls.append(args)
            return torch.zeros((args[0].shape[0], args[8]),
                               dtype=torch.bfloat16)

    from gridbook import cuda_ext
    monkeypatch.setattr(cuda_ext, "get_fused_fp4_ext", lambda: _Ext())
    monkeypatch.setattr(method, "_gf4_ok", lambda _layer: True)
    monkeypatch.setattr(
        method, "_fp4_quant",
        lambda x: (x, torch.zeros(1, dtype=torch.uint8), torch.ones(1)))
    monkeypatch.setattr(
        moe_mod, "apply_moe_activation",
        lambda _act, out, gate_up: out.zero_())

    x = torch.zeros(17, 256, dtype=torch.bfloat16)
    out = method._apply_prefill_grouped_fused_fp4(
        layer, x, torch.ones(17, 1),
        torch.tensor([[i % 2] for i in range(17)]), object(), tile_m=128)
    assert out.shape == x.shape
    assert [(call[10], call[11], call[12]) for call in calls] == [
        (12, 2, 57), (16, 2, 73)]
    assert calls[0][3] is lut13 and calls[0][4] is compose13
    assert calls[1][3] is lut2 and calls[1][4] is compose2


def test_unset_fp4_prefill_selects_the_graded_fused_path(monkeypatch):
    w13 = _scheme("fp4", 12, 2, 57, ("a", "b"))
    w2 = _scheme("fp4", 16, 2, 73, ("c", "d"))
    method = _method(w13, w2)
    method._cuda_moe_ok = lambda _layer: False
    sentinel = torch.full((17, 256), 3.0)
    seen = {}

    def fused(_layer, _x, _weights, _ids, _act, *, tile_m):
        seen["tile_m"] = tile_m
        return sentinel

    method._apply_prefill_grouped_fused_fp4 = fused
    method._apply_prefill_loop = lambda *_args: pytest.fail(
        "unset mode should not fall through to loop when fused succeeds")
    monkeypatch.delenv("PRISMAQUANT_CB_FUSED_FP4_MOE", raising=False)
    monkeypatch.delenv("PRISMAQUANT_CB_PREFILL", raising=False)
    layer = types.SimpleNamespace(
        activation=types.SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False)
    out = method._apply_inline(
        layer, torch.zeros(17, 256), torch.ones(17, 1),
        torch.zeros(17, 1, dtype=torch.long))
    assert out is sentinel
    assert seen == {"tile_m": 128}


def test_graded_autotune_sink_records_both_rungs(tmp_path, monkeypatch):
    method, _layer = _fp8_case()
    path = tmp_path / "graded.jsonl"
    monkeypatch.setenv("PRISMAQUANT_CB_AUTOTUNE_LOG", str(path))
    context = types.SimpleNamespace(_cb_E=2, _cb_inter=256)
    method._log_prefill_choice(
        "stock", {"stock": 1.25}, layer=context, num_tokens=2048)
    row = json.loads(path.read_text().strip())
    assert row["format"] == \
        "FP8_CB[w13=K32/N4/T128,w2=K28/N4/T112]"
    assert row["regime"]["n_experts"] == 2
