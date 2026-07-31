"""R21 — the MoE auto-tuner's timings get a durable sink.

Torch-only, no vLLM, no GPU (`moe_autotune.py` exists precisely so this policy
is testable that way). What is pinned: the rows carry the three keys that make
a serving-cost table comparable across machines — (format, shape regime, box) —
the sink is append-only, an unwritable path degrades to stderr instead of
killing a serve, and the log callback stays compatible with the
three-positional-argument form that predates the sink.
"""
import json
import os

import pytest
import torch

from gridbook.moe_autotune import (
    STOCK,
    autotune_sink_path,
    box_id,
    cb_prefill_auto,
    record_autotune_timings,
    shape_regime,
)


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_sink_path_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_AUTOTUNE_LOG", raising=False)
    assert autotune_sink_path(None) is None
    assert autotune_sink_path(tmp_path).endswith("cb_autotune_timings.jsonl")
    monkeypatch.setenv("PRISMAQUANT_CB_AUTOTUNE_LOG", str(tmp_path / "x.jsonl"))
    assert autotune_sink_path(tmp_path) == str(tmp_path / "x.jsonl")
    monkeypatch.setenv("PRISMAQUANT_CB_AUTOTUNE_LOG", "off")
    assert autotune_sink_path(tmp_path) is None


def test_row_carries_format_regime_and_box(tmp_path):
    path = tmp_path / "t.jsonl"
    written = record_autotune_timings(
        "grouped_fused:tile_m=128",
        {STOCK: 9.0, "grouped_fused:tile_m=128": 4.0},
        layer_prefix="model.layers.3.mlp",
        fmt="FP8_CB_K40",
        regime=shape_regime(4096, n_experts=128, intermediate=768),
        path=str(path),
    )
    assert written == str(path)
    (row,) = _read(path)
    assert row["schema"] == "prismaquant.cb_autotune.v1"
    assert row["format"] == "FP8_CB_K40"
    assert row["box"] == box_id()
    assert row["chosen"] == "grouped_fused:tile_m=128"
    assert row["ms"]["stock"] == pytest.approx(9.0)
    assert row["regime"]["num_tokens"] == 4096
    assert row["regime"]["m_bucket"] == 4096
    assert row["regime"]["n_experts"] == 128
    assert row["regime"]["intermediate"] == 768


def test_shape_regime_buckets_tokens_by_power_of_two():
    assert shape_regime(1024)["m_bucket"] == 1024
    assert shape_regime(1500)["m_bucket"] == 1024
    assert shape_regime(2048)["m_bucket"] == 2048
    assert shape_regime(0)["m_bucket"] == 1


def test_sink_is_append_only(tmp_path):
    path = str(tmp_path / "t.jsonl")
    for i in range(3):
        record_autotune_timings(
            STOCK, {STOCK: float(i)}, layer_prefix=f"l{i}",
            fmt="FP8_CB_K40", regime=shape_regime(2048), path=path)
    rows = _read(path)
    assert [r["layer"] for r in rows] == ["l0", "l1", "l2"]


def test_unwritable_sink_never_fails_the_serve(tmp_path, capsys):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    out = record_autotune_timings(
        STOCK, {STOCK: 1.0}, layer_prefix="l", fmt="FP8_CB_K40",
        regime=shape_regime(2048), path=str(blocked / "sub" / "t.jsonl"))
    assert out is None
    assert "autotune sink unavailable" in capsys.readouterr().err


class _FakeLayer:
    pass


def test_auto_passes_layer_context_to_a_rich_callback(tmp_path):
    seen = {}

    def log(best, timings, forced=False, layer=None, num_tokens=None):
        seen.update(best=best, forced=forced, layer=layer,
                    num_tokens=num_tokens, timings=dict(timings))

    layer = _FakeLayer()
    cb_prefill_auto(
        layer, 4096,
        lambda: [(STOCK, lambda: torch.zeros(2)),
                 ("grouped_fused:tile_m=128", lambda: torch.zeros(2))],
        lambda: torch.ones(2),
        timer=lambda fn: (fn(), 4.0 if fn() is not None else None),
        log=log,
    )
    assert seen["layer"] is layer
    assert seen["num_tokens"] == 4096
    assert seen["forced"] is False


def test_auto_still_accepts_the_legacy_three_arg_callback():
    """The sink is additive: a callback that predates it must not break."""
    calls = []
    layer = _FakeLayer()
    cb_prefill_auto(
        layer, 4096,
        lambda: [(STOCK, lambda: torch.zeros(2))],
        lambda: torch.ones(2),
        timer=lambda fn: (fn(), 4.0),
        log=lambda best, timings, forced: calls.append((best, forced)),
    )
    assert calls == [(STOCK, False)]


def test_forced_choice_is_recorded_too(tmp_path, monkeypatch):
    """A pinned winner (`PRISMAQUANT_CB_PREFILL_AUTO_FORCE`) has no timings but
    still belongs in the table — otherwise a table's gaps look like a box that
    was never measured rather than a run that was pinned."""
    path = str(tmp_path / "t.jsonl")
    out = record_autotune_timings(
        "grouped_fused", {}, layer_prefix="l", fmt="FP8_CB_K40",
        regime=shape_regime(4096), forced=True, path=path)
    assert out == path
    (row,) = _read(path)
    assert row["forced"] is True and row["ms"] == {}


def test_no_sink_configured_is_a_silent_noop(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_AUTOTUNE_LOG", raising=False)
    assert record_autotune_timings(
        STOCK, {STOCK: 1.0}, layer_prefix="l", fmt="FP8_CB_K40",
        regime=shape_regime(2048)) is None


def test_lambda_term_is_still_unimplemented():
    """R21's ruling: the time term stays specified-not-implemented until two
    boxes' tables disagree in ranking. Nothing in the allocator's candidate
    carries a time field, and this pins that so 'persist the timings' is not
    mistaken for 'the allocator now optimizes latency'.

    Monorepo-only: the allocator is not part of the released gridbook package,
    and this file is synced verbatim into the standalone release repo
    (prismaquant scripts/sync_gridbook.py), where the import cannot resolve.
    importorskip, not a bare import, so that run reports "cannot check" rather
    than a red suite."""
    Candidate = pytest.importorskip(
        "prismaquant.allocator_solver",
        reason="allocator lives in the prismaquant monorepo, not in the "
               "released gridbook package").Candidate

    fields = set(getattr(Candidate, "__dataclass_fields__", {}))
    assert not (fields & {"latency_ms", "time_cost", "lambda_time", "ms"})
