"""Low-level gates for behavior-preserving MoE prefill overhead cuts.

Torch-only: CUDA timing is exercised with event/stream stubs, and the immutable
metadata caches run on CPU.  This makes the synchronization and cache-key
contracts fail loudly in the ordinary build environment rather than only on a
serving GPU.
"""
from __future__ import annotations

import gc
import types
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from gridbook.moe_autotune import (
    STOCK,
    cb_autotune_prefill,
    cb_prefill_auto,
    cb_time_candidate,
)
from gridbook.moe_routing import (
    cb_cached_expert_map,
    cb_cached_row_offsets,
)


def test_singleton_candidate_runs_once_without_touching_timer_or_cuda(
        monkeypatch):
    """One candidate has no ranking decision and must allocate no timer events."""
    marker = object()
    calls = {"ran": 0}

    def candidate():
        calls["ran"] += 1
        return marker

    def forbidden_timer(_fn):
        raise AssertionError("singleton candidate reached the timer")

    monkeypatch.setattr(
        torch.cuda, "is_available",
        lambda: (_ for _ in ()).throw(
            AssertionError("singleton candidate queried CUDA timing")))
    best, timings, kept = cb_autotune_prefill(
        [(STOCK, candidate)], timer=forbidden_timer)

    assert best == STOCK
    assert timings == {}
    assert kept is marker
    assert calls == {"ran": 1}


def test_singleton_auto_caches_direct_choice_and_preserves_output():
    layer = types.SimpleNamespace()
    marker = object()
    seen = {"built": 0, "ran": 0, "stock_fallback": 0, "logs": []}

    def build():
        seen["built"] += 1

        def stock():
            seen["ran"] += 1
            return marker

        return [(STOCK, stock)]

    def fallback():
        seen["stock_fallback"] += 1
        raise AssertionError("successful singleton invoked stock twice")

    out = cb_prefill_auto(
        layer, 4096, build, fallback,
        timer=lambda _fn: (_ for _ in ()).throw(
            AssertionError("singleton auto reached timer")),
        log=lambda best, timings, forced: seen["logs"].append(
            (best, dict(timings), forced)),
    )

    assert out is marker
    assert layer._cb_prefill_choice == STOCK
    assert seen == {
        "built": 1,
        "ran": 1,
        "stock_fallback": 0,
        "logs": [(STOCK, {}, False)],
    }


def test_non_stock_singleton_keeps_stock_output_on_tuning_call():
    """Skipping timing must not weaken the tuner's determinism contract."""
    layer = types.SimpleNamespace()
    tuned = object()
    stock = object()
    seen = {"candidate": 0, "fallback": 0}

    def candidate():
        seen["candidate"] += 1
        return tuned

    def fallback():
        seen["fallback"] += 1
        return stock

    out = cb_prefill_auto(
        layer, 4096,
        lambda: [("grouped_fused:tile_m=128", candidate)],
        fallback,
        timer=lambda _fn: (_ for _ in ()).throw(
            AssertionError("non-stock singleton reached timer")),
    )

    assert out is stock
    assert layer._cb_prefill_choice == "grouped_fused:tile_m=128"
    assert seen == {"candidate": 1, "fallback": 1}


@pytest.mark.parametrize("failure", ["none", "raise"])
def test_singleton_disqualification_preserves_fallback_contract(failure,
                                                                 capsys):
    def candidate():
        if failure == "raise":
            raise RuntimeError("unavailable")
        return None

    best, timings, kept = cb_autotune_prefill(
        [(STOCK, candidate)],
        timer=lambda _fn: (_ for _ in ()).throw(
            AssertionError("disqualified singleton reached timer")),
    )
    assert (best, timings, kept) == (None, {}, None)
    if failure == "raise":
        assert "candidate disqualified" in capsys.readouterr().err


def test_cuda_timer_uses_current_stream_and_only_end_event_sync(monkeypatch):
    trace = []
    stream = object()
    events = []

    class FakeEvent:
        def __init__(self, *, enable_timing):
            self.index = len(events)
            self.enable_timing = enable_timing
            events.append(self)
            trace.append(("event", self.index, enable_timing))

        def record(self, recorded_stream=None):
            trace.append(("record", self.index, recorded_stream))

        def synchronize(self):
            trace.append(("event_sync", self.index))

        def elapsed_time(self, end):
            trace.append(("elapsed", self.index, end.index))
            return 7.25

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "current_stream",
        lambda: (trace.append(("current_stream",)), stream)[1])
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda: (_ for _ in ()).throw(
            AssertionError("device-wide synchronize is forbidden")))

    marker = object()

    def candidate():
        trace.append(("candidate",))
        return marker

    out, elapsed = cb_time_candidate(candidate)
    assert out is marker
    assert elapsed == pytest.approx(7.25)
    assert trace == [
        ("current_stream",),
        ("event", 0, True),
        ("event", 1, True),
        ("record", 0, stream),
        ("candidate",),
        ("record", 1, stream),
        ("event_sync", 1),
        ("elapsed", 0, 1),
    ]


def test_cached_expert_maps_key_exact_bounds_device_and_layer():
    layer = types.SimpleNamespace()
    same_alias = torch.device("cpu")

    a = cb_cached_expert_map(layer, 2, 5, 8, "cpu")
    a_again = cb_cached_expert_map(layer, 2, 5, 8, same_alias)
    assert a_again is a
    assert a.dtype == torch.int32 and a.device == same_alias
    assert torch.equal(a, torch.tensor([-1, -1, 0, 1, 2, -1, -1, -1],
                                       dtype=torch.int32))

    # Same chunk width, different bounds: never alias by size alone.
    shifted = cb_cached_expert_map(layer, 3, 6, 8, "cpu")
    assert shifted is not a
    assert torch.equal(
        shifted,
        torch.tensor([-1, -1, -1, 0, 1, 2, -1, -1], dtype=torch.int32),
    )

    # The cache is per layer, so two model layers cannot share tensor lifetime.
    other = cb_cached_expert_map(
        types.SimpleNamespace(), 2, 5, 8, "cpu")
    assert other is not a
    assert torch.equal(other, a)


@pytest.mark.parametrize("bounds", [(-1, 2, 8), (4, 3, 8), (0, 9, 8)])
def test_cached_expert_map_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="invalid expert chunk"):
        cb_cached_expert_map(types.SimpleNamespace(), *bounds, "cpu")


def test_cached_row_offsets_key_exact_size_device_and_share_immutables():
    layer = types.SimpleNamespace()
    row0 = cb_cached_row_offsets(layer, 17, "cpu")
    row0_again = cb_cached_row_offsets(layer, 17, torch.device("cpu"))
    assert row0_again is row0
    assert row0.dtype == torch.int32 and row0.is_contiguous()
    assert torch.equal(row0, torch.zeros(17, dtype=torch.int32))

    other_size = cb_cached_row_offsets(layer, 18, "cpu")
    assert other_size is not row0
    assert other_size.shape == (18,)

    other_layer = cb_cached_row_offsets(types.SimpleNamespace(), 17, "cpu")
    assert other_layer is row0

    empty = cb_cached_row_offsets(layer, 0, "cpu")
    assert empty.shape == (0,)
    with pytest.raises(ValueError, match="non-negative"):
        cb_cached_row_offsets(layer, -1, "cpu")


def test_cached_row_offset_pool_does_not_own_unloaded_layers():
    layer = types.SimpleNamespace()
    row0 = cb_cached_row_offsets(layer, 123_457, "cpu")
    tensor_ref = weakref.ref(row0)

    del row0
    del layer
    gc.collect()

    assert tensor_ref() is None


def test_concurrent_cache_warmup_converges_on_one_object():
    layer = types.SimpleNamespace()

    def get_constants(_):
        return (
            cb_cached_expert_map(layer, 1, 4, 8, "cpu"),
            cb_cached_row_offsets(layer, 32_771, "cpu"),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(get_constants, range(32)))

    expert_maps, row_offsets = zip(*results)
    assert all(value is expert_maps[0] for value in expert_maps)
    assert all(value is row_offsets[0] for value in row_offsets)


@pytest.mark.parametrize("attr", ["_cb_stock_expert_maps",
                                  "_cb_row_offset_cache"])
def test_cache_attribute_collision_fails_loudly(attr):
    layer = types.SimpleNamespace(**{attr: object()})
    with pytest.raises(TypeError, match="must be a dict"):
        if attr == "_cb_stock_expert_maps":
            cb_cached_expert_map(layer, 0, 1, 1, "cpu")
        else:
            cb_cached_row_offsets(layer, 1, "cpu")
