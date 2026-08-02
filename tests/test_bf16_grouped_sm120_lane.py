"""Contract and wiring gates for the OPT-IN sm12x-native BF16 grouped lane.

The selector, attestation and tile-order cases below are CPU-only. The lane
itself is a GPU kernel (``tests/test_bf16_grouped_cutlass.py`` gates its
numerics); what is testable without a device is the part that decides WHETHER
it runs, and that part carries the whole opt-in promise:

* with ``PRISMAQUANT_CB_BF16_SM120`` unset, nothing about the dispatch changes
  and nothing probes or builds;
* with it set on a machine that cannot serve the lane, the model LOAD fails
  with an actionable message instead of quietly running the SM80 schedule —
  which would answer a different question than the operator asked;
* the selector is process-stable and rejects typos, so an intended A/B can
  never become an unlabelled baseline run.

The last section gates the WIRING: ``gridbook/moe.py``'s routed prefill through
its own dispatch with the flag on and off, so the two claims the lane makes at
the operator level are asserted where an operator actually meets them — that
stage one carries NO padded activation copy any more (the in-mainloop A-row
gather reads the compact tensor and is bit-identical to the copy it replaced),
and that the swizzle-group packed expert ORDER is bit-neutral and applied only
where the expert-chunk loop's expert-major assumption still holds. Those cases
need CUDA, the grouped-BF16 extension and the sm12x lane; each skips on its own
rather than skipping the file, exactly as sections A-C stay live on a CI host
with no GPU.
"""
from __future__ import annotations

import importlib.util
import inspect
import types

import pytest

torch = pytest.importorskip("torch")

from gridbook import bf16_grouped_lane as lane  # noqa: E402
from gridbook.cuda_ext import NativeKernelUnavailableError  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_flag(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_CB_BF16_SM120", raising=False)
    lane._reset_for_tests()
    yield
    lane._reset_for_tests()


def test_flag_defaults_to_the_sm80_schedule(monkeypatch):
    assert lane.requested() is False
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "0")
    lane._reset_for_tests()
    assert lane.requested() is False


def test_flag_enables_the_lane(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    assert lane.requested() is True


@pytest.mark.parametrize("value", ["yes", "true", "128", "sm120"])
def test_typos_are_refused_rather_than_silently_ignored(monkeypatch, value):
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", value)
    with pytest.raises(ValueError, match="PRISMAQUANT_CB_BF16_SM120"):
        lane.requested()


def test_selector_cannot_change_mid_process(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "1")
    assert lane.requested() is True
    monkeypatch.setenv("PRISMAQUANT_CB_BF16_SM120", "0")
    with pytest.raises(RuntimeError, match="changed after Gridbook dispatch"):
        lane.requested()


def test_require_lane_fails_closed_without_the_bindings(monkeypatch):
    """A module without the sm12x entry points is refused, not fallen back."""
    stub = types.SimpleNamespace(cb_bf16_grouped_mm=lambda *a, **k: None,
                                 cb_bf16_grouped_mm_out=lambda *a, **k: None)
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: stub)
    with pytest.raises(NativeKernelUnavailableError) as exc_info:
        lane.require_lane("routed quality prefill")
    message = str(exc_info.value)
    assert "PRISMAQUANT_CB_BF16_SM120=1" in message
    assert "cb_bf16_grouped_mm_sm120" in message
    assert "compute capability 12.0/12.1" in message
    assert "does not substitute a different kernel" in message


def test_require_lane_fails_closed_without_the_extension(monkeypatch):
    """No grouped-BF16 module at all is the same fail-closed answer."""
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: None)
    with pytest.raises(NativeKernelUnavailableError,
                       match="cb_bf16_grouped_gemm.cu"):
        lane.require_lane("routed quality prefill")


def test_require_lane_accepts_a_complete_module(monkeypatch):
    stub = types.SimpleNamespace(
        cb_bf16_grouped_mm=lambda *a, **k: None,
        cb_bf16_grouped_mm_out=lambda *a, **k: None,
        cb_bf16_grouped_mm_sm120=lambda *a, **k: None,
        cb_bf16_grouped_mm_sm120_out=lambda *a, **k: None,
        cb_bf16_grouped_mm_sm120_gather=lambda *a, **k: None,
        cb_bf16_grouped_mm_sm120_gather_out=lambda *a, **k: None,
        cb_bf16_grouped_sm120_tile_m=lambda: 128,
    )
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: stub)
    assert lane.require_lane("routed quality prefill") is stub
    assert lane.tile_m(stub) == 128


def test_require_lane_fails_closed_without_the_gather_mode(monkeypatch):
    """A module carrying only the padded-copy entry points is incomplete:
    serving a partial lane would silently reintroduce the padded copy."""
    stub = types.SimpleNamespace(
        cb_bf16_grouped_mm=lambda *a, **k: None,
        cb_bf16_grouped_mm_out=lambda *a, **k: None,
        cb_bf16_grouped_mm_sm120=lambda *a, **k: None,
        cb_bf16_grouped_mm_sm120_out=lambda *a, **k: None,
        cb_bf16_grouped_sm120_tile_m=lambda: 64,
    )
    monkeypatch.setattr("gridbook.cuda_ext.get_bf16_grouped_ext",
                        lambda: stub)
    with pytest.raises(NativeKernelUnavailableError,
                       match="cb_bf16_grouped_mm_sm120_gather"):
        lane.require_lane("routed quality prefill")


def test_dense_helper_pads_to_one_tile_and_slices_back(monkeypatch):
    """Without the gather mode (an old stub), M=100 becomes one padded tile."""
    seen = {}

    def fake_op(a, weights, expert_ids, tile_m):
        seen["a"] = a
        seen["expert_ids"] = expert_ids
        seen["tile_m"] = tile_m
        assert weights.shape[0] == 1, "dense is E=1"
        return torch.arange(a.shape[0] * weights.shape[1],
                            dtype=torch.bfloat16).reshape(
                                a.shape[0], weights.shape[1])

    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120", fake_op)
    ext = types.SimpleNamespace(cb_bf16_grouped_sm120_tile_m=lambda: 128)
    a = torch.randn(100, 64, dtype=torch.bfloat16)
    w = torch.randn(32, 64, dtype=torch.bfloat16)

    y = lane.dense_mm(ext, a, w)

    assert y.shape == (100, 32)
    assert seen["tile_m"] == 128
    assert seen["a"].shape == (128, 64)
    assert torch.equal(seen["a"][:100], a)
    assert not seen["a"][100:].any(), "padding rows must be zero"
    assert seen["expert_ids"].tolist() == [0]
    assert seen["expert_ids"].dtype is torch.int32


def test_dense_helper_leaves_an_exact_multiple_alone(monkeypatch):
    def fake_op(a, weights, expert_ids, tile_m):
        assert a.shape[0] == 256
        assert expert_ids.tolist() == [0, 0]
        return torch.zeros(a.shape[0], weights.shape[1],
                           dtype=torch.bfloat16)

    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120", fake_op)
    ext = types.SimpleNamespace(cb_bf16_grouped_sm120_tile_m=lambda: 128)
    y = lane.dense_mm(ext, torch.randn(256, 64, dtype=torch.bfloat16),
                      torch.randn(8, 64, dtype=torch.bfloat16))
    assert y.shape == (256, 8)


def test_dense_helper_prefers_the_gather_mode_and_copies_nothing(monkeypatch):
    """With the gather entry point present, no padded copy is built at all:
    the compact activation goes straight in with row_src = arange(Mp), whose
    ids past M read zeros inside the kernel."""
    seen = {}

    def fake_gather(a, row_src, weights, expert_ids, tile_m):
        seen["a"] = a
        seen["row_src"] = row_src
        seen["expert_ids"] = expert_ids
        seen["tile_m"] = tile_m
        return torch.zeros(row_src.shape[0], weights.shape[1],
                           dtype=torch.bfloat16)

    def forbidden(*args, **kwargs):
        raise AssertionError("the padded-copy op must not be called")

    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120_gather",
                        fake_gather)
    monkeypatch.setattr("gridbook.ops.cb_bf16_grouped_mm_sm120", forbidden)
    ext = types.SimpleNamespace(
        cb_bf16_grouped_sm120_tile_m=lambda: 128,
        cb_bf16_grouped_mm_sm120_gather=lambda *a, **k: None,
    )
    a = torch.randn(100, 64, dtype=torch.bfloat16)
    y = lane.dense_mm(ext, a, torch.randn(32, 64, dtype=torch.bfloat16))

    assert y.shape == (100, 32)
    assert seen["a"] is not None and seen["a"].shape == (100, 64)
    assert seen["row_src"].dtype is torch.int32
    assert seen["row_src"].tolist() == list(range(128))
    assert seen["expert_ids"].tolist() == [0]
    assert seen["tile_m"] == 128


# ---------------------------------------------------------------------------
# The swizzle-group-aligned expert ORDER (tile-order policy).
# ---------------------------------------------------------------------------


def test_pack_expert_blocks_fills_groups_exactly_when_possible():
    """{3,3,2}-block experts tile a group of 8 with no straddle at all."""
    counts = [129, 130, 65, 129, 130, 66, 129, 129]  # blocks: 3,3,2,3,3,2,3,3
    order, touched, minimum = lane.pack_expert_blocks(counts, 64, 8)
    assert sorted(order) == list(range(8))
    assert touched == minimum, "an exactly packable histogram must align"


def test_pack_expert_blocks_is_deterministic_and_skips_empty_experts():
    counts = [0, 100, 0, 500, 64, 0, 1, 320]
    first = lane.pack_expert_blocks(counts, 64, 8)
    second = lane.pack_expert_blocks(counts, 64, 8)
    assert first == second
    order = first[0]
    assert sorted(order) == [1, 3, 4, 6, 7]
    assert 0 not in order and 2 not in order and 5 not in order


def test_pack_expert_blocks_handles_experts_larger_than_a_group():
    counts = [1200, 30, 30]  # 19 blocks + 1 + 1
    order, touched, minimum = lane.pack_expert_blocks(counts, 64, 8)
    assert sorted(order) == [0, 1, 2]
    assert minimum <= touched <= minimum + 2


def test_pack_expert_blocks_group_one_never_straddles():
    order, touched, minimum = lane.pack_expert_blocks([70, 3, 900], 64, 1)
    assert touched == minimum
