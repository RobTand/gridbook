"""CPU-only gates for the TCQ/CB matched-expand benchmark accounting."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from gridbook import trellis


def _script_path() -> Path:
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        candidate = root / "scripts" / "bench_trellis_r256.py"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("bench_trellis_r256.py")


SCRIPT = _script_path()
SPEC = importlib.util.spec_from_file_location("bench_trellis_r256", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def test_nvfp4_v2_storage_accounting_is_side_inclusive(monkeypatch):
    monkeypatch.setattr(bench, "cb_reader_rungs", lambda _family: (18,))
    got = bench.cb_storage_accounting(
        trellis.TCQ_E2M1_R256, 18, rows=1024, columns=4096)

    assert got["type_size_bytes_per_256"] == 4 * 18 + 9
    assert got["qweight_bytes"] == 1024 * 16 * (4 * 18 + 9)
    # Two product tables, each (2^9, 4), serialized as normative fp16.
    assert got["shared_codebook_fp16_bytes_full_charge"] == 2 * 512 * 4 * 2
    assert got["fp8_row_scale_bytes"] == 0
    assert got["total_bytes"] == 1_335_296
    assert got["total_stored_bits"] == got["total_bytes"] * 8
    assert got["exact_bpw"] == got["total_stored_bits"] / (1024 * 4096)


def test_fp8_storage_accounting_charges_separate_scale(monkeypatch):
    monkeypatch.setattr(bench, "cb_reader_rungs", lambda _family: (32,))
    got = bench.cb_storage_accounting(
        trellis.TCQ_E4M3_R256, 32, rows=1024, columns=4096)

    assert got["type_size_bytes_per_256"] == 4 * 32
    assert got["qweight_bytes"] == 1024 * 16 * (4 * 32)
    assert got["fp8_row_scale_bytes"] == 1024 * 4
    # Four product tables, each (2^8, 2), serialized as normative fp16.
    assert got["shared_codebook_fp16_bytes_full_charge"] == 4 * 256 * 2 * 2
    assert got["total_bytes"] == 2_105_344
    assert got["exact_bpw"] == got["total_stored_bits"] / (1024 * 4096)


@pytest.mark.parametrize(
    ("family", "rungs", "target", "expected"),
    (
        (trellis.TCQ_E2M1_R256, (16, 18, 20), 1_312_867, 18),
        (trellis.TCQ_E4M3_R256, (28, 32, 36), 2_103_419, 32),
    ),
)
def test_match_is_nearest_exact_total_bytes(
    monkeypatch, family, rungs, target, expected,
):
    monkeypatch.setattr(bench, "cb_reader_rungs", lambda _family: rungs)
    got = bench.select_matched_cb_rung(
        family, target, rows=1024, columns=4096)
    assert got["k_bits_per_vec8"] == expected


def test_cb_comparator_refuses_fake_short_tail(monkeypatch):
    monkeypatch.setattr(bench, "cb_reader_rungs", lambda _family: (16,))
    with pytest.raises(ValueError, match="positive multiple of 256"):
        bench.cb_storage_accounting(
            trellis.TCQ_E2M1_R256, 16, rows=4, columns=257)


def test_stored_rate_is_decimal_gb_and_requires_positive_time():
    assert bench.stored_decimal_gb_per_s(1_000_000, 1.0) == 1.0
    with pytest.raises(ValueError, match="must be positive"):
        bench.stored_decimal_gb_per_s(1, 0.0)


@pytest.mark.parametrize(
    ("family", "q256", "expected"),
    (
        (trellis.TCQ_E2M1_R256, 512, {1: 128, 3: 128}),
        (trellis.TCQ_E4M3_R256, 1152, {4: 128, 5: 128}),
    ),
)
def test_synthetic_bandwidth_schedule_is_mixed_and_exact(
    family, q256, expected,
):
    schedule = bench.synthetic_importance_schedule(
        family, q256, columns=512, seed=20260825)
    assert len(schedule) == 512
    for start in (0, 256):
        block = schedule[start:start + 256]
        assert sum(block) == q256
        assert {rate: block.count(rate) for rate in sorted(set(block))} == expected
    assert schedule[:256] != schedule[256:]
