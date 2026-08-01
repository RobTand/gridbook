"""CPU-only tests for the schema-v6 three-arm NVFP4 harness."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _script_path(name: str) -> Path:
    roots = [Path(__file__).resolve().parents[1]]
    for variable in ("GRIDBOOK_SOURCE_ROOT", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    for root in roots:
        candidate = root / "scripts" / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(name)


SCRIPT = _script_path("validate_fused_nvfp4_three_arm.py")
SPEC = importlib.util.spec_from_file_location(
    "validate_fused_nvfp4_three_arm", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
three = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = three
SPEC.loader.exec_module(three)


def test_v6_import_preserves_v5_schema_and_arms():
    assert three.SCHEMA == "gridbook.fused-nvfp4-three-arm.v6"
    assert three.v5.SCHEMA == "gridbook.fused-nvfp4-ab.v5"
    assert three.v5.ARMS == ("baseline", "fused")
    assert three.ARMS == ("baseline", "static", "rowwise")


def test_all_six_permutations_are_counterbalanced():
    orders = [three.three_arm_order(index) for index in range(12)]
    assert len(set(orders[:6])) == math.factorial(3)
    assert orders[:6] == list(three.ARM_PERMUTATIONS)
    assert orders[6:] == orders[:6]
    balance = three.permutation_balance(orders)
    assert balance["all_six_present"] is True
    assert balance["equal_counts"] is True
    assert balance["pass"] is True
    assert set(balance["permutation_counts"].values()) == {2}
    with pytest.raises(ValueError, match="nonnegative"):
        three.three_arm_order(-1)


def test_permutation_balance_fails_missing_or_skewed_orders():
    missing = three.permutation_balance(three.ARM_PERMUTATIONS[:-1])
    assert missing["all_six_present"] is False
    assert missing["pass"] is False
    skewed = three.permutation_balance(
        [*three.ARM_PERMUTATIONS, three.ARM_PERMUTATIONS[0]]
    )
    assert skewed["all_six_present"] is True
    assert skewed["equal_counts"] is False
    assert skewed["pass"] is False


@pytest.mark.parametrize(
    ("arm", "mode", "dense_range", "expected"),
    (
        ("baseline", "dense", "all", ("", "")),
        ("static", "dense", "all", ("1", "")),
        ("rowwise", "dense", "all", ("rowwise", "")),
        ("static", "dense", "midm", ("midm", "")),
        ("rowwise", "dense", "midm", ("rowwise_midm", "")),
        ("baseline", "moe128", "all", ("", "")),
        ("static", "moe128", "all", ("", "128")),
        ("rowwise", "moe128", "all", ("", "rowwise128")),
        ("static", "moe256", "all", ("", "256")),
        ("rowwise", "moe256", "all", ("", "rowwise256")),
    ),
)
def test_requested_modes_are_concrete_and_disjoint(
    arm, mode, dense_range, expected
):
    assert three.requested_modes(
        arm, execution_mode=mode, dense_range=dense_range
    ) == expected


def _cached_selector(environ, name, cache):
    def select():
        current = environ.get(name, "").strip()
        if not cache:
            cache.append(current)
        elif cache[0] != current:
            raise RuntimeError("selector changed without reset")
        return cache[0]

    return select


@pytest.mark.parametrize("arm", three.ARMS)
def test_scoped_selector_resets_attests_and_exactly_restores(arm):
    environ = {
        three.v5.FUSED_ENV: "stale-dense",
        three.v5.FUSED_MOE_ENV: "stale-moe",
    }
    dense_cache = ["stale-dense"]
    moe_cache = ["stale-moe"]
    attestations = []
    with three.scoped_three_arm_selector(
        arm,
        execution_mode="dense",
        dense_range="all",
        environ=environ,
        dense_mode_cache=dense_cache,
        moe_mode_cache=moe_cache,
        dense_selector=_cached_selector(
            environ, three.v5.FUSED_ENV, dense_cache
        ),
        moe_selector=_cached_selector(
            environ, three.v5.FUSED_MOE_ENV, moe_cache
        ),
        label=f"test:{arm}",
        attestations=attestations,
    ) as observed:
        assert observed == three.requested_modes(
            arm, execution_mode="dense", dense_range="all"
        )
        assert dense_cache == [observed[0]]
        assert moe_cache == [observed[1]]
    assert environ[three.v5.FUSED_ENV] == "stale-dense"
    assert environ[three.v5.FUSED_MOE_ENV] == "stale-moe"
    assert dense_cache == ["stale-dense"]
    assert moe_cache == ["stale-moe"]
    assert len(attestations) == 1
    assert attestations[0]["cache_reset_before_selector"] is True
    assert attestations[0]["selector_match"] is True
    assert attestations[0]["exact_restore"] is True
    assert attestations[0]["pass"] is True


def test_scoped_selector_restores_missing_env_and_cache_on_error():
    environ = {}
    dense_cache = []
    moe_cache = []
    attestations = []
    with pytest.raises(RuntimeError, match="expected failure"):
        with three.scoped_three_arm_selector(
            "rowwise",
            execution_mode="moe256",
            dense_range="all",
            environ=environ,
            dense_mode_cache=dense_cache,
            moe_mode_cache=moe_cache,
            dense_selector=_cached_selector(
                environ, three.v5.FUSED_ENV, dense_cache
            ),
            moe_selector=_cached_selector(
                environ, three.v5.FUSED_MOE_ENV, moe_cache
            ),
            label="test:error",
            attestations=attestations,
        ):
            raise RuntimeError("expected failure")
    assert three.v5.FUSED_ENV not in environ
    assert three.v5.FUSED_MOE_ENV not in environ
    assert dense_cache == []
    assert moe_cache == []
    assert attestations[0]["pass"] is True


def test_moe_selector_refuses_legacy_prefill_override():
    environ = {three.v5.PREFILL_ENV: "loop"}
    with pytest.raises(RuntimeError, match="must remain unset"):
        with three.scoped_three_arm_selector(
            "static",
            execution_mode="moe128",
            dense_range="all",
            environ=environ,
            dense_mode_cache=[],
            moe_mode_cache=[],
            dense_selector=lambda: "",
            moe_selector=lambda: "128",
            label="test:prefill",
            attestations=[],
        ):
            pass


def _score(target_logprob: float, token0_logprob: float):
    other = math.log1p(-math.exp(token0_logprob))
    row = three.v5.TopKRow(
        token_ids=(0, 1), logprobs=(token0_logprob, other)
    )
    return three.v5.PromptScore(
        target_logprobs=(target_logprob,), rows=(row,)
    )


def test_quality_summary_contains_all_three_pairwise_directions():
    scores = {
        "baseline": [_score(-1.0, math.log(0.70))],
        "static": [_score(-1.1, math.log(0.65))],
        "rowwise": [_score(-1.05, math.log(0.68))],
    }
    result = three.pairwise_quality_summary(
        scores, kl_mode=three.v5.KL_COARSE_TOPK
    )
    assert set(result["arms"]) == set(three.ARMS)
    assert set(result["comparisons"]) == {
        "baseline_vs_static",
        "baseline_vs_rowwise",
        "static_vs_rowwise",
    }
    baseline_static = result["comparisons"]["baseline_vs_static"]
    assert baseline_static["reference"] == "baseline"
    assert baseline_static["candidate"] == "static"
    assert baseline_static["mean_nll_delta_candidate_minus_reference"] == pytest.approx(
        0.1
    )
    assert baseline_static["ppl_candidate_over_reference"] == pytest.approx(
        math.exp(0.1)
    )
    assert baseline_static["kl_reference_to_candidate"]["mean"] > 0
    assert baseline_static["kl_candidate_to_reference"]["mean"] > 0
    assert "same-process vLLM" in baseline_static["comparison_backend_contract"]


def test_timing_summary_reports_all_pairwise_ratios():
    result = three.timing_summary({
        "baseline": [12.0, 8.0],
        "static": [5.0, 5.0],
        "rowwise": [4.0, 4.0],
    })
    assert result["comparisons"]["baseline_vs_static"][
        "reference_over_candidate_speedup"
    ] == pytest.approx(2.0)
    assert result["comparisons"]["baseline_vs_rowwise"][
        "reference_over_candidate_speedup"
    ] == pytest.approx(2.5)
    assert result["comparisons"]["static_vs_rowwise"][
        "reference_over_candidate_speedup"
    ] == pytest.approx(1.25)


def _dispatch(*, success=3, fallback=0, opportunities=3, attempts=3):
    return {
        "fused_successes": success,
        "fused_fallbacks": fallback,
        "fused_errors": 0,
        "fused_attempts": attempts,
        "candidate_gate_opportunities": opportunities,
        "fused_success_fraction": success / attempts if attempts else None,
        "probe_errors": 0,
        "loop_calls": 0,
        "loop_errors": 0,
        "pids": [os.getpid()],
    }


def test_static_and_rowwise_dispatch_are_independent_hard_gates():
    dispatch = {
        "baseline": _dispatch(success=0, opportunities=0, attempts=0),
        "static": _dispatch(),
        "rowwise": _dispatch(success=2, fallback=1),
    }
    selector_attestations = [{"label": "one", "pass": True}]
    permutations = {
        "quality": {"pass": True},
        "warmup": {"pass": True},
    }
    gates = three.core_integrity_gates(
        dispatch=dispatch,
        execution_mode="dense",
        selector_attestations=selector_attestations,
        permutation_attestations=permutations,
        expected_measurements=1,
    )
    assert gates["static_dispatch"]["zero_fallbacks"]["pass"] is True
    assert gates["static_dispatch"]["complete_fused_coverage"]["pass"] is True
    assert gates["rowwise_dispatch"]["zero_fallbacks"]["pass"] is False
    assert gates["rowwise_dispatch"]["complete_fused_coverage"]["pass"] is False
    assert three._all_gate_leaves_pass(gates) is False


def test_configured_quality_gate_is_applied_to_both_candidates():
    args = SimpleNamespace(
        max_mean_kl=0.2,
        max_mean_nll_regression=0.2,
        max_ppl_relative_regression=0.25,
        max_teacher_mean_kl=None,
        max_teacher_kl_regression=None,
        min_timing_speedup=None,
    )
    report = {
        "quality": {
            "comparisons": {
                "baseline_vs_static": {
                    "kl_reference_to_candidate": {"mean": 0.1},
                    "mean_nll_delta_candidate_minus_reference": 0.1,
                    "ppl_candidate_over_reference": 1.1,
                },
                "baseline_vs_rowwise": {
                    "kl_reference_to_candidate": {"mean": 0.3},
                    "mean_nll_delta_candidate_minus_reference": 0.1,
                    "ppl_candidate_over_reference": 1.1,
                },
            }
        }
    }
    gates = three.configured_gates(args, report)
    assert gates["static"]["max_mean_kl"]["pass"] is True
    assert gates["rowwise"]["max_mean_kl"]["pass"] is False
    assert three._all_gate_leaves_pass(gates) is False


def test_parse_requires_six_prompt_balance_and_explicit_measurement_mode(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "result.json"
    args = three.parse_args([
        "--model", str(model),
        "--output", str(output),
        "--n-samples", "6",
        "--seqlen", "32",
        "--measurement-only",
    ])
    assert args.n_samples == 6
    with pytest.raises(SystemExit):
        three.parse_args([
            "--model", str(model),
            "--output", str(output),
            "--n-samples", "5",
            "--seqlen", "32",
            "--measurement-only",
        ])
    with pytest.raises(SystemExit):
        three.parse_args([
            "--model", str(model),
            "--output", str(output),
            "--n-samples", "6",
            "--seqlen", "32",
        ])


def test_entry_point_help_is_cpu_only_and_does_not_import_vllm():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "baseline/static/rowwise" in completed.stdout
    assert "--dense-range" in completed.stdout
