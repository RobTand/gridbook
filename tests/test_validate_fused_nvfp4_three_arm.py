"""CPU-only tests for the schema-v6 three-arm NVFP4 harness."""

from __future__ import annotations

import importlib.util
import inspect
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


def test_v6_runner_uses_shared_timing_before_quality_contract():
    assert three.validation_common is three.v5.validation_common
    assert three.validation_common.measurement_phase_order(2) == (
        "timing",
        "quality",
    )
    assert "validation_common.measurement_phase_order" in inspect.getsource(
        three.run
    )


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


def _fake_llm_chunked_contract(*, resolved, supported, runner_type="generate"):
    model_config = SimpleNamespace(
        is_chunked_prefill_supported=supported,
        runner_type=runner_type,
    )
    scheduler_config = SimpleNamespace(enable_chunked_prefill=resolved)
    return SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=model_config,
                scheduler_config=scheduler_config,
            )
        )
    )


def test_chunked_prefill_attestation_uses_resolved_vllm_configs():
    common = three.validation_common
    auto = common.attest_chunked_prefill_contract(
        _fake_llm_chunked_contract(resolved=True, supported=True), None
    )
    assert auto == {
        "requested": "auto",
        "engine_kwarg_omitted": True,
        "resolved_enabled": True,
        "model_official_default_enabled": True,
        "model_runner_type": "generate",
        "requested_matches_resolved": None,
        "explicit_override_conflicts_with_official_contract": False,
        "would_trigger_vllm_warning": False,
        "promotion_compatible": True,
        "attestation_source": (
            "LLM.llm_engine.vllm_config.{scheduler_config."
            "enable_chunked_prefill,model_config."
            "is_chunked_prefill_supported,model_config.runner_type}"
        ),
    }

    warned_disable = common.attest_chunked_prefill_contract(
        _fake_llm_chunked_contract(resolved=False, supported=True), False
    )
    assert warned_disable["requested"] == "disable"
    assert warned_disable["requested_matches_resolved"] is True
    assert warned_disable[
        "explicit_override_conflicts_with_official_contract"
    ] is True
    assert warned_disable["would_trigger_vllm_warning"] is True
    assert warned_disable["promotion_compatible"] is False

    overridden = common.attest_chunked_prefill_contract(
        _fake_llm_chunked_contract(resolved=False, supported=True), True
    )
    assert overridden["requested_matches_resolved"] is False
    assert overridden["promotion_compatible"] is False


def test_chunked_prefill_attestation_fails_closed_without_stable_properties():
    with pytest.raises(RuntimeError, match="scheduler config"):
        three.validation_common.attest_chunked_prefill_contract(
            SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=object())),
            None,
        )
    with pytest.raises(RuntimeError, match="auto/True/False"):
        three.validation_common.attest_chunked_prefill_contract(
            _fake_llm_chunked_contract(resolved=True, supported=True),
            "auto",
        )


def test_auto_chunked_prefill_omits_kwarg_then_attests_resolved_state():
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            resolved = kwargs.get("enable_chunked_prefill", True)
            self.llm_engine = _fake_llm_chunked_contract(
                resolved=resolved, supported=True
            ).llm_engine

    class FakeSampling:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    bootstrap = SimpleNamespace(
        candidate_load_revision="resolved",
        quality_logprobs=32,
        llm_class=FakeLLM,
        sampling_params_class=FakeSampling,
    )
    args = SimpleNamespace(
        model="candidate",
        trust_remote_code=False,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        seqlen=64,
        enable_chunked_prefill=None,
        seed=0,
        quantization=None,
        max_num_batched_tokens=None,
    )
    engine = three.validation_common.load_candidate_engine(
        bootstrap,
        args,
        probe=SimpleNamespace(restore=lambda: None),
        attest_chunked_prefill=True,
    )
    assert "enable_chunked_prefill" not in calls[0]
    assert engine.chunked_prefill_contract["requested"] == "auto"
    assert engine.chunked_prefill_contract["resolved_enabled"] is True
    assert engine.chunked_prefill_contract["promotion_compatible"] is True


def test_report_records_requested_and_resolved_chunked_prefill(tmp_path):
    bootstrap = SimpleNamespace(
        candidate_path=tmp_path,
        candidate_load_revision=None,
        quality_logprobs=1024,
        quality_kl_mode=three.v5.KL_COARSE_TOPK,
        candidate_vocab_size=128,
    )
    args = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        teacher_model=None,
        teacher_revision=None,
        teacher_dtype="bfloat16",
        teacher_full_vocab_kl=False,
        trust_remote_code=False,
        allow_downloads=False,
        quantization="gridbook",
        dtype="bfloat16",
        mode="dense",
        enable_chunked_prefill=None,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=None,
        top_k=1024,
        seed=0,
    )
    contract = {
        "requested": "auto",
        "resolved_enabled": True,
        "promotion_compatible": True,
    }
    settings = three.validation_common.shared_report_settings(
        bootstrap,
        args,
        arm_settings={"dense_range": "all"},
        inherited_prefill=None,
        prefill_threshold=16,
        measurement_settings={
            "chunked_prefill": contract["resolved_enabled"],
            "chunked_prefill_contract": contract,
        },
    )
    assert settings["chunked_prefill"] is True
    assert settings["chunked_prefill_contract"] == contract


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
        chunked_prefill_contract={"promotion_compatible": True},
    )
    assert gates["static_dispatch"]["zero_fallbacks"]["pass"] is True
    assert gates["static_dispatch"]["complete_fused_coverage"]["pass"] is True
    assert gates["rowwise_dispatch"]["zero_fallbacks"]["pass"] is False
    assert gates["rowwise_dispatch"]["complete_fused_coverage"]["pass"] is False
    assert three._all_gate_leaves_pass(gates) is False


def test_unsupported_chunked_prefill_override_invalidates_measurement():
    dispatch = {
        "baseline": _dispatch(success=0, opportunities=0, attempts=0),
        "static": _dispatch(),
        "rowwise": _dispatch(),
    }
    gates = three.core_integrity_gates(
        dispatch=dispatch,
        execution_mode="dense",
        selector_attestations=[{"label": "one", "pass": True}],
        permutation_attestations={
            "quality": {"pass": True},
            "warmup": {"pass": True},
        },
        expected_measurements=1,
        chunked_prefill_contract={
            "requested": "disable",
            "resolved_enabled": False,
            "would_trigger_vllm_warning": True,
            "promotion_compatible": False,
        },
    )
    contract = gates["chunked_prefill_execution_contract"]
    assert contract["would_trigger_vllm_warning"] is True
    assert contract["pass"] is False
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
    assert args.enable_chunked_prefill is None
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


def test_chunked_prefill_cli_is_explicit_tristate(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    base = [
        "--model", str(model),
        "--output", str(tmp_path / "result.json"),
        "--measurement-only",
    ]
    assert three.parse_args(base).enable_chunked_prefill is None
    assert three.parse_args([
        *base, "--enable-chunked-prefill"
    ]).enable_chunked_prefill is True
    assert three.parse_args([
        *base, "--disable-chunked-prefill"
    ]).enable_chunked_prefill is False
    with pytest.raises(SystemExit):
        three.parse_args([
            *base,
            "--enable-chunked-prefill",
            "--disable-chunked-prefill",
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
