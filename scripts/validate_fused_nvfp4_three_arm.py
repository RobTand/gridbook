#!/usr/bin/env python3
"""Same-session baseline/static/rowwise fused-NVFP4 validation.

This schema-v6 harness complements, and does not replace, the two-arm v5
``validate_fused_nvfp4_ab.py`` entry point.  It loads exactly one vLLM engine
and measures three activation/execution contracts in that engine:

* ``baseline``: Gridbook's FP32-emulated group-scale activation QDQ;
* ``static``: artifact-attested global FP32 activation scales plus UE4M3
  group factors; and
* ``rowwise``: an independent runtime scale per activation row plus UE4M3
  group factors.

Every quality and timing prompt is run as a three-arm block.  The order cycles
through all six permutations equally, selector environment variables and
process-lifetime caches are reset and attested around every arm, and dispatch
coverage/fallback gates are reported independently for both fused candidates.
The report contains all three pairwise KL/NLL/PPL and timing comparisons.

The implementation deliberately imports the v5 module as a helper library for
model/dataset provenance, scoring, dispatch probes, and serialization.  This
keeps the established v5 CLI and ``gridbook.fused-nvfp4-ab.v5`` schema intact.

Example (inside the pinned vLLM/CUDA image)::

    python3 scripts/validate_fused_nvfp4_three_arm.py \
      --model /models/nvfp4cb_k24_static \
      --teacher-model /models/Qwen3-0.6B \
      --wikitext-text /hfcache/wikitext-2-raw-test.txt \
      --output /evidence/k24-static-rowwise-v6.json \
      --n-samples 6 --seqlen 128 --top-k 1024 \
      --timing-repeats 3 --teacher-full-vocab-kl \
      --max-mean-kl 1e-4 --max-mean-nll-regression 0.00498754 \
      --max-ppl-relative-regression 0.005 \
      --max-teacher-mean-kl 0.02 \
      --max-teacher-kl-regression 1e-4 \
      --min-timing-speedup 1.10

As in v5, timing is offline one-token request wall time, not streaming TTFT,
and a passing run does not replace the shipped workload matrix.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _load_v5_helpers() -> Any:
    path = Path(__file__).with_name("validate_fused_nvfp4_ab.py")
    module_name = "_gridbook_validate_fused_nvfp4_ab_v5"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load v5 validation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


v5 = _load_v5_helpers()
validation_common = v5.validation_common

SCHEMA = "gridbook.fused-nvfp4-three-arm.v6"
ARMS = ("baseline", "static", "rowwise")
CANDIDATE_ARMS = ("static", "rowwise")
ARM_PERMUTATIONS = tuple(itertools.permutations(ARMS))
PAIR_SPECS = (
    ("baseline_vs_static", "baseline", "static"),
    ("baseline_vs_rowwise", "baseline", "rowwise"),
    ("static_vs_rowwise", "static", "rowwise"),
)


def three_arm_order(block_index: int) -> tuple[str, str, str]:
    """Return one of all six permutations, repeating only after a full cycle."""

    if block_index < 0:
        raise ValueError("block_index must be nonnegative")
    return ARM_PERMUTATIONS[block_index % len(ARM_PERMUTATIONS)]


def permutation_balance(
    orders: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Attest that every valid permutation occurred equally often."""

    expected = {"/".join(order) for order in ARM_PERMUTATIONS}
    counts = Counter("/".join(order) for order in orders)
    unexpected = sorted(set(counts) - expected)
    values = [int(counts[key]) for key in sorted(expected)]
    positive = bool(values) and min(values) > 0
    balanced = positive and len(set(values)) == 1
    return {
        "permutation_counts": {key: int(counts[key]) for key in sorted(expected)},
        "unexpected_orders": unexpected,
        "all_six_present": positive,
        "equal_counts": balanced,
        "pass": balanced and not unexpected,
    }


def _dense_modes(dense_range: str) -> tuple[str, str]:
    if dense_range == "all":
        return "1", "rowwise"
    if dense_range == "midm":
        return "midm", "rowwise_midm"
    raise ValueError("dense_range must be one of: all, midm")


def requested_modes(
    arm: str, *, execution_mode: str, dense_range: str
) -> tuple[str, str]:
    """Map a schema-v6 arm to concrete dense and MoE selector values."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    static_dense, rowwise_dense = _dense_modes(dense_range)
    if execution_mode == "dense":
        return {
            "baseline": ("", ""),
            "static": (static_dense, ""),
            "rowwise": (rowwise_dense, ""),
        }[arm]
    if execution_mode not in ("moe128", "moe256"):
        raise ValueError("execution_mode must be one of: dense, moe128, moe256")
    tile_m = execution_mode.removeprefix("moe")
    return {
        "baseline": ("", ""),
        "static": ("", tile_m),
        "rowwise": ("", f"rowwise{tile_m}"),
    }[arm]


_MISSING = object()


@contextmanager
def scoped_three_arm_selector(
    arm: str,
    *,
    execution_mode: str,
    dense_range: str,
    environ: MutableMapping[str, str],
    dense_mode_cache: list,
    moe_mode_cache: list,
    dense_selector: Callable[[], str],
    moe_selector: Callable[[], str],
    label: str,
    attestations: list[dict[str, Any]],
) -> Iterator[tuple[str, str]]:
    """Reset, select, attest, and exactly restore both runtime selectors."""

    expected_dense, expected_moe = requested_modes(
        arm, execution_mode=execution_mode, dense_range=dense_range
    )
    if expected_moe and v5.PREFILL_ENV in environ:
        raise RuntimeError(
            f"{v5.PREFILL_ENV} must remain unset for grouped-MoE validation"
        )
    env_before = {
        name: environ[name] if name in environ else _MISSING
        for name in (v5.FUSED_ENV, v5.FUSED_MOE_ENV)
    }
    dense_before = list(dense_mode_cache)
    moe_before = list(moe_mode_cache)
    record: dict[str, Any] = {
        "label": label,
        "arm": arm,
        "expected_dense_mode": expected_dense,
        "expected_moe_mode": expected_moe,
        "dense_cache_before": list(dense_before),
        "moe_cache_before": list(moe_before),
        "cache_reset_before_selector": False,
        "selector_match": False,
        "exact_restore": False,
        "pass": False,
    }
    try:
        environ[v5.FUSED_ENV] = expected_dense
        environ[v5.FUSED_MOE_ENV] = expected_moe
        dense_mode_cache.clear()
        moe_mode_cache.clear()
        record["cache_reset_before_selector"] = (
            dense_mode_cache == [] and moe_mode_cache == []
        )
        observed_dense = dense_selector()
        observed_moe = moe_selector()
        record.update({
            "observed_dense_mode": observed_dense,
            "observed_moe_mode": observed_moe,
            "dense_cache_after_selector": list(dense_mode_cache),
            "moe_cache_after_selector": list(moe_mode_cache),
        })
        selector_match = (
            observed_dense == expected_dense and observed_moe == expected_moe
        )
        record["selector_match"] = selector_match
        if not selector_match:
            raise RuntimeError(
                f"selector attestation failed for {arm}: expected dense/moe "
                f"{(expected_dense, expected_moe)!r}, observed "
                f"{(observed_dense, observed_moe)!r}"
            )
        yield expected_dense, expected_moe
    finally:
        for name, previous in env_before.items():
            if previous is _MISSING:
                environ.pop(name, None)
            else:
                environ[name] = previous
        dense_mode_cache[:] = dense_before
        moe_mode_cache[:] = moe_before
        exact_restore = (
            dense_mode_cache == dense_before
            and moe_mode_cache == moe_before
            and all(
                (name not in environ if previous is _MISSING
                 else environ.get(name) == previous)
                for name, previous in env_before.items()
            )
        )
        record["exact_restore"] = exact_restore
        record["pass"] = bool(
            record["cache_reset_before_selector"]
            and record["selector_match"]
            and exact_restore
        )
        attestations.append(record)


def _run_generate(
    *,
    llm: Any,
    sampling: Any,
    prompt_ids: list[int],
    arm: str,
    label: str,
    execution_mode: str,
    dense_range: str,
    linear: Any,
    moe: Any,
    probe: Any,
    synchronize: Callable[[], None],
    selector_attestations: list[dict[str, Any]],
) -> tuple[Any, float, dict[str, Any]]:
    with scoped_three_arm_selector(
        arm,
        execution_mode=execution_mode,
        dense_range=dense_range,
        environ=os.environ,
        dense_mode_cache=linear._FP4_FUSED_MODE,
        moe_mode_cache=moe._FUSED_FP4_MOE_STATE,
        dense_selector=linear._fp4_fused_mode,
        moe_selector=moe._requested_fused_fp4_moe_mode,
        label=label,
        attestations=selector_attestations,
    ):
        synchronize()
        started = time.perf_counter()
        # v5 probes intentionally accept only their legacy semantic names.
        # Keep that contract while relabeling the completed record for v6.
        probe_arm = "baseline" if arm == "baseline" else "fused"
        with probe.measurement(probe_arm, label) as raw_record:
            result = llm.generate(
                [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
            )[0]
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_record["arm"] = arm
        return result, elapsed_ms, v5._json_dispatch_record(raw_record)


def _arm_quality(scores: Sequence[Any]) -> dict[str, Any]:
    target_logprobs = [
        value for score in scores for value in score.target_logprobs
    ]
    if not target_logprobs:
        raise RuntimeError("an arm produced no target logprobs")
    coverages = [row.coverage for score in scores for row in score.rows]
    mean_nll = -statistics.fmean(target_logprobs)
    return {
        "tokens_scored": len(target_logprobs),
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "topk_coverage": v5.summarize_values(coverages),
    }


def pairwise_quality_summary(
    arm_scores: Mapping[str, Sequence[Any]], *, kl_mode: str
) -> dict[str, Any]:
    """Compute all three pairwise, bidirectional quality comparisons."""

    comparisons: dict[str, Any] = {}
    for key, reference, candidate in PAIR_SPECS:
        summary = v5._pairwise_score_summary(
            arm_scores[reference],
            arm_scores[candidate],
            reference_name=reference,
            candidate_name=candidate,
            kl_mode=kl_mode,
        )
        summary["comparison_backend_contract"] = (
            "same-process vLLM engine; distinct activation/dispatch arms"
        )
        comparisons[key] = summary
    return {
        "arms": {arm: _arm_quality(arm_scores[arm]) for arm in ARMS},
        "comparisons": comparisons,
        "kl_mode": kl_mode,
        "kl_convention": v5._kl_convention(kl_mode),
    }


def timing_summary(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    arms = {arm: v5.summarize_values(samples[arm]) for arm in ARMS}
    comparisons: dict[str, Any] = {}
    for key, reference, candidate in PAIR_SPECS:
        reference_mean = arms[reference]["mean"]
        candidate_mean = arms[candidate]["mean"]
        comparisons[key] = {
            "reference": reference,
            "candidate": candidate,
            "reference_over_candidate_speedup": (
                float(reference_mean) / float(candidate_mean)
                if reference_mean is not None
                and candidate_mean not in (None, 0.0)
                else None
            ),
            "candidate_over_reference_wall_ratio": (
                float(candidate_mean) / float(reference_mean)
                if candidate_mean is not None
                and reference_mean not in (None, 0.0)
                else None
            ),
        }
    return {
        "metric": "offline_generate_one_token_wall_ms",
        "scope": "scheduler + prefill + logits + one-token output processing",
        "is_streaming_ttft": False,
        "arms": arms,
        "comparisons": comparisons,
    }


def _candidate_dispatch_gates(
    arm: str, dispatch: Mapping[str, Any], *, execution_mode: str
) -> dict[str, Any]:
    gates = {
        "positive_fused_dispatch": {
            "observed_successes": dispatch["fused_successes"],
            "pass": dispatch["fused_successes"] > 0,
        },
        "fused_dispatch_no_errors": {
            "observed_errors": dispatch["fused_errors"],
            "pass": dispatch["fused_errors"] == 0,
        },
        "zero_fallbacks": {
            "observed_fallbacks": dispatch["fused_fallbacks"],
            "pass": dispatch["fused_fallbacks"] == 0,
        },
        "probe_no_errors": {
            "observed_errors": dispatch["probe_errors"],
            "pass": dispatch["probe_errors"] == 0,
        },
    }
    if execution_mode == "dense":
        opportunities = dispatch["candidate_gate_opportunities"]
        attempts = dispatch["fused_attempts"]
        gates["attempted_every_dispatch_opportunity"] = {
            "observed_opportunities": opportunities,
            "observed_attempts": attempts,
            "pass": opportunities > 0 and attempts == opportunities,
        }
        gates["complete_fused_coverage"] = {
            "observed_success_fraction": dispatch["fused_success_fraction"],
            "pass": dispatch["fused_success_fraction"] == 1.0,
        }
    else:
        gates["never_entered_loop"] = {
            "observed_loop_calls": dispatch["loop_calls"],
            "pass": dispatch["loop_calls"] == 0,
        }
    return {"arm": arm, **gates}


def _all_gate_leaves_pass(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "pass" in value:
            return value["pass"] is True
        return all(_all_gate_leaves_pass(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_gate_leaves_pass(item) for item in value)
    return True


def core_integrity_gates(
    *,
    dispatch: Mapping[str, Mapping[str, Any]],
    execution_mode: str,
    selector_attestations: Sequence[Mapping[str, Any]],
    permutation_attestations: Mapping[str, Mapping[str, Any]],
    expected_measurements: int,
) -> dict[str, Any]:
    observed_pids = sorted({
        pid for arm in ARMS for pid in dispatch[arm]["pids"]
    })
    gates: dict[str, Any] = {
        "same_process_dispatch": {
            "expected_pid": os.getpid(),
            "observed_pids": observed_pids,
            "pass": observed_pids == [os.getpid()],
        },
        "baseline_never_entered_fused_dispatch": {
            "observed_attempts": dispatch["baseline"]["fused_attempts"],
            "pass": dispatch["baseline"]["fused_attempts"] == 0,
        },
        "baseline_probe_no_errors": {
            "observed_errors": dispatch["baseline"]["probe_errors"],
            "pass": dispatch["baseline"]["probe_errors"] == 0,
        },
        "selector_cache_attestation": {
            "expected_measurements": expected_measurements,
            "observed_measurements": len(selector_attestations),
            "failed_labels": [
                record["label"]
                for record in selector_attestations
                if record.get("pass") is not True
            ],
            "pass": (
                len(selector_attestations) == expected_measurements
                and all(record.get("pass") is True
                        for record in selector_attestations)
            ),
        },
        "permutation_counterbalance": {
            "phases": dict(permutation_attestations),
            "pass": all(
                record.get("pass") is True
                for record in permutation_attestations.values()
            ),
        },
        "static_dispatch": _candidate_dispatch_gates(
            "static", dispatch["static"], execution_mode=execution_mode
        ),
        "rowwise_dispatch": _candidate_dispatch_gates(
            "rowwise", dispatch["rowwise"], execution_mode=execution_mode
        ),
    }
    if execution_mode != "dense":
        gates.update({
            "baseline_positive_loop_dispatch": {
                "observed_loop_calls": dispatch["baseline"]["loop_calls"],
                "pass": dispatch["baseline"]["loop_calls"] > 0,
            },
            "baseline_loop_no_errors": {
                "observed_errors": dispatch["baseline"]["loop_errors"],
                "pass": dispatch["baseline"]["loop_errors"] == 0,
            },
        })
    return gates


def configured_gates(
    args: argparse.Namespace, report: Mapping[str, Any]
) -> dict[str, Any]:
    gates: dict[str, Any] = {arm: {} for arm in CANDIDATE_ARMS}
    for arm in CANDIDATE_ARMS:
        comparison = report["quality"]["comparisons"][f"baseline_vs_{arm}"]
        arm_gates = gates[arm]
        if args.max_mean_kl is not None:
            observed = comparison["kl_reference_to_candidate"]["mean"]
            arm_gates["max_mean_kl"] = {
                "limit": args.max_mean_kl,
                "observed": observed,
                "pass": observed is not None and observed <= args.max_mean_kl,
            }
        if args.max_mean_nll_regression is not None:
            observed = comparison["mean_nll_delta_candidate_minus_reference"]
            arm_gates["max_mean_nll_regression"] = {
                "limit": args.max_mean_nll_regression,
                "observed": observed,
                "pass": observed <= args.max_mean_nll_regression,
            }
        if args.max_ppl_relative_regression is not None:
            observed = comparison["ppl_candidate_over_reference"] - 1.0
            arm_gates["max_ppl_relative_regression"] = {
                "limit": args.max_ppl_relative_regression,
                "observed": observed,
                "pass": observed <= args.max_ppl_relative_regression,
            }
        if args.min_timing_speedup is not None:
            observed = report["timing"]["comparisons"][
                f"baseline_vs_{arm}"
            ]["reference_over_candidate_speedup"]
            arm_gates["min_timing_speedup"] = {
                "limit": args.min_timing_speedup,
                "observed": observed,
                "pass": (
                    observed is not None and observed >= args.min_timing_speedup
                ),
            }
        if args.max_teacher_mean_kl is not None:
            observed = report["teacher_quality"][arm][
                "kl_reference_to_candidate"
            ]["mean"]
            arm_gates["max_teacher_mean_kl"] = {
                "limit": args.max_teacher_mean_kl,
                "observed": observed,
                "pass": observed is not None and observed <= args.max_teacher_mean_kl,
            }
        if args.max_teacher_kl_regression is not None:
            teacher_quality = report["teacher_quality"]
            observed = (
                teacher_quality[arm]["kl_reference_to_candidate"]["mean"]
                - teacher_quality["baseline"]["kl_reference_to_candidate"]["mean"]
            )
            arm_gates["max_teacher_kl_regression"] = {
                "limit": args.max_teacher_kl_regression,
                "observed": observed,
                "pass": observed <= args.max_teacher_kl_regression,
            }
    return gates


def _configured_limit_values(args: argparse.Namespace) -> tuple[Any, ...]:
    return (
        args.max_mean_kl,
        args.max_mean_nll_regression,
        args.max_ppl_relative_regression,
        args.max_teacher_mean_kl,
        args.max_teacher_kl_regression,
        args.min_timing_speedup,
    )


def _finalize_report(
    args: argparse.Namespace,
    report: dict[str, Any],
    core_gates: Mapping[str, Any],
) -> None:
    measurement_valid = _all_gate_leaves_pass(core_gates)
    gates = configured_gates(args, report)
    has_thresholds = any(
        value is not None for value in _configured_limit_values(args)
    )
    gates_pass = (
        None if args.measurement_only
        else _all_gate_leaves_pass(gates) if has_thresholds else None
    )
    teacher_thresholds = (
        args.max_teacher_mean_kl is not None
        and args.max_teacher_kl_regression is not None
    )
    promotion_contract = {
        "requires_exact_full_vocab_teacher": True,
        "teacher_present": args.teacher_model is not None,
        "teacher_full_vocab_kl": bool(args.teacher_full_vocab_kl),
        "requires_both_teacher_quality_thresholds": True,
        "teacher_quality_thresholds_present": teacher_thresholds,
        "both_candidates_independently_gated": True,
        "complete": bool(
            not args.measurement_only
            and args.teacher_model is not None
            and args.teacher_full_vocab_kl
            and teacher_thresholds
        ),
    }
    report.update({
        "measurement_only": bool(args.measurement_only),
        "measurement_valid": measurement_valid,
        "configured_gates": gates,
        "configured_gates_pass": gates_pass,
        "promotion_contract": promotion_contract,
    })
    if not measurement_valid or gates_pass is False:
        report["status"] = "gate_failed"
        report["promotion_recommendation"] = "do_not_enable"
    elif args.measurement_only:
        report["status"] = "measurement_only"
        report["promotion_recommendation"] = "measurement_only"
    elif not promotion_contract["complete"]:
        report["status"] = "screening_only"
        report["promotion_recommendation"] = "teacher_gate_incomplete"
    else:
        report["status"] = "ok"
        report["promotion_recommendation"] = "candidate_only_requires_served_validation"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if "vllm" in sys.modules:
        raise RuntimeError(
            "vLLM was imported before the harness could force same-process "
            f"execution; launch {Path(__file__).name} in a fresh Python process"
        )
    os.environ[v5.VLLM_MP_ENV] = "0"
    os.environ[v5.FUSED_ENV] = ""
    os.environ[v5.FUSED_MOE_ENV] = ""
    inherited_prefill = os.environ.get(v5.PREFILL_ENV)
    if args.mode != "dense":
        os.environ.pop(v5.PREFILL_ENV, None)
    started = time.monotonic()
    bootstrap = validation_common.prepare_validation(
        args,
        harness_path=Path(__file__),
        helpers=v5,
        extension_none_message=(
            "fused FP4 extension did not build/load; refusing fallback validation"
        ),
    )
    torch = bootstrap.torch
    linear = bootstrap.linear
    moe = bootstrap.moe
    runtime = bootstrap.runtime
    extension = bootstrap.extension
    from gridbook import nvfp4_activation_contract

    candidate_vocab_size = bootstrap.candidate_vocab_size
    candidate_artifact_provenance = bootstrap.candidate_artifact_provenance
    prompts = bootstrap.prompts
    dataset = bootstrap.dataset
    quality_kl_mode = bootstrap.quality_kl_mode
    static_dense_mode, _rowwise_dense_mode = _dense_modes(args.dense_range)
    probe = (
        v5.DenseDispatchProbe(
            linear.PrismaQuantCBLinearMethod,
            prefill_threshold=linear.PREFILL_M_THRESHOLD,
            fused_mode=static_dense_mode,
        )
        if args.mode == "dense"
        else v5.MoEDispatchProbe(moe.PrismaQuantCBMoEMethod)
    )
    probe.install()
    engine = validation_common.load_candidate_engine(
        bootstrap, args, probe=probe
    )
    llm = engine.llm
    quality_sampling = engine.quality_sampling
    timing_sampling = engine.timing_sampling
    model_load_s = engine.model_load_seconds
    dispatch_records: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] = []
    selector_attestations: list[dict[str, Any]] = []
    quality_blocks: list[dict[str, Any]] = []
    timing_samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    quality_orders: list[tuple[str, str, str]] = []
    warmup_orders: list[tuple[str, str, str]] = []
    timing_orders: list[tuple[str, str, str]] = []
    synchronize = torch.cuda.synchronize

    try:
        for cycle in range(args.warmup_cycles):
            for permutation_index in range(len(ARM_PERMUTATIONS)):
                order = three_arm_order(permutation_index)
                warmup_orders.append(order)
                prompt_ids = prompts[
                    (cycle * len(ARM_PERMUTATIONS) + permutation_index)
                    % len(prompts)
                ]
                for arm in order:
                    _result, elapsed_ms, record = _run_generate(
                        llm=llm,
                        sampling=timing_sampling,
                        prompt_ids=prompt_ids,
                        arm=arm,
                        label=f"warmup:{cycle}:{permutation_index}:{arm}",
                        execution_mode=args.mode,
                        dense_range=args.dense_range,
                        linear=linear,
                        moe=moe,
                        probe=probe,
                        synchronize=synchronize,
                        selector_attestations=selector_attestations,
                    )
                    record["wall_ms"] = elapsed_ms
                    warmup_records.append(record)
                    del _result

        # Exact-vocabulary quality materializes very large host objects. Run
        # timing first so deferred GC/lazy host work cannot land on one arm.
        for phase in validation_common.measurement_phase_order(
            args.timing_repeats
        ):
            if phase == "timing":
                validation_common.quiesce_before_timing(torch)
                for repeat in range(args.timing_repeats):
                    for prompt_index, prompt_ids in enumerate(prompts):
                        order = three_arm_order(
                            repeat * len(prompts) + prompt_index
                        )
                        timing_orders.append(order)
                        for arm in order:
                            _output, elapsed_ms, record = _run_generate(
                                llm=llm,
                                sampling=timing_sampling,
                                prompt_ids=prompt_ids,
                                arm=arm,
                                label=(
                                    f"timing:{repeat}:{prompt_index}:{arm}"
                                ),
                                execution_mode=args.mode,
                                dense_range=args.dense_range,
                                linear=linear,
                                moe=moe,
                                probe=probe,
                                synchronize=synchronize,
                                selector_attestations=selector_attestations,
                            )
                            timing_samples[arm].append(elapsed_ms)
                            record["wall_ms"] = elapsed_ms
                            dispatch_records.append(record)
                            del _output
                continue
            if phase != "quality":
                raise RuntimeError(f"unknown measurement phase {phase!r}")
            for prompt_index, prompt_ids in enumerate(prompts):
                order = three_arm_order(prompt_index)
                quality_orders.append(order)
                scores: dict[str, Any] = {}
                walls: dict[str, float] = {}
                for arm in order:
                    output, elapsed_ms, record = _run_generate(
                        llm=llm,
                        sampling=quality_sampling,
                        prompt_ids=prompt_ids,
                        arm=arm,
                        label=f"quality:{prompt_index}:{arm}",
                        execution_mode=args.mode,
                        dense_range=args.dense_range,
                        linear=linear,
                        moe=moe,
                        probe=probe,
                        synchronize=synchronize,
                        selector_attestations=selector_attestations,
                    )
                    scores[arm] = v5.score_prompt_output(
                        output,
                        prompt_ids,
                        args.top_k,
                        full_vocab=args.teacher_full_vocab_kl,
                        expected_vocab_size=(
                            candidate_vocab_size
                            if args.teacher_full_vocab_kl
                            else None
                        ),
                    )
                    walls[arm] = elapsed_ms
                    record["wall_ms"] = elapsed_ms
                    dispatch_records.append(record)
                    del output
                quality_blocks.append({
                    "prompt_index": prompt_index,
                    "arm_order": list(order),
                    "scores": scores,
                    "wall_ms": walls,
                })
    finally:
        probe.restore()

    arm_scores = {
        arm: [block["scores"][arm] for block in quality_blocks]
        for arm in ARMS
    }
    quality = pairwise_quality_summary(
        arm_scores, kl_mode=quality_kl_mode
    )
    quality["per_prompt"] = [
        {
            "prompt_index": block["prompt_index"],
            "arm_order": block["arm_order"],
            "wall_ms": block["wall_ms"],
            "mean_nll": {
                arm: block["scores"][arm].mean_nll for arm in ARMS
            },
        }
        for block in quality_blocks
    ]

    teacher_quality, teacher_record = validation_common.score_teacher(
        bootstrap,
        args,
        arm_scores=arm_scores,
        arms=ARMS,
        helpers=v5,
    )
    if teacher_quality is not None:
        baseline_teacher_kl = teacher_quality["baseline"][
            "kl_reference_to_candidate"
        ]["mean"]
        teacher_quality["candidate_deltas_vs_baseline"] = {
            arm: {
                "teacher_to_candidate_mean_kl_minus_baseline": (
                    teacher_quality[arm]["kl_reference_to_candidate"]["mean"]
                    - baseline_teacher_kl
                ),
                "kl_mode": quality_kl_mode,
            }
            for arm in CANDIDATE_ARMS
        }

    dispatch = {
        arm: v5.aggregate_dispatch(
            [record for record in dispatch_records if record["arm"] == arm]
        )
        for arm in ARMS
    }
    permutation_attestations = {
        "warmup": permutation_balance(warmup_orders),
        "quality": permutation_balance(quality_orders),
    }
    if args.timing_repeats:
        permutation_attestations["timing"] = permutation_balance(timing_orders)
    expected_measurements = (
        args.warmup_cycles * len(ARM_PERMUTATIONS) * len(ARMS)
        + len(prompts) * len(ARMS)
        + args.timing_repeats * len(prompts) * len(ARMS)
    )
    core_gates = core_integrity_gates(
        dispatch=dispatch,
        execution_mode=args.mode,
        selector_attestations=selector_attestations,
        permutation_attestations=permutation_attestations,
        expected_measurements=expected_measurements,
    )
    kl_limitation = (
        "Exact full-vocabulary KL was requested and cardinality-attested."
        if args.teacher_full_vocab_kl
        else "Top-K KL is a coarse-support rejection screen and cannot make "
        "this report promotion-eligible."
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": v5._utc_now(),
        "scope": f"{args.mode} NVFP4-CB prefill; TP=1; one in-process vLLM engine",
        "activation_contract": {
            "baseline": "fp32-emulated group-scale FP4 activation QDQ",
            "static": "artifact-attested global FP32 scale + UE4M3 factors",
            "rowwise": "independent runtime row scale + UE4M3 factors",
            "rowwise_range_multiplier": (
                nvfp4_activation_contract.rowwise_range_multiplier()
            ),
            "all_same_contract": False,
        },
        "settings": validation_common.shared_report_settings(
            bootstrap,
            args,
            arm_settings={
                "dense_range": args.dense_range,
                "concrete_arm_modes": {
                    arm: {
                        "dense": requested_modes(
                            arm,
                            execution_mode=args.mode,
                            dense_range=args.dense_range,
                        )[0],
                        "moe": requested_modes(
                            arm,
                            execution_mode=args.mode,
                            dense_range=args.dense_range,
                        )[1],
                    }
                    for arm in ARMS
                },
            },
            inherited_prefill=inherited_prefill,
            prefill_threshold=linear.PREFILL_M_THRESHOLD,
            measurement_settings={
                "warmup_cycles": args.warmup_cycles,
                "timing_repeats": args.timing_repeats,
            },
        ),
        "runtime": runtime,
        "extension": extension,
        "candidate_artifact_provenance": candidate_artifact_provenance,
        "dataset": dataset,
        "model_load_seconds": model_load_s,
        "warmup_dispatch": warmup_records,
        "quality": quality,
        "teacher": teacher_record,
        "teacher_quality": teacher_quality,
        "dispatch": {
            **dispatch,
            "per_request": dispatch_records,
            "unscoped": v5._json_dispatch_record(probe.unscoped),
        },
        "selector_attestations": selector_attestations,
        "permutation_attestations": permutation_attestations,
        "core_integrity_gates": core_gates,
        "limitations": [
            kl_limitation,
            "Timing is offline one-token request wall time, not served TTFT.",
            (
                "Dense mode does not validate grouped-MoE routing or padding cliffs."
                if args.mode == "dense"
                else "One TileM does not cover the routed-token/padding ladder."
            ),
            "A passing run does not replace served KL/PPL/tasks and workload gates.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.teacher_model is not None:
        report["limitations"].append(
            "Teacher comparisons cross Transformers and vLLM runtimes."
        )
    if args.timing_repeats:
        report["timing"] = timing_summary(timing_samples)
    _finalize_report(args, report, core_gates)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation_common.add_shared_cli_arguments(
        parser, helpers=v5, n_samples_default=6
    )
    parser.add_argument(
        "--dense-range",
        choices=("all", "midm"),
        default="all",
        help="use 1/rowwise for all prefill M, or midm/rowwise_midm through M=128",
    )
    parser.add_argument("--warmup-cycles", type=v5._positive_int, default=1)
    validation_common.add_shared_measurement_cli_arguments(
        parser, helpers=v5
    )
    parser.add_argument("--max-mean-kl", type=v5._nonnegative_float)
    parser.add_argument("--max-mean-nll-regression", type=v5._nonnegative_float)
    parser.add_argument("--max-ppl-relative-regression", type=v5._nonnegative_float)
    parser.add_argument("--max-teacher-mean-kl", type=v5._nonnegative_float)
    parser.add_argument("--max-teacher-kl-regression", type=v5._nonnegative_float)
    parser.add_argument("--min-timing-speedup", type=v5._nonnegative_float)
    args = parser.parse_args(argv)
    if args.n_samples % len(ARM_PERMUTATIONS) != 0:
        parser.error("--n-samples must be a positive multiple of 6")
    teacher_gates = (
        args.max_teacher_mean_kl,
        args.max_teacher_kl_regression,
    )
    validation_common.validate_shared_cli_args(
        parser, args, teacher_gate_values=teacher_gates
    )
    has_thresholds = any(
        value is not None for value in _configured_limit_values(args)
    )
    if not has_thresholds and not args.measurement_only:
        parser.error(
            "no thresholds configured; pass --measurement-only for evidence-only use"
        )
    if has_thresholds and args.measurement_only:
        parser.error("--measurement-only cannot be combined with thresholds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - preserve machine-readable failure
        failure = {
            "schema": SCHEMA,
            "created_at": v5._utc_now(),
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
        }
        v5._atomic_json(args.output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        return 1
    v5._atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "measurement_valid": report["measurement_valid"],
        "configured_gates_pass": report["configured_gates_pass"],
        "promotion_contract": report["promotion_contract"],
        "promotion_recommendation": report["promotion_recommendation"],
        "quality_comparisons": report["quality"]["comparisons"],
        "timing": report.get("timing"),
        "dispatch": {
            arm: report["dispatch"][arm] for arm in ARMS
        },
    }, indent=2), flush=True)
    return 0 if report["status"] in ("ok", "measurement_only") else 2


if __name__ == "__main__":
    raise SystemExit(main())
