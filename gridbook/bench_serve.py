"""Reproducible online serving benchmarks for native/Gridbook parity work.

This module is intentionally a thin orchestrator around ``vllm bench serve``.
vLLM owns the streaming client and therefore the TTFT, TPOT, ITL, and E2EL
definitions; Gridbook only fixes the workload, separates independent blocks,
and records enough provenance to make two runs comparable.

There are no imports from vLLM (or torch) here.  The benchmark executable is a
subprocess, which keeps ``import gridbook`` usable in CPU-only environments and
makes the command builder/unit tests independent of a live server.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "gridbook.vllm-bench-serve.v1"
STREAMING_METRICS = "ttft,tpot,itl,e2el"
DEFAULT_PERCENTILES = "50,90,95,99"

# These are stable vLLM result names.  Percentile keys are added dynamically
# below because the selected percentiles are configurable.
SUMMARY_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
)
REQUIRED_STREAMING_RESULTS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
)
_PERCENTILE_RESULT = re.compile(
    r"^p(?:\d+(?:\.\d+)?)_(?:ttft|tpot|itl|e2el)_ms$"
)
_SENSITIVE_ENV = re.compile(r"(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.I)


class BenchmarkError(RuntimeError):
    """A benchmark failed before it produced a trustworthy block."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _request_rate(value: str) -> str:
    if value.lower() in {"inf", "infinity"}:
        return "inf"
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number or 'inf'") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise argparse.ArgumentTypeError("must be a positive number or 'inf'")
    return value


def _percentiles(value: str) -> str:
    try:
        parsed = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of numbers"
        ) from exc
    if not parsed or any(not 0 < item < 100 for item in parsed):
        raise argparse.ArgumentTypeError("every percentile must be between 0 and 100")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("percentiles must not contain duplicates")
    return ",".join(f"{item:g}" for item in parsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridbook-bench-serve",
        description=(
            "Run fixed-shape, streaming vLLM serving benchmarks in independent "
            "blocks and save metrics plus reproducibility metadata as JSON."
        ),
    )
    server = parser.add_argument_group("online server")
    server.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    server.add_argument(
        "--backend",
        default="openai",
        help="vLLM bench serve backend (default: openai)",
    )
    server.add_argument(
        "--endpoint", default="/v1/completions", help="streaming completion endpoint"
    )
    server.add_argument(
        "--served-model-name",
        help="model name sent to the API (defaults to --model)",
    )
    server.add_argument(
        "--server-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "exact server argument to record as metadata; repeat as needed "
            "(this does not start or modify the server)"
        ),
    )
    server.add_argument(
        "--ready-timeout",
        type=_nonnegative_int,
        default=600,
        metavar="SECONDS",
        help="vLLM endpoint readiness timeout (default: 600)",
    )

    artifact = parser.add_argument_group("artifact identity")
    artifact.add_argument(
        "--model",
        required=True,
        help="model/tokenizer name understood by vLLM bench serve",
    )
    artifact.add_argument(
        "--tokenizer",
        help="tokenizer name or revision (defaults to --model)",
    )
    artifact.add_argument(
        "--model-id",
        required=True,
        help="exact served artifact identifier/revision recorded in the report",
    )
    artifact.add_argument(
        "--image-id",
        required=True,
        help="exact serving image tag or digest recorded in the report",
    )
    artifact.add_argument(
        "--git-commit",
        help="Gridbook commit override; otherwise detected from this checkout",
    )
    artifact.add_argument(
        "--run-label",
        required=True,
        help="short arm/workload label, for example gridbook-27b-decode",
    )

    workload = parser.add_argument_group("fixed workload")
    workload.add_argument("--input-len", type=_positive_int, required=True)
    workload.add_argument("--output-len", type=_positive_int, required=True)
    workload.add_argument(
        "--num-prompts",
        type=_positive_int,
        required=True,
        help="measured prompts in each block",
    )
    workload.add_argument(
        "--max-concurrency", type=_positive_int, required=True
    )
    workload.add_argument(
        "--warmups",
        type=_nonnegative_int,
        default=2,
        help="unmeasured warmup prompts before each block (default: 2)",
    )
    workload.add_argument(
        "--blocks",
        type=_positive_int,
        default=3,
        help="independent client blocks; distinct deterministic seed per block (default: 3)",
    )
    workload.add_argument("--seed", type=int, default=1234)
    workload.add_argument(
        "--request-rate",
        type=_request_rate,
        default="inf",
        help="request arrival rate or 'inf' for a closed saturation block (default: inf)",
    )
    workload.add_argument(
        "--percentiles",
        type=_percentiles,
        default=DEFAULT_PERCENTILES,
        help=f"reported metric percentiles (default: {DEFAULT_PERCENTILES})",
    )

    dispatch = parser.add_argument_group("dispatch provenance")
    dispatch.add_argument(
        "--dispatch-env",
        action="append",
        default=[],
        metavar="NAME",
        help="environment variable to record; repeat as needed",
    )
    dispatch.add_argument(
        "--dispatch-env-prefix",
        action="append",
        default=["PRISMAQUANT_"],
        metavar="PREFIX",
        help=(
            "record variables matching this prefix; repeat to add prefixes "
            "(PRISMAQUANT_ is always included)"
        ),
    )

    output = parser.add_argument_group("runner and output")
    output.add_argument(
        "--vllm-executable", default="vllm", help="vLLM CLI executable"
    )
    output.add_argument(
        "--output", type=Path, required=True, help="structured JSON report path"
    )
    output.add_argument(
        "--overwrite", action="store_true", help="replace an existing report"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_concurrency > args.num_prompts:
        parser.error("--max-concurrency cannot exceed --num-prompts")
    return args


def build_vllm_command(
    args: argparse.Namespace,
    *,
    block_index: int,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    """Build one official streaming ``vllm bench serve`` invocation."""

    seed = args.seed + block_index
    return [
        args.vllm_executable,
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--base-url",
        args.base_url.rstrip("/"),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name or args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.warmups),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        args.request_rate,
        "--seed",
        str(seed),
        "--ignore-eos",
        "--disable-shuffle",
        "--percentile-metrics",
        STREAMING_METRICS,
        "--metric-percentiles",
        args.percentiles,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--ready-check-timeout-sec",
        str(args.ready_timeout),
        "--label",
        args.run_label,
        "--disable-tqdm",
    ]


def _git_state(commit_override: str | None) -> dict[str, Any]:
    if commit_override:
        return {"commit": commit_override, "dirty": None, "source": "argument"}

    checkout = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}
    return {"commit": commit, "dirty": dirty, "source": "checkout"}


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("gridbook")
    except importlib.metadata.PackageNotFoundError:
        try:
            from gridbook import __version__

            return __version__
        except (ImportError, AttributeError):
            return None


def _vllm_version(executable: str) -> str | None:
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output or None


def capture_dispatch_environment(
    env: Mapping[str, str], names: Sequence[str], prefixes: Sequence[str]
) -> dict[str, str | None]:
    selected = set(names)
    selected.update(
        key for key in env if any(key.startswith(prefix) for prefix in prefixes)
    )
    captured: dict[str, str | None] = {}
    for key in sorted(selected):
        value = env.get(key)
        if value is not None and _SENSITIVE_ENV.search(key):
            value = "<redacted>"
        captured[key] = value
    return captured


def collect_metadata(
    args: argparse.Namespace, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    git = _git_state(args.git_commit)
    if not git["commit"]:
        raise BenchmarkError(
            "Gridbook commit is unavailable; run from a checkout or pass --git-commit"
        )
    return {
        "git": git,
        "software": {
            "gridbook_version": _package_version(),
            "vllm_cli_version": _vllm_version(args.vllm_executable),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "artifacts": {
            "image_id": args.image_id,
            "model_id": args.model_id,
            "benchmark_model": args.model,
            "tokenizer": args.tokenizer or args.model,
        },
        "server": {
            "base_url": args.base_url.rstrip("/"),
            "backend": args.backend,
            "endpoint": args.endpoint,
            "served_model_name": args.served_model_name or args.model,
            "recorded_args": list(args.server_arg),
        },
        "dispatch": {
            "environment": capture_dispatch_environment(
                os.environ if env is None else env,
                args.dispatch_env,
                args.dispatch_env_prefix,
            )
        },
        "workload": {
            "dataset": "random",
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_prompts_per_block": args.num_prompts,
            "warmups_per_block": args.warmups,
            "max_concurrency": args.max_concurrency,
            "blocks": args.blocks,
            "base_seed": args.seed,
            "block_seeds": [args.seed + index for index in range(args.blocks)],
            "request_rate": args.request_rate,
            "random_range_ratio": 0,
            "ignore_eos": True,
            "streaming": True,
            "metrics": STREAMING_METRICS.split(","),
            "percentiles": [float(item) for item in args.percentiles.split(",")],
        },
    }


def _load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise BenchmarkError(f"vLLM did not create result file {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"vLLM result is not valid JSON: {path}") from exc
    # ``--append-result`` creates a list; the harness never requests it, but
    # accepting a singleton makes this resilient to vLLM version differences.
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise BenchmarkError("vLLM result JSON must contain one object")
    return payload


def _number(result: Mapping[str, Any], key: str) -> float | None:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def validate_result(result: Mapping[str, Any], args: argparse.Namespace) -> None:
    missing = [key for key in REQUIRED_STREAMING_RESULTS if _number(result, key) is None]
    if missing:
        raise BenchmarkError(
            "vLLM result lacks streaming metrics: " + ", ".join(missing)
        )

    expected = {
        "completed": (("completed",), args.num_prompts),
        "failed": (("failed",), 0),
        # Current vLLM uses *_tokens.  The shorter aliases keep reports made by
        # older bench-serve releases usable without weakening the exact-length
        # gate.
        "total_input_tokens": (
            ("total_input_tokens", "total_input"),
            args.num_prompts * args.input_len,
        ),
        "total_output_tokens": (
            ("total_output_tokens", "total_output"),
            args.num_prompts * args.output_len,
        ),
    }
    mismatches = []
    for label, (aliases, wanted) in expected.items():
        actual = next(
            (_number(result, key) for key in aliases if _number(result, key) is not None),
            None,
        )
        if actual is None:
            mismatches.append(f"{label}=missing (expected {wanted})")
        elif actual != wanted:
            mismatches.append(f"{label}={actual:g} (expected {wanted})")
    if mismatches:
        raise BenchmarkError(
            "fixed-shape block was not completed exactly: " + "; ".join(mismatches)
        )

    for key, wanted in (("input_lens", args.input_len), ("output_lens", args.output_len)):
        lengths = result.get(key)
        if not isinstance(lengths, list) or len(lengths) != args.num_prompts:
            raise BenchmarkError(
                f"vLLM detailed result {key} must contain one value per prompt"
            )
        wrong = [index for index, value in enumerate(lengths) if value != wanted]
        if wrong:
            preview = ", ".join(str(index) for index in wrong[:5])
            raise BenchmarkError(
                f"fixed-shape block has unexpected {key} at request index: {preview}"
            )


def summarize_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results = [block["result"] for block in blocks]
    metric_names = set(SUMMARY_METRICS)
    for result in results:
        metric_names.update(key for key in result if _PERCENTILE_RESULT.match(key))

    metrics: dict[str, Any] = {}
    for name in sorted(metric_names):
        values = [_number(result, name) for result in results]
        if any(value is None for value in values):
            continue
        numeric = [value for value in values if value is not None]
        metrics[name] = {
            "values": numeric,
            "mean": statistics.fmean(numeric),
            "median": statistics.median(numeric),
            "min": min(numeric),
            "max": max(numeric),
            "sample_stdev": statistics.stdev(numeric) if len(numeric) > 1 else None,
        }
    return {"completed_blocks": len(blocks), "metrics": metrics}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run_command(command: Sequence[str]) -> int:
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        raise BenchmarkError(f"could not execute {command[0]!r}: {exc}") from exc


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise BenchmarkError(
            f"output already exists: {output_path} (pass --overwrite to replace it)"
        )

    started_at = _utc_now()
    started_clock = time.monotonic()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "run_label": args.run_label,
        "started_at": started_at,
        "finished_at": None,
        "duration_s": None,
        "metadata": collect_metadata(args),
        "blocks": [],
        "summary": None,
    }

    try:
        with tempfile.TemporaryDirectory(prefix="gridbook-bench-serve-") as raw_dir:
            result_dir = Path(raw_dir)
            for block_index in range(args.blocks):
                filename = f"block-{block_index + 1:03d}.json"
                command = build_vllm_command(
                    args,
                    block_index=block_index,
                    result_dir=result_dir,
                    result_filename=filename,
                )
                block_started = _utc_now()
                block_clock = time.monotonic()
                print(
                    f"[{args.run_label}] block {block_index + 1}/{args.blocks} "
                    f"seed={args.seed + block_index}",
                    flush=True,
                )
                returncode = _run_command(command)
                if returncode != 0:
                    raise BenchmarkError(
                        f"vLLM bench serve failed in block {block_index + 1} "
                        f"with exit code {returncode}"
                    )
                result = _load_result(result_dir / filename)
                validate_result(result, args)
                report["blocks"].append(
                    {
                        "index": block_index + 1,
                        "seed": args.seed + block_index,
                        "started_at": block_started,
                        "finished_at": _utc_now(),
                        "wall_duration_s": time.monotonic() - block_clock,
                        "command": command,
                        "result": result,
                    }
                )
        report["summary"] = summarize_blocks(report["blocks"])
        report["status"] = "success"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = _utc_now()
        report["duration_s"] = time.monotonic() - started_clock
        _atomic_write_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = run_benchmark(args)
    except BenchmarkError as exc:
        print(f"gridbook-bench-serve: {exc}", file=sys.stderr)
        return 2
    print(
        f"saved {len(report['blocks'])} validated block(s) to "
        f"{args.output.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
