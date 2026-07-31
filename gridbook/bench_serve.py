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
import errno
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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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
REQUIRED_THROUGHPUT_RESULTS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)
_PERCENTILE_RESULT = re.compile(
    r"^p(?:\d+(?:\.\d+)?)_(?:ttft|tpot|itl|e2el)_ms$"
)
_SENSITIVE_SEGMENTS = {
    "AUTH",
    "AUTHORIZATION",
    "BEARER",
    "COOKIE",
    "CREDENTIAL",
    "CREDENTIALS",
    "KEY",
    "PASSWORD",
    "PASSWD",
    "PRIVATE",
    "SECRET",
    "SESSION",
    "SIGNATURE",
    "TOKEN",
}
_SENSITIVE_COMPOUNDS = ("APIKEY", "AUTHTOKEN", "ACCESSTOKEN", "REFRESHTOKEN")
_ASSIGNMENT = re.compile(r"(\b[-\w.]+\b\s*(?:=|:)\s*)([^\s,;]+)")
_OPTION_VALUE = re.compile(r"((?:^|\s)(--?[-\w.]+)\s+)([^\s]+)")
_AUTH_HEADER = re.compile(
    r"(?i)((?:proxy-)?authorization\s*:\s*)([^\s,;]+)(?:\s+([^\s,;]+))?"
)
_URL = re.compile(r"https?://[^\s]+", re.I)


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


def _nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def _input_range_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1)") from exc
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1)")
    return ratio


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


def _server_environment(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError("must have the form NAME=VALUE")
    return name, setting


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
        choices=("openai",),
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
    artifact.add_argument(
        "--artifact-bytes",
        type=_positive_int,
        required=True,
        help="exact whole served-artifact bytes, excluding caches",
    )
    artifact.add_argument(
        "--byte-budget",
        type=_positive_int,
        required=True,
        help="maximum allowed whole served-artifact bytes",
    )

    identity = parser.add_argument_group("delegated execution identity")
    identity.add_argument("--format-rung", type=_nonempty, required=True)
    identity.add_argument("--serialized-layout", type=_nonempty, required=True)
    identity.add_argument("--scale-coding", type=_nonempty, required=True)
    identity.add_argument(
        "--quant-contract",
        type=_nonempty,
        required=True,
        help="weight/activation contract, for example W4A4 or W8A8",
    )
    identity.add_argument("--kernel-backend", type=_nonempty, required=True)
    identity.add_argument("--tensor-parallel-size", type=_positive_int, required=True)
    identity.add_argument("--fallback-state", type=_nonempty, required=True)
    identity.add_argument("--client-runtime-id", type=_nonempty, required=True)
    identity.add_argument("--server-runtime-id", type=_nonempty, required=True)
    identity.add_argument("--gpu-id", type=_nonempty, required=True)
    identity.add_argument("--driver-version", type=_nonempty, required=True)
    identity.add_argument("--cuda-version", type=_nonempty, required=True)

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
        help=(
            "independent client blocks; distinct deterministic dataset seed per "
            "block (default: 3)"
        ),
    )
    workload.add_argument(
        "--dataset-seed",
        "--seed",
        dest="dataset_seed",
        type=int,
        default=1234,
        help="random prompt seed; server generation is separately pinned greedy",
    )
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
    workload.add_argument(
        "--input-range-ratio",
        type=_input_range_ratio,
        default=0.0,
        help=(
            "uniform random input-length range ratio in [0,1); output length "
            "remains fixed (default: 0)"
        ),
    )

    dispatch = parser.add_argument_group("dispatch provenance")
    dispatch.add_argument(
        "--runner-env",
        action="append",
        default=[],
        metavar="NAME",
        help="benchmark-runner environment variable to record; repeat as needed",
    )
    dispatch.add_argument(
        "--runner-env-prefix",
        action="append",
        default=["PRISMAQUANT_"],
        metavar="PREFIX",
        help=(
            "record runner variables matching this prefix; repeat to add prefixes "
            "(PRISMAQUANT_ is always included)"
        ),
    )
    dispatch.add_argument(
        "--server-env",
        action="append",
        default=[],
        type=_server_environment,
        metavar="NAME=VALUE",
        help=(
            "explicit environment from the separately managed server; repeat as "
            "needed (never inferred from the runner)"
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
    server_env_names = [name for name, _ in args.server_env]
    if len(server_env_names) != len(set(server_env_names)):
        parser.error("--server-env names must not be repeated")
    if args.artifact_bytes > args.byte_budget:
        parser.error(
            f"--artifact-bytes ({args.artifact_bytes}) exceeds --byte-budget "
            f"({args.byte_budget})"
        )
    return args


def build_vllm_command(
    args: argparse.Namespace,
    *,
    block_index: int,
    result_dir: Path,
    result_filename: str,
) -> list[str]:
    """Build one official streaming ``vllm bench serve`` invocation."""

    dataset_seed = args.dataset_seed + block_index
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
        json.dumps(
            {"input": args.input_range_ratio, "output": 0.0},
            separators=(",", ":"),
            sort_keys=True,
        ),
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.warmups),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        args.request_rate,
        "--seed",
        str(dataset_seed),
        "--temperature",
        "0",
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


def _is_sensitive_key(key: str) -> bool:
    normalized = key.upper().strip("-")
    segments = set(filter(None, re.split(r"[^A-Z0-9]+|_+", normalized)))
    if segments & _SENSITIVE_SEGMENTS:
        return True
    compact = re.sub(r"[^A-Z0-9]", "", normalized)
    return any(marker in compact for marker in _SENSITIVE_COMPOUNDS)


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return value
        hostname = parsed.hostname
        if hostname is None:
            return "<redacted-invalid-url>"
        host = f"[{hostname}]" if ":" in hostname else hostname
        try:
            port = parsed.port
        except ValueError:
            return "<redacted-invalid-url>"
        if port is not None:
            host = f"{host}:{port}"
        netloc = f"<redacted>@{host}" if parsed.username is not None else host
        query = urlencode(
            [
                (key, "<redacted>" if _is_sensitive_key(key) else setting)
                for key, setting in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    except (TypeError, ValueError):
        return "<redacted-invalid-url>"


def _redact_text(value: str) -> str:
    def redact_url(match: re.Match[str]) -> str:
        return _redact_url(match.group(0))

    def redact_assignment(match: re.Match[str]) -> str:
        prefix = match.group(1)
        key = re.split(r"\s*(?:=|:)\s*", prefix, maxsplit=1)[0]
        return prefix + ("<redacted>" if _is_sensitive_key(key) else match.group(2))

    def redact_option(match: re.Match[str]) -> str:
        return match.group(1) + (
            "<redacted>" if _is_sensitive_key(match.group(2)) else match.group(3)
        )

    def redact_authorization(match: re.Match[str]) -> str:
        scheme_or_value = match.group(2)
        if match.group(3) is None:
            return match.group(1) + "<redacted>"
        return match.group(1) + scheme_or_value + " <redacted>"

    redacted = _URL.sub(redact_url, str(value))
    redacted = _ASSIGNMENT.sub(redact_assignment, redacted)
    redacted = _OPTION_VALUE.sub(redact_option, redacted)
    return _AUTH_HEADER.sub(redact_authorization, redacted)


def redact_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for item in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        text = _redact_text(str(item))
        redacted.append(text)
        if str(item).startswith("-") and "=" not in str(item):
            hide_next = _is_sensitive_key(str(item))
    return redacted


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
        if value is not None:
            value = "<redacted>" if _is_sensitive_key(key) else _redact_text(value)
        captured[key] = value
    return captured


def collect_metadata(
    args: argparse.Namespace, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    probe_errors: dict[str, str] = {}

    def probe(name: str, function, fallback=None):
        try:
            return function()
        except Exception as exc:  # noqa: BLE001 - metadata cannot hide run failure
            probe_errors[name] = _redact_text(f"{type(exc).__name__}: {exc}")
            return fallback

    git = probe(
        "git",
        lambda: _git_state(args.git_commit),
        {"commit": None, "dirty": None, "source": "probe-error"},
    )
    runner_env = os.environ if env is None else env
    explicit_server_env = {
        key: "<redacted>" if _is_sensitive_key(key) else _redact_text(value)
        for key, value in sorted(args.server_env)
    }
    metadata = {
        "git": git,
        "software": {
            "gridbook_version": probe("gridbook_version", _package_version),
            "runner_vllm_cli_probe": probe(
                "vllm_cli_version", lambda: _vllm_version(args.vllm_executable)
            ),
            "python": probe("python", platform.python_version),
            "platform": probe("platform", platform.platform),
            "machine": probe("machine", platform.machine),
            "hostname": probe("hostname", platform.node),
        },
        "artifacts": {
            "image_id": _redact_text(args.image_id),
            "model_id": _redact_text(args.model_id),
            "benchmark_model": _redact_text(args.model),
            "tokenizer": _redact_text(args.tokenizer or args.model),
            "whole_served_artifact_bytes": args.artifact_bytes,
            "byte_budget_bytes": args.byte_budget,
            "budget_headroom_bytes": args.byte_budget - args.artifact_bytes,
            "within_byte_budget": args.artifact_bytes <= args.byte_budget,
            "byte_scope": (
                "shipped model shards, configs, tokenizer, and sidecars; "
                "runtime/download/JIT caches excluded"
            ),
        },
        "execution_identity": {
            "format_rung": _redact_text(args.format_rung),
            "serialization": {
                "layout": _redact_text(args.serialized_layout),
                "scale_coding": _redact_text(args.scale_coding),
            },
            "quant_contract": _redact_text(args.quant_contract),
            "kernel_backend": _redact_text(args.kernel_backend),
            "tensor_parallel_size": args.tensor_parallel_size,
            "fallback_state": _redact_text(args.fallback_state),
            "client_runtime_id": _redact_text(args.client_runtime_id),
            "server_runtime_id": _redact_text(args.server_runtime_id),
            "hardware": {
                "gpu_id": _redact_text(args.gpu_id),
                "driver_version": _redact_text(args.driver_version),
                "cuda_version": _redact_text(args.cuda_version),
            },
        },
        "server": {
            "base_url": _redact_url(args.base_url.rstrip("/")),
            "backend": args.backend,
            "endpoint": _redact_text(args.endpoint),
            "served_model_name": _redact_text(
                args.served_model_name or args.model
            ),
            "recorded_args": redact_command(args.server_arg),
        },
        "dispatch": {
            "runner_environment": {
                "source": "benchmark process",
                "values": capture_dispatch_environment(
                    runner_env,
                    args.runner_env,
                    args.runner_env_prefix,
                ),
            },
            "server_environment": {
                "source": "explicit --server-env arguments",
                "values": explicit_server_env,
            },
        },
        "workload": {
            "dataset": "random",
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_prompts_per_block": args.num_prompts,
            "warmups_per_block": args.warmups,
            "max_concurrency": args.max_concurrency,
            "blocks": args.blocks,
            "dataset_base_seed": args.dataset_seed,
            "dataset_block_seeds": [
                args.dataset_seed + index for index in range(args.blocks)
            ],
            "sampling": {
                "strategy": "greedy",
                "temperature": 0.0,
                "sampling_seed": None,
            },
            "request_rate": args.request_rate,
            "input_range_ratio": args.input_range_ratio,
            "vllm_range_ratio": {
                "input": args.input_range_ratio,
                "output": 0.0,
            },
            "input_length_validation": (
                "exact" if args.input_range_ratio == 0 else "conservative-range"
            ),
            "accepted_input_length_bounds": list(
                _input_length_bounds(args.input_len, args.input_range_ratio)
            ),
            "aggregate_reconciliation": {
                "ttft_itl": "mean/median vs detailed arrays; 0.01 ms or 1e-5 relative",
                "e2el": (
                    "mean/median vs TTFT+ITL; 5 ms or 0.5% terminal-SSE allowance"
                ),
                "tpot": (
                    "mean vs (mean_e2el-mean_ttft)/(fixed_output_len-1); "
                    "0.05 ms or 0.5%"
                ),
                "itl_cardinality": (
                    "not equated to output tokens because one SSE chunk may carry "
                    "multiple tokens"
                ),
            },
            "ignore_eos": True,
            "streaming": True,
            "metrics": STREAMING_METRICS.split(","),
            "percentiles": [float(item) for item in args.percentiles.split(",")],
        },
    }
    metadata["collection_errors"] = probe_errors
    return metadata


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
    number = float(value)
    return number if math.isfinite(number) else None


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(_nonfinite_paths(child, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_nonfinite_paths(child, f"{path}[{index}]"))
        return paths
    return []


def _percentile_result_key(percentile: str, metric: str) -> str:
    number = float(percentile)
    label = str(int(number)) if number.is_integer() else f"{number:g}"
    return f"p{label}_{metric}_ms"


def _input_length_bounds(input_len: int, ratio: float) -> tuple[int, int]:
    if ratio == 0:
        return input_len, input_len
    # Pinned vLLM 0.23 samples through ceil(input_len * (1 + ratio)), but its
    # lower endpoint is computed after subtracting tokenizer-added special
    # tokens and its decode/re-encode correction can shorten a prompt.  That
    # special-token count is not exposed in bench-serve JSON.  One token is
    # therefore the only universally safe lower bound; the upper bound remains
    # exact and useful, and totals/per-request types are checked independently.
    return 1, math.ceil(input_len * (1 + ratio))


def _aggregate_tolerance_ms(metric: str, expected_ms: float) -> float:
    if metric == "e2el":
        # vLLM's request latency extends through the terminal SSE event, while
        # detailed ITLs stop at the final token-bearing chunk.  Permit that
        # small transport tail, but not a fabricated decode-length gap.
        return max(5.0, abs(expected_ms) * 0.005)
    return max(0.01, abs(expected_ms) * 1e-5)


def _reconcile_detailed_aggregate(
    result: Mapping[str, Any], metric: str, samples_ms: Sequence[float]
) -> None:
    expected = {
        "mean": statistics.fmean(samples_ms) if samples_ms else 0.0,
        "median": statistics.median(samples_ms) if samples_ms else 0.0,
    }
    for statistic_name, expected_ms in expected.items():
        key = f"{statistic_name}_{metric}_ms"
        actual_ms = _number(result, key)
        tolerance_ms = _aggregate_tolerance_ms(metric, expected_ms)
        if actual_ms is None or abs(actual_ms - expected_ms) > tolerance_ms:
            raise BenchmarkError(
                f"{key} disagrees with detailed streaming evidence: "
                f"reported={actual_ms}, reconstructed={expected_ms:.6f}, "
                f"tolerance={tolerance_ms:.6f} ms"
            )


def validate_result(result: Mapping[str, Any], args: argparse.Namespace) -> None:
    nonfinite = _nonfinite_paths(result)
    if nonfinite:
        raise BenchmarkError(
            "vLLM result contains NaN/Infinity at: " + ", ".join(nonfinite[:10])
        )

    requested_percentiles = args.percentiles.split(",")
    required_metrics = list(REQUIRED_STREAMING_RESULTS)
    required_metrics.extend(REQUIRED_THROUGHPUT_RESULTS)
    for metric in ("ttft", "tpot", "itl", "e2el"):
        required_metrics.extend((f"median_{metric}_ms", f"std_{metric}_ms"))
        required_metrics.extend(
            _percentile_result_key(percentile, metric)
            for percentile in requested_percentiles
        )
    missing = [key for key in required_metrics if _number(result, key) is None]
    if missing:
        raise BenchmarkError(
            "vLLM result lacks finite required metrics: " + ", ".join(missing)
        )

    positive_metrics = list(REQUIRED_THROUGHPUT_RESULTS)
    for metric in ("ttft", "e2el"):
        positive_metrics.extend((f"mean_{metric}_ms", f"median_{metric}_ms"))
        positive_metrics.extend(
            _percentile_result_key(percentile, metric)
            for percentile in requested_percentiles
        )
    if args.output_len > 1:
        for metric in ("tpot", "itl"):
            positive_metrics.extend((f"mean_{metric}_ms", f"median_{metric}_ms"))
            positive_metrics.extend(
                _percentile_result_key(percentile, metric)
                for percentile in requested_percentiles
            )
    nonpositive = [key for key in positive_metrics if _number(result, key) <= 0]
    if nonpositive:
        raise BenchmarkError(
            "vLLM result has non-positive latency/throughput metrics: "
            + ", ".join(nonpositive)
        )
    nonnegative_metrics = [
        f"std_{metric}_ms" for metric in ("ttft", "tpot", "itl", "e2el")
    ]
    if args.output_len == 1:
        for metric in ("tpot", "itl"):
            nonnegative_metrics.extend((f"mean_{metric}_ms", f"median_{metric}_ms"))
            nonnegative_metrics.extend(
                _percentile_result_key(percentile, metric)
                for percentile in requested_percentiles
            )
    negative = [key for key in nonnegative_metrics if _number(result, key) < 0]
    if negative:
        raise BenchmarkError(
            "vLLM result has negative latency metrics: " + ", ".join(negative)
        )

    expected: dict[str, tuple[tuple[str, ...], int]] = {
        "completed": (("completed",), args.num_prompts),
        "failed": (("failed",), 0),
        "total_output_tokens": (
            ("total_output_tokens", "total_output"),
            args.num_prompts * args.output_len,
        ),
    }
    if args.input_range_ratio == 0:
        # Current vLLM uses *_tokens.  The shorter aliases keep results from
        # older bench-serve releases usable without weakening the fixed gate.
        expected["total_input_tokens"] = (
            ("total_input_tokens", "total_input"),
            args.num_prompts * args.input_len,
        )
    mismatches = []
    for label, (aliases, wanted) in expected.items():
        actual = None
        for key in aliases:
            actual = _number(result, key)
            if actual is not None:
                break
        if actual is None:
            mismatches.append(f"{label}=missing (expected {wanted})")
        elif actual != wanted:
            mismatches.append(f"{label}={actual:g} (expected {wanted})")
    if mismatches:
        raise BenchmarkError(
            "fixed-shape block was not completed exactly: " + "; ".join(mismatches)
        )

    input_lens = result.get("input_lens")
    if not isinstance(input_lens, list) or len(input_lens) != args.num_prompts:
        raise BenchmarkError(
            "vLLM detailed result input_lens must contain one value per prompt"
        )
    lower, upper = _input_length_bounds(args.input_len, args.input_range_ratio)
    wrong_inputs = [
        index
        for index, value in enumerate(input_lens)
        if isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= upper
    ]
    if wrong_inputs:
        preview = ", ".join(str(index) for index in wrong_inputs[:5])
        mode = "fixed" if args.input_range_ratio == 0 else f"range [{lower}, {upper}]"
        raise BenchmarkError(
            f"{mode} block has unexpected input_lens at request index: {preview}"
        )
    if args.input_range_ratio > 0:
        total_input = None
        for key in ("total_input_tokens", "total_input"):
            total_input = _number(result, key)
            if total_input is not None:
                break
        observed_total = sum(input_lens)
        if total_input != observed_total:
            raise BenchmarkError(
                "distributed input lengths do not reconcile with total_input_tokens: "
                f"{total_input} != {observed_total}"
            )

    output_lens = result.get("output_lens")
    if not isinstance(output_lens, list) or len(output_lens) != args.num_prompts:
        raise BenchmarkError(
            "vLLM detailed result output_lens must contain one value per prompt"
        )
    wrong_outputs = [
        index
        for index, value in enumerate(output_lens)
        if isinstance(value, bool)
        or not isinstance(value, int)
        or value != args.output_len
    ]
    if wrong_outputs:
        preview = ", ".join(str(index) for index in wrong_outputs[:5])
        raise BenchmarkError(
            "fixed output block has unexpected output_lens at request index: "
            + preview
        )

    errors = result.get("errors")
    if not isinstance(errors, list) or len(errors) != args.num_prompts:
        raise BenchmarkError("vLLM detailed result errors must cover every prompt")
    if any(error is not None and error != "" for error in errors):
        raise BenchmarkError("vLLM detailed result contains request errors")

    ttfts = result.get("ttfts")
    if not isinstance(ttfts, list) or len(ttfts) != args.num_prompts:
        raise BenchmarkError("vLLM detailed result ttfts must cover every prompt")
    bad_ttft = [
        index
        for index, value in enumerate(ttfts)
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ]
    if bad_ttft:
        raise BenchmarkError(
            "vLLM detailed result has non-positive/non-finite TTFT at request "
            f"index: {', '.join(str(index) for index in bad_ttft[:5])}"
        )

    itls = result.get("itls")
    if not isinstance(itls, list) or len(itls) != args.num_prompts:
        raise BenchmarkError("vLLM detailed result itls must cover every prompt")
    for request_index, request_itls in enumerate(itls):
        if not isinstance(request_itls, list):
            raise BenchmarkError(
                f"vLLM detailed result itls[{request_index}] must be an array"
            )
        if args.output_len > 1 and not request_itls:
            raise BenchmarkError(
                "multi-token response has no inter-token latency samples at "
                f"request index {request_index}"
            )
        for interval_index, value in enumerate(request_itls):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise BenchmarkError(
                    "vLLM detailed result has non-positive/non-finite ITL at "
                    f"request {request_index}, interval {interval_index}"
                )

    # vLLM does not currently emit a per-request E2EL array.  Its streaming
    # definition is exactly TTFT plus the subsequent inter-token intervals, so
    # reconstructing it also validates that each detailed request has a finite,
    # positive end-to-end latency.  output_len=1 intentionally has no ITLs.
    e2els = [
        float(ttft) + sum(map(float, intervals))
        for ttft, intervals in zip(ttfts, itls)
    ]
    if any(not math.isfinite(value) or value <= 0 for value in e2els):
        raise BenchmarkError("derived detailed E2EL contains a non-finite value")

    _reconcile_detailed_aggregate(
        result, "ttft", [float(value) * 1000 for value in ttfts]
    )
    _reconcile_detailed_aggregate(
        result,
        "itl",
        [float(value) * 1000 for intervals in itls for value in intervals],
    )
    _reconcile_detailed_aggregate(
        result, "e2el", [value * 1000 for value in e2els]
    )
    if args.output_len > 1:
        # All outputs are fixed length, so linearity makes this exact even
        # though medians cannot be reconstructed from aggregate medians and an
        # SSE chunk may carry multiple speculative tokens.
        expected_mean_tpot = (
            _number(result, "mean_e2el_ms") - _number(result, "mean_ttft_ms")
        ) / (args.output_len - 1)
        reported_mean_tpot = _number(result, "mean_tpot_ms")
        tolerance = max(0.05, abs(expected_mean_tpot) * 0.005)
        if abs(reported_mean_tpot - expected_mean_tpot) > tolerance:
            raise BenchmarkError(
                "mean_tpot_ms disagrees with mean E2EL/TTFT and fixed output "
                f"length: reported={reported_mean_tpot}, "
                f"reconstructed={expected_mean_tpot:.6f}, "
                f"tolerance={tolerance:.6f} ms"
            )


def summarize_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results = [block["raw_result"] for block in blocks]
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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reserve_output(path: Path, *, overwrite: bool) -> None:
    """Atomically claim ``path`` before probes or requests begin.

    ``exists()`` followed by a write has a race in which two benchmark clients
    both believe a result name is free.  O_EXCL makes exactly one of them the
    owner.  Explicit ``--overwrite`` opts out of that protection by design.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BenchmarkError(f"refusing to reserve symlink output: {path}")
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise BenchmarkError(
            f"output already exists: {path} (pass --overwrite to replace it)"
        ) from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP or path.is_symlink():
            raise BenchmarkError(f"refusing to reserve symlink output: {path}") from exc
        raise
    with os.fdopen(descriptor, "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(
            {"schema": SCHEMA, "status": "reserved", "reserved_at": _utc_now()},
            handle,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _json_safe(value: Any) -> Any:
    """Preserve invalid numeric evidence without emitting invalid JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite_float": "NaN"}
        return {"nonfinite_float": "Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _run_command(command: Sequence[str]) -> int:
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        raise BenchmarkError(f"could not execute {command[0]!r}: {exc}") from exc


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    # ``resolve()`` follows the final component and would turn an explicitly
    # rejected output symlink into its target before _reserve_output sees it.
    # abspath-style normalization preserves that final component for the
    # O_NOFOLLOW/symlink gate while still making report paths deterministic.
    output_path = Path(os.path.abspath(os.fspath(args.output.expanduser())))
    _reserve_output(output_path, overwrite=args.overwrite)

    started_at = _utc_now()
    started_clock = time.monotonic()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "run_label": args.run_label,
        "started_at": started_at,
        "finished_at": None,
        "duration_s": None,
        "metadata": None,
        "blocks": [],
        "summary": None,
    }

    try:
        report["metadata"] = collect_metadata(args)
        _atomic_write_json(output_path, report)
        if not report["metadata"]["git"]["commit"]:
            raise BenchmarkError(
                "Gridbook commit is unavailable; run from a checkout or pass "
                "--git-commit"
            )
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
                block: dict[str, Any] = {
                    "index": block_index + 1,
                    "dataset_seed": args.dataset_seed + block_index,
                    "status": "running",
                    "started_at": block_started,
                    "finished_at": None,
                    "wall_duration_s": None,
                    "command": redact_command(command),
                    "returncode": None,
                    "raw_result": None,
                    "validation_error": None,
                }
                report["blocks"].append(block)
                # A durable checkpoint makes the in-flight command available
                # even if the client is interrupted before vLLM returns.
                _atomic_write_json(output_path, report)
                print(
                    f"[{args.run_label}] block {block_index + 1}/{args.blocks} "
                    f"dataset_seed={args.dataset_seed + block_index}",
                    flush=True,
                )
                try:
                    returncode = _run_command(command)
                    block["returncode"] = returncode
                    _atomic_write_json(output_path, report)
                    if returncode != 0:
                        raise BenchmarkError(
                            f"vLLM bench serve failed in block {block_index + 1} "
                            f"with exit code {returncode}"
                        )
                    try:
                        result = _load_result(result_dir / filename)
                        block["raw_result"] = _json_safe(result)
                        _atomic_write_json(output_path, report)
                        validate_result(result, args)
                    except BenchmarkError as exc:
                        block["validation_error"] = _redact_text(str(exc))
                        raise
                    block["status"] = "success"
                except BaseException:
                    block["status"] = "failed"
                    raise
                finally:
                    block["finished_at"] = _utc_now()
                    block["wall_duration_s"] = time.monotonic() - block_clock
        report["summary"] = summarize_blocks(report["blocks"])
        report["status"] = "success"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = _redact_text(f"{type(exc).__name__}: {exc}")
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
