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
import hashlib
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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SCHEMA = "gridbook.vllm-bench-serve.v2"
ARTIFACT_INVENTORY_SCHEMA = "gridbook.artifact-inventory.v1"
PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA = "prismaquant.cb_export_artifact_inventory.v1"
EXECUTION_MANIFEST_SCHEMA = "gridbook.execution-manifest.v1"
STREAMING_METRICS = "ttft,tpot,itl,e2el"
DEFAULT_PERCENTILES = "50,90,95,99"
MIN_WARMUPS = 4
MIN_BLOCKS = 3
REQUEST_BURSTINESS = 1.0

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
    r"^p(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)_(?:ttft|tpot|itl|e2el)_ms$"
)
_VLLM_RESULT_DATE = re.compile(r"^\d{8}-\d{6}$")
_SENSITIVE_SEGMENTS = {
    "AUTH",
    "AUTHORIZATION",
    "BEARER",
    "COOKIE",
    "CREDENTIAL",
    "CREDENTIALS",
    "HEADER",
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
_SENSITIVE_OPTIONS = {
    "-H",
    "-b",
    "--cookie",
    "--extra-header",
    "--extra-headers",
    "--header",
    "--headers",
}
_ASSIGNMENT = re.compile(r"(\b[-\w.]+\b\s*(?:=|:)\s*)([^\s,;]+)")
_OPTION_VALUE = re.compile(r"((?:^|\s)(--?[-\w.]+)\s+)([^\s]+)")
_AUTH_HEADER = re.compile(
    r"(?i)((?:proxy-)?authorization\s*:\s*)([^\s,;]+)(?:\s+([^\s,;]+))?"
)
_SENSITIVE_HEADER = re.compile(
    r"(?i)((?:(?:proxy-)?authorization|cookie|set-cookie|x-api-key|api-key)"
    r"\s*:\s*)(.*)$"
)
_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_RUN_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_]+_$")
_CONCRETE_UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")

_EXECUTION_ASSIGNMENT_FIELDS = (
    "unit",
    "format_rung",
    "serialized_layout",
    "scale_coding",
    "quant_contract",
    "kernel_backend",
    "fallback_state",
)

_PINNED_RESULT_FIELDS = frozenset(
    {
        "date",
        "endpoint_type",
        "backend",
        "label",
        "model_id",
        "tokenizer_id",
        "num_prompts",
        "request_rate",
        "burstiness",
        "max_concurrency",
        "duration",
        "completed",
        "failed",
        "total_input_tokens",
        "total_output_tokens",
        "request_goodput",
        "input_lens",
        "output_lens",
        "ttfts",
        "itls",
        "start_times",
        "generated_texts",
        "errors",
        "max_output_tokens_per_s",
        "max_concurrent_requests",
        "rtfx",
        *SUMMARY_METRICS,
    }
)


class BenchmarkError(RuntimeError):
    """A benchmark failed before it produced a trustworthy block."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
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


def _sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a 64-character SHA-256 hex digest")
    return value.lower()


def _git_commit(value: str) -> str:
    if not _GIT_COMMIT.fullmatch(value):
        raise argparse.ArgumentTypeError("must be an exact 40- or 64-hex commit")
    return value.lower()


def _run_label(value: str) -> str:
    if not _RUN_LABEL.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must be 1-128 characters using only letters, digits, '.', '_', or '-'"
        )
    return value


def _environment_name(value: str) -> str:
    if not _ENV_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("must be an environment variable name")
    return value


def _environment_prefix(value: str) -> str:
    if not _ENV_PREFIX.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must be at least three characters, start with a letter, and end in '_'"
        )
    return value


def _strict_json_loads(value: str | bytes) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number {constant} is forbidden")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = child
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = _strict_json_loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a valid JSON object") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise argparse.ArgumentTypeError("must be a non-empty JSON object")
    return parsed


def _read_hashed_json(
    path: Path, expected_digest: str, label: str
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BenchmarkError(f"cannot read {label} {path}: {exc}") from exc
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise BenchmarkError(
            f"{label} SHA-256 mismatch: declared={expected_digest}, "
            f"actual={actual_digest}"
        )
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"{label} must contain one JSON object")
    return payload, actual_digest


def _validate_server_evidence(
    paths: Sequence[Path], expected_digests: Sequence[str]
) -> list[dict[str, Any]]:
    if not paths or len(paths) != len(expected_digests):
        raise BenchmarkError(
            "server evidence requires one --server-evidence-sha256 per "
            "--server-evidence attachment"
        )
    attachments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (path, expected_digest) in enumerate(zip(paths, expected_digests)):
        normalized = os.path.abspath(os.fspath(path.expanduser()))
        if normalized in seen:
            raise BenchmarkError(f"duplicate server evidence attachment: {path}")
        seen.add(normalized)
        if not path.expanduser().is_file():
            raise BenchmarkError(
                f"server evidence attachment {index + 1} is not a regular file"
            )
        digest = hashlib.sha256()
        size = 0
        try:
            with path.expanduser().open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise BenchmarkError(
                f"cannot read server evidence attachment {index + 1} {path}: {exc}"
            ) from exc
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise BenchmarkError(
                f"server evidence attachment {index + 1} SHA-256 mismatch: "
                f"declared={expected_digest}, actual={actual_digest}"
            )
        attachments.append(
            {
                "reference": str(path),
                "sha256": actual_digest,
                "bytes": size,
            }
        )
    return attachments


def _canonical_inventory_path(name: Any, where: str) -> str:
    if not isinstance(name, str) or not name.strip() or "\\" in name:
        raise BenchmarkError(f"{where} must be a POSIX relative path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or str(path) != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BenchmarkError(f"{where} is not canonical: {name!r}")
    return name


def _inventory_size(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{where} must be a non-negative integer")
    return value


def _extract_prismaquant_inventory(
    document: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if document.get("schema") == PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA:
        return document
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    inventory = provenance.get("artifact_inventory")
    if (
        isinstance(inventory, Mapping)
        and inventory.get("schema") == PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA
    ):
        return inventory
    return None


def _validate_artifact_inventory(path: Path, expected_digest: str) -> dict[str, Any]:
    document, digest = _read_hashed_json(path, expected_digest, "artifact inventory")
    normalized_files: list[dict[str, Any]] = []

    if document.get("schema") == ARTIFACT_INVENTORY_SCHEMA:
        files = document.get("files")
        if not isinstance(files, list) or not files:
            raise BenchmarkError("artifact inventory files must be a non-empty array")
        seen: set[str] = set()
        for index, entry in enumerate(files):
            if not isinstance(entry, Mapping):
                raise BenchmarkError(
                    f"artifact inventory files[{index}] must be an object"
                )
            name = _canonical_inventory_path(
                entry.get("path"), f"artifact inventory files[{index}].path"
            )
            if name in seen:
                raise BenchmarkError(
                    f"artifact inventory contains duplicate path {name!r}"
                )
            seen.add(name)
            size = _inventory_size(
                entry.get("bytes"), f"artifact inventory files[{index}].bytes"
            )
            file_digest = entry.get("sha256")
            if not isinstance(file_digest, str) or not _SHA256.fullmatch(file_digest):
                raise BenchmarkError(
                    f"artifact inventory files[{index}].sha256 must be a SHA-256 digest"
                )
            normalized_files.append(
                {"path": name, "bytes": size, "sha256": file_digest.lower()}
            )
        declared_total = _inventory_size(
            document.get("total_bytes"), "artifact inventory total_bytes"
        )
        schema = ARTIFACT_INVENTORY_SCHEMA
        source = "standalone-gridbook-inventory"
    else:
        inventory = _extract_prismaquant_inventory(document)
        if inventory is None:
            raise BenchmarkError(
                "artifact inventory must be a gridbook.artifact-inventory.v1 "
                "object, a prismaquant.cb_export_artifact_inventory.v1 object, "
                "or a quant_config containing that PrismaQuant inventory"
            )
        if inventory.get("scope") != "all_regular_files_recursive":
            raise BenchmarkError(
                "PrismaQuant artifact inventory scope must be "
                "all_regular_files_recursive"
            )
        file_bytes = inventory.get("file_bytes")
        if not isinstance(file_bytes, Mapping) or not file_bytes:
            raise BenchmarkError(
                "PrismaQuant artifact inventory file_bytes must be a non-empty object"
            )
        for name_value, size_value in file_bytes.items():
            name = _canonical_inventory_path(
                name_value, "PrismaQuant artifact inventory file_bytes key"
            )
            size = _inventory_size(
                size_value, f"PrismaQuant artifact inventory file_bytes[{name!r}]"
            )
            normalized_files.append({"path": name, "bytes": size})
        normalized_files.sort(key=lambda entry: entry["path"])
        declared_total = _inventory_size(
            inventory.get("export_directory_bytes"),
            "PrismaQuant artifact inventory export_directory_bytes",
        )
        schema = PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA
        source = (
            "standalone-prismaquant-inventory"
            if document is inventory
            else "quant-config-provenance"
        )

    computed_total = sum(entry["bytes"] for entry in normalized_files)
    if declared_total != computed_total:
        raise BenchmarkError(
            "artifact inventory total disagrees with its file entries: "
            f"declared={declared_total}, computed={computed_total}"
        )
    return {
        "reference": str(path),
        "sha256": digest,
        "schema": schema,
        "source": source,
        "file_count": len(normalized_files),
        "computed_total_bytes": computed_total,
        "files": normalized_files,
    }


def _validate_execution_manifest(
    path: Path,
    expected_digest: str,
    artifact_inventory_digest: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload, digest = _read_hashed_json(path, expected_digest, "execution manifest")
    if payload.get("schema") != EXECUTION_MANIFEST_SCHEMA:
        raise BenchmarkError(
            "execution manifest schema must be " + EXECUTION_MANIFEST_SCHEMA
        )
    if payload.get("artifact_inventory_sha256") != artifact_inventory_digest:
        raise BenchmarkError(
            "execution manifest is not bound to the selected artifact inventory"
        )
    if payload.get("coverage") != "all_serving_units":
        raise BenchmarkError(
            "execution manifest coverage must declare all_serving_units"
        )
    manifest_tp = payload.get("tensor_parallel_size")
    if (
        isinstance(manifest_tp, bool)
        or not isinstance(manifest_tp, int)
        or manifest_tp <= 0
    ):
        raise BenchmarkError(
            "execution manifest tensor_parallel_size must be a positive integer"
        )
    if manifest_tp != args.tensor_parallel_size:
        raise BenchmarkError(
            "execution manifest tensor_parallel_size disagrees with the command"
        )
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise BenchmarkError("execution manifest assignments must be non-empty")

    units: set[str] = set()
    values: dict[str, set[str]] = {
        field: set() for field in _EXECUTION_ASSIGNMENT_FIELDS if field != "unit"
    }
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping):
            raise BenchmarkError(
                f"execution manifest assignments[{index}] must be an object"
            )
        for field in _EXECUTION_ASSIGNMENT_FIELDS:
            value = assignment.get(field)
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkError(
                    f"execution manifest assignments[{index}].{field} must be non-empty"
                )
            if value.strip().lower() == "mixed":
                raise BenchmarkError(
                    f"execution manifest assignments[{index}].{field} must be "
                    "concrete, not 'mixed'"
                )
            if field == "unit":
                if not _CONCRETE_UNIT.fullmatch(value) or ".." in value:
                    raise BenchmarkError(
                        f"execution manifest assignments[{index}].unit must be a "
                        "concrete serving-unit identifier without wildcards"
                    )
                if value in units:
                    raise BenchmarkError(
                        f"execution manifest contains duplicate unit {value!r}"
                    )
                overlap = next(
                    (
                        existing
                        for existing in units
                        if any(
                            value.startswith(existing + separator)
                            or existing.startswith(value + separator)
                            for separator in (".", "/", ":")
                        )
                    ),
                    None,
                )
                if overlap is not None:
                    raise BenchmarkError(
                        "execution manifest contains overlapping unit identifiers: "
                        f"{overlap!r} and {value!r}"
                    )
                units.add(value)
            else:
                values[field].add(value)

    command_fields = {
        "format_rung": args.format_rung,
        "serialized_layout": args.serialized_layout,
        "scale_coding": args.scale_coding,
        "quant_contract": args.quant_contract,
        "kernel_backend": args.kernel_backend,
        "fallback_state": args.fallback_state,
    }
    for field, command_value in command_fields.items():
        observed = values[field]
        if len(observed) == 1:
            only_value = next(iter(observed))
            if command_value != only_value:
                raise BenchmarkError(
                    f"--{field.replace('_', '-')}={command_value!r} disagrees with "
                    f"the uniform execution manifest value {only_value!r}"
                )
        elif command_value.lower() != "mixed":
            raise BenchmarkError(
                f"--{field.replace('_', '-')} must be 'mixed' because the execution "
                f"manifest contains {len(observed)} values"
            )

    return {
        "reference": str(path),
        "sha256": digest,
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "artifact_inventory_sha256": artifact_inventory_digest,
        "coverage": "all_serving_units",
        "assignment_count": len(assignments),
        "assignments": [
            {field: assignment[field] for field in _EXECUTION_ASSIGNMENT_FIELDS}
            for assignment in assignments
        ],
        "distinct_values": {
            field: sorted(field_values) for field, field_values in values.items()
        },
    }


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


def _percentile_label(value: str | float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


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
    labels = [_percentile_label(item) for item in parsed]
    if len(set(labels)) != len(labels):
        raise argparse.ArgumentTypeError(
            "percentiles collide after pinned-vLLM float normalization"
        )
    return ",".join(labels)


def _server_environment(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError("must have the form NAME=VALUE")
    return name, setting


def _recorded_speculative_config(server_args: Sequence[str]) -> dict[str, Any] | None:
    raw_configs: list[str] = []
    index = 0
    while index < len(server_args):
        argument = server_args[index]
        if argument == "--speculative-config":
            if index + 1 >= len(server_args):
                raise BenchmarkError("recorded --speculative-config has no JSON value")
            raw_configs.append(server_args[index + 1])
            index += 2
            continue
        match = re.fullmatch(r"--speculative-config(?:=|\s+)(.+)", argument, re.DOTALL)
        if match:
            raw_configs.append(match.group(1))
        index += 1
    if len(raw_configs) > 1:
        raise BenchmarkError(
            "recorded server args contain multiple speculative configs"
        )
    if not raw_configs:
        return None
    try:
        config = _strict_json_loads(raw_configs[0])
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkError(
            "recorded server --speculative-config is not strict JSON"
        ) from exc
    if not isinstance(config, dict) or not config:
        raise BenchmarkError(
            "recorded server --speculative-config must be a non-empty JSON object"
        )
    return config


def _validate_recorded_prefix_caching(
    declared_state: str, server_args: Sequence[str]
) -> None:
    enable_flag = "--enable-prefix-caching"
    disable_flag = "--no-enable-prefix-caching"
    recorded = {argument.strip() for argument in server_args}
    enabled = enable_flag in recorded
    disabled = disable_flag in recorded
    if enabled and disabled:
        raise BenchmarkError(
            "recorded server args contain conflicting prefix-caching flags"
        )
    required_flag = enable_flag if declared_state == "on" else disable_flag
    conflicting_flag = disable_flag if declared_state == "on" else enable_flag
    if conflicting_flag in recorded:
        raise BenchmarkError(
            f"--prefix-caching={declared_state} conflicts with recorded "
            f"{conflicting_flag}"
        )
    if required_flag not in recorded:
        raise BenchmarkError(
            f"--prefix-caching={declared_state} requires recorded server arg "
            f"{required_flag}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridbook-bench-serve",
        description=(
            "Run fixed-shape, streaming vLLM serving benchmarks in independent "
            "blocks and save metrics plus reproducibility metadata as JSON."
        ),
    )
    server = parser.add_argument_group("online server")
    server.add_argument(
        "--base-url", type=_nonempty, required=True, help="OpenAI-compatible base URL"
    )
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
        type=_nonempty,
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
        "--prefix-caching",
        choices=("off", "on"),
        required=True,
        help="explicit server prefix-caching state, reconciled with --server-arg",
    )
    server.add_argument(
        "--server-evidence",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help="dispatch/startup log attachment to hash and record; repeatable",
    )
    server.add_argument(
        "--server-evidence-sha256",
        action="append",
        type=_sha256,
        required=True,
        metavar="SHA256",
        help="SHA-256 paired by order with each --server-evidence path",
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
        type=_nonempty,
        required=True,
        help="model/tokenizer name understood by vLLM bench serve",
    )
    artifact.add_argument(
        "--tokenizer",
        type=_nonempty,
        help="tokenizer name or revision (defaults to --model)",
    )
    artifact.add_argument(
        "--model-id",
        type=_nonempty,
        required=True,
        help="exact served artifact identifier/revision recorded in the report",
    )
    artifact.add_argument(
        "--image-id",
        type=_nonempty,
        required=True,
        help="exact serving image tag or digest recorded in the report",
    )
    artifact.add_argument(
        "--git-commit",
        type=_git_commit,
        help="Gridbook commit override; otherwise detected from this checkout",
    )
    artifact.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a dirty source checkout for research-only evidence",
    )
    artifact.add_argument(
        "--run-label",
        type=_run_label,
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
        "--artifact-inventory",
        type=Path,
        required=True,
        help=(
            "canonical standalone inventory or PrismaQuant quant_config.json "
            "whose file-byte sum must equal --artifact-bytes"
        ),
    )
    artifact.add_argument(
        "--artifact-inventory-sha256",
        type=_sha256,
        required=True,
        help="SHA-256 of the exact artifact-inventory JSON bytes",
    )
    artifact.add_argument(
        "--payload-bytes",
        type=_positive_int,
        help="optional model-payload bytes (a diagnostic, never the budget gate)",
    )
    artifact.add_argument(
        "--payload-scope",
        type=_nonempty,
        help="required explicit definition when --payload-bytes is supplied",
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
    accelerator_runtime = identity.add_mutually_exclusive_group(required=True)
    accelerator_runtime.add_argument(
        "--accelerator-runtime",
        dest="accelerator_runtime",
        type=_nonempty,
        help="accelerator runtime identity, for example CUDA 13.0 or ROCm 7.0",
    )
    accelerator_runtime.add_argument(
        "--cuda-version",
        dest="accelerator_runtime",
        type=_nonempty,
        help="deprecated alias for --accelerator-runtime",
    )
    identity.add_argument(
        "--execution-manifest",
        type=Path,
        required=True,
        help=(
            "gridbook.execution-manifest.v1 JSON enumerating uniform or mixed "
            "serving-unit assignments"
        ),
    )
    identity.add_argument(
        "--execution-manifest-sha256",
        type=_sha256,
        required=True,
        help="SHA-256 of the exact execution-manifest JSON bytes",
    )

    speculation = parser.add_argument_group("speculative decoding")
    speculation.add_argument(
        "--speculative-mode",
        choices=("off", "on"),
        required=True,
        help="whether this workload cell executes speculative decoding",
    )
    speculation.add_argument(
        "--speculative-config",
        type=_json_object,
        help=(
            "non-empty JSON object describing the exact server speculation "
            "configuration; required when --speculative-mode=on"
        ),
    )

    workload = parser.add_argument_group("fixed workload")
    workload.add_argument(
        "--input-len",
        type=_positive_int,
        required=True,
        help="requested vLLM random-dataset target before tokenizer adjustment",
    )
    workload.add_argument(
        "--observed-input-len",
        type=_positive_int,
        help=(
            "exact input_lens value expected in vLLM output; required when "
            "--input-range-ratio=0"
        ),
    )
    workload.add_argument(
        "--observed-input-len-min",
        type=_positive_int,
        help=(
            "declared inclusive lower input_lens bound; required with "
            "--observed-input-len-max when --input-range-ratio is nonzero"
        ),
    )
    workload.add_argument(
        "--observed-input-len-max",
        type=_positive_int,
        help=(
            "declared inclusive upper input_lens bound; required with "
            "--observed-input-len-min when --input-range-ratio is nonzero"
        ),
    )
    workload.add_argument("--output-len", type=_positive_int, required=True)
    workload.add_argument(
        "--num-prompts",
        type=_positive_int,
        required=True,
        help="measured prompts in each block",
    )
    workload.add_argument("--max-concurrency", type=_positive_int, required=True)
    workload.add_argument(
        "--warmups",
        type=_nonnegative_int,
        default=MIN_WARMUPS,
        help=f"unmeasured warmup prompts before each block (minimum: {MIN_WARMUPS})",
    )
    workload.add_argument(
        "--blocks",
        type=_positive_int,
        default=MIN_BLOCKS,
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
    workload.add_argument(
        "--expected-input-lens-sha256",
        action="append",
        default=[],
        type=_sha256,
        metavar="SHA256",
        help=(
            "canonical detailed input_lens JSON-array digest for one block; "
            "repeat in block order for nonzero --input-range-ratio"
        ),
    )

    dispatch = parser.add_argument_group("dispatch provenance")
    dispatch.add_argument(
        "--runner-env",
        action="append",
        default=[],
        type=_environment_name,
        metavar="NAME",
        help="benchmark-runner environment variable to record; repeat as needed",
    )
    dispatch.add_argument(
        "--runner-env-prefix",
        action="append",
        default=["PRISMAQUANT_"],
        type=_environment_prefix,
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
    output.add_argument("--vllm-executable", default="vllm", help="vLLM CLI executable")
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
    if args.warmups < MIN_WARMUPS:
        parser.error(f"--warmups must be at least {MIN_WARMUPS}")
    if args.blocks < MIN_BLOCKS:
        parser.error(f"--blocks must be at least {MIN_BLOCKS}")
    if args.input_range_ratio == 0:
        if args.observed_input_len is None:
            parser.error("--observed-input-len is required when --input-range-ratio=0")
        if (
            args.observed_input_len_min is not None
            or args.observed_input_len_max is not None
        ):
            parser.error(
                "--observed-input-len-min/--observed-input-len-max are forbidden "
                "when --input-range-ratio=0"
            )
        if args.expected_input_lens_sha256:
            parser.error(
                "--expected-input-lens-sha256 is forbidden when --input-range-ratio=0"
            )
    else:
        if args.observed_input_len is not None:
            parser.error(
                "--observed-input-len is forbidden when --input-range-ratio is nonzero"
            )
        if args.observed_input_len_min is None or args.observed_input_len_max is None:
            parser.error(
                "--observed-input-len-min and --observed-input-len-max are required "
                "when --input-range-ratio is nonzero"
            )
        if args.observed_input_len_min > args.observed_input_len_max:
            parser.error(
                "--observed-input-len-min cannot exceed --observed-input-len-max"
            )
        if len(args.expected_input_lens_sha256) != args.blocks:
            parser.error(
                "nonzero --input-range-ratio requires exactly one "
                "--expected-input-lens-sha256 per block"
            )
    if 95.0 not in {float(item) for item in args.percentiles.split(",")}:
        parser.error("--percentiles must include 95 for the TTFT/ITL measurement gates")
    server_env_names = [name for name, _ in args.server_env]
    if len(server_env_names) != len(set(server_env_names)):
        parser.error("--server-env names must not be repeated")
    if len(args.runner_env) != len(set(args.runner_env)):
        parser.error("--runner-env names must not be repeated")
    if len(args.runner_env_prefix) != len(set(args.runner_env_prefix)):
        parser.error("--runner-env-prefix values must not be repeated")
    if args.artifact_bytes > args.byte_budget:
        parser.error(
            f"--artifact-bytes ({args.artifact_bytes}) exceeds --byte-budget "
            f"({args.byte_budget})"
        )
    if (args.payload_bytes is None) != (args.payload_scope is None):
        parser.error("--payload-bytes and --payload-scope must be supplied together")
    if args.payload_bytes is not None and args.payload_bytes > args.artifact_bytes:
        parser.error("--payload-bytes cannot exceed --artifact-bytes")
    if args.speculative_mode == "on" and args.speculative_config is None:
        parser.error("--speculative-config is required when --speculative-mode=on")
    if args.speculative_mode == "off" and args.speculative_config is not None:
        parser.error("--speculative-config is forbidden when --speculative-mode=off")
    if args.speculative_mode == "on":
        speculative_tokens = args.speculative_config.get("num_speculative_tokens")
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or speculative_tokens <= 0
        ):
            parser.error(
                "--speculative-config must contain positive integer "
                "num_speculative_tokens"
            )
    try:
        _validate_recorded_prefix_caching(args.prefix_caching, args.server_arg)
        recorded_speculative_config = _recorded_speculative_config(args.server_arg)
        if args.speculative_mode == "on":
            if recorded_speculative_config is None:
                raise BenchmarkError(
                    "--speculative-mode=on requires the exact --speculative-config "
                    "in recorded --server-arg metadata"
                )
            if recorded_speculative_config != args.speculative_config:
                raise BenchmarkError(
                    "structured --speculative-config disagrees with recorded server args"
                )
        elif recorded_speculative_config is not None:
            raise BenchmarkError(
                "--speculative-mode=off conflicts with a recorded server "
                "--speculative-config"
            )
        inventory = _validate_artifact_inventory(
            args.artifact_inventory, args.artifact_inventory_sha256
        )
        if inventory["computed_total_bytes"] != args.artifact_bytes:
            raise BenchmarkError(
                "--artifact-bytes disagrees with the canonical inventory sum: "
                f"declared={args.artifact_bytes}, "
                f"computed={inventory['computed_total_bytes']}"
            )
        execution_manifest = _validate_execution_manifest(
            args.execution_manifest,
            args.execution_manifest_sha256,
            inventory["sha256"],
            args,
        )
        server_evidence = _validate_server_evidence(
            args.server_evidence, args.server_evidence_sha256
        )
    except BenchmarkError as exc:
        parser.error(str(exc))
    args.artifact_inventory_summary = inventory
    args.execution_manifest_summary = execution_manifest
    args.server_evidence_summary = server_evidence
    return args


def _revalidate_bound_inputs(args: argparse.Namespace) -> None:
    """Require every digest-bound input to match its parsed snapshot.

    ``parse_args`` validates these files before a report is reserved.  They are
    external mutable paths, however, so a long-lived caller could otherwise
    change one after parsing while the report continued to claim the old
    digest.  Comparing the complete normalized summaries also protects against
    a future validator accidentally changing semantics without changing this
    call site.
    """

    inventory = _validate_artifact_inventory(
        args.artifact_inventory, args.artifact_inventory_sha256
    )
    if inventory != args.artifact_inventory_summary:
        raise BenchmarkError(
            "artifact inventory changed after command-line validation"
        )
    execution_manifest = _validate_execution_manifest(
        args.execution_manifest,
        args.execution_manifest_sha256,
        inventory["sha256"],
        args,
    )
    if execution_manifest != args.execution_manifest_summary:
        raise BenchmarkError(
            "execution manifest changed after command-line validation"
        )
    server_evidence = _validate_server_evidence(
        args.server_evidence, args.server_evidence_sha256
    )
    if server_evidence != args.server_evidence_summary:
        raise BenchmarkError(
            "server evidence changed after command-line validation"
        )


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
        "--burstiness",
        str(REQUEST_BURSTINESS),
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
        # Unlike a query, a fragment is opaque to the server and may carry an
        # unkeyed credential.  There is no sound allowlist, so fail closed.
        fragment = "<redacted>" if parsed.fragment else ""
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))
    except (TypeError, ValueError):
        return "<redacted-invalid-url>"


def _redact_structured(value: Any, *, parent_key: str | None = None) -> Any:
    if parent_key is not None and _is_sensitive_key(parent_key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(key): _redact_structured(child, parent_key=str(key))
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_structured(child) for child in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_embedded_json(value: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            parsed, end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if value[index + end :].strip():
            continue
        redacted = json.dumps(
            _redact_structured(parsed), separators=(",", ":"), sort_keys=True
        )
        return _redact_plain_text(value[:index]) + redacted
    return None


def _is_sensitive_option(value: str) -> bool:
    return value in _SENSITIVE_OPTIONS or _is_sensitive_key(value)


def _redact_plain_text(value: str) -> str:
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

    redacted = _URL.sub(redact_url, str(value))
    redacted = _SENSITIVE_HEADER.sub(
        lambda match: match.group(1) + "<redacted>", redacted
    )
    redacted = _ASSIGNMENT.sub(redact_assignment, redacted)
    redacted = _OPTION_VALUE.sub(redact_option, redacted)
    return _AUTH_HEADER.sub(lambda match: match.group(1) + "<redacted>", redacted)


def _redact_text(value: str) -> str:
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        return json.dumps(
            _redact_structured(parsed), separators=(",", ":"), sort_keys=True
        )

    option_equals = re.match(r"^(--?[-\w.]+)=(.*)$", text, re.DOTALL)
    if option_equals:
        option, setting = option_equals.groups()
        if _is_sensitive_option(option):
            return option + "=<redacted>"
        return option + "=" + _redact_text(setting)
    option_space = re.match(r"^(--?[-\w.]+)(\s+)(.*)$", text, re.DOTALL)
    if option_space:
        option, spacing, setting = option_space.groups()
        if _is_sensitive_option(option):
            return option + spacing + "<redacted>"
        return option + spacing + _redact_text(setting)

    embedded = _redact_embedded_json(text)
    if embedded is not None:
        return embedded
    return _redact_plain_text(text)


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
            hide_next = _is_sensitive_option(str(item))
    return redacted


def _git_state(
    commit_override: str | None,
    *,
    allow_dirty: bool = False,
    generated_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    if commit_override and not _GIT_COMMIT.fullmatch(commit_override):
        raise BenchmarkError("explicit Gridbook commit is not exact 40/64 hex")

    detected_commit: str | None = None
    detected_dirty: bool | None = None
    unavailable_source = "unavailable-non-source-install"
    source_file = Path(__file__).resolve()
    checkout = source_file.parents[1]
    expected_source = checkout / "gridbook" / "bench_serve.py"
    if (checkout / "pyproject.toml").is_file() and expected_source.is_file():
        unavailable_source = "unavailable"
        try:
            if not source_file.samefile(expected_source):
                raise OSError("source file is not rooted at the candidate checkout")
            top_level = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if Path(top_level).resolve() != checkout:
                unavailable_source = "unavailable-root-mismatch"
            else:
                detected_commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                status_command = [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    ".",
                ]
                for generated_path in generated_paths:
                    normalized_path = Path(
                        os.path.abspath(os.fspath(generated_path.expanduser()))
                    )
                    try:
                        relative_path = normalized_path.relative_to(checkout)
                    except ValueError:
                        continue
                    status_command.append(
                        ":(exclude,top,literal)" + relative_path.as_posix()
                    )
                detected_dirty = bool(
                    subprocess.run(
                        status_command,
                        cwd=checkout,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )
                if not _GIT_COMMIT.fullmatch(detected_commit):
                    detected_commit = None
                    unavailable_source = "unavailable-invalid-commit"
        except (OSError, subprocess.CalledProcessError):
            detected_commit = None
            detected_dirty = None

    if detected_dirty and not allow_dirty:
        raise BenchmarkError(
            "Gridbook checkout is dirty; commit the changes or pass --allow-dirty "
            "for research-only evidence"
        )
    if (
        commit_override is not None
        and detected_commit is not None
        and commit_override.lower() != detected_commit.lower()
    ):
        raise BenchmarkError(
            "explicit Gridbook commit disagrees with the exact source checkout: "
            f"argument={commit_override.lower()}, checkout={detected_commit.lower()}"
        )
    if commit_override:
        return {
            "commit": commit_override.lower(),
            "dirty": detected_dirty,
            "source": (
                "argument+checkout" if detected_commit is not None else "argument"
            ),
            # An assertion supplied beside an installed wheel is useful
            # research metadata, but it is not proof that the executing bytes
            # came from that commit.  Release eligibility requires the exact
            # clean checkout that supplied this module.
            "release_eligible": (
                detected_commit is not None
                and detected_dirty is False
                and not allow_dirty
            ),
        }
    if detected_commit is None:
        return {
            "commit": None,
            "dirty": detected_dirty,
            "source": unavailable_source,
            "release_eligible": False,
        }
    return {
        "commit": detected_commit.lower(),
        "dirty": detected_dirty,
        "source": "checkout",
        "release_eligible": not allow_dirty and not detected_dirty,
    }


def _package_version() -> str | None:
    # Prefer the package that supplied this module.  Distribution metadata can
    # describe a different globally installed Gridbook when a source checkout
    # is selected via PYTHONPATH or an editable test setup.
    try:
        from gridbook import __version__

        return __version__
    except (ImportError, AttributeError):
        try:
            return importlib.metadata.version("gridbook")
        except importlib.metadata.PackageNotFoundError:
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

    _revalidate_bound_inputs(args)
    git = _git_state(
        args.git_commit,
        allow_dirty=args.allow_dirty,
        generated_paths=(args.output,),
    )
    client_runtime_probe = _vllm_version(args.vllm_executable)
    if client_runtime_probe is None:
        raise BenchmarkError(
            "vLLM client version probe failed; --client-runtime-id cannot be verified"
        )
    if client_runtime_probe != args.client_runtime_id:
        raise BenchmarkError(
            "--client-runtime-id disagrees with the successful vLLM --version probe"
        )
    runner_env = os.environ if env is None else env
    explicit_server_env = {
        key: "<redacted>" if _is_sensitive_key(key) else _redact_text(value)
        for key, value in sorted(args.server_env)
    }
    metadata = {
        "git": git,
        "measurement_provenance": {
            "digest_bound_inputs_verified_before_requests": True,
            "digest_bound_inputs_verified_after_requests": False,
            "git_state_verified_after_requests": False,
            "client_runtime_verified_after_requests": False,
        },
        "software": {
            "gridbook_version": probe("gridbook_version", _package_version),
            "runner_vllm_cli_probe": client_runtime_probe,
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
            "whole_served_artifact_bytes": args.artifact_inventory_summary[
                "computed_total_bytes"
            ],
            "byte_budget_bytes": args.byte_budget,
            "budget_headroom_bytes": args.byte_budget - args.artifact_bytes,
            "within_byte_budget": args.artifact_bytes <= args.byte_budget,
            "payload": (
                {
                    "bytes": args.payload_bytes,
                    "scope": _redact_text(args.payload_scope),
                    "is_budget_gate": False,
                }
                if args.payload_bytes is not None
                else None
            ),
            "inventory": _redact_structured(
                {
                    **args.artifact_inventory_summary,
                    "reference": _redact_text(
                        args.artifact_inventory_summary["reference"]
                    ),
                }
            ),
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
            "manifest": _redact_structured(
                {
                    **args.execution_manifest_summary,
                    "reference": _redact_text(
                        args.execution_manifest_summary["reference"]
                    ),
                }
            ),
            "client_runtime_id": _redact_text(args.client_runtime_id),
            "server_runtime_id": _redact_text(args.server_runtime_id),
            "hardware": {
                "gpu_id": _redact_text(args.gpu_id),
                "driver_version": _redact_text(args.driver_version),
                "accelerator_runtime": _redact_text(args.accelerator_runtime),
            },
        },
        "server": {
            "base_url": _redact_url(args.base_url.rstrip("/")),
            "backend": args.backend,
            "endpoint": _redact_text(args.endpoint),
            "served_model_name": _redact_text(args.served_model_name or args.model),
            "recorded_args": redact_command(args.server_arg),
            "prefix_caching": args.prefix_caching,
            "evidence": {
                "scope": (
                    "digest-bound external dispatch/startup attachments; the "
                    "harness verifies bytes, not semantic backend claims"
                ),
                "attachments": [
                    {
                        **attachment,
                        "reference": _redact_text(attachment["reference"]),
                    }
                    for attachment in args.server_evidence_summary
                ],
            },
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
            "requested_random_input_len": args.input_len,
            "observed_input_length_contract": (
                {
                    "mode": "exact",
                    "value": args.observed_input_len,
                }
                if args.input_range_ratio == 0
                else {
                    "mode": "range",
                    "minimum": args.observed_input_len_min,
                    "maximum": args.observed_input_len_max,
                }
            ),
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
            "request_burstiness": REQUEST_BURSTINESS,
            "input_range_ratio": args.input_range_ratio,
            "vllm_range_ratio": {
                "input": args.input_range_ratio,
                "output": 0.0,
            },
            "input_length_validation": (
                "declared-exact" if args.input_range_ratio == 0 else "declared-range"
            ),
            "accepted_input_length_bounds": list(_observed_input_length_bounds(args)),
            "expected_input_lens_sha256_by_block": [
                {
                    "block": index + 1,
                    "dataset_seed": args.dataset_seed + index,
                    "sha256": digest,
                }
                for index, digest in enumerate(args.expected_input_lens_sha256)
            ],
            "aggregate_reconciliation": {
                "throughput": (
                    "duration and exact token/request totals; 1e-9 relative/absolute"
                ),
                "ttft_itl": (
                    "mean/median/population-std/all requested percentiles vs "
                    "detailed arrays; NumPy-linear percentile semantics; "
                    "1e-6 ms or 1e-9 relative"
                ),
                "e2el": (
                    "all aggregates vs per-request TTFT+sum(ITL); 5 ms or 0.5% "
                    "terminal-SSE allowance"
                ),
                "tpot": (
                    "all aggregates vs per-request sum(ITL)/(fixed_output_len-1); "
                    "terminal-SSE allowance divided by output_len-1"
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
            "speculative_decoding": {
                "mode": args.speculative_mode,
                "config": _redact_structured(args.speculative_config),
                "result_contract": (
                    "all vLLM spec_decode_* fields required and reconciled"
                    if args.speculative_mode == "on"
                    else "spec_decode_* fields forbidden"
                ),
            },
        },
    }
    metadata["collection_errors"] = probe_errors
    return metadata


def _load_result(path: Path) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise BenchmarkError(f"vLLM did not create result file {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise BenchmarkError(f"vLLM result is not strict JSON: {path}") from exc
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
    return f"p{_percentile_label(percentile)}_{metric}_ms"


def _observed_input_length_bounds(args: argparse.Namespace) -> tuple[int, int]:
    if args.input_range_ratio == 0:
        return args.observed_input_len, args.observed_input_len
    return args.observed_input_len_min, args.observed_input_len_max


def _canonical_input_lens_sha256(input_lens: Sequence[int]) -> str:
    canonical = json.dumps(
        list(input_lens),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _validate_result_invocation(
    result: Mapping[str, Any], args: argparse.Namespace
) -> None:
    mismatches: list[str] = []
    exact_strings = {
        "backend": args.backend,
        "endpoint_type": args.backend,
        "label": args.run_label,
        "model_id": args.model,
        "tokenizer_id": args.tokenizer or args.model,
    }
    for field, expected in exact_strings.items():
        value = result.get(field)
        if not isinstance(value, str) or value != expected:
            mismatches.append(field)

    exact_integers = {
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
    }
    for field, expected in exact_integers.items():
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            mismatches.append(field)

    request_rate = result.get("request_rate")
    if args.request_rate == "inf":
        request_rate_matches = request_rate == "inf"
    else:
        request_rate_matches = (
            not isinstance(request_rate, bool)
            and isinstance(request_rate, (int, float))
            and math.isfinite(float(request_rate))
            and float(request_rate) == float(args.request_rate)
        )
    if not request_rate_matches:
        mismatches.append("request_rate")

    burstiness = result.get("burstiness")
    if (
        isinstance(burstiness, bool)
        or not isinstance(burstiness, (int, float))
        or not math.isfinite(float(burstiness))
        or float(burstiness) != REQUEST_BURSTINESS
    ):
        mismatches.append("burstiness")

    if mismatches:
        raise BenchmarkError(
            "vLLM result invocation fields disagree with the pinned client command: "
            + ", ".join(mismatches)
        )


def _numpy_linear_percentile(samples: Sequence[float], percentile: float) -> float:
    """NumPy's default percentile method without adding a NumPy dependency."""

    if not samples:
        return 0.0
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sample_aggregates(
    samples_ms: Sequence[float], requested_percentiles: Sequence[str]
) -> dict[str, float]:
    samples = [float(value) for value in samples_ms]
    aggregates = {
        "mean": statistics.fmean(samples) if samples else 0.0,
        "median": statistics.median(samples) if samples else 0.0,
        "std": statistics.pstdev(samples) if samples else 0.0,
    }
    for percentile in requested_percentiles:
        aggregates[f"p{_percentile_label(percentile)}"] = _numpy_linear_percentile(
            samples, float(percentile)
        )
    return aggregates


def _aggregate_tolerance_ms(metric: str, expected_ms: float, output_len: int) -> float:
    if metric == "e2el":
        # vLLM's request latency extends through the terminal SSE event, while
        # detailed ITLs stop at the final token-bearing chunk.  Permit that
        # small transport tail, but not a fabricated decode-length gap.
        return max(5.0, abs(expected_ms) * 0.005)
    if metric == "tpot" and output_len > 1:
        return max(5.0 / (output_len - 1), abs(expected_ms) * 0.005)
    return max(0.000001, abs(expected_ms) * 1e-9)


def _reconcile_detailed_aggregate(
    result: Mapping[str, Any],
    metric: str,
    samples_ms: Sequence[float],
    requested_percentiles: Sequence[str],
    output_len: int,
) -> None:
    expected = _sample_aggregates(samples_ms, requested_percentiles)
    for statistic_name, expected_ms in expected.items():
        if statistic_name.startswith("p"):
            percentile = statistic_name[1:]
            key = _percentile_result_key(percentile, metric)
        else:
            key = f"{statistic_name}_{metric}_ms"
        actual_ms = _number(result, key)
        tolerance_ms = _aggregate_tolerance_ms(metric, expected_ms, output_len)
        if actual_ms is None or abs(actual_ms - expected_ms) > tolerance_ms:
            raise BenchmarkError(
                f"{key} disagrees with detailed streaming evidence: "
                f"reported={actual_ms}, reconstructed={expected_ms:.6f}, "
                f"tolerance={tolerance_ms:.6f} ms"
            )


_SPEC_RESULT_FIELDS = (
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
    "spec_decode_num_drafts",
    "spec_decode_draft_tokens",
    "spec_decode_accepted_tokens",
    "spec_decode_per_position_acceptance_rates",
)


def _exact_nonnegative_int(result: Mapping[str, Any], key: str) -> int | None:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validate_speculative_result(
    result: Mapping[str, Any], args: argparse.Namespace
) -> None:
    present = [key for key in _SPEC_RESULT_FIELDS if key in result]
    if args.speculative_mode == "off":
        if present:
            raise BenchmarkError(
                "non-speculative cell contains speculative result fields: "
                + ", ".join(present)
            )
        return

    missing = [key for key in _SPEC_RESULT_FIELDS if key not in result]
    if missing:
        raise BenchmarkError(
            "speculative cell lacks required vLLM result fields: " + ", ".join(missing)
        )
    drafts = _exact_nonnegative_int(result, "spec_decode_num_drafts")
    draft_tokens = _exact_nonnegative_int(result, "spec_decode_draft_tokens")
    accepted = _exact_nonnegative_int(result, "spec_decode_accepted_tokens")
    if drafts is None or drafts <= 0 or draft_tokens is None or draft_tokens <= 0:
        raise BenchmarkError("speculative draft counts must be positive integers")
    if accepted is None or accepted > draft_tokens:
        raise BenchmarkError(
            "spec_decode_accepted_tokens must be a non-negative integer no larger "
            "than spec_decode_draft_tokens"
        )
    rate = _number(result, "spec_decode_acceptance_rate")
    length = _number(result, "spec_decode_acceptance_length")
    if rate is None or not 0 <= rate <= 100:
        raise BenchmarkError("spec_decode_acceptance_rate must be in [0, 100]")
    if length is None or length < 1:
        raise BenchmarkError("spec_decode_acceptance_length must be at least 1")
    expected_rate = accepted / draft_tokens * 100
    expected_length = 1 + accepted / drafts
    if not math.isclose(rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9):
        raise BenchmarkError(
            "spec_decode_acceptance_rate does not reconcile with accepted/draft tokens"
        )
    if not math.isclose(length, expected_length, rel_tol=1e-9, abs_tol=1e-9):
        raise BenchmarkError(
            "spec_decode_acceptance_length does not reconcile with accepted tokens "
            "per draft"
        )

    per_position = result.get("spec_decode_per_position_acceptance_rates")
    if not isinstance(per_position, list) or not per_position:
        raise BenchmarkError(
            "spec_decode_per_position_acceptance_rates must be a non-empty array"
        )
    bad_positions = [
        index
        for index, value in enumerate(per_position)
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ]
    if bad_positions:
        raise BenchmarkError(
            "speculative per-position acceptance rates must be finite values in [0, 1]"
        )
    configured_tokens = args.speculative_config.get("num_speculative_tokens")
    if (
        isinstance(configured_tokens, bool)
        or not isinstance(configured_tokens, int)
        or configured_tokens <= 0
    ):
        raise BenchmarkError(
            "--speculative-config must contain positive integer num_speculative_tokens"
        )
    if len(per_position) != configured_tokens:
        raise BenchmarkError(
            "speculative per-position rate count disagrees with num_speculative_tokens"
        )
    if not drafts <= draft_tokens <= drafts * configured_tokens:
        raise BenchmarkError(
            "spec_decode_draft_tokens is outside the feasible range from "
            "num_drafts and declared num_speculative_tokens"
        )
    expected_accepted_per_draft = accepted / drafts
    if not math.isclose(
        sum(map(float, per_position)),
        expected_accepted_per_draft,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise BenchmarkError(
            "speculative per-position rates do not reconcile with accepted tokens"
        )


def _validate_result_envelope(
    result: Mapping[str, Any], args: argparse.Namespace
) -> None:
    """Validate the complete saved-result shape of the pinned vLLM client."""

    expected_fields = set(_PINNED_RESULT_FIELDS)
    for percentile in args.percentiles.split(","):
        for metric in ("ttft", "tpot", "itl", "e2el"):
            expected_fields.add(_percentile_result_key(percentile, metric))
    if args.speculative_mode == "on":
        expected_fields.update(_SPEC_RESULT_FIELDS)

    observed_fields = set(result)
    missing = sorted(expected_fields - observed_fields)
    unexpected = sorted(observed_fields - expected_fields)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise BenchmarkError(
            "vLLM result envelope disagrees with the pinned client: "
            + "; ".join(details)
        )

    date = result.get("date")
    if not isinstance(date, str) or not _VLLM_RESULT_DATE.fullmatch(date):
        raise BenchmarkError("vLLM result date does not use YYYYMMDD-HHMMSS")

    generated_texts = result.get("generated_texts")
    if (
        not isinstance(generated_texts, list)
        or len(generated_texts) != args.num_prompts
        or any(not isinstance(value, str) for value in generated_texts)
    ):
        raise BenchmarkError(
            "vLLM detailed result generated_texts must contain one string per prompt"
        )

    start_times = result.get("start_times")
    if not isinstance(start_times, list) or len(start_times) != args.num_prompts:
        raise BenchmarkError(
            "vLLM detailed result start_times must contain one value per prompt"
        )
    bad_start_times = [
        index
        for index, value in enumerate(start_times)
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ]
    if bad_start_times:
        raise BenchmarkError(
            "vLLM detailed result has invalid start_times at request index: "
            + ", ".join(str(index) for index in bad_start_times[:5])
        )

    if result.get("request_goodput") is not None:
        raise BenchmarkError(
            "vLLM result request_goodput must be null without a goodput command"
        )
    peak_output = _number(result, "max_output_tokens_per_s")
    if peak_output is None or peak_output < 0:
        raise BenchmarkError(
            "vLLM result max_output_tokens_per_s must be finite and non-negative"
        )
    peak_concurrency = _exact_nonnegative_int(result, "max_concurrent_requests")
    if peak_concurrency is None or not 1 <= peak_concurrency <= args.num_prompts:
        raise BenchmarkError(
            "vLLM result max_concurrent_requests must be a positive integer no "
            "larger than num_prompts"
        )
    rtfx = _number(result, "rtfx")
    if rtfx != 0:
        raise BenchmarkError("vLLM result rtfx must be zero for the random text dataset")


def validate_result(
    result: Mapping[str, Any], args: argparse.Namespace, *, block_index: int = 0
) -> None:
    nonfinite = _nonfinite_paths(result)
    if nonfinite:
        raise BenchmarkError(
            "vLLM result contains NaN/Infinity at: " + ", ".join(nonfinite[:10])
        )
    _validate_result_invocation(result, args)
    _validate_speculative_result(result, args)

    duration = _number(result, "duration")
    if duration is None or duration <= 0:
        raise BenchmarkError("vLLM result duration must be finite and positive")

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

    expected_counts: dict[str, int] = {
        "completed": args.num_prompts,
        "failed": 0,
        "total_output_tokens": args.num_prompts * args.output_len,
    }
    if args.input_range_ratio == 0:
        expected_counts["total_input_tokens"] = (
            args.num_prompts * args.observed_input_len
        )
    mismatches = []
    for label, wanted in expected_counts.items():
        actual = _exact_nonnegative_int(result, label)
        if actual is None:
            mismatches.append(f"{label}=missing (expected {wanted})")
        elif actual != wanted:
            mismatches.append(f"{label}={actual} (expected {wanted})")
    if mismatches:
        raise BenchmarkError(
            "fixed-shape block was not completed exactly: " + "; ".join(mismatches)
        )

    input_lens = result.get("input_lens")
    if not isinstance(input_lens, list) or len(input_lens) != args.num_prompts:
        raise BenchmarkError(
            "vLLM detailed result input_lens must contain one value per prompt"
        )
    lower, upper = _observed_input_length_bounds(args)
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
    observed_input_lens_digest = _canonical_input_lens_sha256(input_lens)
    if args.input_range_ratio != 0:
        try:
            expected_input_lens_digest = args.expected_input_lens_sha256[block_index]
        except IndexError as exc:
            raise BenchmarkError(
                f"no expected input_lens digest is bound to block {block_index + 1}"
            ) from exc
        if observed_input_lens_digest != expected_input_lens_digest:
            raise BenchmarkError(
                f"block {block_index + 1} input_lens SHA-256 disagrees with the "
                "declared canonical vector digest"
            )
    total_input = _exact_nonnegative_int(result, "total_input_tokens")
    observed_total_input = sum(input_lens)
    if total_input != observed_total_input:
        raise BenchmarkError(
            "input lengths do not reconcile with total_input_tokens: "
            f"{total_input} != {observed_total_input}"
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
            "fixed output block has unexpected output_lens at request index: " + preview
        )
    total_output = _exact_nonnegative_int(result, "total_output_tokens")
    observed_total_output = sum(output_lens)
    if total_output != observed_total_output:
        raise BenchmarkError(
            "output lengths do not reconcile with total_output_tokens: "
            f"{total_output} != {observed_total_output}"
        )

    expected_throughputs = {
        "request_throughput": args.num_prompts / duration,
        "output_throughput": observed_total_output / duration,
        "total_token_throughput": (observed_total_input + observed_total_output)
        / duration,
    }
    for key, expected_throughput in expected_throughputs.items():
        actual_throughput = _number(result, key)
        if actual_throughput is None or not math.isclose(
            actual_throughput,
            expected_throughput,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise BenchmarkError(
                f"{key} disagrees with duration and exact totals: "
                f"reported={actual_throughput}, "
                f"reconstructed={expected_throughput:.12g}"
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
        if args.output_len == 1 and request_itls:
            raise BenchmarkError(
                "single-token response unexpectedly contains inter-token latency "
                f"samples at request index {request_index}"
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
        float(ttft) + sum(map(float, intervals)) for ttft, intervals in zip(ttfts, itls)
    ]
    if any(not math.isfinite(value) or value <= 0 for value in e2els):
        raise BenchmarkError("derived detailed E2EL contains a non-finite value")

    _reconcile_detailed_aggregate(
        result,
        "ttft",
        [float(value) * 1000 for value in ttfts],
        requested_percentiles,
        args.output_len,
    )
    _reconcile_detailed_aggregate(
        result,
        "itl",
        [float(value) * 1000 for intervals in itls for value in intervals],
        requested_percentiles,
        args.output_len,
    )
    _reconcile_detailed_aggregate(
        result,
        "e2el",
        [value * 1000 for value in e2els],
        requested_percentiles,
        args.output_len,
    )
    tpot_samples = (
        [
            sum(map(float, intervals)) * 1000 / (args.output_len - 1)
            for intervals in itls
        ]
        if args.output_len > 1
        else []
    )
    _reconcile_detailed_aggregate(
        result,
        "tpot",
        tpot_samples,
        requested_percentiles,
        args.output_len,
    )
    _validate_result_envelope(result, args)


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
            "block_p05": _numpy_linear_percentile(numeric, 5),
            "block_p95": _numpy_linear_percentile(numeric, 95),
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


def _sanitize_result_for_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Retain metric evidence without persisting arbitrary model/server text.

    ``vllm bench serve --save-detailed`` includes generated completions and may
    include an arbitrary server error body. Neither is needed to reconstruct
    the performance metrics, and generic credential-pattern redaction cannot
    prove such free text is safe to publish. Keep their cardinality/position
    evidence while omitting their contents, then apply the normal recursive
    redaction and non-finite JSON conversion to every remaining field.
    """

    sanitized = dict(result)
    generated = sanitized.pop("generated_texts", None)
    if generated is not None:
        sanitized["generated_texts_omitted"] = {
            "count": len(generated) if isinstance(generated, list) else None,
            "reason": "arbitrary model output is not performance evidence",
        }

    if "errors" in sanitized:
        errors = sanitized.pop("errors")
        sanitized["errors_omitted"] = {
            "count": len(errors) if isinstance(errors, list) else None,
            "reason": "arbitrary server error text is not performance evidence",
        }

    return _json_safe(_redact_structured(sanitized))


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
        "run_label": _redact_text(args.run_label),
        "evidence_scope": "single-arm-serving-measurement",
        "measurement_valid": False,
        "parity_acceptance": False,
        "release_acceptance": False,
        "release_eligible": False,
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
                    "expected_input_lens_sha256": (
                        args.expected_input_lens_sha256[block_index]
                        if args.input_range_ratio != 0
                        else None
                    ),
                    "observed_input_lens_sha256": None,
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
                        block["raw_result"] = _sanitize_result_for_report(result)
                        _atomic_write_json(output_path, report)
                        validate_result(result, args, block_index=block_index)
                        block["observed_input_lens_sha256"] = (
                            _canonical_input_lens_sha256(result["input_lens"])
                        )
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
        _revalidate_bound_inputs(args)
        provenance = report["metadata"]["measurement_provenance"]
        provenance["digest_bound_inputs_verified_after_requests"] = True
        ending_git = _git_state(
            args.git_commit,
            allow_dirty=args.allow_dirty,
            generated_paths=(output_path,),
        )
        if ending_git != report["metadata"]["git"]:
            raise BenchmarkError(
                "Gridbook source provenance changed during the benchmark"
            )
        provenance["git_state_verified_after_requests"] = True
        ending_client_runtime = _vllm_version(args.vllm_executable)
        if ending_client_runtime != args.client_runtime_id:
            raise BenchmarkError(
                "vLLM client runtime identity changed during the benchmark"
            )
        provenance["client_runtime_verified_after_requests"] = True
        report["summary"] = summarize_blocks(report["blocks"])
        report["status"] = "success"
        report["measurement_valid"] = True
        report["release_eligible"] = report["metadata"]["git"]["release_eligible"]
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
