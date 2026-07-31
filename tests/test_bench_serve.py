"""CPU-only tests for the reproducible online parity harness.

The real client is ``vllm bench serve``.  These tests pin command construction,
fixed-shape validation, provenance capture, and the report schema without a
vLLM install, a GPU, or a listening server.
"""

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gridbook import bench_serve

CLIENT_RUNTIME_ID = "vLLM 0.23.1rc1.dev764+g54b16d8a9.d20260703"


def _gridbook_commit_for_test_process():
    """Use the exact checkout commit when the tests import a source tree.

    Installed-wheel CI runs outside a checkout, where an explicit synthetic
    commit is the intended provenance fixture.  A source-tree run must instead
    agree with the checkout that actually supplied ``bench_serve.py``.
    """

    source = Path(bench_serve.__file__).resolve()
    checkout = source.parents[1]
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "d" * 40
    if Path(top_level).resolve() != checkout or not bench_serve._GIT_COMMIT.fullmatch(
        commit
    ):
        return "d" * 40
    return commit.lower()


GRIDBOOK_COMMIT = _gridbook_commit_for_test_process()


def _argv(tmp_path):
    inventory_path = tmp_path / "artifact-inventory.json"
    inventory = {
        "schema": bench_serve.ARTIFACT_INVENTORY_SCHEMA,
        "total_bytes": 800,
        "files": [
            {"path": "model.safetensors", "bytes": 700, "sha256": "a" * 64},
            {"path": "config.json", "bytes": 100, "sha256": "b" * 64},
        ],
    }
    inventory_raw = json.dumps(inventory, sort_keys=True).encode()
    inventory_path.write_bytes(inventory_raw)
    inventory_digest = hashlib.sha256(inventory_raw).hexdigest()

    execution_path = tmp_path / "execution-manifest.json"
    execution = {
        "schema": bench_serve.EXECUTION_MANIFEST_SCHEMA,
        "coverage": "all_serving_units",
        "artifact_inventory_sha256": inventory_digest,
        "tensor_parallel_size": 1,
        "assignments": [
            {
                "unit": "model.layers.0.self_attn.q_proj",
                "format_rung": "FP8_CB_K36",
                "serialized_layout": "product-codebook-indices-v1",
                "scale_coding": "e4m3-per-block",
                "quant_contract": "W8A8",
                "kernel_backend": "gridbook-cuda-cb-gemv-v2",
                "fallback_state": "none-observed",
            }
        ],
    }
    execution_raw = json.dumps(execution, sort_keys=True).encode()
    execution_path.write_bytes(execution_raw)
    execution_digest = hashlib.sha256(execution_raw).hexdigest()

    server_evidence_path = tmp_path / "server-startup.log"
    server_evidence_raw = b"dispatch=gridbook-cuda-cb-gemv-v2 fallback=none\n"
    server_evidence_path.write_bytes(server_evidence_raw)
    server_evidence_digest = hashlib.sha256(server_evidence_raw).hexdigest()
    return [
        "--base-url",
        "http://127.0.0.1:8000/",
        "--model",
        "org/tokenizer",
        "--served-model-name",
        "served",
        "--prefix-caching",
        "off",
        "--server-arg=--no-enable-prefix-caching",
        "--server-evidence",
        str(server_evidence_path),
        "--server-evidence-sha256",
        server_evidence_digest,
        "--model-id",
        "org/artifact@0123456",
        "--image-id",
        "registry/vllm@sha256:abc",
        "--git-commit",
        GRIDBOOK_COMMIT,
        "--run-label",
        "gridbook-0.6b-decode",
        "--artifact-bytes",
        "800",
        "--artifact-inventory",
        str(inventory_path),
        "--artifact-inventory-sha256",
        inventory_digest,
        "--byte-budget",
        "900",
        "--format-rung",
        "FP8_CB_K36",
        "--serialized-layout",
        "product-codebook-indices-v1",
        "--scale-coding",
        "e4m3-per-block",
        "--quant-contract",
        "W8A8",
        "--kernel-backend",
        "gridbook-cuda-cb-gemv-v2",
        "--tensor-parallel-size",
        "1",
        "--fallback-state",
        "none-observed",
        "--client-runtime-id",
        CLIENT_RUNTIME_ID,
        "--server-runtime-id",
        "vllm@g54b16d8a9+gridbook-0.3.0",
        "--gpu-id",
        "NVIDIA-GB10:GPU-1234",
        "--driver-version",
        "580.00",
        "--accelerator-runtime",
        "CUDA 13.0",
        "--execution-manifest",
        str(execution_path),
        "--execution-manifest-sha256",
        execution_digest,
        "--speculative-mode",
        "off",
        "--input-len",
        "32",
        "--observed-input-len",
        "32",
        "--output-len",
        "256",
        "--num-prompts",
        "8",
        "--max-concurrency",
        "1",
        "--output",
        str(tmp_path / "report.json"),
    ]


def _parse(tmp_path, *extra):
    return bench_serve.parse_args([*_argv(tmp_path), *extra])


def _option(command, name):
    return command[command.index(name) + 1]


def _without_option(argv, name):
    index = argv.index(name)
    return [*argv[:index], *argv[index + 2 :]]


def _replace_option(argv, name, value):
    replaced = list(argv)
    index = replaced.index(name)
    replaced[index + 1] = str(value)
    return replaced


def _write_json_digest(path, payload):
    raw = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _range_argv(tmp_path, input_lens, *, ratio="0.25", lower=24, upper=40):
    argv = _without_option(_argv(tmp_path), "--observed-input-len")
    digest = bench_serve._canonical_input_lens_sha256(input_lens)
    argv.extend(
        [
            "--input-range-ratio",
            ratio,
            "--observed-input-len-min",
            str(lower),
            "--observed-input-len-max",
            str(upper),
        ]
    )
    for _ in range(3):
        argv.extend(["--expected-input-lens-sha256", digest])
    return argv


def _reconcile_throughput(result):
    result["request_throughput"] = result["completed"] / result["duration"]
    result["output_throughput"] = result["total_output_tokens"] / result["duration"]
    result["total_token_throughput"] = (
        result["total_input_tokens"] + result["total_output_tokens"]
    ) / result["duration"]


def _valid_result(args, *, offset=0.0):
    multi_token = args.output_len > 1
    ttft_ms = 20.0 + offset
    # Detailed ITLs describe streamed chunk arrivals, not token cardinality.
    # Keeping one interval for a 256-token response protects compatibility with
    # bundled/speculatively accepted token chunks.
    itl_ms = 2.0 + offset if multi_token else 0.0
    e2el_ms = ttft_ms + itl_ms
    tpot_ms = itl_ms / (args.output_len - 1) if multi_token else 0.0
    duration = 4.0
    observed_input_len = (
        args.observed_input_len
        if args.observed_input_len is not None
        else args.observed_input_len_min
    )
    total_input = args.num_prompts * observed_input_len
    total_output = args.num_prompts * args.output_len
    result = {
        "date": "20260731-120000",
        "endpoint_type": args.backend,
        "backend": args.backend,
        "label": args.run_label,
        "model_id": args.model,
        "tokenizer_id": args.tokenizer or args.model,
        "num_prompts": args.num_prompts,
        "request_rate": (
            "inf" if args.request_rate == "inf" else float(args.request_rate)
        ),
        "burstiness": 1.0,
        "max_concurrency": args.max_concurrency,
        "duration": duration,
        "completed": args.num_prompts,
        "failed": 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_lens": [observed_input_len] * args.num_prompts,
        "output_lens": [args.output_len] * args.num_prompts,
        "errors": [""] * args.num_prompts,
        "generated_texts": ["synthetic output"] * args.num_prompts,
        "start_times": [float(index) for index in range(args.num_prompts)],
        "request_goodput": None,
        "max_output_tokens_per_s": total_output / duration,
        "max_concurrent_requests": args.max_concurrency,
        "rtfx": 0.0,
        "ttfts": [ttft_ms / 1000] * args.num_prompts,
        "itls": [
            [itl_ms / 1000] if multi_token else [] for _ in range(args.num_prompts)
        ],
        "request_throughput": args.num_prompts / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "mean_ttft_ms": ttft_ms,
        "median_ttft_ms": ttft_ms,
        "std_ttft_ms": 0.0,
        "mean_tpot_ms": tpot_ms,
        "median_tpot_ms": tpot_ms,
        "std_tpot_ms": 0.0,
        "mean_itl_ms": itl_ms,
        "median_itl_ms": itl_ms,
        "std_itl_ms": 0.0,
        "mean_e2el_ms": e2el_ms,
        "median_e2el_ms": e2el_ms,
        "std_e2el_ms": 0.0,
    }
    percentile_values = {
        "ttft": ttft_ms,
        "tpot": tpot_ms,
        "itl": itl_ms,
        "e2el": e2el_ms,
    }
    for percentile in args.percentiles.split(","):
        for metric, value in percentile_values.items():
            key = bench_serve._percentile_result_key(percentile, metric)
            result[key] = value
    return result


def _add_valid_speculative_result(result, *, drafts=10, tokens=2):
    accepted_by_position = [0.8, 0.4][:tokens]
    if len(accepted_by_position) < tokens:
        accepted_by_position.extend([0.0] * (tokens - len(accepted_by_position)))
    accepted = round(sum(accepted_by_position) * drafts)
    draft_tokens = drafts * tokens
    result.update(
        {
            "spec_decode_acceptance_rate": accepted / draft_tokens * 100,
            "spec_decode_acceptance_length": 1 + accepted / drafts,
            "spec_decode_num_drafts": drafts,
            "spec_decode_draft_tokens": draft_tokens,
            "spec_decode_accepted_tokens": accepted,
            "spec_decode_per_position_acceptance_rates": accepted_by_position,
        }
    )


def test_command_uses_official_streaming_fixed_shape_semantics(tmp_path):
    args = _parse(tmp_path, "--warmups", "4", "--dataset-seed", "900")
    command = bench_serve.build_vllm_command(
        args,
        block_index=2,
        result_dir=tmp_path,
        result_filename="block.json",
    )

    assert command[:3] == ["vllm", "bench", "serve"]
    assert _option(command, "--base-url") == "http://127.0.0.1:8000"
    assert _option(command, "--served-model-name") == "served"
    assert _option(command, "--dataset-name") == "random"
    assert _option(command, "--random-input-len") == "32"
    assert _option(command, "--random-output-len") == "256"
    assert json.loads(_option(command, "--random-range-ratio")) == {
        "input": 0.0,
        "output": 0.0,
    }
    assert _option(command, "--num-warmups") == "4"
    assert _option(command, "--request-rate") == "inf"
    assert _option(command, "--burstiness") == "1.0"
    assert _option(command, "--seed") == "902"
    assert _option(command, "--temperature") == "0"
    assert _option(command, "--percentile-metrics") == "ttft,tpot,itl,e2el"
    assert "--ignore-eos" in command
    assert "--save-detailed" in command
    assert "--no-stream" not in command


def test_requested_64_observed_63_regression_is_validated_exactly(
    tmp_path, monkeypatch
):
    argv = _replace_option(_argv(tmp_path), "--input-len", "64")
    argv = _replace_option(argv, "--observed-input-len", "63")
    args = bench_serve.parse_args(argv)
    command = bench_serve.build_vllm_command(
        args, block_index=0, result_dir=tmp_path, result_filename="raw.json"
    )
    assert _option(command, "--random-input-len") == "64"

    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    assert metadata["workload"]["requested_random_input_len"] == 64
    assert metadata["workload"]["observed_input_length_contract"] == {
        "mode": "exact",
        "value": 63,
    }
    assert metadata["workload"]["accepted_input_length_bounds"] == [63, 63]

    result = _valid_result(args)
    assert result["input_lens"] == [63] * args.num_prompts
    assert result["total_input_tokens"] == 63 * args.num_prompts
    bench_serve.validate_result(result, args)


def test_dataset_seed_and_server_sampling_are_distinct_metadata(tmp_path, monkeypatch):
    args = _parse(tmp_path, "--dataset-seed", "900")
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    workload = metadata["workload"]
    assert workload["dataset_base_seed"] == 900
    assert workload["dataset_block_seeds"] == [900, 901, 902]
    assert workload["sampling"] == {
        "strategy": "greedy",
        "temperature": 0.0,
        "sampling_seed": None,
    }


def test_parser_rejects_ranges_that_break_a_comparable_workload(tmp_path):
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--blocks", "2")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--warmups", "3")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--percentiles", "50,100")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--percentiles", "50,90,99")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--request-rate", "0")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--max-concurrency", "9")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--backend", "vllm")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--server-env", "MISSING_EQUALS")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--server-env", "MODE=a", "--server-env", "MODE=b")
    for prefix in ("", "A_", "___", "NO-DASH_"):
        with pytest.raises(SystemExit):
            _parse(tmp_path, "--runner-env-prefix", prefix)
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--runner-env-prefix", "PRISMAQUANT_")
    for ratio in ("-0.01", "1", "nan", "not-a-number"):
        with pytest.raises(SystemExit):
            _parse(tmp_path, "--input-range-ratio", ratio)


def test_observed_input_contract_is_required_and_unambiguous(tmp_path):
    with pytest.raises(SystemExit):
        bench_serve.parse_args(_without_option(_argv(tmp_path), "--observed-input-len"))
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--observed-input-len-min", "31")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--expected-input-lens-sha256", "a" * 64)
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--input-range-ratio",
            "0.25",
            "--observed-input-len-min",
            "24",
            "--observed-input-len-max",
            "40",
        )

    range_argv = _without_option(_argv(tmp_path), "--observed-input-len")
    with pytest.raises(SystemExit):
        bench_serve.parse_args(
            [
                *range_argv,
                "--input-range-ratio",
                "0.25",
                "--observed-input-len-min",
                "24",
                "--observed-input-len-max",
                "40",
            ]
        )
    with pytest.raises(SystemExit):
        bench_serve.parse_args(
            [
                *range_argv,
                "--input-range-ratio",
                "0.25",
                "--observed-input-len-min",
                "24",
            ]
        )
    with pytest.raises(SystemExit):
        bench_serve.parse_args(
            [
                *range_argv,
                "--input-range-ratio",
                "0.25",
                "--observed-input-len-min",
                "41",
                "--observed-input-len-max",
                "40",
            ]
        )


def test_required_identity_strings_reject_whitespace(tmp_path):
    for option in ("--model-id", "--image-id", "--run-label"):
        with pytest.raises(SystemExit):
            _parse(tmp_path, option, "   ")


def test_git_commit_and_run_label_are_safe_exact_identifiers(tmp_path):
    for commit in ("deadbeef", "g" * 40, "a" * 39, "a" * 41, "a" * 65):
        with pytest.raises(SystemExit):
            _parse(tmp_path, "--git-commit", commit)
    args = _parse(tmp_path, "--git-commit", "A" * 64)
    assert args.git_commit == "a" * 64

    for label in ("has space", "../escape", "slash/value", "line\nbreak", "x" * 129):
        with pytest.raises(SystemExit):
            _parse(tmp_path, "--run-label", label)


def test_execution_identity_and_whole_artifact_bytes_are_required(tmp_path):
    required = (
        "--artifact-bytes",
        "--artifact-inventory",
        "--artifact-inventory-sha256",
        "--byte-budget",
        "--format-rung",
        "--serialized-layout",
        "--scale-coding",
        "--quant-contract",
        "--kernel-backend",
        "--tensor-parallel-size",
        "--fallback-state",
        "--client-runtime-id",
        "--server-runtime-id",
        "--gpu-id",
        "--driver-version",
        "--accelerator-runtime",
        "--prefix-caching",
        "--server-evidence",
        "--server-evidence-sha256",
        "--execution-manifest",
        "--execution-manifest-sha256",
        "--speculative-mode",
        "--model-id",
        "--image-id",
        "--run-label",
    )
    for option in required:
        with pytest.raises(SystemExit):
            bench_serve.parse_args(_without_option(_argv(tmp_path), option))


def test_prefix_caching_and_server_evidence_fail_closed(tmp_path):
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--prefix-caching", "on")
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--prefix-caching",
            "on",
            "--server-arg=--enable-prefix-caching",
        )
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--server-arg=--enable-prefix-caching")

    on_argv = [
        "--server-arg=--enable-prefix-caching"
        if item == "--server-arg=--no-enable-prefix-caching"
        else item
        for item in _replace_option(_argv(tmp_path), "--prefix-caching", "on")
    ]
    assert bench_serve.parse_args(on_argv).prefix_caching == "on"

    argv = _argv(tmp_path)
    evidence_path = Path(_option(argv, "--server-evidence"))
    evidence_path.write_text("mutated after digest declaration")
    with pytest.raises(SystemExit):
        bench_serve.parse_args(argv)

    missing_digest = _without_option(_argv(tmp_path), "--server-evidence-sha256")
    with pytest.raises(SystemExit):
        bench_serve.parse_args(missing_digest)


def test_digest_bound_inputs_are_rechecked_after_argument_parsing(
    tmp_path, monkeypatch
):
    args = _parse(tmp_path)
    Path(args.server_evidence[0]).write_text("changed after parse\n")
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    with pytest.raises(bench_serve.BenchmarkError, match="server evidence.*mismatch"):
        bench_serve.collect_metadata(args, {})


def test_cuda_version_remains_a_deprecated_accelerator_runtime_alias(tmp_path):
    argv = _argv(tmp_path)
    index = argv.index("--accelerator-runtime")
    argv[index] = "--cuda-version"
    args = bench_serve.parse_args(argv)
    assert args.accelerator_runtime == "CUDA 13.0"


def test_whole_artifact_must_fit_exact_byte_budget(tmp_path):
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--byte-budget",
            "799",
        )
    args = _parse(
        tmp_path,
        "--byte-budget",
        "800",
    )
    assert args.artifact_bytes == args.byte_budget == 800


def test_artifact_inventory_digest_sum_and_paths_fail_closed(tmp_path):
    argv = _argv(tmp_path)
    inventory_path = Path(_option(argv, "--artifact-inventory"))

    inventory_path.write_text("{}")
    with pytest.raises(SystemExit):
        bench_serve.parse_args(argv)

    bad_sum = {
        "schema": bench_serve.ARTIFACT_INVENTORY_SCHEMA,
        "total_bytes": 800,
        "files": [{"path": "model", "bytes": 799, "sha256": "a" * 64}],
    }
    digest = _write_json_digest(inventory_path, bad_sum)
    bad_sum_argv = _replace_option(argv, "--artifact-inventory-sha256", digest)
    with pytest.raises(SystemExit):
        bench_serve.parse_args(bad_sum_argv)

    unsafe = {
        "schema": bench_serve.ARTIFACT_INVENTORY_SCHEMA,
        "total_bytes": 800,
        "files": [{"path": "../model", "bytes": 800, "sha256": "a" * 64}],
    }
    digest = _write_json_digest(inventory_path, unsafe)
    unsafe_argv = _replace_option(argv, "--artifact-inventory-sha256", digest)
    with pytest.raises(SystemExit):
        bench_serve.parse_args(unsafe_argv)

    unsafe["files"][0]["path"] = "weights//model"
    digest = _write_json_digest(inventory_path, unsafe)
    noncanonical_argv = _replace_option(argv, "--artifact-inventory-sha256", digest)
    with pytest.raises(SystemExit):
        bench_serve.parse_args(noncanonical_argv)


def test_prismaquant_quant_config_inventory_is_consumed_directly(tmp_path):
    argv = _argv(tmp_path)
    inventory_path = Path(_option(argv, "--artifact-inventory"))
    execution_path = Path(_option(argv, "--execution-manifest"))
    quant_config = {
        "quant_method": "gridbook",
        "provenance": {
            "artifact_inventory": {
                "schema": bench_serve.PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA,
                "scope": "all_regular_files_recursive",
                "file_bytes": {
                    "config.json": 100,
                    "model.safetensors": 700,
                },
                "export_directory_bytes": 800,
            }
        },
    }
    inventory_digest = _write_json_digest(inventory_path, quant_config)
    argv = _replace_option(argv, "--artifact-inventory-sha256", inventory_digest)

    execution = json.loads(execution_path.read_text())
    execution["artifact_inventory_sha256"] = inventory_digest
    execution_digest = _write_json_digest(execution_path, execution)
    argv = _replace_option(argv, "--execution-manifest-sha256", execution_digest)
    args = bench_serve.parse_args(argv)
    assert args.artifact_inventory_summary["schema"] == (
        bench_serve.PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA
    )
    assert args.artifact_inventory_summary["source"] == "quant-config-provenance"
    assert args.artifact_inventory_summary["computed_total_bytes"] == 800
    assert args.artifact_inventory_summary["files"] == [
        {"path": "config.json", "bytes": 100},
        {"path": "model.safetensors", "bytes": 700},
    ]

    quant_config["provenance"]["artifact_inventory"]["export_directory_bytes"] = 801
    bad_digest = _write_json_digest(inventory_path, quant_config)
    bad_argv = _replace_option(argv, "--artifact-inventory-sha256", bad_digest)
    with pytest.raises(SystemExit):
        bench_serve.parse_args(bad_argv)


def test_payload_scope_is_explicit_and_never_substitutes_for_whole_bytes(
    tmp_path, monkeypatch
):
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--payload-bytes", "700")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--payload-scope", "model shards only")
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--payload-bytes",
            "801",
            "--payload-scope",
            "model shards only",
        )
    args = _parse(
        tmp_path,
        "--payload-bytes",
        "700",
        "--payload-scope",
        "model shards plus codebook sidecar",
    )
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    assert metadata["artifacts"]["payload"] == {
        "bytes": 700,
        "scope": "model shards plus codebook sidecar",
        "is_budget_gate": False,
    }
    assert metadata["artifacts"]["whole_served_artifact_bytes"] == 800


def test_execution_manifest_binds_inventory_and_mixed_assignments(tmp_path):
    argv = _argv(tmp_path)
    execution_path = Path(_option(argv, "--execution-manifest"))
    inventory_digest = _option(argv, "--artifact-inventory-sha256")
    manifest = {
        "schema": bench_serve.EXECUTION_MANIFEST_SCHEMA,
        "coverage": "all_serving_units",
        "artifact_inventory_sha256": inventory_digest,
        "tensor_parallel_size": 1,
        "assignments": [
            {
                "unit": "model.layers.0",
                "format_rung": "FP8_CB_K36",
                "serialized_layout": "product-codebook-indices-v1",
                "scale_coding": "e4m3-per-block",
                "quant_contract": "W8A8",
                "kernel_backend": "gridbook-cuda-cb-gemv-v2",
                "fallback_state": "none-observed",
            },
            {
                "unit": "model.layers.1",
                "format_rung": "NVFP4",
                "serialized_layout": "compressed-tensors",
                "scale_coding": "ue4m3",
                "quant_contract": "W4A4",
                "kernel_backend": "vllm-cutlass",
                "fallback_state": "none-observed",
            },
        ],
    }
    manifest_digest = _write_json_digest(execution_path, manifest)
    mixed_argv = _replace_option(argv, "--execution-manifest-sha256", manifest_digest)
    for option in (
        "--format-rung",
        "--serialized-layout",
        "--scale-coding",
        "--quant-contract",
        "--kernel-backend",
    ):
        mixed_argv = _replace_option(mixed_argv, option, "mixed")
    args = bench_serve.parse_args(mixed_argv)
    assert args.execution_manifest_summary["assignment_count"] == 2
    assert args.execution_manifest_summary["sha256"] == manifest_digest

    not_declared_mixed = _replace_option(mixed_argv, "--format-rung", "FP8_CB_K36")
    with pytest.raises(SystemExit):
        bench_serve.parse_args(not_declared_mixed)

    incomplete_manifest = dict(manifest)
    del incomplete_manifest["coverage"]
    incomplete_digest = _write_json_digest(execution_path, incomplete_manifest)
    incomplete_argv = _replace_option(
        mixed_argv, "--execution-manifest-sha256", incomplete_digest
    )
    with pytest.raises(SystemExit):
        bench_serve.parse_args(incomplete_argv)

    manifest["artifact_inventory_sha256"] = "0" * 64
    wrong_binding_digest = _write_json_digest(execution_path, manifest)
    wrong_binding = _replace_option(
        mixed_argv, "--execution-manifest-sha256", wrong_binding_digest
    )
    with pytest.raises(SystemExit):
        bench_serve.parse_args(wrong_binding)


def test_execution_manifest_requires_concrete_nonoverlapping_assignments(tmp_path):
    argv = _argv(tmp_path)
    execution_path = Path(_option(argv, "--execution-manifest"))
    base = json.loads(execution_path.read_text())

    invalid_manifests = []
    bool_tp = json.loads(json.dumps(base))
    bool_tp["tensor_parallel_size"] = True
    invalid_manifests.append(bool_tp)

    wildcard = json.loads(json.dumps(base))
    wildcard["assignments"][0]["unit"] = "model.layers.*"
    invalid_manifests.append(wildcard)

    overlap = json.loads(json.dumps(base))
    overlap["assignments"].append(
        {
            **overlap["assignments"][0],
            "unit": overlap["assignments"][0]["unit"] + ".weight",
        }
    )
    invalid_manifests.append(overlap)

    for field in bench_serve._EXECUTION_ASSIGNMENT_FIELDS:
        literal_mixed = json.loads(json.dumps(base))
        literal_mixed["assignments"][0][field] = "mixed"
        invalid_manifests.append(literal_mixed)

    for manifest in invalid_manifests:
        digest = _write_json_digest(execution_path, manifest)
        candidate = _replace_option(argv, "--execution-manifest-sha256", digest)
        with pytest.raises(SystemExit):
            bench_serve.parse_args(candidate)


def test_speculative_mode_requires_structured_configuration(tmp_path):
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--speculative-mode", "on")
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--speculative-mode",
            "on",
            "--speculative-config",
            '{"method":"mtp"}',
        )
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":2}',
        )
    for config in (
        '{"num_speculative_tokens":2,"num_speculative_tokens":3}',
        '{"num_speculative_tokens":2,"threshold":NaN}',
    ):
        with pytest.raises(SystemExit):
            _parse(
                tmp_path,
                "--speculative-mode",
                "on",
                "--speculative-config",
                config,
            )

    config = '{"method":"mtp","num_speculative_tokens":2}'
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--speculative-mode",
            "on",
            "--speculative-config",
            config,
        )
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            "--speculative-mode",
            "on",
            "--speculative-config",
            config,
            '--server-arg=--speculative-config={"method":"draft-model","num_speculative_tokens":2}',
        )
    args = _parse(
        tmp_path,
        "--speculative-mode",
        "on",
        "--speculative-config",
        config,
        f"--server-arg=--speculative-config={config}",
    )
    assert args.speculative_config["method"] == "mtp"
    with pytest.raises(SystemExit):
        _parse(
            tmp_path,
            '--server-arg=--speculative-config={"method":"mtp","num_speculative_tokens":2}',
        )


def test_dispatch_environment_is_sorted_explicit_and_secret_safe():
    env = {
        "PRISMAQUANT_CB_DECODE": "cuda",
        "PRISMAQUANT_API_TOKEN": "do-not-save",
        "PRISMAQUANT_AUTH_COOKIE": "also-secret",
        "CUDA_VISIBLE_DEVICES": "0",
        "UNRELATED": "ignored",
    }
    captured = bench_serve.capture_dispatch_environment(
        env, ["CUDA_VISIBLE_DEVICES", "MISSING"], ["PRISMAQUANT_"]
    )
    assert list(captured) == [
        "CUDA_VISIBLE_DEVICES",
        "MISSING",
        "PRISMAQUANT_API_TOKEN",
        "PRISMAQUANT_AUTH_COOKIE",
        "PRISMAQUANT_CB_DECODE",
    ]
    assert captured["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["MISSING"] is None
    assert captured["PRISMAQUANT_API_TOKEN"] == "<redacted>"
    assert captured["PRISMAQUANT_AUTH_COOKIE"] == "<redacted>"
    assert captured["PRISMAQUANT_CB_DECODE"] == "cuda"


def test_metadata_captures_supplied_artifact_and_server_identity(tmp_path, monkeypatch):
    args = _parse(
        tmp_path,
        "--server-arg=--enforce-eager",
        "--runner-env",
        "CUDA_VISIBLE_DEVICES",
        "--server-env",
        "PRISMAQUANT_CB_DECODE=v2",
        "--server-env",
        "HF_TOKEN=never-record",
    )
    monkeypatch.setattr(bench_serve, "_package_version", lambda: "0.3.0")
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(
        args,
        {"PRISMAQUANT_CB_DECODE": "cuda", "CUDA_VISIBLE_DEVICES": "0"},
    )

    assert metadata["git"]["commit"] == GRIDBOOK_COMMIT
    assert metadata["git"]["source"] in {"argument", "argument+checkout"}
    assert metadata["git"]["release_eligible"] is (
        metadata["git"]["source"] == "argument+checkout"
    )
    assert metadata["git"]["dirty"] in {None, False}
    assert metadata["software"]["gridbook_version"] == "0.3.0"
    assert metadata["software"]["runner_vllm_cli_probe"] == CLIENT_RUNTIME_ID
    assert metadata["artifacts"]["image_id"].endswith("sha256:abc")
    assert metadata["artifacts"]["model_id"] == "org/artifact@0123456"
    assert metadata["artifacts"]["whole_served_artifact_bytes"] == 800
    assert metadata["artifacts"]["byte_budget_bytes"] == 900
    assert metadata["artifacts"]["budget_headroom_bytes"] == 100
    assert metadata["artifacts"]["within_byte_budget"] is True
    identity = metadata["execution_identity"]
    assert identity["format_rung"] == "FP8_CB_K36"
    assert identity["serialization"] == {
        "layout": "product-codebook-indices-v1",
        "scale_coding": "e4m3-per-block",
    }
    assert identity["quant_contract"] == "W8A8"
    assert identity["kernel_backend"] == "gridbook-cuda-cb-gemv-v2"
    assert identity["tensor_parallel_size"] == 1
    assert identity["fallback_state"] == "none-observed"
    assert identity["manifest"]["assignment_count"] == 1
    assert (
        identity["manifest"]["artifact_inventory_sha256"]
        == metadata["artifacts"]["inventory"]["sha256"]
    )
    assert identity["client_runtime_id"] == CLIENT_RUNTIME_ID
    assert identity["server_runtime_id"] == "vllm@g54b16d8a9+gridbook-0.3.0"
    assert identity["hardware"] == {
        "gpu_id": "NVIDIA-GB10:GPU-1234",
        "driver_version": "580.00",
        "accelerator_runtime": "CUDA 13.0",
    }
    assert metadata["server"]["recorded_args"] == [
        "--no-enable-prefix-caching",
        "--enforce-eager",
    ]
    assert metadata["server"]["prefix_caching"] == "off"
    assert metadata["server"]["evidence"]["attachments"][0]["sha256"] == _option(
        _argv(tmp_path), "--server-evidence-sha256"
    )
    assert (
        "verifies bytes, not semantic backend claims"
        in metadata["server"]["evidence"]["scope"]
    )
    assert metadata["dispatch"]["runner_environment"] == {
        "source": "benchmark process",
        "values": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PRISMAQUANT_CB_DECODE": "cuda",
        },
    }
    assert metadata["dispatch"]["server_environment"] == {
        "source": "explicit --server-env arguments",
        "values": {
            "HF_TOKEN": "<redacted>",
            "PRISMAQUANT_CB_DECODE": "v2",
        },
    }
    assert metadata["dispatch"]["runner_environment"]["values"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "PRISMAQUANT_CB_DECODE": "cuda",
    }
    assert metadata["workload"]["streaming"] is True
    assert metadata["workload"]["dataset_block_seeds"] == [1234, 1235, 1236]
    assert metadata["workload"]["requested_random_input_len"] == 32
    assert metadata["workload"]["observed_input_length_contract"] == {
        "mode": "exact",
        "value": 32,
    }


def test_git_state_only_autodetects_the_exact_clean_source_checkout(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "checkout"
    source = checkout / "gridbook" / "bench_serve.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source\n")
    (checkout / "pyproject.toml").write_text("[project]\nname='gridbook'\n")
    monkeypatch.setattr(bench_serve, "__file__", str(source))

    state = {"top_level": str(checkout), "dirty": ""}

    def fake_git(command, **kwargs):
        if command[-1] == "--show-toplevel":
            output = state["top_level"]
        elif command[-1] == "HEAD":
            output = "a" * 40
        elif command[1:3] == ["status", "--porcelain"]:
            output = state["dirty"]
        else:  # pragma: no cover - protects the test contract itself
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(bench_serve.subprocess, "run", fake_git)
    assert bench_serve._git_state(None) == {
        "commit": "a" * 40,
        "dirty": False,
        "source": "checkout",
        "release_eligible": True,
    }
    with pytest.raises(bench_serve.BenchmarkError, match="disagrees"):
        bench_serve._git_state("b" * 40)

    state["dirty"] = " M gridbook/bench_serve.py\n"
    with pytest.raises(bench_serve.BenchmarkError, match="checkout is dirty"):
        bench_serve._git_state(None)
    dirty = bench_serve._git_state(None, allow_dirty=True)
    assert dirty["dirty"] is True
    assert dirty["release_eligible"] is False

    state["top_level"] = str(tmp_path)
    mismatch = bench_serve._git_state(None)
    assert mismatch["commit"] is None
    assert mismatch["source"] == "unavailable-root-mismatch"

    explicit = bench_serve._git_state(GRIDBOOK_COMMIT, allow_dirty=True)
    assert explicit["commit"] == GRIDBOOK_COMMIT
    assert explicit["release_eligible"] is False
    with pytest.raises(bench_serve.BenchmarkError, match="not exact"):
        bench_serve._git_state("deadbeef")


def test_git_state_requires_a_real_pyproject_source_root(tmp_path, monkeypatch):
    source = tmp_path / "site-packages" / "gridbook" / "bench_serve.py"
    source.parent.mkdir(parents=True)
    source.write_text("# installed source without project root\n")
    monkeypatch.setattr(bench_serve, "__file__", str(source))
    state = bench_serve._git_state(None)
    assert state["commit"] is None
    assert state["source"] == "unavailable-non-source-install"

    explicit = bench_serve._git_state("d" * 40)
    assert explicit["commit"] == "d" * 40
    assert explicit["source"] == "argument"
    assert explicit["dirty"] is None
    assert explicit["release_eligible"] is False


def test_generated_report_is_the_only_checkout_dirty_check_exclusion(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "checkout"
    source = checkout / "gridbook" / "bench_serve.py"
    source.parent.mkdir(parents=True)
    source.write_text("# clean source\n")
    (checkout / "pyproject.toml").write_text("[project]\nname='gridbook'\n")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Gridbook test"], cwd=checkout, check=True
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(bench_serve, "__file__", str(source))

    report = checkout / "results" / "report.json"
    report.parent.mkdir()
    report.write_text("generated evidence\n")
    clean = bench_serve._git_state(commit, generated_paths=(report,))
    assert clean["dirty"] is False
    assert clean["release_eligible"] is True

    source.write_text("# changed source\n")
    with pytest.raises(bench_serve.BenchmarkError, match="checkout is dirty"):
        bench_serve._git_state(commit, generated_paths=(report,))


def test_urls_commands_and_server_args_never_persist_credentials(tmp_path, monkeypatch):
    secret_url = (
        "https://alice:password@example.test:8443/v1?api_key=topsecret&mode=fast"
    )
    args = _parse(
        tmp_path,
        "--base-url",
        secret_url,
        "--server-arg=--hf-token supersecret",
        "--server-arg=Authorization: Bearer abc123",
        "--server-arg=Authorization: Token token-secret",
        "--server-arg=--hf-token",
        "--server-arg=split-secret",
        "--server-arg=MONKEY=visible",
    )
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    command = bench_serve.build_vllm_command(
        args, block_index=0, result_dir=tmp_path, result_filename="raw.json"
    )
    redacted_command = bench_serve.redact_command(command)
    serialized = json.dumps({"metadata": metadata, "command": redacted_command})

    for secret in (
        "alice",
        "password",
        "topsecret",
        "supersecret",
        "abc123",
        "token-secret",
        "split-secret",
    ):
        assert secret not in serialized
    assert "mode=fast" in metadata["server"]["base_url"]
    assert "MONKEY=visible" in serialized
    assert _option(redacted_command, "--tokenizer") == "org/tokenizer"


def test_redaction_covers_nested_json_headers_cookies_and_url_fragments():
    command = [
        "client",
        "--extra-body",
        json.dumps(
            {
                "public": "visible",
                "nested": {
                    "authorization": "Bearer json-secret",
                    "items": [
                        {"api_key": "array-secret"},
                        "https://example.test/#access_token=fragment-secret&mode=ok",
                    ],
                },
            }
        ),
        "--header",
        "X-Api-Key: header-secret",
        "-H",
        "Cookie: session=cookie-secret; Path=/",
        '--server-config={"refresh_token":"embedded-secret","safe":1}',
        "https://example.test/path#token=second-fragment&view=public",
    ]
    serialized = json.dumps(bench_serve.redact_command(command))
    for secret in (
        "json-secret",
        "array-secret",
        "fragment-secret",
        "header-secret",
        "cookie-secret",
        "embedded-secret",
        "second-fragment",
    ):
        assert secret not in serialized
    assert "visible" in serialized
    assert "mode=ok" not in serialized
    assert "view=public" not in serialized
    assert serialized.count("<redacted>") >= 2


def test_redaction_treats_every_nonempty_url_fragment_as_opaque():
    redacted = bench_serve._redact_url(
        "https://example.test/path?mode=fast#opaque-unkeyed-credential"
    )
    assert "mode=fast" in redacted
    assert "opaque-unkeyed-credential" not in redacted
    assert redacted.endswith("#<redacted>")


@pytest.mark.parametrize(
    "header",
    (
        "authorization: Basic lower-secret",
        "Proxy-Authorization: Bearer proxy-secret",
        "Set-Cookie: SID=set-cookie-secret; HttpOnly",
        "API-Key: api-header-secret",
    ),
)
def test_inline_sensitive_header_variations_are_redacted(header):
    redacted = bench_serve._redact_text(header)
    assert "secret" not in redacted
    assert "<redacted>" in redacted


def test_secret_key_detection_does_not_confuse_tokenizer_or_monkey():
    assert bench_serve._is_sensitive_key("HF_TOKEN")
    assert bench_serve._is_sensitive_key("authorization")
    assert bench_serve._is_sensitive_key("AWS_SECRET_ACCESS_KEY")
    assert not bench_serve._is_sensitive_key("tokenizer")
    assert not bench_serve._is_sensitive_key("MONKEY")


def test_validate_result_requires_true_streaming_metrics_and_exact_lengths(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)
    bench_serve.validate_result(result, args)

    missing_itl = dict(result)
    del missing_itl["mean_itl_ms"]
    with pytest.raises(bench_serve.BenchmarkError, match="required metrics"):
        bench_serve.validate_result(missing_itl, args)

    short_decode = dict(result, total_output_tokens=result["total_output_tokens"] - 1)
    with pytest.raises(bench_serve.BenchmarkError, match="not completed exactly"):
        bench_serve.validate_result(short_decode, args)

    uneven = dict(result, output_lens=[255, 257] + [256] * 6)
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected output_lens"):
        bench_serve.validate_result(uneven, args)

    uneven_input = dict(result, input_lens=[31, 33] + [32] * 6)
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected input_lens"):
        bench_serve.validate_result(uneven_input, args)


def test_pinned_vllm_saved_invocation_envelope_is_reconciled_exactly(tmp_path):
    args = _parse(tmp_path, "--request-rate", "2.5")
    result = _valid_result(args)
    assert result["date"] == "20260731-120000"
    assert result["endpoint_type"] == result["backend"] == "openai"
    assert result["model_id"] == args.model
    assert result["tokenizer_id"] == (args.tokenizer or args.model)
    assert result["request_rate"] == 2.5
    assert isinstance(result["request_rate"], float)
    bench_serve.validate_result(result, args)

    mutations = {
        "backend": "other",
        "endpoint_type": args.endpoint,
        "label": "other-label",
        "model_id": args.model_id,
        "tokenizer_id": "other/tokenizer",
        "num_prompts": True,
        "request_rate": "2.5",
        "burstiness": 2.0,
        "max_concurrency": 1.0,
    }
    for field, bad_value in mutations.items():
        with pytest.raises(bench_serve.BenchmarkError, match=field):
            bench_serve.validate_result({**result, field: bad_value}, args)

    for field in mutations:
        missing = dict(result)
        del missing[field]
        with pytest.raises(bench_serve.BenchmarkError, match=field):
            bench_serve.validate_result(missing, args)

    ancillary_mutations = {
        "date": "2026-07-31",
        "request_goodput": 1.0,
        "generated_texts": result["generated_texts"][:-1],
        "start_times": [-1.0, *result["start_times"][1:]],
        "max_output_tokens_per_s": -1.0,
        "max_concurrent_requests": True,
        "rtfx": 0.1,
    }
    for field, bad_value in ancillary_mutations.items():
        with pytest.raises(bench_serve.BenchmarkError, match=field):
            bench_serve.validate_result({**result, field: bad_value}, args)

    with pytest.raises(bench_serve.BenchmarkError, match="unexpected=metadata"):
        bench_serve.validate_result({**result, "metadata": "not requested"}, args)


def test_validate_result_requires_detailed_positive_streaming_arrays(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)

    without_ttfts = dict(result)
    del without_ttfts["ttfts"]
    with pytest.raises(bench_serve.BenchmarkError, match="ttfts must cover"):
        bench_serve.validate_result(without_ttfts, args)

    zero_ttft = dict(result, ttfts=[0.0] + result["ttfts"][1:])
    with pytest.raises(bench_serve.BenchmarkError, match="TTFT"):
        bench_serve.validate_result(zero_ttft, args)

    zero_e2el = dict(result, mean_e2el_ms=0.0)
    with pytest.raises(bench_serve.BenchmarkError, match="non-positive"):
        bench_serve.validate_result(zero_e2el, args)

    missing_itl = dict(result, itls=[[]] + result["itls"][1:])
    with pytest.raises(bench_serve.BenchmarkError, match="no inter-token"):
        bench_serve.validate_result(missing_itl, args)

    zero_itl = dict(result, itls=[[0.0]] + result["itls"][1:])
    with pytest.raises(bench_serve.BenchmarkError, match="ITL"):
        bench_serve.validate_result(zero_itl, args)


def test_validate_result_reconciles_aggregate_and_detailed_timings(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)

    bad_ttft = dict(result, mean_ttft_ms=result["mean_ttft_ms"] + 1.0)
    with pytest.raises(bench_serve.BenchmarkError, match="mean_ttft_ms disagrees"):
        bench_serve.validate_result(bad_ttft, args)

    bad_itl = dict(result, median_itl_ms=result["median_itl_ms"] + 1.0)
    with pytest.raises(bench_serve.BenchmarkError, match="median_itl_ms disagrees"):
        bench_serve.validate_result(bad_itl, args)

    bad_e2el = dict(result, mean_e2el_ms=result["mean_e2el_ms"] + 10.0)
    with pytest.raises(bench_serve.BenchmarkError, match="mean_e2el_ms disagrees"):
        bench_serve.validate_result(bad_e2el, args)

    bad_tpot = dict(result, mean_tpot_ms=result["mean_tpot_ms"] + 1.0)
    with pytest.raises(bench_serve.BenchmarkError, match="mean_tpot_ms disagrees"):
        bench_serve.validate_result(bad_tpot, args)


@pytest.mark.parametrize(
    "metric",
    (
        "std_ttft_ms",
        "p95_ttft_ms",
        "std_itl_ms",
        "p99_itl_ms",
        "std_e2el_ms",
        "p90_e2el_ms",
        "std_tpot_ms",
        "p95_tpot_ms",
    ),
)
def test_every_dispersion_and_percentile_is_reconstructed(tmp_path, metric):
    args = _parse(tmp_path)
    result = _valid_result(args)
    result[metric] += 10.0
    with pytest.raises(bench_serve.BenchmarkError, match="disagrees"):
        bench_serve.validate_result(result, args)


@pytest.mark.parametrize(
    "metric",
    ("request_throughput", "output_throughput", "total_token_throughput"),
)
def test_throughput_is_reconstructed_from_duration_and_totals(tmp_path, metric):
    args = _parse(tmp_path)
    result = _valid_result(args)
    result[metric] += 0.1
    with pytest.raises(bench_serve.BenchmarkError, match="duration and exact totals"):
        bench_serve.validate_result(result, args)


@pytest.mark.parametrize("duration", (0, -1, float("inf"), "4"))
def test_duration_must_be_finite_positive_number(tmp_path, duration):
    args = _parse(tmp_path)
    result = _valid_result(args)
    result["duration"] = duration
    if duration == float("inf"):
        match = "NaN/Infinity"
    else:
        match = "duration"
    with pytest.raises(bench_serve.BenchmarkError, match=match):
        bench_serve.validate_result(result, args)


def test_numpy_linear_percentile_and_population_std_are_pinned(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)
    ttft_ms = [10.0, 11.0, 13.0, 17.0, 23.0, 31.0, 41.0, 53.0]
    result["ttfts"] = [value / 1000 for value in ttft_ms]
    aggregates = bench_serve._sample_aggregates(ttft_ms, args.percentiles.split(","))
    result["mean_ttft_ms"] = aggregates["mean"]
    result["median_ttft_ms"] = aggregates["median"]
    result["std_ttft_ms"] = aggregates["std"]
    for percentile in args.percentiles.split(","):
        result[bench_serve._percentile_result_key(percentile, "ttft")] = aggregates[
            f"p{bench_serve._percentile_label(percentile)}"
        ]

    e2el_ms = [value + 2.0 for value in ttft_ms]
    e2el = bench_serve._sample_aggregates(e2el_ms, args.percentiles.split(","))
    result["mean_e2el_ms"] = e2el["mean"]
    result["median_e2el_ms"] = e2el["median"]
    result["std_e2el_ms"] = e2el["std"]
    for percentile in args.percentiles.split(","):
        result[bench_serve._percentile_result_key(percentile, "e2el")] = e2el[
            f"p{bench_serve._percentile_label(percentile)}"
        ]
    bench_serve.validate_result(result, args)

    result["std_ttft_ms"] = __import__("statistics").stdev(ttft_ms)
    with pytest.raises(bench_serve.BenchmarkError, match="std_ttft_ms disagrees"):
        bench_serve.validate_result(result, args)


def test_percentile_labels_match_pinned_vllm_without_collisions(tmp_path):
    args = _parse(
        tmp_path,
        "--percentiles",
        "1e-7,50,90,95,95.00001,95.00002,99.9999999",
    )
    assert args.percentiles.split(",") == [
        "1e-07",
        "50",
        "90",
        "95",
        "95.00001",
        "95.00002",
        "99.9999999",
    ]
    assert bench_serve._percentile_result_key("95.00001", "ttft") != (
        bench_serve._percentile_result_key("95.00002", "ttft")
    )
    result = _valid_result(args)
    bench_serve.validate_result(result, args)
    summary = bench_serve.summarize_blocks(
        [{"raw_result": dict(result)} for _ in range(3)]
    )
    assert "p1e-07_ttft_ms" in summary["metrics"]
    assert "p99.9999999_e2el_ms" in summary["metrics"]

    unexpected = dict(result, p97_ttft_ms=result["p95_ttft_ms"])
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected=p97_ttft_ms"):
        bench_serve.validate_result(unexpected, args)


def test_itl_chunk_count_is_not_assumed_to_equal_output_tokens(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)
    assert all(len(intervals) == 1 for intervals in result["itls"])
    assert args.output_len - 1 == 255
    bench_serve.validate_result(result, args)


def test_speculative_result_contract_is_required_and_reconciled(tmp_path):
    args = _parse(
        tmp_path,
        "--speculative-mode",
        "on",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
        '--server-arg=--speculative-config={"method":"mtp","num_speculative_tokens":2}',
    )
    result = _valid_result(args)
    with pytest.raises(bench_serve.BenchmarkError, match="lacks required"):
        bench_serve.validate_result(result, args)

    _add_valid_speculative_result(result)
    bench_serve.validate_result(result, args)

    mutations = {
        "spec_decode_acceptance_rate": result["spec_decode_acceptance_rate"] + 1,
        "spec_decode_acceptance_length": result["spec_decode_acceptance_length"] + 1,
        "spec_decode_num_drafts": result["spec_decode_num_drafts"] + 1,
        "spec_decode_draft_tokens": result["spec_decode_draft_tokens"] + 1,
        "spec_decode_accepted_tokens": result["spec_decode_draft_tokens"] + 1,
        "spec_decode_per_position_acceptance_rates": [0.8],
    }
    for key, bad_value in mutations.items():
        bad = dict(result, **{key: bad_value})
        with pytest.raises(bench_serve.BenchmarkError):
            bench_serve.validate_result(bad, args)


def test_non_speculative_cell_rejects_speculative_telemetry(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)
    _add_valid_speculative_result(result)
    with pytest.raises(bench_serve.BenchmarkError, match="non-speculative"):
        bench_serve.validate_result(result, args)


def test_single_token_prefill_explicitly_allows_empty_itl_and_zero_tpot(tmp_path):
    args = _parse(tmp_path, "--output-len", "1")
    result = _valid_result(args)
    assert all(not intervals for intervals in result["itls"])
    assert result["mean_itl_ms"] == result["mean_tpot_ms"] == 0.0
    bench_serve.validate_result(result, args)

    unexpected_itl = dict(result, itls=[[0.001], *result["itls"][1:]])
    with pytest.raises(bench_serve.BenchmarkError, match="single-token"):
        bench_serve.validate_result(unexpected_itl, args)


def test_input_range_ratio_requires_the_exact_block_vector_digest(
    tmp_path, monkeypatch
):
    input_lens = [24, 26, 28, 30, 32, 34, 36, 40]
    args = bench_serve.parse_args(_range_argv(tmp_path, input_lens))
    command = bench_serve.build_vllm_command(
        args, block_index=0, result_dir=tmp_path, result_filename="raw.json"
    )
    assert json.loads(_option(command, "--random-range-ratio")) == {
        "input": 0.25,
        "output": 0.0,
    }
    assert bench_serve._observed_input_length_bounds(args) == (24, 40)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    assert metadata["workload"]["input_range_ratio"] == 0.25
    assert metadata["workload"]["vllm_range_ratio"] == {
        "input": 0.25,
        "output": 0.0,
    }
    assert metadata["workload"]["accepted_input_length_bounds"] == [24, 40]
    assert [
        item["sha256"]
        for item in metadata["workload"]["expected_input_lens_sha256_by_block"]
    ] == args.expected_input_lens_sha256

    result = _valid_result(args)
    result["input_lens"] = input_lens
    result["total_input_tokens"] = sum(input_lens)
    _reconcile_throughput(result)
    bench_serve.validate_result(result, args)

    different_but_in_bounds = dict(result, input_lens=[25, 26, 28, 30, 32, 34, 36, 39])
    different_but_in_bounds["total_input_tokens"] = sum(
        different_but_in_bounds["input_lens"]
    )
    _reconcile_throughput(different_but_in_bounds)
    with pytest.raises(bench_serve.BenchmarkError, match="input_lens SHA-256"):
        bench_serve.validate_result(different_but_in_bounds, args)

    too_long = dict(result, input_lens=[41, *input_lens[1:]])
    too_long["total_input_tokens"] = sum(too_long["input_lens"])
    _reconcile_throughput(too_long)
    with pytest.raises(bench_serve.BenchmarkError, match=r"range \[24, 40\]"):
        bench_serve.validate_result(too_long, args)

    too_short = dict(result, input_lens=[23, *input_lens[1:]])
    too_short["total_input_tokens"] = sum(too_short["input_lens"])
    _reconcile_throughput(too_short)
    with pytest.raises(bench_serve.BenchmarkError, match=r"range \[24, 40\]"):
        bench_serve.validate_result(too_short, args)

    wrong_type = dict(result, input_lens=[24.0, *input_lens[1:]])
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected input_lens"):
        bench_serve.validate_result(wrong_type, args)

    wrong_output_type = dict(result, output_lens=[256.0, *result["output_lens"][1:]])
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected output_lens"):
        bench_serve.validate_result(wrong_output_type, args)

    wrong_total = dict(result, total_input_tokens=sum(input_lens) + 1)
    with pytest.raises(bench_serve.BenchmarkError, match="do not reconcile"):
        bench_serve.validate_result(wrong_total, args)


def test_range_input_vector_digests_are_bound_in_block_order(tmp_path):
    vectors = [
        [24, 25, 26, 27, 28, 29, 30, 31],
        [25, 26, 27, 28, 29, 30, 31, 32],
        [26, 27, 28, 29, 30, 31, 32, 33],
    ]
    argv = _without_option(_argv(tmp_path), "--observed-input-len")
    argv.extend(
        [
            "--input-range-ratio",
            "0.25",
            "--observed-input-len-min",
            "24",
            "--observed-input-len-max",
            "40",
        ]
    )
    for vector in vectors:
        argv.extend(
            [
                "--expected-input-lens-sha256",
                bench_serve._canonical_input_lens_sha256(vector),
            ]
        )
    args = bench_serve.parse_args(argv)
    result = _valid_result(args)
    result["input_lens"] = vectors[1]
    result["total_input_tokens"] = sum(vectors[1])
    _reconcile_throughput(result)
    bench_serve.validate_result(result, args, block_index=1)
    with pytest.raises(bench_serve.BenchmarkError, match="block 1"):
        bench_serve.validate_result(result, args, block_index=0)


def test_validate_result_requires_throughput_percentiles_and_finite_numbers(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)

    no_throughput = dict(result)
    del no_throughput["output_throughput"]
    with pytest.raises(bench_serve.BenchmarkError, match="output_throughput"):
        bench_serve.validate_result(no_throughput, args)

    no_requested_percentile = dict(result)
    del no_requested_percentile["p95_e2el_ms"]
    with pytest.raises(bench_serve.BenchmarkError, match="p95_e2el_ms"):
        bench_serve.validate_result(no_requested_percentile, args)

    nonfinite = dict(result, mean_ttft_ms=float("nan"))
    with pytest.raises(bench_serve.BenchmarkError, match="NaN/Infinity"):
        bench_serve.validate_result(nonfinite, args)

    infinity_in_detail = dict(result, itls=[[float("inf")]] + result["itls"][1:])
    with pytest.raises(bench_serve.BenchmarkError, match="NaN/Infinity"):
        bench_serve.validate_result(infinity_in_detail, args)


def test_summary_keeps_independent_values_and_sample_dispersion(tmp_path):
    args = _parse(tmp_path)
    blocks = [
        {"raw_result": _valid_result(args, offset=0.0)},
        {"raw_result": _valid_result(args, offset=1.0)},
        {"raw_result": _valid_result(args, offset=2.0)},
    ]
    summary = bench_serve.summarize_blocks(blocks)
    ttft = summary["metrics"]["mean_ttft_ms"]

    assert summary["completed_blocks"] == 3
    assert ttft["values"] == [20.0, 21.0, 22.0]
    assert ttft["mean"] == pytest.approx(21.0)
    assert ttft["median"] == pytest.approx(21.0)
    assert ttft["block_p05"] == pytest.approx(20.1)
    assert ttft["block_p95"] == pytest.approx(21.9)
    assert ttft["sample_stdev"] == pytest.approx(1.0)
    assert summary["metrics"]["p99_itl_ms"]["values"] == [2.0, 3.0, 4.0]


def test_run_writes_a_complete_report_without_vllm_or_server(tmp_path, monkeypatch):
    args = _parse(tmp_path, "--blocks", "3", "--allow-dirty")
    calls = []

    def fake_run(command):
        calls.append(command)
        checkpoint = json.loads(args.output.read_text())
        assert checkpoint["status"] == "running"
        assert checkpoint["blocks"][-1]["status"] == "running"
        assert checkpoint["blocks"][-1]["returncode"] is None
        result_dir = Path(_option(command, "--result-dir"))
        filename = _option(command, "--result-filename")
        block_index = int(_option(command, "--seed")) - args.dataset_seed
        (result_dir / filename).write_text(
            json.dumps(_valid_result(args, offset=float(block_index)))
        )
        return 0

    monkeypatch.setattr(bench_serve, "_run_command", fake_run)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    report = bench_serve.run_benchmark(args)
    written = json.loads(args.output.read_text())

    assert len(calls) == 3
    assert report["status"] == "success"
    assert written["schema"] == bench_serve.SCHEMA
    assert written["status"] == "success"
    assert written["evidence_scope"] == "single-arm-serving-measurement"
    assert written["measurement_valid"] is True
    assert written["parity_acceptance"] is False
    assert written["release_acceptance"] is False
    assert written["release_eligible"] is False
    assert [block["dataset_seed"] for block in written["blocks"]] == [
        1234,
        1235,
        1236,
    ]
    assert all(block["status"] == "success" for block in written["blocks"])
    assert all(block["returncode"] == 0 for block in written["blocks"])
    assert all(
        block["expected_input_lens_sha256"] is None for block in written["blocks"]
    )
    assert all(block["observed_input_lens_sha256"] for block in written["blocks"])
    assert all(
        "generated_texts" not in block["raw_result"] for block in written["blocks"]
    )
    assert all("errors" not in block["raw_result"] for block in written["blocks"])
    assert written["summary"]["completed_blocks"] == 3
    assert written["metadata"]["measurement_provenance"] == {
        "digest_bound_inputs_verified_before_requests": True,
        "digest_bound_inputs_verified_after_requests": True,
        "git_state_verified_after_requests": True,
        "client_runtime_verified_after_requests": True,
    }
    assert written["finished_at"].endswith("Z")
    assert os.stat(args.output).st_mode & 0o777 == 0o600


def test_evidence_mutation_during_measurement_invalidates_the_report(
    tmp_path, monkeypatch
):
    args = _parse(tmp_path)
    calls = 0

    def fake_run(command):
        nonlocal calls
        calls += 1
        path = Path(_option(command, "--result-dir")) / _option(
            command, "--result-filename"
        )
        path.write_text(json.dumps(_valid_result(args, offset=float(calls))))
        if calls == args.blocks:
            Path(args.server_evidence[0]).write_text("changed during requests\n")
        return 0

    monkeypatch.setattr(bench_serve, "_run_command", fake_run)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    with pytest.raises(bench_serve.BenchmarkError, match="server evidence.*mismatch"):
        bench_serve.run_benchmark(args)

    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["measurement_valid"] is False
    provenance = report["metadata"]["measurement_provenance"]
    assert provenance["digest_bound_inputs_verified_before_requests"] is True
    assert provenance["digest_bound_inputs_verified_after_requests"] is False


def test_client_runtime_change_during_measurement_invalidates_the_report(
    tmp_path, monkeypatch
):
    args = _parse(tmp_path)

    def fake_run(command):
        path = Path(_option(command, "--result-dir")) / _option(
            command, "--result-filename"
        )
        path.write_text(json.dumps(_valid_result(args)))
        return 0

    runtime_probes = iter((CLIENT_RUNTIME_ID, CLIENT_RUNTIME_ID + "-changed"))
    monkeypatch.setattr(bench_serve, "_run_command", fake_run)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: next(runtime_probes)
    )
    with pytest.raises(bench_serve.BenchmarkError, match="runtime identity changed"):
        bench_serve.run_benchmark(args)

    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["measurement_valid"] is False
    provenance = report["metadata"]["measurement_provenance"]
    assert provenance["digest_bound_inputs_verified_after_requests"] is True
    assert provenance["git_state_verified_after_requests"] is True
    assert provenance["client_runtime_verified_after_requests"] is False


def test_report_omits_generated_text_and_arbitrary_server_errors():
    result = {
        "completed": 2,
        "generated_texts": ["private completion", "https://host/#raw-secret"],
        "errors": [None, "Authorization: Bearer unstructured-secret"],
        "model_url": "https://user:password@example.test/model#fragment-secret",
    }

    sanitized = bench_serve._sanitize_result_for_report(result)
    serialized = json.dumps(sanitized)

    assert "generated_texts" not in sanitized
    assert sanitized["generated_texts_omitted"]["count"] == 2
    assert "errors" not in sanitized
    assert sanitized["errors_omitted"]["count"] == 2
    assert "private completion" not in serialized
    assert "unstructured-secret" not in serialized
    assert "raw-secret" not in serialized
    assert "fragment-secret" not in serialized
    assert "password" not in serialized

    for malformed_errors in (
        "Authorization: Bearer scalar-secret",
        {"message": "mapping-secret"},
        17,
    ):
        malformed = bench_serve._sanitize_result_for_report(
            {"errors": malformed_errors}
        )
        assert "errors" not in malformed
        assert malformed["errors_omitted"]["count"] is None
        assert "secret" not in json.dumps(malformed)


def test_failed_block_leaves_a_structured_failure_report(tmp_path, monkeypatch):
    args = _parse(tmp_path)
    monkeypatch.setattr(bench_serve, "_run_command", lambda command: 7)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )

    with pytest.raises(bench_serve.BenchmarkError, match="exit code 7"):
        bench_serve.run_benchmark(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["measurement_valid"] is False
    assert report["parity_acceptance"] is False
    assert report["release_acceptance"] is False
    assert report["release_eligible"] is False
    assert len(report["blocks"]) == 1
    block = report["blocks"][0]
    assert block["status"] == "failed"
    assert block["returncode"] == 7
    assert block["raw_result"] is None
    assert block["validation_error"] is None
    assert block["command"][:3] == ["vllm", "bench", "serve"]
    assert "exit code 7" in report["error"]


def test_validation_failure_retains_raw_result_and_error(tmp_path, monkeypatch):
    args = _parse(tmp_path)

    def fake_run(command):
        result = _valid_result(args)
        result["total_output_tokens"] -= 1
        path = Path(_option(command, "--result-dir")) / _option(
            command, "--result-filename"
        )
        path.write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(bench_serve, "_run_command", fake_run)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    with pytest.raises(bench_serve.BenchmarkError, match="not completed exactly"):
        bench_serve.run_benchmark(args)

    report = json.loads(args.output.read_text())
    block = report["blocks"][0]
    assert block["returncode"] == 0
    assert block["raw_result"]["total_output_tokens"] > 0
    assert "not completed exactly" in block["validation_error"]


def test_metadata_failure_is_captured_after_output_reservation(tmp_path, monkeypatch):
    args = _parse(tmp_path)

    def fail_metadata(_args):
        raise RuntimeError("metadata probe exploded")

    monkeypatch.setattr(bench_serve, "collect_metadata", fail_metadata)
    with pytest.raises(RuntimeError, match="metadata probe exploded"):
        bench_serve.run_benchmark(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["metadata"] is None
    assert report["blocks"] == []
    assert "metadata probe exploded" in report["error"]


def test_individual_metadata_probe_failure_is_recorded_not_raised(
    tmp_path, monkeypatch
):
    args = _parse(tmp_path)

    def fail_version():
        raise RuntimeError("package metadata unavailable")

    monkeypatch.setattr(bench_serve, "_package_version", fail_version)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    assert metadata["software"]["gridbook_version"] is None
    assert (
        "package metadata unavailable"
        in metadata["collection_errors"]["gridbook_version"]
    )


def test_client_runtime_probe_must_succeed_and_match_exactly(tmp_path, monkeypatch):
    args = _parse(tmp_path)
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: None)
    with pytest.raises(bench_serve.BenchmarkError, match="version probe failed"):
        bench_serve.collect_metadata(args, {})

    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID + "-other"
    )
    with pytest.raises(bench_serve.BenchmarkError, match="disagrees"):
        bench_serve.collect_metadata(args, {})


def test_allow_dirty_marks_report_provenance_release_ineligible(tmp_path, monkeypatch):
    args = _parse(tmp_path, "--allow-dirty")
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    metadata = bench_serve.collect_metadata(args, {})
    assert metadata["git"]["release_eligible"] is False


def test_nonfinite_raw_result_is_rejected_at_strict_json_boundary(
    tmp_path, monkeypatch
):
    args = _parse(tmp_path)

    def fake_run(command):
        result = _valid_result(args)
        result["mean_ttft_ms"] = float("nan")
        path = Path(_option(command, "--result-dir")) / _option(
            command, "--result-filename"
        )
        path.write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(bench_serve, "_run_command", fake_run)
    monkeypatch.setattr(
        bench_serve, "_vllm_version", lambda executable: CLIENT_RUNTIME_ID
    )
    with pytest.raises(bench_serve.BenchmarkError, match="not strict JSON"):
        bench_serve.run_benchmark(args)
    report = json.loads(args.output.read_text())
    block = report["blocks"][0]
    assert block["raw_result"] is None
    assert "not strict JSON" in block["validation_error"]


@pytest.mark.parametrize(
    "raw",
    (
        '{"duration":1,"duration":2}',
        '{"duration":NaN}',
        '{"duration":Infinity}',
        '{"duration":-Infinity}',
    ),
)
def test_non_strict_result_json_is_rejected_at_load_boundary(tmp_path, raw):
    result_path = tmp_path / "invalid.json"
    result_path.write_text(raw)
    with pytest.raises(bench_serve.BenchmarkError, match="not strict JSON"):
        bench_serve._load_result(result_path)


def test_result_loader_rejects_append_style_singleton_lists(tmp_path):
    result_path = tmp_path / "unexpected-list.json"
    result_path.write_text('[{"duration":1}]')
    with pytest.raises(bench_serve.BenchmarkError, match="one object"):
        bench_serve._load_result(result_path)


def test_existing_report_is_never_replaced_without_opt_in(tmp_path, monkeypatch):
    args = _parse(tmp_path)
    args.output.write_text("keep me")
    monkeypatch.setattr(
        bench_serve,
        "_run_command",
        lambda command: pytest.fail("benchmark must not start"),
    )
    with pytest.raises(bench_serve.BenchmarkError, match="already exists"):
        bench_serve.run_benchmark(args)
    assert args.output.read_text() == "keep me"


def test_overwrite_fails_closed_on_symlink_output(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_text("keep target")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    args = _parse(tmp_path, "--overwrite")
    monkeypatch.setattr(
        bench_serve,
        "_run_command",
        lambda command: pytest.fail("benchmark must not start"),
    )
    with pytest.raises(bench_serve.BenchmarkError, match="symlink"):
        bench_serve.run_benchmark(args)
    assert link.is_symlink()
    assert target.read_text() == "keep target"


def test_output_reservation_has_exactly_one_concurrent_winner(tmp_path):
    output = tmp_path / "shared.json"

    def reserve(_index):
        try:
            bench_serve._reserve_output(output, overwrite=False)
            return "won"
        except bench_serve.BenchmarkError:
            return "lost"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(16)))
    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 15
    assert json.loads(output.read_text())["status"] == "reserved"
    assert os.stat(output).st_mode & 0o777 == 0o600


def test_atomic_json_writer_refuses_nonstandard_nan(tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        bench_serve._atomic_write_json(output, {"bad": float("nan")})
