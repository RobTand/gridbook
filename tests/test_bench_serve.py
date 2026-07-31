"""CPU-only tests for the reproducible online parity harness.

The real client is ``vllm bench serve``.  These tests pin command construction,
fixed-shape validation, provenance capture, and the report schema without a
vLLM install, a GPU, or a listening server.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gridbook import bench_serve


def _parse(tmp_path, *extra):
    argv = [
        "--base-url",
        "http://127.0.0.1:8000/",
        "--model",
        "org/tokenizer",
        "--served-model-name",
        "served",
        "--model-id",
        "org/artifact@0123456",
        "--image-id",
        "registry/vllm@sha256:abc",
        "--git-commit",
        "deadbeef",
        "--run-label",
        "gridbook-0.6b-decode",
        "--input-len",
        "32",
        "--output-len",
        "256",
        "--num-prompts",
        "8",
        "--max-concurrency",
        "1",
        "--output",
        str(tmp_path / "report.json"),
        *extra,
    ]
    return bench_serve.parse_args(argv)


def _option(command, name):
    return command[command.index(name) + 1]


def _valid_result(args, *, offset=0.0):
    multi_token = args.output_len > 1
    result = {
        "completed": args.num_prompts,
        "failed": 0,
        "total_input_tokens": args.num_prompts * args.input_len,
        "total_output_tokens": args.num_prompts * args.output_len,
        "input_lens": [args.input_len] * args.num_prompts,
        "output_lens": [args.output_len] * args.num_prompts,
        "errors": [""] * args.num_prompts,
        "ttfts": [0.020 + offset / 1000] * args.num_prompts,
        "itls": [[0.002 + offset / 1000] if multi_token else []
                 for _ in range(args.num_prompts)],
        "request_throughput": 2.0 + offset,
        "output_throughput": 500.0 + offset,
        "total_token_throughput": 600.0 + offset,
        "mean_ttft_ms": 20.0 + offset,
        "median_ttft_ms": 19.0 + offset,
        "std_ttft_ms": 1.0,
        "p99_ttft_ms": 22.0 + offset,
        "mean_tpot_ms": 2.0 + offset if multi_token else 0.0,
        "median_tpot_ms": 1.9 + offset if multi_token else 0.0,
        "std_tpot_ms": 0.1,
        "mean_itl_ms": 2.1 + offset if multi_token else 0.0,
        "median_itl_ms": 2.0 + offset if multi_token else 0.0,
        "std_itl_ms": 0.2,
        "mean_e2el_ms": 530.0 + offset,
        "median_e2el_ms": 525.0 + offset,
        "std_e2el_ms": 5.0,
    }
    percentile_values = {
        "ttft": 22.0 + offset,
        "tpot": 2.2 + offset if multi_token else 0.0,
        "itl": 2.5 + offset if multi_token else 0.0,
        "e2el": 540.0 + offset,
    }
    for percentile in args.percentiles.split(","):
        for metric, value in percentile_values.items():
            key = bench_serve._percentile_result_key(percentile, metric)
            result[key] = value
    return result


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
    assert _option(command, "--random-range-ratio") == "0"
    assert _option(command, "--num-warmups") == "4"
    assert _option(command, "--seed") == "902"
    assert _option(command, "--temperature") == "0"
    assert _option(command, "--percentile-metrics") == "ttft,tpot,itl,e2el"
    assert "--ignore-eos" in command
    assert "--save-detailed" in command
    assert "--no-stream" not in command


def test_dataset_seed_and_server_sampling_are_distinct_metadata(tmp_path, monkeypatch):
    args = _parse(tmp_path, "--dataset-seed", "900")
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: "vLLM")
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
        _parse(tmp_path, "--blocks", "0")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--percentiles", "50,100")
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
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: "vLLM 1.2")
    metadata = bench_serve.collect_metadata(
        args,
        {"PRISMAQUANT_CB_DECODE": "cuda", "CUDA_VISIBLE_DEVICES": "0"},
    )

    assert metadata["git"] == {
        "commit": "deadbeef",
        "dirty": None,
        "source": "argument",
    }
    assert metadata["software"]["gridbook_version"] == "0.3.0"
    assert metadata["software"]["vllm_cli_version"] == "vLLM 1.2"
    assert metadata["artifacts"]["image_id"].endswith("sha256:abc")
    assert metadata["artifacts"]["model_id"] == "org/artifact@0123456"
    assert metadata["server"]["recorded_args"] == ["--enforce-eager"]
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


def test_urls_commands_and_server_args_never_persist_credentials(tmp_path):
    secret_url = (
        "https://alice:password@example.test:8443/v1?api_key=topsecret&mode=fast"
    )
    args = _parse(
        tmp_path,
        "--base-url",
        secret_url,
        "--server-arg=--hf-token supersecret",
        "--server-arg=Authorization: Bearer abc123",
        "--server-arg=MONKEY=visible",
    )
    metadata = bench_serve.collect_metadata(args, {})
    command = bench_serve.build_vllm_command(
        args, block_index=0, result_dir=tmp_path, result_filename="raw.json"
    )
    redacted_command = bench_serve.redact_command(command)
    serialized = json.dumps({"metadata": metadata, "command": redacted_command})

    for secret in ("alice", "password", "topsecret", "supersecret", "abc123"):
        assert secret not in serialized
    assert "mode=fast" in metadata["server"]["base_url"]
    assert "MONKEY=visible" in serialized
    assert _option(redacted_command, "--tokenizer") == "org/tokenizer"


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

    short_decode = dict(
        result, total_output_tokens=result["total_output_tokens"] - 1
    )
    with pytest.raises(bench_serve.BenchmarkError, match="not completed exactly"):
        bench_serve.validate_result(short_decode, args)

    uneven = dict(result, output_lens=[255, 257] + [256] * 6)
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected output_lens"):
        bench_serve.validate_result(uneven, args)


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


def test_single_token_prefill_explicitly_allows_empty_itl_and_zero_tpot(tmp_path):
    args = _parse(tmp_path, "--output-len", "1")
    result = _valid_result(args)
    assert all(not intervals for intervals in result["itls"])
    assert result["mean_itl_ms"] == result["mean_tpot_ms"] == 0.0
    bench_serve.validate_result(result, args)


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
    assert ttft["sample_stdev"] == pytest.approx(1.0)
    assert summary["metrics"]["p99_itl_ms"]["values"] == [2.5, 3.5, 4.5]


def test_run_writes_a_complete_report_without_vllm_or_server(tmp_path, monkeypatch):
    args = _parse(tmp_path, "--blocks", "3")
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
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: None)
    report = bench_serve.run_benchmark(args)
    written = json.loads(args.output.read_text())

    assert len(calls) == 3
    assert report["status"] == "success"
    assert written["schema"] == bench_serve.SCHEMA
    assert written["status"] == "success"
    assert [block["dataset_seed"] for block in written["blocks"]] == [
        1234,
        1235,
        1236,
    ]
    assert all(block["status"] == "success" for block in written["blocks"])
    assert all(block["returncode"] == 0 for block in written["blocks"])
    assert written["summary"]["completed_blocks"] == 3
    assert written["finished_at"].endswith("Z")
    assert os.stat(args.output).st_mode & 0o777 == 0o600


def test_failed_block_leaves_a_structured_failure_report(tmp_path, monkeypatch):
    args = _parse(tmp_path)
    monkeypatch.setattr(bench_serve, "_run_command", lambda command: 7)
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: None)

    with pytest.raises(bench_serve.BenchmarkError, match="exit code 7"):
        bench_serve.run_benchmark(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
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
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: None)
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
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: "vLLM")
    metadata = bench_serve.collect_metadata(args, {})
    assert metadata["software"]["gridbook_version"] is None
    assert "package metadata unavailable" in metadata["collection_errors"][
        "gridbook_version"
    ]


def test_nonfinite_raw_result_is_rejected_but_preserved_as_valid_json(
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
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: None)
    with pytest.raises(bench_serve.BenchmarkError, match="NaN/Infinity"):
        bench_serve.run_benchmark(args)
    report = json.loads(args.output.read_text())
    assert report["blocks"][0]["raw_result"]["mean_ttft_ms"] == {
        "nonfinite_float": "NaN"
    }


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
