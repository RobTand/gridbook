"""CPU-only tests for the reproducible online parity harness.

The real client is ``vllm bench serve``.  These tests pin command construction,
fixed-shape validation, provenance capture, and the report schema without a
vLLM install, a GPU, or a listening server.
"""

import json
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
    return {
        "completed": args.num_prompts,
        "failed": 0,
        "total_input_tokens": args.num_prompts * args.input_len,
        "total_output_tokens": args.num_prompts * args.output_len,
        "input_lens": [args.input_len] * args.num_prompts,
        "output_lens": [args.output_len] * args.num_prompts,
        "request_throughput": 2.0 + offset,
        "output_throughput": 500.0 + offset,
        "total_token_throughput": 600.0 + offset,
        "mean_ttft_ms": 20.0 + offset,
        "median_ttft_ms": 19.0 + offset,
        "std_ttft_ms": 1.0,
        "p99_ttft_ms": 22.0 + offset,
        "mean_tpot_ms": 2.0 + offset,
        "median_tpot_ms": 1.9 + offset,
        "std_tpot_ms": 0.1,
        "p99_tpot_ms": 2.2 + offset,
        "mean_itl_ms": 2.1 + offset,
        "median_itl_ms": 2.0 + offset,
        "std_itl_ms": 0.2,
        "p99_itl_ms": 2.5 + offset,
        "mean_e2el_ms": 530.0 + offset,
        "median_e2el_ms": 525.0 + offset,
        "std_e2el_ms": 5.0,
        "p99_e2el_ms": 540.0 + offset,
    }


def test_command_uses_official_streaming_fixed_shape_semantics(tmp_path):
    args = _parse(tmp_path, "--warmups", "4", "--seed", "900")
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
    assert _option(command, "--percentile-metrics") == "ttft,tpot,itl,e2el"
    assert "--ignore-eos" in command
    assert "--save-detailed" in command
    assert "--no-stream" not in command


def test_parser_rejects_ranges_that_break_a_comparable_workload(tmp_path):
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--blocks", "0")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--percentiles", "50,100")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--request-rate", "0")
    with pytest.raises(SystemExit):
        _parse(tmp_path, "--max-concurrency", "9")


def test_dispatch_environment_is_sorted_explicit_and_secret_safe():
    env = {
        "PRISMAQUANT_CB_DECODE": "cuda",
        "PRISMAQUANT_API_TOKEN": "do-not-save",
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
        "PRISMAQUANT_CB_DECODE",
    ]
    assert captured["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["MISSING"] is None
    assert captured["PRISMAQUANT_API_TOKEN"] == "<redacted>"
    assert captured["PRISMAQUANT_CB_DECODE"] == "cuda"


def test_metadata_captures_supplied_artifact_and_server_identity(tmp_path, monkeypatch):
    args = _parse(
        tmp_path,
        "--server-arg=--enforce-eager",
        "--dispatch-env",
        "CUDA_VISIBLE_DEVICES",
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
    assert metadata["dispatch"]["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "PRISMAQUANT_CB_DECODE": "cuda",
    }
    assert metadata["workload"]["streaming"] is True
    assert metadata["workload"]["block_seeds"] == [1234, 1235, 1236]


def test_validate_result_requires_true_streaming_metrics_and_exact_lengths(tmp_path):
    args = _parse(tmp_path)
    result = _valid_result(args)
    bench_serve.validate_result(result, args)

    missing_itl = dict(result)
    del missing_itl["mean_itl_ms"]
    with pytest.raises(bench_serve.BenchmarkError, match="streaming metrics"):
        bench_serve.validate_result(missing_itl, args)

    short_decode = dict(
        result, total_output_tokens=result["total_output_tokens"] - 1
    )
    with pytest.raises(bench_serve.BenchmarkError, match="not completed exactly"):
        bench_serve.validate_result(short_decode, args)

    uneven = dict(result, output_lens=[255, 257] + [256] * 6)
    with pytest.raises(bench_serve.BenchmarkError, match="unexpected output_lens"):
        bench_serve.validate_result(uneven, args)


def test_summary_keeps_independent_values_and_sample_dispersion(tmp_path):
    args = _parse(tmp_path)
    blocks = [
        {"result": _valid_result(args, offset=0.0)},
        {"result": _valid_result(args, offset=1.0)},
        {"result": _valid_result(args, offset=2.0)},
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
        result_dir = Path(_option(command, "--result-dir"))
        filename = _option(command, "--result-filename")
        block_index = int(_option(command, "--seed")) - args.seed
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
    assert [block["seed"] for block in written["blocks"]] == [1234, 1235, 1236]
    assert written["summary"]["completed_blocks"] == 3
    assert written["finished_at"].endswith("Z")


def test_failed_block_leaves_a_structured_failure_report(tmp_path, monkeypatch):
    args = _parse(tmp_path)
    monkeypatch.setattr(bench_serve, "_run_command", lambda command: 7)
    monkeypatch.setattr(bench_serve, "_vllm_version", lambda executable: None)

    with pytest.raises(bench_serve.BenchmarkError, match="exit code 7"):
        bench_serve.run_benchmark(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert report["blocks"] == []
    assert "exit code 7" in report["error"]


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
