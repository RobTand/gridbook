#!/usr/bin/env python3
"""Profile the research QTIP sign/Hadamard wrappers on one CUDA device.

This benchmark does not qualify serving.  It isolates the current transparent
torch FHT so a future native kernel has an explicit before measurement.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from gridbook.qtip_hadamard import (
    apply_input_transform,
    apply_inverse_output_transform,
    seeded_signs,
)


def _time_ms(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": p95,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--input-block", type=int, default=256)
    parser.add_argument("--output-block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing a CPU timing substitute")
    if args.k % args.input_block or args.n % args.output_block:
        raise SystemExit("block sizes must divide K and N")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260830)
    x = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device,
                    generator=generator)
    y = torch.randn(args.m, args.n, dtype=torch.bfloat16, device=device,
                    generator=generator)
    input_signs = seeded_signs("input", args.k, 11, device=device,
                               dtype=torch.bfloat16)
    output_signs = seeded_signs("output", args.n, 29, device=device,
                                dtype=torch.bfloat16)

    input_fn = lambda: apply_input_transform(
        x, input_signs, args.input_block)
    output_fn = lambda: apply_inverse_output_transform(
        y, output_signs, args.output_block)
    input_samples = _time_ms(input_fn, args.warmup, args.iterations)
    output_samples = _time_ms(output_fn, args.warmup, args.iterations)

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
            activities=activities, record_shapes=True,
            profile_memory=True) as profile:
        with torch.profiler.record_function("qtip_input_transform"):
            input_fn()
        with torch.profiler.record_function("qtip_inverse_output_transform"):
            output_fn()
        torch.cuda.synchronize()
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))

    report = {
        "schema": "gridbook.qtip-hadamard-profile.v1",
        "status": "research_only_not_serving_qualified",
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "block_size": {"input": args.input_block,
                       "output": args.output_block},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "input_transform": _summary(input_samples),
        "inverse_output_transform": _summary(output_samples),
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "profiler_top_cuda": profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=20),
        "trace": str(args.trace) if args.trace else None,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
