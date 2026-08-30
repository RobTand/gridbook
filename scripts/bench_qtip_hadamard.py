#!/usr/bin/env python3
"""Profile reference/native research QTIP sign/Hadamard implementations.

This benchmark does not qualify serving and intentionally computes no speedup.
It can place the transparent torch FHT and the graph-safe warp-128 primitive in
one matched report. Promotion still requires in-process before/after profiles
and time-aligned Netdata power series from both Sparky and Sparklina.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from gridbook import qtip_hadamard as qtip_module
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


def _reference_input(value, signs, block_size):
    return qtip_module._normalized_block_hadamard_rows(
        value * signs.to(dtype=value.dtype), block_size).to(torch.bfloat16)


def _reference_output(value, signs, block_size):
    transformed = qtip_module._normalized_block_hadamard_rows(
        value, block_size)
    return (transformed * signs.float()).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--input-block", type=int, default=128)
    parser.add_argument("--output-block", type=int, default=128)
    parser.add_argument(
        "--implementation", choices=("reference", "native", "both"),
        default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing a CPU timing substitute")
    if args.k % args.input_block or args.n % args.output_block:
        raise SystemExit("block sizes must divide K and N")
    if args.implementation in ("native", "both") and (
            args.input_block != 128 or args.output_block != 128):
        raise SystemExit("native comparison requires both block sizes == 128")

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

    implementations = {}
    if args.implementation in ("reference", "both"):
        implementations["torch_reference"] = (
            lambda: _reference_input(x, input_signs, args.input_block),
            lambda: _reference_output(y, output_signs, args.output_block),
        )
    if args.implementation in ("native", "both"):
        qtip_module.prepare_qtip_hadamard_cuda()
        implementations["cuda_warp128"] = (
            lambda: apply_input_transform(
                x, input_signs, args.input_block),
            lambda: apply_inverse_output_transform(
                y, output_signs, args.output_block),
        )

    timings = {}
    for name, (input_fn, output_fn) in implementations.items():
        timings[name] = {
            "input_transform": _summary(_time_ms(
                input_fn, args.warmup, args.iterations)),
            "inverse_output_transform": _summary(_time_ms(
                output_fn, args.warmup, args.iterations)),
        }

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
            activities=activities, record_shapes=True,
            profile_memory=True) as profile:
        for name, (input_fn, output_fn) in implementations.items():
            with torch.profiler.record_function(
                    f"qtip_{name}_input_transform"):
                input_fn()
            with torch.profiler.record_function(
                    f"qtip_{name}_inverse_output_transform"):
                output_fn()
        torch.cuda.synchronize()
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))

    report = {
        "schema": "gridbook.qtip-hadamard-profile.v2",
        "status": "research_only_not_serving_qualified",
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "block_size": {"input": args.input_block,
                       "output": args.output_block},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "implementation": args.implementation,
        "comparison_scope": (
            "Only matched block_size=128 arms are comparable. Historical "
            "block-256 timings and Arm E input-block<=4096/output-block=256 "
            "cells are "
            "outside this native dispatch cell and are not before results."
        ),
        "timings": timings,
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
        "remaining_measurement_gate": [
            "matched before/after torch.profiler or nsys traces for the "
            "same artifact, M/K/N, residency, warmup, and replay mode",
            "time-aligned Netdata power series from Sparky and Sparklina for "
            "both arms; report work per joule against each ~140 W envelope",
            "exact-artifact vLLM load plus CUDA-graph capture/replay and "
            "served prefill/decode throughput against untransformed W4A4",
            "separate fusion evidence for input FHT + native_fp4_quant and "
            "output FHT + native FP4 epilogue; this standalone op proves "
            "neither fusion",
            "matched quality/KL and quantizable-parameter bpp accounting "
            "before any serving promotion",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
