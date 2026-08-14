#!/usr/bin/env python3
"""GPU correctness, graph, SASS, and microbenchmark harness for K18 D2F.

Do not run as part of CPU CI.  Run only after reviewing/compiling the probe on
an SM120/SM121 host::

    PRISMAQUANT_CB_FP4_D2F_PROBE=1 \
      python scripts/bench_fp4_direct_fragment_probe.py --json out.json

The reference is the existing native-W4A4 smem-decode fused OMMA kernel.  This
script intentionally makes no comparison or quality claim against the shipping
BF16 group-16-QDQ bucket; those activation scales are not UE4M3-exact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

import torch

from gridbook import codec
from gridbook.cuda_ext import (get_fp4_direct_fragment_probe_ext,
                               get_fused_fp4_ext)
from prismaquant import nvfp4_cb_formats as fmt

FLAG = "PRISMAQUANT_CB_FP4_D2F_PROBE"
K_BITS = 18
N_SUB = 2
TYPE_SIZE = 81


def prepare_weight(N: int, K: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    weight = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05
    codebook = fmt._resolve_codebook(
        K_BITS, "fp4", "product", None, torch.device("cuda"))
    packed, fields = fmt.nvfp4_cb_pack(
        weight, K_BITS, grid="fp4", mode="product", codebook=codebook,
        scale_coding=fmt.SCALE_CODING_TWO_TIER)
    del weight
    flat = codec.build_flat_codebook([part.cuda() for part in codebook])
    lut = codec.build_fp4_value_lut(flat, K_BITS, N_SUB).cuda().contiguous()
    compose = codec.build_compose_u8().cuda().contiguous()
    resident_packed = codec.pad_qweight(packed).contiguous()
    assert resident_packed.shape[0] == N
    assert resident_packed.shape[1] >= (K // 256) * TYPE_SIZE
    assert lut.numel() == 2048 and compose.numel() == 4096
    return {"packed": resident_packed, "lut": lut, "compose": compose,
            "fields": fields}


def native_activation(fused, M: int, K: int, seed: int):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    return fused.cb_nvfp4_quantize_rows(x)


def fused_call(fused, a, sfa, w, a_scales, b_scales, N, K):
    return fused.cb_fused_fp4_prefill_mm_scaled(
        a, sfa, w["packed"], w["lut"], w["compose"],
        a_scales, b_scales, N, K, K_BITS, N_SUB, TYPE_SIZE, True)


def direct_call(probe, a, sfa, w, a_scales, b_scales, N, K):
    return probe.cb_fp4_direct_fragment_k18(
        a, sfa, w["packed"], w["lut"], w["compose"],
        a_scales, b_scales, N, K)


def time_call(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return samples


def sass_gate(module) -> dict[str, object]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None and Path("/usr/local/cuda/bin/cuobjdump").is_file():
        cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required for the native-OMMA gate")
    proc = subprocess.run(
        [cuobjdump, "--dump-sass", module.__file__], text=True,
        capture_output=True, check=True)
    sass = proc.stdout + proc.stderr
    omma = len(re.findall(r"OMMA\.SF\.16864", sass, re.IGNORECASE))
    qmma = len(re.findall(r"QMMA", sass, re.IGNORECASE))
    if omma == 0 or qmma != 0:
        raise AssertionError(
            f"native instruction gate failed: OMMA.SF.16864={omma}, "
            f"QMMA={qmma}")
    return {"omma_sf_16864_count": omma, "qmma_count": qmma}


def graph_gate(probe, a, sfa, w, a_scales, b_scales, N, K) -> None:
    M = a.shape[0]
    workspace_bytes = probe.cb_fp4_direct_fragment_k18_workspace_bytes(M, N, K)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    args = (a, sfa, w["packed"], w["lut"], w["compose"],
            a_scales, b_scales, out, workspace, N, K)
    for _ in range(3):
        probe.cb_fp4_direct_fragment_k18_out(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe.cb_fp4_direct_fragment_k18_out(*args)
    for _ in range(128):
        graph.replay()
    torch.cuda.synchronize()
    eager = direct_call(probe, a, sfa, w, a_scales, b_scales, N, K)
    if not torch.equal(out, eager):
        raise AssertionError("captured/replayed output is not bit-identical")


def run_cell(probe, fused, M: int, N: int, K: int,
             warmup: int, iters: int) -> dict[str, object]:
    weight = prepare_weight(N, K, seed=1000 + K)
    a, sfa, a_scales = native_activation(fused, M, K, seed=2000 + M + K)
    b_scales = torch.ones(N, device="cuda", dtype=torch.float32)

    baseline = fused_call(fused, a, sfa, weight, a_scales, b_scales, N, K)
    candidate = direct_call(
        probe, a, sfa, weight, a_scales, b_scales, N, K)
    if not torch.equal(candidate, baseline):
        mismatch = int((candidate.view(torch.int16) !=
                        baseline.view(torch.int16)).sum().item())
        raise AssertionError(
            f"K18 D2F differs from native smem OMMA at M={M},K={K}: "
            f"{mismatch} BF16 words")

    base_samples = time_call(
        lambda: fused_call(fused, a, sfa, weight, a_scales, b_scales, N, K),
        warmup, iters)
    d2f_samples = time_call(
        lambda: direct_call(probe, a, sfa, weight, a_scales, b_scales, N, K),
        warmup, iters)
    base_ms = statistics.median(base_samples)
    d2f_ms = statistics.median(d2f_samples)
    return {
        "M": M, "N": N, "K": K,
        "bit_identical": True,
        "smem_decode_median_ms": base_ms,
        "direct_fragment_median_ms": d2f_ms,
        "speedup": base_ms / d2f_ms,
        "smem_decode_samples_ms": base_samples,
        "direct_fragment_samples_ms": d2f_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if os.environ.get(FLAG) != "1":
        raise SystemExit(f"set {FLAG}=1 explicitly before running the probe")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    cc = torch.cuda.get_device_capability()
    if cc not in ((12, 0), (12, 1)):
        raise SystemExit(f"native OMMA probe needs cc 12.0/12.1, got {cc}")
    probe = get_fp4_direct_fragment_probe_ext()
    fused = get_fused_fp4_ext()
    if probe is None or fused is None:
        raise SystemExit("probe or native fused reference extension unavailable")

    contract = dict(probe.cb_fp4_direct_fragment_probe_contract())
    if contract["arbitrary_fp32_group_scale_exact"] != 0:
        raise AssertionError("probe crossed the frozen activation bucket")
    sass = sass_gate(probe)
    # Graph capture at the small correctness cell; the production cells below
    # exercise the same allocation-free ABI geometry.
    graph_weight = prepare_weight(4096, 2048, seed=777)
    ga, gsfa, gas = native_activation(fused, 128, 2048, seed=778)
    graph_gate(probe, ga, gsfa, graph_weight, gas,
               torch.ones(4096, device="cuda"), 4096, 2048)

    cells = [run_cell(probe, fused, M, 4096, K, args.warmup, args.iters)
             for M in (128, 4096) for K in (2048, 4096)]
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(cc),
        "contract": contract,
        "resources": dict(probe.cb_fp4_direct_fragment_resource_report()),
        "sass": sass,
        "cuda_graph_replays": 128,
        "cells": cells,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
