#!/usr/bin/env python3
"""M <= 12 latency-cliff reproduction driver for the FP4-CB v2 fused mid-M lane.

PROFILING INSTRUMENTATION for the open cliff documented in
docs/BENCHMARKS.md (fused warm ~0.37 ms at M = 8..12 vs ~0.20 ms at M >= 13 on
the 27B qkv shape). This is proposal data only (NATIVE-PARITY): it times the
whole ``cb_fused_fp4v2_prefill_mm`` operator exactly as
``scripts/bench_fp4v2_fused_midm.py`` does, but sweeps the cliff cells
M = 8..13 explicitly, alternates sweep direction so measurement ordering
cannot explain a step, and optionally prints the expand+bridge column for the
same cells.

Run under the GPU bench lock::

    flock /home/rob/dq-runs/gpu-bench.lock \
      python3 scripts/bench_fp4v2_m12_cliff.py

``--profile`` runs a compact fixed schedule (few iterations, no printing)
intended to sit under ``ncu``: one process profiles cleanly per M list.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the container always has it
    print("torch is required", file=sys.stderr)
    raise SystemExit(2)

try:
    from gridbook import codec
    from gridbook.cuda_ext import get_fused_fp4v2_ext
except ModuleNotFoundError:  # pragma: no cover - checkout fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook import codec
    from gridbook.cuda_ext import get_fused_fp4v2_ext


# The exact cell the docs name: 27B qkv/o, hidden-class dense projection.
N_DIM = 5376
K_DIM = 5376


def _prep(k_bits, n, kdim, device, seed):
    """One dense fp4-v2 layer's tensors, identical to the shipped bench."""
    import prismaquant.nvfp4_cb_formats as pq

    cb = pq._resolve_codebook(k_bits, "fp4", "product", None,
                              torch.device(device))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(1, n, kdim, generator=generator) * 0.02).to(device)
    fields = pq.nvfp4_cb_fields(w, k_bits, grid="fp4", mode="product",
                                codebook=cb, scale_coding="two_tier",
                                encode_tier="fast")
    raw = pq.nvfp4_cb_assemble_bytes(fields, k_bits, grid="fp4",
                                     mode="product")
    type_size = pq.nvfp4_cb_type_size(k_bits, "fp4", "two_tier")
    packed = raw.reshape(n, (kdim // 256) * type_size).contiguous().to(device)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    return dict(
        qwp=codec.pad_qweight(packed),
        cb_flat=codec.build_flat_codebook(subs),
        compose=codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(device),
        n=n, K=kdim, k=k_bits)


def _clock_mhz() -> float | None:
    """Best-effort SM clock readback (None when no NVML is available)."""
    try:
        import pynvml
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def _time_op(fn, iters: int, warmup: int):
    """Warm-median ms (CUDA events) plus the median host wall ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples, walls = [], []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    t0 = time.perf_counter()
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    t1 = time.perf_counter()
    walls = [(t1 - t0) * 1e3 / iters]
    return statistics.median(samples), walls[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ms", type=str,
                        default="8,9,10,11,12,13,16,24,32",
                        help="comma-separated M values")
    parser.add_argument("--n", type=int, default=N_DIM)
    parser.add_argument("--K", type=int, default=K_DIM)
    parser.add_argument("--k-bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sweeps", type=int, default=3,
                        help="sweep count; direction alternates each sweep")
    parser.add_argument("--bridge", action="store_true",
                        help="also time the expand+BF16 bridge arm")
    parser.add_argument("--oracle", action="store_true",
                        help="also time the passthrough BF16 fork (same tile/"
                             "MMA/epilogue, plain TMA B, no decode)")
    parser.add_argument("--force-lut", type=int, default=-1,
                        help="force_lut_bytes for the fused arm (0/4096/...)")
    parser.add_argument("--profile", action="store_true",
                        help="compact silent schedule for use under ncu")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("a CUDA device is required", file=sys.stderr)
        return 2
    ext = get_fused_fp4v2_ext()
    if ext is None:
        print("the fused FP4-v2 quality extension could not be built",
              file=sys.stderr)
        return 2
    device = "cuda"
    name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    props = torch.cuda.get_device_properties(device)
    print(f"# device {name} (sm_{major}{minor}) SMs={props.multi_processor_count}"
          f", torch {torch.__version__}")
    tile = [int(v) for v in ext.cb_fused_fp4v2_config()]
    print(f"# k_bits={args.k_bits} tile[m,n,k,stages,cap]={tile}"
          f" max_m={int(ext.cb_fused_fp4v2_max_m())}"
          f" lut_plan={list(ext.cb_fused_fp4v2_lut_plan())}")

    p = _prep(args.k_bits, args.n, args.K, device, args.seed)
    ms = [int(v) for v in args.ms.split(",") if v.strip()]

    # Pre-build every activation once, outside any timed region.
    acts = {}
    for m in sorted(set(ms)):
        g = torch.Generator(device=device).manual_seed(m)
        acts[m] = (torch.randn(m, p["K"], generator=g, device=device)
                   * 0.1).to(torch.bfloat16).contiguous()

    def fused(m):
        return lambda: ext.cb_fused_fp4v2_prefill_mm(
            acts[m], p["qwp"], p["cb_flat"], p["compose"], p["n"], p["K"],
            p["k"], args.force_lut)

    if args.profile:
        # Deterministic compact schedule for ncu: warmup then a handful of
        # launches per M, ascending, one shape total.
        for m in ms:
            f = fused(m)
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            for _ in range(3):
                f()
            torch.cuda.synchronize()
        return 0

    results: dict[int, list[float]] = {}
    bridge_results: dict[int, list[float]] = {}
    oracle_results: dict[int, list[float]] = {}
    clocks: dict[int, list[float]] = {}
    order = list(ms)
    for sweep in range(args.sweeps):
        seq = order if sweep % 2 == 0 else list(reversed(order))
        tag = "asc" if sweep % 2 == 0 else "desc"
        for m in seq:
            warm, wall = _time_op(fused(m), args.iters, args.warmup)
            results.setdefault(m, []).append(warm)
            clk = _clock_mhz()
            if clk is not None:
                clocks.setdefault(m, []).append(float(clk))
            line = (f"sweep{sweep}:{tag:>4} M={m:<3d} fused warm "
                    f"{warm:8.3f} ms  (host {wall:7.3f} ms/call)")
            if args.oracle:
                # Decode-free reference through the SAME tile/MMA/epilogue:
                # needs the expanded BF16 weight, built once per M (untimed).
                from gridbook.expand import expand_fp4_v2_to_weight
                weight = expand_fp4_v2_to_weight(
                    p["qwp"], p["cb_flat"],
                    torch.zeros(p["n"], dtype=torch.int32, device=device),
                    p["compose"], p["n"], p["K"], p["k"], 2,
                    4 * p["k"] + 9)
                warm_o, _ = _time_op(
                    lambda: ext.sm120_fp4v2_bf16_mm_fork(acts[m], weight),
                    args.iters, args.warmup)
                oracle_results.setdefault(m, []).append(warm_o)
                line += f"  oracle {warm_o:8.3f} ms"
            if args.bridge:
                out = torch.empty(m, p["n"], device=device,
                                  dtype=torch.bfloat16)

                def run_bridge():
                    from gridbook.expand import expand_fp4_v2_to_weight
                    from gridbook.ops import cb_bf16_grouped_mm_out
                    weight = expand_fp4_v2_to_weight(
                        p["qwp"], p["cb_flat"],
                        torch.zeros(p["n"], dtype=torch.int32, device=device),
                        p["compose"], p["n"], p["K"], p["k"], 2,
                        4 * p["k"] + 9)
                    ends = torch.full((1,), m, dtype=torch.int32,
                                      device=device)
                    cb_bf16_grouped_mm_out(out, acts[m], weight.unsqueeze(0),
                                           ends, 0)

                _, bwarm = _time_op(run_bridge, args.iters, args.warmup)
                bridge_results.setdefault(m, []).append(bwarm)
                line += f"  bridge {bwarm:8.3f} ms"
            if clk is not None:
                line += f"  smclk {clk:.0f} MHz"
            print(line, flush=True)

    print("\n# summary (median across sweeps)")
    hdr = f"{'M':>4} {'fused med':>10} {'sweeps':>24}"
    if args.oracle:
        hdr += f" {'oracle med':>11}"
    if args.bridge:
        hdr += f" {'bridge med':>11}"
    print(hdr)
    for m in ms:
        r = results.get(m, [])
        row = f"{m:>4} {statistics.median(r):>9.3f}m {'/'.join(f'{x:.3f}' for x in r):>24}"
        if args.oracle:
            o = oracle_results.get(m, [])
            if o:
                row += f" {statistics.median(o):>10.3f}m"
        if args.bridge:
            b = bridge_results.get(m, [])
            if b:
                row += f" {statistics.median(b):>10.3f}m"
        print(row)
    print("# PROPOSAL DATA (NATIVE-PARITY): microbenchmarks propose, only "
          "the served protocol promotes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
