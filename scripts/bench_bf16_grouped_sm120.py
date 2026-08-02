#!/usr/bin/env python3
"""Whole-operator microbenchmark: sm12x-native BF16 lane vs the SM80 bridge.

PROPOSAL DATA ONLY. Per [NATIVE-PARITY](../docs/NATIVE-PARITY.md) a kernel
microbenchmark proposes; only the served protocol promotes. Nothing here is a
serving claim, a TTFT number, or grounds for changing a default.

WHAT IS TIMED. The three ways the same routed BF16 GEMM can be executed, each
measured as the WHOLE operator the serving path would run, not the inner GEMM:

* ``sm120``     — the padded-gather + one grouped launch of the opt-in lane.
                  The gather is charged to the lane, because the lane is what
                  requires it (the sm12x collective has no ptr-array grouping,
                  so every expert's rows must start on a TileM boundary).
* ``sm80``      — today's default: exact per-expert segments through the
                  device-scheduled CUTLASS 2.x grouped kernel, no padding.
* ``segmented`` — the retired reference the published 6-17% deficit was
                  measured against: one BF16 ``F.linear`` per expert writing
                  into a preallocated output.

Everything upstream of the GEMM that the three share — routing, weight
expansion, activation QDQ, the router combine — is excluded, exactly as the
2026-08-01 DSV4 bridge microbenchmark in docs/BENCHMARKS.md excluded it, so the
numbers are comparable to that table.

Run it in the serving container::

    python3 scripts/bench_bf16_grouped_sm120.py                 # default set
    python3 scripts/bench_bf16_grouped_sm120.py --tokens 512    # denser routing

Warm timing is the median of ``--iters`` CUDA-event samples after
``--warmup`` alternating warmups; cold is the first call after a synchronize.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the container always has it
    print("torch is required", file=sys.stderr)
    raise SystemExit(2)
import torch.nn.functional as F

try:
    from gridbook import ops
    from gridbook.cuda_ext import get_bf16_grouped_ext
except ModuleNotFoundError as exc:  # pragma: no cover - checkout fallback
    if exc.name not in {"gridbook", "gridbook.ops", "gridbook.cuda_ext"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook import ops
    from gridbook.cuda_ext import get_bf16_grouped_ext


# (label, experts, K, N) — the published bridge shapes plus a Laguna-class MoE.
SHAPES = (
    ("DSV4 w13   K=4096 N=4096", 32, 4096, 4096),
    ("DSV4 w2    K=2048 N=4096", 32, 2048, 4096),
    ("Laguna s1  K=3072 N=2048", 32, 3072, 2048),
    ("Laguna s2  K=1024 N=3072", 32, 1024, 3072),
)


def _routing(experts: int, tokens: int, top_k: int, seed: int):
    """A uniform synthetic router, as the published DSV4 microbenchmark used."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, experts, (tokens, top_k), generator=generator)
    counts = torch.bincount(ids.reshape(-1), minlength=experts)
    return counts.tolist()


def _padded_layout(counts, tile_m, device):
    expert_ids, source = [], []
    start = 0
    for expert, rows in enumerate(counts):
        for block in range((rows + tile_m - 1) // tile_m):
            expert_ids.append(expert)
            for r in range(tile_m):
                index = block * tile_m + r
                source.append(start + index if index < rows else rows + start)
        start += rows
    return (torch.tensor(expert_ids, dtype=torch.int32, device=device),
            torch.tensor(source, dtype=torch.int64, device=device))


def _time(fn, iters: int, warmup: int):
    """(cold ms, warm-median ms) for one callable."""
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    cold = start.elapsed_time(end)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return cold, samples[len(samples) // 2]


def bench(args) -> int:
    if not torch.cuda.is_available():
        print("a CUDA device is required", file=sys.stderr)
        return 2
    ext = get_bf16_grouped_ext()
    if ext is None:
        print("the grouped-BF16 extension could not be built", file=sys.stderr)
        return 2
    if not hasattr(ext, "cb_bf16_grouped_mm_sm120"):
        print("this build carries no sm12x lane (needs cc 12.0/12.1)",
              file=sys.stderr)
        return 2
    tile_m = int(ext.cb_bf16_grouped_sm120_tile_m())
    device = "cuda"
    name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"# device {name} (sm_{major}{minor}), torch {torch.__version__}")
    print(f"# tile_m={tile_m}, config="
          f"{ext.cb_bf16_grouped_sm120_config()}")
    print(f"# tokens={args.tokens} top_k={args.top_k} seed={args.seed} "
          f"iters={args.iters} warmup={args.warmup}")
    print(f"\n{'shape':>26} {'P':>6} {'Mp':>6} {'sm120 warm':>11} "
          f"{'sm80 warm':>10} {'segmented':>10} {'sm120/sm80':>11} "
          f"{'sm120/segd':>11}")

    for label, experts, k, n in SHAPES:
        counts = _routing(experts, args.tokens, args.top_k, args.seed)
        pairs = sum(counts)
        torch.manual_seed(args.seed)
        a = torch.randn(pairs, k, device=device, dtype=torch.bfloat16)
        weights = torch.randn(experts, n, k, device=device,
                              dtype=torch.bfloat16)
        ends = torch.tensor(counts, dtype=torch.int32, device=device) \
            .cumsum(0, dtype=torch.int32).contiguous()
        out_exact = torch.empty(pairs, n, device=device, dtype=torch.bfloat16)
        expert_ids, source = _padded_layout(counts, tile_m, device)
        mp = int(source.numel())
        zero_extended = torch.cat([a, a.new_zeros((1, k))])
        # The gather index the serving path builds once per prefill; the GATHER
        # itself is inside the timed region because the lane requires it.
        gather = source.clamp(max=pairs)
        out_pad = torch.empty(mp, n, device=device, dtype=torch.bfloat16)

        def run_sm120():
            padded = zero_extended.index_select(0, gather).contiguous()
            ops.cb_bf16_grouped_mm_sm120_out(out_pad, padded, weights,
                                             expert_ids, tile_m)

        def run_sm80():
            ops.cb_bf16_grouped_mm_out(out_exact, a, weights, ends, 0)

        def run_segmented():
            start = 0
            for expert, rows in enumerate(counts):
                if rows:
                    out_exact[start:start + rows] = F.linear(
                        a[start:start + rows], weights[expert])
                start += rows

        _, warm120 = _time(run_sm120, args.iters, args.warmup)
        _, warm80 = _time(run_sm80, args.iters, args.warmup)
        _, warm_seg = _time(run_segmented, args.iters, args.warmup)
        print(f"{label:>26} {pairs:>6} {mp:>6} {warm120:>10.3f}m "
              f"{warm80:>9.3f}m {warm_seg:>9.3f}m "
              f"{warm80 / warm120:>11.3f} {warm_seg / warm120:>11.3f}")

    print("\n# ratios > 1 mean the sm12x lane is FASTER than that baseline.")
    print("# PROPOSAL DATA (NATIVE-PARITY): microbenchmarks propose, only the "
          "served protocol promotes.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=128,
                        help="routed tokens T (DSV4 microbenchmark used 128)")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=731,
                        help="router seed (731 matches the published sweep)")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    return bench(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
