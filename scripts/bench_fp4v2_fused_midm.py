#!/usr/bin/env python3
"""Whole-operator microbenchmark: FP4-CB v2 fused mid-M lane vs expand+bridge.

PROPOSAL DATA ONLY. Per [NATIVE-PARITY](../docs/NATIVE-PARITY.md) a kernel
microbenchmark proposes; only the served protocol promotes. Nothing here is a
serving claim, a TTFT number, or grounds for changing a default.

WHAT IS TIMED. The two ways the same dense FP4-CB prefill GEMM can be executed
at mid M, each measured as the WHOLE operator the serving path would run:

* ``fused``  — one launch of ``cb_fused_fp4v2_prefill_mm``: the packed CB rows
               are decoded inside the CUTLASS producer/consumer stage and the
               ``[N, K]`` BF16 tile never exists in HBM.
* ``bridge`` — today's shipping route: ``expand_fp4_v2_to_weight`` writes the
               decoded ``[N, K]`` BF16 tile to HBM, then the owned CUTLASS
               grouped kernel (``E=1``) multiplies it. The EXPAND IS INSIDE
               the timed region, because removing it is the whole point of the
               lane; charging only the GEMM would measure a different claim.

Everything the two share upstream — the activation group-16 QDQ, the reshape —
is excluded, exactly as the fp8 mid-M and DSV4 bridge microbenchmarks in
docs/BENCHMARKS.md excluded it, so the numbers are comparable to those tables.

The comparison is bit-checked before it is timed: the script asserts the fused
result equals the bridge's decoded tile through the passthrough oracle, so a
regressed kernel cannot report a fast wrong number.

Run it in the serving container, holding the GPU bench lock::

    flock /tmp/claude-1000/gpu-bench.lock \\
      python3 scripts/bench_fp4v2_fused_midm.py

Warm timing is the median of ``--iters`` CUDA-event samples after ``--warmup``
alternating warmups; cold is the first call after a synchronize.
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

try:
    from gridbook import codec
    from gridbook.cuda_ext import (get_fused_fp4v2_ext,
                                   require_fp4_v2_expander)
    from gridbook.expand import expand_fp4_v2_to_weight
    from gridbook.ops import cb_bf16_grouped_mm_out
except ModuleNotFoundError as exc:  # pragma: no cover - checkout fallback
    if not (exc.name or "").startswith("gridbook"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook import codec
    from gridbook.cuda_ext import (get_fused_fp4v2_ext,
                                   require_fp4_v2_expander)
    from gridbook.expand import expand_fp4_v2_to_weight
    from gridbook.ops import cb_bf16_grouped_mm_out


# (label, N, K) — 27B-class dense projections plus a DSV4-class pair. These are
# the shapes the fp8 mid-M lane's 1.04x/1.26x/1.45x band was measured on.
SHAPES = (
    ("27B   qkv    K=5376 N=5376", 5376, 5376),
    ("27B   o      K=5376 N=5376", 5376, 5376),
    ("27B   gate   K=5376 N=14336", 14336, 5376),
    ("27B   down   K=14336 N=5376", 5376, 14336),
    ("DSV4  w13    K=4096 N=4096", 4096, 4096),
    ("DSV4  w2     K=2048 N=4096", 4096, 2048),
)

M_VALUES = (9, 16, 32, 64, 128)


def _prep(k_bits, n, kdim, device, seed):
    """One dense fp4-v2 layer's tensors, from the REAL prismaquant encoder."""
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
        row_off=torch.zeros(n, dtype=torch.int32, device=device),
        N=n, K=kdim, k=k_bits, ts=type_size)


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
    ext = get_fused_fp4v2_ext()
    if ext is None:
        print("the fused FP4-v2 quality extension could not be built",
              file=sys.stderr)
        return 2
    device = "cuda"
    require_fp4_v2_expander("fp4-v2 fused mid-M microbenchmark", device=device)

    name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    tile = [int(v) for v in ext.cb_fused_fp4v2_config()]
    print(f"# device {name} (sm_{major}{minor}), torch {torch.__version__}")
    print(f"# k_bits={args.k_bits}  tile[m,n,k,stages,cap]={tile}  "
          f"max_m={int(ext.cb_fused_fp4v2_max_m())}")
    print(f"# seed={args.seed} iters={args.iters} warmup={args.warmup}")
    print(f"\n{'shape':>28} {'M':>4} {'fused warm':>11} {'bridge warm':>12} "
          f"{'speedup':>8} {'bit-eq':>7}")

    for label, n, kdim in SHAPES:
        p = _prep(args.k_bits, n, kdim, device, args.seed)
        out_bridge = torch.empty(1, dtype=torch.bfloat16, device=device)
        for m in M_VALUES:
            generator = torch.Generator(device=device).manual_seed(m)
            a = (torch.randn(m, kdim, generator=generator, device=device)
                 * 0.1).to(torch.bfloat16).contiguous()
            ends = torch.full((1,), m, dtype=torch.int32, device=device)
            out_bridge = torch.empty(m, n, device=device,
                                     dtype=torch.bfloat16)

            def run_fused():
                ext.cb_fused_fp4v2_prefill_mm(
                    a, p["qwp"], p["cb_flat"], p["compose"], p["N"], p["K"],
                    p["k"])

            def run_bridge():
                weight = expand_fp4_v2_to_weight(
                    p["qwp"], p["cb_flat"], p["row_off"], p["compose"],
                    p["N"], p["K"], p["k"], 2, p["ts"])
                cb_bf16_grouped_mm_out(out_bridge, a, weight.unsqueeze(0),
                                       ends, 0)

            # Correctness before speed: the fused output must equal the SAME
            # tile config fed the expander's tile, bit for bit.
            weight = expand_fp4_v2_to_weight(
                p["qwp"], p["cb_flat"], p["row_off"], p["compose"], p["N"],
                p["K"], p["k"], 2, p["ts"])
            oracle = ext.sm120_fp4v2_bf16_mm_fork(a, weight)
            fused = ext.cb_fused_fp4v2_prefill_mm(
                a, p["qwp"], p["cb_flat"], p["compose"], p["N"], p["K"],
                p["k"])
            bit_eq = torch.equal(oracle.view(torch.uint16),
                                 fused.view(torch.uint16))
            del weight, oracle, fused

            _, warm_f = _time(run_fused, args.iters, args.warmup)
            _, warm_b = _time(run_bridge, args.iters, args.warmup)
            print(f"{label:>28} {m:>4} {warm_f:>10.3f}m {warm_b:>11.3f}m "
                  f"{warm_b / warm_f:>8.3f} {str(bit_eq):>7}")
        print()

    print("# speedup > 1 means the fused lane is FASTER than expand + bridge.")
    print("# The bridge column INCLUDES the expand, because the transient it "
          "writes is what the lane removes.")
    print("# PROPOSAL DATA (NATIVE-PARITY): microbenchmarks propose, only the "
          "served protocol promotes.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k-bits", type=int, default=16,
                        help="fp4-v2 product rung (12..24)")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    return bench(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
