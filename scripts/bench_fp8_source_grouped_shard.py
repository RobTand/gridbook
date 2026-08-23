"""Decode microbench: DSv4 wo_a grouped GEMV at G=8 (TP=1) vs its TP shards.

Not a gate. The requirement the coordinator funded is TP>1 SUPPORT; this number
only says whether a TP=2 wo_a decode is a win, a wash, or a regression per rank.

Geometry is the DSv4 o-projection plane: rows/group 1024, K 4096, batch 1.
G=8 is the whole plane (8192x4096); G=4 is one rank's contiguous group slice at
TP=2 (4096x4096); G=2 is one rank's slice at TP=4 (2048x4096) -- the same bytes
a sharded serve would hold, sliced from the same source plane.

A/B/A ordering, median of the repeats, CUDA-event timing.
"""
from __future__ import annotations

import statistics

import torch

from gridbook import cuda_ext

EXT = cuda_ext.get_fp8_source_w8a16_ext()
assert EXT is not None, "source-FP8 W8A16 extension unavailable"
DEV = torch.device("cuda")

GROUPS, ROWS, K = 8, 1024, 4096
WARMUP = 50
ITERATIONS = 500
REPEATS = 3


def planes(n: int, k: int, seed: int = 61):
    generator = torch.Generator(device=DEV).manual_seed(seed)
    raw_q = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=DEV,
                          generator=generator)
    raw_q.masked_fill_((raw_q & 0x7f) == 0x7f, 0x7e)
    shape = ((n + 127) // 128, (k + 127) // 128)
    raw_scale = torch.empty(shape, dtype=torch.uint8, device=DEV)
    row = torch.arange(shape[0], device=DEV, dtype=torch.int16)[:, None]
    col = torch.arange(shape[1], device=DEV, dtype=torch.int16)[None, :]
    raw_scale.copy_(((119 + 3 * row + 5 * col) % 13 + 113).to(torch.uint8))
    return raw_q.view(torch.float8_e4m3fn), raw_scale.view(torch.float8_e8m0fnu)


FULL_Q, FULL_S = planes(GROUPS * ROWS, K)
FULL_X = torch.randn(1, GROUPS, K, device=DEV, dtype=torch.bfloat16) * 0.125


def arm(shard_degree: int):
    """Rank 0's slice at this shard degree, exactly as vLLM would narrow it."""

    local_groups = GROUPS // shard_degree
    local_rows = local_groups * ROWS
    q = FULL_Q[:local_rows].contiguous()
    scales = FULL_S[:local_rows // 128].contiguous()
    x = FULL_X[:, :local_groups].contiguous()
    return local_groups, q, scales, x


def time_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / ITERATIONS


def main() -> None:
    degrees = (1, 2, 4)
    arms = {d: arm(d) for d in degrees}
    samples: dict[int, list[float]] = {d: [] for d in degrees}
    order = list(degrees) + list(reversed(degrees)) + list(degrees)
    for degree in order[:REPEATS * len(degrees)]:
        groups, q, scales, x = arms[degree]
        samples[degree].append(
            time_us(lambda: EXT.fp8_source_gemv(x, q, scales, groups)))

    print(f"device        : {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(str(v) for v in torch.cuda.get_device_capability())}")
    print(f"geometry      : rows/group {ROWS}, K {K}, batch 1")
    print(f"iterations    : {ITERATIONS} per timed run, {WARMUP} warmup, "
          f"{REPEATS} repeats A/B/A")
    baseline = statistics.median(samples[1])
    for degree in degrees:
        groups = GROUPS // degree
        median = statistics.median(samples[degree])
        spread = max(samples[degree]) - min(samples[degree])
        label = "TP=1 (whole plane)" if degree == 1 else f"TP={degree} (1 rank)"
        print(f"G={groups} {label:<20} {median:8.2f} us/call  "
              f"spread {spread:5.2f} us  "
              f"vs G=8 {median / baseline:5.3f}x  "
              f"runs {[round(v, 2) for v in samples[degree]]}")

    # Cache control. A rank's plane is half the bytes, so a hot-loop microbench
    # can serve it from cache in a way a real serve (which streams every other
    # layer between two wo_a calls) never would. Rotate over enough distinct
    # planes to exceed any on-chip capacity and re-time.
    print()
    for degree in degrees:
        groups = GROUPS // degree
        local_rows = groups * ROWS
        copies = 8 * degree  # constant total bytes across arms
        qs = [planes(local_rows, K, seed=200 + i)[0] for i in range(copies)]
        ss = [planes(local_rows, K, seed=200 + i)[1] for i in range(copies)]
        x = FULL_X[:, :groups].contiguous()
        counter = {"i": 0}

        def rotating():
            i = counter["i"] % copies
            counter["i"] += 1
            return EXT.fp8_source_gemv(x, qs[i], ss[i], groups)

        cold = statistics.median([time_us(rotating) for _ in range(REPEATS)])
        hot = statistics.median(samples[degree])
        total_mb = copies * local_rows * K / 1e6
        print(f"G={groups} rotating over {copies} planes ({total_mb:6.1f} MB "
              f"of E4M3): {cold:8.2f} us/call  vs hot-loop {cold / hot:5.3f}x")
        del qs, ss
        torch.cuda.empty_cache()

    print()
    # Accuracy of the sharded call against the full call's own columns.
    full = EXT.fp8_source_gemv(
        FULL_X, FULL_Q, FULL_S, GROUPS).reshape(1, GROUPS, ROWS)
    for degree in (2, 4):
        groups, q, scales, x = arms[degree]
        got = EXT.fp8_source_gemv(
            x, q, scales, groups).reshape(1, groups, ROWS)
        want = full[:, :groups]
        rel = float((got.float() - want.float()).norm()
                    / want.float().norm().clamp_min(1e-30))
        bitwise = torch.equal(got.view(torch.int16), want.view(torch.int16))
        print(f"G={groups} rank0 vs full-call columns: rel-L2 {rel:.3e}, "
              f"bitwise {bitwise}")


if __name__ == "__main__":
    main()
