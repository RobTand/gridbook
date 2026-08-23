#!/usr/bin/env python3
"""A/B measurement: sm12x grouped-BF16 tile-scheduler swizzle crossover.

PROPOSAL DATA ONLY. The shipped policy (``csrc/cb_bf16_grouped_gemm.cu``)
picks scheduler swizzle {1, 8} by padded M-tile count with threshold 64 — the
MIDPOINT of the two measured grids (32 and 80 M-tiles); the crossover itself is
unmeasured. This script sweeps padded M-tile counts x swizzle {1, 8} on the
published shapes and locates the empirical crossover.

MECHANICS. The swizzle is chosen inside the extension at launch, so the A/B is
driven through a LATCHED diagnostic env override,
``PRISMAQUANT_CB_BF16_SWIZZLE in {auto,1,8}`` (default ``auto`` = today's
behavior bitwise). Because the flag is latched, ONE process carries ONE
setting: this driver spawns one child per arm (``--child``), each of which
times every cell and saves reference outputs for a subset of cells; the parent
then asserts with ``torch.equal`` that

  * swizzle 1 and swizzle 8 outputs are BIT-IDENTICAL everywhere saved (the
    swizzle reorders CTAs; every output tile's accumulation is independent of
    it), and
  * the ``auto`` arm's outputs are bit-identical to the forced arm the grid
    policy selects at that size (<64 tiles -> 1, >=64 -> 8), i.e. the default
    preserves behavior exactly.

Cells: E=32 experts (the published basis), per-expert row counts in whole
64-row tiles so total PADDED M-tiles land exactly on --tiles; four shapes
(w13/w2/Laguna s1/s2). Both A-source modes are timed: ``pad`` (padded-copy
GEMM only — the padded tensor is prebuilt OUTSIDE the timed region, matching
the historical GEMM-only sweep numbers) and ``gather`` (in-mainloop row gather,
one launch, natural expert order). Timing: median of --iters CUDA-event
samples after --warmup launches, per cell, as ``bench_bf16_grouped_sm120.py``.

Run inside the serving container::

    PRISMAQUANT_CB_BF16_SM120=1 python3 scripts/bench_bf16_swizzle_crossover.py \
        --out "$TMPDIR/swizzle_ab" [--tiles 24 32 ... 96]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the container always has it
    print("torch is required", file=sys.stderr)
    raise SystemExit(2)

try:
    from gridbook import ops
    from gridbook.cuda_ext import get_bf16_grouped_ext
except ModuleNotFoundError:  # pragma: no cover - checkout fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook import ops
    from gridbook.cuda_ext import get_bf16_grouped_ext


# (label, K, N) — the published bridge shapes (E=32).
SHAPES = (
    ("w13", 4096, 4096),
    ("w2", 2048, 4096),
    ("laguna_s1", 3072, 2048),
    ("laguna_s2", 1024, 3072),
)

EQUALITY_TILES = (24, 64, 96)  # below / at / above the shipped threshold

THRESHOLD = 64  # the shipped grid-policy threshold, padded M-tiles


def _counts(tiles: int, experts: int) -> list[int]:
    """Per-expert row counts landing EXACTLY ``tiles`` padded 64-row tiles."""
    base, rem = divmod(tiles, experts)
    return [(base + (1 if e < rem else 0)) * 64 for e in range(experts)]


def _padded_layout(counts, tile_m, device):
    """Row-padded tile-indexed layout (natural expert order), bench pattern."""
    starts, s = [0] * len(counts), 0
    for expert, rows in enumerate(counts):
        starts[expert] = s
        s += rows
    expert_ids, source = [], []
    for expert, rows in enumerate(counts):
        for _block in range((rows + tile_m - 1) // tile_m):
            expert_ids.append(expert)
            for r in range(tile_m):
                index = _block * tile_m + r
                source.append(starts[expert] + index if index < rows else s)
    return (torch.tensor(expert_ids, dtype=torch.int32, device=device),
            torch.tensor(source, dtype=torch.int64, device=device))


def _time(fn, iters: int, warmup: int):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _bench_cell(ops, tile_m, weights, a, padded, row_src, expert_ids, out_pad):
    """(pad_ms, gather_ms) for one cell; closures live only this long."""
    def run_pad():
        ops.cb_bf16_grouped_mm_sm120_out(out_pad, padded, weights,
                                         expert_ids, tile_m)

    def run_gather():
        ops.cb_bf16_grouped_mm_sm120_gather_out(out_pad, a, row_src,
                                                weights, expert_ids, tile_m)

    return run_pad, run_gather


def child(args) -> int:
    """One arm: run under exactly one latched PRISMAQUANT_CB_BF16_SWIZZLE."""
    ext = get_bf16_grouped_ext()
    if ext is None or not hasattr(ext, "cb_bf16_grouped_mm_sm120_out"):
        print("the grouped-BF16 extension with the sm12x lane could not be "
              "built", file=sys.stderr)
        return 2
    pin = os.environ.get("PRISMAQUANT_CB_BF16_SWIZZLE")
    if pin not in ("1", "8", None, ""):
        print(f"child requires SWIZZLE in {{1,8,unset}}; got {pin!r}",
              file=sys.stderr)
        return 2
    tile_m = int(ext.cb_bf16_grouped_sm120_tile_m())
    assert tile_m == 64
    # Every child must see IDENTICAL data or the cross-arm torch.equal gates
    # compare different problems.
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Spin the GPU clocks up on a throwaway mid-size cell before any timing,
    # so the first sweep cells do not absorb the idle-clock ramp.
    warm_w = torch.randn(8, 2048, 2048, device="cuda", dtype=torch.bfloat16)
    warm_a = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    warm_e = torch.zeros(8, dtype=torch.int32, device="cuda")
    warm_o = torch.empty(512, 2048, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        ops.cb_bf16_grouped_mm_sm120_out(warm_o, warm_a, warm_w, warm_e,
                                         tile_m)
    torch.cuda.synchronize()
    del warm_w, warm_a, warm_e, warm_o
    torch.cuda.empty_cache()
    results = []
    for label, k, n in SHAPES:
        experts = args.experts
        weights = torch.randn(experts, n, k, device="cuda",
                              dtype=torch.bfloat16)
        for tiles in args.tiles:
            counts = _counts(tiles, experts)
            pairs = sum(counts)
            a = torch.randn(max(pairs, 1), k, device="cuda",
                            dtype=torch.bfloat16)
            expert_ids, source = _padded_layout(counts, tile_m, "cuda")
            mp = int(source.numel())
            zero_extended = torch.cat([a, a.new_zeros((1, k))])
            gather_idx = source.clamp(max=pairs)
            out_pad = torch.empty(mp, n, device="cuda", dtype=torch.bfloat16)
            padded = zero_extended.index_select(0, gather_idx).contiguous()
            row_src = source.to(torch.int32)
            run_pad, run_gather = _bench_cell(ops, tile_m, weights, a, padded,
                                              row_src, expert_ids, out_pad)

            row = {"shape": label, "K": k, "N": n, "tiles": tiles,
                   "mp": mp, "pairs": pairs}
            for mode, fn in (("pad", run_pad), ("gather", run_gather)):
                fn()  # produce the reference output before timing
                if tiles in EQUALITY_TILES:
                    key = f"{args.arm}_{label}_t{tiles}_{mode}.pt"
                    torch.save(out_pad.clone(), out_dir / key)
                med = _time(fn, args.iters, args.warmup)
                row[f"{mode}_ms"] = med
            results.append(row)
            print(f"[{args.arm}] {label:>10} K={k:<5} N={n:<5} tiles={tiles:<3} "
                  f"pad={row['pad_ms']:>8.3f}ms gather={row['gather_ms']:>8.3f}ms",
                  flush=True)
            del a, padded, out_pad, zero_extended, gather_idx
            torch.cuda.empty_cache()
        del weights
        torch.cuda.empty_cache()
    payload = {"arm": args.arm, "env": pin or "auto",
               "device": torch.cuda.get_device_name(),
               "capability": list(torch.cuda.get_device_capability()),
               "rows": results}
    (out_dir / f"results_{args.arm}.json").write_text(json.dumps(payload,
                                                                indent=2))
    return 0


def parent(args) -> int:
    out_dir = Path(args.out)
    env = dict(os.environ)
    env["PRISMAQUANT_CB_BF16_SM120"] = "1"
    here = str(Path(__file__).resolve())
    arms = {}
    for arm, pin in (("sw1", "1"), ("sw8", "8"), ("auto", "")):
        child_env = dict(env)
        if pin:
            child_env["PRISMAQUANT_CB_BF16_SWIZZLE"] = pin
        else:
            child_env.pop("PRISMAQUANT_CB_BF16_SWIZZLE", None)
        print(f"\n########## arm {arm} "
              f"(PRISMAQUANT_CB_BF16_SWIZZLE={pin or 'unset'}) ##########",
              flush=True)
        proc = subprocess.run([sys.executable, here, "--child", "--arm", arm,
                               "--out", str(out_dir),
                               "--tiles", *map(str, args.tiles)],
                              env=child_env)
        if proc.returncode != 0:
            return proc.returncode
        arms[arm] = json.loads(
            (out_dir / f"results_{arm}.json").read_text())

    # Bit gates: order-only difference must not move a bit, anywhere saved.
    print("\n## bit gates (torch.equal)")
    ok = True
    compared = 0
    for path in sorted(out_dir.glob("sw1_*.pt")):
        key = path.name[len("sw1_"):]
        y1 = torch.load(path, weights_only=True)
        y8 = torch.load(out_dir / f"sw8_{key}", weights_only=True)
        same = bool(torch.equal(y1, y8))
        auto_key = f"auto_{key}"
        policy_arm = "sw8" if int(key.split("_t")[1].split("_")[0]) >= THRESHOLD \
            else "sw1"
        ya = torch.load(out_dir / auto_key, weights_only=True)
        yp = torch.load(out_dir / f"{policy_arm}_{key}", weights_only=True)
        auto_same = bool(torch.equal(ya, yp))
        ok &= same and auto_same
        compared += 1
        print(f"#   {key}: sw1==sw8 {same}; auto==policy({policy_arm}) "
              f"{auto_same}")
        del y1, y8, ya, yp
    if compared == 0:
        # A gate that compares nothing must not read as a pass. References are
        # only saved for tile counts in EQUALITY_TILES, so a --tiles set that
        # misses them silently skipped every torch.equal below. Fail closed on
        # the miss count rather than reporting a vacuous OK.
        print(f"BIT GATE VACUOUS: 0 cells compared -- --tiles must include at "
              f"least one of EQUALITY_TILES={EQUALITY_TILES}", file=sys.stderr)
        return 1
    if not ok:
        print("BIT GATE FAILED", file=sys.stderr)
        return 1
    print(f"#   {compared} cell(s) compared, all bit-identical")

    # Combined table + crossover read-out.
    by_arm = {arm: {(r["shape"], r["tiles"]): r for r in data["rows"]}
              for arm, data in arms.items()}
    print(f"\n## warm median ms, {'x'.join(map(str, args.tiles))} padded "
          f"M-tiles x swizzle {{1,8}} ({arms['sw1']['device']}, "
          f"sm_{arms['sw1']['capability'][0]}{arms['sw1']['capability'][1]})")
    for mode in ("pad", "gather"):
        print(f"\n### mode {mode}")
        print(f"{'shape':>10} {'K':>5} {'N':>5} {'tiles':>5} {'sw1':>9} "
              f"{'sw8':>9} {'sw8/sw1':>7}")
        for label, k, n in SHAPES:
            for tiles in args.tiles:
                r1 = by_arm["sw1"][(label, tiles)]
                r8 = by_arm["sw8"][(label, tiles)]
                m1, m8 = r1[f"{mode}_ms"], r8[f"{mode}_ms"]
                print(f"{label:>10} {k:>5} {n:>5} {tiles:>5} {m1:>9.3f} "
                      f"{m8:>9.3f} {m8 / m1:>7.3f}")
        # Empirical crossover per shape: first tile count where sw8 wins.
        print("# per-shape first tile count where sw8 >= sw1 (>0.5% faster):")
        for label, _k, _n in SHAPES:
            winners = [t for t in args.tiles
                       if by_arm["sw8"][(label, t)][f"{mode}_ms"]
                       <= 0.995 * by_arm["sw1"][(label, t)][f"{mode}_ms"]]
            print(f"#   {label}: sw8-first-wins-at="
                  f"{min(winners) if winners else None}")
    (out_dir / "combined.json").write_text(json.dumps(
        {"schema": "gridbook.bf16_swizzle_crossover.v1",
         "threshold_shipped": THRESHOLD,
         "arms": arms}, indent=2))
    print(f"\n# wrote {out_dir}/combined.json")
    print("# PROPOSAL DATA: microbenchmarks propose, only the served protocol "
          "promotes.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--child", action="store_true",
                   help="run one latched arm (internal)")
    p.add_argument("--arm", default="sw1",
                   help="child arm label (sw1|sw8|auto)")
    p.add_argument("--out",
                   default=os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                        "swizzle_ab"),
                   help="output dir; defaults under $TMPDIR so a host run never "
                        "lands in the host /tmp")
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--tiles", type=int, nargs="+",
                   default=[24, 32, 40, 48, 56, 64, 72, 80, 96],
                   help="total padded M-tile counts to sweep")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260821)
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("a CUDA device is required", file=sys.stderr)
        return 2
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
