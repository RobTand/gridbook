#!/usr/bin/env python3
"""A/B measurement: v2 GEMV rpb auto policy (smallest candidate) vs forced arms.

PROPOSAL DATA ONLY (kernel-eval 2026-08-21). The shipping auto policy in
``csrc/cb_gemv_v2.cu`` picks the SMALLEST rpb in {32, 64} with dictionary
re-stage amplification amp = stage_bytes / (rpb * row_bytes) <= 1.5; selecting
the LARGEST was proposed as worth ~5.9% on the measured k24 long-K cell but
"not taken because it has not been re-measured against the second k24 cell".
This script measures BOTH k24 release-width cells (K=2048 and K=4096 — at both,
the ds policy selects HALF staging since dict_bytes=64 KiB > 16 KiB and K >=
2048) with forced rpb, plus the full-auto policy pick as control:

* ``auto        rpb=0 dict_mode=0`` — exactly what ships today.
* ``half_rpb16  rpb=16 dict_mode=2`` — the pre-policy default (context).
* ``half_rpb32  rpb=32 dict_mode=2`` — the shipping policy's pick.
* ``half_rpb64  rpb=64 dict_mode=2`` — the LARGEST-candidate arm.
* ``full_*``    dict_mode=3 context arms, anchoring to the published FULL/HALF
                sweep numbers.

rpb only re-maps output rows to (block, warp); per-row superblock order and
the FMA chain are untouched, so every arm must produce BIT-IDENTICAL y —
asserted with ``torch.equal`` between all arms of a cell.

Calls the extension binding directly (``get_ext_v2().cb_gemv_v2``, which takes
``rpb`` / ``dict_mode`` as plain arguments), so no shipping code changes.

Timing: after --warmup calls, --rounds rounds of --iters CUDA-event-timed
calls; per-round median; reported number is the median across rounds.

GB/s uses the useful stream P*(row_bytes + 2K + 2N) bytes; the staged-dict
re-stage traffic P*nbp*stage_bytes is reported separately as "amp-eff" GB/s
(stream + restage). Run inside the serving container with the tree mounted::

    PRISMAQUANT_CB_EXT_DIR=/home/rob/dq-runs/ext-ab python3 scripts/bench_gemv_v2_rpb_ab.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the container always has it
    print("torch is required", file=sys.stderr)
    raise SystemExit(2)

try:
    from gridbook.cuda_ext import get_ext_v2
except ModuleNotFoundError:  # pragma: no cover - checkout fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook.cuda_ext import get_ext_v2


ARMS = (
    # (label, rpb, dict_mode)  — dict_mode 0=auto 2=half-staged 3=full-staged
    ("auto", 0, 0),
    ("half_rpb16", 16, 2),
    ("half_rpb32", 32, 2),
    ("half_rpb64", 64, 2),
    ("full_rpb32", 32, 3),
    ("full_rpb64", 64, 3),
)


def _time(ext, call_args, warmup: int, rounds: int, iters: int):
    """Median-of-round-medians ms for one launch configuration."""
    for _ in range(warmup):
        ext.cb_gemv_v2(*call_args)
    torch.cuda.synchronize()
    round_medians = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(rounds):
        samples = []
        for _ in range(iters):
            start.record()
            ext.cb_gemv_v2(*call_args)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        round_medians.append(statistics.median(samples))
    return statistics.median(round_medians), min(round_medians), max(round_medians)


def bench(args) -> int:
    if not torch.cuda.is_available():
        print("a CUDA device is required", file=sys.stderr)
        return 2
    ext = get_ext_v2()
    if ext is None:
        print("the CB-GEMV-v2 extension could not be built", file=sys.stderr)
        return 2
    name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"# device {name} (sm_{major}{minor}), torch {torch.__version__}")
    print(f"# k_bits={args.k_bits} N={args.n_out} E={args.experts} "
          f"pairs={args.pairs} rounds={args.rounds}x{args.iters} "
          f"warmup={args.warmup}")

    peak_probe = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    print(f"# machine peak read: {ext.bw_read(peak_probe, 200):.1f} GB/s")
    del peak_probe
    torch.cuda.empty_cache()

    rows = []
    for pairs in args.pairs:
        x_rows = pairs
        for k_feat in args.widths:
            k_bits = args.k_bits
            type_size = 4 * k_bits + 9          # fp4-v2 product contract
            n_sb = k_feat // 256
            row_bytes = n_sb * type_size
            cb_elems = (4 << ((k_bits + 1) // 2)) + (4 << (k_bits // 2))
            dict_bytes = cb_elems * 2           # bf16
            half_bytes = (4 << ((k_bits + 1) // 2)) * 2

            torch.manual_seed(args.seed)
            x = torch.randn(x_rows, k_feat, device="cuda",
                            dtype=torch.bfloat16)
            qw_stack = torch.randint(0, 256, (args.experts, args.n_out,
                                              row_bytes), device="cuda",
                                     dtype=torch.uint8)
            cb_flat = torch.randn(cb_elems, device="cuda",
                                  dtype=torch.bfloat16)
            compose = torch.randn(4096, device="cuda", dtype=torch.float32)
            pair_expert = torch.randint(0, args.experts, (pairs,), device="cuda",
                                        dtype=torch.int32)
            pair_xrow = torch.arange(pairs, device="cuda", dtype=torch.int32)
            stage_bytes = half_bytes               # ds=1 at BOTH cells (policy)
            nbp_of = lambda r: (args.n_out + r - 1) // r
            amp = lambda r: stage_bytes / (r * row_bytes)
            blocks_per_pair = {r: nbp_of(r) for r in (16, 32, 64)}

            label = f"P={pairs} K={k_feat}"
            results, refs = {}, {}
            for arm, rpb, dict_mode in ARMS:
                call = (x, qw_stack, cb_flat, compose, pair_expert, pair_xrow,
                        k_bits, type_size, rpb, dict_mode)
                y = ext.cb_gemv_v2(*call)
                refs[arm] = y
                med, lo, hi = _time(ext, call, args.warmup, args.rounds,
                                    args.iters)
                results[arm] = (med, lo, hi)
                del y

            # BIT-EQUALITY: pure schedule change — every arm must match.
            eq = {arm: bool(torch.equal(refs["auto"], refs[arm]))
                  for arm, _r, _d in ARMS}
            assert all(eq.values()), f"{label}: bit mismatch across arms {eq}"

            stream = pairs * (row_bytes + 2 * k_feat + 2 * args.n_out)
            print(f"\n== cell {label}  (dict {dict_bytes >> 10} KiB, "
                  f"half-stage {half_bytes >> 10} KiB, "
                  f"row_bytes {row_bytes}) ==")
            print(f"{'arm':>12} {'ms':>9} {'min':>9} {'max':>9} "
                  f"{'GB/s':>7} {'+restage':>8} {'blocks/pair':>11}")
            row = {"cell": label, "pairs": pairs, "K": k_feat,
                   "row_bytes": row_bytes, "bit_equal_all_arms": True}
            arms_json = {}
            base = None
            for arm, rpb, dict_mode in ARMS:
                med, lo, hi = results[arm]
                restage = pairs * nbp_of(rpb if rpb else 32) * \
                    (stage_bytes if dict_mode != 0 else half_bytes)
                gbps = stream / (med * 1e6)
                eff = (stream + restage) / (med * 1e6)
                if base is None:
                    base = med
                print(f"{arm:>12} {med:>9.4f} {lo:>9.4f} {hi:>9.4f} "
                      f"{gbps:>7.1f} {eff:>8.1f} "
                      f"{nbp_of(rpb if rpb else 32):>11}")
                arms_json[arm] = {"ms_median": med, "ms_min": lo, "ms_max": hi,
                                  "gbps_stream": gbps, "gbps_eff": eff,
                                  "bit_equal_to_auto": eq[arm]}
            # The policy pick must be half_rpb32 in both cells and time-match.
            pick_amp = amp(32)
            row["policy_pick_predicted"] = ("half_rpb32"
                                            if pick_amp <= 1.5 else "?")
            row["arms"] = arms_json
            rows.append(row)
            del x, qw_stack, cb_flat, compose, pair_expert, pair_xrow, refs
            torch.cuda.empty_cache()

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"schema": "gridbook.gemv_v2_rpb_ab.v1",
             "device": name, "capability": [major, minor],
             "rows": rows}, indent=2))
        print(f"\n# wrote {args.json}")

    print("\n# verdict inputs: half_rpb64 vs half_rpb32 per cell "
          "(>1 means the LARGEST candidate wins):")
    for row in rows:
        a32 = row["arms"]["half_rpb32"]["gbps_stream"]
        a64 = row["arms"]["half_rpb64"]["gbps_stream"]
        ctl = row["arms"]["auto"]["gbps_stream"]
        print(f"#   {row['cell']}: largest/smallest = {a64 / a32:.4f}  "
              f"(auto/half_rpb32 = {ctl / a32:.4f})")
    print("# PROPOSAL DATA: microbenchmarks propose, only the served protocol "
          "promotes.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k-bits", type=int, default=24)
    p.add_argument("--widths", type=int, nargs="+", default=[2048, 4096],
                   help="decoded K widths (the two k24 release-width cells)")
    p.add_argument("--n-out", type=int, default=4096)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--pairs", type=int, nargs="+", default=[30, 96],
                   help="routed pairs (T=5xk6=30 and T=16xk6=96 decode band)")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--json", default=None)
    return bench(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
