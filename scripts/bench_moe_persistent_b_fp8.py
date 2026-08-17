#!/usr/bin/env python3
"""Whole-operator microbenchmark: FP8-CB persistent-B vs the expand+GEMM
bridge (ROADMAP K1.2 twin of ``bench_moe_persistent_b.py``).

PROPOSAL DATA ONLY. Per [NATIVE-PARITY](../docs/NATIVE-PARITY.md) a kernel
microbenchmark proposes; only the served protocol promotes.  Every arm times
the WHOLE routed operator — routing, QDQ, both projection stages, activation,
combine — from ``(x, topk_ids, topk_weights)`` to the combined ``[T, hidden]``
output, exactly as the FP4 bench does.

WHAT IS TIMED. Two ways one FP8-CB MoE layer's routed prefill can execute,
plus the deleted-transient column:

* ``persistent_b`` — the FP8-CB arm of the decode-in-mainloop lane
  (``cb_moe_persistent_b_prefill_fp8``): exact per-expert segments, ONE launch
  per stage, no ``[E, N, K]`` BF16 transient, decode =
  ``bf16_rn(f32(e4m3) * row_scale)`` inside the mainloop.
* ``expand_sm80`` — TODAY'S DEFAULT for FP8-CB above the fused mid-M band
  (``moe.py::_apply_prefill_native_bf16``, FP8 branch): ``cb_expand_fp8``
  materializes each expert chunk as e4m3 bytes, python multiplies by the
  per-row scale and rounds to BF16 in HBM, and the grouped CUTLASS bridge
  reads that transient back.
* ``expand ms`` / ``expand%`` — just the expand+scale+round calls the bridge
  performs, as a fraction of its whole operator: the tax persistent-B deletes.

OPERANDS ARE SYNTHETIC (random byte plane, random e4m3 codebook, random
per-row scales) — legal by construction, since an FP8-CB plane is pure
codeword bits with no packed scale section.  That is timing-grade on purpose:
byte VALUES do not change the memory traffic or the schedule, and the
arms-agree tolerance assert still gates that both arms decode the same bytes
to the same operator (the bit-level decode identity is the test suite's job,
``tests/test_cb_moe_persistent_b_fp8.py``).

The DSv4 row is the built 92 GB body's exact FP8-CB geometry: E=256,
hidden=4096, inter=2048, k=28 (its 11 FP8-CB routed layers), top_k=6.

Run it in the serving container::

    python3 scripts/bench_moe_persistent_b_fp8.py
    python3 scripts/bench_moe_persistent_b_fp8.py --tokens 512 2048
    python3 scripts/bench_moe_persistent_b_fp8.py --k 48   # format ceiling
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    print("torch is required", file=sys.stderr)
    raise SystemExit(2)

_GRIDBOOK_MODULES = {
    "gridbook", "gridbook.codec", "gridbook.cuda_ext", "gridbook.ops",
    "gridbook.moe_persistent_b_lane", "gridbook.native_cutlass",
}
try:
    from gridbook import codec, ops
    from gridbook.cuda_ext import (NativeKernelUnavailableError,
                                   get_bf16_grouped_ext, get_ext,
                                   get_moe_persistent_b_ext)
    from gridbook.moe_persistent_b_lane import (config as pb_config,
                                                resolve_cfg as pb_resolve_cfg,
                                                supports_fp8 as pb_supports)
    from gridbook.native_cutlass import (native_moe_activation,
                                         require_native_moe_activation)
except ModuleNotFoundError as exc:  # pragma: no cover - checkout fallback
    if exc.name not in _GRIDBOOK_MODULES:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook import codec, ops
    from gridbook.cuda_ext import (NativeKernelUnavailableError,
                                   get_bf16_grouped_ext, get_ext,
                                   get_moe_persistent_b_ext)
    from gridbook.moe_persistent_b_lane import (config as pb_config,
                                                resolve_cfg as pb_resolve_cfg,
                                                supports_fp8 as pb_supports)
    from gridbook.native_cutlass import (native_moe_activation,
                                         require_native_moe_activation)


# (label, experts, hidden, inter). The E=256 row is DSv4's exact routed
# geometry; the E=32 row keeps a small-E reference point.
SHAPES = (
    ("DSV4   h4096 i2048", 256, 4096, 2048),
    ("DSV4sm h4096 i2048", 32, 4096, 2048),
)

ACTIVATION = "silu"


def _fp8_widths(k: int) -> list[int]:
    base, extra = divmod(k, 4)
    return [base + (1 if i < extra else 0) for i in range(4)]


def make(k, K, E, N, seed, dev):
    """One synthetic FP8-CB stack: ``(qw, lut_u8, lut_f32, scale)``."""
    type_size = 4 * k
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(2 << w for w in _fp8_widths(k))
    lut_u8 = ((torch.randn(total, generator=g) * 0.5)
              .to(torch.float8_e4m3fn).view(torch.uint8).to(dev))
    lut_f32 = lut_u8.view(torch.float8_e4m3fn).float().contiguous()
    row_bytes = (K // codec.SUPERBLOCK) * type_size
    qw = torch.randint(0, 256, (E, N, row_bytes), dtype=torch.uint8,
                       generator=g).to(dev).contiguous()
    scale = ((torch.rand(E, N, generator=g) * 1.5 + 0.25) * 0.01) \
        .float().to(dev).contiguous()
    return qw, lut_u8, lut_f32, scale


def _native_bf16_chunk(E: int, hidden: int, inter: int) -> int:
    """``moe.py::_native_bf16_chunk`` from a bare layer shape (same budget)."""
    override = os.environ.get("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")
    if override is not None and override.strip():
        value = int(override)
        if value <= 0:
            raise ValueError(
                "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK must be positive")
        return min(E, value)
    raw_budget = os.environ.get("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES")
    budget = int(raw_budget) if raw_budget else (1 << 30)
    per_expert = 2 * inter * hidden * 2
    return max(1, min(E, budget // max(1, per_expert)))


def _router(E, tokens, top_k, seed, dev):
    generator = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(tokens, E, generator=generator), dim=-1)
    weights, ids = probs.topk(top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return ids.to(dev).contiguous(), weights.to(dev).contiguous()


def _rel_l2(value, reference):
    difference = (value.float() - reference.float()).norm()
    return float(difference / reference.float().norm().clamp_min(1e-12))


def _time(fn, iters, warmup):
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


def _arms(shape, tokens, args, cfg, stacks):
    label, E, hidden, inter = shape
    dev = "cuda"
    top_k, k = args.top_k, args.k
    type_size = 4 * k
    chunk = _native_bf16_chunk(E, hidden, inter)
    (w13, w13_lut_u8, w13_lut_f32, w13_scale,
     w2, w2_lut_u8, w2_lut_f32, w2_scale) = stacks

    torch.manual_seed(args.seed)
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16) * 0.1
    topk_ids, topk_weights = _router(E, tokens, top_k, args.seed, dev)
    plan = {"w13": (w13, hidden, w13_lut_u8, w13_scale),
            "w2": (w2, inter, w2_lut_u8, w2_scale)}

    def expand(which, c0, c1):
        """``moe.py::_expand_native_bf16_slice``, FP8 branch, verbatim
        machinery: pad_qweight + cb_expand_fp8 + fp32 scale + one bf16
        round, per expert chunk."""
        stack, in_f, lut_u8, scale = plan[which]
        packed = stack[c0:c1].contiguous()
        n_e = c1 - c0
        out_f = int(packed.shape[1])
        rows = n_e * out_f
        raw = codec.pad_qweight(packed.reshape(rows, -1))
        row0 = torch.zeros(rows, dtype=torch.int32, device=dev)
        value = ops.cb_expand_fp8(raw, lut_u8, row0, rows, in_f, k, 4,
                                  type_size)
        s = scale[c0:c1].reshape(rows).to(torch.float32)
        return (value.float() * s[:, None]).to(torch.bfloat16) \
            .view(n_e, out_f, in_f)

    def exact_route(capture_safe_counts):
        pair_expert = topk_ids.reshape(-1).to(torch.int64)
        order = torch.argsort(pair_expert, stable=True)
        pair_token = torch.arange(tokens, dtype=torch.int64,
                                  device=dev).repeat_interleave(top_k)
        rows = pair_token.index_select(0, order)
        if capture_safe_counts:
            counts = torch.zeros(E, dtype=torch.int64,
                                 device=dev).scatter_add_(
                0, pair_expert, torch.ones_like(pair_expert))
        else:
            counts = torch.bincount(pair_expert, minlength=E)
        ends = torch.cumsum(counts, 0, dtype=torch.int32).contiguous()
        return order, rows, ends

    def combine(pair_output, order, rows):
        pair_weight = topk_weights.reshape(-1).index_select(0, order) \
            .to(pair_output.dtype)
        pair_output.mul_(pair_weight[:, None])
        output = torch.zeros((tokens, hidden), dtype=x.dtype, device=dev)
        output.index_add_(0, rows, pair_output.to(output.dtype))
        return output

    def run_persistent_b():
        order, rows, ends = exact_route(True)
        xq = ops.fp8_act_qdq(x)
        x_sorted = xq.index_select(0, rows).contiguous()
        del xq
        pairs = int(x_sorted.shape[0])
        gate_up = torch.empty((pairs, 2 * inter), dtype=torch.bfloat16,
                              device=dev)
        ops.cb_moe_persistent_b_prefill_fp8(
            gate_up, x_sorted, w13, w13_lut_f32, w13_scale, ends, k,
            type_size, cfg)
        del x_sorted
        activated = torch.empty((pairs, inter), dtype=torch.bfloat16,
                                device=dev)
        native_moe_activation(ACTIVATION, activated, gate_up)
        del gate_up
        aq = ops.fp8_act_qdq(activated)
        del activated
        pair_output = torch.empty((pairs, hidden), dtype=torch.bfloat16,
                                  device=dev)
        ops.cb_moe_persistent_b_prefill_fp8(
            pair_output, aq, w2, w2_lut_f32, w2_scale, ends, k, type_size,
            cfg)
        del aq
        return combine(pair_output, order, rows)

    def run_expand_sm80():
        order, rows, ends = exact_route(False)
        xq = ops.fp8_act_qdq(x)
        x_sorted = xq.index_select(0, rows).contiguous()
        del xq
        pairs = int(x_sorted.shape[0])
        gate_up = torch.empty((pairs, 2 * inter), dtype=torch.bfloat16,
                              device=dev)
        for c0 in range(0, E, chunk):
            weight = expand("w13", c0, min(E, c0 + chunk))
            ops.cb_bf16_grouped_mm_out(gate_up, x_sorted, weight, ends, c0)
            del weight
        del x_sorted
        activated = torch.empty((pairs, inter), dtype=torch.bfloat16,
                                device=dev)
        native_moe_activation(ACTIVATION, activated, gate_up)
        del gate_up
        aq = ops.fp8_act_qdq(activated)
        del activated
        pair_output = torch.empty((pairs, hidden), dtype=torch.bfloat16,
                                  device=dev)
        for c0 in range(0, E, chunk):
            weight = expand("w2", c0, min(E, c0 + chunk))
            ops.cb_bf16_grouped_mm_out(pair_output, aq, weight, ends, c0)
            del weight
        del aq
        return combine(pair_output, order, rows)

    def run_expand_only():
        for which in ("w13", "w2"):
            for c0 in range(0, E, chunk):
                weight = expand(which, c0, min(E, c0 + chunk))
                del weight

    return chunk, run_persistent_b, run_expand_sm80, run_expand_only


def bench(args) -> int:
    if not torch.cuda.is_available():
        print("a CUDA device is required", file=sys.stderr)
        return 2
    if get_ext() is None:
        print("the CB-GEMV extension (cb_expand_fp8 / FP8 QDQ) could not be "
              "built", file=sys.stderr)
        return 2
    pb_ext = get_moe_persistent_b_ext()
    if pb_ext is None or not hasattr(pb_ext,
                                     "cb_moe_persistent_b_prefill_fp8"):
        print("this build carries no FP8-CB persistent-B arm "
              "(needs cc 12.0/12.1 and ABI schema >= 2)", file=sys.stderr)
        return 2
    bf16_ext = get_bf16_grouped_ext()
    if bf16_ext is None or not hasattr(bf16_ext, "cb_bf16_grouped_mm_out"):
        print("the grouped-BF16 extension could not be built", file=sys.stderr)
        return 2
    try:
        require_native_moe_activation(ACTIVATION, "MoE prefill benchmark")
    except NativeKernelUnavailableError as exc:
        print(f"native MoE activation unavailable: {exc}", file=sys.stderr)
        return 2
    type_size = 4 * args.k
    cfg = pb_resolve_cfg(pb_ext, fp8_type_size=type_size)

    device_name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"# device {device_name} (sm_{major}{minor}), "
          f"torch {torch.__version__}")
    print(f"# fp8-CB k={args.k} type_size={type_size} top_k={args.top_k} "
          f"seed={args.seed} tol={args.tol:g} iters={args.iters} "
          f"warmup={args.warmup}")
    eligible = [c for c in range(1, len(pb_config(pb_ext)) + 1)
                if bool(pb_ext.cb_moe_persistent_b_fp8_cfg_eligible(
                    c, type_size))]
    print(f"# persistent_b cfg={cfg} eligible_cfgs_at_k{args.k}={eligible} "
          f"configs={pb_config(pb_ext)}")
    print("# whole-operator arms: routing + QDQ + both stages + activation + "
          "combine (NATIVE-PARITY grouped-MoE rule); operands synthetic "
          "(timing-grade; bit identity is the test suite's gate)")
    print(f"\n{'shape':>19} {'E':>4} {'T':>5} {'P':>6} {'pb warm':>10} "
          f"{'sm80 warm':>10} {'expand ms':>10} {'expand%':>8} "
          f"{'sm80/pb':>8}")

    rows_run = 0
    for shape in SHAPES:
        label, E, hidden, inter = shape
        key = f"{label} E={E}"
        if args.only and args.only.lower() not in key.lower():
            continue
        reason = pb_supports(
            is_fp4=False, n_sub=4, k_bits=args.k, type_size=type_size,
            hidden=hidden, inter=inter, role_split=False)
        if reason is not None:
            print(f"# SKIP {key}: FP8 persistent-B arm cannot serve it "
                  f"({reason})")
            continue
        w13 = make(args.k, hidden, E, 2 * inter, args.seed, "cuda")
        w2 = make(args.k, inter, E, hidden, args.seed + 1, "cuda")
        stacks = (*w13, *w2)
        for tokens in args.tokens:
            chunk, run_pb, run_sm80, run_expand = _arms(
                shape, tokens, args, cfg, stacks)
            pairs = tokens * args.top_k

            reference = run_sm80()
            delta_pb = _rel_l2(run_pb(), reference)
            if delta_pb > args.tol:
                print(f"ARMS DISAGREE at {key} T={tokens}: "
                      f"pb-vs-sm80 rel L2 {delta_pb:.3e} > {args.tol:g}",
                      file=sys.stderr)
                return 3
            del reference

            pb_cold, pb_warm = _time(run_pb, args.iters, args.warmup)
            sm80_cold, sm80_warm = _time(run_sm80, args.iters, args.warmup)
            _, expand_warm = _time(run_expand, args.iters, args.warmup)

            print(f"{label:>19} {E:>4} {tokens:>5} {pairs:>6} "
                  f"{pb_warm:>10.3f} {sm80_warm:>10.3f} "
                  f"{expand_warm:>10.3f} "
                  f"{100 * expand_warm / max(sm80_warm, 1e-9):>7.1f}% "
                  f"{sm80_warm / max(pb_warm, 1e-9):>8.3f}")
            print(f"#   cold: pb {pb_cold:.2f} ms, sm80 {sm80_cold:.2f} ms; "
                  f"chunk={chunk}; pb-vs-sm80 rel L2 {delta_pb:.2e}")
            rows_run += 1
    if rows_run == 0:
        print("no rows matched --only", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[128, 512, 2048])
    parser.add_argument("--top-k", type=int, default=6,
                        help="DSv4's routed top_k")
    parser.add_argument("--k", type=int, default=28,
                        help="FP8-CB rung (DSv4 ships k=28)")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tol", type=float, default=4e-3,
                        help="reassociation-class arms-agree bound")
    parser.add_argument("--only", type=str, default="")
    return bench(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
