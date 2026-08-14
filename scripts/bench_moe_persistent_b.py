#!/usr/bin/env python3
"""Whole-operator microbenchmark: persistent-B MoE vs the expand+GEMM bridge.

PROPOSAL DATA ONLY. Per [NATIVE-PARITY](../docs/NATIVE-PARITY.md) a kernel
microbenchmark proposes; only the served protocol promotes. Nothing here is a
serving claim, a TTFT number, or grounds for changing a default. The same
document requires that a grouped MoE arm "time the whole routed/grouped
operator including routing, packing, launches, kernel, and combine - not an
isolated inner GEMM", and this script obeys that rule literally: every arm is
measured from ``(x, topk_ids, topk_weights)`` to the combined ``[T, hidden]``
layer output, both projection stages included.

WHAT IS TIMED. Four ways ONE FP4-CB MoE layer's routed prefill can be
executed, each the whole operator its serving method would run:

* ``persistent_b``  - the OPT-IN established decode-in-mainloop lane
                      (``moe.py::_apply_prefill_native_bf16_persistent_b``).
                      Exact per-expert segments, ONE
                      ``cb_moe_persistent_b_prefill`` launch per stage, and no
                      expanded ``[E, N, K]`` BF16 transient at all.
* ``d2r``           - the nested, default-off candidate in that SAME
                      persistent-B extension. It keeps routing and both
                      projection stages identical, removes the decoded B
                      shared-memory tile, and constructs MMA B registers with
                      the cooperative BF16 pair helper.
* ``expand_sm80``   - TODAY'S DEFAULT (``_apply_prefill_native_bf16``):
                      ``cb_expand_fp4_v2`` materializes each expert chunk as
                      BF16 in HBM and the device-scheduled CUTLASS 2.x grouped
                      kernel reads it back, over the same exact segments.
* ``expand_sm120``  - the OPT-IN sm12x bridge (``_apply_prefill_native_bf16_
                      sm120``): the same expansion, but feeding the sm12x
                      collective, which has no ptr-array grouping and so also
                      pays a PADDED tile-indexed gather and ONE host read of
                      the per-expert block offsets. Both costs belong to the
                      lane and are inside its timed region, exactly as
                      ``bench_bf16_grouped_sm120.py`` charges the gather to the
                      lane that requires it. Reported ``n/a`` on a build with
                      no sm12x lane.

THE EXPAND-TAX COLUMN is the point of the exercise. ``expand_ms`` times just
the ``cb_expand_fp4_v2`` calls ``expand_sm80`` performs (both stages, same
chunking), and ``expand%`` is that as a fraction of the whole ``expand_sm80``
operator - i.e. how much of today's default is the transient expansion the
persistent-B lane deletes.

SHAPES are ``(label, E, hidden, inter)`` - a whole MoE layer, not a bare GEMM
shape - and both weight stacks are built at their real serving shapes:

    w13 = fused gate_up_proj : [E, 2*inter, hidden]   (stage 1: K=hidden,
                                                       N=2*inter)
    w2  = down_proj          : [E, hidden, inter]     (stage 2: K=inter,
                                                       N=hidden)

The E=32 rows are exactly the DSV4-class and Laguna-class layers whose two
stages ``bench_bf16_grouped_sm120.py`` lists as four separate ``(K, N)`` GEMM
rows; a Laguna-class E=128 row is added because the expand tax scales with E
(the expansion runs over every expert, routed or not) and E=128 is closer to
production routed MoE.

Run it in the serving container::

    python3 scripts/bench_moe_persistent_b.py                    # default set
    python3 scripts/bench_moe_persistent_b.py --tokens 2048      # prefill-ish
    python3 scripts/bench_moe_persistent_b.py --only 'E=128'     # one row

Warm timing is the median of ``--iters`` CUDA-event samples after ``--warmup``
warmups; cold is the first call after a synchronize and is reported on the
per-row ``#`` line. Every row also asserts, OUTSIDE the timed region, that the
arms agree to relative L2 ``--tol`` (reassociation class).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the container always has it
    print("torch is required", file=sys.stderr)
    raise SystemExit(2)

_GRIDBOOK_MODULES = {
    "gridbook", "gridbook.codec", "gridbook.cuda_ext", "gridbook.ops",
    "gridbook.moe_persistent_b_lane", "gridbook.moe_routing",
    "gridbook.native_cutlass",
}
try:
    # gridbook.moe itself is deliberately NOT imported: it pulls in vLLM, and
    # every routing helper the arms need is in these leaf modules.
    from gridbook import codec, ops
    from gridbook.cuda_ext import (NativeKernelUnavailableError,
                                   get_bf16_grouped_ext, get_ext, get_ext_v2,
                                   get_moe_persistent_b_ext)
    from gridbook.moe_persistent_b_lane import (config as pb_config,
                                                resolve_cfg as pb_resolve_cfg,
                                                supports as pb_supports)
    from gridbook.moe_routing import (cb_grouped_block_offsets,
                                      cb_grouped_pad_routing)
    from gridbook.native_cutlass import (native_moe_activation,
                                         require_native_moe_activation)
except ModuleNotFoundError as exc:  # pragma: no cover - checkout fallback
    if exc.name not in _GRIDBOOK_MODULES:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gridbook import codec, ops
    from gridbook.cuda_ext import (NativeKernelUnavailableError,
                                   get_bf16_grouped_ext, get_ext, get_ext_v2,
                                   get_moe_persistent_b_ext)
    from gridbook.moe_persistent_b_lane import (config as pb_config,
                                                resolve_cfg as pb_resolve_cfg,
                                                supports as pb_supports)
    from gridbook.moe_routing import (cb_grouped_block_offsets,
                                      cb_grouped_pad_routing)
    from gridbook.native_cutlass import (native_moe_activation,
                                         require_native_moe_activation)

try:
    import prismaquant.nvfp4_cb_formats as pq
except ModuleNotFoundError as exc:  # pragma: no cover - reported by bench()
    pq, _PQ_REASON = None, str(exc)
else:
    _PQ_REASON = None


# (label, experts, hidden, inter) - a whole MoE LAYER. The E=32 rows are the
# DSV4-class and Laguna-class layers behind bench_bf16_grouped_sm120.py's four
# (K, N) GEMM rows; the E=128 row is the added production-scale expert count.
SHAPES = (
    ("DSV4   h4096 i2048", 32, 4096, 2048),
    ("Laguna h3072 i1024", 32, 3072, 1024),
    ("Laguna h3072 i1024", 128, 3072, 1024),
)

ACTIVATION = "silu"


def make(k, K, E, N, seed, dev):
    """Build one packed FP4-CB-v2 expert stack ``[E, N, (K/256)*type_size]``.

    ``K`` is the stack's in-features and ``N`` its out-features, so a caller
    builds w13 with ``(K=hidden, N=2*inter)`` and w2 with ``(K=inter,
    N=hidden)``. Returns ``(packed, flat_codebook, compose_table, type_size)``
    - the exact operand quartet both the persistent-B kernel and
    ``cb_expand_fp4_v2`` consume, so the arms share bit-identical weights.
    """
    cb = pq._resolve_codebook(k, "fp4", "product", None, dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(E, N, K, generator=g) * 0.02).to(dev)
    fields = pq.nvfp4_cb_fields(w, k, grid="fp4", mode="product", codebook=cb,
                                scale_coding="two_tier", encode_tier="fast")
    packed = pq.nvfp4_cb_assemble_bytes(fields, k, grid="fp4", mode="product")
    ts = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")
    qw = packed.reshape(E, N, (K // 256) * ts).contiguous().to(dev)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    lut = codec.build_flat_codebook(subs)
    comp = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(dev)
    return qw, lut, comp, ts


def _native_bf16_chunk(E: int, hidden: int, inter: int) -> int:
    """``moe.py::_native_bf16_chunk`` computed from a bare layer shape.

    Same authority (the larger w13 transient), same ~1 GiB default budget, same
    env overrides - so the bridge arms chunk their expert dimension exactly as
    serving does. At E=32 these shapes fit in one chunk; the E=128 Laguna row
    does not, and paying two chunks (plus the ``.contiguous()`` slice of the
    packed plane each one takes) is part of what the bridge costs there.
    """
    override = os.environ.get("PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK")
    if override is not None and override.strip():
        value = int(override)
        if value <= 0:
            raise ValueError(
                "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK must be positive")
        return min(E, value)
    raw_budget = os.environ.get("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES")
    budget = int(raw_budget) if raw_budget else (1 << 30)
    if budget <= 0:
        raise ValueError("PRISMAQUANT_CB_PREFILL_CHUNK_BYTES must be positive")
    per_expert = 2 * inter * hidden * 2                  # w13 BF16 bytes
    return max(1, min(E, budget // max(1, per_expert)))


def _router(E: int, tokens: int, top_k: int, seed: int, dev):
    """A synthetic softmax router: ``top_k`` DISTINCT experts per token."""
    generator = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(tokens, E, generator=generator), dim=-1)
    weights, ids = probs.topk(top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return ids.to(dev).contiguous(), weights.to(dev).contiguous()


def _rel_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    difference = (value.float() - reference.float()).norm()
    return float(difference / reference.float().norm().clamp_min(1e-12))


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


def _weights(shape, args):
    """Both packed expert stacks of one layer, at their real serving shapes.

    Built ONCE per shape and shared by every token count, so a sweep quantizes
    each layer once rather than once per T.
    """
    _, E, hidden, inter = shape
    w13, lut, compose, ts = make(args.k, hidden, E, 2 * inter, args.seed,
                                 "cuda")
    w2, _, _, _ = make(args.k, inter, E, hidden, args.seed + 1, "cuda")
    return w13, w2, lut, compose, ts


def _arms(shape, tokens, args, tile_m, cfg, weights):
    """Build four whole-operator closures plus the expand-only closure."""
    label, E, hidden, inter = shape
    dev = "cuda"
    top_k, k = args.top_k, args.k
    chunk = _native_bf16_chunk(E, hidden, inter)

    w13, w2, lut, compose, ts = weights
    torch.manual_seed(args.seed)
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16) * 0.1
    topk_ids, topk_weights = _router(E, tokens, top_k, args.seed, dev)
    stacks = {"w13": (w13, hidden), "w2": (w2, inter)}

    def expand(which, c0, c1):
        """``moe.py::_expand_native_bf16_slice``, FP4-CB-v2 branch."""
        stack, in_f = stacks[which]
        packed = stack[c0:c1].contiguous()
        n_e = c1 - c0
        out_f = int(packed.shape[1])
        rows = n_e * out_f
        weight = ops.cb_expand_fp4_v2(packed.reshape(rows, -1).view(-1), lut,
                                      compose, 0, rows, in_f, k, ts)
        return weight.view(n_e, out_f, in_f)

    def exact_route(capture_safe_counts):
        """The exact-segment routing both non-padded lanes build.

        ``capture_safe_counts`` mirrors the difference between the two lanes
        rather than papering over it: the persistent-B path counts with
        ``scatter_add_`` (pure device work), the default path calls
        ``torch.bincount``, whose CUDA implementation host-syncs. Both produce
        the identical integer counts; timing the one each lane actually runs is
        what keeps this a comparison of the shipped routes.
        """
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

    def run_persistent_b_with(op):
        order, rows, ends = exact_route(True)
        xq = ops.fp4_act_qdq(x)
        x_sorted = xq.index_select(0, rows).contiguous()
        del xq
        pairs = int(x_sorted.shape[0])
        gate_up = torch.empty((pairs, 2 * inter), dtype=torch.bfloat16,
                              device=dev)
        op(gate_up, x_sorted, w13, lut, compose, ends, k, ts, cfg)
        del x_sorted
        activated = torch.empty((pairs, inter), dtype=torch.bfloat16,
                                device=dev)
        native_moe_activation(ACTIVATION, activated, gate_up)
        del gate_up
        aq = ops.fp4_act_qdq(activated)
        del activated
        pair_output = torch.empty((pairs, hidden), dtype=torch.bfloat16,
                                  device=dev)
        op(pair_output, aq, w2, lut, compose, ends, k, ts, cfg)
        del aq
        return combine(pair_output, order, rows)

    def run_persistent_b():
        return run_persistent_b_with(ops.cb_moe_persistent_b_prefill)

    def run_d2r():
        return run_persistent_b_with(ops.cb_moe_persistent_b_prefill_d2r)

    def run_expand_sm80():
        order, rows, ends = exact_route(False)
        xq = ops.fp4_act_qdq(x)
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
        aq = ops.fp4_act_qdq(activated)
        del activated
        pair_output = torch.empty((pairs, hidden), dtype=torch.bfloat16,
                                  device=dev)
        for c0 in range(0, E, chunk):
            weight = expand("w2", c0, min(E, c0 + chunk))
            ops.cb_bf16_grouped_mm_out(pair_output, aq, weight, ends, c0)
            del weight
        del aq
        return combine(pair_output, order, rows)

    def run_expand_sm120():
        # moe.py::_padded_route(trim=True, block_offsets=True), inlined so the
        # bench does not import gridbook.moe (and therefore vLLM).
        pair_expert = topk_ids.reshape(-1).to(torch.long)
        pair_token = torch.arange(tokens, device=dev, dtype=torch.long) \
            .repeat_interleave(top_k)
        order = torch.argsort(pair_expert, stable=True)
        ptok_sorted = pair_token[order]
        pw_sorted = topk_weights.reshape(-1)[order].to(torch.float32)
        expert_ids, row_src, is_pad, _ = cb_grouped_pad_routing(
            topk_ids, E, tile_m)
        # The lane's ONE host read of device data, inside the timed region.
        block_off = cb_grouped_block_offsets(topk_ids, E, tile_m).tolist()
        n_blocks = int(block_off[E])
        expert_ids = expert_ids[:n_blocks].contiguous()
        row_src = row_src[:n_blocks * tile_m]
        is_pad = is_pad[:n_blocks * tile_m]
        rows = ptok_sorted.index_select(0, row_src)
        dest = torch.where(is_pad, torch.full_like(rows, tokens), rows)
        n_rows = n_blocks * tile_m

        xq = ops.fp4_act_qdq(x)
        xq = torch.cat([xq, xq.new_zeros((1, hidden))])
        a_pad = xq.index_select(0, dest).contiguous()
        del xq
        gate_up = torch.empty((n_rows, 2 * inter), dtype=torch.bfloat16,
                              device=dev)
        for c0 in range(0, E, chunk):
            c1 = min(E, c0 + chunk)
            b0, b1 = int(block_off[c0]), int(block_off[c1])
            if b1 == b0:
                continue
            weight = expand("w13", c0, c1)
            ops.cb_bf16_grouped_mm_sm120_out(
                gate_up[b0 * tile_m:b1 * tile_m],
                a_pad[b0 * tile_m:b1 * tile_m], weight,
                (expert_ids[b0:b1] - c0).contiguous(), tile_m)
            del weight
        del a_pad
        activated = torch.empty((n_rows, inter), dtype=torch.bfloat16,
                                device=dev)
        native_moe_activation(ACTIVATION, activated, gate_up)
        del gate_up
        aq = ops.fp4_act_qdq(activated)
        del activated
        y = torch.empty((n_rows, hidden), dtype=torch.bfloat16, device=dev)
        for c0 in range(0, E, chunk):
            c1 = min(E, c0 + chunk)
            b0, b1 = int(block_off[c0]), int(block_off[c1])
            if b1 == b0:
                continue
            weight = expand("w2", c0, c1)
            ops.cb_bf16_grouped_mm_sm120_out(
                y[b0 * tile_m:b1 * tile_m], aq[b0 * tile_m:b1 * tile_m],
                weight, (expert_ids[b0:b1] - c0).contiguous(), tile_m)
            del weight
        del aq
        pw_pad = pw_sorted.index_select(0, row_src)
        y = y * pw_pad[:, None].to(y.dtype)
        out = torch.zeros((tokens + 1, hidden), dtype=x.dtype, device=dev)
        out.index_add_(0, dest, y.to(out.dtype))
        return out[:tokens]

    def run_expand_only():
        """Only the transient the persistent-B lane deletes: both stages'
        ``cb_expand_fp4_v2`` calls, same chunking ``expand_sm80`` uses."""
        for which in ("w13", "w2"):
            for c0 in range(0, E, chunk):
                weight = expand(which, c0, min(E, c0 + chunk))
                del weight

    return (chunk, run_persistent_b, run_d2r, run_expand_sm80,
            run_expand_sm120, run_expand_only)


def bench(args) -> int:
    if not torch.cuda.is_available():
        print("a CUDA device is required", file=sys.stderr)
        return 2
    if pq is None:
        print(f"prismaquant is required to build CB expert stacks "
              f"({_PQ_REASON})", file=sys.stderr)
        return 2
    if get_ext() is None:
        print("the CB-GEMV extension (FP4 activation QDQ) could not be built",
              file=sys.stderr)
        return 2
    ext_v2 = get_ext_v2()
    if ext_v2 is None:
        print("the CB-GEMV-v2 extension (FP4-v2 expander) could not be built",
              file=sys.stderr)
        return 2
    ext_v2.cb_gemv_v2_prepare()
    pb_ext = get_moe_persistent_b_ext()
    if pb_ext is None or not hasattr(pb_ext, "cb_moe_persistent_b_prefill"):
        print("this build carries no persistent-B grouped MoE lane "
              "(needs cc 12.0/12.1)", file=sys.stderr)
        return 2
    d2r_symbols = (
        "cb_moe_persistent_b_prefill_d2r",
        "cb_moe_persistent_b_d2r_decode_pairs",
        "cb_moe_persistent_b_d2r_prepare",
        "cb_moe_persistent_b_d2r_configs",
    )
    missing_d2r = [name for name in d2r_symbols if not hasattr(pb_ext, name)]
    if missing_d2r:
        print("the persistent-B extension does not carry the D2R candidate "
              f"symbols {missing_d2r}", file=sys.stderr)
        return 2
    pb_ext.cb_moe_persistent_b_d2r_prepare()
    bf16_ext = get_bf16_grouped_ext()
    if bf16_ext is None or not hasattr(bf16_ext, "cb_bf16_grouped_mm_out"):
        print("the grouped-BF16 extension could not be built", file=sys.stderr)
        return 2
    try:
        require_native_moe_activation(ACTIVATION, "MoE prefill benchmark")
    except NativeKernelUnavailableError as exc:
        print(f"native MoE activation unavailable: {exc}", file=sys.stderr)
        return 2
    type_size = 4 * args.k + 9
    has_sm120 = hasattr(bf16_ext, "cb_bf16_grouped_mm_sm120_out")
    tile_m = int(bf16_ext.cb_bf16_grouped_sm120_tile_m()) if has_sm120 else 0
    cfg = pb_resolve_cfg(pb_ext)

    device_name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"# device {device_name} (sm_{major}{minor}), "
          f"torch {torch.__version__}")
    print(f"# fp4-CB-v2 k={args.k} type_size={type_size} "
          f"top_k={args.top_k} seed={args.seed} tol={args.tol:g} "
          f"iters={args.iters} warmup={args.warmup}")
    print(f"# persistent_b cfg={cfg} "
          f"tile_k={pb_ext.cb_moe_persistent_b_tile_k()} "
          f"configs={pb_config(pb_ext)}")
    print(f"# d2r same-extension configs="
          f"{pb_ext.cb_moe_persistent_b_d2r_configs()}")
    if has_sm120:
        print(f"# sm120 bridge tile_m={tile_m}, config="
              f"{bf16_ext.cb_bf16_grouped_sm120_config()}")
    else:
        print("# sm120 bridge absent in this build -> expand_sm120 = n/a")
    print("# whole-operator arms: routing + QDQ + both stages + activation + "
          "combine (NATIVE-PARITY grouped-MoE rule)")
    print(f"\n{'shape':>19} {'E':>4} {'T':>5} {'P':>6} {'pb warm':>10} "
          f"{'d2r warm':>10} {'pb/d2r':>8} {'sm80 warm':>10} "
          f"{'sm120 warm':>11} {'expand ms':>10} {'expand%':>8} "
          f"{'sm80/pb':>8} {'sm120/pb':>9}")

    rows_run = 0
    for shape in SHAPES:
        label, E, hidden, inter = shape
        key = f"{label} E={E}"
        if args.only and args.only.lower() not in key.lower():
            continue
        reason = pb_supports(
            is_fp4=True, is_v2=True, n_sub=2, k_bits=args.k,
            type_size=type_size, hidden=hidden, inter=inter)
        if reason is not None:
            print(f"# SKIP {key}: persistent-B lane cannot serve it "
                  f"({reason})")
            continue
        weights = _weights(shape, args)
        for tokens in args.tokens:
            (chunk, run_pb, run_d2r, run_sm80, run_sm120,
             run_expand) = _arms(shape, tokens, args, tile_m, cfg, weights)
            pairs = tokens * args.top_k

            # Agreement, OUTSIDE the timed region: a benchmark whose arms
            # compute different things is worthless.
            reference = run_sm80()
            delta_pb = _rel_l2(run_pb(), reference)
            delta_d2r = _rel_l2(run_d2r(), reference)
            delta_120 = _rel_l2(run_sm120(), reference) if has_sm120 else 0.0
            worst = max(delta_pb, delta_d2r, delta_120)
            if worst > args.tol:
                print(f"arms disagree on {key} T={tokens}: rel-L2 pb="
                      f"{delta_pb:.3e} d2r={delta_d2r:.3e} "
                      f"sm120={delta_120:.3e} > tol "
                      f"{args.tol:.3e}", file=sys.stderr)
                return 2
            del reference

            cold_pb, warm_pb = _time(run_pb, args.iters, args.warmup)
            cold_d2r, warm_d2r = _time(run_d2r, args.iters, args.warmup)
            cold_80, warm_80 = _time(run_sm80, args.iters, args.warmup)
            cold_ex, warm_ex = _time(run_expand, args.iters, args.warmup)
            if has_sm120:
                cold_120, warm_120 = _time(run_sm120, args.iters, args.warmup)
                sm120_warm = f"{warm_120:.3f}m"
                sm120_ratio = f"{warm_120 / warm_pb:.3f}"
                cold_120_text = f"{cold_120:.2f}"
                delta_120_text = f"{delta_120:.2e}"
            else:
                sm120_warm, sm120_ratio, cold_120_text = "n/a", "n/a", "n/a"
                delta_120_text = "n/a"

            print(f"# {key} T={tokens} chunks={-(-E // chunk)} agree: rel-L2 "
                  f"pb={delta_pb:.2e} d2r={delta_d2r:.2e} "
                  f"sm120={delta_120_text} "
                  f"<= {args.tol:.1e}; cold ms pb={cold_pb:.2f} "
                  f"d2r={cold_d2r:.2f} sm80={cold_80:.2f} "
                  f"sm120={cold_120_text} "
                  f"expand={cold_ex:.2f}")
            print(f"{label:>19} {E:>4} {tokens:>5} {pairs:>6} "
                  f"{warm_pb:>9.3f}m {warm_d2r:>9.3f}m "
                  f"{warm_pb / warm_d2r:>8.3f} {warm_80:>9.3f}m "
                  f"{sm120_warm:>11} "
                  f"{warm_ex:>9.3f}m {100.0 * warm_ex / warm_80:>7.1f}% "
                  f"{warm_80 / warm_pb:>8.3f} {sm120_ratio:>9}")
            rows_run += 1
            del run_pb, run_d2r, run_sm80, run_sm120, run_expand
            torch.cuda.empty_cache()
        del weights
        torch.cuda.empty_cache()

    if not rows_run:
        print(f"no shape matched --only {args.only!r}", file=sys.stderr)
        return 2
    print("\n# pb/d2r > 1 means D2R is faster than established persistent-B; "
          "other ratios > 1 mean persistent-B is faster than that baseline; "
          "expand% is the share of the DEFAULT (expand_sm80) "
          "operator spent in the transient persistent-B deletes.")
    print("# PROPOSAL DATA (NATIVE-PARITY): microbenchmarks propose, only the "
          "served protocol promotes.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[128, 512, 2048],
                        help="routed token counts T to sweep")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--k", type=int, default=16,
                        help="FP4-CB codebook bits (type_size = 4k+9)")
    parser.add_argument("--seed", type=int, default=731,
                        help="router/weight seed (731 matches the sweep)")
    parser.add_argument("--tol", type=float, default=4e-3,
                        help="max relative L2 between arms (reassociation)")
    parser.add_argument("--only", type=str, default="",
                        help="run only rows whose 'label E=N' key contains "
                             "this substring (iteration aid)")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    return bench(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
