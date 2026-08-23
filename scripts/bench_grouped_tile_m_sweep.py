"""Routed rho-sweep calibrating GROUPED_WIDE_TILE_MIN_ROWS_PER_EXPERT (K0.4).

Drives the grouped fused FP8-CB lane at k28 with FORCED TileM in {128, 256} —
the caller-side override the selector's own comment names as the measurement
path (moe_routing.py: "skips the selector outright by handing
_apply_prefill_grouped_fused_v2 an explicit tile_m=") — across rho, routed
histogram family, expert count and DSV4-class shapes.  For every cell it

  * gates on PRE-COMBINE stage bit-equality between the tile arms (the
    ``test_both_compiled_tiles_are_bit_identical`` contract), refusing to time
    an unequal pair;
  * times the isolated two-stage grouped GEMM (the quantity the cost model
    T(t) = B(t)*(d + t*m) speaks about) AND the end-to-end operator under the
    production default trim latch;
  * records q / n_hi / sum(floor(c_e/256)) / B(128) / B(256) of the exact
    routed histogram, then fits x = d/m per shape and evaluates Theorem 7's
    three-item acceptance test against the shipped constant 512.

Run on an IDLE GPU:

    flock /home/rob/dq-runs/gpu-bench.lock \
        python3 scripts/bench_grouped_tile_m_sweep.py

or race for a window with scripts/race_run_guarded.sh, which holds both GPU
locks exclusively for the docker job.

``--smoke`` runs tiny cells (also what warms the JIT extension cache).
Cells are written as JSON to --json for downstream analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

import torch

DEV = "cuda"
TOPK = 8
K_BITS = 28
# (name, hidden, inter): w13 is [E, 2*inter, hidden] (N=2*inter, K=hidden),
# w2 is [E, hidden, inter].  "w13k2048" is the w13 K=2048/N=4096 class,
# "dsv4-body" is the DeepSeek-V4 body cell (w13 K=7168, w2 N=7168).
SHAPES = [("w13k2048", 2048, 2048), ("dsv4-body", 7168, 2048)]
EXPERTS = (128, 256)
RHOS = (100, 150, 200, 261, 300, 373, 450, 512, 565, 650, 800, 1000, 1200)
FAMILIES = ("uniform", "adversarial", "skewed", "balanced")
SKEW_ALPHA = 1.0
SKEW_MIX = 0.85
NOISE = 0.01  # sign gate: |diff| below this fraction of the arm time is a tie


def _other_compute_apps() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 — cannot attest means report and stop
        return ["<nvidia-smi unavailable>"]
    me = str(os.getpid())
    others: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, name = line.partition(",")
        if pid.strip() == me or name.strip() == "[Not Found]":
            # "[Not Found]" = the pid exited between nvidia-smi's query and
            # its name resolution.  Across PID namespaces that includes this
            # sweep's own just-torn-down host pid, which os.getpid() can
            # never match — a dead process holds no compute, so skipping it
            # stops a clean run condemning itself; named foreign pids stay
            # fatal.
            continue
        others.append(line)
    return others


def _sm_clocks() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"
    return out.replace("\n", ";")


def _silu_act():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    try:
        return MoEActivation.from_str("silu")
    except Exception:  # noqa: BLE001 - enum spelling differs across vLLM
        return MoEActivation.SILU


def build_layer(*, experts: int, hidden: int, inter: int, seed: int = 0):
    """Synthetic FP8-CB layer at k28, packed one expert at a time so the fp32
    encode transient never exceeds one projection."""
    import types

    import gridbook.codec as codec
    import prismaquant.nvfp4_cb_formats as fmt
    from gridbook.moe import PrismaQuantCBMoEMethod

    n_sub = 4
    type_size = fmt.nvfp4_cb_type_size(K_BITS, "fp8")
    codebook = fmt._resolve_codebook(
        K_BITS, "fp8", "product", None, torch.device(DEV))
    rb13 = (hidden // codec.SUPERBLOCK) * type_size
    rb2 = (inter // codec.SUPERBLOCK) * type_size

    torch.manual_seed(seed)
    p13 = torch.empty(experts, 2 * inter, rb13, dtype=torch.uint8, device=DEV)
    p2 = torch.empty(experts, hidden, rb2, dtype=torch.uint8, device=DEV)
    s13: list[torch.Tensor] = []
    s2: list[torch.Tensor] = []
    for e in range(experts):
        w13e = torch.randn(2 * inter, hidden, device=DEV) * 0.05
        pe, f13 = fmt.nvfp4_cb_pack(
            w13e, K_BITS, grid="fp8", mode="product", codebook=codebook)
        p13[e].copy_(pe.reshape(2 * inter, rb13))
        s13.append(f13["scales"].reshape(-1).float().cpu())
        del w13e, pe, f13
        w2e = torch.randn(hidden, inter, device=DEV) * 0.05
        p2e, f2 = fmt.nvfp4_cb_pack(
            w2e, K_BITS, grid="fp8", mode="product", codebook=codebook)
        p2[e].copy_(p2e.reshape(hidden, rb2))
        s2.append(f2["scales"].reshape(-1).float().cpu())
        del w2e, p2e, f2

    method = PrismaQuantCBMoEMethod.__new__(PrismaQuantCBMoEMethod)
    method.quant_config = None
    method.scheme = {"grid": "fp8", "mode": "product", "k": K_BITS,
                     "n_sub": n_sub, "type_size": type_size}
    method.prefix = "bench.grouped_tile_m_sweep"
    method.is_fp4 = False
    method.is_v2 = False
    method.k = K_BITS
    method.n_sub = n_sub
    method.type_size = type_size
    method._sub_table = None
    layer = types.SimpleNamespace(
        _cb_E=experts,
        _cb_hidden=hidden,
        _cb_inter=inter,
        w13_cb_qweight=p13,
        w2_cb_qweight=p2,
        w13_weight_scale=torch.stack(s13).to(DEV),
        w2_weight_scale=torch.stack(s2).to(DEV),
        _cb_flat=codec.build_flat_codebook([t.to(DEV) for t in codebook]),
        _cb_compose=torch.zeros(1, device=DEV),
        apply_router_weight_on_input=False,
        activation=types.SimpleNamespace(value="silu"),
    )
    return method, layer


def counts_for(family: str, experts: int, rho: int) -> list[int]:
    """Exact integer histogram c_e summing to E*rho per family.

    ``adversarial`` pins every nonzero residue at 128 mod 256 (the worst case
    the derivation is built around; below 128 it mixes 128-row and empty
    experts, which keeps every nonzero residue in the q class).  ``skewed``
    is a Zipf(alpha=1)/uniform mixture with realistic MoE spread.  ``balanced``
    puts exactly rho rows on every expert.
    """
    target = experts * rho
    if family == "balanced":
        return [rho] * experts
    if family == "adversarial":
        if rho < 128:
            active = target // 128
            tail = target - 128 * active
            counts = [128] * active + ([tail] if tail else [])
            return counts + [0] * (experts - len(counts))
        base = 128 + 256 * ((rho - 128) // 256)
        rem = target - experts * base          # in [0, 256*E)
        up, tail = divmod(rem, 256)
        return [base + 256] * up + [base + tail] + \
               [base] * (experts - up - 1)
    if family == "skewed":
        ranks = torch.arange(1, experts + 1, dtype=torch.float64)
        zipf = ranks.pow(-SKEW_ALPHA)
        mix = (1.0 - SKEW_MIX) / experts + SKEW_MIX * zipf / zipf.sum()
        raw = mix * target
        counts = torch.floor(raw).to(torch.long).tolist()
        rem = target - sum(counts)
        frac = (raw - raw.floor()).tolist()
        order = sorted(range(experts), key=lambda i: -frac[i])
        for i in range(rem):
            counts[order[i % experts]] += 1
        return counts
    raise ValueError(family)


def make_routing(family: str, experts: int, rho: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    tokens = (experts * rho) // TOPK
    counts = ([] if family == "uniform"
              else counts_for(family, experts, rho))
    if family == "uniform":
        scores = torch.rand(tokens, experts, generator=gen)
        ids_cpu = scores.topk(TOPK, dim=1).indices
        counts = torch.bincount(ids_cpu.reshape(-1),
                                minlength=experts).tolist()
        ids = ids_cpu
    else:
        pair = torch.tensor(
            [e for e, c in enumerate(counts) for _ in range(c)],
            dtype=torch.long)
        ids = pair[torch.randperm(pair.numel(), generator=gen)]
        ids = ids.reshape(tokens, TOPK)
    weights = torch.rand(tokens, TOPK, generator=gen) + 0.1
    assert sum(counts) == experts * rho, (sum(counts), experts * rho)
    return ids.to(torch.int32).to(DEV), weights.float().to(DEV), counts


def hist_stats(counts: list[int]) -> dict:
    """q / n_hi / S on the exact histogram, plus B(t) for both tiles."""
    q = nhi = s_full = b128 = b256 = 0
    for c in counts:
        if c <= 0:
            continue
        full, res = divmod(c, 256)
        s_full += full
        if 1 <= res <= 128:
            q += 1
            b128 += 2 * full + 1
        elif res >= 129:
            nhi += 1
            b128 += 2 * full + 2
        else:
            b128 += 2 * full
        b256 += full + (1 if res else 0)
    assert b128 == 2 * b256 - q, (b128, b256, q)   # padding lemma, live check
    return {"q": q, "n_hi": nhi, "s_full": s_full,
            "b128": b128, "b256": b256}


def bench_ms(fn, warmup: int, reps: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def bench_ab(fa, fb, warmup: int, reps: int) -> tuple[float, float]:
    """Interleaved A/B medians: both arms sampled on one DVFS trajectory, so
    clock drift between measurements cannot bias either tile."""
    ev_a = (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
    ev_b = (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
    for _ in range(warmup):
        fa()
        fb()
    torch.cuda.synchronize()
    sa: list[float] = []
    sb: list[float] = []
    for _ in range(reps):
        ev_a[0].record(); fa(); ev_a[1].record()
        ev_b[0].record(); fb(); ev_b[1].record()
        torch.cuda.synchronize()
        sa.append(ev_a[0].elapsed_time(ev_a[1]))
        sb.append(ev_b[0].elapsed_time(ev_b[1]))
    return statistics.median(sa), statistics.median(sb)


def capture_arm(method, layer, x, weights, ids, act, tile_m: int):
    """Run the operator once with a forced tile, capturing each stage's exact
    argument tuple and output through the production call path."""
    captured: list = []

    def spy(fext, args, tile, _orig=method._grouped_call, _cap=captured, **kw):
        out = _orig(fext, args, tile, **kw)
        _cap.append((fext, args, tile, out))
        return out

    original = method._grouped_call
    method._grouped_call = spy
    try:
        out = method._apply_prefill_grouped_fused_v2(
            layer, x, weights, ids, act, tile_m=tile_m)
    finally:
        method._grouped_call = original
    assert out is not None, f"tile_m={tile_m} arm declined"
    assert len(captured) == 2, captured
    return out, captured


def pair_order(y: torch.Tensor, ids: torch.Tensor, experts: int,
               tile_m: int) -> torch.Tensor:
    from gridbook.moe_routing import cb_grouped_pad_routing

    _eids, row_src, is_pad, _n = cb_grouped_pad_routing(ids, experts, tile_m)
    keep = ~is_pad[:y.shape[0]]
    order = torch.argsort(row_src[:y.shape[0]][keep], stable=True)
    return y[:y.shape[0]][keep][order]


def run_cell(method, layer, act, *, hidden: int, family: str, rho: int,
             warmup: int, reps: int) -> dict:
    experts = layer._cb_E
    seed = int(hashlib.sha1(
        f"{family}:{rho}:{experts}:{layer._cb_hidden}".encode()
    ).hexdigest()[:8], 16)
    ids, weights, counts = make_routing(family, experts, rho, seed)
    tokens = ids.shape[0]
    torch.manual_seed(seed ^ 0xA5A5)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=DEV) * 0.5

    outs, caps = {}, {}
    for t in (128, 256):
        outs[t], caps[t] = capture_arm(method, layer, x, weights, ids, act, t)
    for stage in (0, 1):
        a = pair_order(caps[128][stage][3], ids, experts, 128)
        b = pair_order(caps[256][stage][3], ids, experts, 256)
        if not torch.equal(a.view(torch.uint16), b.view(torch.uint16)):
            return {"blocked": "stage outputs differ between tiles",
                    "cell": f"{family}/rho{rho}/E{experts}"}
    rel_final = ((outs[256].float() - outs[128].float()).norm()
                 / outs[128].float().norm().clamp_min(1e-6)).item()

    fext = caps[128][0][0]
    stage_args = {t: (caps[t][0][1], caps[t][1][1]) for t in (128, 256)}

    def gemm_pair(t: int):
        a1, a2 = stage_args[t]
        return lambda: (
            fext.cb_fused_moe_grouped(*a1, t),
            fext.cb_fused_moe_grouped(*a2, t))

    def e2e(t: int):
        return lambda: method._apply_prefill_grouped_fused_v2(
            layer, x, weights, ids, act, tile_m=t)

    rec = {
        "family": family, "rho": rho, "E": experts,
        "tokens": tokens, "hidden": hidden, "inter": layer._cb_inter,
        **hist_stats(counts),
        "rel_final": rel_final,
        "clocks": [_sm_clocks()],
    }
    rec["gemm128"], rec["gemm256"] = bench_ab(
        gemm_pair(128), gemm_pair(256), warmup, reps)
    rec["e2e128"], rec["e2e256"] = bench_ab(
        e2e(128), e2e(256), warmup, reps)

    del caps, stage_args, outs, x, ids, weights
    torch.cuda.empty_cache()
    return rec


def fit_xhat(cells: list[dict]) -> tuple[float, float, float]:
    """diff_ms ~= d_hat*(B128-B256) + m_hat*(128*B128 - 256*B256), no
    intercept, pooled across E/family within one shape."""
    u = torch.tensor([float(r["b128"] - r["b256"]) for r in cells],
                     dtype=torch.float64)
    v = torch.tensor([float(128 * r["b128"] - 256 * r["b256"])
                      for r in cells], dtype=torch.float64)
    y = torch.tensor([r["gemm128"] - r["gemm256"] for r in cells],
                     dtype=torch.float64)
    sol = torch.linalg.lstsq(torch.stack([u, v], dim=1), y.unsqueeze(1))
    d_hat, m_hat = (float(s) for s in sol.solution.flatten())
    return d_hat, m_hat, (d_hat / m_hat if m_hat else float("nan"))


def analyse(records: list[dict]) -> str:
    good = [r for r in records if not r.get("blocked")]
    lines: list[str] = ["## Theorem 7 evaluation", ""]
    for shape_h in sorted({r["hidden"] for r in good}):
        cells = [r for r in good if r["hidden"] == shape_h]
        d_hat, m_hat, x_hat = fit_xhat(cells)
        bound = 128.0 * (1.0 + 256.0 / x_hat) if x_hat > 0 else float("inf")
        lines.append(f"### shape hidden={shape_h}: "
                     f"x=d/m = {x_hat:.1f} "
                     f"(d={d_hat:.3e} ms/tile, m={m_hat:.3e} ms/row), "
                     f"bound 128*(1+256/x) = {bound:.0f}")
        mism = []
        for r in cells:
            pred = d_hat * (r["b128"] - r["b256"]) + m_hat * (
                128 * r["b128"] - 256 * r["b256"])
            obs = r["gemm128"] - r["gemm256"]
            floor = NOISE * min(r["gemm128"], r["gemm256"])
            if abs(obs) <= floor or abs(pred) <= floor:
                continue
            if (pred > 0) != (obs > 0):
                mism.append((r, pred, obs))
        lines.append(f"- item 1 sign validity: {len(mism)} mismatches / "
                     f"{len(cells)} cells above the {NOISE:.0%} noise floor")
        for r, pred, obs in mism[:8]:
            lines.append(f"    - {r['family']} rho={r['rho']} E={r['E']}: "
                         f"pred {pred:+.4f} ms vs obs {obs:+.4f} ms")
        viol2 = [r for r in cells
                 if r["rho"] >= bound
                 and r["gemm256"] > r["gemm128"] * (1.0 + NOISE)]
        n_in = sum(1 for r in cells if r["rho"] >= bound)
        lines.append(f"- item 2 bound validity (rho >= {bound:.0f}): "
                     f"{len(viol2)} violations / {n_in} cells")
        for r in viol2[:8]:
            lines.append(f"    - {r['family']} rho={r['rho']} E={r['E']}: "
                         f"g128={r['gemm128']:.3f} g256={r['gemm256']:.3f}")
        lines.append("")

    lines.append("## Crossings per (shape, E, family) — isolated GEMM arms",
                 )
    lines.append("| shape | E | family | first win | re-losses after | last-loss rho | verdict@512 |")
    lines.append("|---|---|---|---|---|---|---|")
    for shape_h in sorted({r["hidden"] for r in good}):
        for e in EXPERTS:
            for fam in FAMILIES:
                pts = sorted(
                    ((r["rho"],
                      r["gemm128"] - r["gemm256"],
                      min(r["gemm128"], r["gemm256"]))
                     for r in good
                     if r["hidden"] == shape_h and r["E"] == e
                     and r["family"] == fam))
                if len(pts) < 3:
                    continue
                signs = [(rho, d, s) for rho, d, s in pts
                         if abs(d) > NOISE * s]
                first_win = next((rho for rho, d, _ in signs if d < 0), None)
                losses = ([rho for rho, d, _ in signs
                           if d > 0 and first_win is not None and rho > first_win])
                last_loss = losses[-1] if losses else None
                at512 = next((d for rho, d, _ in pts if rho == 512), None)
                verdict = ("?" if at512 is None
                           else ("wide-ok" if at512 <= 0 else "WIDE-LOSES"))
                lines.append(
                    f"| {shape_h} | {e} | {fam} | {first_win} | "
                    f"{losses or '-'} | {last_loss or '-'} | {verdict} |")

    lines.append("")
    lines.append("## End-to-end operator arms (production trim latch)")
    lines.append("| shape | E | family | rho | e2e128 ms | e2e256 ms | winner |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(good, key=lambda r: (r["hidden"], r["E"], r["family"],
                                         r["rho"])):
        win = ("256" if r["e2e256"] < r["e2e128"]
               else "128" if r["e2e128"] < r["e2e256"] else "tie")
        lines.append(
            f"| {r['hidden']} | {r['E']} | {r['family']} | {r['rho']} | "
            f"{r['e2e128']:.3f} | {r['e2e256']:.3f} | {win} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--json",
        default=os.environ.get(
            "PRISMAQUANT_SWEEP_STATE",
            "/home/rob/dq-runs/sweep-rho/rho_sweep_cells.json"),
        help="cell checkpoint; PRISMAQUANT_SWEEP_STATE overrides the default")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells already present in --json")
    opts = parser.parse_args()

    def save() -> None:
        tmp = opts.json + ".tmp"
        os.makedirs(os.path.dirname(opts.json), exist_ok=True)
        with open(tmp, "w") as fh:
            json.dump(records, fh, indent=1)
        os.replace(tmp, opts.json)

    records: list[dict] = []
    done: set[tuple] = set()
    if opts.resume and os.path.exists(opts.json):
        with open(opts.json) as fh:
            records = json.load(fh)
        done = {(r.get("hidden"), r.get("E"), r.get("family"), r.get("rho"))
                for r in records if not r.get("blocked")}
        print(f"resuming: {len(records)} cells already recorded")

    others = _other_compute_apps()
    if others:
        print("REFUSED: GPU not idle:", others[:5], file=sys.stderr)
        return 2

    import gridbook
    print(f"gridbook: {gridbook.__file__}")
    import prismaquant
    print(f"prismaquant: {prismaquant.__file__}")
    print(f"device: {torch.cuda.get_device_name()} "
          f"cc={torch.cuda.get_device_capability()} "
          f"sm={torch.cuda.get_device_properties(0).multi_processor_count} "
          f"clocks={_sm_clocks()}")

    smoke = opts.smoke
    warmup = opts.warmup if opts.warmup is not None else (4 if smoke else 20)
    reps = opts.reps if opts.reps is not None else (3 if smoke else 9)
    shapes = SHAPES[:1] if smoke else SHAPES
    es = (8,) if smoke else EXPERTS
    rhos = (96, 200) if smoke else RHOS
    fams = ("uniform",) if smoke else FAMILIES

    act = _silu_act()
    for name, hidden, inter in shapes:
        for e in es:
            print(f"\n=== building {name} hidden={hidden} inter={inter} "
                  f"E={e} ===", flush=True)
            method, layer = build_layer(experts=e, hidden=hidden, inter=inter)
            sizes = method._gf2_tile_sizes(layer)
            print(f"compiled TileM set: {sizes}", flush=True)
            if not smoke and not {128, 256} <= set(sizes):
                print(f"BLOCKED: TileM 256 not compiled for this rung/build "
                      f"(sizes={sizes}); wide-tile cell unmeasurable.",
                      file=sys.stderr)
                return 3
            for fam in fams:
                for rho in rhos:
                    if (hidden, e, fam, rho) in done:
                        print(f"  skip {fam} rho={rho} (recorded)",
                              flush=True)
                        continue
                    rec = run_cell(method, layer, act, hidden=hidden,
                                   family=fam, rho=rho, warmup=warmup,
                                   reps=reps)
                    if rec.get("blocked"):
                        print(f"  BLOCKED {rec['cell']}: {rec['blocked']}",
                              flush=True)
                    else:
                        rec["shape_name"] = name
                        records.append(rec)
                        save()
                        print(f"  {fam:11s} rho={rho:5d} "
                              f"B128={rec['b128']:5d} B256={rec['b256']:5d} "
                              f"q={rec['q']:4d} nhi={rec['n_hi']:4d} | "
                              f"gemm {rec['gemm128']:8.3f}/"
                              f"{rec['gemm256']:8.3f} ms "
                              f"({'256' if rec['gemm256'] < rec['gemm128'] else '128'})"
                              f" | e2e {rec['e2e128']:8.3f}/"
                              f"{rec['e2e256']:8.3f}"
                              f" | rel={rec['rel_final']:.2e}", flush=True)
            del method, layer
            torch.cuda.empty_cache()

    post = _other_compute_apps()
    save()
    if post:
        print("CONTENDED_AFTER_RUN: per-cell data saved but DISCARD timings:",
              post[:5], file=sys.stderr)
        return 2

    print(f"\ncells written: {opts.json}")
    if not smoke:
        print(analyse(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
