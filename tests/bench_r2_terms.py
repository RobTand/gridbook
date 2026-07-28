"""TEMPORARY bench (R2 term decomposition). Not a test; not imported by any
python serving path. Run standalone in a container. Delete when done.

The decode-ALU instrument this bench needs (an #ifdef PQ_BENCH_DECODE_NOALU
block in gridbook/csrc/cutlass_fork/sm120_cb_fused_mma.hpp) is deliberately NOT in the
tree: a wrong-numerics ifdef inside the shipping decode mainloop is a landmine
for the next reader. Re-apply it with
    git apply /home/rob/dq-runs/r2-bench/noalu_instrument.patch
and revert with git checkout when the bench is done.

Arms:
  A2  decode term: cb_fused_prefill_mm(_scaled) vs sm120_fp8_mm_fork64 at
      M in {128,256,384,512} on one expert's shape.
  A3  ALU/TMA split: same, against a SECOND extension compiled with
      -DPQ_BENCH_DECODE_NOALU (wrong numerics by design).
  A1  pad term: grouped kernel, tiles-per-expert sweep.
  A4  launch term: grouped 1 launch vs per-expert loop 2*E launches.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridbook import codec  # noqa: E402

DEV = "cuda"
from gridbook.cuda_ext import _find_cutlass_include, csrc_dir  # noqa: E402

SRC = csrc_dir()


def build(noalu: bool):
    from torch.utils.cpp_extension import load

    # Resolved here, not at module scope: _find_cutlass_include() does
    # `import vllm`, and this module must stay importable (and --help-able)
    # without vLLM installed, as it was when this was a literal path constant.
    CUT_INC = _find_cutlass_include()       # same discovery the plugin uses
    CUT = os.path.dirname(CUT_INC)
    name = "pq_cb_fused_noalu" if noalu else "pq_cb_fused_bench"
    bd = os.path.join(os.path.expanduser("~"), ".cache", "pq-r2-bench", name)
    os.makedirs(bd, exist_ok=True)
    flags = ["-O3", "--expt-relaxed-constexpr"]
    if noalu:
        flags.append("-DPQ_BENCH_DECODE_NOALU")
    return load(name=name, sources=[os.path.join(SRC, "cb_fused_gemm.cu")],
                extra_include_paths=[CUT_INC,
                                     os.path.join(CUT, "tools", "util",
                                                  "include"), SRC],
                extra_cuda_cflags=flags, build_directory=bd, verbose=False)


def synth(k, N, K, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ts = 4 * k
    packed = torch.randint(0, 256, (N, (K // 256) * ts), generator=g,
                           dtype=torch.uint8).to(DEV)
    subs = [(torch.randn(1 << (k // 4), 2, generator=g) * 4.0)
            .to(torch.float8_e4m3fn).float().to(DEV) for _ in range(4)]
    cb8 = codec.build_flat_codebook(subs).to(torch.float8_e4m3fn).view(
        torch.uint8).contiguous()
    return packed, cb8


def timeit(fn, warmup=5, iters=20, label=""):
    """ms per call. JIT build + first-call autotune excluded by warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3 / iters
        best = dt if best is None else min(best, dt)
    return best


def arm_decode(ext, ext_noalu, N, K, k, Ms, tag):
    packed, cb8 = synth(k, N, K, seed=k)
    from gridbook.cuda_ext import get_ext
    gemv = get_ext()
    qwp = codec.pad_qweight(packed)
    off = torch.zeros(N, dtype=torch.int32, device=DEV)
    W = gemv.cb_expand_fp8(qwp, cb8, off, N, K, k, 4, 4 * k)
    bs = torch.rand(N, device=DEV, dtype=torch.float32) + 0.5
    rows = []
    for M in Ms:
        torch.manual_seed(1)
        xq = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
        asc = torch.rand(M, device=DEV, dtype=torch.float32) + 0.5
        t_fork = timeit(lambda: ext.sm120_fp8_mm_fork64(xq, W))
        t_fu = timeit(lambda: ext.cb_fused_prefill_mm(xq, packed, cb8, N, K, k))
        t_fus = timeit(lambda: ext.cb_fused_prefill_mm_scaled(
            xq, packed, cb8, asc, bs, N, K, k))
        t_na = float("nan")
        if ext_noalu is not None:
            t_na = timeit(lambda: ext_noalu.cb_fused_prefill_mm(
                xq, packed, cb8, N, K, k))
        rows.append((M, t_fork, t_fu, t_fus, t_na))
        print(f"[A2/{tag}] M={M:4d} fork={t_fork:8.3f} fused={t_fu:8.3f} "
              f"fused_scaled={t_fus:8.3f} noalu={t_na:8.3f} "
              f"decode={t_fu - t_fork:8.3f}", flush=True)
    return rows


def make_grouped(E, N, K, k, tiles, tile_m):
    packed1, cb8 = synth(k, N, K, seed=k + 1)
    packed = packed1.unsqueeze(0).expand(E, -1, -1).contiguous()
    Mp = E * tiles * tile_m
    torch.manual_seed(2)
    a = (torch.randn(Mp, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    asc = torch.rand(Mp, device=DEV, dtype=torch.float32) + 0.5
    bsc = (torch.rand(E, N, device=DEV, dtype=torch.float32) + 0.5)
    eids = torch.arange(E, device=DEV, dtype=torch.int32).repeat_interleave(
        tiles).contiguous()
    return a, packed, cb8, asc, bsc, eids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A2,A3,A1,A4")
    ap.add_argument("--E-large", type=int, default=64)
    ap.add_argument("--E-small", type=int, default=64)
    ap.add_argument("--noalu", action="store_true")
    args = ap.parse_args()
    arms = set(args.arms.split(","))

    print("building ext...", flush=True)
    t0 = time.time()
    ext = build(False)
    print(f"built in {time.time() - t0:.0f}s", flush=True)
    ext_noalu = None
    if "A3" in arms:
        t0 = time.time()
        ext_noalu = build(True)
        print(f"built noalu in {time.time() - t0:.0f}s", flush=True)

    LARGE = dict(N=2048, K=3072)
    SMALL = dict(N=1024, K=2048)

    if "A2" in arms:
        for k in (44, 48):
            arm_decode(ext, ext_noalu, LARGE["N"], LARGE["K"], k,
                       [128, 256, 384, 512], f"large-k{k}")
        arm_decode(ext, ext_noalu, SMALL["N"], SMALL["K"], 28,
                   [128, 256, 384, 512], "small-k28")

    if "A1" in arms:
        for (tag, cfg, k, E, tms) in (
                ("large-k44", LARGE, 44, args.E_large, [128]),
                ("large-k48", LARGE, 48, args.E_large, [128]),
                ("small-k28", SMALL, 28, args.E_small, [128, 256])):
            for tm in tms:
                for tiles in (1, 2, 3):
                    a, packed, cb8, asc, bsc, eids = make_grouped(
                        E, cfg["N"], cfg["K"], k, tiles, tm)
                    t = timeit(lambda: ext.cb_fused_moe_grouped(
                        a, packed, cb8, asc, bsc, eids, cfg["N"], cfg["K"],
                        k, tm), warmup=3, iters=10)
                    print(f"[A1/{tag}] tile_m={tm} tiles/e={tiles} E={E} "
                          f"Mp={a.shape[0]} t={t:9.3f} ms  "
                          f"per-tile={t / (E * tiles):8.4f}", flush=True)
                    del a, packed, bsc, asc, eids
                    torch.cuda.empty_cache()

    if "A5" in arms:
        arm_skew(ext)

    if "A4" in arms:
        # grouped (1 launch) vs per-expert single-problem loop (E launches).
        for (tag, cfg, k, E, tiles) in (
                ("large-k44", LARGE, 44, args.E_large, 3),
                ("small-k28", SMALL, 28, args.E_small, 1)):
            tm = 128
            a, packed, cb8, asc, bsc, eids = make_grouped(
                E, cfg["N"], cfg["K"], k, tiles, tm)
            N, K = cfg["N"], cfg["K"]
            tg = timeit(lambda: ext.cb_fused_moe_grouped(
                a, packed, cb8, asc, bsc, eids, N, K, k, tm),
                warmup=3, iters=10)
            m_e = tiles * tm
            p0 = packed[0].contiguous()

            def loop():
                for e in range(E):
                    ext.cb_fused_prefill_mm_scaled(
                        a[e * m_e:(e + 1) * m_e], p0, cb8,
                        asc[e * m_e:(e + 1) * m_e], bsc[e], N, K, k)
            tl = timeit(loop, warmup=3, iters=5)
            print(f"[A4/{tag}] E={E} m_e={m_e} grouped={tg:9.3f} "
                  f"loop({E} launches)={tl:9.3f} delta={tl - tg:9.3f} "
                  f"per-launch={(tl - tg) / E * 1e3:7.2f} us", flush=True)
            del a, packed, bsc, asc, eids
            torch.cuda.empty_cache()




def arm_skew(ext, E=64, N=2048, K=3072, k=44, tile_m=128, m_bar=320):
    """A5: same TOTAL rows, uniform m_e vs a lognormal-ish skewed draw.
    Only the tile COUNT sum(ceil(m_e/TileM)) should matter."""
    import math
    g = torch.Generator().manual_seed(11)
    w = torch.distributions.LogNormal(0.0, 0.9).sample((E,), ).numpy() \
        if False else torch.exp(torch.randn(E, generator=g) * 0.9).numpy()
    tot = E * m_bar
    m = [max(1, int(round(tot * x / w.sum()))) for x in w]
    tiles = [math.ceil(x / tile_m) for x in m]
    unif_tiles = [math.ceil(m_bar / tile_m)] * E
    packed1, cb8 = synth(k, N, K, seed=k + 1)
    packed = packed1.unsqueeze(0).expand(E, -1, -1).contiguous()
    bsc = torch.rand(E, N, device=DEV, dtype=torch.float32) + 0.5
    for tag, tl in (("uniform", unif_tiles), ("skewed", tiles)):
        Mp = sum(tl) * tile_m
        a = (torch.randn(Mp, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
        asc = torch.rand(Mp, device=DEV, dtype=torch.float32) + 0.5
        eids = torch.tensor(
            sum(([e] * t for e, t in enumerate(tl)), []),
            device=DEV, dtype=torch.int32).contiguous()
        t = timeit(lambda: ext.cb_fused_moe_grouped(
            a, packed, cb8, asc, bsc, eids, N, K, k, tile_m),
            warmup=3, iters=10)
        used = sum(m) if tag == "skewed" else E * m_bar
        print(f"[A5/{tag}] tiles={sum(tl)} Mp={Mp} useful_rows={used} "
              f"pad_frac={(Mp - used) / Mp:.3f} t={t:8.3f} ms "
              f"per-tile={t / sum(tl):7.4f}", flush=True)
        del a, asc, eids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
