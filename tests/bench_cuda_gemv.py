"""Microbench: CUDA FP8_CB decode-GEMV vs the Triton decode-GEMM on real 27B
shapes. Run inside the serving container (needs nvcc):

  docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
    --entrypoint bash vllm-node:latest -c \\
    'PYTHONPATH=/repo:/repo/plugins/gridbook \\
     python3 /repo/plugins/gridbook/tests/bench_cuda_gemv.py'

Reports us/call and effective packed-weight GB/s (the bandwidth-bound target:
GB10 LPDDR5X ~= 273 GB/s peak, ~230-250 achievable).
"""
import sys
import time

import torch

sys.path.insert(0, "/repo/plugins/gridbook")
from gridbook import codec  # noqa: E402
from gridbook.cuda_ext import get_ext  # noqa: E402
from gridbook.kernels import cb_decode_linear  # noqa: E402

DEV = "cuda"
# (name, N, K) — Qwen3.6-27B: fused qkv, o_proj (24*256=6144), gate_up, down.
SHAPES = [
    ("qkv_proj", 8192, 5120),
    ("o_proj", 5120, 6144),
    ("gate_up", 34816, 5120),
    ("down_proj", 5120, 17408),
]
KS = [44, 48]
MS = [1, 2, 4, 8, 16]
ITERS = 50


def synth(k, N, K, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ts = 4 * k
    packed = torch.randint(0, 256, (N, (K // 256) * ts), generator=g,
                           dtype=torch.uint8).to(DEV)
    sub_w = k // 4
    subs = [(torch.randn(1 << sub_w, 2, generator=g) * 4.0)
            .to(torch.float8_e4m3fn).float().to(DEV) for _ in range(4)]
    ws = (torch.rand(N, generator=g).to(DEV) + 0.5) * 0.02
    cb_flat = codec.build_flat_codebook(subs)
    return dict(qwp=codec.pad_qweight(packed), cb_flat=cb_flat,
                cb8=cb_flat.to(torch.float8_e4m3fn).view(
                    torch.uint8).contiguous(),
                row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
                N=N, K=K, k=k, ts=ts, ws=ws.float())


def timeit(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ext = get_ext()
    assert ext is not None, "CUDA extension failed to build"
    print(f"device: {torch.cuda.get_device_name()}")
    total = {("cuda", m): 0.0 for m in MS} | {("triton", m): 0.0 for m in MS}
    for k in KS:
        print(f"\n=== FP8_CB_K{k} ===")
        print(f"{'shape':>10} {'M':>3} {'triton us':>10} {'cuda us':>9} "
              f"{'speedup':>8} {'GB/s':>7}")
        for name, N, K in SHAPES:
            p = synth(k, N, K)
            gbytes = N * (K // 256) * p["ts"] / 1e9
            for M in MS:
                x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
                xq = codec.fp8_dynamic_act_qdq(x)

                def cuda_fn():
                    ext.cb_gemv_fp8(x, p["qwp"], p["cb8"], p["row_off"],
                                    p["ws"], p["N"], p["K"], p["k"], 4,
                                    p["ts"], True)

                def triton_fn():
                    cb_decode_linear(xq, p["qwp"], p["cb_flat"], p["row_off"],
                                     p["ws"], torch.zeros(1, device=DEV),
                                     N=p["N"], K=p["K"], k_bits=p["k"],
                                     n_sub=4, type_size=p["ts"], is_fp4=False)

                tc = timeit(cuda_fn)
                tt = timeit(triton_fn)
                total[("cuda", M)] += tc
                total[("triton", M)] += tt
                print(f"{name:>10} {M:>3} {tt * 1e6:>10.1f} {tc * 1e6:>9.1f} "
                      f"{tt / tc:>7.2f}x {gbytes / tc:>7.1f}")
    print("\nper-layer-set totals (sum of the 4 shapes, both rungs):")
    for M in MS:
        print(f"  M={M:>2}: triton {total[('triton', M)] * 1e6:8.1f} us | "
              f"cuda {total[('cuda', M)] * 1e6:8.1f} us | "
              f"{total[('triton', M)] / total[('cuda', M)]:5.2f}x")


if __name__ == "__main__":
    main()
