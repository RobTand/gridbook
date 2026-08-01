"""Microbench: native FP8-CB GEMV vs native expand+CUTLASS.

Both arms include activation quantization.  The comparator is Gridbook's
native transient expander followed by vLLM's CUTLASS scaled GEMM, i.e. the
fail-closed native fallback for shapes outside the direct decode kernel.
Run inside the serving container (needs nvcc):

  docker run --rm --gpus all -v /home/rob/gridbook:/gridbook \\
    --entrypoint bash vllm-node:latest -c \\
    'PYTHONPATH=/gridbook python3 /gridbook/tests/bench_cuda_gemv.py'

Reports us/call and effective packed-weight GB/s (the bandwidth-bound target:
GB10 LPDDR5X ~= 273 GB/s peak, ~230-250 achievable).
"""
import time

import torch

from gridbook import codec  # noqa: E402
from gridbook.cuda_ext import get_ext  # noqa: E402
from vllm import _custom_ops as vllm_ops  # noqa: E402

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
    total = ({("gemv", m): 0.0 for m in MS}
             | {("expand_cutlass", m): 0.0 for m in MS})
    for k in KS:
        print(f"\n=== FP8_CB_K{k} ===")
        print(f"{'shape':>10} {'M':>3} {'expand+CL us':>12} {'gemv us':>9} "
              f"{'speedup':>8} {'GB/s':>7}")
        for name, N, K in SHAPES:
            p = synth(k, N, K)
            gbytes = N * (K // 256) * p["ts"] / 1e9
            for M in MS:
                x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
                def gemv_fn():
                    ext.cb_gemv_fp8(x, p["qwp"], p["cb8"], p["row_off"],
                                    p["ws"], p["N"], p["K"], p["k"], 4,
                                    p["ts"], True)

                def expand_cutlass_fn():
                    xq, scale_a = vllm_ops.scaled_fp8_quant(
                        x, use_per_token_if_dynamic=True)
                    weight = ext.cb_expand_fp8(
                        p["qwp"], p["cb8"], p["row_off"], p["N"], p["K"],
                        p["k"], 4, p["ts"])
                    vllm_ops.cutlass_scaled_mm(
                        xq, weight.t(), scale_a, p["ws"].reshape(-1, 1),
                        torch.bfloat16, None)

                tg = timeit(gemv_fn)
                te = timeit(expand_cutlass_fn)
                total[("gemv", M)] += tg
                total[("expand_cutlass", M)] += te
                print(f"{name:>10} {M:>3} {te * 1e6:>12.1f} {tg * 1e6:>9.1f} "
                      f"{te / tg:>7.2f}x {gbytes / tg:>7.1f}")
    print("\nper-layer-set totals (sum of the 4 shapes, both rungs):")
    for M in MS:
        print(
            f"  M={M:>2}: expand+CUTLASS "
            f"{total[('expand_cutlass', M)] * 1e6:8.1f} us | native GEMV "
            f"{total[('gemv', M)] * 1e6:8.1f} us | "
            f"{total[('expand_cutlass', M)] / total[('gemv', M)]:5.2f}x")


if __name__ == "__main__":
    main()
