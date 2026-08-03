"""The pending NATIVE-PARITY timing half of the MXFP8 dense lane.

Run on an IDLE GPU (the harness refuses otherwise — a timing number taken
against a contended GPU is invalid and must not stand):

    flock /tmp/claude-1000/gpu-bench.lock \
        python -m gridbook.bench_mxfp8_dense

Correctness parity is already audited (see ``source_passthrough.py``); this
bench supplies the serve-side timing evidence the OPT-IN promotion note in
``docs/PLUGIN.md`` (``GRIDBOOK_MXFP8_DENSE``) says is pending: kernel vs the
native BF16 route at the real DSV4-Flash body shapes, plus the end-to-end
path including dynamic activation quantization, which is the number a serve
decision actually rides on.
"""
from __future__ import annotations

import subprocess
import sys
import time


def _other_compute_apps() -> list[str]:
    """Compute apps on the GPU that are not this process.

    The post-run check runs while this process still holds a CUDA context, so
    its own pid must not count as contention; an unparseable pid line is kept
    (unknown is contention, not a pass).
    """
    import os

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 — no nvidia-smi means we cannot attest
        return ["<nvidia-smi unavailable: cannot attest an idle GPU>"]
    me = str(os.getpid())
    lines = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid = line.split(",", 1)[0].strip()
        if pid == me:
            continue
        lines.append(line)
    return lines


def main() -> int:
    import torch

    from . import cuda_ext as ce
    from .mxfp8 import fill_sf_plane, quantize_mxfp8

    others = _other_compute_apps()
    if others:
        print("REFUSED: the GPU is not idle; a timing number taken under "
              "contention is invalid. Live compute apps:", file=sys.stderr)
        for line in others[:5]:
            print(f"  {line}", file=sys.stderr)
        return 2

    ext = ce.get_mxfp8_dense_ext()
    if ext is None:
        print("REFUSED: MXFP8 dense extension unavailable (see stderr above).",
              file=sys.stderr)
        return 2
    torch.manual_seed(0)

    def bench(fn, iters=30, warmup=8):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    # The distinct DSV4-Flash body (N, K) shapes, benched at decode and
    # prefill M.
    shapes = [(1, 8192, 4096), (1, 32768, 1024),
              (512, 8192, 4096), (512, 32768, 1024), (512, 4096, 8192),
              (2048, 8192, 4096)]
    print(f"device: {torch.cuda.get_device_name()} "
          f"capability {torch.cuda.get_device_capability()}")
    for m, n, k in shapes:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        a_q, a_sf = quantize_mxfp8(a)
        b_q, b_sf = quantize_mxfp8(w)
        offa = ext.mxfp8_sf_offsets(m, k, False).cuda()
        offb = ext.mxfp8_sf_offsets(n, k, True).cuda()
        pa = fill_sf_plane(a_sf, offa, int(ext.mxfp8_sf_plane_numel(m, k)))
        pb = fill_sf_plane(b_sf, offb, int(ext.mxfp8_sf_plane_numel(n, k)))
        flops = 2.0 * m * n * k

        t_kernel = bench(lambda: ext.mxfp8_dense_mm(a_q, pa, b_q, pb))
        t_bf16 = bench(lambda: a @ w.t())

        def e2e():
            q, sf = quantize_mxfp8(a)
            p = fill_sf_plane(sf, offa, int(ext.mxfp8_sf_plane_numel(m, k)))
            return ext.mxfp8_dense_mm(q, p, b_q, pb)

        t_e2e = bench(e2e, iters=15)
        print(f"M={m:5d} N={n:6d} K={k:5d}: "
              f"kernel {t_kernel*1e3:7.3f} ms ({flops/t_kernel/1e12:6.2f} TF)"
              f" | bf16 mm {t_bf16*1e3:7.3f} ms ({flops/t_bf16/1e12:6.2f} TF)"
              f" | e2e+quant {t_e2e*1e3:7.3f} ms"
              f" | kernel/bf16 {t_bf16/t_kernel:5.2f}x")

    others = _other_compute_apps()
    if others:
        print("CONTENDED_AFTER_RUN: another compute app appeared during the "
              "bench; DISCARD these numbers.", file=sys.stderr)
        return 2
    print("BENCH_OK (GPU idle before and after)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
