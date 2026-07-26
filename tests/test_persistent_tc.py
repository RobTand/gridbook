"""Parity + bench for the §4b persistent-N tensor-core prefill kernel.

Parity contract: D_unscaled from cb_prefill_persistent_tc must match
  ref = (A_f32 @ expand_cb_to_fp8(packed).f32.T)   (fp32 reference)
within fp8-MMA reassociation tolerance, and the DECODED resident tile is
bit-exact by construction (same decode as cb_expand_fp8 — cross-checked
here via the expander).

Run (GPU): PYTHONPATH=. pytest plugins/gridbook/tests/test_persistent_tc.py -v
Bench:     python plugins/gridbook/tests/test_persistent_tc.py bench
"""
import sys

import pytest
import torch

try:
    from gridbook.cuda_ext import get_ext, get_persistent_ext
except Exception:  # pragma: no cover
    get_ext = get_persistent_ext = lambda: None

cuda_ok = torch.cuda.is_available()


def _mk(N, K, kbits, M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tsize = 4 * kbits
    rb = K // 256 * tsize
    packed = torch.randint(0, 255, (N, rb), dtype=torch.uint8,
                           device="cuda", generator=g)
    # valid e4m3 values only — random BYTES would include NaN codes
    # (0x7f/0xff) and poison both the kernel and the reference equally.
    # Size by rung: 4 sub-tables x 2^(k/4) entries x 2 bytes (even splits).
    cb_bytes = 4 * (1 << (kbits // 4)) * 2
    cb = (torch.randn((cb_bytes,), device="cuda", generator=g) * 0.1).to(
        torch.float8_e4m3fn).view(torch.uint8)
    a = (torch.randn(M, K, device="cuda", generator=g) * 0.05).to(
        torch.float8_e4m3fn)
    return a, packed, cb, tsize


@pytest.mark.skipif(not cuda_ok, reason="needs CUDA")
@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("kbits", [36, 40, 44, 48])
@pytest.mark.parametrize("shape", [(256, 1024, 512), (2048, 3072, 1400),
                                   (192, 3072, 300)])
def test_persistent_tc_parity(kbits, shape, variant):
    N, K, M = shape
    ext, ptc = get_ext(), get_persistent_ext()
    if ext is None or ptc is None:
        pytest.skip("extensions unavailable")
    a, packed, cb, tsize = _mk(N, K, kbits, M)
    offs = torch.zeros(N, dtype=torch.int32, device="cuda")

    d = ptc.cb_prefill_persistent_tc(a, packed, cb, N, K, kbits, tsize,
                                     variant=variant)
    w = ext.cb_expand_fp8(packed, cb, offs, N, K, kbits, 4, tsize)
    ref = a.float() @ w.float().t()

    err = (d.float() - ref).abs()
    rel = err.max() / ref.abs().max().clamp_min(1e-6)
    # fp8 tensor-core accumulation reassociates vs the fp32 reference; the
    # bound is the fork-kernel parity bound from test_fused_prefill.
    assert rel < 2e-2, f"rel={rel:.3e}"


def bench():  # pragma: no cover
    ext, ptc = get_ext(), get_persistent_ext()
    assert ext is not None and ptc is not None
    import time
    for (N, K, kbits, M) in [(2048, 3072, 48, 1400), (2048, 3072, 48, 8192),
                             (9216, 3072, 48, 8192)]:
        a, packed, cb, tsize = _mk(N, K, kbits, M)
        offs = torch.zeros(N, dtype=torch.int32, device="cuda")

        def t(f, rep=5):
            f(); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(rep):
                f()
            torch.cuda.synchronize()
            return (time.time() - t0) / rep

        for variant in (1, 2):
            t_v = t(lambda: ptc.cb_prefill_persistent_tc(
                a, packed, cb, N, K, kbits, tsize, variant=variant))
            print(f"  v{variant}: {t_v*1e3:6.1f} ms")
        t_ptc = t(lambda: ptc.cb_prefill_persistent_tc(
            a, packed, cb, N, K, kbits, tsize))
        def serial():
            w = ext.cb_expand_fp8(packed, cb, offs, N, K, kbits, 4, tsize)
            return a.float() @ w.float().t()   # stand-in GEMM; fork ext is
                                               # the served comparator
        t_ser = t(serial)
        print(f"N={N} K={K} M={M} K{kbits}: persistent {t_ptc*1e3:6.1f} ms | "
              f"expand+matmul {t_ser*1e3:6.1f} ms")


if __name__ == "__main__" and "bench" in sys.argv:  # pragma: no cover
    bench()
