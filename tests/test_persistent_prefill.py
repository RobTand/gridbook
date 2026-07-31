"""Parity gate for the persistent-N FP8_CB prefill reference kernel
(gridbook/csrc/cb_persistent_prefill.cu). See docs/lanes/nvfp4-cb/persistent-n-prefill.md.

Container-only (JIT-builds an isolated extension; needs nvcc):

  docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
    --entrypoint bash vllm-node:latest -c \\
    'pip install -q pytest; PYTHONPATH=/repo:/repo/plugins/gridbook \\
     python3 -m pytest /repo/plugins/gridbook/tests/test_persistent_prefill.py -v'

Contract: cb_prefill_persistent_n_fp8 decodes B bit-exactly to cb_expand_fp8 and
accumulates the GEMM in f32, so its UNSCALED bf16 output matches
(cb_expand_fp8 -> f32 matmul -> bf16) to within the reassociation tolerance
(<=1 bf16 output ULP + a norm backstop) — the f32 sum order differs from the
per-row dot. Shapes are kept modest: the reference kernel is an f32-FMA
schedule/INV-1 validator, not a large-M perf path.
"""
import os

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")
from gridbook.cuda_ext import csrc_dir, get_ext  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

gemv_ext = get_ext()
if gemv_ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)", allow_module_level=True)


def _build():
    from torch.utils.cpp_extension import load
    build = os.path.join(os.path.expanduser("~"), ".cache", "pq-persistent-build")
    os.makedirs(build, exist_ok=True)
    src = os.path.join(csrc_dir(), "cb_persistent_prefill.cu")
    return load(name="pq_cb_persistent", sources=[src],
                extra_cuda_cflags=["-O3"], build_directory=build, verbose=False)


try:
    ext = _build()
except Exception as exc:  # pragma: no cover - env dependent
    pytest.skip(f"persistent ext build failed: {exc}", allow_module_level=True)

DEV = "cuda"


def _assert_close(y, ref, tag):
    a, b = y.float(), ref.float()
    d = (a - b).abs()
    tol = torch.maximum(a.abs(), b.abs()) * 2.0 ** -7 + 1e-5
    nbad = int((d > tol).sum())
    assert nbad == 0, (f"{tag}: {nbad} elems beyond 1 bf16 ULP "
                       f"(max d {d.max():.3e})")
    rel = (d.norm() / b.norm().clamp_min(1e-6)).item()
    assert rel <= 1e-3, f"{tag}: norm backstop rel {rel:.3e}"


def _synth(k, N, K, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ts = 4 * k
    packed = torch.randint(0, 256, (N, (K // 256) * ts), generator=g,
                           dtype=torch.uint8).to(DEV)
    subs = [(torch.randn(1 << (k // 4), 2, generator=g) * 4.0)
            .to(torch.float8_e4m3fn).float().to(DEV) for _ in range(4)]
    cb8 = codec.build_flat_codebook(subs).to(torch.float8_e4m3fn).view(
        torch.uint8).contiguous()
    return packed, cb8, ts


def _ref_and_got(packed, cb8, N, K, k, M, seed=1):
    ts = 4 * k
    qwp = codec.pad_qweight(packed)
    off = torch.zeros(N, dtype=torch.int32, device=DEV)
    W = gemv_ext.cb_expand_fp8(qwp, cb8, off, N, K, k, 4, ts)     # [N,K] e4m3
    torch.manual_seed(seed)
    a = (torch.randn(M, K, device=DEV) * 0.1).to(torch.float8_e4m3fn)
    ref = (a.float() @ W.float().t()).to(torch.bfloat16)          # UNSCALED
    got = ext.cb_prefill_persistent_n_fp8(a, qwp, cb8, N, K, k)
    return ref, got


@pytest.mark.parametrize("k", [36, 40, 44, 48])
def test_persistent_parity_rungs(k):
    packed, cb8, _ = _synth(k, N=64, K=1024, seed=k)
    ref, got = _ref_and_got(packed, cb8, 64, 1024, k, M=128)
    _assert_close(got, ref, f"persistent k={k}")


def test_persistent_parity_ragged():
    # N not a multiple of TILE_N(=8), M not a multiple of the block (256).
    packed, cb8, _ = _synth(44, N=70, K=1536, seed=7)
    ref, got = _ref_and_got(packed, cb8, 70, 1536, 44, M=100)
    _assert_close(got, ref, "persistent ragged")


def test_persistent_parity_wideN():
    packed, cb8, _ = _synth(40, N=256, K=1024, seed=3)
    ref, got = _ref_and_got(packed, cb8, 256, 1024, 40, M=64)
    _assert_close(got, ref, "persistent wideN")
