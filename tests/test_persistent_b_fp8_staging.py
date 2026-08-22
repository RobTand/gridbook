"""Staging gates for the FP8-CB persistent-B mainloop (ox/pb-salvage-s3).

The S3 change vectorizes the packed-superblock staging for the FP8 family
ONLY (u32 words when the source plane is word-aligned, the baseline byte loop
otherwise, then the baseline zero pass).  These tests gate that copy with
``torch.equal`` at every shipped FP8-CB rung k in {28, 36, 44, 48}:

* the decode probe is bit-identical to the pure-Torch reference (the suite's
  standard decode-identity pattern, extended to all four shipped rungs);
* a one-hot readout drives ``cb_moe_persistent_b_prefill_fp8`` — the binding
  whose mainloop contains the changed staging — with exactly one unit
  activation per routed row, so every output element is ONE exact product of
  a bf16 weight and equals the reference-decoded weight BITWISE, whatever the
  kernel's accumulation order.  A full K-column sweep per expert reads out
  every decoded weight of a fully staged superblock plane, so any staging
  byte error on either the aligned path or the misaligned fallback (an
  odd-byte-offset qw view) fails here.

Container-only, like the rest of the native-kernel suite: skips cleanly
without CUDA or off the cc 12.0/12.1 devices the lane is compiled for.
"""
from __future__ import annotations

import pytest
import torch

codec = pytest.importorskip("gridbook.codec")

from cb_torch_reference import reconstruct_cb_weight  # noqa: E402
from gridbook.cuda_ext import get_moe_persistent_b_ext  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

ext = get_moe_persistent_b_ext()
if ext is None:
    pytest.skip("the persistent-B grouped MoE extension is unavailable "
                "(needs cc 12.0/12.1 + nvcc)", allow_module_level=True)

DEV = torch.device("cuda")

# The shipped FP8-CB rungs: the ceil-first 4-way split changes shape at every
# k mod 4; k=28 is DSv4's shipping rung, k=48 the format ceiling.
_RUNGS = [28, 36, 44, 48]


def _fp8_lut_elems(k: int) -> int:
    base, extra = divmod(k, 4)
    return sum(2 << (base + (1 if i < extra else 0)) for i in range(4))


def _pack_fp8(k: int, K: int, E: int, N: int, seed: int):
    """``(qw [E,N,row_bytes], lut_f32, scale [E,N], type_size)``."""
    type_size = 4 * k
    g = torch.Generator(device="cpu").manual_seed(seed)
    lut_u8 = ((torch.randn(_fp8_lut_elems(k), generator=g) * 0.5)
              .to(torch.float8_e4m3fn).view(torch.uint8).to(DEV))
    lut_f32 = lut_u8.view(torch.float8_e4m3fn).float().contiguous()
    row_bytes = (K // codec.SUPERBLOCK) * type_size
    qw = torch.randint(0, 256, (E, N, row_bytes), dtype=torch.uint8,
                       generator=g).to(DEV).contiguous()
    scale = ((torch.rand(E, N, generator=g) * 1.5 + 0.25) * 0.01) \
        .float().to(DEV).contiguous()
    return qw, lut_f32, scale, type_size


def _torch_reference_decode(qw, lut_f32, scale, K, k, type_size):
    """The pure-Torch serving chain: values(f32) * row_scale -> bf16."""
    E, N, _ = qw.shape
    rows = E * N
    return reconstruct_cb_weight(
        qw.reshape(rows, -1), lut_f32,
        torch.zeros(rows, dtype=torch.int32, device=DEV),
        scale.reshape(rows), torch.zeros(1, device=DEV),
        N=rows, K=K, k_bits=k, n_sub=4, type_size=type_size,
        is_fp4=False)


@pytest.mark.parametrize("k", _RUNGS, ids=[f"k{k}" for k in _RUNGS])
def test_decode_probe_is_bit_identical_to_the_torch_reference_at_shipped_rungs(
        k):
    K, E, N = 512, 2, 24
    qw, lut_f32, scale, type_size = _pack_fp8(k, K, E, N, seed=k * 7919)

    got = ext.cb_moe_persistent_b_decode_fp8(
        qw.reshape(-1), lut_f32, scale.reshape(-1), 0, E * N,
        K, k, type_size)
    want = _torch_reference_decode(qw, lut_f32, scale, K, k, type_size)

    assert got.shape == (E * N, K)
    assert got.dtype is torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("k", _RUNGS, ids=[f"k{k}" for k in _RUNGS])
@pytest.mark.parametrize("misaligned", [False, True],
                         ids=["aligned-src", "misaligned-src"])
def test_prefill_one_hot_readout_is_bitwise_the_torch_reference(
        k, misaligned):
    """Every routed row carries ONE unit activation: y[p, n] is a single
    exact product, so the whole staged/decoded plane is read out bitwise
    through the mainloop under test.  ``misaligned`` offsets the qw storage
    by one byte so the staging falls back to the baseline byte loop."""
    # K = one superblock: n_sb == 1, so each row slot is staged exactly once.
    K, E, N = codec.SUPERBLOCK, 2, 32
    qw, lut_f32, scale, type_size = _pack_fp8(k, K, E, N,
                                              seed=k * 104729 + 1)
    if misaligned:
        g = torch.Generator(device="cpu").manual_seed(k + 555)
        pad = torch.randint(0, 256, (1,), dtype=torch.uint8,
                            generator=g).to(DEV)
        flat = torch.cat([pad, qw.reshape(-1)])
        qw = flat[1:].view(E, N, (K // codec.SUPERBLOCK) * type_size)

    want = _torch_reference_decode(qw, lut_f32, scale, K, k,
                                   type_size).view(E, N, K)

    P = E * K
    ends = torch.arange(1, E + 1, dtype=torch.int32,
                        device=DEV) * K
    a = torch.zeros(P, K, dtype=torch.bfloat16, device=DEV)
    a[torch.arange(P, device=DEV), torch.arange(P, device=DEV) % K] = 1.0
    out = torch.empty(P, N, dtype=torch.bfloat16, device=DEV)
    ext.cb_moe_persistent_b_prefill_fp8(
        out, a, qw, lut_f32, scale, ends, k, type_size, 0)

    # Row r probes expert r//K at column r%K; y[r, n] must equal that
    # expert's reference-decoded weight [n, r%K] exactly.  Signed zero is the
    # one sanctioned difference: the epilogue's fp32 accumulation turns a
    # -0.0 weight (e4m3 subnormal-rounding) into +0.0 (IEEE +0 + -0 = +0),
    # while every non-zero weight survives the one-product reduction
    # bit-exactly.
    expected = torch.cat([want[e].t() for e in range(E)])
    assert out.shape == expected.shape
    out_bits = out.view(torch.int16)
    want_bits = expected.view(torch.int16)
    signed_zero_ok = (out == 0) & (expected == 0)
    mismatch = (out_bits != want_bits) & ~signed_zero_ok
    if bool(mismatch.any()):
        bad = mismatch.nonzero()[:8]
        probe = ", ".join(
            f"[{r},{n}] out={out[r, n].item():.6e} "
            f"want={expected[r, n].item():.6e}"
            for r, n in bad.tolist())
        pytest.fail(
            f"staged-plane readout diverges from the Torch reference at "
            f"k={k} ({'misaligned' if misaligned else 'aligned'} src): "
            f"nbad={int(mismatch.sum())} of {mismatch.numel()} "
            f"(signed-zero-only pairs: "
            f"{int(signed_zero_ok.sum())}); first: {probe}")
