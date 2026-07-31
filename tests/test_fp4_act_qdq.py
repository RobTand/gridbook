"""Bit-exactness battery for the fp4 group-16 activation-QDQ CUDA op.

The contract is EQUALITY, not tolerance. ``gridbook/linear.py`` and
``gridbook/moe.py`` both document that the fp4 act-QDQ deliberately runs OUTSIDE
the CB kernel so CUDA-vs-Triton numerics stay aligned; a tolerance here would
silently break that alignment and the divergence would only surface as a quality
delta nobody could attribute. Finite outputs are compared through an integer
view of the raw bf16 bits, including the sign bit of zero.

The eager codec is the oracle. It is still live at the per-expert reference loop
(``moe._apply_prefill_loop``), the batched prefill (``moe._apply_prefill_batched``)
and the Triton ``cb_gemm`` path (``linear._apply_inline``) — this change
deliberately did NOT rewrite those — so the oracle is an independent
implementation, not the thing under test wearing a hat.

Entirely CUDA-gated: with no extension build (no nvcc / no GPU) the whole module
skips, in line with the suite's runtime self-selection convention
(.github/scripts/run_cpu_tests.sh, "WHY NO PYTEST MARKERS").
"""
import time

import pytest
import torch

codec = pytest.importorskip("gridbook.codec",
                            reason="gridbook plugin not importable")
from gridbook.cuda_ext import get_ext  # noqa: E402

ext = get_ext()
if ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)",
                allow_module_level=True)
if not hasattr(ext, "fp4_act_qdq"):
    pytest.skip("extension build predates fp4_act_qdq — delete the JIT build "
                "dir (PRISMAQUANT_CB_EXT_DIR or ~/.cache/prismaquant-cb-ext) "
                "and re-import so the new cb_gemv.cu is compiled",
                allow_module_level=True)

DEV = "cuda"

# 3072 / 1024 / 12288 / 1536 are shipped hidden, moe-intermediate, dense-MLP and
# HY3-intermediate widths; the rest are awkward-but-legal K and M.
SHAPES = [(1, 3072), (1, 1024), (1, 12288), (1, 1536), (1, 16),
          (2, 3072), (8, 3072), (64, 1024), (313, 3072), (1024, 1024),
          (7, 48), (1, 128)]


def _inputs(m, k, gen):
    base = torch.randn(m, k, device=DEV, dtype=torch.bfloat16, generator=gen)
    yield "randn", base
    yield "randn*1e-4", base * 1e-4
    yield "randn*1e+4", base * 1e4
    yield "zeros", torch.zeros(m, k, device=DEV, dtype=torch.bfloat16)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _bit_equal(a, b):
    return torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize("m,k", SHAPES, ids=lambda v: str(v))
def test_bit_exact_vs_codec(m, k):
    """CUDA op == codec.fp4_group16_act_qdq, bit for bf16 bit."""
    gen = torch.Generator(device=DEV).manual_seed(1234)
    for name, x in _inputs(m, k, gen):
        ref = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
        got = ext.fp4_act_qdq(x)
        if not _bit_equal(ref, got):
            bad = _bits(ref) != _bits(got)
            n = int(bad.sum())
            i = int(bad.flatten().nonzero()[0])
            pytest.fail(
                f"M={m} K={k} {name}: {n}/{ref.numel()} elements differ; "
                f"first at flat index {i}: ref={ref.flatten()[i].item()} "
                f"got={got.flatten()[i].item()} x={x.flatten()[i].item()}")


def test_exact_midpoints_resolve_low():
    """Exact E2M1 midpoints must round to the LOWER rung.

    ``codec.fp4_group16_act_qdq`` picks ``hi`` only on a STRICT ``<``
    (codec.py, ``torch.where((hi - xg).abs() < (xg - lo).abs(), hi, lo)``), so an
    exact tie falls through to ``lo``. Adjacent E2M1 rungs differ by up to 2x, so
    getting this backwards is a 2x error on the affected elements — and it is
    invisible to any tolerance-based test.

    amax is 6.0 here, so scale == 1.0 and the listed values ARE the quotients.
    """
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
            -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 6.0, 0.0]
    x = torch.tensor([ties], device=DEV, dtype=torch.bfloat16)
    ref = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    got = ext.fp4_act_qdq(x)
    assert _bit_equal(ref, got), (
        f"midpoint tie-break diverged\n  ref={ref.flatten().tolist()}\n"
        f"  got={got.flatten().tolist()}")


def test_reciprocal_scale_midpoint_case():
    """The `x == +/-amax/8` case that distinguishes reciprocal-multiply from divide.

    ``codec.py`` writes ``amax.clamp_min(1e-8) / NVFP4_GRID_MAX``; torch lowers
    tensor-by-SCALAR division to a multiply by the float32 reciprocal, which for
    1/6 rounds one ulp HIGH. A kernel using a correctly-rounded ``__fdiv_rn``
    lands one ulp LOW, and at ``x == +/-amax/8`` the exact quotient is the
    midpoint -0.75, so the two implementations pick rungs that differ by exactly
    2x. This test pins that case directly rather than hoping randn hits it.
    """
    amax = 1.7265625                      # exactly representable in bf16
    row = [amax] + [-amax / 8.0] * 15
    x = torch.tensor([row], device=DEV, dtype=torch.bfloat16)
    ref = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    got = ext.fp4_act_qdq(x)
    assert _bit_equal(ref, got), (
        f"reciprocal-vs-divide divergence\n  ref={ref.flatten().tolist()}\n"
        f"  got={got.flatten().tolist()}")


def test_nonfinite_groups_match_codec_classification():
    """NaN propagation must match ``torch.amax`` group semantics.

    CUDA ``fmaxf`` ignores a lone NaN, whereas the codec's ``torch.amax``
    propagates it.  Without an explicit group flag the fused path would return
    plausible finite values for a group the reference turns entirely into NaN.
    NaN payload bits are not part of the contract, but the value classes and
    all finite outputs are.
    """
    rows = torch.tensor([
        [float("nan"), 1.0] + [0.0] * 14,
        [float("inf"), -1.0] + [0.0] * 14,
        [float("-inf"), 1.0] + [0.0] * 14,
    ], device=DEV, dtype=torch.bfloat16)
    ref = codec.fp4_group16_act_qdq(rows).to(torch.bfloat16)
    got = ext.fp4_act_qdq(rows)
    assert torch.equal(torch.isnan(ref), torch.isnan(got))
    assert torch.equal(torch.isposinf(ref), torch.isposinf(got))
    assert torch.equal(torch.isneginf(ref), torch.isneginf(got))
    finite = torch.isfinite(ref)
    assert torch.equal(ref[finite], got[finite])


def test_signed_zero_and_bf16_subnormal_bits():
    # Construct +/- minimum bf16 subnormals by bit pattern so the test does not
    # depend on a host float conversion preserving them.
    raw = torch.tensor([0, -32768, 1, -32767] * 4, dtype=torch.int16)
    x = raw.view(torch.bfloat16).reshape(1, 16).to(DEV)
    ref = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    got = ext.fp4_act_qdq(x)
    assert _bit_equal(ref, got)


def test_noncontiguous_higher_rank_input_returns_contiguous_output():
    x = torch.randn(2, 48, 3, device=DEV,
                    dtype=torch.bfloat16).transpose(1, 2)
    assert not x.is_contiguous() and x.shape == (2, 3, 48)
    ref = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    got = ext.fp4_act_qdq(x)
    assert got.is_contiguous()
    assert _bit_equal(ref, got)


def test_refuses_k_not_multiple_of_group():
    """A last dim that is not a multiple of 16 must raise, not silently corrupt."""
    x = torch.randn(1, 30, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiple of the fp4 group"):
        ext.fp4_act_qdq(x)


def test_refuses_non_bf16():
    x = torch.randn(1, 32, device=DEV, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA bf16"):
        ext.fp4_act_qdq(x)


def test_refuses_zero_last_dim_without_dividing_by_zero():
    x = torch.empty(2, 0, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="positive last dimension"):
        ext.fp4_act_qdq(x)


def test_accepts_empty_leading_dimension():
    x = torch.empty(0, 32, device=DEV, dtype=torch.bfloat16)
    got = ext.fp4_act_qdq(x)
    assert got.shape == x.shape and got.dtype == x.dtype and got.numel() == 0


def test_fp8_twin_rejects_zero_last_dim_without_dividing_by_zero():
    x = torch.empty(2, 0, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="positive last dimension"):
        ext.fp8_act_qdq(x)


def test_resolver_matches_codec():
    """``ops.fp4_act_qdq_or_codec`` is bit-identical on both of its branches."""
    from gridbook import ops
    x = torch.randn(4, 3072, device=DEV, dtype=torch.bfloat16)
    ref = codec.fp4_group16_act_qdq(x).to(torch.bfloat16)
    assert _bit_equal(ref, ops.fp4_act_qdq_or_codec(x))
    # fp32 input takes the eager branch by dtype (the op is bf16-only).
    xf = x.float()
    ref32 = codec.fp4_group16_act_qdq(xf).to(torch.bfloat16)
    assert _bit_equal(ref32, ops.fp4_act_qdq_or_codec(xf))

    # An available CUDA extension must not make a CPU bf16 tensor call a CUDA-
    # only op.  The resolver's eager fallback is device-generic.
    x_cpu = torch.randn(2, 32, dtype=torch.bfloat16)
    ref_cpu = codec.fp4_group16_act_qdq(x_cpu).to(torch.bfloat16)
    assert _bit_equal(ref_cpu, ops.fp4_act_qdq_or_codec(x_cpu))


def test_cuda_graph_replay_matches_eager_bits():
    from gridbook import ops
    x = torch.randn(4, 3072, device=DEV, dtype=torch.bfloat16)
    ref = ops.fp4_act_qdq_or_codec(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.fp4_act_qdq_or_codec(x)
    graph.replay()
    torch.cuda.synchronize()
    assert _bit_equal(ref, captured)


def test_torch_compile_custom_op_matches_eager_bits():
    from gridbook import ops
    x = torch.randn(4, 3072, device=DEV, dtype=torch.bfloat16)
    expected = ops.fp4_act_qdq_or_codec(x)
    compiled = torch.compile(ops.fp4_act_qdq_or_codec, fullgraph=True)
    actual = compiled(x)
    assert _bit_equal(expected, actual)


def _host_us(fn, x, iters=300, warmup=30):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@pytest.mark.parametrize("m,k", [(1, 3072), (1, 1024), (8, 3072), (256, 3072)])
def test_cuda_cheaper_than_eager_codec(m, k, capsys):
    """One launch must beat ~two dozen eager dispatches.

    The assertion threshold is 1x — deliberately not a tuned number, because a
    tuned one would flake on a shared GPU. The point of the test is the printed
    comparison; the assert only catches a catastrophic regression (e.g. the
    resolver silently falling back).
    """
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    eager = _host_us(lambda t: codec.fp4_group16_act_qdq(t).to(torch.bfloat16), x)
    cuda = _host_us(ext.fp4_act_qdq, x)
    with capsys.disabled():
        print(f"\n  M={m:4d} K={k:5d}  eager-codec {eager:7.2f} us   "
              f"cuda-fp4 {cuda:6.2f} us   saved {eager - cuda:7.2f} us "
              f"[{eager / cuda:5.1f}x]")
    assert cuda < eager
