"""Correctness gates for the persistent-B grouped MoE decode-in-mainloop lane.

``csrc/cb_moe_persistent_b.cu`` (ROADMAP K1.1) fuses the FP4-CB v2 weight
decode into a grouped-MoE mainloop: a CTA owns one ``(expert, N-tile)``,
decodes that weight tile from the packed CB bytes into shared memory, and
streams the expert's routed rows through it.  There is no ``[E, N, K]`` BF16
HBM transient and no dense entry point in the translation unit.

WHAT IS PROVEN HERE, AND WITH WHICH INSTRUMENT

* The DECODE is an identity, so it is gated with ``torch.equal`` on an integer
  view: ``cb_moe_persistent_b_decode`` exposes the mainloop's own decode stage
  and must be BIT-IDENTICAL to the shipping expander ``cb_expand_v2`` on every
  rung.  The kernel's decode assembles its 64-bit codeword window from u32
  shared-memory reads where the expander uses byte reads, so this is the test
  that the substitution really is algebraic.  Odd and even ``k`` are both
  covered because the ceil-first bit split puts a different width in the low
  half; k=24 is the largest LUT.
* The whole OPERATOR is not an identity and cannot be.  It is an
  fp32-accumulate GEMM with its own tile shape, K walk and warp partition, so
  it differs from a per-expert GEMM by REDUCTION ORDER.  The discipline is the
  one ``test_bf16_grouped_cutlass.py`` established: measure the lane's relative
  L2 against an FP32 reference built from the SAME BF16 operands, and require
  it to be no worse than a per-segment BF16 ``F.linear`` computing the same.
  The BF16 operands are the activations the caller passes and the weights the
  kernel's own (bit-exact) decode produces, so the reference describes exactly
  the arithmetic the kernel claims to do.
* ROUTING breadth, config equivalence, stream/graph capture and the contract
  rejections are ordinary behavioural gates.

N SCOPE NOTE.  ``cb_moe_persistent_b_prefill`` requires ``N % 8 == 0`` (the
epilogue stores pairs of BF16), and ``cfg=0`` additionally refuses an N
narrower than the narrowest compiled TN.  Those two floors do not agree, so
this file gates the operator only down to the narrowest compiled TN and takes
no position on the gap; see the accompanying report.  The decode probe has
neither constraint, because it walks a flat byte plane.  So the "N is not a
nice multiple" coverage splits: the decode tests use N=17 and N=33 (not
multiples of 8 or 32, so the flat plane's rows line up with no codeword or
LUT-gather boundary), the prefill tests use N=40 and N=200 (multiples of 8,
never of 32, so every compiled TN masks a partial tile) down to the narrowest
compiled TN, and the N%8 rejection is gated explicitly.  Every tile-shaped
constant is ENUMERATED from ``cb_moe_persistent_b_configs()``; this file
hardcodes no tile.

Container-only, like the rest of the native-kernel suite: the module skips
cleanly (never vacuously) without CUDA, without prismaquant, or off the
cc 12.0/12.1 devices the lane is compiled for.
"""
from __future__ import annotations

import contextlib
import functools
import re

import pytest
import torch
import torch.nn.functional as F

codec = pytest.importorskip("gridbook.codec")
pq = pytest.importorskip("prismaquant.nvfp4_cb_formats")

from gridbook.cuda_ext import (get_ext_v2,  # noqa: E402
                               get_moe_persistent_b_ext)

if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

ext = get_moe_persistent_b_ext()
if ext is None:
    pytest.skip("the persistent-B grouped MoE extension is unavailable "
                "(needs cc 12.0/12.1 + nvcc)", allow_module_level=True)

v2ext = get_ext_v2()
if v2ext is None:
    pytest.skip("the CB-GEMV-v2 extension is unavailable, so there is no "
                "cb_expand_v2 oracle for the decode identity",
                allow_module_level=True)
v2ext.cb_gemv_v2_prepare()

DEV = torch.device("cuda")
TILE_K = int(ext.cb_moe_persistent_b_tile_k())
CONFIGS = [list(map(int, row)) for row in ext.cb_moe_persistent_b_configs()]
# ALWAYS enumerated, never hardcoded: a (TM, TN) pair can be added or dropped
# for shared-memory reasons and this file must not be able to name one that was
# not compiled.  The narrowest compiled TN is the smallest N the auto selector
# can serve, so it is the N floor these tests exercise.
MIN_TN = min(cfg[1] for cfg in CONFIGS)
MAX_TN = max(cfg[1] for cfg in CONFIGS)


@contextlib.contextmanager
def _true_fp32_matmul():
    """The fp32 reference must be a REAL fp32 reference, not TF32.

    Scoped and restored rather than set at import: pytest imports every
    selected module during collection, so a module-level mutation of a global
    numeric mode would silently change the arithmetic of unrelated test files.
    """
    tf32 = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_float32_matmul_precision(precision)


# ===========================================================================
# Operand construction — PrismaQuant's real two-tier v2 encoder, never a
# fabricated packed plane.
# ===========================================================================
@functools.lru_cache(maxsize=None)
def _pack(k: int, K: int, E: int, N: int, seed: int):
    """``(qw [E,N,row_bytes], lut, compose, type_size)`` for FP4-CB v2."""
    cb = pq._resolve_codebook(k, "fp4", "product", None, DEV)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = (torch.randn(E, N, K, generator=generator) * 0.02).to(DEV)
    fields = pq.nvfp4_cb_fields(
        weight, k, grid="fp4", mode="product", codebook=cb,
        scale_coding="two_tier", encode_tier="fast")
    packed = pq.nvfp4_cb_assemble_bytes(fields, k, grid="fp4", mode="product")
    type_size = pq.nvfp4_cb_type_size(k, "fp4", "two_tier")
    qw = packed.reshape(E, N, (K // 256) * type_size).contiguous().to(DEV)
    subs = list(cb) if isinstance(cb, (tuple, list)) else [cb]
    lut = codec.build_flat_codebook(subs)
    compose = codec.build_compose_table(codec.TWO_TIER_SUB_TABLE).to(DEV)
    return qw, lut, compose, type_size


def _ends(counts) -> torch.Tensor:
    """The exact-segment routing the kernel consumes: cumsum(bincount)."""
    return torch.tensor(counts, dtype=torch.int32,
                        device=DEV).cumsum(0, dtype=torch.int32).contiguous()


def _decoded_weights(qw, lut, compose, k, type_size, K) -> torch.Tensor:
    """``[E, N, K]`` BF16 weights, produced by the kernel's OWN decode stage.

    Using the probe rather than ``cb_expand_v2`` keeps the numerics reference
    self-contained; the two are gated bit-identical above, so this is the same
    tensor either way.
    """
    E, N, _ = qw.shape
    flat = ext.cb_moe_persistent_b_decode(
        qw.reshape(-1), lut, compose, 0, E * N, K, k, type_size)
    return flat.view(E, N, K)


def _prefill(a, qw, lut, compose, ends, k, type_size, cfg=0, out=None):
    if out is None:
        out = torch.empty((a.shape[0], qw.shape[1]),
                          dtype=torch.bfloat16, device=DEV)
    ext.cb_moe_persistent_b_prefill(
        out, a, qw, lut, compose, ends, k, type_size, cfg)
    return out


def _per_segment_references(a, weights, expert_ends):
    """BF16 ``F.linear`` and highest-precision FP32 references, per segment.

    Rows past the last cumulative end belong to no expert; the kernel must not
    write them, so the references leave them alone too (the caller compares the
    routed prefix only).
    """
    rows = int(expert_ends[-1].item()) if expert_ends.numel() else 0
    n = weights.shape[1]
    y_bf16 = torch.zeros((rows, n), dtype=torch.bfloat16, device=DEV)
    y_fp32 = torch.zeros((rows, n), dtype=torch.float32, device=DEV)
    with _true_fp32_matmul():
        start = 0
        for expert, stop in enumerate(expert_ends.cpu().tolist()):
            if stop > start:
                y_bf16[start:stop] = F.linear(a[start:stop], weights[expert])
                y_fp32[start:stop] = (
                    a[start:stop].float() @ weights[expert].float().t())
            start = stop
    return y_bf16, y_fp32


def _rel_l2(y, reference):
    return ((y.float() - reference).norm()
            / reference.norm().clamp_min(1e-12))


def _assert_reassociation_only(y, bf16_linear, fp32, label):
    """The two-part gate: absolute backstop + relative to ``F.linear``."""
    kernel_rel = _rel_l2(y, fp32)
    linear_rel = _rel_l2(bf16_linear, fp32)
    assert torch.isfinite(y).all(), f"{label}: non-finite output"
    assert kernel_rel <= 2e-3, (
        f"{label}: persistent-B relative L2 {kernel_rel:.6e} exceeds the BF16 "
        f"output-rounding backstop")
    assert kernel_rel <= torch.maximum(1.25 * linear_rel, linear_rel + 2e-5), (
        f"{label}: persistent-B {kernel_rel:.6e} vs per-segment BF16 "
        f"F.linear {linear_rel:.6e}: the reduction-order difference is larger "
        f"than reassociation explains")


# The rungs: k=12..24 spans odd and even k (the ceil-first split puts
# ceil(k/2) bits in the low half, so the two parities take different paths),
# and k=24 is the largest LUT the format admits.  N=17 and N=33 are NOT
# multiples of 8 or 32, so the flat plane's rows do not line up with any
# codeword or LUT-gather boundary.
_RUNGS = [
    (12, 256, 4, 8),
    (13, 512, 3, 17),
    (14, 768, 3, 33),
    (16, 1536, 2, 64),
    (20, 2048, 2, 40),
    (24, 4096, 2, 24),
]
_RUNG_IDS = [f"k{k}-K{K}-E{E}-N{N}" for k, K, E, N in _RUNGS]


# ===========================================================================
# 1. The decode stage is an IDENTITY, gated bitwise.
# ===========================================================================
@pytest.mark.parametrize("k,K,E,N", _RUNGS, ids=_RUNG_IDS)
def test_decode_probe_is_bit_identical_to_cb_expand_v2(k, K, E, N):
    """The mainloop's decode must be the shipping expander, bit for bit.

    ``cb_moe_persistent_b_decode`` runs the same ``cb_decode_codeword`` device
    function the mainloop calls, out of the same 4-byte-aligned shared staging,
    so a disagreement here is a disagreement inside the GEMM.
    """
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=k * 1009 + K)
    flat = qw.reshape(-1)
    rows = E * N

    got = ext.cb_moe_persistent_b_decode(
        flat, lut, compose, 0, rows, K, k, type_size)
    want = v2ext.cb_expand_v2(flat, lut, compose, 0, rows, K, k, type_size)

    assert got.shape == (rows, K)
    assert got.dtype is torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("k,K,E,N", _RUNGS, ids=_RUNG_IDS)
def test_decode_probe_window_at_row0_is_bit_identical(k, K, E, N):
    """A ``row0 > 0`` window: the byte-plane base offset is per-row, not per
    call, so an off-by-one row stride shows up only when the window moves."""
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=k * 1009 + K)
    flat = qw.reshape(-1)
    rows = E * N
    row0 = max(1, rows // 3)
    nrows = max(1, rows - row0 - 1)
    assert row0 > 0 and row0 + nrows <= rows

    got = ext.cb_moe_persistent_b_decode(
        flat, lut, compose, row0, nrows, K, k, type_size)
    want = v2ext.cb_expand_v2(
        flat, lut, compose, row0, nrows, K, k, type_size)

    assert got.shape == (nrows, K)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # ...and the window really is a window: it equals the same rows of the
    # full-plane decode, so the two tests cannot both pass on a constant.
    full = ext.cb_moe_persistent_b_decode(
        flat, lut, compose, 0, rows, K, k, type_size)
    assert torch.equal(got.view(torch.int16),
                       full[row0:row0 + nrows].view(torch.int16))


def test_decode_probe_zero_rows_is_a_well_formed_empty_result():
    k, K, E, N = 16, 512, 2, 64
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=5)
    got = ext.cb_moe_persistent_b_decode(
        qw.reshape(-1), lut, compose, 3, 0, K, k, type_size)
    assert got.shape == (0, K)
    assert got.dtype is torch.bfloat16


# ===========================================================================
# 2. Whole-operator numerics: reassociation, and nothing else.
# ===========================================================================
@pytest.mark.parametrize("counts,k,K,N", [
    ([10, 0, 130, 3], 16, 1024, 256),
    ([200, 1, 0], 13, 512, 128),
    ([1, 1], 24, 1024, 64),
    ([0, 0, 5], 20, 768, 40),
    ([64, 64, 64, 64, 64], 12, 2048, 512),
    ([3, 0, 4], 12, 256, MIN_TN),
], ids=["empty-middle-longtail", "skewed-trailing-empty", "one-row-each",
        "leading-empty-ragged-n", "exact-tile-multiples", "narrowest-tile-n"])
def test_whole_operator_error_matches_a_per_segment_bf16_reference(
        counts, k, K, N):
    """The fused-decode GEMM may not consume more error than a per-expert GEMM.

    Both sides consume the SAME BF16 operands — the caller's activations and
    the weights this kernel's own decode produces — so the only admissible
    difference is the fp32 summation order.

    MEASURED (GB10, cc 12.1) on ``counts=[10,0,130,3]``, k=16, K=1024, N=256:
    kernel 1.667688e-3 against fp32, per-segment BF16 ``F.linear`` 1.726041e-3,
    ratio 0.966 — the fused-decode lane is if anything the more accurate of the
    two.  The gate is not vacuous: shifting the routing by ONE expert takes the
    same measurement to 3.749e-1, and transposing a single pair of weight
    columns (the decode/swizzle failure class) takes it to 5.868e-2.  Both are
    one to two orders of magnitude past the 2e-3 backstop.
    """
    torch.manual_seed(20260801)
    E, P = len(counts), sum(counts)
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=k * 31 + N)
    weights = _decoded_weights(qw, lut, compose, k, type_size, K)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)

    y = _prefill(a, qw, lut, compose, ends, k, type_size)
    bf16_linear, fp32 = _per_segment_references(a, weights, ends)
    _assert_reassociation_only(y, bf16_linear, fp32, f"counts={counts}")


# ===========================================================================
# 3. Routing breadth, and the promise that nothing outside a segment is
#    written.
# ===========================================================================
_ROUTING = [
    ([0, 7, 5], "leading-empty"),
    ([6, 0, 9], "middle-empty"),
    ([4, 11, 0], "trailing-empty"),
    ([0, 0, 33, 0, 0], "all-empty-but-one"),
    ([0, 0, 0, 0], "all-empty"),
    ([17], "single-expert"),
    ([1, 1, 1, 1, 1, 1], "one-row-per-expert"),
    ([700, 3, 0, 1, 0, 2], "skewed-past-tile-m"),
    ([1, 0, 2] + [0] * 13, "more-experts-than-rows"),
]


@pytest.mark.parametrize("counts", [c for c, _ in _ROUTING],
                         ids=[i for _, i in _ROUTING])
def test_routing_breadth_is_correct_and_writes_only_routed_rows(counts):
    """Every routing shape the served operator can produce.

    ``out`` is pre-poisoned with NaN and given more rows than the routing
    claims.  Rows at or past ``expert_ends[-1]`` belong to no expert, so the
    kernel must leave their exact bit patterns alone — which is the observable
    form of "an empty expert costs two int32 loads and a return".  The
    comparison is on the integer view because NaN is not equal to itself.
    """
    torch.manual_seed(4242 + len(counts))
    k, K, N = 16, 512, 64
    spare = 5
    E, total = len(counts), sum(counts)
    P = total + spare
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=777)
    weights = _decoded_weights(qw, lut, compose, k, type_size, K)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)
    assert int(ends[-1].item()) == total

    out = torch.full((P, N), float("nan"), dtype=torch.bfloat16, device=DEV)
    _prefill(a, qw, lut, compose, ends, k, type_size, out=out)
    torch.cuda.synchronize()

    tail = out[total:]
    sentinel = torch.full_like(tail, float("nan"))
    assert torch.equal(tail.view(torch.int16),
                       sentinel.view(torch.int16)), (
        f"counts={counts}: the kernel wrote {int((~tail.isnan()).sum())} "
        f"values past the last routed row")

    routed = out[:total]
    if total:
        bf16_linear, fp32 = _per_segment_references(a, weights, ends)
        _assert_reassociation_only(routed, bf16_linear, fp32,
                                   f"counts={counts}")
    else:
        assert routed.numel() == 0


def test_zero_routed_rows_returns_a_well_formed_empty_output():
    """P == 0 is the degenerate prefill the operator can still be handed."""
    k, K, E, N = 13, 512, 4, 64
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=11)
    a = torch.empty(0, K, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(0, N, dtype=torch.bfloat16, device=DEV)
    ends = torch.zeros(E, dtype=torch.int32, device=DEV)
    _prefill(a, qw, lut, compose, ends, k, type_size, out=out)
    torch.cuda.synchronize()
    assert out.shape == (0, N)
    assert out.dtype is torch.bfloat16


# ===========================================================================
# 4. Config selection: different tiles, same operator.
# ===========================================================================
def _config_case():
    """The widest rung on purpose.

    Shared memory per CTA grows with ``type_size``, so k=24 is the strictest
    case a config has to survive — and it is the rung
    ``cb_moe_persistent_b_configs()`` quotes its ``smem`` figure at.  Running
    every attested config here therefore gates ATTESTATION AGAINST REALITY: a
    tile the module advertises but would refuse to launch (over the sm_120
    budget, or under the two-CTAs-per-SM occupancy floor) fails right here
    instead of at a serving call.
    """
    counts = [0, 5, 200, 1, 0, 70]
    k, K, N = 24, 1024, 200            # N % 8 == 0, N % 32 != 0
    E, P = len(counts), sum(counts)
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=1234)
    torch.manual_seed(9090)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    return counts, k, K, N, qw, lut, compose, type_size, a


@pytest.mark.parametrize("cfg", list(range(1, len(CONFIGS) + 1)),
                         ids=[f"cfg{i + 1}-tm{c[0]}-tn{c[1]}"
                              for i, c in enumerate(CONFIGS)])
def test_every_compiled_config_agrees_with_auto(cfg):
    """A compiled tile may be selected explicitly and must compute the same op.

    Distinct (TM, TN) pairs are distinct warp partitions and distinct K walks,
    so they reassociate the fp32 accumulation differently.  The gate is
    therefore a bound sized to reassociation, the same one the two BF16 grouped
    lanes are held to, plus the absolute error gate against fp32 for each
    config independently.

    MEASURED (GB10, cc 12.1) on the shape below: all six compiled tiles were in
    fact BIT-IDENTICAL to the auto choice.  That is not a promise the kernel
    makes and this test does not assert it — bf16xbf16 products are exact in
    fp32, so the tiles differ only in the ~2^-24 rounding of their fp32 partial
    sums, an order of magnitude below the 2^-8 quantum of the bf16 result, and
    a disagreement needs an accumulator within half an ulp of a rounding
    boundary.  Possible, not observed.
    """
    counts, k, K, N, qw, lut, compose, type_size, a = _config_case()
    ends = _ends(counts)
    weights = _decoded_weights(qw, lut, compose, k, type_size, K)

    auto = _prefill(a, qw, lut, compose, ends, k, type_size, cfg=0)
    chosen = _prefill(a, qw, lut, compose, ends, k, type_size, cfg=cfg)

    bf16_linear, fp32 = _per_segment_references(a, weights, ends)
    _assert_reassociation_only(auto, bf16_linear, fp32, "cfg=0")
    _assert_reassociation_only(chosen, bf16_linear, fp32, f"cfg={cfg}")

    disagreement = _rel_l2(chosen.float(), auto.float())
    assert disagreement <= 4e-3, (
        f"cfg={cfg} {CONFIGS[cfg - 1][:3]} disagrees with the auto config by "
        f"{disagreement:.6e} relative L2, beyond a reassociation difference")


def test_auto_config_steps_down_to_a_narrower_n_tile():
    """An N under the widest compiled TN must take the auto step-down."""
    counts = [0, 9, 40]
    k, K, N = 20, 512, MIN_TN
    E, P = len(counts), sum(counts)
    assert N < MAX_TN, "this test needs an N under the widest compiled TN"
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=606)
    torch.manual_seed(607)
    a = torch.randn(P, K, device=DEV, dtype=torch.bfloat16)
    ends = _ends(counts)
    weights = _decoded_weights(qw, lut, compose, k, type_size, K)

    y = _prefill(a, qw, lut, compose, ends, k, type_size, cfg=0)
    bf16_linear, fp32 = _per_segment_references(a, weights, ends)
    _assert_reassociation_only(y, bf16_linear, fp32, "auto-small-N")


# ===========================================================================
# 5/6. Stream and CUDA-graph contract.
# ===========================================================================
def _stream_case():
    counts = [0, 5, 130, 1, 0, 12]
    k, K, N = 16, 512, 128
    E, P = len(counts), sum(counts)
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=1911)
    torch.manual_seed(1912)
    a = torch.randn(P + 3, K, device=DEV, dtype=torch.bfloat16)
    return (a, qw, lut, compose, _ends(counts), k, type_size, P + 3, N)


def test_nondefault_stream_matches_the_default_stream_result():
    """It honours ``getCurrentCUDAStream``; the answer does not move."""
    a, qw, lut, compose, ends, k, type_size, P, N = _stream_case()

    default = torch.full((P, N), float("nan"),
                         dtype=torch.bfloat16, device=DEV)
    _prefill(a, qw, lut, compose, ends, k, type_size, out=default)
    torch.cuda.synchronize()

    producer = torch.cuda.current_stream()
    worker = torch.cuda.Stream()
    assert worker.cuda_stream != producer.cuda_stream, (
        "this test is vacuous unless the worker really is a second stream")
    worker.wait_stream(producer)
    with torch.cuda.stream(worker):
        on_worker = torch.full((P, N), float("nan"),
                               dtype=torch.bfloat16, device=DEV)
        ext.cb_moe_persistent_b_prefill(
            on_worker, a, qw, lut, compose, ends, k, type_size, 0)
    worker.synchronize()

    assert torch.equal(on_worker.view(torch.int16), default.view(torch.int16))


@pytest.mark.parametrize("on_worker_stream", [False, True],
                         ids=["default-stream", "worker-stream"])
def test_cuda_graph_capture_and_replay_match_eager(on_worker_stream):
    """The launch geometry is a pure function of (E, N) and the tile, so the
    call captures with no host sync and no routing-dependent grid.

    The kernel writes into a CALLER-OWNED ``out``, so capture and every replay
    reuse the identical buffer; each replay is re-poisoned first and compared
    as a snapshot, which also re-proves that replay leaves the unrouted tail
    alone.  Bitwise, because a graph replay of a deterministic kernel is an
    identity, not an approximation.
    """
    a, qw, lut, compose, ends, k, type_size, P, N = _stream_case()
    out = torch.full((P, N), float("nan"), dtype=torch.bfloat16, device=DEV)
    poison = torch.full_like(out, float("nan"))

    ext.cb_moe_persistent_b_prefill(
        out, a, qw, lut, compose, ends, k, type_size, 0)
    torch.cuda.synchronize()
    eager = out.clone()
    assert not torch.equal(eager.view(torch.int16), poison.view(torch.int16))

    graph = torch.cuda.CUDAGraph()
    worker = None
    if on_worker_stream:
        worker = torch.cuda.Stream()
        assert worker.cuda_stream != torch.cuda.current_stream().cuda_stream
        worker.wait_stream(torch.cuda.current_stream())
        capture = torch.cuda.graph(graph, stream=worker)
    else:
        capture = torch.cuda.graph(graph)

    out.copy_(poison)
    with capture:
        ext.cb_moe_persistent_b_prefill(
            out, a, qw, lut, compose, ends, k, type_size, 0)

    for replay in range(3):
        out.copy_(poison)
        graph.replay()
        if worker is not None:
            worker.synchronize()
        torch.cuda.synchronize()
        assert torch.equal(out.clone().view(torch.int16),
                           eager.view(torch.int16)), (
            f"replay {replay} on the "
            f"{'worker' if on_worker_stream else 'default'} stream diverged "
            f"from the eager result")


# ===========================================================================
# 7. Contract violations fail loudly at the API boundary.
# ===========================================================================
def _valid_args():
    k, K, E, N = 16, 512, 3, 64
    counts = [2, 0, 6]
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=4242)
    P = sum(counts)
    torch.manual_seed(4243)
    return {
        "out": torch.empty((P, N), dtype=torch.bfloat16, device=DEV),
        "a": torch.randn(P, K, device=DEV, dtype=torch.bfloat16),
        "qw": qw,
        "lut": lut,
        "compose": compose,
        "expert_ends": _ends(counts),
        "k_bits": k,
        "type_size": type_size,
        "cfg": 0,
    }


def _noncontiguous_like(a):
    wide = torch.randn(a.shape[0], 2 * a.shape[1],
                       device=DEV, dtype=torch.bfloat16)
    view = wide.narrow(1, 0, a.shape[1])
    assert not view.is_contiguous() and view.shape == a.shape
    return view


def _wrong_k(a):
    """Same P, a K that is not a CB superblock multiple."""
    return torch.randn(a.shape[0], 640, device=DEV, dtype=torch.bfloat16)


def _permuted_qw(qw):
    view = qw.transpose(1, 2).contiguous().transpose(1, 2)
    assert view.shape == qw.shape and not view.is_contiguous()
    return view


# The pattern is a substring of the kernel's real TORCH_CHECK text.
_VIOLATIONS = [
    ("noncontiguous-a",
     lambda c: {**c, "a": _noncontiguous_like(c["a"])},
     "must be contiguous"),
    ("fp32-a",
     lambda c: {**c, "a": c["a"].float()},
     "activations must be BF16"),
    ("fp32-out",
     lambda c: {**c, "out": c["out"].float()},
     "output must be BF16"),
    ("int64-expert-ends",
     lambda c: {**c, "expert_ends": c["expert_ends"].to(torch.int64)},
     "expert_ends must be int32"),
    ("cpu-expert-ends",
     lambda c: {**c, "expert_ends": c["expert_ends"].cpu()},
     "every operand must be a CUDA tensor"),
    ("short-expert-ends",
     lambda c: {**c, "expert_ends": c["expert_ends"][:-1].contiguous()},
     "one cumulative count per expert"),
    ("k-not-superblock-multiple",
     lambda c: {**c, "a": _wrong_k(c["a"])},
     re.escape("K must be a multiple of the CB superblock")),
    ("bad-type-size",
     lambda c: {**c, "type_size": c["type_size"] + 1},
     "FP4-CB layout v2 requires"),
    ("k-bits-out-of-range",
     lambda c: {**c, "k_bits": 25},
     re.escape("k_bits in [1,24]")),
    ("short-lut",
     lambda c: {**c, "lut": c["lut"][:-4].contiguous()},
     "flat product codebook must hold"),
    ("wrong-size-compose",
     lambda c: {**c, "compose": c["compose"][:-16].contiguous()},
     re.escape("compose table must be 256*16 floats")),
    ("fp64-compose",
     lambda c: {**c, "compose": c["compose"].double()},
     "compose table must be FP32"),
    ("permuted-qw-stack",
     lambda c: {**c, "qw": _permuted_qw(c["qw"])},
     "every operand must be contiguous|fully contiguous"),
    ("wrong-out-shape",
     lambda c: {**c, "out": torch.empty((c["out"].shape[0],
                                         c["out"].shape[1] + 8),
                                        dtype=torch.bfloat16, device=DEV)},
     re.escape("out must be [P,N]")),
    ("uncompiled-cfg",
     lambda c: {**c, "cfg": len(CONFIGS) + 1},
     re.escape(f"cfg must be 0 (auto) or 1..{len(CONFIGS)}")),
    ("negative-cfg",
     lambda c: {**c, "cfg": -1},
     re.escape("cfg must be 0 (auto) or 1..")),
]


@pytest.mark.parametrize("mutate,message",
                         [(m, msg) for _, m, msg in _VIOLATIONS],
                         ids=[name for name, _, _ in _VIOLATIONS])
def test_rejects_contract_violations(mutate, message):
    args = mutate(_valid_args())
    with pytest.raises(RuntimeError, match=message):
        ext.cb_moe_persistent_b_prefill(
            args["out"], args["a"], args["qw"], args["lut"], args["compose"],
            args["expert_ends"], args["k_bits"], args["type_size"],
            args["cfg"])


def test_rejects_an_n_that_is_not_a_multiple_of_eight():
    """The epilogue stores BF16 PAIRS, so a ragged N is refused."""
    k, K, E, N = 16, 512, 2, 12
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=808)
    a = torch.randn(4, K, device=DEV, dtype=torch.bfloat16)
    out = torch.empty((4, N), dtype=torch.bfloat16, device=DEV)
    ends = _ends([1, 3])
    with pytest.raises(RuntimeError, match="N must be a multiple of 8"):
        ext.cb_moe_persistent_b_prefill(
            out, a, qw, lut, compose, ends, k, type_size, 0)


@pytest.mark.parametrize("mutate,message", [
    (lambda f, l, c, K, k, ts: (f.cpu(), l, c, 0, 1, K, k, ts),
     "operands must be CUDA tensors"),
    (lambda f, l, c, K, k, ts: (f.view(torch.int8), l, c, 0, 1, K, k, ts),
     "contiguous 1-D"),
    (lambda f, l, c, K, k, ts: (f, l, c.cpu(), 0, 1, K, k, ts),
     "operands must be CUDA tensors"),
    (lambda f, l, c, K, k, ts: (f, l, c[:-16].contiguous(), 0, 1, K, k, ts),
     re.escape("256*16 FP32 table")),
    (lambda f, l, c, K, k, ts: (f, l, c, 0, 1, 511, k, ts),
     re.escape("K must be a multiple of 256")),
    (lambda f, l, c, K, k, ts: (f, l, c, 0, 1, K, k, ts + 1),
     re.escape("type_size == 4*k+9")),
    (lambda f, l, c, K, k, ts: (f, l, c, -1, 1, K, k, ts),
     "must be non-negative"),
    (lambda f, l, c, K, k, ts: (f, l, c, 0, 1 << 20, K, k, ts),
     "byte plane holds"),
], ids=["cpu-plane", "int8-plane", "cpu-compose", "short-compose", "ragged-K",
        "bad-type-size", "negative-row0", "past-the-end"])
def test_decode_probe_rejects_contract_violations(mutate, message):
    k, K, E, N = 16, 512, 2, 64
    qw, lut, compose, type_size = _pack(k, K, E, N, seed=4242)
    args = mutate(qw.reshape(-1), lut, compose, K, k, type_size)
    with pytest.raises(RuntimeError, match=message):
        ext.cb_moe_persistent_b_decode(*args)


# ===========================================================================
# 8. The ROADMAP K1.3 firewall, and 9. what was actually compiled.
# ===========================================================================
def test_the_module_has_no_dense_entry_point():
    """K1.3 keeps dense large-M closed; this module must not reopen it.

    The firewall is structural, not documentary: every ``cb_*`` binding either
    takes ``expert_ends`` (so a dense caller cannot reach the schedule) or is a
    decode probe / host-only attestation that computes no GEMM at all.
    """
    assert ext.cb_moe_persistent_b_is_moe_only() is True

    routed = {"cb_moe_persistent_b_prefill"}
    non_gemm = {"cb_moe_persistent_b_decode", "cb_moe_persistent_b_configs",
                "cb_moe_persistent_b_tile_k",
                "cb_moe_persistent_b_is_moe_only",
                # Device attestation: sets the dynamic-smem opt-in for every
                # compiled tile and validates the capability. No operands, no
                # GEMM, so it cannot be a dense entry point.
                "cb_moe_persistent_b_prepare"}
    exported = {name for name in dir(ext) if name.startswith("cb_")}
    assert exported == routed | non_gemm, (
        f"unexpected bindings {sorted(exported - (routed | non_gemm))}: a new "
        f"public entry point must be classified against the K1.3 firewall")

    for name in routed:
        doc = getattr(ext, name).__doc__ or ""
        assert "expert_ends" in doc, (
            f"{name} computes a GEMM but its signature does not take the "
            f"routing; that is a dense entry point")
    for name in non_gemm - {"cb_moe_persistent_b_decode"}:
        assert not getattr(ext, name).__doc__.count("Tensor"), (
            f"{name} is supposed to be a host-only attestation")


def test_compiled_configs_are_sane():
    """Enumerate what was compiled — never assert a hardcoded tile list."""
    assert TILE_K == 64, (
        "TK=64 is load-bearing: it is 8 sixteen-byte chunks (a conflict-free "
        "XOR swizzle over 8 ldmatrix rows) and it divides the 256-column CB "
        "superblock evenly")
    assert CONFIGS, "the module compiled no tile configs at all"

    for tm, tn, warps, threads, smem, capacity in CONFIGS:
        label = f"TM={tm} TN={tn} warps={warps}"
        assert threads == warps * 32, label
        assert 0 < threads <= 1024, label
        assert tn % 32 == 0, f"{label}: TN must tile the 32-wide warp columns"
        assert tm % 16 == 0, f"{label}: TM must tile the m16 MMA atom"
        assert 0 < smem <= capacity, (
            f"{label}: needs {smem} B of dynamic shared memory against a "
            f"{capacity} B sm_120 budget")
        # Two CTAs per SM is the occupancy floor the schedule is built around;
        # W4's grouped-BF16 collective measured what a slip to one costs.
        # Stated in ATTESTED numbers only, so it survives a retuned budget.
        assert 2 * smem <= capacity, (
            f"{label}: {smem} B leaves room for only one CTA in the "
            f"{capacity} B budget")
        # The kernel's own static_asserts, restated on the host side.
        wn = tn // 32
        assert warps % wn == 0, f"{label}: the warp grid must cover the CTA"
        wm = warps // wn
        assert tm % (16 * wm) == 0, f"{label}: TM must tile the warp rows"
        assert ((tm // wm) // 16) * ((tn // wn) // 8) * 4 <= 128, (
            f"{label}: accumulator register budget")
