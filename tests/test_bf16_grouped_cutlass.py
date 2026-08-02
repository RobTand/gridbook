"""Correctness gates for the owned CUTLASS BF16 grouped bridge.

The bridge consumes already-quantized BF16 activations and already-expanded
BF16 weights. Those two tensors are the quality contract; the grouped kernel
may differ from a per-expert GEMM only by FP32 summation order. Tests therefore
compare both implementations to an FP32 reference made from the *same* BF16
operands, and separately ratchet the chunked in-place serving topology.

BOTH LANES ARE GATED HERE. The default SM80 device-scheduled lane takes exact
per-expert segments; the OPT-IN sm12x-native lane (audit §3 P1) takes the
row-padded, tile-indexed layout. Neither is bit-exact against the other, and
neither can be: they are fp32-accumulate GEMMs with different tile shapes and
K-iteration, so they differ by REDUCTION ORDER. The comparison discipline is
therefore the one this file already used — measure each lane's relative error
against an FP32 reference built from the SAME BF16 operands, and require it to
be no worse than a per-segment BF16 ``F.linear`` computing the same thing.
That is a real gate (it fails on any indexing, expert-selection, padding or
alignment bug, all of which move the error by orders of magnitude) and it is
the only defensible one for a reassociating kernel.

CUDA-only: the module skips in build environments without the serving
toolchain, matching the native-kernel suite convention.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

ops = pytest.importorskip("gridbook.ops")
from gridbook.cuda_ext import get_bf16_grouped_ext  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA device unavailable", allow_module_level=True)

ext = get_bf16_grouped_ext()
if ext is None:
    pytest.skip("CUTLASS grouped BF16 extension unavailable",
                allow_module_level=True)

DEV = "cuda"


def _routing_counts(experts: int, active: dict[int, int]):
    counts = torch.zeros(experts, dtype=torch.int32, device=DEV)
    for expert, rows in active.items():
        counts[expert] = rows
    return counts, counts.cumsum(0, dtype=torch.int32).contiguous()


def _per_expert_references(a, weights, expert_ends):
    """Return BF16 F.linear and highest-precision FP32 references."""
    y_bf16 = torch.empty(
        (a.shape[0], weights.shape[1]), dtype=torch.bfloat16, device=DEV)
    y_fp32 = torch.empty(
        (a.shape[0], weights.shape[1]), dtype=torch.float32, device=DEV)
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


def test_repeated_endpoints_and_chunked_output_match_single_launch():
    """Sixty-one empty experts and chunk boundaries cannot drop/shift rows."""
    torch.manual_seed(7)
    experts, k, n = 64, 512, 256
    _, ends = _routing_counts(experts, {0: 3, 31: 5, 63: 4})
    pairs = 12
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(
        experts, n, k, device=DEV, dtype=torch.bfloat16)

    single = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    chunked = torch.empty_like(single)
    for start in range(0, experts, 16):
        stop = min(experts, start + 16)
        ops.cb_bf16_grouped_mm_out(
            chunked, a, weights[start:stop].contiguous(), ends, start)

    assert torch.equal(chunked, single)
    assert torch.isfinite(single).all()


def test_fp32_accumulation_error_matches_bf16_linear_reference():
    """CUTLASS cannot consume materially more error than per-expert GEMM.

    K=4096 makes accumulation order visible while the sparse E=32 routing
    keeps the test bounded. Twenty-nine repeated endpoints cover the empty-
    expert scheduler case in the same numerical gate.
    """
    torch.manual_seed(20260801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    experts, k, n = 32, 4096, 512
    active = {0: 7, 13: 19, 31: 11}
    _, ends = _routing_counts(experts, active)
    pairs = sum(active.values())
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(
        experts, n, k, device=DEV, dtype=torch.bfloat16)

    cutlass = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    bf16_linear, fp32 = _per_expert_references(a, weights, ends)
    cutlass_rel = _rel_l2(cutlass, fp32)
    linear_rel = _rel_l2(bf16_linear, fp32)

    # Absolute backstop for BF16 output rounding plus a relative comparison to
    # the prior per-expert BF16 GEMM. Live GB10 attestation at E=64/N=1024 gave
    # 1.673277e-3 vs 1.673275e-3 (ratio 1.0000015).
    assert cutlass_rel <= 2e-3
    assert cutlass_rel <= torch.maximum(
        1.25 * linear_rel, linear_rel + 2e-5)


def test_all_empty_problem_has_a_well_formed_empty_output():
    experts, k, n = 16, 256, 128
    ends = torch.zeros(experts, dtype=torch.int32, device=DEV)
    a = torch.empty(0, k, dtype=torch.bfloat16, device=DEV)
    weights = torch.randn(
        experts, n, k, dtype=torch.bfloat16, device=DEV)
    out = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    assert out.shape == (0, n)
    assert out.dtype is torch.bfloat16


# ===========================================================================
# The OPT-IN sm12x-native lane (audit §3 P1).
# ===========================================================================

HAS_SM120 = hasattr(ext, "cb_bf16_grouped_mm_sm120")
sm120 = pytest.mark.skipif(
    not HAS_SM120,
    reason="the grouped-BF16 module was built without its sm12x lane "
           "(compiled only for compute capability 12.0/12.1)")
TILE_M = int(ext.cb_bf16_grouped_sm120_tile_m()) if HAS_SM120 else 0


def _padded_layout(counts):
    """Build the row-padded tile-indexed layout the sm12x lane consumes.

    Deliberately constructed HERE rather than through
    ``moe_routing.cb_grouped_pad_routing``: this file gates the kernel, so the
    layout it is fed must be an independent statement of the contract.

    Returns ``(expert_ids, row_of_pad_row)`` where ``row_of_pad_row[i]`` is the
    index into the expert-sorted activation array, or ``-1`` for a padding row.
    """
    expert_ids, source = [], []
    start = 0
    for expert, rows in enumerate(counts):
        blocks = (rows + TILE_M - 1) // TILE_M
        for b in range(blocks):
            expert_ids.append(expert)
            for r in range(TILE_M):
                index = b * TILE_M + r
                source.append(start + index if index < rows else -1)
        start += rows
    return (torch.tensor(expert_ids, dtype=torch.int32, device=DEV),
            torch.tensor(source, dtype=torch.int64, device=DEV))


def _run_padded(a_sorted, weights, counts):
    """Gather into the padded layout, run the lane, return (padded, layout)."""
    expert_ids, source = _padded_layout(counts)
    zero_extended = torch.cat(
        [a_sorted, a_sorted.new_zeros((1, a_sorted.shape[1]))])
    gather = torch.where(source < 0,
                         torch.full_like(source, a_sorted.shape[0]), source)
    a_pad = zero_extended.index_select(0, gather).contiguous()
    out = ops.cb_bf16_grouped_mm_sm120(a_pad, weights, expert_ids, TILE_M)
    return out, source


def _real_rows(padded, source, total):
    """Drop the padding rows, restoring the expert-sorted [P, N] result."""
    real = source >= 0
    y = torch.empty((total, padded.shape[1]), dtype=padded.dtype,
                    device=padded.device)
    y.index_copy_(0, source[real], padded[real])
    return y


@sm120
def test_sm120_config_fits_the_sm120_shared_memory_budget():
    """The compiled rung, pinned: pingpong 64x128x64, 3 stages, in budget.

    TileM=64 is the padding granularity and the reason the PINGPONG kernel
    layer is used (cooperative's 256-thread TiledMma floors TileM at 128); the
    128 MMA threads assert that layer is really the one instantiated.
    """
    (tile_m, tile_n, tile_k, stages, smem, capacity, mma_threads,
     swizzle_small, swizzle_large, threshold, gather_mainloop) = \
        ext.cb_bf16_grouped_sm120_config()
    assert (tile_m, tile_n, tile_k) == (64, 128, 64)
    assert mma_threads == 128, "TileM=64 requires the pingpong kernel layer"
    assert stages >= 2, "a TMA warp-specialized mainloop needs two stages"
    assert 0 < smem <= capacity
    assert ext.cb_bf16_grouped_sm120_tile_sizes() == [tile_m]
    # The swizzle policy is a tile ORDER knob: it may never be zero (invalid to
    # CUTLASS) and its two measured values must straddle the threshold.
    assert swizzle_small >= 1 and swizzle_large >= swizzle_small
    assert threshold > 0
    # The in-mainloop A-row gather mode is part of the compiled lane.
    assert gather_mainloop == 1


@sm120
@pytest.mark.parametrize("counts,k,n", [
    ([0, 3, 0, 300, 0, 129, 1, 0], 512, 256),          # uneven + empty
    ([0, 0, 0, 0, 0, 0, 0, 900], 4096, 512),           # single expert, long K
    ([128, 128, 128, 128, 128, 128, 128, 128], 256, 1024),  # exact multiples
    ([1] * 8, 256, 128),                               # one row per expert
    ([64, 65, 191, 192, 193, 0, 7, 256], 1024, 4096),  # tile boundaries
    ([65] * 64, 256, 128),                             # large grid: swizzle 8
], ids=["uneven-empty", "single-expert-longK", "exact-multiple",
        "one-row-each", "tile-boundaries", "large-grid-swizzle"])
def test_sm120_error_matches_a_per_segment_bf16_reference(counts, k, n):
    """The lane's fp32-accumulate error may not exceed a BF16 F.linear's.

    Every routing shape the served operator can produce: empty experts (absent
    from ``expert_ids`` by construction), a single expert, segments that are
    exact tile multiples, one-row segments (the whole rest of the tile is
    padding), and lengths straddling the tile boundary at 64/65/191/192/193.

    The last case crosses the grid size at which the kernel switches its
    tile-scheduler swizzle. That switch is a tile-ORDER argument and must not
    move a bit, which is exactly what this gate then measures.
    """
    torch.manual_seed(20260801)
    torch.backends.cuda.matmul.allow_tf32 = False
    experts = len(counts)
    pairs = sum(counts)
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(experts, n, k, device=DEV, dtype=torch.bfloat16)
    ends = torch.tensor(counts, dtype=torch.int32,
                        device=DEV).cumsum(0, dtype=torch.int32).contiguous()

    padded, source = _run_padded(a, weights, counts)
    y = _real_rows(padded, source, pairs)
    bf16_linear, fp32 = _per_expert_references(a, weights, ends)

    lane_rel = _rel_l2(y, fp32)
    linear_rel = _rel_l2(bf16_linear, fp32)
    assert torch.isfinite(y).all()
    assert lane_rel <= 2e-3, (
        f"sm12x lane relative L2 {lane_rel:.6e} exceeds the BF16 output "
        f"rounding backstop")
    assert lane_rel <= torch.maximum(1.25 * linear_rel, linear_rel + 2e-5), (
        f"sm12x lane {lane_rel:.6e} vs per-segment BF16 F.linear "
        f"{linear_rel:.6e}: the reduction-order difference is larger than "
        f"reassociation explains")


@sm120
@pytest.mark.parametrize("m,k,n", [
    (1, 256, 8),        # N at the 8-element alignment floor
    (100, 8, 128),      # K at the alignment floor
    (128, 4096, 4096),  # exactly one tile
    (300, 1024, 3072),  # M not a tile multiple (pre-padded by the caller)
], ids=["n-align-floor", "k-align-floor", "one-tile", "ragged-m"])
def test_sm120_dense_e1_matches_the_reference(m, k, n):
    """Dense FP4-CB prefill calls this lane with E=1 (linear.py)."""
    torch.manual_seed(4242)
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)

    padded, source = _run_padded(a, weight.unsqueeze(0), [m])
    y = _real_rows(padded, source, m)
    fp32 = a.float() @ weight.float().t()
    bf16_linear = F.linear(a, weight)

    assert torch.isfinite(y).all()
    assert _rel_l2(y, fp32) <= 2e-3
    assert _rel_l2(y, fp32) <= torch.maximum(
        1.25 * _rel_l2(bf16_linear, fp32), _rel_l2(bf16_linear, fp32) + 2e-5)


@sm120
def test_sm120_and_sm80_lanes_differ_only_by_reduction_order():
    """Both lanes, same operands: each near fp32, and close to each other.

    This is the requalification surface stated in one assertion.

    MEASURED (GB10, cc 12.1, 2026-08-01) across the seven shapes gated in this
    file plus DSV4/Laguna projections: both lanes and a per-segment BF16
    ``F.linear`` all land on the SAME relative L2 against fp32 — 1.612e-3 to
    1.663e-3, ratio 1.0000 — and the two lanes' BF16 outputs were bit-identical
    on every one of them. That is not a promise the kernels make and this test
    does not assert it: bf16×bf16 products are exact in fp32, so the lanes
    differ only in the ~2^-24 rounding of their fp32 partial sums, an order of
    magnitude below the 2^-8 quantum of the bf16 result — a disagreement needs
    an accumulator within half an ulp of a rounding boundary. Possible, not
    observed. The gate is therefore a bound, sized to reassociation.
    """
    torch.manual_seed(97)
    counts = [17, 0, 260, 33, 128, 5]
    experts, k, n = len(counts), 2048, 512
    pairs = sum(counts)
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(experts, n, k, device=DEV, dtype=torch.bfloat16)
    ends = torch.tensor(counts, dtype=torch.int32,
                        device=DEV).cumsum(0, dtype=torch.int32).contiguous()

    sm80 = ops.cb_bf16_grouped_mm(a, weights, ends, 0)
    padded, source = _run_padded(a, weights, counts)
    lane = _real_rows(padded, source, pairs)
    _, fp32 = _per_expert_references(a, weights, ends)

    assert _rel_l2(lane, fp32) <= 2e-3
    assert _rel_l2(sm80, fp32) <= 2e-3
    # Row scale is ~sqrt(K); one BF16 ulp there is ~0.25. Disagreement is
    # bounded RELATIVE to the reference magnitude, not absolutely, so the gate
    # travels to other shapes.
    disagreement = _rel_l2(lane.float(), sm80.float())
    assert disagreement <= 4e-3, (
        f"the two lanes disagree by {disagreement:.6e} relative L2, which is "
        f"beyond a reassociation difference")


@sm120
def test_sm120_padding_only_tiles_are_computed_and_discardable():
    """A tile whose expert id is -1 must not fault; its rows are throwaway.

    The routed caller trims those tiles, but the kernel's contract is that a
    negative id is clamped rather than read out of bounds — the fused lanes
    make the same promise.
    """
    k, n = 256, 128
    a = torch.randn(2 * TILE_M, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(3, n, k, device=DEV, dtype=torch.bfloat16)
    ids = torch.tensor([2, -1], dtype=torch.int32, device=DEV)
    out = ops.cb_bf16_grouped_mm_sm120(a, weights, ids, TILE_M)
    assert torch.isfinite(out).all()
    expected = F.linear(a[:TILE_M], weights[2])
    assert _rel_l2(out[:TILE_M], expected.float()) <= 2e-3


@sm120
def test_sm120_out_variant_writes_the_same_bytes_in_place():
    torch.manual_seed(5)
    counts = [200, 0, 60]
    k, n = 512, 256
    a = torch.randn(sum(counts), k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(len(counts), n, k, device=DEV, dtype=torch.bfloat16)
    expert_ids, source = _padded_layout(counts)
    zero_extended = torch.cat([a, a.new_zeros((1, k))])
    gather = torch.where(source < 0, torch.full_like(source, a.shape[0]),
                         source)
    a_pad = zero_extended.index_select(0, gather).contiguous()

    allocating = ops.cb_bf16_grouped_mm_sm120(a_pad, weights, expert_ids,
                                              TILE_M)
    into = torch.empty_like(allocating)
    ops.cb_bf16_grouped_mm_sm120_out(into, a_pad, weights, expert_ids, TILE_M)
    assert torch.equal(into, allocating)

    # A contiguous ROW SLICE is what the chunked serving path passes.
    half = into.shape[0] // 2
    if half % TILE_M == 0 and half:
        chunk = torch.empty_like(allocating)
        ops.cb_bf16_grouped_mm_sm120_out(
            chunk[:half], a_pad[:half], weights,
            expert_ids[:half // TILE_M].contiguous(), TILE_M)
        assert torch.equal(chunk[:half], allocating[:half])


@sm120
@pytest.mark.parametrize("mutate,message", [
    (lambda a, w, e: (a[:-1], w, e), "multiple of the grouped tile_m"),
    (lambda a, w, e: (a, w, e[:-1]), "expert_ids must be contiguous"),
    (lambda a, w, e: (a, w, e.cpu()), "must be CUDA tensors"),
    (lambda a, w, e: (a, w, e.to(torch.int64)),
     "expert_ids must be contiguous"),
    (lambda a, w, e: (a.float(), w, e), "must be BF16"),
    (lambda a, w, e: (a, w.transpose(1, 2).contiguous().transpose(1, 2), e),
     "fully contiguous"),
], ids=["ragged-mp", "short-ids", "cpu-ids", "int64-ids", "fp32-a",
        "strided-weights"])
def test_sm120_rejects_contract_violations(mutate, message):
    k, n = 256, 128
    a = torch.randn(TILE_M, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(2, n, k, device=DEV, dtype=torch.bfloat16)
    ids = torch.zeros(1, dtype=torch.int32, device=DEV)
    bad_a, bad_w, bad_ids = mutate(a, weights, ids)
    with pytest.raises(RuntimeError, match=message):
        ops.cb_bf16_grouped_mm_sm120(bad_a, bad_w, bad_ids, TILE_M)


# ===========================================================================
# The IN-MAINLOOP A-ROW GATHER mode of the same collective.
#
# Its contract is BIT-IDENTITY with the padded-copy mode: the producer loads
# the same smem bytes (indexed cp.async with zero-fill instead of TMA over a
# materialized padded copy), the consumer/MMA/epilogue are the same code, so
# the outputs must be equal to the bit — a stronger gate than the tolerance
# band, and it is exactly why the gather mode is not a requalification event.
# ===========================================================================


def _row_src32(source):
    """The padded-copy layout's source vector, as the gather kernel takes it.

    Padding rows may carry ANY id outside [0, S); -1 is what the routed
    layout uses, so that is what the gate feeds."""
    return source.to(torch.int32)


@sm120
@pytest.mark.parametrize("counts,k,n", [
    ([0, 3, 0, 300, 0, 129, 1, 0], 512, 256),          # uneven + empty
    ([0, 0, 0, 0, 0, 0, 0, 900], 4096, 512),           # single expert, long K
    ([128, 128, 128, 128, 128, 128, 128, 128], 256, 1024),  # exact multiples
    ([1] * 8, 256, 128),                               # one row per expert
    ([64, 65, 191, 192, 193, 0, 7, 256], 1024, 4096),  # tile boundaries
    ([65] * 64, 256, 128),                             # large grid: swizzle 8
    ([17, 0, 260, 33, 128, 5], 1032, 264),             # K-residue: K%64 != 0
    ([5, 2, 9], 8, 128),                               # K below one k-tile
], ids=["uneven-empty", "single-expert-longK", "exact-multiple",
        "one-row-each", "tile-boundaries", "large-grid-swizzle",
        "k-residue", "k-align-floor"])
def test_sm120_gather_matches_padded_copy_bitwise(counts, k, n):
    """Gather mode == padded-copy mode, to the BIT, on every routing shape.

    The K-residue case matters: the TMA path zero-fills the out-of-bounds
    K tail of its box, and the gather path's predicate must reproduce those
    zeros exactly or the last k-tile's MMA consumes different bytes.
    """
    torch.manual_seed(20260802)
    experts = len(counts)
    pairs = sum(counts)
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(experts, n, k, device=DEV, dtype=torch.bfloat16)

    padded_ref, source = _run_padded(a, weights, counts)
    expert_ids, _ = _padded_layout(counts)
    gathered = ops.cb_bf16_grouped_mm_sm120_gather(
        a, _row_src32(source), weights, expert_ids, TILE_M)

    assert gathered.shape == padded_ref.shape
    assert torch.equal(gathered, padded_ref), (
        "the in-mainloop gather loaded different bytes than the padded copy")


@sm120
def test_sm120_gather_reads_duplicated_rows_like_the_routed_operator():
    """row_src may name one source row many times (top_k duplication)."""
    torch.manual_seed(11)
    counts = [70, 70, 70]
    k, n = 512, 256
    source_rows = 70  # every expert reads the same 70 rows
    a = torch.randn(source_rows, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(len(counts), n, k, device=DEV,
                          dtype=torch.bfloat16)
    expert_ids, source = _padded_layout(counts)
    dup_source = torch.where(source < 0, source, source % source_rows)

    zext = torch.cat([a, a.new_zeros((1, k))])
    gather = torch.where(dup_source < 0,
                         torch.full_like(dup_source, source_rows), dup_source)
    a_pad = zext.index_select(0, gather).contiguous()
    padded_ref = ops.cb_bf16_grouped_mm_sm120(a_pad, weights, expert_ids,
                                              TILE_M)
    gathered = ops.cb_bf16_grouped_mm_sm120_gather(
        a, dup_source.to(torch.int32), weights, expert_ids, TILE_M)
    assert torch.equal(gathered, padded_ref)


@sm120
def test_sm120_gather_dense_e1_matches_the_padded_helper():
    """The dense helper's layout: row_src = arange(Mp), ids >= M read zeros."""
    torch.manual_seed(4242)
    m, k, n = 300, 1024, 3072
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    blocks = (m + TILE_M - 1) // TILE_M
    mp = blocks * TILE_M

    row_src = torch.arange(mp, dtype=torch.int32, device=DEV)
    ids = torch.zeros(blocks, dtype=torch.int32, device=DEV)
    gathered = ops.cb_bf16_grouped_mm_sm120_gather(
        a, row_src, weight.unsqueeze(0), ids, TILE_M)

    a_pad = torch.cat([a, a.new_zeros((mp - m, k))]).contiguous()
    padded_ref = ops.cb_bf16_grouped_mm_sm120(a_pad, weight.unsqueeze(0),
                                              ids, TILE_M)
    assert torch.equal(gathered, padded_ref)
    assert not gathered[m:].any(), "rows past M gather zeros"


@sm120
def test_sm120_gather_oob_ids_produce_zero_rows():
    """Any id outside [0, S) — negative or too large — is a zero row.

    The reference is the padded-copy MODE on the equivalent materialized
    tile (bit-identity is the gather contract); rows whose ids are -1, == S
    or far beyond S must all come out exactly as the zero rows the padded
    copy would carry there.
    """
    k, n = 256, 128
    a = torch.randn(10, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(2, n, k, device=DEV, dtype=torch.bfloat16)
    ids = torch.zeros(1, dtype=torch.int32, device=DEV)
    row_src = torch.full((TILE_M,), -1, dtype=torch.int32, device=DEV)
    row_src[0] = 3
    row_src[1] = 10        # == S: out of bounds
    row_src[2] = 2 ** 30   # far out of bounds
    out = ops.cb_bf16_grouped_mm_sm120_gather(a, row_src, weights, ids,
                                              TILE_M)
    assert torch.isfinite(out).all()
    a_pad = torch.zeros(TILE_M, k, device=DEV, dtype=torch.bfloat16)
    a_pad[0] = a[3]
    padded_ref = ops.cb_bf16_grouped_mm_sm120(a_pad, weights, ids, TILE_M)
    assert torch.equal(out, padded_ref)
    assert not out[1:].any(), "every out-of-range id must read a zero row"


@sm120
def test_sm120_gather_out_variant_writes_the_same_bytes_in_place():
    torch.manual_seed(5)
    counts = [200, 0, 60]
    k, n = 512, 256
    a = torch.randn(sum(counts), k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(len(counts), n, k, device=DEV, dtype=torch.bfloat16)
    expert_ids, source = _padded_layout(counts)
    allocating = ops.cb_bf16_grouped_mm_sm120_gather(
        a, _row_src32(source), weights, expert_ids, TILE_M)
    into = torch.empty_like(allocating)
    ops.cb_bf16_grouped_mm_sm120_gather_out(
        into, a, _row_src32(source), weights, expert_ids, TILE_M)
    assert torch.equal(into, allocating)


@sm120
@pytest.mark.parametrize("mutate,message", [
    (lambda a, r, w, e: (a, r[:-1], w, e), "multiple of the grouped tile_m"),
    (lambda a, r, w, e: (a, r.to(torch.int64), w, e),
     "row_src must be a contiguous int32"),
    (lambda a, r, w, e: (a, r.cpu(), w, e),
     "row_src must be a contiguous int32"),
    (lambda a, r, w, e: (a, r, w, e[:-1]), "expert_ids must be contiguous"),
    (lambda a, r, w, e: (a.float(), r, w, e), "must be BF16"),
], ids=["ragged-row-src", "int64-row-src", "cpu-row-src", "short-ids",
        "fp32-a"])
def test_sm120_gather_rejects_contract_violations(mutate, message):
    k, n = 256, 128
    a = torch.randn(40, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(2, n, k, device=DEV, dtype=torch.bfloat16)
    row_src = torch.arange(2 * TILE_M, dtype=torch.int32, device=DEV)
    ids = torch.zeros(2, dtype=torch.int32, device=DEV)
    bad = mutate(a, row_src, weights, ids)
    with pytest.raises(RuntimeError, match=message):
        ops.cb_bf16_grouped_mm_sm120_gather(*bad, TILE_M)


@sm120
def test_sm120_packed_expert_order_is_a_pure_block_permutation():
    """The swizzle-group-aligned expert ORDER changes no output bit.

    A tile's result depends on its A rows and its expert's B slice, never on
    the tile's position in the launch — position is scheduler order, the same
    thing the swizzle already permutes. The packed order must therefore give
    bit-identical per-row results after undoing the permutation. This is the
    bit gate for the tile-order policy the T=512 measurements motivate.
    """
    from gridbook.bf16_grouped_lane import pack_expert_blocks

    torch.manual_seed(20260802)
    counts = [130, 0, 61, 258, 64, 5, 129, 190]
    k, n = 512, 256
    pairs = sum(counts)
    a = torch.randn(pairs, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(len(counts), n, k, device=DEV,
                          dtype=torch.bfloat16)

    order, touched, minimum = pack_expert_blocks(counts, TILE_M, 8)
    assert sorted(order) == [e for e, r in enumerate(counts) if r]
    assert touched >= minimum

    def layout(expert_order):
        expert_ids, source = [], []
        starts = [0] * len(counts)
        s = 0
        for e, r in enumerate(counts):
            starts[e] = s
            s += r
        for e in expert_order:
            for b in range((counts[e] + TILE_M - 1) // TILE_M):
                expert_ids.append(e)
                for i in range(TILE_M):
                    idx = b * TILE_M + i
                    source.append(starts[e] + idx if idx < counts[e] else -1)
        return (torch.tensor(expert_ids, dtype=torch.int32, device=DEV),
                torch.tensor(source, dtype=torch.int32, device=DEV))

    natural = [e for e, r in enumerate(counts) if r]
    y = {}
    for name, expert_order in (("natural", natural), ("packed", order)):
        expert_ids, source = layout(expert_order)
        out = ops.cb_bf16_grouped_mm_sm120_gather(a, source, weights,
                                                  expert_ids, TILE_M)
        real = source >= 0
        dense = torch.empty((pairs, n), dtype=out.dtype, device=DEV)
        dense.index_copy_(0, source[real].long(), out[real])
        y[name] = dense
    assert torch.equal(y["natural"], y["packed"])


@sm120
@pytest.mark.parametrize("factor", [2, 4], ids=["double", "quadruple"])
def test_sm120_refuses_an_uncompiled_tile_m(factor):
    """A tile_m the module did not instantiate is refused, not approximated.

    Derived from the binding rather than hardcoded: the compiled rung is a
    schedule decision that has already changed once (128 -> 64), and a test
    naming a literal would silently start asserting the wrong thing.
    """
    uncompiled = TILE_M * factor
    assert uncompiled not in ext.cb_bf16_grouped_sm120_tile_sizes()
    k, n = 256, 128
    a = torch.randn(TILE_M, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(1, n, k, device=DEV, dtype=torch.bfloat16)
    ids = torch.zeros(1, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="compiles tile_m"):
        ops.cb_bf16_grouped_mm_sm120(a, weights, ids, uncompiled)
