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
    tile_m, tile_n, tile_k, stages, smem, capacity = \
        ext.cb_bf16_grouped_sm120_config()
    assert (tile_m, tile_n, tile_k) == (128, 128, 64)
    assert stages >= 2, "a TMA warp-specialized mainloop needs two stages"
    assert 0 < smem <= capacity
    assert ext.cb_bf16_grouped_sm120_tile_sizes() == [tile_m]


@sm120
@pytest.mark.parametrize("counts,k,n", [
    ([0, 3, 0, 300, 0, 129, 1, 0], 512, 256),          # uneven + empty
    ([0, 0, 0, 0, 0, 0, 0, 900], 4096, 512),           # single expert, long K
    ([128, 128, 128, 128, 128, 128, 128, 128], 256, 1024),  # exact multiples
    ([1] * 8, 256, 128),                               # one row per expert
    ([64, 65, 191, 192, 193, 0, 7, 256], 1024, 4096),  # tile boundaries
], ids=["uneven-empty", "single-expert-longK", "exact-multiple",
        "one-row-each", "tile-boundaries"])
def test_sm120_error_matches_a_per_segment_bf16_reference(counts, k, n):
    """The lane's fp32-accumulate error may not exceed a BF16 F.linear's.

    Every routing shape the served operator can produce: empty experts (absent
    from ``expert_ids`` by construction), a single expert, segments that are
    exact tile multiples, one-row segments (127 padding rows per tile), and
    lengths straddling the tile boundary at 64/65/191/192/193.
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


@sm120
def test_sm120_refuses_an_uncompiled_tile_m():
    k, n = 256, 128
    a = torch.randn(TILE_M, k, device=DEV, dtype=torch.bfloat16)
    weights = torch.randn(1, n, k, device=DEV, dtype=torch.bfloat16)
    ids = torch.zeros(1, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="compiles tile_m"):
        ops.cb_bf16_grouped_mm_sm120(a, weights, ids, 64)
