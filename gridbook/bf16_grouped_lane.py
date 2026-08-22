"""Selection and attestation of the sm12x-native BF16 grouped lane.

Gridbook's quality-preserving prefill bridge has two compiled lanes (see
``csrc/cb_bf16_grouped_gemm.cu``):

* the DEFAULT device-scheduled CUTLASS 2.x ``DefaultGemmGrouped`` on an
  ``arch::Sm80`` schedule, which takes exact per-expert segments; and
* the sm12x-native CUTLASS 3.x collective (TMA warp-specialized mainloop,
  row-padded tile-indexed grouping), compiled only on cc 12.x.

The second is **OPT-IN** behind ``PRISMAQUANT_CB_BF16_SM120=1``, per
[NATIVE-PARITY](../docs/NATIVE-PARITY.md): it is bit-gated against the torch
reference but has not been through the served grouped-MoE protocol, and it
changes the FP32 REDUCTION ORDER of every default NVFP4-CB prefill. Nothing
here runs, probes or builds anything when the flag is unset — with the flag
off the dispatch is byte-for-byte what it was.

This module owns three things so ``linear.py`` and ``moe.py`` share one
answer: the process-stable flag read, the load-time attestation (the repo
convention is that nothing resolves at first forward), and the dense
``E=1`` padding helper.
"""
from __future__ import annotations

from . import lane_select

_FLAG = "PRISMAQUANT_CB_BF16_SM120"


def requested() -> bool:
    """Whether the operator asked for the sm12x-native lane.

    Process-stable like every other Gridbook dispatch selector: a value that
    changes after dispatch was fixed would silently mix two reduction orders
    inside one run and make an A/B unreadable. A typo raises rather than
    quietly selecting the baseline.
    """
    return lane_select.latched_bool(
        _FLAG, meaning="the sm12x-native BF16 grouped lane")


def _reset_for_tests() -> None:
    """Clear the process-stable latch (tests only)."""
    lane_select.reset_for_tests(_FLAG)


def require_lane(operation: str = "this operation", *, device=None):
    """Return the extension whose sm12x lane is usable, or fail closed.

    Called at model load, never at first forward. Failing closed is the point:
    with the flag on, quietly serving the SM80 lane would produce a run whose
    numbers describe the wrong kernel.

    The symbol list is ``cuda_ext``'s own strict tuple, imported rather than
    restated. This function used to carry a six-name local copy while the
    loader enforced seven, under a comment asserting "the two lists now agree"
    — the exact drift a shared constant makes impossible. It is passed through
    ``lane_select.require_lane`` so the device check, which this lane used to
    compute and then discard, is real for all three lanes at once.
    """
    from .cuda_ext import (_BF16_GROUPED_SM120_SYMBOLS,
                           bf16_grouped_sm120_buildable,
                           get_bf16_grouped_ext, require_bf16_grouped_ext)

    # Keep the bridge's own fail-closed diagnostic for "no module at all": it
    # names the nvcc hint and the default route, which a lane-level message
    # would not.
    require_bf16_grouped_ext(operation)
    return lane_select.require_lane(
        operation, flag=_FLAG,
        lane="the sm12x-native BF16 grouped lane",
        source="grouped-BF16 extension (cb_bf16_grouped_gemm.cu)",
        alternative="the default SM80-schedule bridge",
        get_ext=get_bf16_grouped_ext,
        symbols=_BF16_GROUPED_SM120_SYMBOLS,
        buildable=bf16_grouped_sm120_buildable,
        device=device)


def tile_m(ext) -> int:
    """The lane's row-padding granularity, as the kernel reports it."""
    return int(ext.cb_bf16_grouped_sm120_tile_m())


def dense_mm(ext, a, weight):
    """Dense ``E=1`` GEMM through the sm12x lane.

    ``a`` is ``[M, K]`` BF16 and ``weight`` is ``[N, K]`` BF16; returns
    ``[M, N]``. When the extension carries the in-mainloop A-row gather mode
    (every cc 12.x build since it was added), no padded copy exists at all:
    the kernel reads rows ``0..M-1`` directly and the rounded-up remainder of
    the last tile reads zeros (``row_src`` ids ``>= M``). Older stubs without
    the gather entry point fall back to materializing the zero-padded copy —
    the two are bit-identical (the kernel-level gate asserts it).
    """
    import torch

    granularity = tile_m(ext)
    m = int(a.shape[0])
    blocks = (m + granularity - 1) // granularity
    padded = blocks * granularity
    expert_ids = torch.zeros(blocks, dtype=torch.int32, device=a.device)
    if hasattr(ext, "cb_bf16_grouped_mm_sm120_gather"):
        from .ops import cb_bf16_grouped_mm_sm120_gather

        row_src = torch.arange(padded, dtype=torch.int32, device=a.device)
        y = cb_bf16_grouped_mm_sm120_gather(a.contiguous(), row_src,
                                            weight.unsqueeze(0), expert_ids,
                                            granularity)
        return y[:m]

    from .ops import cb_bf16_grouped_mm_sm120

    if padded != m:
        a = torch.cat([a, a.new_zeros((padded - m, a.shape[1]))])
    y = cb_bf16_grouped_mm_sm120(a.contiguous(), weight.unsqueeze(0),
                                 expert_ids, granularity)
    return y[:m]


# Index of the swizzle field inside ``cb_bf16_grouped_sm120_config()``. Named
# rather than spelled as a bare ``[8]`` at the read below — the kernel returns
# a flat int vector, so an inserted field would silently shift the meaning of a
# positional index and the packed expert ORDER would be aligned to the wrong
# number with no error anywhere. The sibling lanes validate their config reads
# the same way (``fp4v2_fused_midm_lane._facts``, ``moe_persistent_b_lane
# .resolve_cfg``).
_CONFIG_SWIZZLE_INDEX = 8


def swizzle_group(ext) -> int:
    """The large-grid tile-scheduler swizzle the compiled lane uses.

    This is the group size the packed tile ORDER below aligns expert
    boundaries to; below the kernel's grid threshold the scheduler runs
    swizzle 1 and the order is measured neutral, so one order serves both
    regimes.
    """
    config = ext.cb_bf16_grouped_sm120_config()
    if len(config) <= _CONFIG_SWIZZLE_INDEX:
        raise IndexError(
            f"cb_bf16_grouped_sm120_config() returned {len(config)} fields; "
            f"the swizzle is field {_CONFIG_SWIZZLE_INDEX}. The loaded module "
            f"and this reader disagree about the config layout")
    return int(config[_CONFIG_SWIZZLE_INDEX])


def pack_expert_blocks(counts, tile_m, group):
    """Deterministic expert ORDER aligning expert boundaries to swizzle groups.

    The tile scheduler processes M-tiles in groups of ``group`` (its
    large-grid swizzle): all tiles of a group sweep the N dimension together,
    so a B (expert weight) slice is fetched from DRAM once per GROUP its
    tiles touch, not once per tile. With the natural expert-major order an
    expert's tiles regularly straddle a group boundary and its B slice is
    fetched twice. This packs experts into groups first-fit by padded block
    count (largest fitting expert first, smallest spills when none fits), so
    group boundaries coincide with expert boundaries wherever the block
    histogram allows.

    Pure host math on the routing histogram; the order permutes WHOLE expert
    segments only, never rows within an expert, so each padded row's GEMM
    result is unchanged — tile order is scheduler order, which the kernel
    already treats as bit-neutral (same guarantee as the swizzle itself).

    Returns ``(order, groups_touched, groups_minimum)`` — the last two are
    dispatch telemetry: how many swizzle groups the packed order's expert
    tile-ranges touch, against the unpackable minimum ``sum(ceil(b/group))``.
    """
    blocks = [(int(e), (int(r) + tile_m - 1) // tile_m)
              for e, r in enumerate(counts) if int(r) > 0]
    blocks.sort(key=lambda t: (-t[1], t[0]))
    remaining = list(blocks)
    order, cap = [], 0
    while remaining:
        if cap == 0:
            cap = group
        pick = None
        for i, (_, b) in enumerate(remaining):
            if b <= cap:
                pick = i
                break
        if pick is None:
            pick = len(remaining) - 1  # smallest remaining; spills the group
        expert, b = remaining.pop(pick)
        order.append(expert)
        cap = (cap - b) % group
    by_expert = dict(blocks)
    pos, touched, minimum = 0, 0, 0
    for expert in order:
        b = by_expert[expert]
        touched += (pos + b - 1) // group - pos // group + 1
        minimum += (b + group - 1) // group
        pos += b
    return order, touched, minimum


def pack_expert_blocks_chunked(counts, tile_m, group, chunk):
    """``pack_expert_blocks`` over each consecutive ``chunk``-wide subrange.

    A layer whose experts do not fit ONE decode chunk is served as several
    grouped launches, each over its own contiguous block range
    ``[block_off[c0], block_off[c1])`` with the weight stack expanded in
    EXPERT-ID order for ``[c0, c1)`` and blocks remapped to local ids as
    ``expert_ids - c0``. A globally packed order would hand a chunk blocks of
    FOREIGN experts — rows multiplied by the wrong weight slice — so the
    constraint the packing must respect is range membership, not row order:
    each chunk packs ITS OWN experts only. Every ``[c0, c1)`` range then keeps
    exactly the blocks it started with, merely reordered, and the alignment
    win applies per chunk (ROADMAP K1.5).

    ``chunk=None`` (or ``chunk >= len(counts)``) is the whole-layer packing
    and delegates to :func:`pack_expert_blocks` unchanged. Pure host math like
    the single-range version: the counts arrive as Python ints from the
    block-offset host read the caller already paid for, so nothing here may
    touch a device tensor. Returns ``(order, touched, minimum)`` with the
    order concatenated across chunks (original expert ids) and the telemetry
    summed.
    """
    if chunk is None:
        return pack_expert_blocks(counts, tile_m, group)
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError(f"chunk width must be >= 1, got {chunk}")
    experts = len(counts)
    if chunk >= experts:
        return pack_expert_blocks(counts, tile_m, group)
    order, touched, minimum = [], 0, 0
    for c0 in range(0, experts, chunk):
        c1 = min(experts, c0 + chunk)
        sub_order, sub_touched, sub_minimum = pack_expert_blocks(
            counts[c0:c1], tile_m, group)
        order.extend(e + c0 for e in sub_order)
        touched += sub_touched
        minimum += sub_minimum
    return order, touched, minimum
