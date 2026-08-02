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

import os

_FLAG = "PRISMAQUANT_CB_BF16_SM120"
_ALLOWED = frozenset(("", "0", "1"))
_STATE: list[str] = []


def requested() -> bool:
    """Whether the operator asked for the sm12x-native lane.

    Process-stable like every other Gridbook dispatch selector: a value that
    changes after dispatch was fixed would silently mix two reduction orders
    inside one run and make an A/B unreadable. A typo raises rather than
    quietly selecting the baseline.
    """
    current = os.environ.get(_FLAG, "").strip()
    if current not in _ALLOWED:
        raise ValueError(
            f"invalid {_FLAG}={current!r}; expected '1' to enable the "
            f"sm12x-native BF16 grouped lane, or '' / '0' for the default "
            f"SM80-schedule bridge")
    if not _STATE:
        _STATE.append(current)
    elif current != _STATE[0]:
        raise RuntimeError(
            f"{_FLAG} changed after Gridbook dispatch was fixed; restart the "
            f"process instead of mixing GEMM reduction orders within one run")
    return _STATE[0] == "1"


def _reset_for_tests() -> None:
    """Clear the process-stable latch (tests only)."""
    _STATE.clear()


def require_lane(operation: str = "this operation", *, device=None):
    """Return the extension whose sm12x lane is usable, or fail closed.

    Called at model load, never at first forward. Failing closed is the point:
    with the flag on, quietly serving the SM80 lane would produce a run whose
    numbers describe the wrong kernel.
    """
    from .cuda_ext import (NativeKernelUnavailableError,
                           bf16_grouped_sm120_buildable,
                           require_bf16_grouped_ext)

    ext = require_bf16_grouped_ext(operation)
    capability = None
    try:
        import torch

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 — reported below as "unavailable"
        capability = None
    # Every symbol the lane's FORWARD PATH dereferences, not just its GEMM
    # entry points. `cb_bf16_grouped_sm120_config` was missing here while
    # `swizzle_group()` (the packed expert order's group size) reads it on
    # every routed prefill — so a module this function called "complete" could
    # still AttributeError at first forward, which is precisely what attesting
    # at load is supposed to make impossible. cuda_ext's own
    # `_BF16_GROUPED_SM120_SYMBOLS` already required it; the two lists now agree.
    missing = [name for name in ("cb_bf16_grouped_mm_sm120",
                                 "cb_bf16_grouped_mm_sm120_out",
                                 "cb_bf16_grouped_mm_sm120_gather",
                                 "cb_bf16_grouped_mm_sm120_gather_out",
                                 "cb_bf16_grouped_sm120_tile_m",
                                 "cb_bf16_grouped_sm120_config")
               if not hasattr(ext, name)]
    if missing:
        raise NativeKernelUnavailableError(
            f"{operation} requested the sm12x-native BF16 grouped lane "
            f"({_FLAG}=1), but the loaded grouped-BF16 extension does not "
            f"carry it (missing {missing}; device capability {capability}). "
            f"The lane is compiled only for compute capability 12.0/12.1"
            + ("" if capability is None or
               bf16_grouped_sm120_buildable(capability)
               else f", and this device reports "
                    f"{capability[0]}.{capability[1]}")
            + f". Unset {_FLAG} to use the default SM80-schedule bridge; "
            f"Gridbook does not substitute a different kernel behind an "
            f"explicit lane selection.")
    return ext


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


def swizzle_group(ext) -> int:
    """The large-grid tile-scheduler swizzle the compiled lane uses.

    This is the group size the packed tile ORDER below aligns expert
    boundaries to; below the kernel's grid threshold the scheduler runs
    swizzle 1 and the order is measured neutral, so one order serves both
    regimes.
    """
    return int(ext.cb_bf16_grouped_sm120_config()[8])


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
