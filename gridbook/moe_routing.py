"""Torch-only MoE routing construction and immutable routing constants.

Lives in its own module rather than in ``moe.py`` because ``moe.py`` imports
vLLM at module scope: keeping this torch-only makes the routing construction —
the part with all the index arithmetic and therefore all the risk — testable in
the build venv on CPU, with no vLLM and no GPU.  The cached tensors below are
likewise pure routing/expander metadata: read-only after construction and keyed
by exact shape and device. ``moe.py`` consumes them directly.
"""
from __future__ import annotations

import threading
import weakref

import torch


# Deduplicate the much larger row-zero constants across identically shaped
# layers without making this module their owner.  Each consuming layer keeps a
# strong reference in its private cache (which is what stabilises pointers for
# graph capture); this process-wide index is weak so unloading the last model
# also releases the Tensor objects.  The lock covers the rare first creation so
# concurrent layer initialisation cannot allocate duplicate GPU buffers.
_ROW_OFFSET_POOL: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_ROW_OFFSET_POOL_LOCK = threading.Lock()
_LAYER_CACHE_INIT_LOCK = threading.Lock()


def _layer_tensor_cache(layer, attr: str) -> dict:
    """Return a private per-layer tensor cache, creating it on first use."""
    cache = getattr(layer, attr, None)
    if cache is None:
        # Layer warmup is normally serial, but make first use convergent even
        # when two request threads reach a previously untouched layer together.
        with _LAYER_CACHE_INIT_LOCK:
            cache = getattr(layer, attr, None)
            if cache is None:
                cache = {}
                setattr(layer, attr, cache)
    if not isinstance(cache, dict):
        raise TypeError(f"{attr} must be a dict, got {type(cache).__name__}")
    return cache


def cb_cached_expert_map(layer, c0: int, c1: int, E: int,
                         device) -> torch.Tensor:
    """Read-only global-to-local map for stock-prefill expert chunk ``[c0,c1)``.

    The chunk loop's bounds and expert count are Python/config integers, and a
    model layer does not change device while serving.  The exact tuple is the
    cache key nevertheless: a changed chunk override, a moved layer, or two
    same-sized but differently positioned chunks can never alias.  Retaining
    every exact tensor (rather than growing/replacing one buffer) also keeps a
    captured graph's pointer alive for the layer's lifetime.
    """
    c0, c1, E = int(c0), int(c1), int(E)
    if not 0 <= c0 <= c1 <= E:
        raise ValueError(f"invalid expert chunk [{c0}, {c1}) for E={E}")
    dev = torch.device(device)
    cache = _layer_tensor_cache(layer, "_cb_stock_expert_maps")
    key = (dev, c0, c1, E)
    cached = cache.get(key)
    if cached is not None:
        return cached
    value = torch.full((E,), -1, dtype=torch.int32, device=dev)
    value[c0:c1] = torch.arange(c1 - c0, dtype=torch.int32, device=dev)
    # ``setdefault`` makes a concurrent first construction converge on one
    # persistent object.  Either contender is byte-identical and read-only.
    return cache.setdefault(key, value)


def cb_cached_row_offsets(layer, rows: int, device) -> torch.Tensor:
    """Return an exact-size, read-only int32 zero vector for CB expanders.

    Every row in a stacked stock-prefill decode uses codebook row zero.  The old
    path allocated and zero-filled this constant once per projection and chunk.
    Exact-size/device keys avoid stride tricks, mutable slicing, cross-device
    reuse, and pointer replacement under graph capture.  Layers retain strong
    references, while a weak process-wide pool shares identical immutable
    buffers across layers and stops owning them when the last model unloads.
    """
    rows = int(rows)
    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    dev = torch.device(device)
    cache = _layer_tensor_cache(layer, "_cb_row_offset_cache")
    key = (dev, rows)
    cached = cache.get(key)
    if cached is not None:
        return cached
    with _ROW_OFFSET_POOL_LOCK:
        value = _ROW_OFFSET_POOL.get(key)
        if value is None:
            value = torch.zeros(rows, dtype=torch.int32, device=dev)
            _ROW_OFFSET_POOL[key] = value
    return cache.setdefault(key, value)


def cb_grouped_pad_routing(topk_ids: torch.Tensor, E: int, tile_m: int):
    """Build the block-aligned padded row layout the grouped CUTLASS kernel
    consumes: every expert's rows start on a TileM boundary, so an M-tile
    belongs to exactly ONE expert and the kernel can select its B operand from
    a per-tile expert id.

    Returns ``(expert_ids, row_src, is_pad, n_blocks)``:

    * ``expert_ids`` ``[cap_blocks]`` int32 — the expert owning each M-tile,
      ``-1`` for tiles past the real total (unused capacity).
    * ``row_src`` ``[cap_blocks*tile_m]`` int64 — for each padded row, the index
      into the STABLE-argsorted pair array it gathers from. Padding rows carry 0
      (a safe in-range sentinel); use ``is_pad`` to neutralise them.
    * ``is_pad`` ``[cap_blocks*tile_m]`` bool.
    * ``n_blocks`` 0-dim int64 CUDA/CPU tensor — the REAL block count
      ``sum_e ceil(c_e/tile_m)``. Reading it costs one sync; the caller decides.

    ORDER. Callers must gather through a STABLE ``argsort(pair_expert)``, which
    leaves each expert's rows in token-major order — the loop path's
    ``torch.where(topk_ids == e)`` order — so per-segment GEMMs bit-match the
    per-expert loop and only the combine reassociates.

    STATIC CAPACITY. ``cap_blocks = P // tile_m + E`` with ``P = T*top_k``, and
    this bound is known from SHAPES alone (no device read). Proof: only nonzero
    experts consume blocks; for the ``n <= E`` of them,
    ``ceil(c_e/tile_m) <= floor((c_e - 1)/tile_m) + 1``, so the total is at most
    ``n + sum_e floor((c_e - 1)/tile_m) <= n + floor((P - n)/tile_m)
    <= E + floor(P/tile_m)``, using ``floor(a) + floor(b) <= floor(a + b)`` and
    ``sum_e (c_e - 1) = P - n``.

    NO HOST READS. Block->expert is a ``searchsorted`` over the block-offset
    cumsum rather than a ``repeat_interleave`` by a tensor count, because the
    latter has a data-dependent output size and would itself sync.
    """
    dev = topk_ids.device
    pair_expert = topk_ids.reshape(-1).to(torch.long)              # [P]
    P = pair_expert.numel()
    cap_blocks = P // tile_m + E

    counts = torch.bincount(pair_expert, minlength=E)              # [E]
    pair_off = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    blocks_e = (counts + (tile_m - 1)) // tile_m                   # [E]
    block_off = torch.cat([blocks_e.new_zeros(1),
                           torch.cumsum(blocks_e, 0)])             # [E+1]
    n_blocks = block_off[E]

    bpos = torch.arange(cap_blocks, device=dev, dtype=torch.long)
    # First e with block_off[e+1] > b == the owner of block b; returns E for
    # blocks past the real total. Duplicate cumsum entries (zero-row experts,
    # which consume zero blocks) are skipped by construction.
    eid = torch.searchsorted(block_off[1:].contiguous(), bpos, right=True)
    live_blk = eid < E
    eid_c = eid.clamp(max=E - 1)
    expert_ids = torch.where(live_blk, eid, torch.full_like(eid, -1)) \
        .to(torch.int32)

    r = torch.arange(cap_blocks * tile_m, device=dev, dtype=torch.long)
    b = r // tile_m
    e_row = eid_c[b]
    # Rank of this padded row WITHIN its expert's segment: whole blocks before
    # it inside the segment, plus its offset in its own block.
    rank = (b - block_off[e_row]) * tile_m + (r - b * tile_m)
    is_pad = (~live_blk[b]) | (rank >= counts[e_row])
    row_src = torch.where(is_pad, torch.zeros_like(rank),
                          pair_off[e_row] + rank)
    return expert_ids, row_src, is_pad, n_blocks
