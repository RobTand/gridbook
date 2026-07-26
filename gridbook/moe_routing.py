"""Padded, block-aligned MoE routing for the ROUND-2 grouped-fused prefill.

Lives in its own module rather than in ``moe.py`` because ``moe.py`` imports
vLLM at module scope: keeping this torch-only makes the routing construction —
the part with all the index arithmetic and therefore all the risk — testable in
the build venv on CPU, with no vLLM and no GPU. ``moe.py`` re-exports it.
"""
from __future__ import annotations

import torch


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
