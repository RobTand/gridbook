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


def _expert_counts(pair_expert: torch.Tensor, E: int) -> torch.Tensor:
    """``[E]`` int64 routed-pair counts per expert, WITHOUT a host sync.

    scatter_add_, NOT bincount. ATen's CUDA ``bincount`` sizes its output from
    ``self.min().item()`` / ``self.max().item()``, so it host-syncs and cannot
    be captured ("Cannot copy between CPU and CUDA tensors during CUDA graph
    capture", measured on torch 2.11.0+cu130). The persistent-B lane hit this
    exact trap: its docstring claimed no host read while a bincount two lines
    away made the operator uncapturable, and the fix — this form — is now gated
    with a negative control (tests/test_cb_moe_persistent_b.py). The padded
    tile-indexed lanes carried the same latent bincount until 2026-08-02; the
    "NO HOST READS" paragraph below was false for as long as they did.

    Pure device work at a static shape, producing the identical integers.
    """
    return torch.zeros(E, dtype=torch.int64, device=pair_expert.device) \
        .scatter_add_(0, pair_expert, torch.ones_like(pair_expert))


def cb_grouped_block_offsets(topk_ids: torch.Tensor, E: int, tile_m: int):
    """``[E+1]`` cumulative TileM-block offsets of the padded row layout.

    ``block_offsets[e]`` is the first padded M-tile owned by expert ``e`` in
    the layout :func:`cb_grouped_pad_routing` builds, so a contiguous expert
    chunk ``[c0, c1)`` owns tiles ``[block_offsets[c0], block_offsets[c1])``
    and rows ``[block_offsets[c0]*tile_m, block_offsets[c1]*tile_m)``.
    ``block_offsets[E]`` is the real block total (what ``n_blocks`` reports).

    Callers that CHUNK the expert dimension — the BF16 bridge bounds its
    decoded weight transient that way — need these boundaries on the HOST to
    slice each launch, which costs one device read. Kept out of
    :func:`cb_grouped_pad_routing` so the fused paths, which launch the whole
    collective at once, keep their single optional sync.
    """
    counts = _expert_counts(topk_ids.reshape(-1).to(torch.long), E)
    blocks_e = (counts + (tile_m - 1)) // tile_m
    return torch.cat([blocks_e.new_zeros(1), torch.cumsum(blocks_e, 0)])


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
    latter has a data-dependent output size and would itself sync — and the
    per-expert counts come from :func:`_expert_counts`, not ``bincount``, for
    the same reason. Until 2026-08-02 this paragraph was FALSE: a
    ``torch.bincount`` here host-synced on every call, so the padded grouped
    lanes could not be captured no matter what their callers did.
    """
    dev = topk_ids.device
    pair_expert = topk_ids.reshape(-1).to(torch.long)              # [P]
    P = pair_expert.numel()
    cap_blocks = P // tile_m + E

    counts = _expert_counts(pair_expert, E)                        # [E]
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


# ===========================================================================
# K0.4 — the grouped-MoE TileM SELECTOR.
#
# It replaces two hand choices: the FP8 grouped path resolved ``tile_m=None``
# to the kernel's compiled default (so serving could never reach 256 at all),
# and the FP4 grouped path read its tile off the SUFFIX of an activation POLICY
# env string ("static_lsq256") — a performance knob riding on a numerics
# selector.
#
# It lives here, not in moe.py, for the reason this module exists: it is pure
# integer arithmetic over shapes, so it is testable on CPU with no vLLM and no
# GPU — which is where the risk in this kind of code actually is.
# ===========================================================================

# Keep in lockstep with the compiled tile shapes — csrc/cb_fused_gemm.cu
# (MoeTile<TM>, moe_tile_supported) and csrc/cb_fused_fp4_gemm.cu. Python never
# ASSUMES the compiled set: the caller passes what the extension reports
# (``moe._gf2_tile_sizes``). These two names exist so the arithmetic can talk
# about the pair without hardcoding a ladder.
GROUPED_TILE_M_BASE = 128
GROUPED_TILE_M_WIDE = 256

# TileN is a property of the compiled tile shape, not of the layer: the fp8
# grouped tile is Shape<TM,_64,_128>; the fp4 grouped tile is
# Shape<TM,_128,_128> (TileN pinned at 128 by the blockscaled scale-factor smem
# atom).
GROUPED_TILE_N = {"fp8": 64, "fp4": 128}

# The one calibrated constant. Derivation (also in docs/KERNELS.md):
#   Per CTA at tile t: decode TileN*K weights ONCE, then issue t*TileN*K MACs.
#   With d = per-tile decode cost and m = per-row MMA+traffic cost,
#   T(t) ~ B(t) * (d + t*m), where B(t) = sum_e ceil(c_e/t) is the M-tile count.
#   Exact padding lemma: pad_256(c) - pad_128(c) = 128 iff (c mod 256) lies in
#   [1,128], else 0. So with q = #{e : (c_e mod 256) in [1,128]},
#   B(128) = 2*B(256) - q, and TileM=256 wins iff (B128 - q)/q > 256/x for
#   x = d/m. The decode:MMA ratio is 1:t independent of N and K, so BOTH stages
#   give the same condition. Minimising the left side over EVERY histogram with
#   sum c_e = P (worst case c_e = 128 mod 256, which maximises q and minimises
#   B128) gives the host-knowable sufficient condition
#       rho = P/E  >  128 * (1 + 256/x).
#   Inverting the dense fused TileM A/B (22.57-50.32% at fixed occupancy =>
#   2(x+128)/(x+256) in [1.226, 1.503]) bounds x to [75, 259], putting the
#   threshold in [254, 565]. 512 is the pessimistic end of that interval.
# PROPOSAL DATA for this kernel family until a routed sweep pins it on the
# grouped lanes; PRISMAQUANT_CB_GROUPED_TILE_M overrides for measurement.
GROUPED_WIDE_TILE_MIN_ROWS_PER_EXPERT = 512

# (tile_m, k_bits) the extension compiles at ZERO shared-memory margin.
# cb_fused_gemm.cu: "TileM=256/k_bits=32 lands on EXACTLY the 101376-byte
# ceiling (zero margin) — it is compiled, but must be launch-verified before
# being trusted." Until that verification lands the SELECTOR will not choose
# it; an explicit operator override still can.
GROUPED_TILE_M_UNVERIFIED = frozenset({(GROUPED_TILE_M_WIDE, 32)})

_SM_COUNTS: dict[int, int] = {}


def cb_sm_count(device) -> int:
    """Multiprocessor count for ``device``, cached; 0 when unavailable.

    ``get_device_properties`` reads cached runtime metadata and does NOT
    synchronize the device; caching it here also keeps that query out of
    steady-state dispatch. A failed probe caches 0, so every selector that
    consumes it fails closed to the narrow tile.
    """
    if getattr(device, "type", None) != "cuda":
        return 0
    try:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        index = int(index)
    except Exception:  # noqa: BLE001 — optional optimization, fail closed
        return 0
    cached = _SM_COUNTS.get(index)
    if cached is not None:
        return cached
    try:
        count = int(torch.cuda.get_device_properties(index)
                    .multi_processor_count)
        if count <= 0:
            count = 0
    except Exception:  # noqa: BLE001 — optional optimization, fail closed
        count = 0
    _SM_COUNTS[index] = count
    return count


def cb_grouped_tile_m(*, tokens: int, top_k: int, experts: int,
                      hidden: int, inter: int, tile_n: int,
                      compiled, sm_count: int, k_bits: int = 0,
                      min_rows_per_expert: int =
                      GROUPED_WIDE_TILE_MIN_ROWS_PER_EXPERT) -> int:
    """Grouped-fused TileM, from HOST-KNOWN integers only.

    CUDA-GRAPH SAFETY — the claim, and why it holds by construction.

    Every input is a Python ``int``. ``tokens``/``top_k`` come from
    ``topk_ids.shape`` (host tensor METADATA, never values); ``experts``,
    ``hidden``, ``inter`` and ``k_bits`` are layer constants fixed at load;
    ``tile_n`` and ``compiled`` are properties of the BUILD (the extension's own
    per-rung tile list, never a hardcoded set); ``sm_count`` is the cached,
    non-synchronizing device-property read. No ATen op runs on a device tensor,
    so there is no device->host copy and hence no synchronization on any path —
    and nothing for capture to trip over.

    That matters because ``tile_m`` decides BOTH the kernel symbol
    (``run_moe_grouped<TM,KB>`` is a distinct ``__global__`` per TM) and every
    routing tensor's SHAPE (``cap_blocks = P//tile_m + E``). Both must be fixed
    before the first node is recorded; a selector over host-known values is
    resolved at record time by definition, so one capture gets one tile, one
    symbol, one grid and one shape set.

    THE ROUTED HISTOGRAM IS DELIBERATELY NOT AN INPUT. It is device data, and
    ``tile_m`` is an INPUT to :func:`cb_grouped_pad_routing` while the trim read
    is an OUTPUT of it — so a histogram-reading selector must run strictly
    upstream, and its read would be a NEW, EARLIER sync that cannot be folded
    into the one the trim already spends. Under capture it would make the
    launched symbol and every routing shape a function of device data, which is
    not a limitation to work around but outside what a graph is. This is the
    persistent-B ``scatter_add_`` lesson applied before the fact instead of
    after: that lane's "no host read" claim was false because a ``bincount`` two
    lines away host-synced, and the repair was to REMOVE the read, not to guard
    it. (The same latent ``bincount`` sat in this module until 2026-08-02; see
    :func:`_expert_counts`.)

    PRICE. A histogram-free rule must hold for EVERY histogram consistent with
    ``(P, E)``, so it fires the wide tile later than an oracle would — rho about
    ``128 + 32768/x`` against ``16384/x``. Both thresholds sit above ordinary
    chunked-prefill batch sizes, so the practical loss is nil while the
    sync-freedom is permanent.

    FAIL-CLOSED ESTIMATION. Every device-unknown quantity enters at the end of
    its provable interval that favours the INCUMBENT narrow tile: the number of
    routed experts becomes ``E`` (maximum padding), and the grid becomes
    ``ceil(P/t)`` (minimum occupancy). Both criteria are monotone in that
    direction, so a wide-tile verdict is justified UNIFORMLY — for every
    histogram — rather than on average.

    Returns a member of ``compiled``, or ``0`` when the build offers no tile —
    which the caller must treat as "no fused route", never as "use the default".
    """
    legal = sorted({int(t) for t in compiled if int(t) > 0})
    if not legal:
        return 0
    base = GROUPED_TILE_M_BASE if GROUPED_TILE_M_BASE in legal else legal[0]
    wide = GROUPED_TILE_M_WIDE
    if wide not in legal:
        return base                       # not compiled for this rung/build
    if k_bits and (wide, int(k_bits)) in GROUPED_TILE_M_UNVERIFIED:
        return base                       # zero smem margin, not yet verified
    if sm_count <= 0:
        return base                       # device metadata unavailable

    pairs = int(tokens) * int(top_k)      # P, exactly host-known
    if pairs < wide:
        return base                       # shape guard (mirrors the dense one)
    rho = pairs // max(1, int(experts))
    if rho <= int(min_rows_per_expert):
        return base

    # OCCUPANCY. Both projection stages share ONE tile and ONE row layout
    # (``expert_ids`` is built once and reused), but their N differs — stage 1
    # is N=2*inter, stage 2 is N=hidden. The NARROWER projection bounds the
    # grid, and ceil(P/t) lower-bounds the real tile count for every histogram,
    # so this is a lower bound on the true occupancy of the worse stage.
    n_min = min(2 * int(inter), int(hidden))
    grid_lo = ((pairs + wide - 1) // wide) * ((n_min + tile_n - 1) // tile_n)
    occupancy_floor = (2 * int(sm_count) + 2) // 3
    return wide if grid_lo >= occupancy_floor else base
