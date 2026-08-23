"""Independent pure-Torch reference decoder for Gridbook CB test fixtures.

This module intentionally does not import any Gridbook runtime kernel.  It
decodes the on-disk index stream, applies the declared codebook/scale contract,
rounds each reconstructed weight to BF16, and uses a normal FP32-reference
matmul.  Native CUDA/CUTLASS tests use it as the correctness oracle.

It also SYNTHESIZES layout-v2 fixtures (see ``synth_two_tier_v2_plane``), so a
lane whose gates compare two decoders over the same bytes does not need the
separate ``prismaquant`` producer package installed to run at all.  Format
constants are restated here rather than imported from ``gridbook.codec`` on
purpose: this module is the independent side of every comparison it feeds.
"""
from __future__ import annotations

import torch


SUPERBLOCK = 256
TWO_TIER_SUPER_BIAS = 127
E4M3_MAX = 448.0
# The E2M1 magnitude grid every fp4 codebook value sits on.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _split_widths(k_bits: int, n_sub: int) -> list[int]:
    if n_sub not in (1, 2, 4) or 8 % n_sub:
        raise ValueError(f"unsupported sub-codebook count: {n_sub}")
    base, extra = divmod(k_bits, n_sub)
    return [base + (1 if i < extra else 0) for i in range(n_sub)]


def extract_codewords(
    qw_padded: torch.Tensor,
    *,
    N: int,
    K: int,
    k_bits: int,
    type_size: int,
) -> torch.Tensor:
    """Return the 32 little-endian ``k_bits`` codewords per superblock.

    The returned shape is ``[N, K/256, 32]``.  Eight bytes of explicit zero
    padding make the final unaligned window well-defined without relying on a
    runtime kernel's row-read slack.
    """
    if qw_padded.dtype is not torch.uint8 or qw_padded.dim() != 2:
        raise TypeError("qw_padded must be a rank-2 uint8 tensor")
    if tuple(qw_padded.shape[:1]) != (N,):
        raise ValueError(f"expected N={N} packed rows, got {qw_padded.shape[0]}")
    if K <= 0 or K % SUPERBLOCK:
        raise ValueError("K must be a positive multiple of 256")
    if not 0 < k_bits < 64:
        raise ValueError("the Torch reference supports 1..63-bit codewords")
    n_sb = K // SUPERBLOCK
    row_bytes = n_sb * type_size
    if qw_padded.shape[1] < row_bytes:
        raise ValueError(
            f"packed row has {qw_padded.shape[1]} bytes, needs {row_bytes}")

    raw = qw_padded[:, :row_bytes].contiguous()
    pad = torch.zeros((N, 8), dtype=torch.uint8, device=raw.device)
    padded = torch.cat((raw, pad), dim=1).to(torch.int64)
    vector = torch.arange(32, device=raw.device, dtype=torch.int64)
    byte_base = (vector * k_bits) // 8
    bit_shift = (vector * k_bits) % 8
    sb_base = (torch.arange(n_sb, device=raw.device, dtype=torch.int64)
               * type_size)
    starts = sb_base[:, None] + byte_base[None, :]
    window = torch.zeros((N, n_sb, 32), dtype=torch.int64,
                         device=raw.device)
    for byte in range(8):
        window |= padded[:, starts + byte] << (8 * byte)
    return (window >> bit_shift[None, None, :]) & ((1 << k_bits) - 1)


def decode_cb_values(
    qw_padded: torch.Tensor,
    cb_flat: torch.Tensor,
    cb_row_offset: torch.Tensor,
    *,
    N: int,
    K: int,
    k_bits: int,
    n_sub: int,
    type_size: int,
) -> torch.Tensor:
    """Decode codebook values, before the FP8/FP4 weight scale is applied."""
    if cb_flat.dim() != 1:
        raise ValueError("cb_flat must be a vector")
    if cb_row_offset.dtype is not torch.int32 or cb_row_offset.shape != (N,):
        raise TypeError(f"cb_row_offset must be int32 with shape ({N},)")
    if cb_flat.device != qw_padded.device \
            or cb_row_offset.device != qw_padded.device:
        raise ValueError("packed weights, codebook, and row offsets must colocate")

    codes = extract_codewords(
        qw_padded, N=N, K=K, k_bits=k_bits, type_size=type_size)
    row_base = cb_row_offset.to(torch.int64)[:, None, None, None]

    sub_dim = 8 // n_sub
    widths = _split_widths(k_bits, n_sub)
    bit_offset = 0
    table_base = 0
    pieces = []
    local = torch.arange(sub_dim, device=qw_padded.device,
                         dtype=torch.int64)
    for width in widths:
        index = (codes >> bit_offset) & ((1 << width) - 1)
        gather = (row_base + table_base
                  + index[..., None] * sub_dim + local)
        pieces.append(cb_flat[gather])
        bit_offset += width
        table_base += (1 << width) * sub_dim
    values = torch.cat(pieces, dim=-1)

    return values.reshape(N, K)


def reconstruct_cb_weight(
    qw_padded: torch.Tensor,
    cb_flat: torch.Tensor,
    cb_row_offset: torch.Tensor,
    scale: torch.Tensor,
    compose: torch.Tensor,
    *,
    N: int,
    K: int,
    k_bits: int,
    n_sub: int,
    type_size: int,
    is_fp4: bool,
    is_v2: bool = False,
) -> torch.Tensor:
    """Reconstruct the exact BF16 weight consumed by decode-contract v1."""
    values = decode_cb_values(
        qw_padded, cb_flat, cb_row_offset, N=N, K=K, k_bits=k_bits,
        n_sub=n_sub, type_size=type_size).to(torch.float32)

    if not is_fp4:
        if scale.numel() != N:
            raise ValueError(f"FP8-CB scale must contain N={N} values")
        weight = values * scale.reshape(N, 1).to(torch.float32)
        return weight.to(torch.bfloat16)

    n_sb = K // SUPERBLOCK
    if is_v2:
        if compose.numel() != 256 * 16:
            raise ValueError("v2 compose table must contain 4096 values")
        row_bytes = n_sb * type_size
        blocks = qw_padded[:, :row_bytes].contiguous().reshape(
            N, n_sb, type_size)
        super_e = blocks[..., 4 * k_bits].to(torch.int64)
        packed_sub = blocks[..., 4 * k_bits + 1:4 * k_bits + 9].to(
            torch.int64)
        sub_codes = torch.stack((packed_sub & 0xF, packed_sub >> 4), dim=-1)
        sub_codes = sub_codes.reshape(N, n_sb, 16)
        scales = compose.reshape(256, 16)[
            super_e[..., None].expand_as(sub_codes), sub_codes]
        scales = scales.reshape(N, n_sb * 16)
    else:
        if scale.numel() != N * n_sb * 16:
            raise ValueError(
                f"FP4-CB v1 scale must contain {N * n_sb * 16} values")
        scales = scale.reshape(N, n_sb * 16).to(torch.float32)

    weight = values * scales.repeat_interleave(16, dim=1)
    return weight.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Fixture synthesis for FP4-CB layout v2 (two-tier scale coding).
#
# The v2 lanes' correctness gates run one decoder against another over the SAME
# packed bytes, so what they need from a fixture is LEGALITY, not optimality:
# every ``(super, sub)`` pair must compose to an exact, in-range E4M3 scale
# (two-tier-scale-spec 1.2).  An illegal pair is not something any artifact can
# hold, and a gate that decodes one proves nothing about serving.  The producer
# guarantees legality as a side effect of encoding real weights; the helpers
# below guarantee it DIRECTLY, by building the legality mask from the same rule
# and drawing only from it.  That is what lets the bit-exact gates run on any
# CUDA box, with the producer-backed variants kept as additional tests.
# ---------------------------------------------------------------------------
def two_tier_v2_type_size(k_bits: int) -> int:
    """Bytes per superblock in layout v2: ``4*k`` index + 1 super + 8 sub."""
    return 4 * int(k_bits) + 9


def two_tier_compose_legality(sub_table) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(compose (256,16) fp32, legal (256,16) bool)`` for v2 scales.

    ``compose[E, c] = sub_table[c] * 2**(E - 127)``.  A pair is LEGAL exactly
    when that value is finite and in ``(0, 448]``, survives a
    ``float8_e4m3fn`` round trip bit-for-bit, and loses nothing to the fp32
    cast — the rule the producer's encoder emits under.
    """
    table = torch.tensor(list(sub_table), dtype=torch.float64)
    exact = table[None, :] * torch.pow(
        2.0, torch.arange(256, dtype=torch.float64)[:, None]
        - TWO_TIER_SUPER_BIAS)
    compose = exact.to(torch.float32)
    finite = torch.isfinite(compose)
    trip = torch.where(finite, compose, torch.zeros_like(compose)).to(
        torch.float8_e4m3fn).to(torch.float32)
    legal = (finite & (compose > 0) & (compose <= E4M3_MAX)
             & (trip == compose) & (compose.to(torch.float64) == exact))
    return compose, legal


def two_tier_full_legal_supers(sub_table, *, span=None) -> torch.Tensor:
    """The super exponents whose sixteen sub codes are ALL legal (int64).

    Drawing the super byte from these lets the sub nibbles be uniform over
    ``0..15``, so a synthesized plane exercises the whole compose gather
    without ever emitting a pair no artifact could contain.

    ``span`` narrows the result to that many exponents around the middle of
    the legal run.  A gate on BIT-EXACTNESS wants the whole run (widest
    coverage of the gather, and decode identity does not care how the scales
    are conditioned).  A gate on NUMERICS wants a narrow one: the super
    exponent is per ``(row, superblock)``, so the full run makes output rows
    differ in magnitude by up to 2**12, and a whole-matrix relative L2 would
    then be set by the loudest rows and go blind to the quiet ones.
    """
    _, legal = two_tier_compose_legality(sub_table)
    full = torch.nonzero(legal.all(dim=1)).reshape(-1)
    if span is None:
        return full
    span = int(span)
    if not 0 < span <= full.numel():
        raise ValueError(f"span must be in 1..{full.numel()}, got {span}")
    start = (full.numel() - span) // 2
    return full[start:start + span]


def synth_product_codebook(
    k_bits: int, *, seed: int, device="cpu"
) -> list[torch.Tensor]:
    """Two E2M1-valued product sub-tables for rung ``k_bits``.

    Shapes are ``(2**ceil(k/2), 4)`` then ``(2**floor(k/2), 4)`` — the split
    ``decode_cb_values`` reads and the one zero-based dictionary the v2
    kernels gather from.  Values are drawn from the signed E2M1 grid, so the
    table is on the grid the fp4 format declares and survives BF16 exactly.
    """
    grid = torch.tensor(
        sorted({v for m in E2M1_MAGNITUDES for v in (m, -m)}),
        dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    tables = []
    for width in _split_widths(int(k_bits), 2):
        pick = torch.randint(0, grid.numel(), ((1 << width) * 4,),
                             generator=generator)
        tables.append(grid[pick].reshape(1 << width, 4).to(device))
    return tables


def synth_two_tier_v2_plane(
    rows: int, K: int, k_bits: int, *, sub_table, seed: int, device="cpu",
    super_span=None
) -> torch.Tensor:
    """A random but format-legal FP4-CB layout-v2 packed plane.

    Returns ``[rows, (K // 256) * (4 * k_bits + 9)]`` uint8: per superblock,
    ``4 * k_bits`` bytes of index stream, one super-exponent byte, and eight
    sub-nibble bytes.  Thirty-two ``k_bits`` codewords fill exactly
    ``4 * k_bits`` bytes, so every bit of the index section is a codeword bit
    and any byte pattern there is a valid stream over full power-of-two
    sub-tables.  The scale section is the part that has a legality rule, and
    it is drawn to satisfy it.  ``super_span`` is forwarded to
    :func:`two_tier_full_legal_supers`.  Generated from a CPU generator so the
    fixture is reproducible and identical on every device.
    """
    rows, K, k_bits = int(rows), int(K), int(k_bits)
    if K <= 0 or K % SUPERBLOCK:
        raise ValueError("K must be a positive multiple of 256")
    if not 0 < k_bits <= 24:
        raise ValueError("FP4-CB layout v2 codewords are 1..24 bits")
    type_size = two_tier_v2_type_size(k_bits)
    n_sb = K // SUPERBLOCK
    generator = torch.Generator().manual_seed(int(seed))
    block = torch.empty((rows, n_sb, type_size), dtype=torch.uint8)
    block[:, :, :4 * k_bits] = torch.randint(
        0, 256, (rows, n_sb, 4 * k_bits), generator=generator,
        dtype=torch.uint8)
    supers = two_tier_full_legal_supers(sub_table, span=super_span)
    if not supers.numel():
        raise ValueError("no super exponent makes every sub code legal")
    pick = torch.randint(0, supers.numel(), (rows, n_sb), generator=generator)
    block[:, :, 4 * k_bits] = supers[pick].to(torch.uint8)
    # Two legal nibbles per byte, so any byte is a legal sub-code pair.
    block[:, :, 4 * k_bits + 1:] = torch.randint(
        0, 256, (rows, n_sb, 8), generator=generator, dtype=torch.uint8)
    return block.reshape(rows, n_sb * type_size).contiguous().to(device)


def cb_linear_reference(
    x: torch.Tensor,
    qw_padded: torch.Tensor,
    cb_flat: torch.Tensor,
    cb_row_offset: torch.Tensor,
    scale: torch.Tensor,
    compose: torch.Tensor,
    *,
    N: int,
    K: int,
    k_bits: int,
    n_sub: int,
    type_size: int,
    is_fp4: bool,
    is_v2: bool = False,
) -> torch.Tensor:
    """Decode-contract-v1 oracle: BF16 W, FP32-reference matrix product."""
    weight = reconstruct_cb_weight(
        qw_padded, cb_flat, cb_row_offset, scale, compose, N=N, K=K,
        k_bits=k_bits, n_sub=n_sub, type_size=type_size, is_fp4=is_fp4,
        is_v2=is_v2)
    shape = x.shape
    out = x.reshape(-1, K).float() @ weight.float().t()
    return out.to(x.dtype).reshape(*shape[:-1], N)
