"""Independent pure-Torch reference decoder for Gridbook CB test fixtures.

This module intentionally does not import any Gridbook runtime kernel.  It
decodes the on-disk index stream, applies the declared codebook/scale contract,
rounds each reconstructed weight to BF16, and uses a normal FP32-reference
matmul.  Native CUDA/CUTLASS tests use it as the correctness oracle.
"""
from __future__ import annotations

import torch


SUPERBLOCK = 256


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
    local8 = torch.arange(8, device=qw_padded.device, dtype=torch.int64)
    row_base = cb_row_offset.to(torch.int64)[:, None, None, None]

    if n_sub == 1:
        # Signed layout: eight low bits are per-coordinate signs and the
        # remaining high bits select one non-negative 8-vector.
        magnitude = codes >> 8
        gather = row_base + magnitude[..., None] * 8 + local8
        values = cb_flat[gather]
        negative = ((codes[..., None] >> local8) & 1).bool()
        values = torch.where(negative, -values, values)
    else:
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
