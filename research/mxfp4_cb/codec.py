"""Superblock codec for MXFP4-CB (CPU reference).

Wire per row (K multiple of 256):
  [ index_stream 4*k bytes | scale_plane 8 bytes ]  repeated per SB
  index_stream: 32 k-bit codewords LSB-first (§1.1)
  scale_plane: 8 × E8M0 bytes, one per 32-weight MX block

Validation mirrors gridbook/expand.py and docs/SPEC.md §§1–1.4.
"""
from __future__ import annotations

import torch

from .format import (
    Mxfp4CbFormat,
    SUPERBLOCK,
    CODEWORDS_PER_SB,
    MX_BLOCK,
    SCALES_PER_SB,
    E2M1_GRID_MAX,
)
from .e2m1 import (
    e2m1_encode,
    e2m1_decode,
    e8m0_encode_amax,
    e8m0_decode,
)

__all__ = [
    "pack_indices",
    "unpack_indices",
    "validate_qweight",
    "encode_mxfp4_cb",
    "decode_mxfp4_cb",
    "reference_decode_weight",
]


def _split_widths(k: int, n_sub: int):
    base, extra = divmod(k, n_sub)
    return [base + (1 if i < extra else 0) for i in range(n_sub)]


def pack_indices(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Pack 32 k-bit codes per superblock into 4*k bytes LSB-first.

    Args:
        codes: [num_sb, 32] int64 (0 <= code < 2^k)
        k: bits per codeword
    Returns:
        [num_sb, 4*k] uint8 LSB-first stream
    """
    if codes.dtype not in (torch.int64, torch.int32, torch.int16):
        codes = codes.to(torch.int64)
    if codes.dim() != 2 or codes.shape[1] != CODEWORDS_PER_SB:
        raise ValueError(f"codes must be [num_sb, 32], got {tuple(codes.shape)}")
    if bool((codes < 0).any()) or bool((codes >= (1 << k)).any()):
        raise ValueError(f"codes out of range for k={k}")
    num_sb = codes.shape[0]
    index_bytes = 4 * k
    # Build bitstream as 64-bit integer per SB chunk, then emit little-endian bytes.
    # For k≤24, 32*k ≤768 bits -> need multi-word.  Use Python int per SB then
    # to_bytes, which is deterministic and avoids overflow.
    out = torch.zeros((num_sb, index_bytes), dtype=torch.uint8)
    for sb in range(num_sb):
        stream = 0
        for i in range(CODEWORDS_PER_SB):
            stream |= int(codes[sb, i].item()) << (i * k)
        b = stream.to_bytes(index_bytes, "little")
        out[sb] = torch.tensor(list(b), dtype=torch.uint8)
    return out


def unpack_indices(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of pack_indices."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if packed.dim() != 2:
        raise ValueError(f"packed must be 2-D [num_sb, 4*k], got {tuple(packed.shape)}")
    index_bytes = 4 * k
    if packed.shape[1] != index_bytes:
        raise ValueError(f"packed width {packed.shape[1]} != 4*k={index_bytes}")
    num_sb = packed.shape[0]
    out = torch.zeros((num_sb, CODEWORDS_PER_SB), dtype=torch.int64)
    for sb in range(num_sb):
        raw = bytes(packed[sb].tolist())
        stream = int.from_bytes(raw, "little")
        for i in range(CODEWORDS_PER_SB):
            out[sb, i] = (stream >> (i * k)) & ((1 << k) - 1)
    return out


def validate_qweight(qw: torch.Tensor, fmt: Mxfp4CbFormat, K: int) -> int:
    """Validate packed qweight [rows, row_bytes] and return row_bytes.

    Raises on dtype/rank/stride/range violations.  Mirrors expand.py.
    """
    if not isinstance(fmt, Mxfp4CbFormat):
        raise TypeError("fmt must be Mxfp4CbFormat")
    if qw.dtype != torch.uint8:
        raise TypeError(f"qweight must be uint8, got {qw.dtype}")
    if qw.dim() != 2:
        raise ValueError(f"qweight must be 2-D [rows, row_bytes], got {tuple(qw.shape)}")
    if qw.stride(1) != 1:
        raise ValueError("qweight rows must be contiguous in bytes (stride 1)")
    if K % SUPERBLOCK != 0 or K <= 0:
        raise ValueError(f"K={K} must be positive multiple of {SUPERBLOCK}")
    n_sb = K // SUPERBLOCK
    need = n_sb * fmt.type_size
    if qw.shape[1] < need:
        raise ValueError(f"qweight row_bytes {qw.shape[1]} < need {need} for K={K} type_size={fmt.type_size}")
    return need


def _validate_codebook_parts(codebook_parts: list[torch.Tensor], fmt: Mxfp4CbFormat) -> None:
    if fmt.mode == "product":
        if len(codebook_parts) != 2:
            raise ValueError(f"product mode needs 2 codebook parts, got {len(codebook_parts)}")
        w0, w1 = fmt.sub_widths
        cb0, cb1 = codebook_parts
        if cb0.shape != (1 << w0, 4) or cb1.shape != (1 << w1, 4):
            raise ValueError(
                f"product codebook shape mismatch for k={fmt.k}: got {cb0.shape}, {cb1.shape}, want {(1<<w0,4)},{(1<<w1,4)}"
            )
    elif fmt.mode == "signed":
        if len(codebook_parts) != 1:
            raise ValueError("signed mode needs 1 codebook part")
        if codebook_parts[0].shape != (1 << (fmt.k - 8), 8):
            raise ValueError(f"signed codebook shape mismatch for k={fmt.k}: got {codebook_parts[0].shape}")
    else:
        if len(codebook_parts) != 1:
            raise ValueError("full mode needs 1 codebook part")
        if codebook_parts[0].shape != (1 << fmt.k, 8):
            raise ValueError(f"full codebook shape mismatch for k={fmt.k}: got {codebook_parts[0].shape}")


def _decode_values_from_codes(
    codes: torch.Tensor,  # [num_sb, 32]
    codebook_parts: list[torch.Tensor],
    fmt: Mxfp4CbFormat,
) -> torch.Tensor:
    """Map codes -> [num_sb, 256] values before scaling.

    codebook_parts: for product, two tensors [(2^w0,4),(2^w1,4)];
                    for signed, one [(2^(k-8),8)] magnitude;
                    for full, one [(2^k,8)].
    Returns flat values per SB as float32.
    """
    _validate_codebook_parts(codebook_parts, fmt)
    num_sb = codes.shape[0]
    if fmt.mode == "product":
        w0, w1 = fmt.sub_widths  # ceil-first
        cb0, cb1 = codebook_parts
        assert cb0.shape == (1 << w0, 4) and cb1.shape == (1 << w1, 4), \
            f"product codebook shape mismatch for k={fmt.k}: got {cb0.shape}, {cb1.shape}, want {(1<<w0,4)},{(1<<w1,4)}"
        idx0 = codes & ((1 << w0) - 1)
        idx1 = (codes >> w0) & ((1 << w1) - 1)
        v0 = cb0[idx0]  # [num_sb,32,4]
        v1 = cb1[idx1]
        vals = torch.cat([v0, v1], dim=-1)  # [num_sb,32,8]
        return vals.reshape(num_sb, SUPERBLOCK).to(torch.float32)
    elif fmt.mode == "signed":
        mag = codebook_parts[0]  # [(2^(k-8),8)]
        mag_idx = (codes >> 8) & ((1 << (fmt.k - 8)) - 1) if fmt.k > 8 else torch.zeros_like(codes)
        signs = codes & 0xFF  # 8 bits
        base = mag[mag_idx]  # [num_sb,32,8]
        # apply signs
        for j in range(8):
            bit = ((signs >> j) & 1).bool().unsqueeze(-1)
            base[:, :, j] = torch.where(bit.squeeze(-1) if False else (signs >> j) & 1, -base[:, :, j], base[:, :, j])
            # Simpler per-coord: vectorize
        # Recompute correctly:
        vals = mag[mag_idx].clone()
        sign_mask = signs  # [num_sb,32]
        for j in range(8):
            neg = ((sign_mask >> j) & 1).bool()
            vals[neg, j] *= -1
        # Above indexing is wrong shape; do loop with gather
        # Fallback: element-wise
        vals2 = torch.zeros_like(vals)
        for sb in range(num_sb):
            for ci in range(CODEWORDS_PER_SB):
                m = mag[mag_idx[sb, ci]]
                s = int(signs[sb, ci].item())
                for j in range(8):
                    vals2[sb, ci, j] = -m[j] if (s >> j) & 1 else m[j]
        return vals2.reshape(num_sb, SUPERBLOCK).to(torch.float32)
    else:  # full
        cb = codebook_parts[0]
        assert cb.shape == (1 << fmt.k, 8)
        vals = cb[codes]  # [num_sb,32,8]
        return vals.reshape(num_sb, SUPERBLOCK).to(torch.float32)


def _encode_codes_for_vectors(
    vecs: torch.Tensor,  # [num_sb,32,8] float32 (scaled)
    codebook_parts: list[torch.Tensor],
    fmt: Mxfp4CbFormat,
) -> torch.Tensor:
    """Nearest code assignment per 8-vector (L2). Returns codes [num_sb,32]."""
    _validate_codebook_parts(codebook_parts, fmt)
    num_sb, nvec, dim = vecs.shape
    assert nvec == CODEWORDS_PER_SB and dim == 8
    codes = torch.zeros((num_sb, CODEWORDS_PER_SB), dtype=torch.int64)
    if fmt.mode == "product":
        w0, w1 = fmt.sub_widths
        cb0, cb1 = codebook_parts  # each float32
        # Split vecs into two 4-d halves
        v0 = vecs[:, :, :4]  # [sb,32,4]
        v1 = vecs[:, :, 4:]
        # brute force per vector: find nearest sub-code
        for sb in range(num_sb):
            for ci in range(CODEWORDS_PER_SB):
                # sub0
                d0 = ((cb0.to(torch.float32) - v0[sb, ci]) ** 2).sum(dim=-1)
                i0 = int(d0.argmin().item())
                d1 = ((cb1.to(torch.float32) - v1[sb, ci]) ** 2).sum(dim=-1)
                i1 = int(d1.argmin().item())
                codes[sb, ci] = i0 | (i1 << w0)
    elif fmt.mode == "signed":
        mag = codebook_parts[0]
        mag_f = mag.to(torch.float32)
        for sb in range(num_sb):
            for ci in range(CODEWORDS_PER_SB):
                v = vecs[sb, ci]  # [8]
                best = None
                best_code = 0
                # enumerate magnitudes
                for mi in range(mag.shape[0]):
                    m = mag_f[mi]  # [8] positive
                    # best signs for this magnitude is sign of v where feasible
                    # distance with sign choice per coord: choose sign to match v
                    cand = torch.where(v < 0, -m, m)
                    dist = ((cand - v) ** 2).sum().item()
                    # But signed mode encodes arbitrary signs, so the nearest
                    # code for this magnitude is signs = (v<0)
                    if best is None or dist < best:
                        best = dist
                        signs = 0
                        for j in range(8):
                            if v[j] < 0:
                                signs |= 1 << j
                            # zeros tie to positive (matches encode)
                            # For zero magnitude entry, sign irrelevant, but we keep.
                        best_code = (mi << 8) | signs
                codes[sb, ci] = best_code
    else:  # full
        cb = codebook_parts[0].to(torch.float32)
        for sb in range(num_sb):
            for ci in range(CODEWORDS_PER_SB):
                v = vecs[sb, ci]
                d = ((cb - v) ** 2).sum(dim=-1)
                codes[sb, ci] = int(d.argmin().item())
    return codes


def encode_mxfp4_cb(
    weight: torch.Tensor,
    fmt: Mxfp4CbFormat,
    codebook_parts: list[torch.Tensor],
) -> torch.Tensor:
    """Reference encode: weight [rows, K] -> packed qweight [rows, row_bytes].

    Steps per superblock:
      1. partition into 8 MX blocks of 32 -> amax -> E8M0 byte -> scale
      2. divide weight block by scale -> scaled vectors [32,8]
      3. assign nearest codeword per 8-vector
      4. LSB-first pack + append 8 scale bytes

    Validates shape/dtype/invariants; deterministic (argmin ties favor smallest
    index — deterministic by torch.argmin's first-occurrence rule).
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [rows,K], got {tuple(weight.shape)}")
    if weight.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"weight dtype must be float, got {weight.dtype}")
    rows, K = weight.shape
    fmt.validate_weights_shape((rows, K))
    n_sb = K // SUPERBLOCK
    row_bytes = fmt.packed_row_bytes(K)
    out = torch.zeros((rows, row_bytes), dtype=torch.uint8)

    w_f = weight.to(torch.float32)
    for r in range(rows):
        for sb in range(n_sb):
            base = sb * SUPERBLOCK
            block = w_f[r, base: base + SUPERBLOCK]  # [256]
            # Per-MX-block scales
            groups = block.reshape(SCALES_PER_SB, MX_BLOCK)
            amax = groups.abs().amax(dim=-1)
            sf_bytes = e8m0_encode_amax(amax)  # [8] uint8
            scales = e8m0_decode(sf_bytes)  # [8] float32
            # Scale each 32-block
            scaled = (groups / scales.unsqueeze(-1)).reshape(CODEWORDS_PER_SB, 8)
            scaled = scaled.clamp(-E2M1_GRID_MAX, E2M1_GRID_MAX)
            # assign
            vecs = scaled.unsqueeze(0)  # [1,32,8]
            codes = _encode_codes_for_vectors(vecs, codebook_parts, fmt)  # [1,32]
            packed = pack_indices(codes, fmt.k)  # [1,4k]
            # write row
            off = sb * fmt.type_size
            out[r, off: off + fmt.index_bytes] = packed[0]
            out[r, off + fmt.index_bytes: off + fmt.type_size] = sf_bytes
    return out


def decode_mxfp4_cb(
    qweight: torch.Tensor,
    fmt: Mxfp4CbFormat,
    codebook_parts: list[torch.Tensor],
    rows: int,
    K: int,
) -> torch.Tensor:
    """Reference decode: packed qweight -> BF16 weight [rows, K].

    Mirrors encode's scale handling; uses the same codebook.
    """
    need = validate_qweight(qweight, fmt, K)
    if qweight.shape[0] != rows:
        raise ValueError(f"qweight rows {qweight.shape[0]} != rows {rows}")
    n_sb = K // SUPERBLOCK
    w = torch.zeros((rows, K), dtype=torch.bfloat16)

    for r in range(rows):
        for sb in range(n_sb):
            off = sb * fmt.type_size
            idx_bytes = qweight[r, off: off + fmt.index_bytes]
            sf_bytes = qweight[r, off + fmt.index_bytes: off + fmt.type_size]
            codes = unpack_indices(idx_bytes.unsqueeze(0), fmt.k)  # [1,32]
            vals = _decode_values_from_codes(codes, codebook_parts, fmt)  # [1,256] float32
            vals = vals[0]  # [256]
            scales = e8m0_decode(sf_bytes)  # [8]
            # expand scales per 32
            scale_expanded = scales.repeat_interleave(MX_BLOCK)  # [256]
            block = (vals * scale_expanded).to(torch.bfloat16)
            w[r, sb * SUPERBLOCK: (sb + 1) * SUPERBLOCK] = block
    return w


def reference_decode_weight(
    qweight: torch.Tensor,
    fmt: Mxfp4CbFormat,
    codebook_parts: list[torch.Tensor],
) -> torch.Tensor:
    """Convenience: infer rows/K from qweight and decode."""
    if qweight.dim() != 2:
        raise ValueError("qweight must be 2-D")
    rows = qweight.shape[0]
    n_sb = qweight.shape[1] // fmt.type_size
    if qweight.shape[1] % fmt.type_size != 0:
        raise ValueError(f"qweight width {qweight.shape[1]} not multiple of type_size {fmt.type_size}")
    K = n_sb * SUPERBLOCK
    return decode_mxfp4_cb(qweight, fmt, codebook_parts, rows, K)

