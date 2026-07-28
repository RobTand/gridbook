"""Transient CB->value expander (docs/nvfp4-cb-plan/serving-kernel.md §1a,
prototype ii+ / M-gated prefill dispatch).

``expand_cb_to_value`` is the existing decode-GEMM kernel MINUS the matmul MINUS
the per-channel/group scale: it decodes the codebook VALUE for every ``(n, j)``
into a bounded ``[N, K]`` tile. Because an FP8_CB codebook value already lives on
the e4m3 grid (‖·‖<=448), that tile is a *standard per-output-channel FP8
weight* — the caller casts it to ``float8_e4m3fn`` (lossless) and feeds vLLM's
stock W8A8 fp8 GEMM with the layer's existing ``weight_scale``. That is the whole
trick: an expanded FP8_CB weight IS a plain fp8 checkpoint, so prefill reaches
the native tensor cores instead of re-decoding per M-tile in a bf16-MMA kernel.

**INV-1 (docs §0), honored precisely.** The ``[N, K]`` tile is a per-LAYER
TRANSIENT: the caller expands ONE layer, GEMMs, and frees it before the next.
This is *not* the NVINT2 trap — that died from a RESIDENT, model-wide dense
expansion (92.9 GB artifact -> 115.7 GiB resident, OOM). Here the resident weight
stays the packed k-bit index stream + the tiny flat codebook + the per-channel
fp32 scale; only a single layer's decoded tile is ever live (peak ~9 MiB for the
0.6B MLP rung), and it is released each forward. The bounded transient is the
point, not a compromise.

Self-contained: imports only ``torch`` + ``triton`` (no vLLM, no ``prismaquant``,
no ``.kernels``), so the build-venv correctness gate imports it directly. The
codeword byte-window extraction + product sub-index gather below is copied from
``kernels._cb_decode_gemm_kernel`` (kept in lockstep on purpose — the two must
decode bit-identically). **Even-split product only** (FP8_CB_K44: k=44, n_sub=4,
sub_dim=2), matching the decode prototype's scope.

FP8_CB only. NVFP4_CB stays on the Triton decode path: a *transient* NVFP4 tile
would still need the Blackwell FP4-MMA plumbing (prototype (iii), INV-2) to be
worth expanding, and a decoded fp4 value is not a standalone tensor without its
group-16 scale plane — so this expander refuses ``is_fp4``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cb_expand_value_kernel(
    qw_ptr, cb_ptr, cboff_ptr, w_ptr,
    N, K,
    stride_qn,                 # padded row stride (bytes) of qw
    stride_wn, stride_wk,      # output [N, K] strides
    K_BITS: tl.constexpr,
    SUB_DIM: tl.constexpr,
    SUB_W0: tl.constexpr, SUB_W1: tl.constexpr,
    SUB_W2: tl.constexpr, SUB_W3: tl.constexpr,
    SUB_OFF1: tl.constexpr, SUB_OFF2: tl.constexpr, SUB_OFF3: tl.constexpr,
    SUB_BASE1: tl.constexpr, SUB_BASE2: tl.constexpr, SUB_BASE3: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)                    # one 256-weight superblock
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n_i = offs_n.to(tl.int64)

    # --- per-column (within one 256-superblock) decode constants ------------
    # (identical to kernels._cb_decode_gemm_kernel; must stay bit-for-bit).
    kcol = tl.arange(0, 256)
    v_local = kcol // 8                         # which of the 32 codewords
    coord = kcol % 8                            # coord inside the 8-dim vector
    sub = coord // SUB_DIM                       # which sub-codebook
    local = coord % SUB_DIM                      # coord inside the sub-vector
    bitpos = v_local * K_BITS
    byte_base = (bitpos // 8).to(tl.int64)       # first byte of the codeword
    bit_in_byte = bitpos % 8
    mask_k = (1 << K_BITS) - 1
    # Ceil-first per-sub split (encoder _bit_split); even k reduces to the
    # historical uniform layout.
    shift_sub = tl.where(
        sub == 0, 0, tl.where(sub == 1, SUB_OFF1,
                              tl.where(sub == 2, SUB_OFF2, SUB_OFF3)))
    mask_sub = tl.where(
        sub == 0, (1 << SUB_W0) - 1,
        tl.where(sub == 1, (1 << SUB_W1) - 1,
                 tl.where(sub == 2, (1 << SUB_W2) - 1, (1 << SUB_W3) - 1)))
    cb_base = tl.where(
        sub == 0, 0, tl.where(sub == 1, SUB_BASE1,
                              tl.where(sub == 2, SUB_BASE2, SUB_BASE3)))     # flat-codebook block base

    # Per-output-row codebook base offset (0 for a single-codebook Linear; a
    # fused qkv/gate_up module points each shard's rows at that role's block of
    # the concatenated flat codebook — same fusion mechanism as the decode
    # kernel, so the transient path is fusion-correct too).
    cb_off = tl.load(cboff_ptr + offs_n_i, mask=mask_n, other=0).to(tl.int64)

    s = pid_s
    col_byte = s * TYPE_SIZE + byte_base                       # [256] int64
    # 8-byte little-endian window; masked to K_BITS so the extra bytes (next
    # superblock / row pad) fall away. INV-1: no dense weight materialized here,
    # only this one [BLOCK_N, 256] tile in registers.
    code = tl.zeros((BLOCK_N, 256), dtype=tl.int64)
    base_ptr = offs_n_i[:, None] * stride_qn + col_byte[None, :]
    for i in range(0, 8):
        b = tl.load(qw_ptr + base_ptr + i, mask=mask_n[:, None],
                    other=0).to(tl.int64)
        code = code | (b << (8 * i))
    code = (code >> bit_in_byte[None, :]) & mask_k
    sub_idx = (code >> shift_sub[None, :]) & mask_sub          # [BN, 256]
    gather = (cb_off[:, None] + cb_base[None, :]
              + sub_idx * SUB_DIM + local[None, :])
    # The raw codebook VALUE (bf16), NOT * scale — this is the decode kernel
    # minus the `* scale` and minus the `tl.dot`.
    val = tl.load(cb_ptr + gather)                             # [BN, 256] bf16

    xcols = (s * 256 + kcol).to(tl.int64)
    w_out = w_ptr + offs_n_i[:, None] * stride_wn + xcols[None, :] * stride_wk
    tl.store(w_out, val, mask=mask_n[:, None])


@triton.jit
def _cb_expand_fp8_kernel(
    qw_ptr, cb_ptr, cboff_ptr, w_ptr,
    N, K,
    stride_qn,                 # padded row stride (bytes) of qw
    stride_wn, stride_wk,      # output [N, K] strides
    K_BITS: tl.constexpr,
    SUB_DIM: tl.constexpr,
    SUB_W0: tl.constexpr, SUB_W1: tl.constexpr,
    SUB_W2: tl.constexpr, SUB_W3: tl.constexpr,
    SUB_OFF1: tl.constexpr, SUB_OFF2: tl.constexpr, SUB_OFF3: tl.constexpr,
    SUB_BASE1: tl.constexpr, SUB_BASE2: tl.constexpr, SUB_BASE3: tl.constexpr,
    TYPE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FP8-direct variant of ``_cb_expand_value_kernel`` (same codeword
    extraction, kept in lockstep): the codebook is pre-converted to E4M3 BYTES
    (uint8), so the expand is a pure byte gather -> byte store. No bf16
    intermediate, no cast pass — the [N,K] transient is written once as fp8,
    halving the expand-side HBM traffic (the cutlass-kernel-notes stopgap)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)                    # one 256-weight superblock
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n_i = offs_n.to(tl.int64)

    kcol = tl.arange(0, 256)
    v_local = kcol // 8
    coord = kcol % 8
    sub = coord // SUB_DIM
    local = coord % SUB_DIM
    bitpos = v_local * K_BITS
    byte_base = (bitpos // 8).to(tl.int64)
    bit_in_byte = bitpos % 8
    mask_k = (1 << K_BITS) - 1
    # Ceil-first per-sub split (encoder _bit_split); even k reduces to the
    # historical uniform layout.
    shift_sub = tl.where(
        sub == 0, 0, tl.where(sub == 1, SUB_OFF1,
                              tl.where(sub == 2, SUB_OFF2, SUB_OFF3)))
    mask_sub = tl.where(
        sub == 0, (1 << SUB_W0) - 1,
        tl.where(sub == 1, (1 << SUB_W1) - 1,
                 tl.where(sub == 2, (1 << SUB_W2) - 1, (1 << SUB_W3) - 1)))
    cb_base = tl.where(
        sub == 0, 0, tl.where(sub == 1, SUB_BASE1,
                              tl.where(sub == 2, SUB_BASE2, SUB_BASE3)))

    cb_off = tl.load(cboff_ptr + offs_n_i, mask=mask_n, other=0).to(tl.int64)

    s = pid_s
    col_byte = s * TYPE_SIZE + byte_base
    code = tl.zeros((BLOCK_N, 256), dtype=tl.int64)
    base_ptr = offs_n_i[:, None] * stride_qn + col_byte[None, :]
    for i in range(0, 8):
        b = tl.load(qw_ptr + base_ptr + i, mask=mask_n[:, None],
                    other=0).to(tl.int64)
        code = code | (b << (8 * i))
    code = (code >> bit_in_byte[None, :]) & mask_k
    sub_idx = (code >> shift_sub[None, :]) & mask_sub
    gather = (cb_off[:, None] + cb_base[None, :]
              + sub_idx * SUB_DIM + local[None, :])
    val = tl.load(cb_ptr + gather)                             # [BN, 256] uint8

    xcols = (s * 256 + kcol).to(tl.int64)
    w_out = w_ptr + offs_n_i[:, None] * stride_wn + xcols[None, :] * stride_wk
    tl.store(w_out, val, mask=mask_n[:, None])


@triton.jit
def _cb_expand_weight_v2_kernel(
    qw_ptr, cb_ptr, cboff_ptr, compose_ptr, w_ptr,
    N, K,
    stride_qn, stride_wn, stride_wk,
    K_BITS: tl.constexpr, SUB_DIM: tl.constexpr,
    SUB_W0: tl.constexpr, SUB_W1: tl.constexpr,
    SUB_W2: tl.constexpr, SUB_W3: tl.constexpr,
    SUB_OFF1: tl.constexpr, SUB_OFF2: tl.constexpr, SUB_OFF3: tl.constexpr,
    SUB_BASE1: tl.constexpr, SUB_BASE2: tl.constexpr, SUB_BASE3: tl.constexpr,
    SIGNED: tl.constexpr,
    TYPE_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """fp4 two-tier v2 weight expander (spec §4b): decode the codebook value AND
    compose the E4M3 group scale in-register from the packed 9-byte plane, write
    the full ``value × scale`` bf16 weight tile into the transient ``[N,K]``
    buffer. Same in-kernel compose as the decode kernel (bit-exact); the plane
    is composed during expansion so the transient carries a ready E4M3-scaled
    weight (a future CUTLASS block-scaled prefill would instead stage the
    swizzled SF plane here — zero CUTLASS surgery). INV-1: bounded per layer."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n_i = offs_n.to(tl.int64)

    kcol = tl.arange(0, 256)
    v_local = kcol // 8
    coord = kcol % 8
    sub = coord // SUB_DIM
    local = coord % SUB_DIM
    bitpos = v_local * K_BITS
    byte_base = (bitpos // 8).to(tl.int64)
    bit_in_byte = bitpos % 8
    mask_k = (1 << K_BITS) - 1
    # Ceil-first per-sub split (encoder _bit_split); even k reduces to the
    # historical uniform layout.
    shift_sub = tl.where(
        sub == 0, 0, tl.where(sub == 1, SUB_OFF1,
                              tl.where(sub == 2, SUB_OFF2, SUB_OFF3)))
    mask_sub = tl.where(
        sub == 0, (1 << SUB_W0) - 1,
        tl.where(sub == 1, (1 << SUB_W1) - 1,
                 tl.where(sub == 2, (1 << SUB_W2) - 1, (1 << SUB_W3) - 1)))
    cb_base = tl.where(
        sub == 0, 0, tl.where(sub == 1, SUB_BASE1,
                              tl.where(sub == 2, SUB_BASE2, SUB_BASE3)))
    grp16v = tl.arange(0, 16)
    cb_off = tl.load(cboff_ptr + offs_n_i, mask=mask_n, other=0).to(tl.int64)

    s = pid_s
    col_byte = s * TYPE_SIZE + byte_base
    code = tl.zeros((BLOCK_N, 256), dtype=tl.int64)
    base_ptr = offs_n_i[:, None] * stride_qn + col_byte[None, :]
    for i in range(0, 8):
        b = tl.load(qw_ptr + base_ptr + i, mask=mask_n[:, None],
                    other=0).to(tl.int64)
        code = code | (b << (8 * i))
    code = (code >> bit_in_byte[None, :]) & mask_k
    if SIGNED:
        sub_idx = code >> 8
        gather = cb_off[:, None] + sub_idx * SUB_DIM + local[None, :]
        val = tl.load(cb_ptr + gather).to(tl.float32)             # [BN,256]
        neg = ((code >> local[None, :]) & 1) == 1
        val = tl.where(neg, -val, val)
    else:
        sub_idx = (code >> shift_sub[None, :]) & mask_sub
        gather = (cb_off[:, None] + cb_base[None, :] + sub_idx * SUB_DIM
                  + local[None, :])
        val = tl.load(cb_ptr + gather).to(tl.float32)             # [BN,256]

    # v2 compose (bit-exact to the decode kernel / reconstruct).
    super_off = s * TYPE_SIZE + 4 * K_BITS
    super_e = tl.load(qw_ptr + offs_n_i * stride_qn + super_off,
                      mask=mask_n, other=0).to(tl.int64)
    sub_off = s * TYPE_SIZE + 4 * K_BITS + 1 + (grp16v // 2)
    nib = (grp16v % 2) * 4
    sub_byte = tl.load(
        qw_ptr + offs_n_i[:, None] * stride_qn + sub_off[None, :],
        mask=mask_n[:, None], other=0).to(tl.int64)
    code16 = (sub_byte >> nib[None, :]) & 0xF
    sc16 = tl.load(compose_ptr + super_e[:, None] * 16 + code16)   # [BN,16]
    sc = tl.reshape(tl.broadcast_to(sc16[:, :, None], (BLOCK_N, 16, 16)),
                    (BLOCK_N, 256))
    w = (val * sc).to(tl.bfloat16)

    xcols = (s * 256 + kcol).to(tl.int64)
    tl.store(w_ptr + offs_n_i[:, None] * stride_wn + xcols[None, :] * stride_wk,
             w, mask=mask_n[:, None])


def _ceil_first_split(k_bits: int, n_sub: int, sub_dim: int):
    """Per-sub widths/offsets/table bases (ceil-first, = encoder _bit_split),
    padded to 4 subs for the constexpr plumbing."""
    base, extra = divmod(k_bits, n_sub)
    ws = [base + (1 if i < extra else 0) for i in range(n_sub)] \
        + [0] * (4 - n_sub)
    offs = [sum(ws[:i]) for i in range(4)]
    bases = [sum(sub_dim << w for w in ws[:i] if w) for i in range(4)]
    return ws, offs, bases


def expand_fp4_v2_to_weight(cb_qweight_padded, cb_flat, cb_row_offset, compose,
                            N, K, k_bits, n_sub, type_size):
    """Transient [N,K] bf16 weight (value × composed E4M3 scale) for a fp4 v2
    layer — the prefill counterpart of the decode kernel, amortising the decode
    over M via one cuBLAS bf16 GEMM. Bounded per-layer transient (INV-1)."""
    sub_dim = 8 // n_sub
    ws, offs, bases = _ceil_first_split(k_bits, n_sub, sub_dim)
    dev = cb_qweight_padded.device
    W = torch.empty((N, K), dtype=torch.bfloat16, device=dev)
    n_sb = K // 256
    block_n = 64
    grid = (triton.cdiv(N, block_n), n_sb)
    _cb_expand_weight_v2_kernel[grid](
        cb_qweight_padded, cb_flat, cb_row_offset, compose, W, N, K,
        cb_qweight_padded.stride(0), W.stride(0), W.stride(1),
        K_BITS=k_bits, SUB_DIM=sub_dim,
        SUB_W0=ws[0], SUB_W1=ws[1], SUB_W2=ws[2], SUB_W3=ws[3],
        SUB_OFF1=offs[1], SUB_OFF2=offs[2], SUB_OFF3=offs[3],
        SUB_BASE1=bases[1], SUB_BASE2=bases[2], SUB_BASE3=bases[3],
        SIGNED=(n_sub == 1),
        TYPE_SIZE=type_size,
        BLOCK_N=block_n, num_warps=4)
    return W


def expand_cb_to_fp8(
    cb_qweight_padded: torch.Tensor,   # (N, row_bytes + PAD) uint8 (codec.PAD_BYTES)
    cb_flat_fp8: torch.Tensor,         # (cb_total,) uint8 E4M3-byte codebook(s)
    cb_row_offset: torch.Tensor,       # (N,) int32 per-row base into cb_flat
    N: int, K: int,
    k_bits: int, n_sub: int, type_size: int,
) -> torch.Tensor:
    """FP8-direct transient expand: decode the codebook VALUE for every
    ``(n, j)`` straight into a fresh ``[N, K]`` **float8_e4m3fn** tile.

    Byte-identical to ``expand_cb_to_value(...).to(torch.float8_e4m3fn)`` (the
    codebook is e4m3-grid-valued, so the byte gather IS the lossless cast) but
    writes 1 B/elt once instead of a 2 B/elt bf16 tile plus a separate cast
    pass — the cheap prefill-traffic win from cutlass-kernel-notes.md. INV-1:
    the returned tile is a bounded per-layer transient the caller frees.
    """
    if cb_flat_fp8.dtype != torch.uint8:
        raise TypeError("expand_cb_to_fp8 wants the E4M3-byte (uint8) codebook")
    if K % 256 != 0:
        raise ValueError(f"K={K} must be a multiple of the 256-weight superblock")
    sub_dim = 8 // n_sub
    ws, offs, bases = _ceil_first_split(k_bits, n_sub, sub_dim)
    dev = cb_qweight_padded.device
    W = torch.empty((N, K), dtype=torch.uint8, device=dev)
    n_sb = K // 256
    block_n = 64
    grid = (triton.cdiv(N, block_n), n_sb)
    _cb_expand_fp8_kernel[grid](
        cb_qweight_padded, cb_flat_fp8, cb_row_offset, W,
        N, K,
        cb_qweight_padded.stride(0),
        W.stride(0), W.stride(1),
        K_BITS=k_bits, SUB_DIM=sub_dim,
        SUB_W0=ws[0], SUB_W1=ws[1], SUB_W2=ws[2], SUB_W3=ws[3],
        SUB_OFF1=offs[1], SUB_OFF2=offs[2], SUB_OFF3=offs[3],
        SUB_BASE1=bases[1], SUB_BASE2=bases[2], SUB_BASE3=bases[3],
        TYPE_SIZE=type_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return W.view(torch.float8_e4m3fn)


def expand_cb_to_value(
    cb_qweight_padded: torch.Tensor,   # (N, row_bytes + PAD) uint8 (codec.PAD_BYTES)
    cb_flat: torch.Tensor,             # (cb_total,) bf16 flat codebook(s)
    cb_row_offset: torch.Tensor,       # (N,) int32 per-row base into cb_flat
    N: int, K: int,
    k_bits: int, n_sub: int, type_size: int, is_fp4: bool,
) -> torch.Tensor:
    """Decode the codebook VALUE for every ``(n, j)`` into a fresh ``[N, K]``
    bf16 transient (no per-channel/group scale applied).

    The result is the FP8_CB weight's decoded e4m3-grid values; the caller pairs
    it with the layer's per-output ``weight_scale`` to run a stock fp8 W8A8 GEMM.
    Every value is exactly representable in e4m3 (the codebook is e4m3-valued),
    so ``result.to(torch.float8_e4m3fn)`` is lossless.

    INV-1: the returned tile is a bounded per-layer transient; the caller frees
    it after the GEMM. It is never resident/model-wide.
    """
    if is_fp4:
        raise NotImplementedError(
            "expand_cb_to_value is FP8_CB-only (prototype ii+). NVFP4_CB stays "
            "on the Triton decode path: a transient FP4 tile still needs the "
            "Blackwell FP4-MMA to be worth expanding (prototype iii / INV-2), "
            "and a decoded fp4 value is not a standalone tensor without its "
            "group-16 scale plane. See docs/nvfp4-cb-plan/serving-kernel.md.")
    if K % 256 != 0:
        raise ValueError(f"K={K} must be a multiple of the 256-weight superblock")
    sub_dim = 8 // n_sub
    ws, offs, bases = _ceil_first_split(k_bits, n_sub, sub_dim)
    dev = cb_qweight_padded.device
    W = torch.empty((N, K), dtype=torch.bfloat16, device=dev)
    n_sb = K // 256
    block_n = 64
    grid = (triton.cdiv(N, block_n), n_sb)
    _cb_expand_value_kernel[grid](
        cb_qweight_padded, cb_flat, cb_row_offset, W,
        N, K,
        cb_qweight_padded.stride(0),
        W.stride(0), W.stride(1),
        K_BITS=k_bits, SUB_DIM=sub_dim,
        SUB_W0=ws[0], SUB_W1=ws[1], SUB_W2=ws[2], SUB_W3=ws[3],
        SUB_OFF1=offs[1], SUB_OFF2=offs[2], SUB_OFF3=offs[3],
        SUB_BASE1=bases[1], SUB_BASE2=bases[2], SUB_BASE3=bases[3],
        TYPE_SIZE=type_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return W
