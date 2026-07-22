"""Triton decode-GEMM kernels for the NVFP4-CB / FP8-CB codebook formats.

**CORRECTNESS-FIRST PROTOTYPE — NOT PRODUCTION-ELIGIBLE.**

This is prototype (i) of docs/nvfp4-cb-plan/serving-kernel.md. It exists to
measure served KL and get a first speed reading; it is explicitly disqualified
as the production prefill path:

* **INV-1 (honored):** the resident weight is the packed k-bit index stream +
  the tiny flat codebook + the (pre-decoded) E4M3/fp32 scales. The dense [N,K]
  weight is NEVER materialized in HBM — each superblock's [256, BLOCK_N] weight
  tile is expanded inside the kernel, in registers, then consumed immediately by
  the matmul. That is the whole point (the NVINT2 OOM trap was exactly a
  load-time dense expansion).
* **INV-2 (WAIVED for this prototype):** we decode FP4/FP8 codes to bf16 and run
  `tl.dot` (bf16 MMA). Triton cannot emit the Blackwell sm_121 block-scaled FP4
  MMA, so this kernel reaches only bf16 tensor cores. The production prefill
  (CUTLASS/CuTe fused-expand, prototype (iii)) is what routes decoded codes to
  the FP4 MMA; THIS kernel will fail the perf gate by construction and must not
  be promoted. Comments below say so at each relevant point.

**Product** mode is implemented for ANY integer k: even splits (NVFP4_CB_K16
-> (8,8); FP8_CB_K44 -> (11,11,11,11)) and ceil-first uneven splits (k=13 ->
(7,6)), matching the encoder's _bit_split. Signed/full modes are out of scope
for this prototype.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cb_decode_gemm_kernel(
    x_ptr, qw_ptr, cb_ptr, cboff_ptr, scale_ptr, compose_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qn,                 # padded row stride (bytes) of qw
    stride_ym, stride_yn,
    stride_sn,                 # scale-row stride (fp4 v1 only; ignored otherwise)
    K_BITS: tl.constexpr,
    N_SUB: tl.constexpr,
    SUB_DIM: tl.constexpr,
    # Ceil-first per-sub bit widths (encoder _bit_split): sub i occupies
    # SUB_OFF_i..SUB_OFF_i+SUB_W_i-1 of the codeword, its table at flat-
    # codebook element SUB_BASE_i. Unused trailing subs carry zeros.
    SUB_W0: tl.constexpr, SUB_W1: tl.constexpr,
    SUB_W2: tl.constexpr, SUB_W3: tl.constexpr,
    SUB_OFF1: tl.constexpr, SUB_OFF2: tl.constexpr, SUB_OFF3: tl.constexpr,
    SUB_BASE1: tl.constexpr, SUB_BASE2: tl.constexpr, SUB_BASE3: tl.constexpr,
    SIGNED: tl.constexpr,      # S-rungs: 8 LSB sign bits + (k-8)-bit magnitude
    TYPE_SIZE: tl.constexpr,
    IS_FP4: tl.constexpr,
    LAYOUT_V2: tl.constexpr,   # fp4 two-tier scale plane (9 B, in-kernel compose)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_n_i = offs_n.to(tl.int64)

    # --- per-column (within one 256-superblock) decode constants ------------
    kcol = tl.arange(0, 256)
    coord = kcol % 8                          # coord inside the 8-dim vector
    sub = coord // SUB_DIM                     # which sub-codebook
    local = coord % SUB_DIM                    # coord inside the sub-vector
    # Per-CODEWORD byte addressing (32 distinct codewords, one per 8-dim vector):
    # 8 consecutive columns share codeword v = kcol//8, so the byte window is
    # loaded once per v (not per column) and broadcast — ~8x fewer byte loads.
    v = tl.arange(0, 32)
    byte_base_v = ((v * K_BITS) // 8).to(tl.int64)   # first byte of codeword [32]
    bit_in_byte_v = (v * K_BITS) % 8                  # [32]
    mask_k = (1 << K_BITS) - 1
    # Per-column sub-table constants via the (constexpr) ceil-first split.
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
    grp16v = tl.arange(0, 16)                   # 16 distinct group-16 scales [16]

    # Per-output-row codebook base offset (0 for a single-codebook Linear; for a
    # fused qkv/gate_up module each shard's rows point at that role's block of
    # the concatenated flat codebook — this is how per-role shared codebooks
    # survive vLLM's qkv/gate_up fusion).
    cb_off = tl.load(cboff_ptr + offs_n_i, mask=mask_n, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_sb = K // 256

    if not IS_FP4:
        # fp8: one per-output-channel fp32 scale, hoisted out of the K loop.
        sc_row = tl.load(scale_ptr + offs_n_i, mask=mask_n, other=0.0)  # [BN]

    for s in range(0, n_sb):
        col_byte_v = s * TYPE_SIZE + byte_base_v                   # [32] int64
        # --- decode 32 codewords ONCE, then expand to 256 cols IN REGISTERS -
        # 8-byte little-endian window per codeword; masked to K_BITS -> the
        # extra bytes (scale plane / next superblock) fall away. Only 32
        # distinct codewords per superblock, so load them once (~8x fewer byte
        # loads) and broadcast each across its 8 columns. INV-1: no dense weight.
        code32 = tl.zeros((BLOCK_N, 32), dtype=tl.int64)
        base_ptr = offs_n_i[:, None] * stride_qn + col_byte_v[None, :]
        for i in range(0, 8):
            b = tl.load(qw_ptr + base_ptr + i, mask=mask_n[:, None],
                        other=0).to(tl.int64)
            code32 = code32 | (b << (8 * i))
        code32 = (code32 >> bit_in_byte_v[None, :]) & mask_k       # [BN,32]
        # reshape [BN,32,8]->[BN,256] maps column kcol -> codeword kcol//8.
        code = tl.reshape(tl.broadcast_to(code32[:, :, None],
                                          (BLOCK_N, 32, 8)), (BLOCK_N, 256))
        if SIGNED:
            # signed mode: magnitude index above the 8 sign bits; ONE 8-dim
            # non-negative table (SUB_DIM==8, sub==0). Sign applied to the
            # magnitude BEFORE the scale multiply (exact negation), matching
            # nvfp4_cb_reconstruct's signed branch bit-for-bit.
            sub_idx = code >> 8
            gather = cb_off[:, None] + sub_idx * SUB_DIM + local[None, :]
            val = tl.load(cb_ptr + gather).to(tl.float32)         # [BN,256]
            neg = ((code >> local[None, :]) & 1) == 1
            val = tl.where(neg, -val, val)
        else:
            sub_idx = (code >> shift_sub[None, :]) & mask_sub      # [BN,256]
            gather = (cb_off[:, None] + cb_base[None, :]
                      + sub_idx * SUB_DIM + local[None, :])
            val = tl.load(cb_ptr + gather).to(tl.float32)         # [BN,256]

        if IS_FP4:
            # 16 distinct group-16 scales per superblock, broadcast across their
            # 16 columns (reshape [BN,16,16]->[BN,256] maps column kcol ->
            # group kcol//16).
            if LAYOUT_V2:
                # Two-tier v2 (two-tier-scale-spec.md §4a): compose the plane
                # IN-KERNEL from the packed 9 bytes (1 E8M0 super + 8 sub nibble
                # bytes) — NO resident fp32 plane (G4). scale_g = compose[E, c_g]
                # via the (256,16) table; bit-exact to nvfp4_cb_reconstruct.
                super_off = s * TYPE_SIZE + 4 * K_BITS
                super_e = tl.load(qw_ptr + offs_n_i * stride_qn + super_off,
                                  mask=mask_n, other=0).to(tl.int64)   # [BN]
                sub_off = s * TYPE_SIZE + 4 * K_BITS + 1 + (grp16v // 2)  # [16]
                nib = (grp16v % 2) * 4                                 # [16]
                sub_byte = tl.load(
                    qw_ptr + offs_n_i[:, None] * stride_qn + sub_off[None, :],
                    mask=mask_n[:, None], other=0).to(tl.int64)       # [BN,16]
                code16 = (sub_byte >> nib[None, :]) & 0xF             # [BN,16]
                sc16 = tl.load(
                    compose_ptr + super_e[:, None] * 16 + code16)     # [BN,16]
            else:
                grpv = s * 16 + grp16v                                # [16]
                sc16 = tl.load(
                    scale_ptr + offs_n_i[:, None] * stride_sn + grpv[None, :],
                    mask=mask_n[:, None], other=0.0)                 # [BN,16]
            sc = tl.reshape(tl.broadcast_to(sc16[:, :, None],
                                            (BLOCK_N, 16, 16)), (BLOCK_N, 256))
            w = (val * sc).to(tl.bfloat16)
        else:
            w = (val * sc_row[:, None]).to(tl.bfloat16)            # [BN,256]

        xcols = (s * 256 + kcol).to(tl.int64)
        x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_xm
                    + xcols[None, :] * stride_xk,
                    mask=mask_m[:, None], other=0.0).to(tl.bfloat16)  # [BM,256]
        # bf16 MMA (INV-2 waived): y += x @ w^T.
        acc += tl.dot(x, tl.trans(w))

    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             y, mask=mask_m[:, None] & mask_n[None, :])


def cb_decode_linear(
    x: torch.Tensor,             # (..., K) activations (bf16/fp16)
    qw_padded: torch.Tensor,     # (N, row_bytes+8) uint8, +8 pad for the window
    cb_flat: torch.Tensor,       # (cb_total,) bf16 flat codebook(s), concatenated
    cb_row_offset: torch.Tensor,  # (N,) int32 per-row base into cb_flat
    scale: torch.Tensor,         # fp4 v1: (N, n_sb*16) fp32 ; fp8: (N,) fp32 ;
                                 # fp4 v2: unused dummy
    compose: torch.Tensor,       # fp4 v2: (4096,) fp32 compose table ; else dummy
    *, N: int, K: int,
    k_bits: int, n_sub: int, type_size: int, is_fp4: bool, is_v2: bool = False,
) -> torch.Tensor:
    """Launch the decode-GEMM. Returns (..., N). M-gated: a small BLOCK_M for
    the decode regime (M<=16), a larger tile for prefill — mirrors GGUF's
    MMVQ/MMQ split (quantization/linear.py:34-57), one Triton kernel either
    way. The dense weight is never materialized (INV-1)."""
    sub_dim = 8 // n_sub
    # Ceil-first split (encoder _bit_split): widths, codeword offsets, and
    # flat-codebook element bases per sub; trailing unused subs zeroed.
    base, extra = divmod(k_bits, n_sub)
    ws = [base + (1 if i < extra else 0) for i in range(n_sub)] \
        + [0] * (4 - n_sub)
    offs = [sum(ws[:i]) for i in range(4)]
    bases = [sum(sub_dim << w for w in ws[:i] if w) for i in range(4)]
    orig_shape = x.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    block_m = 16 if M <= 16 else 64
    block_n = 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    stride_sn = scale.stride(0) if (is_fp4 and not is_v2) else 0
    _cb_decode_gemm_kernel[grid](
        x2, qw_padded, cb_flat, cb_row_offset, scale, compose, y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        qw_padded.stride(0),
        y.stride(0), y.stride(1),
        stride_sn,
        K_BITS=k_bits, N_SUB=n_sub, SUB_DIM=sub_dim,
        SUB_W0=ws[0], SUB_W1=ws[1], SUB_W2=ws[2], SUB_W3=ws[3],
        SUB_OFF1=offs[1], SUB_OFF2=offs[2], SUB_OFF3=offs[3],
        SUB_BASE1=bases[1], SUB_BASE2=bases[2], SUB_BASE3=bases[3],
        SIGNED=(n_sub == 1),
        TYPE_SIZE=type_size, IS_FP4=is_fp4, LAYOUT_V2=is_v2,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    return y.reshape(*orig_shape[:-1], N)
