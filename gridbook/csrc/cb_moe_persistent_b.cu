// SPDX-License-Identifier: Apache-2.0
// Large-M grouped MoE decode-in-mainloop — PERSISTENT-B ALONG M (ROADMAP K1.1,
// 2026-08-01 performance audit §3 P2b).  FP4-CB two-tier v2 experts, BF16
// activations, sm_120/sm_121.
//
// ===========================================================================
// WHAT THIS REPLACES
// ===========================================================================
// The FP4-CB MoE quality prefill (moe.py `_apply_prefill_native_bf16`) is:
//
//   for each expert chunk:  cb_expand_fp4_v2(w[chunk]) -> BF16 [Ec,N,K] in HBM
//                           cb_bf16_grouped_mm_out(a_sorted, that tile, ...)
//
// The expansion runs over EVERY expert, routed or not, and materializes 2
// bytes/weight of HBM transient that the GEMM then reads back.  At Laguna
// scale that expand is ~35% of layer time (docs/BENCHMARKS.md, quoted by the
// audit).  It is 4x the transient traffic of FP8-CB's direct-to-E4M3 expand,
// which is structural cause (b) of the NVFP4-CB deficit.
//
// This kernel deletes the transient outright: the packed CB bytes (~2.3-2.9
// bpw, i.e. ~0.3 B/weight) are read ONCE, decoded to BF16 in shared memory,
// and consumed in-place by the MMA.  Nothing per-weight is ever written to
// HBM.  Experts with no routed rows are never touched at all.
//
// ===========================================================================
// WHY THIS IS NOT THE MEASURED-NEGATIVE PERSISTENT-N KERNEL
// ===========================================================================
// The quarantined dense persistent-N schedule owned an N-slice PER CTA and
// walked M with CUDA-core / SM89-class atoms on the DENSE path; it measured
// 2-5.7x slower and stays closed (ROADMAP K1.3, and dense large-M is K1.3's
// business, not this file's).  This kernel:
//
//   * is MoE-only — it takes `expert_ends` and a stacked [E,N,W] expert
//     tensor, and there is no dense entry point in this translation unit;
//   * uses tensor cores (`mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`), not
//     CUDA cores;
//   * owns an (expert, N-tile) pair, not a bare N-slice, and the win comes
//     from amortizing a DECODE over the expert's routed rows — a cost the
//     dense persistent-N kernel did not have and could not amortize.
//
// ===========================================================================
// THE SCHEDULE
// ===========================================================================
// Work unit = (expert e, N-tile j).  Grid geometry is `E * ceil(N/TN)` CTAs,
// a pure function of the LAYER SHAPE — no routing value, no device read, no
// host sync — so the launch is CUDA-graph capturable as-is.  A CTA:
//
//   1. reads its expert's exact routed segment [expert_ends[e-1],
//      expert_ends[e]) from device memory.  An EMPTY expert exits here: two
//      int32 loads and a return, which is what "empty experts cost ~zero"
//      means concretely;
//   2. loops that segment in M-tiles of TM rows -- the M-LOOP LIVES INSIDE
//      THE KERNEL.  Everything hoisted above it (the segment bounds, the
//      expert's byte-plane base pointer, the N-tile column base, the format
//      descriptor) is computed once per (expert, N-tile), not once per
//      M-tile as an M-tile-indexed grid would force;
//   3. for each M-tile, walks K in TK=64 columns.  Every 256 K columns (one
//      CB superblock) it stages that superblock's packed bytes for all TN
//      weight rows into shared memory ONCE, and decodes four TK-slices out of
//      that one staging.  Each decoded BF16 value is then multiplied by TM
//      activation rows before it is discarded.
//
// So the decode is amortized TM-fold over M, and TM is chosen as large as the
// accumulator register budget allows.  For the routed-MoE regime this file
// targets (E=128..256, top_k=4..8, T=128..2048 => tens of rows per expert),
// TM=128 means an expert's whole segment is ONE M-tile and its weight tile is
// decoded EXACTLY ONCE — the K1.1 objective, reached without any padding.
//
// Contrast with `cb_fused_gemm.cu` / `cb_fused_fp4_gemm.cu`, whose CTAs are
// M-TILE-INDEXED: there the tile scheduler hands each CTA one (m_tile,
// n_tile) pair, so a second M-tile of the same expert is a different CTA and
// re-decodes B from scratch.  That is correct and fast for mid-M (it is why
// FP8-CB's fused lane is gated at M<=128) but it is exactly the property that
// makes it lose at large M (FP8 measures 0.22x at M~1400).  Inverting the
// ownership is the whole point of this file.
//
// ===========================================================================
// ROUTING LAYOUT: EXACT SEGMENTS, NOT PADDED TILES
// ===========================================================================
// Gridbook's CUTLASS grouped lanes consume the row-padded, tile-indexed
// construction (`csrc/cb_grouped_common.hpp`) because upstream CUTLASS 4.3.4
// has no sm120 ptr-array collective and a uniform tile cannot straddle two
// experts.  This kernel is hand-assembled, so it has no such constraint and
// takes the EXACT per-expert segments (`expert_ends`, the cumulative routed
// row count) that `_apply_prefill_native_bf16` already builds.  That choice:
//
//   * costs zero padded rows.  The padded layout rounds every expert up to a
//     whole TileM: at E=256/TileM=128 a T=512,top_k=8 prefill (4096 routed
//     rows) can expand to as many as 4096+256*127 rows of gathered
//     activations and GEMM work.  The exact-segment layout gathers 4096;
//   * costs ZERO host reads.  The padded lanes spend one `.item()` on the
//     real block total (`PRISMAQUANT_CB_GROUPED_TRIM`) to avoid launching the
//     static-capacity tail.  This kernel's grid does not depend on the
//     routing at all, so there is nothing to trim and nothing to read;
//   * reuses moe.py's existing machinery verbatim — the same stable argsort,
//     the same `expert_ends = cumsum(bincount(...))`, the same
//     `index_select`/`index_add_` combine that the DEFAULT quality path uses
//     today.  No new gather machinery was invented for this lane.
//
// The price is intra-CTA M-tile quantization (an expert with 1 routed row
// still runs one TM-row MMA tile).  That waste is ARITHMETIC ONLY — it costs
// no HBM traffic, and this kernel is decode/bandwidth-bound — and the
// epilogue masks it, so it never reaches a token.
//
// ===========================================================================
// NUMERICS
// ===========================================================================
// * WEIGHTS.  `cb_decode_codeword` below is a line-by-line transcription of
//   `cb_gemv_v2.cu::cb_expand_v2_kernel`'s inner body: the same ceil-first
//   two-way bit split, the same 8-byte codeword window, the same
//   `compose[super_e*16 + code16]` two-tier scale compose, the same
//   `bf16_rn(f32(cb_entry) * sc)` per-value round.  The only difference is
//   that the 8-byte window is assembled from three u32 shared-memory reads
//   instead of eight byte reads — algebraically the identical 64-bit window
//   (the same substitution `cb_gemv_v2_kernel` already makes).  It is a
//   TESTED identity, not an asserted one: `cb_moe_persistent_b_decode`
//   exposes this exact device function and the suite compares it to
//   `cb_expand_v2` with `torch.equal`.
// * ACTIVATIONS.  Untouched.  The caller passes the same BF16 group-16 RTN
//   QDQ payload the quality path already computes; this kernel consumes it as
//   plain BF16.  There is no new activation contract, unlike the fused
//   native-NVFP4 lane.
// * ACCUMULATION.  FP32, one BF16 round in the epilogue, alpha=1/beta=0 —
//   identical in kind to `cb_bf16_grouped_gemm.cu`.  What differs from the
//   shipping bridge is the FP32 REDUCTION ORDER (different tile shape,
//   different K walk, different warp partition).  Reassociation-class: the
//   same requalification surface W4's sm12x-native BF16 lane and the promoted
//   FP8 mid-M fused kernel cleared.  Nothing else about the served numerics
//   changes.
//
// ===========================================================================
// SMEM / OCCUPANCY BUDGET  (measured; csrc/tools/persistent_b_probe.cu)
// ===========================================================================
// Per-CTA dynamic shared memory, K-major 16-byte-XOR-swizzled tiles:
//
//     A stages   2 * TM * TK * 2 B      (double-buffered, cp.async)
//     B decoded  1 * TN * TK * 2 B      (rewritten every TK stage)
//     packed     TN * ts_pad B          (ts_pad = round4(4k+9) + 8)
//
// TK is fixed at 64 BF16 = 128 B per row = exactly the 8 sixteen-byte chunks
// an XOR swizzle needs to make `ldmatrix` conflict-free over 8 rows, and it
// divides the 256-column CB superblock evenly (4 TK-slices per superblock).
// The codebook LUT stays in GLOBAL memory and is read through `__ldg`; see
// `kSmemReservedPerCta` below for the measurement that rejected staging it.
//
// Compiled ladder, shared memory quoted at the largest rung (k=24, the widest
// packed superblock).  `cb_moe_persistent_b_configs()` attests these at
// runtime; python enumerates THAT, never this comment.
//
//  cfg  TM   TN  warps thr      A      B     pk    smem   CTAs/SM  accum/thr
//  ---  ---  ---  ----- ---  ------  -----  -----  -----  -------  ---------
//   1   128   64    8   256  32,768  8,192  7,424  48,384    2         32
//   2    64   64    4   128  16,384  8,192  7,424  32,000    3         32
//   3   128   32    4   128  32,768  4,096  3,712  40,576    2         32
//   4    64  128    8   256  16,384 16,384 14,848  47,616    2         32
//
// Every config holds >= 2 CTAs/SM INCLUDING the ~1 KiB the hardware reserves
// per CTA, and `run_persistent_b` enforces that as a TORCH_CHECK so a future
// tile cannot quietly slip to one.  That is the lesson W4 paid for: its first
// BF16 collective wanted 75,776 B, got 1 CTA/SM on GB10, and the sweep of
// alternatives never recovered it.
//
// TWO WIDER TILES WERE COMPILED, MEASURED AND DROPPED: 128x128 (64,000 B) and
// 256x64 (81,152 B).  Both fall to 1 CTA/SM, and neither won a single cell of
// the sweep — 256x64 halves the decode repetition at large rows-per-expert,
// which is exactly the regime it should own, and still lost to 128x64 there
// (18.6 vs 7.8 ms on DSV4 w2 at T=2048).  Occupancy dominates the decode
// amortization on this device.  Full sweep: docs/KERNELS.md.
//
// AUTO SELECTION picks cfg 4 below ~64 mean routed rows per expert and cfg 1
// above, from SHAPES alone (P and E) so the choice stays a trace-time constant
// -- see `run_persistent_b`.
//
// ===========================================================================
// FORMAT SCOPE
// ===========================================================================
// FP4-CB two-tier v2, product mode n_sub=2, `type_size == 4*k + 9`, k in
// [1,24], K % 256 == 0.  This is the quality path that has no fused
// alternative at any M.  The decode stage is factored behind
// `cb_decode_codeword` + a format descriptor so an FP8-CB payload (a flat
// e4m3 plane plus a per-(expert,row) fp32 scale) can be slotted in later
// without touching the schedule; that is deliberately NOT implemented here
// (ROADMAP K1.2 owns the FP8 rung surface).
//
// ===========================================================================
// EXPERIMENTAL DIRECT-TO-REGISTER B VARIANT
// ===========================================================================
// `cb_moe_persistent_b_prefill_d2r` is a DEFAULT-OFF sibling entry point in
// THIS extension.  It preserves the activation and BF16 weight-decode
// contract above, but does not materialize the decoded `TN x TK` B tile in
// shared memory.  For `mma.sync.m16n8k16.row.col`, a lane owns B row
// `n = lane >> 2` and the pair `kpair = 2 * (lane & 3)`: its two B registers
// hold `(kpair, kpair+1)` and `(kpair+8, kpair+9)`.  `cb_decode_pair` produces
// exactly those two rounded BF16 values from each of the two codewords.
//
// Removing B's `ldmatrix` also removes the reason the baseline partitions
// warps over 32-column N slices.  The candidate sets WN=WARPS and WM=1: each
// warp owns a UNIQUE 8- or 16-column N slice, and the four lanes sharing an N
// row decode the four distinct pairs of a codeword exactly once.  There is no
// full-codeword shuffle or shared-memory exchange: three width-4 shuffles
// broadcast the two index pairs and shared scale inside each four-lane N-row
// subgroup.  There is no cross-warp duplicate B decode.  The trade is 2-4x
// more `ldmatrix` reads of the already-resident A tile, which is deliberately
// left for the baseline-vs-candidate benchmark to measure.
//
// The candidate retains double-buffered A and the packed-superblock staging.
// Its packed staging happens AFTER barrier (1): unlike the baseline, MMA
// reads `sPk` directly, so that CTA barrier is the WAR fence between the last
// pair decode of superblock s and the first overwrite with superblock s+1.
// Each D2R warp stages exactly the contiguous sPk rows only that warp consumes
// and publishes them with `__syncwarp`; no D2R consumer crosses a warp, so the
// baseline's CTA-wide decoded-B publication barrier is unnecessary.  This
// ownership and ordering are load-bearing.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#define DEVINL __device__ __forceinline__

namespace {

// ---------------------------------------------------------------------------
// Numeric primitives — byte-identical to cb_gemv_v2.cu.
// ---------------------------------------------------------------------------
DEVINL float bf16_to_f32(uint16_t v) {
  __nv_bfloat16_raw r;
  r.x = v;
  return __bfloat162float(__nv_bfloat16(r));
}

DEVINL uint16_t f32_to_bf16_rn(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// ---------------------------------------------------------------------------
// FP4-CB v2 format descriptor.  `Split2` in cb_gemv_v2.cu, carried by value so
// the decode stage is one self-contained device function.
//
// Ceil-first bit split (== the producer's cb_layout.bit_split(k, 2)):
// sub0 takes ceil(k/2) bits in the LOW bits, sub1 takes floor(k/2).  sub_dim
// is 4, so sub1's flat element base is `4 << w0`.
// ---------------------------------------------------------------------------
struct CbFp4V2Fmt {
  int k_bits;
  int type_size;
  int scale_off;   // 4 * k_bits: first byte of the 9-byte two-tier plane
  int w0;
  uint32_t m0;
  uint32_t m1;
  int e1;          // flat element offset of sub-codebook 1

  __host__ __device__ CbFp4V2Fmt() = default;

  __host__ explicit CbFp4V2Fmt(int k, int ts) {
    k_bits = k;
    type_size = ts;
    scale_off = 4 * k;
    w0 = (k + 1) >> 1;
    const int w1 = k >> 1;
    m0 = (1u << w0) - 1u;
    m1 = (1u << w1) - 1u;
    e1 = 4 << w0;
  }
};

// ---------------------------------------------------------------------------
// THE DECODE STAGE.  One CB codeword -> 8 consecutive BF16 weights.
//
// `s32` points at a 4-byte-aligned copy of ONE superblock's `type_size` bytes
// (staged in shared memory by the mainloop, or in global memory by the probe
// binding); `c` is the codeword index 0..31 within that superblock, which owns
// output columns [8*c, 8*c+8).
//
// Every line below has a counterpart in cb_expand_v2_kernel.  The one
// substitution is the codeword window: the expander assembles 64 bits from
// eight byte loads starting at byte `b0`, this assembles the same 64-bit field
// from u32 loads starting at byte `b0 & ~3` and compensates in `rem` --
// exactly what cb_gemv_v2_kernel does for its own smem row stage.  The bits
// selected are identical, so `code` is identical.
// ---------------------------------------------------------------------------
//
// `lut_smem` selects a plain generic load over `__ldg` because `__ldg` is
// defined only for GLOBAL addresses; it changes where the identical bytes are
// read from, never what they are.
DEVINL uint64_t cb_extract_code(const uint32_t* __restrict__ s32,
                               int c,
                               const CbFp4V2Fmt& f) {
  const int bitpos = c * f.k_bits;
  const int b0 = bitpos >> 3;
  const int rem = ((b0 & 3) << 3) + (bitpos & 7);
  const int widx = b0 >> 2;
  const uint32_t wa = s32[widx];
  const uint32_t wb = s32[widx + 1];
  uint64_t code = (((uint64_t)wb << 32) | (uint64_t)wa) >> rem;
  if (rem + f.k_bits > 64) {
    code |= (uint64_t)s32[widx + 2] << (64 - rem);
  }
  code &= (f.k_bits >= 64) ? ~0ull : ((1ull << f.k_bits) - 1ull);
  return code;
}

template <bool LutSmem>
DEVINL uint4 cb_decode_codeword(const uint32_t* __restrict__ s32,
                                const uint8_t* __restrict__ sbytes,
                                int c,
                                const CbFp4V2Fmt& f,
                                const uint16_t* lut,
                                const float* __restrict__ compose) {
  const uint64_t code = cb_extract_code(s32, c, f);

  // Two-tier scale: one uint8 super exponent per superblock, then 8 bytes of
  // nibbles, one nibble per group of two codewords (== per 16 output columns).
  const int grp = c >> 1;
  const uint32_t super_e = (uint32_t)sbytes[f.scale_off];
  const uint32_t sub_byte = (uint32_t)sbytes[f.scale_off + 1 + (grp >> 1)];
  const uint32_t code16 = (sub_byte >> ((grp & 1) * 4)) & 0xFu;
  const float sc = __ldg(compose + super_e * 16u + code16);

  const uint32_t i0 = (uint32_t)code & f.m0;
  const uint32_t i1 = (uint32_t)(code >> f.w0) & f.m1;
  const uint2* p0 = reinterpret_cast<const uint2*>(lut + (int64_t)i0 * 4);
  const uint2* p1 =
      reinterpret_cast<const uint2*>(lut + (int64_t)f.e1 + (int64_t)i1 * 4);
  const uint2 q0 = LutSmem ? *p0 : __ldg(p0);
  const uint2 q1 = LutSmem ? *p1 : __ldg(p1);

  const uint16_t h[8] = {
      (uint16_t)(q0.x & 0xffffu), (uint16_t)(q0.x >> 16),
      (uint16_t)(q0.y & 0xffffu), (uint16_t)(q0.y >> 16),
      (uint16_t)(q1.x & 0xffffu), (uint16_t)(q1.x >> 16),
      (uint16_t)(q1.y & 0xffffu), (uint16_t)(q1.y >> 16)};
  uint16_t o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    o[j] = f32_to_bf16_rn(bf16_to_f32(h[j]) * sc);
  }
  uint4 out;
  out.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
  out.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
  out.z = (uint32_t)o[4] | ((uint32_t)o[5] << 16);
  out.w = (uint32_t)o[6] | ((uint32_t)o[7] << 16);
  return out;
}

DEVINL uint32_t cb_scale_pair(uint32_t q, float sc) {
  const uint16_t h0 = (uint16_t)(q & 0xffffu);
  const uint16_t h1 = (uint16_t)(q >> 16);
  const uint16_t o0 = f32_to_bf16_rn(bf16_to_f32(h0) * sc);
  const uint16_t o1 = f32_to_bf16_rn(bf16_to_f32(h1) * sc);
  return (uint32_t)o0 | ((uint32_t)o1 << 16);
}

// Direct-to-register twin of TWO `cb_decode_codeword` calls: return the two
// adjacent BF16 pairs one lane needs for a complete k16 MMA B fragment.
//
// The four lanes in a width-4 subgroup share one output N row and call this
// with the same EVEN `c0`; `pair == lane & 3`.  Subgroup lane 0 performs the
// two bit-window extractions and the ONE scale compose shared by c0/c0+1,
// then broadcasts two packed (i0,i1) index pairs and the scale bits.  Every
// lane loads exactly its own uint32 BF16 pair from each codeword.  Thus one
// k16/N-row fragment costs two code extracts, one compose load, four pair LUT
// loads per codeword and three width-4 shuffles -- no 4x bit/compose repeat,
// no full-codeword broadcast, and no decoded B shared-memory exchange.
//
// k<=24 makes each half-index <=12 bits, so packing i0|i1<<16 is exact.
DEVINL uint2 cb_decode_k16_pairs(const uint32_t* __restrict__ s32,
                                const uint8_t* __restrict__ sbytes,
                                int c0,
                                int pair,
                                const CbFp4V2Fmt& f,
                                const uint16_t* lut,
                                const float* __restrict__ compose) {
  uint32_t packed_idx0 = 0;
  uint32_t packed_idx1 = 0;
  uint32_t scale_bits = 0;
  if (pair == 0) {
    const uint64_t code0 = cb_extract_code(s32, c0, f);
    const uint64_t code1 = cb_extract_code(s32, c0 + 1, f);
    const uint32_t i00 = (uint32_t)code0 & f.m0;
    const uint32_t i01 = (uint32_t)(code0 >> f.w0) & f.m1;
    const uint32_t i10 = (uint32_t)code1 & f.m0;
    const uint32_t i11 = (uint32_t)(code1 >> f.w0) & f.m1;
    packed_idx0 = i00 | (i01 << 16);
    packed_idx1 = i10 | (i11 << 16);

    // c0 is even, so c0 and c0+1 are exactly the two codewords in `grp` and
    // share the same two-tier scale by the serialized format contract.
    const int grp = c0 >> 1;
    const uint32_t super_e = (uint32_t)sbytes[f.scale_off];
    const uint32_t sub_byte =
        (uint32_t)sbytes[f.scale_off + 1 + (grp >> 1)];
    const uint32_t code16 = (sub_byte >> ((grp & 1) * 4)) & 0xFu;
    scale_bits = __float_as_uint(
        __ldg(compose + super_e * 16u + code16));
  }
  // Every warp lane calls the helper. `width=4` partitions the warp into
  // eight independent four-lane groups, so srcLane=0 names each subgroup's
  // own leader. A uniform full mask is the least ambiguous sync contract.
  packed_idx0 = __shfl_sync(0xffffffffu, packed_idx0, 0, 4);
  packed_idx1 = __shfl_sync(0xffffffffu, packed_idx1, 0, 4);
  scale_bits = __shfl_sync(0xffffffffu, scale_bits, 0, 4);
  const float sc = __uint_as_float(scale_bits);

  const bool second_sub = pair >= 2;
  const int64_t elem_base = second_sub ? (int64_t)f.e1 : 0;
  const uint32_t idx0 = second_sub ? (packed_idx0 >> 16)
                                   : (packed_idx0 & 0xffffu);
  const uint32_t idx1 = second_sub ? (packed_idx1 >> 16)
                                   : (packed_idx1 & 0xffffu);
  const uint32_t* p0 = reinterpret_cast<const uint32_t*>(
      lut + elem_base + (int64_t)idx0 * 4);
  const uint32_t* p1 = reinterpret_cast<const uint32_t*>(
      lut + elem_base + (int64_t)idx1 * 4);
  return make_uint2(cb_scale_pair(__ldg(p0 + (pair & 1)), sc),
                    cb_scale_pair(__ldg(p1 + (pair & 1)), sc));
}

// ---------------------------------------------------------------------------
// PTX helpers.
// ---------------------------------------------------------------------------
DEVINL uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVINL void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

DEVINL void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                         const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 16-byte cp.async with a predicated source size: `pred == false` zero-fills
// the destination, which is exactly the out-of-segment A row behaviour we
// want (those rows are masked in the epilogue anyway, but a zero tile keeps
// the accumulator finite regardless of what is in HBM past the segment).
DEVINL void cp_async_16(uint32_t dst, const void* src, bool pred) {
  const int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(dst), "l"(src), "r"(bytes));
}

DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
DEVINL void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// ---------------------------------------------------------------------------
// Shared-memory tile addressing.
//
// A [rows][TK] BF16 tile is stored as `rows * (TK/8)` sixteen-byte chunks with
// the chunk index XOR-swizzled by the low 3 bits of the row.  TK == 64 makes a
// row exactly 8 chunks == 128 B == the full 32-bank width, so the 8 rows an
// `ldmatrix` tile touches land on 8 distinct chunk columns: conflict-free.
// ---------------------------------------------------------------------------
constexpr int kTK = 64;                 // BF16 columns per mainloop stage
constexpr int kChunks = kTK / 8;        // 16-byte chunks per row
constexpr int kSuperblock = 256;        // CB v2 columns per superblock
constexpr int kStagesPerSb = kSuperblock / kTK;

DEVINL int swz_chunk(int row, int chunk) {
  return row * kChunks + (chunk ^ (row & 7));
}

// Address of BF16 element (row, kcol) where kcol is a multiple of 8.
DEVINL uint16_t* tile_at(uint16_t* base, int row, int kcol) {
  return base + (int64_t)swz_chunk(row, kcol >> 3) * 8;
}

// ---------------------------------------------------------------------------
// The persistent-B mainloop.
//
// Template parameters are the tile shape and whether B is decoded directly
// into registers; WARPS*32 threads per CTA.  The baseline warp grid is
// WN=TN/32, WM=WARPS/WN.  D2R uses WN=WARPS, WM=1 so each warp owns a unique
// N slice.  Both arrangements hold exactly 32 accumulator registers/thread
// for every compiled config.
// ---------------------------------------------------------------------------
template <int TM, int TN, int WARPS, bool D2R = false>
__global__ __launch_bounds__(WARPS * 32) void cb_moe_persistent_b_kernel(
    const uint16_t* __restrict__ a,          // [P, K] bf16 (routed, sorted)
    const uint8_t* __restrict__ qw,          // [E, N, row_bytes] packed CB
    const uint16_t* __restrict__ lut,        // flat product codebook (bf16)
    const float* __restrict__ compose,       // [256*16] two-tier compose
    const int32_t* __restrict__ expert_ends, // [E] cumulative routed rows
    uint16_t* __restrict__ y,                // [P, N] bf16
    const int64_t P, const int64_t N, const int64_t K,
    const CbFp4V2Fmt fmt,
    const int ts_pad,
    const int n_tiles,
    const int64_t total_wu) {
  constexpr int kThreads = WARPS * 32;
  constexpr int WN = D2R ? WARPS : TN / 32;
  constexpr int WM = WARPS / WN;
  constexpr int MATOM = (TM / WM) / 16;
  constexpr int NATOM = (TN / WN) / 8;
  static_assert(TN % 32 == 0, "TN must be a multiple of 32");
  static_assert(TN % (WN * 8) == 0, "warp N slices must tile in m16n8 atoms");
  static_assert(WM * WN == WARPS, "warp grid must cover the CTA");
  static_assert(TM % (16 * WM) == 0, "TM must tile the warp rows");
  static_assert(MATOM * NATOM * 4 <= 128, "accumulator register budget");
  // The MMA block issues N-atoms in PAIRS (one `ldmatrix.x4` feeds two), so an
  // odd NATOM would silently drop the last 8 output columns instead of failing
  // to compile.
  static_assert(D2R || NATOM % 2 == 0,
                "baseline N-atoms are consumed in ldmatrix.x4 pairs");

  extern __shared__ __align__(16) uint8_t smem_raw[];
  uint16_t* sA = reinterpret_cast<uint16_t*>(smem_raw);
  uint16_t* sB = sA + 2 * TM * kTK;
  uint8_t* sPk = reinterpret_cast<uint8_t*>(
      D2R ? sB : sB + TN * kTK);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp / WN;
  const int wn = warp % WN;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int n_sb = (int)(K / kSuperblock);
  const int64_t row_bytes = (int64_t)n_sb * fmt.type_size;
  const int total_stages = n_sb * kStagesPerSb;

  for (int64_t wu = blockIdx.x; wu < total_wu; wu += gridDim.x) {
    const int e = (int)(wu / n_tiles);
    const int n0 = (int)(wu % n_tiles) * TN;

    // Exact routed segment for this expert.  An empty expert stops here.
    //
    // CLAMPED TO [0, P] because `expert_ends` is DEVICE data: the host can
    // check its dtype, length and device, but not its VALUES, and these bounds
    // drive both the A-row predicate and the epilogue's store predicate. An
    // out-of-range or non-monotone entry would otherwise read and write past
    // `a` and `y` -- a fault at best, and at worst two CTAs writing the same
    // rows. Serving builds this as a cumsum of a bincount so it always ends at
    // exactly P, but the binding is public and the clamp is one register `min`
    // on a value already loaded once per work unit.
    const int Pi = (int)P;
    const int raw_lo = (e == 0) ? 0 : expert_ends[e - 1];
    const int raw_hi = expert_ends[e];
    const int m_lo = raw_lo < 0 ? 0 : (raw_lo > Pi ? Pi : raw_lo);
    const int m_hi = raw_hi < m_lo ? m_lo : (raw_hi > Pi ? Pi : raw_hi);
    if (m_hi <= m_lo) {
      continue;
    }

    const uint8_t* qw_e = qw + (int64_t)e * N * row_bytes;

    // ---- the in-kernel M loop: stream this expert's rows through B --------
    for (int m0 = m_lo; m0 < m_hi; m0 += TM) {
      float acc[MATOM][NATOM][4];
#pragma unroll
      for (int i = 0; i < MATOM; ++i)
#pragma unroll
        for (int j = 0; j < NATOM; ++j)
#pragma unroll
          for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

      // Prologue: A stage 0.
      {
#pragma unroll
        for (int it = 0; it < (TM * kChunks + kThreads - 1) / kThreads; ++it) {
          const int i = tid + it * kThreads;
          if (TM * kChunks % kThreads != 0 && i >= TM * kChunks) break;
          const int m = i / kChunks;
          const int c = i - m * kChunks;
          const int gm = m0 + m;
          cp_async_16(smem_addr(tile_at(sA, m, c * 8)),
                      a + (int64_t)gm * K + c * 8, gm < m_hi);
        }
      }
      cp_async_commit();

      for (int st = 0; st < total_stages; ++st) {
        // Packed superblock staging, once per kStagesPerSb TK-slices.  Its
        // only reader is `decode_B`, and barrier (2) of the previous stage
        // separates that read from this write.  It touches neither `sA` nor
        // `sB`, so it is safe this side of barrier (1).
        if constexpr (!D2R) {
          if (st % kStagesPerSb == 0) {
            const int sbi = st / kStagesPerSb;
            for (int n = warp; n < TN; n += WARPS) {
              const int gn = n0 + n;
              uint8_t* dst = sPk + (int64_t)n * ts_pad;
              if (gn < N) {
                const uint8_t* src = qw_e + (int64_t)gn * row_bytes +
                                     (int64_t)sbi * fmt.type_size;
                for (int b = lane; b < fmt.type_size; b += 32) {
                  dst[b] = __ldg(src + b);
                }
              } else {
                for (int b = lane; b < fmt.type_size; b += 32) dst[b] = 0;
              }
              for (int b = fmt.type_size + lane; b < ts_pad; b += 32) {
                dst[b] = 0;
              }
            }
          }
        }

        // Wait for THIS stage's A slice (the only group outstanding here) and
        // publish it.  Barrier (1) does double duty, and the second duty is
        // what makes the two-barrier loop correct: every thread reaches it
        // only after finishing `mma(st-1)`, so once past it no thread is
        // still reading `sA[(st-1)&1]` or `sB`.  The prefetch below and the
        // decode below therefore both write buffers nobody is reading.
        cp_async_wait<0>();
        __syncthreads();                                            // (1)

        // Prefetch the NEXT A slice into the other buffer.  This must come
        // AFTER barrier (1): `(st+1) & 1 == (st-1) & 1`, so issuing it before
        // the barrier would let one warp's async copy overwrite the exact tile
        // another warp is still feeding to `mma(st-1)` -- a WAR race that is
        // invisible while warps happen to run in lockstep and corrupts as soon
        // as they do not.
        if (st + 1 < total_stages) {
          const int kb = (st + 1) * kTK;
          uint16_t* dstbuf = sA + ((st + 1) & 1) * (TM * kTK);
#pragma unroll
          for (int it = 0; it < (TM * kChunks + kThreads - 1) / kThreads;
               ++it) {
            const int i = tid + it * kThreads;
            if (TM * kChunks % kThreads != 0 && i >= TM * kChunks) break;
            const int m = i / kChunks;
            const int c = i - m * kChunks;
            const int gm = m0 + m;
            cp_async_16(smem_addr(tile_at(dstbuf, m, c * 8)),
                        a + (int64_t)gm * K + kb + c * 8, gm < m_hi);
          }
        }
        cp_async_commit();

        // D2R consumes sPk inside MMA, so only barrier (1) proves every warp
        // finished reading the PREVIOUS superblock.  Stage the replacement on
        // this side of that barrier. Unlike the baseline's interleaved staging
        // above, warp w owns the exact contiguous rows its D2R N slice reads;
        // a warp-local publication is therefore sufficient. The baseline
        // keeps its established CTA barrier below unchanged.
        if constexpr (D2R) {
          if (st % kStagesPerSb == 0) {
            const int sbi = st / kStagesPerSb;
            constexpr int kNPerWarp = TN / WARPS;
            const int n_begin = warp * kNPerWarp;
            for (int n = n_begin; n < n_begin + kNPerWarp; ++n) {
              const int gn = n0 + n;
              uint8_t* dst = sPk + (int64_t)n * ts_pad;
              if (gn < N) {
                const uint8_t* src = qw_e + (int64_t)gn * row_bytes +
                                     (int64_t)sbi * fmt.type_size;
                for (int b = lane; b < fmt.type_size; b += 32) {
                  dst[b] = __ldg(src + b);
                }
              } else {
                for (int b = lane; b < fmt.type_size; b += 32) dst[b] = 0;
              }
              for (int b = fmt.type_size + lane; b < ts_pad; b += 32) {
                dst[b] = 0;
              }
            }
            __syncwarp();
          }
        }

        // ---- DECODE: TN weight rows x TK columns, once per stage ---------
        if constexpr (!D2R) {
          const int t = st % kStagesPerSb;
          for (int i = tid; i < TN * kChunks; i += kThreads) {
            const int n = i / kChunks;
            const int cc = i - n * kChunks;
            uint4 v;
            if (n0 + n < N) {
              const uint8_t* rowb = sPk + (int64_t)n * ts_pad;
              const uint32_t* r32 =
                  reinterpret_cast<const uint32_t*>(rowb);
              const int cw = t * kChunks + cc;
              v = cb_decode_codeword<false>(r32, rowb, cw, fmt, lut, compose);
            } else {
              v = make_uint4(0u, 0u, 0u, 0u);
            }
            *reinterpret_cast<uint4*>(tile_at(sB, n, cc * 8)) = v;
          }
        }
        if constexpr (!D2R) {
          __syncthreads();                                          // (2)
        }

        // ---- MMA: 4 k16 steps over the staged/direct tile ----------------
        if constexpr (!D2R) {
          const uint16_t* abuf = sA + (st & 1) * (TM * kTK);
#pragma unroll
          for (int kk = 0; kk < kTK / 16; ++kk) {
            const int kbase = kk * 16;
            uint32_t af[MATOM][4];
#pragma unroll
            for (int i = 0; i < MATOM; ++i) {
              const int r = wm * (TM / WM) + i * 16 + (lane & 7) +
                            8 * ((lane >> 3) & 1);
              const int kcol = kbase + 8 * (lane >> 4);
              ldmatrix_x4(af[i], smem_addr(tile_at(
                                     const_cast<uint16_t*>(abuf), r, kcol)));
            }
            uint32_t bf[NATOM / 2][4];
#pragma unroll
            for (int j2 = 0; j2 < NATOM / 2; ++j2) {
              const int tile_idx = lane >> 3;
              const int r = wn * (TN / WN) + j2 * 16 + 8 * (tile_idx >> 1) +
                            (lane & 7);
              const int kcol = kbase + 8 * (tile_idx & 1);
              ldmatrix_x4(bf[j2], smem_addr(tile_at(sB, r, kcol)));
            }
#pragma unroll
            for (int i = 0; i < MATOM; ++i) {
#pragma unroll
              for (int j2 = 0; j2 < NATOM / 2; ++j2) {
                mma_m16n8k16(acc[i][2 * j2], af[i], &bf[j2][0]);
                mma_m16n8k16(acc[i][2 * j2 + 1], af[i], &bf[j2][2]);
              }
            }
          }
        } else {
          const uint16_t* abuf = sA + (st & 1) * (TM * kTK);
#pragma unroll
          for (int kk = 0; kk < kTK / 16; ++kk) {
            const int kbase = kk * 16;
            const int cw0 = (st % kStagesPerSb) * kChunks + 2 * kk;
            // At most four live B registers (NATOM<=2 in the compiled cfgs).
            // Decode them BEFORE loading A, then load one A fragment and
            // immediately consume it.  Materializing af[MATOM][4] here would
            // add 32 live regs in the TM=128 cfgs and risk spills/occupancy.
            uint32_t bf[NATOM][2];
#pragma unroll
            for (int j = 0; j < NATOM; ++j) {
              const int n = wn * (TN / WN) + j * 8 + (lane >> 2);
              const uint8_t* rowb = sPk + (int64_t)n * ts_pad;
              const uint2 v = cb_decode_k16_pairs(
                  reinterpret_cast<const uint32_t*>(rowb), rowb, cw0,
                  lane & 3, fmt, lut, compose);
              if (n0 + n < N) {
                // mma B mapping for row.col m16n8k16:
                //   n=lane>>2, kpair=2*(lane&3)
                //   reg0=(kpair,kpair+1), reg1=(kpair+8,kpair+9).
                bf[j][0] = v.x;
                bf[j][1] = v.y;
              } else {
                bf[j][0] = 0u;
                bf[j][1] = 0u;
              }
            }
#pragma unroll
            for (int i = 0; i < MATOM; ++i) {
              uint32_t af[4];
              const int r = i * 16 + (lane & 7) +
                            8 * ((lane >> 3) & 1);  // WM=1 for D2R
              const int kcol = kbase + 8 * (lane >> 4);
              ldmatrix_x4(af, smem_addr(tile_at(
                                  const_cast<uint16_t*>(abuf), r, kcol)));
#pragma unroll
              for (int j = 0; j < NATOM; ++j) {
                mma_m16n8k16(acc[i][j], af, &bf[j][0]);
              }
            }
          }
        }
      }

      // ---- epilogue: fp32 -> ONE bf16 round, masked to the segment -------
      cp_async_wait<0>();
#pragma unroll
      for (int i = 0; i < MATOM; ++i) {
#pragma unroll
        for (int j = 0; j < NATOM; ++j) {
          const int64_t col = n0 + wn * (TN / WN) + j * 8 + tig * 2;
          if (col >= N) continue;
          const uint32_t packed =
              (uint32_t)f32_to_bf16_rn(acc[i][j][0]) |
              ((uint32_t)f32_to_bf16_rn(acc[i][j][1]) << 16);
          const uint32_t packed_hi =
              (uint32_t)f32_to_bf16_rn(acc[i][j][2]) |
              ((uint32_t)f32_to_bf16_rn(acc[i][j][3]) << 16);
          const int r0 = m0 + wm * (TM / WM) + i * 16 + gid;
          if (r0 < m_hi) {
            *reinterpret_cast<uint32_t*>(y + (int64_t)r0 * N + col) = packed;
          }
          const int r1 = r0 + 8;
          if (r1 < m_hi) {
            *reinterpret_cast<uint32_t*>(y + (int64_t)r1 * N + col) = packed_hi;
          }
        }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Decode probe.  Exposes EXACTLY the mainloop's decode stage as a standalone
// expander so the suite can prove bit-identity against cb_expand_v2 with
// torch.equal rather than a tolerance.  One block per row, one warp per
// superblock; the packed superblock is copied into the same 4-byte-aligned
// shared staging the mainloop uses, so the u32 window path is the one under
// test.
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_moe_persistent_b_decode_kernel(
    const uint8_t* __restrict__ qw,
    const uint16_t* __restrict__ lut,
    const float* __restrict__ compose,
    uint16_t* __restrict__ w,
    const int64_t row0, const int64_t nrows, const int64_t K,
    const CbFp4V2Fmt fmt, const int ts_pad) {
  const int n_sb = (int)(K / kSuperblock);
  const int64_t row_bytes = (int64_t)n_sb * fmt.type_size;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  extern __shared__ __align__(16) uint8_t smem_raw[];
  uint8_t* sPk = smem_raw;

  const int64_t r = row0 + blockIdx.x;
  if (r >= row0 + nrows) return;
  const uint8_t* row = qw + r * row_bytes;

  for (int s = warp; s < n_sb; s += WARPS) {
    uint8_t* dst = sPk + (int64_t)warp * ts_pad;
    const uint8_t* src = row + (int64_t)s * fmt.type_size;
    for (int b = lane; b < fmt.type_size; b += 32) dst[b] = __ldg(src + b);
    for (int b = fmt.type_size + lane; b < ts_pad; b += 32) dst[b] = 0;
    __syncwarp();
    const uint4 v = cb_decode_codeword<false>(
        reinterpret_cast<const uint32_t*>(dst), dst, lane, fmt, lut, compose);
    *reinterpret_cast<uint4*>(w + (r - row0) * K + ((int64_t)s << 8) +
                              (lane << 3)) = v;
    __syncwarp();
  }
}

// Probe the EXACT cooperative helper used by D2R.  One warp stages one
// superblock; each width-4 subgroup owns one k16 fragment, so its four lanes
// cover all four pair values while subgroup lane 0 owns the shared bit/scale
// extraction.  Eight subgroups times two iterations cover all sixteen k16
// codewords / 256 output values.  The output layout is the ordinary dense BF16
// row, making a bitwise comparison to the full decoder non-vacuous.
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void cb_moe_persistent_b_d2r_decode_pairs_kernel(
    const uint8_t* __restrict__ qw,
    const uint16_t* __restrict__ lut,
    const float* __restrict__ compose,
    uint16_t* __restrict__ w,
    const int64_t row0, const int64_t nrows, const int64_t K,
    const CbFp4V2Fmt fmt, const int ts_pad) {
  const int n_sb = (int)(K / kSuperblock);
  const int64_t row_bytes = (int64_t)n_sb * fmt.type_size;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2;
  const int pair = lane & 3;

  extern __shared__ __align__(16) uint8_t smem_raw[];
  uint8_t* sPk = smem_raw;

  const int64_t r = row0 + blockIdx.x;
  if (r >= row0 + nrows) return;
  const uint8_t* row = qw + r * row_bytes;

  for (int s = warp; s < n_sb; s += WARPS) {
    uint8_t* dst = sPk + (int64_t)warp * ts_pad;
    const uint8_t* src = row + (int64_t)s * fmt.type_size;
    for (int b = lane; b < fmt.type_size; b += 32) dst[b] = __ldg(src + b);
    for (int b = fmt.type_size + lane; b < ts_pad; b += 32) dst[b] = 0;
    __syncwarp();
#pragma unroll
    for (int batch = 0; batch < 2; ++batch) {
      const int k16 = group + batch * 8;
      const int c0 = 2 * k16;
      const uint2 v = cb_decode_k16_pairs(
          reinterpret_cast<const uint32_t*>(dst), dst, c0, pair, fmt, lut,
          compose);
      const int64_t base = (r - row0) * K + ((int64_t)s << 8) +
                           (int64_t)c0 * 8 + pair * 2;
      *reinterpret_cast<uint32_t*>(w + base) = v.x;
      *reinterpret_cast<uint32_t*>(w + base + 8) = v.y;
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
constexpr int kSm120SmemCapacity = 101376;   // 99 KiB opt-in dynamic budget

int ts_padded(int type_size) {
  return ((type_size + 3) / 4) * 4 + 8;
}

struct TileCfg {
  int tm;
  int tn;
  int warps;
};

// Compiled configurations, in dispatch order.  Config 1 (index 0) is the
// default: TM=128 decodes an expert's weight tile exactly once whenever the
// expert has at most 128 routed rows, which is the routed-MoE regime.
constexpr TileCfg kCfgs[] = {
    {128, 64, 8},
    {64, 64, 4},
    {128, 32, 4},
    {64, 128, 8},
};
constexpr int kNumCfgs = int(sizeof(kCfgs) / sizeof(kCfgs[0]));

// Shared-memory floor of a config: A stages + (baseline-only) decoded B + the
// packed staging.  D2R removes exactly `TN * TK * sizeof(bf16)` bytes.
int64_t cfg_smem_bytes(TileCfg c, int type_size, bool d2r = false) {
  return (int64_t)2 * c.tm * kTK * 2 +
         (d2r ? 0 : (int64_t)c.tn * kTK * 2) +
         (int64_t)c.tn * ts_padded(type_size);
}

// GB10 / sm_120 shared memory per SM.  Two CTAs is the occupancy floor every
// compiled config holds to: W4's grouped-BF16 lane measured what happens when
// a schedule slips to one CTA per SM (docs/BENCHMARKS.md).
constexpr int64_t kSmemPerSm = 102400;
// Every CTA also reserves ~1 KiB of the SM's shared memory beyond what it
// requests.  MEASURED, not assumed: staging the 4 KiB k=16 codebook pushed the
// default config from 46,336 to 50,432 B, which is still under half the 102,400
// B SM budget but NOT under half of it once the reservation is counted — and
// the config went from 2 CTAs/SM to 1, costing 1.9x on DSV4 w13 T=128 (4.09 ->
// 7.76 ms) while buying about 1% where it did fit.  The resident-codebook idea
// (`cb_gemv_v2.cu`'s DS=2) is therefore REJECTED for this kernel: its gathers
// already hit L1 at the rungs where the table is small, and occupancy is the
// binding constraint on this device.  The budget check below counts the
// reservation so no future config can repeat the mistake silently.
constexpr int64_t kSmemReservedPerCta = 1024;

// ---------------------------------------------------------------------------
// Per-device preparation, on the model of `cb_gemv_v2.cu::cb_gemv_v2_prepare`.
//
// WHY IT IS A SEPARATE ENTRY POINT RATHER THAN A LAZY FIRST-LAUNCH SETUP.
// Opting a kernel in to the 99 KiB dynamic shared-memory budget is a
// `cudaFuncSetAttribute` call, which is NOT stream-ordered work. Doing it
// lazily on first launch means the very first call decides where it happens —
// and for a prefill path that may be the inside of a CUDA-graph capture, or a
// thread that has never launched before. The loader therefore attests this at
// MODEL LOAD (`moe_persistent_b_lane.require_lane`), exactly as the FP4-v2
// expander's contract is attested, so by the time any capture or first forward
// runs, every compiled configuration is already prepared on that device. The
// launcher still calls it, but the atomic fast path makes that a load.
// ---------------------------------------------------------------------------
constexpr int kMaxTrackedDevices = 16;
std::array<std::atomic<bool>, kMaxTrackedDevices> pb_prepared{};
std::mutex pb_prepare_mutex;

void pb_prepare_device(int device) {
  TORCH_CHECK(device >= 0 && device < kMaxTrackedDevices,
              "cb_moe_persistent_b cannot track CUDA device index ", device,
              " (maximum ", kMaxTrackedDevices - 1, ")");
  if (pb_prepared[device].load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(pb_prepare_mutex);
  if (pb_prepared[device].load(std::memory_order_relaxed)) return;

  const c10::cuda::OptionalCUDAGuard guard(c10::Device(c10::kCUDA, device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(prop.major == 12 && (prop.minor == 0 || prop.minor == 1),
              "cb_moe_persistent_b supports only native compute capability "
              "12.0/12.1, but cuda:", device, " reports ", prop.major, ".",
              prop.minor);
  // The REAL requirement is the largest compiled tile at the widest rung, not
  // the whole 99 KiB budget: gating on the budget would reject a device this
  // schedule can serve. (Every cc-12.x part advertises the full 101,376 B, so
  // this has never fired -- which is exactly why it should state the true
  // bound rather than a convenient one.)
  int64_t need = 0;
  for (int i = 0; i < kNumCfgs; ++i) {
    const int64_t s = cfg_smem_bytes(kCfgs[i], 4 * 24 + 9);
    if (s > need) need = s;
  }
  TORCH_CHECK(prop.sharedMemPerBlockOptin >= need,
              "cb_moe_persistent_b needs ", need,
              " B of opt-in shared memory for its largest compiled tile, but "
              "cuda:", device, " advertises ", prop.sharedMemPerBlockOptin,
              " B");
  TORCH_CHECK(prop.sharedMemPerMultiprocessor >= kSmemPerSm,
              "cb_moe_persistent_b sizes its tiles against ", kSmemPerSm,
              " B of shared memory per SM, but cuda:", device, " advertises ",
              prop.sharedMemPerMultiprocessor, " B");

#define PB_SET_MAX_SMEM(TM_, TN_, W_)                                        \
  C10_CUDA_CHECK(cudaFuncSetAttribute(                                       \
      reinterpret_cast<const void*>(                                         \
          cb_moe_persistent_b_kernel<TM_, TN_, W_>),                         \
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSm120SmemCapacity))
  PB_SET_MAX_SMEM(128, 64, 8);
  PB_SET_MAX_SMEM(64, 64, 4);
  PB_SET_MAX_SMEM(128, 32, 4);
  PB_SET_MAX_SMEM(64, 128, 8);
#undef PB_SET_MAX_SMEM

  pb_prepared[device].store(true, std::memory_order_release);
}

void cb_moe_persistent_b_prepare() {
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  pb_prepare_device(device);
}

template <int TM, int TN, int WARPS, bool D2R = false>
void launch_cfg(const uint16_t* a, const uint8_t* qw, const uint16_t* lut,
                const float* compose, const int32_t* expert_ends, uint16_t* y,
                int64_t P, int64_t N, int64_t K, CbFp4V2Fmt fmt, int ts_pad,
                int n_tiles, int64_t total_wu, int64_t smem, unsigned grid,
                cudaStream_t stream) {
  auto kern = cb_moe_persistent_b_kernel<TM, TN, WARPS, D2R>;
  kern<<<grid, WARPS * 32, (size_t)smem, stream>>>(
      a, qw, lut, compose, expert_ends, y, P, N, K, fmt, ts_pad, n_tiles,
      total_wu);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_persistent_b(torch::Tensor out, torch::Tensor a, torch::Tensor qw,
                      torch::Tensor lut, torch::Tensor compose,
                      torch::Tensor expert_ends, int64_t k_bits,
                      int64_t type_size, int64_t cfg_index, bool d2r) {
  TORCH_CHECK(a.is_cuda() && qw.is_cuda() && lut.is_cuda() &&
                  compose.is_cuda() && expert_ends.is_cuda() && out.is_cuda(),
              "cb_moe_persistent_b: every operand must be a CUDA tensor");
  TORCH_CHECK(a.device() == qw.device() && a.device() == lut.device() &&
                  a.device() == compose.device() &&
                  a.device() == expert_ends.device() &&
                  a.device() == out.device(),
              "cb_moe_persistent_b: every operand must be on one CUDA device");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16,
              "cb_moe_persistent_b: activations must be BF16 (the quality "
              "path's QDQ payload), got ",
              a.scalar_type());
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16,
              "cb_moe_persistent_b: output must be BF16");
  TORCH_CHECK(lut.scalar_type() == torch::kBFloat16,
              "cb_moe_persistent_b: the flat product codebook must be BF16");
  TORCH_CHECK(compose.scalar_type() == torch::kFloat32,
              "cb_moe_persistent_b: the two-tier compose table must be FP32");
  TORCH_CHECK(qw.scalar_type() == torch::kUInt8,
              "cb_moe_persistent_b: packed CB weights must be uint8");
  TORCH_CHECK(expert_ends.scalar_type() == torch::kInt32,
              "cb_moe_persistent_b: expert_ends must be int32");
  TORCH_CHECK(a.dim() == 2 && out.dim() == 2 && qw.dim() == 3 &&
                  expert_ends.dim() == 1 && lut.dim() == 1 &&
                  compose.dim() == 1,
              "cb_moe_persistent_b: expected a [P,K], out [P,N], "
              "qw [E,N,row_bytes], expert_ends [E], lut [L], compose [C]");
  TORCH_CHECK(a.is_contiguous() && out.is_contiguous() && qw.is_contiguous() &&
                  lut.is_contiguous() && compose.is_contiguous() &&
                  expert_ends.is_contiguous(),
              "cb_moe_persistent_b: every operand must be contiguous");
  TORCH_CHECK(compose.numel() == 256 * 16,
              "cb_moe_persistent_b: the two-tier compose table must be "
              "256*16 floats, got ",
              compose.numel());

  const int64_t P = a.size(0);
  const int64_t K = a.size(1);
  const int64_t E = qw.size(0);
  const int64_t N = qw.size(1);
  const int64_t row_bytes = qw.size(2);

  TORCH_CHECK(out.size(0) == P && out.size(1) == N,
              "cb_moe_persistent_b: out must be [P,N] = [", P, ",", N,
              "], got ", out.sizes());
  TORCH_CHECK(expert_ends.numel() == E,
              "cb_moe_persistent_b: expert_ends must have one cumulative "
              "count per expert (expected ",
              E, ", got ", expert_ends.numel(), ")");
  TORCH_CHECK(E > 0 && N > 0 && K > 0,
              "cb_moe_persistent_b: E, N and K must be positive");
  TORCH_CHECK(K % kSuperblock == 0,
              "cb_moe_persistent_b: K must be a multiple of the CB superblock "
              "(256), got ",
              K);
  TORCH_CHECK(N % 8 == 0,
              "cb_moe_persistent_b: N must be a multiple of 8 BF16 elements, "
              "got ",
              N);
  TORCH_CHECK(k_bits >= 1 && k_bits <= 24,
              "cb_moe_persistent_b: FP4-CB v2 supports k_bits in [1,24], got ",
              k_bits);
  TORCH_CHECK(type_size == 4 * k_bits + 9,
              "cb_moe_persistent_b: FP4-CB layout v2 requires "
              "type_size == 4*k+9 (n_sub=2 product mode); got type_size=",
              type_size, " for k=", k_bits);
  TORCH_CHECK(row_bytes == (K / kSuperblock) * type_size,
              "cb_moe_persistent_b: packed row stride ", row_bytes,
              " != (K/256)*type_size = ", (K / kSuperblock) * type_size);
  TORCH_CHECK(qw.stride(2) == 1 && qw.stride(1) == row_bytes &&
                  qw.stride(0) == N * row_bytes,
              "cb_moe_persistent_b: the expert stack must be fully "
              "contiguous [E,N,row_bytes]; the kernel derives the per-expert "
              "byte-plane stride as N*row_bytes");
  const int64_t need_lut = (int64_t)(4 << ((k_bits + 1) >> 1)) +
                           (int64_t)(4 << (k_bits >> 1));
  TORCH_CHECK(lut.numel() == need_lut,
              "cb_moe_persistent_b: the flat product codebook must hold "
              "4*(2^ceil(k/2) + 2^floor(k/2)) BF16 values (expected ",
              need_lut, ", got ", lut.numel(), ")");
  TORCH_CHECK(P <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max() &&
                  K <= std::numeric_limits<int>::max() &&
                  E <= std::numeric_limits<int>::max(),
              "cb_moe_persistent_b: dimensions exceed int32");

  // Validated BEFORE the zero-row early exit, so an out-of-range selector is
  // rejected identically whatever the routing produced.
  TORCH_CHECK(cfg_index >= 0 && cfg_index <= kNumCfgs,
              "cb_moe_persistent_b: cfg must be 0 (auto) or 1..", kNumCfgs,
              ", got ", cfg_index);

  // Alignment the instruction selection depends on, and that contiguity does
  // NOT imply: `cp.async ..., 16` needs a 16-byte-aligned global source, the
  // epilogue stores two BF16 as one `uint32_t`, and the decode gathers eight
  // BF16 as one `uint2`. Fresh allocations satisfy all three; an odd-offset
  // view through this public binding would not.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "cb_moe_persistent_b: `a` must be 16-byte aligned (the mainloop "
              "stages it with 16-byte cp.async); pass a fresh contiguous "
              "tensor rather than an offset view");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0,
              "cb_moe_persistent_b: `out` must be 4-byte aligned (the epilogue "
              "stores two BF16 columns per instruction)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(lut.data_ptr()) % 8 == 0,
              "cb_moe_persistent_b: the flat codebook must be 8-byte aligned "
              "(each decode gathers four BF16 per sub-table in one load)");

  if (P == 0) {
    return;
  }

  const CbFp4V2Fmt fmt((int)k_bits, (int)type_size);
  const int ts_pad = ts_padded((int)type_size);

  int idx = int(cfg_index) - 1;
  if (cfg_index == 0) {
    // AUTO SELECTION — a pure function of SHAPES (P, N, E), never of a
    // routing VALUE.  P is `a.size(0)` and E is `qw.size(0)`, both known to
    // the host without reading device memory, so the choice is a trace-time
    // constant and the launch stays capturable.  It is deliberately NOT a
    // function of the per-expert counts, which would need a device read.
    //
    // The knob it turns is decode amortization vs wasted MMA.  `P / E` is the
    // mean routed rows per expert; a TM far above it wastes MMA on masked
    // rows, and a TM far below it re-decodes the weight tile once per extra
    // M-tile.  Measured on GB10 (docs/KERNELS.md): TM=64/TN=128 wins below
    // ~64 mean rows, TM=128/TN=64 through ~192, TM=256/TN=64 above.
    const int64_t mean_rows = P / E;
    idx = (mean_rows <= 64) ? 3 : 0;   // TM=64,TN=128 : TM=128,TN=64
    // A tile wider than N decodes columns that do not exist, so step down to
    // the widest compiled tile that fits.  This is purely a WORK-SAVING step:
    // correctness never depends on it, because a wide TN is masked in all
    // three places it could matter (`gn < N` in the packed staging, `n0+n < N`
    // in the decode, `col >= N` in the epilogue).  When N is narrower than
    // every compiled tile — a legal FP4-CB layer only needs N % 8 == 0 — the
    // narrowest tile is kept and the masking carries it, rather than failing a
    // shape the kernel computes correctly.
    if (kCfgs[idx].tn > N) {
      // Rank: prefer a tile that fits N; among those prefer the TM the
      // rows-per-expert rule already chose (dropping it would discard the
      // whole point of the selection at exactly the narrow shapes); then the
      // widest TN.  If nothing fits, take the narrowest TN and let the column
      // masking carry it.
      const int want_tm = kCfgs[idx].tm;
      int next = -1;
      for (int i = 0; i < kNumCfgs; ++i) {
        if (next < 0) {
          next = i;
          continue;
        }
        const bool fits = kCfgs[i].tn <= N;
        const bool best_fits = kCfgs[next].tn <= N;
        if (fits != best_fits) {
          if (fits) next = i;
          continue;
        }
        if (!fits) {
          if (kCfgs[i].tn < kCfgs[next].tn) next = i;
          continue;
        }
        const bool same_tm = kCfgs[i].tm == want_tm;
        const bool best_same_tm = kCfgs[next].tm == want_tm;
        if (same_tm != best_same_tm) {
          if (same_tm) next = i;
        } else if (kCfgs[i].tn > kCfgs[next].tn) {
          next = i;
        }
      }
      idx = next;
    }
  }
  const TileCfg cfg = kCfgs[idx];
  const int64_t smem = cfg_smem_bytes(cfg, (int)type_size, d2r);
  TORCH_CHECK(smem + kSmemReservedPerCta <= kSmemPerSm / 2,
              "cb_moe_persistent_b: config TM=", cfg.tm, " TN=", cfg.tn,
              " needs ", smem, " B of shared memory, which drops the kernel "
              "below two CTAs per SM");
  TORCH_CHECK(smem <= kSm120SmemCapacity,
              "cb_moe_persistent_b: config TM=", cfg.tm, " TN=", cfg.tn,
              " needs ", smem, " B of shared memory, over the sm_120 budget (",
              kSm120SmemCapacity, ")");

  const int n_tiles = int((N + cfg.tn - 1) / cfg.tn);
  const int64_t total_wu = E * (int64_t)n_tiles;
  // Launch geometry depends ONLY on the layer shape (E, N) and the compiled
  // tile — never on a routing value and never on a device read.  That is what
  // makes the call capturable in a CUDA graph without a host sync.
  const unsigned grid =
      (unsigned)std::min<int64_t>(total_wu, 2147483647LL);

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  pb_prepare_device((int)a.device().index());
  auto stream = at::cuda::getCurrentCUDAStream();

  const uint16_t* ap = reinterpret_cast<const uint16_t*>(a.data_ptr());
  const uint8_t* qp = qw.data_ptr<uint8_t>();
  const uint16_t* lp = reinterpret_cast<const uint16_t*>(lut.data_ptr());
  const float* cp = compose.data_ptr<float>();
  const int32_t* ep = expert_ends.data_ptr<int32_t>();
  uint16_t* yp = reinterpret_cast<uint16_t*>(out.data_ptr());

#define GB_LAUNCH(TM_, TN_, W_, D2R_)                                        \
  launch_cfg<TM_, TN_, W_, D2R_>(                                           \
      ap, qp, lp, cp, ep, yp, P, N, K, fmt, ts_pad, n_tiles, total_wu,       \
      smem, grid, stream)
  if (d2r) {
    switch (idx) {
      case 0: GB_LAUNCH(128, 64, 8, true); break;
      case 1: GB_LAUNCH(64, 64, 4, true); break;
      case 2: GB_LAUNCH(128, 32, 4, true); break;
      case 3: GB_LAUNCH(64, 128, 8, true); break;
      default:
        TORCH_CHECK(false, "cb_moe_persistent_b_d2r: unreachable config index");
    }
  } else {
    switch (idx) {
      case 0: GB_LAUNCH(128, 64, 8, false); break;
      case 1: GB_LAUNCH(64, 64, 4, false); break;
      case 2: GB_LAUNCH(128, 32, 4, false); break;
      case 3: GB_LAUNCH(64, 128, 8, false); break;
      default:
        TORCH_CHECK(false, "cb_moe_persistent_b: unreachable config index");
    }
  }
#undef GB_LAUNCH
}

void cb_moe_persistent_b_prefill(torch::Tensor out, torch::Tensor a,
                                 torch::Tensor qw, torch::Tensor lut,
                                 torch::Tensor compose,
                                 torch::Tensor expert_ends, int64_t k_bits,
                                 int64_t type_size, int64_t cfg) {
  run_persistent_b(out, a, qw, lut, compose, expert_ends, k_bits, type_size,
                   cfg, false);
}

void cb_moe_persistent_b_prefill_d2r(torch::Tensor out, torch::Tensor a,
                                     torch::Tensor qw, torch::Tensor lut,
                                     torch::Tensor compose,
                                     torch::Tensor expert_ends,
                                     int64_t k_bits, int64_t type_size,
                                     int64_t cfg) {
  run_persistent_b(out, a, qw, lut, compose, expert_ends, k_bits, type_size,
                   cfg, true);
}

torch::Tensor run_decode_probe(torch::Tensor qw_flat,
                               torch::Tensor lut,
                               torch::Tensor compose, int64_t row0,
                               int64_t nrows, int64_t K,
                               int64_t k_bits, int64_t type_size,
                               bool d2r_pairs) {
  TORCH_CHECK(qw_flat.is_cuda() && lut.is_cuda() && compose.is_cuda(),
              "cb_moe_persistent_b_decode: operands must be CUDA tensors");
  TORCH_CHECK(qw_flat.scalar_type() == torch::kUInt8 && qw_flat.dim() == 1 &&
                  qw_flat.is_contiguous(),
              "cb_moe_persistent_b_decode: qw_flat must be a contiguous 1-D "
              "uint8 byte plane");
  TORCH_CHECK(lut.scalar_type() == torch::kBFloat16 && lut.is_contiguous(),
              "cb_moe_persistent_b_decode: lut must be contiguous BF16");
  TORCH_CHECK(compose.scalar_type() == torch::kFloat32 &&
                  compose.is_contiguous() && compose.numel() == 256 * 16,
              "cb_moe_persistent_b_decode: compose must be a contiguous "
              "256*16 FP32 table");
  TORCH_CHECK(K % kSuperblock == 0,
              "cb_moe_persistent_b_decode: K must be a multiple of 256");
  TORCH_CHECK(k_bits >= 1 && k_bits <= 24 && type_size == 4 * k_bits + 9,
              "cb_moe_persistent_b_decode: FP4-CB v2 requires k in [1,24] and "
              "type_size == 4*k+9");
  TORCH_CHECK(nrows >= 0 && row0 >= 0,
              "cb_moe_persistent_b_decode: row0/nrows must be non-negative");
  const int64_t row_bytes = (K / kSuperblock) * type_size;
  TORCH_CHECK(qw_flat.numel() >= (row0 + nrows) * row_bytes,
              "cb_moe_persistent_b_decode: byte plane holds ", qw_flat.numel(),
              " bytes, needs ", (row0 + nrows) * row_bytes);

  auto out = torch::empty({nrows, K},
                          torch::TensorOptions()
                              .dtype(torch::kBFloat16)
                              .device(qw_flat.device()));
  if (nrows == 0) {
    return out;
  }
  const CbFp4V2Fmt fmt((int)k_bits, (int)type_size);
  const int ts_pad = ts_padded((int)type_size);
  constexpr int kDecodeWarps = 8;
  const int64_t smem = (int64_t)kDecodeWarps * ts_pad;

  const c10::cuda::OptionalCUDAGuard guard(qw_flat.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (d2r_pairs) {
    cb_moe_persistent_b_d2r_decode_pairs_kernel<kDecodeWarps>
        <<<(unsigned)nrows, kDecodeWarps * 32, (size_t)smem, stream>>>(
            qw_flat.data_ptr<uint8_t>(),
            reinterpret_cast<const uint16_t*>(lut.data_ptr()),
            compose.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(out.data_ptr()), row0, nrows, K, fmt,
            ts_pad);
  } else {
    cb_moe_persistent_b_decode_kernel<kDecodeWarps>
        <<<(unsigned)nrows, kDecodeWarps * 32, (size_t)smem, stream>>>(
            qw_flat.data_ptr<uint8_t>(),
            reinterpret_cast<const uint16_t*>(lut.data_ptr()),
            compose.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(out.data_ptr()), row0, nrows, K, fmt,
            ts_pad);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor cb_moe_persistent_b_decode(torch::Tensor qw_flat,
                                         torch::Tensor lut,
                                         torch::Tensor compose, int64_t row0,
                                         int64_t nrows, int64_t K,
                                         int64_t k_bits, int64_t type_size) {
  return run_decode_probe(qw_flat, lut, compose, row0, nrows, K, k_bits,
                          type_size, false);
}

torch::Tensor cb_moe_persistent_b_d2r_decode_pairs(
    torch::Tensor qw_flat, torch::Tensor lut, torch::Tensor compose,
    int64_t row0, int64_t nrows, int64_t K, int64_t k_bits,
    int64_t type_size) {
  return run_decode_probe(qw_flat, lut, compose, row0, nrows, K, k_bits,
                          type_size, true);
}

// Host-only attestation of what was actually compiled — no launch, no device
// needed.  Per config: [tm, tn, warps, threads, smem_at_k24, capacity].
std::vector<std::vector<int64_t>> cb_moe_persistent_b_configs() {
  std::vector<std::vector<int64_t>> out;
  for (int i = 0; i < kNumCfgs; ++i) {
    const TileCfg c = kCfgs[i];
    out.push_back({int64_t(c.tm), int64_t(c.tn), int64_t(c.warps),
                   int64_t(c.warps) * 32,
                   cfg_smem_bytes(c, 4 * 24 + 9),
                   int64_t(kSm120SmemCapacity)});
  }
  return out;
}

// Candidate attestation.  Per row:
// [tm, tn, warps, threads, smem_at_k24, capacity, wn, wm, matom, natom].
// All four rows have 4*matom*natom == 32 accumulator registers/thread.
std::vector<std::vector<int64_t>> cb_moe_persistent_b_d2r_configs() {
  std::vector<std::vector<int64_t>> out;
  for (int i = 0; i < kNumCfgs; ++i) {
    const TileCfg c = kCfgs[i];
    const int wn = c.warps;
    const int wm = 1;
    const int matom = c.tm / 16;
    const int natom = (c.tn / wn) / 8;
    out.push_back({int64_t(c.tm), int64_t(c.tn), int64_t(c.warps),
                   int64_t(c.warps) * 32,
                   cfg_smem_bytes(c, 4 * 24 + 9, true),
                   int64_t(kSm120SmemCapacity), int64_t(wn), int64_t(wm),
                   int64_t(matom), int64_t(natom)});
  }
  return out;
}

void cb_moe_persistent_b_d2r_prepare() {
  // D2R's largest dynamic-smem request is below the default per-block limit,
  // but reuse the established preparation to attest cc 12.0/12.1 and the
  // device's shared-memory capacity at model load.  No first-forward setup.
  cb_moe_persistent_b_prepare();
}

int64_t cb_moe_persistent_b_tile_k() { return kTK; }

// The K1.3 firewall in code: this translation unit has no dense entry point.
// Every binding takes `expert_ends` (or is a decode probe), so a dense caller
// cannot reach the schedule at all.
bool cb_moe_persistent_b_is_moe_only() { return true; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_moe_persistent_b_prefill", &cb_moe_persistent_b_prefill,
        "Large-M grouped MoE decode-in-mainloop (persistent-B along M): one "
        "launch over the exact routed segments [P,K] x stacked FP4-CB-v2 "
        "experts [E,N,row_bytes] -> [P,N] BF16. A CTA owns one (expert, "
        "N-tile), decodes that weight tile from packed CB bytes into shared "
        "memory, and streams the expert's routed rows through it. No "
        "expanded [E,N,K] HBM transient. FP32 accumulate, one BF16 round.",
        pybind11::arg("out"), pybind11::arg("a"), pybind11::arg("qw"),
        pybind11::arg("lut"), pybind11::arg("compose"),
        pybind11::arg("expert_ends"), pybind11::arg("k_bits"),
        pybind11::arg("type_size"), pybind11::arg("cfg") = 0);
  m.def("cb_moe_persistent_b_prefill_d2r",
        &cb_moe_persistent_b_prefill_d2r,
        "Experimental BF16 direct-to-register sibling of persistent-B. Same "
        "exact routed segments and packed FP4-CB-v2 operands; removes the "
        "decoded B shared-memory tile and fills each mma.sync B fragment "
        "directly with the cooperative pair helper. Still MoE-only and "
        "takes expert_ends. Default serving never calls this symbol.",
        pybind11::arg("out"), pybind11::arg("a"), pybind11::arg("qw"),
        pybind11::arg("lut"), pybind11::arg("compose"),
        pybind11::arg("expert_ends"), pybind11::arg("k_bits"),
        pybind11::arg("type_size"), pybind11::arg("cfg") = 0);
  m.def("cb_moe_persistent_b_decode", &cb_moe_persistent_b_decode,
        "The mainloop's decode stage, standalone: rows [row0, row0+nrows) of "
        "a flattened FP4-CB-v2 byte plane -> [nrows, K] BF16. Bit-identical "
        "to cb_expand_v2 by construction and by test.",
        pybind11::arg("qw_flat"), pybind11::arg("lut"),
        pybind11::arg("compose"), pybind11::arg("row0"), pybind11::arg("nrows"),
        pybind11::arg("K"), pybind11::arg("k_bits"), pybind11::arg("type_size"));
  m.def("cb_moe_persistent_b_d2r_decode_pairs",
        &cb_moe_persistent_b_d2r_decode_pairs,
        "Standalone probe of the exact cooperative pair helper used by D2R: "
        "packed FP4-CB-v2 rows -> dense BF16, for bitwise comparison with "
        "the established full-codeword decoder.",
        pybind11::arg("qw_flat"), pybind11::arg("lut"),
        pybind11::arg("compose"), pybind11::arg("row0"),
        pybind11::arg("nrows"), pybind11::arg("K"),
        pybind11::arg("k_bits"), pybind11::arg("type_size"));
  m.def("cb_moe_persistent_b_prepare", &cb_moe_persistent_b_prepare,
        "opt every compiled configuration in to the 99 KiB dynamic "
        "shared-memory budget on the CURRENT device, and attest the device is "
        "one this schedule supports. Called at model load so no "
        "cudaFuncSetAttribute happens inside a first forward or a graph "
        "capture.");
  m.def("cb_moe_persistent_b_configs", &cb_moe_persistent_b_configs,
        "compiled tile configs: [tm, tn, warps, threads, smem_bytes_at_k24, "
        "sm120 capacity] each (enumerate THIS, never a hardcoded list)");
  m.def("cb_moe_persistent_b_d2r_prepare",
        &cb_moe_persistent_b_d2r_prepare,
        "load-time device attestation for the experimental D2R sibling");
  m.def("cb_moe_persistent_b_d2r_configs",
        &cb_moe_persistent_b_d2r_configs,
        "D2R configs: [tm,tn,warps,threads,smem_k24,capacity,wn,wm,matom,"
        "natom]; same cfg indices as persistent-B");
  m.def("cb_moe_persistent_b_tile_k", &cb_moe_persistent_b_tile_k,
        "mainloop K-stage width in BF16 columns");
  m.def("cb_moe_persistent_b_is_moe_only", &cb_moe_persistent_b_is_moe_only,
        "ROADMAP K1.3 firewall: this module exposes no dense entry point");
}
