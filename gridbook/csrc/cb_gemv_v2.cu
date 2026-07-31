// SPDX-License-Identifier: Apache-2.0
// CB decode GEMV v2 — smem-resident dictionary, M<=16 grouped MoE decode
// (GB10 / sm_121a).
//
// PRIOR ART IN THIS TREE. Staging the codebook in shared memory is NOT a new
// idea here: cutlass_fork/sm120_cb_fused_mma.hpp:140-196 already does it for
// the FUSED PREFILL mainloop (CbLutResidentSubs / CbLutSmemBytes / load_lut /
// the LutStorage empty-base trick), where it removed the k48 L1 cliff and cut
// the decode-ALU term ~9.1x, bit-exact. This file is the SAME insight applied
// to the OTHER half of the model: the M<=16 grouped decode GEMV, which is a
// disjoint kernel, a disjoint regime (bandwidth- not ALU-bound) and a disjoint
// smem budget. Read the two as corroborating each other.
//
// THESIS. The inherited CB decode (cb_gemv.cu) gathers its sub-codebook entries
// from global/L2 per lane per superblock — a serialized, latency-bound
// indirection. At 2.3-2.9 bpw a CB expert streams FEWER bytes than 4.5-bpw
// dense NVFP4; this kernel converts that byte win into latency by removing the
// L2 round-trip:
//
//   1. the WHOLE per-rung sub-codebook (product n_sub=2: k13 1.5 KB, k16 4 KB,
//      k20 16 KB, k24 64 KB — all fit the sm_121a 99 KB opt-in smem budget) is
//      staged to shared memory ONCE per block by all threads (vectorized
//      uint4), then every decode gather is an LDS hit instead of an L2/DRAM
//      round-trip;
//   2. one WARP owns one output row and stages the row's ENTIRE packed byte
//      stream (n_sb * type_size bytes) to smem in one burst of coalesced u64
//      loads (all 32 lanes issuing) — maximal memory-level parallelism on the
//      DRAM stream, vs the inherited kernel's one-superblock-at-a-time stage;
//   3. one block covers RPB consecutive output rows of one routed (pair), so
//      the dict load is amortized over RPB rows (host-tunable).
//
// SMEM BUDGET — the reason this could not live inside cb_gemv.cu unchanged.
// The inherited decode GEMV's rowpack schedule hard-caps its request at 48 KB
// (cb_gemv.cu:1445, `if (rp_smem <= 48 * 1024)`, else it falls through to the
// non-rowpack schedule) and never calls cudaFuncSetAttribute. This kernel opts
// in to the sm_121a 99 KB dynamic-smem budget explicitly, which is what makes
// the k20 (16 KB) and k24 (64 KB) staged-dictionary configurations expressible
// at all.
//
// NUMERICS — NOT bit-exact against the default inherited schedule. The byte
// format, the decode semantics, the wv rounding (v1: w = bf16_rn(f32(cb)*sc);
// v2 contract: raw values with sc applied to the lane partial), the FMA chain
// order, the ascending-superblock per-warp accumulation and the 32-lane tree
// reduce are all IDENTICAL to cb_gemv.cu's `cb_moe_gemv_fp4_v2_rowpack_kernel`
// (PRISMAQUANT_CB_W2_SCHED=rowpack), against which this kernel is bit-exact.
// The inherited DEFAULT schedule sums in a different order, so v2-vs-default is
// a REASSOCIATION-class difference, the same class as the CUDA-vs-Triton gate
// in tests/test_cuda_gemv.py. Measured on a 204-cell synthetic sweep: 9 cells
// differ, worst max_rel 5.88e-03. Do not describe this kernel as bit-exact.
//
// Scope: fp4 grid, product mode n_sub=2, two-tier v2 scale plane
// (type_size = 4k+9), stacked [E, N, row_bytes] MoE experts. Signed/full/fp8
// rungs stay on the inherited kernels.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#define DEVINL __device__ __forceinline__

namespace {

inline bool pq_env_is(const char* name, const char* val) {
  const char* e = std::getenv(name);
  return e != nullptr && std::strcmp(e, val) == 0;
}

DEVINL float bf16_to_f32(uint16_t v) {
  __nv_bfloat16_raw r; r.x = v;
  return __bfloat162float(__nv_bfloat16(r));
}
DEVINL uint16_t f32_to_bf16_rn(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// Ceil-first bit split (== encoder _bit_split(k,2)): sub0 = ceil(k/2) bits in
// the LOW bits, sub1 = floor(k/2). elt base of sub1 = 4 << w0 (sub_dim=4).
struct Split2 {
  int w0, w1;
  uint32_t m0, m1;
  int64_t e1;
  DEVINL explicit Split2(int k) {
    w0 = (k + 1) >> 1; w1 = k >> 1;
    m0 = (1u << w0) - 1u; m1 = (1u << w1) - 1u;
    e1 = (int64_t)4 << w0;
  }
};

// ---------------------------------------------------------------------------
// Grouped MoE fp4-v2 decode GEMV, smem-resident dictionary + whole-row stage.
//   x        [Xrows, K] bf16 (as u16), act-QDQ'd outside (same as inherited)
//   qw       [E, Nout, row_bytes] uint8 packed (row_bytes = n_sb * type_size)
//   cb       [cb_elems] bf16 flat sub-codebooks (sub0 then sub1, 4 vals/entry)
//   compose  [256*16] fp32 two-tier scale table (tiny, L1-hot -> stays global)
//   pair_expert/pair_xrow [P] int32 routed pairs (capture-safe: fixed shapes)
//   y        [P, Nout] bf16
// Block = WARPS warps; block b covers rows [rg*RPB, rg*RPB+RPB) of pair p,
// warp w handles rows rg*RPB + w, +WARPS, ... Each row: stage whole packed row
// to smem (u64 burst), decode superblocks ascending, lane v = codeword v.
// Dynamic smem = [dict cb_bytes][WARPS slots of slot_bytes].
// ---------------------------------------------------------------------------
// DS (dict residency) — measured trade, not a guess:
//   2 FULL  : both sub-tables staged to smem; every gather is an LDS hit.  Costs
//             cb_bytes of the 99 KB opt-in budget, and the opt-in forces the
//             unified-cache carveout toward shared, shrinking L1.  For k24
//             (64 KB) that is 1 block/SM.
//   0 GLOBAL: dict stays in global, gathered with __ldg, and the launcher asks
//             for the MAX-L1 carveout so the table is L1-resident.  Keeps every
//             OTHER v2 win (whole-row burst stage, __ldcs, one gather per 8
//             weights) at full occupancy.
//   1 HALF  : sub0 in smem, sub1 by __ldg — halves both the smem bill and the
//             divergent-gather rate.  The k24 compromise.
// sub0 has 2^ceil(k/2) entries (elements [0, e1)), sub1 the rest.
template <int WARPS, int DS>
__global__ __launch_bounds__(WARPS * 32) void cb_gemv_v2_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ qw,
    const uint16_t* __restrict__ cb,
    const float* __restrict__ compose,
    const int32_t* __restrict__ pair_expert,
    const int32_t* __restrict__ pair_xrow,
    uint16_t* __restrict__ y,
    const int64_t P, const int64_t Nout, const int64_t K,
    const int k_bits, const int type_size,
    const int rpb, const int slot_bytes, const int cb_elems,
    const int64_t last_row_byte,   // total bytes of qw (OOB guard for u64 tail)
    const int v2) {
  const int n_sb = (int)(K >> 8);
  const int64_t row_bytes = (int64_t)n_sb * type_size;
  const int64_t nblk_per_pair = (Nout + rpb - 1) / rpb;
  const int64_t p = blockIdx.x / nblk_per_pair;
  const int64_t rg = blockIdx.x % nblk_per_pair;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // Staged element count: FULL = whole flat dict, HALF = sub0 only (= e1).
  const int stage_elems =
      (DS == 2) ? cb_elems : ((DS == 1) ? (int)((int64_t)4 << ((k_bits + 1) >> 1))
                                        : 0);
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* dict = reinterpret_cast<uint16_t*>(smem);
  uint8_t* slots = smem + (size_t)stage_elems * 2;

  // --- Phase 0: cooperative dict load (uint4-vectorized; cb_bytes % 16 == 0) -
  if (DS != 0) {
    const uint4* src = reinterpret_cast<const uint4*>(cb);
    uint4* dst = reinterpret_cast<uint4*>(dict);
    const int nv = (stage_elems * 2) >> 4;
    for (int i = threadIdx.x; i < nv; i += WARPS * 32) dst[i] = src[i];
    __syncthreads();
  }

  const int64_t e = (int64_t)pair_expert[p];
  const uint16_t* xr = x + (int64_t)pair_xrow[p] * K;
  const Split2 sp(k_bits);
  const int scale_off = 4 * k_bits;
  uint8_t* slot = slots + (size_t)warp * slot_bytes;

  for (int nl = warp; nl < rpb; nl += WARPS) {
    const int64_t n = (int64_t)rg * rpb + nl;
    if (n >= Nout) break;
    const uint8_t* row = qw + (e * Nout + n) * row_bytes;

    // --- Phase 1: stage the WHOLE packed row to smem (one coalesced burst) --
    // Aligned-down u64 loads; the final word could over-read past the tensor
    // end on the very last row, so the last <=15 bytes go byte-granular.
    const uintptr_t a = reinterpret_cast<uintptr_t>(row);
    const int off8 = (int)(a & 7u);
    const int64_t row_end = (e * Nout + n) * row_bytes + row_bytes; // in qw
    {
      const uint64_t* g8 = reinterpret_cast<const uint64_t*>(a - off8);
      uint64_t* d8 = reinterpret_cast<uint64_t*>(slot);
      const int total = off8 + (int)row_bytes;
      int nv_full = total >> 3;                    // whole u64s inside the row
      // last u64 of the row region: safe unless it crosses the tensor end.
      const bool tail_safe = (row_end + 8 <= last_row_byte);
      const int nv_ld = tail_safe ? ((total + 7) >> 3) : nv_full;
      for (int i = lane; i < nv_ld; i += 32) d8[i] = __ldcs(g8 + i);
      if (!tail_safe) {
        for (int b = (nv_full << 3) - off8 + lane; b < (int)row_bytes; b += 32)
          if (b >= 0) slot[off8 + b] = __ldcs(row + b);
      }
    }
    __syncwarp();

    // --- Phase 2: decode superblocks ascending (inherited rowpack order) ----
    float acc = 0.0f;
    for (int s = 0; s < n_sb; ++s) {
      const int sb_off = off8 + s * type_size;
      const int bitpos = sb_off * 8 + lane * k_bits;
      const int b0 = bitpos >> 3;
      const int rem = ((b0 & 3) << 3) + (bitpos & 7);
      const uint32_t* s32 = reinterpret_cast<const uint32_t*>(slot);
      const int widx = b0 >> 2;
      const uint32_t w0_ = s32[widx];
      const uint32_t w1_ = s32[widx + 1];
      const uint32_t w2_ = (rem + k_bits > 64) ? s32[widx + 2] : 0u;
      const int grp = lane >> 1;
      const uint8_t super_e = slot[sb_off + scale_off];
      const uint8_t sub_byte = slot[sb_off + scale_off + 1 + (grp >> 1)];
      const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
      uint64_t code = lo >> rem;
      if (rem + k_bits > 64) code |= (uint64_t)w2_ << (64 - rem);
      code &= (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);

      const uint32_t code16 = (uint32_t)((sub_byte >> ((grp & 1) * 4)) & 0xFu);
      const float sc = __ldg(compose + (int)super_e * 16 + (int)code16);

      // Sub-index gathers -> SMEM (the entire point of this kernel) or, when
      // the dict is too large to stage without starving occupancy, __ldg.
      const uint32_t i0 = (uint32_t)code & sp.m0;
      const uint32_t i1 = (uint32_t)(code >> sp.w0) & sp.m1;
      uint2 q0, q1;
      if (DS != 0) {
        q0 = *reinterpret_cast<const uint2*>(dict + (int64_t)i0 * 4);
      } else {
        q0 = __ldg(reinterpret_cast<const uint2*>(cb + (int64_t)i0 * 4));
      }
      if (DS == 2) {
        q1 = *reinterpret_cast<const uint2*>(dict + sp.e1 + (int64_t)i1 * 4);
      } else {
        q1 = __ldg(reinterpret_cast<const uint2*>(cb + sp.e1 +
                                                  (int64_t)i1 * 4));
      }
      float wv[8];
      if (v2) {
        wv[0] = bf16_to_f32((uint16_t)(q0.x & 0xffffu));
        wv[1] = bf16_to_f32((uint16_t)(q0.x >> 16));
        wv[2] = bf16_to_f32((uint16_t)(q0.y & 0xffffu));
        wv[3] = bf16_to_f32((uint16_t)(q0.y >> 16));
        wv[4] = bf16_to_f32((uint16_t)(q1.x & 0xffffu));
        wv[5] = bf16_to_f32((uint16_t)(q1.x >> 16));
        wv[6] = bf16_to_f32((uint16_t)(q1.y & 0xffffu));
        wv[7] = bf16_to_f32((uint16_t)(q1.y >> 16));
      } else {
        wv[0] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q0.x & 0xffffu)) * sc));
        wv[1] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q0.x >> 16)) * sc));
        wv[2] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q0.y & 0xffffu)) * sc));
        wv[3] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q0.y >> 16)) * sc));
        wv[4] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q1.x & 0xffffu)) * sc));
        wv[5] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q1.x >> 16)) * sc));
        wv[6] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q1.y & 0xffffu)) * sc));
        wv[7] = bf16_to_f32(f32_to_bf16_rn(
            bf16_to_f32((uint16_t)(q1.y >> 16)) * sc));
      }

      // FMA against x: one 16-byte L2 load per lane (identical to inherited).
      const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
      const uint4 xv = __ldg(reinterpret_cast<const uint4*>(xr + xbase));
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
      if (v2) {
        float ppart = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ppart = fmaf(wv[2 * i], bf16_to_f32((uint16_t)(xw[i] & 0xffffu)),
                       ppart);
          ppart = fmaf(wv[2 * i + 1], bf16_to_f32((uint16_t)(xw[i] >> 16)),
                       ppart);
        }
        acc = fmaf(sc, ppart, acc);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc = fmaf(wv[2 * i], bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), acc);
          acc = fmaf(wv[2 * i + 1], bf16_to_f32((uint16_t)(xw[i] >> 16)), acc);
        }
      }
    }

    // 32-lane tree reduce (identical to inherited rowpack), direct write.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[p * Nout + n] = f32_to_bf16_rn(acc);
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// Expand twin (the bit-exactness oracle): same decode, writes the rendered
// bf16 weight w = bf16_rn(f32(cb) * f32(scale)) — must equal the CPU codec
// nvfp4_cb_reconstruct(...).to(bf16) and expand.expand_fp4_v2_to_weight
// bit-for-bit. rows over [row0, row0+nrows) of the FLATTENED (E*Nout) stack.
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_expand_v2_kernel(
    const uint8_t* __restrict__ qw,        // flat [rows_total * row_bytes]
    const uint16_t* __restrict__ cb,
    const float* __restrict__ compose,
    uint16_t* __restrict__ w,              // [nrows, K] bf16 out
    const int64_t row0, const int64_t nrows, const int64_t K,
    const int k_bits, const int type_size, const int cb_elems,
    const int64_t rows_total) {
  const int n_sb = (int)(K >> 8);
  const int64_t row_bytes = (int64_t)n_sb * type_size;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* dict = reinterpret_cast<uint16_t*>(smem);
  {
    const uint4* src = reinterpret_cast<const uint4*>(cb);
    uint4* dst = reinterpret_cast<uint4*>(dict);
    const int nv = (cb_elems * 2) >> 4;
    for (int i = threadIdx.x; i < nv; i += WARPS * 32) dst[i] = src[i];
  }
  __syncthreads();

  const Split2 sp(k_bits);
  const int scale_off = 4 * k_bits;
  const int64_t r = row0 + blockIdx.x;
  if (r >= row0 + nrows) return;
  const uint8_t* row = qw + r * row_bytes;

  for (int s = warp; s < n_sb; s += WARPS) {
    // byte-granular local window (correctness path; expand is not perf-bound)
    const uint8_t* bsrc = row + (int64_t)s * type_size;
    const int bitpos = lane * k_bits;
    const int b0 = bitpos >> 3;
    const int sh = bitpos & 7;
    uint64_t v = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t gb = (int64_t)s * type_size + b0 + i;
      const uint64_t byte = (gb < row_bytes) ? (uint64_t)bsrc[b0 + i] : 0ull;
      v |= byte << (8 * i);
    }
    uint64_t code = (v >> sh) & ((k_bits >= 64) ? ~0ull
                                                : ((1ull << k_bits) - 1ull));
    const int grp = lane >> 1;
    const uint8_t super_e = bsrc[scale_off];
    const uint8_t sub_byte = bsrc[scale_off + 1 + (grp >> 1)];
    const uint32_t code16 = (uint32_t)((sub_byte >> ((grp & 1) * 4)) & 0xFu);
    const float sc = __ldg(compose + (int)super_e * 16 + (int)code16);

    const uint32_t i0 = (uint32_t)code & sp.m0;
    const uint32_t i1 = (uint32_t)(code >> sp.w0) & sp.m1;
    uint16_t out[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = f32_to_bf16_rn(bf16_to_f32(dict[(int64_t)i0 * 4 + j]) * sc);
      out[4 + j] = f32_to_bf16_rn(bf16_to_f32(dict[sp.e1 + (int64_t)i1 * 4 + j])
                                  * sc);
    }
    uint16_t* dst = w + (r - row0) * K + ((int64_t)s << 8) + (lane << 3);
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[j] = out[j];
  }
  (void)rows_total;
}

// ---------------------------------------------------------------------------
// Peak-bandwidth probes (measure the roofline denominator on THIS box):
//   bw_read : grid-stride uint4 read-and-reduce (GEMV-like: read-dominated)
//   bw_triad: c[i] = a[i] + s*b[i] uint4/f32 (classic streaming triad)
// ---------------------------------------------------------------------------
__global__ void bw_read_kernel(const uint4* __restrict__ a, float* __restrict__ o,
                               int64_t n4) {
  float acc = 0.f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 v = __ldcs(a + i);
    acc += __uint_as_float(v.x ^ v.y ^ v.z ^ v.w);
  }
  if (acc == 12345.678f) o[0] = acc;   // never true: keeps the loads live
}

__global__ void bw_triad_kernel(const float4* __restrict__ a,
                                const float4* __restrict__ b,
                                float4* __restrict__ c, float s, int64_t n4) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (int64_t)gridDim.x * blockDim.x) {
    const float4 av = a[i], bv = b[i];
    c[i] = make_float4(av.x + s * bv.x, av.y + s * bv.y,
                       av.z + s * bv.z, av.w + s * bv.w);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
// OCCUPANCY / RESIDENCY ARITHMETIC — one place, used by the residency policy,
// the rpb policy and the dispatch predicate so they cannot drift apart.
//
// sm_121a opt-in dynamic-smem budget = 99 KiB (101,376 B). A block's bill is
// `stage_bytes + WARPS * slot_bytes`; blocks/SM is that divided into the budget
// and then clamped by the cc-12.x hardware limits (32 resident blocks/SM and
// 48 warps/SM, so an 8-warp block is capped at 6 blocks = 100 % of the warp
// slots). Reproduces the measured ladder exactly: k24 FULL 1 block = 8 of 48
// warps = 16.7 %, k24 HALF 2 blocks = 33.3 %, k24 GLOBAL 6 blocks = 100 %.
static constexpr int CBV2_SMEM_BUDGET = 99 * 1024;
static inline int cbv2_slot_bytes(int64_t row_bytes) {
  return (int)(((8 + row_bytes + 24) + 15) / 16) * 16;
}
static inline int cbv2_blocks_per_sm(size_t stage_bytes, int slot_bytes,
                                     int warps) {
  const size_t bill = stage_bytes + (size_t)warps * slot_bytes;
  int b = (int)(CBV2_SMEM_BUDGET / bill);
  if (b > 32) b = 32;                    // cc 12.x resident-blocks/SM cap
  if (b > 48 / warps) b = 48 / warps;    // cc 12.x 48-warp/SM cap
  return b < 1 ? 1 : b;
}

// cudaFuncSetAttribute is device-specific state. The submitted implementation
// used one process-global bool per residency template, so the first GPU to run
// could make every later GPU skip its own setup; two first launches could also
// race. Prepare all call families once per device under a mutex, then make the
// hot/capture path a single acquire-load. Model loading calls the exported
// prepare binding before capture; the launch-side ensure is a defensive guard
// for direct pybind users and tests.
constexpr int CBV2_MAX_TRACKED_DEVICES = 64;
std::array<std::atomic<bool>, CBV2_MAX_TRACKED_DEVICES> cbv2_prepared{};
std::mutex cbv2_prepare_mutex;

static void cbv2_prepare_device(int device) {
  TORCH_CHECK(device >= 0 && device < CBV2_MAX_TRACKED_DEVICES,
              "CB-GEMV-v2 cannot track CUDA device index ", device,
              " (maximum ", CBV2_MAX_TRACKED_DEVICES - 1, ")");
  if (cbv2_prepared[device].load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(cbv2_prepare_mutex);
  if (cbv2_prepared[device].load(std::memory_order_relaxed)) return;

  const c10::cuda::OptionalCUDAGuard guard(c10::Device(c10::kCUDA, device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(prop.major == 12 && (prop.minor == 0 || prop.minor == 1),
              "CB-GEMV-v2 supports only native compute capability 12.0/12.1, "
              "but cuda:", device, " reports ", prop.major, ".",
              prop.minor);
  TORCH_CHECK(prop.sharedMemPerBlockOptin >= CBV2_SMEM_BUDGET,
              "CB-GEMV-v2 requires ", CBV2_SMEM_BUDGET,
              " B opt-in shared memory, but cuda:", device, " advertises ",
              prop.sharedMemPerBlockOptin, " B");

#define CBV2_SET_MAX_SMEM(FN)                                                \
  C10_CUDA_CHECK(cudaFuncSetAttribute(                                      \
      FN, cudaFuncAttributeMaxDynamicSharedMemorySize, CBV2_SMEM_BUDGET))
  CBV2_SET_MAX_SMEM((cb_gemv_v2_kernel<8, 0>));
  CBV2_SET_MAX_SMEM((cb_gemv_v2_kernel<8, 1>));
  CBV2_SET_MAX_SMEM((cb_gemv_v2_kernel<8, 2>));
  CBV2_SET_MAX_SMEM((cb_expand_v2_kernel<8>));
#undef CBV2_SET_MAX_SMEM

  cbv2_prepared[device].store(true, std::memory_order_release);
}

void cb_gemv_v2_prepare() {
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cbv2_prepare_device(device);
}
// dict_mode: 0 = auto (the measured policy below — smem only while the dict
// still leaves >=3 blocks/SM of the 99 KB budget, else global).
// Forced modes map `ds = dict_mode - 1`, and ds is the STAGE size, so:
//     1 = force GLOBAL dict (ds 0, stage_bytes 0)
//     2 = force HALF-staged dict (ds 1)
//     3 = force FULL smem dict (ds 2)
// (This legend was once written as the exact INVERSE of the code. A wrapper
// author forcing a mode off the wrong legend would have pinned the WORST
// residency for the rung and read it as a kernel regression. Keep this legend
// and `ds = dict_mode - 1` below in lockstep; any bench harness that forces a
// mode must quote the same mapping.)
torch::Tensor cb_gemv_v2(torch::Tensor x, torch::Tensor qw_stack,
                         torch::Tensor cb_flat, torch::Tensor compose,
                         torch::Tensor pair_expert, torch::Tensor pair_xrow,
                         int64_t k_bits, int64_t type_size, int64_t rpb,
                         int64_t dict_mode) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 &&
              x.dim() == 2 && x.is_contiguous(),
              "x must be a contiguous CUDA bf16 [rows,K] tensor");
  TORCH_CHECK(qw_stack.dim() == 3 && qw_stack.scalar_type() == torch::kUInt8 &&
              qw_stack.is_cuda() && qw_stack.is_contiguous(),
              "qw_stack must be a contiguous CUDA uint8 [E,N,row_bytes] tensor");
  TORCH_CHECK(cb_flat.is_cuda() && cb_flat.scalar_type() == torch::kBFloat16 &&
              cb_flat.dim() == 1 && cb_flat.is_contiguous(),
              "cb_flat must be a contiguous CUDA bf16 vector");
  TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kFloat32 &&
              compose.numel() == 256 * 16 && compose.is_contiguous(),
              "compose must be a contiguous CUDA float32 tensor with 4096 elements");
  TORCH_CHECK(pair_expert.is_cuda() &&
              pair_expert.scalar_type() == torch::kInt32 &&
              pair_expert.dim() == 1 && pair_expert.is_contiguous(),
              "pair_expert must be a contiguous CUDA int32 vector");
  TORCH_CHECK(pair_xrow.is_cuda() && pair_xrow.scalar_type() == torch::kInt32 &&
              pair_xrow.dim() == 1 && pair_xrow.is_contiguous(),
              "pair_xrow must be a contiguous CUDA int32 vector");
  TORCH_CHECK(x.device() == qw_stack.device() &&
              x.device() == cb_flat.device() && x.device() == compose.device() &&
              x.device() == pair_expert.device() &&
              x.device() == pair_xrow.device(),
              "all CB-GEMV-v2 tensors must be on the same CUDA device");
  TORCH_CHECK(k_bits > 0 && k_bits <= 24,
              "k_bits must be in [1,24], got ", k_bits);
  TORCH_CHECK(type_size == 4 * k_bits + 9, "fp4-v2 type_size must be 4k+9");
  TORCH_CHECK(dict_mode >= 0 && dict_mode <= 3,
              "dict_mode must be 0(auto), 1(global), 2(half), or 3(full)");
  TORCH_CHECK(rpb >= 0 && rpb <= 1024,
              "rpb must be in [0,1024] (0 selects auto)");
  TORCH_CHECK(qw_stack.size(0) > 0, "qw_stack must contain at least one expert");
  const int64_t Nout = qw_stack.size(1);
  const int64_t row_bytes = qw_stack.size(2);
  TORCH_CHECK(row_bytes > 0 && row_bytes % type_size == 0,
              "row_bytes must be a positive multiple of type_size");
  const int64_t K = (row_bytes / type_size) << 8;
  TORCH_CHECK(K > 0 && K % 256 == 0 &&
              K <= std::numeric_limits<int>::max(),
              "decoded K must be a positive int-sized multiple of 256");
  TORCH_CHECK(row_bytes <= std::numeric_limits<int>::max() - 32,
              "packed row is too wide for the kernel's slot indexing");
  TORCH_CHECK(x.size(1) == K, "x width != decoded row width");
  const int64_t P = pair_expert.numel();
  TORCH_CHECK(pair_xrow.numel() == P,
              "pair_expert and pair_xrow must have the same length");
  const int64_t cb_elems = cb_flat.numel();
  TORCH_CHECK((cb_elems * 2) % 16 == 0, "cb bytes must be 16B-aligned");
  const int expect =
      (int)((4ll << ((k_bits + 1) / 2)) + (4ll << (k_bits / 2)));
  TORCH_CHECK(cb_elems == expect, "cb_flat elems ", cb_elems,
              " != product-mode expectation ", expect, " for k=", k_bits);

  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cbv2_prepare_device(x.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto y = torch::empty({P, Nout}, x.options());
  if (P == 0 || Nout == 0) return y;

  constexpr int WARPS = 8;
  const int slot_bytes = cbv2_slot_bytes(row_bytes);
  const size_t slot_smem = (size_t)WARPS * slot_bytes;
  const size_t dict_bytes = (size_t)cb_elems * 2;
  const size_t half_bytes = (size_t)(4ll << ((k_bits + 1) / 2)) * 2;

  // Residency decision (host-side; shapes + python ints only -> capture-safe).
  // MEASURED policy, fitted to the GB10 ladder, not a guess.  The
  // discriminator is K: the whole-row stage pushes row_bytes of streaming
  // traffic through L1 per row, so a LONG row evicts a large global dict while
  // a SHORT one does not.  Hence:
  //   dict <=  4 KB (k13/k16) -> GLOBAL: the table is L1-trivial at any K, so
  //                              spend the smem on occupancy instead.
  //   dict <= 16 KB (k20)     -> SMEM at K >= 2048 (measured 198 vs 173 GB/s
  //                              at K=4096), GLOBAL below (213 vs 197 GB/s).
  //   dict  > 16 KB (k24)     -> GLOBAL at K <= 1536 (86% roofline); above
  //                              that the rung hits the occupancy wall and the
  //                              DISPATCHER should send it to the inherited
  //                              kernel — HALF is only the least-bad in-kernel
  //                              choice if it does not.
  // Re-derivation of that table from the two structural quantities above (this
  // is WHY it is K-dependent, not just THAT it is): blocks/SM at the STAGED
  // residency is 6/6/6 for k13/k16 at every measured shape, 3-5 for k20, and
  // 1 (FULL) / 2 (HALF) for k24 — and the dict re-stage amplification
  // stage_bytes/(rpb*row_bytes) is <=0.6 for k13/k16, ~0.7-1.9 for k20 and
  // 2.4-19.5 for k24.  Occupancy alone does not decide it and amplification
  // alone does not decide it; the winner flips exactly where BOTH turn.  The
  // HALF choice for k24 long-K stands on the measurement (K=2048: HALF 97.8 vs
  // FULL 87.5 GB/s at their own best rpb; K=4096 they tie at 114.5/114.6).
  int ds;
  if (dict_mode >= 1 && dict_mode <= 3) ds = (int)dict_mode - 1;   // 0/1/2
  else if (dict_bytes <= 4 * 1024) ds = 0;
  else if (dict_bytes <= 16 * 1024) ds = (K >= 2048) ? 2 : 0;
  else ds = (K <= 1536) ? 0 : 1;

  const size_t stage_bytes = (ds == 2) ? dict_bytes
                                       : ((ds == 1) ? half_bytes : 0);
  const size_t smem = stage_bytes + slot_smem;
  TORCH_CHECK(smem <= CBV2_SMEM_BUDGET, "smem ", smem,
              " exceeds the 99KB sm_121a cap");

  // rpb (rows/block) POLICY.  rpb only decides which warp of which block owns
  // output row n; the per-row superblock order, the FMA chain and the 32-lane
  // reduce are untouched, so every rpb produces BIT-IDENTICAL y.  It is a pure
  // scheduling knob, and it is the one that governs DICTIONARY RE-STAGE
  // AMPLIFICATION: phase 0 pulls `stage_bytes` per BLOCK, while the block's
  // useful stream is `rpb * row_bytes`, so the block reads
  //     amp = stage_bytes / (rpb * row_bytes)
  // bytes of dictionary per byte of weight.  That single ratio is what the
  // sweep measured as a "K-dependent winner": k16 at K=4096 amp=0.22
  // (invisible), k20 0.72, k24 2.44 — the k24 block reads 2.4x more dictionary
  // than weight.
  //
  // Amplification only BITES when occupancy is already starved, because with
  // >=3 blocks/SM one block's phase-0 burst overlaps another's compute.  Hence:
  // rpb 16 (the measured default) unless the staged dict leaves <=2 blocks/SM,
  // then the smallest rpb in {32,64} that brings amp <= 1.5.  Fitted to the
  // four k24 cells of the sweep and within 7% of the per-cell oracle there.
  // BE PRECISE ABOUT WHICH NUMBER THIS BUYS: the headline "97.1 -> 114.6 GB/s,
  // +18%" on the k24 long-K cell is the FULL-smem residency, and the ds policy
  // above never selects FULL for k24 at K>=2048 (it selects HALF).  On the path
  // that actually ships, this rule picks rpb=32 and the delta is
  // v2_half_rpb16 107.3 -> v2_half_rpb32 107.9 GB/s = +0.6%, leaving 5.9%
  // against v2_half_rpb64 = 114.5.  So the honest claim is "within 7% of the
  // oracle", not "+18%", and an acceptance test written at 2% WILL FAIL on this
  // cell.  Selecting the LARGEST rather than the smallest candidate in {32,64}
  // satisfying amp<=1.5 would recover that 5.9%; it is not taken here because
  // it has not been re-measured against the second k24 cell, whose own best is
  // at a different rpb.  rpb cannot reassociate anything, so any such change is
  // bit-identical and gated only on a speed re-measure.  The rule yields
  // EXACTLY 16 — i.e. bit-identical AND launch-identical to the pre-policy
  // kernel — for every ladder without a k24 rung, because every k13/k16/k20
  // cell sits at >=3 blocks/SM.
  if (rpb <= 0) {
    rpb = 16;
    if (stage_bytes &&
        cbv2_blocks_per_sm(stage_bytes, slot_bytes, WARPS) <= 2) {
      for (int cand = 32; cand <= 64; cand *= 2) {
        rpb = cand;
        if ((double)stage_bytes <= 1.5 * (double)cand * (double)row_bytes) break;
      }
    }
  }
  if (rpb < WARPS) rpb = WARPS;

  // Carveout: MEASURED, and the naive move is wrong.  Asking for
  // cudaFuncAttributePreferredSharedMemoryCarveout = MaxL1 (so a global 64 KB
  // dict could be L1-resident) costs 1.9-2.9x ACROSS EVERY rung and shape:
  // the driver then hands the block the smallest shared partition that fits
  // the request (16 KB for a 13.7 KB slot bill) = 1 block/SM.  The 128 KB
  // unified cache cannot be both a big L1 and an occupancy-sized shared pool.
  // So: always opt in to the 99 KB budget and let shared win.  (Do not
  // re-propose the MaxL1 carveout without re-running that counter-measurement.)
  //
  // NOTE this is the opt-in the inherited decode GEMV never takes: its rowpack
  // schedule caps itself at 48 KB (cb_gemv.cu:1445) and therefore cannot host
  // the k20/k24 staged-dictionary configurations at all.
  const int64_t nbp = (Nout + rpb - 1) / rpb;
  TORCH_CHECK(nbp > 0 &&
              P <= std::numeric_limits<int>::max() / nbp,
              "CB-GEMV-v2 launch grid exceeds CUDA's x dimension");
  const int v2 = pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT", "v2") ? 1 : 0;
#define LAUNCH_CBV2(DS)                                                      \
  cb_gemv_v2_kernel<WARPS, DS><<<(unsigned)(P * nbp), WARPS * 32, smem,      \
                                 stream>>>(                                  \
      reinterpret_cast<const uint16_t*>(x.data_ptr()),                       \
      qw_stack.data_ptr<uint8_t>(),                                          \
      reinterpret_cast<const uint16_t*>(cb_flat.data_ptr()),                 \
      compose.data_ptr<float>(), pair_expert.data_ptr<int32_t>(),            \
      pair_xrow.data_ptr<int32_t>(),                                         \
      reinterpret_cast<uint16_t*>(y.data_ptr()),                             \
      P, Nout, K, (int)k_bits, (int)type_size, (int)rpb, slot_bytes,         \
      (int)cb_elems, qw_stack.numel(), (int)v2)
  if (ds == 2) { LAUNCH_CBV2(2); }
  else if (ds == 1) { LAUNCH_CBV2(1); }
  else { LAUNCH_CBV2(0); }
#undef LAUNCH_CBV2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// THE DISPATCH PREDICATE, as ONE host function so the plugin does not re-derive
// it from magic constants. Returns true when the per-(layer, stack) dispatcher
// must send this cell to the INHERITED kernel instead of v2.
//
// The k24 long-K wall, stated as arithmetic rather than as a rung name:
//  (1) The product sub-tables are 2 * 2^ceil(k/2) * 4 * 2 B.  At k=24 that is
//      65,536 B.  Two blocks/SM of the 99 KiB budget need <= 50,688 B EACH, so
//      the FULL dict alone exceeds a 2-block bill at EVERY slot size and warp
//      count.  1 block/SM = 8 of 48 warps = 16.7 % occupancy BY CONSTRUCTION.
//  (2) At 1 block/SM a warp alternates load-burst / compute with no other
//      block resident to keep global loads outstanding, so the DRAM duty cycle
//      ceiling is ~50 %.  Measured 48.8 %, i.e. the cell is already at ~98 % of
//      its own structural ceiling — there is no tuning left in it.
//  (3) rpb DOES buy the last of it (dict amplification 2.44 -> 0.61 at rpb 64,
//      97.1 -> 114.6 GB/s) and it is still not enough: the inherited kernel
//      runs the same cell at 142.7 GB/s (61.2 %) precisely because its ~1.7 KiB
//      smem footprint buys maximum occupancy.  A 1.24x gap is not a tuning gap.
//  The only untried lever is double-buffered row staging (cp.async a superblock
//  chunk ahead while decoding the current one), which is the only way to fill
//  DRAM at 1 block/SM.  The smem bill permits it exactly up to K=4096:
//  65,536 + 2*8*1,712 = 92,928 B <= 101,376; at K=8192 it is 119,808 B and does
//  NOT fit at WARPS=8.  Nobody has built it; do not assume it works.
//
// The gate below reproduces the v2-vs-inherited verdict on ALL SIXTEEN measured
// cells of the GB10 sweep (4 rungs x 4 shapes): every k13/k16/k20 cell sits at
// >=3 blocks/SM and v2 wins by 1.13-1.58x; the four k24 cells sit at 1 block/SM
// and v2 wins at K=512/1536 (146 vs 113, 201 vs 170) and LOSES at K=2048/4096
// (98 vs 121, 115 vs 143).  K >= 2048 is MEASURED, not derived — the GLOBAL
// residency's K-dependence is the one term the model never reduced to a
// formula.
bool cb_gemv_v2_prefers_inherited(int64_t k_bits, int64_t type_size,
                                  int64_t in_features) {
  if (k_bits <= 0 || k_bits > 24) return true;         // keep shifts in range
  if (type_size != 4 * k_bits + 9) return true;        // not fp4-v2 product mode
  if (in_features <= 0 || in_features % 256 != 0) return true;
  const int64_t row_bytes = (in_features >> 8) * type_size;
  const int slot_bytes = cbv2_slot_bytes(row_bytes);
  const size_t dict_bytes =
      (size_t)((4ll << ((k_bits + 1) / 2)) + (4ll << (k_bits / 2))) * 2;
  if (dict_bytes + 8ull * slot_bytes > CBV2_SMEM_BUDGET)
    return true;                                       // cannot stage at all
  return cbv2_blocks_per_sm(dict_bytes, slot_bytes, 8) <= 1 &&
         in_features >= 2048;
}

torch::Tensor cb_expand_v2(torch::Tensor qw_flat, torch::Tensor cb_flat,
                           torch::Tensor compose, int64_t row0, int64_t nrows,
                           int64_t K, int64_t k_bits, int64_t type_size) {
  TORCH_CHECK(qw_flat.is_cuda() && qw_flat.scalar_type() == torch::kUInt8 &&
              qw_flat.dim() >= 1 && qw_flat.is_contiguous(),
              "qw_flat must be a contiguous CUDA uint8 tensor");
  TORCH_CHECK(cb_flat.is_cuda() && cb_flat.scalar_type() == torch::kBFloat16 &&
              cb_flat.dim() == 1 && cb_flat.is_contiguous(),
              "cb_flat must be a contiguous CUDA bf16 vector");
  TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kFloat32 &&
              compose.numel() == 256 * 16 && compose.is_contiguous(),
              "compose must be a contiguous CUDA float32 tensor with 4096 elements");
  TORCH_CHECK(qw_flat.device() == cb_flat.device() &&
              qw_flat.device() == compose.device(),
              "qw_flat, cb_flat and compose must share a CUDA device");
  TORCH_CHECK(k_bits > 0 && k_bits <= 24,
              "k_bits must be in [1,24], got ", k_bits);
  TORCH_CHECK(type_size == 4 * k_bits + 9,
              "fp4-v2 type_size must be 4k+9");
  TORCH_CHECK(K > 0 && K % 256 == 0 &&
              K <= std::numeric_limits<int>::max(),
              "K must be a positive int-sized multiple of 256");
  const int64_t row_bytes = (K >> 8) * type_size;
  TORCH_CHECK(row_bytes > 0 && qw_flat.numel() % row_bytes == 0,
              "packed byte count must be divisible by row_bytes");
  const int64_t rows_total = qw_flat.numel() / row_bytes;
  TORCH_CHECK(row0 >= 0 && nrows >= 0 && row0 <= rows_total &&
              nrows <= rows_total - row0,
              "requested row0=", row0, " and nrows=", nrows,
              " are outside a packed tensor with ", rows_total, " rows");
  TORCH_CHECK(nrows <= std::numeric_limits<int>::max(),
              "nrows exceeds CUDA's grid dimension");
  const int64_t cb_elems = cb_flat.numel();
  const int64_t expect =
      (4ll << ((k_bits + 1) / 2)) + (4ll << (k_bits / 2));
  TORCH_CHECK(cb_elems == expect, "cb_flat elems ", cb_elems,
              " != product-mode expectation ", expect, " for k=", k_bits);
  TORCH_CHECK((cb_elems * 2) % 16 == 0,
              "cb bytes must be 16B-aligned");

  const c10::cuda::OptionalCUDAGuard guard(qw_flat.device());
  cbv2_prepare_device(qw_flat.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto w = torch::empty({nrows, K},
                        torch::dtype(torch::kBFloat16).device(qw_flat.device()));
  if (nrows == 0) return w;
  constexpr int WARPS = 8;
  const size_t smem = (size_t)cb_elems * 2;
  // The oracle always stages the FULL dict, so it has no residency fallback:
  // say so at the API boundary rather than failing as an opaque launch error.
  TORCH_CHECK(smem <= CBV2_SMEM_BUDGET, "cb_expand_v2 needs ", smem,
              " B of shared memory for the flat codebook, over the 99KB "
              "sm_121a cap (k=", k_bits, ")");
  cb_expand_v2_kernel<WARPS><<<(unsigned)nrows, WARPS * 32, smem, stream>>>(
      qw_flat.data_ptr<uint8_t>(),
      reinterpret_cast<const uint16_t*>(cb_flat.data_ptr()),
      compose.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(w.data_ptr()),
      row0, nrows, K, (int)k_bits, (int)type_size, (int)cb_elems, rows_total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w;
}

double bw_read(torch::Tensor a, int64_t iters) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kUInt8 &&
              a.is_contiguous() && a.numel() >= 16 && a.numel() % 16 == 0,
              "a must be contiguous CUDA uint8 with a positive 16B-aligned size");
  TORCH_CHECK(iters > 0, "iters must be positive");
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto o = torch::zeros({1}, torch::dtype(torch::kFloat32).device(a.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n4 = a.numel() / 16;
  cudaEvent_t t0, t1;
  C10_CUDA_CHECK(cudaEventCreate(&t0));
  C10_CUDA_CHECK(cudaEventCreate(&t1));
  // warmup
  bw_read_kernel<<<1024, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(a.data_ptr()), o.data_ptr<float>(), n4);
  C10_CUDA_CHECK(cudaEventRecord(t0, stream));
  for (int64_t i = 0; i < iters; ++i)
    bw_read_kernel<<<1024, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(a.data_ptr()), o.data_ptr<float>(), n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaEventRecord(t1, stream));
  C10_CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  C10_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  C10_CUDA_CHECK(cudaEventDestroy(t0));
  C10_CUDA_CHECK(cudaEventDestroy(t1));
  TORCH_CHECK(ms > 0.f, "bandwidth timer returned zero elapsed time");
  return (double)a.numel() * iters / ((double)ms * 1e6);   // GB/s
}

double bw_triad(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                int64_t iters) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda() &&
              a.scalar_type() == torch::kFloat32 &&
              b.scalar_type() == torch::kFloat32 &&
              c.scalar_type() == torch::kFloat32 &&
              a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "a, b and c must be contiguous CUDA float32 tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
              "a, b and c must share a CUDA device");
  TORCH_CHECK(a.numel() >= 4 && a.numel() % 4 == 0 &&
              a.numel() == b.numel() && a.numel() == c.numel(),
              "triad tensors must have the same positive 16B-aligned size");
  TORCH_CHECK(iters > 0, "iters must be positive");
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n4 = a.numel() / 4;
  cudaEvent_t t0, t1;
  C10_CUDA_CHECK(cudaEventCreate(&t0));
  C10_CUDA_CHECK(cudaEventCreate(&t1));
  bw_triad_kernel<<<1024, 256, 0, stream>>>(
      reinterpret_cast<const float4*>(a.data_ptr()),
      reinterpret_cast<const float4*>(b.data_ptr()),
      reinterpret_cast<float4*>(c.data_ptr()), 1.5f, n4);
  C10_CUDA_CHECK(cudaEventRecord(t0, stream));
  for (int64_t i = 0; i < iters; ++i)
    bw_triad_kernel<<<1024, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(a.data_ptr()),
        reinterpret_cast<const float4*>(b.data_ptr()),
        reinterpret_cast<float4*>(c.data_ptr()), 1.5f, n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaEventRecord(t1, stream));
  C10_CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  C10_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  C10_CUDA_CHECK(cudaEventDestroy(t0));
  C10_CUDA_CHECK(cudaEventDestroy(t1));
  TORCH_CHECK(ms > 0.f, "bandwidth timer returned zero elapsed time");
  return (double)a.numel() * 4 * 3 * iters / ((double)ms * 1e6);  // GB/s
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_gemv_v2", &cb_gemv_v2,
        "grouped fp4-v2 CB decode GEMV, smem-resident dict "
        "(x, qw[E,N,rb], cb_flat bf16, compose, pair_expert, pair_xrow, "
        "k, type_size, rpb, dict_mode) -> y[P,N]; rpb=0 = auto; the decode "
        "contract is read from PRISMAQUANT_CB_DECODE_CONTRACT per call");
  m.def("cb_gemv_v2_prefers_inherited", &cb_gemv_v2_prefers_inherited,
        "dispatch predicate: true -> route (k, type_size, K) to the inherited "
        "kernel (the k24 long-K occupancy wall)");
  m.def("cb_gemv_v2_prepare", &cb_gemv_v2_prepare,
        "validate the current GB10 device and configure all v2 kernels' "
        "opt-in dynamic shared-memory attributes once for that device");
  m.def("cb_expand_v2", &cb_expand_v2,
        "bit-exactness oracle: decode rows [row0,row0+nrows) to bf16");
  m.def("bw_read", &bw_read, "peak read bandwidth probe (GB/s)");
  m.def("bw_triad", &bw_triad, "peak triad bandwidth probe (GB/s)");
}
