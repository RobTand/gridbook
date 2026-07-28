// CUDA decode-GEMV for the FP8_CB codebook format (prototype ii — the
// production decode path; docs/nvfp4-cb-plan/serving-kernel.md §1b).
//
// Replaces the Triton `_cb_decode_gemm_kernel` in the decode regime (M<=16).
// The Triton prototype is ~2.4x below the bandwidth bound on GB10 (4.20 tok/s
// vs AURA's 10.26 on the 27B); this kernel is a straight bandwidth-bound
// dequant-GEMV:
//   * one thread block per output row; 8 warps stride the row's 256-weight
//     superblocks; the packed bytes are staged to smem with coalesced uint4
//     loads and each lane extracts one k-bit codeword (32 codewords <-> 32
//     lanes) with aligned 32-bit reads — the packed stream is read from HBM
//     exactly once;
//   * codebook sub-entries are 2 adjacent bf16 values = one 32-bit __ldg
//     gather (L1/L2-resident table);
//   * INV-1: the dense [N,K] weight is never materialized — decode lives in
//     registers, exactly like the Triton kernel it replaces.
//
// Numerics contract (must preserve the served KL of the Triton path):
//   w_j   = bf16_rn(f32(codebook_j) * weight_scale[n])   — identical rounding
//   y_mn  = f32 accumulation of f32(w_j) * f32(xq_mj)    — reassociation-only
//                                                          difference vs tl.dot
//   xq    = fp8 dynamic per-token QDQ of x, bit-exact to
//           codec.fp8_dynamic_act_qdq (fused here as one kernel: f32 amax ->
//           scale = max(amax/448, 1/(448*512)) -> clamp -> e4m3 rn-satfinite
//           -> f32 -> * scale -> bf16_rn).
//
// Scope: fp8 grid, `product` mode, n_sub=4 (sub_dim=2) — the shipped
// FP8_CB_K{36,40,44,48} rungs. Anything else stays on the Triton fallback.
// Compiled by torch.utils.cpp_extension WITHOUT fast-math (division and
// conversion rounding must match torch exactly).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DEVINL __device__ __forceinline__

namespace {

// Host-side env helpers for the (host-only, CUDA-graph-capture-safe) launch-time
// schedule switches. Reads happen in the launcher at capture time, never on a
// hot device path.
inline bool pq_env_is(const char* name, const char* val) {
  const char* e = std::getenv(name);
  return e != nullptr && std::strcmp(e, val) == 0;
}
inline int pq_env_int(const char* name, int fallback) {
  const char* e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') return fallback;
  return std::atoi(e);
}

constexpr int kThreads = 256;            // 8 warps
constexpr int kWarps = kThreads / 32;
constexpr int kSlotBytes = 208;          // >= max type_size (192) + 16 slack
                                         // for the aligned 3-word extraction
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0f / (448.0f * 512.0f);
// torch computes tensor/scalar as a RECIPROCAL MULTIPLY (a * f32(1/448)), not
// a true division — 1 f32 ULP off correctly-rounded for some amax. The scale
// must match codec.fp8_dynamic_act_qdq bit-for-bit, so replicate that chain.
constexpr float kInvFp8Max = 1.0f / 448.0f;

DEVINL float bf16_to_f32(uint16_t v) {
  __nv_bfloat16_raw r;
  r.x = v;
  return __bfloat162float(__nv_bfloat16(r));
}

DEVINL uint16_t f32_to_bf16_rn(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// Single-rounded f32 -> e4m3 (RN-even, saturating region pre-clamped by the
// caller). Verbatim port of c10::detail::fp8e4m3fn_from_fp32_value — the
// hardware/`__nv_cvt_float_to_fp8` route can double-round via f16 and differs
// from torch at half-ULP boundaries, which would break the bit-exact QDQ
// contract (seen live: x=0.7265625 rounding to the adjacent code).
DEVINL uint8_t f32_to_e4m3_c10(float f) {
  constexpr uint32_t fp8_max = 1087u << 20;      // 480.0f, first non-e4m3fn
  constexpr uint32_t denorm_mask = 141u << 23;   // 2^(-121+127) subnormal magic
  uint32_t f_bits = __float_as_uint(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = 0x7f;                               // NaN (unreachable: pre-clamp)
  } else if (f_bits < (121u << 23)) {            // < 2^-6: subnormal result
    f_bits = __float_as_uint(__uint_as_float(f_bits)
                             + __uint_as_float(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;       // RN-even tie break
    f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

// ---------------------------------------------------------------------------
// Fused per-token fp8 dynamic QDQ (bit-exact mirror of
// codec.fp8_dynamic_act_qdq): one block per token row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void fp8_act_qdq_kernel(
    const uint16_t* __restrict__ x,   // [M, K] bf16 (as u16)
    uint16_t* __restrict__ out,       // [M, K] bf16 (as u16)
    int64_t K) {
  const int64_t m = blockIdx.x;
  const uint16_t* row = x + m * K;
  uint16_t* orow = out + m * K;
  __shared__ float red[kWarps];
  __shared__ float s_scale;

  float amax = 0.0f;
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(bf16_to_f32(row[i])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) red[warp] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) a = fmaxf(a, red[w]);
    s_scale = fmaxf(a * kInvFp8Max, kMinScale);
  }
  __syncthreads();
  const float scale = s_scale;

  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) {
    float v = bf16_to_f32(row[i]);
    float q = fminf(fmaxf(v / scale, -kFp8Max), kFp8Max);
    __nv_fp8_storage_t f8 = (__nv_fp8_storage_t)f32_to_e4m3_c10(q);
    float dq = __half2float(__nv_cvt_fp8_to_halfraw(f8, __NV_E4M3));
    orow[i] = f32_to_bf16_rn(dq * scale);
  }
}

// ---------------------------------------------------------------------------
// FP8_CB product-mode decode-GEMV. One block per output row n; warp w handles
// superblocks s = w, w+kWarps, ...; lane v owns codeword v (32 codewords per
// superblock, one per 8-weight vector).
// ---------------------------------------------------------------------------
DEVINL float e4m3_to_f32(uint8_t b) {
  return __half2float(
      __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
}

// Per-sub bit-split descriptor, matching the encoder's _bit_split (ceil-first):
// sub i holds w_i = k/N + (i < k%N) bits at codeword offset off[i] =
// sum_{j<i} w_j, and its table begins elt[i] = SUBDIM * sum_{j<i} 2^{w_j}
// ELEMENTS into the row's flat-codebook slice. Even k reduces to the
// historical uniform split (off[i] = i*(k/N), equal tables), so this is a
// strict generalization — all-constant unrolled scalar math, computed once at
// kernel top and held in registers.
template <int NSUB, int SUBDIM>
struct SubSplit {
  int off[NSUB];
  uint32_t mask[NSUB];
  int64_t elt[NSUB];
  DEVINL explicit SubSplit(int k_bits) {
    const int base = k_bits / NSUB, extra = k_bits % NSUB;
    int o = 0;
    int64_t e = 0;
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
      const int w = base + (i < extra ? 1 : 0);
      off[i] = o;
      mask[i] = (1u << w) - 1u;
      elt[i] = e;
      o += w;
      e += (int64_t)SUBDIM << w;
    }
  }
};

// Decode one superblock's already-read packed words (w0/w1/w2, rem) into wv[8]
// (e4m3 LUT, Triton-bit-exact rounding) and FMA against x for each active m.
// Factored out of cb_gemv_fp8_kernel so the single- and double-buffer schedules
// share IDENTICAL compute — the only difference between the two is when the
// packed load is issued, so this is the clean bit-identity A/B unit. The word
// read is left inline in each loop so the DB path can inject its prefetch
// between the read and this compute (where the overlap lives).
// DECODE CONTRACT v2 (PRISMAQUANT_CB_DECODE_CONTRACT, default v1 until the
// served A/B gate): the per-weight bf16(val*scale) round is REMOVED and the
// scale hoisted out of the hot loop — per-channel scales multiply once per
// output row at write-out (fp8), per-group-16 scales once per lane-partial
// (fp4 two-tier: each lane's 8 weights share one group scale). Removes 1-2
// ops/weight from a chain ncu showed compute-bound (SM 71% vs mem 44%).
// Mathematically sc*Σ(val*x) == Σ((val*sc)*x) in f32; only v1's per-weight
// bf16 round differs (ulp-class). The expand/prefill paths already apply
// scales in the GEMM epilogue, so v2 makes serving MORE self-consistent.
// cb_fma_x_scaled: per-m 8-weight partial, then acc += sc*partial.
template <int MT>
DEVINL void cb_fma_x_scaled(int s, int lane, int M, const uint16_t* x,
                            int64_t K, const float* wv, float sc,
                            float* acc) {
  const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m < M) {
      const uint4 xv = __ldg(
          reinterpret_cast<const uint4*>(x + (int64_t)m * K + xbase));
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
      float p = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        p = fmaf(wv[2 * i], bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), p);
        p = fmaf(wv[2 * i + 1], bf16_to_f32((uint16_t)(xw[i] >> 16)), p);
      }
      acc[m] = fmaf(sc, p, acc[m]);
    }
  }
}

// Shared x-FMA tail for the dense decode helpers: identical for the fp8
// and fp4-v2 (product AND signed) paths — ONE bit-identical epilogue.
template <int MT>
DEVINL void cb_fma_x(int s, int lane, int M, const uint16_t* x, int64_t K,
                        const float* wv, float* acc) {
  const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m < M) {
      const uint4 xv = __ldg(
          reinterpret_cast<const uint4*>(x + (int64_t)m * K + xbase));
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[m] = fmaf(wv[2 * i],
                      bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), acc[m]);
        acc[m] = fmaf(wv[2 * i + 1],
                      bf16_to_f32((uint16_t)(xw[i] >> 16)), acc[m]);
      }
    }
  }
}

template <int MT>
DEVINL void fp8_decode_fma(uint32_t w0_, uint32_t w1_, uint32_t w2_, int rem,
                           int s, int lane, int M, const uint16_t* x, int64_t K,
                           const uint16_t* cb16, int64_t cb_base, int k_bits,
                           const SubSplit<4, 2>& sp,
                           uint64_t code_mask, float sc_row, int v2,
                           float* acc) {
  const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
  uint64_t code = lo >> rem;
  if (rem + k_bits > 64) code |= (uint64_t)w2_ << (64 - rem);
  code &= code_mask;
  float wv[8];
  if (v2) {
    // contract v2: raw e4m3 values; sc_row multiplies ONCE at row write-out
    // (the caller's epilogue), not per weight, and no per-weight round.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
      const int64_t elt = cb_base + sp.elt[i] + (int64_t)idx * 2;
      const uint16_t pair = __ldg(cb16 + (elt >> 1));
      wv[2 * i] = e4m3_to_f32((uint8_t)(pair & 0xffu));
      wv[2 * i + 1] = e4m3_to_f32((uint8_t)(pair >> 8));
    }
    cb_fma_x<MT>(s, lane, M, x, K, wv, acc);
    return;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
    const int64_t elt = cb_base + sp.elt[i] + (int64_t)idx * 2;
    const uint16_t pair = __ldg(cb16 + (elt >> 1));
    wv[2 * i] = bf16_to_f32(
        f32_to_bf16_rn(e4m3_to_f32((uint8_t)(pair & 0xffu)) * sc_row));
    wv[2 * i + 1] = bf16_to_f32(
        f32_to_bf16_rn(e4m3_to_f32((uint8_t)(pair >> 8)) * sc_row));
  }
  cb_fma_x<MT>(s, lane, M, x, K, wv, acc);
}

template <int MT, int WARPS, bool DB>
__global__ __launch_bounds__(WARPS * 32) void cb_gemv_fp8_kernel(
    const uint16_t* __restrict__ x,        // [M, K] bf16 (as u16), QDQ'd
    const uint8_t* __restrict__ qw,        // [N, qw_stride] packed rows
    const uint16_t* __restrict__ cb16,     // E4M3-byte codebook as u16 pairs
    const int32_t* __restrict__ cboff,     // [N] element base into cb_flat
    const float* __restrict__ scale,       // [N] per-output-channel fp32
    uint16_t* __restrict__ y,              // [M, N] bf16 (as u16)
    const int M, const int64_t N, const int64_t K,
    const int64_t qw_stride,
    const int k_bits, const int type_size, const int v2) {
  const int64_t n = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);          // K / 256

  __shared__ __align__(16) uint8_t stage[WARPS][DB ? 2 : 1][kSlotBytes];
  __shared__ float red[WARPS][MT > 0 ? MT : 1];

  const uint8_t* row = qw + n * qw_stride;
  const float sc_row = __ldg(scale + n);
  const int64_t cb_base = (int64_t)__ldg(cboff + n);
  const SubSplit<4, 2> sp(k_bits);
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);

  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

  // Even-k rungs: type_size and the row stride are 8-aligned, so the stage
  // uses 8-byte loads — type_size/8 <= 24 lanes cover one superblock in a
  // single coalesced round. Odd-k rungs (type_size = 4k, k odd) have
  // superblock bases at 4-byte phase; they take the byte-granular stage path
  // below (correctness-first — no served rung depends on odd-k speed yet).
  const bool st8 =
      ((type_size & 7) == 0) && (((int)(qw_stride & 7)) == 0);
  const int stage_vecs = type_size >> 3;
  const int bitpos = lane * k_bits;         // codeword position (fixed per lane)
  const int b0 = bitpos >> 3;
  const int rem = ((b0 & 3) << 3) + (bitpos & 7);
  const int widx = b0 >> 2;

  if constexpr (DB) {
    // Software-pipelined double buffer: while decoding+FMA'ing the current
    // superblock, prefetch superblock s+WARPS into the OTHER smem slot so its
    // evict-first packed load overlaps compute (a win only when this kernel is
    // latency-exposed — few blocks, e.g. the dense M<=16 GEMV; measured vs the
    // single-buffer schedule via PRISMAQUANT_CB_FP8_SCHED). ONE __syncwarp/iter
    // (a slot is not reused for two iterations, so the top-of-loop barrier
    // serves both the incoming RAW and the outgoing WAR). Bit-identical to the
    // single-buffer path — only the load schedule changes, not the sum.
    int s = warp;
    if (st8 && s < n_sb) {
      const uint64_t* g0 =
          reinterpret_cast<const uint64_t*>(row + (int64_t)s * type_size);
      uint64_t* d0 = reinterpret_cast<uint64_t*>(stage[warp][0]);
      if (lane < stage_vecs) d0[lane] = __ldcs(g0 + lane);
      int buf = 0;
      while (s < n_sb) {
        __syncwarp();
        const uint32_t* s32 =
            reinterpret_cast<const uint32_t*>(stage[warp][buf]);
        const uint32_t w0_ = s32[widx];
        const uint32_t w1_ = s32[widx + 1];
        const uint32_t w2_ = s32[widx + 2];
        const int s_next = s + WARPS;
        if (s_next < n_sb) {
          const uint64_t* gN = reinterpret_cast<const uint64_t*>(
              row + (int64_t)s_next * type_size);
          uint64_t* dN = reinterpret_cast<uint64_t*>(stage[warp][buf ^ 1]);
          if (lane < stage_vecs) dN[lane] = __ldcs(gN + lane);
        }
        fp8_decode_fma<MT>(w0_, w1_, w2_, rem, s, lane, M, x, K, cb16, cb_base,
                           k_bits, sp, code_mask, sc_row, v2, acc);
        s = s_next;
        buf ^= 1;
      }
    } else if (s < n_sb) {
      // Odd-k byte-granular stage (no prefetch; correctness-first).
      for (; s < n_sb; s += WARPS) {
        const uint8_t* gsrc = row + (int64_t)s * type_size;
        for (int b = lane; b < type_size; b += 32)
          stage[warp][0][b] = __ldcs(gsrc + b);
        __syncwarp();
        const uint32_t* s32 =
            reinterpret_cast<const uint32_t*>(stage[warp][0]);
        const uint32_t w0_ = s32[widx];
        const uint32_t w1_ = s32[widx + 1];
        const uint32_t w2_ = s32[widx + 2];
        __syncwarp();
        fp8_decode_fma<MT>(w0_, w1_, w2_, rem, s, lane, M, x, K, cb16, cb_base,
                           k_bits, sp, code_mask, sc_row, v2, acc);
      }
    }
  } else {
    // Single-buffer schedule (the original): stage this superblock, read its
    // words, release the slot, then decode+FMA. Two __syncwarp/iteration.
    for (int s = warp; s < n_sb; s += WARPS) {
      const uint8_t* gsrc = row + (int64_t)s * type_size;
      if (st8) {
        const uint64_t* g8 = reinterpret_cast<const uint64_t*>(gsrc);
        uint64_t* gdst = reinterpret_cast<uint64_t*>(stage[warp][0]);
        if (lane < stage_vecs) gdst[lane] = __ldcs(g8 + lane);
      } else {
        for (int b = lane; b < type_size; b += 32)
          stage[warp][0][b] = __ldcs(gsrc + b);
      }
      __syncwarp();
      const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp][0]);
      const uint32_t w0_ = s32[widx];
      const uint32_t w1_ = s32[widx + 1];
      const uint32_t w2_ = s32[widx + 2];
      __syncwarp();
      fp8_decode_fma<MT>(w0_, w1_, w2_, rem, s, lane, M, x, K, cb16, cb_base,
                         k_bits, sp, code_mask, sc_row, v2, acc);
    }
  }

  // --- reduce: 32 lanes -> warp leader -> block --------------------------
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) red[warp][m] = v;
  }
  __syncthreads();
  if (warp == 0 && lane < MT && lane < M) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += red[w][lane];
    if (v2) total *= sc_row;               // contract v2: scale in epilogue
    y[(int64_t)lane * N + n] = f32_to_bf16_rn(total);
  }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
torch::Tensor fp8_act_qdq(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "fp8_act_qdq wants a CUDA bf16 tensor");
  auto x2 = x.contiguous();
  const int64_t K = x2.size(-1);
  const int64_t M = x2.numel() / K;
  auto out = torch::empty_like(x2);
  if (M == 0 || K == 0) return out;
  const c10::cuda::OptionalCUDAGuard guard(x2.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  fp8_act_qdq_kernel<<<(unsigned)M, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x2.data_ptr()),
      reinterpret_cast<uint16_t*>(out.data_ptr()), K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int MT>
void launch_gemv(const torch::Tensor& xq, const torch::Tensor& qw,
                 const torch::Tensor& cb, const torch::Tensor& cboff,
                 const torch::Tensor& scale, torch::Tensor& y,
                 int M, int64_t N, int64_t K, int k_bits,
                 int type_size, cudaStream_t stream) {
  // Warp count: superblocks per row are warp-strided, so a row count that is
  // a multiple of 4 but not 8 (e.g. K=5120 -> 20 superblocks) leaves a 20%
  // tail at 8 warps; 4 warps divide it exactly. Large rows amortize the tail
  // and prefer 8 warps for block-level parallelism.
  const int n_sb = (int)(K >> 8);
  const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
  // Default: double-buffer (prefetch) schedule — the dense fp8 GEMV is one block
  // per output row (comparatively few blocks) and thus latency-exposed, so
  // prefetching the next superblock overlaps its evict-first load with compute.
  // Both schedules are BIT-IDENTICAL (same partial-sum order); the env switch
  // PRISMAQUANT_CB_FP8_SCHED=legacy selects the original single-buffer path (for
  // an interleaved A/B and serving bisection). Read per call: host-only,
  // CUDA-graph-capture-safe, negligible.
  const bool fp8_db = !pq_env_is("PRISMAQUANT_CB_FP8_SCHED", "legacy");
  const int v2 = pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT", "v2") ? 1 : 0;
#define PQ_LAUNCH(W, DBFLAG)                                               \
  cb_gemv_fp8_kernel<MT, W, DBFLAG><<<(unsigned)N, (W)*32, 0, stream>>>(   \
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),                    \
      qw.data_ptr<uint8_t>(),                                              \
      reinterpret_cast<const uint16_t*>(cb.data_ptr()),                    \
      cboff.data_ptr<int32_t>(), scale.data_ptr<float>(),                  \
      reinterpret_cast<uint16_t*>(y.data_ptr()),                           \
      M, N, K, qw.stride(0), k_bits, type_size, v2)
  if (fp8_db) {
    if (use4) { PQ_LAUNCH(4, true); } else { PQ_LAUNCH(8, true); }
  } else {
    if (use4) { PQ_LAUNCH(4, false); } else { PQ_LAUNCH(8, false); }
  }
#undef PQ_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor cb_gemv_fp8(torch::Tensor x, torch::Tensor qw_padded,
                          torch::Tensor cb_flat, torch::Tensor cb_row_offset,
                          torch::Tensor scale, int64_t N, int64_t K,
                          int64_t k_bits, int64_t n_sub, int64_t type_size,
                          bool qdq_input) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp8 wants bf16 activations");
  TORCH_CHECK(qw_padded.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cb_flat.scalar_type() == torch::kUInt8,
              "cb_gemv_fp8 wants the E4M3-byte (uint8) codebook");
  TORCH_CHECK(cb_row_offset.scalar_type() == torch::kInt32);
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(n_sub == 4, "CUDA GEMV supports the fp8 n_sub=4 rungs only");
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of the 256 superblock");
  TORCH_CHECK(type_size <= 192, "type_size beyond the fp8 rung range (<=K48)");
  TORCH_CHECK(type_size == 4 * k_bits, "fp8 type_size must equal 4*k");
  // Widest sub-table (ceil-first split) must stay within the shipped range.
  TORCH_CHECK((k_bits + (int64_t)n_sub - 1) / n_sub <= 12,
              "sub-table beyond the shipped fp8 rungs");
  TORCH_CHECK(qw_padded.dim() == 2 && qw_padded.size(0) == N);
  TORCH_CHECK(qw_padded.stride(1) == 1, "qw rows must be contiguous");
  TORCH_CHECK(cb_row_offset.numel() == N, "cb_row_offset must cover every row");

  auto sizes = x.sizes().vec();
  auto x2 = x.reshape({-1, K}).contiguous();
  const int64_t M = x2.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "decode GEMV handles M in [1,16]");

  const c10::cuda::OptionalCUDAGuard guard(x2.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor xq = qdq_input ? fp8_act_qdq(x2) : x2;
  auto y = torch::empty({M, N}, x2.options());

  const int m = (int)M;
  if (M <= 1) {
    launch_gemv<1>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, (int)type_size, stream);
  } else if (M <= 2) {
    launch_gemv<2>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, (int)type_size, stream);
  } else if (M <= 4) {
    launch_gemv<4>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, (int)type_size, stream);
  } else if (M <= 8) {
    launch_gemv<8>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                   (int)k_bits, (int)type_size, stream);
  } else {
    launch_gemv<16>(xq, qw_padded, cb_flat, cb_row_offset, scale, y, m, N, K,
                    (int)k_bits, (int)type_size, stream);
  }

  sizes.back() = N;
  return y.reshape(sizes);
}

// ---------------------------------------------------------------------------
// Dense fp4 two-tier (v2) decode-GEMV. Merges the two halves the shipped
// fp4-v2 paths already prove:
//   * the dense fp8 kernel's addressing — one block per output row n, the
//     multi-M (MT) register accumulator, and the per-output-row codebook base
//     cb_row_offset (a fused qkv/gate_up module concatenates its roles'
//     codebooks and points each row at its role's block, exactly as fp8);
//   * the grouped MoE fp4-v2 kernel's decode — n_sub=2 sub-codebooks of
//     sub_dim=4 gathered from the BF16 flat codebook, the byte-granular smem
//     stage (type_size = 4k+9 is ODD/unaligned), and the in-register two-tier
//     scale compose (1 E8M0 super byte at offset 4k, then 8 bytes holding 16
//     4-bit sub codes; group g in byte g//2, even g = LOW nibble, LSB-first)
//     via the (256,16) compose table: scale_g = compose[super_e*16 + code16_g].
// Weight rounding matches expand_fp4_v2_to_weight / the Triton decode-GEMM
// bit-for-bit: w = bf16(f32(codebook_value) * f32(scale)); the FMA is
// f32-accumulate, so it differs from the Triton path only by summation
// reassociation (its bf16 tl.dot). Unlike the fp8 dense kernel this one does
// NOT fuse the activation QDQ: fp4 group-16 RTN runs OUTSIDE (python
// codec.fp4_group16_act_qdq), so x arrives already QDQ'd — which keeps the CUDA
// path bit-aligned with the Triton fp4 path (QDQ is outside there too).
// Shared decode+FMA for the dense fp4-v2 kernel (compose two-tier scale, gather
// the bf16 sub-codebook, FMA against x) from a superblock's already-read words +
// scale bytes. Factored so the single- and double-buffer schedules share
// IDENTICAL compute — only the load schedule differs (the clean bit-identity
// A/B unit). The word/scale read stays inline so the DB path can inject its
// prefetch between the read and this compute.
// Signed-mode (NVFP4_CB_S*, n_sub==1) codeword decode: 8 LSB sign bits
// (bit j -> coordinate j negative) + a (k-8)-bit magnitude index into ONE
// non-negative half-grid table of 8-dim entries. The 8 bf16 magnitudes are a
// single 16-byte gather (entries are 8-element aligned); the sign is applied
// by flipping the bf16 sign bit (exact), THEN the two-tier scale multiplies
// and rounds — bit-exact to nvfp4_cb_reconstruct's signed branch.
DEVINL void fp4v2_signed_gather(uint64_t code, const uint16_t* cb_bf16,
                                int64_t cb_base, float sc, int v2, float* wv) {
  const uint32_t sign8 = (uint32_t)(code & 0xffu);
  const int64_t elt = cb_base + (int64_t)(code >> 8) * 8;
  const uint4 q = __ldg(reinterpret_cast<const uint4*>(cb_bf16 + elt));
  const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint16_t b = (uint16_t)((qw[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
    b = (uint16_t)(b ^ (((sign8 >> j) & 1u) << 15));   // exact sign flip
    // v2: raw signed magnitude — the group scale multiplies the lane
    // PARTIAL once (cb_fma_x_scaled), not each weight, and no round.
    wv[j] = v2 ? bf16_to_f32(b)
               : bf16_to_f32(f32_to_bf16_rn(bf16_to_f32(b) * sc));
  }
}

template <int MT>
DEVINL void fp4v2_decode_fma(uint32_t w0_, uint32_t w1_, uint32_t w2_, int rem,
                             uint8_t super_e, uint8_t sub_byte, int grp, int s,
                             int lane, int M, const uint16_t* x, int64_t K,
                             const uint16_t* cb_bf16, int64_t cb_base,
                             int k_bits, int n_sub, const SubSplit<2, 4>& sp,
                             uint64_t code_mask,
                             const float* compose, int v2, float* acc) {
  const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
  uint64_t code = lo >> rem;
  if (rem + k_bits > 64) code |= (uint64_t)w2_ << (64 - rem);
  code &= code_mask;
  const uint32_t code16 = (uint32_t)((sub_byte >> ((grp & 1) * 4)) & 0xFu);
  const float sc = __ldg(compose + (int)super_e * 16 + (int)code16);
  float wv[8];
  if (n_sub == 1) {                              // signed mode (S-rungs)
    fp4v2_signed_gather(code, cb_bf16, cb_base, sc, v2, wv);
    if (v2) cb_fma_x_scaled<MT>(s, lane, M, x, K, wv, sc, acc);
    else cb_fma_x<MT>(s, lane, M, x, K, wv, acc);
    return;
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
    const int64_t elt = cb_base + sp.elt[i] + (int64_t)idx * 4;
#pragma unroll
    for (int local = 0; local < 4; ++local) {
      const float val = bf16_to_f32(__ldg(cb_bf16 + elt + local));
      // v2: raw codebook value; the lane's 8 weights share ONE group-16
      // scale, applied once to the partial in cb_fma_x_scaled.
      wv[i * 4 + local] = v2 ? val
                             : bf16_to_f32(f32_to_bf16_rn(val * sc));
    }
  }
  if (v2) cb_fma_x_scaled<MT>(s, lane, M, x, K, wv, sc, acc);
  else cb_fma_x<MT>(s, lane, M, x, K, wv, acc);
}

template <int MT, int WARPS, bool DB>
__global__ __launch_bounds__(WARPS * 32) void cb_gemv_fp4_v2_kernel(
    const uint16_t* __restrict__ x,        // [M, K] bf16 (as u16), QDQ'd
    const uint8_t* __restrict__ qw,        // [N, qw_stride] packed rows
    const uint16_t* __restrict__ cb_bf16,  // BF16 flat codebook as u16
    const int32_t* __restrict__ cboff,     // [N] element base into cb_flat
    const float* __restrict__ compose,     // [256*16] fp32 two-tier compose
    uint16_t* __restrict__ y,              // [M, N] bf16 (as u16)
    const int M, const int64_t N, const int64_t K,
    const int64_t qw_stride,
    const int k_bits, const int n_sub, const int type_size, const int v2) {
  const int64_t n = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);          // K / 256

  __shared__ __align__(16) uint8_t stage[WARPS][DB ? 2 : 1][kSlotBytes];
  __shared__ float red[WARPS][MT > 0 ? MT : 1];

  const uint8_t* row = qw + n * qw_stride;
  const int64_t cb_base = (int64_t)__ldg(cboff + n);
  const SubSplit<2, 4> sp(k_bits);        // unused in signed (n_sub==1) mode
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  const int scale_off = 4 * k_bits;             // base of the 9-byte scale sec.

  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.0f;

  const int bitpos = lane * k_bits;             // codeword position (per lane)
  const int b0 = bitpos >> 3;
  const int rem = ((b0 & 3) << 3) + (bitpos & 7);
  const int widx = b0 >> 2;
  const int grp = lane >> 1;                     // group-16 index = codeword/2
  const int sub_off = scale_off + 1 + (grp >> 1);

  if constexpr (DB) {
    // Software-pipelined double buffer: prefetch superblock s+WARPS into the
    // OTHER slot while decoding+FMA'ing the current one (a win only when this
    // kernel is latency-exposed — one block/row -> few blocks; measured vs the
    // single-buffer schedule via PRISMAQUANT_CB_FP4V2_SCHED). ONE __syncwarp/iter
    // (a slot is not reused for two iterations). Bit-identical — only the load
    // schedule changes.
    int s = warp;
    if (s < n_sb) {
      const uint8_t* g0 = row + (int64_t)s * type_size;
      for (int b = lane; b < type_size; b += 32)
        stage[warp][0][b] = __ldcs(g0 + b);
      int buf = 0;
      while (s < n_sb) {
        __syncwarp();
        const uint32_t* s32 =
            reinterpret_cast<const uint32_t*>(stage[warp][buf]);
        const uint32_t w0_ = s32[widx];
        const uint32_t w1_ = s32[widx + 1];
        const uint32_t w2_ = s32[widx + 2];
        const uint8_t super_e = stage[warp][buf][scale_off];
        const uint8_t sub_byte = stage[warp][buf][sub_off];
        const int s_next = s + WARPS;
        if (s_next < n_sb) {
          const uint8_t* gN = row + (int64_t)s_next * type_size;
          for (int b = lane; b < type_size; b += 32)
            stage[warp][buf ^ 1][b] = __ldcs(gN + b);
        }
        fp4v2_decode_fma<MT>(w0_, w1_, w2_, rem, super_e, sub_byte, grp, s, lane,
                             M, x, K, cb_bf16, cb_base, k_bits, n_sub, sp,
                             code_mask, compose, v2, acc);
        s = s_next;
        buf ^= 1;
      }
    }
  } else {
    // Single-buffer schedule (the original): byte-granular stage (type_size =
    // 4k+9 is odd/unaligned), read words + scale bytes, release, decode+FMA.
    for (int s = warp; s < n_sb; s += WARPS) {
      const uint8_t* gsrc = row + (int64_t)s * type_size;
      for (int b = lane; b < type_size; b += 32)
        stage[warp][0][b] = __ldcs(gsrc + b);
      __syncwarp();
      const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp][0]);
      const uint32_t w0_ = s32[widx];
      const uint32_t w1_ = s32[widx + 1];
      const uint32_t w2_ = s32[widx + 2];
      const uint8_t super_e = stage[warp][0][scale_off];
      const uint8_t sub_byte = stage[warp][0][sub_off];
      __syncwarp();
      fp4v2_decode_fma<MT>(w0_, w1_, w2_, rem, super_e, sub_byte, grp, s, lane,
                           M, x, K, cb_bf16, cb_base, k_bits, n_sub, sp,
                           code_mask, compose, v2, acc);
    }
  }

  // --- reduce: 32 lanes -> warp leader -> block --------------------------
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) red[warp][m] = v;
  }
  __syncthreads();
  if (warp == 0 && lane < MT && lane < M) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += red[w][lane];
    y[(int64_t)lane * N + n] = f32_to_bf16_rn(total);
  }
}

template <int MT>
void launch_gemv_fp4_v2(const torch::Tensor& xq, const torch::Tensor& qw,
                        const torch::Tensor& cb, const torch::Tensor& cboff,
                        const torch::Tensor& compose, torch::Tensor& y,
                        int M, int64_t N, int64_t K, int k_bits, int n_sub,
                        int type_size, cudaStream_t stream) {
  // Same warp-count heuristic as the fp8 dense kernel: a superblock count that
  // divides 4 but not 8 (K=1024 -> 4) leaves a tail at 8 warps; 4 warps divide
  // it exactly. Large rows amortize the tail and prefer 8 warps.
  const int n_sb = (int)(K >> 8);
  const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
  // Default: single-buffer (legacy) — the DENSE fp4-v2 kernel is the same
  // one-block-per-row (latency-exposed) shape as the dense fp8 kernel, so the
  // prefetch double buffer is a candidate; it is measured vs legacy via the env
  // switch and only defaulted on after a served-KL-safe A/B (bit-identical).
  const bool fp4v2_db = pq_env_is("PRISMAQUANT_CB_FP4V2_SCHED", "db");
  const int v2 = pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT", "v2") ? 1 : 0;
#define PQ_LAUNCH_FP4V2(W, DBFLAG)                                          \
  cb_gemv_fp4_v2_kernel<MT, W, DBFLAG><<<(unsigned)N, (W)*32, 0, stream>>>( \
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),                    \
      qw.data_ptr<uint8_t>(),                                              \
      reinterpret_cast<const uint16_t*>(cb.data_ptr()),                    \
      cboff.data_ptr<int32_t>(), compose.data_ptr<float>(),                \
      reinterpret_cast<uint16_t*>(y.data_ptr()),                           \
      M, N, K, qw.stride(0), k_bits, n_sub, type_size, v2)
  if (fp4v2_db) {
    if (use4) { PQ_LAUNCH_FP4V2(4, true); } else { PQ_LAUNCH_FP4V2(8, true); }
  } else {
    if (use4) { PQ_LAUNCH_FP4V2(4, false); } else { PQ_LAUNCH_FP4V2(8, false); }
  }
#undef PQ_LAUNCH_FP4V2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor cb_gemv_fp4_v2(torch::Tensor x, torch::Tensor qw_padded,
                             torch::Tensor cb_flat, torch::Tensor cb_row_offset,
                             torch::Tensor compose, int64_t N, int64_t K,
                             int64_t k_bits, int64_t n_sub, int64_t type_size) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp4_v2 wants bf16 activations (act-QDQ'd outside)");
  TORCH_CHECK(qw_padded.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cb_flat.scalar_type() == torch::kBFloat16,
              "cb_gemv_fp4_v2 wants the BF16 flat codebook");
  TORCH_CHECK(cb_row_offset.scalar_type() == torch::kInt32);
  TORCH_CHECK(compose.scalar_type() == torch::kFloat32 &&
                  compose.numel() == 256 * 16,
              "compose must be the (256*16,) fp32 two-tier table");
  TORCH_CHECK(n_sub == 2 || n_sub == 1,
              "fp4-v2 GEMV: n_sub=2 (product) or n_sub=1 (signed S-rungs)");
  TORCH_CHECK(n_sub == 2 || k_bits > 8,
              "signed mode needs k > 8 (8 sign bits + magnitude index)");
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of the 256 superblock");
  TORCH_CHECK(type_size == 4 * k_bits + 9,
              "fp4-v2 type_size must be 4k+9 (E8M0 super + 8 sub-nibble bytes)");
  TORCH_CHECK(type_size <= kSlotBytes, "type_size exceeds the smem stage slot");
  TORCH_CHECK(qw_padded.dim() == 2 && qw_padded.size(0) == N);
  TORCH_CHECK(qw_padded.stride(1) == 1, "qw rows must be contiguous");
  TORCH_CHECK(cb_row_offset.numel() == N, "cb_row_offset must cover every row");

  auto sizes = x.sizes().vec();
  auto x2 = x.reshape({-1, K}).contiguous();
  const int64_t M = x2.size(0);
  TORCH_CHECK(M >= 1 && M <= 16, "decode GEMV handles M in [1,16]");

  const c10::cuda::OptionalCUDAGuard guard(x2.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  auto y = torch::empty({M, N}, x2.options());

  const int m = (int)M;
  if (M <= 1) {
    launch_gemv_fp4_v2<1>(x2, qw_padded, cb_flat, cb_row_offset, compose, y, m,
                          N, K, (int)k_bits, (int)n_sub, (int)type_size,
                          stream);
  } else if (M <= 2) {
    launch_gemv_fp4_v2<2>(x2, qw_padded, cb_flat, cb_row_offset, compose, y, m,
                          N, K, (int)k_bits, (int)n_sub, (int)type_size,
                          stream);
  } else if (M <= 4) {
    launch_gemv_fp4_v2<4>(x2, qw_padded, cb_flat, cb_row_offset, compose, y, m,
                          N, K, (int)k_bits, (int)n_sub, (int)type_size,
                          stream);
  } else if (M <= 8) {
    launch_gemv_fp4_v2<8>(x2, qw_padded, cb_flat, cb_row_offset, compose, y, m,
                          N, K, (int)k_bits, (int)n_sub, (int)type_size,
                          stream);
  } else {
    launch_gemv_fp4_v2<16>(x2, qw_padded, cb_flat, cb_row_offset, compose, y, m,
                           N, K, (int)k_bits, (int)n_sub, (int)type_size,
                           stream);
  }

  sizes.back() = N;
  return y.reshape(sizes);
}

// ---------------------------------------------------------------------------
// Grouped MoE decode GEMV: one launch covers every routed (token, expert)
// pair of a layer (the per-expert python loop it replaces cost ~10k host
// syncs + launches per token — 3.52 tok/s on the 35B A3B vs BF16's 28.4).
// Same decode inner loop and numerics as the dense kernel; addressing swaps
// the per-row codebook offset (stacks share ONE codebook) for a per-pair
// expert index into the stacked (E, out, row_bytes) tensor and an x-row
// index (phase A: the pair's token row; phase B: the pair itself).
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_moe_gemv_fp8_kernel(
    const uint16_t* __restrict__ x,        // [Xrows, K] bf16 (as u16), QDQ'd
    const uint8_t* __restrict__ qw,        // [E, out, row_bytes] packed
    const uint16_t* __restrict__ cb16,     // E4M3-byte codebook as u16 pairs
    const float* __restrict__ scale,       // [E, out] per-(expert,out) fp32
    const int32_t* __restrict__ pair_expert,  // [P]
    const int32_t* __restrict__ pair_xrow,    // [P] row of x per pair
    uint16_t* __restrict__ y,              // [P, out] bf16 (as u16)
    const int64_t P, const int64_t Nout, const int64_t K,
    const int k_bits, const int type_size, const int v2) {
  const int64_t g = blockIdx.x;
  const int64_t p = g / Nout;
  const int64_t n = g % Nout;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);
  const int64_t row_bytes = (int64_t)n_sb * type_size;

  __shared__ __align__(16) uint8_t stage[WARPS][kSlotBytes];
  __shared__ float red[WARPS];

  const int64_t e = (int64_t)pair_expert[p];
  const uint8_t* row = qw + (e * Nout + n) * row_bytes;
  const uint16_t* xr = x + (int64_t)pair_xrow[p] * K;
  const float sc_row = __ldg(scale + e * Nout + n);
  const SubSplit<4, 2> sp(k_bits);
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  // u64 staging needs each superblock base 8-aligned: type_size%8==0 (even
  // k) AND this row's base 8-aligned (row_bytes*[e*Nout+n]). Odd-k rungs
  // stage byte-granular (correctness-first).
  const bool st8 = ((type_size & 7) == 0) &&
                   ((((int64_t)(e * Nout + n) * row_bytes) & 7) == 0);
  const int stage_vecs = type_size >> 3;

  float acc = 0.0f;
  for (int s = warp; s < n_sb; s += WARPS) {
    const uint8_t* bsrc = row + (int64_t)s * type_size;
    if (st8) {
      const uint64_t* gsrc = reinterpret_cast<const uint64_t*>(bsrc);
      uint64_t* gdst = reinterpret_cast<uint64_t*>(stage[warp]);
      if (lane < stage_vecs) gdst[lane] = __ldcs(gsrc + lane);
    } else {
      for (int b = lane; b < type_size; b += 32)
        stage[warp][b] = __ldcs(bsrc + b);
    }
    __syncwarp();

    const int bitpos = lane * k_bits;
    const int b0 = bitpos >> 3;
    const int rem = ((b0 & 3) << 3) + (bitpos & 7);
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp]);
    const int widx = b0 >> 2;
    const uint32_t w0_ = s32[widx];
    const uint32_t w1_ = s32[widx + 1];
    const uint32_t w2_ = s32[widx + 2];
    __syncwarp();
    const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
    uint64_t code = lo >> rem;
    if (rem + k_bits > 64) {
      code |= (uint64_t)w2_ << (64 - rem);
    }
    code &= code_mask;

    float wv[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
      const int64_t elt = sp.elt[i] + (int64_t)idx * 2;
      const uint16_t pair2 = __ldg(cb16 + (elt >> 1));
      if (v2) {                     // contract v2: raw values, sc in epilogue
        wv[2 * i] = e4m3_to_f32((uint8_t)(pair2 & 0xffu));
        wv[2 * i + 1] = e4m3_to_f32((uint8_t)(pair2 >> 8));
      } else {
        wv[2 * i] = bf16_to_f32(
            f32_to_bf16_rn(e4m3_to_f32((uint8_t)(pair2 & 0xffu)) * sc_row));
        wv[2 * i + 1] = bf16_to_f32(
            f32_to_bf16_rn(e4m3_to_f32((uint8_t)(pair2 >> 8)) * sc_row));
      }
    }

    const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(xr + xbase));
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc = fmaf(wv[2 * i],
                 bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), acc);
      acc = fmaf(wv[2 * i + 1],
                 bf16_to_f32((uint16_t)(xw[i] >> 16)), acc);
    }
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += red[w];
    if (v2) total *= sc_row;               // contract v2: scale in epilogue
    y[p * Nout + n] = f32_to_bf16_rn(total);
  }
}

// ---------------------------------------------------------------------------
// Grouped MoE decode GEMV for the fp4 two-tier (v2) codebook format
// (docs/nvfp4-cb-plan/two-tier-scale-spec.md §4/§5, moe_cb_design.md). Same
// grouped design as cb_moe_gemv_fp8_kernel (one block per routed (pair, out);
// warps stride the 256-weight superblocks; lane v owns codeword v, 32 per
// superblock). Two structural differences from the fp8 path:
//   * decode is n_sub=2 sub-codebooks of sub_dim=4, gathered from the BF16 flat
//     codebook (_cb_flat), not the e4m3-byte LUT — the fp4 grid values live in
//     bf16 and are scaled by the composed group scale;
//   * the per-group-16 scale is a TWO-TIER code composed in-register from the
//     packed 9-byte scale section per superblock (1 E8M0 super byte at offset
//     4k, then 8 bytes holding 16 4-bit sub codes; group g in byte g//2, even
//     g = LOW nibble, LSB-first) via the (256,16) fp32 compose table:
//     scale_g = compose[super_e*16 + code16_g].
// Weight rounding matches expand_fp4_v2_to_weight (== the per-expert loop's
// _decode_expert) bit-for-bit: w = bf16(f32(codebook_value) * f32(scale)); the
// FMA is f32-accumulate exactly like the fp8 kernel, so grouped-vs-loop differs
// only by reassociation (the loop's bf16 F.linear). type_size = 4k+9 is ODD and
// the row/superblock base is not 8-byte aligned, so the superblock is staged to
// smem BYTE-granular (vs the fp8 u64 stage); the aligned u32 extraction then
// reads from smem unchanged. Activation QDQ (fp4 group-16 RTN) stays OUTSIDE
// the kernel (python codec.fp4_group16_act_qdq), bit-identical to the loop.
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_moe_gemv_fp4_v2_kernel(
    const uint16_t* __restrict__ x,        // [Xrows, K] bf16 (as u16), QDQ'd
    const uint8_t* __restrict__ qw,        // [E, out, row_bytes] packed
    const uint16_t* __restrict__ cb_bf16,  // BF16 flat codebook as u16
    const float* __restrict__ compose,     // [256*16] fp32 two-tier compose
    const int32_t* __restrict__ pair_expert,  // [P]
    const int32_t* __restrict__ pair_xrow,    // [P] row of x per pair
    uint16_t* __restrict__ y,              // [P, out] bf16 (as u16)
    const int64_t P, const int64_t Nout, const int64_t K,
    const int k_bits, const int n_sub, const int type_size, const int v2) {
  const int64_t gb = blockIdx.x;
  const int64_t p = gb / Nout;
  const int64_t n = gb % Nout;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);
  const int64_t row_bytes = (int64_t)n_sb * type_size;

  __shared__ __align__(16) uint8_t stage[WARPS][kSlotBytes];
  __shared__ float red[WARPS];

  const int64_t e = (int64_t)pair_expert[p];
  const uint8_t* row = qw + (e * Nout + n) * row_bytes;
  const uint16_t* xr = x + (int64_t)pair_xrow[p] * K;
  const SubSplit<2, 4> sp(k_bits);
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  const int scale_off = 4 * k_bits;             // base of the 9-byte scale sec.

  float acc = 0.0f;
  for (int s = warp; s < n_sb; s += WARPS) {
    // Wide aligned stage. type_size = 4k+9 is ODD, so the superblock base is
    // not vector-aligned — but staging from the 8-aligned base8 = base & ~7
    // with one u64 __ldcs (evict-first, read once) replaces ceil(ts/32)
    // byte loads + byte stores with a single wide load + store; the head
    // offset off8 (0..7) is carried into the extraction so the u32 reads stay
    // aligned. The LAST superblock of a row keeps the byte path: a u64 there
    // could read up to 7 bytes past the row's final byte (past the tensor for
    // the last row), whereas an interior superblock reads into the next
    // in-row superblock, always in bounds.
    const uint8_t* gsrc = row + (int64_t)s * type_size;
    int off8;
    if (s + 1 < n_sb) {
      const uintptr_t a = reinterpret_cast<uintptr_t>(gsrc);
      off8 = (int)(a & 7u);
      const uint64_t* g8 = reinterpret_cast<const uint64_t*>(a - off8);
      uint64_t* d8 = reinterpret_cast<uint64_t*>(stage[warp]);
      const int nv = (off8 + type_size + 7) >> 3;
      if (lane < nv) d8[lane] = __ldcs(g8 + lane);
    } else {
      off8 = 0;
      for (int b = lane; b < type_size; b += 32)
        stage[warp][b] = __ldcs(gsrc + b);
    }
    __syncwarp();

    // --- extract this lane's k-bit codeword (aligned 32-bit smem reads) ----
    // bitpos carries the head-offset shift (off8*8) so the slot's u32 words
    // line up with the aligned-down stage; the extracted bits are identical.
    const int bitpos = off8 * 8 + lane * k_bits;
    const int b0 = bitpos >> 3;
    const int rem = ((b0 & 3) << 3) + (bitpos & 7);
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp]);
    const int widx = b0 >> 2;
    const uint32_t w0_ = s32[widx];
    const uint32_t w1_ = s32[widx + 1];
    // s32[widx+2] only contributes when rem + k spills past 64 bits (k >= ~34);
    // the shipped fp4-v2 rungs are k<=20 (rem+k <= 51), so predicate the third
    // word away — one fewer L1TEX smem load per superblock on the hot path.
    const uint32_t w2_ = (rem + k_bits > 64) ? s32[widx + 2] : 0u;
    // --- two-tier scale bytes for this lane's group (read before release) ---
    const int grp = lane >> 1;                    // group-16 index = codeword/2
    const uint8_t super_e = stage[warp][off8 + scale_off];
    const uint8_t sub_byte = stage[warp][off8 + scale_off + 1 + (grp >> 1)];
    __syncwarp();
    const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
    uint64_t code = lo >> rem;
    if (rem + k_bits > 64) {
      code |= (uint64_t)w2_ << (64 - rem);
    }
    code &= code_mask;

    // --- compose the group-16 scale (bit-exact to expand_fp4_v2_to_weight) --
    const uint32_t code16 = (uint32_t)((sub_byte >> ((grp & 1) * 4)) & 0xFu);
    const float sc = __ldg(compose + (int)super_e * 16 + (int)code16);

    // --- decode 2 sub-indices -> 8 weights (BF16 codebook, composed scale) --
    // The 4 sub_dim values of a sub-index are contiguous and 8-byte aligned
    // (stacks share one codebook so elt is a multiple of 4 bf16 elements), so
    // gather them as ONE aligned 8-byte load instead of 4 scalar bf16 __ldg —
    // value-identical, 4x fewer codebook load instructions. (The packed
    // __floats2bfloat162_rn round was tried and is bit-identical but slower
    // here — the extra pack/unpack outweighs the halved conversions.)
    float wv[8];
    if (n_sub == 1) {                          // signed mode (S-rungs)
      fp4v2_signed_gather(code, cb_bf16, 0, sc, v2, wv);
    } else {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
      const int64_t elt = sp.elt[i] + (int64_t)idx * 4;
      const uint2 quad = __ldg(reinterpret_cast<const uint2*>(cb_bf16 + elt));
      if (v2) {            // contract v2: raw values, sc scales the partial
        wv[i * 4 + 0] = bf16_to_f32((uint16_t)(quad.x & 0xffffu));
        wv[i * 4 + 1] = bf16_to_f32((uint16_t)(quad.x >> 16));
        wv[i * 4 + 2] = bf16_to_f32((uint16_t)(quad.y & 0xffffu));
        wv[i * 4 + 3] = bf16_to_f32((uint16_t)(quad.y >> 16));
      } else {
      // Match the loop bit-for-bit: w = bf16_rn(f32(cb_value) * f32(scale)).
      wv[i * 4 + 0] = bf16_to_f32(
          f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.x & 0xffffu)) * sc));
      wv[i * 4 + 1] = bf16_to_f32(
          f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.x >> 16)) * sc));
      wv[i * 4 + 2] = bf16_to_f32(
          f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.y & 0xffffu)) * sc));
      wv[i * 4 + 3] = bf16_to_f32(
          f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.y >> 16)) * sc));
      }
    }
    }

    // --- FMA against x: one 16-byte load per lane -------------------------
    const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
    const uint4 xv = __ldg(reinterpret_cast<const uint4*>(xr + xbase));
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
    if (v2) {
      float ppart = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ppart = fmaf(wv[2 * i],
                     bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), ppart);
        ppart = fmaf(wv[2 * i + 1],
                     bf16_to_f32((uint16_t)(xw[i] >> 16)), ppart);
      }
      acc = fmaf(sc, ppart, acc);
    } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc = fmaf(wv[2 * i],
                 bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), acc);
      acc = fmaf(wv[2 * i + 1],
                 bf16_to_f32((uint16_t)(xw[i] >> 16)), acc);
    }
    }
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += red[w];
    y[p * Nout + n] = f32_to_bf16_rn(total);
  }
}

// ---------------------------------------------------------------------------
// fp4-v2 grouped MoE decode GEMV — ROWPACK schedule (opt-in; round-3 item A).
//
// One block covers RPB consecutive output rows of ONE routed (pair): the pair's
// activation row x is staged into smem ONCE and reused by ALL RPB rows, whereas
// the default schedule launches a separate block per (pair, row) that each
// re-reads x from L2. Each of the RPB warps owns one output row and strides ALL
// n_sb superblocks of that row's packed weights, so a block runs RPB
// independent decode streams (deep cross-warp latency hiding) over a single
// resident x. The bet vs the round-2 default (2 warps, 3 superblocks/warp, one
// (pair,row) per block): at the LOW rungs (K14/K16) the per-superblock decode is
// light, so the default is latency- and launch-bound at ~37-41% of peak; RPB=8
// cuts the block count 8x, hides the evict-first weight-load latency behind 8
// concurrent rows, and serves every FMA's x from smem (1-cycle) instead of an
// L2 __ldg.
//
// NUMERICS: one warp accumulates its row's n_sb superblocks in ascending order,
// then a 32-lane tree reduce — a DIFFERENT fp32 partial-sum order than the
// default 2-warp (or legacy 8-warp) schedule, so outputs REASSOCIATE. Same
// tolerance contract as the round-2 w2 schedule (<=1 bf16 ULP + norm backstop
// vs the per-expert loop); gated behind PRISMAQUANT_CB_W2_SCHED=rowpack so a
// served-KL run can bisect it.
//
// Dynamic smem = [x row: K bf16, 16-aligned] ++ [RPB weight stage slots]. For
// the Hy3 w2 shape (K=1536, RPB=8): 3072 + 8*208 = 4736 B — well under the 48 KB
// default (the launcher only takes this path when it fits, else falls through).
template <int RPB>
__global__ __launch_bounds__(RPB * 32) void cb_moe_gemv_fp4_v2_rowpack_kernel(
    const uint16_t* __restrict__ x,        // [Xrows, K] bf16 (as u16), QDQ'd
    const uint8_t* __restrict__ qw,        // [E, out, row_bytes] packed
    const uint16_t* __restrict__ cb_bf16,  // BF16 flat codebook as u16
    const float* __restrict__ compose,     // [256*16] fp32 two-tier compose
    const int32_t* __restrict__ pair_expert,  // [P]
    const int32_t* __restrict__ pair_xrow,    // [P] row of x per pair
    uint16_t* __restrict__ y,              // [P, out] bf16 (as u16)
    const int64_t P, const int64_t Nout, const int64_t K,
    const int k_bits, const int n_sub, const int type_size, const int v2) {
  const int64_t nblk_per_pair = (Nout + RPB - 1) / RPB;
  const int64_t gb = blockIdx.x;
  const int64_t p = gb / nblk_per_pair;
  const int64_t rg = gb % nblk_per_pair;          // row group within the pair
  const int warp = threadIdx.x >> 5;              // 0..RPB-1 -> row in the group
  const int lane = threadIdx.x & 31;
  const int64_t n = rg * RPB + warp;              // this warp's output row
  const int n_sb = (int)(K >> 8);
  const int64_t row_bytes = (int64_t)n_sb * type_size;

  // Dynamic smem: [x row (K bf16, 16-aligned)] then [RPB weight stage slots].
  extern __shared__ __align__(16) uint8_t rp_smem[];
  uint16_t* x_smem = reinterpret_cast<uint16_t*>(rp_smem);
  uint8_t* stage_base = rp_smem + (size_t)K * 2;  // K*2 is a 512B multiple -> 16B

  // --- Phase 1: stage this pair's x row into smem ONCE (all threads) --------
  const uint16_t* xr = x + (int64_t)pair_xrow[p] * K;
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) x_smem[i] = xr[i];
  __syncthreads();

  const SubSplit<2, 4> sp(k_bits);
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  const int scale_off = 4 * k_bits;

  // --- Phase 2: this warp computes ONE output row against the shared x -------
  if (n < Nout) {
    const int64_t e = (int64_t)pair_expert[p];
    const uint8_t* row = qw + (e * Nout + n) * row_bytes;
    uint8_t* slot = stage_base + (size_t)warp * kSlotBytes;
    float acc = 0.0f;
    for (int s = 0; s < n_sb; ++s) {
      // Stage superblock s into this warp's slot (identical to the default
      // kernel: u64 head-aligned for interior sb, byte path for the last).
      const uint8_t* gsrc = row + (int64_t)s * type_size;
      int off8;
      if (s + 1 < n_sb) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(gsrc);
        off8 = (int)(a & 7u);
        const uint64_t* g8 = reinterpret_cast<const uint64_t*>(a - off8);
        uint64_t* d8 = reinterpret_cast<uint64_t*>(slot);
        const int nv = (off8 + type_size + 7) >> 3;
        if (lane < nv) d8[lane] = __ldcs(g8 + lane);
      } else {
        off8 = 0;
        for (int b = lane; b < type_size; b += 32) slot[b] = __ldcs(gsrc + b);
      }
      __syncwarp();

      const int bitpos = off8 * 8 + lane * k_bits;
      const int b0 = bitpos >> 3;
      const int rem = ((b0 & 3) << 3) + (bitpos & 7);
      const uint32_t* s32 = reinterpret_cast<const uint32_t*>(slot);
      const int widx = b0 >> 2;
      const uint32_t w0_ = s32[widx];
      const uint32_t w1_ = s32[widx + 1];
      const uint32_t w2_ = (rem + k_bits > 64) ? s32[widx + 2] : 0u;
      const int grp = lane >> 1;
      const uint8_t super_e = slot[off8 + scale_off];
      const uint8_t sub_byte = slot[off8 + scale_off + 1 + (grp >> 1)];
      __syncwarp();
      const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
      uint64_t code = lo >> rem;
      if (rem + k_bits > 64) code |= (uint64_t)w2_ << (64 - rem);
      code &= code_mask;

      const uint32_t code16 = (uint32_t)((sub_byte >> ((grp & 1) * 4)) & 0xFu);
      const float sc = __ldg(compose + (int)super_e * 16 + (int)code16);

      float wv[8];
      if (n_sub == 1) {                        // signed mode (S-rungs)
        fp4v2_signed_gather(code, cb_bf16, 0, sc, v2, wv);
      } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
        const int64_t elt = sp.elt[i] + (int64_t)idx * 4;
        const uint2 quad = __ldg(reinterpret_cast<const uint2*>(cb_bf16 + elt));
        if (v2) {
          wv[i * 4 + 0] = bf16_to_f32((uint16_t)(quad.x & 0xffffu));
          wv[i * 4 + 1] = bf16_to_f32((uint16_t)(quad.x >> 16));
          wv[i * 4 + 2] = bf16_to_f32((uint16_t)(quad.y & 0xffffu));
          wv[i * 4 + 3] = bf16_to_f32((uint16_t)(quad.y >> 16));
        } else {
        wv[i * 4 + 0] = bf16_to_f32(
            f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.x & 0xffffu)) * sc));
        wv[i * 4 + 1] = bf16_to_f32(
            f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.x >> 16)) * sc));
        wv[i * 4 + 2] = bf16_to_f32(
            f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.y & 0xffffu)) * sc));
        wv[i * 4 + 3] = bf16_to_f32(
            f32_to_bf16_rn(bf16_to_f32((uint16_t)(quad.y >> 16)) * sc));
        }
      }
      }

      // FMA against the SHARED smem x (16-byte LDS.128; no gmem/L2 traffic).
      // xbase*2 = (s*512 + lane*16) bytes -> 16-aligned for the uint4 read.
      const int64_t xbase = ((int64_t)s << 8) + (lane << 3);
      const uint4 xv = *reinterpret_cast<const uint4*>(x_smem + xbase);
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
      if (v2) {
        float ppart = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ppart = fmaf(wv[2 * i],
                       bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), ppart);
          ppart = fmaf(wv[2 * i + 1],
                       bf16_to_f32((uint16_t)(xw[i] >> 16)), ppart);
        }
        acc = fmaf(sc, ppart, acc);
      } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc = fmaf(wv[2 * i],
                   bf16_to_f32((uint16_t)(xw[i] & 0xffffu)), acc);
        acc = fmaf(wv[2 * i + 1],
                   bf16_to_f32((uint16_t)(xw[i] >> 16)), acc);
      }
      }
    }
    // One warp owns the whole row -> 32-lane tree reduce, direct write.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[p * Nout + n] = f32_to_bf16_rn(acc);
  }
}

// Deterministic per-token combine: out[t] = sum over its pairs (pre-sorted
// ascending-expert, matching the python loop's index_add_ order) of
// weight[p] * y[p], accumulated in bf16 exactly like the loop's bf16
// index_add_ (per-add rounding, same order).
__global__ void cb_moe_combine_kernel(
    const uint16_t* __restrict__ y,        // [P, H] bf16
    const float* __restrict__ pair_w,      // [P] router weights
    const int32_t* __restrict__ tok_start,  // [T+1] pair range per token
    uint16_t* __restrict__ out,            // [T, H] bf16
    const int64_t H) {
  const int64_t t = blockIdx.y;
  const int64_t h = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (h >= H) return;
  const int32_t p0 = tok_start[t], p1 = tok_start[t + 1];
  __nv_bfloat16 acc = __float2bfloat16_rn(0.0f);
  for (int32_t p = p0; p < p1; ++p) {
    const float v = bf16_to_f32(y[(int64_t)p * H + h]) * pair_w[p];
    // Match the loop: oe * weight rounds to bf16, then bf16 += (index_add_).
    const __nv_bfloat16 vb = __float2bfloat16_rn(v);
    acc = __hadd(acc, vb);
  }
  out[t * H + h] = __bfloat16_as_ushort(acc);
}

torch::Tensor cb_moe_gemv_fp8(torch::Tensor xq, torch::Tensor qw_stack,
                              torch::Tensor cb_flat_fp8, torch::Tensor scale,
                              torch::Tensor pair_expert,
                              torch::Tensor pair_xrow, int64_t k_bits,
                              int64_t n_sub, int64_t type_size) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(qw_stack.dim() == 3 && qw_stack.scalar_type() == torch::kUInt8);
  TORCH_CHECK(qw_stack.is_contiguous(), "stacked qw must be contiguous");
  TORCH_CHECK(cb_flat_fp8.scalar_type() == torch::kUInt8);
  TORCH_CHECK(scale.dim() == 2 && scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(pair_expert.scalar_type() == torch::kInt32);
  TORCH_CHECK(pair_xrow.scalar_type() == torch::kInt32);
  TORCH_CHECK(n_sub == 4);
  TORCH_CHECK((k_bits + n_sub - 1) / n_sub <= 12 && type_size == 4 * k_bits &&
              type_size <= 192);
  const int64_t Nout = qw_stack.size(1);
  const int64_t row_bytes = qw_stack.size(2);
  const int64_t K = (row_bytes / type_size) << 8;
  TORCH_CHECK(xq.size(-1) == K, "x width != decoded row width");
  const int64_t P = pair_expert.numel();
  const c10::cuda::OptionalCUDAGuard guard(xq.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto y = torch::empty({P, Nout}, xq.options());
  if (P == 0) return y;
  const int n_sb = (int)(K >> 8);
  const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
  const int v2 = pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT", "v2") ? 1 : 0;
  const int64_t grid = P * Nout;
  if (use4) {
    cb_moe_gemv_fp8_kernel<4><<<(unsigned)grid, 128, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(xq.data_ptr()),
        qw_stack.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
        scale.data_ptr<float>(), pair_expert.data_ptr<int32_t>(),
        pair_xrow.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(y.data_ptr()),
        P, Nout, K, (int)k_bits, (int)type_size, v2);
  } else {
    cb_moe_gemv_fp8_kernel<8><<<(unsigned)grid, 256, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(xq.data_ptr()),
        qw_stack.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
        scale.data_ptr<float>(), pair_expert.data_ptr<int32_t>(),
        pair_xrow.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(y.data_ptr()),
        P, Nout, K, (int)k_bits, (int)type_size, v2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor cb_moe_gemv_fp4_v2(torch::Tensor xq, torch::Tensor qw_stack,
                                 torch::Tensor cb_flat, torch::Tensor compose,
                                 torch::Tensor pair_expert,
                                 torch::Tensor pair_xrow, int64_t k_bits,
                                 int64_t n_sub, int64_t type_size) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(qw_stack.dim() == 3 && qw_stack.scalar_type() == torch::kUInt8);
  TORCH_CHECK(qw_stack.is_contiguous(), "stacked qw must be contiguous");
  TORCH_CHECK(cb_flat.scalar_type() == torch::kBFloat16,
              "cb_moe_gemv_fp4_v2 wants the BF16 flat codebook");
  TORCH_CHECK(compose.scalar_type() == torch::kFloat32 &&
                  compose.numel() == 256 * 16,
              "compose must be the (256*16,) fp32 two-tier table");
  TORCH_CHECK(pair_expert.scalar_type() == torch::kInt32);
  TORCH_CHECK(pair_xrow.scalar_type() == torch::kInt32);
  TORCH_CHECK(n_sub == 2 || n_sub == 1,
              "fp4-v2 MoE GEMV: n_sub=2 (product) or n_sub=1 (signed)");
  TORCH_CHECK(n_sub == 2 || k_bits > 8, "signed mode needs k > 8");
  TORCH_CHECK(type_size == 4 * k_bits + 9,
              "fp4-v2 type_size must be 4k+9 (E8M0 super + 8 sub-nibble bytes)");
  TORCH_CHECK(type_size <= kSlotBytes, "type_size exceeds the smem stage slot");
  const int64_t Nout = qw_stack.size(1);
  const int64_t row_bytes = qw_stack.size(2);
  TORCH_CHECK(row_bytes % type_size == 0,
              "row_bytes must be a multiple of type_size");
  const int64_t K = (row_bytes / type_size) << 8;
  TORCH_CHECK(xq.size(-1) == K, "x width != decoded row width");
  const int64_t P = pair_expert.numel();
  const c10::cuda::OptionalCUDAGuard guard(xq.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto y = torch::empty({P, Nout}, xq.options());
  if (P == 0) return y;
  const int n_sb = (int)(K >> 8);
  const int64_t grid = P * Nout;

  // Warp schedule.
  //   legacy (PRISMAQUANT_CB_W2_SCHED=legacy): the original heuristic — 4 warps
  //     when n_sb divides 4 but not 8, else 8. For a small n_sb this leaves
  //     idle warps (w2 n_sb=6 -> 8 warps, 2 idle, ONE superblock/warp) and is
  //     numerics-preserving (the serving path before this change).
  //   default: pick the warp count that gives every warp >=2 superblocks when
  //     n_sb allows (all warps active, more per-warp memory-level parallelism to
  //     hide the evict-first load latency this kernel is bound on), capped at 8.
  //     w13 (n_sb=16) -> 8 warps, 2 superblocks/warp -> IDENTICAL block schedule
  //     to legacy's 8 (BIT-IDENTICAL output). w2 (n_sb=6) -> 3 warps, 2
  //     superblocks/warp, all active -> this REASSOCIATES the fp32 partial-sum
  //     order vs legacy's 8 warps, so it is gated by the env switch for serving
  //     bisection. (PRISMAQUANT_CB_W2_WARPS overrides the count, for retuning.)
  // Schedule selector (PRISMAQUANT_CB_W2_SCHED). Read per call -> host-only and
  // CUDA-graph-capture-safe, and lets a bench toggle schedules within one
  // process:
  //   legacy  -> original 8/4-warp heuristic (numerics-preserving baseline)
  //   rowpack -> round-3 item A: RPB rows/block sharing a smem-resident x row
  //   default (unset / anything else) -> round-2 schedule (~3 superblocks/warp)
  const bool w2_legacy = pq_env_is("PRISMAQUANT_CB_W2_SCHED", "legacy");
  const int v2 = pq_env_is("PRISMAQUANT_CB_DECODE_CONTRACT", "v2") ? 1 : 0;
  const bool w2_rowpack = pq_env_is("PRISMAQUANT_CB_W2_SCHED", "rowpack");

  // --- ROWPACK: one block = RPB output rows of one pair, sharing a smem x ----
  if (w2_rowpack) {
    int rpb = pq_env_int("PRISMAQUANT_CB_W2_ROWS", 8);   // rows (= warps) / block
    if (rpb != 4 && rpb != 8 && rpb != 16) rpb = 8;
    const size_t rp_smem = (size_t)K * 2 + (size_t)rpb * kSlotBytes;
    if (rp_smem <= 48 * 1024) {                          // else fall through
      const int64_t nbp = (Nout + rpb - 1) / rpb;
      const unsigned rp_grid = (unsigned)(P * nbp);
#define LAUNCH_FP4V2_ROWPACK(R)                                              \
  cb_moe_gemv_fp4_v2_rowpack_kernel<R><<<rp_grid, (R) * 32, rp_smem, stream>>>(\
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),                      \
      qw_stack.data_ptr<uint8_t>(),                                          \
      reinterpret_cast<const uint16_t*>(cb_flat.data_ptr()),                 \
      compose.data_ptr<float>(), pair_expert.data_ptr<int32_t>(),            \
      pair_xrow.data_ptr<int32_t>(),                                         \
      reinterpret_cast<uint16_t*>(y.data_ptr()),                             \
      P, Nout, K, (int)k_bits, (int)n_sub, (int)type_size, v2)
      switch (rpb) {
        case 4: LAUNCH_FP4V2_ROWPACK(4); break;
        case 16: LAUNCH_FP4V2_ROWPACK(16); break;
        default: LAUNCH_FP4V2_ROWPACK(8); break;
      }
#undef LAUNCH_FP4V2_ROWPACK
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return y;
    }
    // smem too large for this K -> fall through to the default warp schedule.
  }

  int warps;
  if (w2_legacy) {
    const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
    warps = use4 ? 4 : 8;
  } else {
    // Non-w2 shapes keep their legacy warp count (bit-identical): w13 (n_sb=16,
    // a multiple of 8) -> 8 warps, exactly as before. Only a small n_sb (the w2
    // case, where 8 warps idle two and run one superblock/warp) is retuned to
    // ~3 superblocks/warp — the measured Hy3-w2 optimum (n_sb=6 -> 2 warps:
    // 37-47% of peak vs 25-31% at 8; 2 beats 3 beats 1 across all rungs). This
    // reassociates the w2 fp32 partial-sum order, hence the env gate above.
    const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
    warps = use4 ? 4 : 8;
    if (n_sb < 8) {
      warps = (n_sb + 2) / 3;                // ~3 superblocks/warp; n_sb=6 -> 2
      if (warps < 1) warps = 1;
    }
    const int ov = pq_env_int("PRISMAQUANT_CB_W2_WARPS", 0);
    if (ov >= 1 && ov <= 8) warps = ov;      // tuning override (0 = disabled)
  }
  const unsigned tpb = (unsigned)warps * 32;
#define LAUNCH_FP4V2_MOE(W)                                                  \
  cb_moe_gemv_fp4_v2_kernel<W><<<(unsigned)grid, tpb, 0, stream>>>(          \
      reinterpret_cast<const uint16_t*>(xq.data_ptr()),                      \
      qw_stack.data_ptr<uint8_t>(),                                          \
      reinterpret_cast<const uint16_t*>(cb_flat.data_ptr()),                 \
      compose.data_ptr<float>(), pair_expert.data_ptr<int32_t>(),            \
      pair_xrow.data_ptr<int32_t>(),                                         \
      reinterpret_cast<uint16_t*>(y.data_ptr()),                             \
      P, Nout, K, (int)k_bits, (int)n_sub, (int)type_size, v2)
  switch (warps) {
    case 1: LAUNCH_FP4V2_MOE(1); break;
    case 2: LAUNCH_FP4V2_MOE(2); break;
    case 3: LAUNCH_FP4V2_MOE(3); break;
    case 4: LAUNCH_FP4V2_MOE(4); break;
    case 6: LAUNCH_FP4V2_MOE(6); break;
    default: LAUNCH_FP4V2_MOE(8); break;
  }
#undef LAUNCH_FP4V2_MOE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor cb_moe_combine(torch::Tensor y, torch::Tensor pair_w,
                             torch::Tensor tok_start, int64_t T) {
  TORCH_CHECK(y.dim() == 2 && y.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(pair_w.scalar_type() == torch::kFloat32);
  TORCH_CHECK(tok_start.scalar_type() == torch::kInt32 &&
              tok_start.numel() == T + 1);
  const int64_t H = y.size(1);
  const c10::cuda::OptionalCUDAGuard guard(y.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({T, H}, y.options());
  if (T == 0) return out;
  dim3 grid((unsigned)((H + 255) / 256), (unsigned)T);
  cb_moe_combine_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(y.data_ptr()),
      pair_w.data_ptr<float>(), tok_start.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(out.data_ptr()), H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---------------------------------------------------------------------------
// FP8-direct transient expand (prefill): decode the whole packed weight into
// a [N, K] e4m3-byte tile. Same stage/extract/LUT structure as the GEMV with
// the FMA replaced by one coalesced 8-byte store per codeword — the Triton
// byte-gather expander ran at 61-86 GB/s and serialized ~half the prefill;
// this one is stream-bandwidth-bound.
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ __launch_bounds__(WARPS * 32) void cb_expand_fp8_kernel(
    const uint8_t* __restrict__ qw,        // [N, qw_stride] packed rows
    const uint16_t* __restrict__ cb16,     // E4M3-byte codebook as u16 pairs
    const int32_t* __restrict__ cboff,     // [N] element base into cb_flat
    uint8_t* __restrict__ w,               // [N, K] e4m3 bytes out
    const int64_t N, const int64_t K, const int64_t qw_stride,
    const int k_bits, const int type_size) {
  const int64_t n = blockIdx.x;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int n_sb = (int)(K >> 8);

  __shared__ __align__(16) uint8_t stage[WARPS][kSlotBytes];

  const uint8_t* row = qw + n * qw_stride;
  const int64_t cb_base = (int64_t)__ldg(cboff + n);
  const SubSplit<4, 2> sp(k_bits);
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  const bool st8 = ((type_size & 7) == 0) && (((int)(qw_stride & 7)) == 0);
  const int stage_vecs = type_size >> 3;

  for (int s = warp; s < n_sb; s += WARPS) {
    const uint8_t* bsrc = row + (int64_t)s * type_size;
    if (st8) {
      const uint64_t* gsrc = reinterpret_cast<const uint64_t*>(bsrc);
      uint64_t* gdst = reinterpret_cast<uint64_t*>(stage[warp]);
      if (lane < stage_vecs) gdst[lane] = __ldcs(gsrc + lane);
    } else {
      for (int b = lane; b < type_size; b += 32)
        stage[warp][b] = __ldcs(bsrc + b);
    }
    __syncwarp();

    const int bitpos = lane * k_bits;
    const int b0 = bitpos >> 3;
    const int rem = ((b0 & 3) << 3) + (bitpos & 7);
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(stage[warp]);
    const int widx = b0 >> 2;
    const uint32_t w0_ = s32[widx];
    const uint32_t w1_ = s32[widx + 1];
    const uint32_t w2_ = s32[widx + 2];
    __syncwarp();
    const uint64_t lo = ((uint64_t)w1_ << 32) | (uint64_t)w0_;
    uint64_t code = lo >> rem;
    if (rem + k_bits > 64) {
      code |= (uint64_t)w2_ << (64 - rem);
    }
    code &= code_mask;

    uint64_t out8 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t idx = (uint32_t)(code >> sp.off[i]) & sp.mask[i];
      const int64_t elt = cb_base + sp.elt[i] + (int64_t)idx * 2;
      const uint64_t pair = (uint64_t)__ldg(cb16 + (elt >> 1));
      out8 |= pair << (16 * i);
    }
    // One coalesced 8-byte store per codeword: 256 B per warp-superblock.
    *reinterpret_cast<uint64_t*>(
        w + n * K + ((int64_t)s << 8) + (lane << 3)) = out8;
  }
}

// Shared body for both expander entries: identical validation, identical
// launch-config selection, identical kernel. `w_ptr` is the N*K byte
// destination (allocated by the caller). Keeping this in ONE place is
// deliberate: the allocating entry is on the shipping stock path, and any
// drift between it and the out-variant would silently corrupt served weights.
static void cb_expand_fp8_launch(const torch::Tensor& qw_padded,
                                 const torch::Tensor& cb_flat_fp8,
                                 const torch::Tensor& cb_row_offset,
                                 uint8_t* w_ptr, int64_t N, int64_t K,
                                 int64_t k_bits, int64_t n_sub,
                                 int64_t type_size) {
  TORCH_CHECK(qw_padded.is_cuda() && qw_padded.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cb_flat_fp8.scalar_type() == torch::kUInt8,
              "cb_expand_fp8 wants the E4M3-byte (uint8) codebook");
  TORCH_CHECK(cb_row_offset.scalar_type() == torch::kInt32 &&
              cb_row_offset.numel() == N);
  TORCH_CHECK(n_sub == 4);
  TORCH_CHECK(K % 256 == 0 && type_size == 4 * k_bits && type_size <= 192);
  TORCH_CHECK(qw_padded.dim() == 2 && qw_padded.size(0) == N &&
              qw_padded.stride(1) == 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n_sb = (int)(K >> 8);
  const bool use4 = (n_sb % 8 != 0) && (n_sb % 4 == 0) && (n_sb < 48);
  if (use4) {
    cb_expand_fp8_kernel<4><<<(unsigned)N, 128, 0, stream>>>(
        qw_padded.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
        cb_row_offset.data_ptr<int32_t>(), w_ptr,
        N, K, qw_padded.stride(0), (int)k_bits, (int)type_size);
  } else {
    cb_expand_fp8_kernel<8><<<(unsigned)N, 256, 0, stream>>>(
        qw_padded.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
        cb_row_offset.data_ptr<int32_t>(), w_ptr,
        N, K, qw_padded.stride(0), (int)k_bits, (int)type_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor cb_expand_fp8(torch::Tensor qw_padded, torch::Tensor cb_flat_fp8,
                            torch::Tensor cb_row_offset, int64_t N, int64_t K,
                            int64_t k_bits, int64_t n_sub, int64_t type_size) {
  TORCH_CHECK(qw_padded.is_cuda() && qw_padded.scalar_type() == torch::kUInt8);
  const c10::cuda::OptionalCUDAGuard guard(qw_padded.device());
  auto w = torch::empty({N, K}, qw_padded.options());
  cb_expand_fp8_launch(qw_padded, cb_flat_fp8, cb_row_offset,
                       w.data_ptr<uint8_t>(), N, K, k_bits, n_sub, type_size);
  return w.view(torch::kFloat8_e4m3fn);
}

// Out-variant: decode into a caller-owned buffer. Required by the l2_pipeline
// prefill mode — a persisting-L2 access-policy window pins a FIXED address
// range, so the destination must be a stable scratch buffer, not a fresh
// allocation per decode. `out` may be LARGER than N*K (python slices a
// rotating pair of scratch buffers out of one pinned arena); only the first
// N*K bytes are written.
void cb_expand_fp8_into(torch::Tensor out, torch::Tensor qw_padded,
                        torch::Tensor cb_flat_fp8, torch::Tensor cb_row_offset,
                        int64_t N, int64_t K, int64_t k_bits, int64_t n_sub,
                        int64_t type_size) {
  TORCH_CHECK(qw_padded.is_cuda() && qw_padded.scalar_type() == torch::kUInt8);
  TORCH_CHECK(out.is_cuda(), "cb_expand_fp8_into: out must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kUInt8,
              "cb_expand_fp8_into: out must be uint8 (raw E4M3 bytes)");
  TORCH_CHECK(out.is_contiguous(), "cb_expand_fp8_into: out must be contiguous");
  TORCH_CHECK(out.numel() >= N * K,
              "cb_expand_fp8_into: out too small (need >= N*K bytes)");
  TORCH_CHECK(out.device() == qw_padded.device(),
              "cb_expand_fp8_into: out and qw_padded must share a device");
  const c10::cuda::OptionalCUDAGuard guard(qw_padded.device());
  cb_expand_fp8_launch(qw_padded, cb_flat_fp8, cb_row_offset,
                       out.data_ptr<uint8_t>(), N, K, k_bits, n_sub, type_size);
}

// ---------------------------------------------------------------------------
// L2 persisting-window helpers (host-only; torch exposes no access-policy API)
//
// Two independent CUDA calls are needed and BOTH are required:
//   1. cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes) RESERVES the
//      set-aside carve-out of L2 that persisting lines are allowed to occupy.
//      Without it the carve-out is 0 and the window below is advisory only.
//   2. cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow) marks a
//      specific address range as persisting FOR THAT STREAM. The window is a
//      per-stream attribute, so every stream that touches the range (decode
//      and GEMM alike) must carry it.
// Everything degrades to false/no-op: an un-pinnable device must never be
// fatal, python falls through to the stock path.
// ---------------------------------------------------------------------------
static int64_t l2_dev_attr(cudaDeviceAttr attr) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  int v = 0;
  if (cudaDeviceGetAttribute(&v, attr, dev) != cudaSuccess) return 0;
  return (int64_t)v;
}

int64_t l2_persisting_max_bytes() {
#if CUDART_VERSION >= 11000
  return l2_dev_attr(cudaDevAttrMaxPersistingL2CacheSize);
#else
  return 0;
#endif
}

int64_t l2_max_window_bytes() {
#if CUDART_VERSION >= 11000
  return l2_dev_attr(cudaDevAttrMaxAccessPolicyWindowSize);
#else
  return 0;
#endif
}

// The device-wide carve-out reservation, remembered per process.
//
// WHY THIS IS SPLIT OUT. The two calls below are NOT the same kind of thing and
// must not share a lifetime:
//   * cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize) is DEVICE-WIDE and
//     implicitly synchronizing. Re-issuing it per layer per forward drove a
//     live serve's throughput to zero, so it fires only when the reservation
//     must GROW — in practice once per process.
//   * cudaStreamSetAttribute(accessPolicyWindow) is a cheap PER-STREAM
//     attribute. It is safe on the hot path, and it MUST be reset before the
//     stream is handed back: leaving our window attached to vLLM's serving
//     stream points every later kernel on that stream at a foreign address
//     range, which outlives the forward that set it.
static int64_t g_l2_reserved_bytes = 0;

int64_t l2_max_window_bytes();
int64_t l2_persisting_max_bytes();

bool l2_pin_region(torch::Tensor buf, int64_t num_bytes) {
#if CUDART_VERSION >= 11000
  if (!buf.is_cuda() || num_bytes <= 0) return false;
  const c10::cuda::OptionalCUDAGuard guard(buf.device());

  const int64_t win_max = l2_max_window_bytes();
  const int64_t persist_max = l2_persisting_max_bytes();
  if (win_max <= 0 || persist_max <= 0) return false;

  // The window is hardware-capped; asking for more is an error, so clamp.
  int64_t bytes = num_bytes < win_max ? num_bytes : win_max;
  const int64_t reserve = bytes < persist_max ? bytes : persist_max;

  // (1) reserve the L2 set-aside — GROW-ONLY, so the synchronizing call is
  // paid once per process rather than once per layer per forward.
  if (reserve > g_l2_reserved_bytes) {
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           (size_t)reserve) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    g_l2_reserved_bytes = reserve;
  }
  // (2) ...then mark the range persisting on the CURRENT stream.
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = buf.data_ptr();
  attr.accessPolicyWindow.num_bytes = (size_t)bytes;
  attr.accessPolicyWindow.hitRatio = 1.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow,
                             &attr) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
#else
  (void)buf; (void)num_bytes;
  return false;
#endif
}

// Clear ONLY the per-stream window. No device-wide call, so this is cheap
// enough to run at the end of every forward — which is the point: the serving
// stream must never carry our access-policy window outside the forward that
// set it. The carve-out reservation deliberately survives (see above).
void l2_reset_window() {
#if CUDART_VERSION >= 11000
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio = 0.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
  cudaGetLastError();  // best-effort, never fatal
#endif
}

void l2_unpin() {
#if CUDART_VERSION >= 11000
  g_l2_reserved_bytes = 0;
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio = 0.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
  cudaCtxResetPersistingL2Cache();
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
  cudaGetLastError();  // swallow: unpin is best-effort, never fatal
#endif
}

// ---------------------------------------------------------------------------
// Debug probes (test-only): isolate the conversion and the scale reduction.
// ---------------------------------------------------------------------------
__global__ void e4m3_probe_kernel(const float* __restrict__ q,
                                  uint8_t* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = f32_to_e4m3_c10(q[i]);
}

torch::Tensor e4m3_probe(torch::Tensor q) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat32);
  auto qc = q.contiguous();
  auto out = torch::empty(qc.sizes(), qc.options().dtype(torch::kUInt8));
  const int64_t n = qc.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  e4m3_probe_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
      qc.data_ptr<float>(), out.data_ptr<uint8_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ __launch_bounds__(kThreads) void qdq_scale_kernel(
    const uint16_t* __restrict__ x, float* __restrict__ scales, int64_t K) {
  const int64_t m = blockIdx.x;
  const uint16_t* row = x + m * K;
  __shared__ float red[kWarps];
  float amax = 0.0f;
  for (int64_t i = threadIdx.x; i < K; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(bf16_to_f32(row[i])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) red[warp] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) a = fmaxf(a, red[w]);
    scales[m] = fmaxf(a * kInvFp8Max, kMinScale);
  }
}

torch::Tensor qdq_scale_probe(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16);
  auto x2 = x.contiguous();
  const int64_t K = x2.size(-1);
  const int64_t M = x2.numel() / K;
  auto out = torch::empty({M}, x2.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  qdq_scale_kernel<<<(unsigned)M, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x2.data_ptr()),
      out.data_ptr<float>(), K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_act_qdq", &fp8_act_qdq,
        "Fused per-token fp8 dynamic QDQ (bit-exact to codec.fp8_dynamic_act_qdq)");
  m.def("cb_gemv_fp8", &cb_gemv_fp8,
        "FP8_CB product-mode decode GEMV (bandwidth-bound, INV-1)");
  m.def("cb_gemv_fp4_v2", &cb_gemv_fp4_v2,
        "dense fp4 two-tier (v2) decode GEMV (per-row codebook offset + "
        "in-register two-tier scale compose)");
  m.def("cb_expand_fp8", &cb_expand_fp8,
        "FP8-direct transient expand (prefill; bounded per-layer tile)");
  m.def("cb_expand_fp8_into", &cb_expand_fp8_into,
        "FP8-direct expand into a caller-owned (L2-pinnable) scratch buffer");
  m.def("l2_pin_region", &l2_pin_region,
        "pin a buffer range as L2-persisting on the current stream "
        "(false if unavailable)");
  m.def("l2_reset_window", &l2_reset_window,
        "clear ONLY the per-stream access-policy window (cheap; no device-wide call)");
  m.def("l2_unpin", &l2_unpin,
        "reset the access-policy window + persisting L2 carve-out");
  m.def("l2_persisting_max_bytes", &l2_persisting_max_bytes,
        "cudaDevAttrMaxPersistingL2CacheSize");
  m.def("l2_max_window_bytes", &l2_max_window_bytes,
        "cudaDevAttrMaxAccessPolicyWindowSize");
  m.def("cb_moe_gemv_fp8", &cb_moe_gemv_fp8,
        "grouped MoE decode GEMV over routed (token, expert) pairs");
  m.def("cb_moe_gemv_fp4_v2", &cb_moe_gemv_fp4_v2,
        "grouped MoE decode GEMV for the fp4 two-tier (v2) codebook format");
  m.def("cb_moe_combine", &cb_moe_combine,
        "deterministic per-token weighted combine (loop-order bf16 adds)");
  m.def("e4m3_probe", &e4m3_probe, "debug: f32 -> e4m3 codes via the c10 port");
  m.def("qdq_scale_probe", &qdq_scale_probe,
        "debug: per-token scale exactly as the QDQ kernel computes it");
}
