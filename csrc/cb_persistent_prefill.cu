// Persistent-N large-M FP8_CB prefill — INV-1 reference (round-3 Task 9).
// See docs/nvfp4-cb-plan/persistent-n-prefill.md.
//
// Each persistent CTA owns TILE_N output columns and grid-strides over N. Phase
// 1 decodes packed B[n0:n0+TILE_N, :K] ONCE into a smem e4m3 tile (bit-exact to
// cb_expand_fp8: same codeword-window extraction + LUT gather, UNSCALED — the
// per-output-channel scale is applied OUTSIDE, matching the transient-expand /
// fork64 prefill convention). Phase 2 streams the whole M dimension through the
// resident tile with a register-blocked f32-FMA GEMM (A stationary per thread,
// TILE_N accumulators in registers).
//
// PURPOSE: validate the persistent-N SCHEDULE and INV-1 (B is never
// materialized in HBM — only the packed stream is read, once). It is a
// correctness/decode-amortization REFERENCE, deliberately NOT a cuBLAS
// competitor: it uses f32 FMA, not the FP8 tensor cores (INV-2), so at large M
// it is far slower than the transient-expand path. The tensor-core endgame is
// the CUTLASS kernel specified in §4b of the design doc; whether to build it is
// gated on the opportunity-sizing bench, not on this kernel's throughput.
//
// Built/run only by tests/test_persistent_prefill.py + the bench (a separate
// JIT extension); nothing in the shipping decode path references it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#define DEVINL __device__ __forceinline__

namespace {

DEVINL float e4m3_to_f32(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
}
DEVINL uint16_t f32_to_bf16_rn(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// TILE_N output columns per CTA (smem-resident full-K, e4m3 bytes = TILE_N*K B).
// 8 keeps TILE_N*K <= 48 KB for K <= 6144 (no smem opt-in); larger K opts in.
template <int TILE_N, int THREADS>
__global__ __launch_bounds__(THREADS) void persistent_n_fp8_kernel(
    const uint8_t* __restrict__ A,        // [M, K] e4m3 (row-major, as u8)
    const uint8_t* __restrict__ packed,   // [N, row_stride] uint8 (padded rows)
    const uint16_t* __restrict__ lut16,   // e4m3-byte codebook as u16 pairs
    uint16_t* __restrict__ D,             // [M, N] bf16 (as u16), UNSCALED
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t row_stride, const int k_bits, const int type_size) {
  const int n_sb = (int)(K >> 8);
  // Ceil-first per-sub bit split (n_sub = 4), matching the encoder's
  // _bit_split: sub i holds base + (i < k%4) bits; its table starts at the
  // cumulative PAIR offset (sub_dim=2 bytes = 1 u16 pair per entry). Even k
  // reduces to the historical uniform split.
  int sub_off[4];
  uint32_t sub_mask4[4];
  int sub_pairbase[4];
  {
    const int base = k_bits >> 2, extra = k_bits & 3;
    int o = 0, pb = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int w = base + (i < extra ? 1 : 0);
      sub_off[i] = o;
      sub_mask4[i] = (1u << w) - 1u;
      sub_pairbase[i] = pb;
      o += w;
      pb += 1 << w;
    }
  }
  const uint64_t code_mask =
      (k_bits >= 64) ? ~0ull : ((1ull << k_bits) - 1ull);
  const int tid = (int)threadIdx.x;

  extern __shared__ __align__(16) uint8_t sB[];      // [TILE_N * K] e4m3 bytes

  for (int64_t n0 = (int64_t)blockIdx.x * TILE_N; n0 < N;
       n0 += (int64_t)gridDim.x * TILE_N) {
    int tn = TILE_N;
    if (N - n0 < (int64_t)TILE_N) tn = (int)(N - n0);   // last (ragged) N-tile

    // --- Phase 1: decode TILE_N rows x K into smem (e4m3 bytes), ONCE -------
    // 32 codewords/superblock; each -> 8 e4m3 bytes via 4 sub-index LUT gathers.
    const int total_cw = tn * n_sb * 32;
    for (int cw = tid; cw < total_cw; cw += THREADS) {
      const int r = cw / (n_sb * 32);                // row within tile
      const int rc = cw - r * (n_sb * 32);
      const int sb = rc >> 5;                        // superblock
      const int v = rc & 31;                         // codeword in superblock
      const uint8_t* rp = packed + (n0 + r) * row_stride + (int64_t)sb * type_size;
      const uint32_t* row32 = reinterpret_cast<const uint32_t*>(rp);
      const int bitpos = v * k_bits;
      const int b0 = bitpos >> 3;
      const int rem = ((b0 & 3) << 3) + (bitpos & 7);
      const int widx = b0 >> 2;
      const uint32_t w0 = row32[widx];
      const uint32_t w1 = row32[widx + 1];
      const uint32_t w2 = (rem + k_bits > 64) ? row32[widx + 2] : 0u;
      const uint64_t lo = ((uint64_t)w1 << 32) | (uint64_t)w0;
      uint64_t code = lo >> rem;
      if (rem + k_bits > 64) code |= (uint64_t)w2 << (64 - rem);
      code &= code_mask;
      uint8_t* dst = sB + (size_t)r * K + (size_t)sb * 256 + (size_t)v * 8;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint32_t idx = (uint32_t)(code >> sub_off[s]) & sub_mask4[s];
        const uint16_t pair = __ldg(lut16 + sub_pairbase[s] + idx);
        dst[2 * s] = (uint8_t)(pair & 0xff);
        dst[2 * s + 1] = (uint8_t)(pair >> 8);
      }
    }
    __syncthreads();

    // --- Phase 2: stream M, register-blocked f32 GEMM against the smem B ----
    // Thread owns a strip of M rows; for each row it reads A[m,:] once and
    // accumulates against all TILE_N columns (register accumulators), so decode
    // is amortized over the whole M dimension and B stays on chip.
    for (int64_t m = tid; m < M; m += THREADS) {
      const uint8_t* arow = A + m * K;
      float acc[TILE_N];
#pragma unroll
      for (int c = 0; c < TILE_N; ++c) acc[c] = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float a = e4m3_to_f32(arow[k]);
#pragma unroll
        for (int c = 0; c < TILE_N; ++c) {
          if (c < tn) acc[c] = fmaf(a, e4m3_to_f32(sB[(size_t)c * K + k]), acc[c]);
        }
      }
#pragma unroll
      for (int c = 0; c < TILE_N; ++c)
        if (c < tn) D[m * N + (n0 + c)] = f32_to_bf16_rn(acc[c]);
    }
    __syncthreads();                                 // before reusing sB
  }
}

torch::Tensor cb_prefill_persistent_n_fp8(torch::Tensor a, torch::Tensor packed,
                                          torch::Tensor lut, int64_t N,
                                          int64_t K, int64_t k_bits) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn,
              "a must be fp8 e4m3 [M,K]");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) == K,
              "a must be contiguous [M,K]");
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
                  packed.dim() == 2 && packed.size(0) == N &&
                  packed.stride(1) == 1,
              "packed must be [N, row_stride] uint8, row-contiguous");
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kUInt8);
  TORCH_CHECK(K % 256 == 0);
  const int type_size = 4 * (int)k_bits;
  const int64_t row_stride = packed.stride(0);
  TORCH_CHECK(row_stride >= (K / 256) * type_size,
              "packed row stride too small for K/k_bits");

  const int M = (int)a.size(0);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));
  if (M == 0 || N == 0) return d;

  constexpr int TILE_N = 8;
  constexpr int THREADS = 256;
  const size_t smem = (size_t)TILE_N * K;            // e4m3 bytes
  if (smem > 48 * 1024) {
    // Opt in to >48 KB dynamic smem (host-only, idempotent, capture-safe).
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        persistent_n_fp8_kernel<TILE_N, THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  }
  // Persistent grid: a couple of waves of CTAs; grid-strides over the N-tiles.
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                         a.get_device());
  if (sm_count < 1) sm_count = 1;
  const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
  const int64_t waves = (int64_t)sm_count * 2;
  unsigned grid = (unsigned)(n_tiles < waves ? n_tiles : waves);
  if (grid == 0) grid = 1;
  persistent_n_fp8_kernel<TILE_N, THREADS><<<grid, THREADS, smem, stream>>>(
      reinterpret_cast<const uint8_t*>(a.data_ptr()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<const uint16_t*>(lut.data_ptr()),
      reinterpret_cast<uint16_t*>(d.data_ptr()),
      M, N, K, row_stride, (int)k_bits, type_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_prefill_persistent_n_fp8", &cb_prefill_persistent_n_fp8,
        "persistent-N FP8_CB prefill reference (INV-1, decode-once, UNSCALED "
        "bf16 out); f32-FMA schedule/correctness reference, not the tensor-core "
        "perf path (see docs/nvfp4-cb-plan/persistent-n-prefill.md)");
}
