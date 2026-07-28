// ============================================================================
// Persistent-N FP8_CB prefill, TENSOR-CORE phase-2 (§4b v1, self-contained).
//
// Schedule identical to the §4a reference (cb_persistent_prefill.cu, parity-
// green): each CTA grid-strides over N-tiles; per N-tile, phase 1 decodes the
// packed CB rows for [TILE_N, K] ONCE into resident smem (e4m3 bytes,
// bit-exact to cb_expand_fp8 — same window extraction, same LUT); phase 2
// streams the whole M dimension through the resident tile. INV-1: decoded
// bytes never touch HBM. INV-2 (new vs §4a): phase 2 runs on the fp8 tensor
// cores via cute::TiledMma over SM89_16x8x32_F32E4M3E4M3F32_TN — 8 warps
// stacked along M => a [128, 8]x[k32] atom step; TILE_N=16 = two N-halves.
//
// A is streamed in [TILE_M, KCHUNK] double-buffered smem stages filled with
// cp.async; B fragments come from the resident buffer. Output D is UNSCALED
// bf16 (per-output-channel scale applied by the caller's epilogue, matching
// the transient-expand/fork convention and the §4a reference).
//
// v1 correctness choices (tune later, measured):
//   - default cute::copy for smem->rmem fragments (no ldmatrix yet)
//   - TILE_N=16 (smem: 16*K B-resident + 2*8 KB A stages; K<=4096 fits the
//     99 KB sm120 opt-in budget)
//   - one CTA per SM (grid = min(n_tiles, sm_count) from the host)
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm89.hpp>

#include <cstdint>

#define DEVINL __device__ __forceinline__

namespace pq_ptc {

using namespace cute;

constexpr int kThreads = 256;          // 8 warps
constexpr int kTileN = 16;
constexpr int kTileM = 128;
constexpr int kKChunk = 64;            // A stage width (bytes == elems, e4m3)

DEVINL float e4m3_to_f32(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
}

// --------------------------------------------------------------------------
// Phase 1: decode packed CB rows into the resident smem B tile.
// Verbatim transplant of the §4a reference decode (bit-exact contract).
// --------------------------------------------------------------------------
DEVINL void decode_resident_b(
    const uint8_t* __restrict__ packed, const uint16_t* __restrict__ lut16,
    uint8_t* __restrict__ sB, int64_t n0, int tn, int64_t K,
    int64_t row_stride, int k_bits, int type_size,
    const int* sub_off, const uint32_t* sub_mask4, const int* sub_pairbase,
    uint64_t code_mask) {
  const int n_sb = (int)(K >> 8);
  const int tid = (int)threadIdx.x;
  const int total_cw = tn * n_sb * 32;
  for (int cw = tid; cw < total_cw; cw += kThreads) {
    const int r = cw / (n_sb * 32);
    const int rc = cw - r * (n_sb * 32);
    const int sb = rc >> 5;
    const int v = rc & 31;
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
}

// --------------------------------------------------------------------------
// The kernel.
//   A      [M, K]  e4m3 (row-major u8)
//   packed [N, row_stride] u8 CB rows
//   D      [M, N]  bf16 (as u16), UNSCALED
// smem plan (extern, opt-in):
//   [0, kTileN*K)                       resident decoded B
//   [bOff, bOff + 2*kTileM*kKChunk)     A double-buffer stages
// --------------------------------------------------------------------------
template <bool kLdsmA>
__global__ __launch_bounds__(kThreads) void cb_persistent_tc_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ packed,
    const uint16_t* __restrict__ lut16, uint16_t* __restrict__ D,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t row_stride, const int k_bits, const int type_size) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* sB = smem;
  const size_t bBytes = (size_t)kTileN * (size_t)K;
  uint8_t* sA = smem + ((bBytes + 15) & ~size_t(15));

  // per-sub split tables (runtime k_bits, ceil-first — §4a convention)
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

  // TiledMma: 8 warps along M -> [128, 8] x k32 per pass; two N-halves.
  auto tiled_mma = make_tiled_mma(
      MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>{},
      Layout<Shape<_8, _1, _1>>{});
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  const int tid = (int)threadIdx.x;
  const int n_ktiles = (int)(K / kKChunk);

  for (int64_t n0 = (int64_t)blockIdx.x * kTileN; n0 < N;
       n0 += (int64_t)gridDim.x * kTileN) {
    const int tn = (N - n0 < (int64_t)kTileN) ? (int)(N - n0) : kTileN;

    // ---- Phase 1: decode once ------------------------------------------
    decode_resident_b(packed, lut16, sB, n0, tn, K, row_stride, k_bits,
                      type_size, sub_off, sub_mask4, sub_pairbase, code_mask);
    // Zero the ragged tail rows so the MMA reads defined bytes; their D
    // columns are never written.
    for (int z = tid; z < (kTileN - tn) * (int)K; z += kThreads) {
      sB[(size_t)tn * K + z] = 0;
    }
    __syncthreads();

    // Resident B as a cute tensor: [kTileN, K] row-major (K contiguous).
    auto sB_t = make_tensor(
        make_smem_ptr(reinterpret_cast<float_e4m3_t*>(sB)),
        make_layout(make_shape(Int<kTileN>{}, (int)K),
                    make_stride((int)K, Int<1>{})));

    // ---- Phase 2: sweep M ---------------------------------------------
    for (int64_t m0 = 0; m0 < M; m0 += kTileM) {
      const int tm = (M - m0 < (int64_t)kTileM) ? (int)(M - m0) : kTileM;

      // accumulators: [128,16] tile as two [128,8] halves
      auto acc0 = partition_fragment_C(
          tiled_mma, Shape<Int<kTileM>, Int<8>>{});
      auto acc1 = partition_fragment_C(
          tiled_mma, Shape<Int<kTileM>, Int<8>>{});
      clear(acc0);
      clear(acc1);

      // A stage tensors (double buffer)
      auto stage_layout = make_layout(
          make_shape(Int<kTileM>{}, Int<kKChunk>{}),
          make_stride(Int<kKChunk>{}, Int<1>{}));

      // prime stage 0
      int buf = 0;
      {
        const uint8_t* gsrc = A + m0 * K;
        uint8_t* gdst = sA;                     // stage 0
        for (int i = tid; i < kTileM * kKChunk / 16; i += kThreads) {
          const int row = (i * 16) / kKChunk;
          const int col = (i * 16) % kKChunk;
          if (row < tm) {
            __pipeline_memcpy_async(gdst + row * kKChunk + col,
                                    gsrc + (int64_t)row * K + col, 16);
          }
        }
        __pipeline_commit();
      }

      for (int kt = 0; kt < n_ktiles; ++kt) {
        // prefetch next stage
        if (kt + 1 < n_ktiles) {
          const uint8_t* gsrc = A + m0 * K + (int64_t)(kt + 1) * kKChunk;
          uint8_t* gdst = sA + (buf ^ 1) * (kTileM * kKChunk);
          for (int i = tid; i < kTileM * kKChunk / 16; i += kThreads) {
            const int row = (i * 16) / kKChunk;
            const int col = (i * 16) % kKChunk;
            if (row < tm) {
              __pipeline_memcpy_async(gdst + row * kKChunk + col,
                                      gsrc + (int64_t)row * K + col, 16);
            }
          }
        }
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();

        auto sA_t = make_tensor(
            make_smem_ptr(reinterpret_cast<float_e4m3_t*>(
                sA + buf * (kTileM * kKChunk))),
            stage_layout);

        // Two k32 MMA steps per 64-wide chunk, two N-halves each.
        CUTLASS_PRAGMA_UNROLL
        for (int ks = 0; ks < kKChunk / 32; ++ks) {
          auto sA_k = local_tile(sA_t, Shape<Int<kTileM>, _32>{},
                                 make_coord(0, ks));
          auto sB_k0 = local_tile(sB_t, Shape<_8, _32>{},
                                  make_coord(0, kt * (kKChunk / 32) + ks));
          auto sB_k1 = local_tile(sB_t, Shape<_8, _32>{},
                                  make_coord(1, kt * (kKChunk / 32) + ks));

          auto tCrA = thr_mma.partition_fragment_A(sA_k);
          if constexpr (kLdsmA) {
            // v2: ldmatrix.x4 A loads — one 128-bit LDSM per thread per
            // 8x16B matrix quad instead of scalar LDS.8 per element. The
            // plain row-major stage keeps every lane's 16 B segment
            // 16-aligned (kKChunk=64 rows, 32 B k-steps); bank conflicts
            // without a swizzle are accepted for v2 (measured next).
            auto ldsm_copy = make_tiled_copy_A(
                Copy_Atom<SM75_U32x4_LDSM_N, float_e4m3_t>{}, tiled_mma);
            auto thr_ldsm = ldsm_copy.get_thread_slice(threadIdx.x);
            auto tXsA = thr_ldsm.partition_S(sA_k);
            auto tXrA = thr_ldsm.retile_D(tCrA);
            copy(ldsm_copy, tXsA, tXrA);
          } else {
            auto tCsA = thr_mma.partition_A(sA_k);
            copy(tCsA, tCrA);
          }

          auto tCsB0 = thr_mma.partition_B(sB_k0);
          auto tCrB0 = thr_mma.make_fragment_B(tCsB0);
          copy(tCsB0, tCrB0);
          gemm(tiled_mma, tCrA, tCrB0, acc0);

          auto tCsB1 = thr_mma.partition_B(sB_k1);
          auto tCrB1 = thr_mma.make_fragment_B(tCsB1);
          copy(tCsB1, tCrB1);
          gemm(tiled_mma, tCrA, tCrB1, acc1);
        }
        __syncthreads();
        buf ^= 1;
      }

      // ---- epilogue: write the [tm, tn] block of D (bf16, unscaled) ----
      auto cD = make_identity_tensor(Shape<Int<kTileM>, Int<8>>{});
      auto tCcD = thr_mma.partition_C(cD);
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        auto& acc = (h == 0) ? acc0 : acc1;
        for (int i = 0; i < size(acc); ++i) {
          const auto coord = tCcD(i);
          const int mi = get<0>(coord);
          const int ni = get<1>(coord) + h * 8;
          if (mi < tm && ni < tn) {
            D[(m0 + mi) * N + (n0 + ni)] =
                __bfloat16_as_ushort(__float2bfloat16_rn(acc(i)));
          }
        }
      }
      __syncthreads();
    }
    // Teardown hygiene: no cp.async group may be outstanding when the CTA
    // retires (the final empty commit group otherwise stays in flight).
    __pipeline_wait_prior(0);
    __syncthreads();
  }
}

}  // namespace pq_ptc

// ---------------------------------------------------------------------------
torch::Tensor cb_prefill_persistent_tc(
    torch::Tensor a, torch::Tensor packed, torch::Tensor cb_flat_fp8,
    int64_t N, int64_t K, int64_t k_bits, int64_t type_size,
    int64_t variant) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8);
  TORCH_CHECK(cb_flat_fp8.scalar_type() == torch::kUInt8);
  TORCH_CHECK(a.dim() == 2 && (int64_t)a.size(1) == K);
  TORCH_CHECK(K % 256 == 0 && K % pq_ptc::kKChunk == 0);
  TORCH_CHECK(type_size == 4 * k_bits && type_size <= 192);
  TORCH_CHECK(packed.dim() == 2 && packed.size(0) == N &&
              packed.stride(1) == 1);
  // The LUT must cover all four sub-tables: 4 * 2^(k/4) u16 pairs for the
  // even splits this kernel serves. An undersized table turns into silent
  // out-of-bounds __ldg on unified memory — a UVM fault-state hazard (the
  // suspected wedge mechanism from the 2026-07-23 undersized-test runs).
  TORCH_CHECK(k_bits % 4 == 0, "persistent-TC v1 serves even splits only");
  const int64_t need_bytes = 4 * (int64_t(1) << (k_bits / 4)) * 2;
  TORCH_CHECK(cb_flat_fp8.numel() >= need_bytes,
              "codebook too small: need ", need_bytes, " bytes for K",
              k_bits, ", got ", cb_flat_fp8.numel());
  const int64_t M = a.size(0);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  const size_t bBytes = ((size_t)pq_ptc::kTileN * (size_t)K + 15) & ~size_t(15);
  const size_t smem = bBytes + 2 * (size_t)pq_ptc::kTileM * pq_ptc::kKChunk;
  TORCH_CHECK(smem <= 99 * 1024, "smem plan exceeds sm120 budget: K too large");
  const auto* kfn = (variant == 2)
      ? (const void*)pq_ptc::cb_persistent_tc_kernel<true>
      : (const void*)pq_ptc::cb_persistent_tc_kernel<false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, a.device().index()));
  const int n_tiles = (int)((N + pq_ptc::kTileN - 1) / pq_ptc::kTileN);
  const int grid = std::min(n_tiles, sm_count);

  const auto launch = (variant == 2)
      ? &pq_ptc::cb_persistent_tc_kernel<true>
      : &pq_ptc::cb_persistent_tc_kernel<false>;
  launch<<<grid, pq_ptc::kThreads, smem, stream>>>(
      reinterpret_cast<const uint8_t*>(a.data_ptr()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<const uint16_t*>(cb_flat_fp8.data_ptr()),
      reinterpret_cast<uint16_t*>(d.data_ptr()),
      M, N, K, packed.stride(0), (int)k_bits, (int)type_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_prefill_persistent_tc", &cb_prefill_persistent_tc,
        "Persistent-N FP8_CB prefill, tensor-core phase-2 (D unscaled bf16); "
        "variant: 1=scalar-copy v1, 2=ldmatrix-A v2",
        pybind11::arg("a"), pybind11::arg("packed"),
        pybind11::arg("cb_flat_fp8"), pybind11::arg("N"), pybind11::arg("K"),
        pybind11::arg("k_bits"), pybind11::arg("type_size"),
        pybind11::arg("variant") = 1);
}
