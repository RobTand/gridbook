// Native W8A16 execution for verbatim DeepSeek block-128 source FP8 weights.
//
// Resident state remains exactly the checkpoint's E4M3 [N,K] value plane and
// UE8M0 [ceil(N/128), ceil(K/128)] scale plane.  Decode (M<=8) streams those
// bytes directly through a bandwidth-bound GEMV.  Prefill expands one bounded
// layer tile to BF16 for Gridbook's owned CUTLASS BF16 bridge; the tile is
// caller-scoped and never becomes resident model state.
//
// Both paths reconstruct each weight identically:
//
//   w_bf16 = bf16_rn(e4m3(value_byte) * 2^(scale_byte - 127))
//
// GEMV converts that BF16 value back to FP32 before FMA, matching a BF16 GEMM
// consuming the expanded tile.  Activations enter as BF16 and are never QDQ'd.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlock = 128;
constexpr int kMaxM = 8;

__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
  __nv_bfloat16_raw raw;
  raw.x = bits;
  return __bfloat162float(__nv_bfloat16(raw));
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t bits) {
  return __half2float(
      __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)bits, __NV_E4M3));
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t bits) {
  // float8_e8m0fnu reserves 0xff as NaN.  Every other byte is exactly
  // 2^(byte-127), including byte zero (2^-127).
  return bits == 0xffu ? __int_as_float(0x7f800001)
                       : ldexpf(1.0f, int(bits) - 127);
}

template <int MT>
__global__ __launch_bounds__(kThreads) void fp8_source_gemv_kernel(
    const uint16_t* __restrict__ x,       // [M,G,K] BF16
    const uint8_t* __restrict__ q,        // [N,K] E4M3 bytes
    const uint8_t* __restrict__ scales,   // [ceil(N/128),ceil(K/128)]
    uint16_t* __restrict__ out,           // [M,N] BF16
    int m, int groups, int n_total, int k, int scale_cols) {
  const int n = int(blockIdx.x);
  const int group_rows = n_total / groups;
  const int group = n / group_rows;
  const int scale_row = n / kBlock;

  extern __shared__ uint8_t scale_cache[];
  for (int col = int(threadIdx.x); col < scale_cols; col += kThreads) {
    scale_cache[col] = scales[scale_row * scale_cols + col];
  }
  __shared__ float warp_partial[kMaxM][kWarps];
  __syncthreads();

  float accum[MT];
#pragma unroll
  for (int row = 0; row < MT; ++row) {
    accum[row] = 0.0f;
  }

  const uint8_t* q_row = q + int64_t(n) * k;
  for (int col = int(threadIdx.x); col < k; col += kThreads) {
    const float scale = ue8m0_to_f32(scale_cache[col / kBlock]);
    const float decoded = e4m3_to_f32(q_row[col]) * scale;
    // Establish the W8A16 weight value exactly once, just as the transient
    // expander does before the CUTLASS bridge consumes it.
    const float weight = bf16_to_f32(f32_to_bf16_rn(decoded));
#pragma unroll
    for (int row = 0; row < MT; ++row) {
      if (row < m) {
        const int64_t x_index =
            (int64_t(row) * groups + group) * k + col;
        accum[row] = fmaf(weight, bf16_to_f32(x[x_index]), accum[row]);
      }
    }
  }

  const int lane = int(threadIdx.x) & 31;
  const int warp = int(threadIdx.x) >> 5;
#pragma unroll
  for (int row = 0; row < MT; ++row) {
    float value = accum[row];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
      warp_partial[row][warp] = value;
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int row = 0; row < MT; ++row) {
      float value = lane < kWarps ? warp_partial[row][lane] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
      }
      if (lane == 0 && row < m) {
        out[int64_t(row) * n_total + n] = f32_to_bf16_rn(value);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads) void fp8_source_expand_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ scales,
    uint16_t* __restrict__ out,
    int n_total, int k, int scale_cols) {
  const int n = int(blockIdx.x);
  const int scale_row = n / kBlock;
  extern __shared__ uint8_t scale_cache[];
  for (int col = int(threadIdx.x); col < scale_cols; col += kThreads) {
    scale_cache[col] = scales[scale_row * scale_cols + col];
  }
  __syncthreads();

  const int64_t row = int64_t(n) * k;
  for (int col = int(threadIdx.x); col < k; col += kThreads) {
    const float decoded =
        e4m3_to_f32(q[row + col]) *
        ue8m0_to_f32(scale_cache[col / kBlock]);
    out[row + col] = f32_to_bf16_rn(decoded);
  }
}

void check_source_planes(const torch::Tensor& q,
                         const torch::Tensor& scales) {
  TORCH_CHECK(q.is_cuda() && scales.is_cuda(),
              "FP8 source value and scale planes must be CUDA tensors");
  TORCH_CHECK(q.device() == scales.device(),
              "FP8 source value and scale planes must share one CUDA device");
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn,
              "FP8 source value plane must be float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == torch::kUInt8 ||
                  scales.scalar_type() == torch::kFloat8_e8m0fnu,
              "FP8 source scale plane must be uint8/float8_e8m0fnu");
  TORCH_CHECK(q.dim() == 2 && scales.dim() == 2,
              "expected FP8 source q [N,K] and scales [ceil(N/128),"
              "ceil(K/128)]");
  TORCH_CHECK(q.is_contiguous() && scales.is_contiguous(),
              "FP8 source value and scale planes must be contiguous");

  const int64_t n = q.size(0);
  const int64_t k = q.size(1);
  TORCH_CHECK(n > 0 && k > 0,
              "FP8 source value plane must have positive N and K");
  TORCH_CHECK(scales.size(0) == (n + kBlock - 1) / kBlock &&
                  scales.size(1) == (k + kBlock - 1) / kBlock,
              "FP8 source block128 scale shape does not match q [N,K]");
  TORCH_CHECK(n <= std::numeric_limits<int>::max() &&
                  k <= std::numeric_limits<int>::max(),
              "FP8 source dimensions exceed int32");
}

torch::Tensor fp8_source_gemv(torch::Tensor x,
                              torch::Tensor q,
                              torch::Tensor scales,
                              int64_t groups64) {
  check_source_planes(q, scales);
  TORCH_CHECK(x.is_cuda() && x.device() == q.device(),
              "FP8 source activation must be on the planes' CUDA device");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "FP8 source W8A16 GEMV requires BF16 activations");
  TORCH_CHECK(x.dim() == 3 && x.is_contiguous(),
              "FP8 source W8A16 GEMV expects contiguous x [M,G,K]");
  TORCH_CHECK(groups64 > 0 && groups64 <= std::numeric_limits<int>::max(),
              "FP8 source W8A16 groups must be a positive int32 value");

  const int64_t m64 = x.size(0);
  const int64_t groups = groups64;
  const int64_t n64 = q.size(0);
  const int64_t k64 = q.size(1);
  TORCH_CHECK(m64 >= 1 && m64 <= kMaxM,
              "FP8 source W8A16 GEMV owns 1 <= M <= ", kMaxM,
              ", got M=", m64);
  TORCH_CHECK(x.size(1) == groups && x.size(2) == k64,
              "FP8 source activation shape must be [M,groups,K]");
  TORCH_CHECK(n64 % groups == 0,
              "FP8 source output rows N must be divisible by groups");

  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto output = torch::empty({m64, n64}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int m = int(m64);
  const int g = int(groups);
  const int n = int(n64);
  const int k = int(k64);
  const int scale_cols = int(scales.size(1));
  const size_t shared_bytes = size_t(scale_cols) * sizeof(uint8_t);

#define LAUNCH_SOURCE_GEMV(MT)                                             \
  fp8_source_gemv_kernel<MT><<<(unsigned)n, kThreads, shared_bytes, stream>>>(\
      reinterpret_cast<const uint16_t*>(x.data_ptr()),                     \
      reinterpret_cast<const uint8_t*>(q.data_ptr()),                      \
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),                 \
      reinterpret_cast<uint16_t*>(output.data_ptr()),                      \
      m, g, n, k, scale_cols)
  switch (m) {
    case 1: LAUNCH_SOURCE_GEMV(1); break;
    case 2: LAUNCH_SOURCE_GEMV(2); break;
    case 3: LAUNCH_SOURCE_GEMV(3); break;
    case 4: LAUNCH_SOURCE_GEMV(4); break;
    case 5: LAUNCH_SOURCE_GEMV(5); break;
    case 6: LAUNCH_SOURCE_GEMV(6); break;
    case 7: LAUNCH_SOURCE_GEMV(7); break;
    case 8: LAUNCH_SOURCE_GEMV(8); break;
    default: TORCH_CHECK(false, "unreachable FP8 source GEMV M");
  }
#undef LAUNCH_SOURCE_GEMV
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fp8_source_expand_bf16(torch::Tensor q,
                                     torch::Tensor scales) {
  check_source_planes(q, scales);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int n = int(q.size(0));
  const int k = int(q.size(1));
  const int scale_cols = int(scales.size(1));
  auto output = torch::empty(q.sizes(), q.options().dtype(torch::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = size_t(scale_cols) * sizeof(uint8_t);
  fp8_source_expand_kernel<<<(unsigned)n, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),
      reinterpret_cast<uint16_t*>(output.data_ptr()),
      n, k, scale_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_source_gemv", &fp8_source_gemv,
        "Raw-resident block128 FP8 x BF16 activation GEMV (W8A16)",
        pybind11::arg("x"), pybind11::arg("q"), pybind11::arg("scales"),
        pybind11::arg("groups"));
  m.def("fp8_source_expand_bf16", &fp8_source_expand_bf16,
        "Bounded block128 FP8 -> BF16 source-weight expansion",
        pybind11::arg("q"), pybind11::arg("scales"));
}
