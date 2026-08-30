// Research-only QTIP sign + normalized-Hadamard CUDA primitive.
//
// The one-warp/128-value decomposition was selected after studying EXL3's
// MIT-licensed hadamard_inner.cuh at commit
// 0c49587a7c235e6303a6bbedc8b665272ad3a2ea (Copyright (c) 2025
// Turboderp).  No EXL3 source is imported or linked.  This implementation is
// original Gridbook code for BF16 tensors and Gridbook's x D H / y H D ABI:
// each lane owns four adjacent values, computes their H4 in registers, then
// completes H128 with five warp shuffles.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kBlockSize = 128;
constexpr int kWarpSize = 32;
constexpr int kValuesPerLane = 4;
constexpr int64_t kAbiSchema = 1;
constexpr float kInvSqrt128 = 0.08838834764831843f;

struct __align__(8) Bf16x4 {
  __nv_bfloat16 values[kValuesPerLane];
};

#define QTIP_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define QTIP_CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

// Apply five H2 factors across the 32 lanes.  A lane whose partner index is
// lower needs partner-current; a lower lane needs current+partner.  This is
// the same Sylvester stage order as the transparent torch reference.
__device__ __forceinline__ float hadamard_across_warp(
    float value, int lane) {
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    float const partner =
        __shfl_xor_sync(0xffffffffu, value, offset, kWarpSize);
    value = (lane & offset) ? partner - value : value + partner;
  }
  return value;
}

// One template argument avoids a CUDA 13 host-stub macro bug on kernels with
// comma-separated template arguments. Bit 0 is sign-before; bit 1 is vector
// loads.
template <int Mode>
__global__ __launch_bounds__(kWarpSize)
void qtip_hadamard_warp128_kernel(
    __nv_bfloat16 const* __restrict__ input,
    __nv_bfloat16 const* __restrict__ signs,
    __nv_bfloat16* __restrict__ output,
    int dimension,
    int blocks_per_row) {
  int const lane = int(threadIdx.x);
  int64_t const transform = int64_t(blockIdx.x);
  int64_t const row = transform / blocks_per_row;
  int const block = int(transform - row * blocks_per_row);
  int64_t const base = row * dimension + int64_t(block) * kBlockSize;
  int const within = lane * kValuesPerLane;
  int const sign_base = block * kBlockSize + within;

  constexpr bool SignBefore = (Mode & 1) != 0;
  constexpr bool VectorLoads = (Mode & 2) != 0;
  Bf16x4 input_values;
  Bf16x4 sign_values;
  if constexpr (VectorLoads) {
    // One aligned 8-byte transaction per plane/lane, the memory-side half of
    // the 4-values-per-lane decomposition. PyTorch allocations are normally
    // far more aligned; a sliced contiguous view can be the exception.
    input_values = reinterpret_cast<Bf16x4 const*>(input + base)[lane];
    sign_values = reinterpret_cast<Bf16x4 const*>(
        signs + block * kBlockSize)[lane];
  } else {
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      input_values.values[i] = input[base + within + i];
      sign_values.values[i] = signs[sign_base + i];
    }
  }
  float v0 = bf16_to_float(input_values.values[0]);
  float v1 = bf16_to_float(input_values.values[1]);
  float v2 = bf16_to_float(input_values.values[2]);
  float v3 = bf16_to_float(input_values.values[3]);
  if constexpr (SignBefore) {
    v0 *= bf16_to_float(sign_values.values[0]);
    v1 *= bf16_to_float(sign_values.values[1]);
    v2 *= bf16_to_float(sign_values.values[2]);
    v3 *= bf16_to_float(sign_values.values[3]);
  }

  // H4 over the adjacent low two index bits, entirely in registers.
  float const s01 = v0 + v1;
  float const d01 = v0 - v1;
  float const s23 = v2 + v3;
  float const d23 = v2 - v3;
  float h0 = s01 + s23;
  float h1 = d01 + d23;
  float h2 = s01 - s23;
  float h3 = d01 - d23;

  // H32 over the high five index bits.  Together H32 (x) H4 is the exact
  // normalized Sylvester H128 used by gridbook.qtip_hadamard.
  h0 = hadamard_across_warp(h0, lane) * kInvSqrt128;
  h1 = hadamard_across_warp(h1, lane) * kInvSqrt128;
  h2 = hadamard_across_warp(h2, lane) * kInvSqrt128;
  h3 = hadamard_across_warp(h3, lane) * kInvSqrt128;

  if constexpr (!SignBefore) {
    h0 *= bf16_to_float(sign_values.values[0]);
    h1 *= bf16_to_float(sign_values.values[1]);
    h2 *= bf16_to_float(sign_values.values[2]);
    h3 *= bf16_to_float(sign_values.values[3]);
  }
  Bf16x4 output_values;
  output_values.values[0] = float_to_bf16(h0);
  output_values.values[1] = float_to_bf16(h1);
  output_values.values[2] = float_to_bf16(h2);
  output_values.values[3] = float_to_bf16(h3);
  reinterpret_cast<Bf16x4*>(output + base)[lane] = output_values;
}

torch::Tensor qtip_hadamard_warp128_cuda(
    torch::Tensor input, torch::Tensor signs, bool sign_before) {
  QTIP_CHECK_CUDA(input);
  QTIP_CHECK_CUDA(signs);
  QTIP_CHECK_CONTIGUOUS(input);
  QTIP_CHECK_CONTIGUOUS(signs);
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must be bfloat16");
  TORCH_CHECK(signs.scalar_type() == torch::kBFloat16,
              "signs must be bfloat16");
  TORCH_CHECK(input.device() == signs.device(),
              "input and signs must share one CUDA device");
  TORCH_CHECK(input.dim() == 2,
              "input must be a contiguous 2-D [rows,dimension] tensor");
  TORCH_CHECK(signs.dim() == 1,
              "signs must be a contiguous 1-D [dimension] tensor");
  int64_t const rows64 = input.size(0);
  int64_t const dimension64 = input.size(1);
  TORCH_CHECK(dimension64 > 0 && dimension64 % kBlockSize == 0,
              "input dimension must be a positive multiple of 128, got ",
              dimension64);
  TORCH_CHECK(signs.numel() == dimension64,
              "signs length must equal input dimension ", dimension64,
              ", got ", signs.numel());
  TORCH_CHECK(rows64 <= INT_MAX && dimension64 <= INT_MAX,
              "input dimensions exceed int32 kernel limits");
  int64_t const blocks_per_row64 = dimension64 / kBlockSize;
  TORCH_CHECK(rows64 == 0 ||
                  blocks_per_row64 <= INT_MAX / rows64,
              "rows * (dimension / 128) exceeds CUDA grid limits");

  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty(
      input.sizes(), input.options().dtype(torch::kBFloat16));
  int64_t const transforms = rows64 * blocks_per_row64;
  if (transforms == 0) return output;
  auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());
  auto const* input_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
  auto const* sign_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(signs.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  bool const vector_loads =
      reinterpret_cast<uintptr_t>(input_ptr) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(sign_ptr) % alignof(Bf16x4) == 0;
  if (sign_before && vector_loads) {
    qtip_hadamard_warp128_kernel<3>
        <<<unsigned(transforms), kWarpSize, 0, stream>>>(input_ptr, sign_ptr,
            output_ptr, int(dimension64), int(blocks_per_row64));
  } else if (sign_before) {
    qtip_hadamard_warp128_kernel<1>
        <<<unsigned(transforms), kWarpSize, 0, stream>>>(input_ptr, sign_ptr,
            output_ptr, int(dimension64), int(blocks_per_row64));
  } else if (vector_loads) {
    qtip_hadamard_warp128_kernel<2>
        <<<unsigned(transforms), kWarpSize, 0, stream>>>(input_ptr, sign_ptr,
            output_ptr, int(dimension64), int(blocks_per_row64));
  } else {
    qtip_hadamard_warp128_kernel<0>
        <<<unsigned(transforms), kWarpSize, 0, stream>>>(input_ptr, sign_ptr,
            output_ptr, int(dimension64), int(blocks_per_row64));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qtip_hadamard_warp128", &qtip_hadamard_warp128_cuda,
        "Research QTIP BF16 sign + normalized H128 (one warp per block)");
  m.def("qtip_hadamard_abi_schema", []() { return kAbiSchema; });
}
