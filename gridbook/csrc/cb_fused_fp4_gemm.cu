// NVFP4_CB fused decode-in-prologue BLOCK-SCALED prefill GEMM (fp4-MMA lane).
//
// The fp4 counterpart of cb_fused_gemm.cu: B is the PACKED NVFP4_CB byte
// stream + smem-resident value/compose LUTs; the dense fp4 tile never exists
// in HBM. The MMA is the sm120/sm121 block-scaled NVF4 path
// (SM120_16x8x64_TN_VS = mma.kind::mxf4nvf4.block_scale ue4m3 -> OMMA.SF.16864,
// k=64 — twice the per-instruction K of fp8's QMMA.16832). The trap this file
// exists to avoid: kind::f8f6f4 also ACCEPTS e2m1 operands but issues at the
// fp8 k=32 rate; the fork static_asserts the NVF4 atom, and the SASS gate in
// docs/lanes/nvfp4-cb/fp4-fused-prefill.md disassembles the built module to
// prove OMMA.SF.16864 is present and QMMA is not.
//
// Entry points:
//  - sm120_nvf4_mm_scaled: STOCK block-scaled collective (dense packed-e2m1
//    B + swizzled SFB from gmem) at the identical TileShape/TiledMma/epilogue
//    config — the bit-exactness REFERENCE for the fused kernel (the fork64
//    role from the fp8 workstream).
//  - cb_fused_fp4_prefill_mm_scaled: the decode-in-prologue NVFP4_CB GEMM.
//    k_bits/n_sub/type_size/scale-coding are RUNTIME parameters — one
//    instantiation serves K12..K24 product, S13..S16 signed, v1 and v2.
//
// Scale convention (weight side is EXACT, activation side is native NVFP4):
//  - SFB = the per-group-16 e4m3 weight scale, composed in-prologue (v2
//    two-tier compose is exact e4m3 by construction; v1 plane bytes are
//    already e4m3) and applied INSIDE the MMA. No per-channel weight scale
//    exists for fp4-CB, so b_scales is normally ones (kept in the EVT for
//    signature symmetry with the fp8 entry).
//  - SFA = the activation group scale in ue4m3 from native NVFP4 activation
//    quantization; the per-tensor (or per-token) fp32 residual 1/global_scale
//    arrives as a_scales via the fp32 EVT epilogue:
//      D = bf16_rn(b_scale[n] * (a_scale[m] * acc_f32))
//    — the same node tree and rounding order as cb_fused_prefill_mm_scaled.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <climits>
#include <cmath>
#include <optional>

#include <pybind11/stl.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass_fork/sm120_cb_fused_fp4_mma.hpp"

namespace {

using namespace cute;

using ElementPairAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementSF = cutlass::float_ue4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignAB = 32;    // e2m1 elements (16 bytes)
constexpr int AlignD = 8;
using ClusterShape = Shape<_1, _1, _1>;
// TileN is pinned at 128 by the blockscaled SF smem atom (Blk_MN = 128);
// TileK = 128 is half a 256-weight CB superblock (the fork's decode grain).
using TileShapeFp4 = Shape<_128, _128, _128>;
using TileShapeFp4M256 = Shape<_256, _128, _128>;

using Sm1xxCfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// Same EVT node tree as cb_fused_gemm.cu's ScaledFusion (fp32 multiply chain,
// one round to bf16): D = bf16_rn(b_scale[n] * (a_scale[m] * acc)).
template <class TileShape>
struct ScaledFusion {
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, TileShape, float, float, Stride<_1, _0, _0>>;
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, TileShape, float, float, Stride<_0, _1, _0>>;
  using AccFetch = cutlass::epilogue::fusion::Sm90AccFetch;
  using MulA = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementAcc, ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using MulB = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTA = cutlass::epilogue::fusion::Sm90EVT<MulA, ScaleA, AccFetch>;
  using type = cutlass::epilogue::fusion::Sm90EVT<MulB, ScaleB, EVTA>;
};

template <class TileShape>
struct CfgFp4 {
  using Fusion = typename ScaledFusion<TileShape>::type;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      void, LayoutD, AlignD,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      Fusion>::CollectiveOp;
  using BuilderMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPairAB, LayoutA, AlignAB,
      ElementPairAB, LayoutB, AlignAB,
      ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

// Hard smem gate (see cb_fused_gemm.cu — the sm90 cooperative kernel layer
// does not static_assert its own SharedStorageSize).
template <class GemmKernel>
struct AssertSmemFits {
  static_assert((int)GemmKernel::SharedStorageSize <=
                    cutlass::arch::sm120_smem_capacity_bytes,
                "fused CB fp4 kernel exceeds the sm_120 shared-memory capacity");
  static constexpr bool value = true;
};

void check_same_cuda_device(torch::Tensor const& anchor,
                            torch::Tensor const& tensor,
                            char const* name) {
  TORCH_CHECK(tensor.device() == anchor.device(), name,
              " must be on the same CUDA device as a (", anchor.device(),
              "), got ", tensor.device());
}

void check_dense_packed_row_storage(torch::Tensor const& packed,
                                    int64_t rows,
                                    int64_t required_row_bytes) {
  TORCH_CHECK(packed.size(1) >= required_row_bytes,
              "packed visible row width ", packed.size(1),
              " is smaller than the required ", required_row_bytes,
              " bytes; passing a narrow view may not hide bytes read by the "
              "fused kernel");
  TORCH_CHECK(packed.stride(0) >= required_row_bytes,
              "packed row stride ", packed.stride(0),
              " is smaller than the required ", required_row_bytes,
              " bytes");

  // A logical shape/stride check alone does not attest a tensor whose backing
  // storage was externally resized after the view was created.  The fused
  // producer reads ``required_row_bytes`` from every row starting at
  // data_ptr()+row*stride(0), so prove that the final read remains inside the
  // underlying uint8 storage.  Division avoids overflow in (rows-1)*stride.
  TORCH_CHECK(packed.storage_offset() >= 0,
              "packed storage offset must be nonnegative");
  const uint64_t storage_bytes = packed.storage().nbytes();
  const uint64_t storage_offset =
      static_cast<uint64_t>(packed.storage_offset());
  const uint64_t row_bytes = static_cast<uint64_t>(required_row_bytes);
  const uint64_t row_stride = static_cast<uint64_t>(packed.stride(0));
  const uint64_t rows_before_last = static_cast<uint64_t>(rows - 1);
  TORCH_CHECK(storage_offset <= storage_bytes &&
                  row_bytes <= storage_bytes - storage_offset,
              "packed backing storage is too small for even the first "
              "required row");
  const uint64_t bytes_before_last =
      storage_bytes - storage_offset - row_bytes;
  TORCH_CHECK(rows_before_last == 0 ||
                  row_stride <= bytes_before_last / rows_before_last,
              "packed backing storage is too small for ", rows,
              " rows of ", required_row_bytes, " bytes at stride ",
              packed.stride(0), " from storage offset ",
              packed.storage_offset());
}

// ---------------------------------------------------------------------------
// Row-wise native-NVFP4 activation quantization.
//
// vLLM's public scaled_fp4_quant takes one global scale for the complete
// [M,K] tensor.  That makes one row's bytes depend on the other rows present
// in the request.  The GEMMs in this file already consume one residual
// a_scale per M row, so derive a full-range global scale independently for
// each row and preserve the standard UE4M3-group/E2M1 payload contract.  This
// is one activation primitive shared by future dense and grouped dispatch; it
// does not duplicate either weight decoder or GEMM.

__device__ __forceinline__ float cb_rcp_approx_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n"
               : "=f"(result) : "f"(value));
  return result;
}

template <typename InputT>
__device__ __forceinline__ float cb_input_to_float(InputT value);

template <>
__device__ __forceinline__ float cb_input_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float cb_input_to_float(half value) {
  return __half2float(value);
}

__device__ __forceinline__ uint32_t cb_pack_e2m1x8(float const (&values)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t packed;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(packed)
      : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]),
        "f"(values[4]), "f"(values[5]), "f"(values[6]), "f"(values[7]));
  return packed;
#else
  return 0;
#endif
}

__device__ __forceinline__ int64_t cb_sfa_swizzle_offset(
    int row, int group, int groups) {
  // K is required to be a multiple of 256, hence groups=K/16 is already a
  // multiple of four and no separate K padding term is needed.
  return (row % 32) * 16 + ((row / 32) % 4) * 4 + (group % 4)
      + (group / 4) * 512
      + (row / 128) * (512 * (groups / 4));
}

template <typename InputT>
__device__ __forceinline__ void cb_nvfp4_quantize_rows_body(
    InputT const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    int M, int K, float range_multiplier) {
  int const row = static_cast<int>(blockIdx.x);
  int const groups = K / 16;
  if (row >= M) {
    for (int group = static_cast<int>(threadIdx.x); group < groups;
         group += static_cast<int>(blockDim.x)) {
      sfa[cb_sfa_swizzle_offset(row, group, groups)] = 0;
    }
    return;
  }

  InputT const* row_input = input + static_cast<int64_t>(row) * K;
  float local_max = 0.0f;
  for (int col = static_cast<int>(threadIdx.x); col < K;
       col += static_cast<int>(blockDim.x)) {
    local_max = fmaxf(local_max, fabsf(cb_input_to_float(row_input[col])));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max,
                      __shfl_down_sync(0xffffffffu, local_max, offset));
  }
  __shared__ float warp_max[8];
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_max[warp] = local_max;
  }
  __syncthreads();
  if (warp == 0) {
    float block_max = lane < 8 ? warp_max[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_max = fmaxf(block_max,
                        __shfl_down_sync(0xffffffffu, block_max, offset));
    }
    if (lane == 0) {
      warp_max[0] = block_max;
    }
  }
  __syncthreads();

  float const row_max = warp_max[0];
  float global_scale = 1.0f;
  if (row_max > 0.0f && isfinite(row_max)) {
    global_scale = __fdiv_rn(range_multiplier * 6.0f, row_max);
  }
  if (threadIdx.x == 0) {
    a_scales[row] = row_max > 0.0f && isfinite(row_max)
        ? __fdiv_rn(1.0f, global_scale)
        : 0.0f;
  }

  uint8_t* row_packed = packed + static_cast<int64_t>(row) * (K / 2);
  for (int group = static_cast<int>(threadIdx.x); group < groups;
       group += static_cast<int>(blockDim.x)) {
    float values[16];
    float group_max = 0.0f;
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      float const value = cb_input_to_float(row_input[group * 16 + item]);
      values[item] = value;
      group_max = fmaxf(group_max, fabsf(value));
    }

    float sf_value = 0.0f;
    if (group_max > 0.0f && row_max > 0.0f && isfinite(row_max)) {
      sf_value = global_scale * (group_max * cb_rcp_approx_ftz(6.0f));
      sf_value = fminf(sf_value, 448.0f);
    }
    __nv_fp8_e4m3 stored_sf = __nv_fp8_e4m3(sf_value);
    uint8_t const sf_byte = stored_sf.__x;
    sfa[cb_sfa_swizzle_offset(row, group, groups)] = sf_byte;
    float const narrowed_sf = static_cast<float>(stored_sf);

    float normalize = 0.0f;
    if (narrowed_sf != 0.0f) {
      normalize = cb_rcp_approx_ftz(
          narrowed_sf * cb_rcp_approx_ftz(global_scale));
    }
    float low[8];
    float high[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      // A zero UE4M3 scale makes the complete group numerical zero. Preserve
      // the input sign on the E2M1 zero payload, matching vLLM/PTX.
      low[item] = narrowed_sf == 0.0f
          ? copysignf(0.0f, values[item])
          : values[item] * normalize;
      high[item] = narrowed_sf == 0.0f
          ? copysignf(0.0f, values[item + 8])
          : values[item + 8] * normalize;
    }
    uint32_t* group_out = reinterpret_cast<uint32_t*>(row_packed + group * 8);
    group_out[0] = cb_pack_e2m1x8(low);
    group_out[1] = cb_pack_e2m1x8(high);
  }
}

// Keep the registered CUDA entry points non-templated and outside the
// anonymous namespace. NVCC emits host launch stubs for __global__ functions
// and cannot disambiguate this file's anonymous namespace from CUTE's own
// anonymous namespace on CUDA 13. The typed device body remains shared, so
// the FP16/BF16 paths cannot drift.
}  // namespace

__global__ void cb_nvfp4_quantize_rows_bf16_kernel(
    __nv_bfloat16 const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    int M, int K, float range_multiplier) {
  cb_nvfp4_quantize_rows_body(
      input, packed, sfa, a_scales, M, K, range_multiplier);
}

__global__ void cb_nvfp4_quantize_rows_fp16_kernel(
    half const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    int M, int K, float range_multiplier) {
  cb_nvfp4_quantize_rows_body(
      input, packed, sfa, a_scales, M, K, range_multiplier);
}

namespace {

void cb_nvfp4_quantize_rows_out(torch::Tensor input,
                                torch::Tensor packed,
                                torch::Tensor sfa,
                                torch::Tensor a_scales,
                                double range_multiplier) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous() &&
              input.stride(1) == 1 && input.stride(0) == input.size(1),
              "input must be contiguous [M,K]");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 ||
              input.scalar_type() == torch::kFloat16,
              "input must be bfloat16 or float16");
  TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0,
              "input M and K must be positive");
  TORCH_CHECK(input.size(0) <= INT_MAX && input.size(1) <= INT_MAX,
              "input dimensions exceed int32 kernel limits");
  TORCH_CHECK(input.size(1) % 256 == 0,
              "input K must be a multiple of 256");
  TORCH_CHECK(std::isfinite(range_multiplier) && range_multiplier > 0.0 &&
              range_multiplier <= 448.0,
              "range_multiplier must be finite and in (0,448]");

  int const M = static_cast<int>(input.size(0));
  int const K = static_cast<int>(input.size(1));
  int const padded_m = ((M + 127) / 128) * 128;
  int64_t const sfa_size = static_cast<int64_t>(padded_m) * (K / 16);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
              packed.dim() == 2 && packed.size(0) == M &&
              packed.size(1) == K / 2 && packed.is_contiguous(),
              "packed must be contiguous CUDA uint8 [M,K/2]");
  TORCH_CHECK(sfa.is_cuda() && sfa.scalar_type() == torch::kUInt8 &&
              sfa.numel() == sfa_size && sfa.is_contiguous(),
              "sfa must be contiguous CUDA uint8 with ", sfa_size,
              " elements");
  TORCH_CHECK(a_scales.is_cuda() &&
              a_scales.scalar_type() == torch::kFloat32 &&
              a_scales.numel() == M && a_scales.is_contiguous(),
              "a_scales must be contiguous CUDA float32 [M]");
  check_same_cuda_device(input, packed, "packed");
  check_same_cuda_device(input, sfa, "sfa");
  check_same_cuda_device(input, a_scales, "a_scales");

  const c10::cuda::OptionalCUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 const block(256);
  dim3 const grid(padded_m);
  if (input.scalar_type() == torch::kBFloat16) {
    ::cb_nvfp4_quantize_rows_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
        packed.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
        a_scales.data_ptr<float>(), M, K,
        static_cast<float>(range_multiplier));
  } else {
    ::cb_nvfp4_quantize_rows_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half const*>(input.data_ptr()),
        packed.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
        a_scales.data_ptr<float>(), M, K,
        static_cast<float>(range_multiplier));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cb_nvfp4_quantize_rows(torch::Tensor input, double range_multiplier) {
  TORCH_CHECK(input.dim() == 2, "input must be rank 2");
  int64_t const M = input.size(0);
  int64_t const K = input.size(1);
  int64_t const padded_m = ((M + 127) / 128) * 128;
  auto packed = torch::empty({M, K / 2},
      input.options().dtype(torch::kUInt8));
  auto sfa = torch::empty({padded_m * (K / 16)},
      input.options().dtype(torch::kUInt8));
  auto a_scales = torch::empty({M}, input.options().dtype(torch::kFloat32));
  cb_nvfp4_quantize_rows_out(
      input, packed, sfa, a_scales, range_multiplier);
  return {packed, sfa, a_scales};
}

// Policy rebind: builder-constructed blockscaled collective -> the CB fused
// fp4 mainloop (Stages pinned to 2, everything else inherited verbatim).
template <int NewStages, class T>
struct SwapToFusedFp4;
template <int NewStages, int S, int SP, class CS, class KS, class... Rest>
struct SwapToFusedFp4<NewStages, cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<S, SP, CS, KS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120CbFusedFp4TmaWarpSpecialized<NewStages, SP, CS, KS>, Rest...>;
};

// ---------------------------------------------------------------------------
// Reference: STOCK block-scaled NVF4 GEMM at the fused kernel's exact config.
// ---------------------------------------------------------------------------
torch::Tensor sm120_nvf4_mm_scaled(torch::Tensor a, torch::Tensor sfa,
                                   torch::Tensor b, torch::Tensor sfb,
                                   torch::Tensor a_scales,
                                   torch::Tensor b_scales,
                                   int64_t N, int64_t K) {
  using TileShape = TileShapeFp4;
  using Mainloop = typename CfgFp4<TileShape>::BuilderMainloop;
  using Epilogue = typename CfgFp4<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(AssertSmemFits<GemmKernel>::value);

  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kUInt8 && a.dim() == 2 &&
              a.stride(1) == 1 && a.size(1) * 2 == K && a.stride(0) == K / 2,
              "a must be contiguous packed-e2m1 uint8 [M, K/2]");
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == torch::kUInt8 && b.dim() == 2 &&
              b.stride(1) == 1 && b.size(0) == N && b.size(1) * 2 == K &&
              b.stride(0) == K / 2,
              "b must be contiguous packed-e2m1 uint8 [N, K/2]");
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  const int M = (int)a.size(0);
  const int64_t sfa_need = ((M + 127) / 128) * 128 * (K / 16);
  const int64_t sfb_need = ((N + 127) / 128) * 128 * (K / 16);
  TORCH_CHECK(sfa.is_cuda() && sfa.numel() == sfa_need && sfa.is_contiguous(),
              "sfa must be the swizzled ue4m3 plane, numel ", sfa_need);
  TORCH_CHECK(sfa.scalar_type() == torch::kUInt8,
              "sfa must have uint8 storage");
  TORCH_CHECK(sfb.is_cuda() && sfb.scalar_type() == torch::kUInt8 &&
              sfb.numel() == sfb_need && sfb.is_contiguous(),
              "sfb must be the swizzled ue4m3 plane, numel ", sfb_need);
  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
              a_scales.numel() == M && a_scales.is_contiguous());
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
              b_scales.numel() == N && b_scales.is_contiguous());
  check_same_cuda_device(a, sfa, "sfa");
  check_same_cuda_device(a, b, "b");
  check_same_cuda_device(a, sfb, "sfb");
  check_same_cuda_device(a, a_scales, "a_scales");
  check_same_cuda_device(a, b_scales, "b_scales");

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, (int)K, 1});
  StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {(int)N, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, (int)N, 1});
  auto layout_sfa = Sm1xxCfg::tile_atom_to_shape_SFA(
      cute::make_shape(M, (int)N, (int)K, 1));
  auto layout_sfb = Sm1xxCfg::tile_atom_to_shape_SFB(
      cute::make_shape(M, (int)N, (int)K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, (int)N, (int)K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a.data_ptr()), sa,
       reinterpret_cast<const cutlass::float_e2m1_t*>(b.data_ptr()), sb,
       reinterpret_cast<const ElementSF*>(sfa.data_ptr()), layout_sfa,
       reinterpret_cast<const ElementSF*>(sfb.data_ptr()), layout_sfb},
      {// EVT args: children first, then node op args (empty for multiplies).
       {{b_scales.data_ptr<float>(), 0.0f, Stride<_0, _1, _0>{}},
        {{a_scales.data_ptr<float>(), 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, typename GemmKernel::StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "nvf4 ref can_implement failed");
  auto init_status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "nvf4 ref initialize failed: ",
              cutlass::cutlassGetStatusString(init_status));
  auto run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "nvf4 ref launch failed: ",
              cutlass::cutlassGetStatusString(run_status));
  return d;
}

// ---------------------------------------------------------------------------
// Fused decode-in-prologue entry (runtime rung parameters).
// ---------------------------------------------------------------------------
torch::Tensor cb_fused_fp4_prefill_mm_scaled(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, int64_t N, int64_t K, int64_t k_bits,
    int64_t n_sub, int64_t type_size, bool is_v2,
    std::optional<torch::Tensor> lut_tile_ids) {
  using TileShape = TileShapeFp4;
  using Mainloop = typename SwapToFusedFp4<
      2, typename CfgFp4<TileShape>::BuilderMainloop>::type;
  using Epilogue = typename CfgFp4<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(AssertSmemFits<GemmKernel>::value);

  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kUInt8 && a.dim() == 2 &&
              a.stride(1) == 1 && a.size(1) * 2 == K && a.stride(0) == K / 2,
              "a must be contiguous packed-e2m1 uint8 [M, K/2]");
  TORCH_CHECK(K > 0 && K % 256 == 0,
              "K must be a positive multiple of 256");
  TORCH_CHECK(N > 0 && N % 8 == 0,
              "N must be a positive multiple of 8 (bf16 TMA epilogue "
              "alignment; every exported CB Linear satisfies this)");
  TORCH_CHECK(k_bits >= 9 && k_bits <= 24, "fp4 rung k_bits out of range");
  TORCH_CHECK(n_sub == 1 || n_sub == 2, "fp4 n_sub must be 1 (signed) or 2");
  TORCH_CHECK(type_size == 4 * k_bits + (is_v2 ? 9 : 16),
              "type_size inconsistent with k_bits/scale coding");
  const int M = (int)a.size(0);
  const int64_t n_sb = K / 256;
  const int64_t sfa_need = ((M + 127) / 128) * 128 * (K / 16);
  TORCH_CHECK(sfa.is_cuda() && sfa.scalar_type() == torch::kUInt8 &&
              sfa.numel() == sfa_need && sfa.is_contiguous(),
              "sfa must be contiguous uint8 swizzled ue4m3 storage, numel ",
              sfa_need);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
              packed.dim() == 2 && packed.size(0) == N && packed.stride(1) == 1,
              "packed must be uint8 [N, row_bytes(+pad)]");
  check_dense_packed_row_storage(packed, N, n_sb * type_size);
  // No tail-slack requirement: the consumer's gmem gathers stay inside each
  // row's own superblock (aligned-u32 index windows end <= ts-1; the scale
  // plane is read with u8 loads).
  // Value LUT size must match the rung exactly. Product: two ceil-first
  // sub-tables of u16 nibble-quads; signed: 2^(k-8) u32 nibble-octets.
  const int64_t w0 = k_bits - k_bits / 2;
  const int64_t lut_need = (n_sub == 2)
      ? ((1LL << w0) + (1LL << (k_bits / 2))) * 2
      : (1LL << (k_bits - 8)) * 4;
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kUInt8 &&
              lut.is_contiguous() && lut.numel() >= lut_need &&
              lut.numel() % lut_need == 0,
              "value LUT must contain one or more contiguous uint8[", lut_need,
              "] blocks for this rung");
  TORCH_CHECK(lut_need <= 16384, "value LUT exceeds the smem carve");
  TORCH_CHECK(lut.numel() / lut_need <= INT32_MAX,
              "too many concatenated value LUT blocks");
  const int32_t num_lut_blocks = static_cast<int32_t>(lut.numel() / lut_need);
  const int64_t num_lut_tiles_64 = (N + 127) / 128;
  TORCH_CHECK(num_lut_tiles_64 <= INT32_MAX, "too many dense N tiles");
  int32_t const* lut_tile_ids_ptr = nullptr;
  int32_t num_lut_tiles = 0;
  if (lut_tile_ids.has_value()) {
    torch::Tensor const& ids = *lut_tile_ids;
    TORCH_CHECK(ids.defined() && ids.is_cuda() &&
                ids.scalar_type() == torch::kInt32 && ids.is_contiguous() &&
                ids.numel() == num_lut_tiles_64,
                "lut_tile_ids must be contiguous CUDA int32[ceil(N/128)] (",
                num_lut_tiles_64, " entries)");
    check_same_cuda_device(a, ids, "lut_tile_ids");
    lut_tile_ids_ptr = ids.data_ptr<int32_t>();
    num_lut_tiles = static_cast<int32_t>(num_lut_tiles_64);
  } else {
    TORCH_CHECK(num_lut_blocks == 1,
                "multiple value LUT blocks require lut_tile_ids");
  }
  TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kUInt8 &&
              compose.is_contiguous(),
              "compose must be contiguous CUDA uint8 storage");
  if (is_v2) {
    TORCH_CHECK(compose.numel() == 4096,
                "v2 compose table must be uint8[4096] (256x16 e4m3 bytes)");
  }
  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
              a_scales.numel() == M && a_scales.is_contiguous());
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
              b_scales.numel() == N && b_scales.is_contiguous());
  check_same_cuda_device(a, sfa, "sfa");
  check_same_cuda_device(a, packed, "packed");
  check_same_cuda_device(a, lut, "lut");
  check_same_cuda_device(a, compose, "compose");
  check_same_cuda_device(a, a_scales, "a_scales");
  check_same_cuda_device(a, b_scales, "b_scales");

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, (int)N, 1});
  auto layout_sfa = Sm1xxCfg::tile_atom_to_shape_SFA(
      cute::make_shape(M, (int)N, (int)K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, (int)N, (int)K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a.data_ptr()), sa,
       reinterpret_cast<const ElementSF*>(sfa.data_ptr()), layout_sfa,
       packed.data_ptr<uint8_t>(), packed.stride(0),
       lut.data_ptr<uint8_t>(), (int32_t)lut_need,
       lut_tile_ids_ptr, num_lut_blocks, num_lut_tiles,
       is_v2 ? compose.data_ptr<uint8_t>() : nullptr,
       (int32_t)k_bits, (int32_t)n_sub, (int32_t)type_size,
       (int32_t)(is_v2 ? 1 : 0), nullptr, 0, 0, nullptr},
      {// EVT args: children first, then node op args (empty for multiplies).
       {{b_scales.data_ptr<float>(), 0.0f, Stride<_0, _1, _0>{}},
        {{a_scales.data_ptr<float>(), 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, typename GemmKernel::StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fused fp4 can_implement failed (K%256? type_size? lut?)");
  auto init_status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "fused fp4 initialize failed: ",
              cutlass::cutlassGetStatusString(init_status));
  auto run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "fused fp4 launch failed: ",
              cutlass::cutlassGetStatusString(run_status));
  return d;
}

// ---------------------------------------------------------------------------
// GROUPED (MoE) fused entry — TILE-INDEXED grouping, the cb_fused_gemm.cu
// mechanism verbatim: the caller pre-gathers and PADS A's rows so every
// expert's segment spans whole TileM blocks; expert_ids[m_tile] picks the
// expert whose PACKED rows the producer stages for that M-tile. One ordinary
// launch, no ptr-arrays, no tensormap updates. Unlike fp8 there is no
// per-expert-channel weight scale (SFB carries the whole weight scale), so
// the epilogue stays the plain ScaledFusion with b_scales = ones.
// ---------------------------------------------------------------------------
torch::Tensor run_fp4_moe_grouped_m128(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, torch::Tensor expert_ids, int64_t N, int64_t K,
    int64_t k_bits, int64_t n_sub, int64_t type_size, bool is_v2) {
  // Keep this concrete (rather than templating the host runner on TileShape).
  // nvcc 13 otherwise preserves unsigned C<16u>/C<2u> layout aliases in the
  // host pass, producing an unregistered device_kernel duplicate.
  using TileShape = TileShapeFp4;
  using Mainloop = typename SwapToFusedFp4<
      2, typename CfgFp4<TileShape>::BuilderMainloop>::type;
  using Epilogue = typename CfgFp4<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(AssertSmemFits<GemmKernel>::value);

  const int Mp = (int)a.size(0);
  const int64_t row_bytes = packed.size(2);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({Mp, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {Mp, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {Mp, (int)N, 1});
  auto layout_sfa = Sm1xxCfg::tile_atom_to_shape_SFA(
      cute::make_shape(Mp, (int)N, (int)K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mp, (int)N, (int)K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a.data_ptr()), sa,
       reinterpret_cast<const ElementSF*>(sfa.data_ptr()), layout_sfa,
       packed.data_ptr<uint8_t>(), row_bytes,
       lut.data_ptr<uint8_t>(), (int32_t)lut.numel(),
       nullptr, 1, 0,
       is_v2 ? compose.data_ptr<uint8_t>() : nullptr,
       (int32_t)k_bits, (int32_t)n_sub, (int32_t)type_size,
       (int32_t)(is_v2 ? 1 : 0),
       expert_ids.data_ptr<int>(), N * row_bytes, (int32_t)packed.size(0),
       nullptr},
      {{{b_scales.data_ptr<float>(), 0.0f, Stride<_0, _1, _0>{}},
        {{a_scales.data_ptr<float>(), 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, typename GemmKernel::StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fused fp4 moe can_implement failed");
  auto init_status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "fused fp4 moe initialize failed: ",
              cutlass::cutlassGetStatusString(init_status),
              "; CUDA status: ", cudaGetErrorString(cudaPeekAtLastError()));
  auto run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "fused fp4 moe launch failed: ",
              cutlass::cutlassGetStatusString(run_status),
              "; CUDA status: ", cudaGetErrorString(cudaPeekAtLastError()));
  return d;
}

torch::Tensor run_fp4_moe_grouped_m256(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, torch::Tensor expert_ids, int64_t N, int64_t K,
    int64_t k_bits, int64_t n_sub, int64_t type_size, bool is_v2) {
  // See the M128 runner: the kernel config must remain concrete at this host
  // call site so nvcc emits and registers the matching TileM=256 host stub.
  using TileShape = TileShapeFp4M256;
  using Mainloop = typename SwapToFusedFp4<
      2, typename CfgFp4<TileShape>::BuilderMainloop>::type;
  using Epilogue = typename CfgFp4<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(AssertSmemFits<GemmKernel>::value);

  const int Mp = (int)a.size(0);
  const int64_t row_bytes = packed.size(2);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({Mp, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {Mp, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {Mp, (int)N, 1});
  auto layout_sfa = Sm1xxCfg::tile_atom_to_shape_SFA(
      cute::make_shape(Mp, (int)N, (int)K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mp, (int)N, (int)K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a.data_ptr()), sa,
       reinterpret_cast<const ElementSF*>(sfa.data_ptr()), layout_sfa,
       packed.data_ptr<uint8_t>(), row_bytes,
       lut.data_ptr<uint8_t>(), (int32_t)lut.numel(),
       nullptr, 1, 0,
       is_v2 ? compose.data_ptr<uint8_t>() : nullptr,
       (int32_t)k_bits, (int32_t)n_sub, (int32_t)type_size,
       (int32_t)(is_v2 ? 1 : 0),
       expert_ids.data_ptr<int>(), N * row_bytes, (int32_t)packed.size(0),
       nullptr},
      {{{b_scales.data_ptr<float>(), 0.0f, Stride<_0, _1, _0>{}},
        {{a_scales.data_ptr<float>(), 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, typename GemmKernel::StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fused fp4 moe can_implement failed");
  auto init_status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "fused fp4 moe initialize failed: ",
              cutlass::cutlassGetStatusString(init_status),
              "; CUDA status: ", cudaGetErrorString(cudaPeekAtLastError()));
  auto run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "fused fp4 moe launch failed: ",
              cutlass::cutlassGetStatusString(run_status),
              "; CUDA status: ", cudaGetErrorString(cudaPeekAtLastError()));
  return d;
}

torch::Tensor cb_fused_fp4_moe_grouped(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, torch::Tensor expert_ids, int64_t N, int64_t K,
    int64_t k_bits, int64_t n_sub, int64_t type_size, bool is_v2,
    int64_t tile_m) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kUInt8 && a.dim() == 2 &&
              a.stride(1) == 1 && a.size(1) * 2 == K && a.stride(0) == K / 2,
              "a must be contiguous packed-e2m1 uint8 [Mp, K/2]");
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  TORCH_CHECK(N % 8 == 0, "N must be a multiple of 8 (bf16 TMA epilogue)");
  TORCH_CHECK(k_bits >= 9 && k_bits <= 24 && (n_sub == 1 || n_sub == 2));
  TORCH_CHECK(type_size == 4 * k_bits + (is_v2 ? 9 : 16),
              "type_size inconsistent with k_bits/scale coding");
  const int Mp = (int)a.size(0);
  TORCH_CHECK(tile_m == 128 || tile_m == 256, "tile_m must be 128 or 256");
  TORCH_CHECK(Mp % tile_m == 0, "Mp (", Mp, ") must be a multiple of tile_m (",
              tile_m, "); pad each expert's row segment");
  const int64_t n_sb = K / 256;
  const int64_t sfa_need = ((Mp + 127) / 128) * 128 * (K / 16);
  TORCH_CHECK(sfa.is_cuda() && sfa.scalar_type() == torch::kUInt8 &&
              sfa.numel() == sfa_need && sfa.is_contiguous(),
              "sfa must be contiguous uint8 swizzled ue4m3 storage, numel ",
              sfa_need);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
              packed.dim() == 3 && packed.size(0) > 0 && packed.size(1) == N &&
              packed.stride(2) == 1 && packed.stride(1) == packed.size(2) &&
              packed.stride(0) == N * packed.size(2),
              "packed must be fully contiguous uint8 [E, N, row_bytes]");
  TORCH_CHECK(packed.size(2) == n_sb * type_size,
              "row_bytes must equal n_sb*type_size for stacked experts");
  // No tail-slack requirement (gmem gathers stay in-superblock; see the
  // dense entry note), so registered expert stacks are consumed as-is.
  const int64_t w0 = k_bits - k_bits / 2;
  const int64_t lut_need = (n_sub == 2)
      ? ((1LL << w0) + (1LL << (k_bits / 2))) * 2
      : (1LL << (k_bits - 8)) * 4;
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kUInt8 &&
              lut.is_contiguous() && lut.numel() == lut_need &&
              lut_need <= 16384, "value LUT must be uint8[", lut_need, "]");
  TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kUInt8 &&
              compose.is_contiguous(),
              "compose must be contiguous CUDA uint8 storage");
  if (is_v2) {
    TORCH_CHECK(compose.numel() == 4096,
                "v2 compose table must be uint8[4096]");
  }
  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
              a_scales.numel() == Mp && a_scales.is_contiguous());
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
              b_scales.numel() == N && b_scales.is_contiguous());
  TORCH_CHECK(expert_ids.is_cuda() &&
              expert_ids.scalar_type() == torch::kInt32 &&
              expert_ids.is_contiguous() &&
              expert_ids.numel() == Mp / tile_m,
              "expert_ids must be contiguous int32 [Mp/tile_m]");
  check_same_cuda_device(a, sfa, "sfa");
  check_same_cuda_device(a, packed, "packed");
  check_same_cuda_device(a, lut, "lut");
  check_same_cuda_device(a, compose, "compose");
  check_same_cuda_device(a, a_scales, "a_scales");
  check_same_cuda_device(a, b_scales, "b_scales");
  check_same_cuda_device(a, expert_ids, "expert_ids");
  if (tile_m == 256) {
    return run_fp4_moe_grouped_m256(
        a, sfa, packed, lut, compose, a_scales, b_scales, expert_ids, N, K,
        k_bits, n_sub, type_size, is_v2);
  }
  return run_fp4_moe_grouped_m128(
      a, sfa, packed, lut, compose, a_scales, b_scales, expert_ids, N, K,
      k_bits, n_sub, type_size, is_v2);
}

// Every grouped TileM compiled for the fp4 lane (runtime-k => rung-independent,
// unlike the fp8 ladder's per-rung smem feasibility).
std::vector<int64_t> cb_fused_fp4_moe_tile_sizes() { return {128, 256}; }

// SharedStorageSize of [fused, ref] kernels + the sm_120 ceiling (host-only).
std::vector<int64_t> smem_report_fp4() {
  using MF = typename SwapToFusedFp4<2, typename CfgFp4<TileShapeFp4>::BuilderMainloop>::type;
  using KF = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, MF, typename CfgFp4<TileShapeFp4>::Epilogue>;
  using KR = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, typename CfgFp4<TileShapeFp4>::BuilderMainloop,
      typename CfgFp4<TileShapeFp4>::Epilogue>;
  using MF256 = typename SwapToFusedFp4<2, typename CfgFp4<TileShapeFp4M256>::BuilderMainloop>::type;
  using KF256 = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, MF256, typename CfgFp4<TileShapeFp4M256>::Epilogue>;
  return {(int64_t)KF::SharedStorageSize,
          (int64_t)KR::SharedStorageSize,
          (int64_t)KF256::SharedStorageSize,
          (int64_t)cutlass::arch::sm120_smem_capacity_bytes};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_nvfp4_quantize_rows", &cb_nvfp4_quantize_rows,
        py::arg("input"), py::arg("range_multiplier") = 448.0,
        "Row-wise native NVFP4 activation quantization: contiguous BF16/FP16 "
        "[M,K] -> packed E2M1 A, CUTLASS-swizzled UE4M3 SFA, and one FP32 "
        "residual scale per row. Each row is independent of its batch peers.");
  m.def("cb_nvfp4_quantize_rows_out", &cb_nvfp4_quantize_rows_out,
        py::arg("input"), py::arg("packed"), py::arg("sfa"),
        py::arg("a_scales"), py::arg("range_multiplier") = 448.0,
        "Allocation-free row-wise native NVFP4 activation quantization for "
        "CUDA graphs and cached serving workspaces.");
  m.def("sm120_nvf4_mm_scaled", &sm120_nvf4_mm_scaled,
        "STOCK sm120 block-scaled NVF4 GEMM (packed e2m1 A/B + swizzled ue4m3 "
        "SF planes) at the fused kernel's exact TiledMma/tile/epilogue config "
        "— the bit-exactness reference (OMMA.SF.16864).");
  m.def("cb_fused_fp4_prefill_mm_scaled", &cb_fused_fp4_prefill_mm_scaled,
        py::arg("a"), py::arg("sfa"), py::arg("packed"), py::arg("lut"),
        py::arg("compose"), py::arg("a_scales"), py::arg("b_scales"),
        py::arg("N"), py::arg("K"), py::arg("k_bits"), py::arg("n_sub"),
        py::arg("type_size"), py::arg("is_v2"),
        py::arg("lut_tile_ids") = py::none(),
        "NVFP4_CB decode-in-prologue fused BLOCK-SCALED GEMM: packed CB rows "
        "+ smem value/compose LUTs decoded straight into the e2m1/SFB smem "
        "operands of the NVF4 MMA; per-token activation scale x per-channel "
        "scale applied in the fp32 EVT epilogue (cutlass_scaled_mm rounding "
        "order). Runtime k_bits/n_sub/type_size/is_v2 — one kernel for "
        "K12..K24, S13..S16, v1+v2.");
  m.def("cb_fused_fp4_moe_grouped", &cb_fused_fp4_moe_grouped,
        "NVFP4_CB grouped (MoE) fused BLOCK-SCALED GEMM: ONE launch over "
        "row-padded A [Mp,K/2] where each TileM block's B rows are staged "
        "from expert_ids[tile]'s slice of the stacked packed [E,N,row_bytes]; "
        "same decode-in-prologue, same fp32 EVT epilogue. Runtime rung "
        "params; tile_m in {128, 256}.",
        py::arg("a"), py::arg("sfa"), py::arg("packed"), py::arg("lut"),
        py::arg("compose"), py::arg("a_scales"), py::arg("b_scales"),
        py::arg("expert_ids"), py::arg("N"), py::arg("K"), py::arg("k_bits"),
        py::arg("n_sub"), py::arg("type_size"), py::arg("is_v2"),
        py::arg("tile_m") = 128);
  m.def("cb_fused_fp4_moe_tile_sizes", &cb_fused_fp4_moe_tile_sizes,
        "grouped TileM values compiled for the fp4 lane");
  m.def("smem_report_fp4", &smem_report_fp4,
        "[fused SharedStorageSize, ref SharedStorageSize, sm120 capacity]");
}
