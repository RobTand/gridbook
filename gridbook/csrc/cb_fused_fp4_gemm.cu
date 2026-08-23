// NVFP4_CB fused decode-in-prologue BLOCK-SCALED prefill GEMM (fp4-MMA lane).
//
// The fp4 counterpart of cb_fused_gemm.cu: B is the PACKED NVFP4_CB byte
// stream + smem-resident value/compose LUTs; the dense fp4 tile never exists
// in HBM. The MMA is the sm120/sm121 block-scaled NVF4 path
// (SM120_16x8x64_TN_VS = mma.kind::mxf4nvf4.block_scale ue4m3 -> OMMA.SF.16864,
// k=64 — twice the per-instruction K of fp8's QMMA.16832). The trap this file
// exists to avoid: kind::f8f6f4 also ACCEPTS e2m1 operands but issues at the
// fp8 k=32 rate; the fork static_asserts the NVF4 atom, and the SASS gate in
// tests/test_fused_fp4_prefill.py disassembles the built module to prove
// OMMA.SF.16864 is present in BOTH concrete fused symbols (TileM=128 and
// TileM=256) and that QMMA appears in neither — a module-wide search would
// pass on the stock reference kernel's own OMMA. docs/RELEASING.md §2.2 makes
// running that suite on the release GPU image a pre-tag gate.
//
// Entry points:
//  - sm120_nvf4_mm_scaled: STOCK block-scaled collective (dense packed-e2m1
//    B + swizzled SFB from gmem) at the identical TileShape/TiledMma/epilogue
//    config — the bit-exactness REFERENCE for the fused kernel (the fork64
//    role from the fp8 workstream).
//  - cb_fused_fp4_prefill_mm_scaled: the decode-in-prologue NVFP4_CB GEMM.
//    k_bits/n_sub/type_size/scale-coding are RUNTIME parameters — one
//    instantiation serves K12..K24 product, v1 and v2.
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

// Shared glue for the row-padded tile-indexed grouped construction (EVT tree,
// smem gate, host validation) — see cb_grouped_common.hpp. The types this file
// takes from there are proven identical to its former verbatim spellings by
// the static_asserts below, so the generated kernels are unchanged.
#include "cb_grouped_common.hpp"

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

// The EVT node tree (fp32 multiply chain, one round to bf16):
// D = bf16_rn(b_scale[n] * (a_scale[m] * acc)). It was verbatim in three files
// and now lives once in cb_grouped_common.hpp; binding it to THIS file's
// element types reproduces the same types (proven immediately below).
template <class TileShape>
using ScaledFusion = gridbook::grouped::ScaledFusion<TileShape, ElementAcc,
                                                     ElementD>;

// BIT-IDENTITY PROOF for the dedupe extraction in
// docs/audits/ultraplan_perf_2026-08-01.md §4: the shared tree is the SAME
// TYPE this file spelled before it, so the epilogue collective, its smem
// layout and its rounding order are unchanged.
static_assert(
    cute::is_same_v<
        typename ScaledFusion<TileShapeFp4>::type,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiplies, ElementD, ElementAcc,
                cutlass::FloatRoundStyle::round_to_nearest>,
            cutlass::epilogue::fusion::Sm90RowBroadcast<
                0, TileShapeFp4, float, float, Stride<_0, _1, _0>>,
            cutlass::epilogue::fusion::Sm90EVT<
                cutlass::epilogue::fusion::Sm90Compute<
                    cutlass::multiplies, ElementAcc, ElementAcc,
                    cutlass::FloatRoundStyle::round_to_nearest>,
                cutlass::epilogue::fusion::Sm90ColBroadcast<
                    0, TileShapeFp4, float, float, Stride<_1, _0, _0>>,
                cutlass::epilogue::fusion::Sm90AccFetch>>>,
    "shared ScaledFusion must reproduce the pre-extraction EVT node tree");

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

// Hard smem gate (shared — the sm90 cooperative kernel layer does not
// static_assert its own SharedStorageSize).
template <class GemmKernel>
using AssertSmemFits = gridbook::grouped::AssertSmemFits<GemmKernel>;

using gridbook::grouped::check_same_cuda_device;

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
// Native-NVFP4 activation quantization.
//
// vLLM's public scaled_fp4_quant takes one global scale for the complete
// [M,K] tensor.  That makes one row's bytes depend on the other rows present
// in the request.  The GEMMs in this file already consume one residual
// a_scale per M row, so derive a full-range global scale independently for
// each row and preserve the standard UE4M3-group/E2M1 payload contract.  This
// is one activation primitive shared by future dense and grouped dispatch; it
// does not duplicate either weight decoder or GEMM.
//
// The experimental static-LSQ policy deliberately keeps the artifact's fixed
// global scale G, and therefore emits the same E2M1/SFA bytes as vLLM's
// scaled_fp4_quant.  Once those native-QDQ bytes are fixed, it computes the
// least-squares residual independently for every row.  If q_raw is the
// E2M1*UE4M3 value consumed by the MMA, then
//
//   r / G = (x dot (q_raw/G)) / ((q_raw/G) dot (q_raw/G)) / G
//         = (x dot q_raw) / (q_raw dot q_raw).
//
// That final expression is both cheaper and numerically better conditioned.
// Only the existing per-row EVT operand changes; the packed activation bytes,
// weight representation, decoder, and GEMM stay exactly the same.

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

__device__ __forceinline__ float cb_unpack_e2m1(uint32_t packed, int item) {
  uint32_t const code = (packed >> (item * 4)) & 0xfu;
  // E2M1 finite magnitudes in encoding order.  Preserve signed zero only in
  // the serialized nibble; it is immaterial to either dot product.  A switch
  // avoids a dynamically indexed local array in this register-sensitive path.
  float value;
  switch (code & 7u) {
    case 0: value = 0.0f; break;
    case 1: value = 0.5f; break;
    case 2: value = 1.0f; break;
    case 3: value = 1.5f; break;
    case 4: value = 2.0f; break;
    case 5: value = 3.0f; break;
    case 6: value = 4.0f; break;
    default: value = 6.0f; break;
  }
  return (code & 8u) ? -value : value;
}

__device__ __forceinline__ int64_t cb_sfa_swizzle_offset(
    int row, int group, int groups) {
  // K is required to be a multiple of 256, hence groups=K/16 is already a
  // multiple of four and no separate K padding term is needed.
  return (row % 32) * 16 + ((row / 32) % 4) * 4 + (group % 4)
      + (group / 4) * 512
      + (row / 128) * (512 * (groups / 4));
}

enum class CbNvfp4ActivationPolicy : int {
  kRowwiseRange = 0,
  kStaticLsq = 1,
};

template <typename InputT, CbNvfp4ActivationPolicy Policy>
__device__ __forceinline__ void cb_nvfp4_quantize_rows_body(
    InputT const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    float fixed_global_scale,
    int M, int K, float range_multiplier) {
  constexpr bool kStaticLsq = Policy == CbNvfp4ActivationPolicy::kStaticLsq;
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
  __shared__ float warp_reduce_0[8];
  __shared__ float warp_reduce_1[8];
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp = static_cast<int>(threadIdx.x) >> 5;

  float row_max = 0.0f;
  float global_scale = 1.0f;
  if constexpr (kStaticLsq) {
    global_scale = fixed_global_scale;
  } else {
    float local_max = 0.0f;
    for (int col = static_cast<int>(threadIdx.x); col < K;
         col += static_cast<int>(blockDim.x)) {
      local_max = fmaxf(local_max, fabsf(cb_input_to_float(row_input[col])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_max = fmaxf(local_max,
                        __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    if (lane == 0) {
      warp_reduce_0[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
      float block_max = lane < 8 ? warp_reduce_0[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        block_max = fmaxf(
            block_max,
            __shfl_down_sync(0xffffffffu, block_max, offset));
      }
      if (lane == 0) {
        warp_reduce_0[0] = block_max;
      }
    }
    __syncthreads();

    row_max = warp_reduce_0[0];
    if (row_max > 0.0f && isfinite(row_max)) {
      global_scale = __fdiv_rn(range_multiplier * 6.0f, row_max);
    }
    if (threadIdx.x == 0) {
      a_scales[row] = row_max > 0.0f && isfinite(row_max)
          ? __fdiv_rn(1.0f, global_scale)
          : 0.0f;
    }
  }

  uint8_t* row_packed = packed + static_cast<int64_t>(row) * (K / 2);
  float local_xq = 0.0f;
  float local_qq = 0.0f;
  bool valid_quant_scale;
  if constexpr (kStaticLsq) {
    valid_quant_scale = global_scale > 0.0f && isfinite(global_scale);
  } else {
    valid_quant_scale = row_max > 0.0f && isfinite(row_max);
  }
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
    if (group_max > 0.0f && valid_quant_scale) {
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
    uint32_t const packed_low = cb_pack_e2m1x8(low);
    uint32_t const packed_high = cb_pack_e2m1x8(high);
    group_out[0] = packed_low;
    group_out[1] = packed_high;

    if constexpr (kStaticLsq) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float const q_low = cb_unpack_e2m1(packed_low, item) * narrowed_sf;
        float const q_high = cb_unpack_e2m1(packed_high, item) * narrowed_sf;
        local_xq = fmaf(values[item], q_low, local_xq);
        local_qq = fmaf(q_low, q_low, local_qq);
        local_xq = fmaf(values[item + 8], q_high, local_xq);
        local_qq = fmaf(q_high, q_high, local_qq);
      }
    }
  }

  if constexpr (kStaticLsq) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_xq += __shfl_down_sync(0xffffffffu, local_xq, offset);
      local_qq += __shfl_down_sync(0xffffffffu, local_qq, offset);
    }
    if (lane == 0) {
      warp_reduce_0[warp] = local_xq;
      warp_reduce_1[warp] = local_qq;
    }
    __syncthreads();
    if (warp == 0) {
      float block_xq = lane < 8 ? warp_reduce_0[lane] : 0.0f;
      float block_qq = lane < 8 ? warp_reduce_1[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        block_xq += __shfl_down_sync(0xffffffffu, block_xq, offset);
        block_qq += __shfl_down_sync(0xffffffffu, block_qq, offset);
      }
      if (lane == 0) {
        a_scales[row] = (block_qq > 0.0f && isfinite(block_qq)
                         && isfinite(block_xq))
            ? __fdiv_rn(block_xq, block_qq)
            : __fdiv_rn(1.0f, global_scale);
      }
    }
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
  cb_nvfp4_quantize_rows_body<
      __nv_bfloat16, CbNvfp4ActivationPolicy::kRowwiseRange>(
      input, packed, sfa, a_scales, 0.0f, M, K, range_multiplier);
}

__global__ void cb_nvfp4_quantize_rows_fp16_kernel(
    half const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    int M, int K, float range_multiplier) {
  cb_nvfp4_quantize_rows_body<
      half, CbNvfp4ActivationPolicy::kRowwiseRange>(
      input, packed, sfa, a_scales, 0.0f, M, K, range_multiplier);
}

__global__ void cb_nvfp4_quantize_static_lsq_bf16_kernel(
    __nv_bfloat16 const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    float global_scale,
    int M, int K) {
  cb_nvfp4_quantize_rows_body<
      __nv_bfloat16, CbNvfp4ActivationPolicy::kStaticLsq>(
      input, packed, sfa, a_scales, global_scale, M, K, 0.0f);
}

__global__ void cb_nvfp4_quantize_static_lsq_fp16_kernel(
    half const* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    float* __restrict__ a_scales,
    float global_scale,
    int M, int K) {
  cb_nvfp4_quantize_rows_body<
      half, CbNvfp4ActivationPolicy::kStaticLsq>(
      input, packed, sfa, a_scales, global_scale, M, K, 0.0f);
}

namespace {

void cb_nvfp4_quantize_out_impl(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor sfa,
    torch::Tensor a_scales,
    double range_multiplier,
    std::optional<float> fixed_global_scale) {
  bool const static_lsq = fixed_global_scale.has_value();
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
  if (!static_lsq) {
    TORCH_CHECK(std::isfinite(range_multiplier) && range_multiplier > 0.0 &&
                range_multiplier <= 448.0,
                "range_multiplier must be finite and in (0,448]");
  }

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
    if (static_lsq) {
      ::cb_nvfp4_quantize_static_lsq_bf16_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
          packed.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
          a_scales.data_ptr<float>(), *fixed_global_scale, M, K);
    } else {
      ::cb_nvfp4_quantize_rows_bf16_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
          packed.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
          a_scales.data_ptr<float>(), M, K,
          static_cast<float>(range_multiplier));
    }
  } else {
    if (static_lsq) {
      ::cb_nvfp4_quantize_static_lsq_fp16_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<half const*>(input.data_ptr()),
          packed.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
          a_scales.data_ptr<float>(), *fixed_global_scale, M, K);
    } else {
      ::cb_nvfp4_quantize_rows_fp16_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<half const*>(input.data_ptr()),
          packed.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
          a_scales.data_ptr<float>(), M, K,
          static_cast<float>(range_multiplier));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cb_nvfp4_quantize_rows_out(torch::Tensor input,
                                torch::Tensor packed,
                                torch::Tensor sfa,
                                torch::Tensor a_scales,
                                double range_multiplier) {
  cb_nvfp4_quantize_out_impl(
      input, packed, sfa, a_scales, range_multiplier, std::nullopt);
}

void cb_nvfp4_quantize_static_lsq_out(torch::Tensor input,
                                      double global_scale,
                                      torch::Tensor packed,
                                      torch::Tensor sfa,
                                      torch::Tensor a_scales) {
  float const global_scale_f32 = static_cast<float>(global_scale);
  TORCH_CHECK(std::isfinite(global_scale) && std::isfinite(global_scale_f32) &&
              global_scale_f32 > 0.0f,
              "global_scale must round to a finite positive float32 value");
  cb_nvfp4_quantize_out_impl(
      input, packed, sfa, a_scales, 0.0, global_scale_f32);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
allocate_nvfp4_quant_outputs(torch::Tensor input) {
  TORCH_CHECK(input.dim() == 2, "input must be rank 2");
  int64_t const M = input.size(0);
  int64_t const K = input.size(1);
  int64_t const padded_m = ((M + 127) / 128) * 128;
  auto packed = torch::empty({M, K / 2},
      input.options().dtype(torch::kUInt8));
  auto sfa = torch::empty({padded_m * (K / 16)},
      input.options().dtype(torch::kUInt8));
  auto a_scales = torch::empty({M}, input.options().dtype(torch::kFloat32));
  return {packed, sfa, a_scales};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cb_nvfp4_quantize_rows(torch::Tensor input, double range_multiplier) {
  auto [packed, sfa, a_scales] = allocate_nvfp4_quant_outputs(input);
  cb_nvfp4_quantize_rows_out(
      input, packed, sfa, a_scales, range_multiplier);
  return {packed, sfa, a_scales};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cb_nvfp4_quantize_static_lsq(torch::Tensor input,
                             double global_scale) {
  auto [packed, sfa, a_scales] = allocate_nvfp4_quant_outputs(input);
  cb_nvfp4_quantize_static_lsq_out(
      input, global_scale, packed, sfa, a_scales);
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

// Dense TileM=256 reuses the concrete kernel runner compiled for grouped
// serving.  Keep this a non-template declaration: nvcc 13 otherwise preserves
// unsigned layout aliases in the host pass and emits an unregistered duplicate
// device stub (the reason the two concrete runners below exist).
torch::Tensor run_fp4_fused_m256(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, int32_t const* lut_tile_ids,
    int32_t num_lut_blocks, int32_t num_lut_tiles,
    int const* expert_ids, int64_t N, int64_t K, int64_t k_bits,
    int64_t n_sub, int64_t type_size, bool is_v2,
    int64_t packed_row_bytes, int64_t packed_expert_stride,
    int32_t num_experts);

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
    std::optional<torch::Tensor> lut_tile_ids, int64_t tile_m) {
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
  TORCH_CHECK(tile_m == 128 || tile_m == 256,
              "dense fused fp4 tile_m must be 128 or 256");
  TORCH_CHECK(k_bits >= 9 && k_bits <= 24, "fp4 rung k_bits out of range");
  TORCH_CHECK(n_sub == 2,
              "fp4 n_sub must be 2 (product); the signed n_sub=1 family was "
              "removed from the runtime");
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
  // Value LUT size must match the rung exactly: two ceil-first sub-tables of
  // u16 nibble-quads (product).
  const int64_t w0 = k_bits - k_bits / 2;
  const int64_t lut_need = ((1LL << w0) + (1LL << (k_bits / 2))) * 2;
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

  if (tile_m == 256) {
    return run_fp4_fused_m256(
        a, sfa, packed, lut, compose, a_scales, b_scales,
        lut_tile_ids_ptr, num_lut_blocks, num_lut_tiles,
        nullptr, N, K, k_bits, n_sub, type_size, is_v2,
        packed.stride(0), 0, 0);
  }

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

torch::Tensor run_fp4_fused_m256(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, int32_t const* lut_tile_ids,
    int32_t num_lut_blocks, int32_t num_lut_tiles,
    int const* expert_ids, int64_t N, int64_t K, int64_t k_bits,
    int64_t n_sub, int64_t type_size, bool is_v2,
    int64_t packed_row_bytes, int64_t packed_expert_stride,
    int32_t num_experts) {
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
       packed.data_ptr<uint8_t>(), packed_row_bytes,
       lut.data_ptr<uint8_t>(), (int32_t)(lut.numel() / num_lut_blocks),
       lut_tile_ids, num_lut_blocks, num_lut_tiles,
       is_v2 ? compose.data_ptr<uint8_t>() : nullptr,
       (int32_t)k_bits, (int32_t)n_sub, (int32_t)type_size,
       (int32_t)(is_v2 ? 1 : 0),
       expert_ids, packed_expert_stride, num_experts,
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
  TORCH_CHECK(k_bits >= 9 && k_bits <= 24 && n_sub == 2);
  TORCH_CHECK(type_size == 4 * k_bits + (is_v2 ? 9 : 16),
              "type_size inconsistent with k_bits/scale coding");
  const int Mp = (int)a.size(0);
  TORCH_CHECK(tile_m == 128 || tile_m == 256, "tile_m must be 128 or 256");
  // Padding granularity, stacked-expert contiguity and the per-tile expert id
  // vector belong to the SHARED grouping construction (cb_grouped_common.hpp).
  gridbook::grouped::check_padded_rows(Mp, tile_m);
  const int64_t n_sb = K / 256;
  const int64_t sfa_need = ((Mp + 127) / 128) * 128 * (K / 16);
  TORCH_CHECK(sfa.is_cuda() && sfa.scalar_type() == torch::kUInt8 &&
              sfa.numel() == sfa_need && sfa.is_contiguous(),
              "sfa must be contiguous uint8 swizzled ue4m3 storage, numel ",
              sfa_need);
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8,
              "packed must be fully contiguous uint8 [E, N, row_bytes]");
  gridbook::grouped::check_stacked_experts(packed, N, "packed");
  TORCH_CHECK(packed.size(2) == n_sb * type_size,
              "row_bytes must equal n_sb*type_size for stacked experts");
  // No tail-slack requirement (gmem gathers stay in-superblock; see the
  // dense entry note), so registered expert stacks are consumed as-is.
  const int64_t w0 = k_bits - k_bits / 2;
  const int64_t lut_need = ((1LL << w0) + (1LL << (k_bits / 2))) * 2;
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
  gridbook::grouped::check_expert_ids(a, expert_ids, Mp, tile_m,
                                      packed.size(0));
  check_same_cuda_device(a, sfa, "sfa");
  check_same_cuda_device(a, packed, "packed");
  check_same_cuda_device(a, lut, "lut");
  check_same_cuda_device(a, compose, "compose");
  check_same_cuda_device(a, a_scales, "a_scales");
  check_same_cuda_device(a, b_scales, "b_scales");
  // (expert_ids' device is attested inside check_expert_ids above.)
  if (tile_m == 256) {
    return run_fp4_fused_m256(
        a, sfa, packed, lut, compose, a_scales, b_scales,
        nullptr, 1, 0, expert_ids.data_ptr<int>(), N, K,
        k_bits, n_sub, type_size, is_v2, packed.size(2),
        N * packed.size(2), (int32_t)packed.size(0));
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
  m.def("cb_nvfp4_quantize_static_lsq", &cb_nvfp4_quantize_static_lsq,
        py::arg("input"), py::arg("global_scale"),
        "Experimental fixed-G native NVFP4 activation quantization with a "
        "per-row least-squares EVT residual. Packed E2M1 and SFA bytes match "
        "scaled_fp4_quant(input, global_scale); no alternate weight or GEMM "
        "representation is created.");
  m.def("cb_nvfp4_quantize_static_lsq_out",
        &cb_nvfp4_quantize_static_lsq_out,
        py::arg("input"), py::arg("global_scale"), py::arg("packed"),
        py::arg("sfa"), py::arg("a_scales"),
        "Allocation-free fixed-G NVFP4 quantization plus per-row LSQ residual "
        "for non-default streams and CUDA graphs.");
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
        py::arg("tile_m") = 128,
        "NVFP4_CB decode-in-prologue fused BLOCK-SCALED GEMM: packed CB rows "
        "+ smem value/compose LUTs decoded straight into the e2m1/SFB smem "
        "operands of the NVF4 MMA; per-token activation scale x per-channel "
        "scale applied in the fp32 EVT epilogue (cutlass_scaled_mm rounding "
        "order). Runtime k_bits/n_sub/type_size/is_v2 and dense tile_m in "
        "{128,256} — one kernel for K12..K24, S13..S16, v1+v2.");
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
