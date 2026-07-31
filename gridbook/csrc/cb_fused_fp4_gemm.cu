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
  TORCH_CHECK(sfb.is_cuda() && sfb.numel() == sfb_need && sfb.is_contiguous(),
              "sfb must be the swizzled ue4m3 plane, numel ", sfb_need);
  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
              a_scales.numel() == M && a_scales.is_contiguous());
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
              b_scales.numel() == N && b_scales.is_contiguous());

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
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
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
    c10::optional<torch::Tensor> debug_out = c10::nullopt) {
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
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  TORCH_CHECK(N % 8 == 0, "N must be a multiple of 8 (bf16 TMA epilogue "
              "alignment; every exported CB Linear satisfies this)");
  TORCH_CHECK(k_bits >= 9 && k_bits <= 24, "fp4 rung k_bits out of range");
  TORCH_CHECK(n_sub == 1 || n_sub == 2, "fp4 n_sub must be 1 (signed) or 2");
  TORCH_CHECK(type_size == 4 * k_bits + (is_v2 ? 9 : 16),
              "type_size inconsistent with k_bits/scale coding");
  const int M = (int)a.size(0);
  const int64_t n_sb = K / 256;
  const int64_t sfa_need = ((M + 127) / 128) * 128 * (K / 16);
  TORCH_CHECK(sfa.is_cuda() && sfa.numel() == sfa_need && sfa.is_contiguous(),
              "sfa must be the swizzled ue4m3 plane, numel ", sfa_need);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
              packed.dim() == 2 && packed.size(0) == N && packed.stride(1) == 1,
              "packed must be uint8 [N, row_bytes(+pad)]");
  TORCH_CHECK(packed.stride(0) >= n_sb * type_size,
              "packed row stride too small for K/type_size");
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
              lut.is_contiguous() && lut.numel() == lut_need,
              "value LUT must be uint8[", lut_need, "] for this rung");
  TORCH_CHECK(lut_need <= 16384, "value LUT exceeds the smem carve");
  if (is_v2) {
    TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kUInt8 &&
                compose.is_contiguous() && compose.numel() == 4096,
                "v2 compose table must be uint8[4096] (256x16 e4m3 bytes)");
  }
  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
              a_scales.numel() == M && a_scales.is_contiguous());
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
              b_scales.numel() == N && b_scales.is_contiguous());

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
       lut.data_ptr<uint8_t>(), (int32_t)lut.numel(),
       is_v2 ? compose.data_ptr<uint8_t>() : nullptr,
       (int32_t)k_bits, (int32_t)n_sub, (int32_t)type_size,
       (int32_t)(is_v2 ? 1 : 0), nullptr, 0,
       debug_out.has_value() ? debug_out->data_ptr<uint8_t>() : nullptr},
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
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
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
using TileShapeFp4M256 = Shape<_256, _128, _128>;

template <class TileShape>
torch::Tensor run_fp4_moe_grouped(torch::Tensor a, torch::Tensor sfa,
                                  torch::Tensor packed, torch::Tensor lut,
                                  torch::Tensor compose,
                                  torch::Tensor a_scales,
                                  torch::Tensor b_scales,
                                  torch::Tensor expert_ids, int64_t N,
                                  int64_t K, int64_t k_bits, int64_t n_sub,
                                  int64_t type_size, bool is_v2) {
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
       is_v2 ? compose.data_ptr<uint8_t>() : nullptr,
       (int32_t)k_bits, (int32_t)n_sub, (int32_t)type_size,
       (int32_t)(is_v2 ? 1 : 0),
       expert_ids.data_ptr<int>(), N * row_bytes, nullptr},
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
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
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
  TORCH_CHECK(sfa.is_cuda() && sfa.numel() == sfa_need && sfa.is_contiguous(),
              "sfa must be the swizzled ue4m3 plane, numel ", sfa_need);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
              packed.dim() == 3 && packed.size(1) == N &&
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
  if (is_v2) {
    TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kUInt8 &&
                compose.is_contiguous() && compose.numel() == 4096);
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
  if (tile_m == 256) {
    return run_fp4_moe_grouped<TileShapeFp4M256>(
        a, sfa, packed, lut, compose, a_scales, b_scales, expert_ids, N, K,
        k_bits, n_sub, type_size, is_v2);
  }
  return run_fp4_moe_grouped<TileShapeFp4>(
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
  m.def("sm120_nvf4_mm_scaled", &sm120_nvf4_mm_scaled,
        "STOCK sm120 block-scaled NVF4 GEMM (packed e2m1 A/B + swizzled ue4m3 "
        "SF planes) at the fused kernel's exact TiledMma/tile/epilogue config "
        "— the bit-exactness reference (OMMA.SF.16864).");
  m.def("cb_fused_fp4_prefill_mm_scaled", &cb_fused_fp4_prefill_mm_scaled,
        py::arg("a"), py::arg("sfa"), py::arg("packed"), py::arg("lut"),
        py::arg("compose"), py::arg("a_scales"), py::arg("b_scales"),
        py::arg("N"), py::arg("K"), py::arg("k_bits"), py::arg("n_sub"),
        py::arg("type_size"), py::arg("is_v2"),
        py::arg("debug_out") = py::none(),
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
