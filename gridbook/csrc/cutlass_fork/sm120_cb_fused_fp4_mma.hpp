/***************************************************************************************************
 * PrismaQuant NVFP4_CB fused decode-in-prologue BLOCK-SCALED mainloop for
 * sm120/sm121 (GB10).
 *
 * Derived from the vendored CUTLASS 4.3.4 sm120_blockscaled_mma_tma.hpp
 * (pristine copy: sm120_blockscaled_mma_tma_orig.hpp), the same way
 * sm120_cb_fused_mma.hpp forked sm120_mma_tma.hpp for FP8_CB. The A side
 * (packed e2m1 activations + swizzled ue4m3 SFA, both TMA-pipelined) is kept
 * verbatim; the ENTIRE B side is replaced:
 *
 *   - gmem B is the PACKED NVFP4_CB byte stream ([N_rows, n_sb*type_size]
 *     uint8). fp4 rows have an ODD type_size (4k+9 two-tier v2 / 4k+16 v1),
 *     so the row stride is NOT a 16-byte multiple and TMA is structurally
 *     unusable for it (16B global-stride rule). Instead the producer
 *     WARP (32 threads) manually stages, per K-tile, each row's half-
 *     superblock index window (aligned-u32 window loads over the misaligned
 *     source) plus its 16 scale-plane bytes into a fixed 72 B/row smem
 *     stage. Ordering into the TMA mbarrier: plain stores -> __syncwarp ->
 *     leader's mbarrier.arrive(expect_tx) (release.cta) issued by the A/SFA
 *     TMA copies -> consumer_wait (acquire.cta). No async-proxy fence is
 *     needed: writer and reader are both generic proxy.
 *   - After consumer_wait, the 256 MMA threads cooperatively decode the
 *     staged bytes into (a) the standard swizzled e2m1 SmemLayoutB buffer
 *     (packed nibbles — the codebook is E2M1-grid-valued by construction,
 *     nvfp4_cb_formats._snap_to_grid) and (b) the standard SmemLayoutSFB
 *     ue4m3 buffer (two-tier v2 compose is EXACT e4m3 by construction,
 *     two-tier-scale-spec.md §1.2; v1 plane bytes are already e4m3).
 *     Single decoded buffer + NamedBarrier serialization per K-tile — the
 *     sm120_cb_fused_mma.hpp pattern verbatim.
 *   - The MMA is the UNCHANGED block-scaled path: zipped (B,SFB) fragments
 *     into SM120_16x8x64_TN_VS = mma.kind::mxf4nvf4.block_scale ue4m3
 *     (OMMA.SF.16864, k=64 — the true fp4 rate). Deliberately NOT
 *     kind::f8f6f4, which accepts e2m1 but issues at the fp8 k=32 rate.
 *
 * Unlike the fp8 fork, k_bits/n_sub/type_size are RUNTIME parameters: the
 * packed stream never touches a TMA descriptor or a k-sized smem layout here,
 * so one instantiation serves the whole rung ladder (K12..K24 product,
 * S13..S16 signed, v1 + v2 scale coding). The value LUT (max 16 KiB at k24)
 * and the v2 compose table (4 KiB) are smem-resident (R6), staged once per
 * CTA.
 *
 * INV-1: the dense fp4 tile exists only as one [TileN, TileK] nibble smem
 * buffer + one SFB smem plane. Weight-side numerics are EXACT (e2m1 values,
 * e4m3 scales — both representable losslessly in the MMA operands); the
 * activation side is native NVFP4 quantization (per-tensor fp32 global scale
 * x per-group-16 ue4m3 SF), which is a DIFFERENT activation bucket from the
 * Triton path's fp32-scale group-16 QDQ — the hardware SF operand is ue4m3,
 * so an fp32 group scale is unrepresentable. See the parity notes in
 * docs/lanes/nvfp4-cb/fp4-fused-prefill.md.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

namespace cutlass::gemm {

template <
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120CbFusedFp4TmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

}  // namespace cutlass::gemm

namespace cutlass::gemm::collective {
using namespace cute;

template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120CbFusedFp4TmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {
  using DispatchPolicy = MainloopSm120CbFusedFp4TmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;

  using TiledMma = TiledMma_;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;

  static constexpr int SFVecSize = TiledMma::Traits::SFVecSize;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyA    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;

  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;
  using SmemLayoutAtomA   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomB   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsB{}))>;

  using SmemCopyAtomsA = SmemCopyAtomsA_;
  using SmemCopyAtomsB = SmemCopyAtomsB_;
  using SmemCopyAtomA   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomSFA = remove_cvref_t<decltype(get<1>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomB   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsB{}))>;
  using SmemCopyAtomSFB = remove_cvref_t<decltype(get<1>(SmemCopyAtomsB{}))>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  static constexpr int NumProducerThreadEvents = 1;

  // --- CB fp4 geometry (compile-time) -------------------------------------
  static constexpr int TileM = size<0>(TileShape{});
  static constexpr int TileN = size<1>(TileShape{});
  static constexpr int TileK = size<2>(TileShape{});
  static_assert(TileK == 128, "CB fused fp4 mainloop assumes TileK = 128 (half a 256-weight superblock)");
  static_assert(TileN == 128, "blockscaled SF smem atoms require TileN == Blk_MN == 128");

  // Per-stage packed-B DESCRIPTOR (attempt-2 of the staging design, measured
  // — see fp4-fused-prefill.md §7): the producer warp does NOT stage packed
  // bytes (a single 32-thread warp staging ~9 KB/K-tile was the pipeline
  // bottleneck: ~13x off native at one-M-tile shapes). It publishes 16 bytes
  // per stage — the M-tile's packed base pointer (expert-resolved) and the
  // CTA's first output row — and the 256 MMA threads gather the packed
  // bytes STRAIGHT FROM GMEM during decode (L1/L2-hot; the decode-GEMV
  // pattern). All gmem windows stay inside the row's own superblock (u8
  // loads for the scale plane), so NO tail slack is required of the buffer.
  static constexpr int CbStageDescBytes = 16;

  // Fixed smem LUT carve (R6): value LUT sized for the LARGEST fp4 rung
  // (product k24: two 2^12-entry sub-tables x 2 B = 16 KiB; every signed rung
  // S13..S16 is <= 1 KiB) + the two-tier compose table (256 x 16 e4m3 bytes).
  // Runtime lut_bytes selects how much is staged; the carve is constant so
  // ONE instantiation serves the whole ladder.
  static constexpr int CbLutMaxBytes = 16384;
  static constexpr int CbComposeBytes = 4096;

  // A/SFA: unchanged pipelined layouts (pristine).
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutSFA = decltype(make_layout(
      append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
      append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));

  // B decoded: ONE buffer in the standard swizzled layout (the MMA-side
  // contract), exactly like the fp8 fork. SFB decoded: ONE buffer in the
  // standard blockscaled SF layout.
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<1>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutSFB = decltype(make_layout(
      append(shape(SmemLayoutAtomSFB{}), Int<1>{}),
      append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "CB fused fp4 mainloop: A uses plain SM90_TMA_LOAD.");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();
  static_assert(!IsF8F6F4 && cute::is_same_v<ElementB, cutlass::float_e2m1_t> &&
                cute::is_same_v<ElementSF, cutlass::float_ue4m3_t> && SFVecSize == 16,
                "CB fused fp4 mainloop requires the NVF4 block-scaled MMA "
                "(e2m1 x e2m1, ue4m3 SF, vec16 = kind::mxf4nvf4 k=64). The "
                "f8f6f4 path would run e2m1 at the fp8 k=32 rate.");

  using TmaInternalElementA = ElementA;          // packed e2m1 TMA (non-f8f6f4)
  using SmemAllocTypeA = typename TiledMma::ValTypeA;
  using SmemAllocTypeB = typename TiledMma::ValTypeB;

  // TMA transaction: A + SFA ONLY (B is producer-staged manually, SFB is
  // decoded in-prologue — neither goes through the TMA barrier's byte count).
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = 0;
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(16) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(16) cute::array_aligned<uint8_t, CbStageDescBytes * DispatchPolicy::Stages> smem_BP;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(16) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
      alignas(16) cute::array_aligned<uint8_t, CbLutMaxBytes> smem_lut;
      alignas(16) cute::array_aligned<uint8_t, CbComposeBytes> smem_compose;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    // --- packed CB weight stream (replaces ptr_B/dB/ptr_SFB) --------------
    uint8_t const* ptr_packed{nullptr};   // [N_rows, >= n_sb*type_size] (+>=8B row slack)
    int64_t packed_row_bytes{0};          // explicit row stride (any parity)
    uint8_t const* ptr_lut{nullptr};      // value LUT: product -> u16 nibble
                                          // quads (tbl0 then tbl1); signed ->
                                          // u32 nibble octets (magnitudes)
    int32_t lut_bytes{0};                 // <= CbLutMaxBytes
    uint8_t const* ptr_compose{nullptr};  // v2: (256*16) e4m3 bytes; else null
    int32_t k_bits{0};
    int32_t n_sub{2};                     // 2 = product, 1 = signed
    int32_t type_size{0};                 // 4k+9 (v2) or 4k+16 (v1)
    int32_t is_v2{1};
    // --- OPTIONAL tile-indexed grouping (MoE), the fp8 fork's mechanism:
    // ptr_expert_ids[m_tile] selects which expert of a stacked
    // [E, N, row_bytes] packed buffer this M-tile's B rows come from; the
    // indirection lives ENTIRELY in the producer staging. nullptr => dense.
    int const* ptr_expert_ids{nullptr};
    int64_t packed_expert_stride{0};      // = N * row_bytes for stacked packs
    // DEBUG ONLY (tests): when non-null, the CTA at (m=0, n=0) dumps its
    // first-K-tile decoded smem_B bytes, smem_SFB bytes, staged packed rows
    // and the smem LUT here after the first decode barrier. Never set in
    // serving dispatch.
    uint8_t* ptr_debug{nullptr};
  };

  // Device side kernel params
  struct Params {
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    using TMA_SFA = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    TMA_A tma_load_a;
    TMA_SFA tma_load_sfa;
    LayoutSFA layout_SFA;
    uint8_t const* ptr_packed;
    int64_t packed_row_bytes;
    uint8_t const* ptr_lut;
    uint8_t const* ptr_compose;
    int32_t lut_bytes;
    int32_t k_bits;
    int32_t n_sub;
    int32_t type_size;
    int32_t is_v2;
    int32_t N_rows;
    int const* ptr_expert_ids;
    int64_t packed_expert_stride;
    uint8_t* ptr_debug;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{});

    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    typename Params::TMA_SFA tma_load_sfa = make_tma_copy<uint16_t>(
        GmemTiledCopySFA{}, tensor_sfa, SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{});

    return {
      tma_load_a, tma_load_sfa, args.layout_SFA,
      args.ptr_packed, args.packed_row_bytes,
      args.ptr_lut, args.ptr_compose, args.lut_bytes,
      args.k_bits, args.n_sub, args.type_size, args.is_v2,
      static_cast<int32_t>(N), args.ptr_expert_ids,
      args.packed_expert_stride, args.ptr_debug,
      TmaTransactionBytes, TmaTransactionBytesMK, TmaTransactionBytesNK
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    bool implementable = cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    implementable = implementable && (K % 256 == 0) && (L == 1);
    implementable = implementable && (args.k_bits >= 9) && (args.k_bits <= 24);
    implementable = implementable && (args.n_sub == 1 || args.n_sub == 2);
    implementable = implementable && (args.type_size == 4 * args.k_bits + (args.is_v2 ? 9 : 16));
    implementable = implementable && (args.packed_row_bytes >= (K / 256) * args.type_size);
    implementable = implementable && (args.lut_bytes > 0) && (args.lut_bytes <= CbLutMaxBytes);
    implementable = implementable && (!args.is_v2 || args.ptr_compose != nullptr);
    return implementable;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
  }

  // ---- SF partitioning helpers: pristine copies (unchanged) ---------------
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr auto
  thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr auto
  thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);
    auto b_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto b_tensor = zipped_divide(t_tensor, b_tile);
    auto tv_tensor = b_tensor.compose(AtomLayoutSFB_TV{},_);
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr auto
  partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr auto
  partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template<class TiledMma_Arg>
  CUTE_HOST_DEVICE constexpr auto
  get_layoutSFA_TV(TiledMma_Arg& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto atile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<1>{} ,                Int<0>{} )),
                                    _));
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template<class TiledMma_Arg>
  CUTE_HOST_DEVICE constexpr auto
  get_layoutSFB_TV(TiledMma_Arg& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto btile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<0>{} ,                Int<1>{} )),
                                    _));
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  /// load_init: A + SFA only (B is decoded from the producer-staged bytes).
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M,K,L));
    Tensor mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(params.layout_SFA));
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    // gB_nkl stand-in so the kernel layer's idx2crd(N_idx, shape<2>(...))
    // keeps working: the kernel layer only reads shapes from element 1.
    Tensor gB_nkl = local_tile(
        make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)),
                    make_layout(make_shape(N, K, L))),
        TileShape{}, make_coord(_,_,_), Step<X, _1,_1>{});
    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl);
  }

  /// Producer: TMA A + SFA, manual staging of the packed CB byte stream.
  /// Called by the WHOLE producer warp (32 threads).
  template <class TensorA, class TensorB, class TensorSFA, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFA> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    (void)block_rank_in_cluster;
    const int lane = thread_idx & 31;
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gSFA_mkl = get<2>(load_inputs);

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);
    Tensor gSFA = gSFA_mkl(_,_,m_coord,_,l_coord);

    Tensor tAgA = block_tma_a.partition_S(gA);
    Tensor tAsA = block_tma_a.partition_D(sA);
    Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
    Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);

    // Publish the packed-B descriptor for this work tile: base pointer
    // (expert-resolved for the grouped/MoE path — padding tiles' -1 clamps
    // to 0; their outputs are discarded by the caller) + the CTA's first
    // output row. The consumers gather packed bytes from gmem directly.
    const int n_base = int(n_coord) * TileN;
    const uint8_t* __restrict__ gp = params.ptr_packed;
    if (params.ptr_expert_ids != nullptr) {
      int eid = __ldg(params.ptr_expert_ids + int(m_coord));
      gp += int64_t(eid < 0 ? 0 : eid) * params.packed_expert_stride;
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      if (lane_predicate) {
        pipeline.producer_acquire(smem_pipe_write);
      }
      __syncwarp();

      int write_stage = smem_pipe_write.index();
      const int kt = int(*k_tile_iter);
      if (lane == 0) {
        uint8_t* slot = shared_tensors.smem_BP.data()
            + write_stage * CbStageDescBytes;
        *reinterpret_cast<uint64_t*>(slot) =
            reinterpret_cast<uint64_t>(gp);
        *reinterpret_cast<int32_t*>(slot + 8) = n_base;
      }
      __syncwarp();

      if (lane_predicate) {
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        // The leader's mbarrier arrive (inside .with()) carries release.cta
        // semantics and is ordered AFTER the __syncwarp above, so the manual
        // stores are visible to consumers once the barrier phase flips.
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,kt), tAsA(_,_,_,write_stage));
        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_,_,_,kt), tAsSFA(_,_,_,write_stage));
      }
      ++k_tile_iter;
      ++smem_pipe_write;
    }
    __syncwarp();
  }

  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // --- once-per-CTA LUT staging (R6) --------------------------------------
  CUTLASS_DEVICE void
  load_lut(TensorStorage& shared_tensors, Params const& params, int thread_idx) const {
    {
      uint32_t* dst = reinterpret_cast<uint32_t*>(shared_tensors.smem_lut.data());
      const uint32_t* src = reinterpret_cast<const uint32_t*>(params.ptr_lut);
      const int nwords = (params.lut_bytes + 3) >> 2;
      for (int w = thread_idx; w < nwords; w += ThreadCount) {
        dst[w] = __ldg(src + w);
      }
    }
    if (params.is_v2) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(shared_tensors.smem_compose.data());
      const uint32_t* src = reinterpret_cast<const uint32_t*>(params.ptr_compose);
      constexpr int NWords = CbComposeBytes / 4;
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < NWords / ThreadCount; ++w) {
        dst[thread_idx + w * ThreadCount] = __ldg(src + thread_idx + w * ThreadCount);
      }
    }
  }

  // --- the decode: staged packed bytes -> swizzled e2m1 B + SFB plane ------
  // Bit-exact to the Triton/CUDA decode kernels' codeword extraction and to
  // nvfp4_cb_reconstruct's value/scale composition; the difference is only
  // the OUTPUT encoding (e2m1 nibbles + ue4m3 SF instead of bf16 products),
  // which is lossless for this format by construction.
  static constexpr int CwPerTile = TileN * 16;             // codewords / K-tile
  static constexpr int CwPerThread = CwPerTile / ThreadCount;
  static_assert(CwPerTile % ThreadCount == 0, "codeword count must divide thread count");
  static constexpr int SfPerTile = TileN * (TileK / 16);   // SF bytes / K-tile
  static constexpr int SfPerThread = SfPerTile / ThreadCount;
  static_assert(SfPerTile % ThreadCount == 0, "SF count must divide thread count");

  template <class SBNoSw, class SBSw, class SFBLayout1>
  CUTLASS_DEVICE void
  decode_stage(TensorStorage& shared_tensors, uint8_t* sB_base,
               SBNoSw const& sB_nosw, SBSw const& sB_sw,
               SFBLayout1 const& sfb_layout, Params const& params,
               int read_stage, int sb, int half, int thread_idx) const {
    // Packed-B descriptor for this stage (published by the producer under
    // the stage's mbarrier): expert-resolved base pointer + first row.
    const uint8_t* slot = shared_tensors.smem_BP.data()
        + read_stage * CbStageDescBytes;
    const uint8_t* __restrict__ gp = reinterpret_cast<const uint8_t*>(
        *reinterpret_cast<const uint64_t*>(slot));
    const int n_base = *reinterpret_cast<const int32_t*>(slot + 8);

    const int kb = params.k_bits;
    const int ts = params.type_size;
    const int64_t row_stride = params.packed_row_bytes;
    const int N_rows = params.N_rows;
    const uint32_t mask_k = (uint32_t)((1ull << kb) - 1ull);
    const bool signed_mode = (params.n_sub == 1);
    const int w0 = kb - (kb >> 1);                   // ceil-first split
    const uint32_t mask0 = (1u << w0) - 1u;
    const uint32_t mask1 = (1u << (kb >> 1)) - 1u;
    const uint16_t* lut16 = reinterpret_cast<const uint16_t*>(shared_tensors.smem_lut.data());
    const uint32_t* lut32 = reinterpret_cast<const uint32_t*>(shared_tensors.smem_lut.data());
    const int64_t sb_off = int64_t(sb) * ts;
    const int idx_off = half * 2 * kb;               // half-superblock base

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < CwPerThread; ++i) {
      const int linear = thread_idx + i * ThreadCount;
      const int r = linear & (TileN - 1);
      const int vl = linear / TileN;                 // 0..15 in this K-tile
      int n_glob = n_base + r;
      n_glob = n_glob < N_rows ? n_glob : (N_rows - 1);
      const uint8_t* row_g = gp + int64_t(n_glob) * row_stride + sb_off;
      // Aligned-u32 window straight from gmem: the window stays inside this
      // row's superblock for every rung (max end byte 2k + (15k>>3 & ~3) + 7
      // <= ts-1), so no buffer tail slack is ever needed.
      const int bitpos = vl * kb;
      const uint8_t* pbyte = row_g + idx_off + (bitpos >> 3);
      const uintptr_t pa = reinterpret_cast<uintptr_t>(pbyte);
      const uint32_t* al = reinterpret_cast<const uint32_t*>(pa & ~uintptr_t(3));
      const int rem = int((pa & 3) * 8) + (bitpos & 7);
      const uint64_t w64 = (uint64_t(__ldg(al + 1)) << 32) | __ldg(al);
      const uint32_t code = uint32_t(w64 >> rem) & mask_k;

      uint32_t out32;
      if (signed_mode) {
        // 8 LSB sign bits + magnitude index above them; positive-half-grid
        // nibbles get their e2m1 sign bit from the code's sign byte.
        out32 = lut32[code >> 8];
        const uint32_t signs = code & 0xFFu;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 8; ++j) {
          out32 ^= ((signs >> j) & 1u) << (4 * j + 3);
        }
      } else {
        const uint32_t p0 = lut16[code & mask0];
        const uint32_t p1 = lut16[(1u << w0) + ((code >> w0) & mask1)];
        out32 = p0 | (p1 << 16);
      }
      // Codeword vl covers K elements [vl*8, vl*8+8) of this K-tile ->
      // 4 whole bytes of the K-major nibble stream, owned by THIS thread
      // (full-byte stores, no sub-byte RMW). Addressing is deliberately
      // MANUAL: the UMMA sub-byte atoms' Swizzle<2,4,3> operates on BYTE
      // addresses (the smem_ptr_flag_bits<4> convention the LDSM read path
      // honors), while element-wise subbyte writes through the
      // position-independent tensor apply it to NIBBLE offsets — measured
      // via the smem debug dump as a row-dependent 16/32-nibble block
      // permutation. So: plain (non-swizzle) layout -> nibble offset ->
      // byte offset -> swizzle functor in byte space.
      const int lnib = sB_nosw(r, vl * 8);
      const int lb = lnib >> 1;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        sB_base[sB_sw(lb + j)] = uint8_t(out32 >> (8 * j));
      }
    }

    // SFB: one ue4m3 byte per (row, group-of-16), gathered from the row's
    // scale plane at gmem byte offset 4k (u8 loads — always in-row). v2
    // composes via the e4m3-byte compose table (exact); v1 copies the plane
    // byte (already e4m3, positive by construction).
    uint8_t* sfb_smem = reinterpret_cast<uint8_t*>(shared_tensors.smem_SFB.begin());
    const uint8_t* comp = shared_tensors.smem_compose.data();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SfPerThread; ++i) {
      const int linear = thread_idx + i * ThreadCount;
      const int r = linear & (TileN - 1);
      const int g = linear / TileN;                  // 0..7 in this K-tile
      const int gs = half * 8 + g;                   // group in superblock
      int n_glob = n_base + r;
      n_glob = n_glob < N_rows ? n_glob : (N_rows - 1);
      const uint8_t* srow = gp + int64_t(n_glob) * row_stride + sb_off + 4 * kb;
      uint8_t sf;
      if (params.is_v2) {
        const int e = __ldg(srow);
        const int c = (__ldg(srow + 1 + (gs >> 1)) >> ((gs & 1) * 4)) & 0xF;
        sf = comp[e * 16 + c];
      } else {
        sf = __ldg(srow + gs);
      }
      sfb_smem[sfb_layout(make_coord(r, g * 16))] = sf;
    }
  }

  /// Consumer: decode-then-MMA per K-tile (fp8-fork control flow + the
  /// pristine block-scaled fragment plumbing).
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& params) {
    using namespace cute;
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));
    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA            = smem_thr_copy_SFA.partition_S(
        as_position_independent_swizzle_tensor(sSFA));
    Tensor tCrSFA_copy_view  = smem_thr_copy_SFA.retile_D(tCrSFA);

    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);

    // Decode-write addressing pieces: the single decoded buffer's PLAIN
    // layout (nibble offsets, swizzle stripped) + its swizzle functor
    // (applied in BYTE space — see decode_stage) + the raw byte base.
    auto sB_l = SmemLayoutB{}(_,_,Int<0>{});
    auto sB_nosw = sB_l.layout_b();
    auto sB_sw = sB_l.layout_a();
    uint8_t* sB_base = reinterpret_cast<uint8_t*>(&shared_tensors.smem_B);
    // Plain (non-swizzled) SFB layout for direct offset writes.
    auto sfb_layout = SmemLayoutSFB{}(_,_,Int<0>{});

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));

    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage   = tCsA(_,_,_,read_stage);
    auto tCsB_stage   = tCsB(_,_,_,Int<0>{});
    auto tCsSFA_stage = tCsSFA(_,_,_,read_stage);
    auto tCsSFB_stage = tCsSFB(_,_,_,Int<0>{});

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));
      // Left shift A,B for FP4 (no-op for the non-f8f6f4 VS atom; kept to
      // mirror the pristine mainloop exactly).
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));
      copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
    };
    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma,
                 make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)),
                 make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)),
                 accum);
    };

    // Once-per-CTA codebook staging (same race-freedom argument as the fp8
    // fork: only the MMA warpgroups touch these regions, they enter mma() in
    // lockstep, and the NamedBarrier below orders staging before every read).
    if (!lut_resident_) {
      load_lut(shared_tensors, params, thread_idx);
      cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      lut_resident_ = true;
    }

    int t_abs = 0;
    pipeline.consumer_wait(smem_pipe_read);
    decode_stage(shared_tensors, sB_base, sB_nosw, sB_sw, sfb_layout, params,
                 read_stage, t_abs >> 1, t_abs & 1, thread_idx);
    cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    if (params.ptr_debug != nullptr && blockIdx.x == 0 && blockIdx.y == 0 &&
        !debug_dumped_) {
      // raw physical bytes: [smem_B cosize/2 | smem_SFB cosize | lut 16K |
      // compose 4K]
      constexpr int NB = cute::cosize_v<SmemLayoutB> / 2;
      constexpr int NSFB = cute::cosize_v<SmemLayoutSFB>;
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(&shared_tensors.smem_B);
      const uint8_t* psfb = reinterpret_cast<const uint8_t*>(&shared_tensors.smem_SFB);
      uint8_t* d = params.ptr_debug;
      for (int i = thread_idx; i < NB; i += ThreadCount) d[i] = pb[i];
      d += NB;
      for (int i = thread_idx; i < NSFB; i += ThreadCount) d[i] = psfb[i];
      d += NSFB;
      for (int i = thread_idx; i < CbLutMaxBytes; i += ThreadCount) d[i] = shared_tensors.smem_lut.data()[i];
      d += CbLutMaxBytes;
      for (int i = thread_idx; i < CbComposeBytes; i += ThreadCount) d[i] = shared_tensors.smem_compose.data()[i];
      debug_dumped_ = true;
      cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    }

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
              thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage   = tCsA(_,_,_,read_stage);
          tCsSFA_stage = tCsSFA(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
          ++t_abs;
          decode_stage(shared_tensors, sB_base, sB_nosw, sB_sw, sfb_layout,
                       params, read_stage, t_abs >> 1, t_abs & 1, thread_idx);
          cutlass::arch::NamedBarrier::sync(
              thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        }
        copy_kblock(k_block_next);
        gemm_kblock(k_block);
      });
    }

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }
      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);
    });
  }

  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }

private:
  // Per-thread, whole-kernel-lifetime flag (the kernel declares ONE
  // CollectiveMainloop outside the scheduler loop) — see the fp8 fork.
  bool lut_resident_ = false;
  bool debug_dumped_ = false;
};

}  // namespace cutlass::gemm::collective
