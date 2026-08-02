/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/memory_sm80.h"   // cp_async_zfill (A-row gather)
#include "cutlass/arch/barrier.h"       // cpasync_barrier_arrive_noinc

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------------
// PrismaQuant EXPERT-INDEXED fork of the sm120 TMA warp-specialized mainloop.
//
// Source: vendored CUTLASS 4.3.4 sm120_mma_tma.hpp — the same file
// sm120_cb_mma_tma.hpp forks without change (kept pristine alongside as
// sm120_mma_tma_orig.hpp for diffing). This fork carries exactly FOUR
// additions, each marked "PRISMAQUANT ADDITION":
//
//   1. Arguments/Params gain {int const* ptr_expert_ids; int num_experts;}.
//   2. B's TMA tensor is built with `num_experts` as its BATCH (l) extent
//      instead of the problem's L, so a stacked [E, N, K] weight buffer is one
//      tensor whose per-expert stride is N*K.
//   3. `load()` takes the B tile's l-coordinate from expert_ids[m_tile]
//      (one scalar gmem read, uniform across the CTA) rather than from the
//      problem's l-coordinate.
//   4. An IN-MAINLOOP A-ROW GATHER: with {int const* ptr_row_src;
//      int source_rows;} set, the producer warp reads each padded row m of
//      the launch from row ptr_row_src[m] of a COMPACT [source_rows, K]
//      activation tensor via predicated 16-byte cp.async (ids outside
//      [0, source_rows) zero-fill), instead of TMA-reading a materialized
//      row-padded copy. The smem stage bytes are IDENTICAL to what the TMA
//      path loads from the padded copy — padding rows are zero either way,
//      and TMA's own out-of-bounds K-residue zero-fill is matched by the
//      cp.async zfill predicate — so the two A paths produce bit-identical
//      output (gated by test_bf16_grouped_cutlass.py). The pipeline pattern
//      (NumProducerThreadEvents = 33 = one TMA leader arrive_and_expect_tx
//      + 32 per-lane cp.async noinc arrivals, transaction bytes NK-only in
//      gather mode) is exactly upstream's own
//      sm120_mma_tma_blockwise_scaling.hpp producer idiom.
//
// With ptr_expert_ids == nullptr and num_experts == 1 and
// ptr_row_src == nullptr this is the unmodified upstream mainloop schedule,
// which is what the dense (E=1) path uses.
//
// WHY. Upstream CUTLASS 4.3.4 has NO sm120 ptr-array/grouped collective
// (`sm120_mma_builder.inl` static_asserts `!IsPtrArrayKernel`), so Gridbook
// groups the way its two fused kernels already do: every expert's rows span
// whole TileM blocks, and the launch is ONE ordinary single-problem GEMM of
// shape (Mp, N, K) in which each M-tile reads a different expert's B. No
// tensormap updates, no ptr-arrays; the descriptors stay host-built and
// immutable. Addition 4 removes the one real cost of that construction the
// 2026-08-01 T=512 measurements identified: the row-padded activation COPY
// (an HBM write plus a padded re-read too large for L2), replacing it with
// direct indexed reads of the compact activation, which IS L2-resident at
// serving sizes. See cb_grouped_common.hpp.
//
// WHAT IS NOT HERE. No packed-B/LUT machinery: B is plain BF16, TMA'd straight
// into smem. This mainloop is a pure GEMM schedule change relative to the SM80
// `DefaultGemmGrouped` bridge it replaces; the only numerical consequence is
// FP32 accumulation ORDER (the epilogue still rounds once to bf16), and the
// gather path does not even change that — same smem bytes, same MMA schedule.
// ---------------------------------------------------------------------------
namespace cutlass::gemm {
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120CbBf16ExpertTmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};
}  // namespace cutlass::gemm

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120CbBf16ExpertTmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm120CbBf16ExpertTmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  // PRISMAQUANT ADDITION 4: 33 producer events per stage — the TMA leader's
  // arrive_and_expect_tx plus one cp.async `noinc` arrival from each of the
  // 32 producer-warp lanes. This is the exact producer accounting upstream's
  // sm120_mma_tma_blockwise_scaling.hpp uses for its TMA + cp.async stage.
  // In TMA-A mode a lane's noinc arrival simply fires immediately (no prior
  // cp.async), so both A paths satisfy the same barrier arithmetic.
  static constexpr int NumProducerThreadEvents = 33;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::conditional_t<cute::is_same_v<ElementA, float>,
                                                  cutlass::tfloat32_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementA>>>>>>;
  using TmaInternalElementB = cute::conditional_t<cute::is_same_v<ElementB, float>,
                                                  cutlass::tfloat32_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::array_aligned<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::array_aligned<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
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
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    // --- PRISMAQUANT ADDITION 1: tile-indexed grouping (MoE). ---
    // ptr_expert_ids[m_tile] selects which batch (expert) of a stacked
    // [E, N, K] B buffer this M-tile multiplies against.
    // nullptr => dense path: the B batch coordinate is the problem's l.
    int const* ptr_expert_ids{nullptr};
    int num_experts{1};              // E (defaults to 1 == dense)
    // --- PRISMAQUANT ADDITION 4: in-mainloop A-row gather. ---
    // When ptr_row_src != nullptr, ptr_A is a COMPACT row-major
    // [source_rows, K] tensor (row stride exactly K elements) and padded row
    // m of the launch loads source row ptr_row_src[m]; any id outside
    // [0, source_rows) yields a zero row (the padding rows). nullptr => A is
    // TMA'd from a row-padded [Mp, K] tensor exactly as upstream.
    int const* ptr_row_src{nullptr};
    int source_rows{0};
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,0),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    // PRISMAQUANT ADDITION 1 (device side).
    int const* ptr_expert_ids = nullptr;
    int num_experts = 1;
    // PRISMAQUANT ADDITION 4 (device side). ptr_a duplicates the raw A
    // pointer because in gather mode A is read by address arithmetic, not
    // through the (built but unissued) TMA descriptor. a_cols is K.
    int const* ptr_row_src = nullptr;
    int source_rows = 0;
    ElementA const* ptr_a = nullptr;
    int a_cols = 0;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    // PRISMAQUANT ADDITION 2: B's batch mode is the EXPERT mode, not the
    // problem's L. For the dense path num_experts == 1 == L, so the descriptor
    // is bit-identical to the upstream one.
    const int32_t b_batch = args.num_experts > 0 ? args.num_experts : 1;
    // PRISMAQUANT ADDITION 4: in gather mode the A descriptor is built (the
    // kernel layer prefetches it and load_init derives the k-tile count from
    // its shape) but never issued — the problem's M is the padded Mp while
    // ptr_A holds only source_rows rows, so issuing it would be out of
    // bounds. The transaction bytes the pipeline expects are NK-only, since
    // A arrives by per-lane cp.async arrivals instead of TMA bytes.
    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,b_batch), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    const bool gather_a = args.ptr_row_src != nullptr;
    return {
      tma_load_a,
      tma_load_b,
      gather_a ? TmaTransactionBytesNK : TmaTransactionBytes,
      TmaTransactionBytesMK,
      TmaTransactionBytesNK,
      args.ptr_expert_ids,
      b_batch,
      args.ptr_row_src,
      args.source_rows,
      args.ptr_A,
      int(K)
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    // PRISMAQUANT ADDITION 2 (alignment): B is checked at its own batch extent.
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,args.num_experts), StrideB{});
    implementable = implementable && (args.num_experts >= 1);
    // PRISMAQUANT ADDITION 4: gather mode needs a non-negative source extent.
    // (Row alignment is the same K % 8 the A TMA check above enforces; the
    // 16-byte cp.async chunks start at multiples of 8 elements.)
    implementable = implementable && (args.ptr_row_src == nullptr || args.source_rows >= 0);

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    // PRISMAQUANT ADDITION 2: B's batch extent is the EXPERT count.
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
        make_shape(N,K,mainloop_params.num_experts));                                                        // (n,k,e)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,e)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    // PRISMAQUANT ADDITION 4: the whole producer WARP participates. TMA
    // issue stays behind elect_one; the A-row gather (when enabled) uses all
    // 32 lanes; every lane contributes one `noinc` arrival per stage (see
    // NumProducerThreadEvents). This is the produce structure of upstream's
    // sm120_mma_tma_blockwise_scaling.hpp.
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
    // PRISMAQUANT ADDITION 3: tile-indexed grouping. For the MoE path the B
    // batch coordinate is the expert this M-tile belongs to (uniform across
    // the CTA, one scalar gmem read). Padding tiles may carry -1; clamp so
    // the TMA never goes out of bounds (their output rows are discarded by
    // the caller's unpermute).
    int b_l = l_coord;
    if (mainloop_params.ptr_expert_ids != nullptr) {
      int eid = __ldg(mainloop_params.ptr_expert_ids + m_coord);
      b_l = (eid < 0) ? 0 : eid;
    }
    Tensor gB = gB_nkl(_,_,n_coord,_,b_l);                                                         // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
      }
    }

    if (mainloop_params.ptr_row_src == nullptr) {
      // ----- TMA-A mode: the upstream mainloop body, plus the per-lane
      // arrival every stage now expects. -----
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        int write_stage = smem_pipe_write.index();
        if (lane_predicate) {
          using BarrierType = typename MainloopPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

          copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
          copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        }
        // With no cp.async pending, each lane's arrival fires immediately.
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
      return;
    }

    // ----- Gather-A mode (PRISMAQUANT ADDITION 4): B is TMA'd exactly as
    // above; each A stage is gathered row-by-row from the compact source by
    // all 32 lanes with predicated, zero-filling 16-byte cp.async. The smem
    // bytes are identical to what TMA-A mode loads from a materialized
    // padded copy: ids outside [0, source_rows) produce zero rows (the
    // padded copy gathers an appended zero row there) and chunks beyond K
    // produce zeros (TMA zero-fills its out-of-bounds K residue). -----
    constexpr int kTileM = size<0>(TileShape{});
    constexpr int kTileK = size<2>(TileShape{});
    constexpr int kChunkElems = 16 / int(sizeof(ElementA));       // one cp.async
    constexpr int kRowChunks = kTileK / kChunkElems;              // chunks/row
    static_assert(kTileK % kChunkElems == 0,
                  "TileK must be a whole number of 16-byte cp.async chunks");
    static_assert((kTileM * kRowChunks) % 32 == 0,
                  "the A tile must divide evenly over the 32 producer lanes");
    constexpr int kIters = kTileM * kRowChunks / 32;
    // Lane l, iteration i covers row (l / kRowChunks) + i * (32 / kRowChunks),
    // chunk l % kRowChunks: each warp instruction reads 32/kRowChunks rows in
    // kRowChunks consecutive 16-byte pieces — coalesced per row, and the row
    // ids broadcast from L1 across the lanes that share a row.
    int const* row_ids = mainloop_params.ptr_row_src + m_coord * kTileM;
    ElementA const* src_a = mainloop_params.ptr_a;
    int const src_rows = mainloop_params.source_rows;
    int const src_cols = mainloop_params.a_cols;
    int const lane_row0 = thread_idx / kRowChunks;
    int const lane_chunk = (thread_idx % kRowChunks) * kChunkElems;
    int k_tile_base = 0;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_ (all lanes wait; the leader lane
      // performs the arrive_and_expect_tx for the NK transaction bytes).
      pipeline.producer_acquire(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (lane_predicate) {
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIters; ++i) {
        int const row = lane_row0 + i * (32 / kRowChunks);
        int const kk = k_tile_base + lane_chunk;                // global K offset
        int const src_row = __ldg(row_ids + row);
        bool const pred =
            (unsigned(src_row) < unsigned(src_rows)) && (kk < src_cols);
        // A well-formed (in-tensor) address even when predicated off; a
        // zero-size cp.async performs no read.
        int64_t const src_off =
            pred ? int64_t(src_row) * src_cols + kk : int64_t(0);
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            raw_pointer_cast(&sA(row, lane_chunk, write_stage)),
            src_a + src_off, pred);
      }
      // Each lane's arrival fires when its cp.asyncs above have completed;
      // together with the leader's expect_tx these are the 33 producer
      // events the barrier was initialized with.
      pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);

      ++k_tile_iter;
      k_tile_base += kTileK;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});    // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});    // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)

    //
    // Copy Atom A and B retiling
    //

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                            //      (CPY,CPY_M,CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                            //      (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));

    //
    // PIPELINED MAIN LOOP
    //

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage   = tCsA(_,_,_,read_stage);
    auto tCsB_stage   = tCsB(_,_,_,read_stage);

    auto copy_kblock = [&](auto k_block) {
        // copy smem->rmem for A/B operand
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

      // Left shift A,B for FP4 (no-op for every other MMA op)
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
    };

    pipeline.consumer_wait(smem_pipe_read);

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      //
      // Compute on k_tile
      //
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          // UNLOCK smem_pipe_read, done _computing_ on it
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage   = tCsA(_,_,_,read_stage);
          tCsB_stage   = tCsB(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);

      });
    } // k_tile_count

    //
    // Hoist out last k_tile
    //
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        // UNLOCK smem_pipe_read, done _computing_ on it
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);

    });
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
