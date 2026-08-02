/***************************************************************************************************
 * PrismaQuant NVFP4_CB **v2 quality** decode-in-prologue mainloop for sm120 /
 * sm121 (GB10) — CB -> BF16, contract-preserving.
 *
 * WHAT THIS IS, AND WHAT IT IS NOT.
 *
 * There are two fused FP4 mainloops in this tree and they serve DIFFERENT
 * numeric contracts:
 *
 *   * `sm120_cb_fused_fp4_mma.hpp` decodes CB into the NATIVE NVFP4 operands
 *     (e2m1 nibbles + ue4m3 SFB) and multiplies them with packed-e2m1
 *     activations on the block-scaled MMA. That is true W4A4 and it CHANGES
 *     the served activation bucket, which is why it is opt-in behind
 *     `PRISMAQUANT_CB_FUSED_FP4` and its six promotion gates.
 *
 *   * THIS file decodes CB into BF16 values that are **bit-identical to
 *     `cb_expand_v2`'s output** (csrc/cb_gemv_v2.cu::cb_expand_v2_kernel) and
 *     multiplies them against the same BF16 group-16-QDQ'd activations the
 *     shipping quality path already feeds the BF16 bridge. Nothing about the
 *     served activation contract or the decoded weight values moves: the only
 *     difference from `expand -> cb_bf16_grouped_mm` is the FP32 GEMM
 *     REDUCTION ORDER — the same requalification class the promoted FP8 mid-M
 *     fused kernel cleared (2026-08-01 performance audit §3 P2a).
 *
 * WHY IT EXISTS. The quality FP4 path materializes the decoded [N,K] BF16 tile
 * in HBM before every dense prefill GEMM (~2x the GEMM's own bytes). At mid M
 * that transient dominates. Decoding inside the CUTLASS producer/consumer
 * stage removes it entirely. The mechanism is exactly the FP8-CB fused
 * kernel's (`sm120_cb_fused_mma.hpp`): decode the packed bytes into the
 * standard swizzled SmemLayoutB buffer between `consumer_wait` and the
 * smem->rmem copy, behind a NamedBarrier.
 *
 * MID-M ONLY BY CONSTRUCTION. Decode-in-prologue re-decodes B once per M-tile,
 * so its work is `ceil(M/TileM)` x a one-shot expand. The FP8 twin measured
 * 0.22x of the serial transient at M~1400 and 1.04x/1.26x/1.45x at
 * M=32/64/128. This lane is therefore gated at `9 <= M <= 128` (ONE M-tile)
 * by the python dispatch, and the binding refuses anything above.
 *
 * B-SIDE DESIGN (differs from the FP8 fork; follows the FP4 fork).
 *   - fp4-v2 rows have an ODD `type_size = 4k+9`, so the packed row stride is
 *     never a 16-byte multiple and TMA is structurally unusable for B (TMA
 *     requires 16-byte global strides). The producer therefore publishes a
 *     tiny per-stage DESCRIPTOR (the CTA's first output row) into smem under
 *     the stage's mbarrier, and the consumer threads gather the packed bytes
 *     straight from gmem with aligned-u32 windows. This is the same
 *     construction `sm120_cb_fused_fp4_mma.hpp` ships for this exact payload.
 *     Ordering: plain stores -> __syncwarp -> the leader's mbarrier arrive
 *     (release.cta, carried by the A TMA copy) -> consumer_wait (acquire.cta).
 *     Writer and reader are both generic proxy, so no async-proxy fence.
 *   - k_bits / type_size are RUNTIME parameters: the packed stream never
 *     touches a TMA descriptor or a k-sized smem layout here, so ONE
 *     instantiation serves the whole K12..K24 product ladder. The only
 *     compile-time B-side parameter is `LutBytes`, the codebook smem stage
 *     capacity (see below).
 *
 * CODEBOOK RESIDENCY. The fp4-v2 product dictionary is BF16 VALUES — 4 per
 * entry, so `(8 << ceil(k/2)) + (8 << floor(k/2))` bytes: 1 KiB at k12 up to
 * 64 KiB at k24. The high rungs cannot be staged next to the GEMM's own smem,
 * so the stage is a PREFIX of the flat codebook (`[sub0 | sub1]`) whose length
 * the host chooses per rung, and the two gather pointers are selected ONCE per
 * decode from that length:
 *
 *     d0 = (stage_bytes >= sub0_bytes)  ? smem_lut      : cb_global
 *     d1 = (stage_bytes >= total_bytes) ? smem_lut + e1 : cb_global + e1
 *
 * so a partially staged table costs a pointer select, never a per-gather
 * branch. Both loads are GENERIC (`ld`), which is correct for either window;
 * `__ldg` is deliberately NOT used on these pointers because a shared-memory
 * address through the non-coherent path is illegal.
 *
 * The (256,16) fp32 compose table (16 KiB) stays in GLOBAL and is gathered
 * with `__ldg`, exactly as `cb_expand_v2` and the decode GEMV do: it is read
 * once per 8 decoded weights and is L1-hot.
 *
 * INV-1: the dense BF16 tile exists only as ONE [TileN, TileK] smem buffer;
 * it is never written to HBM.
 **************************************************************************************************/
#pragma once

#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
// The PRIMARY CollectiveMma template this file specializes. The other
// cutlass_fork headers rely on their TU including collective_builder.hpp
// first; declaring the dependency here makes the fork include-order
// independent, which the host-only smem probe needs (it builds no collective
// through the builder at all).
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/layout.hpp"
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
  class KernelSchedule_,
  int LutBytes_
>
struct MainloopSm120CbFp4V2Bf16TmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  constexpr static int LutBytes = LutBytes_;
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
  int LutBytes,
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
    MainloopSm120CbFp4V2Bf16TmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType, LutBytes>,
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
  using DispatchPolicy = MainloopSm120CbFp4V2Bf16TmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType, LutBytes>;
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
  static constexpr int NumWarps = ThreadCount / 32;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  static constexpr int NumProducerThreadEvents = 1;

  static constexpr int TileM = size<0>(TileShape{});
  static constexpr int TileN = size<1>(TileShape{});
  static constexpr int TileK = size<2>(TileShape{});

  // --- fp4-v2 CB geometry -------------------------------------------------
  // 256 weights per superblock; one codeword decodes 8 (4 from each of the two
  // product sub-tables), so a superblock holds 32 codewords and the K-tile
  // must divide it evenly.
  static constexpr int CbWeightsPerSuperblock = 256;
  static constexpr int CbWeightsPerCodeword = 8;
  static constexpr int CwPerRowTile = TileK / CbWeightsPerCodeword;
  static constexpr int TilesPerSuperblock = CbWeightsPerSuperblock / TileK;
  static_assert(CbWeightsPerSuperblock % TileK == 0,
                "the fp4-v2 CB fused mainloop needs TileK to divide the "
                "256-weight superblock");
  static_assert(TileK % CbWeightsPerCodeword == 0,
                "TileK must be a whole number of 8-weight codewords");

  // Decode work assignment. One warp owns `32 / CwPerRowTile` whole rows per
  // iteration and reads their codewords contiguously, so each 8/16-lane group
  // issues ONE short in-row burst instead of 32 unrelated odd-stride gathers
  // (the coalescing lesson already banked in the fp4 fused fork).
  static constexpr int CwPerTile = TileN * CwPerRowTile;
  static constexpr int CwPerThread = CwPerTile / ThreadCount;
  static constexpr int RowsPerWarpIter = 32 / CwPerRowTile;
  static_assert(CwPerTile % ThreadCount == 0,
                "codeword count must divide the MMA thread count");
  static_assert(32 % CwPerRowTile == 0,
                "codewords per row-tile must divide the warp width");
  static_assert(RowsPerWarpIter * NumWarps * CwPerThread == TileN,
                "decode row assignment must cover TileN exactly once");

  // --- codebook smem stage (see the header note) ---------------------------
  static constexpr int CbLutSmemBytes = LutBytes;
  static_assert(CbLutSmemBytes % 1024 == 0,
                "the LUT stage must be a 1024-byte multiple so it cannot "
                "perturb the alignment of the 1024-aligned A/B buffers");

  // A zero-sized stage must cost ZERO bytes, and cute::array_aligned<uint8_t,0>
  // is still CUTE_ALIGNAS(16) -> 16 B. So the stage is an empty BASE of
  // TensorStorage (empty-base optimization), the fp8 fork's trick.
  struct LutStorageNone {
    CUTLASS_HOST_DEVICE static uint8_t const* lut_smem() { return nullptr; }
    CUTLASS_HOST_DEVICE static uint8_t* lut_smem_mut() { return nullptr; }
  };
  struct LutStorageArray {
    alignas(16) cute::array_aligned<
        uint8_t, (CbLutSmemBytes > 0 ? CbLutSmemBytes : 1024)> smem_lut;
    CUTLASS_HOST_DEVICE uint8_t const* lut_smem() const { return smem_lut.data(); }
    CUTLASS_HOST_DEVICE uint8_t* lut_smem_mut() { return smem_lut.data(); }
  };
  using LutStorage =
      cute::conditional_t<(CbLutSmemBytes > 0), LutStorageArray, LutStorageNone>;

  // Per-stage producer descriptor: the CTA's first output row. 16 bytes so the
  // array keeps the 16-byte alignment the buffers around it assume.
  static constexpr int CbStageDescBytes = 16;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(not cute::is_void_v<SmemCopyAtomA>, "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>, "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");
  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>,
                "CB fp4-v2 BF16 fused mainloop: A uses plain SM90_TMA_LOAD (no multicast).");
  static_assert(ThreadCount % 32 == 0, "the decoder assumes whole MMA warps");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();
  static_assert(!IsF8F6F4 &&
                cute::is_same_v<ElementA, cutlass::bfloat16_t> &&
                cute::is_same_v<ElementB, cutlass::bfloat16_t>,
                "the fp4-v2 quality fused mainloop decodes CB to BF16 and "
                "multiplies BF16 activations (contract-preserving lane)");

  using TmaInternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;

  using SmemAllocTypeA = typename TiledMma::ValTypeA;
  using SmemAllocTypeB = typename TiledMma::ValTypeB;

  // A: unchanged pipelined layout. B decoded: ONE buffer in the standard
  // swizzled layout (the MMA-side contract) — the decode target.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<1>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  // TMA moves A ONLY: B is decoded in-prologue and never crosses the barrier's
  // byte count.
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = 0;
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK;

  struct SharedStorage {
    // LutStorage is a BASE (not a member) so a zero-sized stage really is zero
    // bytes; its size is always a 1024-byte multiple, so it never perturbs the
    // 1024-byte alignment of the buffers below it.
    struct TensorStorage : cute::aligned_struct<128, _0>, LutStorage {
      alignas(1024) cute::array_aligned<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(16) cute::array_aligned<uint8_t, CbStageDescBytes * DispatchPolicy::Stages> smem_BP;
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
    // --- packed fp4-v2 CB weight stream (replaces ptr_B / dB) -------------
    uint8_t const* ptr_packed{nullptr};   // [N, >= n_sb*type_size] uint8
    int64_t packed_row_bytes{0};          // explicit row stride (odd parity ok)
    uint16_t const* ptr_cb{nullptr};      // flat BF16 product dictionary
                                          // [sub0 (4<<w0) | sub1 (4<<w1)]
    float const* ptr_compose{nullptr};    // (256*16) fp32 two-tier table
    int32_t cb_stage_bytes{0};            // prefix of ptr_cb staged to smem
    int32_t k_bits{0};
    int32_t type_size{0};                 // == 4*k_bits + 9
    // TESTS ONLY. Replaces each decoded value with a coordinate of the decode
    // write, so a one-hot read-out recovers the decoder's exact
    // (row, column, k-tile) -> smem mapping. Never set in serving dispatch;
    // 0 is the only value the binding's default path passes.
    //   1 = the tile row r, 2 = the column within the K-tile, 3 = the
    //   absolute K-tile index.
    int32_t debug_mode{0};
  };

  // Device side kernel params
  struct Params {
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    TMA_A tma_load_a;
    uint8_t const* ptr_packed;
    int64_t packed_row_bytes;
    uint16_t const* ptr_cb;
    float const* ptr_compose;
    int32_t cb_stage_bytes;
    int32_t k_bits;
    int32_t type_size;
    int32_t debug_mode;
    int32_t N_rows;
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

    return {
      tma_load_a, args.ptr_packed, args.packed_row_bytes, args.ptr_cb,
      args.ptr_compose, args.cb_stage_bytes, args.k_bits, args.type_size,
      args.debug_mode, static_cast<int32_t>(N),
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
    implementable = implementable && (K % CbWeightsPerSuperblock == 0) && (L == 1);
    implementable = implementable && (args.k_bits >= 12) && (args.k_bits <= 24);
    implementable = implementable && (args.type_size == 4 * args.k_bits + 9);
    implementable = implementable &&
        (args.packed_row_bytes >= int64_t(K / CbWeightsPerSuperblock) * args.type_size);
    implementable = implementable && (args.ptr_cb != nullptr);
    implementable = implementable && (args.ptr_compose != nullptr);
    implementable = implementable && (args.cb_stage_bytes >= 0) &&
                    (args.cb_stage_bytes <= CbLutSmemBytes) &&
                    (args.cb_stage_bytes % 16 == 0);
    return implementable;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    // gB_nkl stand-in: the kernel layer only reads SHAPES from element 1 of the
    // returned tuple (its idx2crd over the N tiling), never its data.
    Tensor gB_nkl = local_tile(
        make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
                    make_layout(make_shape(N, K, L))),
        TileShape{}, make_coord(_,_,_), Step<X, _1,_1>{});
    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Producer: TMA A, plus the per-stage packed-B descriptor. Called by the
  /// whole producer warp (32 threads).
  template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
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
    (void)block_rank_in_cluster;
    const int lane = thread_idx & 31;
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
    Tensor gA_mkl = get<0>(load_inputs);
    auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                          // (BLK_M,BLK_K,k)
    Tensor tAgA = block_tma_a.partition_S(gA);
    Tensor tAsA = block_tma_a.partition_D(sA);

    const int n_base = int(n_coord) * TileN;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      if (lane_predicate) {
        pipeline.producer_acquire(smem_pipe_write);
      }
      __syncwarp();

      int write_stage = smem_pipe_write.index();
      if (lane == 0) {
        int32_t* slot = reinterpret_cast<int32_t*>(
            shared_tensors.smem_BP.data() + write_stage * CbStageDescBytes);
        slot[0] = n_base;
      }
      __syncwarp();

      if (lane_predicate) {
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        // The leader's mbarrier arrive (inside .with()) carries release.cta
        // semantics and is ordered AFTER the __syncwarp above, so the manual
        // descriptor store is visible once the barrier phase flips.
        copy(mainloop_params.tma_load_a.with(*tma_barrier, 0),
             tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
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

  // --- once-per-CTA codebook staging --------------------------------------
  // Called by ALL MMA threads; the caller issues the NamedBarrier. Stages a
  // PREFIX of the flat BF16 dictionary (`[sub0 | sub1]`), 4 bytes per thread
  // per step (the codebook is 16-byte aligned in every producer artifact, so
  // any 4-byte granularity is safe).
  CUTLASS_DEVICE void
  load_lut(TensorStorage& shared_tensors, Params const& params,
           int thread_idx) const {
    if constexpr (CbLutSmemBytes > 0) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(shared_tensors.lut_smem_mut());
      const uint32_t* src = reinterpret_cast<const uint32_t*>(params.ptr_cb);
      const int nwords = params.cb_stage_bytes >> 2;
      for (int w = thread_idx; w < nwords; w += ThreadCount) {
        dst[w] = __ldg(src + w);
      }
    }
  }

  // --- the decode: packed gmem bytes -> the swizzled BF16 B buffer ---------
  //
  // BIT-EXACT to csrc/cb_gemv_v2.cu::cb_expand_v2_kernel, statement for
  // statement:
  //
  //   code   : k_bits window at bit `v*k_bits` of the superblock's index plane
  //   i0/i1  : ceil-first product split, i0 = code & m0, i1 = (code>>w0) & m1
  //   sc     : compose[super_e*16 + code16] (fp32 gather), where super_e is the
  //            superblock's exponent byte at offset 4k and code16 is the group
  //            nibble from byte 4k+1+(grp>>1), grp = v>>1
  //   value  : bf16_rn( f32(cb[...]) * sc )   -- one fp32 multiply, one round
  //
  // The window read differs in FORM only: `cb_expand_v2` assembles 8 bytes
  // byte-by-byte, this assembles the same bits from two aligned u32 loads. The
  // aligned window `[(b0 & ~3), (b0 & ~3) + 8)` always stays inside the
  // superblock for every compiled rung — max end byte is (31k>>3)+7 <= 4k+8 =
  // type_size-1 for all k — so no tail slack is ever required.
  template <class SBDecTensor>
  CUTLASS_DEVICE void
  decode_stage(TensorStorage& shared_tensors, SBDecTensor& sBx,
               Params const& params, int read_stage, int sb, int q,
               int t_abs, int thread_idx) const {
    const int32_t* slot = reinterpret_cast<const int32_t*>(
        shared_tensors.smem_BP.data() + read_stage * CbStageDescBytes);
    const int n_base = slot[0];

    const int kb = params.k_bits;
    const int ts = params.type_size;
    const int64_t row_stride = params.packed_row_bytes;
    const int N_rows = params.N_rows;
    const uint32_t mask_k = (uint32_t)((1u << kb) - 1u);
    const int w0 = (kb + 1) >> 1;                    // ceil-first split
    const int w1 = kb >> 1;
    const uint32_t mask0 = (1u << w0) - 1u;
    const uint32_t mask1 = (1u << w1) - 1u;
    const int64_t e1 = int64_t(4) << w0;             // sub1 element base

    // ONE pointer select per decode (never a per-gather branch); generic loads
    // are correct for either the smem stage or the global dictionary.
    const int32_t staged = params.cb_stage_bytes;
    const int32_t sub0_bytes = (int32_t)(8 << w0);
    const int32_t total_bytes = sub0_bytes + (int32_t)(8 << w1);
    const uint16_t* lut_s =
        reinterpret_cast<const uint16_t*>(shared_tensors.lut_smem());
    const uint16_t* d0 = (staged >= sub0_bytes) ? lut_s : params.ptr_cb;
    const uint16_t* d1 =
        ((staged >= total_bytes) ? lut_s : params.ptr_cb) + e1;

    const uint8_t* __restrict__ gp = params.ptr_packed;
    const int64_t sb_off = int64_t(sb) * ts;
    const int warp = thread_idx >> 5;
    const int lane = thread_idx & 31;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < CwPerThread; ++i) {
      const int r = (i * NumWarps + warp) * RowsPerWarpIter
                  + (lane / CwPerRowTile);
      const int vl = lane % CwPerRowTile;            // 0..CwPerRowTile-1
      const int v = q * CwPerRowTile + vl;           // codeword in superblock
      int n_glob = n_base + r;
      n_glob = n_glob < N_rows ? n_glob : (N_rows - 1);
      const uint8_t* row_g = gp + int64_t(n_glob) * row_stride + sb_off;

      // --- codeword: aligned-u32 window over the misaligned bit position ---
      const int bitpos = v * kb;
      const uint8_t* pbyte = row_g + (bitpos >> 3);
      const uintptr_t pa = reinterpret_cast<uintptr_t>(pbyte);
      const uint32_t* al = reinterpret_cast<const uint32_t*>(pa & ~uintptr_t(3));
      const int rem = int((pa & 3) << 3) + (bitpos & 7);
      const uint64_t w64 = (uint64_t(__ldg(al + 1)) << 32) | uint64_t(__ldg(al));
      const uint32_t code = uint32_t(w64 >> rem) & mask_k;

      // --- two-tier v2 scale compose (fp32, gathered from the global table) -
      const int grp = v >> 1;
      const uint8_t* srow = row_g + 4 * kb;
      const int super_e = (int)__ldg(srow);
      const int sub_byte = (int)__ldg(srow + 1 + (grp >> 1));
      const int code16 = (sub_byte >> ((grp & 1) * 4)) & 0xF;
      const float sc = __ldg(params.ptr_compose + super_e * 16 + code16);

      // --- product-mode value gather (4 bf16 per sub-table) ----------------
      const uint32_t i0 = code & mask0;
      const uint32_t i1 = (code >> w0) & mask1;
      const uint2 g0 = *reinterpret_cast<const uint2*>(d0 + int64_t(i0) * 4);
      const uint2 g1 = *reinterpret_cast<const uint2*>(d1 + int64_t(i1) * 4);
      const uint32_t raw[4] = {g0.x, g0.y, g1.x, g1.y};

      const int c0 = vl * CbWeightsPerCodeword;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < CbWeightsPerCodeword; ++j) {
        const uint16_t src =
            (uint16_t)((j & 1) ? (raw[j >> 1] >> 16) : (raw[j >> 1] & 0xffffu));
        __nv_bfloat16_raw rb; rb.x = src;
        const float val = __bfloat162float(__nv_bfloat16(rb));
        float scaled = val * sc;
        if (params.debug_mode != 0) {                 // TESTS ONLY (see above)
          scaled = (params.debug_mode == 1) ? float(r)
                 : (params.debug_mode == 2) ? float(c0 + j)
                                            : float(t_abs);
        }
        const uint16_t out = __bfloat16_as_ushort(__float2bfloat16_rn(scaled));
        sBx(r, c0 + j, 0) = cutlass::bfloat16_t::bitcast(out);
      }
    }
  }

  /// Consumer: decode-then-MMA per K-tile.
  template <class FrgTensorC>
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

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});   // (BLK_N,BLK_K,1)

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));

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

    // Element-addressable view of the single decoded buffer for the decode
    // writes. It is built by applying `as_position_independent_swizzle_tensor`
    // to the SAME 3-mode `sB` the LDSM read path partitions, and the pipe
    // coordinate is supplied at INDEX time.
    //
    // This is load-bearing and was a measured bug: slicing the pipe mode FIRST
    // (`as_position_independent_swizzle_tensor(sB(_,_,Int<0>{}))`, which is
    // what the fp8 fork can do because its layout aliases differently) yields a
    // DIFFERENT physical mapping than the read view for this bf16 128-byte
    // swizzle atom — a coordinate-write probe showed columns permuted by
    // XOR 8 on odd rows and rows 8/9 transposed. Deriving the write view from
    // the identical expression the reader uses makes the two mappings the same
    // object by construction, so they cannot drift again. BF16 needs no manual
    // byte addressing (unlike the sub-byte fp4 NATIVE fork): the swizzle
    // functor applies to element offsets exactly as LDSM expects.
    auto sBx = as_position_independent_swizzle_tensor(sB);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));

    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage = tCsA(_,_,_,read_stage);
    auto tCsB_stage = tCsB(_,_,_,Int<0>{});

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));
      // Left shift A,B for FP4 (no-op for the bf16 atom; kept to mirror the
      // pristine mainloop exactly).
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));
    };
    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
    };

    // --- once-per-CTA codebook staging ------------------------------------
    // The kernel is persistent: mma() is re-entered for every work tile this
    // CTA visits, so the staging is guarded by a member flag that lives in the
    // consumer threads' registers for the whole kernel. The staged prefix does
    // not depend on the work tile (one codebook per launch — the python layer
    // asserts a single interned LUT block), so it is staged once and the
    // region is read-only for the CTA's lifetime. Race-freedom is the fp8
    // fork's argument verbatim: only the MMA warpgroups touch smem_lut, they
    // enter mma() in lockstep for the same work tile, and the NamedBarrier
    // orders every staging store before every decode load.
    if constexpr (CbLutSmemBytes > 0) {
      if (!lut_resident_) {
        load_lut(shared_tensors, mainloop_params, thread_idx);
        cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        lut_resident_ = true;
      }
    }

    int t_abs = 0;
    pipeline.consumer_wait(smem_pipe_read);
    decode_stage(shared_tensors, sBx, mainloop_params, read_stage,
                 t_abs / TilesPerSuperblock, t_abs % TilesPerSuperblock,
                 t_abs, thread_idx);
    cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);

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
          tCsA_stage = tCsA(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
          ++t_abs;
          decode_stage(shared_tensors, sBx, mainloop_params, read_stage,
                       t_abs / TilesPerSuperblock, t_abs % TilesPerSuperblock,
                       t_abs, thread_idx);
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
  // Per-thread, whole-kernel-lifetime flag: has this CTA already staged the
  // codebook prefix into smem? See the staging block in mma().
  bool lut_resident_ = false;
};

}  // namespace cutlass::gemm::collective
