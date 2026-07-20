/***************************************************************************************************
 * PrismaQuant FP8_CB fused decode-in-prologue mainloop for sm120 (GB10).
 *
 * Derived from the vendored CUTLASS 4.3.4 sm120_mma_tma.hpp (pristine copy:
 * sm120_mma_tma_orig.hpp; passthrough fork: sm120_cb_mma_tma.hpp). The
 * B-operand path is replaced:
 *
 *   - gmem B is the PACKED FP8_CB byte stream ([N_rows, n_sb*type_size]
 *     uint8, UNPADDED rows so the row stride is a 16-byte multiple). Each
 *     K-tile of 128 weights lives in one 256-weight superblock (sb = kt/2);
 *     the TMA loads the full superblock row-slice [TileN x type_size] per
 *     stage (contiguous bytes — codewords are LSB-first in order). Two
 *     consecutive K-tiles re-load the same superblock; L2 absorbs the
 *     second read (v1 tax, documented).
 *   - After consumer_wait, the (256) MMA threads cooperatively decode the
 *     packed stage -> the standard swizzled SmemLayoutB DECODED buffer
 *     (single buffer, not per-stage), then a NamedBarrier, then the
 *     unchanged smem->rmem copy + MMA. Decode-then-MMA is serialized per
 *     K-tile (v1); the packed TMA pipeline still prefetches ahead.
 *   - Codebook lookups gather u16 pairs from GLOBAL memory (L1-hot, <=32 KB
 *     table) — no smem LUT, same pattern the decode GEMV proved at
 *     bandwidth-bound speeds. One codebook per launch (lattice artifacts;
 *     asserted at the python layer).
 *
 * INV-1: the dense fp8 tile exists only as one [TileN, TileK] smem buffer.
 * The k-bit width is a compile-time policy parameter (KBits in
 * {36,40,44,48}); type_size = 4*KBits bytes, n_sub = 4.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/collective.hpp"
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
  int KBits_
>
struct MainloopSm120CbFusedTmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  constexpr static int KBits = KBits_;
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
  int KBits,
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
    MainloopSm120CbFusedTmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType, KBits>,
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
  using DispatchPolicy = MainloopSm120CbFusedTmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType, KBits>;
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

  static constexpr int NumProducerThreadEvents = 1;

  // --- CB format constants (fp8 grid, product mode, n_sub = 4) ------------
  static constexpr int CbKBits = KBits;
  static constexpr int CbTypeSize = 4 * KBits;             // bytes / superblock
  static constexpr int CbSubW = KBits / 4;
  static constexpr uint32_t CbSubMask = (1u << CbSubW) - 1u;
  static constexpr int TileN = size<1>(TileShape{});
  static constexpr int TileK = size<2>(TileShape{});
  static_assert(TileK == 128, "CB fused mainloop assumes TileK = 128 (half a 256-weight superblock)");
  static_assert(KBits == 36 || KBits == 40 || KBits == 44 || KBits == 48,
                "shipped FP8_CB rungs only");
  static_assert(CbTypeSize % 16 == 0, "type_size must be a 16-byte multiple (TMA box)");

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(not cute::is_void_v<SmemCopyAtomA>, "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>, "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");
  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "CB fused mainloop: A uses plain SM90_TMA_LOAD (no multicast).");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "CB fused mainloop: B uses plain SM90_TMA_LOAD (no multicast).");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();
  static_assert(IsF8F6F4 && cute::is_same_v<ElementB, cutlass::float_e4m3_t>,
                "CB fused mainloop decodes to e4m3 (FP8_CB)");

  using TmaInternalElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, cutlass::tfloat32_t,
                              uint_bit_t<sizeof_bits_v<ElementA>>>;

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  // A: unchanged pipelined layout. B decoded: ONE buffer in the standard
  // swizzled layout (the MMA-side contract). B packed: dense row-major
  // [TileN, type_size] per stage.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<1>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutBPacked = Layout<
      Shape<Int<TileN>, Int<CbTypeSize>, Int<DispatchPolicy::Stages>>,
      Stride<Int<CbTypeSize>, _1, Int<TileN * CbTypeSize>>>;

  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(TileN * CbTypeSize);
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::array_aligned<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(128) cute::array_aligned<uint8_t, cute::cosize_v<SmemLayoutBPacked>> smem_BP;
      // 16-byte tail so the last row's aligned-u32 window overread (max
      // widx+2 -> byte type_size+3) stays inside the allocation.
      alignas(16) cute::array_aligned<uint8_t, 16> smem_BP_pad;
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
    uint8_t const* ptr_packed{nullptr};   // [N_rows, row_stride_bytes] UNPADDED
    int64_t packed_row_bytes{0};          // = n_sb * type_size (16B multiple)
    uint8_t const* ptr_lut{nullptr};      // flat e4m3-byte codebook (global)
  };

  // Device side kernel params
  struct Params {
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    using ShapeBP = decltype(make_shape(int32_t(0), int32_t(0), int32_t(0)));
    using StrideBP = decltype(make_stride(int64_t(0), Int<1>{}, int64_t(0)));
    using TMA_BP = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<uint8_t const*>(nullptr),
                    make_layout(ShapeBP{}, StrideBP{})),
        SmemLayoutBPacked{}(_,_,0),
        make_shape(Int<TileN>{}, Int<CbTypeSize>{}),
        _1{}));
    TMA_A tma_load_a;
    TMA_BP tma_load_bp;
    uint8_t const* ptr_lut;
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

    const int n_sb = K / 256;
    const int total_bytes = n_sb * CbTypeSize;
    Tensor tensor_bp = make_tensor(
        args.ptr_packed,
        make_layout(
            make_shape(int32_t(N), int32_t(total_bytes), int32_t(L)),
            make_stride(args.packed_row_bytes, Int<1>{},
                        int64_t(N) * args.packed_row_bytes)));
    typename Params::TMA_BP tma_load_bp = make_tma_copy(
        SM90_TMA_LOAD{}, tensor_bp, SmemLayoutBPacked{}(_,_,cute::Int<0>{}),
        make_shape(Int<TileN>{}, Int<CbTypeSize>{}), _1{});
    return {
      tma_load_a, tma_load_bp, args.ptr_lut,
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
    implementable = implementable && (K % 256 == 0);
    implementable = implementable && (args.packed_row_bytes % 16 == 0);
    implementable = implementable && (args.packed_row_bytes >= (K / 256) * CbTypeSize);
    return implementable;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_bp.get_tma_descriptor());
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));
    const int n_sb = K / 256;
    Tensor mBP_nkl = mainloop_params.tma_load_bp.get_tma_tensor(
        make_shape(int32_t(N), int32_t(n_sb * CbTypeSize), int32_t(L)));

    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});   // (BLK_M,BLK_K,m,k,l)
    Tensor gBP_nkl = local_tile(mBP_nkl, make_tile(Int<TileN>{}, Int<CbTypeSize>{}),
                                make_coord(_,_,_));                                          // (BLK_N,TS,n,sb,l)
    return cute::make_tuple(gA_mkl, gBP_nkl);
  }

  /// Producer: TMA A (dense) + packed-B superblock slices.
  template <class TensorA, class TensorBP, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorBP> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    (void)block_rank_in_cluster;
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
      Tensor sBP = make_tensor(make_smem_ptr(shared_tensors.smem_BP.data()), SmemLayoutBPacked{});

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gBP_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
      auto block_tma_bp = mainloop_params.tma_load_bp.get_slice(0);

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                        // (BLK_M,BLK_K,k)
      Tensor gBP = gBP_nkl(_,_,n_coord,_,l_coord);                      // (BLK_N,TS,sb)

      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);
      Tensor tBgBP = block_tma_bp.partition_S(gBP);
      Tensor tBsBP = block_tma_bp.partition_D(sBP);

      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        int write_stage = smem_pipe_write.index();
        int kt = int(*k_tile_iter);
        copy(mainloop_params.tma_load_a.with(*tma_barrier, 0), tAgA(_,_,_,kt), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_bp.with(*tma_barrier, 0), tBgBP(_,_,_,kt >> 1), tBsBP(_,_,_,write_stage));
        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
  }

  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // --- the decode: packed stage -> swizzled decoded B buffer --------------
  // 2048 codewords per K-tile (TileN=128 rows x 16 codewords of the needed
  // half-superblock); ThreadCount threads, CW_PER_THREAD each. Bit-exact to
  // cb_expand_fp8 / the Triton expander (same window extraction, same LUT).
  static constexpr int CwPerTile = TileN * 16;
  static constexpr int CwPerThread = CwPerTile / ThreadCount;
  static_assert(CwPerTile % ThreadCount == 0, "codeword count must divide thread count");

  template <class SBDecTensor>
  CUTLASS_DEVICE void
  decode_stage(TensorStorage& shared_tensors, SBDecTensor& sBx,
               uint8_t const* __restrict__ lut, int read_stage, int half,
               int thread_idx) const {
    const uint8_t* stage_base =
        shared_tensors.smem_BP.data() + read_stage * (TileN * CbTypeSize);
    const uint16_t* lut16 = reinterpret_cast<const uint16_t*>(lut);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < CwPerThread; ++i) {
      const int linear = thread_idx + i * ThreadCount;
      const int r = linear & (TileN - 1);
      const int vl = linear / TileN;             // 0..15
      const int v = half * 16 + vl;              // codeword index in superblock
      const uint32_t* row32 =
          reinterpret_cast<const uint32_t*>(stage_base + r * CbTypeSize);
      const int bitpos = v * CbKBits;
      const int b0 = bitpos >> 3;
      const int rem = ((b0 & 3) << 3) + (bitpos & 7);
      const int widx = b0 >> 2;
      const uint32_t w0 = row32[widx];
      const uint32_t w1 = row32[widx + 1];
      const uint32_t w2 = row32[widx + 2];
      const uint64_t lo = ((uint64_t)w1 << 32) | (uint64_t)w0;
      uint64_t code = lo >> rem;
      if (rem + CbKBits > 64) {
        code |= (uint64_t)w2 << (64 - rem);
      }
      code &= (CbKBits >= 64) ? ~0ull : ((1ull << CbKBits) - 1ull);

      uint64_t out8 = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < 4; ++s) {
        const uint32_t idx = (uint32_t)(code >> (s * CbSubW)) & CbSubMask;
        const uint16_t pair = __ldg(lut16 + (s << CbSubW) + idx);
        out8 |= (uint64_t)pair << (16 * s);
      }
      const int c0 = vl * 8;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        sBx(r, c0 + j) = (uint8_t)(out8 >> (8 * j));
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
    auto smem_thr_copy_B   = smem_thr_copy_B_init(smem_tiled_copy_B, thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

    // Element-addressable view of the single decoded buffer for the decode
    // writes (position-independent swizzle so (row, col) indexing is exact).
    Tensor sBdec = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()),
                               SmemLayoutB{}(_,_,Int<0>{}));
    auto sBx = as_position_independent_swizzle_tensor(sBdec);

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
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));
    };
    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
    };

    int t_abs = 0;
    pipeline.consumer_wait(smem_pipe_read);
    decode_stage(shared_tensors, sBx, mainloop_params.ptr_lut, read_stage,
                 t_abs & 1, thread_idx);
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
          decode_stage(shared_tensors, sBx, mainloop_params.ptr_lut, read_stage,
                       t_abs & 1, thread_idx);
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
  template <class TiledCopyB>
  CUTLASS_DEVICE static auto
  smem_thr_copy_B_init(TiledCopyB& tiled_copy_b, int thread_idx) {
    return tiled_copy_b.get_thread_slice(thread_idx);
  }
};

}  // namespace cutlass::gemm::collective
