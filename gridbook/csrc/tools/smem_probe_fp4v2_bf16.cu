// HOST-ONLY smem budget probe for the fp4-v2 quality fused mid-M lane
// (csrc/cb_fused_fp4v2_gemm.cu + cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp).
//
// Sibling of smem_probe_tilem.cu, same discipline: a standalone main()
// developer tool, NOT a serving source. Nothing loads it; everything under
// csrc/tools/ stays in the repo and the sdist but is excluded from the wheel.
// Its OUTPUT is what ships — the smem feasibility table baked into
// cb_fused_fp4v2_gemm.cu — so re-run it whenever a tile shape, stage count,
// codebook-stage size or the collective's storage changes.
//
// It instantiates the SAME hand-assembled 16-bit collective and epilogue the
// serving TU does, for (TileN x TileK x Stages x LutBytes), and printf()s
// `GemmKernel::SharedStorageSize`. There is NO kernel launch and NO CUDA
// runtime call, so this binary runs on a GPU-less box. Instantiating
// GemmUniversal is itself the feasibility oracle: the kernel layer
// static_asserts SharedStorageSize <= sm120_smem_capacity_bytes only for the
// asymmetric-DMA kernel, so the sizes are printed and compared here instead
// (cb_grouped_common.hpp::AssertSmemFits is what makes it a compile error in
// the serving TU).
//
// build (from the repo root; -I must name the csrc directory itself, since the
// cutlass_fork/ include below is relative to it):
//   nvcc -std=c++17 -arch=sm_120a -O3 --expt-relaxed-constexpr \
//     -I$CUTLASS/include -I$CUTLASS/tools/util/include -Igridbook/csrc \
//     gridbook/csrc/tools/smem_probe_fp4v2_bf16.cu -o probe_fp4v2
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp"

namespace {

using namespace cute;

namespace cutlass_detail = cutlass::gemm::collective::detail;

using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
constexpr int kAlign = 8;
using ClusterShape = Shape<_1, _1, _1>;

// Verbatim the serving TU's operand selection (see cb_fused_fp4v2_gemm.cu for
// why the collective is hand-assembled).
template <class TileShape>
struct MmaCfg {
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
      cute::Layout<cute::Shape<_4, _2, _1>>{},
      cute::Tile<decltype(cute::min(size<0>(TileShape{}), _128{})), _32, _16>{}));
  using SmemLayoutAtomA = decltype(cutlass_detail::rs_smem_selector<
      cute::GMMA::Major::K, Element,
      decltype(cute::get<0>(TileShape{})),
      decltype(cute::get<2>(TileShape{}))>());
  using SmemLayoutAtomB = decltype(cutlass_detail::rs_smem_selector<
      cute::GMMA::Major::K, Element,
      decltype(cute::get<1>(TileShape{})),
      decltype(cute::get<2>(TileShape{}))>());
  using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
};

template <class TileShape, int Stages, int LutBytes>
struct Cfg {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      void, LayoutC, kAlign,
      Element, LayoutC, kAlign,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using DispatchPolicy =
      cutlass::gemm::MainloopSm120CbFp4V2Bf16TmaWarpSpecialized<
          Stages, 2, ClusterShape,
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<2>, LutBytes>;

  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy, TileShape,
      Element, cutlass::gemm::TagToStrideA_t<LayoutA>,
      Element, cutlass::gemm::TagToStrideB_t<LayoutB>,
      typename MmaCfg<TileShape>::TiledMma,
      cute::SM90_TMA_LOAD, typename MmaCfg<TileShape>::SmemLayoutAtomA,
      typename MmaCfg<TileShape>::SmemCopyAtom, cute::identity,
      cute::SM90_TMA_LOAD, typename MmaCfg<TileShape>::SmemLayoutAtomB,
      typename MmaCfg<TileShape>::SmemCopyAtom, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
};

template <int TN, int TK, int Stages, int LutBytes>
void row() {
  using TileShape = Shape<_128, Int<TN>, Int<TK>>;
  using C = Cfg<TileShape, Stages, LutBytes>;
  const int total = (int)C::GemmKernel::SharedStorageSize;
  const int cap = cutlass::arch::sm120_smem_capacity_bytes;
  printf("TileM=128 TileN=%3d TileK=%3d Stages=%d Lut=%5d | "
         "mlTensor=%6d epiTensor=%5d | SharedStorageSize=%6d  cap=%d  %-4s  "
         "ctas/sm(cap)=%d\n",
         TN, TK, Stages, LutBytes,
         (int)sizeof(typename C::Mainloop::TensorStorage),
         (int)sizeof(typename C::Epilogue::TensorStorage),
         total, cap, total <= cap ? "FIT" : "OVER",
         total > 0 ? cap / total : 0);
}

template <int TN, int TK, int Stages>
void lut_block() {
  row<TN, TK, Stages, 0>();
  row<TN, TK, Stages, 4096>();
  row<TN, TK, Stages, 16384>();
  row<TN, TK, Stages, 32768>();
  row<TN, TK, Stages, 49152>();
  printf("\n");
}

}  // namespace

int main() {
  printf("sm120_smem_capacity_bytes = %d\n\n",
         cutlass::arch::sm120_smem_capacity_bytes);
  // The shipped shape family: narrow-N mid-M (the fp8 fused lane's proven
  // pattern) at the two K depths a 128-byte / 256-byte swizzle atom admits.
  lut_block<64, 64, 2>();
  lut_block<64, 64, 3>();
  lut_block<64, 128, 2>();
  lut_block<128, 64, 2>();
  return 0;
}
