// HOST-ONLY smem budget probe for the grouped (MoE) fused CB GEMM.
//
// A standalone main() developer tool, NOT a serving source: nothing loads it,
// and everything under csrc/tools/ is kept in the repo and the sdist but
// excluded from the wheel (see pyproject.toml / MANIFEST.in). Its OUTPUT is
// what ships -- the smem table baked into cb_fused_gemm.cu -- so re-run it
// whenever a tile shape, stage count, or the collective's storage changes.
//
// Compiles the SAME collective/epilogue types cb_fused_gemm.cu instantiates,
// for TileM x k_bits, and printf()s the SharedStorage byte sizes. There is NO
// kernel launch and NO CUDA runtime call, so this binary runs on a GPU-less
// box. It deliberately does NOT instantiate GemmUniversal for infeasible
// configs (the kernel layer static_asserts SharedStorageSize <=
// sm120_smem_capacity_bytes, which would be a compile error rather than a
// report); instead it sums the two SharedStorage blocks the kernel layer sums:
//
//   SharedStorageSize ~= sizeof(Epilogue::TensorStorage)
//                      + sizeof(Mainloop::TensorStorage)
//                      + pipeline/scheduler storage
//
// build (from the repo root; -I must name the csrc directory itself, since the
// cutlass_fork/ includes below are relative to it):
//   nvcc -std=c++17 -arch=sm_120a -O3 --expt-relaxed-constexpr \
//     -I$CUTLASS/include -I$CUTLASS/tools/util/include -Igridbook/csrc \
//     -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
//     -I/usr/include/python3.12 gridbook/csrc/tools/smem_probe_tilem.cu -o probe
#include <cstdio>
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

#include "cutlass_fork/sm120_cb_fused_mma.hpp"
#include "cutlass_fork/sm120_expert_row_broadcast.hpp"
// The expert-indexed EVT tree comes from the same shared header the serving
// kernels use, so this probe can never report the sizes of a DIFFERENT
// epilogue than the one cb_fused_gemm.cu instantiates (2026-08-01 audit §4
// dedupe #2). It is why the build line above needs the torch/python includes.
#include "cb_grouped_common.hpp"

namespace {

using namespace cute;

using ElementAB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignAB = 16;
constexpr int AlignD = 8;
using ClusterShape = Shape<_1, _1, _1>;

template <class TileShape>
using MoeScaledFusion = gridbook::grouped::MoeScaledFusion<TileShape,
                                                           ElementAcc, ElementD>;

template <class TileShape>
struct CfgMoeScaled {
  using Fusion = typename MoeScaledFusion<TileShape>::type;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      void, LayoutD, AlignD,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      Fusion>::CollectiveOp;
  using BuilderMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementAB, LayoutA, AlignAB,
      ElementAB, LayoutB, AlignAB,
      ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

template <int KB, int NewStages, class T>
struct SwapToFused;
template <int KB, int NewStages, int S, int SP, class CS, class KS, class... Rest>
struct SwapToFused<KB, NewStages, cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm120TmaWarpSpecialized<S, SP, CS, KS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120CbFusedTmaWarpSpecialized<NewStages, SP, CS, KS, KB>,
      Rest...>;
};

template <int TM, int KB>
void row() {
  using TileShape = Shape<Int<TM>, _64, _128>;
  using Cfg = CfgMoeScaled<TileShape>;
  using Mainloop = typename SwapToFused<KB, 2, typename Cfg::BuilderMainloop>::type;
  using Epi = typename Cfg::Epilogue;

  const int ml_tensors = (int)sizeof(typename Mainloop::TensorStorage);
  const int ml_pipe = (int)sizeof(typename Mainloop::PipelineStorage);
  const int ml_total = (int)sizeof(typename Mainloop::SharedStorage);
  const int epi_total = (int)sizeof(typename Epi::SharedStorage);
  const int epi_tensors = (int)sizeof(typename Epi::TensorStorage);
  // kernel layer SUMS epilogue tensors + mainloop tensors + pipelines.
  const int est = ml_tensors + epi_tensors + ml_pipe +
                  (int)sizeof(typename Epi::PipelineStorage);
  printf("TileM=%3d k_bits=%2d | mlTensor=%6d mlPipe=%4d mlTotal=%6d | "
         "epiTensor=%5d epiTotal=%5d | SUM~=%6d  limit=%d  %s\n",
         TM, KB, ml_tensors, ml_pipe, ml_total, epi_tensors, epi_total, est,
         cutlass::arch::sm120_smem_capacity_bytes,
         est <= cutlass::arch::sm120_smem_capacity_bytes ? "FIT" : "OVER");
}

// THE RUNG LAW (K1.2; v10 producer extension 2026-08-24).
//
// v10 producers emit K4..K48 step 4. The accepted reader domain also retains
// every integer K28..K48 for legacy artifacts. The fused mid-M collective can
// serve only the multiples of 4, a property of the FORMAT and of TMA:
//
//   1. TMA BOX. Packed B is read by SM90_TMA_LOAD with a box of
//      (TileN, CbTypeSize) bytes over the [N, n_sb*CbTypeSize] byte stream,
//      where CbTypeSize = 4*k_bits is one 256-weight superblock. TMA requires
//      the box's contiguous extent to be a 16-byte multiple, i.e.
//      4*k_bits % 16 == 0  <=>  k_bits % 4 == 0.
//      (sm120_cb_fused_mma.hpp: `static_assert(CbTypeSize % 16 == 0`.)
//   2. UNIFORM SUB-TABLE WIDTH. The collective decodes a codeword with
//      CbSubW = KBits/4 and indexes the flat codebook at `(s << CbSubW) + idx`
//      -- one width, one stride, for all four sub-tables. The FORMAT splits
//      k_bits over n_sub=4 RAGGEDLY (csrc/cb_gemv.cu `SubSplit`:
//      `base = k_bits/NSUB, extra = k_bits%NSUB`, width_i = base + (i<extra)),
//      so at k_bits=37 the true widths are (10,9,9,9) with non-uniform table
//      offsets. A uniform-width decode would therefore be WRONG, not merely
//      unaligned, for every rung with k_bits % 4 != 0.
//
// Both laws bite at the same place, so the probe walks the canonical producer
// rungs. Legacy irregular reader rungs cannot instantiate this collective and
// use the generic decode/expand paths instead.
constexpr int kKbLo = 4;
constexpr int kKbHi = 48;
constexpr int kKbStep = 4;
constexpr int kKbCount = (kKbHi - kKbLo) / kKbStep + 1;

template <int TM, int... I>
void block_impl(std::integer_sequence<int, I...>) {
  (row<TM, kKbLo + kKbStep * I>(), ...);
}

template <int TM>
void block() {
  block_impl<TM>(std::make_integer_sequence<int, kKbCount>{});
  printf("\n");
}

// EXACT kernel-layer size -- the number the compiled matrix is decided on.
//
// CORRECTED 2026-08-02: an earlier revision of this comment claimed `exact` was
// a COMPILE-TIME oracle (that an over-budget cell would fail to instantiate).
// It is not, and cb_fused_gemm.cu says why: the sm90-family cooperative kernel
// layer does NOT static_assert its own SharedStorageSize against the arch
// capacity -- only the sm120 asymmetric-DMA kernel does. Measured: `exact<256,
// 48>` compiles cleanly and prints 109,568 (OVER). The serving TU's gate is
// `gridbook::grouped::AssertSmemFits` (cb_grouped_common.hpp), which is where
// an over-budget instantiation actually becomes a compile error. So the
// verdict to read here is the PRINTED FIT/OVER, and every cell is instantiated.
template <int TM, int KB>
void exact() {
  using TileShape = Shape<Int<TM>, _64, _128>;
  using Cfg = CfgMoeScaled<TileShape>;
  using Mainloop = typename SwapToFused<KB, 2, typename Cfg::BuilderMainloop>::type;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, typename Cfg::Epilogue>;
  printf("EXACT TileM=%3d k_bits=%2d  SharedStorageSize=%6d  limit=%d  %s\n",
         TM, KB, (int)GemmKernel::SharedStorageSize,
         cutlass::arch::sm120_smem_capacity_bytes,
         (int)GemmKernel::SharedStorageSize <=
                 cutlass::arch::sm120_smem_capacity_bytes ? "FIT" : "OVER");
}

template <int TM, int... I>
void exact_block_impl(std::integer_sequence<int, I...>) {
  (exact<TM, kKbLo + kKbStep * I>(), ...);
}

// Exact sizes for a CLOSED rung range [kKbLo, HI] on the law's step. Because
// `exact` is only instantiable for feasible cells, "the largest HI for which
// this TU compiles" IS the compiled-matrix boundary for that TileM.
template <int TM, int HI>
void exact_block() {
  exact_block_impl<TM>(
      std::make_integer_sequence<int, (HI - kKbLo) / kKbStep + 1>{});
}

}  // namespace

// Feasibility oracle knobs (see the K1.2 rung-coverage note in
// cb_fused_gemm.cu). Recompiling with -DPROBE_256_HI=<k> is the ONLY honest way
// to establish the TileM=256 boundary: the kernel layer static_asserts its own
// SharedStorageSize against the arch capacity, so an over-budget cell is a
// COMPILE ERROR here rather than a printed "OVER" line. Bisect it: the largest
// value of PROBE_256_HI that compiles is the last rung TileM=256 admits.
#ifndef PROBE_256_HI
#define PROBE_256_HI 32
#endif

int main() {
  printf("sm120_smem_capacity_bytes = %d\n\n",
         cutlass::arch::sm120_smem_capacity_bytes);
  block<128>();
  block<256>();
  block<384>();
  // TileM=128 is feasible across the whole range, so the exact oracle runs the
  // full compiled ladder there.
  exact_block<128, kKbHi>();
#ifndef PROBE_NO_256
  exact_block<256, PROBE_256_HI>();
#endif
  return 0;
}
