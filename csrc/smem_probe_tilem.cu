// HOST-ONLY smem budget probe for the grouped (MoE) fused CB GEMM.
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
// build:
//   nvcc -std=c++17 -arch=sm_120a -O3 --expt-relaxed-constexpr \
//     -I$CUTLASS/include -I$CUTLASS/tools/util/include -Icsrc \
//     -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
//     -I/usr/include/python3.12 smem_probe_tilem.cu -o probe
#include <cstdio>

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
struct MoeScaledFusion {
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, TileShape, float, float, Stride<_1, _0, _0>>;
  using ScaleB = cutlass::epilogue::fusion::Sm120CbExpertRowBroadcast<
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

template <int TM>
void block() {
  row<TM, 28>(); row<TM, 32>(); row<TM, 36>();
  row<TM, 40>(); row<TM, 44>(); row<TM, 48>();
  printf("\n");
}

// EXACT kernel-layer size. Only instantiable for configs that pass the kernel's
// own `SharedStorageSize <= sm120_smem_capacity_bytes` static_assert, so this is
// itself the feasibility oracle: a config that does not compile here is
// infeasible, full stop.
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

}  // namespace

int main() {
  printf("sm120_smem_capacity_bytes = %d\n\n",
         cutlass::arch::sm120_smem_capacity_bytes);
  block<128>();
  block<256>();
  block<384>();
  exact<128, 28>(); exact<128, 32>(); exact<128, 36>();
  exact<128, 40>(); exact<128, 44>(); exact<128, 48>();
#ifndef PROBE_NO_256
  exact<256, 28>(); exact<256, 32>();
#endif
#ifdef PROBE_256_HIGH
  exact<256, 36>(); exact<256, 40>(); exact<256, 44>(); exact<256, 48>();
#endif
  return 0;
}
