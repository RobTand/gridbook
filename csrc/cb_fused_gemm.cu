// Fused-prefill workstream (Task 7).
//
// MEASURED VERDICT (2026-07-19, GB10, 27B shapes; bench under bench-lock):
//   * v1 decode-in-prologue at M=1400: 0.22x of the serial transient
//     (36.0 vs 8.1 ms/layer-set). NOT a bug — structural: every M-tile CTA
//     re-decodes the same B tiles, so decode work = ceil(M/128) x the
//     transient's one-shot expand (11 x 2.7 ms + 5.5 ms GEMM ~= the
//     measured 36 ms exactly). Overlapping decode with MMA cannot fix a
//     ~10x compute redundancy.
//   * chunked expand + fork-GEMM overlap (bit-safe with OUR fixed-config
//     kernel, unlike cutlass_scaled_mm): 0.74-0.79x of serial — on the
//     ~273 GB/s unified-memory part the M=1400 GEMM is already partially
//     memory-bound, so the expander has no spare bandwidth to hide in, and
//     narrow-N chunks lose GEMM efficiency.
//   * fused at M<=128 (ONE M-tile -> no redundancy): WINS — 1.04x / 1.26x /
//     1.45x vs serial at M=32/64/128. This is the fused kernel's honest
//     serving niche today: the mid-M band (17..128) between the decode GEMV
//     and the transient path. Serving dispatch not yet wired (default path
//     unchanged); see tests/test_fused_prefill.py for the bit-exact gates.
//   * The large-M endgame requires a weight-stationary/persistent-N
//     schedule (decode each B tile ONCE, loop M inside the CTA) — a
//     kernel-layer restructure, not a collective fork.
//
// Entry points:
//  - sm120_fp8_mm_fork:  128x128x128 passthrough through the UNCHANGED forked
//    collective (fork-without-change gate, kept as regression).
//  - sm120_fp8_mm_fork64: same passthrough at 64x128x128 — the bit-exactness
//    REFERENCE for the fused kernel (identical TiledMma/layout config).
//  - cb_fused_prefill_mm: the decode-in-prologue FP8_CB GEMM — B is the
//    PACKED byte stream + a global e4m3-byte LUT; the dense tile never
//    exists in HBM. KBits in {36,40,44,48} template-dispatched.
//  - smem_report: per-config SharedStorage sizes (budget sanity).
//
// All entries return the UNSCALED bf16 accumulation (per-token x per-channel
// scales applied outside, as the baseline gate established).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_fork/sm120_cb_mma_tma.hpp"
#include "cutlass_fork/sm120_cb_fused_mma.hpp"

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

// ---------------------------------------------------------------------------
// Config factory: builder collective + epilogue at a given tile shape, plus
// rebinders onto the passthrough / fused policies.
// ---------------------------------------------------------------------------
template <class TileShape>
struct Cfg {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      void, LayoutD, AlignD,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
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

template <class T>
struct SwapToCb;
template <int S, int SP, class CS, class KS, class... Rest>
struct SwapToCb<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm120TmaWarpSpecialized<S, SP, CS, KS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120CbTmaWarpSpecialized<S, SP, CS, KS>, Rest...>;
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

// ---------------------------------------------------------------------------
// Passthrough runner (dense fp8 B) — shared by the 128- and 64-tile entries.
// ---------------------------------------------------------------------------
template <class TileShape, class Mainloop>
torch::Tensor run_dense(torch::Tensor a, torch::Tensor b) {
  using Epilogue = typename Cfg<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1));
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(K % AlignAB == 0);
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == K);
  TORCH_CHECK(b.stride(1) == 1 && b.stride(0) == K);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const ElementAB*>(a.data_ptr()), sa,
       reinterpret_cast<const ElementAB*>(b.data_ptr()), sb},
      {{1.0f, 0.0f}, nullptr, StrideD{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "can_implement failed");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
  return d;
}

using Tile128 = Shape<_128, _128, _128>;
// Fused config: cooperative kernel requires TileM >= 128; the smem
// budget (A stages + packed stages + decoded buffer + additive epilogue)
// is met by narrowing N instead: 128x64x128.
using TileF = Shape<_128, _64, _128>;
using Fork128 = typename SwapToCb<typename Cfg<Tile128>::BuilderMainloop>::type;
using ForkF = typename SwapToCb<typename Cfg<TileF>::BuilderMainloop>::type;

torch::Tensor sm120_fp8_mm_fork(torch::Tensor a, torch::Tensor b) {
  return run_dense<Tile128, Fork128>(a, b);
}
torch::Tensor sm120_fp8_mm_fork64(torch::Tensor a, torch::Tensor b) {
  return run_dense<TileF, ForkF>(a, b);
}

// ---------------------------------------------------------------------------
// Fused decode-in-prologue entry.
// ---------------------------------------------------------------------------
template <int KB>
torch::Tensor run_fused(torch::Tensor a, torch::Tensor packed,
                        torch::Tensor lut, int64_t N, int64_t K) {
  constexpr int Stages = 2;
  using TileShape = TileF;
  using Mainloop = typename SwapToFused<
      KB, Stages, typename Cfg<TileShape>::BuilderMainloop>::type;
  using Epilogue = typename Cfg<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  const int M = (int)a.size(0);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, (int)N, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, (int)N, (int)K, 1},
      {reinterpret_cast<const ElementAB*>(a.data_ptr()), sa,
       packed.data_ptr<uint8_t>(), packed.stride(0),
       lut.data_ptr<uint8_t>()},
      {{1.0f, 0.0f}, nullptr, StrideD{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fused can_implement failed (K%256? row stride %16?)");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
  return d;
}

torch::Tensor cb_fused_prefill_mm(torch::Tensor a, torch::Tensor packed,
                                  torch::Tensor lut, int64_t N, int64_t K,
                                  int64_t k_bits) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn,
              "a must be fp8 e4m3 [M,K]");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
              a.stride(0) == K);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
              packed.dim() == 2 && packed.size(0) == N &&
              packed.stride(1) == 1);
  TORCH_CHECK(packed.stride(0) % 16 == 0,
              "packed row stride must be a 16-byte multiple (UNPADDED rows)");
  TORCH_CHECK(packed.stride(0) >= (K / 256) * 4 * k_bits);
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kUInt8);
  TORCH_CHECK(K % 256 == 0);
  switch (k_bits) {
    case 36: return run_fused<36>(a, packed, lut, N, K);
    case 40: return run_fused<40>(a, packed, lut, N, K);
    case 44: return run_fused<44>(a, packed, lut, N, K);
    case 48: return run_fused<48>(a, packed, lut, N, K);
    default: TORCH_CHECK(false, "unsupported k_bits ", k_bits);
  }
}

std::vector<int64_t> smem_report() {
  using F44 = typename SwapToFused<44, 2, typename Cfg<TileF>::BuilderMainloop>::type;
  using F48 = typename SwapToFused<48, 2, typename Cfg<TileF>::BuilderMainloop>::type;
  using K44 = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, F44, typename Cfg<TileF>::Epilogue>;
  using K48 = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, F48, typename Cfg<TileF>::Epilogue>;
  return {(int64_t)sizeof(typename F44::SharedStorage),
          (int64_t)sizeof(typename F48::SharedStorage),
          (int64_t)K44::SharedStorageSize,
          (int64_t)K48::SharedStorageSize,
          (int64_t)sizeof(typename Cfg<TileF>::Epilogue::SharedStorage)};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm120_fp8_mm_fork", &sm120_fp8_mm_fork,
        "sm120 fp8 GEMM through the FORKED collective (128-tile passthrough)");
  m.def("sm120_fp8_mm_fork64", &sm120_fp8_mm_fork64,
        "passthrough at 64x128x128 (fused kernel's reference config)");
  m.def("cb_fused_prefill_mm", &cb_fused_prefill_mm,
        "FP8_CB decode-in-prologue fused GEMM (unscaled bf16 out)");
  m.def("smem_report", &smem_report, "SharedStorage sizes [F44, F48, K44, K48, Epi]");
}
