// Native CUTLASS grouped BF16 GEMM used by the quality-preserving FP4-CB MoE
// bridge.  The packed FP4-v2 weights are decoded to BF16 by Gridbook's native
// CUDA expander before this op; activations are likewise the exact BF16 result
// of Gridbook's group-16 QDQ.  This kernel therefore changes only the GEMM
// launch topology: one device-scheduled CUTLASS group instead of one GEMM per
// expert.  Accumulation is FP32 and the epilogue rounds once to BF16.
//
// TWO LANES LIVE HERE (2026-08-01 performance audit, §3 P1):
//
//  1. `cb_bf16_grouped_mm[_out]` — the ORIGINAL device-scheduled CUTLASS 2.x
//     `DefaultGemmGrouped` on an `arch::Sm80` schedule (Ampere `m16n8k16`, no
//     TMA, no warp specialization). It consumes EXACT per-expert segments via
//     cumulative `expert_ends`, needs no padding, and runs on every device from
//     cc 8.0 up. It is compiled on every device and stays the DEFAULT.
//
//  2. `cb_bf16_grouped_mm_sm120[_out]` — the sm12x-NATIVE lane: a CUTLASS 3.x
//     collective with a TMA warp-specialized mainloop, stages carved out of the
//     sm120 smem budget, and the row-padded TILE-INDEXED grouping Gridbook's
//     two fused kernels already use (see cb_grouped_common.hpp). Compiled only
//     when the loader defines `PRISMAQUANT_CB_BF16_SM120` (cc 12.x), and
//     dispatched only behind `PRISMAQUANT_CB_BF16_SM120=1` — OPT-IN, per
//     docs/NATIVE-PARITY.md.
//
// NUMERICS, both lanes: FP32 accumulate, alpha=1/beta=0, ONE round to BF16 in
// the epilogue. There are no scales in this bridge and none are added. What
// differs between the lanes is the REDUCTION ORDER of the FP32 accumulation
// (different tile shape, different K-iteration, different warp partitioning) —
// that difference, and nothing else, is the requalification surface for the
// new lane. It is the same class of change the promoted FP8 mid-M fused kernel
// cleared: bit-level unit gates first, then the NATIVE-PARITY served protocol.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#if defined(PRISMAQUANT_CB_BF16_SM120)
#include <cctype>
#include <cstdlib>
#include <string>

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_fork/sm120_bf16_expert_mma.hpp"
#include "cb_grouped_common.hpp"
#endif

namespace {

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int kAlignment = 8;  // one 128-bit BF16 access
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element, kAlignment, Accumulator, Accumulator>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, LayoutA, cutlass::ComplexTransform::kNone, kAlignment,
    Element, LayoutB, cutlass::ComplexTransform::kNone, kAlignment,
    Element, LayoutC, Accumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
using GroupedGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

__global__ void prepare_grouped_bf16_args(
    int const* __restrict__ expert_ends,
    int expert_start, int experts, int n, int k,
    Element const* __restrict__ a,
    Element const* __restrict__ b,
    Element* __restrict__ d,
    cutlass::gemm::GemmCoord* __restrict__ problems,
    Element const** __restrict__ ptr_a,
    Element const** __restrict__ ptr_b,
    Element** __restrict__ ptr_c,
    Element** __restrict__ ptr_d,
    int64_t* __restrict__ lda,
    int64_t* __restrict__ ldb,
    int64_t* __restrict__ ldc,
    int64_t* __restrict__ ldd) {
  int e = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (e >= experts) {
    return;
  }
  int global_e = expert_start + e;
  int begin = global_e == 0 ? 0 : expert_ends[global_e - 1];
  int end = expert_ends[global_e];
  problems[e] = cutlass::gemm::GemmCoord(end - begin, n, k);
  ptr_a[e] = a + int64_t(begin) * k;
  ptr_b[e] = b + int64_t(e) * n * k;
  ptr_c[e] = d + int64_t(begin) * n;
  ptr_d[e] = d + int64_t(begin) * n;
  lda[e] = k;
  ldb[e] = k;
  ldc[e] = n;
  ldd[e] = n;
}

void check_and_run_grouped_mm(torch::Tensor output,
                              torch::Tensor a,
                              torch::Tensor weights,
                              torch::Tensor expert_ends,
                              int64_t expert_start64) {
  TORCH_CHECK(a.is_cuda() && weights.is_cuda() && expert_ends.is_cuda(),
              "a, weights and expert_ends must be CUDA tensors");
  TORCH_CHECK(a.device() == weights.device() && a.device() == expert_ends.device(),
              "a, weights and expert_ends must be on one CUDA device");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
              weights.scalar_type() == torch::kBFloat16,
              "a and weights must be BF16");
  TORCH_CHECK(expert_ends.scalar_type() == torch::kInt32,
              "expert_ends must be int32");
  TORCH_CHECK(a.dim() == 2 && weights.dim() == 3 && expert_ends.dim() == 1,
              "expected a [P,K], weights [E,N,K], expert_ends [E]");

  int64_t p64 = a.size(0);
  int64_t k64 = a.size(1);
  int64_t e64 = weights.size(0);
  int64_t n64 = weights.size(1);
  int64_t global_e64 = expert_ends.size(0);
  TORCH_CHECK(weights.size(2) == k64,
              "shape mismatch: a [P,K] and weights [E,N,K]");
  TORCH_CHECK(expert_start64 >= 0 && expert_start64 <= global_e64 &&
              e64 <= global_e64 - expert_start64,
              "expert range [", expert_start64, ", ",
              expert_start64 + e64, ") is outside expert_ends [",
              global_e64, "]");
  TORCH_CHECK(p64 <= std::numeric_limits<int>::max() &&
              n64 <= std::numeric_limits<int>::max() &&
              k64 <= std::numeric_limits<int>::max() &&
              e64 <= std::numeric_limits<int>::max() &&
              expert_start64 <= std::numeric_limits<int>::max(),
              "grouped GEMM dimensions exceed int32");
  TORCH_CHECK(k64 % kAlignment == 0 && n64 % kAlignment == 0,
              "K and N must be multiples of 8 BF16 elements");
  TORCH_CHECK(a.is_contiguous() && weights.is_contiguous() &&
              expert_ends.is_contiguous(), "all inputs must be contiguous");
  TORCH_CHECK(output.is_cuda() && output.device() == a.device() &&
              output.scalar_type() == torch::kBFloat16 &&
              output.dim() == 2 && output.size(0) == p64 &&
              output.size(1) == n64 && output.is_contiguous(),
              "output must be a contiguous CUDA BF16 [P,N] tensor on a's device");

  int p = int(p64), k = int(k64), experts = int(e64), n = int(n64);
  int expert_start = int(expert_start64);
  if (p == 0 || experts == 0) {
    return;
  }

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto i32 = a.options().dtype(torch::kInt32);
  auto i64 = a.options().dtype(torch::kInt64);
  auto problems = torch::empty({e64, 3}, i32);
  auto ptr_a = torch::empty({e64}, i64);
  auto ptr_b = torch::empty({e64}, i64);
  auto ptr_c = torch::empty({e64}, i64);
  auto ptr_d = torch::empty({e64}, i64);
  auto lda = torch::empty({e64}, i64);
  auto ldb = torch::empty({e64}, i64);
  auto ldc = torch::empty({e64}, i64);
  auto ldd = torch::empty({e64}, i64);

  int threads = 128;
  int blocks = (experts + threads - 1) / threads;
  prepare_grouped_bf16_args<<<blocks, threads, 0, stream>>>(
      expert_ends.data_ptr<int>(), expert_start, experts, n, k,
      reinterpret_cast<Element const*>(a.data_ptr()),
      reinterpret_cast<Element const*>(weights.data_ptr()),
      reinterpret_cast<Element*>(output.data_ptr()),
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problems.data_ptr()),
      reinterpret_cast<Element const**>(ptr_a.data_ptr<int64_t>()),
      reinterpret_cast<Element const**>(ptr_b.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_c.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
      lda.data_ptr<int64_t>(), ldb.data_ptr<int64_t>(),
      ldc.data_ptr<int64_t>(), ldd.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int threadblock_count = GroupedGemm::sufficient();
  TORCH_CHECK(threadblock_count > 0,
              "CUTLASS grouped BF16 kernel has zero occupancy");
  typename GroupedGemm::Arguments args(
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problems.data_ptr()),
      experts, threadblock_count,
      typename Epilogue::Params(Accumulator(1), Accumulator(0)),
      reinterpret_cast<Element**>(ptr_a.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_b.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_c.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
      lda.data_ptr<int64_t>(), ldb.data_ptr<int64_t>(),
      ldc.data_ptr<int64_t>(), ldd.data_ptr<int64_t>());
  TORCH_CHECK(GroupedGemm::get_workspace_size(args) == 0,
              "device-scheduled CUTLASS grouped BF16 unexpectedly requested "
              "workspace; this binding does not permit a host-precomputed "
              "scheduler");

  GroupedGemm gemm;
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS grouped BF16 can_implement failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.initialize(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS grouped BF16 initialize failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS grouped BF16 run failed: ",
              cutlass::cutlassGetStatusString(status));
}

torch::Tensor cb_bf16_grouped_mm(torch::Tensor a,
                                 torch::Tensor weights,
                                 torch::Tensor expert_ends,
                                 int64_t expert_start) {
  TORCH_CHECK(weights.dim() == 3,
              "weights must have shape [E,N,K]");
  auto output = torch::zeros({a.size(0), weights.size(1)}, a.options());
  check_and_run_grouped_mm(output, a, weights, expert_ends, expert_start);
  return output;
}

void cb_bf16_grouped_mm_out(torch::Tensor output,
                            torch::Tensor a,
                            torch::Tensor weights,
                            torch::Tensor expert_ends,
                            int64_t expert_start) {
  check_and_run_grouped_mm(output, a, weights, expert_ends, expert_start);
}

// ===========================================================================
// LANE 2 — sm12x-native CUTLASS 3.x collective, row-padded tile-indexed
// grouping.  Compiled only for cc 12.x (loader-defined macro).
// ===========================================================================
#if defined(PRISMAQUANT_CB_BF16_SM120)

// ---------------------------------------------------------------------------
// WHY THE COLLECTIVE IS HAND-BUILT.
//
// Upstream CUTLASS 4.3.4's sm120 dense CollectiveBuilder refuses 16-bit input:
// `sm120_mma_builder.inl` carries
//   static_assert(is_sm10x_f8f6f4_element<ElementA>() && ...,
//                 "SM120 TmaWarpSpecialized builder currently only supports
//                  F8F6F4 MMA.")
// and its `rr_op_selector_sm120` unconditionally returns the 8-bit
// `SM120_16x8x32_TN` atom. The MAINLOOP, however, is type-generic — it
// branches on `IsF8F6F4` for its smem allocation type and its fp4 shifts are
// no-ops for every other atom — so what is missing is only the builder's
// operand selection. We therefore assemble the same four choices the builder
// makes for f8f6f4, in their 16-bit forms, and hand them to the (forked)
// mainloop:
//
//   MMA atom      SM80_16x8x16_F32BF16BF16F32_TN — the bf16 tensor-core
//                 instruction on sm_120; rmem-sourced, which is exactly what
//                 this mainloop requires (it static_asserts no GMMA
//                 descriptor iterators).
//   Atom layout   2x2x1 warps = 128 threads, the builder's PINGPONG shape.
//                 (Its cooperative shape is 4x2x1 = 256 threads, whose kernel
//                 layer floors TileM at 128 — see the tile choice below.)
//   Permutation   Tile<_64,_32,_16>: M covers the tile with the 2-warp row,
//                 and N is widened to 32 (2 n-atoms) so ONE ldmatrix.x4 fills
//                 a thread's B fragment — the same reason upstream widens
//                 PermTileN to 32 for 8-bit.
//   Smem atom     rs_smem_selector<K-major> — CUTLASS's own selector for a
//                 swizzled K-major tile that TMA writes and LDSM reads (the
//                 sm90 "RS" mainloop uses it for precisely this pairing).
//
// Everything downstream (TMA descriptors, pipeline, tile scheduler, epilogue)
// is stock CUTLASS. The stage count comes from the same
// StageCountAutoCarveout helper the builder uses, against the sm120 smem
// budget with the epilogue's storage carved out.
//
// WHAT THE MEASUREMENTS SAID (GB10, cc 12.1; docs/BENCHMARKS.md has the
// tables). The operator is bound by B traffic: this construction re-reads an
// expert's B slice once per PADDED M-tile, so cost tracks the padded tile
// count, and the ragged rounding of each expert's rows up to TileM is the
// whole tax. With the padding removed (a synthetic routing whose expert counts
// are exact tile multiples) this collective runs at 1.08-1.13x segmented
// cuBLAS and 1.05-1.24x the SM80 lane — the SCHEDULE is not the problem. What
// closes most of the remaining gap is halving the rounding granularity, which
// is why the compiled rung is PINGPONG at TileM=64 rather than cooperative at
// 128.
// ---------------------------------------------------------------------------
namespace sm120_lane {

// NOTE: `using namespace cute;` stays INSIDE this named namespace. cute opens
// its own anonymous namespace (cute/atom/mma_traits_sm70.hpp), so a
// file-scope using-directive would make this TU's anonymous namespace name
// ambiguous with cute's — and nvcc's generated device-stub, which references
// the SM80 lane's __global__ through _NV_ANON_NAMESPACE, then fails to
// compile. Confining it keeps both lanes in one translation unit.
using namespace cute;

namespace cutlass_detail = cutlass::gemm::collective::detail;

using ElementAcc = float;
using ClusterShape = Shape<_1, _1, _1>;

// The bf16 tensor-core atom, 4 warps pingpong, N widened for ldmatrix.x4.
template <class TileShape>
struct MmaCfg {
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>{},
      cute::Layout<cute::Shape<_2, _2, _1>>{},
      cute::Tile<decltype(cute::min(size<0>(TileShape{}), _128{})),
                 _32, _16>{}));
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

template <class TileShape>
struct Cfg {
  // Plain alpha=1/beta=0 epilogue: this bridge has no scales, and adding any
  // would change the served numerics rather than only the reduction order.
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      void, LayoutC, kAlignment,
      Element, LayoutC, kAlignment,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // Stage count: (sm120 capacity - epilogue storage) / per-stage bytes, the
  // builder's own arithmetic. A/B smem is the only per-stage tensor storage
  // here — no packed stream, no decoded buffer — and at 64x128x64 that is
  // 24 KiB/stage, which auto-carves to 3 stages (83,968 B of 101,376).
  static constexpr int kStages =
      cutlass_detail::sm100_compute_stage_count_or_override<
          cutlass_detail::sm120_smem_capacity_bytes,
          Element, Element, TileShape,
          typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage>(
              cutlass::gemm::collective::StageCountAutoCarveout<
                  static_cast<int>(sizeof(
                      typename Epilogue::SharedStorage))>{});
  static_assert(kStages >= 2,
                "the sm120 BF16 collective needs at least two mainloop stages");

  using DispatchPolicy =
      cutlass::gemm::MainloopSm120CbBf16ExpertTmaWarpSpecialized<
          kStages, /*SchedulerPipelineStageCount=*/2, ClusterShape,
          cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<2>>;

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
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// TileM is the row-padding granularity python must respect, and on this
// construction it is the single most important number in the file: an expert's
// rows are rounded UP to it, and every rounded-up tile re-reads that expert's
// whole B slice. TileM=64 is therefore chosen over 128 even though the 128
// tile is the more efficient GEMM per FLOP — measured across the DSV4 and
// Laguna cells at T=128/512, halving the rounding granularity wins by more
// than the tile efficiency loses (BENCHMARKS has both columns).
//
// TileM=64 requires the PINGPONG kernel layer: the cooperative one
// static_asserts a 256-thread TiledMma, whose 4x2x1 warp layout floors TileM
// at 128.
//
// ONE rung is compiled. 256x128x64, 128x256x64 and 128x64x128 each need 48 KiB
// per stage, which StageCountAutoCarveout resolves to ONE stage — below the
// mainloop's Stages>=2 static_assert — and the feasible alternatives
// (128x128x64 coop, 128x128x32 at 2-5 stages, 64x256x64, 64x64x64, 64x128x32)
// all measured slower or no better. TileN=128/TileK=64 keeps the 128-byte
// swizzle atom (K=64 bf16 = 128 B).
using GroupedTile = Shape<_64, _128, _64>;
constexpr int64_t kSm120TileM = size<0>(GroupedTile{});

// Tile-scheduler swizzle: a RUNTIME argument (it reorders CTAs and changes
// nothing numerically), chosen by the padded grid because that is how it
// measured. At the DSV4/Laguna cells with 32 padded M-tiles, swizzle 1 was
// fastest on three of four shapes; at 80 M-tiles, swizzle 8 was worth up to
// 1.36x on the long-K w13 shape (11.17 -> 8.16 ms) and never cost more than 7%
// elsewhere, so the policy optimises the WORST cell rather than the mean.
// Between them the crossover is unmeasured; 64 is the midpoint of the two
// measured grids.
constexpr int kSm120SwizzleSmallGrid = 1;
constexpr int kSm120SwizzleLargeGrid = 8;
constexpr int kSm120SwizzleGridThreshold = 64;  // padded M-tiles

constexpr int sm120_swizzle_for(int64_t m_tiles) {
  return m_tiles >= kSm120SwizzleGridThreshold ? kSm120SwizzleLargeGrid
                                               : kSm120SwizzleSmallGrid;
}

// DIAGNOSTIC override, latched (measurement A/B only): $PRISMAQUANT_CB_BF16_SWIZZLE
// accepts "" (unset), "auto", "1" or "8" — "" / "auto" keep sm120_swizzle_for()
// exactly; "1" / "8" pin that scheduler swizzle for EVERY launch of the
// process. Parsed strictly (a typo raises and names the accepted spellings)
// and read ONCE, latched to its first value: a selector that moved mid-process
// would mix tile orders across launches of one run — the same class of drift
// gridbook/lane_select.py exists to prevent. The shipped policy is unchanged
// when the variable is unset, and a pin cannot move a bit (tile ORDER only).
int sm120_swizzle_launch_choice(int64_t m_tiles) {
  static const int pinned = [] {
    const char* raw = std::getenv("PRISMAQUANT_CB_BF16_SWIZZLE");
    if (raw == nullptr) return 0;
    std::string v(raw);
    const char* ws = " \t\n\r\f\v";
    v.erase(0, v.find_first_not_of(ws));
    v.erase(v.find_last_not_of(ws) + 1);
    for (char& c : v) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (v.empty() || v == "auto") return 0;
    if (v == "1") return kSm120SwizzleSmallGrid;
    if (v == "8") return kSm120SwizzleLargeGrid;
    TORCH_CHECK(false, "invalid PRISMAQUANT_CB_BF16_SWIZZLE=", raw,
                "; expected 'auto' (the measured grid policy), '1' or '8' "
                "(diagnostic pin), or leave it unset");
    return 0;
  }();
  return pinned != 0 ? pinned : sm120_swizzle_for(m_tiles);
}

using GroupedCfg = Cfg<GroupedTile>;
static_assert(gridbook::grouped::AssertSmemFits<
                  typename GroupedCfg::GemmKernel>::value);

// Shared launch core for the two A-source modes of the ONE compiled
// collective. `row_src_ptr == nullptr` is the row-padded TMA-A mode: `a` is
// the materialized [Mp, K] padded activation. Otherwise `a` is a COMPACT
// [S, K] activation and padded row m reads source row row_src[m] inside the
// mainloop (ids outside [0, S) are zero rows) — the padded copy never
// exists. Both modes load byte-identical smem tiles, so their outputs are
// bit-identical (gated in tests/test_bf16_grouped_cutlass.py).
void run_sm120_launch(torch::Tensor output, torch::Tensor a,
                      torch::Tensor weights, torch::Tensor expert_ids,
                      int64_t tile_m, int64_t mp,
                      int const* row_src_ptr, int64_t source_rows) {
  using Gemm = typename GroupedCfg::Gemm;
  using GemmKernel = typename GroupedCfg::GemmKernel;

  TORCH_CHECK(a.is_cuda() && weights.is_cuda() && expert_ids.is_cuda(),
              "a, weights and expert_ids must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
                  weights.scalar_type() == torch::kBFloat16,
              "a and weights must be BF16");
  TORCH_CHECK(tile_m == kSm120TileM,
              "the sm120 grouped BF16 lane compiles tile_m=", kSm120TileM,
              " only (got ", tile_m,
              "); query cb_bf16_grouped_sm120_tile_sizes()");

  const int64_t k = a.size(1);
  const int64_t e = weights.size(0);
  const int64_t n = weights.size(1);
  gridbook::grouped::check_stacked_experts(weights, weights.size(1),
                                           "weights");
  TORCH_CHECK(weights.size(2) == k,
              "shape mismatch: a [.,K] and weights [E,N,K]");
  TORCH_CHECK(mp <= std::numeric_limits<int>::max() &&
                  n <= std::numeric_limits<int>::max() &&
                  k <= std::numeric_limits<int>::max() &&
                  e <= std::numeric_limits<int>::max() &&
                  source_rows <= std::numeric_limits<int>::max(),
              "grouped GEMM dimensions exceed int32");
  TORCH_CHECK(k % kAlignment == 0 && n % kAlignment == 0,
              "K and N must be multiples of 8 BF16 elements");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  gridbook::grouped::check_padded_rows(mp, tile_m);
  gridbook::grouped::check_expert_ids(a, expert_ids, mp, tile_m, e);
  gridbook::grouped::check_same_cuda_device(a, weights, "weights");
  TORCH_CHECK(output.is_cuda() && output.device() == a.device() &&
                  output.scalar_type() == torch::kBFloat16 &&
                  output.dim() == 2 && output.size(0) == mp &&
                  output.size(1) == n && output.is_contiguous(),
              "output must be a contiguous CUDA BF16 [Mp,N] tensor on a's "
              "device");
  if (mp == 0) {
    return;
  }

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  const int mpi = int(mp), ni = int(n), ki = int(k), ei = int(e);
  // The problem's M is the PADDED row count in both modes; in gather mode
  // the A TMA descriptor this stride shapes is built but never issued (the
  // mainloop reads A by row index instead).
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {mpi, ki, 1});
  // B's batch mode is the EXPERT mode: per-expert stride N*K, which is exactly
  // a contiguous [E,N,K] stack. The problem's own L stays 1.
  StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {ni, ki, ei});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {mpi, ni, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {mpi, ni, ki, 1},
      {reinterpret_cast<const Element*>(a.data_ptr()), sa,
       reinterpret_cast<const Element*>(weights.data_ptr()), sb,
       expert_ids.data_ptr<int>(), ei,
       row_src_ptr, int(source_rows)},
      {{1.0f, 0.0f}, nullptr, StrideC{},
       reinterpret_cast<Element*>(output.data_ptr()), sd},
      cutlass::KernelHardwareInfo{},
      // Tile ORDER only — every output tile's accumulation is independent of
      // it, so this cannot move a bit.
      {sm120_swizzle_launch_choice(mp / tile_m),
       cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({int64_t(ws)},
                                a.options().dtype(torch::kUInt8));
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "sm120 grouped BF16 can_implement failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "sm120 grouped BF16 initialize failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "sm120 grouped BF16 run failed: ",
              cutlass::cutlassGetStatusString(status));
}

void run_sm120_grouped(torch::Tensor output, torch::Tensor a,
                       torch::Tensor weights, torch::Tensor expert_ids,
                       int64_t tile_m) {
  TORCH_CHECK(a.dim() == 2, "expected a padded activation [Mp,K]");
  run_sm120_launch(output, a, weights, expert_ids, tile_m,
                   /*mp=*/a.size(0), /*row_src_ptr=*/nullptr,
                   /*source_rows=*/0);
}

void run_sm120_grouped_gather(torch::Tensor output, torch::Tensor a,
                              torch::Tensor row_src, torch::Tensor weights,
                              torch::Tensor expert_ids, int64_t tile_m) {
  TORCH_CHECK(a.dim() == 2, "expected a compact activation [S,K]");
  TORCH_CHECK(tile_m > 0, "grouped tile_m must be positive, got ", tile_m);
  const int64_t mp = row_src.numel();
  gridbook::grouped::check_row_src(a, row_src, mp, tile_m);
  run_sm120_launch(output, a, weights, expert_ids, tile_m, mp,
                   row_src.data_ptr<int>(), /*source_rows=*/a.size(0));
}

torch::Tensor cb_bf16_grouped_mm_sm120(torch::Tensor a, torch::Tensor weights,
                                       torch::Tensor expert_ids,
                                       int64_t tile_m) {
  TORCH_CHECK(a.dim() == 2 && weights.dim() == 3,
              "expected a [Mp,K] and weights [E,N,K]");
  auto output = torch::empty({a.size(0), weights.size(1)}, a.options());
  run_sm120_grouped(output, a, weights, expert_ids, tile_m);
  return output;
}

void cb_bf16_grouped_mm_sm120_out(torch::Tensor output, torch::Tensor a,
                                  torch::Tensor weights,
                                  torch::Tensor expert_ids, int64_t tile_m) {
  run_sm120_grouped(output, a, weights, expert_ids, tile_m);
}

torch::Tensor cb_bf16_grouped_mm_sm120_gather(torch::Tensor a,
                                              torch::Tensor row_src,
                                              torch::Tensor weights,
                                              torch::Tensor expert_ids,
                                              int64_t tile_m) {
  TORCH_CHECK(a.dim() == 2 && weights.dim() == 3,
              "expected a [S,K] and weights [E,N,K]");
  auto output = torch::empty({row_src.numel(), weights.size(1)}, a.options());
  run_sm120_grouped_gather(output, a, row_src, weights, expert_ids, tile_m);
  return output;
}

void cb_bf16_grouped_mm_sm120_gather_out(torch::Tensor output,
                                         torch::Tensor a,
                                         torch::Tensor row_src,
                                         torch::Tensor weights,
                                         torch::Tensor expert_ids,
                                         int64_t tile_m) {
  run_sm120_grouped_gather(output, a, row_src, weights, expert_ids, tile_m);
}

int64_t cb_bf16_grouped_sm120_tile_m() { return kSm120TileM; }

std::vector<int64_t> cb_bf16_grouped_sm120_tile_sizes() {
  return {kSm120TileM};
}

// Host-only config attestation: what was actually compiled. Used by the tests
// and by the KERNELS.md evidence table (no launch, no device needed).
// [tile_m, tile_n, tile_k, stages, SharedStorageSize, sm120 capacity,
//  mma_threads, swizzle(small grid), swizzle(large grid), grid threshold,
//  gather_mainloop (1 = the in-mainloop A-row gather mode is compiled)].
std::vector<int64_t> cb_bf16_grouped_sm120_config() {
  return {int64_t(size<0>(GroupedTile{})),
          int64_t(size<1>(GroupedTile{})),
          int64_t(size<2>(GroupedTile{})),
          int64_t(GroupedCfg::kStages),
          int64_t(GroupedCfg::GemmKernel::SharedStorageSize),
          int64_t(cutlass::arch::sm120_smem_capacity_bytes),
          int64_t(size(typename MmaCfg<GroupedTile>::TiledMma{})),
          int64_t(kSm120SwizzleSmallGrid),
          int64_t(kSm120SwizzleLargeGrid),
          int64_t(kSm120SwizzleGridThreshold),
          int64_t(1)};
}

}  // namespace sm120_lane

#endif  // PRISMAQUANT_CB_BF16_SM120

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_bf16_grouped_mm", &cb_bf16_grouped_mm,
        "Native CUTLASS grouped BF16 GEMM", pybind11::arg("a"),
        pybind11::arg("weights"), pybind11::arg("expert_ends"),
        pybind11::arg("expert_start") = 0);
  m.def("cb_bf16_grouped_mm_out", &cb_bf16_grouped_mm_out,
        "Native CUTLASS grouped BF16 GEMM into a caller-owned output",
        pybind11::arg("output"), pybind11::arg("a"),
        pybind11::arg("weights"), pybind11::arg("expert_ends"),
        pybind11::arg("expert_start") = 0);
#if defined(PRISMAQUANT_CB_BF16_SM120)
  m.def("cb_bf16_grouped_mm_sm120", &sm120_lane::cb_bf16_grouped_mm_sm120,
        "sm12x-native CUTLASS BF16 grouped GEMM (TMA warp-specialized): ONE "
        "launch over row-padded A [Mp,K] where each tile_m block multiplies "
        "expert_ids[tile]'s slice of the stacked weights [E,N,K]. Mp must be a "
        "multiple of tile_m (= cb_bf16_grouped_sm120_tile_m()). fp32 "
        "accumulate, alpha=1/beta=0, one bf16 round — same numerics class as "
        "the SM80 lane, different FP32 reduction order.",
        pybind11::arg("a"), pybind11::arg("weights"),
        pybind11::arg("expert_ids"),
        pybind11::arg("tile_m") = sm120_lane::kSm120TileM);
  m.def("cb_bf16_grouped_mm_sm120_out", &sm120_lane::cb_bf16_grouped_mm_sm120_out,
        "sm12x-native grouped BF16 GEMM into a caller-owned [Mp,N] output",
        pybind11::arg("output"), pybind11::arg("a"), pybind11::arg("weights"),
        pybind11::arg("expert_ids"),
        pybind11::arg("tile_m") = sm120_lane::kSm120TileM);
  m.def("cb_bf16_grouped_mm_sm120_gather",
        &sm120_lane::cb_bf16_grouped_mm_sm120_gather,
        "sm12x-native grouped BF16 GEMM, IN-MAINLOOP A-row gather mode: the "
        "same collective as cb_bf16_grouped_mm_sm120, but padded row m reads "
        "row row_src[m] of a COMPACT [S,K] activation inside the mainloop "
        "(ids outside [0,S) are zero rows), so the row-padded activation "
        "copy is never materialized. row_src is int32 [Mp], Mp a multiple of "
        "tile_m, one expert id per tile as usual. The smem tiles are byte-"
        "identical to the padded-copy mode's, so the two modes' outputs are "
        "bit-identical.",
        pybind11::arg("a"), pybind11::arg("row_src"), pybind11::arg("weights"),
        pybind11::arg("expert_ids"),
        pybind11::arg("tile_m") = sm120_lane::kSm120TileM);
  m.def("cb_bf16_grouped_mm_sm120_gather_out",
        &sm120_lane::cb_bf16_grouped_mm_sm120_gather_out,
        "gather-mode sm12x grouped BF16 GEMM into a caller-owned [Mp,N] "
        "output",
        pybind11::arg("output"), pybind11::arg("a"), pybind11::arg("row_src"),
        pybind11::arg("weights"), pybind11::arg("expert_ids"),
        pybind11::arg("tile_m") = sm120_lane::kSm120TileM);
  m.def("cb_bf16_grouped_sm120_tile_m", &sm120_lane::cb_bf16_grouped_sm120_tile_m,
        "row-padding granularity of the sm12x-native lane");
  m.def("cb_bf16_grouped_sm120_tile_sizes", &sm120_lane::cb_bf16_grouped_sm120_tile_sizes,
        "every TileM compiled for the sm12x-native lane (enumerate THIS)");
  m.def("cb_bf16_grouped_sm120_config", &sm120_lane::cb_bf16_grouped_sm120_config,
        "[tile_m, tile_n, tile_k, stages, SharedStorageSize, sm120 capacity, "
        "mma_threads (128 = pingpong layer, 256 = cooperative), "
        "swizzle below the grid threshold, swizzle at/above it, threshold in "
        "padded M-tiles]");
#endif
}
