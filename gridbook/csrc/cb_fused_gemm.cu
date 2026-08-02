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
//     and the transient path. Serving dispatch is on by default with an
//     explicit opt-out; see tests/test_fused_prefill.py for the bit-exact gates.
//   * Large-M needs a weight-stationary/no-HBM-materialization design that
//     amortizes each B decode across M. The first dense persistent-N build
//     measured negative (2-5.7x slower than transient at 27B shapes), so any
//     replacement starts from a fresh roofline rather than this collective.
//
// Entry points:
//  - sm120_fp8_mm_fork:  128x128x128 passthrough through the UNCHANGED forked
//    collective (fork-without-change gate, kept as regression).
//  - sm120_fp8_mm_fork64: same passthrough at 64x128x128 — the bit-exactness
//    REFERENCE for the fused kernel (identical TiledMma/layout config).
//  - cb_fused_prefill_mm: the decode-in-prologue FP8_CB GEMM — B is the
//    PACKED byte stream + a global e4m3-byte LUT; the dense tile never
//    exists in HBM. KBits template-dispatched over the RUNG LAW below
//    (28..48 step 4) — a strict subset of the 21-rung product ladder;
//    cb_fused_kbits() reports what this build carries.
//  - smem_report: per-config SharedStorage sizes (budget sanity).
//
// Scale convention:
//  - the ORIGINAL entries return the UNSCALED bf16 accumulation (per-token x
//    per-channel scales applied outside, as the baseline gate established);
//  - cb_fused_prefill_mm_scaled applies BOTH scales inside an fp32 EVT
//    epilogue -- out = bf16_rn(acc_f32 * a_scale[m] * b_scale[n]) -- which is
//    the same rounding ORDER as ops.cutlass_scaled_mm. The unscaled entry
//    rounds to bf16 first, and that order difference is measurable on served
//    prompt logprobs (mean 0.10 / max 0.86 nats on the 27B artifact), so the
//    serving call site uses the _scaled entry.

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

#include "cutlass_fork/sm120_cb_mma_tma.hpp"
#include "cutlass_fork/sm120_cb_fused_mma.hpp"
#include "cutlass_fork/sm120_expert_row_broadcast.hpp"

// The EVT trees, the smem gate, the tile-feasibility filter and the grouped
// host validation are shared with cb_fused_fp4_gemm.cu and the sm12x BF16
// bridge (2026-08-01 audit §4 dedupe #1/#2). Every type this file takes from
// there is proven identical to its former verbatim spelling by the
// static_asserts below — the generated kernels are unchanged.
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

// ===========================================================================
// THE FUSED RUNG LAW (K1.2, established 2026-08-02).
//
// The producer's FP8-CB ladder is EVERY INTEGER k_bits in [28, 48]
// (prismaquant/format_registry.py `for _k in range(28, 49)` — 3.5..6.0 bpw in
// 0.125 steps; gridbook/runtime_contract.json carries the same 21 rungs). The
// fused mid-M lane serves the MULTIPLES OF 4 and no others, and the K1.2
// investigation established that this is a property of the FORMAT and of TMA
// — not a missing template instantiation that could be added:
//
//  1. TMA BOX. Packed B is fetched by SM90_TMA_LOAD with a box of
//     (TileN, CbTypeSize) BYTES over the [N, n_sb*CbTypeSize] byte stream,
//     where CbTypeSize = 4*k_bits is one 256-weight superblock. TMA requires
//     the box's contiguous extent to be a 16-byte multiple:
//         4*k_bits % 16 == 0   <=>   k_bits % 4 == 0.
//     (cutlass_fork/sm120_cb_fused_mma.hpp: `static_assert(CbTypeSize % 16
//     == 0, "type_size must be a 16-byte multiple (TMA box)")`.)
//
//  2. UNIFORM SUB-TABLE WIDTH. The fused mainloop decodes a codeword with a
//     SINGLE width `CbSubW = KBits/4` and indexes the flat codebook at
//     `(s << CbSubW) + idx` — one width and one stride for all four
//     sub-tables. The FORMAT splits k_bits over n_sub=4 RAGGEDLY: see
//     `SubSplit` in csrc/cb_gemv.cu, the decode reference this kernel is
//     bit-gated against — `base = k_bits/NSUB, extra = k_bits%NSUB`, and
//     sub-table i has width `base + (i < extra)`. At k_bits=37 the true
//     widths are (10,9,9,9) with non-uniform table offsets, so a uniform
//     decode would be WRONG, not merely unaligned.
//
// The two laws coincide exactly, which is why one predicate expresses both.
// Consequence for dispatch: the 21-rung product ladder is NOT fully backed by
// this lane, and Python must therefore ASK for the backed set rather than
// carry a literal (`cb_fused_kbits()` below). ROADMAP K1.2 offers exactly two
// arms — instantiate every product rung, or "encode the concrete route so the
// allocator cannot price an unbacked fast path". Arm one is closed by the
// laws above; this file implements arm two, and makes the surface queryable.
// ===========================================================================
constexpr int64_t kFusedKbLo = 28;
constexpr int64_t kFusedKbHi = 48;
constexpr int64_t kFusedKbStep = 4;   // = 16 / gcd(4, 16), i.e. law 1 above.

constexpr bool fused_kbits_supported(int64_t kb) {
  return kb >= kFusedKbLo && kb <= kFusedKbHi &&
         (kb - kFusedKbLo) % kFusedKbStep == 0;
}

// THE one rung list. Every switch, every report and every binding below is
// generated from it, because 21-case switches hand-written N times is a bug
// farm — the whole point of K1.2's dispatch half.
#define PQ_FUSED_RUNGS(X) X(28) X(32) X(36) X(40) X(44) X(48)

// ...and the list is proved equal to the law in BOTH directions: every member
// satisfies the predicate, and the member count equals the law's cardinality.
#define PQ_RUNG_IN_LAW(KB) \
  static_assert(fused_kbits_supported(KB), "rung " #KB " violates the fused rung law");
PQ_FUSED_RUNGS(PQ_RUNG_IN_LAW)
#undef PQ_RUNG_IN_LAW
#define PQ_RUNG_ONE(KB) +1
static_assert((0 PQ_FUSED_RUNGS(PQ_RUNG_ONE)) ==
                  (kFusedKbHi - kFusedKbLo) / kFusedKbStep + 1,
              "PQ_FUSED_RUNGS must enumerate the law's rungs exhaustively");
#undef PQ_RUNG_ONE

// The TileM values with at least one compiled rung (see the grouped section).
#define PQ_FUSED_MOE_TILES(X) X(128) X(256)

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

// ---------------------------------------------------------------------------
// Scaled config: identical mainloop, but the epilogue is an EVT that applies
// the per-token (row / M) activation scale and the per-channel (col / N)
// weight scale in fp32 BEFORE the bf16 round -- matching cutlass_scaled_mm.
//
//   D = convert<bf16>( b_scale[n] * ( a_scale[m] * acc_f32 ) )
//
// Node tree (the same shape vLLM's ScaledEpilogue uses):
//   Sm90EVT<Compute<multiplies, bf16>, RowBroadcast(b_scales),
//           Sm90EVT<Compute<multiplies, f32>, ColBroadcast(a_scales),
//                   Sm90AccFetch>>
//
// The tree itself now lives in cb_grouped_common.hpp (it was verbatim in three
// files); binding it to THIS file's element types reproduces the same types.
// ---------------------------------------------------------------------------
template <class TileShape>
using ScaledFusion = gridbook::grouped::ScaledFusion<TileShape, ElementAcc,
                                                     ElementD>;

template <class TileShape>
struct CfgScaled {
  using Fusion = typename ScaledFusion<TileShape>::type;
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

// Hard smem gate (shared): the sm90 cooperative kernel layer does NOT
// static_assert its own SharedStorageSize against the arch capacity (only the
// sm120 asymmetric-DMA kernel does), so an over-budget config would compile and
// then fail at launch. Every instantiated fused config passes through this.
template <class GemmKernel>
using AssertSmemFits = gridbook::grouped::AssertSmemFits<GemmKernel>;

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

// BIT-IDENTITY PROOF for the §4-dedupe extraction: the EVT trees taken from
// cb_grouped_common.hpp are the SAME TYPES this file spelled verbatim before
// it. Type identity is what decides the generated kernel here (the fusion type
// selects the epilogue collective, its smem layout and its rounding order), so
// this is the same class of proof as the MoeTile<128> == TileF assert below.
static_assert(
    cute::is_same_v<
        typename ScaledFusion<TileF>::type,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiplies, ElementD, ElementAcc,
                cutlass::FloatRoundStyle::round_to_nearest>,
            cutlass::epilogue::fusion::Sm90RowBroadcast<
                0, TileF, float, float, Stride<_0, _1, _0>>,
            cutlass::epilogue::fusion::Sm90EVT<
                cutlass::epilogue::fusion::Sm90Compute<
                    cutlass::multiplies, ElementAcc, ElementAcc,
                    cutlass::FloatRoundStyle::round_to_nearest>,
                cutlass::epilogue::fusion::Sm90ColBroadcast<
                    0, TileF, float, float, Stride<_1, _0, _0>>,
                cutlass::epilogue::fusion::Sm90AccFetch>>>,
    "shared ScaledFusion must reproduce the pre-extraction EVT node tree");

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
  static_assert(AssertSmemFits<GemmKernel>::value);

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

// ONE rung rejection, phrased as the LAW rather than as a list — so an operator
// who lands here with a k_bits=37 artifact is told the reason (and that the
// rung is served, just not by this lane) instead of "unsupported".
void check_fused_kbits(int64_t k_bits) {
  TORCH_CHECK(
      fused_kbits_supported(k_bits),
      "k_bits=", k_bits, " is outside the FUSED lane's rung law. The fused "
      "mid-M FP8-CB kernel serves k_bits in [", kFusedKbLo, ", ", kFusedKbHi,
      "] on a step of ", kFusedKbStep, " (multiples of 4). This is a format + "
      "TMA law, not a build option: type_size = 4*k_bits must be a 16-byte "
      "multiple for the packed-B TMA box, and the mainloop's single "
      "CbSubW = k_bits/4 sub-table width is only the format's real layout when "
      "k_bits % 4 == 0 (csrc/cb_gemv.cu SubSplit splits raggedly otherwise). "
      "Every integer rung 28..48 IS served by Gridbook — through the decode "
      "GEMV and the expand+GEMM quality bridge — just not through this lane. "
      "Enumerate cb_fused_kbits() rather than assuming a ladder.");
}

void check_fused_inputs(torch::Tensor a, torch::Tensor packed,
                        torch::Tensor lut, int64_t N, int64_t K,
                        int64_t k_bits) {
  // FIRST, before any shape check. Every downstream bound is a function of
  // k_bits (row_bytes = (K/256)*4*k_bits above all), so an off-law rung makes
  // the shape checks fire first and report a stride mismatch — burying the
  // actual reason under a symptom. The rung is the most fundamental
  // precondition and carries the most informative message.
  check_fused_kbits(k_bits);
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
}

torch::Tensor cb_fused_prefill_mm(torch::Tensor a, torch::Tensor packed,
                                  torch::Tensor lut, int64_t N, int64_t K,
                                  int64_t k_bits) {
  check_fused_inputs(a, packed, lut, N, K, k_bits);
#define PQ_DENSE_CASE(KB) case KB: return run_fused<KB>(a, packed, lut, N, K);
  switch (k_bits) { PQ_FUSED_RUNGS(PQ_DENSE_CASE) default: break; }
#undef PQ_DENSE_CASE
  TORCH_CHECK(false, "unsupported k_bits ", k_bits);
}

// ---------------------------------------------------------------------------
// Fused decode-in-prologue entry, SCALED epilogue (cutlass_scaled_mm-faithful).
// ---------------------------------------------------------------------------
template <int KB>
torch::Tensor run_fused_scaled(torch::Tensor a, torch::Tensor packed,
                               torch::Tensor lut, torch::Tensor a_scales,
                               torch::Tensor b_scales, int64_t N, int64_t K) {
  constexpr int Stages = 2;
  using TileShape = TileF;
  using Mainloop = typename SwapToFused<
      KB, Stages, typename CfgScaled<TileShape>::BuilderMainloop>::type;
  using Epilogue = typename CfgScaled<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(AssertSmemFits<GemmKernel>::value);

  const int M = (int)a.size(0);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, (int)N, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, (int)N, (int)K, 1},
      {reinterpret_cast<const ElementAB*>(a.data_ptr()), sa,
       packed.data_ptr<uint8_t>(), packed.stride(0),
       lut.data_ptr<uint8_t>()},
      {// EVT args: children first, then node op args (empty for multiplies).
       {{b_scales.data_ptr<float>(), 0.0f, Stride<_0, _1, _0>{}},
        {{a_scales.data_ptr<float>(), 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "fused(scaled) can_implement failed (K%256? row stride %16?)");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
  return d;
}

torch::Tensor cb_fused_prefill_mm_scaled(torch::Tensor a, torch::Tensor packed,
                                         torch::Tensor lut,
                                         torch::Tensor a_scales,
                                         torch::Tensor b_scales, int64_t N,
                                         int64_t K, int64_t k_bits) {
  check_fused_inputs(a, packed, lut, N, K, k_bits);
  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
                  a_scales.numel() == a.size(0) && a_scales.is_contiguous(),
              "a_scales must be contiguous fp32 [M] (per-token)");
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
                  b_scales.numel() == N && b_scales.is_contiguous(),
              "b_scales must be contiguous fp32 [N] (per-output-channel)");
#define PQ_DENSE_SCALED_CASE(KB) \
  case KB: return run_fused_scaled<KB>(a, packed, lut, a_scales, b_scales, N, K);
  switch (k_bits) { PQ_FUSED_RUNGS(PQ_DENSE_SCALED_CASE) default: break; }
#undef PQ_DENSE_SCALED_CASE
  TORCH_CHECK(false, "unsupported k_bits ", k_bits);
}

// ---------------------------------------------------------------------------
// GROUPED (MoE) fused entry — TILE-INDEXED grouping.
//
// True CUTLASS ptr-array grouping is unavailable on this builder (the vendored
// sm120_mma_builder.inl static_asserts !IsPtrArrayKernel and
// MainloopSm120ArrayTmaWarpSpecialized has no implementation). Instead we do
// what vLLM's Triton fused-MoE kernel does: the caller pre-gathers and PADS
// A's rows so every expert's segment starts on a multiple of TileM and spans a
// whole number of TileM blocks. The launch is then ONE ordinary single-problem
// GEMM of shape (Mp, N, K) in which each M-tile reads a DIFFERENT expert's
// weights: the fork's packed-B TMA tensor already carries a batch mode whose
// stride is N * packed_row_bytes -- exactly the per-expert stride of a stacked
// [E, N, row_bytes] buffer -- so the expert index is simply the B tile's
// l-coordinate, chosen per M-tile from expert_ids[]. No tensormap updates, no
// ptr-arrays; the descriptors stay host-built and immutable.
//
// The B-scale epilogue node is Sm120CbExpertRowBroadcast (same body as
// Sm90RowBroadcast, base pointer offset by expert_ids[m] * N). Everything else
// -- the A-scale ColBroadcast and the multiply/round order
// bf16_rn(b_scale * (a_scale * acc)) -- is identical to ScaledFusion.
// ---------------------------------------------------------------------------
template <class TileShape>
using MoeScaledFusion = gridbook::grouped::MoeScaledFusion<TileShape,
                                                           ElementAcc, ElementD>;

// BIT-IDENTITY PROOF (see the ScaledFusion assert above): the expert-indexed
// tree from the shared header is the type this file spelled verbatim before.
static_assert(
    cute::is_same_v<
        typename MoeScaledFusion<TileF>::type,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::multiplies, ElementD, ElementAcc,
                cutlass::FloatRoundStyle::round_to_nearest>,
            cutlass::epilogue::fusion::Sm120CbExpertRowBroadcast<
                0, TileF, float, float, Stride<_0, _1, _0>>,
            cutlass::epilogue::fusion::Sm90EVT<
                cutlass::epilogue::fusion::Sm90Compute<
                    cutlass::multiplies, ElementAcc, ElementAcc,
                    cutlass::FloatRoundStyle::round_to_nearest>,
                cutlass::epilogue::fusion::Sm90ColBroadcast<
                    0, TileF, float, float, Stride<_1, _0, _0>>,
                cutlass::epilogue::fusion::Sm90AccFetch>>>,
    "shared MoeScaledFusion must reproduce the pre-extraction EVT node tree");

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

// The grouped path's DEFAULT TileM (python must pad Mp to a multiple of the
// tile_m it selects; 128 stays the default so existing callers are unchanged).
constexpr int64_t kMoeTileM = size<0>(TileF{});

// ---------------------------------------------------------------------------
// TileM ladder for the grouped path.
//
// MOTIVATION: every M-tile CTA re-decodes its expert's B weights, so an expert
// covering ceil(m_e/TileM) tiles pays that many redundant decodes. A larger
// TileM cuts both the redundancy and (per expert) the number of padded tiles.
//
// FEASIBILITY IS SMEM-BOUND, and the bound bites the HIGH rungs first, because
// smem grows with BOTH TileM (smem_A = TileM*TileK*Stages) and k_bits
// (smem_BP = TileN*4*k_bits*Stages), and since R6 also with the staged
// codebook LUT. Measured by csrc/tools/smem_probe_tilem.cu (host-only, no
// launch) against cutlass::arch::sm120_smem_capacity_bytes = 101376 (= 99 KiB,
// the CUDA cc-12.0 max opt-in dynamic smem per block):
//
//   GemmKernel::SharedStorageSize, TileN=64 TileK=128 Stages=2
//   (REGENERATED 2026-08-02, GB10 / CUTLASS 4.3.4 — the previous table quoted
//    the PRE-R6 base and was stale by up to 16,384 B once the LUT stage landed)
//   TileM   k28     k32      k36      k40      k44      k48
//   128    67584   70656    74752    80896    91136    93184     all FIT
//   256   100352  101376   103424  105472  107520  109568        FIT only 28/32
//   384  >=132096 (smem_A alone = 98304)                         none FIT
//
// So the compiled matrix is: TileM=128 x all six law rungs, plus TileM=256 x
// {28, 32} ONLY. TileM=384 is infeasible at every rung and is not compiled.
// (TileM must be a multiple of the TiledMma's 128-row M, so 128/256/384 are the
// only candidates at all — there is no 192 rung to rescue the high k_bits.)
// NOTE: TileM=256/k_bits=32 lands on EXACTLY the 101376-byte ceiling (zero
// margin) — it is compiled, but must be launch-verified before being trusted.
//
// The predicate below is a LAW, not a transcription of that table: it is the
// closed form the collective's own storage policy implies, and it is
// static_asserted cell-by-cell against the twelve measured numbers, so a future
// storage change that breaks the law is a compile error rather than a stale
// comment. (The old hand-listed `kb == 28 || kb == 32` could not notice.)
//
// Nothing else changed: same SwapToFused mainloop, same MoeScaledFusion
// expert-indexed epilogue, same Stages=2, same TileN=64/TileK=128, same
// bf16_rn(b_scale * (a_scale * acc)) rounding order.
// ---------------------------------------------------------------------------
template <int TM>
using MoeTile = Shape<Int<TM>, _64, _128>;

// BIT-IDENTITY PROOF for the pre-existing TileM=128 grouped path: the templated
// tile at TM=128 is the SAME TYPE as the TileF the old code hardcoded, so
// run_moe_grouped<128, KB> instantiates a byte-for-byte identical kernel
// (identical TiledMma, smem layouts, pipeline, epilogue and rounding order).
static_assert(cute::is_same_v<MoeTile<128>, TileF>,
              "TileM=128 grouped path must remain the original TileF config");

// --- the smem closed form (mirrors sm120_cb_fused_mma.hpp's storage policy) --
//
// LUT stage: the flat codebook is 4 sub-tables x 2^CbSubW rows x 2 e4m3 bytes.
// The collective stages `CbLutResidentSubs` of them, choosing 4 / 2 / 0 by the
// headroom its TileM leaves (16 KB at TileM=128, 1 KB at TileM=256).
constexpr int64_t moe_lut_bytes(int64_t tm, int64_t kb) {
  const int64_t sub_bytes = (int64_t{1} << (kb / 4)) * 2;   // 2^CbSubW * 2
  const int64_t subs = (tm <= 128) ? ((4 * sub_bytes <= 16384) ? 4 : 2)
                                   : ((4 * sub_bytes <= 1024) ? 4 : 0);
  return subs * sub_bytes;
}

// smem_A (TileM*TileK*Stages = 256*tm) + smem_BP (TileN*4*kb*Stages = 512*kb)
// + 19,456 B fixed (decoded-B buffer + epilogue tensors + pipelines) + the LUT.
constexpr int64_t moe_smem_bytes(int64_t tm, int64_t kb) {
  return 256 * tm + 512 * kb + 19456 + moe_lut_bytes(tm, kb);
}

// Compiled-matrix predicate: the rung law AND the smem budget, both as laws.
constexpr bool moe_tile_supported(int64_t tm, int64_t kb) {
  return fused_kbits_supported(kb) && (tm == 128 || tm == 256 || tm == 384) &&
         moe_smem_bytes(tm, kb) <=
             (int64_t)cutlass::arch::sm120_smem_capacity_bytes;
}

// The closed form REPRODUCES the probe, cell by cell. If a CUTLASS bump or a
// storage-policy edit moves any of these, this file stops compiling and the
// table above must be regenerated (csrc/tools/smem_probe_tilem.cu) — which is
// exactly the failure mode the stale pre-R6 table had no way to signal.
#define PQ_ASSERT_SMEM(TM, KB, BYTES)                                       \
  static_assert(moe_smem_bytes(TM, KB) == BYTES,                            \
                "measured smem for (TileM=" #TM ", k" #KB ") no longer "    \
                "matches the closed form; re-run smem_probe_tilem");
PQ_ASSERT_SMEM(128, 28, 67584)  PQ_ASSERT_SMEM(128, 32, 70656)
PQ_ASSERT_SMEM(128, 36, 74752)  PQ_ASSERT_SMEM(128, 40, 80896)
PQ_ASSERT_SMEM(128, 44, 91136)  PQ_ASSERT_SMEM(128, 48, 93184)
PQ_ASSERT_SMEM(256, 28, 100352) PQ_ASSERT_SMEM(256, 32, 101376)
PQ_ASSERT_SMEM(256, 36, 103424) PQ_ASSERT_SMEM(256, 40, 105472)
PQ_ASSERT_SMEM(256, 44, 107520) PQ_ASSERT_SMEM(256, 48, 109568)
#undef PQ_ASSERT_SMEM

// ...and therefore the predicate reproduces the compiled matrix exactly.
static_assert(moe_tile_supported(256, 32) &&
                  moe_smem_bytes(256, 32) ==
                      (int64_t)cutlass::arch::sm120_smem_capacity_bytes,
              "TileM=256/k32 is the documented ZERO-MARGIN cell; if it stops "
              "landing exactly on the ceiling the caution above is wrong");
static_assert(!moe_tile_supported(256, 36) && !moe_tile_supported(384, 28),
              "smem-infeasible cells must stay out of the compiled matrix");
static_assert(!moe_tile_supported(128, 30),
              "the rung law must gate the grouped path too");

template <int TM, int KB>
torch::Tensor run_moe_grouped(torch::Tensor a, torch::Tensor packed,
                              torch::Tensor lut, torch::Tensor a_scales,
                              torch::Tensor b_scales, torch::Tensor expert_ids,
                              int64_t N, int64_t K) {
  static_assert(moe_tile_supported(TM, KB),
                "attempted to instantiate an smem-infeasible (TileM, k_bits)");
  constexpr int Stages = 2;
  using TileShape = MoeTile<TM>;
  using Mainloop = typename SwapToFused<
      KB, Stages, typename CfgMoeScaled<TileShape>::BuilderMainloop>::type;
  using Epilogue = typename CfgMoeScaled<TileShape>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(AssertSmemFits<GemmKernel>::value);

  const int Mp = (int)a.size(0);
  const int E = (int)packed.size(0);
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({Mp, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {Mp, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {Mp, (int)N, 1});

  const int* ptr_eids = expert_ids.data_ptr<int>();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {Mp, (int)N, (int)K, 1},
      {reinterpret_cast<const ElementAB*>(a.data_ptr()), sa,
       packed.data_ptr<uint8_t>(), packed.stride(1),
       lut.data_ptr<uint8_t>(),
       ptr_eids, E},
      {// EVT args: children first, then node op args (empty for multiplies).
       {{b_scales.data_ptr<float>(), 0.0f, Stride<_0, _1, _0>{},
         ptr_eids, (int)N},
        {{a_scales.data_ptr<float>(), 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "moe grouped can_implement failed (K%256? row stride %16?)");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr()) == cutlass::Status::kSuccess);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess);
  return d;
}

// Rung dispatch for ONE TileM, generated from the single rung list. The
// `if constexpr` is what makes generation safe: a cell the law rejects emits no
// code at all, so `run_moe_grouped`'s own static_assert (and AssertSmemFits
// behind it) is never reached for an over-budget config. Adding a TileM or a
// rung is therefore one edit to one macro list, not two nested switches kept in
// sync by hand — the K1.2 "bug farm" this replaces.
template <int TM, class... Args>
torch::Tensor moe_dispatch_kbits(int64_t k_bits, Args&&... args) {
#define PQ_MOE_RUNG(KB)                                                    \
  if constexpr (moe_tile_supported(TM, KB)) {                              \
    if (k_bits == KB) return run_moe_grouped<TM, KB>(args...);             \
  }
  PQ_FUSED_RUNGS(PQ_MOE_RUNG)
#undef PQ_MOE_RUNG
  TORCH_CHECK(false, "no compiled grouped kernel for (tile_m=", TM,
              ", k_bits=", k_bits, ")");
}

int64_t cb_fused_moe_tile_m() { return kMoeTileM; }

// THE compiled rung set, as the build actually carries it. Python must
// enumerate this instead of duplicating a literal ladder: the fused lane backs
// a strict SUBSET of the 21-rung product ladder (see the rung law at the top),
// and a duplicated literal is how a dispatch silently misses a compiled rung or
// selects an uncompiled one. Generated from the same list the switches are.
std::vector<int64_t> cb_fused_kbits() {
#define PQ_RUNG_VALUE(KB) KB,
  return {PQ_FUSED_RUNGS(PQ_RUNG_VALUE)};
#undef PQ_RUNG_VALUE
}

// Every TileM value for which AT LEAST ONE rung is compiled.
std::vector<int64_t> cb_fused_moe_tile_sizes() {
#define PQ_TILE_VALUE(TM) TM,
  return {PQ_FUSED_MOE_TILES(PQ_TILE_VALUE)};
#undef PQ_TILE_VALUE
}

// Per-rung candidate list — python should enumerate THIS, never the union,
// so it can never select an uncompiled (TileM, k_bits) pair.
std::vector<int64_t> cb_fused_moe_tile_sizes_for_kbits(int64_t k_bits) {
  return gridbook::grouped::tile_sizes_where(
      cb_fused_moe_tile_sizes(),
      [k_bits](int64_t tm) { return moe_tile_supported(tm, k_bits); });
}

torch::Tensor cb_fused_moe_grouped(torch::Tensor a, torch::Tensor packed,
                                   torch::Tensor lut, torch::Tensor a_scales,
                                   torch::Tensor b_scales,
                                   torch::Tensor expert_ids, int64_t N,
                                   int64_t K, int64_t k_bits,
                                   int64_t tile_m = kMoeTileM) {
  // --- mirrors check_fused_inputs, but packed is [E, N, row_bytes] ---
  check_fused_kbits(k_bits);   // first, for the reason given in that function
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn,
              "a must be fp8 e4m3 [Mp,K]");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) == K,
              "a must be contiguous [Mp,K]");
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kUInt8);
  TORCH_CHECK(K % 256 == 0);
  TORCH_CHECK(moe_tile_supported(tile_m, k_bits),
              "grouped tile_m=", tile_m, " is not compiled for k_bits=", k_bits,
              " (smem-infeasible on sm_120: limit 101376 B). Query "
              "cb_fused_moe_tile_sizes_for_kbits(k_bits) for the legal set.");
  // Padding granularity, stacked-expert contiguity and the per-tile expert id
  // vector are properties of the SHARED grouping construction, so those three
  // checks come from cb_grouped_common.hpp (same conditions, one wording).
  gridbook::grouped::check_padded_rows(a.size(0), tile_m);

  TORCH_CHECK(packed.scalar_type() == torch::kUInt8,
              "packed must be uint8 [E,N,row_bytes] on cuda");
  gridbook::grouped::check_stacked_experts(packed, N, "packed");
  const int64_t row_bytes = packed.size(2);
  TORCH_CHECK(row_bytes % 16 == 0,
              "packed row stride must be a 16-byte multiple (UNPADDED rows)");
  TORCH_CHECK(row_bytes >= (K / 256) * 4 * k_bits,
              "packed row_bytes too small for K/k_bits");

  TORCH_CHECK(a_scales.is_cuda() && a_scales.scalar_type() == torch::kFloat32 &&
                  a_scales.numel() == a.size(0) && a_scales.is_contiguous(),
              "a_scales must be contiguous fp32 [Mp] (per padded row)");
  TORCH_CHECK(b_scales.is_cuda() && b_scales.scalar_type() == torch::kFloat32 &&
                  b_scales.dim() == 2 && b_scales.size(0) == packed.size(0) &&
                  b_scales.size(1) == N && b_scales.is_contiguous(),
              "b_scales must be contiguous fp32 [E,N]");
  gridbook::grouped::check_expert_ids(a, expert_ids, a.size(0), tile_m,
                                      packed.size(0));

#define PQ_MOE_TILE_CASE(TM)                                                 \
  case TM: return moe_dispatch_kbits<TM>(k_bits, a, packed, lut, a_scales,   \
                                         b_scales, expert_ids, N, K);
  switch (tile_m) { PQ_FUSED_MOE_TILES(PQ_MOE_TILE_CASE) default: break; }
#undef PQ_MOE_TILE_CASE
  TORCH_CHECK(false, "no compiled grouped kernel for (tile_m=", tile_m,
              ", k_bits=", k_bits, ")");
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

// Per-rung SharedStorageSize of the fused kernel (host-only; no launch), plus
// the smem the LUT stage costs at that rung and the sm_120 ceiling. Used to
// verify the R6 codebook-residency arithmetic on a GPU-less box.
template <int KB>
static void push_rung(std::vector<int64_t>& out) {
  using ML = typename SwapToFused<KB, 2, typename Cfg<TileF>::BuilderMainloop>::type;
  using KN = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, ML, typename Cfg<TileF>::Epilogue>;
  out.push_back(KB);
  out.push_back((int64_t)KN::SharedStorageSize);
  out.push_back((int64_t)ML::CbLutSmemBytes);
  out.push_back((int64_t)ML::CbLutResidentSubs);
}

// Flat [k_bits, SharedStorageSize, lut_smem_bytes, resident_sub_tables] per
// COMPILED rung — generated from the one rung list, so it can never fall behind
// the dispatch (it did: this used to be a hand-written six-call sequence).
// "All the rungs" here means all the rungs that EXIST in this lane; the 15
// integer rungs the law excludes cannot be instantiated to be reported on.
std::vector<int64_t> smem_report_rungs() {
  std::vector<int64_t> out;
#define PQ_PUSH_RUNG(KB) push_rung<KB>(out);
  PQ_FUSED_RUNGS(PQ_PUSH_RUNG)
#undef PQ_PUSH_RUNG
  out.push_back(-1);
  out.push_back((int64_t)cutlass::arch::sm120_smem_capacity_bytes);
  out.push_back(-1);
  out.push_back(-1);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm120_fp8_mm_fork", &sm120_fp8_mm_fork,
        "sm120 fp8 GEMM through the FORKED collective (128-tile passthrough)");
  m.def("sm120_fp8_mm_fork64", &sm120_fp8_mm_fork64,
        "passthrough at 64x128x128 (fused kernel's reference config)");
  m.def("cb_fused_prefill_mm", &cb_fused_prefill_mm,
        "FP8_CB decode-in-prologue fused GEMM (unscaled bf16 out)");
  m.def("cb_fused_prefill_mm_scaled", &cb_fused_prefill_mm_scaled,
        "FP8_CB decode-in-prologue fused GEMM, per-token x per-channel scales "
        "applied in the fp32 EVT epilogue (cutlass_scaled_mm rounding order)");
  m.def("cb_fused_moe_grouped", &cb_fused_moe_grouped,
        "FP8_CB grouped (MoE) fused GEMM: ONE launch over row-padded A "
        "[Mp,K] where each TileM block reads expert_ids[tile]'s weights from "
        "packed [E,N,row_bytes]; per-token a_scales[Mp] x per-expert-channel "
        "b_scales[E,N] applied in the fp32 EVT epilogue (same rounding order "
        "as cb_fused_prefill_mm_scaled). Mp must be a multiple of tile_m "
        "(default 128 = cb_fused_moe_tile_m()); legal tile_m values for a rung "
        "come from cb_fused_moe_tile_sizes_for_kbits(k_bits).",
        py::arg("a"), py::arg("packed"), py::arg("lut"), py::arg("a_scales"),
        py::arg("b_scales"), py::arg("expert_ids"), py::arg("N"), py::arg("K"),
        py::arg("k_bits"), py::arg("tile_m") = kMoeTileM);
  m.def("cb_fused_kbits", &cb_fused_kbits,
        "the k_bits rungs this build actually COMPILED for the fused mid-M "
        "lane, ascending. The product FP8-CB ladder is every integer 28..48; "
        "this lane serves the multiples of 4 only, because type_size = 4*k "
        "must be a 16-byte TMA box multiple AND the mainloop's single "
        "CbSubW = k/4 sub-table width is the format's real layout only then. "
        "Enumerate this to decide fused eligibility — never a literal ladder.");
  m.def("cb_fused_moe_tile_m", &cb_fused_moe_tile_m,
        "DEFAULT TileM of the grouped (MoE) path — the row-padding granularity");
  m.def("cb_fused_moe_tile_sizes", &cb_fused_moe_tile_sizes,
        "every TileM with at least one compiled k_bits rung");
  m.def("cb_fused_moe_tile_sizes_for_kbits", &cb_fused_moe_tile_sizes_for_kbits,
        "the TileM values ACTUALLY compiled for this k_bits rung — enumerate "
        "this to pick a grouped tile_m; anything else TORCH_CHECKs");
  m.def("smem_report", &smem_report, "SharedStorage sizes [F44, F48, K44, K48, Epi]");
  m.def("smem_report_rungs", &smem_report_rungs,
        "flat [k_bits, SharedStorageSize, lut_smem_bytes, resident_subs] per "
        "COMPILED rung (see cb_fused_kbits), "
        "then [-1, sm120_capacity, -1, -1]");
}
