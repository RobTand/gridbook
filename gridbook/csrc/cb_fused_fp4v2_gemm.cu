// FP4-CB **v2 quality** fused mid-M lane (2026-08-01 performance audit §3 P2a).
//
// WHAT THIS CLOSES. The audit's structural cause (c): "FP4 has no mid-M lane at
// all". FP8-CB owns M = 9..128 with its fused decode-in-prologue kernel and
// measures 1.04x / 1.26x / 1.45x at M = 32/64/128 against the serial
// expand+GEMM transient. The FP4 quality path had nothing there: M = 9..16
// always took the BF16 bridge and so did everything above it, because the only
// fused FP4 kernel in the tree (cb_fused_fp4_gemm.cu) serves a DIFFERENT
// activation contract and is opt-in behind its own six promotion gates.
//
// This module is the contract-preserving twin of the FP8 mid-M lane:
//
//   packed FP4-v2 CB rows --(decode in the CUTLASS producer/consumer stage)-->
//   BF16 values BIT-IDENTICAL to cb_expand_v2 --> 16-bit tensor-core MMA
//   against the same BF16 group-16-QDQ'd activations --> fp32 accumulate -->
//   ONE bf16 round.
//
// The served weight values and the served activation bucket are untouched. The
// ONLY difference from today's shipping `expand_fp4_v2_to_weight` +
// `cb_bf16_grouped_mm` route is the FP32 GEMM REDUCTION ORDER — the same
// requalification class the promoted FP8 mid-M kernel cleared, and the same one
// the sm12x-native BF16 grouped lane (P1) carries.
//
// SEPARATE MODULE, deliberately. cb_fused_fp4_gemm.cu is the NATIVE-NVFP4
// lane: different payload (packed e2m1 activations + swizzled ue4m3 scale
// factors), different served numerics, different promotion chain. Nothing here
// touches its kernels or its instantiation lists.
//
// ---------------------------------------------------------------------------
// WHY THE COLLECTIVE IS HAND-BUILT (P1's finding, reused verbatim).
//
// Upstream CUTLASS 4.3.4's sm120 dense CollectiveBuilder refuses 16-bit input:
// `sm120_mma_builder.inl` static_asserts `is_sm10x_f8f6f4_element<ElementA>()
// && ...` and its `rr_op_selector_sm120` unconditionally returns the 8-bit
// `SM120_16x8x32_TN` atom. The MAINLOOP is type-generic, so what is missing is
// only the builder's operand selection. We therefore assemble the same four
// choices the builder makes for f8f6f4, in their 16-bit forms — the identical
// selection cb_bf16_grouped_gemm.cu's sm12x lane proved:
//
//   MMA atom      SM80_16x8x16_F32BF16BF16F32_TN (rmem-sourced, as required)
//   Atom layout   4x2x1 warps = 256 threads, the builder's cooperative shape
//   Permutation   Tile<_128,_32,_16> (one ldmatrix.x4 fills a B fragment)
//   Smem atom     rs_smem_selector<K-major> — CUTLASS's own selector
//
// ---------------------------------------------------------------------------
// TILE SHAPE AND SMEM (measured by csrc/tools/smem_probe_fp4v2_bf16.cu,
// host-only, no launch; re-run it whenever anything below changes).
//
// TileF = 128 x 64 x 64, Stages = 2. TileM=128 because the sm90 cooperative
// kernel layer requires it; narrow N (64) is the fp8 fused lane's proven mid-M
// pattern and it is what buys the codebook its smem headroom. TileK=64 keeps
// the 128-byte K-major swizzle atom (64 bf16 = 128 B) — the same reason the
// P1 grouped lane chose TileK=64 — and divides the 256-weight CB superblock
// exactly 4 ways.
//
//   GemmKernel::SharedStorageSize, TileM=128 TileN=64 (sm120 cap = 101,376 B)
//   TileK  Stages  Lut=0    Lut=4Ki   Lut=16Ki  Lut=32Ki  Lut=48Ki
//    64      2     52,224   56,320    68,608    84,992    101,376  (0 margin)
//    64      3     68,608   72,704    84,992    101,376   117,760  OVER
//   128      2     93,184   97,280    109,568   OVER      OVER
//   TileN=128/TileK=64/Stages=2: 60,416 / 64,512 / 76,800 / 93,184 / OVER
//
// SHIPPED: TileN=64, TileK=64, Stages=2, with Lut in {0, 4Ki, 16Ki, 32Ki}.
// The 48 KiB stage lands on EXACTLY the 101,376-byte ceiling with zero margin
// and is deliberately NOT compiled (the fp8 lane's TileM=256/k32 entry is the
// precedent for treating a zero-margin config as untrusted). Every shipped
// configuration is 1 CTA/SM, which is what the fp8 mid-M twin also runs at
// while winning 1.04-1.45x — at M <= 128 there is exactly ONE M-tile, so the
// grid is N/64 CTAs and occupancy per SM is not the limiter the P1 grouped
// lane found it to be.
//
// ---------------------------------------------------------------------------
// CODEBOOK RESIDENCY LADDER. The fp4-v2 product dictionary is BF16 VALUES —
// (8 << ceil(k/2)) + (8 << floor(k/2)) bytes — so it grows from 1 KiB at k12 to
// 64 KiB at k24 and cannot be staged whole at the top of the ladder. The smem
// stage holds a PREFIX of the flat `[sub0 | sub1]` codebook and the mainloop
// selects its two gather pointers from the staged length once per decode, so a
// partially staged table costs a pointer select and never a per-gather branch.
//
//   k    cb bytes   compiled Lut   staged      residency
//   12     1,024        4,096       1,024      full
//   13     1,536        4,096       1,536      full
//   14     2,048        4,096       2,048      full
//   15     3,072        4,096       3,072      full
//   16     4,096        4,096       4,096      full
//   17     6,144       16,384       6,144      full
//   18     8,192       16,384       8,192      full
//   19    12,288       16,384      12,288      full
//   20    16,384       16,384      16,384      full
//   21    24,576       32,768      24,576      full
//   22    32,768       32,768      32,768      full
//   23    49,152       32,768      32,768      sub0 only (sub1 global)
//   24    65,536       32,768      32,768      sub0 only (sub1 global)
//
// The whole K12..K24 product ladder is served by FOUR compiled kernels because
// k_bits is a RUNTIME parameter here: unlike the fp8 fork (whose CbTypeSize
// sizes a TMA box and a k-sized smem layout, forcing compile-time dispatch),
// this lane's packed stream never touches a TMA descriptor or a k-sized smem
// layout. That is the same property cb_fused_fp4_gemm.cu already exploits.
//
// ---------------------------------------------------------------------------
// NUMERICS. Plain alpha=1 / beta=0 epilogue — there is NO scale epilogue and
// none may be added. The quality path's activations are BF16 QDQ'd upstream and
// the v2-decoded weights already carry their two-tier scale, exactly as the
// BF16 bridge consumes them. fp32 accumulate, one bf16 round: the bridge path's
// numerics class, unchanged.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_fork/sm120_cb_fp4v2_bf16_mma.hpp"
// The PASSTHROUGH mainloop (plain BF16 B through TMA) at the SAME tile and
// schedule — the bit-exactness REFERENCE, see `sm120_fp4v2_bf16_mm_fork`
// below. It is P1's expert-indexed fork used with ptr_expert_ids == nullptr,
// i.e. the unmodified upstream mainloop.
#include "cutlass_fork/sm120_bf16_expert_mma.hpp"
// AssertSmemFits — the hard smem gate the sm90 cooperative kernel layer does
// not perform for itself. Shared with every other Gridbook CUTLASS lane
// (2026-08-01 audit §4 dedupe #1).
#include "cb_grouped_common.hpp"

namespace {

using namespace cute;

namespace cutlass_detail = cutlass::gemm::collective::detail;

using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
constexpr int kAlign = 8;                       // one 128-bit BF16 access
using ClusterShape = Shape<_1, _1, _1>;

// The shipped tile and stage count (see the smem table above).
using TileF = Shape<_128, _64, _64>;
constexpr int kStages = 2;
constexpr int64_t kMaxM = size<0>(TileF{});     // HARD mid-M gate: one M-tile

// Compiled codebook-stage classes, smallest first. 49,152 is deliberately
// absent: it lands on EXACTLY the sm120 ceiling with zero margin.
constexpr int64_t kLutClasses[] = {0, 4096, 16384, 32768};

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
  // Plain alpha=1 / beta=0: this lane has no scales and adding any would change
  // the served numerics rather than only the reduction order.
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
          Stages, /*SchedulerPipelineStageCount=*/2, ClusterShape,
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
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// The epilogue's TILE-SHAPE-dependent storage is what the P1 grouped lane also
// uses, so the two lanes' numerics classes are the same by construction. This
// asserts the shape family stays what the smem table above measured.
static_assert(size<0>(TileF{}) == 128 && size<1>(TileF{}) == 64 &&
                  size<2>(TileF{}) == 64,
              "the shipped fp4-v2 fused mid-M tile is 128x64x64; re-run "
              "csrc/tools/smem_probe_fp4v2_bf16.cu before changing it");

// ---------------------------------------------------------------------------
// Host-side rung arithmetic. `cb_expand_v2`'s flat product dictionary is
// [sub0 (4 << ceil(k/2) elements) | sub1 (4 << floor(k/2) elements)] BF16.
// ---------------------------------------------------------------------------
constexpr int64_t cb_sub0_bytes(int64_t k) { return int64_t(8) << ((k + 1) / 2); }
constexpr int64_t cb_sub1_bytes(int64_t k) { return int64_t(8) << (k / 2); }
constexpr int64_t cb_total_bytes(int64_t k) {
  return cb_sub0_bytes(k) + cb_sub1_bytes(k);
}
constexpr int64_t cb_elems(int64_t k) {
  return (int64_t(4) << ((k + 1) / 2)) + (int64_t(4) << (k / 2));
}

constexpr bool kbits_supported(int64_t k) { return k >= 12 && k <= 24; }

// Smallest compiled class that holds the FULL table; else the largest class
// that holds sub0; else no smem stage at all.
int64_t resolve_lut_bytes(int64_t k) {
  const int64_t total = cb_total_bytes(k);
  for (int64_t c : kLutClasses) {
    if (c > 0 && total <= c) return c;
  }
  const int64_t s0 = cb_sub0_bytes(k);
  for (int i = (int)(sizeof(kLutClasses) / sizeof(kLutClasses[0])) - 1;
       i >= 0; --i) {
    if (kLutClasses[i] > 0 && s0 <= kLutClasses[i]) return kLutClasses[i];
  }
  return 0;
}

// How many bytes of the flat codebook prefix actually go to smem for a given
// compiled class: the whole table when it fits, else sub0 alone, else nothing.
int64_t resolve_stage_bytes(int64_t k, int64_t lut_bytes) {
  if (lut_bytes <= 0) return 0;
  const int64_t total = cb_total_bytes(k);
  if (total <= lut_bytes) return total;
  const int64_t s0 = cb_sub0_bytes(k);
  return (s0 <= lut_bytes) ? s0 : 0;
}

// ---------------------------------------------------------------------------
// Runner.
// ---------------------------------------------------------------------------
template <int LutBytes>
torch::Tensor run_fused(torch::Tensor a, torch::Tensor packed,
                        torch::Tensor cb_flat, torch::Tensor compose,
                        int64_t N, int64_t K, int64_t k_bits,
                        int64_t stage_bytes, int64_t debug_mode) {
  using C = Cfg<TileF, kStages, LutBytes>;
  using GemmKernel = typename C::GemmKernel;
  using Gemm = typename C::Gemm;
  static_assert(gridbook::grouped::AssertSmemFits<GemmKernel>::value);

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
      {reinterpret_cast<const Element*>(a.data_ptr()), sa,
       packed.data_ptr<uint8_t>(), packed.stride(0),
       reinterpret_cast<const uint16_t*>(cb_flat.data_ptr()),
       compose.data_ptr<float>(),
       (int32_t)stage_bytes, (int32_t)k_bits, (int32_t)(4 * k_bits + 9),
       (int32_t)debug_mode},
      {{1.0f, 0.0f}, nullptr, StrideC{},
       reinterpret_cast<Element*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp4-v2 fused mid-M can_implement failed (K%256? row stride? "
              "k_bits in [12,24]?): ", cutlass::cutlassGetStatusString(status));
  status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp4-v2 fused mid-M initialize failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp4-v2 fused mid-M run failed: ",
              cutlass::cutlassGetStatusString(status));
  return d;
}

// ---------------------------------------------------------------------------
// PASSTHROUGH reference: the SAME TileF / TiledMma / smem layouts / epilogue,
// but B arrives as a plain dense BF16 [N, K] tile through TMA instead of being
// decoded in the prologue.
//
// THIS IS THE DECODE BIT-EXACTNESS ORACLE, and it is the fp8 lane's
// `sm120_fp8_mm_fork64` construction verbatim in spirit: because the two
// configs share the TiledMma, the k-block ordering and the epilogue, their
// FP32 accumulation ORDER is identical, so
//
//     cb_fused_fp4v2_prefill_mm(a, packed, cb, compose, ...)
//       ==  sm120_fp4v2_bf16_mm_fork(a, cb_expand_v2(packed, cb, compose))
//
// bit-for-bit IF AND ONLY IF the in-prologue decode reproduces `cb_expand_v2`
// exactly. It doubles as the fork-without-change regression for the
// passthrough mainloop at this tile.
// ---------------------------------------------------------------------------
template <class TileShape>
struct PassthroughCfg {
  using Epilogue = typename Cfg<TileShape, kStages, 0>::Epilogue;

  using DispatchPolicy =
      cutlass::gemm::MainloopSm120CbBf16ExpertTmaWarpSpecialized<
          kStages, /*SchedulerPipelineStageCount=*/2, ClusterShape,
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<2>>;

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

// The oracle and the lane must share every type that decides the reduction
// order. Proving it by TYPE IDENTITY is how this tree already proves kernel
// equivalence (see the `MoeTile<128> == TileF` assert in cb_fused_gemm.cu).
static_assert(
    cute::is_same_v<typename Cfg<TileF, kStages, 0>::Mainloop::TiledMma,
                    typename PassthroughCfg<TileF>::Mainloop::TiledMma>,
    "the passthrough oracle must share the fused lane's TiledMma: the atom, "
    "the warp layout and the permutation are what fix the FP32 accumulation "
    "order the bit-exactness gate relies on");
static_assert(
    cute::is_same_v<typename Cfg<TileF, kStages, 0>::Mainloop::SmemLayoutA,
                    typename PassthroughCfg<TileF>::Mainloop::SmemLayoutA>,
    "the passthrough oracle must share the fused lane's A staging, or the "
    "k-block iteration the two mainloops perform is not the same one");
static_assert(
    cute::is_same_v<typename Cfg<TileF, kStages, 0>::Epilogue,
                    typename PassthroughCfg<TileF>::Epilogue>,
    "the passthrough oracle must share the fused lane's epilogue collective, "
    "or output differences would not isolate the decode");

torch::Tensor sm120_fp4v2_bf16_mm_fork(torch::Tensor a, torch::Tensor b) {
  using C = PassthroughCfg<TileF>;
  using GemmKernel = typename C::GemmKernel;
  using Gemm = typename C::Gemm;
  static_assert(gridbook::grouped::AssertSmemFits<GemmKernel>::value);

  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 &&
                  b.is_cuda() && b.scalar_type() == torch::kBFloat16,
              "a and b must be CUDA bf16 tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1),
              "expected a [M,K] and b [N,K]");
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(K % kAlign == 0 && N % kAlign == 0,
              "K and N must be multiples of ", kAlign, " BF16 elements");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == K && b.stride(1) == 1 &&
                  b.stride(0) == K,
              "a and b must be contiguous row-major");

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const Element*>(a.data_ptr()), sa,
       reinterpret_cast<const Element*>(b.data_ptr()), sb,
       /*ptr_expert_ids=*/nullptr, /*num_experts=*/1},
      {{1.0f, 0.0f}, nullptr, StrideC{},
       reinterpret_cast<Element*>(d.data_ptr()), sd}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp4-v2 passthrough can_implement failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp4-v2 passthrough initialize failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "fp4-v2 passthrough run failed: ",
              cutlass::cutlassGetStatusString(status));
  return d;
}

// ---------------------------------------------------------------------------
// Host validation. Mirrors cb_fused_gemm.cu::check_fused_inputs; the extra
// checks are the fp4-v2 payload's own (odd type_size, product dictionary size,
// the two-tier compose table) and the HARD mid-M gate.
// ---------------------------------------------------------------------------
void check_fused_inputs(torch::Tensor a, torch::Tensor packed,
                        torch::Tensor cb_flat, torch::Tensor compose,
                        int64_t N, int64_t K, int64_t k_bits) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16,
              "a must be a CUDA bf16 [M,K] activation (the quality path's "
              "group-16 QDQ output)");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) == K,
              "a must be contiguous [M,", K, "]");
  TORCH_CHECK(kbits_supported(k_bits),
              "unsupported k_bits ", k_bits,
              " (the fp4-v2 product ladder is K12..K24)");
  const int64_t type_size = 4 * k_bits + 9;
  TORCH_CHECK(K > 0 && K % 256 == 0, "K must be a positive multiple of 256");
  TORCH_CHECK(N > 0 && N % kAlign == 0,
              "N must be a positive multiple of ", kAlign, " BF16 elements");
  // Mid-M ONLY, by construction: decode-in-prologue re-decodes B once per
  // M-tile, so beyond one tile the redundancy dominates (the fp8 twin measured
  // 0.22x at M~1400). Refusing here means a mis-set dispatch gate can never
  // quietly serve a slow kernel.
  TORCH_CHECK(a.size(0) >= 1 && a.size(0) <= kMaxM,
              "the fp4-v2 fused mid-M lane serves 1 <= M <= ", kMaxM,
              " (ONE M-tile) by construction, got M=", a.size(0),
              "; larger M must use the expand + BF16 bridge route");
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
                  packed.dim() == 2 && packed.size(0) == N &&
                  packed.stride(1) == 1,
              "packed must be a CUDA uint8 [N, row_bytes] tensor with unit "
              "column stride");
  TORCH_CHECK(packed.stride(0) >= (K / 256) * type_size,
              "packed row stride ", packed.stride(0),
              " is too small for K=", K, " at k_bits=", k_bits,
              " (need >= ", (K / 256) * type_size, ")");
  TORCH_CHECK(cb_flat.is_cuda() && cb_flat.scalar_type() == torch::kBFloat16 &&
                  cb_flat.dim() == 1 && cb_flat.is_contiguous() &&
                  cb_flat.numel() == cb_elems(k_bits),
              "cb_flat must be a contiguous CUDA bf16 vector of ",
              cb_elems(k_bits), " elements (the zero-based product dictionary "
              "for k=", k_bits, "), got ", cb_flat.numel());
  TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kFloat32 &&
                  compose.is_contiguous() && compose.numel() == 256 * 16,
              "compose must be a contiguous CUDA float32 tensor with 4096 "
              "elements (the two-tier v2 table)");
  TORCH_CHECK(a.device() == packed.device() && a.device() == cb_flat.device() &&
                  a.device() == compose.device(),
              "every fp4-v2 fused mid-M tensor must be on one CUDA device");
}

torch::Tensor cb_fused_fp4v2_prefill_mm(torch::Tensor a, torch::Tensor packed,
                                        torch::Tensor cb_flat,
                                        torch::Tensor compose, int64_t N,
                                        int64_t K, int64_t k_bits,
                                        int64_t force_lut_bytes,
                                        int64_t debug_mode) {
  check_fused_inputs(a, packed, cb_flat, compose, N, K, k_bits);
  // TESTS ONLY, and validated so a typo cannot silently return coordinates
  // instead of weights (see the mainloop's `debug_mode`).
  TORCH_CHECK(debug_mode >= 0 && debug_mode <= 3,
              "debug_mode must be 0 (decode), or 1/2/3 to write the decoder's "
              "row / column / K-tile coordinate; got ", debug_mode);
  int64_t lut_bytes = force_lut_bytes;
  if (lut_bytes < 0) {
    lut_bytes = resolve_lut_bytes(k_bits);
  } else {
    bool ok = false;
    for (int64_t c : kLutClasses) ok = ok || (c == lut_bytes);
    TORCH_CHECK(ok, "force_lut_bytes=", lut_bytes,
                " is not a compiled codebook-stage class; query "
                "cb_fused_fp4v2_lut_classes()");
  }
  const int64_t stage_bytes = resolve_stage_bytes(k_bits, lut_bytes);
  switch (lut_bytes) {
    case 0:     return run_fused<0>(a, packed, cb_flat, compose, N, K, k_bits, stage_bytes, debug_mode);
    case 4096:  return run_fused<4096>(a, packed, cb_flat, compose, N, K, k_bits, stage_bytes, debug_mode);
    case 16384: return run_fused<16384>(a, packed, cb_flat, compose, N, K, k_bits, stage_bytes, debug_mode);
    case 32768: return run_fused<32768>(a, packed, cb_flat, compose, N, K, k_bits, stage_bytes, debug_mode);
    default:
      TORCH_CHECK(false, "no compiled fp4-v2 fused kernel for a ", lut_bytes,
                  "-byte codebook stage");
  }
}

// --- attestation surface (host-only; no launch, no device needed) ----------

// The HARD mid-M ceiling, read by the python dispatch so the gate can never
// drift from the kernel that enforces it.
int64_t cb_fused_fp4v2_max_m() { return kMaxM; }

// Every k_bits the lane compiles a kernel for. Python must enumerate THIS.
std::vector<int64_t> cb_fused_fp4v2_kbits() {
  std::vector<int64_t> out;
  for (int64_t k = 12; k <= 24; ++k) {
    if (kbits_supported(k) && resolve_lut_bytes(k) >= 0) out.push_back(k);
  }
  return out;
}

// The compiled codebook-stage classes, smallest first.
std::vector<int64_t> cb_fused_fp4v2_lut_classes() {
  std::vector<int64_t> out;
  for (int64_t c : kLutClasses) out.push_back(c);
  return out;
}

// [tile_m, tile_n, tile_k, stages, sm120 capacity].
std::vector<int64_t> cb_fused_fp4v2_config() {
  return {int64_t(size<0>(TileF{})), int64_t(size<1>(TileF{})),
          int64_t(size<2>(TileF{})), int64_t(kStages),
          int64_t(cutlass::arch::sm120_smem_capacity_bytes)};
}

template <int LutBytes>
static void push_class(std::vector<int64_t>& out) {
  using C = Cfg<TileF, kStages, LutBytes>;
  out.push_back(LutBytes);
  out.push_back((int64_t)C::GemmKernel::SharedStorageSize);
}

// Flat [lut_bytes, SharedStorageSize] x |classes|, then
// [-1, sm120_smem_capacity_bytes].
std::vector<int64_t> cb_fused_fp4v2_smem_report() {
  std::vector<int64_t> out;
  push_class<0>(out);
  push_class<4096>(out);
  push_class<16384>(out);
  push_class<32768>(out);
  out.push_back(-1);
  out.push_back((int64_t)cutlass::arch::sm120_smem_capacity_bytes);
  return out;
}

// Flat [k_bits, cb_bytes, lut_bytes, stage_bytes] x 13 — the residency ladder
// exactly as the kernel resolves it, so a test can assert the shipped table.
std::vector<int64_t> cb_fused_fp4v2_lut_plan() {
  std::vector<int64_t> out;
  for (int64_t k = 12; k <= 24; ++k) {
    const int64_t lb = resolve_lut_bytes(k);
    out.push_back(k);
    out.push_back(cb_total_bytes(k));
    out.push_back(lb);
    out.push_back(resolve_stage_bytes(k, lb));
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm120_fp4v2_bf16_mm_fork", &sm120_fp4v2_bf16_mm_fork,
        "PASSTHROUGH bf16 GEMM through the same 128x64x64 config the fused "
        "lane uses (plain dense B via TMA). This is the fused kernel's "
        "bit-exactness ORACLE: identical TiledMma / k-block ordering / "
        "epilogue means identical FP32 reduction order, so equality with "
        "cb_fused_fp4v2_prefill_mm on the cb_expand_v2 tile is exactly the "
        "claim that the in-prologue decode reproduces cb_expand_v2.",
        py::arg("a"), py::arg("b"));
  m.def("cb_fused_fp4v2_prefill_mm", &cb_fused_fp4v2_prefill_mm,
        "FP4-CB v2 QUALITY decode-in-prologue fused GEMM (mid-M). B is the "
        "packed fp4-v2 byte stream plus its flat bf16 product dictionary and "
        "the (256,16) fp32 two-tier compose table; the decoded BF16 tile never "
        "exists in HBM and is bit-identical to cb_expand_v2's output. A is the "
        "shipping quality path's BF16 group-16-QDQ'd activation. fp32 "
        "accumulate, alpha=1/beta=0, ONE bf16 round — the BF16 bridge's "
        "numerics class with a different fp32 reduction order. 1 <= M <= "
        "cb_fused_fp4v2_max_m() is enforced.",
        py::arg("a"), py::arg("packed"), py::arg("cb_flat"), py::arg("compose"),
        py::arg("N"), py::arg("K"), py::arg("k_bits"),
        py::arg("force_lut_bytes") = -1, py::arg("debug_mode") = 0);
  m.def("cb_fused_fp4v2_max_m", &cb_fused_fp4v2_max_m,
        "the HARD mid-M ceiling (one M-tile) this lane enforces");
  m.def("cb_fused_fp4v2_kbits", &cb_fused_fp4v2_kbits,
        "every k_bits rung the lane serves — enumerate THIS, never a union");
  m.def("cb_fused_fp4v2_lut_classes", &cb_fused_fp4v2_lut_classes,
        "the compiled codebook smem-stage classes, smallest first");
  m.def("cb_fused_fp4v2_config", &cb_fused_fp4v2_config,
        "[tile_m, tile_n, tile_k, stages, sm120 capacity]");
  m.def("cb_fused_fp4v2_smem_report", &cb_fused_fp4v2_smem_report,
        "flat [lut_bytes, SharedStorageSize] per compiled class, then "
        "[-1, sm120 capacity]");
  m.def("cb_fused_fp4v2_lut_plan", &cb_fused_fp4v2_lut_plan,
        "flat [k_bits, cb_bytes, lut_bytes, stage_bytes] x 13 — the codebook "
        "residency ladder as the kernel resolves it");
}
