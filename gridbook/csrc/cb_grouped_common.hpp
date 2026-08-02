// Shared glue for Gridbook's ROW-PADDED, TILE-INDEXED grouped GEMM
// construction (2026-08-01 performance audit, §4 dedupe #1/#2).
//
// WHY THIS FILE EXISTS. Upstream CUTLASS 4.3.4 has no sm120 ptr-array/grouped
// collective (`sm120_mma_builder.inl` static_asserts `!IsPtrArrayKernel`), so
// every Gridbook grouped kernel on this architecture uses the same
// construction instead: the CALLER pre-gathers and PADS A's rows so each
// expert's segment spans whole TileM blocks, B carries a batch (expert) mode,
// and each M-tile reads `expert_ids[m_tile]` as its B l-coordinate. One
// ordinary single-problem GEMM launch; no tensormap updates, no ptr-arrays.
//
// That construction was implemented independently three times — FP8-CB fused
// (`cb_fused_gemm.cu`), NVFP4-CB fused (`cb_fused_fp4_gemm.cu`) and, as of P1,
// the sm12x-native BF16 bridge (`cb_bf16_grouped_gemm.cu`). Becoming its third
// consumer is what forces the extraction. What is genuinely COMMON lives here:
//
//   * the `ScaledFusion` EVT node tree (verbatim x3 before this file),
//   * its expert-indexed `MoeScaledFusion` variant,
//   * `AssertSmemFits` — the hard smem gate the sm90-family kernel layers
//     (cooperative AND pingpong) do not perform for themselves,
//   * the tile-feasibility predicate scaffolding (`tile_sizes_where`),
//   * the host-side shape/stride/expert_ids validation every grouped binding
//     repeats.
//
// What is NOT common stays in each translation unit: element types, tile
// shapes, the mainloop fork, the packed-weight layout, and the rung ladders.
// Everything here is therefore PARAMETERIZED by element type / tile shape, and
// each consumer proves the parameterized types are the SAME TYPES its verbatim
// copy produced (see the `is_same_v` static_asserts at each use site). Bit
// identity of the generated kernels is a type identity question, and that is
// how the tree already proves it (`static_assert(is_same_v<MoeTile<128>,
// TileF>)` in `cb_fused_gemm.cu`).
//
// Gridbook-owned glue, deliberately NOT under `cutlass_fork/`: nothing here is
// a fork of a CUTLASS file. It composes public CUTLASS templates and Gridbook's
// own expert-indexed epilogue node.

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

#include "cutlass_fork/sm120_expert_row_broadcast.hpp"

namespace gridbook {
namespace grouped {

using namespace cute;

// ---------------------------------------------------------------------------
// EVT epilogue: per-token (row / M) activation scale x per-channel (col / N)
// weight scale applied in fp32 BEFORE the single round to ElementD — the same
// rounding ORDER as ops.cutlass_scaled_mm:
//
//   D = convert<ElementD>( b_scale[n] * ( a_scale[m] * acc_f32 ) )
//
// Node tree (the shape vLLM's ScaledEpilogue uses):
//   Sm90EVT<Compute<multiplies, D>, RowBroadcast(b_scales),
//           Sm90EVT<Compute<multiplies, f32>, ColBroadcast(a_scales),
//                   Sm90AccFetch>>
// ---------------------------------------------------------------------------
template <class TileShape, class ElementAcc, class ElementD>
struct ScaledFusion {
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, TileShape, float, float, Stride<_1, _0, _0>>;
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
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

// The grouped (MoE) variant: identical tree except that the B-scale node is
// Sm120CbExpertRowBroadcast (same body as Sm90RowBroadcast, base pointer
// offset by expert_ids[m_tile] * N), because the per-output-channel weight
// scale of a tile-indexed grouped launch comes from b_scales[E, N] row
// expert_ids[m], not from a single [N] vector. The A-scale ColBroadcast and
// the multiply/round order are untouched.
template <class TileShape, class ElementAcc, class ElementD>
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

// Hard smem gate. Neither sm90-family kernel layer — cooperative (the two
// fused lanes) nor pingpong (the sm12x BF16 lane) — static_asserts its own
// SharedStorageSize against the arch capacity; only the sm120 asymmetric-DMA
// kernel does. An over-budget config would therefore compile and then fail at
// launch. Every instantiated Gridbook config passes through this.
template <class GemmKernel>
struct AssertSmemFits {
  static_assert((int)GemmKernel::SharedStorageSize <=
                    cutlass::arch::sm120_smem_capacity_bytes,
                "Gridbook grouped CB kernel exceeds the sm_120 shared-memory "
                "capacity");
  static constexpr bool value = true;
};

// ---------------------------------------------------------------------------
// Tile-feasibility scaffolding.
//
// Each grouped lane compiles a MATRIX of (TileM x rung) kernels bounded by
// smem, and python must enumerate exactly what was compiled — never the union,
// or it can select an uninstantiated pair. Every lane therefore exposes a
// constexpr predicate plus this filter; only the predicate differs.
// ---------------------------------------------------------------------------
template <class Predicate>
inline std::vector<int64_t> tile_sizes_where(std::vector<int64_t> const& all,
                                             Predicate&& supported) {
  std::vector<int64_t> out;
  for (int64_t tm : all) {
    if (supported(tm)) out.push_back(tm);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Host-side validation shared by every tile-indexed grouped binding.
//
// These are the checks whose CONTRACT is the grouping construction itself
// (padding granularity, per-tile expert ids, stacked-expert contiguity), as
// opposed to a lane's own packed-format checks. Error text is preserved
// verbatim where a test matches on it.
// ---------------------------------------------------------------------------
inline void check_same_cuda_device(torch::Tensor const& anchor,
                                   torch::Tensor const& tensor,
                                   char const* name) {
  TORCH_CHECK(tensor.device() == anchor.device(), name,
              " must be on the same CUDA device as a (", anchor.device(),
              "), got ", tensor.device());
}

// Mp must span whole TileM blocks: an M-tile that straddled two experts would
// have no single B l-coordinate.
inline void check_padded_rows(int64_t mp, int64_t tile_m) {
  TORCH_CHECK(tile_m > 0, "grouped tile_m must be positive, got ", tile_m);
  TORCH_CHECK(mp % tile_m == 0,
              "Mp (", mp, ") must be a multiple of the grouped tile_m (",
              tile_m, "); pad each expert's row segment");
}

// One int32 expert id per M-tile, on the same device, contiguous. Padding
// tiles legally carry -1 (the mainloop clamps them; their rows are discarded
// by the caller's unpermute), so only the UPPER bound is enforced here.
inline void check_expert_ids(torch::Tensor const& anchor,
                             torch::Tensor const& expert_ids,
                             int64_t mp, int64_t tile_m, int64_t experts) {
  TORCH_CHECK(expert_ids.is_cuda() &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  expert_ids.is_contiguous() &&
                  expert_ids.numel() == mp / tile_m,
              "expert_ids must be contiguous int32 cuda [Mp/tile_m] (expected ",
              mp / tile_m, ", got ", expert_ids.numel(), ")");
  check_same_cuda_device(anchor, expert_ids, "expert_ids");
  TORCH_CHECK(experts > 0, "the stacked expert dimension must be positive");
}

// Row-source index vector for an IN-MAINLOOP A-row gather: one int32 per
// padded row, contiguous, on the anchor's device. Ids outside
// [0, source_rows) are the padding rows (they load zeros); like expert_ids,
// the CONTENT is the caller's contract — validating values would cost a
// device sync in the hot path.
inline void check_row_src(torch::Tensor const& anchor,
                          torch::Tensor const& row_src,
                          int64_t mp, int64_t tile_m) {
  TORCH_CHECK(row_src.is_cuda() && row_src.scalar_type() == torch::kInt32 &&
                  row_src.is_contiguous() && row_src.dim() == 1 &&
                  row_src.numel() == mp,
              "row_src must be a contiguous int32 cuda [Mp] vector (expected ",
              mp, ", got ", row_src.numel(), ")");
  check_same_cuda_device(anchor, row_src, "row_src");
  check_padded_rows(mp, tile_m);
}

// A stacked [E, N, W] weight buffer must be FULLY contiguous: the grouped
// mainloop turns the expert index into a single batch stride (N*W), which is
// only the expert's slice if no dimension is strided or permuted.
inline void check_stacked_experts(torch::Tensor const& stack, int64_t n,
                                  char const* name) {
  TORCH_CHECK(stack.is_cuda() && stack.dim() == 3 && stack.size(0) > 0 &&
                  stack.size(1) == n,
              name, " must be a CUDA [E, ", n, ", W] stack, got ",
              stack.sizes());
  const int64_t w = stack.size(2);
  TORCH_CHECK(stack.stride(2) == 1 && stack.stride(1) == w &&
                  stack.stride(0) == n * w,
              name, " must be fully contiguous [E, N, W]; the grouped kernel "
              "derives the per-expert batch stride as N*W");
}

}  // namespace grouped
}  // namespace gridbook
