/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------------
// PrismaQuant expert-indexed row-broadcast epilogue node.
//
// VERBATIM COPY of cutlass::epilogue::fusion::Sm90RowBroadcast from the
// vendored CUTLASS 4.3.4
// (include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp,
// "Row vector broadcast"), renamed Sm120CbExpertRowBroadcast, with exactly two
// additions, both marked "PRISMAQUANT ADDITION":
//   1. Arguments gains {int const* ptr_expert_ids; int scale_stride;}
//   2. get_consumer_store_callbacks offsets ptr_row by
//      expert_ids[m_tile] * scale_stride (negative ids clamped to 0).
// With ptr_expert_ids == nullptr the node is bit-identical to the original,
// which is what the dense (non-MoE) path relies on.
//
// Motivation: the tile-indexed MoE grouping scheme (one ordinary single-problem
// GEMM over row-padded/permuted A, each M-tile reading a different expert's
// weights) needs the per-output-channel weight scale to come from
// b_scales[E, N] row expert_ids[m], not from a single [N] vector.
// ---------------------------------------------------------------------------

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector broadcast, EXPERT-INDEXED (see PRISMAQUANT ADDITION markers)
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct Sm120CbExpertRowBroadcast {
  using StrideMNL = StrideMNL_;
  // Get base element input type.
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  // Check if input is an array of pointers.
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrRowType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static_assert(Stages == 0, "Row broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<1>(StrideMNL{}))>, bool>; // row vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{} || IsDynamicBroadcast);

  struct SharedStorage {
    array_aligned<ElementInput, size<1>(CtaTileShapeMNK{})> smem;
  };

  struct Arguments {
    PtrRowType ptr_row = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dRow = {};
    // ---- PRISMAQUANT ADDITION (tile-indexed MoE grouping) ----
    // ptr_expert_ids[m_tile] selects which expert's row-vector slice this
    // output tile reads: base = ptr_row + expert_ids[m] * scale_stride.
    // nullptr => byte-for-byte the upstream Sm90RowBroadcast behaviour.
    int const* ptr_expert_ids = nullptr;
    int scale_stride = 0;                 // per-expert element stride (= N)
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm120CbExpertRowBroadcast() { }

  CUTLASS_HOST_DEVICE
  Sm120CbExpertRowBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false),
        smem(const_cast<ElementInput*>(shared_storage.smem.data())) {
    auto const& [stride_M, stride_N, stride_L] = params.dRow;
    // Nullptr default
    if (EnableNullptr && params.ptr_row == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_N == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_row[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class GS_GTensor, class GS_STensor, class GS_CTensor, class Tiled_G2S, class SR_STensor, class SR_RTensor, class Residue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        GS_GTensor tGS_gRow_, GS_STensor tGS_sRow_,
        GS_CTensor tGS_cRow_, Tiled_G2S tiled_g2s_,
        SR_STensor tSR_sRow_, SR_RTensor tSR_rRow_,
        Residue residue_cRow_, Params const& params_)
      : tGS_gRow(tGS_gRow_)
      , tGS_sRow(tGS_sRow_)
      , tGS_cRow(tGS_cRow_)
      , tiled_G2S(tiled_g2s_)
      , tSR_sRow(tSR_sRow_)
      , tSR_rRow(tSR_rRow_)
      , residue_cRow(residue_cRow_)
      , params(params_) {
    }

    GS_GTensor tGS_gRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_STensor tGS_sRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_CTensor tGS_cRow;                                                         // (CPY,CPY_M,CPY_N)
    Tiled_G2S tiled_G2S;

    SR_STensor tSR_sRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    SR_RTensor tSR_rRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    Residue residue_cRow;                                                        // (m, n)
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      bool is_nullptr = EnableNullptr && params.ptr_row == nullptr;

      Tensor tGS_gRow_flt = filter_zeros(tGS_gRow);
      Tensor tGS_sRow_flt = filter_zeros(tGS_sRow);
      Tensor tGS_cRow_flt = filter_zeros(tGS_cRow, tGS_gRow.stride());

      for (int i = 0; i < size(tGS_gRow_flt); ++i) {
        if (get<1>(tGS_cRow_flt(i)) >= size<1>(CtaTileShapeMNK{})) {
          continue; // OOB of SMEM,
        }
        if (not is_nullptr && elem_less(tGS_cRow_flt(i), residue_cRow)) {
          tGS_sRow_flt(i) = tGS_gRow_flt(i); // issue async gmem to smem load
        }
        else {
          tGS_sRow_flt(i) = params.null_default; // fill OOB values so smem to RF load can issue without predication
        }
      }
    }

    CUTLASS_DEVICE bool
    begin_sync_needed() const {
      return true; // Ensure visibility of async gmem to smem loads
    }

    CUTLASS_DEVICE void
    begin_loop(int epi_m, int epi_n) {
      if (epi_m == 0) { // Assumes M-major subtile loop
        Tensor tSR_sRow_flt = filter_zeros(tSR_sRow(_,_,_,epi_m,epi_n));
        Tensor tSR_rRow_flt = make_tensor_like<ElementInput>(tSR_sRow_flt);
        copy_aligned(tSR_sRow_flt, tSR_rRow_flt);

        constexpr int FrgSize = size(tSR_rRow_flt);
        using FrgInput = Array<ElementInput, FrgSize>;
        using FrgCompute = Array<ElementCompute, FrgSize>;
        using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FrgSize>;

        Tensor tSR_rRow_input_frg = recast<FrgInput>(coalesce(tSR_rRow_flt));
        Tensor tSR_rRow_compute_frg = recast<FrgCompute>(filter(tSR_rRow));
        ConvertInput convert_input{};

        tSR_rRow_compute_frg(_0{}) = convert_input(tSR_rRow_input_frg(_0{}));
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_row;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_row[i] = tSR_rRow(epi_v * FragmentSize + i);
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using ThreadCount = decltype(size(args.tiled_copy));

    auto layout_N = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_N = get<1>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_N = repeat_like(shape_N, int(0));
        if (get<1>(params.dRow) == bool(1)) {
          stride_N = transform_leaf(compact_major<LayoutLeft>(shape_N),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_N, stride_N);
      }
      else {
        return make_layout(shape_N);
      }
    }();

    auto layout_M = make_layout(M, repeat_like(M, _0{}));
    auto layout_L = make_layout(L, get<2>(params.dRow));
    ElementInput const* ptr_row = nullptr;
    if constexpr(IsArrayOfPointers) {
      if (!(EnableNullptr && params.ptr_row == nullptr)) {
        ptr_row = params.ptr_row[l];
      }
    } else {
      ptr_row = params.ptr_row;
    }
    // ---- PRISMAQUANT ADDITION: expert-indexed base pointer ----
    // Only this pointer offset differs from the upstream node; every other
    // line (smem staging, predication, EnableNullptr, conversion order) is
    // verbatim, so the numerics and rounding order are unchanged.
    if (params.ptr_expert_ids != nullptr && ptr_row != nullptr) {
      int expert_id = __ldg(params.ptr_expert_ids + m);
      if (expert_id < 0) { expert_id = 0; }
      ptr_row += static_cast<int64_t>(expert_id) * static_cast<int64_t>(params.scale_stride);
    }
    Tensor mRow = make_tensor(make_gmem_ptr(ptr_row), make_layout(layout_M,layout_N,layout_L));
    Tensor gRow = local_tile(mRow(_,_,l), take<0,2>(args.tile_shape_mnk), make_coord(m, n));          // (CTA_M, CTA_N)
    Tensor sRow = make_tensor(make_smem_ptr(smem),
        make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_0{}, _1{}));  // (CTA_M, CTA_N)
    //// G2S: Gmem to Smem
    auto tiled_g2s = make_tiled_copy(Copy_Atom<DefaultCopy, ElementInput>{},
                                     Layout< Shape<_1, ThreadCount>,
                                            Stride<_0,          _1>>{},
                                     Layout<_1>{});
    auto thr_g2s = tiled_g2s.get_slice(args.thread_idx);
    Tensor tGS_gRow = thr_g2s.partition_S(gRow);
    Tensor tGS_sRow = thr_g2s.partition_D(sRow);

    //// G2S: Coord
    Tensor tGS_cRow = thr_g2s.partition_S(args.cD);

    //// S2R: Smem to Reg
    Tensor tSR_sRow = sm90_partition_for_epilogue<ReferenceSrc>(sRow, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tSR_rRow = make_tensor_like<ElementCompute>(take<0,3>(tSR_sRow));                        // (CPY,CPY_M,CPY_N)

    return ConsumerStoreCallbacks(
      tGS_gRow,
      tGS_sRow,
      tGS_cRow, tiled_g2s,
      tSR_sRow,
      tSR_rRow,
      args.residue_cD,
      params);
  }
};



/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue::fusion
