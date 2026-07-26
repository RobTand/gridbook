/***************************************************************************************************
 * PrismaQuant FP8_CB PERSISTENT-N fused mainloop for sm120 (GB10) — §4b DRAFT.
 *
 * STATUS: design-complete draft, NOT yet compiled (authored 2026-07-23 during
 * the driver-mismatch GPU outage; iterate in the next GPU window). Derived
 * from sm120_cb_fused_mma.hpp (the mid-M decode-in-prologue collective).
 *
 * The schedule (docs/nvfp4-cb-plan/persistent-n-prefill.md §4/§7):
 *   - Grid = one CTA per (N-tile stream); CTA c owns N-tiles {c, c+G, ...}
 *     (block-cyclic over the N-tile axis, G = gridDim.x). For each owned
 *     N-tile it sweeps ALL M-tiles before advancing (N-major, M-innermost —
 *     the opposite of both the default raster and the strided persistent
 *     scheduler, either of which changes N every visit and re-decodes).
 *   - On the FIRST M-tile of an owned N-tile, the packed-B TMA pipeline runs
 *     over all K-tiles and decode_stage() fills a FULL-K resident smem
 *     buffer sB_res[TileN, K] (e4m3 bytes). Every subsequent M-tile of that
 *     N-tile skips B-load and decode entirely: only the A pipeline runs, and
 *     the MMA reads B fragments from sB_res. Decode is paid once per
 *     N-tile per kernel — INV-1 (no [N,K] in HBM) and decode-amortization
 *     both hold.
 *   - v1 smem budget (K <= 3072): TileM=128, TileN=16, TileK=128.
 *       sB_res   : 16 x 3072      = 48 KB (e4m3 bytes, swizzled per K-tile)
 *       sA       : 128 x 128 x 2s = 32 KB (fp8, 2 stages)
 *       sBP      : 16 x 192  x 2s =  6 KB (packed stages, K48 worst case)
 *       epilogue + barriers       ~  4 KB
 *       total                     ~ 90 KB  (opt-in dynamic smem, 1 CTA/SM)
 *     TileN=16 costs MMA-N efficiency; acceptable for v1 correctness+measure.
 *     v2 (if v1 wins but wants wider N): decoded panel in a per-CTA HBM
 *     scratch pinned via L2 accessPolicyWindow, TileN=128.
 *
 * Work decomposition invariants:
 *   - ceil(N/TileN) streams; M-sweep length ceil(M/TileM). The epilogue is
 *     unchanged (per-tile accumulator writeout; no cross-tile state).
 *   - K % 256 == 0 (superblock), KBits in {36,40,44,48} (even sub-splits),
 *     one codebook per launch — all inherited from the mid-M kernel.
 *
 * Remaining GPU-window work (marked TODO(4b-gpu) inline):
 *   1. TMA descriptor for the A-only steady-state (reuse tma_load_a; the
 *      B-packed TMA fires only during first-M-tile visits — producer skips
 *      the copy but must still arrive on the barrier with the right
 *      transaction bytes, or use a separate A-only pipeline for the steady
 *      state. Draft uses the expected-bytes-adjust approach.)
 *   2. NamedBarrier ordering for the decode->resident handoff (single sync
 *      after the K-sweep decode, then none in the steady state).
 *   3. Scheduler integration: GemmUniversal's TileScheduler slot with
 *      CbPersistentNScheduler below; verify the epilogue's tile-coord
 *      derivation uses scheduler-provided coords (it does for the SM90-style
 *      cooperative kernels this forks).
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_math.h"

#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

namespace cutlass::gemm {

// ---------------------------------------------------------------------------
// N-major contiguous-range persistent scheduler.
//
// Linear work ids are (n_tile, m_tile) with m innermost:
//     work_id = n_idx * m_tiles + m_idx
// CTA c starts at n_idx = c and advances n_idx += grid_size when its M-sweep
// completes. is_first_m_tile() gates the decode+resident-fill phase.
// ---------------------------------------------------------------------------
struct CbPersistentNSchedulerParams {
  int m_tiles = 0;
  int n_tiles = 0;
  int grid = 0;
};

struct CbPersistentNScheduler {
  using Params = CbPersistentNSchedulerParams;

  struct WorkTileInfo {
    int m_idx = 0;
    int n_idx = 0;
    bool first_m = false;
    bool valid = false;
  };

  Params params;
  int cta_id = 0;
  int cur_n = 0;
  int cur_m = 0;

  CUTLASS_DEVICE
  CbPersistentNScheduler(Params const& p, int block_idx)
      : params(p), cta_id(block_idx), cur_n(block_idx), cur_m(0) {}

  CUTLASS_DEVICE WorkTileInfo
  initial_work() const {
    return {0, cur_n, /*first_m=*/true, cur_n < params.n_tiles};
  }

  // Advance M-innermost; roll to the CTA's next owned N stream at M end.
  CUTLASS_DEVICE WorkTileInfo
  next_work() {
    ++cur_m;
    if (cur_m >= params.m_tiles) {
      cur_m = 0;
      cur_n += params.grid;
      return {0, cur_n, /*first_m=*/true, cur_n < params.n_tiles};
    }
    return {cur_m, cur_n, /*first_m=*/false, true};
  }

  CUTLASS_HOST
  static dim3 grid_shape(Params const& p) {
    // One CTA per N stream up to the persistent grid bound; the host caps
    // grid at min(n_tiles, #SMs) so every CTA owns >= 1 stream.
    return dim3(static_cast<unsigned>(p.grid), 1, 1);
  }
};

template <
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_,
  int KBits_
>
struct MainloopSm120CbPersistentNTmaWarpSpecialized {
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

// The collective below forks sm120_cb_fused_mma.hpp's CollectiveMma. Only the
// deltas are sketched here in full; unchanged members (smem A plan, TMA
// params for A and packed-B, epilogue plumbing) are inherited structurally
// by copy at integration time — the draft keeps the file focused on the two
// load/mma entry points whose CONTROL FLOW changes.
//
// TODO(4b-gpu): materialize the full CollectiveMma specialization by copying
// sm120_cb_fused_mma.hpp and applying exactly the deltas below; kept as a
// patch-plan here so the GPU window starts from a reviewed control flow.
//
// ---- DELTA 1: TensorStorage ----------------------------------------------
//   struct TensorStorage {
//     ...unchanged smem_A[Stages], smem_BP[BPStages]...
//     alignas(128) cute::ArrayEngine<uint8_t, TileN * KMax> smem_B_res;
//   };
//   KMax is a compile-time bound (3072 for v1; static_assert K <= KMax at
//   host dispatch).
//
// ---- DELTA 2: load(): producer -------------------------------------------
//   for (WorkTileInfo w = sched.initial_work(); w.valid; w = sched.next_work()) {
//     for (kt in K-tiles) {
//       producer_acquire(stage);
//       tma_load_a(...m=w.m_idx, kt...);                  // every M-tile
//       if (w.first_m) tma_load_bp(...n=w.n_idx, sb=kt>>1...);
//       // expected-bytes must match what was issued:
//       //   first_m ? bytes(A_tile)+bytes(BP_slice) : bytes(A_tile)
//       // -> producer_commit with per-visit transaction bytes.
//     }
//   }
//
// ---- DELTA 3: mma(): consumer --------------------------------------------
//   for (WorkTileInfo w = sched.initial_work(); w.valid; w = sched.next_work()) {
//     clear(accum);
//     for (kt in K-tiles) {
//       consumer_wait(stage);
//       if (w.first_m) {
//         decode_stage(shared, sB_res_slice(kt), lut, stage, kt & 1, tid);
//         NamedBarrier::sync(ThreadCount, Sm120MainloopBarrier);
//       }
//       copy A stage -> rmem;  copy sB_res_slice(kt) -> rmem (B fragments);
//       gemm(tiled_mma, tCrA, tCrB, accum);
//       consumer_release(stage);
//     }
//     epilogue(accum, tile_coord(w.m_idx, w.n_idx));
//   }
//   The steady state (first_m == false) never touches smem_BP and never
//   syncs the decode barrier — B fragments come straight from smem_B_res.
//
// ---- DELTA 4: kernel/scheduler glue --------------------------------------
//   GemmUniversal<ProblemShape, MainloopPersistent, Epilogue,
//                 CbPersistentNScheduler>
//   Host computes Params{m_tiles, n_tiles, grid=min(n_tiles, sm_count)}.

}  // namespace cutlass::gemm::collective
