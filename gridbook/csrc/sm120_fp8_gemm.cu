// CUTLASS baseline-parity gate (cutlass-kernel-notes.md build step 1):
// a plain sm120 fp8 W8A8 GEMM with per-token x per-channel scale epilogue,
// built from the vendored CUTLASS headers, matching vLLM's
// `cutlass_scaled_mm(a_fp8, b_fp8.t(), scale_a, scale_b, bf16)` numerically
// and (near enough) on speed for real 27B shapes.
//
// Purpose: prove we can own the exact collective/epilogue configuration the
// fused CB prefill will fork, BEFORE touching the mainloop. The CB fork then
// replaces only the B-operand global->shared producer (packed k-bit indices ->
// smem LUT decode -> fp8 tile), keeping this scale/MMA/epilogue path verbatim.
//
// D[m,n] = bf16( scale_a[m] * scale_b[n] * sum_k A[m,k]*B[n,k] )
// A: [M,K] e4m3 row-major (per-token dynamic-quantized activations)
// B: [N,K] e4m3 (weights; passed K-major = CUTLASS column-major N x K)
// scale_a: [M] fp32 ; scale_b: [N] fp32 ; D: [M,N] bf16 row-major.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace {

using namespace cute;

using ElementAB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;

// A row-major [M,K]; B column-major [K,N] view of the K-major [N,K] weight.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;  // 16
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;    // 8

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Baseline-gate epilogue: plain acc -> bf16 (alpha=1). The per-token x
// per-channel scale EVT is aligned with vLLM's ScaledEpilogue in a follow-up;
// for the gate the scales are applied outside (numerics-identical, and the
// mainloop — the part the CB fork will own — is what is being de-risked).
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        void, LayoutD, AlignD,          // no C source
        ElementD, LayoutD, AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementAB, LayoutA, AlignAB,
        ElementAB, LayoutB, AlignAB,
        ElementAcc,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor sm120_fp8_scaled_mm(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor scale_a, torch::Tensor scale_b) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1),
              "a [M,K], b [N,K] (K-major weight)");
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  TORCH_CHECK(K % AlignAB == 0, "K must be 16-aligned for fp8 TMA");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == K, "a must be contiguous");
  TORCH_CHECK(b.stride(1) == 1 && b.stride(0) == K, "b must be K-contiguous");
  TORCH_CHECK(scale_a.numel() == M && scale_b.numel() == N);

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d = torch::empty({M, N},
                        a.options().dtype(torch::kBFloat16));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideD stride_d =
      cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const ElementAB*>(a.data_ptr()), stride_a,
       reinterpret_cast<const ElementAB*>(b.data_ptr()), stride_b},
      {{1.0f, 0.0f},
       nullptr, StrideD{},  // no C
       reinterpret_cast<ElementD*>(d.data_ptr()), stride_d}};

  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({(int64_t)ws},
                                a.options().dtype(torch::kUInt8));
  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "can_implement failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.initialize(args, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "initialize failed");
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "run failed");
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm120_fp8_scaled_mm", &sm120_fp8_scaled_mm,
        "baseline sm120 fp8 GEMM w/ per-token x per-channel scales");
}
