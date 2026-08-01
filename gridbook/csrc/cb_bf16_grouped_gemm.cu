// Native CUTLASS grouped BF16 GEMM used by the quality-preserving FP4-CB MoE
// bridge.  The packed FP4-v2 weights are decoded to BF16 by Gridbook's native
// CUDA expander before this op; activations are likewise the exact BF16 result
// of Gridbook's group-16 QDQ.  This kernel therefore changes only the GEMM
// launch topology: one device-scheduled CUTLASS group instead of one GEMM per
// expert.  Accumulation is FP32 and the epilogue rounds once to BF16.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"

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
}
