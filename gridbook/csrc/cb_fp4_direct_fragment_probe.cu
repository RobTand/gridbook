// Research-only SM120 native-NVFP4 direct-to-register-fragment probe.
//
// This is deliberately a SEPARATE, default-off extension.  It answers one
// narrow feasibility question: can a resident K18 product-codebook row be
// decoded directly into the architectural B and SFB registers of
// OMMA.SF.16864, without materialising a dense decoded B tile in shared
// memory?  It does not register an operator, alter dispatch, or accept the
// shipping BF16 group-16-QDQ activation bucket.
//
// Numerical boundary (fail closed): SM120_16x8x64_TN_VS is typed as
//
//   E2M1 A x E2M1 B, float accumulator, UE4M3 SFA/SFB, scale_vec::4X.
//
// UE4M3 is an 8-bit, unsigned E4M3 lattice (CUTLASS float8.h: range 0..448,
// four exponent bits, three mantissa bits).  An arbitrary FP32 group-16 scale
// is therefore not exactly encodable.  This probe consumes the already-native
// activation factorisation used by cb_fused_fp4_gemm.cu instead: packed E2M1
// values + UE4M3 group bytes + one FP32 residual per row in the EVT.  Its only
// valid reference is the existing native-W4A4 fused/stock OMMA bucket, never
// the quality-preserving BF16-QDQ path.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <map>

#include <pybind11/stl.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass_fork/sm120_cb_fused_fp4_mma.hpp"
#include "cb_grouped_common.hpp"

namespace {

using namespace cute;

using ElementPairAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementSF = cutlass::float_ue4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using ClusterShape = Shape<_1, _1, _1>;
using TileShape = Shape<_128, _128, _128>;
using Sm1xxCfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int AlignAB = 32;
constexpr int AlignD = 8;
constexpr int ProbeKBits = 18;
constexpr int ProbeTypeSize = 4 * ProbeKBits + 9;
constexpr int ProbeLutBytes = 2048;
constexpr int ProbeComposeBytes = 4096;

using Fusion = typename gridbook::grouped::ScaledFusion<
    TileShape, ElementAcc, ElementD>::type;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    void, LayoutD, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    Fusion>::CollectiveOp;

using BuilderMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairAB, LayoutA, AlignAB,
    ElementPairAB, LayoutB, AlignAB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

template <int NewStages, class T>
struct SwapToK18DirectFragment;

template <int NewStages, int S, int SP, class CS, class KS, class... Rest>
struct SwapToK18DirectFragment<NewStages,
    cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<
            S, SP, CS, KS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120CbFusedFp4TmaWarpSpecialized<
          NewStages, SP, CS, KS, true, ProbeKBits>, Rest...>;
};

using Mainloop = typename SwapToK18DirectFragment<2, BuilderMainloop>::type;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
static_assert(gridbook::grouped::AssertSmemFits<GemmKernel>::value);
static_assert(Mainloop::DispatchPolicy::DirectFragment);
static_assert(Mainloop::DispatchPolicy::DirectKBits == ProbeKBits);

void check_same_device(torch::Tensor const& anchor,
                       torch::Tensor const& tensor,
                       char const* name) {
  gridbook::grouped::check_same_cuda_device(anchor, tensor, name);
}

void check_packed_row_storage(torch::Tensor const& packed, int64_t rows,
                              int64_t required_row_bytes) {
  TORCH_CHECK(packed.size(1) >= required_row_bytes &&
                  packed.stride(0) >= required_row_bytes,
              "packed visible width/stride is smaller than the required ",
              required_row_bytes, " K18 bytes per row");
  TORCH_CHECK(packed.storage_offset() >= 0,
              "packed storage offset must be nonnegative");
  uint64_t const storage_bytes = packed.storage().nbytes();
  uint64_t const storage_offset = packed.storage_offset();
  uint64_t const row_bytes = required_row_bytes;
  uint64_t const row_stride = packed.stride(0);
  uint64_t const rows_before_last = rows - 1;
  TORCH_CHECK(storage_offset <= storage_bytes &&
                  row_bytes <= storage_bytes - storage_offset,
              "packed backing storage is too small for the first K18 row");
  uint64_t const bytes_before_last =
      storage_bytes - storage_offset - row_bytes;
  TORCH_CHECK(rows_before_last == 0 ||
                  row_stride <= bytes_before_last / rows_before_last,
              "packed backing storage is too small for all K18 rows");
}

void check_problem_shape(int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(M == 128 || M == 4096,
              "K18 direct-fragment probe M must be 128 (correctness cell) "
              "or 4096 (production cell), got ", M);
  TORCH_CHECK(N == 4096,
              "K18 direct-fragment probe N is pinned to 4096, got ", N);
  TORCH_CHECK(K == 2048 || K == 4096,
              "K18 direct-fragment probe K must be 2048 or 4096, got ", K);
}

void check_inputs(
    torch::Tensor const& a, torch::Tensor const& sfa,
    torch::Tensor const& packed, torch::Tensor const& lut,
    torch::Tensor const& compose, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, int64_t N, int64_t K) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kUInt8 &&
                  a.dim() == 2 && a.is_contiguous() && a.size(1) * 2 == K,
              "a must be contiguous CUDA uint8 packed-E2M1 [M,K/2]");
  int64_t const M = a.size(0);
  check_problem_shape(M, N, K);

  int64_t const sfa_need = ((M + 127) / 128) * 128 * (K / 16);
  TORCH_CHECK(sfa.is_cuda() && sfa.scalar_type() == torch::kUInt8 &&
                  sfa.is_contiguous() && sfa.numel() == sfa_need,
              "sfa must be the native CUTLASS-swizzled UE4M3 plane with ",
              sfa_need, " bytes");

  int64_t const row_bytes = (K / 256) * ProbeTypeSize;
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 &&
                  packed.dim() == 2 && packed.stride(1) == 1 &&
                  packed.size(0) == N,
              "packed must be resident K18-v2 uint8[N,row_bytes] with unit "
              "inner stride (the existing right-padded residency tensor is "
              "accepted; no alternate streaming representation is built)");
  check_packed_row_storage(packed, N, row_bytes);
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kUInt8 &&
                  lut.is_contiguous() && lut.numel() == ProbeLutBytes,
              "lut must be exactly one resident K18 product LUT (2048 bytes)");
  TORCH_CHECK(compose.is_cuda() && compose.scalar_type() == torch::kUInt8 &&
                  compose.is_contiguous() &&
                  compose.numel() == ProbeComposeBytes,
              "compose must be the exact v2 E4M3 table (4096 bytes)");
  TORCH_CHECK(a_scales.is_cuda() &&
                  a_scales.scalar_type() == torch::kFloat32 &&
                  a_scales.is_contiguous() && a_scales.numel() == M,
              "a_scales must be the native per-row FP32 residual [M]");
  TORCH_CHECK(b_scales.is_cuda() &&
                  b_scales.scalar_type() == torch::kFloat32 &&
                  b_scales.is_contiguous() && b_scales.numel() == N,
              "b_scales must be contiguous CUDA float32 [N]");
  check_same_device(a, sfa, "sfa");
  check_same_device(a, packed, "packed");
  check_same_device(a, lut, "lut");
  check_same_device(a, compose, "compose");
  check_same_device(a, a_scales, "a_scales");
  check_same_device(a, b_scales, "b_scales");
}

typename Gemm::Arguments make_arguments(
    int M, int N, int K,
    uint8_t const* a, uint8_t const* sfa, uint8_t const* packed,
    uint8_t const* lut, uint8_t const* compose,
    float const* a_scales, float const* b_scales,
    cutlass::bfloat16_t* d, int64_t packed_row_bytes) {
  using StrideA = typename GemmKernel::StrideA;
  using StrideD = typename GemmKernel::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = Sm1xxCfg::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));

  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a), sa,
       reinterpret_cast<const ElementSF*>(sfa), layout_sfa,
       packed, packed_row_bytes,
       lut, int32_t(ProbeLutBytes),
       nullptr, int32_t(1), int32_t(0),
       compose,
       int32_t(ProbeKBits), int32_t(2), int32_t(ProbeTypeSize), int32_t(1),
       nullptr, int64_t(0), int32_t(0), nullptr},
      {{{b_scales, 0.0f, Stride<_0, _1, _0>{}},
        {{a_scales, 0.0f, Stride<_1, _0, _0>{}}, {}, {}},
        {}},
       nullptr, typename GemmKernel::StrideC{}, d, sd}};
}

int64_t cb_fp4_direct_fragment_k18_workspace_bytes(
    int64_t M, int64_t N, int64_t K) {
  check_problem_shape(M, N, K);
  auto args = make_arguments(
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      (K / 256) * ProbeTypeSize);
  return static_cast<int64_t>(Gemm::get_workspace_size(args));
}

void cb_fp4_direct_fragment_k18_out(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, torch::Tensor out, torch::Tensor workspace,
    int64_t N, int64_t K) {
  check_inputs(a, sfa, packed, lut, compose, a_scales, b_scales, N, K);
  int const M = static_cast<int>(a.size(0));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 &&
                  out.is_contiguous() && out.dim() == 2 &&
                  out.size(0) == M && out.size(1) == N,
              "out must be preallocated contiguous CUDA BF16 [M,N]");
  TORCH_CHECK(workspace.is_cuda() &&
                  workspace.scalar_type() == torch::kUInt8 &&
                  workspace.is_contiguous(),
              "workspace must be preallocated contiguous CUDA uint8");
  check_same_device(a, out, "out");
  check_same_device(a, workspace, "workspace");

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto args = make_arguments(
      M, static_cast<int>(N), static_cast<int>(K),
      a.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(),
      packed.data_ptr<uint8_t>(), lut.data_ptr<uint8_t>(),
      compose.data_ptr<uint8_t>(), a_scales.data_ptr<float>(),
      b_scales.data_ptr<float>(),
      reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
      packed.stride(0));
  size_t const ws = Gemm::get_workspace_size(args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(ws),
              "workspace has ", workspace.numel(), " bytes, needs ", ws);

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "K18 direct-fragment can_implement rejected the pinned cell");
  auto status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "K18 direct-fragment initialize failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "K18 direct-fragment launch failed: ",
              cutlass::cutlassGetStatusString(status));
}

torch::Tensor cb_fp4_direct_fragment_k18(
    torch::Tensor a, torch::Tensor sfa, torch::Tensor packed,
    torch::Tensor lut, torch::Tensor compose, torch::Tensor a_scales,
    torch::Tensor b_scales, int64_t N, int64_t K) {
  check_inputs(a, sfa, packed, lut, compose, a_scales, b_scales, N, K);
  auto out = torch::empty({a.size(0), N},
                          a.options().dtype(torch::kBFloat16));
  int64_t const ws = cb_fp4_direct_fragment_k18_workspace_bytes(
      a.size(0), N, K);
  auto workspace = torch::empty({ws}, a.options().dtype(torch::kUInt8));
  cb_fp4_direct_fragment_k18_out(
      a, sfa, packed, lut, compose, a_scales, b_scales,
      out, workspace, N, K);
  return out;
}

std::map<std::string, int64_t> cb_fp4_direct_fragment_probe_contract() {
  return {{"k_bits", ProbeKBits},
          {"n_sub", 2},
          {"scale_coding_v2", 1},
          {"type_size", ProbeTypeSize},
          {"lut_bytes", ProbeLutBytes},
          {"compose_bytes", ProbeComposeBytes},
          {"tile_m", 128}, {"tile_n", 128}, {"tile_k", 128},
          {"production_m", 4096}, {"production_n", 4096},
          {"production_k0", 2048}, {"production_k1", 4096},
          {"quality_bucket_native_w4a4_only", 1},
          {"arbitrary_fp32_group_scale_exact", 0}};
}

std::map<std::string, int64_t> cb_fp4_direct_fragment_resource_report() {
  // Operand/accumulator lower bound for the concrete 4x2 atom layout and
  // 128x128x128 CTA: C64 + A16 + SFA4 + B32 + SFB16 = 132 architectural
  // 32-bit registers per MMA thread before coordinates, decode temporaries,
  // pipeline state, and compiler allocation.  ptxas is the authoritative
  // post-compile number; this report makes the pre-compile risk explicit.
  return {{"kernel_shared_storage_bytes", GemmKernel::SharedStorageSize},
          {"mainloop_tensor_storage_bytes",
           sizeof(typename Mainloop::SharedStorage::TensorStorage)},
          {"operand_accumulator_register_floor", 132},
          {"sm120_shared_storage_ceiling",
           cutlass::arch::sm120_smem_capacity_bytes}};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cb_fp4_direct_fragment_k18", &cb_fp4_direct_fragment_k18,
        "Allocation-bearing K18 native-W4A4 correctness/microbenchmark probe");
  m.def("cb_fp4_direct_fragment_k18_out", &cb_fp4_direct_fragment_k18_out,
        "Allocation-free K18 native-W4A4 probe ABI for CUDA graph capture");
  m.def("cb_fp4_direct_fragment_k18_workspace_bytes",
        &cb_fp4_direct_fragment_k18_workspace_bytes,
        "Workspace bytes for a fail-closed pinned K18 probe cell");
  m.def("cb_fp4_direct_fragment_probe_contract",
        &cb_fp4_direct_fragment_probe_contract,
        "Static numerical and shape contract of the research probe");
  m.def("cb_fp4_direct_fragment_resource_report",
        &cb_fp4_direct_fragment_resource_report,
        "Shared-memory sizes and the pre-ptxas register lower bound");
}
