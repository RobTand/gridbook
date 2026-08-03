// MXFP8 dense W8A8 block-scaled GEMM (sm120/sm121).
//
// Serves two producer spellings of the SAME on-device format — E4M3 elements
// with one UE8M0 (F8_E8M0) exponent scale per 32 contiguous K elements:
//
//  * ``mxfp8_e4m3_e8m0_g32``: producer-emitted MXFP8, scales stored row-major
//    [rows, ceil(K/32)] on disk;
//  * ``fp8_e4m3_ue8m0_block128``: DeepSeek-convention block quantization
//    (one UE8M0 scale per 128x128 tile).  128 = 4 * 32, so a 32-wide MX chunk
//    never straddles a block boundary and the block form embeds EXACTLY into
//    MXFP8 by pure scale replication: SF_mx[n, c] = S_ds[n // 128, c // 4].
//    No element byte changes, no scale arithmetic, no rounding.  The Python
//    lane (gridbook/mxfp8.py) performs that broadcast at model load.
//
// The GEMM itself is the STOCK sm120 block-scaled CollectiveBuilder mainloop —
// kind::mxf8f6f4 with hardware SF application every 32 elements — the one
// collective this architecture family provides natively (the same builder the
// NVFP4-CB fused lane rebinds; here there is no CB decode, so no fork header
// is needed and the builder product is used unmodified).
//
// Scale-factor plane layout: CUTLASS Sm1xxBlockScaledConfig<32>.  Its
// tile_atom_to_shape_SF{A,B} layouts have an ELEMENT-indexed domain
// (m, k_element, l) whose inner SFVec mode has stride 0 — all 32 elements of a
// chunk map to the chunk's one SF byte.  ``mxfp8_sf_offsets`` walks that exact
// layout object on the host, so the Python-side scatter that fills the plane
// is consistent with the mainloop's reading of it BY CONSTRUCTION — there is
// no second spelling of the swizzle to drift.
//
// Entry points:
//  - mxfp8_dense_mm[_out]: D = A x B^T in bf16 from quantized A and B plus
//    their swizzled SF planes.
//  - mxfp8_sf_offsets(rows, K, is_b): int64 CPU tensor, one plane offset per
//    (row, k_group) in row-major order.
//  - mxfp8_sf_plane_numel(rows, K): allocation size (bytes / uint8 elements)
//    of the padded swizzled plane.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <climits>
#include <optional>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace {

using namespace cute;

// e4m3 element + ue8m0 scale per 32, the OCP MXFP8 pairing.  The pair type is
// what selects kind::mxf8f6f4 with SFVecSize 32 in the block-scaled builder.
using ElementPairAB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementSF = cutlass::float_ue8m0_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
// e4m3 is one byte per element: 16 elements = the 16-byte TMA box edge.
constexpr int AlignAB = 16;
constexpr int AlignD = 8;
using ClusterShape = Shape<_1, _1, _1>;
// TileN is pinned at 128 by the block-scaled SF smem atom (Blk_MN = 128),
// exactly as in the NVFP4-CB lane; TileM/TileK 128 keep one proven shape.
using TileShapeMx = Shape<_128, _128, _128>;

constexpr int kSFVec = 32;
using Sm1xxCfg = cutlass::detail::Sm1xxBlockScaledConfig<kSFVec>;

struct CfgMx {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShapeMx, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc,
      void, LayoutD, AlignD,
      ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPairAB, LayoutA, AlignAB,
      ElementPairAB, LayoutB, AlignAB,
      ElementAcc,
      TileShapeMx, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
};

using GemmKernelMx = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, typename CfgMx::Mainloop,
    typename CfgMx::Epilogue>;
using GemmMx = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelMx>;

int64_t pad_up(int64_t v, int64_t m) { return ((v + m - 1) / m) * m; }

// Padded plane extent implied by the SfAtom tiling: rows to a multiple of the
// 128-row MN atom, K to a multiple of the (SFVec * 4)-element K atom, one SF
// byte per SFVec elements.
int64_t plane_numel(int64_t rows, int64_t k) {
  return pad_up(rows, 128) * (pad_up(k, int64_t{kSFVec} * 4) / kSFVec);
}

void check_quantized_operand(torch::Tensor const& t, char const* name,
                             int64_t rows, int64_t k) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == torch::kUInt8 ||
                  t.scalar_type() == torch::kFloat8_e4m3fn,
              name, " must be uint8 or float8_e4m3fn storage of e4m3 bytes");
  TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == k &&
                  t.stride(1) == 1 && t.stride(0) == k,
              name, " must be contiguous [", rows, ", ", k, "], got shape [",
              t.size(0), ", ", t.size(1), "] strides [", t.stride(0), ", ",
              t.stride(1), "]");
}

void check_sf_plane(torch::Tensor const& t, char const* name, int64_t rows,
                    int64_t k) {
  const int64_t need = plane_numel(rows, k);
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                  (t.scalar_type() == torch::kUInt8 ||
                   t.scalar_type() == torch::kFloat8_e8m0fnu),
              name, " must be a contiguous CUDA uint8/float8_e8m0fnu plane");
  TORCH_CHECK(t.numel() == need, name, " numel ", t.numel(),
              " does not match the padded swizzled plane (", need,
              " for ", rows, " rows, K=", k, ")");
}

void run_mxfp8_dense(torch::Tensor a, torch::Tensor sfa, torch::Tensor b,
                     torch::Tensor sfb, torch::Tensor d) {
  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(0);

  TORCH_CHECK(M > 0 && N > 0 && K > 0, "empty problem");
  TORCH_CHECK(K % kSFVec == 0,
              "K must be a multiple of ", kSFVec,
              " (one UE8M0 scale per ", kSFVec, " elements), got ", K);
  TORCH_CHECK(K % AlignAB == 0, "K must be a multiple of ", AlignAB,
              " e4m3 elements (16-byte TMA alignment), got ", K);
  TORCH_CHECK(N % AlignD == 0,
              "N must be a multiple of ", AlignD,
              " (bf16 TMA epilogue alignment), got ", N);
  check_quantized_operand(a, "a", M, K);
  check_quantized_operand(b, "b", N, K);
  check_sf_plane(sfa, "sfa", M, K);
  check_sf_plane(sfb, "sfb", N, K);
  TORCH_CHECK(d.is_cuda() && d.scalar_type() == torch::kBFloat16 &&
                  d.dim() == 2 && d.size(0) == M && d.size(1) == N &&
                  d.stride(1) == 1 && d.stride(0) == N,
              "out must be contiguous bf16 [", M, ", ", N, "]");
  TORCH_CHECK(sfa.device() == a.device(),
              "sfa must be on the same CUDA device as a");
  TORCH_CHECK(b.device() == a.device(),
              "b must be on the same CUDA device as a");
  TORCH_CHECK(sfb.device() == a.device(),
              "sfb must be on the same CUDA device as a");
  TORCH_CHECK(d.device() == a.device(),
              "out must be on the same CUDA device as a");

  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  using StrideA = typename GemmKernelMx::StrideA;
  using StrideB = typename GemmKernelMx::StrideB;
  using StrideD = typename GemmKernelMx::StrideD;
  StrideA sa = cutlass::make_cute_packed_stride(
      StrideA{}, {(int)M, (int)K, 1});
  StrideB sb = cutlass::make_cute_packed_stride(
      StrideB{}, {(int)N, (int)K, 1});
  StrideD sd = cutlass::make_cute_packed_stride(
      StrideD{}, {(int)M, (int)N, 1});
  auto layout_sfa = Sm1xxCfg::tile_atom_to_shape_SFA(
      cute::make_shape((int)M, (int)N, (int)K, 1));
  auto layout_sfb = Sm1xxCfg::tile_atom_to_shape_SFB(
      cute::make_shape((int)M, (int)N, (int)K, 1));

  typename GemmMx::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {(int)M, (int)N, (int)K, 1},
      {reinterpret_cast<const cutlass::float_e4m3_t*>(a.data_ptr()), sa,
       reinterpret_cast<const cutlass::float_e4m3_t*>(b.data_ptr()), sb,
       reinterpret_cast<const ElementSF*>(sfa.data_ptr()), layout_sfa,
       reinterpret_cast<const ElementSF*>(sfb.data_ptr()), layout_sfb},
      {{1.0f, 0.0f},
       nullptr, typename GemmKernelMx::StrideC{},
       reinterpret_cast<ElementD*>(d.data_ptr()), sd}};

  GemmMx gemm;
  size_t ws = GemmMx::get_workspace_size(args);
  auto workspace =
      torch::empty({(int64_t)ws}, a.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
              "mxfp8 dense can_implement failed for M=", M, " N=", N,
              " K=", K);
  auto init_status = gemm.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "mxfp8 dense initialize failed: ",
              cutlass::cutlassGetStatusString(init_status));
  auto run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "mxfp8 dense launch failed: ",
              cutlass::cutlassGetStatusString(run_status));
}

torch::Tensor mxfp8_dense_mm(torch::Tensor a, torch::Tensor sfa,
                             torch::Tensor b, torch::Tensor sfb) {
  auto d = torch::empty({a.size(0), b.size(0)},
                        a.options().dtype(torch::kBFloat16));
  run_mxfp8_dense(a, sfa, b, sfb, d);
  return d;
}

torch::Tensor mxfp8_dense_mm_out(torch::Tensor a, torch::Tensor sfa,
                                 torch::Tensor b, torch::Tensor sfb,
                                 torch::Tensor out) {
  run_mxfp8_dense(a, sfa, b, sfb, out);
  return out;
}

// One plane offset per (row, k_group), row-major, computed by indexing the
// SAME CuTe layout the mainloop consumes.  The layout's domain is
// (m, k_element, l) with the inner SFVec mode at stride 0, so any element of
// the group addresses the group's byte; k_group * SFVec is used.  Host-side
// int64 loop: the largest real plane (7168 x 7168/32) is ~1.6M entries.
torch::Tensor mxfp8_sf_offsets(int64_t rows, int64_t k, bool is_b) {
  TORCH_CHECK(rows > 0 && k > 0 && k % kSFVec == 0,
              "rows must be positive and K a positive multiple of ", kSFVec);
  const int64_t groups = k / kSFVec;
  const int64_t plane = plane_numel(rows, k);
  auto out = torch::empty({rows * groups},
                          torch::TensorOptions().dtype(torch::kInt64));
  auto* p = out.data_ptr<int64_t>();
  // M and N enter tile_atom_to_shape_SF{A,B} independently; pass ``rows`` in
  // the slot the requested side reads and 1 in the other.
  if (is_b) {
    auto layout = Sm1xxCfg::tile_atom_to_shape_SFB(
        cute::make_shape(1, (int)rows, (int)k, 1));
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t g = 0; g < groups; ++g) {
        const auto off = layout(
            cute::make_coord((int)r, (int)(g * kSFVec), 0));
        p[r * groups + g] = static_cast<int64_t>(off);
      }
    }
  } else {
    auto layout = Sm1xxCfg::tile_atom_to_shape_SFA(
        cute::make_shape((int)rows, 1, (int)k, 1));
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t g = 0; g < groups; ++g) {
        const auto off = layout(
            cute::make_coord((int)r, (int)(g * kSFVec), 0));
        p[r * groups + g] = static_cast<int64_t>(off);
      }
    }
  }
  // The layout maps inside the padded plane by construction; make the
  // contract explicit so a CUTLASS-side layout change cannot silently write
  // past the allocation the Python side sized with mxfp8_sf_plane_numel.
  TORCH_CHECK(out.max().item<int64_t>() < plane,
              "internal: SF offset beyond the padded plane");
  return out;
}

int64_t mxfp8_sf_plane_numel(int64_t rows, int64_t k) {
  TORCH_CHECK(rows > 0 && k > 0 && k % kSFVec == 0,
              "rows must be positive and K a positive multiple of ", kSFVec);
  return plane_numel(rows, k);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mxfp8_dense_mm", &mxfp8_dense_mm,
        "MXFP8 (e4m3 + ue8m0/32) dense W8A8 block-scaled GEMM -> bf16");
  m.def("mxfp8_dense_mm_out", &mxfp8_dense_mm_out,
        "mxfp8_dense_mm writing into a preallocated bf16 tensor");
  m.def("mxfp8_sf_offsets", &mxfp8_sf_offsets,
        "swizzled-plane offset per (row, k_group), from the mainloop's own "
        "CuTe layout");
  m.def("mxfp8_sf_plane_numel", &mxfp8_sf_plane_numel,
        "allocation numel of the padded swizzled SF plane");
}
