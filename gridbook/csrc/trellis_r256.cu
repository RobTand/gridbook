// Research-only scan-free decoder for gridbook.trellis.wire.v1.
//
// This translation unit has a dedicated symbol/module namespace.  It does not
// implement, alias, or consume any CB ABI.  The payload remains packed in HBM:
// decode_codes/expand are explicit test operations, while dequant_gemv decodes
// each weight in-register and never constructs a resident expanded matrix.
// The prevalidated native-code path emits nibble-packed E2M1 or byte-packed
// E4M3 for a future tensor-core mainloop; it never emits a float weight plane.
// No prefill GEMM is claimed here (INV-2 remains a release gate).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kFamilyE2M1 = 1;
constexpr int kFamilyE4M3 = 2;
constexpr int kMemory = 8;
constexpr int kSuperblock = 256;
constexpr unsigned kG0 = 0561u;
constexpr unsigned kG1 = 0753u;

enum DecodeError : int {
  kDecodeOk = 0,
  kDecodeInvalidSchedule = 1,
  kDecodeInvalidColumnOffset = 2,
  kDecodeInvalidRowStride = 3,
  kDecodeInvalidPreviousOffset = 4,
  kDecodeInvalidAlphabetCode = 5,
  kDecodeInvalidDecodedCode = 6,
  kDecodeInvalidScale = 7,
  kDecodeNoncanonicalPadding = 8,
};

#define TCQ_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define TCQ_CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

__device__ __forceinline__ unsigned read_bits(
    uint8_t const* row, int64_t bit_offset, int width) {
  unsigned value = 0;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    if (bit < width) {
      int64_t const absolute = bit_offset + bit;
      value |= unsigned((row[absolute >> 3] >> (absolute & 7)) & 1u) << bit;
    }
  }
  return value;
}

__device__ __forceinline__ void record_decode_error(
    int* error, DecodeError code) {
  atomicCAS(error, kDecodeOk, int(code));
}

__device__ __forceinline__ bool valid_native_code(
    uint8_t code, int family) {
  if (family == kFamilyE2M1) return code <= 0x0fu;
  return code != 0x7fu && code != 0xffu;
}

// The public Python helper derives these tensors, but the pybind symbols are a
// raw ABI too.  Validate the complete derived plan on device before any decode
// thread is allowed to use a schedule-controlled shift or bit address.  This
// one-thread pass is intentionally conservative: the research prototype pays
// a small launch cost in exchange for making malformed workspace fail closed.
__global__ void validate_decode_plan_kernel(
    uint8_t const* schedule,
    int64_t const* column_offsets,
    int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut,
    int columns,
    int row_stride,
    int family,
    int terminal_rate,
    int* error,
    int64_t* row_body_bits) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int64_t expected_offset = 0;
  int64_t const stride_bits = int64_t(row_stride) * 8;
  for (int column = 0; column < columns; ++column) {
    int const rate = int(schedule[column]);
    if (rate < 1 || rate > terminal_rate) {
      *error = kDecodeInvalidSchedule;
      return;
    }
    if (column_offsets[column] != expected_offset) {
      *error = kDecodeInvalidColumnOffset;
      return;
    }
    if (expected_offset > stride_bits - rate) {
      *error = kDecodeInvalidColumnOffset;
      return;
    }
    expected_offset += rate;
  }

  int64_t const body_bytes = (expected_offset + 7) / 8;
  int64_t const canonical_stride =
      ((body_bytes + 15) / 16) * 16;
  if (canonical_stride != row_stride) {
    *error = kDecodeInvalidRowStride;
    return;
  }
  *row_body_bits = expected_offset;

  // Validate every address in the scan-free tail-biting history plan, not
  // merely its outer bound.  This rejects an in-bounds pointer into a point
  // field as well as an OOB pointer.  Each superblock is independent.
  for (int block_start = 0; block_start < columns;
       block_start += kSuperblock) {
    int const block_stop = min(block_start + kSuperblock, columns);
    int64_t coded_offsets[kSuperblock];
    int coded_count = 0;
    for (int column = block_start; column < block_stop; ++column) {
      if (int(schedule[column]) < terminal_rate) {
        coded_offsets[coded_count++] = column_offsets[column];
      }
    }
    if (coded_count < kMemory) {
      *error = kDecodeInvalidPreviousOffset;
      return;
    }
    int coded_ordinal = 0;
    for (int column = block_start; column < block_stop; ++column) {
      int64_t const* previous =
          previous_u_offsets + int64_t(column) * kMemory;
      if (int(schedule[column]) == terminal_rate) {
        for (int i = 0; i < kMemory; ++i) {
          if (previous[i] != -1) {
            *error = kDecodeInvalidPreviousOffset;
            return;
          }
        }
        continue;
      }
      for (int i = 0; i < kMemory; ++i) {
        int const expected_ordinal =
            (coded_ordinal - 1 - i + coded_count) % coded_count;
        int64_t const prior = previous[i];
        if (prior < 0 || prior >= expected_offset ||
            prior != coded_offsets[expected_ordinal]) {
          *error = kDecodeInvalidPreviousOffset;
          return;
        }
      }
      ++coded_ordinal;
    }
  }

  // The dense LUT has terminal_rate rows of 256 bytes.  Unused slots are
  // canonical zeroes, so validating the whole supplied ABI footprint is both
  // cheap and deterministic.
  for (int index = 0; index < terminal_rate * 256; ++index) {
    if (!valid_native_code(alphabet_lut[index], family)) {
      *error = kDecodeInvalidAlphabetCode;
      return;
    }
  }
}

__global__ void validate_payload_padding_kernel(
    uint8_t const* payload,
    int rows,
    int row_stride,
    int64_t const* row_body_bits,
    int* error) {
  int const row = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || *error != kDecodeOk) return;
  int64_t const body_bits = *row_body_bits;
  int64_t const used_bytes = (body_bits + 7) / 8;
  uint8_t const* packed_row = payload + int64_t(row) * row_stride;
  if (body_bits & 7) {
    uint8_t const mask = uint8_t((1u << (body_bits & 7)) - 1u);
    if (packed_row[used_bytes - 1] & uint8_t(~mask)) {
      record_decode_error(error, kDecodeNoncanonicalPadding);
      return;
    }
  }
  for (int64_t byte = used_bytes; byte < row_stride; ++byte) {
    if (packed_row[byte] != 0) {
      record_decode_error(error, kDecodeNoncanonicalPadding);
      return;
    }
  }
}

__global__ void validate_scales_kernel(
    float const* scales, int64_t count, int* error) {
  int64_t const index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count || *error != kDecodeOk) return;
  float const scale = scales[index];
  if (!isfinite(scale) || scale <= 0.0f) {
    record_decode_error(error, kDecodeInvalidScale);
  }
}

__device__ __forceinline__ uint8_t decode_code(
    uint8_t const* row,
    uint8_t const* schedule,
    int64_t const* column_offsets,
    int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut,
    int column,
    int family,
    int terminal_rate,
    int64_t row_body_bits,
    int* error) {
  if (*error != kDecodeOk) return 0;
  int const rate = int(schedule[column]);
  if (rate < 1 || rate > terminal_rate) {
    record_decode_error(error, kDecodeInvalidSchedule);
    return 0;
  }
  int64_t const offset = column_offsets[column];
  if (offset < 0 || offset > row_body_bits - rate) {
    record_decode_error(error, kDecodeInvalidColumnOffset);
    return 0;
  }
  if (rate == terminal_rate) {
    uint8_t const code = uint8_t(read_bits(row, offset, terminal_rate));
    if (!valid_native_code(code, family)) {
      record_decode_error(error, kDecodeInvalidDecodedCode);
      return 0;
    }
    return code;
  }
  unsigned const u = read_bits(row, offset, 1);
  unsigned state = 0;
#pragma unroll
  for (int i = 0; i < kMemory; ++i) {
    int64_t const prior = previous_u_offsets[int64_t(column) * kMemory + i];
    if (prior < 0 || prior >= row_body_bits) {
      record_decode_error(error, kDecodeInvalidPreviousOffset);
      return 0;
    }
    state |= read_bits(row, prior, 1) << (kMemory - 1 - i);
  }
  unsigned const reg = (u << kMemory) | state;
  unsigned const subset =
      2u * (__popc(reg & kG0) & 1u) + (__popc(reg & kG1) & 1u);
  unsigned const point = read_bits(row, offset + 1, rate - 1);
  unsigned const points = 1u << (rate - 1);
  uint8_t const code =
      alphabet_lut[rate * 256 + subset * points + point];
  if (!valid_native_code(code, family)) {
    record_decode_error(error, kDecodeInvalidDecodedCode);
    return 0;
  }
  return code;
}

// Fast path used only after validate_wire_cuda has accepted immutable wire
// tensors.  Bounds guards remain here so mutation cannot cause an out-of-range
// load; malformed post-validation data decodes as zero.  It is intentionally
// not an integrity boundary: callers that need a diagnostic exception must
// use the validating operation before admitting the tensors.
__device__ __forceinline__ uint8_t decode_code_prevalidated(
    uint8_t const* row,
    uint8_t const* schedule,
    int64_t const* column_offsets,
    int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut,
    int column,
    int family,
    int terminal_rate,
    int64_t row_body_bits) {
  int const rate = int(schedule[column]);
  if (rate < 1 || rate > terminal_rate) return 0;
  int64_t const offset = column_offsets[column];
  if (offset < 0 || offset > row_body_bits - rate) return 0;
  if (rate == terminal_rate) {
    uint8_t const code = uint8_t(read_bits(row, offset, terminal_rate));
    return valid_native_code(code, family) ? code : 0;
  }
  unsigned const u = read_bits(row, offset, 1);
  unsigned state = 0;
#pragma unroll
  for (int i = 0; i < kMemory; ++i) {
    int64_t const prior = previous_u_offsets[int64_t(column) * kMemory + i];
    if (prior < 0 || prior >= row_body_bits) return 0;
    state |= read_bits(row, prior, 1) << (kMemory - 1 - i);
  }
  unsigned const reg = (u << kMemory) | state;
  unsigned const subset =
      2u * (__popc(reg & kG0) & 1u) + (__popc(reg & kG1) & 1u);
  unsigned const point = read_bits(row, offset + 1, rate - 1);
  unsigned const points = 1u << (rate - 1);
  uint8_t const code =
      alphabet_lut[rate * 256 + subset * points + point];
  return valid_native_code(code, family) ? code : 0;
}

__device__ __forceinline__ float e2m1_value(uint8_t code) {
  constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                  2.0f, 3.0f, 4.0f, 6.0f};
  float const value = magnitude[code & 7u];
  return (code & 8u) ? -value : value;
}

__device__ __forceinline__ float e4m3fn_value(uint8_t code) {
  int const sign = (code & 0x80u) ? -1 : 1;
  int const exponent = (code >> 3) & 0xf;
  int const mantissa = code & 7;
  float value;
  if (exponent == 0) {
    value = ldexpf(float(mantissa) / 8.0f, 1 - 7);
  } else {
    value = ldexpf(1.0f + float(mantissa) / 8.0f, exponent - 7);
  }
  return float(sign) * value;
}

__device__ __forceinline__ float decoded_value(
    uint8_t code, float const* scales, int family, int row, int column,
    int scale_stride) {
  if (family == kFamilyE2M1) {
    return e2m1_value(code) * scales[int64_t(row) * scale_stride + column / 16];
  }
  return e4m3fn_value(code) * scales[row];
}

__global__ void decode_codes_kernel(
    uint8_t const* payload, uint8_t const* schedule,
    int64_t const* column_offsets, int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut, uint8_t* output, int rows, int columns,
    int row_stride, int family, int terminal_rate,
    int64_t const* row_body_bits, int* error) {
  int64_t const linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const count = int64_t(rows) * columns;
  if (linear >= count) return;
  if (*error != kDecodeOk) return;
  int const row = int(linear / columns);
  int const column = int(linear - int64_t(row) * columns);
  output[linear] = decode_code(
      payload + int64_t(row) * row_stride, schedule, column_offsets,
      previous_u_offsets, alphabet_lut, column, family, terminal_rate,
      *row_body_bits, error);
}

__global__ void decode_native_packed_prevalidated_kernel(
    uint8_t const* payload, uint8_t const* schedule,
    int64_t const* column_offsets, int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut, uint8_t* output, int rows, int columns,
    int output_columns, int row_stride, int family, int terminal_rate,
    int64_t row_body_bits) {
  int64_t const linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const count = int64_t(rows) * output_columns;
  if (linear >= count) return;
  int const row = int(linear / output_columns);
  int const output_column = int(linear - int64_t(row) * output_columns);
  uint8_t const* packed_row = payload + int64_t(row) * row_stride;
  if (family == kFamilyE2M1) {
    int const first = output_column * 2;
    uint8_t const low = decode_code_prevalidated(
        packed_row, schedule, column_offsets, previous_u_offsets,
        alphabet_lut, first, family, terminal_rate, row_body_bits);
    uint8_t high = 0;
    if (first + 1 < columns) {
      high = decode_code_prevalidated(
          packed_row, schedule, column_offsets, previous_u_offsets,
          alphabet_lut, first + 1, family, terminal_rate, row_body_bits);
    }
    output[linear] = uint8_t((low & 0x0fu) | ((high & 0x0fu) << 4));
    return;
  }
  output[linear] = decode_code_prevalidated(
      packed_row, schedule, column_offsets, previous_u_offsets,
      alphabet_lut, output_column, family, terminal_rate, row_body_bits);
}

__global__ void expand_kernel(
    uint8_t const* payload, uint8_t const* schedule,
    int64_t const* column_offsets, int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut, float const* scales, float* output,
    int rows, int columns, int row_stride, int family, int scale_stride,
    int64_t const* row_body_bits, int* error) {
  int64_t const linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const count = int64_t(rows) * columns;
  if (linear >= count) return;
  if (*error != kDecodeOk) return;
  int const row = int(linear / columns);
  int const column = int(linear - int64_t(row) * columns);
  uint8_t const code = decode_code(
      payload + int64_t(row) * row_stride, schedule, column_offsets,
      previous_u_offsets, alphabet_lut, column,
      family, family == kFamilyE2M1 ? 4 : 8, *row_body_bits, error);
  if (*error != kDecodeOk) return;
  output[linear] = decoded_value(
      code, scales, family, row, column, scale_stride);
}

__global__ void dequant_gemv_kernel(
    float const* x, uint8_t const* payload, uint8_t const* schedule,
    int64_t const* column_offsets, int64_t const* previous_u_offsets,
    uint8_t const* alphabet_lut, float const* scales, float* output,
    int batches, int rows, int columns, int row_stride, int family,
    int scale_stride, int64_t const* row_body_bits, int* error) {
  int64_t const linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const count = int64_t(batches) * rows;
  if (linear >= count) return;
  if (*error != kDecodeOk) return;
  int const batch = int(linear / rows);
  int const row = int(linear - int64_t(batch) * rows);
  uint8_t const* packed_row = payload + int64_t(row) * row_stride;
  float total = 0.0f;
  for (int column = 0; column < columns; ++column) {
    uint8_t const code = decode_code(
        packed_row, schedule, column_offsets, previous_u_offsets,
        alphabet_lut, column, family, family == kFamilyE2M1 ? 4 : 8,
        *row_body_bits, error);
    if (*error != kDecodeOk) return;
    total = fmaf(
        x[int64_t(batch) * columns + column],
        decoded_value(code, scales, family, row, column, scale_stride), total);
  }
  output[linear] = total;
}

int64_t checked_product(int64_t left, int64_t right, char const* name) {
  TORCH_CHECK(left >= 0 && right >= 0,
              name, " dimensions must be nonnegative");
  TORCH_CHECK(left == 0 ||
                  right <= std::numeric_limits<int64_t>::max() / left,
              name, " element count overflows int64");
  return left * right;
}

int checked_grid_blocks(
    int64_t count, int threads, int device, char const* operation) {
  TORCH_CHECK(count > 0, operation, " launch count must be positive");
  TORCH_CHECK(count <= std::numeric_limits<int64_t>::max() - (threads - 1),
              operation, " launch rounding overflows int64");
  int64_t const blocks = (count + threads - 1) / threads;
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  TORCH_CHECK(blocks <= properties.maxGridSize[0],
              operation, " launch requires ", blocks,
              " x-grid blocks, device limit is ", properties.maxGridSize[0]);
  TORCH_CHECK(blocks <= INT_MAX,
              operation, " launch grid exceeds int32");
  return int(blocks);
}

void launch_plan_validation(
    torch::Tensor const& payload,
    torch::Tensor const& schedule,
    torch::Tensor const& column_offsets,
    torch::Tensor const& previous_u_offsets,
    torch::Tensor const& alphabet_lut,
    int columns,
    int row_stride,
    int family,
    torch::Tensor const& error,
    torch::Tensor const& row_body_bits,
    at::cuda::CUDAStream stream) {
  int const terminal_rate = family == kFamilyE2M1 ? 4 : 8;
  validate_decode_plan_kernel<<<1, 1, 0, stream>>>(
      schedule.data_ptr<uint8_t>(), column_offsets.data_ptr<int64_t>(),
      previous_u_offsets.data_ptr<int64_t>(),
      alphabet_lut.data_ptr<uint8_t>(), columns, row_stride, family,
      terminal_rate, error.data_ptr<int>(), row_body_bits.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int const threads = 256;
  int const blocks = checked_grid_blocks(
      payload.size(0), threads, payload.device().index(),
      "payload padding validation");
  validate_payload_padding_kernel<<<blocks, threads, 0, stream>>>(
      payload.data_ptr<uint8_t>(), int(payload.size(0)),
      int(payload.size(1)), row_body_bits.data_ptr<int64_t>(),
      error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_scale_validation(
    torch::Tensor const& scales,
    torch::Tensor const& error,
    at::cuda::CUDAStream stream) {
  int const threads = 256;
  int64_t const count = scales.numel();
  int const blocks = checked_grid_blocks(
      count, threads, scales.device().index(), "scale validation");
  validate_scales_kernel<<<blocks, threads, 0, stream>>>(
      scales.data_ptr<float>(), count, error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

char const* decode_error_message(int error) {
  switch (error) {
    case kDecodeInvalidSchedule:
      return "schedule rate is outside [1,native_bits]";
    case kDecodeInvalidColumnOffset:
      return "column offset is noncanonical or outside the row body";
    case kDecodeInvalidRowStride:
      return "row stride is not the canonical aligned body stride";
    case kDecodeInvalidPreviousOffset:
      return "previous-u offset plan is noncanonical or outside the row body";
    case kDecodeInvalidAlphabetCode:
      return "alphabet LUT contains an invalid native code";
    case kDecodeInvalidDecodedCode:
      return "payload decoded to an invalid native code";
    case kDecodeInvalidScale:
      return "decoded scales must be finite and positive";
    case kDecodeNoncanonicalPadding:
      return "payload body/row padding must be canonical zero";
    default:
      return "unknown device validation error";
  }
}

void check_device_error(
    torch::Tensor const& error, at::cuda::CUDAStream stream) {
  int host_error = kDecodeOk;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      &host_error, error.data_ptr<int>(), sizeof(host_error),
      cudaMemcpyDeviceToHost, stream));
  // A synchronous boundary is deliberate for these raw research ops: without
  // it malformed input could be reported only after a corrupted result escaped.
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  TORCH_CHECK(host_error == kDecodeOk,
              "TCQ R256 raw CUDA ABI rejected malformed input (error ",
              host_error, ": ", decode_error_message(host_error), ")");
}

void validate_common(
    torch::Tensor const& payload, torch::Tensor const& schedule,
    torch::Tensor const& column_offsets,
    torch::Tensor const& previous_u_offsets,
    torch::Tensor const& alphabet_lut,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  TCQ_CHECK_CUDA(payload);
  TCQ_CHECK_CUDA(schedule);
  TCQ_CHECK_CUDA(column_offsets);
  TCQ_CHECK_CUDA(previous_u_offsets);
  TCQ_CHECK_CUDA(alphabet_lut);
  TCQ_CHECK_CONTIGUOUS(payload);
  TCQ_CHECK_CONTIGUOUS(schedule);
  TCQ_CHECK_CONTIGUOUS(column_offsets);
  TCQ_CHECK_CONTIGUOUS(previous_u_offsets);
  TCQ_CHECK_CONTIGUOUS(alphabet_lut);
  TORCH_CHECK(payload.scalar_type() == torch::kUInt8,
              "payload must be uint8");
  TORCH_CHECK(schedule.scalar_type() == torch::kUInt8,
              "schedule must be uint8");
  TORCH_CHECK(column_offsets.scalar_type() == torch::kInt64,
              "column_offsets must be int64");
  TORCH_CHECK(previous_u_offsets.scalar_type() == torch::kInt64,
              "previous_u_offsets must be int64");
  TORCH_CHECK(alphabet_lut.scalar_type() == torch::kUInt8,
              "alphabet_lut must be uint8");
  TORCH_CHECK(rows > 0 && columns > 0 && row_stride > 0,
              "rows, columns, and row_stride must be positive");
  TORCH_CHECK(rows <= INT_MAX, "rows must fit int32");
  TORCH_CHECK(columns <= INT_MAX, "columns must fit int32");
  TORCH_CHECK(row_stride <= INT_MAX, "row_stride must fit int32");
  TORCH_CHECK(columns <= std::numeric_limits<int64_t>::max() / kMemory,
              "columns*8 overflows int64");
  TORCH_CHECK(row_stride % 16 == 0,
              "row_stride must preserve the wire's 16-byte alignment");
  TORCH_CHECK(payload.dim() == 2 && payload.size(0) == rows &&
                  payload.size(1) == row_stride,
              "payload must have shape [rows,row_stride]");
  TORCH_CHECK(schedule.dim() == 1 && schedule.size(0) == columns,
              "expanded schedule must have shape [columns]");
  TORCH_CHECK(column_offsets.dim() == 1 &&
                  column_offsets.size(0) == columns,
              "column_offsets must have shape [columns]");
  TORCH_CHECK(previous_u_offsets.dim() == 2 &&
                  previous_u_offsets.size(0) == columns &&
                  previous_u_offsets.size(1) == kMemory,
              "previous_u_offsets must have shape [columns,8]");
  TORCH_CHECK(family == kFamilyE2M1 || family == kFamilyE4M3,
              "family must be 1 (E2M1) or 2 (E4M3)");
  int const trellis_rates = family == kFamilyE2M1 ? 4 : 8;
  TORCH_CHECK(alphabet_lut.dim() == 2 &&
                  alphabet_lut.size(0) == trellis_rates &&
                  alphabet_lut.size(1) == 256,
              "alphabet_lut must have exact shape [native_bits,256]");
  TORCH_CHECK(payload.device() == schedule.device() &&
                  payload.device() == column_offsets.device() &&
                  payload.device() == previous_u_offsets.device() &&
                  payload.device() == alphabet_lut.device(),
              "all trellis tensors must be on one CUDA device");
}

torch::Tensor decode_codes_cuda(
    torch::Tensor payload, torch::Tensor schedule,
    torch::Tensor column_offsets, torch::Tensor previous_u_offsets,
    torch::Tensor alphabet_lut, int64_t rows, int64_t columns,
    int64_t row_stride, int64_t family) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  c10::cuda::CUDAGuard guard(payload.device());
  int64_t const count = checked_product(rows, columns, "decode_codes");
  int const threads = 256;
  int const blocks = checked_grid_blocks(
      count, threads, payload.device().index(), "decode_codes");
  auto output = torch::empty({rows, columns}, payload.options());
  auto error = torch::zeros(
      {1}, payload.options().dtype(torch::kInt32));
  auto row_body_bits = torch::zeros(
      {1}, payload.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
  launch_plan_validation(
      payload, schedule, column_offsets, previous_u_offsets, alphabet_lut,
      int(columns), int(row_stride), int(family), error, row_body_bits, stream);
  decode_codes_kernel<<<blocks, threads, 0, stream>>>(
      payload.data_ptr<uint8_t>(), schedule.data_ptr<uint8_t>(),
      column_offsets.data_ptr<int64_t>(),
      previous_u_offsets.data_ptr<int64_t>(),
      alphabet_lut.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
      int(rows), int(columns), int(row_stride), int(family),
      family == kFamilyE2M1 ? 4 : 8, row_body_bits.data_ptr<int64_t>(),
      error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  check_device_error(error, stream);
  return output;
}

torch::Tensor validate_wire_cuda(
    torch::Tensor payload, torch::Tensor schedule,
    torch::Tensor column_offsets, torch::Tensor previous_u_offsets,
    torch::Tensor alphabet_lut, torch::Tensor scales,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  TCQ_CHECK_CUDA(scales);
  TCQ_CHECK_CONTIGUOUS(scales);
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "decoded scales must be float32");
  TORCH_CHECK(scales.device() == payload.device(),
              "scales must be on the payload device");
  int64_t const scale_stride =
      family == kFamilyE2M1 ? (columns + 15) / 16 : 1;
  checked_product(rows, scale_stride, "wire validation scales");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == scale_stride,
              "decoded scales must have exact shape [rows,scale_stride]");
  c10::cuda::CUDAGuard guard(payload.device());
  auto error = torch::zeros(
      {1}, payload.options().dtype(torch::kInt32));
  auto row_body_bits = torch::zeros(
      {1}, payload.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
  launch_plan_validation(
      payload, schedule, column_offsets, previous_u_offsets, alphabet_lut,
      int(columns), int(row_stride), int(family), error, row_body_bits, stream);
  launch_scale_validation(scales, error, stream);
  check_device_error(error, stream);
  return row_body_bits;
}

void decode_native_packed_prevalidated_out_cuda(
    torch::Tensor payload, torch::Tensor schedule,
    torch::Tensor column_offsets, torch::Tensor previous_u_offsets,
    torch::Tensor alphabet_lut, torch::Tensor output,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family,
    int64_t row_body_bits) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  TCQ_CHECK_CUDA(output);
  TCQ_CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == torch::kUInt8,
              "native packed output must be uint8");
  TORCH_CHECK(output.device() == payload.device(),
              "native packed output must be on the payload device");
  TORCH_CHECK(row_body_bits > 0 && row_body_bits <= row_stride * 8,
              "row_body_bits must be inside the packed row stride");
  int64_t const output_columns =
      family == kFamilyE2M1 ? (columns + 1) / 2 : columns;
  TORCH_CHECK(output.dim() == 2 && output.size(0) == rows &&
                  output.size(1) == output_columns,
              "native packed output has the wrong shape");
  c10::cuda::CUDAGuard guard(payload.device());
  int64_t const count = checked_product(
      rows, output_columns, "prevalidated native packed decode");
  int const threads = 256;
  int const blocks = checked_grid_blocks(
      count, threads, payload.device().index(),
      "prevalidated native packed decode");
  auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
  decode_native_packed_prevalidated_kernel<<<blocks, threads, 0, stream>>>(
      payload.data_ptr<uint8_t>(), schedule.data_ptr<uint8_t>(),
      column_offsets.data_ptr<int64_t>(),
      previous_u_offsets.data_ptr<int64_t>(),
      alphabet_lut.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
      int(rows), int(columns), int(output_columns), int(row_stride),
      int(family), family == kFamilyE2M1 ? 4 : 8, row_body_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor expand_cuda(
    torch::Tensor payload, torch::Tensor schedule,
    torch::Tensor column_offsets, torch::Tensor previous_u_offsets,
    torch::Tensor alphabet_lut, torch::Tensor scales,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  TCQ_CHECK_CUDA(scales);
  TCQ_CHECK_CONTIGUOUS(scales);
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
              "decoded scales must be float32");
  TORCH_CHECK(scales.device() == payload.device(),
              "scales must be on the payload device");
  int64_t const scale_stride64 =
      family == kFamilyE2M1 ? (columns + 15) / 16 : 1;
  TORCH_CHECK(scale_stride64 <= INT_MAX, "scale_stride must fit int32");
  int const scale_stride = int(scale_stride64);
  checked_product(rows, scale_stride64, "expand scales");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == scale_stride64,
              "decoded scales must have exact shape [rows,scale_stride]");
  c10::cuda::CUDAGuard guard(payload.device());
  int64_t const count = checked_product(rows, columns, "expand");
  int const threads = 256;
  int const blocks = checked_grid_blocks(
      count, threads, payload.device().index(), "expand");
  auto output = torch::empty({rows, columns},
                             payload.options().dtype(torch::kFloat32));
  auto error = torch::zeros(
      {1}, payload.options().dtype(torch::kInt32));
  auto row_body_bits = torch::zeros(
      {1}, payload.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
  launch_plan_validation(
      payload, schedule, column_offsets, previous_u_offsets, alphabet_lut,
      int(columns), int(row_stride), int(family), error, row_body_bits, stream);
  launch_scale_validation(scales, error, stream);
  expand_kernel<<<blocks, threads, 0, stream>>>(
      payload.data_ptr<uint8_t>(), schedule.data_ptr<uint8_t>(),
      column_offsets.data_ptr<int64_t>(),
      previous_u_offsets.data_ptr<int64_t>(),
      alphabet_lut.data_ptr<uint8_t>(), scales.data_ptr<float>(),
      output.data_ptr<float>(), int(rows), int(columns), int(row_stride),
      int(family), scale_stride, row_body_bits.data_ptr<int64_t>(),
      error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  check_device_error(error, stream);
  return output;
}

torch::Tensor dequant_gemv_cuda(
    torch::Tensor x, torch::Tensor payload, torch::Tensor schedule,
    torch::Tensor column_offsets, torch::Tensor previous_u_offsets,
    torch::Tensor alphabet_lut, torch::Tensor scales,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  TCQ_CHECK_CUDA(x);
  TCQ_CHECK_CUDA(scales);
  TCQ_CHECK_CONTIGUOUS(x);
  TCQ_CHECK_CONTIGUOUS(scales);
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                  scales.scalar_type() == torch::kFloat32,
              "x and decoded scales must be float32");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == columns,
              "x must have shape [batch,columns]");
  TORCH_CHECK(x.device() == payload.device() && scales.device() == payload.device(),
              "x, scales, and wire tensors must share a CUDA device");
  int64_t const scale_stride64 =
      family == kFamilyE2M1 ? (columns + 15) / 16 : 1;
  TORCH_CHECK(scale_stride64 <= INT_MAX, "scale_stride must fit int32");
  int const scale_stride = int(scale_stride64);
  checked_product(rows, scale_stride64, "dequant_gemv scales");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == scale_stride64,
              "decoded scales must have exact shape [rows,scale_stride]");
  int64_t const batches = x.size(0);
  TORCH_CHECK(batches > 0, "dequant_gemv batch must be positive");
  TORCH_CHECK(batches <= INT_MAX, "dequant_gemv batch must fit int32");
  c10::cuda::CUDAGuard guard(payload.device());
  int64_t const count = checked_product(batches, rows, "dequant_gemv");
  int const threads = 128;
  int const blocks = checked_grid_blocks(
      count, threads, payload.device().index(), "dequant_gemv");
  auto output = torch::empty({batches, rows}, x.options());
  auto error = torch::zeros(
      {1}, payload.options().dtype(torch::kInt32));
  auto row_body_bits = torch::zeros(
      {1}, payload.options().dtype(torch::kInt64));
  auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
  launch_plan_validation(
      payload, schedule, column_offsets, previous_u_offsets, alphabet_lut,
      int(columns), int(row_stride), int(family), error, row_body_bits, stream);
  launch_scale_validation(scales, error, stream);
  dequant_gemv_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<float>(), payload.data_ptr<uint8_t>(),
      schedule.data_ptr<uint8_t>(), column_offsets.data_ptr<int64_t>(),
      previous_u_offsets.data_ptr<int64_t>(), alphabet_lut.data_ptr<uint8_t>(),
      scales.data_ptr<float>(), output.data_ptr<float>(), int(batches),
      int(rows), int(columns), int(row_stride), int(family), scale_stride,
      row_body_bits.data_ptr<int64_t>(), error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  check_device_error(error, stream);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trellis_r256_validate_wire", &validate_wire_cuda,
        "TCQ R256 one-time device wire validation");
  m.def("trellis_r256_decode_codes", &decode_codes_cuda,
        "TCQ R256 scan-free native-code decoder");
  m.def("trellis_r256_decode_native_packed_prevalidated_out",
        &decode_native_packed_prevalidated_out_cuda,
        "TCQ R256 prevalidated native packed-code decoder (out variant)");
  m.def("trellis_r256_expand", &expand_cuda,
        "TCQ R256 explicit transient value expansion");
  m.def("trellis_r256_dequant_gemv", &dequant_gemv_cuda,
        "TCQ R256 direct decode/dequant GEMV");
  m.def("trellis_r256_wire_schema", []() {
    return std::string("gridbook.trellis.wire.v1");
  });
  m.def("trellis_r256_abi_schema", []() { return int64_t(2); });
}
