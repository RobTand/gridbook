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

// ---------------------------------------------------------------------------
// v2 warp-resident decode core.
//
// The scan-free kernels above give one THREAD one weight and re-read the
// columns x MEMORY_ORDER predecessor table for every row.  v2 makes the decode
// unit a WARP instead: one warp owns one (row, superblock), 32 lanes x 8
// columns = 256 columns, the block's u-bit mask is built with ballots (warp
// synchronous, no CTA barrier in the inner loop), and the tensor-shared plan
// and alphabet LUT are staged into shared memory once per CTA and amortised
// over kWarps x kRowsPerWarp rows.
//
// It consumes the compact plan of gridbook.trellis.derived_block_plan rather
// than the (column_offsets, previous_u_offsets) tables; that plan is validated
// on device against the already-validated scan-free tables by
// validate_block_plan_kernel before any decode thread runs, so a malformed
// raw-ABI plan still fails closed.
//
// The decode core is one device function with three epilogues (native code
// tile, scaled dense transient, fused GEMV) so the three paths cannot drift
// numerically.  Output is bit-identical to the scan-free decoder; that is a
// gate, not an aspiration.

constexpr unsigned kFull = 0xffffffffu;
constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarp * kWarps;
constexpr int kColsPerLane = kSuperblock / kWarp;   // 8
constexpr int kRowsPerWarp = 8;
constexpr int kRowsPerCta = kWarps * kRowsPerWarp;  // 64
// 256 columns * 8 bits max = 2048 bits = 64 words, + 1 funnel-shift guard.
constexpr int kBodyWords = 65;
constexpr int kMaskWords = kSuperblock / 32 + 1;    // 9

__device__ __forceinline__ unsigned bits_from(
    unsigned const* words, int bit, int width) {
  if (width <= 0) return 0u;
  int const word = bit >> 5;
  unsigned const value = __funnelshift_r(words[word], words[word + 1], bit & 31);
  return value & ((1u << width) - 1u);
}

__device__ __forceinline__ unsigned history8(
    unsigned const* mask, int at, int count) {
  if (at >= kMemory) {
    int const start = at - kMemory;
    return __funnelshift_r(mask[start >> 5], mask[(start >> 5) + 1],
                           start & 31) & 0xffu;
  }
  unsigned out = 0u;
#pragma unroll
  for (int j = 0; j < kMemory; ++j) {
    int index = at - kMemory + j;
    if (index < 0) index += count;
    out |= ((mask[index >> 5] >> (index & 31)) & 1u) << j;
  }
  return out;
}

struct BlockPlan {
  int start_column;
  int columns;
  int coded_count;
  int bit_offset;
  int bit_length;
};

__device__ __forceinline__ BlockPlan load_block_plan(
    int32_t const* __restrict__ block_meta, int block) {
  BlockPlan plan;
  plan.start_column = block_meta[block * 5 + 0];
  plan.columns = block_meta[block * 5 + 1];
  plan.coded_count = block_meta[block * 5 + 2];
  plan.bit_offset = block_meta[block * 5 + 3];
  plan.bit_length = block_meta[block * 5 + 4];
  return plan;
}

// Stage one row's superblock body into a per-warp buffer, bit-aligned to the
// block's own start so relative column offsets need no further shifting.
__device__ __forceinline__ void stage_body(
    uint8_t const* __restrict__ payload_row, int row_stride,
    BlockPlan const& plan, unsigned* body, int lane) {
  int const row_words = row_stride >> 2;      // stride is 16-byte aligned
  int const word_base = plan.bit_offset >> 5;
  int const bit_shift = plan.bit_offset & 31;
  int const need = ((plan.bit_length + bit_shift + 31) >> 5) + 1;
  unsigned const* row_words_ptr =
      reinterpret_cast<unsigned const*>(payload_row);
  for (int w = lane; w < need && w < kBodyWords; w += kWarp) {
    int const index = word_base + w;
    unsigned const lo = (index < row_words) ? row_words_ptr[index] : 0u;
    unsigned const hi =
        (index + 1 < row_words) ? row_words_ptr[index + 1] : 0u;
    body[w] = bit_shift ? __funnelshift_r(lo, hi, bit_shift) : lo;
  }
  for (int w = need + lane; w < kBodyWords; w += kWarp) body[w] = 0u;
  __syncwarp();
}

// Materialise the block's u-bit mask with ballots.  Warp-synchronous: no
// shared-memory atomic and no CTA barrier.  Valid only when column order and
// coded-ordinal order coincide, i.e. the block has no bypass column.
__device__ __forceinline__ void stage_mask(
    BlockPlan const& plan, int terminal_rate,
    uint8_t const* s_rate, int32_t const* s_offset,
    unsigned const* body, unsigned* mask, int lane) {
#pragma unroll
  for (int k = 0; k < kColsPerLane; ++k) {
    int const local = k * kWarp + lane;
    unsigned u = 0u;
    if (local < plan.columns) {
      int const rate = int(s_rate[local]);
      if (rate < terminal_rate) u = bits_from(body, s_offset[local], 1);
    }
    unsigned const word = __ballot_sync(kFull, u != 0u);
    if (lane == 0) mask[k] = word;
  }
  if (lane == 0) mask[kColsPerLane] = 0u;   // funnel-shift guard
  __syncwarp();
}

// General path: history8 indexes the mask in CODED-ORDINAL order, which
// diverges from column order as soon as the block carries a bypass column.
__device__ __forceinline__ void stage_mask_by_ordinal(
    BlockPlan const& plan, int terminal_rate,
    uint8_t const* s_rate, int32_t const* s_offset, int16_t const* s_ordinal,
    unsigned const* body, unsigned* mask, int lane) {
  for (int w = lane; w < kMaskWords; w += kWarp) mask[w] = 0u;
  __syncwarp();
#pragma unroll
  for (int k = 0; k < kColsPerLane; ++k) {
    int const local = k * kWarp + lane;
    unsigned u = 0u;
    int ordinal = -1;
    if (local < plan.columns) {
      int const rate = int(s_rate[local]);
      ordinal = int(s_ordinal[local]);
      if (rate < terminal_rate && ordinal >= 0) {
        u = bits_from(body, s_offset[local], 1);
      }
    }
    // Ordinals are strictly increasing in column order, so each lane's bit has
    // a distinct destination; a ballot cannot be used, but the write is to
    // shared memory the warp owns exclusively.
    if (u) atomicOr(&mask[ordinal >> 5], 1u << (ordinal & 31));
  }
  __syncwarp();
}

__device__ __forceinline__ uint8_t decode_lane_column(
    BlockPlan const& plan, int terminal_rate,
    uint8_t const* s_rate, int32_t const* s_offset, int16_t const* s_ordinal,
    unsigned const* body, unsigned const* mask, uint8_t const* s_lut,
    int local) {
  int const rate = int(s_rate[local]);
  int const offset = s_offset[local];
  if (rate >= terminal_rate) {
    return uint8_t(bits_from(body, offset, terminal_rate));
  }
  unsigned const u = bits_from(body, offset, 1);
  unsigned const state = history8(mask, int(s_ordinal[local]), plan.coded_count);
  unsigned const reg = (u << kMemory) | state;
  unsigned const subset =
      2u * (__popc(reg & kG0) & 1u) + (__popc(reg & kG1) & 1u);
  unsigned const point = bits_from(body, offset + 1, rate - 1);
  return s_lut[rate * 256 + subset * (1u << (rate - 1)) + point];
}

struct CtaPlan {
  int32_t offset[kSuperblock];
  int16_t ordinal[kSuperblock];
  uint8_t rate[kSuperblock];
  uint8_t all_coded;   // 1 when the block has no bypass column
};

__device__ __forceinline__ void stage_cta_plan(
    uint8_t const* __restrict__ col_rate,
    int32_t const* __restrict__ col_bit_offset,
    int16_t const* __restrict__ col_ordinal,
    uint8_t const* __restrict__ alphabet_lut, int lut_bytes,
    BlockPlan const& plan, CtaPlan* s_plan, uint8_t* s_lut) {
  for (int i = threadIdx.x; i < plan.columns; i += blockDim.x) {
    int const column = plan.start_column + i;
    s_plan->rate[i] = col_rate[column];
    s_plan->offset[i] = col_bit_offset[column] - plan.bit_offset;
    s_plan->ordinal[i] = col_ordinal[column];
  }
  for (int i = threadIdx.x; i < lut_bytes; i += blockDim.x) {
    s_lut[i] = alphabet_lut[i];
  }
  if (threadIdx.x == 0) {
    s_plan->all_coded = uint8_t(plan.coded_count == plan.columns);
  }
  __syncthreads();
}

extern __shared__ unsigned char smem_raw[];

struct Shared {
  CtaPlan* plan;
  uint8_t* lut;
  unsigned* body;   // [kWarps][kBodyWords]
  unsigned* mask;   // [kWarps][kMaskWords]
};

__device__ __forceinline__ Shared shared_layout(int lut_bytes) {
  Shared s;
  s.plan = reinterpret_cast<CtaPlan*>(smem_raw);
  s.lut = smem_raw + sizeof(CtaPlan);
  unsigned char* cursor = s.lut + lut_bytes;
  // 4-byte align the word buffers
  cursor += (4 - (reinterpret_cast<uintptr_t>(cursor) & 3)) & 3;
  s.body = reinterpret_cast<unsigned*>(cursor);
  s.mask = s.body + kWarps * kBodyWords;
  return s;
}

size_t shared_bytes(int lut_bytes) {
  return sizeof(CtaPlan) + size_t(lut_bytes) + 4
       + size_t(kWarps) * (kBodyWords + kMaskWords) * sizeof(unsigned);
}

// Stage the per-warp body + mask for one row.  Shared by all three epilogues.
__device__ __forceinline__ void stage_row(
    uint8_t const* __restrict__ payload, int row, int row_stride,
    BlockPlan const& plan, int terminal_rate, Shared const& s,
    unsigned* body, unsigned* mask, int lane) {
  uint8_t const* payload_row = payload + int64_t(row) * row_stride;
  stage_body(payload_row, row_stride, plan, body, lane);
  if (s.plan->all_coded) {
    stage_mask(plan, terminal_rate, s.plan->rate, s.plan->offset,
               body, mask, lane);
  } else {
    stage_mask_by_ordinal(plan, terminal_rate, s.plan->rate, s.plan->offset,
                          s.plan->ordinal, body, mask, lane);
  }
}

// The compact plan is derived workspace, but the pybind symbols are a raw ABI
// too.  Cross-check it against the scan-free tables that
// validate_decode_plan_kernel has already proven consistent with the payload,
// so a malformed compact plan can never reach a bit address.
__global__ void validate_block_plan_kernel(
    uint8_t const* schedule,
    int64_t const* column_offsets,
    uint8_t const* col_rate,
    int32_t const* col_bit_offset,
    int16_t const* col_ordinal,
    int32_t const* block_meta,
    int columns, int blocks, int terminal_rate, int lut_bytes,
    int64_t const* row_body_bits, int* error) {
  if (*error != kDecodeOk) return;
  if (lut_bytes < terminal_rate * 256) {
    record_decode_error(error, kDecodeInvalidAlphabetCode);
    return;
  }
  int const expected_blocks = (columns + kSuperblock - 1) / kSuperblock;
  if (blocks != expected_blocks) {
    record_decode_error(error, kDecodeInvalidSchedule);
    return;
  }
  for (int block = 0; block < blocks; ++block) {
    int const start = block_meta[block * 5 + 0];
    int const count = block_meta[block * 5 + 1];
    int const coded = block_meta[block * 5 + 2];
    int const bit_offset = block_meta[block * 5 + 3];
    int const bit_length = block_meta[block * 5 + 4];
    int const expected_start = block * kSuperblock;
    int const expected_count =
        min(kSuperblock, columns - expected_start);
    if (start != expected_start || count != expected_count ||
        count <= 0 || count > kSuperblock) {
      record_decode_error(error, kDecodeInvalidSchedule);
      return;
    }
    if (bit_offset != int(column_offsets[start])) {
      record_decode_error(error, kDecodeInvalidColumnOffset);
      return;
    }
    int ordinal = 0;
    int cursor = bit_offset;
    for (int i = 0; i < count; ++i) {
      int const column = start + i;
      int const rate = int(schedule[column]);
      if (int(col_rate[column]) != rate) {
        record_decode_error(error, kDecodeInvalidSchedule);
        return;
      }
      if (int64_t(col_bit_offset[column]) != column_offsets[column] ||
          col_bit_offset[column] != cursor) {
        record_decode_error(error, kDecodeInvalidColumnOffset);
        return;
      }
      int const want = (rate < terminal_rate) ? ordinal : -1;
      if (int(col_ordinal[column]) != want) {
        record_decode_error(error, kDecodeInvalidPreviousOffset);
        return;
      }
      if (rate < terminal_rate) ++ordinal;
      cursor += rate;
    }
    if (ordinal != coded || coded < kMemory) {
      record_decode_error(error, kDecodeInvalidPreviousOffset);
      return;
    }
    // A full superblock at the terminal rate is 256*8 = 2048 bits, exactly the
    // capacity the per-warp staging buffer addresses: relative offsets reach
    // bit_length-rate, so bits_from touches at most word 64 of kBodyWords=65.
    if (cursor - bit_offset != bit_length ||
        bit_length > kSuperblock * terminal_rate ||
        bit_length > (kBodyWords - 1) * 32) {
      record_decode_error(error, kDecodeInvalidColumnOffset);
      return;
    }
  }
  int const last = columns - 1;
  if (int64_t(col_bit_offset[last]) + int64_t(col_rate[last]) !=
      *row_body_bits) {
    record_decode_error(error, kDecodeInvalidColumnOffset);
  }
}

__global__ void __launch_bounds__(kThreads)
decode_native_v2_kernel(
    uint8_t const* __restrict__ payload,
    uint8_t const* __restrict__ col_rate,
    int32_t const* __restrict__ col_bit_offset,
    int16_t const* __restrict__ col_ordinal,
    int32_t const* __restrict__ block_meta,
    uint8_t const* __restrict__ alphabet_lut,
    uint8_t* __restrict__ out,
    int rows, int out_columns, int row_stride,
    int family, int terminal_rate, int lut_bytes) {
  BlockPlan const plan = load_block_plan(block_meta, blockIdx.x);
  Shared const s = shared_layout(lut_bytes);
  stage_cta_plan(col_rate, col_bit_offset, col_ordinal, alphabet_lut,
                 lut_bytes, plan, s.plan, s.lut);

  int const warp = int(threadIdx.x) >> 5;
  int const lane = int(threadIdx.x) & 31;
  unsigned* body = s.body + warp * kBodyWords;
  unsigned* mask = s.mask + warp * kMaskWords;

  int const row_begin = int(blockIdx.y) * kRowsPerCta + warp * kRowsPerWarp;
  int const row_end = min(row_begin + kRowsPerWarp, rows);
  for (int row = row_begin; row < row_end; ++row) {
    stage_row(payload, row, row_stride, plan, terminal_rate, s, body, mask,
              lane);
#pragma unroll
    for (int k = 0; k < kColsPerLane; ++k) {
      int const local = k * kWarp + lane;
      uint8_t code = 0;
      if (local < plan.columns) {
        code = decode_lane_column(plan, terminal_rate, s.plan->rate,
                                  s.plan->offset, s.plan->ordinal, body, mask,
                                  s.lut, local);
      }
      if (family == kFamilyE2M1) {
        uint8_t const partner =
            uint8_t(__shfl_down_sync(kFull, unsigned(code), 1));
        bool const even = (lane & 1) == 0;
        int const column = plan.start_column + local;
        if (even && local < plan.columns) {
          uint8_t const high =
              (local + 1 < plan.columns) ? partner : uint8_t(0);
          int const out_column = column >> 1;
          if (out_column < out_columns) {
            out[int64_t(row) * out_columns + out_column] =
                uint8_t((code & 0x0fu) | ((high & 0x0fu) << 4));
          }
        }
      } else if (local < plan.columns) {
        out[int64_t(row) * out_columns + plan.start_column + local] = code;
      }
    }
    __syncwarp();
  }
}

__global__ void __launch_bounds__(kThreads)
expand_v2_kernel(
    uint8_t const* __restrict__ payload,
    uint8_t const* __restrict__ col_rate,
    int32_t const* __restrict__ col_bit_offset,
    int16_t const* __restrict__ col_ordinal,
    int32_t const* __restrict__ block_meta,
    uint8_t const* __restrict__ alphabet_lut,
    float const* __restrict__ scales,
    float* __restrict__ out,
    int rows, int columns, int row_stride,
    int family, int terminal_rate, int lut_bytes, int scale_stride) {
  BlockPlan const plan = load_block_plan(block_meta, blockIdx.x);
  Shared const s = shared_layout(lut_bytes);
  stage_cta_plan(col_rate, col_bit_offset, col_ordinal, alphabet_lut,
                 lut_bytes, plan, s.plan, s.lut);

  int const warp = int(threadIdx.x) >> 5;
  int const lane = int(threadIdx.x) & 31;
  unsigned* body = s.body + warp * kBodyWords;
  unsigned* mask = s.mask + warp * kMaskWords;

  int const row_begin = int(blockIdx.y) * kRowsPerCta + warp * kRowsPerWarp;
  int const row_end = min(row_begin + kRowsPerWarp, rows);
  for (int row = row_begin; row < row_end; ++row) {
    stage_row(payload, row, row_stride, plan, terminal_rate, s, body, mask,
              lane);
#pragma unroll
    for (int k = 0; k < kColsPerLane; ++k) {
      int const local = k * kWarp + lane;
      if (local >= plan.columns) continue;
      int const column = plan.start_column + local;
      uint8_t const code = decode_lane_column(
          plan, terminal_rate, s.plan->rate, s.plan->offset, s.plan->ordinal,
          body, mask, s.lut, local);
      out[int64_t(row) * columns + column] =
          decoded_value(code, scales, family, row, column, scale_stride);
    }
    __syncwarp();
  }
}

// NOTE: the fused GEMV deliberately stays on the scan-free kernel above.  The
// warp-resident form reduces per-superblock partials with atomicAdd, which is
// ~2.7x faster but not run-to-run deterministic; this repo quarantines
// irreproducible numbers, and the GEMV is a research decode rung on no serving
// path.  A deterministic fixed-order reduction is the way to claim that speedup.

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

int terminal_rate_of(int family) {
  return family == kFamilyE2M1 ? 4 : 8;
}

// The v2 kernels tile (superblock, row-block); both grid extents are checked
// against the device limits before launch, as the scan-free path does.
dim3 v2_grid(torch::Tensor const& block_meta, int64_t rows, int device,
             char const* operation) {
  int64_t const blocks = block_meta.size(0);
  int64_t const row_tiles = (rows + kRowsPerCta - 1) / kRowsPerCta;
  TORCH_CHECK(blocks > 0 && row_tiles > 0,
              operation, " launch extents must be positive");
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  TORCH_CHECK(blocks <= properties.maxGridSize[0] && blocks <= INT_MAX,
              operation, " launch requires ", blocks,
              " x-grid blocks, device limit is ", properties.maxGridSize[0]);
  TORCH_CHECK(row_tiles <= properties.maxGridSize[1] && row_tiles <= INT_MAX,
              operation, " launch requires ", row_tiles,
              " y-grid blocks, device limit is ", properties.maxGridSize[1]);
  TORCH_CHECK(
      shared_bytes(0) + 8 * 256 <= properties.sharedMemPerBlock,
      operation, " needs more shared memory than the device provides");
  return dim3(int(blocks), int(row_tiles));
}

void validate_compact_plan(
    torch::Tensor const& payload,
    torch::Tensor const& col_rate,
    torch::Tensor const& col_bit_offset,
    torch::Tensor const& col_ordinal,
    torch::Tensor const& block_meta,
    int64_t columns) {
  TCQ_CHECK_CUDA(col_rate);
  TCQ_CHECK_CUDA(col_bit_offset);
  TCQ_CHECK_CUDA(col_ordinal);
  TCQ_CHECK_CUDA(block_meta);
  TCQ_CHECK_CONTIGUOUS(col_rate);
  TCQ_CHECK_CONTIGUOUS(col_bit_offset);
  TCQ_CHECK_CONTIGUOUS(col_ordinal);
  TCQ_CHECK_CONTIGUOUS(block_meta);
  TORCH_CHECK(col_rate.scalar_type() == torch::kUInt8,
              "col_rate must be uint8");
  TORCH_CHECK(col_bit_offset.scalar_type() == torch::kInt32,
              "col_bit_offset must be int32");
  TORCH_CHECK(col_ordinal.scalar_type() == torch::kInt16,
              "col_ordinal must be int16");
  TORCH_CHECK(block_meta.scalar_type() == torch::kInt32,
              "block_meta must be int32");
  TORCH_CHECK(col_rate.dim() == 1 && col_rate.size(0) == columns,
              "col_rate must have shape [columns]");
  TORCH_CHECK(col_bit_offset.dim() == 1 && col_bit_offset.size(0) == columns,
              "col_bit_offset must have shape [columns]");
  TORCH_CHECK(col_ordinal.dim() == 1 && col_ordinal.size(0) == columns,
              "col_ordinal must have shape [columns]");
  int64_t const blocks = (columns + kSuperblock - 1) / kSuperblock;
  TORCH_CHECK(block_meta.dim() == 2 && block_meta.size(0) == blocks &&
                  block_meta.size(1) == 5,
              "block_meta must have exact shape [blocks,5]");
  TORCH_CHECK(payload.device() == col_rate.device() &&
                  payload.device() == col_bit_offset.device() &&
                  payload.device() == col_ordinal.device() &&
                  payload.device() == block_meta.device(),
              "all trellis tensors must be on one CUDA device");
}

void launch_block_plan_validation(
    torch::Tensor const& schedule,
    torch::Tensor const& column_offsets,
    torch::Tensor const& col_rate,
    torch::Tensor const& col_bit_offset,
    torch::Tensor const& col_ordinal,
    torch::Tensor const& block_meta,
    torch::Tensor const& alphabet_lut,
    int columns, int family,
    torch::Tensor const& error,
    torch::Tensor const& row_body_bits,
    at::cuda::CUDAStream stream) {
  int const terminal_rate = family == kFamilyE2M1 ? 4 : 8;
  validate_block_plan_kernel<<<1, 1, 0, stream>>>(
      schedule.data_ptr<uint8_t>(), column_offsets.data_ptr<int64_t>(),
      col_rate.data_ptr<uint8_t>(), col_bit_offset.data_ptr<int32_t>(),
      col_ordinal.data_ptr<int16_t>(), block_meta.data_ptr<int32_t>(),
      columns, int(block_meta.size(0)), terminal_rate,
      int(alphabet_lut.numel()), row_body_bits.data_ptr<int64_t>(),
      error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    torch::Tensor col_rate, torch::Tensor col_bit_offset,
    torch::Tensor col_ordinal, torch::Tensor block_meta,
    torch::Tensor alphabet_lut, torch::Tensor scales,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  validate_compact_plan(payload, col_rate, col_bit_offset, col_ordinal,
                        block_meta, columns);
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
  launch_block_plan_validation(
      schedule, column_offsets, col_rate, col_bit_offset, col_ordinal,
      block_meta, alphabet_lut, int(columns), int(family), error,
      row_body_bits, stream);
  launch_scale_validation(scales, error, stream);
  check_device_error(error, stream);
  return row_body_bits;
}

void decode_native_packed_prevalidated_out_cuda(
    torch::Tensor payload, torch::Tensor col_rate,
    torch::Tensor col_bit_offset, torch::Tensor col_ordinal,
    torch::Tensor block_meta, torch::Tensor alphabet_lut,
    torch::Tensor output,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  TCQ_CHECK_CUDA(payload);
  TCQ_CHECK_CONTIGUOUS(payload);
  TCQ_CHECK_CUDA(alphabet_lut);
  TCQ_CHECK_CONTIGUOUS(alphabet_lut);
  TORCH_CHECK(payload.scalar_type() == torch::kUInt8, "payload must be uint8");
  TORCH_CHECK(alphabet_lut.scalar_type() == torch::kUInt8,
              "alphabet_lut must be uint8");
  TORCH_CHECK(rows > 0 && columns > 0 && row_stride > 0,
              "rows, columns, and row_stride must be positive");
  TORCH_CHECK(rows <= INT_MAX && columns <= INT_MAX && row_stride <= INT_MAX,
              "rows, columns, and row_stride must fit int32");
  TORCH_CHECK(row_stride % 16 == 0,
              "row_stride must preserve the wire's 16-byte alignment");
  TORCH_CHECK(payload.dim() == 2 && payload.size(0) == rows &&
                  payload.size(1) == row_stride,
              "payload must have shape [rows,row_stride]");
  TORCH_CHECK(family == kFamilyE2M1 || family == kFamilyE4M3,
              "family must be 1 (E2M1) or 2 (E4M3)");
  int const terminal_rate = family == kFamilyE2M1 ? 4 : 8;
  TORCH_CHECK(alphabet_lut.dim() == 2 &&
                  alphabet_lut.size(0) == terminal_rate &&
                  alphabet_lut.size(1) == 256,
              "alphabet_lut must have exact shape [native_bits,256]");
  validate_compact_plan(payload, col_rate, col_bit_offset, col_ordinal,
                        block_meta, columns);
  TCQ_CHECK_CUDA(output);
  TCQ_CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == torch::kUInt8,
              "native packed output must be uint8");
  TORCH_CHECK(output.device() == payload.device(),
              "native packed output must be on the payload device");
  int64_t const output_columns =
      family == kFamilyE2M1 ? (columns + 1) / 2 : columns;
  TORCH_CHECK(output.dim() == 2 && output.size(0) == rows &&
                  output.size(1) == output_columns,
              "native packed output has the wrong shape");
  c10::cuda::CUDAGuard guard(payload.device());
  checked_product(rows, output_columns, "prevalidated native packed decode");
  auto stream = at::cuda::getCurrentCUDAStream(payload.device().index());
  int const lut_bytes = int(alphabet_lut.numel());
  decode_native_v2_kernel<<<v2_grid(block_meta, rows, payload.device().index(),
                                   "prevalidated native packed decode"),
                            kThreads, shared_bytes(lut_bytes), stream>>>(
      payload.data_ptr<uint8_t>(), col_rate.data_ptr<uint8_t>(),
      col_bit_offset.data_ptr<int32_t>(), col_ordinal.data_ptr<int16_t>(),
      block_meta.data_ptr<int32_t>(), alphabet_lut.data_ptr<uint8_t>(),
      output.data_ptr<uint8_t>(), int(rows), int(output_columns),
      int(row_stride), int(family), terminal_rate, lut_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor expand_cuda(
    torch::Tensor payload, torch::Tensor schedule,
    torch::Tensor column_offsets, torch::Tensor previous_u_offsets,
    torch::Tensor col_rate, torch::Tensor col_bit_offset,
    torch::Tensor col_ordinal, torch::Tensor block_meta,
    torch::Tensor alphabet_lut, torch::Tensor scales,
    int64_t rows, int64_t columns, int64_t row_stride, int64_t family) {
  validate_common(payload, schedule, column_offsets, previous_u_offsets,
                  alphabet_lut, rows, columns, row_stride, family);
  validate_compact_plan(payload, col_rate, col_bit_offset, col_ordinal,
                        block_meta, columns);
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
  checked_product(rows, columns, "expand");
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
  launch_block_plan_validation(
      schedule, column_offsets, col_rate, col_bit_offset, col_ordinal,
      block_meta, alphabet_lut, int(columns), int(family), error,
      row_body_bits, stream);
  launch_scale_validation(scales, error, stream);
  check_device_error(error, stream);
  int const lut_bytes = int(alphabet_lut.numel());
  expand_v2_kernel<<<v2_grid(block_meta, rows, payload.device().index(),
                             "expand"),
                     kThreads, shared_bytes(lut_bytes), stream>>>(
      payload.data_ptr<uint8_t>(), col_rate.data_ptr<uint8_t>(),
      col_bit_offset.data_ptr<int32_t>(), col_ordinal.data_ptr<int16_t>(),
      block_meta.data_ptr<int32_t>(), alphabet_lut.data_ptr<uint8_t>(),
      scales.data_ptr<float>(), output.data_ptr<float>(),
      int(rows), int(columns), int(row_stride), int(family), terminal_rate_of(
          int(family)), lut_bytes, scale_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  m.def("trellis_r256_abi_schema", []() { return int64_t(3); });
}
