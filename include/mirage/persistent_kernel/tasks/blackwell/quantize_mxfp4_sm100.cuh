/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "../common/common_header.cuh"

#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

namespace kernel {

// MXFP4 quantizer. Layout & control flow mirrors quantize_nvfp4_sm100.cuh; the
// MXFP4-specific deltas are:
//   * GROUP_SIZE = 32 (MXFP4 spec) vs 16 for NVFP4 — twice as many e2m1
//   elements
//     share one scale, halving the number of scale bytes per row.
//   * Scale dtype is e8m0 (1-byte unbiased exponent). The scale is the
//     power-of-two ceiling of group_max / max_4bit, stored as `floor(log2(s)) +
//     127`. Computed via frexpf so the result is exact and IEEE-correct without
//     touching f32 bit patterns.
//   * Interleaved SF layout: [PADDED_BATCH/128, HIDDEN/64, 32, 4, 2] — same
//     outer indexing as NVFP4 but the innermost dim halves (4 → 2) because each
//     64-K-element "block" now contains only 2 scales instead of 4.

template <int SUBWARP_SIZE>
__device__ __forceinline__ float group_reduce_max_mx(float val) {
#pragma unroll
  for (int offset = SUBWARP_SIZE >> 1; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset, SUBWARP_SIZE));
  }
  return val;
}

// MXFP4 interleaved SF offset.
//
// CUTLASS's Sm1xxBlockScaledBasicChunk<SFVecSize> declares an SF atom of shape
// `((32, 4), (SFVecSize, 4))` with strides `((16, 4), (0, 1))` = 512 bytes
// regardless of SFVecSize (the SFVecSize axis is broadcast, stride 0). For
// NVFP4 (SFVecSize=16) that atom maps to 64 K-elements. The same 512-byte
// atom — used by the MMA's `scale_vec::2X` instruction for MXFP4 — covers
// half as many *usable* scales (2 NSF instead of 4), so it still represents
// only 64 K-elements but with 2 padding bytes per (within, row_group).
//
// We therefore use the SAME gmem layout as NVFP4: [PADDED_BATCH/128, K/64,
// 32, 4, 4]. The quantizer writes the e8m0 byte into the first 2 slots of
// the inner `4` axis; the other 2 slots are unused padding.
__device__ __forceinline__ int interleaved_mxfp4_scale_offset(
    int row_idx, int group_idx, int num_k_outer, int scale_outer_stride) {
  int row_in_block = row_idx & 127;
  // 2 MXFP4 scales per 64-K atom (1 scale per 32 K-elements; vec::2X uses
  // NSF=2). group_idx = scale_id along K. k_outer = group_idx >> 1; k_inner =
  // group_idx & 1.
  return (row_idx >> 7) * num_k_outer * scale_outer_stride +
         (group_idx >> 1) * scale_outer_stride + (row_in_block & 31) * 16 +
         ((row_in_block >> 5) & 3) * 4 + (group_idx & 1);
}

// Per-tile swapAB layout for MXFP4: same 512-B atom shape as NVFP4.
__device__ __forceinline__ int swapab_mxfp4_scale_offset(int row_idx,
                                                         int group_idx,
                                                         int num_k_outer,
                                                         int mma_n) {
  int n_tile = row_idx / mma_n;
  int i = row_idx % mma_n;
  int row_group = i >> 5;
  int within_32 = i & 31;
  int k_outer = group_idx >> 1;
  int k_inner = group_idx & 1;
  return n_tile * (num_k_outer * 32 * 4 * 4) + k_outer * (32 * 4 * 4) +
         within_32 * 16 + row_group * 4 + k_inner;
}

// Encode a positive float as an e8m0 byte: stores the unbiased exponent of the
// smallest power of two >= x. NaN / non-positive inputs map to 0x00
// (subnormal).
__device__ __forceinline__ uint8_t encode_e8m0(float x) {
  if (!(x > 0.0f)) {
    return 0x00;
  }
  int e;
  float m = frexpf(x, &e); // x = m * 2^e, m in [0.5, 1.0). So x <= 2^e.
  // Clamp to e8m0 representable range: unbiased exponent in [-127, 127].
  int unbiased = e - 1; // 2^(e-1) <= x <= 2^e
  // Round up: if mantissa > 0.5 we need the next exponent so 2^exp >= x.
  if (m > 0.5f) {
    unbiased = e;
  }
  if (unbiased < -127) {
    unbiased = -127;
  }
  if (unbiased > 127) {
    unbiased = 127;
  }
  return static_cast<uint8_t>(unbiased + 127);
}

__device__ __forceinline__ float decode_e8m0(uint8_t s) {
  if (s == 0x00) {
    return 0.0f;
  }
  int unbiased = static_cast<int>(s) - 127;
  return ldexpf(1.0f, unbiased);
}

// Quantizes a single logical row `row_idx` (mirrors quantize_nvfp4_one_row).
// `row_in`/`row_q` point at that row's input / packed-output base; `output_s_ptr`
// is the scale-tensor base (the scale write computes its own absolute offset
// from `row_idx`). `valid_row` selects real vs padded rows. Shared by the
// per-CTA entry and the MPK whole-batch loop.
template <int HIDDEN_SIZE, int GROUP_SIZE, int GLOBAL_STRIDE, typename T>
__device__ __forceinline__ void
    quantize_mxfp4_one_row(T const *__restrict__ row_in,
                           uint8_t *__restrict__ row_q,
                           void *__restrict__ output_s_ptr,
                           int row_idx,
                           bool valid_row,
                           float eps,
                           float min_4bit,
                           float max_4bit,
                           int scale_outer_stride,
                           int mma_n) {
  constexpr int WARP_SIZE = 32;
  constexpr int SUBWARP_SIZE = WARP_SIZE; // MXFP4: one full warp per group of 32
  constexpr int GROUPS_PER_WARP = WARP_SIZE / SUBWARP_SIZE; // = 1
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
  auto *output_s = static_cast<uint8_t *>(output_s_ptr);

  int const lane_idx = threadIdx.x & (WARP_SIZE - 1);
  int const warp_idx = threadIdx.x / WARP_SIZE;
  int const subwarp_idx = lane_idx / SUBWARP_SIZE; // == 0 (SUBWARP == WARP)
  int const sublane_idx = lane_idx % SUBWARP_SIZE;
  int const groups_per_block = (blockDim.x / WARP_SIZE) * GROUPS_PER_WARP;
  int const num_k_outer = NUM_GROUPS_PER_ROW / 2; // 2 MXFP4 scales per 64-K atom

#pragma unroll
  for (int group_idx = warp_idx * GROUPS_PER_WARP + subwarp_idx;
       group_idx < NUM_GROUPS_PER_ROW;
       group_idx += groups_per_block) {
    int const element_idx = group_idx * GROUP_SIZE + sublane_idx;
    float const orig_val =
        valid_row ? static_cast<float>(row_in[element_idx]) : 0.0f;
    float const group_max =
        group_reduce_max_mx<SUBWARP_SIZE>(fmaxf(fabsf(orig_val), eps));
    // MXFP4 scale: smallest power of two such that group_max / scale <=
    // max_4bit.
    float const raw_scale = valid_row ? group_max / max_4bit : 1.0f;
    const uint8_t scale_e8m0 = encode_e8m0(raw_scale);
    float const applied_scale = decode_e8m0(scale_e8m0);

    if (sublane_idx == 0 && (mma_n == 0 || valid_row)) {
      int sf_offset =
          (mma_n > 0)
              ? swapab_mxfp4_scale_offset(
                    row_idx, group_idx, num_k_outer, mma_n)
              : interleaved_mxfp4_scale_offset(
                    row_idx, group_idx, num_k_outer, scale_outer_stride);
      output_s[sf_offset] = scale_e8m0;
    }

    float const inv_scale = applied_scale > 0.0f ? 1.0f / applied_scale : 0.0f;
    const uint8_t nibble =
        static_cast<uint8_t>(
            cutlass::float_e2m1_t(
                fminf(fmaxf(orig_val * inv_scale, min_4bit), max_4bit))
                .raw()) &
        0x0f;
    const uint8_t pair = __shfl_xor_sync(0xffffffffu, nibble, 1, SUBWARP_SIZE);

    if ((sublane_idx & 1) == 0) {
      row_q[group_idx * (GROUP_SIZE / 2) + (sublane_idx >> 1)] =
          nibble | static_cast<uint8_t>(pair << 4);
    }
  }
}

// A single CTA quantizes the whole (padded) batch, looping over every row via
// quantize_mxfp4_one_row. MPK dispatches one task per worker CTA with no
// meaningful blockIdx, and the standalone launcher invokes this with a 1-CTA
// grid; both share this one entry. Padded rows (>= batch_size, up to a multiple
// of 128) get zero data + scale = 1. Layout/API matches
// quantize_nvfp4_sm100_task_impl.
//
// input_ptr:   row-major [BATCH_SIZE, GLOBAL_STRIDE] input
// output_q_ptr: row-major [PADDED_BATCH_SIZE, GLOBAL_STRIDE/2] packed e2m1 bytes
// output_s_ptr: interleaved scale bytes [PADDED_BATCH_SIZE/128, HIDDEN/128,
//   32, 4, 4] (CUTLASS Sm1xxBlockScaledBasicChunk<SFVecSize=32>)
// scale_outer_stride: 32*4*4 = 512 for contiguous storage (same atom as NVFP4).
template <int HIDDEN_SIZE,
          int GROUP_SIZE,
          int GLOBAL_STRIDE,
          typename T,
          typename PACKED_T = uint8_t,
          typename SCALE_T = uint8_t>
__device__ __forceinline__ void
    quantize_mxfp4_sm100_task_impl(void const *__restrict__ input_ptr,
                                   void *__restrict__ output_q_ptr,
                                   void *__restrict__ output_s_ptr,
                                   int batch_size,
                                   float eps,
                                   float min_4bit = -6.0f,
                                   float max_4bit = 6.0f,
                                   int scale_outer_stride = 32 * 4 * 4,
                                   int mma_n = 0) {
  static_assert(GROUP_SIZE == 32, "MXFP4 requires GROUP_SIZE == 32");
  static_assert(HIDDEN_SIZE % GROUP_SIZE == 0,
                "HIDDEN_SIZE must be divisible by GROUP_SIZE");
  static_assert(HIDDEN_SIZE % (GROUP_SIZE * 2) == 0,
                "HIDDEN_SIZE must be divisible by 64 (one SF atom = 2 MXFP4 "
                "scales = 64 K-elements)");
  static_assert(GLOBAL_STRIDE >= HIDDEN_SIZE,
                "GLOBAL_STRIDE must cover at least one logical row");
  static_assert((GLOBAL_STRIDE % 2) == 0,
                "GLOBAL_STRIDE must be even for packed MXFP4 output");
  static_assert(std::is_same_v<PACKED_T, uint8_t>,
                "MXFP4 output must be stored as packed uint8_t bytes");
  static_assert(std::is_same_v<SCALE_T, uint8_t>,
                "MXFP4 scales must be stored as e8m0 bytes");

  constexpr int OUTPUT_Q_STRIDE = GLOBAL_STRIDE / 2;
  int const padded_batch_size = ((batch_size + 127) / 128) * 128;

  T const *input = static_cast<T const *>(input_ptr);
  auto *output_q = static_cast<PACKED_T *>(output_q_ptr);

  for (int row = 0; row < padded_batch_size; row++) {
    bool const valid_row = row < batch_size;
    quantize_mxfp4_one_row<HIDDEN_SIZE, GROUP_SIZE, GLOBAL_STRIDE, T>(
        input + (valid_row ? row : 0) * GLOBAL_STRIDE,
        output_q + row * OUTPUT_Q_STRIDE,
        output_s_ptr,
        row,
        valid_row,
        eps,
        min_4bit,
        max_4bit,
        scale_outer_stride,
        mma_n);
  }
}

} // namespace kernel
