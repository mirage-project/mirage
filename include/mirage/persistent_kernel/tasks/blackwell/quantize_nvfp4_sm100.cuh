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

template <int SUBWARP_SIZE>
__device__ __forceinline__ float group_reduce_max(float val) {
#pragma unroll
  for (int offset = SUBWARP_SIZE >> 1; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset, SUBWARP_SIZE));
  }
  return val;
}

__device__ __forceinline__ int interleaved_nvfp4_scale_offset(int row_idx,
                                                              int group_idx,
                                                              int num_k_outer,
                                                              int scale_outer_stride) {
  int row_in_block = row_idx & 127;
  return (row_idx >> 7) * num_k_outer * scale_outer_stride +
         (group_idx >> 2) * scale_outer_stride +
         (row_in_block & 31) * 16 +
         ((row_in_block >> 5) & 3) * 4 +
         (group_idx & 3);
}

// Offset into the per-tile swapAB layout: [num_n_tiles, sf_k_outer, 32, 4, 4]
//   n_tile   = row_idx / mma_n
//   i        = row_idx % mma_n  (row within tile)
//   row_group = i / 32,  within_32 = i % 32
//   k_outer  = group_idx / 4,  k_inner = group_idx % 4
__device__ __forceinline__ int swapab_nvfp4_scale_offset(int row_idx,
                                                         int group_idx,
                                                         int num_k_outer,
                                                         int mma_n) {
  int n_tile    = row_idx / mma_n;
  int i         = row_idx % mma_n;
  int row_group = i >> 5;
  int within_32 = i & 31;
  int k_outer   = group_idx >> 2;
  int k_inner   = group_idx & 3;
  // stride of [num_n_tiles, sf_k_outer, 32, 4, 4]: innermost = 1, then 4, 16, 32*4*4, ...
  return n_tile * (num_k_outer * 32 * 4 * 4) +
         k_outer * (32 * 4 * 4) +
         within_32 * 16 +
         row_group * 4 +
         k_inner;
}

// One CTA handles one row. Launch over ceil_div(BATCH_SIZE, 128) * 128 rows so
// padded rows can be filled with zero data and scale=1.
//
// input_ptr:
//   row-major [BATCH_SIZE, GLOBAL_STRIDE] input
// output_q_ptr:
//   row-major [PADDED_BATCH_SIZE, GLOBAL_STRIDE / 2] packed NVFP4 bytes
// output_s_ptr:
//   interleaved scale bytes laid out like
//   [PADDED_BATCH_SIZE / 128, HIDDEN_SIZE / 64, 32, 4, 4]
// scale_outer_stride:
//   byte stride between adjacent k_outer slices. Use 32 * 4 * 4 for contiguous
//   storage.
template <int HIDDEN_SIZE,
          int GROUP_SIZE,
          int GLOBAL_STRIDE,
          typename T,
          typename PACKED_T = uint8_t,
          typename SCALE_T = uint8_t>
__device__ __forceinline__ void
quantize_nvfp4_sm100_task_impl(const void *__restrict__ input_ptr,
                               void *__restrict__ output_q_ptr,
                               void *__restrict__ output_s_ptr,
                               int batch_size,
                               float eps,
                               float min_4bit = -6.0f,
                               float max_4bit = 6.0f,
                               int scale_outer_stride = 32 * 4 * 4,
                               int mma_n = 0) {
  static_assert(GROUP_SIZE == 16, "NVFP4 requires GROUP_SIZE == 16");
  static_assert(HIDDEN_SIZE % GROUP_SIZE == 0,
                "HIDDEN_SIZE must be divisible by GROUP_SIZE");
  static_assert(HIDDEN_SIZE % (GROUP_SIZE * 4) == 0,
                "HIDDEN_SIZE must be divisible by 64 for interleaved NVFP4 scales");
  static_assert(GLOBAL_STRIDE >= HIDDEN_SIZE,
                "GLOBAL_STRIDE must cover at least one logical row");
  static_assert((GLOBAL_STRIDE % 2) == 0,
                "GLOBAL_STRIDE must be even for packed NVFP4 output");
  static_assert(std::is_same_v<PACKED_T, uint8_t>,
                "NVFP4 output must be stored as packed uint8_t bytes");
  static_assert(std::is_same_v<SCALE_T, uint8_t>,
                "NVFP4 scales must be stored as ue4m3 bytes");

  constexpr int WARP_SIZE = 32;
  constexpr int SUBWARP_SIZE = GROUP_SIZE;
  constexpr int GROUPS_PER_WARP = WARP_SIZE / SUBWARP_SIZE;
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
  constexpr int OUTPUT_Q_STRIDE = GLOBAL_STRIDE / 2;
  const int padded_batch_size = ((batch_size + 127) / 128) * 128;

  const int row_idx = blockIdx.x;
  if (row_idx >= padded_batch_size) {
    return;
  }

  const T *input = static_cast<const T *>(input_ptr);
  auto *output_q = static_cast<PACKED_T *>(output_q_ptr);
  auto *output_s = static_cast<SCALE_T *>(output_s_ptr);

  const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int subwarp_idx = lane_idx / SUBWARP_SIZE;
  const int sublane_idx = lane_idx % SUBWARP_SIZE;
  const int groups_per_block = (blockDim.x / WARP_SIZE) * GROUPS_PER_WARP;
  const bool valid_row = row_idx < batch_size;

  const int input_row_offset = row_idx * GLOBAL_STRIDE;
  const int output_q_row_offset = row_idx * OUTPUT_Q_STRIDE;
  const int num_k_outer = NUM_GROUPS_PER_ROW / 4;

#pragma unroll
  for (int group_idx = warp_idx * GROUPS_PER_WARP + subwarp_idx;
       group_idx < NUM_GROUPS_PER_ROW;
       group_idx += groups_per_block) {
    const int element_idx = group_idx * GROUP_SIZE + sublane_idx;
    const float orig_val =
        valid_row ? static_cast<float>(input[input_row_offset + element_idx]) : 0.0f;
    const float group_max =
        group_reduce_max<SUBWARP_SIZE>(fmaxf(fabsf(orig_val), eps));
    const cutlass::float_ue4m3_t scale_quant(
        valid_row ? group_max / max_4bit : 1.0f);
    const float applied_scale = static_cast<float>(scale_quant);

    if (sublane_idx == 0 && (mma_n == 0 || valid_row)) {
      int sf_offset = (mma_n > 0)
          ? swapab_nvfp4_scale_offset(row_idx, group_idx, num_k_outer, mma_n)
          : interleaved_nvfp4_scale_offset(row_idx, group_idx, num_k_outer, scale_outer_stride);
      output_s[sf_offset] = scale_quant.raw();
    }

    const uint8_t nibble = static_cast<uint8_t>(
        cutlass::float_e2m1_t(
            fminf(fmaxf(orig_val / applied_scale, min_4bit), max_4bit))
            .raw()) &
        0x0f;
    const uint8_t pair = __shfl_xor_sync(0xffffffffu, nibble, 1, SUBWARP_SIZE);

    if ((sublane_idx & 1) == 0) {
      output_q[output_q_row_offset + group_idx * (GROUP_SIZE / 2) +
               (sublane_idx >> 1)] =
          nibble | static_cast<uint8_t>(pair << 4);
    }
  }
}

}  // namespace kernel
