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
#include "../common//common_header.cuh"
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <cuda_fp8.h>
namespace kernel {

template <int SUBWARP_SIZE>
__device__ __forceinline__ float group_reduce_max(float val) {
#pragma unroll
  for (int offset = SUBWARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, val, offset, SUBWARP_SIZE);
    val = fmaxf(val, other);
  }
  if (threadIdx.x == 0) {
    printf("group_reduce_max: %f, SUBWARP_SIZE: %d\n", val, SUBWARP_SIZE);
  }
  return val;
}

template <
    int BATCH_SIZE,
    int HIDDEN_SIZE,
    int GROUP_SIZE,
    int GLOBAL_STRIDE,
    typename T,
    typename DST_T,
    bool SCALE_UE8M0,
    typename SCALE_PACKED_T = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__device__ __forceinline__ void
    per_token_group_quantize_fp8_task_impl(void const *__restrict__ input_ptr,
                                           void *__restrict__ output_q_ptr,
                                           void *__restrict__ output_s_ptr,
                                           float const eps,
                                           float const min_8bit,
                                           float const max_8bit) {
  // Pointers
  T const *input = static_cast<T const *>(input_ptr);
  DST_T *output_q = static_cast<DST_T *>(output_q_ptr);
  SCALE_PACKED_T *output_s = static_cast<SCALE_PACKED_T *>(output_s_ptr);

  // Assume each thread handles 32B of data
  // each subwarp has n threads, n = group_size * 2 / 32,
  // when group_size = 128, n = 8, i.e. each subwarp has 8 threads handling each
  // group
  constexpr int ELEMENTS_PER_THREAD = 32 / sizeof(T);
  constexpr int SUBWARP_SIZE = GROUP_SIZE * sizeof(T) / 32;
  constexpr int NUM_SUBWARPS = HIDDEN_SIZE / GROUP_SIZE;
  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  constexpr int NUM_ELEMENTS_PER_PACK =
      sizeof(SCALE_PACKED_T) / sizeof(scale_element_t);
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
  constexpr int GLOBAL_GROUP_STRIDE = GLOBAL_STRIDE / GROUP_SIZE;

  // Assertions
  static_assert(SUBWARP_SIZE == 4 || SUBWARP_SIZE == 8 || SUBWARP_SIZE == 16,
                "SUBWARP_SIZE must be equal to 4, 8, or 16");

  // Calculate indices
  int const warp_idx = warp_id();
  int const thread_idx = threadIdx.x;
  int const lane_idx = thread_idx % SUBWARP_SIZE;
  int const subwarp_idx = thread_idx / SUBWARP_SIZE;
  int const num_groups_per_block = 256 / SUBWARP_SIZE; // 2wg

  if (threadIdx.x == 0) {
    printf(
        "subwarp_idx: %d, num_groups_per_block: %d, NUM_GROUPS_PER_ROW: %d\n",
        subwarp_idx,
        num_groups_per_block,
        NUM_GROUPS_PER_ROW);
  }
#pragma unroll
  for (int group_idx = subwarp_idx; group_idx < NUM_GROUPS_PER_ROW;
       group_idx += num_groups_per_block) {

    // Step 1: Find the max in each subwarp (group)
    float local_max = eps;
#pragma unroll
    for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
      float const abs_val = fabsf(static_cast<float const>(
          input[ele_idx + ELEMENTS_PER_THREAD * lane_idx +
                GROUP_SIZE * group_idx]));
      local_max = fmaxf(abs_val, local_max);
    }
    float group_max = group_reduce_max<SUBWARP_SIZE>(local_max);

    // Step 2: Compute and store the scale
    float y_scale = group_max / max_8bit;
    scale_element_t scale_quant;

    if constexpr (SCALE_UE8M0) {
      scale_quant = (uint8_t)(ceilf(log2f(fmaxf(y_scale, 1e-10f))) + 127);
    } else {
      scale_quant = y_scale;
      // TODO(Yu): decide which one is better，sglang uses this but has some
      // precision issues scale_quant = exp2f(ceilf(log2f(fmaxf(y_scale,
      // 1e-10f))));
    }

    if (threadIdx.x == 0) {
      printf("group_idx: %d, scale_quant: %f, group_max: %f, y_scale: %f\n",
             group_idx,
             scale_quant,
             group_max,
             y_scale);
    }
    if (lane_idx == 0) {
      int const pack_idx = group_idx / NUM_ELEMENTS_PER_PACK;
      int const in_pack_idx = group_idx % NUM_ELEMENTS_PER_PACK;
      // SCALE_UE8M0 and non-SCALE_UE8M0 share the same logic for simplicity
      output_s[pack_idx * NUM_ELEMENTS_PER_PACK + in_pack_idx] = scale_quant;
    }

    // Step 3: Compute and store the quantized output
#pragma unroll
    for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
      // TODO(Yu): use vectorized load to registers
      float const orig_val = static_cast<float const>(
          input[ele_idx + ELEMENTS_PER_THREAD * lane_idx +
                GROUP_SIZE * group_idx]);
      float const quant_val = fminf(fmaxf(orig_val / y_scale, min_8bit),
                                    max_8bit); // clip to [min_8bit, max_8bit]
      output_q[ele_idx + ELEMENTS_PER_THREAD * lane_idx +
               GROUP_SIZE * group_idx] = DST_T(quant_val);
    }
  }
}

} // namespace kernel
