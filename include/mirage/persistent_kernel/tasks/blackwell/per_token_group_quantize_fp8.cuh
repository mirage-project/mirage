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
#include "../common//common_header.cuh"

#include <cuda_fp8.h>
namespace kernel {

template <int SUBWARP_SIZE>
__device__ __forceinline__ float group_reduce_max(float val) {
#pragma unroll
  for (int offset = SUBWARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, val, offset, SUBWARP_SIZE);
    val = fmaxf(val, other);
  }
  return val;
}

__device__ __forceinline__ uint8_t encode_ue8m0(float scale) {
  int ue8m0 = static_cast<int>(ceilf(log2f(fmaxf(scale, 1e-30f)))) + 127;
  ue8m0 = max(0, min(255, ue8m0));
  return static_cast<uint8_t>(ue8m0);
}

template <int BATCH_SIZE,
          int HIDDEN_SIZE,
          int GROUP_SIZE,
          int GLOBAL_STRIDE,
          typename T,
          typename DST_T,
          bool SCALE_UE8M0,
          typename SCALE_PACKED_T =
              std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__device__ __forceinline__ void
per_token_group_quantize_fp8_task_impl(const void *__restrict__ input_ptr,
                       void *__restrict__ output_q_ptr,
                       void *__restrict__ output_s_ptr,
                       const float eps,
                       const float min_8bit,
                       const float max_8bit) {
  // Pointers
  const T *input = static_cast<const T *>(input_ptr);
  DST_T *output_q = static_cast<DST_T *>(output_q_ptr);
  SCALE_PACKED_T *output_s = static_cast<SCALE_PACKED_T *>(output_s_ptr);

  // Assume each thread handles 32B of data
  // each subwarp has n threads, n = group_size * 2 / 32,
  // when group_size = 128, n = 8, i.e. each subwarp has 8 threads handling each group
  constexpr int ELEMENTS_PER_THREAD = 32 / sizeof(T); 
  constexpr int SUBWARP_SIZE = GROUP_SIZE * sizeof(T) / 32;
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
  constexpr int SCALE_ALIGNMENT = SCALE_UE8M0 ? 4 : 1;
  constexpr int PADDED_SCALE_K =
      ((NUM_GROUPS_PER_ROW + SCALE_ALIGNMENT - 1) / SCALE_ALIGNMENT) *
      SCALE_ALIGNMENT;

  // Assertions
  static_assert(SUBWARP_SIZE == 4 || SUBWARP_SIZE == 8 || SUBWARP_SIZE == 16, "SUBWARP_SIZE must be equal to 4, 8, or 16");
  if constexpr (SCALE_UE8M0) {
    static_assert(GROUP_SIZE == 128,
                  "Packed UE8M0 scale currently requires GROUP_SIZE == 128");
    static_assert(std::is_same_v<SCALE_PACKED_T, uint32_t>,
                  "Packed UE8M0 scale must be stored as uint32");
    static_assert(SUBWARP_SIZE == 8,
                  "Packed UE8M0 scale expects 8 threads per 128-element group");
  }
  
  // Calculate indices
  const int thread_idx = threadIdx.x;
  const int lane_idx = thread_idx % SUBWARP_SIZE;
  const int subwarp_idx = thread_idx / SUBWARP_SIZE;
  const int num_groups_per_block = blockDim.x / SUBWARP_SIZE;

#pragma unroll
  for (int padded_idx = thread_idx; padded_idx < PADDED_SCALE_K;
       padded_idx += blockDim.x) {
    if (padded_idx >= NUM_GROUPS_PER_ROW) {
      output_s[padded_idx] = static_cast<SCALE_PACKED_T>(0);
    }
  }

#pragma unroll
  for (int group_idx = subwarp_idx; group_idx < NUM_GROUPS_PER_ROW; group_idx += num_groups_per_block) {
    const int group_base_idx = GROUP_SIZE * group_idx;

    float local_max = eps;
  #pragma unroll
    for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
      const float abs_val = fabsf(static_cast<float>(
          input[group_base_idx + ele_idx + ELEMENTS_PER_THREAD * lane_idx]));
      local_max = fmaxf(abs_val, local_max);
    }

    float y_scale = 0.0f;
    if constexpr (SCALE_UE8M0) {
      // Every two lanes cover one 32-element subtile inside the 128-element
      // group, which matches the packed UE8M0 scale contract used by GEMM.
      float pair_max = fmaxf(
          local_max, __shfl_xor_sync(0xffffffff, local_max, 1, SUBWARP_SIZE));
      pair_max = fmaxf(pair_max, 1e-10f);
      y_scale = pair_max / max_8bit;
      const uint8_t scale_quant = encode_ue8m0(y_scale);
      const uint8_t sub_scale0 =
          __shfl_sync(0xffffffff, scale_quant, 0, SUBWARP_SIZE);
      const uint8_t sub_scale1 =
          __shfl_sync(0xffffffff, scale_quant, 2, SUBWARP_SIZE);
      const uint8_t sub_scale2 =
          __shfl_sync(0xffffffff, scale_quant, 4, SUBWARP_SIZE);
      const uint8_t sub_scale3 =
          __shfl_sync(0xffffffff, scale_quant, 6, SUBWARP_SIZE);

      if (lane_idx == 0) {
        uint32_t packed_scale = static_cast<uint32_t>(sub_scale0) |
                                (static_cast<uint32_t>(sub_scale1) << 8) |
                                (static_cast<uint32_t>(sub_scale2) << 16) |
                                (static_cast<uint32_t>(sub_scale3) << 24);
        output_s[group_idx] = packed_scale;
      }
    } else {
      float group_max = group_reduce_max<SUBWARP_SIZE>(local_max);
      group_max = fmaxf(group_max, 1e-10f);
      y_scale = group_max / max_8bit;
      if (lane_idx == 0) {
        output_s[group_idx] = static_cast<SCALE_PACKED_T>(y_scale);
      }
    }

  #pragma unroll
    for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
      const int output_idx =
          group_base_idx + ele_idx + ELEMENTS_PER_THREAD * lane_idx;
      const float orig_val = static_cast<float>(input[output_idx]);
      const float quant_val =
          fminf(fmaxf(orig_val / y_scale, min_8bit), max_8bit);
      output_q[output_idx] = __nv_fp8_e4m3(quant_val);
    }
  }
}

} // namespace kernel
