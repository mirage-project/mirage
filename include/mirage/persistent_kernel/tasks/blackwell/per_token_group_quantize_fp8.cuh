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
  return val;
}

__device__ __forceinline__ uint8_t encode_ue8m0(float scale) {
  int ue8m0 = static_cast<int>(ceilf(log2f(fmaxf(scale, 1e-30f)))) + 127;
  ue8m0 = max(0, min(255, ue8m0));
  return static_cast<uint8_t>(ue8m0);
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
                                           float const max_8bit,
                                           int const scale_outer_stride) {
  // Pointers
  T const *input = static_cast<T const *>(input_ptr);
  DST_T *output_q = static_cast<DST_T *>(output_q_ptr);
  SCALE_PACKED_T *output_s = static_cast<SCALE_PACKED_T *>(output_s_ptr);

  // Assume each thread handles 32B of data
  // each subwarp has n threads, n = group_size * 2 / 32,
  // when group_size = 128, n = 8, i.e. each subwarp has 8 threads handling each
  // group
  constexpr int WARP_SIZE = 32;
  constexpr int ELEMENTS_PER_THREAD = GROUP_SIZE / WARP_SIZE;
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
  constexpr int SCALE_ALIGNMENT = SCALE_UE8M0 ? 4 : 1;
  constexpr int PACKED_SCALE_K =
      ((NUM_GROUPS_PER_ROW + SCALE_ALIGNMENT - 1) / SCALE_ALIGNMENT);
  __shared__ uint8_t packed_scale_bytes[NUM_GROUPS_PER_ROW];

  // Assertions
  if constexpr (SCALE_UE8M0) {
    static_assert(GROUP_SIZE == 128,
                  "Packed UE8M0 scale currently requires GROUP_SIZE == 128");
    static_assert(std::is_same_v<SCALE_PACKED_T, uint32_t>,
                  "Packed UE8M0 scale must be stored as uint32");
  }

  // Calculate indices
  int const thread_idx = threadIdx.x;
  int const lane_idx = thread_idx % WARP_SIZE;
  int const warp_idx = thread_idx / WARP_SIZE;
  int const num_groups_per_block = blockDim.x / WARP_SIZE;

  // Each CTA quantizes exactly ONE batch row, selected by blockIdx.x.
  // Callers must launch grid_dim=(BATCH_SIZE, 1, 1) — one block per row.
  // BATCH_SIZE is retained as a template parameter for callers to assert
  // intent, but only blockIdx.x controls which row this CTA processes.
  int const batch_idx = static_cast<int>(blockIdx.x);
  int const row_base = batch_idx * GLOBAL_STRIDE;

#pragma unroll
  for (int group_idx = warp_idx; group_idx < NUM_GROUPS_PER_ROW;
       group_idx += num_groups_per_block) {
    int const group_base_idx = row_base + GROUP_SIZE * group_idx;

    float local_max = eps;
#pragma unroll
    for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
      int const input_idx = group_base_idx + lane_idx + ele_idx * WARP_SIZE;
      float const abs_val = fabsf(static_cast<float>(input[input_idx]));
      local_max = fmaxf(abs_val, local_max);
    }

    float y_scale = 0.0f;
    if constexpr (SCALE_UE8M0) {
      float group_max = group_reduce_max<WARP_SIZE>(local_max);
      group_max = fmaxf(group_max, 1e-10f);
      y_scale = group_max / max_8bit;
      const uint8_t scale_quant =
          __shfl_sync(0xffffffff, encode_ue8m0(y_scale), 0, WARP_SIZE);
      y_scale = exp2f(static_cast<float>(scale_quant) - 127.0f);
      if (lane_idx == 0) {
        packed_scale_bytes[group_idx] = scale_quant;
      }
    } else {
      float group_max = group_reduce_max<WARP_SIZE>(local_max);
      group_max = fmaxf(group_max, 1e-10f);
      y_scale = group_max / max_8bit;
      if (lane_idx == 0) {
        // float32 scale is stored as [batch, num_groups] row-major.
        output_s[batch_idx * NUM_GROUPS_PER_ROW + group_idx] =
            static_cast<SCALE_PACKED_T>(y_scale);
      }
    }

#pragma unroll
    for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
      int const output_idx = group_base_idx + lane_idx + ele_idx * WARP_SIZE;
      float const orig_val = static_cast<float>(input[output_idx]);
      float const quant_val =
          fminf(fmaxf(orig_val / y_scale, min_8bit), max_8bit);
      output_q[output_idx] = __nv_fp8_e4m3(quant_val);
    }
  }

  if constexpr (SCALE_UE8M0) {
    // Ensure every warp finished writing its byte into packed_scale_bytes
    // before any thread packs four bytes into a uint32 below.
    __syncthreads();
    // UE8M0 scale layout: column-major [packed_k, aligned_batch].
    // This row's packed scales go in column `batch_idx`.
#pragma unroll
    for (int packed_idx = thread_idx; packed_idx < PACKED_SCALE_K;
         packed_idx += blockDim.x) {
      uint32_t packed_scale = 0;
#pragma unroll
      for (int pack_idx = 0; pack_idx < SCALE_ALIGNMENT; ++pack_idx) {
        int const group_idx = packed_idx * SCALE_ALIGNMENT + pack_idx;
        const uint8_t encoded =
            group_idx < NUM_GROUPS_PER_ROW ? packed_scale_bytes[group_idx] : 0;
        packed_scale |= static_cast<uint32_t>(encoded) << (pack_idx * 8);
      }
      output_s[packed_idx * scale_outer_stride + batch_idx] =
          static_cast<SCALE_PACKED_T>(packed_scale);
    }
  }
}

} // namespace kernel
