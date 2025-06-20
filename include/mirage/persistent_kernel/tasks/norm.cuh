/* Copyright 2025 CMU
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
#include "common.h"
#include "utils.cuh"
namespace kernel {
template <typename T, typename InputSmem, int NUM_HEAD, int HEAD_DIM = 128>
__device__ __forceinline__ void rms_norm(InputSmem smem_input,
                                         T const *weight_ptr,
                                         float *smem_sum,
                                         float eps,
                                         int row_offset = 0,
                                         bool rotary_emd = false,
                                         T const *cos_ptr = nullptr,
                                         T const *sin_ptr = nullptr) {
  // smem_input: NUM_HEADS, HEAD_DIM
  int warp_idx = warp_id();
#pragma unroll
  for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {
    float sum = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < (HEAD_DIM / 128); i++) {
      sum +=
          (float)smem_input.at(head_idx + row_offset, threadIdx.x + i * 128) *
          (float)smem_input.at(head_idx + row_offset, threadIdx.x + i * 128);
    }

#pragma unroll
    for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
      smem_sum[warp_idx] = sum;
    }
    __syncthreads();
    sum = threadIdx.x < NUM_WARPS ? smem_sum[threadIdx.x] : 0.0f;

#pragma unroll
    for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }

    if (threadIdx.x == 0) {
      smem_sum[0] = sum;
    }

    __syncthreads();

    float rms_rcp = rsqrt(smem_sum[0] / float(HEAD_DIM) + eps);

    // multiply with weight
#pragma unroll
    for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += 128) {
      float val = smem_input.at(head_idx + row_offset, i);
      float w = (float)weight_ptr[i];
      val *= rms_rcp * w;
      smem_input.at(head_idx + row_offset, i) = (T)val;

      if (rotary_emd) {
        __syncthreads();
        int offset = (i / HEAD_DIM) * HEAD_DIM + i;
        float cos = (float)cos_ptr[offset];
        float sin = (float)sin_ptr[offset];
        if (i < HEAD_DIM / 2) {
          float v1 = (float)smem_input.at(head_idx + row_offset, i);
          float v2 =
              (float)smem_input.at(head_idx + row_offset, i + HEAD_DIM / 2);
          float v_rot = v1 * cos - v2 * sin;
          smem_input.at(head_idx + row_offset, i) = (T)v_rot;
        } else {
          float v1 = (float)smem_input.at(head_idx + row_offset, i);
          float v2 =
              (float)smem_input.at(head_idx + row_offset, i - HEAD_DIM / 2);
          float v_rot = v1 * cos + v2 * sin;
          smem_input.at(head_idx + row_offset, i) = (T)v_rot;
        }
      }
    }
  }
}
} // namespace kernel