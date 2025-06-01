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
                                         float eps) {
  // smem_input: NUM_HEADS, HEAD_DIM
  int warp_idx = warp_id();
#pragma unroll
  for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {
    float sum = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < (HEAD_DIM / 128); i++) {
      sum = smem_input.at(head_idx, threadIdx.x + i * 128) *
            smem_input.at(head_idx, threadIdx.x + i * 128);
    }

#pragma unroll
    for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
      smem_sum[warp_idx] = sum;
    }
    sum = threadIdx.x < NUM_WARPS ? smem_sum[threadIdx.x] : 0.0f;

#pragma unroll
    for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }
    smem_sum[0] = sum;

    float rms_rcp = rsqrt(smem_sum[0] / float(HEAD_DIM) + eps);

    // multiply with weight
#pragma unroll
    for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += 128) {
      float val = smem_input.at(head_idx, i);
      float w = (float)weight_ptr[i];
      val *= sum * w;
      smem_input.at(head_idx, i) = (T)val;
    }
  }
}
} // namespace kernel