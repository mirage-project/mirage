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
 #include "copy_sm80.cuh"
 #include "dmem_layout.cuh"
 #include "element_binary.cuh"
 #include "element_unary.cuh"
 #include "mma.cuh"
 #include "reduction.cuh"
 #include "smem_layout.cuh"
 #include "utils.cuh"

template<typename T, typename InputSmem, int NUM_HEAD, int HEAD_DIM=128>
__device__ __forceinline__ void rms_norm(InputSmem smem_input,  InputSmem weight, float * smem_sum, float eps){
//smem_input: NUM_HEADS, HEAD_DIM
#pragma unroll
for (int head_idx = 0; head_idx < num_head; ++head_idx) {
float sum = 0.0f;

#pragma unroll
for(uint32_t i = 0; i < (HEAD_DIM / 128); i ++){
    sum = smem_input.at(head_idx, threadIdx+ i * 128) * smem_input.at(head_idx, threadIdx+ i * 128);
}

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum += math::shfl_xor_sync(sum_sq, offset);
  }
  
  if(threadIdx.x % 32 == 0){
    smem_sum[warp_idx] = sum;
  }
  sum = threadIdx.x < num_warps ? warp_sums[tid] : 0.0f;

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum += math::shfl_xor_sync(sum_sq, offset);
  }
  smem_sum[0] = sum;
  
  float rms_rcp = rsqrt(smem_sum[0] / float(HEAD_DIM) + eps);
  
  //multiply with weight
#pragma unroll
for (uint32_t i = 0; i < HEAD_DIM; i += 128) {
    float val = smem_input.at(head_idx, i);
    float w = weight.at(0, i);
    val *= sum * w;
    smem_input.at(head_idx, i) = val;
}
}

}