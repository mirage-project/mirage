/* Copyright 2025 Mirage Team
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

template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void rms_norm_impl(void const *input_ptr,
                                              void const *weight_ptr,
                                              void *output_ptr,
                                              float eps) {
  static_assert(BATCH_SIZE == 1);
  extern __shared__ char smem[];
  float *reduce_smem = reinterpret_cast<float *>(smem);

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  float sum = 0.0f;
#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
    float val = (float)d_input[i];
    sum += val * val;
  }
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum += shfl_xor_sync(sum, offset);
  }
  if (threadIdx.x % 32 == 0) {
    reduce_smem[threadIdx.x / 32] = sum;
  }
  __syncthreads();
  sum = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
  for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
    sum += shfl_xor_sync(sum, offset);
  }
  if (threadIdx.x == 0) {
    reduce_smem[0] = sum;
  }
  __syncthreads();

  float rms_rcp = rsqrt(reduce_smem[0] / float(HIDDEN_DIM) + eps);

#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
    float val = (float)d_input[i];
    float w = (float)d_weight[i];
    val *= rms_rcp * w;
    d_output[i] = (T)val;
  }
}

} // namespace kernel