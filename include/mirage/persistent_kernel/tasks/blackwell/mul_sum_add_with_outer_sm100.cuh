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
#include "tasks/common/common_header.cuh"

// mHC K5: HC Post + Residual Fusion.
//
// y[k, c] = post[k] * x[c] + sum_i comb[k, i] * residual[i, c]
//
// Reads: residual[NUM_TOPK, OUTPUT_SIZE], x[OUTPUT_SIZE], post[NUM_TOPK],
//        comb[NUM_TOPK*NUM_TOPK].
// Writes: y[NUM_TOPK, OUTPUT_SIZE].
//
// Per output row k, this matches mul_sum_add but the residual is the rank-1
// outer product post[k] * x[c] (computed inline) rather than a precomputed
// dense buffer; avoids the (n*C) intermediate read/write.

namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int NUM_TOPK,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void
    mul_sum_add_with_outer_sm100_task_impl(void const *residual_ptr,
                                           void const *x_ptr,
                                           void const *comb_ptr,
                                           void const *post_ptr,
                                           void *output_ptr) {
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  float const *__restrict__ d_comb = static_cast<float const *>(comb_ptr);
  float const *__restrict__ d_post = static_cast<float const *>(post_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  for (int row_idx = 0; row_idx < BATCH_SIZE; ++row_idx) {
    for (int k = 0; k < NUM_TOPK; ++k) {
      float post_k = d_post[row_idx * NUM_TOPK + k];
      for (int i = threadIdx.x; i < OUTPUT_SIZE; i += blockDim.x) {
        float x_val = static_cast<float>(d_x[row_idx * OUTPUT_STRIDE + i]);
        float sum_val = post_k * x_val;
#pragma unroll
        for (int t = 0; t < NUM_TOPK; ++t) {
          T r_val = d_residual[row_idx * OUTPUT_STRIDE * NUM_TOPK +
                               t * OUTPUT_STRIDE + i];
          float c = d_comb[row_idx * NUM_TOPK * NUM_TOPK + k * NUM_TOPK + t];
          sum_val += static_cast<float>(r_val) * c;
        }
        d_output[row_idx * OUTPUT_STRIDE * NUM_TOPK + k * OUTPUT_STRIDE + i] =
            T(sum_val);
      }
    }
  }
}

} // namespace kernel
