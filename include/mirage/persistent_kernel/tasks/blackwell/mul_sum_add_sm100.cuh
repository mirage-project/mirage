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
#include "tasks/common/common_header.cuh"
namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int NUM_TOPK,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void
    mul_sum_add_sm100_task_impl(void const *input_ptr,
                                void const *weight_ptr,
                                void const *residual_ptr,
                                void *output_ptr) {
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  float const *__restrict__ d_weight = static_cast<float const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  for (int row_idx = 0; row_idx < BATCH_SIZE; ++row_idx) {
    for (int i = threadIdx.x; i < OUTPUT_SIZE; i += blockDim.x) {
      T res_val = d_residual[row_idx * OUTPUT_STRIDE + i];
      float sum_val = float(res_val);
#pragma unroll
      for (int topk_idx = 0; topk_idx < NUM_TOPK; ++topk_idx) {
        T val = d_input[row_idx * OUTPUT_STRIDE * NUM_TOPK +
                        topk_idx * OUTPUT_STRIDE + i];
        float weight = d_weight[row_idx * NUM_TOPK + topk_idx];
        sum_val += float(val) * weight;
      }
      d_output[row_idx * OUTPUT_STRIDE + i] = T(sum_val);
    }
  }
} // mul_sum_add_sm100_task_impl

} // namespace kernel
