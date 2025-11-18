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
          int OUTER_DIM_SIZE,
          int INNER_DIM_SIZE,
          int OUTER_DIM_STRIDE,
          int OUTPUT_SIZE>
__device__ __forceinline__ void identity_task_impl(void const *input_ptr,
                                                   void *output_ptr) {
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

#pragma unroll
  for (int i = threadIdx.x; i < OUTER_DIM_SIZE * OUTPUT_SIZE;
       i += blockDim.x) {
    int outer_dim_idx = i / OUTPUT_SIZE;
    int inner_dim_idx = i % OUTPUT_SIZE;
    d_output[outer_dim_idx * OUTER_DIM_STRIDE + inner_dim_idx] =
        d_input[outer_dim_idx * OUTER_DIM_STRIDE + inner_dim_idx];
  }
}

} // namespace kernel
