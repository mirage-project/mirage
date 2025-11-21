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

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    tensor_init_sm100_task_impl(void *input_ptr, float init_val = 0.0f) {
  T *__restrict__ d_input = static_cast<T *>(input_ptr);
  for (int row_idx = 0; row_idx < BATCH_SIZE; ++row_idx) {
    for (int i = threadIdx.x; i < OUTPUT_SIZE; i += blockDim.x) {
      d_input[row_idx * OUTPUT_STRIDE + i] = T(init_val);
    }
  }
} // tensor_init_sm100_task_impl

} // namespace kernel
