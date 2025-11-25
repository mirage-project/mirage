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
#include "../common/utils.cuh"
#include "../common/worker_config.h"
namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void silu_mul_task_impl_hopper(
    void const *input_ptr, void *output_ptr, int num_active_tokens) {
  if (threadIdx.x >= CONSUMER_NUM_THREADS) {
    return;
  }
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_mul = static_cast<T const *>(input_ptr) + OUTPUT_SIZE;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
#pragma unroll
  for (int i = threadIdx.x; i < num_active_tokens * OUTPUT_SIZE;
       i += CONSUMER_NUM_THREADS) {
    int batch_idx = i / OUTPUT_SIZE;
    int offset = i % OUTPUT_SIZE;
    float input_val = float(d_input[batch_idx * I_STRIDE + offset]);
    T mul_val = d_mul[batch_idx * I_STRIDE + offset];
    d_output[batch_idx * O_STRIDE + offset] =
        T(input_val / (1.0f + expf(-input_val))) * mul_val;
  }
}

} // namespace kernel
