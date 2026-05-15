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
#include "tasks/common/common_header.cuh"

// Simple element-wise addition: output = input_a + input_b
// Used for residual connections (self.x = self.x + attn_out / mlp_out)
// when the fused with_residual kernels are broken.
//
// Grid: (num_blocks, 1, 1) where num_blocks partitions the output.
// Block: (128, 1, 1) or (256, 1, 1).
// Each task handles BATCH_SIZE * OUTPUT_SIZE elements.

namespace kernel {

// IN_A_STRIDE / IN_A_OFFSET let input_a read a column slice of a wider
// buffer (e.g. when input_a is qkv_a_out and we want only its q_a slice).
// Defaults keep the legacy contiguous (STRIDE-wide) behaviour for input_b
// and the output.
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int STRIDE,
          int IN_A_STRIDE = STRIDE,
          int IN_A_OFFSET = 0>
__device__ __forceinline__ void elementwise_add_task_impl(
    void const *input_a_ptr, void const *input_b_ptr, void *output_ptr) {
  T const *__restrict__ a = static_cast<T const *>(input_a_ptr);
  T const *__restrict__ b = static_cast<T const *>(input_b_ptr);
  T *__restrict__ out = static_cast<T *>(output_ptr);

#pragma unroll
  for (int i = threadIdx.x; i < BATCH_SIZE * OUTPUT_SIZE; i += blockDim.x) {
    int row = i / OUTPUT_SIZE;
    int col = i % OUTPUT_SIZE;
    int idx_a = row * IN_A_STRIDE + IN_A_OFFSET + col;
    int idx = row * STRIDE + col;
    out[idx] = T(float(a[idx_a]) + float(b[idx]));
  }
}

} // namespace kernel
