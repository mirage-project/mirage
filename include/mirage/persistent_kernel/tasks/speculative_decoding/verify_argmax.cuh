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

template <typename T, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    verify_argmax_reduce_kernel(void const *__restrict__ input_val_ptr,
                         void const *__restrict__ input_idx_ptr,
                         void *__restrict__ final_output_ptr,
                         int step,
                         long long *tokens) {
  T const *__restrict__ partial_vals = static_cast<T const *>(input_val_ptr);
  long long const *__restrict__ partial_idxs =
      static_cast<long long const *>(input_idx_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);

  int tidx = threadIdx.x;
  T local_max = T(-inf);
  // Pack (chunk_index, relative_index) into a single 64-bit integer
  long long local_packed_idx = -1;

  for (int i = tidx; i < NUM_PARTIAL_TASKS; i += blockDim.x) {
    T current_val = partial_vals[i];
    if (current_val > local_max) {
      local_max = current_val;
      // Higher 32 bits for chunk_index (i), lower 32 for relative_index
      local_packed_idx = ((long long)i << 32) | partial_idxs[i];
    }
  }

  block_reduce_max_idx(local_max, local_packed_idx);

  if (tidx == 0) {
    if (local_packed_idx != -1) {
      long long winning_chunk_idx = local_packed_idx >> 32;
      long long winning_relative_idx = local_packed_idx & 0xFFFFFFFF;
      final_output[0] = winning_chunk_idx * CHUNK_SIZE + winning_relative_idx;
      tokens[step + 1] = winning_chunk_idx * CHUNK_SIZE + winning_relative_idx;
    } else {
      final_output[0] = -1;
      tokens[step + 1] = -1;
    }
  }
}

} // namespace kernel
