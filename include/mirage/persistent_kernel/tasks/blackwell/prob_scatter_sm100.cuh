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

// Scatter a per-token probability into a per-position buffer.
// Used for accumulating P_target(input_token) across verify iterations.
//
// Reads the runtime step counter to determine the write position.
// Each iteration writes prob[batch] → buffer[batch, step].
//
// Grid: (batch_size, 1, 1), Block: (1, 1, 1)
// Very lightweight — single scalar write per batch element.

namespace kernel {

template <int BATCH_SIZE, int MAX_POSITIONS>
__device__ __forceinline__ void
    prob_scatter_task_impl(void const *__restrict__ prob_ptr,
                           void *__restrict__ buffer_ptr,
                           int const *__restrict__ step_ptr) {
  float const *__restrict__ prob = static_cast<float const *>(prob_ptr);
  float *__restrict__ buffer = static_cast<float *>(buffer_ptr);

  int const tid = threadIdx.x;
  if (tid > 0) {
    return;
  }

  for (int b = 0; b < BATCH_SIZE; ++b) {
    int const pos = step_ptr[b];
    if (pos >= 0 && pos < MAX_POSITIONS) {
      buffer[b * MAX_POSITIONS + pos] = prob[b];
    }
  }
}

// --- Extract slice from prob buffer at runtime offset ---
// Reads buffer[batch, offset..offset+K-1] into output[batch, 0..K-1].
// Used to extract target_probs for verify positions from the accumulated
// buffer.
template <int BATCH_SIZE, int MAX_POSITIONS, int NUM_EXTRACT>
__device__ __forceinline__ void
    prob_extract_task_impl(void const *__restrict__ buffer_ptr,
                           void *__restrict__ output_ptr,
                           int const *__restrict__ offset_ptr) {
  float const *__restrict__ buffer = static_cast<float const *>(buffer_ptr);
  float *__restrict__ output = static_cast<float *>(output_ptr);

  int const tid = threadIdx.x;
  if (tid >= NUM_EXTRACT) {
    return;
  }

  for (int b = 0; b < BATCH_SIZE; ++b) {
    int const offset = offset_ptr[b] + 1; // +1 because verify starts at step+1
    int const src_pos = offset + tid;
    if (src_pos >= 0 && src_pos < MAX_POSITIONS) {
      output[b * NUM_EXTRACT + tid] = buffer[b * MAX_POSITIONS + src_pos];
    } else {
      output[b * NUM_EXTRACT + tid] = 0.0f;
    }
  }
}

} // namespace kernel
