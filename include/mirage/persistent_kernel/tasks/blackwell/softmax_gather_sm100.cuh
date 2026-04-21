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

// Fused softmax + gather: given logits [BATCH_SIZE, VOCAB_SIZE] and
// token_ids [BATCH_SIZE], output prob[batch] =
// softmax(logits[batch])[token_id].
//
// Does NOT materialize the full probability distribution — computes:
//   max_val = max(logits[batch, :])
//   log_sum = log(sum(exp(logits[batch, :] - max_val)))
//   prob = exp(logits[batch, token_id] - max_val - log_sum)
//
// Uses warp-cooperative parallel reduction over the vocab dimension.
// Grid: (BATCH_SIZE, 1, 1), Block: (256, 1, 1).
// Each block handles one batch element.

namespace kernel {

template <typename T, int BATCH_SIZE, int VOCAB_SIZE>
__device__ __forceinline__ void
    softmax_gather_task_impl(void const *__restrict__ logits_ptr,
                             void const *__restrict__ token_ids_ptr,
                             void *__restrict__ output_probs_ptr) {
  T const *__restrict__ logits = static_cast<T const *>(logits_ptr);
  long long const *__restrict__ token_ids =
      static_cast<long long const *>(token_ids_ptr);
  float *__restrict__ output_probs = static_cast<float *>(output_probs_ptr);

  int const tid = threadIdx.x;
  int const num_threads = blockDim.x;

  for (int batch_idx = 0; batch_idx < BATCH_SIZE; ++batch_idx) {
    T const *row = logits + batch_idx * VOCAB_SIZE;
    int const target_id = static_cast<int>(token_ids[batch_idx]);

    // Phase 1: Find max across vocab (parallel reduction)
    float local_max = -1e30f;
    for (int i = tid; i < VOCAB_SIZE; i += num_threads) {
      float val = static_cast<float>(row[i]);
      local_max = fmaxf(local_max, val);
    }
    // Warp shuffle reduce max
    for (int mask = 16; mask > 0; mask >>= 1) {
      local_max =
          fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
    }
    // Cross-warp reduce via shared memory
    __shared__ float smem_max[8]; // max 8 warps
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
      smem_max[warp_id] = local_max;
    }
    __syncthreads();
    if (tid < 8) {
      float wmax = smem_max[tid];
      for (int mask = 4; mask > 0; mask >>= 1) {
        wmax = fmaxf(wmax, __shfl_xor_sync(0xff, wmax, mask));
      }
      smem_max[0] = wmax;
    }
    __syncthreads();
    float const global_max = smem_max[0];

    // Phase 2: Sum exp(logit - max) across vocab
    float local_sum = 0.0f;
    for (int i = tid; i < VOCAB_SIZE; i += num_threads) {
      local_sum += __expf(static_cast<float>(row[i]) - global_max);
    }
    // Warp shuffle reduce sum
    for (int mask = 16; mask > 0; mask >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
    }
    __shared__ float smem_sum[8];
    if (lane_id == 0) {
      smem_sum[warp_id] = local_sum;
    }
    __syncthreads();
    if (tid < 8) {
      float wsum = smem_sum[tid];
      for (int mask = 4; mask > 0; mask >>= 1) {
        wsum += __shfl_xor_sync(0xff, wsum, mask);
      }
      smem_sum[0] = wsum;
    }
    __syncthreads();
    float const global_sum = smem_sum[0];

    // Phase 3: Compute probability at target token
    if (tid == 0) {
      float logit_at_target = (target_id >= 0 && target_id < VOCAB_SIZE)
                                  ? static_cast<float>(row[target_id])
                                  : -1e30f;
      float prob = __expf(logit_at_target - global_max) / global_sum;
      output_probs[batch_idx] = prob;
    }
    __syncthreads();
  }
}

} // namespace kernel
