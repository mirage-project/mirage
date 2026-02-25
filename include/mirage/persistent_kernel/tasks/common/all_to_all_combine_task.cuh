/* Copyright 2023-2024 CMU
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

#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace mirage {
namespace kernel {

// output[t, h] = residual[t, h] + sum_k(routing_weights[t,k] * expert_outputs[t,k,h])
template <typename T,
          int BATCH_SIZE,
          int HIDDEN_DIM,
          int TOPK,
          int WORLD_SIZE,
          bool ADD_RESIDUAL = true>
__global__ void all_to_all_combine_task_impl(
    T const *expert_outputs,         // [BATCH_SIZE, TOPK, HIDDEN_DIM]
    int const *routing_indices,      // [BATCH_SIZE, TOPK]
    T const *routing_weights,        // [BATCH_SIZE, TOPK]
    T const *residual,               // [BATCH_SIZE, HIDDEN_DIM]
    T *output,                       // [BATCH_SIZE, HIDDEN_DIM]
    int const *recv_counts,          // [WORLD_SIZE]
    int const *recv_offsets,         // [WORLD_SIZE]
    int num_experts,
    int experts_per_rank,
    int rank,
    volatile int *sync_flags) {

  using namespace cooperative_groups;
  auto grid = this_grid();
  auto block = this_thread_block();

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // Phase 1: Wait for all source GPUs to finish sending
  if (tid == 0 && bid == 0) {
    for (int src_rank = 0; src_rank < WORLD_SIZE; src_rank++) {
      if (src_rank == rank) continue;
      while (sync_flags[src_rank] == 0) {}
    }
    __threadfence_system();
  }

  __syncthreads();

  // Phase 2: Compute this block's token range
  const int num_blocks = gridDim.x;
  const int batch_tokens = BATCH_SIZE / num_blocks;
  const int start_token = bid * batch_tokens;
  const int end_token = start_token + batch_tokens;

  __syncthreads();

  // Phase 3: Weighted accumulation — output[t,h] = sum_k(w_k * expert_k[h])
  constexpr int ELEMS_PER_VEC = 16 / sizeof(T);
  constexpr int NUM_VECS = HIDDEN_DIM / ELEMS_PER_VEC;

  for (int token_idx = start_token + tid;
       token_idx < end_token;
       token_idx += blockDim.x) {

    #pragma unroll
    for (int v = 0; v < NUM_VECS; v++) {
      float accum[ELEMS_PER_VEC];
      #pragma unroll
      for (int i = 0; i < ELEMS_PER_VEC; i++) {
        accum[i] = 0.0f;
      }

      #pragma unroll
      for (int k = 0; k < TOPK; k++) {
        float weight = static_cast<float>(routing_weights[token_idx * TOPK + k]);
        const T* expert_ptr = expert_outputs +
                              token_idx * TOPK * HIDDEN_DIM +
                              k * HIDDEN_DIM +
                              v * ELEMS_PER_VEC;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_VEC; i++) {
          accum[i] += weight * static_cast<float>(expert_ptr[i]);
        }
      }

      T* out_ptr = output + token_idx * HIDDEN_DIM + v * ELEMS_PER_VEC;
      #pragma unroll
      for (int i = 0; i < ELEMS_PER_VEC; i++) {
        out_ptr[i] = static_cast<T>(accum[i]);
      }
    }
  }

  __syncthreads();

  // Phase 4: Add residual — output[t,h] += residual[t,h]
  if constexpr (ADD_RESIDUAL) {
    for (int token_idx = start_token + tid;
         token_idx < end_token;
         token_idx += blockDim.x) {

      #pragma unroll
      for (int v = 0; v < NUM_VECS; v++) {
        T *out_ptr       = output   + token_idx * HIDDEN_DIM + v * ELEMS_PER_VEC;
        const T *res_ptr = residual + token_idx * HIDDEN_DIM + v * ELEMS_PER_VEC;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_VEC; i++) {
          float val = static_cast<float>(out_ptr[i])
                    + static_cast<float>(res_ptr[i]);
          out_ptr[i] = static_cast<T>(val);
        }
      }
    }
  }
}

} // namespace kernel
} // namespace mirage
