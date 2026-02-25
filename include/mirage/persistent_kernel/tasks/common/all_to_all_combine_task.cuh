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

// __device__ version: operates on pre-offset pointers for a single token.
// Called from within the persistent kernel's _execute_task (one task per token).
// Threads within the block parallelize over the HIDDEN_DIM vectors.
// For world_size=1, the spin-wait on sync_flags is a no-op.
template <typename T,
          int HIDDEN_DIM,
          int TOPK,
          int WORLD_SIZE,
          bool ADD_RESIDUAL = true>
__device__ void all_to_all_combine_device_impl(
    T const *expert_outputs_row,    // [TOPK, HIDDEN_DIM] - pre-offset to this token
    int const *routing_indices_row, // [TOPK] - pre-offset (reserved for future use)
    T const *routing_weights_row,   // [TOPK] - pre-offset to this token
    T const *residual_row,          // [HIDDEN_DIM] - pre-offset, or nullptr
    T       *output_row,            // [HIDDEN_DIM] - pre-offset to this token
    int const *recv_counts,         // [WORLD_SIZE] - reserved
    int const *recv_offsets,        // [WORLD_SIZE] - reserved
    int num_experts,
    int experts_per_rank,
    int rank,
    volatile int *sync_flags) {     // [WORLD_SIZE] - spin-wait for cross-GPU ordering

  const int tid = threadIdx.x;

  // Phase 1: Wait for all source GPUs (no-op for world_size=1).
  if (tid == 0) {
    for (int src_rank = 0; src_rank < WORLD_SIZE; src_rank++) {
      if (src_rank == rank) continue;
      while (sync_flags[src_rank] == 0) { __threadfence_system(); }
    }
    __threadfence_system();
  }
  __syncthreads();

  // Phase 2: Weighted accumulation — output[h] = sum_k(w_k * expert_k[h])
  // Threads parallelize over HIDDEN_DIM vectors within this single token.
  constexpr int ELEMS_PER_VEC = 16 / sizeof(T);
  constexpr int NUM_VECS = HIDDEN_DIM / ELEMS_PER_VEC;

  for (int v = tid; v < NUM_VECS; v += blockDim.x) {
    float accum[ELEMS_PER_VEC];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_VEC; i++) accum[i] = 0.0f;

    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      float weight = static_cast<float>(routing_weights_row[k]);
      const T* expert_ptr = expert_outputs_row + k * HIDDEN_DIM + v * ELEMS_PER_VEC;
      #pragma unroll
      for (int i = 0; i < ELEMS_PER_VEC; i++) {
        accum[i] += weight * static_cast<float>(expert_ptr[i]);
      }
    }

    T* out_ptr = output_row + v * ELEMS_PER_VEC;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_VEC; i++) {
      out_ptr[i] = static_cast<T>(accum[i]);
    }
  }
  __syncthreads();

  // Phase 3: Add residual — output[h] += residual[h]
  if constexpr (ADD_RESIDUAL) {
    if (residual_row != nullptr) {
      for (int v = tid; v < NUM_VECS; v += blockDim.x) {
        T *out_ptr       = output_row   + v * ELEMS_PER_VEC;
        const T *res_ptr = residual_row + v * ELEMS_PER_VEC;
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

// __global__ wrapper: one block per token range, used for standalone tests.
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

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;
  const int batch_tokens = BATCH_SIZE / num_blocks;
  const int start_token = bid * batch_tokens;
  const int end_token = start_token + batch_tokens;

  // Phase 1: Wait for all source GPUs (only block 0 waits to reduce contention).
  if (tid == 0 && bid == 0) {
    for (int src_rank = 0; src_rank < WORLD_SIZE; src_rank++) {
      if (src_rank == rank) continue;
      while (sync_flags[src_rank] == 0) {}
    }
    __threadfence_system();
  }
  __syncthreads();

  // Phase 2: Weighted accumulation — output[t,h] = sum_k(w_k * expert_k[h])
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

  // Phase 3: Add residual — output[t,h] += residual[t,h]
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
