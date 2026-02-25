/* Copyright 2023-2025 CMU
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
#include "cutlass/arch/memory_sm80.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace mirage {
namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int HIDDEN_DIM,
          int TOPK,
          int WORLD_SIZE,
          bool USE_NVLINK = true>
__global__ void all_to_all_dispatch_task_impl(
    T const *input_tokens,
    int const *routing_indices,
    T const *routing_weights,
    T *send_buffer,
    T **recv_ptrs,
    int *send_counts,
    int *send_offsets,
    int num_experts,
    int experts_per_rank,
    int rank,
    int node_size,
    int *grid_counter,
    volatile int *sync_flags) {

  using namespace cooperative_groups;
  auto grid = this_grid();

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;

  __shared__ int local_counts[WORLD_SIZE];
  __shared__ int global_offsets[WORLD_SIZE];

  // PHASE 1: COUNT
  if (tid < WORLD_SIZE) {
    local_counts[tid] = 0;
  }
  __syncthreads();

  const int batch_tokens = BATCH_SIZE / num_blocks;
  const int start_token = bid * batch_tokens;
  const int end_token = start_token + batch_tokens;

  for (int token_idx = start_token + tid;
       token_idx < end_token;
       token_idx += blockDim.x) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = routing_indices[token_idx * TOPK + k];
      int dest_rank = expert_id / experts_per_rank;
      atomicAdd(&local_counts[dest_rank], 1);
    }
  }
  __syncthreads();

  // PHASE 2: OFFSET
  if (tid < WORLD_SIZE) {
    atomicAdd(&send_counts[tid], local_counts[tid]);
  }
  grid.sync();

  if (bid == 0 && tid == 0) {
    int cumulative = 0;
    for (int r = 0; r < WORLD_SIZE; r++) {
      send_offsets[r] = cumulative;
      cumulative += send_counts[r];
    }
  }
  grid.sync();

  if (tid < WORLD_SIZE) {
    global_offsets[tid] = send_offsets[tid];
  }
  __syncthreads();

  // PHASE 3: DISPATCH
  for (int token_idx = start_token + tid;
       token_idx < end_token;
       token_idx += blockDim.x) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = routing_indices[token_idx * TOPK + k];
      int dest_rank = expert_id / experts_per_rank;
      int write_pos = atomicAdd(&send_offsets[dest_rank], 1);

      T* dst_ptr;
      bool same_node = (dest_rank / node_size) == (rank / node_size);
      if (same_node && recv_ptrs[dest_rank] != nullptr) {
        dst_ptr = recv_ptrs[dest_rank] + write_pos * HIDDEN_DIM;
      } else {
        dst_ptr = send_buffer + dest_rank * BATCH_SIZE * HIDDEN_DIM + write_pos * HIDDEN_DIM;
      }

      const T* src_ptr = input_tokens + token_idx * HIDDEN_DIM;
      constexpr int ELEMS_PER_VEC = 16 / sizeof(T);
      constexpr int NUM_VECS = HIDDEN_DIM / ELEMS_PER_VEC;

      #pragma unroll
      for (int v = 0; v < NUM_VECS; v++) {
        uint4 data = *reinterpret_cast<const uint4*>(src_ptr + v * ELEMS_PER_VEC);
        *reinterpret_cast<uint4*>(dst_ptr + v * ELEMS_PER_VEC) = data;
      }

      constexpr int REMAINDER = HIDDEN_DIM % ELEMS_PER_VEC;
      if constexpr (REMAINDER > 0) {
        #pragma unroll
        for (int r = 0; r < REMAINDER; r++) {
          dst_ptr[NUM_VECS * ELEMS_PER_VEC + r] = src_ptr[NUM_VECS * ELEMS_PER_VEC + r];
        }
      }
    }
  }
  __syncthreads();

  // PHASE 4: SYNC
  grid.sync();

  if (bid == 0 && tid == 0) {
    __threadfence_system();
    sync_flags[rank] = 1;
  }
}

} // namespace kernel
} // namespace mirage
