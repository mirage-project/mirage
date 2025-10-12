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
#include "tasks/common/utils.cuh"

#include <cutlass/arch/barrier.h>

namespace kernel {
template <typename T>
__device__ __forceinline__ void warp_reduce_max_idx_sm100(T &val,
                                                          long long &idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float tmp = __shfl_down_sync(0xffffffff, (float)val, offset);
    T other_val = (T)tmp;
    long long other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <typename T>
__device__ __forceinline__ void block_reduce_max_idx_sm100(T &val,
                                                           long long &idx) {
  // Align the shared memory to 128 bytes
  extern __shared__ char smem[];
  long long *smem_idxs =
      (long long *)((reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
  T *smem_vals = reinterpret_cast<T *>(smem_idxs + 32); // max 32 warps

  cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, /*bar-id*/ 6);

  warp_reduce_max_idx_sm100(val, idx);

  int my_lane_id = lane_id();
  int my_warp_id = warp_id();

  if (my_lane_id == 0) {
    smem_vals[my_warp_id] = val;
    smem_idxs[my_warp_id] = idx;
  }

  wg_barrier.arrive_and_wait();

  // Only thread 0 holds the final result
  if (my_warp_id == 0) {
    T block_max_val = T(-inf);
    long long block_max_idx = -1;

    int num_warps = (NUM_THREADS + 31) >> log2_constexpr(NUM_THREADS_PER_WARP);
    if (my_lane_id < num_warps) {
      block_max_val = smem_vals[my_lane_id];
      block_max_idx = smem_idxs[my_lane_id];
    }
    warp_reduce_max_idx_sm100(block_max_val, block_max_idx);

    if (my_lane_id == 0) {
      val = block_max_val;
      idx = block_max_idx;
    }
  }
}

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    argmax_partial_sm100_kernel(void const *__restrict__ input_ptr,
                                void *__restrict__ output_val_ptr,
                                void *__restrict__ output_idx_ptr,
                                int num_active_tokens) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  T *__restrict__ output_val = static_cast<T *>(output_val_ptr);
  long long *__restrict__ output_idx = static_cast<long long *>(output_idx_ptr);

  int tidx = threadIdx.x;

  if (tidx < NUM_THREADS) {

// TODO: try vectorize
#pragma unroll
    for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
      T local_max = T(-inf);
      long long local_idx = -1;
#pragma unroll
      for (int i = tidx; i < CHUNK_SIZE; i += NUM_THREADS) {
        T val = input[i + batch_idx * CHUNK_SIZE * NUM_PARTIAL_TASKS];
        if (val > local_max) {
          local_max = val;
          local_idx = i;
        }
      }

      block_reduce_max_idx_sm100<T>(local_max, local_idx);

      if (tidx == 0) {
        output_val[batch_idx * NUM_PARTIAL_TASKS] = local_max;
        output_idx[batch_idx * NUM_PARTIAL_TASKS] = local_idx;
      }
    }
  }
}

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    argmax_reduce_sm100_kernel(void const *__restrict__ input_val_ptr,
                               void const *__restrict__ input_idx_ptr,
                               void *__restrict__ final_output_ptr,
                               int num_active_tokens) {
  // int step,
  // long long *tokens) {
  T const *__restrict__ partial_vals = static_cast<T const *>(input_val_ptr);
  long long const *__restrict__ partial_idxs =
      static_cast<long long const *>(input_idx_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);

  int tidx = threadIdx.x;
  if (tidx < NUM_THREADS) {
// TODO: try vectorize
#pragma unroll
    for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
      T local_max = T(-inf);
      // Pack (chunk_index, relative_index) into a single 64-bit integer
      long long local_packed_idx = -1;

#pragma unroll
      for (int i = tidx; i < NUM_PARTIAL_TASKS; i += NUM_THREADS) {
        T current_val = partial_vals[i + batch_idx * NUM_PARTIAL_TASKS];
        if (current_val > local_max) {
          local_max = current_val;
          // Higher 32 bits for chunk_index (i), lower 32 for relative_index
          local_packed_idx = ((long long)i << 32) |
                             partial_idxs[i + batch_idx * NUM_PARTIAL_TASKS];
        }
      }

      block_reduce_max_idx_sm100(local_max, local_packed_idx);

      if (tidx == 0) {
        if (local_packed_idx != -1) {
          long long winning_chunk_idx = local_packed_idx >> 32;
          long long winning_relative_idx = local_packed_idx & 0xFFFFFFFF;
          final_output[batch_idx] =
              winning_chunk_idx * CHUNK_SIZE + winning_relative_idx;
          // tokens[step + 1] = winning_chunk_idx * CHUNK_SIZE +
          // winning_relative_idx;
        } else {
          final_output[batch_idx] = -1;
          // tokens[step + 1] = -1;
        }
      }
    }
  }
}

} // namespace kernel
