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
#include "element_unary.cuh"
#include <climits>

namespace kernel {

// Find the first n-gram in the sequence
template <typename T, int NGRAM_SIZE, int NUM_WORKERS>
static __device__ __forceinline__ void 
      find_ngram_partial_kernel(void const *__restrict__ input_ptr,
                                int *__restrict__ ngram_id_ptr,
                                long long *__restrict__ output_id_ptr,
                                int input_token_num) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  long long *__restrict__ output = output_id_ptr;

  int block_id = blockIdx.x;
  int t_id = threadIdx.x;

  __shared__ int ngram[NGRAM_SIZE];
  __shared__ int input_tokens[NUM_THREADS + NGRAM_SIZE - 1];
  __shared__ int block_min_idx;

  if (t_id == 0) {
    block_min_idx = INT_MAX;
  }
  if (t_id < NGRAM_SIZE) {
    ngram[t_id] = ngram_id_ptr[t_id];
  }

  for (int idx = t_id + block_id * NUM_THREADS; idx < input_token_num; idx += NUM_WORKERS * NUM_THREADS) {
    
    // Load input tokens into shared memory
    input_tokens[t_id] = input[idx];
    if (t_id >= NUM_THREADS_PER_WARP && t_id < NUM_THREADS_PER_WARP + NGRAM_SIZE - 1) {
        input_tokens[NUM_THREADS + t_id - NUM_THREADS_PER_WARP] =
            input[idx + (NUM_THREADS - NUM_THREADS_PER_WARP) + t_id - NUM_THREADS_PER_WARP];
    }
    __syncthreads();

    // Each thread checks if an n-gram starts at its position
    bool is_ngram = true;
    if (idx > input_token_num - NGRAM_SIZE) {
      is_ngram = false;
    } else {
      #pragma unroll
      for (int i = 0; i < NGRAM_SIZE; i++) {
        if (ngram[i] != input_tokens[t_id + i]) {
            is_ngram = false;
            break;
        }
      }
    }

    if (is_ngram) {
      atomicMin(&block_min_idx, idx);
    }
    __syncthreads();
    // Synchronize to make sure all threads see the updated block_min_idx
    // If a thread in this block has already found a match, exit the loop
    if (block_min_idx != INT_MAX) {
      break;
    }
  }

  // After the loop, thread 0 writes the block's result to the global output
  if (t_id == 0) {
    output[0] = block_min_idx;
  }
}

// Find the first n-gram in the sequence
template <int NUM_PARTIAL_TASKS>
static __device__ __forceinline__ void 
find_ngram_global_kernel(long long const *__restrict__ input_array,
                         long long *__restrict__ output_result) {
    
    __shared__ long long block_min_idx_shared;
    
    if (threadIdx.x == 0) {
        block_min_idx_shared = INT_MAX;
    }
    __syncthreads();
    
    // Grid-stride loop for a single block to process the array
    for (int i = threadIdx.x; i < NUM_PARTIAL_TASKS; i += NUM_THREADS) {
        if (input_array[i] < INT_MAX) {
            atomicMin(&block_min_idx_shared, input_array[i]);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        if (block_min_idx_shared != INT_MAX) {
            output_result[0] = block_min_idx_shared;
        } else {
            output_result[0] = -1;
        }
    }
}

template <int NUM_PARTIAL_TASKS>
static __device__ __forceinline__ void 
find_ngram_global_kernel_sequential(long long const *__restrict__ input_array,
                         long long *__restrict__ output_result) {
    for (int i = 0; i < NUM_PARTIAL_TASKS; i++) {
        if (input_array[i] < INT_MAX) {
            output_result[0] = input_array[i];
            return;
        }
    }
    output_result[0] = -1;
}

} // namespace kernel