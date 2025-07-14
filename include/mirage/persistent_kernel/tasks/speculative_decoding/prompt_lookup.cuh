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
#include "../common.h"
#include <climits>

namespace kernel {
// Find the first n-gram in the sequence
template <int NGRAM_SIZE, int NUM_WORKERS>
static __device__ __forceinline__ void
    find_ngram_partial_kernel(void const *__restrict__ _input_ptr,
                              void *__restrict__ _output_id_ptr,
                              int input_token_num) {
  int t_id = threadIdx.x;
  int task_id = blockIdx.x; // TODO: Is this good?
  if (input_token_num <= NGRAM_SIZE) {
    return;
  }
  long long const *__restrict__ input_ptr =
      static_cast<long long const *>(_input_ptr);
  long long *__restrict__ output_ptr = static_cast<long long *>(_output_id_ptr);
  long long const *__restrict__ ngram_id_ptr =
      input_ptr + input_token_num - NGRAM_SIZE;

  long long *__restrict__ output = output_ptr;

  __shared__ long long ngram[NGRAM_SIZE];
  __shared__ long long input_tokens[NUM_THREADS + NGRAM_SIZE - 1];
  __shared__ long long block_min_idx;

  if (t_id == 0) {
    block_min_idx = LLONG_MAX;
  }
  __syncthreads();
  if (t_id < NGRAM_SIZE) {
    ngram[t_id] = ngram_id_ptr[t_id];
  }

  // Fix: Use a unified loop condition that all threads evaluate the same way
  int total_elements = input_token_num - NGRAM_SIZE;
  int elements_per_iteration = NUM_WORKERS * NUM_THREADS;

  for (int iteration = 0; iteration * elements_per_iteration < total_elements &&
                          block_min_idx == LLONG_MAX;
       iteration++) {
    int idx = t_id + task_id * NUM_THREADS + iteration * elements_per_iteration;

    // Load input tokens into shared memory - only if idx is valid
    if (idx < total_elements) {
      input_tokens[t_id] = input_ptr[idx];
      if (t_id >= NUM_THREADS_PER_WARP &&
          t_id < NUM_THREADS_PER_WARP + NGRAM_SIZE - 1) {
        int load_idx = idx + (NUM_THREADS - NUM_THREADS_PER_WARP) + t_id -
                       NUM_THREADS_PER_WARP;
        if (load_idx < total_elements) {
          input_tokens[NUM_THREADS + t_id - NUM_THREADS_PER_WARP] =
              input_ptr[load_idx];
        }
      }
    }
    __syncthreads();

    // Each thread checks if an n-gram starts at its position
    bool is_ngram = false;
    if (idx < total_elements) {
      is_ngram = true;
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
  }

  // After the loop, thread 0 writes the block's result to the global output
  if (t_id == 0) {
    output[0] = block_min_idx;
  }
}

// Find the first n-gram in the sequence
template <int NGRAM_SIZE, int SPEC_LENGTH, int NUM_PARTIAL_TASKS>
static __device__ __forceinline__ void
    find_ngram_global_kernel(void const *__restrict__ _input_array,
                             void const *__restrict__ _tokens_ptr,
                             void *__restrict__ _output_result,
                             int step) {
  long long const *__restrict__ input_array =
      static_cast<long long const *>(_input_array);
  long long const *__restrict__ tokens_ptr =
      static_cast<long long const *>(_tokens_ptr);
  long long *__restrict__ output_result =
      static_cast<long long *>(_output_result);

  int t_id = threadIdx.x;
  __shared__ long long block_min_idx_shared;

  if (t_id == 0) {
    block_min_idx_shared = LLONG_MAX;
  }
  __syncthreads();

  // Grid-stride loop for a single block to process the array
  for (int i = threadIdx.x; i < NUM_PARTIAL_TASKS; i += NUM_THREADS) {
    if (input_array[i] < LLONG_MAX) {
      atomicMin(&block_min_idx_shared, input_array[i]);
    }
  }

  __syncthreads();
  if (t_id == 32) {
    output_result[0] = tokens_ptr[step];
  } else if (t_id < SPEC_LENGTH) {
    int spec_token_idx = block_min_idx_shared + NGRAM_SIZE + t_id;
    if (block_min_idx_shared != LLONG_MAX && spec_token_idx <= step) {
      output_result[t_id + 1] = tokens_ptr[spec_token_idx];
    } else {
      output_result[t_id + 1] = -1;
    }
  }
}

template <int NUM_PARTIAL_TASKS>
static __device__ __forceinline__ void find_ngram_global_kernel_sequential(
    long long const *__restrict__ input_array,
    long long *__restrict__ output_result) {
  for (int i = 0; i < NUM_PARTIAL_TASKS; i++) {
    if (input_array[i] < LLONG_MAX) {
      output_result[0] = input_array[i];
      return;
    }
  }
  output_result[0] = -1;
}

} // namespace kernel