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
#include "reduction.cuh"
#include <cub/cub.cuh>

namespace kernel {

using bfloat16 = type::bfloat16_t;
using half = __half;

// Multi-token softmax kernel
// Each block processes one token's softmax independently
// Input: [1, num_tokens * vocab_size] - concatenated logits
// Output: [1, num_tokens * vocab_size] - concatenated probabilities
template <typename T, int VOCAB_SIZE, int MAX_TOKENS>
__device__ __forceinline__ void
    multi_token_softmax_kernel(void const *__restrict__ input_ptr,
                              void *__restrict__ output_ptr,
                              int num_tokens,
                              float temperature = 1.0f) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);
  
  // Validate input
  if (num_tokens > MAX_TOKENS) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("ERROR: num_tokens (%d) exceeds MAX_TOKENS (%d)\n", num_tokens, MAX_TOKENS);
    }
    return;
  }
  
  // Each block processes one token
  int token_id = blockIdx.x;
  if (token_id >= num_tokens) return;
  
  // Calculate offsets for this token's data
  int token_offset = token_id * VOCAB_SIZE;
  T const *token_input = input + token_offset;
  T *token_output = output + token_offset;
  
  int tid = threadIdx.x;
  
  // Step 1: Find max value for numerical stability
  float local_max = -INFINITY;
  for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
    float val = float(token_input[i]) / temperature;
    local_max = fmaxf(local_max, val);
  }
  
  // Block-level reduction for max
  __shared__ float smem_max[32]; // max 32 warps per block
  
  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
    local_max = fmaxf(local_max, other_val);
  }
  
  // Write warp results to shared memory
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  if (lane_id == 0) {
    smem_max[warp_id] = local_max;
  }
  
  __syncthreads();
  
  // Final reduction in first warp
  if (warp_id == 0) {
    float block_max = -INFINITY;
    int num_warps = (blockDim.x + 31) / 32;
    if (lane_id < num_warps) {
      block_max = smem_max[lane_id];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xffffffff, block_max, offset);
      block_max = fmaxf(block_max, other_val);
    }
    
    if (lane_id == 0) {
      smem_max[0] = block_max;
    }
  }
  
  __syncthreads();
  float max_val = smem_max[0];
  
  // Step 2: Compute exp(x - max) and sum
  float local_sum = 0.0f;
  for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
    float val = float(token_input[i]) / temperature;
    float exp_val = expf(val - max_val);
    token_output[i] = T(exp_val); // Store exp values temporarily
    local_sum += exp_val;
  }
  
  // Block-level reduction for sum
  __shared__ float smem_sum[32]; // max 32 warps per block
  
  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  
  // Write warp results to shared memory
  if (lane_id == 0) {
    smem_sum[warp_id] = local_sum;
  }
  
  __syncthreads();
  
  // Final reduction in first warp
  if (warp_id == 0) {
    float block_sum = 0.0f;
    int num_warps = (blockDim.x + 31) / 32;
    if (lane_id < num_warps) {
      block_sum = smem_sum[lane_id];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    
    if (lane_id == 0) {
      smem_sum[0] = block_sum;
    }
  }
  
  __syncthreads();
  float sum_val = smem_sum[0];
  
  // Step 3: Normalize by sum
  float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
    float normalized = float(token_output[i]) * inv_sum;
    token_output[i] = T(normalized);
  }
}

} // namespace kernel