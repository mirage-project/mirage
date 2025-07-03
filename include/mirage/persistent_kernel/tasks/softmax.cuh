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

// Warp-level reduction for max value
template <typename T>
__device__ __forceinline__ void warp_reduce_max(T &val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    T other_val = __shfl_down_sync(0xffffffff, val, offset);
    val = fmaxf(val, other_val);
  }
}

// Warp-level reduction for sum
template <typename T>
__device__ __forceinline__ void warp_reduce_sum(T &val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
}

// Block-level reduction for max
template <typename T>
__device__ __forceinline__ void block_reduce_max(T &val) {
  __shared__ T smem[32]; // max 32 warps
  
  warp_reduce_max(val);
  
  int my_lane_id = lane_id();
  int my_warp_id = warp_id();
  
  if (my_lane_id == 0) {
    smem[my_warp_id] = val;
  }
  
  __syncthreads();
  
  if (my_warp_id == 0) {
    T block_max = T(-INFINITY);
    int num_warps = (blockDim.x + 31) >> 5;
    if (my_lane_id < num_warps) {
      block_max = smem[my_lane_id];
    }
    warp_reduce_max(block_max);
    
    if (my_lane_id == 0) {
      val = block_max;
    }
  }
}

// Block-level reduction for sum
template <typename T>
__device__ __forceinline__ void block_reduce_sum(T &val) {
  __shared__ T smem[32]; // max 32 warps
  
  warp_reduce_sum(val);
  
  int my_lane_id = lane_id();
  int my_warp_id = warp_id();
  
  if (my_lane_id == 0) {
    smem[my_warp_id] = val;
  }
  
  __syncthreads();
  
  if (my_warp_id == 0) {
    T block_sum = 0.0f;
    int num_warps = (blockDim.x + 31) >> 5;
    if (my_lane_id < num_warps) {
      block_sum = smem[my_lane_id];
    }
    warp_reduce_sum(block_sum);
    
    if (my_lane_id == 0) {
      val = block_sum;
    }
  }
}

// Softmax kernel for a single batch
// Computes softmax over the last dimension (vocab_size)
template <typename T, int CHUNK_SIZE>
__device__ __forceinline__ void
    softmax_kernel(void const *__restrict__ input_ptr,
                   void *__restrict__ output_ptr,
                   float temperature = 1.0f) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);
  
  int tidx = threadIdx.x;
  
  // Step 1: Find max value for numerical stability
  float local_max = -INFINITY;
  for (int i = tidx; i < CHUNK_SIZE; i += blockDim.x) {
    float val = float(input[i]) / temperature;
    local_max = fmaxf(local_max, val);
  }
  
  block_reduce_max(local_max);
  __shared__ float shared_max;
  if (tidx == 0) {
    shared_max = local_max;
  }
  __syncthreads();
  float max_val = shared_max;
  
  // Step 2: Compute exp(x - max) and sum
  float local_sum = 0.0f;
  for (int i = tidx; i < CHUNK_SIZE; i += blockDim.x) {
    float val = float(input[i]) / temperature;
    float exp_val = expf(val - max_val);
    output[i] = T(exp_val); // Store exp values temporarily
    local_sum += exp_val;
  }
  
  block_reduce_sum(local_sum);
  __shared__ float shared_sum;
  if (tidx == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();
  float sum_val = shared_sum;
  
  // Step 3: Normalize by sum
  float inv_sum = 1.0f / sum_val;
  for (int i = tidx; i < CHUNK_SIZE; i += blockDim.x) {
    float normalized = float(output[i]) * inv_sum;
    output[i] = T(normalized);
  }
}


} // namespace kernel