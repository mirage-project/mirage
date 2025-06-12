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
namespace kernel {

template <typename T>
__device__ __forceinline__ void warp_reduce_max_idx(T &val, int &idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float tmp = __shfl_down_sync(0xffffffff, (float)val, offset);
    T other_val = T(tmp);
    int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <typename T>
__device__ __forceinline__ void block_reduce_max_idx(T &val, int &idx) {
  __shared__ T smem_vals[32]; // max 32 warps
  __shared__ int smem_idxs[32];

  warp_reduce_max_idx(val, idx);

  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  if (lane_id == 0) {
    smem_vals[warp_id] = val;
    smem_idxs[warp_id] = idx;
  }

  __syncthreads();

  T block_max_val = T(-inf);
  int block_max_idx = -1;

  if (warp_id == 0) {
    int num_warps = (blockDim.x + 31) / 32;
    if (lane_id < num_warps) {
      block_max_val = smem_vals[lane_id];
      block_max_idx = smem_idxs[lane_id];
    }
    warp_reduce_max_idx(block_max_val, block_max_idx);
  }

  __syncthreads();

  if (warp_id == 0 && lane_id == 0) {
    smem_vals[0] = block_max_val;
    smem_idxs[0] = block_max_idx;
  }

  __syncthreads();

  val = smem_vals[0];
  idx = smem_idxs[0];
}

template <typename T>
struct PartialMax {
  T val;
  int idx;
};

template <typename T, int VOCAB_SIZE, int NUM_BLOCKS>
__device__ __forceinline__ void argmax_partial_kernel(
    void const *__restrict__ input_ptr,
    void *__restrict__ partial_output_ptr,
    int block_idx) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  PartialMax<T> *__restrict__ partial_output =
      static_cast<PartialMax<T> *>(partial_output_ptr);

  int tidx = threadIdx.x;
  T local_max = T(-inf);
  int local_idx = -1;

  int part_size = (VOCAB_SIZE + NUM_BLOCKS - 1) / NUM_BLOCKS;
  int start_offset = block_idx * part_size;
  int end_offset = min((block_idx + 1) * part_size, VOCAB_SIZE);

  for (int i = start_offset + tidx; i < end_offset; i += blockDim.x) {
    T val = input[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
  }

  block_reduce_max_idx(local_max, local_idx);

  if (tidx == 0) {
    partial_output[block_idx].val = local_max;
    partial_output[block_idx].idx = local_idx;
  }
}

template <typename T, int NUM_BLOCKS>
__device__ __forceinline__ void
argmax_reduce_kernel(void const *__restrict__ partial_input_ptr,
                     void *__restrict__ final_output_ptr) {
  PartialMax<T> const *__restrict__ partial_input =
      static_cast<PartialMax<T> const *>(partial_input_ptr);
  int *__restrict__ final_output = static_cast<int *>(final_output_ptr);

  int tidx = threadIdx.x;
  T local_max = T(-inf);
  int local_idx = -1;

  for (int i = tidx; i < NUM_BLOCKS; i += blockDim.x) {
    if (partial_input[i].val > local_max) {
      local_max = partial_input[i].val;
      local_idx = partial_input[i].idx;
    }
  }

  block_reduce_max_idx(local_max, local_idx);

  if (tidx == 0) {
    final_output[0] = local_idx;
  }
}

} // namespace kernel
