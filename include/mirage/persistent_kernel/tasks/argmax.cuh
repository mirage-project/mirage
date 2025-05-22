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
__device__ __forceinline__ void
    warp_argmax(T val, int idx, T &warp_max_val, int &warp_max_idx) {
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
  warp_max_val = val;
  warp_max_idx = idx;
}

template <typename T, int BATCH_SIZE, int VOCAB_SIZE>
__device__ __forceinline__ void
    argmax_kernel(void const *__restrict__ input_ptr,
                  void *__restrict__ output_ptr) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  int *__restrict__ output = static_cast<int *>(output_ptr);

  // assume batch size is 1 for a single task
  int tidx = threadIdx.x;
  int warp_idx = warp_id();
  T local_max = T(-inf);
  int local_idx = -1;
  __shared__ T warp_max_vals[4];
  __shared__ int warp_max_idxs[4];

  for (int i = tidx; i < VOCAB_SIZE; i += blockDim.x) {
    T val = input[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
  }
  T warp_max_val;
  int warp_max_idx;
  warp_argmax(local_max, local_idx, warp_max_val, warp_max_idx);

  __syncthreads();

  if ((tidx % 32) == 0) {
    warp_max_vals[warp_idx] = warp_max_val;
    warp_max_idxs[warp_idx] = warp_max_idx;
  }

  T final_max_val = T(-inf);
  int final_max_idx = -1;

  if (warp_idx == 0 && tidx < 32) {
    if (tidx < 4) {
      final_max_val = warp_max_vals[tidx];
      final_max_idx = warp_max_idxs[tidx];
    }
    warp_argmax(final_max_val, final_max_idx, warp_max_val, warp_max_idx);

    if (tidx == 0) {
      output[0] = warp_max_idx;
    }
  }
}

} // namespace kernel
