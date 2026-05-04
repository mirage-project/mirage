/* Copyright 2026 CMU
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
#include "tasks/common/common_header.cuh"

// mHC K1 (rmsnorm half): per-token RMSNorm with implicit unit weight.
//   y[t, i] = x[t, i] * rsqrt(mean(x[t, :]**2) + eps)
//
// One block per token. Threads cooperate on the sum-of-squares reduction
// using warp shuffles + a small shared-memory cross-warp combine.

namespace kernel {

template <typename T_in, typename T_out, int HIDDEN, int BLOCK_THREADS>
__device__ __forceinline__ void mHC_rmsnorm_task_impl(void const *x_ptr,
                                                      void *y_ptr,
                                                      float eps) {
  static_assert(BLOCK_THREADS % 32 == 0, "block size must be a multiple of 32");
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;

  T_in const *__restrict__ x = static_cast<T_in const *>(x_ptr);
  T_out *__restrict__ y = static_cast<T_out *>(y_ptr);

  __shared__ float warp_sums[NUM_WARPS];
  __shared__ float rsqrt_shared;

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < HIDDEN; i += BLOCK_THREADS) {
    float v = static_cast<float>(x[i]);
    local_sum += v * v;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
  }

  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = local_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (lane == 0) {
      rsqrt_shared = rsqrtf(v / static_cast<float>(HIDDEN) + eps);
    }
  }
  __syncthreads();

  float const rsqrt_val = rsqrt_shared;
  for (int i = threadIdx.x; i < HIDDEN; i += BLOCK_THREADS) {
    float v = static_cast<float>(x[i]);
    y[i] = static_cast<T_out>(v * rsqrt_val);
  }
}

} // namespace kernel
