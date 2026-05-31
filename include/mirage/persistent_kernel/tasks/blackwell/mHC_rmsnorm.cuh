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
__device__ __forceinline__ void
    mHC_rmsnorm_task_impl(void const *x_ptr, void *y_ptr, float eps) {
  static_assert(BLOCK_THREADS % 32 == 0, "block size must be a multiple of 32");
  static_assert(HIDDEN % BLOCK_THREADS == 0,
                "HIDDEN must be a multiple of BLOCK_THREADS");
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  static_assert(NUM_WARPS <= 32, "NUM_WARPS must fit in a single warp");
  // Each thread holds ELEMS_PER_THREAD inputs in registers across the two
  // passes (sum-of-squares + multiply-by-rsqrt). Halves x's gmem reads.
  constexpr int ELEMS_PER_THREAD = HIDDEN / BLOCK_THREADS;

  T_in const *__restrict__ x = static_cast<T_in const *>(x_ptr);
  T_out *__restrict__ y = static_cast<T_out *>(y_ptr);

  __shared__ float warp_sums[NUM_WARPS];

  // Pass 1: load + accumulate, keep values in register-resident array.
  float xs[ELEMS_PER_THREAD];
  float local_sum = 0.0f;
#pragma unroll
  for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
    int const i = threadIdx.x + k * BLOCK_THREADS;
    xs[k] = static_cast<float>(x[i]);
    local_sum += xs[k] * xs[k];
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

  // Every warp redundantly reads `warp_sums` and reduces; produces the
  // same global rsqrt in every thread without a second __syncthreads().
  float global_sum = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    global_sum += __shfl_xor_sync(0xffffffff, global_sum, offset);
  }
  float const rsqrt_val = rsqrtf(global_sum / static_cast<float>(HIDDEN) + eps);

  // Pass 2: reuse register-resident xs to compute output. No second load.
#pragma unroll
  for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
    int const i = threadIdx.x + k * BLOCK_THREADS;
    y[i] = static_cast<T_out>(xs[k] * rsqrt_val);
  }
}

} // namespace kernel
