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

// MLA Reduce Task for Blackwell (SM100)
// Merges partial O and LSE outputs from split-KV MLA decode tasks.
//
// Each thread handles one head. Streams through splits to find
// lse_max + sum_exp, then accumulates scaled partial outputs.

#include <cuda_bf16.h>

namespace kernel {

template <int NUM_HEADS, // e.g. 128
          int D_V>       // e.g. 512
__device__ __noinline__ void mla_reduce_sm100_task_impl(
    float const *Oa, // partial outputs: [B, sk, D_V, NUM_HEADS] (strided)
    float const *La, // partial LSE:     [B, sk, NUM_HEADS]
    nv_bfloat16 *O,  // final output:    [B, NUM_HEADS, D_V]
    int num_splits,  // sk
    int batch_idx,   // which batch element
    int d_start,     // starting D_V index for this task
    int d_count      // number of D_V elements this task handles
) {
  int const tid = threadIdx.x;
  if (tid >= NUM_HEADS) {
    return;
  }

  int const h = tid; // head index

  // Pass 1: find lse_max across all splits
  float lse_max = -1e30f;
  for (int s = 0; s < num_splits; s++) {
    float lse_val = La[(batch_idx * num_splits + s) * NUM_HEADS + h];
    lse_max = fmaxf(lse_max, lse_val);
  }

  // Pass 2: compute sum_exp
  float sum_exp = 0.0f;
  for (int s = 0; s < num_splits; s++) {
    float lse_val = La[(batch_idx * num_splits + s) * NUM_HEADS + h];
    sum_exp += __expf(lse_val - lse_max);
  }
  float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

  // Pass 3: accumulate weighted O for each d dimension
  for (int di = d_start; di < d_start + d_count; di++) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; s++) {
      float lse_val = La[(batch_idx * num_splits + s) * NUM_HEADS + h];
      float scale = __expf(lse_val - lse_max) * inv_sum;
      // Oa layout: [batch*sk, D_V, NUM_HEADS] — from mla_decode output
      float oval = Oa[(batch_idx * num_splits + s) * D_V * NUM_HEADS +
                      di * NUM_HEADS + h];
      acc += scale * oval;
    }
    // O layout: [B, NUM_HEADS, D_V]
    O[(batch_idx * NUM_HEADS + h) * D_V + di] = __float2bfloat16(acc);
  }
}

} // namespace kernel
