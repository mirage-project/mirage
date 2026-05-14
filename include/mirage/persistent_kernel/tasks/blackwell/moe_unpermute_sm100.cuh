/* Copyright 2025 Mirage Team
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
#include "../common/utils.cuh"
#include "../common/worker_config.h"
namespace kernel {

// ============================================================================
// moe_unpermute_sm100
// ============================================================================
//
// Reverse of moe_permute_sm100. Reads the W2-output in permuted layout
// `(M_TOTAL, HIDDEN)` BF16 + the packed `meta` buffer produced by
// moe_permute_sm100 (which holds permuted_weights + token_to_permuted),
// then accumulates `out[t] = residual[t] +
//                  sum_k(permuted_output[token_to_permuted[t,k]-1]
//                         * permuted_weights[same row])`.
//
// One CTA per token (grid_dim = (MBT, 1, 1)). The token id comes from
// `task_desc->task_metadata.request_id = bid.x` (set by the runtime —
// see runtime.cc near TASK_MOE_UNPERMUTE_SM100).
//
// meta layout (must match moe_permute_sm100):
//   meta[0       : M_TOTAL]                 = permuted_weights (float32
//                                              reinterpret-cast as int32)
//   meta[M_TOTAL : M_TOTAL + MBT * TOPK]    = token_to_permuted (int32,
//                                              1-indexed; 0 = not routed
//                                              locally; zero-init each
//                                              iter upstream)
//

template <int MBT, int TOPK, int HIDDEN, int M_TOTAL, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    moe_unpermute_sm100_task_impl(void const *permuted_output_ptr,
                                  void const *meta_ptr,
                                  void const *residual_ptr,
                                  void *output_ptr,
                                  int my_token) {
  using bf16 = cute::bfloat16_t;
  bf16 const *__restrict__ d_in =
      static_cast<bf16 const *>(permuted_output_ptr);
  int32_t const *__restrict__ d_meta = static_cast<int32_t const *>(meta_ptr);
  float const *__restrict__ d_weights =
      reinterpret_cast<float const *>(d_meta); // [0 : M_TOTAL)
  int32_t const *__restrict__ d_t2p =
      d_meta + M_TOTAL; // [M_TOTAL : M_TOTAL + MBT*TOPK)
  bf16 const *__restrict__ d_res = static_cast<bf16 const *>(residual_ptr);
  bf16 *__restrict__ d_out = static_cast<bf16 *>(output_ptr);

  // Load this token's topk → permuted_row mapping (small, registers OK).
  int rows[TOPK];
  float weights[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    int row_1idx = d_t2p[(size_t)my_token * TOPK + k];
    rows[k] = row_1idx - 1; // -1 if not routed locally (was 0)
    weights[k] = (row_1idx > 0) ? d_weights[row_1idx - 1] : 0.0f;
  }

  // Accumulate per-element: out[i] = residual[i]
  //                                + sum_k(d_in[rows[k]][i] * weights[k]).
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    float acc = float(d_res[(size_t)my_token * OUTPUT_STRIDE + i]);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if (rows[k] >= 0 && weights[k] != 0.0f) {
        float v = float(d_in[(size_t)rows[k] * HIDDEN + i]);
        acc += v * weights[k];
      }
    }
    d_out[(size_t)my_token * OUTPUT_STRIDE + i] = bf16(acc);
  }
}

} // namespace kernel
