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
#include "attention_prep_sm100.cuh" // Q_STAGED_ROWS
#include "tasks/common/common_header.cuh"

namespace kernel {

// ============================================================================
// Decode-attention FINALIZE: copy the valid rows of the padded generated-core
// output into the packed attention output.
//   attn_pad : FOLD-DIM [kv_heads * max_reqs, Q_STAGED_ROWS, HEAD_DIM]
//              (rows >= NUM_QO_PER_KV are pad garbage -- never copied)
//   output   : [tokens, num_q_heads * HEAD_DIM]
// One task per request; grid (reqs, 1, 1).
// ============================================================================
template <typename T,
          int NUM_KV_HEADS,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int O_STRIDE,
          int MAX_REQS> // attn_pad is HEAD-MAJOR: [kvh, max_reqs, 8, hd]
__device__ __forceinline__ void attention_finalize_sm100_impl(
    void const *attn_pad_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int16_t request_id) {
  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  constexpr int VALID = NUM_KV_HEADS * NUM_QO_PER_KV * HEAD_DIM;
  T const *pad = static_cast<T const *>(attn_pad_ptr);
  // The generated core computes attention for the LAST new token only (see
  // attention_prep); its output row is the one downstream generation reads.
  T *out = static_cast<T *>(output_ptr) +
           (size_t)(last_token_pos - 1) * O_STRIDE;
  for (int idx = threadIdx.x; idx < VALID; idx += NUM_THREADS) {
    int const c = idx % HEAD_DIM;
    int const qh = (idx / HEAD_DIM) % NUM_QO_PER_KV;
    int const kvh = idx / (HEAD_DIM * NUM_QO_PER_KV);
    out[(size_t)(kvh * NUM_QO_PER_KV + qh) * HEAD_DIM + c] =
        pad[(((size_t)kvh * MAX_REQS + request_id) * Q_STAGED_ROWS + qh) * HEAD_DIM + c];
  }
}

} // namespace kernel
