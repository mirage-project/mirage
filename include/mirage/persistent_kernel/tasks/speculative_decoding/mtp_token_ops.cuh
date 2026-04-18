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
#include "tasks/common/common_header.cuh"

namespace kernel {

// ============================================================================
// MTP Token Operations
//
// Small utility kernels for MTP speculative decoding token management.
// ============================================================================

// --- Token Scatter ---
// Copy a single token ID per batch element from src[batch, 1] to a specific
// column of dst[batch, num_draft_tokens].
// Used after each MTP argmax step to collect draft tokens.
//
// Inputs:
//   src: [BATCH_SIZE, 1] int64 — single draft token from argmax
//   dst: [BATCH_SIZE, NUM_SLOTS] int64 — collection buffer
// Params:
//   SLOT_IDX: which column to write (compile-time, from static unroll)
//   NUM_SLOTS: total columns in dst
template <int BATCH_SIZE, int NUM_SLOTS, int SLOT_IDX>
__device__ __forceinline__ void mtp_token_scatter_kernel(
    void const *__restrict__ src_ptr,
    void *__restrict__ dst_ptr) {

  long long const *__restrict__ src =
      static_cast<long long const *>(src_ptr);
  long long *__restrict__ dst = static_cast<long long *>(dst_ptr);

  int b = threadIdx.x;
  if (b < BATCH_SIZE) {
    dst[b * NUM_SLOTS + SLOT_IDX] = src[b];
  }
}

// --- Float Scatter (for draft probabilities) ---
// Same pattern as token scatter but for float32 values.
// Used to accumulate P_draft(token) per draft step.
template <int BATCH_SIZE, int NUM_SLOTS, int SLOT_IDX>
__device__ __forceinline__ void mtp_float_scatter_kernel(
    void const *__restrict__ src_ptr,
    void *__restrict__ dst_ptr) {
  float const *__restrict__ src = static_cast<float const *>(src_ptr);
  float *__restrict__ dst = static_cast<float *>(dst_ptr);
  int b = threadIdx.x;
  if (b < BATCH_SIZE) {
    dst[b * NUM_SLOTS + SLOT_IDX] = src[b];
  }
}

// --- Draft Tokens to Sequence Buffer ---
// After MTP draft generation, copy K draft tokens from all_draft_ids into the
// main token sequence buffer (config.tokens) at the correct positions.
// Also writes the main model's token (bonus/base) at position 0.
//
// This prepares the input for the next iteration's verification forward:
//   tokens[request, step+1] = main_token
//   tokens[request, step+2] = draft_0
//   ...
//   tokens[request, step+K+1] = draft_{K-1}
//
// Inputs:
//   main_token:     [BATCH_SIZE, 1] int64 — main model's argmax output
//   draft_tokens:   [BATCH_SIZE, NUM_DRAFT] int64 — collected draft tokens
//   tokens_buffer:  [MAX_REQUESTS, MAX_SEQ_LEN] int64 — full token sequence
//   step:           [MAX_REQUESTS] int32 — current step per request
// Outputs:
//   num_new_tokens: [MAX_REQUESTS] int32 — set to NUM_DRAFT + 1
template <int NUM_DRAFT, int MAX_SEQ_LEN>
__device__ __forceinline__ void mtp_prepare_verify_input_kernel(
    void const *__restrict__ main_token_ptr,
    void const *__restrict__ draft_tokens_ptr,
    void *__restrict__ tokens_buffer_ptr,
    void const *__restrict__ step_ptr,
    void *__restrict__ num_new_tokens_ptr,
    int request_id) {

  long long const *__restrict__ main_token =
      static_cast<long long const *>(main_token_ptr);
  long long const *__restrict__ draft_tokens =
      static_cast<long long const *>(draft_tokens_ptr);
  long long *__restrict__ tokens =
      static_cast<long long *>(tokens_buffer_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int *__restrict__ num_new_tokens =
      static_cast<int *>(num_new_tokens_ptr);

  int t_id = threadIdx.x;
  // Use task metadata request_id (not blockIdx.x which is worker block ID
  // in persistent kernel)
  int req = request_id;

  int cur_step = step[req];

  // Thread 0: write main token at step+1, set num_new_tokens
  if (t_id == 0) {
    if (cur_step + 1 < MAX_SEQ_LEN) {
      tokens[req * MAX_SEQ_LEN + cur_step + 1] = main_token[req];
    }
    // Clamp num_new_tokens so we don't exceed MAX_SEQ_LEN
    int max_new = MAX_SEQ_LEN - cur_step - 1;
    if (max_new < 0) max_new = 0;
    num_new_tokens[req] = (NUM_DRAFT + 1 < max_new) ? NUM_DRAFT + 1 : max_new;
  }

  // Threads 0..NUM_DRAFT-1: write draft tokens (bounds-checked)
  if (t_id < NUM_DRAFT) {
    int write_pos = cur_step + 2 + t_id;
    if (write_pos < MAX_SEQ_LEN) {
      tokens[req * MAX_SEQ_LEN + write_pos] =
          draft_tokens[req * NUM_DRAFT + t_id];
    }
  }
}

} // namespace kernel
