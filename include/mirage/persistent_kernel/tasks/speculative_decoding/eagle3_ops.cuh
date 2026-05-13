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
// Eagle3 Operations
//
// Kernels supporting Eagle3 speculative decoding:
//   1. copy_layer_kernel        — capture target's aux hidden states (memcpy)
//   2. eagle3_aux_concat_kernel — concat 3 H-dim tensors → 3H (for fc / eh_proj)
//   3. eagle3_input_concat_kernel — concat(embed, hidden) → 2H (for draft QKV)
//   4. eagle3_d2t_remap_kernel  — hot vocab id → target vocab id via d2t table
// ============================================================================

// --- Generic Memcpy ---
// Copy a contiguous (BATCH_SIZE, HIDDEN_DIM) tensor from src to dst.
// Used by Eagle3 to capture target's intermediate hidden states into dedicated
// aux buffers (since MPK's per-layer intermediates are reused across layers).
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void
    copy_layer_kernel(void const *__restrict__ src_ptr,
                      void *__restrict__ dst_ptr) {
  T const *__restrict__ src = static_cast<T const *>(src_ptr);
  T *__restrict__ dst = static_cast<T *>(dst_ptr);

  int const total = BATCH_SIZE * HIDDEN_DIM;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  for (int i = tid; i < total; i += stride) {
    dst[i] = src[i];
  }
}

// --- Aux Hidden State Concatenation ---
// Concatenates 3 (BATCH_SIZE, HIDDEN_DIM) bf16 tensors along dim 1, producing
// (BATCH_SIZE, 3 * HIDDEN_DIM).
//
// Layout: output[b, 0..H)        = h0[b, :]
//         output[b, H..2H)       = h1[b, :]
//         output[b, 2H..3H)      = h2[b, :]
//
// Used at Eagle3 draft step 0 to combine h_low / h_mid / h_high before the
// `eh_proj` (fc) linear (3H → H).
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void
    eagle3_aux_concat_kernel(void const *__restrict__ h0_ptr,
                             void const *__restrict__ h1_ptr,
                             void const *__restrict__ h2_ptr,
                             void *__restrict__ output_ptr) {
  T const *__restrict__ h0 = static_cast<T const *>(h0_ptr);
  T const *__restrict__ h1 = static_cast<T const *>(h1_ptr);
  T const *__restrict__ h2 = static_cast<T const *>(h2_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);

  int const total = BATCH_SIZE * HIDDEN_DIM;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  for (int i = tid; i < total; i += stride) {
    int b = i / HIDDEN_DIM;
    int d = i % HIDDEN_DIM;
    int out_base = b * 3 * HIDDEN_DIM;
    output[out_base + d] = h0[i];
    output[out_base + HIDDEN_DIM + d] = h1[i];
    output[out_base + 2 * HIDDEN_DIM + d] = h2[i];
  }
}

// --- Eagle3 Attention Input Concatenation ---
// Concatenates token embedding and per-step hidden into the Llama-style draft
// attention QKV input of shape (BATCH_SIZE, 2 * HIDDEN_DIM).
//
// Layout: output[b, 0..H)  = embed[b, :]
//         output[b, H..2H) = hidden[b, :]
//
// The Eagle3 draft model's qkv_proj weight is (out_dim, 2H), so this matches
// sglang's `torch.cat([embeds, hidden_states], dim=-1)` before attention.
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void
    eagle3_input_concat_kernel(void const *__restrict__ embed_ptr,
                               void const *__restrict__ hidden_ptr,
                               void *__restrict__ output_ptr) {
  T const *__restrict__ embed = static_cast<T const *>(embed_ptr);
  T const *__restrict__ hidden = static_cast<T const *>(hidden_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);

  int const total = BATCH_SIZE * HIDDEN_DIM;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  for (int i = tid; i < total; i += stride) {
    int b = i / HIDDEN_DIM;
    int d = i % HIDDEN_DIM;
    int out_base = b * 2 * HIDDEN_DIM;
    output[out_base + d] = embed[i];
    output[out_base + HIDDEN_DIM + d] = hidden[i];
  }
}

// --- Eagle3 d2t Remap (hot vocab → target vocab) ---
// Eagle3 draft head outputs an id in the draft's smaller hot vocabulary
// (draft_vocab_size, typically 32000). The d2t table converts this back to
// the target's full vocab id via:
//
//   target_id = hot_id + d2t[hot_id]
//
// (sglang convention; d2t is signed int64). One thread per batch element.
//
// Inputs:
//   hot_token: [BATCH_SIZE] int64    — argmax over draft logits
//   d2t:      [DRAFT_VOCAB_SIZE] int64
// Outputs:
//   target_token: [BATCH_SIZE] int64 — target vocab id for downstream tasks
template <int BATCH_SIZE>
__device__ __forceinline__ void
    eagle3_d2t_remap_kernel(void const *__restrict__ hot_token_ptr,
                            void const *__restrict__ d2t_table_ptr,
                            void *__restrict__ target_token_ptr) {
  long long const *__restrict__ hot =
      static_cast<long long const *>(hot_token_ptr);
  long long const *__restrict__ d2t =
      static_cast<long long const *>(d2t_table_ptr);
  long long *__restrict__ target =
      static_cast<long long *>(target_token_ptr);

  int b = threadIdx.x;
  if (b < BATCH_SIZE) {
    long long hot_id = hot[b];
    target[b] = hot_id + d2t[hot_id];
  }
}

} // namespace kernel
