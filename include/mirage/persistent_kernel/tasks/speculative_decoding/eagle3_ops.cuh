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
// Padding guard: when lm_head is row-padded (to satisfy TMA 16B alignment),
// argmax can land in the padded range [DRAFT_VOCAB_REAL, padded). d2t only
// has DRAFT_VOCAB_REAL entries, so an OOB read would write garbage. When
// hot_id is out of range, emit 0 — verify will reject (won't match target
// argmax) and the bonus token is committed normally.
//
// Inputs:
//   hot_token: [BATCH_SIZE] int64    — argmax over draft logits
//   d2t:      [DRAFT_VOCAB_REAL] int64
// Outputs:
//   target_token: [BATCH_SIZE] int64 — target vocab id for downstream tasks
template <int BATCH_SIZE, int DRAFT_VOCAB_REAL>
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
    if (hot_id >= 0 && hot_id < (long long)DRAFT_VOCAB_REAL) {
      target[b] = hot_id + d2t[hot_id];
    } else {
      target[b] = 0; // padded-row argmax → sentinel; verify will reject
    }
  }
}

// --- Eagle3 Commit (verify-aware token buffer write + step-advance signal +
//                    cross-iter draft snapshot) ---
//
// Replaces mtp_prepare_verify + mtp_accept_commit for Eagle3's K=1+ flow.
// Runs once at end of iter, handling four responsibilities atomically:
//
//  1. Write the verified prefix (accepted drafts + bonus) into the tokens
//     buffer at positions [step+1 .. step+accepted_count], guarded against
//     overwriting the prompt (`pos >= prompt_len`).
//  2. Write the K new draft tokens (for next iter's input) at positions
//     [step+accepted_count+1 .. step+accepted_count+K], same guard.
//  3. Publish `accepted_count` to `new_token_nums[req]` so the OFFLINE
//     runtime's prepare_next_batch can advance step by accept_count past
//     prefill (gated by MPK_SPEC_DECODE).
//  4. Copy `draft_tokens_new` into `drafts_prev_attached` (an attach_input
//     tensor not tracked as a graph edge). Next iter's verify_strict reads
//     `drafts_prev_attached`, which still holds iter N's value when iter N+1
//     runs — solving the cross-iter "verify needs prev iter's drafts" problem
//     without violating the unique-edge-per-pair MPK invariant.
//
// `accepted_count` here is verify_strict's output = final_accepted + 1,
// so it lies in [1, K+1].
//
// Inputs:
//   tokens_buffer       [MAX_REQUESTS, MAX_SEQ_LEN] int64 — full seq buffer
//   argmax_out          [BATCH_SIZE]                int64 — target argmax (K+1)
//   draft_tokens_new    [BATCH_SIZE, K]             int64 — this iter's draft (next iter input)
//   accepted_count      [BATCH_SIZE]                int32 — from verify_strict
//   step                [MAX_REQUESTS]              int32 — current step
//   prompt_length       [MAX_REQUESTS]              int32 — req's prompt length
// Outputs:
//   new_token_nums      [MAX_REQUESTS]              int32 — accepted_count (for runtime)
//   drafts_prev         [MAX_REQUESTS, K]           int64 — attach_input snapshot
template <int K, int BATCH_SIZE, int MAX_SEQ_LEN>
__device__ __forceinline__ void
    eagle3_commit_kernel(void *__restrict__ tokens_buffer_ptr,
                         void const *__restrict__ argmax_out_ptr,
                         void const *__restrict__ draft_tokens_new_ptr,
                         void const *__restrict__ accepted_count_ptr,
                         void const *__restrict__ step_ptr,
                         void const *__restrict__ prompt_length_ptr,
                         void *__restrict__ new_token_nums_ptr,
                         void *__restrict__ drafts_prev_ptr,
                         void *__restrict__ debug_stats_ptr,
                         int request_id) {
  // Single-edge per (producer, consumer) pair design:
  //   argmax_out (from argmax_reduce)             : 1 edge
  //   draft_tokens_new (from mtp_token_scatter)   : 1 edge
  //   accepted_count (from mtp_verify_strict)     : 1 edge
  //   tokens_buffer / new_token_nums / drafts_prev: attach_input (no edges)
  long long *__restrict__ tokens =
      static_cast<long long *>(tokens_buffer_ptr);
  long long const *__restrict__ argmax =
      static_cast<long long const *>(argmax_out_ptr);
  long long const *__restrict__ drafts =
      static_cast<long long const *>(draft_tokens_new_ptr);
  int const *__restrict__ accepted_count =
      static_cast<int const *>(accepted_count_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int const *__restrict__ prompt_length =
      static_cast<int const *>(prompt_length_ptr);
  int *__restrict__ new_token_nums =
      static_cast<int *>(new_token_nums_ptr);
  long long *__restrict__ drafts_prev =
      static_cast<long long *>(drafts_prev_ptr);

  int t_id = threadIdx.x;
  int req = request_id;

  int cur_step = step[req];
  int prompt_len = prompt_length[req];
  int ac = accepted_count[0];

  // Write verified prefix at step+1 .. step+ac (only past prompt). Values
  // come directly from argmax_out (target's argmax over K+1 positions).
  if (t_id < ac) {
    int pos = cur_step + 1 + t_id;
    if (pos < MAX_SEQ_LEN && pos >= prompt_len) {
      tokens[req * MAX_SEQ_LEN + pos] = argmax[t_id];
    }
  }

  // Write K new drafts at step+ac+1 .. step+ac+K (only past prompt).
  if (t_id < K) {
    int pos = cur_step + ac + 1 + t_id;
    if (pos < MAX_SEQ_LEN && pos >= prompt_len) {
      tokens[req * MAX_SEQ_LEN + pos] = drafts[t_id];
    }
  }

  // Snapshot current iter's drafts (shape [BATCH_SIZE, K]) into attach_input
  // slot for next iter's verify_strict to consume. Copy the whole tensor so
  // verify can index draft_token_ids[bid * K + k] across all q-positions.
  int const total = BATCH_SIZE * K;
  for (int i = t_id; i < total; i += blockDim.x) {
    drafts_prev[i] = drafts[i];
  }

  // Publish step-advance signal to runtime.
  if (t_id == 0) {
    new_token_nums[req] = ac;
  }

  // Debug stats: int32 buffer of size 2 = [iter_count, sum_accepted_drafts].
  // sum_accepted_drafts = sum of (ac - 1) across iters (excludes bonus token).
  if (t_id == 0) {
    int *stats = static_cast<int *>(debug_stats_ptr);
    atomicAdd(&stats[0], 1);
    atomicAdd(&stats[1], ac - 1);
  }
}

} // namespace kernel
