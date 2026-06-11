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
//   2. concat_kernel            — concat N H-dim tensors → N*H along dim 1
//                                 (N=3 for fc/eh_proj, N=2 for draft QKV input)
//   3. eagle3_d2t_remap_kernel  — hot vocab id → target vocab id via d2t table
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

// --- Tensor Concatenation along dim 1 ---
// Concatenates N (BATCH_SIZE, HIDDEN_DIM) tensors along dim 1, producing
// (BATCH_SIZE, N * HIDDEN_DIM).
//
// Layout: output[b, k*H .. (k+1)*H) = inputs[k][b, :]   for k in [0, N)
//
// Generic helper (not Eagle3-specific). Current Eagle3 uses:
//   - N=3: combine aux h_low / h_mid / h_high before `eh_proj` (fc) (3H → H).
//   - N=2: concat(embed, hidden) for the draft attention QKV input (2H),
//          matching sglang's torch.cat([embeds, hidden_states], dim=-1).
template <typename T, int BATCH_SIZE, int HIDDEN_DIM, int N>
__device__ __forceinline__ void
    concat_kernel(void const *const *__restrict__ input_ptrs,
                  void *__restrict__ output_ptr) {
  T *__restrict__ output = static_cast<T *>(output_ptr);

  int const total = BATCH_SIZE * HIDDEN_DIM;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  for (int i = tid; i < total; i += stride) {
    int b = i / HIDDEN_DIM;
    int d = i % HIDDEN_DIM;
    int out_base = b * N * HIDDEN_DIM;
#pragma unroll
    for (int k = 0; k < N; k++) {
      T const *__restrict__ in = static_cast<T const *>(input_ptrs[k]);
      output[out_base + k * HIDDEN_DIM + d] = in[i];
    }
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
  long long *__restrict__ target = static_cast<long long *>(target_token_ptr);

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
//   tokens_buffer    [MAX_REQ, MAX_SEQ_LEN] int64 — full seq buffer
//   argmax_out       [BATCH_SIZE]           int64 — target argmax (K+1)
//   draft_tokens_new [BATCH_SIZE, K]        int64 — this iter's draft (next in)
//   accepted_count   [BATCH_SIZE]           int32 — from verify_strict
//   step             [MAX_REQ]              int32 — current step
//   prompt_length    [MAX_REQ]              int32 — req's prompt length
// Outputs:
//   new_token_nums   [MAX_REQ]              int32 — accepted_count for runtime
//   drafts_prev      [MAX_REQ, K]           int64 — attach_input snapshot
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
                         void *__restrict__ accept_hist_ptr,
                         int request_id) {
  // Single-edge per (producer, consumer) pair design:
  //   argmax_out (from argmax_reduce)             : 1 edge
  //   draft_tokens_new (from mtp_token_scatter)   : 1 edge
  //   accepted_count (from mtp_verify_strict)     : 1 edge
  //   tokens_buffer / new_token_nums / drafts_prev: attach_input (no edges)
  long long *__restrict__ tokens = static_cast<long long *>(tokens_buffer_ptr);
  long long const *__restrict__ argmax =
      static_cast<long long const *>(argmax_out_ptr);
  long long const *__restrict__ drafts =
      static_cast<long long const *>(draft_tokens_new_ptr);
  int const *__restrict__ accepted_count =
      static_cast<int const *>(accepted_count_ptr);
  int const *__restrict__ step = static_cast<int const *>(step_ptr);
  int const *__restrict__ prompt_length =
      static_cast<int const *>(prompt_length_ptr);
  int *__restrict__ new_token_nums = static_cast<int *>(new_token_nums_ptr);
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

  // Select the source slot in all_draft_ids for the new K-token chain.
  //
  // K=1 produces mbt parallel chains (default attention processes all mbt
  // input slots, each predicting one token under a different "ac assumption").
  // Slot b's prediction is for position step_N + b + 2, so commit picks
  // src_slot = ac - 1 to match the next iter's expected position.
  //
  // K>1 instead runs a sequential per-iter loop with Q_LEN_OVERRIDE=1, so the
  // draft kernel writes only slot 0 of attn_out (see attention_sm100.cuh
  // output-write loop bound by `num_tokens * NUM_QO_PER_KV * HEAD_DIM`).
  // Slots 1..mbt-1 propagate stale/garbage values through the downstream
  // rmsnorm → MLP → argmax → d2t_remap pipeline (each grid_dim=(mbt, ...))
  // and into all_draft_ids[1..mbt-1, :]. Picking those rows commits garbage
  // for next iter and collapses K>1 accept rate. Until a proper mbt-parallel
  // K>1 chain is implemented, force slot 0. ac>=2 cases still write the
  // chain at positions misaligned by (ac-1), but slot 0's chain is at least
  // a valid prediction.
  int src_slot = (K > 1) ? 0 : (ac - 1);
  if (src_slot < 0) {
    src_slot = 0;
  }
  if (src_slot >= BATCH_SIZE) {
    src_slot = BATCH_SIZE - 1;
  }

  // Write K new drafts at step+ac+1 .. step+ac+K (only past prompt), drawn
  // from the slot ac-1 chain.
  if (t_id < K) {
    int pos = cur_step + ac + 1 + t_id;
    if (pos < MAX_SEQ_LEN && pos >= prompt_len) {
      tokens[req * MAX_SEQ_LEN + pos] = drafts[src_slot * K + t_id];
    }
  }

  // Snapshot current iter's drafts (shape [BATCH_SIZE=K+1, K]) into the
  // attach_input slot for next iter's verify_strict to consume. The next iter's
  // verify reads drafts_prev[0..K-1] (the first K entries).
  // BUT FIRST: record the OLD drafts_prev (what THIS iter's verify just
  // compared against argmax) into the trace, so we can byte-compare what was
  // verified vs what the target predicted.
  long long old_drafts_prev[8]; // K <= 8 (well bounded for spec decode)
  if (t_id == 0) {
    for (int i = 0; i < K; i++) {
      old_drafts_prev[i] = drafts_prev[i];
    }
  }
  if (t_id < K) {
    drafts_prev[t_id] = drafts[src_slot * K + t_id];
  }

  // Debug-instrumentation: atomically increment accept-rate histogram bin
  // for this iter's `ac`. `ac` lies in [1, K+1] (verify_strict guarantees);
  // bin 0 is reserved for "iters that ran the commit kernel" sanity counter.
  if (t_id == 0 && accept_hist_ptr != nullptr) {
    int *hist = static_cast<int *>(accept_hist_ptr);
    atomicAdd(&hist[ac], 1);
    // Tail trace: layout = [hist 0..K+1, trace_counter at K+2,
    // then per-iter records of 2K+2 ints: (ac, argmax[0..K],
    // drafts_prev[0..K-1])]. Capacity is host-allocated; we cap captured iters
    // at 16 here.
    int const TRACE_COUNTER_OFFSET = K + 2;
    int const RECORD_SIZE = 2 * K + 2;
    int const MAX_TRACE_ITERS = 16;
    int trace_idx = atomicAdd(&hist[TRACE_COUNTER_OFFSET], 1);
    if (trace_idx < MAX_TRACE_ITERS) {
      int base = TRACE_COUNTER_OFFSET + 1 + trace_idx * RECORD_SIZE;
      hist[base + 0] = ac;
      for (int i = 0; i < K + 1; i++) {
        hist[base + 1 + i] = (int)(argmax[i] & 0xFFFFFFFF);
      }
      for (int i = 0; i < K; i++) {
        hist[base + 1 + (K + 1) + i] = (int)(old_drafts_prev[i] & 0xFFFFFFFF);
      }
    }
  }

  // Publish step-advance signal to runtime.
  if (t_id == 0) {
    new_token_nums[req] = ac;
  }
}

} // namespace kernel
