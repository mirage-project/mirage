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

namespace kernel {

// The generated core's Q operand is padded to this many rows per kv head:
// swapAB puts the token count into the MMA's N dimension, which tcgen05
// requires to be a multiple of 8. Prep zeroes the pad rows so they are
// benign in the matmul. Must match Q_PAD_ROWS in compiled_attention.py.
constexpr int Q_STAGED_ROWS = 8;

// ============================================================================
// Decode-attention PREP for the compiler-generated attention core.
//
// One task per (request, kv_head); the layer slices qkv / k_cache / v_cache /
// and the staging buffers to this task's kv head via imaps. For every NEW
// token of the step (one at decode; several during chunked prefill) it:
//   1. applies q_norm/k_norm (RMSNorm over HEAD_DIM, fp32 internal) and NeoX
//      RoPE at the token's absolute position,
//   2. appends k and v to the (contiguous, page_size >= max_seq) KV cache,
//      k also TRANSPOSED into kt_staged (the generated core needs a
//      physically row-major K^T; TMA cannot transpose) and v into v_staged,
//   3. clears the additive mask at the token's position (mask starts at
//      -30000 everywhere, so unwritten positions vanish from the softmax).
// Only the LAST token's q is staged, scaled by 1/sqrt(HEAD_DIM), into the
// zero-padded q_staged rows [0, NUM_QO_PER_KV) -- decode-only generation
// consumes no other row's attention output, and the last token sees the
// full appended history, so causality is free.
//
// One THREAD per (token, row) does a full HEAD_DIM norm+rope; the fp32
// x[]/y[] scratch (1KB per thread at HEAD_DIM=128) spills to local memory,
// which is fine at decode and acceptable during prefill.
// ============================================================================
template <typename T,
          int NUM_QO_PER_KV,       // q heads per kv head (2 for Qwen3)
          int HEAD_DIM,            // 128
          int QKV_STRIDE,          // full fused row stride in elements
          int KV_CACHE_ROW_STRIDE, // num_kv_heads * HEAD_DIM (element stride
                                   // between cache rows within a page)
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int Q_STAGED_REQ_STRIDE,  // 8 * HEAD_DIM (per-kvh slice)
          int KT_STAGED_REQ_STRIDE> // HEAD_DIM * MAX_SEQ_LEN (per-kvh slice)
__device__ __forceinline__ void attention_prep_sm100_impl(
    void const *qkv_ptr,     // sliced to this kv head: [tokens, 512-col slice]
    void *paged_k_cache_ptr, // sliced to this kv head
    void *paged_v_cache_ptr, // sliced to this kv head
    void const *q_norm_weight_ptr, // [HEAD_DIM]
    void const *k_norm_weight_ptr, // [HEAD_DIM]
    void const *cos_ptr,           // [max_seq, HEAD_DIM] NeoX duplicated
    void const *sin_ptr,
    void *q_staged_ptr,  // per-kvh slice: [reqs, 8, HEAD_DIM]
    void *mask_ptr,      // per-kvh slice: [reqs, 1, MAX_SEQ_LEN], init -30000
    void *kt_staged_ptr, // per-kvh slice: K TRANSPOSED [reqs, HEAD_DIM, S]
                         // row-major -- the generated core needs physically
                         // row-major K^T (TMA cannot transpose)
    void *v_staged_ptr,  // per-kvh slice: [reqs, MAX_SEQ_LEN, HEAD_DIM]
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    float eps) {
  constexpr int HALF = HEAD_DIM / 2;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return; // inactive request
  }
  int const num_tokens = last_token_pos - first_token_pos;
  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const seq_len = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[request_id];
  int const first_new_pos = seq_len - num_tokens;
  int const page_idx = paged_kv_indices_buffer_ptr[first_page_pos];
  // The compiler-generated attention core reads K/V through STATIC
  // per-request views of the cache (kv[request_id's page]); that is only
  // sound when the page allocator handed request r page r. page_size >=
  // max_seq guarantees one page per request, and offline batch admission
  // assigns them in order. Trap loudly if the assumption ever breaks
  // instead of silently attending to another request's history.
  if (threadIdx.x == 0 && page_idx != request_id) {
    printf("attention_prep_sm100: page_idx %d != request_id %d -- the "
           "generated attention core's static KV views are invalid\n",
           page_idx,
           (int)request_id);
    // Unconditional: assert() is NDEBUG-elided, which would silently remove
    // this protection from release builds.
    __trap();
  }

  T const *qkv_base =
      static_cast<T const *>(qkv_ptr) + (size_t)first_token_pos * QKV_STRIDE;
  T *k_cache = static_cast<T *>(paged_k_cache_ptr);
  T *v_cache = static_cast<T *>(paged_v_cache_ptr);
  T *q_staged =
      static_cast<T *>(q_staged_ptr) + (size_t)request_id * Q_STAGED_REQ_STRIDE;
  T *mask = static_cast<T *>(mask_ptr) + (size_t)request_id * MAX_SEQ_LEN;
  T *kt_staged = static_cast<T *>(kt_staged_ptr) +
                 (size_t)request_id * KT_STAGED_REQ_STRIDE;
  T *v_staged = static_cast<T *>(v_staged_ptr) +
                (size_t)request_id * (size_t)MAX_SEQ_LEN * HEAD_DIM;
  float const q_scale = rsqrtf(static_cast<float>(HEAD_DIM));

  // Multi-token (prefill) steps append EVERY new token's k/v; only the LAST
  // token's q is staged -- in decode-only generation nothing downstream
  // consumes the other prefill rows' attention output, and the last token
  // attends over the full (now complete) history, so causality is free.
  //
  // One thread per (token, row) with row in [0, NUM_QO_PER_KV] (q rows + k);
  // v rows and bookkeeping handled by a second pass below.
  int const norm_rows = num_tokens * (NUM_QO_PER_KV + 1);
  for (int p = threadIdx.x; p < norm_rows; p += NUM_THREADS) {
    int const tok = p / (NUM_QO_PER_KV + 1);
    int const row = p % (NUM_QO_PER_KV + 1);
    bool const is_k = (row == NUM_QO_PER_KV);
    bool const is_last_tok = (tok == num_tokens - 1);
    if (!is_k && !is_last_tok) {
      continue; // q rows of non-final prefill tokens are never used
    }
    int const pos = first_new_pos + tok;
    T const *src = qkv_base + (size_t)tok * QKV_STRIDE + (size_t)row * HEAD_DIM;
    T const *w =
        static_cast<T const *>(is_k ? k_norm_weight_ptr : q_norm_weight_ptr);
    T const *cos = static_cast<T const *>(cos_ptr) + (size_t)pos * HEAD_DIM;
    T const *sin = static_cast<T const *>(sin_ptr) + (size_t)pos * HEAD_DIM;
    float x[HEAD_DIM];
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
      x[i] = float(src[i]);
      sum_sq += x[i] * x[i];
    }
    float const r = rsqrtf(sum_sq / HEAD_DIM + eps);
#pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
      x[i] = x[i] * r * float(w[i]);
    }
    // NeoX rope: pair (i, i+HALF)
    float y[HEAD_DIM];
#pragma unroll
    for (int i = 0; i < HALF; i++) {
      float const c0 = float(cos[i]), s0 = float(sin[i]);
      float const c1 = float(cos[i + HALF]), s1 = float(sin[i + HALF]);
      y[i] = x[i] * c0 - x[i + HALF] * s0;
      y[i + HALF] = x[i + HALF] * c1 + x[i] * s1;
    }
    if (is_k) {
      size_t const cache_row =
          ((size_t)page_idx * PAGE_SIZE + pos) * KV_CACHE_ROW_STRIDE;
#pragma unroll
      for (int i = 0; i < HEAD_DIM; i++) {
        k_cache[cache_row + i] = T(y[i]);
        kt_staged[(size_t)i * MAX_SEQ_LEN + pos] = T(y[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < HEAD_DIM; i++) {
        q_staged[(size_t)row * HEAD_DIM + i] = T(y[i] * q_scale);
      }
    }
  }
  // v append (raw, no norm/rope -- matches the monolith), mask flips, and
  // staging pad rows.
  for (int p = threadIdx.x; p < num_tokens * HEAD_DIM; p += NUM_THREADS) {
    int const tok = p / HEAD_DIM;
    int const i = p % HEAD_DIM;
    int const pos = first_new_pos + tok;
    T const *src = qkv_base + (size_t)tok * QKV_STRIDE +
                   (size_t)(NUM_QO_PER_KV + 1) * HEAD_DIM;
    size_t const cache_row =
        ((size_t)page_idx * PAGE_SIZE + pos) * KV_CACHE_ROW_STRIDE;
    v_cache[cache_row + i] = src[i];
    v_staged[(size_t)pos * HEAD_DIM + i] = src[i];
    if (i == 0) {
      // Idempotent across the kv-head tasks of this request.
      mask[pos] = T(0.0f);
    }
  }
  for (int p = threadIdx.x; p < (Q_STAGED_ROWS - NUM_QO_PER_KV) * HEAD_DIM;
       p += NUM_THREADS) {
    int const row = NUM_QO_PER_KV + p / HEAD_DIM;
    q_staged[(size_t)row * HEAD_DIM + (p % HEAD_DIM)] = T(0.0f);
  }
}

} // namespace kernel
