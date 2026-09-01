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

constexpr int Q_STAGED_ROWS = 8;

template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int QKV_STRIDE,
          int KV_CACHE_ROW_STRIDE,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int Q_STAGED_REQ_STRIDE,
          int KT_STAGED_REQ_STRIDE>
__device__ __forceinline__ void attention_prep_sm100_impl(
    void const *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    void *q_staged_ptr,
    void *mask_ptr,
    void *kt_staged_ptr,
    void *v_staged_ptr,
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
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;
  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const seq_len = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[request_id];
  int const first_new_pos = seq_len - num_tokens;
  int const page_idx = paged_kv_indices_buffer_ptr[first_page_pos];
  if (threadIdx.x == 0 && page_idx != request_id) {
    printf("attention_prep_sm100: page_idx %d != request_id %d -- the "
           "generated attention core's static KV views are invalid\n",
           page_idx,
           (int)request_id);
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

  int const norm_rows = num_tokens * (NUM_QO_PER_KV + 1);
  for (int p = threadIdx.x; p < norm_rows; p += NUM_THREADS) {
    int const tok = p / (NUM_QO_PER_KV + 1);
    int const row = p % (NUM_QO_PER_KV + 1);
    bool const is_k = (row == NUM_QO_PER_KV);
    bool const is_last_tok = (tok == num_tokens - 1);
    if (!is_k && !is_last_tok) {
      continue;
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
      mask[pos] = T(0.0f);
    }
  }
  for (int p = threadIdx.x; p < (Q_STAGED_ROWS - NUM_QO_PER_KV) * HEAD_DIM;
       p += NUM_THREADS) {
    int const row = NUM_QO_PER_KV + p / HEAD_DIM;
    q_staged[(size_t)row * HEAD_DIM + (p % HEAD_DIM)] = T(0.0f);
  }
}

}
