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

// MLA KV Cache Gather for SM100
//
// Gathers paged KV cache entries into a contiguous buffer for TMA-based
// MLA decode kernel. Also handles:
//   1. Appending new c_latent + k_pe to the paged cache
//   2. Gathering the full KV sequence into a contiguous buffer
//
// The contiguous buffer layout matches what mla_decode_sm100 expects:
//   [kv_len, D_K] where D_K = 576 (stored as bf16)
//
// Grid:  (max_num_batched_requests, 1, 1)
// Block: (128, 1, 1)

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {

template <int D_K,       // Total KV dim (576 = 512 latent + 64 rope)
          int D_V,       // Latent dim (512)
          int PAGE_SIZE> // Page size (e.g., 128)
__device__ __forceinline__ void mla_kv_cache_gather_sm100_task_impl(
    void const *c_latent_new_ptr, // [num_tokens, D_V] new c_latent (normed)
    void const *k_pe_new_ptr,     // [num_tokens, D_K - D_V] new k_pe
    void *paged_cache_ptr,        // [num_pages, PAGE_SIZE, D_K] paged KV cache
    void *contiguous_kv_ptr,      // [max_seq_len, D_K] output: contiguous KV
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id) {
  using T = __nv_bfloat16;
  int const tid = threadIdx.x;
  int const NUM_THREADS = 128;
  int const ROPE_DIM = D_K - D_V; // 64

  // Get sequence metadata for this request
  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  int const num_new_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const last_page_len = paged_kv_last_page_len_buffer_ptr[request_id];
  int const seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len;

  // Bounds check: skip if page table looks uninitialized
  bool valid =
      (num_pages > 0 && num_pages <= 512 && num_new_tokens > 0 && seq_len > 0);

  T *paged_cache = reinterpret_cast<T *>(paged_cache_ptr);
  T *contiguous_kv = reinterpret_cast<T *>(contiguous_kv_ptr);
  T const *c_latent_new = reinterpret_cast<T const *>(c_latent_new_ptr);
  T const *k_pe_new = reinterpret_cast<T const *>(k_pe_new_ptr);

  // Load page indices into shared memory
  __shared__ int page_indices[512]; // max pages per request
  if (valid) {
    for (int i = tid; i < num_pages; i += NUM_THREADS) {
      page_indices[i] = paged_kv_indices_buffer_ptr[first_page_pos + i];
    }
  }
  __syncthreads();
  if (!valid) {
    return;
  }

  // Step 1: Append new tokens to paged cache
  // Write c_latent (D_V=512 dims) + k_pe (ROPE_DIM=64 dims) into the correct
  // page positions
  int const kv_start_pos = seq_len - num_new_tokens;
  for (int tok = 0; tok < num_new_tokens; tok++) {
    int const seq_pos = kv_start_pos + tok;
    int const page_idx = page_indices[seq_pos / PAGE_SIZE];
    int const pos_in_page = seq_pos % PAGE_SIZE;
    T *dst = paged_cache + (page_idx * PAGE_SIZE + pos_in_page) * D_K;
    T const *src_lat = c_latent_new + tok * D_V;
    T const *src_pe = k_pe_new + tok * ROPE_DIM;

    // Copy c_latent (512 dims) — vectorized uint4 loads (8 bf16 per load)
    for (int d = tid * 8; d < D_V; d += NUM_THREADS * 8) {
      if (d + 8 <= D_V) {
        *reinterpret_cast<uint4 *>(dst + d) =
            *reinterpret_cast<uint4 const *>(src_lat + d);
      }
    }
    // Copy k_pe (64 dims)
    for (int d = tid * 8; d < ROPE_DIM; d += NUM_THREADS * 8) {
      if (d + 8 <= ROPE_DIM) {
        *reinterpret_cast<uint4 *>(dst + D_V + d) =
            *reinterpret_cast<uint4 const *>(src_pe + d);
      }
    }
  }
  __syncthreads();

  // Step 2: Gather all pages into contiguous buffer
  // For each sequence position, copy D_K elements from the paged cache
  // to the contiguous buffer
  for (int seq_pos = 0; seq_pos < seq_len; seq_pos++) {
    int const page_idx = page_indices[seq_pos / PAGE_SIZE];
    int const pos_in_page = seq_pos % PAGE_SIZE;
    T const *src = paged_cache + (page_idx * PAGE_SIZE + pos_in_page) * D_K;
    T *dst = contiguous_kv + seq_pos * D_K;

    // Vectorized copy: D_K=576 / 8 = 72 uint4 loads, with 128 threads
    for (int d = tid * 8; d < D_K; d += NUM_THREADS * 8) {
      if (d + 8 <= D_K) {
        *reinterpret_cast<uint4 *>(dst + d) =
            *reinterpret_cast<uint4 const *>(src + d);
      }
    }
  }
}

} // namespace kernel
