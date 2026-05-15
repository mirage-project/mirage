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

// 2026-05-12 (user #2 part-a) FuseTensor support: C_LATENT_ROW_STRIDE controls
// the per-token stride of the c_latent input. Defaults to D_V (legacy
// contiguous (mbt, D_V) buffer). For the QKV-a fused path c_latent lives at
// cols [1536:2048) of a wider (mbt, 2176) qkv_a_out buffer, so pass
// C_LATENT_ROW_STRIDE=2176 and pre-offset c_latent_new_ptr by 1536 elements.
// K_PE_ROW_STRIDE already supports the same idea for k_pe.
template <int D_K,       // Total KV dim (576 = 512 latent + 64 rope)
          int D_V,       // Latent dim (512)
          int PAGE_SIZE, // Page size (e.g., 128)
          int K_PE_ROW_STRIDE = D_K - D_V,
          int C_LATENT_ROW_STRIDE = D_V>
__device__ __forceinline__ void mla_kv_cache_gather_sm100_task_impl(
    void const *c_latent_new_ptr, // [num_tokens, C_LATENT_ROW_STRIDE] new
                                  // c_latent (normed)
    void const *k_pe_new_ptr,     // [num_tokens, K_PE_ROW_STRIDE] new k_pe
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
  bool valid = (num_pages > 0 && num_new_tokens > 0 && seq_len > 0);

  T *paged_cache = reinterpret_cast<T *>(paged_cache_ptr);
  T *contiguous_kv = reinterpret_cast<T *>(contiguous_kv_ptr);
  T const *c_latent_new = reinterpret_cast<T const *>(c_latent_new_ptr);
  T const *k_pe_new = reinterpret_cast<T const *>(k_pe_new_ptr);

  // Page indices are read directly from global memory (L2-cached)
  int const *page_indices = paged_kv_indices_buffer_ptr + first_page_pos;
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
    T const *src_lat = c_latent_new + tok * C_LATENT_ROW_STRIDE;
    T const *src_pe = k_pe_new + tok * K_PE_ROW_STRIDE;

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

  if (contiguous_kv_ptr == paged_cache_ptr) {
    return;
  }

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

// B14 (2026-05-15): multi-CTA stripe across seq_pos in Phase 2.
//
// Pre-B14: single CTA per request did Phase 1 (append num_new_tokens
// to paged_cache) + Phase 2 (read paged_cache, write ckv_sep/kpe_sep
// for prefill OR contiguous_kv for decode). seq_pos loop was serial
// → 117 μs per call for ~128 prefill rows / decode iter.
//
// Post-B14: grid_dim now (num_requests, NUM_SEQ_CHUNKS, 1). Phase 1
// only runs on chunk 0 (q_len writes, tiny). Phase 2 strides seq_pos
// across NUM_SEQ_CHUNKS CTAs. To avoid cross-CTA race with Phase 1's
// paged_cache writes, Phase 2 reads c_latent_new/k_pe_new DIRECTLY
// for seq_pos >= kv_start_pos (the just-appended rows), and only
// reads paged_cache for seq_pos < kv_start_pos (genuine cache hits).
// This makes Phase 2 independent of Phase 1's writes — no fence
// required.
template <int D_K,       // Total KV dim (576 = 512 latent + 64 rope)
          int D_V,       // Latent dim (512)
          int PAGE_SIZE, // Page size (e.g., 128)
          int K_PE_ROW_STRIDE = D_K - D_V,
          int C_LATENT_ROW_STRIDE = D_V>
__device__ __forceinline__ void mla_kv_cache_gather_unified_sm100_task_impl(
    void const *c_latent_new_ptr, // [num_tokens, C_LATENT_ROW_STRIDE] new
                                  // c_latent (normed)
    void const *k_pe_new_ptr,     // [num_tokens, K_PE_ROW_STRIDE] new k_pe
    void *paged_cache_ptr,        // [num_pages, PAGE_SIZE, D_K] paged KV cache
    void *contiguous_kv_ptr,      // [max_seq_len, D_K] decode-layout output
    void *ckv_sep_ptr,            // [max_seq_len, D_V] prefill CKV output
    void *kpe_sep_ptr,            // [max_seq_len, D_K-D_V] prefill KPE output
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    bool prompt_prefill,
    int request_id,
    int seq_chunk_idx, // bid.y; chunk 0 also runs Phase 1
    int num_seq_chunks) {
  using T = __nv_bfloat16;
  int const tid = threadIdx.x;
  int const NUM_THREADS = 128;
  int const ROPE_DIM = D_K - D_V;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  int const num_new_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const last_page_len = paged_kv_last_page_len_buffer_ptr[request_id];
  int const seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len;

  bool valid = (num_pages > 0 && num_new_tokens > 0 && seq_len > 0);

  T *paged_cache = reinterpret_cast<T *>(paged_cache_ptr);
  T *contiguous_kv = reinterpret_cast<T *>(contiguous_kv_ptr);
  T *ckv_sep = reinterpret_cast<T *>(ckv_sep_ptr);
  T *kpe_sep = reinterpret_cast<T *>(kpe_sep_ptr);
  T const *c_latent_new = reinterpret_cast<T const *>(c_latent_new_ptr);
  T const *k_pe_new = reinterpret_cast<T const *>(k_pe_new_ptr);

  int const *page_indices = paged_kv_indices_buffer_ptr + first_page_pos;
  __syncthreads();
  if (!valid) {
    return;
  }

  int const kv_start_pos = seq_len - num_new_tokens;

  // Phase 1: only chunk 0 appends new tokens to paged_cache. Other
  // chunks skip and proceed straight to Phase 2 (using c_latent_new/
  // k_pe_new directly for new positions, avoiding cross-CTA race).
  if (seq_chunk_idx == 0) {
    for (int tok = 0; tok < num_new_tokens; tok++) {
      int const seq_pos = kv_start_pos + tok;
      int const page_idx = page_indices[seq_pos / PAGE_SIZE];
      int const pos_in_page = seq_pos % PAGE_SIZE;
      T *dst = paged_cache + (page_idx * PAGE_SIZE + pos_in_page) * D_K;
      T const *src_lat = c_latent_new + tok * C_LATENT_ROW_STRIDE;
      T const *src_pe = k_pe_new + tok * K_PE_ROW_STRIDE;

      for (int d = tid * 8; d < D_V; d += NUM_THREADS * 8) {
        if (d + 8 <= D_V) {
          *reinterpret_cast<uint4 *>(dst + d) =
              *reinterpret_cast<uint4 const *>(src_lat + d);
        }
      }
      for (int d = tid * 8; d < ROPE_DIM; d += NUM_THREADS * 8) {
        if (d + 8 <= ROPE_DIM) {
          *reinterpret_cast<uint4 *>(dst + D_V + d) =
              *reinterpret_cast<uint4 const *>(src_pe + d);
        }
      }
    }
  }

  if (!prompt_prefill && contiguous_kv_ptr == paged_cache_ptr) {
    return;
  }

  // Phase 2: stride seq_pos across NUM_SEQ_CHUNKS. For seq_pos >=
  // kv_start_pos (just-appended rows) read directly from
  // c_latent_new/k_pe_new — bypasses cross-CTA paged_cache race.
  for (int seq_pos = seq_chunk_idx; seq_pos < seq_len;
       seq_pos += num_seq_chunks) {
    T const *src_lat; // source for the D_V latent half
    T const *src_pe;  // source for the ROPE_DIM rope half
    bool combined = false;
    T const *src_combined = nullptr;
    if (seq_pos >= kv_start_pos) {
      int const tok = seq_pos - kv_start_pos;
      src_lat = c_latent_new + tok * C_LATENT_ROW_STRIDE;
      src_pe = k_pe_new + tok * K_PE_ROW_STRIDE;
    } else {
      int const page_idx = page_indices[seq_pos / PAGE_SIZE];
      int const pos_in_page = seq_pos % PAGE_SIZE;
      src_combined = paged_cache + (page_idx * PAGE_SIZE + pos_in_page) * D_K;
      combined = true;
    }

    if (prompt_prefill) {
      T *ckv_dst = ckv_sep + seq_pos * D_V;
      T *kpe_dst = kpe_sep + seq_pos * ROPE_DIM;
      for (int d = tid * 8; d < D_V; d += NUM_THREADS * 8) {
        if (d + 8 <= D_V) {
          T const *s = combined ? (src_combined + d) : (src_lat + d);
          *reinterpret_cast<uint4 *>(ckv_dst + d) =
              *reinterpret_cast<uint4 const *>(s);
        }
      }
      for (int d = tid * 8; d < ROPE_DIM; d += NUM_THREADS * 8) {
        if (d + 8 <= ROPE_DIM) {
          T const *s = combined ? (src_combined + D_V + d) : (src_pe + d);
          *reinterpret_cast<uint4 *>(kpe_dst + d) =
              *reinterpret_cast<uint4 const *>(s);
        }
      }
    } else {
      T *dst = contiguous_kv + seq_pos * D_K;
      // decode layout = [D_V latent | ROPE_DIM rope] packed into D_K.
      for (int d = tid * 8; d < D_V; d += NUM_THREADS * 8) {
        if (d + 8 <= D_V) {
          T const *s = combined ? (src_combined + d) : (src_lat + d);
          *reinterpret_cast<uint4 *>(dst + d) =
              *reinterpret_cast<uint4 const *>(s);
        }
      }
      for (int d = tid * 8; d < ROPE_DIM; d += NUM_THREADS * 8) {
        if (d + 8 <= ROPE_DIM) {
          T const *s = combined ? (src_combined + D_V + d) : (src_pe + d);
          *reinterpret_cast<uint4 *>(dst + D_V + d) =
              *reinterpret_cast<uint4 const *>(s);
        }
      }
    }
  }
}

} // namespace kernel
