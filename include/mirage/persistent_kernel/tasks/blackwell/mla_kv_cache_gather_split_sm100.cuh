/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

// MLA KV Cache Gather (SPLIT output) for SM100
//
// Same as mla_kv_cache_gather_sm100.cuh but outputs CKV and KPE as SEPARATE
// dense buffers instead of a single concatenated [S, D_K=576] tensor. This
// is the layout expected by mla_prefill_sm100 (chunked prefill kernel).
//
// Steps per request:
//   1. Append new c_latent + k_pe to the paged cache (unchanged from
//      mla_kv_cache_gather_sm100)
//   2. Gather the full sequence into:
//        ckv_sep  [max_seq_len, D_V=512]
//        kpe_sep  [max_seq_len, D_K-D_V=64]
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
__device__ __forceinline__ void mla_kv_cache_gather_split_sm100_task_impl(
    void const *c_latent_new_ptr, // [num_tokens, D_V] new c_latent
    void const *k_pe_new_ptr,     // [num_tokens, D_K - D_V] new k_pe
    void *paged_cache_ptr,        // [num_pages, PAGE_SIZE, D_K] paged cache
    void *ckv_sep_ptr,            // [max_seq_len, D_V=512] output
    void *kpe_sep_ptr,            // [max_seq_len, D_K-D_V=64] output
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id) {
  using T = __nv_bfloat16;
  int const tid = threadIdx.x;
  int const NUM_THREADS = 128;
  constexpr int ROPE_DIM = D_K - D_V; // 64

  // Sequence metadata
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
  T *ckv_sep = reinterpret_cast<T *>(ckv_sep_ptr);
  T *kpe_sep = reinterpret_cast<T *>(kpe_sep_ptr);
  T const *c_latent_new = reinterpret_cast<T const *>(c_latent_new_ptr);
  T const *k_pe_new = reinterpret_cast<T const *>(k_pe_new_ptr);

  int const *page_indices = paged_kv_indices_buffer_ptr + first_page_pos;
  __syncthreads();
  if (!valid) {
    return;
  }

  // Step 1: append new tokens to paged cache (identical to non-split variant —
  // the cache layout is still [c_latent | k_pe] concatenated per position).
  int const kv_start_pos = seq_len - num_new_tokens;
  for (int tok = 0; tok < num_new_tokens; tok++) {
    int const seq_pos = kv_start_pos + tok;
    int const page_idx = page_indices[seq_pos / PAGE_SIZE];
    int const pos_in_page = seq_pos % PAGE_SIZE;
    T *dst = paged_cache + (page_idx * PAGE_SIZE + pos_in_page) * D_K;
    T const *src_lat = c_latent_new + tok * D_V;
    T const *src_pe = k_pe_new + tok * ROPE_DIM;
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
  __syncthreads();

  // Step 2: gather into TWO separate contiguous buffers (CKV and KPE).
  for (int seq_pos = 0; seq_pos < seq_len; seq_pos++) {
    int const page_idx = page_indices[seq_pos / PAGE_SIZE];
    int const pos_in_page = seq_pos % PAGE_SIZE;
    T const *src = paged_cache + (page_idx * PAGE_SIZE + pos_in_page) * D_K;
    T *ckv_dst = ckv_sep + seq_pos * D_V;
    T *kpe_dst = kpe_sep + seq_pos * ROPE_DIM;

    // Copy 512-element c_latent into ckv_sep
    for (int d = tid * 8; d < D_V; d += NUM_THREADS * 8) {
      if (d + 8 <= D_V) {
        *reinterpret_cast<uint4 *>(ckv_dst + d) =
            *reinterpret_cast<uint4 const *>(src + d);
      }
    }
    // Copy 64-element k_pe into kpe_sep
    for (int d = tid * 8; d < ROPE_DIM; d += NUM_THREADS * 8) {
      if (d + 8 <= ROPE_DIM) {
        *reinterpret_cast<uint4 *>(kpe_dst + d) =
            *reinterpret_cast<uint4 const *>(src + D_V + d);
      }
    }
  }
}

} // namespace kernel
