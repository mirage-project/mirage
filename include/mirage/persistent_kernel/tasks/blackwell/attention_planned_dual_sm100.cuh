/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Fused planned-attention consumer (SM100).
//
// One task per bucket — handles BOTH prefill and decode works assigned to
// that bucket in a single CTA. Replaces the separate planned_prefill +
// planned_decode consumers, halving the number of consumer CTAs the planner
// flow launches (NUM_BUCKETS instead of 2*NUM_BUCKETS).
//
// Per-CTA flow:
//   1. Read this bucket's prefill work slice (plan_prefill_indptr[b..b+1)).
//      For each work, invoke the wide-Q prefill kernel.
//   2. Read this bucket's decode work slice (plan_decode_indptr[b..b+1)).
//      For each work, invoke the split-K decode kernel.
//
// This is optimization #4 from issue #627's perf follow-up. The shared-memory
// footprint is max(prefill, decode); since prefill is the larger kernel,
// no SMEM increase. Register pressure may rise (compiler reserves union)
// but B200 register budget tolerates this for our configs.

#pragma once
#include "attention_prefill_sm100.cuh"
#include "attention_sm100.cuh"
#include "tasks/common/common_header.cuh"

namespace kernel {

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS_TOTAL,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int NUM_BUCKETS,
          int MAX_WORKS,
          int KV_SPLIT_SIZE = 0,
          int NUM_KV_CHUNKS = 1>
__device__ __forceinline__ void planned_dual_attention_sm100_task_impl(
    void const *qkv_base_ptr,
    void *paged_k_cache_base_ptr,
    void *paged_v_cache_base_ptr,
    void *output_base_ptr,
    void *final_output_base_ptr,
    void *lse_ptr,
    int const *plan_buffer,
    int const *qo_indptr_buffer,
    int const *paged_kv_indptr_buffer,
    int const *paged_kv_indices_buffer,
    int const *paged_kv_last_page_len_buffer,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps,
    int bucket_idx) {
  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS_TOTAL;
  constexpr int QKV_GROUP_STRIDE = (NUM_QO_PER_KV + 2) * HEAD_DIM;
  constexpr int O_GROUP_STRIDE = NUM_QO_PER_KV * HEAD_DIM;

  int const *plan_prefill_indptr = plan_buffer;
  int const *plan_decode_indptr = plan_buffer + (NUM_BUCKETS + 1);
  int const *worker_batch_indices = plan_buffer + 2 * (NUM_BUCKETS + 1);
  int const *worker_kv_head_indices = worker_batch_indices + MAX_WORKS;
  int const *worker_qo_tile_indices = worker_kv_head_indices + MAX_WORKS;
  int const *worker_kv_tile_indices = worker_qo_tile_indices + MAX_WORKS;

  int p_start = plan_prefill_indptr[bucket_idx];
  int p_end = plan_prefill_indptr[bucket_idx + 1];
  int d_start = plan_decode_indptr[bucket_idx];
  int d_end = plan_decode_indptr[bucket_idx + 1];
  int total_works = plan_decode_indptr[NUM_BUCKETS];

  (void)total_works;

  // Optimization #5: cheap early-exit when bucket has no work.
  if (p_start == p_end && d_start == d_end) {
    return;
  }

#define RUN_PREFILL_WORK(WORK_ID)                                             \
  do {                                                                        \
    int const _w = (WORK_ID);                                                 \
    int batch = worker_batch_indices[_w];                                     \
    int kv_head = worker_kv_head_indices[_w];                                 \
    int qo_tile = worker_qo_tile_indices[_w];                                 \
    int kv_tile = worker_kv_tile_indices[_w];                                 \
    if constexpr (KV_SPLIT_SIZE > 0) {                                        \
      if (kv_tile < 0) {                                                      \
        T const *qkv_w = static_cast<T const *>(qkv_base_ptr) +               \
                         kv_head * QKV_GROUP_STRIDE;                         \
        T *k_w = static_cast<T *>(paged_k_cache_base_ptr) +                   \
                 kv_head * HEAD_DIM;                                         \
        T *v_w = static_cast<T *>(paged_v_cache_base_ptr) +                   \
                 kv_head * HEAD_DIM;                                         \
        T *o_w = static_cast<T *>(final_output_base_ptr) +                    \
                 kv_head * O_GROUP_STRIDE;                                   \
        multitoken_paged_attention_prefill_sm100_task_impl<T,                 \
                                                       NUM_QO_PER_KV,         \
                                                       /*NUM_KV_HEADS=*/1,    \
                                                       KV_CACHE_STRIDE,       \
                                                       QKV_STRIDE,            \
                                                       NUM_QO_HEADS *         \
                                                           HEAD_DIM,          \
                                                       HEAD_DIM,              \
                                                       MAX_SEQ_LEN,           \
                                                       PAGE_SIZE,             \
                                                       MAX_TOKENS>(           \
            qkv_w, k_w, v_w, o_w,                                             \
            qo_indptr_buffer, paged_kv_indptr_buffer,                         \
            paged_kv_indices_buffer, paged_kv_last_page_len_buffer,           \
            static_cast<int16_t>(batch), qk_norm, rope,                       \
            q_norm_weight_ptr, k_norm_weight_ptr,                             \
            cos_ptr, sin_ptr, q_eps, k_eps, -1, qo_tile, -1,                  \
            MAX_SEQ_LEN, nullptr, -1, NUM_KV_HEADS_TOTAL, NUM_KV_CHUNKS);     \
        break;                                                               \
      }                                                                       \
    }                                                                         \
                                                                              \
    T const *qkv_w = static_cast<T const *>(qkv_base_ptr);                    \
    T *k_w = static_cast<T *>(paged_k_cache_base_ptr);                        \
    T *v_w = static_cast<T *>(paged_v_cache_base_ptr);                        \
    T *o_w = static_cast<T *>(output_base_ptr);                               \
    int flat_kv_head_idx = kv_head;                                           \
    int kv_chunk_idx = kv_tile;                                               \
    int kv_chunk_size = KV_SPLIT_SIZE;                                        \
    void *work_lse_ptr = lse_ptr;                                             \
    if constexpr (KV_SPLIT_SIZE == 0) {                                       \
      qkv_w += kv_head * QKV_GROUP_STRIDE;                                    \
      k_w += kv_head * HEAD_DIM;                                              \
      v_w += kv_head * HEAD_DIM;                                              \
      o_w += kv_head * O_GROUP_STRIDE;                                        \
      flat_kv_head_idx = -1;                                                  \
      kv_chunk_idx = -1;                                                      \
      kv_chunk_size = 256;                                                    \
      work_lse_ptr = nullptr;                                                 \
    }                                                                         \
    multitoken_paged_attention_prefill_sm100_task_impl<T,                     \
                                                       NUM_QO_PER_KV,         \
                                                       /*NUM_KV_HEADS=*/1,    \
                                                       KV_CACHE_STRIDE,       \
                                                       QKV_STRIDE,            \
                                                       O_STRIDE,              \
                                                       HEAD_DIM,              \
                                                       MAX_SEQ_LEN,           \
                                                       PAGE_SIZE,             \
                                                       MAX_TOKENS>(           \
        qkv_w, k_w, v_w, o_w,                                                 \
        qo_indptr_buffer, paged_kv_indptr_buffer,                             \
        paged_kv_indices_buffer, paged_kv_last_page_len_buffer,               \
        static_cast<int16_t>(batch), qk_norm, rope,                           \
        q_norm_weight_ptr, k_norm_weight_ptr,                                 \
        cos_ptr, sin_ptr, q_eps, k_eps, -1, qo_tile, kv_chunk_idx,             \
        kv_chunk_size, work_lse_ptr, flat_kv_head_idx, NUM_KV_HEADS_TOTAL,    \
        NUM_KV_CHUNKS);                                                       \
  } while (0)

#define RUN_DECODE_WORK(WORK_ID)                                              \
  do {                                                                        \
    int const _w = (WORK_ID);                                                 \
    int batch = worker_batch_indices[_w];                                     \
    int kv_head = worker_kv_head_indices[_w];                                 \
    int qo_tile = worker_qo_tile_indices[_w];                                 \
    int kv_tile = worker_kv_tile_indices[_w];                                 \
                                                                              \
    if constexpr (KV_SPLIT_SIZE > 0) {                                        \
      if (kv_tile < 0) {                                                      \
        T const *qkv_w = static_cast<T const *>(qkv_base_ptr) +               \
                         kv_head * QKV_GROUP_STRIDE;                         \
        T *k_w = static_cast<T *>(paged_k_cache_base_ptr) +                   \
                 kv_head * HEAD_DIM;                                         \
        T *v_w = static_cast<T *>(paged_v_cache_base_ptr) +                   \
                 kv_head * HEAD_DIM;                                         \
        T *o_w = static_cast<T *>(final_output_base_ptr) +                    \
                 kv_head * O_GROUP_STRIDE;                                   \
        multitoken_paged_attention_sm100_task_impl<T,                         \
                                                   NUM_QO_PER_KV,             \
                                                   /*NUM_KV_HEADS=*/1,        \
                                                   KV_CACHE_STRIDE,           \
                                                   QKV_STRIDE,                \
                                                   /*O_STRIDE=*/NUM_QO_HEADS *\
                                                       HEAD_DIM,              \
                                                   HEAD_DIM,                  \
                                                   MAX_SEQ_LEN,               \
                                                   PAGE_SIZE>(                \
            qkv_w, k_w, v_w, o_w,                                             \
            qo_indptr_buffer, paged_kv_indptr_buffer,                         \
            paged_kv_indices_buffer, paged_kv_last_page_len_buffer,           \
            static_cast<int16_t>(batch), qk_norm, rope,                       \
            q_norm_weight_ptr, k_norm_weight_ptr,                             \
            cos_ptr, sin_ptr, q_eps, k_eps);                                  \
      } else {                                                                \
        T const *qkv_w = static_cast<T const *>(qkv_base_ptr) +               \
                         kv_head * QKV_GROUP_STRIDE;                         \
        T *k_w = static_cast<T *>(paged_k_cache_base_ptr) +                   \
                 kv_head * HEAD_DIM;                                         \
        T *v_w = static_cast<T *>(paged_v_cache_base_ptr) +                   \
                 kv_head * HEAD_DIM;                                         \
        multitoken_paged_attention_sm100_task_impl<T,                         \
                                                   NUM_QO_PER_KV,             \
                                                   /*NUM_KV_HEADS=*/1,        \
                                                   KV_CACHE_STRIDE,           \
                                                   QKV_STRIDE,                \
                                                   O_STRIDE,                  \
                                                   HEAD_DIM,                  \
                                                   MAX_SEQ_LEN,               \
                                                   PAGE_SIZE>(                \
            qkv_w, k_w, v_w, static_cast<T *>(output_base_ptr),               \
            qo_indptr_buffer, paged_kv_indptr_buffer,                         \
            paged_kv_indices_buffer, paged_kv_last_page_len_buffer,           \
            static_cast<int16_t>(batch), qk_norm, rope,                       \
            q_norm_weight_ptr, k_norm_weight_ptr,                             \
            cos_ptr, sin_ptr, q_eps, k_eps, -1, kv_tile, KV_SPLIT_SIZE,       \
            lse_ptr, kv_head, NUM_KV_HEADS_TOTAL, NUM_KV_CHUNKS);             \
      }                                                                       \
    } else {                                                                  \
      (void)qo_tile;                                                          \
      (void)kv_tile;                                                          \
      (void)final_output_base_ptr;                                            \
      T const *qkv_w = static_cast<T const *>(qkv_base_ptr) +                 \
                       kv_head * QKV_GROUP_STRIDE;                           \
      T *k_w = static_cast<T *>(paged_k_cache_base_ptr) +                     \
               kv_head * HEAD_DIM;                                           \
      T *v_w = static_cast<T *>(paged_v_cache_base_ptr) +                     \
               kv_head * HEAD_DIM;                                           \
      T *o_w = static_cast<T *>(output_base_ptr) + kv_head * O_GROUP_STRIDE;  \
                                                                              \
      multitoken_paged_attention_sm100_task_impl<T,                           \
                                                 NUM_QO_PER_KV,               \
                                                 /*NUM_KV_HEADS=*/1,          \
                                                 KV_CACHE_STRIDE,             \
                                                 QKV_STRIDE,                  \
                                                 O_STRIDE,                    \
                                                 HEAD_DIM,                    \
                                                 MAX_SEQ_LEN,                 \
                                                 PAGE_SIZE>(                  \
          qkv_w, k_w, v_w, o_w,                                               \
          qo_indptr_buffer, paged_kv_indptr_buffer,                           \
          paged_kv_indices_buffer, paged_kv_last_page_len_buffer,             \
          static_cast<int16_t>(batch), qk_norm, rope,                         \
          q_norm_weight_ptr, k_norm_weight_ptr,                               \
          cos_ptr, sin_ptr, q_eps, k_eps);                                    \
    }                                                                         \
  } while (0)

  // Prefill works first. The wide-Q prefill kernel uses per-warp split-M,
  // and individual warps can finish their M-blocks at slightly different
  // times. Without an explicit sync between phases, when we enter the decode
  // loop below, the decode kernel's first __syncthreads() can deadlock
  // because some warps are still inside the prefill kernel's body.
  for (int w = p_start; w < p_end; w++) {
    RUN_PREFILL_WORK(w);
  }

  // Sync ALL threads/warps before switching to the decode kernel (see comment
  // above the prefill loop). Only needed when real prefill work ran in this
  // bucket; p_start/p_end are uniform across the block (read from the plan
  // buffer), so the guard does not diverge. Skips the barrier on decode-only
  // buckets (the common decode-iteration case).
  if (p_start != p_end) {
    __syncthreads();
  }

  // Decode works second.
  for (int w = d_start; w < d_end; w++) {
    RUN_DECODE_WORK(w);
  }

#undef RUN_PREFILL_WORK
#undef RUN_DECODE_WORK
}

} // namespace kernel
