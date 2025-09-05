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
#include "../common.h"
#include "../copy_sm80.cuh"
#include "../dmem_layout.cuh"
#include "../element_binary.cuh"
#include "../element_unary.cuh"
#include "../mma.cuh"
#include "../norm.cuh"
#include "../reduction.cuh"
#include "../rotary_embedding.cuh"
#include "../smem_layout.cuh"
#include "../utils.cuh"
#include "norm_wg.cuh"
#include "rotary_embedding_wg.cuh"
#include "smem_layout_tma.cuh"
#include "tma.cuh"
#include "utils.cuh"
#include "wgmma.cuh"

namespace kernel {

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          typename TMA_Q,
          typename TMA_KV,
          typename TMA_PAGED_KV_CACHE,
          typename TMA_PAGED_KV_CACHE_TAIL_PAGE,
          typename TMA_OUTPUT,
          int MAX_TOKENS = 8>
__device__ __forceinline__ void multitoken_paged_attention_hopper_impl(
    const TMA_Q &tma_q,
    const TMA_KV &tma_k,
    const TMA_KV &tma_v,
    const TMA_PAGED_KV_CACHE &tma_paged_k_cache,
    const TMA_PAGED_KV_CACHE &tma_paged_v_cache,
    const TMA_PAGED_KV_CACHE_TAIL_PAGE &tma_paged_k_cache_tail_page,
    const TMA_PAGED_KV_CACHE_TAIL_PAGE &tma_paged_v_cache_tail_page,
    const TMA_OUTPUT &tma_output,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps) {
  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;

  constexpr int KV_TILE_SIZE = 64;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int THREADS_PER_WARPGROUP = 128;
  constexpr int CONSUMER_WARPGROUPS = 1;
  constexpr int PRODUCER_WARPGROUPS = 1;
  constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
  constexpr int Kstages = 2;
  // NOTE(Yu): we use m64n64k16 mma atom to compute matrix multiplication
  constexpr int MMA_ITERS_M = (MAX_TOKENS * NUM_QO_PER_KV + 63) / 64;
  constexpr int QSMEM_ROW = 64;

  // the scale factor for normalization in softmax
  float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

  int warp_idx = warp_id();
  int lane_idx = lane_id();
  int warpgroup_id = warp_idx / WARPGROUP_WARPS;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  // Exit the current task is number of query tokens is zero
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const seq_len = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[request_id];

  // Load the paged KV indices into shared memory
  __shared__ int page_indices[MAX_PAGES_PER_REQUEST];
#pragma unroll
  for (int i = threadIdx.x; i < num_pages * sizeof(int) / 16;
       i += NUM_THREADS) {
    __uint128_t const *src_ptr =
        reinterpret_cast<__uint128_t const *>(paged_kv_indices_buffer_ptr +
                                              first_page_pos) +
        i;
    __uint128_t *dst_ptr = reinterpret_cast<__uint128_t *>(page_indices) + i;
    *dst_ptr = *src_ptr;
  }
  if (num_pages % (16 / sizeof(int)) != 0) {
    int tail_pages = num_pages % (16 / sizeof(int));
    int tail_offset = num_pages - tail_pages;
    for (int i = threadIdx.x; i < tail_pages; i += NUM_THREADS) {
      page_indices[tail_offset + i] =
          paged_kv_indices_buffer_ptr[first_page_pos + tail_offset + i];
    }
  }

  __syncthreads();

  T *__restrict__ d_paged_k_cache = reinterpret_cast<T *>(paged_k_cache_ptr);
  T *__restrict__ d_paged_v_cache = reinterpret_cast<T *>(paged_v_cache_ptr);

  //  DTensors' layouts
  using KVCacheDmem = dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_CACHE_STRIDE>;

  KVCacheDmem paged_k_cache_dmem(d_paged_k_cache),
      paged_v_cache_dmem(d_paged_v_cache);

  // STensors' offsets and sizes

  // since smem is 1024 bytes aligned, S_Q_OFFSET is set to zero
  constexpr size_t S_Q_OFFSET = 0;
  constexpr size_t S_Q_SIZE = sizeof(T) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;

  constexpr size_t S_K_OFFSET = (S_Q_OFFSET + S_Q_SIZE + 1023) / 1024 * 1024;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;

  constexpr size_t S_K_BUFFER_OFFSET =
      (S_K_OFFSET + S_K_SIZE + 1023) / 1024 * 1024;
  constexpr size_t S_K_BUFFER_SIZE = S_K_SIZE;

  constexpr size_t S_V_OFFSET =
      (S_K_BUFFER_OFFSET + S_K_BUFFER_SIZE + 1023) / 1024 * 1024;
  constexpr size_t S_V_SIZE = S_K_SIZE;

  constexpr size_t S_V_BUFFER_OFFSET =
      (S_V_OFFSET + S_V_SIZE + 1023) / 1024 * 1024;
  constexpr size_t S_V_BUFFER_SIZE = S_K_SIZE;

  constexpr size_t S_O_OFFSET =
      (S_V_BUFFER_OFFSET + S_V_BUFFER_SIZE + 1023) / 1024 * 1024;
  constexpr size_t S_O_SIZE = S_Q_SIZE;

  // align to size of float
  constexpr size_t S_Q_NORM_SUM_OFFSET =
      ((S_O_OFFSET + S_O_SIZE + sizeof(float) - 1) &
       ~size_t(sizeof(float) - 1));
  constexpr size_t S_Q_NORM_SUM_SIZE =
      sizeof(float) * 4; // 4 floats for 4 warps

  constexpr size_t S_K_NORM_SUM_OFFSET =
      S_Q_NORM_SUM_OFFSET + S_Q_NORM_SUM_SIZE;
  constexpr size_t S_K_NORM_SUM_SIZE = sizeof(float) * 4;

  constexpr size_t S_M_BUFFER_OFFSET = S_K_NORM_SUM_OFFSET + S_K_NORM_SUM_SIZE;
  constexpr size_t S_M_BUFFER_SIZE =
      sizeof(float) * MMA_ITERS_M * NUM_THREADS * 2;

  constexpr size_t S_D_BUFFER_OFFSET = S_M_BUFFER_OFFSET + S_M_BUFFER_SIZE;
  constexpr size_t S_D_BUFFER_SIZE =
      sizeof(float) * MMA_ITERS_M * NUM_THREADS * 2;

  constexpr size_t S_O_BUFFER_OFFSET = S_D_BUFFER_OFFSET + S_D_BUFFER_SIZE;
  constexpr size_t S_O_BUFFER_SIZE =
      sizeof(float) * MMA_ITERS_M * NUM_THREADS * 64;

  constexpr size_t S_Q_BARRIER_OFFSET =
      (S_O_BUFFER_OFFSET + S_O_BUFFER_SIZE + 7) / 8 * 8;
  constexpr size_t S_Q_BARRIER_SIZE = 8 * Kstages;

  constexpr size_t S_K_BARRIER_OFFSET =
      (S_Q_BARRIER_OFFSET + S_Q_BARRIER_SIZE + 7) / 8 * 8;
  constexpr size_t S_K_BARRIER_SIZE = 8 * Kstages;

  constexpr size_t S_V_BARRIER_OFFSET =
      (S_K_BARRIER_OFFSET + S_K_BARRIER_SIZE + 7) / 8 * 8;
  constexpr size_t S_V_BARRIER_SIZE = 8 * Kstages;

  constexpr size_t S_COMPUTE_DONE_OFFSET =
      (S_V_BARRIER_OFFSET + S_V_BARRIER_SIZE + 7) / 8 * 8;
  constexpr size_t S_COMPUTE_DONE_SIZE = 8 * Kstages;

  constexpr size_t S_TOTAL_OFFSET = S_COMPUTE_DONE_OFFSET + S_COMPUTE_DONE_SIZE;
  static_assert(S_TOTAL_OFFSET <= 224 * 1024);

  extern __shared__ char smem_ptr[];

  uintptr_t smem = (reinterpret_cast<uintptr_t>(smem_ptr) + 1023) / 1024 * 1024;

  T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
  T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
  T *s_k_buffer = reinterpret_cast<T *>(smem + S_K_BUFFER_OFFSET);
  T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
  T *s_v_buffer = reinterpret_cast<T *>(smem + S_V_BUFFER_OFFSET);
  T *s_o = reinterpret_cast<T *>(smem + S_O_OFFSET);
  float *s_q_norm_sum = reinterpret_cast<float *>(smem + S_Q_NORM_SUM_OFFSET);
  float *s_k_norm_sum = reinterpret_cast<float *>(smem + S_K_NORM_SUM_OFFSET);
  float *s_m_buffer = reinterpret_cast<float *>(smem + S_M_BUFFER_OFFSET);
  float *s_d_buffer = reinterpret_cast<float *>(smem + S_D_BUFFER_OFFSET);
  float *s_o_buffer = reinterpret_cast<float *>(smem + S_O_BUFFER_OFFSET);

  // STensors' layouts
  using QOSmem = smem_tma<T,
                          3,
                          3,
                          3,
                          MAX_TOKENS * NUM_QO_PER_KV,
                          64,
                          (HEAD_DIM + 63) / 64>;
  using KVSmem = smem_tma<T, 3, 3, 3, KV_TILE_SIZE, 64, (HEAD_DIM + 63) / 64>;

  using Q_DESC = wgmma::mma_descriptor<QOSmem>;
  using K_DESC = wgmma::mma_descriptor<KVSmem, false>;
  using V_DESC = wgmma::mma_descriptor<KVSmem, true>;

  QOSmem q_smem(s_q), o_smem(s_o);
  KVSmem k_smem(s_k), v_smem(s_v);
  KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);

  int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
  int curr_iter_len = min(seq_len, KV_TILE_SIZE);
  int cp_finished_seq_len = 0;

  // Currently assume that PAGE_SIZE is a multiplier of KV_TILE_SIZE
  // so that we access a single page in one iteration
  static_assert(PAGE_SIZE % KV_TILE_SIZE == 0);

  int const TMA_TRANS_BYTES_Q =
      num_tokens * NUM_QO_PER_KV * HEAD_DIM * sizeof(T);
  int TMA_TRANS_BYTES_KV = curr_iter_len * HEAD_DIM * sizeof(T);

  //  define barries
  Barrier *q_barrier = reinterpret_cast<Barrier *>(smem + S_Q_BARRIER_OFFSET);
  Barrier *k_barrier = reinterpret_cast<Barrier *>(smem + S_K_BARRIER_OFFSET);
  Barrier *v_barrier = reinterpret_cast<Barrier *>(smem + S_V_BARRIER_OFFSET);
  Barrier *compute_done =
      reinterpret_cast<Barrier *>(smem + S_COMPUTE_DONE_OFFSET);

  // init barrier
  if (threadIdx.x == 0) {
    for (int i = 0; i < Kstages; i++) {
      initialize_barrier(q_barrier[i], 1);
      initialize_barrier(k_barrier[i], 1);
      initialize_barrier(v_barrier[i], 1);
      initialize_barrier(compute_done[i], 1);
    }
  }
  __syncthreads();

  if (warpgroup_id == NUM_WARPGROUPS - 1) {
    // prefetch
    // load q
    if (lane_idx == 0 && warp_idx % 4 == 0) {
      set_barrier_transaction_bytes(q_barrier[0], TMA_TRANS_BYTES_Q);
      tma_q.tma_cp_async(q_barrier[0], q_smem(0, 0), {0, 0});
    }

    wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);

    // load k and v
    if (lane_idx == 0 && warp_idx % 4 == 0) {
      set_barrier_transaction_bytes(k_barrier[0],
                                    curr_iter_len * HEAD_DIM * sizeof(T));
      set_barrier_transaction_bytes(v_barrier[0],
                                    curr_iter_len * HEAD_DIM * sizeof(T));
      int begin = cp_finished_seq_len;
      int end = begin + curr_iter_len;
      int boundary = seq_len - num_tokens;
      int cache_rows = (begin >= boundary) ? 0
                       : (end <= boundary) ? (end - begin)
                                           : (boundary - begin);

      int kv_rows = curr_iter_len - cache_rows;

      // load from tail page
      if (cache_rows < KV_TILE_SIZE) {
        int page = page_indices[begin / PAGE_SIZE];
        int in_page_row = begin % PAGE_SIZE;
        int coords[3] = {0, in_page_row, page};

        tma_paged_k_cache_tail_page.tma_cp_async(
            k_barrier[0], k_buffer_smem(0, 0), coords);
        tma_paged_v_cache_tail_page.tma_cp_async(
            v_barrier[0], v_buffer_smem(0, 0), coords);

      } else if (cache_rows > 0) {
        int page = page_indices[begin / PAGE_SIZE];
        int in_page_row = begin % PAGE_SIZE;
        int coords[3] = {0, in_page_row, page};

        tma_paged_k_cache.tma_cp_async(
            k_barrier[0], k_buffer_smem(0, 0), coords);
        tma_paged_v_cache.tma_cp_async(
            v_barrier[0], v_buffer_smem(0, 0), coords);
      }

      if (kv_rows > 0) {
        int src_row = begin + cache_rows - boundary;
        int per_token_stride = NUM_QO_PER_KV + 2 * NUM_KV_HEADS;

        // assume tma_k, tma_v both have qkv_ptr as src_ptr
        int coords_k[3] = {0, NUM_QO_PER_KV, 0};
        int coords_v[3] = {0, NUM_QO_PER_KV + NUM_KV_HEADS, 0};
        // each time copy a head (kv head is only 1)
        tma_k.tma_cp_async(
            k_barrier[0], k_buffer_smem(cache_rows, 0), coords_k);
        tma_v.tma_cp_async(
            v_barrier[0], v_buffer_smem(cache_rows, 0), coords_v);
      }
    }
    cp_finished_seq_len += curr_iter_len;

    // start loading next tile in kv smem
    for (int iter = 0; iter < num_iters - 1; iter++) {
      int phase = ((iter + 1) / Kstages) % 2;
      int slot = (iter + 1) % Kstages;
      wait(compute_done[0], phase);

      int next_iter_len = iter + 1 < num_iters
                              ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                              : 0;
      if (next_iter_len > 0) {

        if (lane_idx == 0 && warp_idx % 4 == 0) {
          set_barrier_transaction_bytes(k_barrier[slot],
                                        next_iter_len * HEAD_DIM * sizeof(T));
          set_barrier_transaction_bytes(v_barrier[slot],
                                        next_iter_len * HEAD_DIM * sizeof(T));
          int begin = cp_finished_seq_len;
          int end = begin + next_iter_len;
          int boundary = seq_len - num_tokens;
          int cache_rows = (begin >= boundary) ? 0
                           : (end <= boundary) ? (end - begin)
                                               : (boundary - begin);

          int kv_rows = next_iter_len - cache_rows;

          // load from tail page
          if (cache_rows < KV_TILE_SIZE) {
            int page = page_indices[begin / PAGE_SIZE];
            int in_page_row = begin % PAGE_SIZE;
            int coords[3] = {0, in_page_row, page};

            tma_paged_k_cache_tail_page.tma_cp_async(
                k_barrier[slot], k_smem(0, 0), coords);
            tma_paged_v_cache_tail_page.tma_cp_async(
                v_barrier[slot], v_smem(0, 0), coords);

          } else if (cache_rows > 0) {
            // printf("load from paged kv cache\n");
            int page = page_indices[begin / PAGE_SIZE];
            int in_page_row = begin % PAGE_SIZE;
            int coords[3] = {0, in_page_row, page};

            tma_paged_k_cache.tma_cp_async(
                k_barrier[slot], k_smem(0, 0), coords);
            tma_paged_v_cache.tma_cp_async(
                v_barrier[slot], v_smem(0, 0), coords);
          }

          if (kv_rows > 0) {
            int src_row = begin + cache_rows - boundary;
            int per_token_stride = NUM_QO_PER_KV + 2 * NUM_KV_HEADS;

            int coords_k[3] = {0, NUM_QO_PER_KV, 0};
            int coords_v[3] = {0, NUM_QO_PER_KV + NUM_KV_HEADS, 0};
            // each time copy a head (kv head is only 1)
            tma_k.tma_cp_async(
                k_barrier[slot], k_smem(cache_rows, 0), coords_k);
            tma_v.tma_cp_async(
                v_barrier[slot], v_smem(cache_rows, 0), coords_v);
          }
        }
      }

      wait(compute_done[slot], phase);

      // rotate the buffers
      if ((iter & 0x1) == 0) {
        k_smem.set_ptr(s_k_buffer);
        k_buffer_smem.set_ptr(s_k);
        v_smem.set_ptr(s_v_buffer);
        v_buffer_smem.set_ptr(s_v);
      } else {
        k_smem.set_ptr(s_k);
        k_buffer_smem.set_ptr(s_k_buffer);
        v_smem.set_ptr(s_v);
        v_buffer_smem.set_ptr(s_v_buffer);
      }
      wg_sync<THREADS_PER_WARPGROUP * PRODUCER_WARPGROUPS>(8);
    }

  } else {

    float m_local[MMA_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      m_local[m][0] = -inf;
      m_local[m][1] = -inf;
    }
    float d[MMA_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      d[m][0] = 1.f;
      d[m][1] = 1.f;
    }

    float o[MMA_ITERS_M][HEAD_DIM / 64][32];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
      for (int n = 0; n < HEAD_DIM / 64; n++) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          clear_8_floats(o[m][n] + k * 8);
        }
      }
    }

    wait(q_barrier[0], 0);

    for (int iter = 0; iter < num_iters; iter++) {

      int next_iter_len = iter + 1 < num_iters
                              ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                              : 0;

      int phase = (iter / Kstages) % 2;
      int slot = iter % Kstages;

      wait(k_barrier[slot], phase);

      wait(v_barrier[slot], phase);

      // rotate the buffers
      if ((iter & 0x1) == 0) {
        k_smem.set_ptr(s_k_buffer);
        k_buffer_smem.set_ptr(s_k);
        v_smem.set_ptr(s_v_buffer);
        v_buffer_smem.set_ptr(s_v);
      } else {
        k_smem.set_ptr(s_k);
        k_buffer_smem.set_ptr(s_k_buffer);
        v_smem.set_ptr(s_v);
        v_buffer_smem.set_ptr(s_v_buffer);
      }

      int kv_tokens_to_process = min(
          curr_iter_len,
          max(iter * KV_TILE_SIZE + curr_iter_len - (seq_len - num_tokens), 0));
      int first_kv_token_to_process =
          iter * KV_TILE_SIZE + curr_iter_len - kv_tokens_to_process;
      if (qk_norm) {
        // Q norm
        if (iter == 0) {
          rms_norm_wg<T,
                      QOSmem,
                      NUM_QO_PER_KV,
                      HEAD_DIM,
                      THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
              q_smem,
              static_cast<T const *>(q_norm_weight_ptr),
              s_q_norm_sum,
              q_eps,
              num_tokens /*window_size*/,
              0 /*token_offset*/,
              rope,
              static_cast<T const *>(cos_ptr) +
                  (seq_len - num_tokens) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (seq_len - num_tokens) * HEAD_DIM,
              9);
        }
        // K norm
        if (kv_tokens_to_process > 0) {
          rms_norm_wg<T,
                      KVSmem,
                      1,
                      HEAD_DIM,
                      THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
              k_smem,
              static_cast<T const *>(k_norm_weight_ptr),
              s_k_norm_sum,
              k_eps,
              kv_tokens_to_process /*window_size*/,
              curr_iter_len - kv_tokens_to_process,
              rope,
              static_cast<T const *>(cos_ptr) +
                  first_kv_token_to_process * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  first_kv_token_to_process * HEAD_DIM,
              9);
        }
      } else if (rope) {
        if (iter == 0) {
#pragma unroll
          for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
            // q rope
            rotary_embedding_wg<T,
                                QOSmem,
                                NUM_QO_PER_KV,
                                HEAD_DIM,
                                THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
                q_smem,
                static_cast<T const *>(cos_ptr) +
                    (token_idx + seq_len - num_tokens) * HEAD_DIM,
                static_cast<T const *>(sin_ptr) +
                    (token_idx + seq_len - num_tokens) * HEAD_DIM,
                token_idx * NUM_QO_PER_KV,
                9);
          }
        }
        if (kv_tokens_to_process > 0) {
          for (int token_idx = 0; token_idx < kv_tokens_to_process;
               token_idx++) {
            // k rope
            rotary_embedding_wg<T,
                                KVSmem,
                                1,
                                HEAD_DIM,
                                THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(
                k_smem,
                static_cast<T const *>(cos_ptr) +
                    (token_idx + first_kv_token_to_process) * HEAD_DIM,
                static_cast<T const *>(sin_ptr) +
                    (token_idx + first_kv_token_to_process) * HEAD_DIM,
                token_idx + curr_iter_len - kv_tokens_to_process,
                9);
          }
        }
      }

      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);

      // update the KV Cache
      if (kv_tokens_to_process > 0) {
        int page_idx = page_indices[first_kv_token_to_process / PAGE_SIZE];
        for (int elem_idx = threadIdx.x;
             elem_idx < kv_tokens_to_process * HEAD_DIM;
             elem_idx += NUM_THREADS) {
          int token_idx = elem_idx / HEAD_DIM;
          int col = elem_idx % HEAD_DIM;
          int page_offset = (token_idx + first_kv_token_to_process) % PAGE_SIZE;
          int src_row = (token_idx + first_kv_token_to_process) % KV_TILE_SIZE;
          int dst_row = page_idx * PAGE_SIZE + page_offset;
          paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
          paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
        }
      }
      // compute X = QK^T
      // NOTE(Yu): we use m64n64k16 mma atom, and wrapped it as m64n64kK mma,
      // i.e. we don't need to iterate over k explicitly
      float x_frag_f[MMA_ITERS_M][32];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          clear_8_floats(x_frag_f[m] + k * 8);
        }
      }
      for (int m = 0; m < MMA_ITERS_M; m++) {
        Q_DESC q_desc(q_smem(m * 64, 0));
        K_DESC k_desc(k_smem(0, 0));

        wgmma::warpgroup_arrive();
        wgmma::mma<T, 64, 64, 16, QOSmem, KVSmem, Q_DESC, K_DESC, false, false>(
            x_frag_f[m], q_desc, k_desc);
        wgmma::mma_commit_group();
        wgmma::mma_async_wait();
      }

      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);

      // update m_local: get partial max
      // NOTE(Yu): We do 64x64x16 mma, and each thread saves 32 register values,
      // each thread maintains MMA_ITERS_M * 2 partial max values. For a given
      // m, the first value is the maximum of x_frag_f[m][0, 1, 4, 5, 8, 9, 12,
      // 13, 16, 17, 20, 21, 24, 25, 28, 29], and the second value is the
      // maximum of x_frag_f[m][2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26,
      // 27, 30, 31]
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#:~:text=Figure%20149%20WGMMA%20.m64nNk16%20register%20fragment%20layout%20for%20accumulator%20matrix%20D.
      float m_prev[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        m_prev[m][0] = m_local[m][0];
        m_prev[m][1] = m_local[m][1];
#pragma unroll
        for (int frag_idx = 0; frag_idx < 32; frag_idx++) {
          int row = (m << 6) + (warp_idx << 4) + (lane_idx >> 2) +
                    (((frag_idx & 0x3) >> 1) << 3);
          int col = ((lane_idx & 0x3) << 1) + ((frag_idx >> 2) << 3) +
                    (frag_idx & 0x1);
          int token_idx = row / NUM_QO_PER_KV;
          bool is_valid =
              (row < num_tokens * NUM_QO_PER_KV) &&
              (col + iter * KV_TILE_SIZE <= token_idx + seq_len - num_tokens);
          x_frag_f[m][frag_idx] = is_valid ? x_frag_f[m][frag_idx] : -inf;
          m_local[m][(frag_idx & 0x3) >> 1] =
              max(m_local[m][(frag_idx & 0x3) >> 1], x_frag_f[m][frag_idx]);
        }
      }
      // update m_local: get local max across 4 threads in a row
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x1));
        m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x2));
        m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x1));
        m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x2));
      }

      float rescale[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        rescale[m][0] =
            expf(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
        rescale[m][1] =
            expf(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
      }

      // update d: get partial sum
      float d_partial[MMA_ITERS_M][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        d_partial[m][0] = 0.f;
        d_partial[m][1] = 0.f;
#pragma unroll
        for (int frag_idx = 0; frag_idx < 32; frag_idx++) {
          x_frag_f[m][frag_idx] =
              x_frag_f[m][frag_idx] != -inf
                  ? expf(x_frag_f[m][frag_idx] * sm_scale -
                         m_local[m][(frag_idx & 0x3) >> 1] * sm_scale)
                  : 0.f;
          d_partial[m][(frag_idx & 0x3) >> 1] += x_frag_f[m][frag_idx];
        }
      }
      // update d: get local sum across 4 threads in a row
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
        d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x1);
        d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x2);
        d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x1);
        d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x2);
        d[m][0] *= rescale[m][0];
        d[m][1] *= rescale[m][1];
        d[m][0] += d_partial[m][0];
        d[m][1] += d_partial[m][1];
      }

      // update o: rescale
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        for (int n = 0; n < HEAD_DIM / 64; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 32; frag_idx++) {
            o[m][n][frag_idx] *= rescale[m][(frag_idx & 0x3) >> 1];
          }
        }
      }

      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);

      uint32_t x_frag[MMA_ITERS_M][16];
      for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
        convert_32_f32_to_16_bf16_uint32(x_frag_f[m], x_frag[m]);
        for (int n = 0; n < HEAD_DIM / 64; n++) {
          V_DESC v_desc(v_smem(m * 64, n * 64));
          wgmma::warpgroup_arrive();
          wgmma::mma_rs<T, 64, 64, 16, KVSmem, V_DESC, true>(
              o[m][n], x_frag[m], v_desc);
          wgmma::mma_commit_group();
          wgmma::mma_async_wait();
        }
      }

      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);

      curr_iter_len = next_iter_len;

      if (warp_idx == 0 && lane_idx == 0) {
        arrive(compute_done[slot], 1);
      }
    }

    // write intermediate results to buffer in shared memory
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      m_local[m][0] *= m_local[m][0] != -inf ? sm_scale : 1.f;
      m_local[m][1] *= m_local[m][1] != -inf ? sm_scale : 1.f;
      s_m_buffer[m * CONSUMER_WARPGROUPS * THREADS_PER_WARPGROUP * 2 +
                 threadIdx.x * 2] = m_local[m][0];
      s_m_buffer[m * CONSUMER_WARPGROUPS * THREADS_PER_WARPGROUP * 2 +
                 threadIdx.x * 2 + 1] = m_local[m][1];
      s_d_buffer[m * CONSUMER_WARPGROUPS * THREADS_PER_WARPGROUP * 2 +
                 threadIdx.x * 2] = d[m][0];
      s_d_buffer[m * CONSUMER_WARPGROUPS * THREADS_PER_WARPGROUP * 2 +
                 threadIdx.x * 2 + 1] = d[m][1];
      for (int n = 0; n < HEAD_DIM / 64; n++) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < 32; frag_idx++) {
          s_o_buffer[m * CONSUMER_WARPGROUPS * THREADS_PER_WARPGROUP * 64 +
                     threadIdx.x * 64 + n * 32 + frag_idx] = o[m][n][frag_idx];
        }
      }
    }
    wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);

    // get global m, d, and o
    // each thread handles an element in o in each iteration
    for (int elem_idx = threadIdx.x;
         elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
         elem_idx += NUM_THREADS) {
      int row = elem_idx / HEAD_DIM;
      int col = elem_idx % HEAD_DIM;
      int t_idx = (row / 16) * 32 + (row % 8) * 4 + (col % 8) / 2;
      int mma_iter_n = col / 64;

      int frag_idx = ((col % 64) / 8) * 4 + ((row % 16) / 8) * 2 + (col % 2);

      float m_global = -inf;
      float d_global = 1.f;
      float o_global = 0.f;

      int md_smem_offset = (row / 64) * CONSUMER_WARPGROUPS *
                               THREADS_PER_WARPGROUP * 2 // mma iter m
                           + t_idx * 2                   // corresponding thread
                           + (frag_idx % 4) / 2; // first half or second half

      m_global = s_m_buffer[md_smem_offset];
      d_global = s_d_buffer[md_smem_offset];
      o_global = s_o_buffer[(row / 64) * CONSUMER_WARPGROUPS *
                                THREADS_PER_WARPGROUP * 64 // mma iter m
                            + t_idx * 64      // corresponding thread
                            + mma_iter_n * 32 // mma iter n
                            + frag_idx];

      o_smem.at(row, col) = bfloat16(o_global / d_global);
    }
    wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(9);

    async_proxy_fence();

    // copy back to dmem
    if (warp_idx % 4 == 0 && lane_id() == 0) {
      tma_output.tma_store_async(o_smem(0, 0), {0, 0});
      store_commit_group();
    }
  }
}
} // namespace kernel