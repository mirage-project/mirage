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
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "norm.cuh"
#include "reduction.cuh"
#include "rotary_embedding.cuh"
#include "smem_layout.cuh"
#include "tasks/common/common_header.cuh"
#include "tasks/hopper/norm_hopper.cuh"
#include "tasks/hopper/rotary_embedding_hopper.cuh"
#include "tasks/hopper/utils.cuh"

namespace kernel {

// NOTE(Jinchen): this task implements the paged attention where a causal mask
// is applied. In each task, we process one request with one or more tokens
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int SEQ_LEN,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8,
          bool PARTITION_KV = true,
          int NUM_KV_CHUNKS = 1>
__device__ __forceinline__ void
    multitoken_paged_attention_task_impl_32_64_split_kv(
        void const *qkv_ptr,
        void *paged_k_cache_ptr,
        void *paged_v_cache_ptr,
        void *output_ptr,
        int const *qo_indptr_buffer_ptr,
        int const *paged_kv_indptr_buffer_ptr,
        int const *paged_kv_indices_buffer_ptr,
        int const *paged_kv_last_page_len_buffer_ptr,
        int16_t request_id,
        bool qk_norm,
        bool rope,
        void const *q_norm_weight_ptr,
        void const *k_norm_weight_ptr,
        void const *cos_ptr,
        void const *sin_ptr,
        float q_eps,
        float k_eps,
        void *lse,
        int kv_idx) {

  if (threadIdx.x >= 128) {
    return;
  }

  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;

  constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
  constexpr int KV_TILE_SIZE = 128;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int THREADS_PER_WARPGROUP = 128;
  constexpr int CONSUMER_WARPGROUPS = 1;

  constexpr int CONSUMER_WARPGROUP_SYNC_BARRIER_ID = 6;

  // NOTE(Jinchen): we use m16n16k16 mma to compute matrix multiplication
  // constexpr int MMA_ITERS_M = (MAX_TOKENS * NUM_QO_PER_KV  + 15) / 16;

  constexpr int NUM_Q = MAX_TOKENS * NUM_QO_PER_KV;

  // constexpr int NUM_WARPS_Q = (NUM_Q  <= 16) ? 1 : 4;
  constexpr int HEAD_DIM_COPY_ITER = HEAD_DIM / CP_CHUNK_SIZE;

  constexpr int HEAD_DIM_ITER = HEAD_DIM / 16;

  constexpr int NUM_ITER_QK_N = (KV_TILE_SIZE / (16));

  constexpr int GLOBAL_ITERS_M = (NUM_Q + 64 - 1) / 64;

  // now we don't support iteration over Q
  //  assert(GLOBAL_ITERS_M == 1);
  //  the scale factor for normalization in softmax
  float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM)) * log2e;

  // size_t KV_CACHE_OFFSET = kv_idx * SEQ_LEN;

  int warp_idx = warp_id();
  int lane_idx = lane_id();

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  // Exit the current task is number of query tokens is zero
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  // NOTE(Jinchen): to simplify the implementation, we assume that the metadata
  // of the paged KV cache includes the new tokens, i.e., spaces are allocated
  // before the request is processed, while the real data is copied into the
  // corresponding pages after that
  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int seq_len = (num_pages - 1) * PAGE_SIZE +
                paged_kv_last_page_len_buffer_ptr[request_id];

  int const global_seq_len = seq_len;

  seq_len = PARTITION_KV ? (((seq_len - kv_idx * SEQ_LEN) >= SEQ_LEN)
                                ? SEQ_LEN
                                : (seq_len - kv_idx * SEQ_LEN))
                         : seq_len;

  if (seq_len <= 0) {
    return;
  }

  int kv_cache_offset = PARTITION_KV ? kv_idx * SEQ_LEN : 0;
  // if(threadIdx.x == 0){
  //   printf("seq_len %d, global_seq_len %d, kv_cache_offset %d\n", seq_len,
  //   global_seq_len, kv_cache_offset);
  // }

  // valid_lens = [seq_len - num_tokens + 1 + i for i in range(num_tokens)]
  // seq_len = 7 * 64 + 64 = 512
  // num tokens = 8
  // Load the paged KV indices into shared memory
  __shared__ __align__(16) int page_indices[MAX_PAGES_PER_REQUEST];
#pragma unroll
  for (int i = threadIdx.x; i < num_pages * sizeof(int) / 16;
       i += NUM_THREADS) {
    __uint128_t const *src_ptr =
        reinterpret_cast<__uint128_t const *>(paged_kv_indices_buffer_ptr) + i;
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
  wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);

  T const *__restrict__ d_q =
      reinterpret_cast<T const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  T const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  T const *__restrict__ d_v = d_k + HEAD_DIM;
  T *__restrict__ d_paged_k_cache = reinterpret_cast<T *>(paged_k_cache_ptr);
  T *__restrict__ d_paged_v_cache = reinterpret_cast<T *>(paged_v_cache_ptr);
  T *__restrict__ d_output =
      reinterpret_cast<T *>(output_ptr) + first_token_pos * O_STRIDE;

  // DTensors' layouts
  using QDmem =
      dmem_row_const<T, MAX_TOKENS, HEAD_DIM * NUM_QO_PER_KV, QKV_STRIDE>;
  using KVDmem = dmem_row_const<T, MAX_TOKENS, HEAD_DIM, QKV_STRIDE>;
  using KVCacheDmem = dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_CACHE_STRIDE>;
  using ODmem = dmem_row<T, MAX_TOKENS, HEAD_DIM * NUM_QO_PER_KV, O_STRIDE>;

  QDmem q_dmem(d_q);
  KVDmem k_dmem(d_k), v_dmem(d_v);
  KVCacheDmem paged_k_cache_dmem(d_paged_k_cache),
      paged_v_cache_dmem(d_paged_v_cache);
  ODmem o_dmem(d_output);

  // STensors' offsets and sizes
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  constexpr size_t ZERO_BUFFER_SIZE = sizeof(T) * 8;

  constexpr size_t S_Q_OFFSET = ZERO_BUFFER_OFFSET + ZERO_BUFFER_SIZE;
  constexpr size_t S_Q_SIZE = sizeof(T) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;

  constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;

  constexpr size_t S_K_BUFFER_OFFSET = S_K_OFFSET + S_K_SIZE;
  constexpr size_t S_K_BUFFER_SIZE = S_K_SIZE;

  constexpr size_t S_V_OFFSET = S_K_BUFFER_OFFSET + S_K_BUFFER_SIZE;
  constexpr size_t S_V_SIZE = S_K_SIZE;

  constexpr size_t S_V_BUFFER_OFFSET = S_V_OFFSET + S_V_SIZE;
  constexpr size_t S_V_BUFFER_SIZE = S_K_SIZE;

  // align to size of float
  constexpr size_t S_Q_NORM_SUM_OFFSET =
      ((S_V_BUFFER_OFFSET + S_V_BUFFER_SIZE + sizeof(float) - 1) &
       ~size_t(sizeof(float) - 1));
  constexpr size_t S_Q_NORM_SUM_SIZE =
      sizeof(float) * 4; // 4 floats for 4 warps

  constexpr size_t S_K_NORM_SUM_OFFSET =
      S_Q_NORM_SUM_OFFSET + S_Q_NORM_SUM_SIZE;

  constexpr size_t S_K_NORM_SUM_SIZE = sizeof(float) * 4;

  // stage2 flash meta buffer

  constexpr size_t S_M_BUFFER_OFFSET = ZERO_BUFFER_OFFSET + ZERO_BUFFER_SIZE;
  constexpr size_t S_M_BUFFER_SIZE =
      sizeof(float) * GLOBAL_ITERS_M * NUM_THREADS * 2;

  constexpr size_t S_D_BUFFER_OFFSET = S_M_BUFFER_OFFSET + S_M_BUFFER_SIZE;
  constexpr size_t S_D_BUFFER_SIZE =
      sizeof(float) * GLOBAL_ITERS_M * NUM_THREADS * 2;

  constexpr size_t S_O_BUFFER_OFFSET = S_D_BUFFER_OFFSET + S_D_BUFFER_SIZE;
  constexpr size_t S_O_BUFFER_SIZE =
      sizeof(float) * GLOBAL_ITERS_M * NUM_THREADS * 64;

  // stage3 output buffer
  constexpr size_t S_O_OFFSET = ZERO_BUFFER_OFFSET + ZERO_BUFFER_OFFSET;
  // constexpr size_t S_O_SIZE = S_Q_SIZE;

  constexpr size_t S_TOTAL_OFFSET =
      (S_O_BUFFER_OFFSET + S_O_BUFFER_SIZE >
       S_K_NORM_SUM_OFFSET + S_K_NORM_SUM_SIZE)
          ? (S_O_BUFFER_OFFSET + S_O_BUFFER_SIZE)
          : (S_K_NORM_SUM_OFFSET + S_K_NORM_SUM_SIZE);
  // if (threadIdx.x == 0){
  //   printf("S_TOTAL_OFFSET %llu, MAX_DYNAMIC_SHARED_MEMORY_SIZE %llu\n",
  //   S_TOTAL_OFFSET, mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  // }
  assert(S_TOTAL_OFFSET <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);

  extern __shared__ char smem[];

  T *zero_buf = reinterpret_cast<T *>(smem + ZERO_BUFFER_OFFSET);
  clear_smem_buffer<T, 8>(zero_buf);
  T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
  T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
  T *s_k_buffer = reinterpret_cast<T *>(smem + S_K_BUFFER_OFFSET);
  T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
  T *s_v_buffer = reinterpret_cast<T *>(smem + S_V_BUFFER_OFFSET);
  T *s_o = reinterpret_cast<T *>(smem + S_O_OFFSET);

  float *s_q_norm_sum = reinterpret_cast<float *>(smem + S_Q_NORM_SUM_OFFSET);
  float *s_k_norm_sum = reinterpret_cast<float *>(smem + S_K_NORM_SUM_OFFSET);
  // float *s_m_buffer = reinterpret_cast<float *>(smem + S_M_BUFFER_OFFSET);
  // float *s_d_buffer = reinterpret_cast<float *>(smem + S_D_BUFFER_OFFSET);
  // float *s_o_buffer = reinterpret_cast<float *>(smem + S_O_BUFFER_OFFSET);

  // STensors' layouts
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using QOSmem = smem_row_tiled<T,
                                3,
                                3,
                                3,
                                MAX_TOKENS * NUM_QO_PER_KV,
                                HEAD_DIM,
                                HEAD_DIM>;
  using KVSmem = smem_row_tiled<T, 3, 3, 3, KV_TILE_SIZE, HEAD_DIM, HEAD_DIM>;

  // using QSmem = smem_row_tiled_new<T, 3, 3, 3, MAX_TOKENS * NUM_QO_PER_KV,
  // HEAD_DIM, HEAD_DIM>;
  ZeroBufferSmem zero_buffer(zero_buf);
  QOSmem q_smem(s_q);
  QOSmem o_smem(s_o);
  KVSmem k_smem(s_k), v_smem(s_v);
  KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);

  int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
  int curr_iter_len = min(seq_len, KV_TILE_SIZE);

  // printf("num_iters %d, curr_iter_len %d, seq_len %d, num_tokens %d\n",
  // num_iters, curr_iter_len, seq_len, num_tokens);
  // PAGE_SIZE = 4
  // if(threadIdx.x == 0){
  //   printf("seq_length %d, %d, %d, %d, %d\n", seq_len, KV_TILE_SIZE,
  //   num_pages, PAGE_SIZE, paged_kv_last_page_len_buffer_ptr[request_id]);
  // }
  int cp_finished_seq_len = 0;
  // assert no leafover to be handled when loading qkv
  static_assert(HEAD_DIM % CP_CHUNK_SIZE == 0);

  // Currently assume that PAGE_SIZE is a multiplier of KV_TILE_SIZE
  // so that we access a single page in one iteration
  static_assert(PAGE_SIZE % KV_TILE_SIZE == 0);

  // 8 * 4 * 16(8 bytes per load)
#pragma unroll
  for (int chunk_idx = threadIdx.x;
       chunk_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM_COPY_ITER;
       chunk_idx += NUM_THREADS) {

    int src_row = chunk_idx / (NUM_QO_PER_KV * HEAD_DIM_COPY_ITER);
    int src_col =
        (chunk_idx % (NUM_QO_PER_KV * HEAD_DIM_COPY_ITER)) * CP_CHUNK_SIZE;

    int dst_row = src_row * NUM_QO_PER_KV + src_col / HEAD_DIM;
    int dst_col = src_col % HEAD_DIM;
    load_smem(q_smem(dst_row, dst_col), (q_dmem(src_row, src_col)));
  }

  int page_idx_0 = page_indices[kv_cache_offset / PAGE_SIZE];
#pragma unroll
  for (int chunk_idx = threadIdx.x;
       chunk_idx < curr_iter_len * HEAD_DIM_COPY_ITER;
       chunk_idx += NUM_THREADS) {
    int dst_row = chunk_idx / HEAD_DIM_COPY_ITER;
    int col = (chunk_idx % HEAD_DIM_COPY_ITER) * CP_CHUNK_SIZE;
    if (dst_row + cp_finished_seq_len + kv_cache_offset <
        global_seq_len - num_tokens) {
      // load from KV Cache
      // int page_idx = page_indices[(dst_row + cp_finished_seq_len) /
      // PAGE_SIZE];
      int page_offset =
          (dst_row + cp_finished_seq_len + kv_cache_offset) % PAGE_SIZE;
      int src_row = page_idx_0 * PAGE_SIZE + page_offset;
      load_smem(k_buffer_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
      load_smem(v_buffer_smem(dst_row, col), paged_v_cache_dmem(src_row, col));

      // if((float)k_buffer_smem.at(dst_row, col) >=  0.2f){
      //   printf("kv idx %d, dst_row %d, dst_col %d, src_row %d, value %f\n",
      //   kv_idx, dst_row, col, src_row, (float)k_buffer_smem.at(dst_row,
      //   col));
      // }

    } else {
      // load from QKV
      int src_row = dst_row + cp_finished_seq_len - (seq_len - num_tokens);
      load_smem(k_buffer_smem(dst_row, col), k_dmem(src_row, col));
      load_smem(v_buffer_smem(dst_row, col), v_dmem(src_row, col));
      // if((float)k_buffer_smem.at(dst_row, col) >=  0.2f){
      //   printf("kv idx %d, dst_row %d, dst_col %d, src_row %d, value %f\n",
      //   kv_idx, dst_row, col, src_row, (float)k_buffer_smem.at(dst_row,
      //   col));
      // }
    }
  }
  cp_async_fence();
  cp_finished_seq_len += curr_iter_len;

  float m_local[GLOBAL_ITERS_M][2]; // 2 * 128 threads = 256
#pragma unroll
  for (int m = 0; m < GLOBAL_ITERS_M; m++) {
    m_local[m][0] = -inf;
    m_local[m][1] = -inf;
  }
  float d[GLOBAL_ITERS_M][2];
#pragma unroll
  for (int m = 0; m < GLOBAL_ITERS_M; m++) {
    d[m][0] = 1.f;
    d[m][1] = 1.f;
  }
  float o[GLOBAL_ITERS_M][HEAD_DIM / 16][8];
#pragma unroll
  for (int m = 0; m < GLOBAL_ITERS_M; m++) {
#pragma unroll
    for (int n = 0; n < HEAD_DIM_ITER; n++) {
      clear_8_floats(o[m][n]);
    }
  }

  for (int iter = 0; iter < num_iters; iter++) {
    int next_iter_len = iter + 1 < num_iters
                            ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                            : 0;
    if (next_iter_len > 0) {
      int page_idx =
          page_indices[(cp_finished_seq_len + kv_cache_offset) / PAGE_SIZE];

#pragma unroll
      for (int chunk_idx = threadIdx.x;
           chunk_idx < curr_iter_len * HEAD_DIM_COPY_ITER;
           chunk_idx += NUM_THREADS) {
        int dst_row = chunk_idx / HEAD_DIM_COPY_ITER;
        int col = (chunk_idx % HEAD_DIM_COPY_ITER) * CP_CHUNK_SIZE;
        if (dst_row + cp_finished_seq_len + kv_cache_offset <
            global_seq_len - num_tokens) {
          // load from KV Cache
          // int page_idx =
          //    page_indices[(dst_row + cp_finished_seq_len) / PAGE_SIZE];
          int page_offset =
              (dst_row + cp_finished_seq_len + kv_cache_offset) % PAGE_SIZE;
          int src_row = page_idx * PAGE_SIZE + page_offset;

          // if((float)k_buffer_smem.at(dst_row, col) >= 0.2f){
          //   printf("kv idx %d, dst_row %d, dst_col %d, src_row %d, value
          //   %f\n", kv_idx, dst_row, col, src_row,
          //   (float)k_buffer_smem.at(dst_row, col));
          // }
          load_smem(k_smem(dst_row, col), (paged_k_cache_dmem(src_row, col)));
          load_smem(v_smem(dst_row, col), (paged_v_cache_dmem(src_row, col)));
        } else {
          // load from QKV

          int src_row = dst_row + cp_finished_seq_len - (seq_len - num_tokens);
          // if(threadIdx.x == 0){
          // printf("blockIdx.x %d, src_row %d, dst_row %d, col %d\n",
          // blockIdx.x, src_row, dst_row, col);
          // }
          // if((float)k_buffer_smem.at(dst_row, col) >= 0.2f){
          //   printf("kv idx %d, dst_row %d, dst_col %d, src_row %d, value
          //   %f\n", kv_idx, dst_row, col, src_row,
          //   (float)k_buffer_smem.at(dst_row, col));
          // }
          load_smem(k_smem(dst_row, col), (k_dmem(src_row, col)));
          load_smem(v_smem(dst_row, col), (v_dmem(src_row, col)));
        }
      }
      cp_async_fence();
      cp_async_wait<1>();
      cp_finished_seq_len += next_iter_len;
    } else {
      cp_async_wait<0>();
    }

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
    wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);

    int kv_tokens_to_process =
        min(curr_iter_len,
            max(iter * KV_TILE_SIZE + curr_iter_len + kv_cache_offset -
                    (global_seq_len - num_tokens),
                0));
    int first_kv_token_to_process =
        iter * KV_TILE_SIZE + curr_iter_len - kv_tokens_to_process;

    if (qk_norm) {
      // Q norm
      if (iter == 0) {
        rms_norm_hopper<T,
                        QOSmem,
                        NUM_QO_PER_KV,
                        HEAD_DIM,
                        THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS,
                        CONSUMER_WARPGROUP_SYNC_BARRIER_ID>(
            q_smem,
            static_cast<T const *>(q_norm_weight_ptr),
            s_q_norm_sum,
            q_eps,
            num_tokens /*window_size*/,
            0 /*token_offset*/,
            rope,
            static_cast<T const *>(cos_ptr) +
                (global_seq_len - num_tokens) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) +
                (global_seq_len - num_tokens) * HEAD_DIM);
      }
      // K norm
      if (kv_tokens_to_process > 0) {
        rms_norm_hopper<T,
                        KVSmem,
                        1,
                        HEAD_DIM,
                        THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS,
                        CONSUMER_WARPGROUP_SYNC_BARRIER_ID>(
            k_smem,
            static_cast<T const *>(k_norm_weight_ptr),
            s_k_norm_sum,
            k_eps,
            kv_tokens_to_process /*window_size*/,
            curr_iter_len - kv_tokens_to_process,
            rope,
            static_cast<T const *>(cos_ptr) +
                (first_kv_token_to_process + kv_cache_offset) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) +
                (first_kv_token_to_process + kv_cache_offset) * HEAD_DIM);
      }
    } else if (rope) {
      if (iter == 0) {
#pragma unroll
        for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
          // q rope
          rotary_embedding_hopper<T,
                                  QOSmem,
                                  NUM_QO_PER_KV,
                                  1,
                                  HEAD_DIM,
                                  THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS,
                                  CONSUMER_WARPGROUP_SYNC_BARRIER_ID>(
              q_smem,
              static_cast<T const *>(cos_ptr) +
                  (token_idx + global_seq_len - num_tokens) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (token_idx + global_seq_len - num_tokens) * HEAD_DIM,
              token_idx);
        }
      }
      if (kv_tokens_to_process > 0) {
        for (int token_idx = 0; token_idx < kv_tokens_to_process; token_idx++) {
          // k rope
          rotary_embedding_hopper<T,
                                  KVSmem,
                                  1,
                                  1,
                                  HEAD_DIM,
                                  THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS,
                                  CONSUMER_WARPGROUP_SYNC_BARRIER_ID>(
              k_smem,
              static_cast<T const *>(cos_ptr) +
                  (token_idx + first_kv_token_to_process + kv_cache_offset) *
                      HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (token_idx + first_kv_token_to_process + kv_cache_offset) *
                      HEAD_DIM,
              token_idx + curr_iter_len - kv_tokens_to_process);
        }
      }
    }

    wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);

    // update the KV Cache
    if (kv_tokens_to_process > 0) {
      int page_idx =
          page_indices[(first_kv_token_to_process + kv_cache_offset) /
                       PAGE_SIZE];
      for (int elem_idx = threadIdx.x;
           elem_idx < kv_tokens_to_process * HEAD_DIM;
           elem_idx += NUM_THREADS) {
        int token_idx = elem_idx / HEAD_DIM;
        int col = elem_idx % HEAD_DIM;
        // int page_idx = page_indices[(token_idx + first_kv_token_to_process) /
        // PAGE_SIZE];
        int page_offset =
            (token_idx + first_kv_token_to_process + kv_cache_offset) %
            PAGE_SIZE;
        int src_row = (token_idx + first_kv_token_to_process) % KV_TILE_SIZE;
        int dst_row = page_idx * PAGE_SIZE + page_offset;
        paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
        paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
      }
    }

    // printf("kv idx %d, kv cache first value %f, kv_tokens_to_process %d\n",
    // kv_idx, (float)paged_k_cache_dmem.at(0, 0), kv_tokens_to_process);

    // compute X = QK^T
    // NOTE(Jinchen): we use m16n16k16 mma, and let warp layout be
    // 1x4x1, so mma iterates over m and k dimensions

    // mma from 16X16X64 -> 64X16X16
    float x_frag_f[GLOBAL_ITERS_M][NUM_ITER_QK_N][8];
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      for (int n = 0; n < NUM_ITER_QK_N; n++) {
        clear_8_floats(x_frag_f[m][n]);
      }
    }
    uint32_t q_frag[4], kt_frag[4];

#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      // if partition Q across 4 warps we use 4 warps for q (64X16)
      // else we only process 16 Q tokens per mma instruction(16X16)
      //  int q_row = (m << 4) + (lane_idx & 0xF);
      int q_row = (m << 6) + (warp_idx << 4) + (lane_idx & 0xF);
#pragma unroll
      // loop for N dim when NUM_WARP_Q is 4
      for (int n = 0; n < NUM_ITER_QK_N; n++) {
        // if partition Q across 4 warps, we need to loop over this N
        // else we partition N across 4 warps(N is KV_CHUNK_SIZE = 64)
        int kt_col = (n << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
        for (int k = 0; k < HEAD_DIM_ITER; k++) {
          int q_col = (k << 4) + ((lane_idx >> 4) << 3);
          int kt_row = (k << 4) + (((lane_idx & 0xF) >> 3) << 3);
          T *src_ptr_Q = q_row < num_tokens * NUM_QO_PER_KV
                             ? q_smem(q_row, q_col)
                             : (zero_buffer(0, 0));
          T *src_ptr_KT = kt_col < curr_iter_len ? k_smem(kt_col, kt_row)
                                                 : (zero_buffer(0, 0));
          ldsm(src_ptr_Q, q_frag);
          ldsm(src_ptr_KT, kt_frag);

          // if(threadIdx.x == 0 ){
          //     printf("kv idx %d, q frag is n %d, k %d, value %f\n", kv_idx,
          //     n,k, (float)src_ptr_Q[0]); printf("kv idx %d, kt_frag frag is n
          //     %d, k %d, kt_col %d,curr_iter_len %d, value %f\n", kv_idx, n,k,
          //     kt_col, curr_iter_len , (float)src_ptr_KT[0]);
          // }
          mma_m16n16k16_bf16bf16bf32(
              x_frag_f[m][n], q_frag, kt_frag, x_frag_f[m][n]);
        }
      }
    }
    wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);

    // if(threadIdx.x == 0){
    //   printf("QKT, kv idx %d, x_frag_f[0][0][0] %f\n", kv_idx,
    //   x_frag_f[0][0][0]);
    // }

    float m_prev[GLOBAL_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      m_prev[m][0] = m_local[m][0];
      m_prev[m][1] = m_local[m][1];
#pragma unroll
      for (int n = 0; n < NUM_ITER_QK_N; n++) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
          // row_base = (m * 16) + (lane_idx / 4)
          // col_base = (warp_idx * 16) + ((lane_idx % 4) * 2)
          // row_offset = ((frag_idx % 4) / 2) * 8
          // col_offset = ((frag_idx / 4) * 8) + (frag_idx % 2)
          // todo: Xinhao, this is m << 4 or m << 6?
          int row = (m << 4) + (lane_idx >> 2) +
                    (((frag_idx & 0x3) >> 1) << 3) + (warp_idx << 4);
          int col = (n << 4) + ((lane_idx & 0x3) << 1) +
                    ((frag_idx >> 2) << 3) + (frag_idx & 0x1);
          int token_idx = row / NUM_QO_PER_KV;

          // col <= token idx + 512 - 8
          bool is_valid = (row < num_tokens * NUM_QO_PER_KV) &&
                          ((col + iter * KV_TILE_SIZE + kv_cache_offset) <=
                           (token_idx + global_seq_len - num_tokens));
          //  if(threadIdx.x < 64){
          //   printf("blockIdx.x %d, is valid %d, row %d, col%d, val %d,
          //   token_idx %d, global_seq_len %d, num_tokens %d, \n", blockIdx.x,
          //   is_valid, row, col, col + iter * KV_TILE_SIZE + kv_cache_offset,
          //   token_idx, global_seq_len, num_tokens);
          // }
          x_frag_f[m][n][frag_idx] = is_valid ? x_frag_f[m][n][frag_idx] : -inf;
          m_local[m][(frag_idx & 0x3) >> 1] =
              max(m_local[m][(frag_idx & 0x3) >> 1], x_frag_f[m][n][frag_idx]);
        }
      }
    }

// update m_local: get local max across 4 threads in a row
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x1));
      m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x2));
      m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x1));
      m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x2));
    }

    float rescale[GLOBAL_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      rescale[m][0] =
          ptx_exp2(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
      rescale[m][1] =
          ptx_exp2(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
    }

    // update d: get partial sum
    float d_partial[GLOBAL_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      d_partial[m][0] = 0.f;
      d_partial[m][1] = 0.f;
#pragma unroll
      for (int n = 0; n < NUM_ITER_QK_N; n++) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
          // x_frag_f[m][n][frag_idx] =
          //     x_frag_f[m][n][frag_idx] != -inf
          //         ? ptx_exp2(x_frag_f[m][n][frag_idx] * sm_scale -
          //                m_local[m][(frag_idx & 0x3) >> 1] * sm_scale)
          //         : 0.f;

          x_frag_f[m][n][frag_idx] =
              ptx_exp2(x_frag_f[m][n][frag_idx] * sm_scale -
                       m_local[m][(frag_idx & 0x3) >> 1] * sm_scale);

          // x_frag_f[m][n][frag_idx] = ptx_exp2(x_frag_f[m][n][frag_idx] *
          // sm_scale -  m_local[m][(frag_idx & 0x3) >> 1] * sm_scale);
          d_partial[m][(frag_idx & 0x3) >> 1] += x_frag_f[m][n][frag_idx];
        }
      }
    }
    // update d: get local sum across 4 threads in a row
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
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
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
#pragma unroll
      for (int n = 0; n < HEAD_DIM_ITER; n++) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
          o[m][n][frag_idx] *= rescale[m][(frag_idx & 0x3) >> 1];
        }
      }
    }
    // QK = 64 X 64, v = 64 X 128
    // 16 * 64 64 * 128 -> 16 * 128, K in different warps
    // 64 * 64 * 64 * 128, M in different warps

    // update o: compute O = exp(X - m) * V and accumulate
    // use m16n16k16 mma to compute and let warp layout be 1x1x4
    uint32_t x_frag[GLOBAL_ITERS_M][NUM_ITER_QK_N][4], v_frag[4];
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
#pragma unroll
      for (int n = 0; n < HEAD_DIM_ITER; n++) {
#pragma unroll
        for (int k = 0; k < NUM_ITER_QK_N; k++) {
          convert_f32_to_bf16_uint32(x_frag_f[m][k], x_frag[m][k]);
          int v_row = (k << 4) + (lane_idx & 0xF);
          int v_col = (n << 4) + ((lane_idx >> 4) << 3);
          T *src_ptr_V = v_row < curr_iter_len ? v_smem(v_row, v_col)
                                               : (zero_buffer(0, 0));
          ldsm_t(src_ptr_V, v_frag);
          mma_m16n16k16_bf16bf16bf32(o[m][n], x_frag[m][k], v_frag, o[m][n]);
        }
      }
    }
    wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);
    curr_iter_len = next_iter_len;
  }

  // update m
  if (PARTITION_KV) {
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
      m_local[m][0] *= m_local[m][0] != -inf ? sm_scale : 1.f;
      m_local[m][1] *= m_local[m][1] != -inf ? sm_scale : 1.f;
    }
  }

  // result: 64 * 128
#pragma unroll
  for (int m = 0; m < GLOBAL_ITERS_M; m++) {
#pragma unroll
    for (int i = 0; i < HEAD_DIM_ITER; i++) {
      int row = ((threadIdx.x >> 5) << 4) + (lane_idx >> 2);
      int col = ((lane_idx & 3) << 1) + (i << 4);

      float d0 = d[m][0];
      float d1 = d[m][1];

      int const c16 = col >> 4; // same for col, col+8

      o_smem.at(row, col) = bfloat16(o[m][c16][0] / d0);
      o_smem.at(row, col + 1) = bfloat16(o[m][c16][1] / d0);

      o_smem.at(row + 8, col) = bfloat16(o[m][c16][2] / d1);
      o_smem.at(row + 8, col + 1) = bfloat16(o[m][c16][3] / d1);

      o_smem.at(row, col + 8) = bfloat16(o[m][c16][4] / d0);
      o_smem.at(row, col + 9) = bfloat16(o[m][c16][5] / d0);

      o_smem.at(row + 8, col + 8) = bfloat16(o[m][c16][6] / d1);
      o_smem.at(row + 8, col + 9) = bfloat16(o[m][c16][7] / d1);
    }
  }
  wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);

  // store the output
  // 32 * 128
  // 32, 128
  // dst_row = 32 / 4
  // dst_col = 128 + (32 % 4) * 128
  // 8 * 4 * 128 head1, head2, head3 head4, 8 token 4 heads
  for (int elem_idx = threadIdx.x;
       elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
       elem_idx += NUM_THREADS) {
    // int src_row = (elem_idx / HEAD_DIM) % 2;
    int src_row = elem_idx / HEAD_DIM;
    int src_col = elem_idx % HEAD_DIM;
    int dst_row = src_row / NUM_QO_PER_KV;
    int dst_col = src_col + (src_row % NUM_QO_PER_KV) * HEAD_DIM;

    // printf("blockIdx.x %d, threadIdx.x %d, dst_row %d, dst_col %d, val
    // %f\n",blockIdx.x,  threadIdx.x, dst_row, dst_col,
    // (float)o_smem.at(src_row, src_col));

    o_dmem.at(dst_row, dst_col) = o_smem.at(src_row, src_col);
  }

  // store the log exp sum if use split KV

  if constexpr (PARTITION_KV) {
#pragma unroll
    for (int m = 0; m < GLOBAL_ITERS_M; m++) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        int idx = m * 64 + warp_idx * 16 + j * 8 + lane_idx / 4;
        if (idx < (num_tokens * NUM_QO_HEADS)) {

          int token_idx = idx / NUM_QO_HEADS;
          int head_idx = idx % NUM_QO_HEADS;

          int offset = head_idx +
                       token_idx * NUM_KV_CHUNKS * NUM_QO_HEADS * NUM_QO_GROUPS;

          reinterpret_cast<float *>(lse)[offset] =
              ptx_log2(d[m][j]) + m_local[m][j];
        }
      }
    }
    wg_sync<128>(CONSUMER_WARPGROUP_SYNC_BARRIER_ID);
  }
}
} // namespace kernel