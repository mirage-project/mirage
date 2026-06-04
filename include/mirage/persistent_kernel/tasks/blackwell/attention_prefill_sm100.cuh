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
#include "norm_sm100.cuh"
#include "rotary_embedding_sm100.cuh"
#include "tasks/ampere/mma.cuh"
#include "tasks/ampere/smem_layout.cuh"
#include "tasks/common/common_header.cuh"
#include "tasks/hopper/rotary_embedding_hopper.cuh"

#include <cutlass/arch/barrier.h>

namespace kernel {

// SM100 wide-Q paged attention for prefill (causal).
//
// Compared with multitoken_paged_attention_sm100_task_impl (decode shape),
// this kernel partitions the Q dimension across 4 warps instead of the
// KV dimension. Each warp owns (NUM_Q / 4) packed-Q rows and walks the full
// KV tile per outer iteration, so there is no cross-warp m/d/o merge buffer.
// That removes the O(MMA_ITERS_M * NUM_THREADS * 64) smem term and lets
// MAX_TOKENS scale up to 32 (packed_Q = 128 for Qwen3 GQA group=4) without
// exceeding B200's dynamic smem budget.
//
// Requirements:
//   * MAX_TOKENS * NUM_QO_PER_KV must be divisible by 64
//     (each warp gets at least one m16-mma iteration).
//   * KV_TILE_SIZE = 64 (matches existing decode kernel & page granularity).
//   * Causal mask only.
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 32>
__device__ __forceinline__ void
    multitoken_paged_attention_prefill_sm100_task_impl(
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
        int prefill_threshold = -1,
        int qo_tile_idx = 0,
        int kv_chunk_idx = -1,
        int kv_chunk_size = 256,
        void *lse_ptr = nullptr,
        int flat_kv_head_idx = -1,
        int flat_num_kv_heads = NUM_KV_HEADS,
        int lse_num_kv_chunks = -1) {
  constexpr int CONSUMER_WARPGROUP_SYNC_BARRIER_ID = 6;
  constexpr int ROTARY_SYNC_BARRIER_ID = 7;
  cutlass::arch::NamedBarrier wg_barrier(
      NUM_THREADS, /*bar-id*/ CONSUMER_WARPGROUP_SYNC_BARRIER_ID);
  if (threadIdx.x < NUM_THREADS) {
    constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;
    constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
    constexpr int KV_TILE_SIZE = 64;
    constexpr int NUM_WARPS = NUM_THREADS / 32; // = 4
    constexpr int NUM_Q = MAX_TOKENS * NUM_QO_PER_KV;
    constexpr int HEAD_DIM_ITER = HEAD_DIM / 16;
    constexpr int NUM_ITER_QK_N = KV_TILE_SIZE / 16; // KV cols per warp / 16

    // Each warp owns (NUM_Q / NUM_WARPS) Q rows. mma m=16 iterations per warp:
    constexpr int MMA_ITERS_M_WARP = NUM_Q / (NUM_WARPS * 16);
    static_assert(NUM_Q % (NUM_WARPS * 16) == 0,
                  "MAX_TOKENS * NUM_QO_PER_KV must be divisible by 64 "
                  "for split-M warp layout");

    // Layout of the packed Q axis: warp 0 owns rows [0, 16*M), warp 1 owns
    // [16*M, 32*M), etc. (m << 6) gives the M-iter offset (=NUM_WARPS*16).
    // (warp_idx << 4) places this warp's 16-row band within that M block.

    // softmax scale (no log2e — we use expf for clarity / parity with decode)
    float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    int warp_idx = warp_id();
    int lane_idx = lane_id();

    int const request_first_token_pos = qo_indptr_buffer_ptr[request_id];
    int const request_last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
    if (request_first_token_pos == request_last_token_pos) {
      return;
    }
    int const request_num_tokens =
        request_last_token_pos - request_first_token_pos;
    if (prefill_threshold >= 0 &&
        request_num_tokens * NUM_QO_PER_KV <= prefill_threshold) {
      return;
    }
    int const tile_token_offset = qo_tile_idx * MAX_TOKENS;
    if (tile_token_offset >= request_num_tokens) {
      return;
    }
    int num_tokens = min(MAX_TOKENS, request_num_tokens - tile_token_offset);
    int first_token_pos = request_first_token_pos + tile_token_offset;

    int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
    int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
    int const num_pages = last_page_pos - first_page_pos;
    int const allocated_seq_len = (num_pages - 1) * PAGE_SIZE +
                                  paged_kv_last_page_len_buffer_ptr[request_id];
    int const history_len = allocated_seq_len - request_num_tokens;
    int q_abs_start = history_len + tile_token_offset;
    int const full_seq_len = q_abs_start + num_tokens;
    int const split_start =
        kv_chunk_idx >= 0 ? kv_chunk_idx * kv_chunk_size : 0;
    int const split_end =
        kv_chunk_idx >= 0 ? min(full_seq_len, split_start + kv_chunk_size)
                          : full_seq_len;
    if (split_start >= split_end) {
      return;
    }
    if (kv_chunk_idx >= 0 && split_start > q_abs_start) {
      int const skipped_q_tokens = min(num_tokens, split_start - q_abs_start);
      first_token_pos += skipped_q_tokens;
      q_abs_start += skipped_q_tokens;
      num_tokens -= skipped_q_tokens;
      if (num_tokens <= 0) {
        return;
      }
    }
    int const seq_len = split_end - split_start;

    int const *page_indices = paged_kv_indices_buffer_ptr + first_page_pos;
    wg_barrier.arrive_and_wait();

    int const flat_qkv_offset =
        flat_kv_head_idx >= 0
            ? flat_kv_head_idx * (NUM_QO_PER_KV + 2) * HEAD_DIM
            : 0;
    int const flat_kv_offset =
        flat_kv_head_idx >= 0 ? flat_kv_head_idx * HEAD_DIM : 0;
    int const partial_group_offset =
        kv_chunk_idx >= 0
            ? (kv_chunk_idx * flat_num_kv_heads + flat_kv_head_idx) *
                  NUM_QO_PER_KV
            : 0;

    T const *__restrict__ d_q_request =
        reinterpret_cast<T const *>(qkv_ptr) +
        request_first_token_pos * QKV_STRIDE + flat_qkv_offset;
    T const *__restrict__ d_q =
        d_q_request + (first_token_pos - request_first_token_pos) * QKV_STRIDE;
    T const *__restrict__ d_k_request =
        d_q_request + NUM_QO_PER_KV * HEAD_DIM;
    T const *__restrict__ d_v_request = d_k_request + HEAD_DIM;
    T *__restrict__ d_paged_k_cache =
        reinterpret_cast<T *>(paged_k_cache_ptr) + flat_kv_offset;
    T *__restrict__ d_paged_v_cache =
        reinterpret_cast<T *>(paged_v_cache_ptr) + flat_kv_offset;
    T *__restrict__ d_output =
        reinterpret_cast<T *>(output_ptr) + first_token_pos * O_STRIDE +
        partial_group_offset * HEAD_DIM;

    using QDmem =
        dmem_row_const<T, MAX_TOKENS, HEAD_DIM * NUM_QO_PER_KV, QKV_STRIDE>;
    using KVDmem = dmem_row_const<T, MAX_TOKENS, HEAD_DIM, QKV_STRIDE>;
    using KVCacheDmem = dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_CACHE_STRIDE>;
    using ODmem = dmem_row<T, MAX_TOKENS, HEAD_DIM * NUM_QO_PER_KV, O_STRIDE>;

    QDmem q_dmem(d_q);
    KVDmem k_dmem(d_k_request), v_dmem(d_v_request);
    KVCacheDmem paged_k_cache_dmem(d_paged_k_cache),
        paged_v_cache_dmem(d_paged_v_cache);
    ODmem o_dmem(d_output);

    // ---- smem layout ----------------------------------------------------
    // Reuse the same buffer layout as the decode kernel but drop S_M / S_D /
    // S_O cross-warp buffers — they're unused with the split-M design.
    constexpr size_t ZERO_BUFFER_OFFSET = 0;
    constexpr size_t ZERO_BUFFER_SIZE = sizeof(T) * 8;

    constexpr size_t S_Q_OFFSET = ZERO_BUFFER_OFFSET + ZERO_BUFFER_SIZE;
    constexpr size_t S_Q_SIZE =
        sizeof(T) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;

    constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
    constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;

    constexpr size_t S_K_BUFFER_OFFSET = S_K_OFFSET + S_K_SIZE;
    constexpr size_t S_K_BUFFER_SIZE = S_K_SIZE;

    constexpr size_t S_V_OFFSET = S_K_BUFFER_OFFSET + S_K_BUFFER_SIZE;
    constexpr size_t S_V_SIZE = S_K_SIZE;

    constexpr size_t S_V_BUFFER_OFFSET = S_V_OFFSET + S_V_SIZE;
    constexpr size_t S_V_BUFFER_SIZE = S_K_SIZE;

    // S_O reuses S_Q (Q is in registers / no longer needed once we begin
    // writing the output).
    constexpr size_t S_O_OFFSET = S_Q_OFFSET;
    constexpr size_t S_O_SIZE = S_Q_SIZE;

    constexpr size_t S_Q_NORM_SUM_OFFSET =
        ((S_V_BUFFER_OFFSET + S_V_BUFFER_SIZE + sizeof(float) - 1) &
         ~size_t(sizeof(float) - 1));
    constexpr size_t S_Q_NORM_SUM_SIZE = sizeof(float) * 4;

    constexpr size_t S_K_NORM_SUM_OFFSET =
        S_Q_NORM_SUM_OFFSET + S_Q_NORM_SUM_SIZE;
    constexpr size_t S_K_NORM_SUM_SIZE = sizeof(float) * 4;

    constexpr size_t S_TOTAL_OFFSET = S_K_NORM_SUM_OFFSET + S_K_NORM_SUM_SIZE;
    static_assert(S_TOTAL_OFFSET <=
                  mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);

    extern __shared__ char smem[];
    T *zero_buf = reinterpret_cast<T *>(smem + ZERO_BUFFER_OFFSET);
    clear_smem_buffer<T, 8>(zero_buf);
    T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
    T *s_o = reinterpret_cast<T *>(smem + S_O_OFFSET);
    T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
    T *s_k_buffer = reinterpret_cast<T *>(smem + S_K_BUFFER_OFFSET);
    T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
    T *s_v_buffer = reinterpret_cast<T *>(smem + S_V_BUFFER_OFFSET);
    float *s_q_norm_sum = reinterpret_cast<float *>(smem + S_Q_NORM_SUM_OFFSET);
    float *s_k_norm_sum = reinterpret_cast<float *>(smem + S_K_NORM_SUM_OFFSET);

    using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
    using QOSmem =
        smem_row<T, 3, 3, 3, MAX_TOKENS * NUM_QO_PER_KV, HEAD_DIM, HEAD_DIM>;
    using KVSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, HEAD_DIM, HEAD_DIM>;

    ZeroBufferSmem zero_buffer(zero_buf);
    QOSmem q_smem(s_q), o_smem(s_o);
    KVSmem k_smem(s_k), v_smem(s_v);
    KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);

    int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
    int curr_iter_len = min(seq_len, KV_TILE_SIZE);
    int cp_finished_seq_len = 0;
    static_assert(HEAD_DIM % CP_CHUNK_SIZE == 0);
    static_assert(PAGE_SIZE % KV_TILE_SIZE == 0);

    // ---- prologue: load Q & first KV tile -------------------------------
#pragma unroll
    for (int chunk_idx = threadIdx.x;
         chunk_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM / CP_CHUNK_SIZE;
         chunk_idx += NUM_THREADS) {
      int src_row = chunk_idx / (NUM_QO_PER_KV * HEAD_DIM / CP_CHUNK_SIZE);
      int src_col = (chunk_idx % (NUM_QO_PER_KV * HEAD_DIM / CP_CHUNK_SIZE)) *
                    CP_CHUNK_SIZE;
      int dst_row = src_row * NUM_QO_PER_KV + src_col / HEAD_DIM;
      int dst_col = src_col % HEAD_DIM;
      load_smem(q_smem(dst_row, dst_col), q_dmem(src_row, src_col));
    }

#pragma unroll
    for (int chunk_idx = threadIdx.x;
         chunk_idx < curr_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
         chunk_idx += NUM_THREADS) {
      int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
      int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
      int global_pos = split_start + dst_row + cp_finished_seq_len;
      // Only true history lives exclusively in the paged cache. Tokens from
      // this prefill request are already present in qkv input, so read them
      // from qkv instead of depending on earlier Q tiles to populate cache.
      if (global_pos < history_len) {
        int page_idx = page_indices[global_pos / PAGE_SIZE];
        int page_offset = global_pos % PAGE_SIZE;
        int src_row = page_idx * PAGE_SIZE + page_offset;
        load_smem(k_buffer_smem(dst_row, col),
                  paged_k_cache_dmem(src_row, col));
        load_smem(v_buffer_smem(dst_row, col),
                  paged_v_cache_dmem(src_row, col));
      } else {
        int src_row = global_pos - history_len;
        load_smem(k_buffer_smem(dst_row, col), k_dmem(src_row, col));
        load_smem(v_buffer_smem(dst_row, col), v_dmem(src_row, col));
      }
    }
    cp_async_fence();
    cp_finished_seq_len += curr_iter_len;

    // ---- per-thread online-softmax accumulators (registers) --------------
    // Per warp: MMA_ITERS_M_WARP m-iters × HEAD_DIM_ITER n-iters × 8 frags.
    float m_local[MMA_ITERS_M_WARP][2];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
      m_local[m][0] = -inf;
      m_local[m][1] = -inf;
    }
    float d[MMA_ITERS_M_WARP][2];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
      d[m][0] = 0.f;
      d[m][1] = 0.f;
    }
    float o[MMA_ITERS_M_WARP][HEAD_DIM / 16][8];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
#pragma unroll
      for (int n = 0; n < HEAD_DIM_ITER; n++) {
        clear_8_floats(o[m][n]);
      }
    }

    // ---- main KV loop ----------------------------------------------------
    for (int iter = 0; iter < num_iters; iter++) {
      int next_iter_len = iter + 1 < num_iters
                              ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                              : 0;
      if (next_iter_len > 0) {
#pragma unroll
        for (int chunk_idx = threadIdx.x;
             chunk_idx < next_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
             chunk_idx += NUM_THREADS) {
          int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
          int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
          int global_pos = split_start + dst_row + cp_finished_seq_len;
          if (global_pos < history_len) {
            int page_idx = page_indices[global_pos / PAGE_SIZE];
            int page_offset = global_pos % PAGE_SIZE;
            int src_row = page_idx * PAGE_SIZE + page_offset;
            load_smem(k_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
            load_smem(v_smem(dst_row, col), paged_v_cache_dmem(src_row, col));
          } else {
            int src_row = global_pos - history_len;
            load_smem(k_smem(dst_row, col), k_dmem(src_row, col));
            load_smem(v_smem(dst_row, col), v_dmem(src_row, col));
          }
        }
        cp_async_fence();
        cp_async_wait<1>();
        cp_finished_seq_len += next_iter_len;
      } else {
        cp_async_wait<0>();
      }

      // rotate the buffers so that the just-finished load is "current" and
      // the next prefetched tile fills the "buffer" slot
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
      wg_barrier.arrive_and_wait();

      int tile_global_start = split_start + iter * KV_TILE_SIZE;
      int tile_global_end = tile_global_start + curr_iter_len;
      int prompt_kv_begin = max(tile_global_start, history_len);
      int prompt_kv_end =
          min(tile_global_end, history_len + request_num_tokens);
      int kv_tokens_to_process = max(prompt_kv_end - prompt_kv_begin, 0);
      int first_kv_token_to_process = prompt_kv_begin - tile_global_start;
      int cache_write_begin = max(tile_global_start, q_abs_start);
      int cache_write_end = min(tile_global_end, q_abs_start + num_tokens);
      int kv_tokens_to_write = max(cache_write_end - cache_write_begin, 0);
      int first_kv_token_to_write = cache_write_begin - tile_global_start;
      if (qk_norm) {
        if (iter == 0) {
          rms_norm_sm100<T,
                         QOSmem,
                         NUM_QO_PER_KV,
                         HEAD_DIM,
                         CONSUMER_WARPGROUP_SYNC_BARRIER_ID,
                         ROTARY_SYNC_BARRIER_ID>(
              q_smem,
              static_cast<T const *>(q_norm_weight_ptr),
              s_q_norm_sum,
              q_eps,
              num_tokens,
              0,
              rope,
              static_cast<T const *>(cos_ptr) +
                  q_abs_start * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  q_abs_start * HEAD_DIM);
        }
        if (kv_tokens_to_process > 0) {
          rms_norm_sm100<T,
                         KVSmem,
                         1,
                         HEAD_DIM,
                         CONSUMER_WARPGROUP_SYNC_BARRIER_ID,
                         ROTARY_SYNC_BARRIER_ID>(
              k_smem,
              static_cast<T const *>(k_norm_weight_ptr),
              s_k_norm_sum,
              k_eps,
              kv_tokens_to_process,
              first_kv_token_to_process,
              rope,
              static_cast<T const *>(cos_ptr) +
                  prompt_kv_begin * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  prompt_kv_begin * HEAD_DIM);
        }
      } else if (rope) {
        if (iter == 0) {
#pragma unroll
          for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
            rotary_embedding_hopper<T,
                                    QOSmem,
                                    NUM_QO_PER_KV,
                                    1,
                                    HEAD_DIM,
                                    128,
                                    CONSUMER_WARPGROUP_SYNC_BARRIER_ID>(
                q_smem,
                static_cast<T const *>(cos_ptr) +
                    (token_idx + q_abs_start) * HEAD_DIM,
                static_cast<T const *>(sin_ptr) +
                    (token_idx + q_abs_start) * HEAD_DIM,
                token_idx);
          }
        }
        if (kv_tokens_to_process > 0) {
          for (int token_idx = 0; token_idx < kv_tokens_to_process;
               token_idx++) {
            rotary_embedding_hopper<T,
                                    KVSmem,
                                    1,
                                    1,
                                    HEAD_DIM,
                128,
                CONSUMER_WARPGROUP_SYNC_BARRIER_ID>(
                k_smem,
                static_cast<T const *>(cos_ptr) +
                    (token_idx + prompt_kv_begin) * HEAD_DIM,
                static_cast<T const *>(sin_ptr) +
                    (token_idx + prompt_kv_begin) * HEAD_DIM,
                token_idx + first_kv_token_to_process);
          }
        }
      }

      wg_barrier.arrive_and_wait();

      // write new tokens' K/V back into the paged cache
      if (kv_tokens_to_write > 0) {
        for (int elem_idx = threadIdx.x;
             elem_idx < kv_tokens_to_write * HEAD_DIM;
             elem_idx += NUM_THREADS) {
          int token_idx = elem_idx / HEAD_DIM;
          int col = elem_idx % HEAD_DIM;
          int global_pos = cache_write_begin + token_idx;
          int page_idx = page_indices[global_pos / PAGE_SIZE];
          int page_offset = global_pos % PAGE_SIZE;
          int src_row = first_kv_token_to_write + token_idx;
          int dst_row = page_idx * PAGE_SIZE + page_offset;
          paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
          paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
        }
      }

      // ---- compute X = Q K^T (split-M, full-KV-per-warp) -----------------
      // For each warp's M iters, walk the full KV tile (NUM_ITER_QK_N cols of
      // 16). x_frag_f[m][n][frag] holds this warp's partial scores.
      float x_frag_f[MMA_ITERS_M_WARP][NUM_ITER_QK_N][8];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
#pragma unroll
        for (int n = 0; n < NUM_ITER_QK_N; n++) {
          clear_8_floats(x_frag_f[m][n]);
        }
      }
      uint32_t q_frag[4], kt_frag[4];

#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        // Q rows owned by this thread of this warp for this m-iter:
        //   q_row = m*(NUM_WARPS*16) + warp_idx*16 + (lane_idx & 0xF)
        int q_row_block =
            (m * NUM_WARPS * 16) + (warp_idx << 4);
        if (q_row_block >= num_tokens * NUM_QO_PER_KV) {
          continue;
        }
        int q_row =
            q_row_block + (lane_idx & 0xF);
#pragma unroll
        for (int n = 0; n < NUM_ITER_QK_N; n++) {
          int kt_col = (n << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (int k = 0; k < HEAD_DIM_ITER; k++) {
            int q_col = (k << 4) + ((lane_idx >> 4) << 3);
            int kt_row = (k << 4) + (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr_Q = q_row < num_tokens * NUM_QO_PER_KV
                               ? q_smem(q_row, q_col)
                               : zero_buffer(0, 0);
            T *src_ptr_KT = kt_col < curr_iter_len ? k_smem(kt_col, kt_row)
                                                   : zero_buffer(0, 0);
            ldsm(src_ptr_Q, q_frag);
            ldsm(src_ptr_KT, kt_frag);
            mma_m16n16k16_bf16bf16bf32(
                x_frag_f[m][n], q_frag, kt_frag, x_frag_f[m][n]);
          }
        }
      }
      wg_barrier.arrive_and_wait();

      // ---- online softmax: causal mask, m_local, rescale, d --------------
      float m_prev[MMA_ITERS_M_WARP][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        m_prev[m][0] = m_local[m][0];
        m_prev[m][1] = m_local[m][1];
#pragma unroll
        for (int n = 0; n < NUM_ITER_QK_N; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
            int row = (m * NUM_WARPS * 16) + (warp_idx << 4) +
                      (lane_idx >> 2) + (((frag_idx & 0x3) >> 1) << 3);
            int col = (n << 4) + ((lane_idx & 0x3) << 1) +
                      ((frag_idx >> 2) << 3) + (frag_idx & 0x1);
            int token_idx = row / NUM_QO_PER_KV;
            int global_col = split_start + iter * KV_TILE_SIZE + col;
            bool is_valid =
                (row < num_tokens * NUM_QO_PER_KV) &&
                (global_col <= q_abs_start + token_idx);
            x_frag_f[m][n][frag_idx] =
                is_valid ? x_frag_f[m][n][frag_idx] : -inf;
            m_local[m][(frag_idx & 0x3) >> 1] = max(
                m_local[m][(frag_idx & 0x3) >> 1], x_frag_f[m][n][frag_idx]);
          }
        }
      }

      // reduce max across 4 threads in a row (intra-warp)
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x1));
        m_local[m][0] = max(m_local[m][0], shfl_xor_sync(m_local[m][0], 0x2));
        m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x1));
        m_local[m][1] = max(m_local[m][1], shfl_xor_sync(m_local[m][1], 0x2));
      }

      float rescale[MMA_ITERS_M_WARP][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        rescale[m][0] = (m_local[m][0] == -inf)
                            ? 1.f
                            : expf(m_prev[m][0] * sm_scale -
                                   m_local[m][0] * sm_scale);
        rescale[m][1] = (m_local[m][1] == -inf)
                            ? 1.f
                            : expf(m_prev[m][1] * sm_scale -
                                   m_local[m][1] * sm_scale);
      }

      float d_partial[MMA_ITERS_M_WARP][2];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        d_partial[m][0] = 0.f;
        d_partial[m][1] = 0.f;
#pragma unroll
        for (int n = 0; n < NUM_ITER_QK_N; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
            x_frag_f[m][n][frag_idx] =
                x_frag_f[m][n][frag_idx] != -inf
                    ? expf(x_frag_f[m][n][frag_idx] * sm_scale -
                           m_local[m][(frag_idx & 0x3) >> 1] * sm_scale)
                    : 0.f;
            d_partial[m][(frag_idx & 0x3) >> 1] += x_frag_f[m][n][frag_idx];
          }
        }
      }
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x1);
        d_partial[m][0] += shfl_xor_sync(d_partial[m][0], 0x2);
        d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x1);
        d_partial[m][1] += shfl_xor_sync(d_partial[m][1], 0x2);
        d[m][0] = d[m][0] * rescale[m][0] + d_partial[m][0];
        d[m][1] = d[m][1] * rescale[m][1] + d_partial[m][1];
      }

      // rescale running o
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
#pragma unroll
        for (int n = 0; n < HEAD_DIM_ITER; n++) {
#pragma unroll
          for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
            o[m][n][frag_idx] *= rescale[m][(frag_idx & 0x3) >> 1];
          }
        }
      }

      // ---- accumulate o += softmax(x) * V --------------------------------
      // x is (M=warp's rows) × (N=KV_TILE_SIZE), V is (N=KV_TILE_SIZE)×HEAD_DIM
      // We use m16n16k16 mma with the outer k loop over KV tile blocks.
      uint32_t x_frag[MMA_ITERS_M_WARP][NUM_ITER_QK_N][4], v_frag[4];
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
        int q_row_block =
            (m * NUM_WARPS * 16) + (warp_idx << 4);
        if (q_row_block >= num_tokens * NUM_QO_PER_KV) {
          continue;
        }
#pragma unroll
        for (int n = 0; n < HEAD_DIM_ITER; n++) {
#pragma unroll
          for (int k = 0; k < NUM_ITER_QK_N; k++) {
            convert_f32_to_bf16_uint32(x_frag_f[m][k], x_frag[m][k]);
            int v_row = (k << 4) + (lane_idx & 0xF);
            int v_col = (n << 4) + ((lane_idx >> 4) << 3);
            T *src_ptr_V = v_row < curr_iter_len ? v_smem(v_row, v_col)
                                                 : zero_buffer(0, 0);
            ldsm_t(src_ptr_V, v_frag);
            mma_m16n16k16_bf16bf16bf32(o[m][n], x_frag[m][k], v_frag, o[m][n]);
          }
        }
      }
      wg_barrier.arrive_and_wait();
      curr_iter_len = next_iter_len;
    }

    // ---- write output into smem (no cross-warp merge needed) -------------
    // Each warp owns its M block; write o[m][n][frag] / d[m][frag-row] for
    // this warp's q-row range into o_smem.
    wg_barrier.arrive_and_wait();
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
#pragma unroll
      for (int i = 0; i < HEAD_DIM_ITER; i++) {
        // Fragment layout (m16n16 acc): each thread holds 8 values for a
        // 16×16 tile.
        //   row_base = (m * NUM_WARPS * 16) + (warp_idx * 16) + (lane_idx / 4)
        //   col_base = (i * 16) + (lane_idx % 4) * 2
        // Per-thread layout within the 16×16 tile (frag layout same as
        // multitoken_paged_attention_task_impl_32_64):
        //   frag 0,1 -> row+0, col+0/1
        //   frag 2,3 -> row+8, col+0/1
        //   frag 4,5 -> row+0, col+8/9
        //   frag 6,7 -> row+8, col+8/9
        int row =
            (m * NUM_WARPS * 16) + (warp_idx << 4) + (lane_idx >> 2);
        int col = ((lane_idx & 3) << 1) + (i << 4);
        float d0 = d[m][0];
        float d1 = d[m][1];

        o_smem.at(row, col) = bfloat16(d0 > 0.f ? o[m][i][0] / d0 : 0.f);
        o_smem.at(row, col + 1) = bfloat16(d0 > 0.f ? o[m][i][1] / d0 : 0.f);
        o_smem.at(row + 8, col) = bfloat16(d1 > 0.f ? o[m][i][2] / d1 : 0.f);
        o_smem.at(row + 8, col + 1) =
            bfloat16(d1 > 0.f ? o[m][i][3] / d1 : 0.f);
        o_smem.at(row, col + 8) = bfloat16(d0 > 0.f ? o[m][i][4] / d0 : 0.f);
        o_smem.at(row, col + 9) = bfloat16(d0 > 0.f ? o[m][i][5] / d0 : 0.f);
        o_smem.at(row + 8, col + 8) =
            bfloat16(d1 > 0.f ? o[m][i][6] / d1 : 0.f);
        o_smem.at(row + 8, col + 9) =
            bfloat16(d1 > 0.f ? o[m][i][7] / d1 : 0.f);
      }
    }
    wg_barrier.arrive_and_wait();

    // ---- write smem output to global memory ------------------------------
    for (int elem_idx = threadIdx.x;
         elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
         elem_idx += NUM_THREADS) {
      int src_row = elem_idx / HEAD_DIM;
      int src_col = elem_idx % HEAD_DIM;
      int dst_row = src_row / NUM_QO_PER_KV;
      int dst_col = src_col + (src_row % NUM_QO_PER_KV) * HEAD_DIM;
      o_dmem.at(dst_row, dst_col) = o_smem.at(src_row, src_col);
    }

    if (kv_chunk_idx >= 0 && lse_ptr != nullptr) {
      constexpr float log2e_const = 1.4426950408889634f;
      int const num_kv_chunks =
          lse_num_kv_chunks > 0
              ? lse_num_kv_chunks
              : (MAX_SEQ_LEN + kv_chunk_size - 1) / kv_chunk_size;
      float *lse_out = reinterpret_cast<float *>(lse_ptr);
#pragma unroll
      for (int m = 0; m < MMA_ITERS_M_WARP; m++) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          int idx = (m * NUM_WARPS * 16) + (warp_idx << 4) + j * 8 +
                    lane_idx / 4;
          if (idx < num_tokens * NUM_QO_PER_KV) {
            int token_idx = idx / NUM_QO_PER_KV;
            int head_idx = idx % NUM_QO_PER_KV;
            int offset = (first_token_pos + token_idx) * num_kv_chunks *
                             flat_num_kv_heads * NUM_QO_PER_KV +
                         partial_group_offset + head_idx;
            lse_out[offset] =
                d[m][j] > 0.f
                    ? log2f(d[m][j]) +
                          m_local[m][j] * sm_scale * log2e_const
                    : -inf;
          }
        }
      }
    }
  } // threadIdx.x < NUM_THREADS
}

} // namespace kernel
