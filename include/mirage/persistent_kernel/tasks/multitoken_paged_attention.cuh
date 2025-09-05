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
#include "common.h"
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "norm.cuh"
#include "reduction.cuh"
#include "rotary_embedding.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"

namespace kernel {

// NOTE(Jinchen): this task implements the paged attention where a causal mask
// is applied. In each task, we process one request with one or more tokens
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int MAX_TOKENS = 1>
__device__ __forceinline__ void multitoken_paged_attention_task_impl(
    void const *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *output_ptr,
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

  // NOTE(Jinchen): The input is a packed QKV tensor, which may contain
  // multiple tokens. The shape of the packed QKV tensor is
  // [num_tokens, head_dim * (num_qo_heads + num_kv_heads * 2)]
  constexpr int QKV_STRIDE = (NUM_QO_HEADS + NUM_KV_HEADS * 2) * HEAD_DIM;
  constexpr int O_STRIDE = NUM_QO_HEADS * HEAD_DIM;
  // NOTE(Jinchen): assume the layout of KV Cache is NHD,
  // i.e., the shape of KV Cache is
  // [max_num_pages, page_size, num_kv_heads, head_dim]
  constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;

  constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
  constexpr int KV_TILE_SIZE = 64;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;

  // NOTE(Jinchen): we use m16n16k16 mma to compute matrix multiplication
  constexpr int MMA_ITERS_M = (MAX_TOKENS * NUM_QO_PER_KV + 15) / 16;

  // the scale factor for normalization in softmax
  float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

  int warp_idx = warp_id();
  int lane_idx = lane_id();

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  int const num_tokens = last_token_pos - first_token_pos;

  // NOTE(Jinchen): to simplify the implementation, we assume that the metadata
  // of the paged KV cache includes the new tokens, i.e., spaces are allocated
  // before the request is processed, while the real data is copied into the
  // corresponding pages after that
  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const seq_len = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[request_id];
  // valid_lens = [seq_len - num_tokens + 1 + i for i in range(num_tokens)]

  // Load the paged KV indices into shared memory
  __shared__ int page_indices[MAX_PAGES_PER_REQUEST];
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
          paged_kv_indices_buffer_ptr[tail_offset + i];
    }
  }
  __syncthreads();

  T const *__restrict__ d_q =
      reinterpret_cast<T const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  T const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  T const *__restrict__ d_v = d_k + HEAD_DIM;
  T *__restrict__ d_paged_k_cache = reinterpret_cast<T *>(paged_k_cache_ptr);
  T *__restrict__ d_paged_v_cache = reinterpret_cast<T *>(paged_v_cache_ptr);
  T *__restrict__ d_output =
      reinterpret_cast<T *>(output_ptr) + first_token_pos * QKV_STRIDE;

  // DTensors' layouts
  using QDmem =
      dmem_row_const<T, MAX_TOKENS * NUM_QO_PER_KV, HEAD_DIM, QKV_STRIDE>;
  using KVDmem = dmem_row_const<T, MAX_TOKENS, HEAD_DIM, QKV_STRIDE>;
  using KVCacheDmem = dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_CACHE_STRIDE>;
  using ODmem = dmem_row<T, MAX_TOKENS * NUM_QO_PER_KV, HEAD_DIM, O_STRIDE>;

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

  constexpr size_t S_O_OFFSET = S_V_BUFFER_OFFSET + S_V_BUFFER_SIZE;
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
  [[maybe_unused]] constexpr size_t S_O_BUFFER_SIZE =
      sizeof(float) * MMA_ITERS_M * NUM_THREADS * 64;

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
  float *s_m_buffer = reinterpret_cast<float *>(smem + S_M_BUFFER_OFFSET);
  float *s_d_buffer = reinterpret_cast<float *>(smem + S_D_BUFFER_OFFSET);
  float *s_o_buffer = reinterpret_cast<float *>(smem + S_O_BUFFER_OFFSET);

  // STensors' layouts
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using QOSmem =
      smem_row<T, 3, 3, 3, MAX_TOKENS * NUM_QO_PER_KV, HEAD_DIM, HEAD_DIM>;
  using KVSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, HEAD_DIM, HEAD_DIM>;

  ZeroBufferSmem zero_buffer(zero_buf);
  QOSmem q_smem(s_q), o_smem(s_o);
  KVSmem k_smem(s_k), v_smem(s_v);
  KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);

  size_t const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
  int curr_iter_len = min(seq_len, KV_TILE_SIZE);
  int cp_finished_seq_len = 0;

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
    if (dst_row + cp_finished_seq_len < seq_len - num_tokens) {
      // load from KV Cache
      int page_idx = page_indices[(dst_row + cp_finished_seq_len) / PAGE_SIZE];
      int page_offset = (dst_row + cp_finished_seq_len) % PAGE_SIZE;
      int src_row = page_idx * PAGE_SIZE + page_offset;
      load_smem(k_buffer_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
      load_smem(v_buffer_smem(dst_row, col), paged_v_cache_dmem(src_row, col));
    } else {
      // load from QKV
      int src_row = dst_row + cp_finished_seq_len - (seq_len - num_tokens);
      load_smem(k_buffer_smem(dst_row, col), k_dmem(src_row, col));
      load_smem(v_buffer_smem(dst_row, col), v_dmem(src_row, col));
    }
  }
  cp_async_fence();
  cp_finished_seq_len += curr_iter_len;

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
  float o[MMA_ITERS_M][HEAD_DIM / 16][8];
#pragma unroll
  for (int m = 0; m < MMA_ITERS_M; m++) {
#pragma unroll
    for (int n = 0; n < HEAD_DIM / 16; n++) {
      clear_8_floats(o[m][n]);
    }
  }

  for (size_t iter = 0; iter < num_iters; iter++) {
    int next_iter_len = iter + 1 < num_iters
                            ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                            : 0;
    if (next_iter_len > 0) {
#pragma unroll
      for (int chunk_idx = threadIdx.x;
           chunk_idx < curr_iter_len * HEAD_DIM / CP_CHUNK_SIZE;
           chunk_idx += NUM_THREADS) {
        int dst_row = chunk_idx / (HEAD_DIM / CP_CHUNK_SIZE);
        int col = (chunk_idx % (HEAD_DIM / CP_CHUNK_SIZE)) * CP_CHUNK_SIZE;
        if (dst_row + cp_finished_seq_len < seq_len - num_tokens) {
          // load from KV Cache
          int page_idx =
              page_indices[(dst_row + cp_finished_seq_len) / PAGE_SIZE];
          int page_offset = (dst_row + cp_finished_seq_len) % PAGE_SIZE;
          int src_row = page_idx * PAGE_SIZE + page_offset;
          load_smem(k_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
          load_smem(v_smem(dst_row, col), paged_v_cache_dmem(src_row, col));
        } else {
          // load from QKV
          int src_row = dst_row + cp_finished_seq_len - (seq_len - num_tokens);
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
    __syncthreads();

    if (qk_norm) {
      // Q norm
      if (iter == 0) {
#pragma unroll
        for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
          rms_norm<T, QOSmem, NUM_QO_PER_KV, HEAD_DIM>(
              q_smem,
              static_cast<T const *>(q_norm_weight_ptr),
              s_q_norm_sum,
              q_eps,
              token_idx * NUM_QO_PER_KV,
              rope,
              static_cast<T const *>(cos_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM);
        }
      }
      // K norm
      // NOTE(Jinchen): assume that MAX_TOKENS is less than or equal to
      // KV_TILE_SIZE, so the new tokens are always in the last tile
      else if (iter == num_iters - 1) {
        for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
          rms_norm<T, KVSmem, 1, HEAD_DIM>(
              k_smem,
              static_cast<T const *>(k_norm_weight_ptr),
              s_k_norm_sum,
              k_eps,
              token_idx + curr_iter_len - num_tokens,
              rope,
              static_cast<T const *>(cos_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM);
        }
      }
    } else {
      if (rope && iter == 0) {
#pragma unroll
        for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
          // q rope
          rotary_embedding<T, QOSmem, NUM_QO_PER_KV, HEAD_DIM>(
              q_smem,
              static_cast<T const *>(cos_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM,
              token_idx * NUM_QO_PER_KV);
        }
      } else if (rope && iter == num_iters - 1) {
        for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
          // k rope
          rotary_embedding<T, KVSmem, 1, HEAD_DIM>(
              k_smem,
              static_cast<T const *>(cos_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) +
                  (token_idx + seq_len - num_tokens) * HEAD_DIM,
              token_idx + curr_iter_len - num_tokens);
        }
      }
    }

    __syncthreads();

    // compute X = QK^T
    // NOTE(Jinchen): we use m16n16k16 mma, and let warp layout be
    // 1x4x1, so mma iterates over m and k dimensions
    float x_frag_f[MMA_ITERS_M][8];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      clear_8_floats(x_frag_f[m]);
    }
    uint32_t q_frag[4], kt_frag[4];

    int kt_col = (warp_idx << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      int q_row = (m << 4) + (lane_idx & 0xF);
#pragma unroll
      for (int k = 0; k < HEAD_DIM / 16; k++) {
        int q_col = (k << 4) + ((lane_idx >> 4) << 3);
        int kt_row = (k << 4) + (((lane_idx & 0xF) >> 3) << 3);
        T *src_ptr_Q = q_row < num_tokens * NUM_QO_PER_KV ? q_smem(q_row, q_col)
                                                          : zero_buffer(0, 0);
        T *src_ptr_KT =
            kt_col < curr_iter_len ? k_smem(kt_col, kt_row) : zero_buffer(0, 0);
        ldsm(src_ptr_Q, q_frag);
        ldsm(src_ptr_KT, kt_frag);
        mma_m16n16k16_bf16bf16bf32(x_frag_f[m], q_frag, kt_frag, x_frag_f[m]);
      }
    }
    __syncthreads();

    // update m_local: get partial max
    // NOTE(Jinchen): each thread maintains MMA_ITERS_M * 2 partial max
    // values. For a given m, the first value is the maximum of
    // x_frag_f[m][0, 1, 4, 5], and the second value is the maximum of
    // x_frag_f[m][2, 3, 6, 7]
    float m_prev[MMA_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      m_prev[m][0] = m_local[m][0];
      m_prev[m][1] = m_local[m][1];
#pragma unroll
      for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
        // row_base = (m * 16) + (lane_idx / 4)
        // col_base = (warp_idx * 16) + ((lane_idx % 4) * 2)
        // row_offset = ((frag_idx % 4) / 2) * 8
        // col_offset = ((frag_idx / 4) * 8) + (frag_idx % 2)
        int row = (m << 4) + (lane_idx >> 2) + (((frag_idx & 0x3) >> 1) << 3);
        int col = (warp_idx << 4) + ((lane_idx & 0x3) << 1) +
                  ((frag_idx >> 2) << 3) + (frag_idx & 0x1);
        int token_idx = row / NUM_QO_PER_KV;
        bool is_valid =
            row < num_tokens * NUM_QO_PER_KV &&
            col + iter * KV_TILE_SIZE <= token_idx + seq_len - num_tokens;
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
      rescale[m][0] = expf(m_prev[m][0] * sm_scale - m_local[m][0] * sm_scale);
      rescale[m][1] = expf(m_prev[m][1] * sm_scale - m_local[m][1] * sm_scale);
    }

    // update d: get partial sum
    float d_partial[MMA_ITERS_M][2];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      d_partial[m][0] = 0.f;
      d_partial[m][1] = 0.f;
#pragma unroll
      for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
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
      for (int n = 0; n < HEAD_DIM / 16; n++) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
          o[m][n][frag_idx] *= rescale[m][(frag_idx & 0x3) >> 1];
        }
      }
    }

    // update o: compute O = exp(X - m) * V and accumulate
    // use m16n16k16 mma to compute and let warp layout be 1x1x4
    uint32_t x_frag[MMA_ITERS_M][4], v_frag[4];
#pragma unroll
    for (int m = 0; m < MMA_ITERS_M; m++) {
      convert_f32_to_bf16_uint32(x_frag_f[m], x_frag[m]);
      int v_row = (warp_idx << 4) + (lane_idx & 0xF);
#pragma unroll
      for (int n = 0; n < HEAD_DIM / 16; n++) {
        int v_col = (n << 4) + ((lane_idx >> 4) << 3);
        T *src_ptr_V =
            v_row < curr_iter_len ? v_smem(v_row, v_col) : zero_buffer(0, 0);
        ldsm_t(src_ptr_V, v_frag);
        mma_m16n16k16_bf16bf16bf32(o[m][n], x_frag[m], v_frag, o[m][n]);
      }
    }
    __syncthreads();

    curr_iter_len = next_iter_len;
  }

  // write intermediate results to buffer in shared memory
#pragma unroll
  for (int m = 0; m < MMA_ITERS_M; m++) {
    m_local[m][0] *= m_local[m][0] != -inf ? sm_scale : 1.f;
    m_local[m][1] *= m_local[m][1] != -inf ? sm_scale : 1.f;
    s_m_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2] = m_local[m][0];
    s_m_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2 + 1] = m_local[m][1];
    s_d_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2] = d[m][0];
    s_d_buffer[m * NUM_THREADS * 2 + threadIdx.x * 2 + 1] = d[m][1];
    for (int n = 0; n < HEAD_DIM / 16; n++) {
#pragma unroll
      for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
        s_o_buffer[m * NUM_THREADS * 64 + threadIdx.x * 64 + n * 8 + frag_idx] =
            o[m][n][frag_idx];
      }
    }
  }
  __syncthreads();

  // get global m, d, and o
  // each thread handles an element in o in each iteration
  for (int elem_idx = threadIdx.x;
       elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
       elem_idx += NUM_THREADS) {
    int row = elem_idx / HEAD_DIM;
    int col = elem_idx % HEAD_DIM;
    int t_idx = (row % 8) * 4 + (col % 8) / 2;
    int mma_iter_n = col / 16;
    /* The fragment layout is as follows:
     *
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 0 1 0 1 0 1 0 1 4 5 4 5 4 5 4 5
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     * 2 3 2 3 2 3 2 3 6 7 6 7 6 7 6 7
     */
    int frag_idx = ((col % 16) / 8) * 4 + ((row % 16) / 8) * 2 + (col % 2);

    float m_global = -inf;
    float d_global = 1.f;
    float o_global = 0.f;
    // 4 local values per row
#pragma unroll
    for (int local_idx = 0; local_idx < 4; local_idx++) {
      // access the shared memory buffer
      int md_smem_offset = (row / 16) * NUM_THREADS * 2 // mma iter m
                           + local_idx * 32 * 2  // 32 threads per local value
                           + t_idx * 2           // corresponding thread
                           + (frag_idx % 4) / 2; // first half or second half
      float m_prev = m_global,
            d_prev = d_global; // save previous values
      float other_m = s_m_buffer[md_smem_offset],
            other_d = s_d_buffer[md_smem_offset];
      m_global = max(m_prev, other_m);
      d_global =
          d_prev * expf(m_prev - m_global) + other_d * expf(other_m - m_global);
      // accumulate o
      float other_o =
          s_o_buffer[(row / 16) * NUM_THREADS * 64 // mma iter m
                     + local_idx * 32 * 64         // 32 threads per local value
                     + t_idx * 64                  // corresponding thread
                     + mma_iter_n * 8              // mma iter n
                     + frag_idx];
      o_global = o_global * expf(m_prev - m_global) +
                 other_o * expf(other_m - m_global);
    }
    o_smem.at(row, col) = bfloat16(o_global / d_global);
  }
  __syncthreads();

  // update the KV Cache
  for (int elem_idx = threadIdx.x; elem_idx < num_tokens * HEAD_DIM;
       elem_idx += NUM_THREADS) {
    int token_idx = elem_idx / HEAD_DIM;
    int col = elem_idx % HEAD_DIM;
    int page_idx = page_indices[(token_idx + seq_len - num_tokens) / PAGE_SIZE];
    int page_offset = (token_idx + seq_len - num_tokens) % PAGE_SIZE;
    int src_row = (token_idx + seq_len - num_tokens) % KV_TILE_SIZE;
    int dst_row = page_idx * PAGE_SIZE + page_offset;
    paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
    paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
  }

  // store the output
  for (int elem_idx = threadIdx.x;
       elem_idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
       elem_idx += NUM_THREADS) {
    int src_row = elem_idx / HEAD_DIM;
    int src_col = elem_idx % HEAD_DIM;
    int dst_row = src_row / NUM_QO_PER_KV;
    int dst_col = src_col + (src_row % NUM_QO_PER_KV) * HEAD_DIM;
    o_dmem.at(dst_row, dst_col) = o_smem.at(src_row, src_col);
  }
}

} // namespace kernel