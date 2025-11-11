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

// NOTE(Jinchen): for each task, we process one batch
// and one KV head at a time
template <typename T,
          int NUM_Q_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int KV_STRIDE>
// NOTE(Jinchen): assume the layout of KV Cache is NHD,
// i.e., the shape of KV Cache is
// [max_num_pages, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ void
    paged_attention_task_impl(void const *qkv_ptr,
                              void *paged_k_cache_ptr,
                              void *paged_v_cache_ptr,
                              void *output_ptr,
                              void const *paged_kv_indices_buffer_ptr,
                              size_t seq_len,
                              bool qk_norm,
                              bool rope,
                              void const *q_norm_weight_ptr,
                              void const *k_norm_weight_ptr,
                              void const *cos_ptr,
                              void const *sin_ptr,
                              float q_eps,
                              float k_eps) {
  constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);
  constexpr size_t KV_TILE_SIZE = 64;
  constexpr int NUM_CHUNKS_Q = NUM_Q_PER_KV * HEAD_DIM / CP_CHUNK_SIZE;
  constexpr int NUM_CHUNKS_PER_ROW = HEAD_DIM / CP_CHUNK_SIZE;
  constexpr size_t MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;

  float const sm_scale = 1.f / sqrtf(float(HEAD_DIM));

  int warp_idx = warp_id();
  int lane_idx = lane_id();

  // NOTE(Jinchen): assume qkv_ptr points to the corresponding request
  // rather than the beginning of the whole QKV tensor
  T const *__restrict__ d_q = static_cast<T const *>(qkv_ptr);
  T const *__restrict__ d_k =
      static_cast<T const *>(qkv_ptr) + NUM_Q_PER_KV * HEAD_DIM;
  T const *__restrict__ d_v =
      static_cast<T const *>(qkv_ptr) + (NUM_Q_PER_KV + 1) * HEAD_DIM;
  T *__restrict__ d_paged_k_cache = static_cast<T *>(paged_k_cache_ptr);
  T *__restrict__ d_paged_v_cache = static_cast<T *>(paged_v_cache_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  int const *__restrict__ d_indices_buffer =
      static_cast<int const *>(paged_kv_indices_buffer_ptr);

  // DTensors' layouts
  using QDmem = dmem_row_const<T, NUM_Q_PER_KV, HEAD_DIM, HEAD_DIM>;
  using KVDmem = dmem_row_const<T, 1, HEAD_DIM, HEAD_DIM>;
  using KVCacheDmem = dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_STRIDE>;
  using ODmem = dmem_row<T, NUM_Q_PER_KV, HEAD_DIM, HEAD_DIM>;

  QDmem q_dmem(d_q);
  KVDmem k_dmem(d_k), v_dmem(d_v);
  KVCacheDmem paged_k_cache_dmem(d_paged_k_cache),
      paged_v_cache_dmem(d_paged_v_cache);
  ODmem o_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets and sizes
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  constexpr size_t ZERO_BUFFER_SIZE = sizeof(T) * 8;

  constexpr size_t S_Q_OFFSET = ZERO_BUFFER_OFFSET + ZERO_BUFFER_SIZE;
  constexpr size_t S_Q_SIZE = sizeof(T) * NUM_Q_PER_KV * HEAD_DIM;

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
  constexpr size_t S_M_BUFFER_SIZE = sizeof(float) * NUM_THREADS;

  constexpr size_t S_D_BUFFER_OFFSET = S_M_BUFFER_OFFSET + S_M_BUFFER_SIZE;
  constexpr size_t S_D_BUFFER_SIZE = sizeof(float) * NUM_THREADS;

  constexpr size_t S_O_BUFFER_OFFSET = S_D_BUFFER_OFFSET + S_D_BUFFER_SIZE;
  // constexpr size_t S_O_BUFFER_SIZE = sizeof(float) * NUM_THREADS * 32;

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
  using QOSmem = smem_row<T, 3, 3, 3, NUM_Q_PER_KV, HEAD_DIM, HEAD_DIM>;
  using KVSmem = smem_row<T, 3, 3, 3, KV_TILE_SIZE, HEAD_DIM, HEAD_DIM>;

  ZeroBufferSmem zero_buffer(zero_buf);
  QOSmem q_smem(s_q), o_smem(s_o);
  KVSmem k_smem(s_k), v_smem(s_v);
  KVSmem k_buffer_smem(s_k_buffer), v_buffer_smem(s_v_buffer);

  int const num_pages = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
  __shared__ int page_indices[MAX_PAGES_PER_REQUEST];

  if (num_pages % (16 / sizeof(int)) != 0) {
    int tail_pages = num_pages % (16 / sizeof(int));
    int tail_offset = num_pages - tail_pages;
    for (int i = threadIdx.x; i < tail_pages; i += NUM_THREADS) {
      page_indices[tail_offset + i] = d_indices_buffer[tail_offset + i];
    }
  }
  __syncthreads();

  size_t const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
  int curr_iter_len = min(seq_len, KV_TILE_SIZE);
  int cp_finished_seq_len = 0;

#pragma unroll
  for (int chunk_idx = threadIdx.x; chunk_idx < NUM_CHUNKS_Q;
       chunk_idx += NUM_THREADS) {
    int row = chunk_idx >> log2_constexpr(NUM_CHUNKS_PER_ROW);
    int col = (chunk_idx & (NUM_CHUNKS_PER_ROW - 1))
              << log2_constexpr(CP_CHUNK_SIZE);
    load_smem(q_smem(row, col), q_dmem(row, col));
  }

#pragma unroll
  for (int chunk_idx = threadIdx.x;
       chunk_idx < curr_iter_len << log2_constexpr(NUM_CHUNKS_PER_ROW);
       chunk_idx += NUM_THREADS) {
    int dst_row = chunk_idx >> log2_constexpr(NUM_CHUNKS_PER_ROW);
    int col = (chunk_idx & (NUM_CHUNKS_PER_ROW - 1))
              << log2_constexpr(CP_CHUNK_SIZE);
    if (dst_row + cp_finished_seq_len < seq_len - 1) {
      // load from KV Cache
      int page_idx = page_indices[(dst_row + cp_finished_seq_len) / PAGE_SIZE];
      int page_offset = (dst_row + cp_finished_seq_len) % PAGE_SIZE;
      int src_row = page_idx * PAGE_SIZE + page_offset;
      load_smem(k_buffer_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
      load_smem(v_buffer_smem(dst_row, col), paged_v_cache_dmem(src_row, col));
    } else {
      // load from QKV
      int src_row = 0;
      load_smem(k_buffer_smem(dst_row, col), k_dmem(src_row, col));
      load_smem(v_buffer_smem(dst_row, col), v_dmem(src_row, col));
    }
  }
  cp_async_fence();
  cp_finished_seq_len += curr_iter_len;

  // metadata for FlashAttention
  float m = -inf;
  float d = 1.f;
  float o[HEAD_DIM / 16][8];
#pragma unroll
  for (int n = 0; n < HEAD_DIM / 16; n++) {
    clear_8_floats(o[n]);
  }

  for (size_t iter = 0; iter < num_iters; iter++) {
    int next_iter_len = iter + 1 < num_iters
                            ? min(seq_len - cp_finished_seq_len, KV_TILE_SIZE)
                            : 0;
    if (next_iter_len > 0) {
#pragma unroll
      for (int chunk_idx = threadIdx.x;
           chunk_idx < next_iter_len << log2_constexpr(NUM_CHUNKS_PER_ROW);
           chunk_idx += NUM_THREADS) {
        int dst_row = chunk_idx >> log2_constexpr(NUM_CHUNKS_PER_ROW);
        int col = (chunk_idx & (NUM_CHUNKS_PER_ROW - 1))
                  << log2_constexpr(CP_CHUNK_SIZE);
        if (dst_row + cp_finished_seq_len < seq_len - 1) {
          // load from KV Cache
          int page_idx =
              page_indices[(dst_row + cp_finished_seq_len) / PAGE_SIZE];
          int page_offset = (dst_row + cp_finished_seq_len) % PAGE_SIZE;
          int src_row = page_idx * PAGE_SIZE + page_offset;
          load_smem(k_smem(dst_row, col), paged_k_cache_dmem(src_row, col));
          load_smem(v_smem(dst_row, col), paged_v_cache_dmem(src_row, col));
        } else {
          // load from QKV
          int src_row = 0;
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
        rms_norm<T, QOSmem, NUM_Q_PER_KV, HEAD_DIM>(
            q_smem,
            static_cast<T const *>(q_norm_weight_ptr),
            s_q_norm_sum,
            q_eps,
            1 /*num_tokens*/,
            0,
            rope,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
      }
      // K norm
      else if (iter == num_iters - 1) {
        rms_norm<T, KVSmem, 1, HEAD_DIM>(
            k_smem,
            static_cast<T const *>(k_norm_weight_ptr),
            s_k_norm_sum,
            k_eps,
            1 /*num_tokens*/,
            curr_iter_len - 1,
            rope,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
      }
    } else {
      if (rope) {
        // q rope
        if (iter == 0) {
          rotary_embedding<T, QOSmem, NUM_Q_PER_KV, 1, HEAD_DIM>(
              q_smem,
              static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM,
              0);
        }
        // k rope
        else if (iter == num_iters - 1) {
          rotary_embedding<T, KVSmem, 1, 1, HEAD_DIM>(
              k_smem,
              static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
              static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM,
              curr_iter_len - 1);
        }
      }
    }

    __syncthreads();

    // compute X = QK^T
    // NOTE(Jinchen): assume NUM_Q_PER_KV <= 8, KV_TILE_SIZE == 64
    // and HEAD_DIM % 16 == 0. Here we use m16n16k16 mma, and let warp
    // layout be 1x4x1, so mma only iterates over k dimension
    float x_frag_f[8];
    clear_8_floats(x_frag_f);
    uint32_t q_frag[4], kt_frag[4];

    int q_row = lane_idx & 0xF;
    int kt_col = (warp_idx << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
    for (int k = 0; k < HEAD_DIM / 16; k++) {
      int q_col = (k << 4) + ((lane_idx >> 4) << 3);
      int kt_row = (k << 4) + (((lane_idx & 0xF) >> 3) << 3);
      T *src_ptr_Q =
          q_row < NUM_Q_PER_KV ? q_smem(q_row, q_col) : zero_buffer(0, 0);
      T *src_ptr_KT =
          kt_col < curr_iter_len ? k_smem(kt_col, kt_row) : zero_buffer(0, 0);
      ldsm(src_ptr_Q, q_frag);
      ldsm(src_ptr_KT, kt_frag);
      mma_m16n16k16_bf16bf16bf32(x_frag_f, q_frag, kt_frag, x_frag_f);
    }
    __syncthreads();

    // update m: get partial max
    // NOTE(Jinchen): assume NUM_Q_PER_KV <= 8, then each thread
    // only needs to maintain 1 partial max, which is just the maximum
    // of x_frag_f[0], [1], [4], [5]
    float m_prev = m;
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        int idx = (i << 2) + j;
        int col = (warp_idx << 4) + ((lane_idx & 0x3) << 1) + (i << 3) + j;
        x_frag_f[idx] = col < curr_iter_len ? x_frag_f[idx] : -inf;
        m = max(x_frag_f[idx], m);
      }
    }

    // update m: get local max across 4 threads in a row
    m = max(m, shfl_xor_sync(m, 0x1));
    m = max(m, shfl_xor_sync(m, 0x2));

    float rescale = expf(m_prev * sm_scale - m * sm_scale);

    // update d: get partial sum
    float d_partial = 0.f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        int idx = (i << 2) + j;
        int row = lane_idx >> 2;
        int col = (warp_idx << 4) + ((lane_idx & 0x3) << 1) + (i << 3) + j;
        x_frag_f[idx] =
            row < NUM_Q_PER_KV && col < curr_iter_len && x_frag_f[idx] != -inf
                ? expf(x_frag_f[idx] * sm_scale - m * sm_scale)
                : 0.f;
        d_partial += x_frag_f[idx];
      }
    }

    // update d: get local sum across 4 threads in a row
    d_partial += shfl_xor_sync(d_partial, 0x1);
    d_partial += shfl_xor_sync(d_partial, 0x2);
    d *= rescale;
    d += d_partial;

    // update o: rescale
#pragma unroll
    for (int n = 0; n < HEAD_DIM / 16; n++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        o[n][i] *= rescale;
      }
    }

    // update o: compute O = exp(X - m) * V and accumulate
    // use m16n16k16 mma to compute and let warp layout be 1x1x4
    uint32_t x_frag[4], v_frag[4];
    convert_f32_to_bf16_uint32(x_frag_f, x_frag);
    int v_row = (warp_idx << 4) + (lane_idx & 0xF);
    for (int n = 0; n < HEAD_DIM / 16; n++) {
      int v_col = (n << 4) + ((lane_idx >> 4) << 3);
      T *src_ptr_V =
          v_row < curr_iter_len ? v_smem(v_row, v_col) : zero_buffer(0, 0);
      ldsm_t(src_ptr_V, v_frag);
      mma_m16n16k16_bf16bf16bf32(o[n], x_frag, v_frag, o[n]);
    }
    __syncthreads();

    curr_iter_len = next_iter_len;
  }

  // write intermediate results to buffer in shared memory
  m *= m != -inf ? sm_scale : 1.f;
  s_m_buffer[threadIdx.x] = m;
  s_d_buffer[threadIdx.x] = d;
  for (int n = 0; n < HEAD_DIM / 16; n++) {
    for (int i = 0; i < 2; i++) {
      s_o_buffer[(threadIdx.x << 5) + (n << 2) + (i << 1)] = o[n][i << 2];
      s_o_buffer[(threadIdx.x << 5) + (n << 2) + (i << 1) + 1] =
          o[n][(i << 2) + 1];
    }
  }
  __syncthreads();

  // get global m, d, and o
  m = -inf;
  d = 1.f;
  if (warp_idx == 0) {
#pragma unroll
    for (int t_idx = 0; t_idx < 4; t_idx++) {
      int smem_idx = (t_idx << 5) + ((lane_idx >> 2) << 2) + (lane_idx & 0x3);
      float other_m = s_m_buffer[smem_idx];
      float other_d = s_d_buffer[smem_idx];
      float m_prev = m, d_prev = d;
      m = max(m_prev, other_m);
      d = d_prev * expf(m_prev - m) + other_d * expf(other_m - m);
      // accumulate o
#pragma unroll
      for (int n = 0; n < HEAD_DIM / 16; n++) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
          float other_o_0 = s_o_buffer[(smem_idx << 5) + (n << 2) + (i << 1)];
          float other_o_1 =
              s_o_buffer[(smem_idx << 5) + (n << 2) + (i << 1) + 1];
          o[n][i << 2] =
              o[n][i << 2] * expf(m_prev - m) + other_o_0 * expf(other_m - m);
          o[n][(i << 2) + 1] = o[n][(i << 2) + 1] * expf(m_prev - m) +
                               other_o_1 * expf(other_m - m);
        }
      }
    }
  }
  __syncthreads();

  // compute O / d and store to shared memory
  if (warp_idx == 0) {
#pragma unroll
    for (int n = 0; n < HEAD_DIM / 16; n++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int row = (lane_idx >> 2) + ((i & 0x1) << 3);
        int col = (n << 4) + ((i >> 1) << 3) + ((lane_idx & 0x3) << 1);
        if (row < NUM_Q_PER_KV) {
          o_smem.at(row, col) = bfloat16(o[n][i << 1] / d);
          o_smem.at(row, col + 1) = bfloat16(o[n][(i << 1) + 1] / d);
        }
      }
    }
  }
  __syncthreads();

// update KV Cache
#pragma unroll
  for (int col = threadIdx.x; col < HEAD_DIM; col += NUM_THREADS) {
    int src_row = (seq_len - 1) & (KV_TILE_SIZE - 1);
    // NOTE(Jinchen): page allocation is done outside of the task,
    // so we assume there are always available slots in the last page,
    int page_idx = page_indices[num_pages - 1];
    int page_offset = (seq_len - 1) % PAGE_SIZE;
    int dst_row = page_idx * PAGE_SIZE + page_offset;
    paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
    paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
  }

// store O to device memory
#pragma unroll
  for (int elem_idx = threadIdx.x; elem_idx < NUM_Q_PER_KV * HEAD_DIM;
       elem_idx += NUM_THREADS) {
    int row = elem_idx >> log2_constexpr(HEAD_DIM);
    int col = elem_idx & (HEAD_DIM - 1);
    o_dmem.at(row, col) = o_smem.at(row, col);
  }
}

} // namespace kernel
