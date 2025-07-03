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
#include "smem_layout.cuh"
#include "utils.cuh"
namespace kernel {

// kernel Input: 9X128, K_Cache: 4KX128, V_Cache:4KX128
// Load Q = 8 X 128, K = 1 X 128, V = 1 X 128
// load K into K_Cache, V into V_cache
template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int WEIGHT_STRIDE,
          int EXTEND_NUM>
__device__ __forceinline__ void
    single_batch_extend_kernel(void const *qkv_ptr,
                                 void *k_cache_ptr,
                                 void *v_cache_ptr,
                                 void *output_ptr,
                                 size_t seq_len,
                                 bool qk_norm,
                                 bool rotary_emd,
                                 void const *qnorm_weight_ptr,
                                 void const *knorm_weight_ptr,
                                 void const *cos_ptr,
                                 void const *sin_ptr,
                                 float q_eps,
                                 float k_eps) {
  // constexpr int chunk_size = 16 / sizeof(T);
  constexpr size_t MAX_SEQ_LEN = 512;
  constexpr size_t KV_CHUNK_SIZE = 64;
  float const sm_scale = (1.f / sqrt((float)HEAD_DIM));
  // float const sm_scale = 1.f;

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  size_t total_seq_len = seq_len + EXTEND_NUM;

  size_t num_iterations = (total_seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(total_seq_len, KV_CHUNK_SIZE);
  int cp_finished_seq_len = curr_iter_len;
  int last_seq_len = curr_iter_len;

  const __restrict__ T *d_q = static_cast<T const *>(qkv_ptr);
  const __restrict__ T *d_k =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * NUM_Q_HEADS;
  const __restrict__ T *d_v =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * (NUM_Q_HEADS + NUM_KV_HEADS);
  T __restrict__ *d_k_cache = static_cast<T *>(k_cache_ptr);
  T __restrict__ *d_v_cache = static_cast<T *>(v_cache_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  dmem_row_const<T, NUM_Q_HEADS, 128, 128> q_dmem(d_q); // [4 * 128 * 2B]
  dmem_row_const<T, 1, 128, 128> k_dmem(d_k); // [1 * 128 * 2B]
  dmem_row_const<T, 1, 128, 128> v_dmem(d_v); // [1 * 128 * 2B]
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> k_cache_dmem(d_k_cache);
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> v_cache_dmem(d_v_cache);
  dmem_row<T, NUM_Q_HEADS, 128, 128> output_dmem(d_output);

  extern __shared__ char smem[];

  constexpr size_t SHARED_Q_OFFSET = 128;

  constexpr size_t SHARED_K_OFFSET = SHARED_Q_OFFSET + EXTEND_NUM * HEAD_DIM * sizeof(T);
  constexpr size_t SHARED_K_BUFFER_OFFSET = SHARED_K_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  constexpr size_t SHARED_V_OFFSET = SHARED_K_BUFFER_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  constexpr size_t SHARED_V_BUFFER_OFFSET = SHARED_V_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  
  constexpr size_t D_OFFSET = SHARED_V_BUFFER_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  constexpr size_t MAX_OFFSET = D_OFFSET + HEAD_DIM * sizeof(float);
  constexpr size_t O_OFFSET = MAX_OFFSET + HEAD_DIM * sizeof(float);

  constexpr size_t Q_NORM_SUM_OFFSET = O_OFFSET + HEAD_DIM * 32 * sizeof(float); // TODO: check
  constexpr size_t K_NORM_SUM_OFFSET = Q_NORM_SUM_OFFSET + NUM_WARPS * sizeof(float);
  
  constexpr size_t SHARED_OUTPUT_OFFSET = 128;
  constexpr size_t ZERO_BUFFER_OFFSET = 0;

  // copy input
  T *shared_q = (T *)(smem + SHARED_Q_OFFSET); // 1792 bytes (7 * 128 * 2B)
  // copy weight
  T *shared_k = (T *)(smem + SHARED_K_OFFSET); // 16384 bytes (64 * 128 * 2B)
  T *shared_k_buffer = (T *)(smem + SHARED_K_BUFFER_OFFSET); // 16384 bytes (64 * 128 * 2B)

  T *shared_v = (T *)(smem + SHARED_V_OFFSET); // 16384 bytes (64 * 128 * 2B)
  T *shared_v_buffer = (T *)(smem + SHARED_V_BUFFER_OFFSET); // 16384 bytes (64 * 128 * 2B)
  // intermidiate
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET); // reuse shared_q
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET); // 16 bytes (8 * 2B)

  // flashattn metadata
  float *d_smem = (float *)(smem + D_OFFSET); // 512 bytes (128 * 4B)
  float *max_smem = (float *)(smem + MAX_OFFSET); // 512 bytes (128 * 4B)
  float *o_smem = (float *)(smem + O_OFFSET); // 16384 bytes (128 * 32 * 4B)

  float *qnorm_sum = (float *)(smem + Q_NORM_SUM_OFFSET); // 16 bytes (4 * 4B)
  float *knorm_sum = (float *)(smem + K_NORM_SUM_OFFSET); // 16 bytes (4 * 4B)
  // define the swizzle mode

  // zero buffer
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);

  using QSmem = smem_row<T, 3, 3, 3, NUM_Q_HEADS * EXTEND_NUM, 128, 128>;
  using KSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using VSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using OSmem = smem_row<T, 3, 3, 3, NUM_Q_HEADS * EXTEND_NUM, 128, 128>;
  QSmem q_smem(shared_q);

  KSmem k_cache_smem(shared_k); // 16384 bytes (64 * 128 * 2B)
  KSmem k_cache_smem_buffer(shared_k_buffer); // 16384 bytes (64 * 128 * 2B)
  VSmem v_cache_smem(shared_v); // 16384 bytes (64 * 128 * 2B)
  VSmem v_cache_smem_buffer(shared_v_buffer); // 16384 bytes (64 * 128 * 2B)
  OSmem output_smem(shared_output); // [4 * 128 * 2B]

  // smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128> output_smem(shared_output);

  // todo, add a chunk assigned function
  for (int i = 0; i < 8; i++) {
    zero_buffer.at(i) = (bfloat16)0.0f;
  }

  // load first Q, K, V
#pragma unroll
  for (int i = threadIdx.x; i < NUM_Q_HEADS * EXTEND_NUM * (HEAD_DIM / 8);
       i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    load_smem(q_smem(row, col), q_dmem(row, col));
  }

  // metadata for flashattention
  // TODO: check if this is enough
  float o[8][8];
#pragma unroll
  for (int n = 0; n < 8; n++) {
    clear_8_floats(o[n]);
  }
  float d_sum = 1.f;
  float m = -inf;

#pragma unroll
  for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    if (row >= seq_len - 1) { // last and extended tokens
      // from qkv
      load_smem(k_cache_smem_buffer(row, col), k_dmem(row - (seq_len - 1), col));
    } else {
      // from cache
      load_smem(k_cache_smem_buffer(row, col), k_cache_dmem(row, col));
    }
  }

  #pragma unroll
  for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    if (row >= seq_len - 1) { // last and extended tokens
      load_smem(v_cache_smem_buffer(row, col), v_dmem(row - (seq_len - 1), col));
    } else {
      load_smem(v_cache_smem_buffer(row, col), v_cache_dmem(row, col));
    }
  }
  cp_async_fence();

  // KV iteration
  //  N = 64 per iter
  for (uint32_t kv_idx = 0; kv_idx < num_iterations; kv_idx += 1) {
    // load next k, v
    int next_iter_len =
        kv_idx + 1 < num_iterations
            ? static_cast<int>(std::min(total_seq_len, (kv_idx + 2) * KV_CHUNK_SIZE) -
                               (kv_idx + 1) * KV_CHUNK_SIZE)
            : -1;

      // async load next k, v
      if (kv_idx + 1 != num_iterations) {
#pragma unroll
      for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
        // offset
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row + cp_finished_seq_len >= seq_len - 1) {
          load_smem(k_cache_smem(row, col), k_dmem(row + cp_finished_seq_len - (seq_len - 1), col));
        } else {
          load_smem(k_cache_smem(row, col),
                    k_cache_dmem(cp_finished_seq_len + row, col));
        }
      }
#pragma unroll
      for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
        // offset
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row + cp_finished_seq_len >= seq_len - 1) { 
          load_smem(v_cache_smem(row, col), v_dmem(row + cp_finished_seq_len - (seq_len - 1), col));
        } else {
          load_smem(v_cache_smem(row, col),
                    v_cache_dmem(cp_finished_seq_len + row, col));
        }
      }
      cp_async_fence();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }

    if ((kv_idx & 1) == 0) {
      k_cache_smem.set_ptr(shared_k_buffer);
      k_cache_smem_buffer.set_ptr(shared_k);
      v_cache_smem.set_ptr(shared_v_buffer);
      v_cache_smem_buffer.set_ptr(shared_v);
    } else {
      k_cache_smem.set_ptr(shared_k);
      k_cache_smem_buffer.set_ptr(shared_k_buffer);
      v_cache_smem.set_ptr(shared_v);
      v_cache_smem_buffer.set_ptr(shared_v_buffer);
    }
    __syncthreads();

    // q_norm
    if (qk_norm && kv_idx == 0) {
      window_rms_norm<T, QSmem, NUM_Q_HEADS, EXTEND_NUM + 1, HEAD_DIM>(
          q_smem,
          static_cast<T const *>(qnorm_weight_ptr),
          qnorm_sum,
          q_eps,
          rotary_emd,
          static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM, //TODO: check the index of cos and sin
          static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
    }

    // knorm
    if (qk_norm && kv_idx == num_iterations - 1) {
      window_rms_norm<T, KSmem, NUM_KV_HEADS, EXTEND_NUM + 1, HEAD_DIM>(
          k_cache_smem,
          static_cast<T const *>(knorm_weight_ptr),
          knorm_sum,
          k_eps,
          rotary_emd,
          static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM, //TODO: check the index of cos and sin
          static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
    }
    __syncthreads();

    // MMA
    constexpr int NUM_Q_TOKEN_DIM_ITER = (NUM_Q_HEADS * EXTEND_NUM + 16 - 1) / 16;
    float s_frag[NUM_Q_TOKEN_DIM_ITER][8];
    #pragma unroll
    for(int i = 0; i < NUM_Q_TOKEN_DIM_ITER; i++) {
      clear_8_floats(s_frag[i]);
    }

    uint32_t a_frag[4], b_frag[4], v_frag[4];

    // QK^T
    //  MNK = 4 * (extend_num + 1), 64, 128, tiledMMA 16, 64, 16, thread layout ()
    //  int n_col = warp_idx * 16 + idx_in_warp / 16 * 8;
    int n_row = (idx_in_warp / 16) * 8 + (idx_in_warp % 8) + warp_idx * 16;

    #pragma unroll
    for (uint32_t k = 0; k < 8; k++) {

      int m_col = k * 16 + idx_in_warp / 16 * 8;
      int n_col = ((idx_in_warp % 16) / 8) * 8 + k * 16;

      #pragma unroll
      for(int q_head_i = 0; q_head_i < NUM_Q_TOKEN_DIM_ITER; q_head_i++) {

        int m_row = idx_in_warp % 16 + q_head_i * 16;

        bool is_valid_A = (m_row < NUM_Q_HEADS * EXTEND_NUM);
        T *src_ptr_A = is_valid_A ? q_smem(m_row, m_col) : zero_buffer(0, 0);
        ldsm(src_ptr_A, &a_frag[0]);

        bool is_valid_B = (n_row < curr_iter_len);
        T *src_ptr_B =
            is_valid_B ? k_cache_smem(n_row, n_col) : zero_buffer(0, 0);
        ldsm(src_ptr_B, &b_frag[0]);

        mma_m16n16k16_bf16bf16bf32(s_frag, a_frag, b_frag, s_frag);
      }
    }
    __syncthreads();
    // To be continued...
  }



  // To be continued...


}

} // namespace kernel
