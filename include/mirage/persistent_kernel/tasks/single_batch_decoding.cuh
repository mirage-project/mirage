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

// kernel Input: 9X128, K_Cache: 4KX128, V_Cache:4KX128
// Load Q = 8 X 128, K = 1 X 128, V = 1 X 128
// load K into K_Cache, V into V_cache
template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int WEIGHT_STRIDE>
__device__ __forceinline__ void
    single_batch_decoding_kernel(void const *qkv_ptr,
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

  size_t num_iterations = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(seq_len, KV_CHUNK_SIZE);
  int cp_finished_seq_len = curr_iter_len;
  int last_seq_len = curr_iter_len;

  __restrict__ T const *d_q = static_cast<T const *>(qkv_ptr);
  __restrict__ T const *d_k =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * NUM_Q_HEADS;
  __restrict__ T const *d_v =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * (NUM_Q_HEADS + NUM_KV_HEADS);
  T __restrict__ *d_k_cache = static_cast<T *>(k_cache_ptr);
  T __restrict__ *d_v_cache = static_cast<T *>(v_cache_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  dmem_row_const<T, NUM_Q_HEADS, 128, 128> q_dmem(d_q);
  dmem_row_const<T, 1, 128, 128> k_dmem(d_k);
  dmem_row_const<T, 1, 128, 128> v_dmem(d_v);
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> k_cache_dmem(d_k_cache);
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> v_cache_dmem(d_v_cache);
  dmem_row<T, NUM_Q_HEADS, 128, 128> output_dmem(d_output);
  extern __shared__ char smem[];

  // copy input
  T *shared_q = (T *)(smem + 128);
  // copy weight
  T *shared_k = (T *)(smem + 1920);
  T *shared_k_buffer = (T *)(smem + 18304);

  T *shared_v = (T *)(smem + 34688);
  T *shared_v_buffer = (T *)(smem + 51072);
  // intermidiate
  T *shared_output = (T *)(smem + 128);
  T *zero_buf = (T *)(smem);

  // flashattn metadata
  float *d_smem = (float *)(smem + 67456);
  float *max_smem = (float *)(smem + 67968);
  float *o_smem = (float *)(smem + 68480);

  float *qnorm_sum = (float *)(smem + 84864);
  float *knorm_sum = (float *)(smem + 84880);
  // define the swizzle mode

  // zero buffer
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);
  // smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128> q_smem(shared_q);
  // //(16, 128) per stage
  // smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128> k_cache_smem(shared_k);
  // smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128> k_cache_smem_buffer(
  //     shared_k_buffer);

  // smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128> v_cache_smem(shared_v);
  // smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128> v_cache_smem_buffer(
  //     shared_v_buffer);

  using QSmem = smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128>;
  using KSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using VSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using OSmem = smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128>;
  QSmem q_smem(shared_q);

  KSmem k_cache_smem(shared_k);
  KSmem k_cache_smem_buffer(shared_k_buffer);
  VSmem v_cache_smem(shared_v);
  VSmem v_cache_smem_buffer(shared_v_buffer);
  OSmem output_smem(shared_output);

  // smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128> output_smem(shared_output);

  // todo, add a chunk assigned function
  vec_zero_t<T, 8>::fill_zero(zero_buf);

// load first Q, K, V
#pragma unroll
  for (int i = threadIdx.x; i < NUM_Q_HEADS * (HEAD_DIM / 8);
       i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    load_smem(q_smem(row, col), q_dmem(row, col));
  }

  // metadata for flashattention
  float o[8][8];
#pragma unroll
  for (int n = 0; n < 8; n++) {
    clear_8_floats(o[n]);
  }
  float d_sum = 1.f;
  float m = -inf;

  // 16 * 128
#pragma unroll
  for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    if (row == seq_len - 1) {
      // from qkv
      load_smem(k_cache_smem_buffer(row, col), k_dmem(0, col));
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
    if (row == seq_len - 1) {
      // from qkv
      load_smem(v_cache_smem_buffer(row, col), v_dmem(0, col));
    } else {
      // from cache
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
            ? static_cast<int>(std::min(seq_len, (kv_idx + 2) * KV_CHUNK_SIZE) -
                               (kv_idx + 1) * KV_CHUNK_SIZE)
            : -1;

    if (kv_idx + 1 != num_iterations) {
#pragma unroll
      for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
        // offset
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row + cp_finished_seq_len == seq_len - 1) {
          load_smem(k_cache_smem(row, col), k_dmem(0, col));
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
        if (row + cp_finished_seq_len == seq_len - 1) {
          load_smem(v_cache_smem(row, col), v_dmem(0, col));
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

    if (qk_norm) {
      if (kv_idx == 0) {
        // q_norm && rope
        rms_norm<T, QSmem, NUM_Q_HEADS, 1, HEAD_DIM>(
            q_smem,
            static_cast<T const *>(qnorm_weight_ptr),
            qnorm_sum,
            q_eps,
            0,
            rotary_emd,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
      } else if (kv_idx == num_iterations - 1) {
        // knorm && rope
        rms_norm<T, KSmem, NUM_KV_HEADS, 1, HEAD_DIM>(
            k_cache_smem,
            static_cast<T const *>(knorm_weight_ptr),
            knorm_sum,
            k_eps,
            curr_iter_len - 1,
            rotary_emd,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
      }
    } else {
      if (rotary_emd && kv_idx == 0) {
        // q rope
        rotary_embedding<T, QSmem, NUM_Q_HEADS, 1, HEAD_DIM>(
            q_smem,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM,
            0);
      } else if (rotary_emd && kv_idx == num_iterations - 1) {
        // k rope
        rotary_embedding<T, KSmem, NUM_KV_HEADS, 1, HEAD_DIM>(
            k_cache_smem,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM,
            curr_iter_len - 1);
      }
    }

    __syncthreads();

    float s_frag[8];
    clear_8_floats(s_frag);

    uint32_t a_frag[4], b_frag[4], v_frag[4];

    // QK^T
    //  MNK = 7, 64, 128, tiledMMA 7, 64, 16, thread layout 1,4,1
    int m_row = idx_in_warp % 16;
    int n_row = (idx_in_warp / 16) * 8 + (idx_in_warp % 8) + warp_idx * 16;
    //  int n_col = warp_idx * 16 + idx_in_warp / 16 * 8;
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      int m_col = k * 16 + idx_in_warp / 16 * 8;
      //  int n_col = idx_in_warp / 16 * 8 + k * 16;
      int n_col = ((idx_in_warp % 16) / 8) * 8 + k * 16;
      //  int n_row = idx_in_warp % 16 + k * 16;
      bool is_valid_A = (m_row < NUM_Q_HEADS);
      T *src_ptr_A = is_valid_A ? q_smem(m_row, m_col) : zero_buffer(0, 0);
      ldsm(src_ptr_A, &a_frag[0]);
      bool is_valid_B = (n_row < curr_iter_len);
      T *src_ptr_B =
          is_valid_B ? k_cache_smem(n_row, n_col) : zero_buffer(0, 0);
      ldsm(src_ptr_B, &b_frag[0]);
      mma_m16n16k16_bf16bf16bf32(s_frag, a_frag, b_frag, s_frag);
    }
    __syncthreads();

    // update flashattention
    float m_prev = m;

    // get local max
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // update M, apply mask when length doesn't match the padding length 16
        int idx = i * 4 + j;
        // int row = idx_in_warp / 4;
        int col = (idx_in_warp % 4) * 2 + i * 8 + j + warp_idx * 16;
        s_frag[idx] = (col < curr_iter_len) ? s_frag[idx] : -inf;
        m = max(s_frag[idx], m);
      }
    }

    // get global max across 4 threads
    m = max(m, shfl_xor_sync(m, 0x2));
    m = max(m, shfl_xor_sync(m, 0x1));

    // update m, d, o
    // float o_scale = expf(m_prev * sm_scale - m * sm_scale);
    float o_scale = expf(m_prev * sm_scale - m * sm_scale);
    float d_local = 0.f;
    d_sum *= o_scale;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int idx = i * 4 + j;
        // s_frag[idx] = expf(s_frag[idx] * sm_scale - m * sm_scale);
        int col = (idx_in_warp % 4) * 2 + i * 8 + j + warp_idx * 16;
        int row = idx_in_warp / 4;
        // 0 means exp(-inf)
        s_frag[idx] = ((col < curr_iter_len) && (row < NUM_Q_HEADS))
                          ? expf(s_frag[idx] * sm_scale - m * sm_scale)
                          : 0;
        d_local += s_frag[idx];
      }
    }

// update o
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[n][i] *= o_scale;
      }
    }

    // sum the d across 4threads
    d_local += shfl_xor_sync(d_local, 0X1);
    d_local += shfl_xor_sync(d_local, 0X2);
    d_sum += d_local;
    // m *= sm_scale;

    // //QK^T * V
    uint32_t o_frag[4];
    convert_f32_to_bf16_uint32(s_frag, o_frag);

    for (int n = 0; n < 8; n++) {
      int v_row = idx_in_warp % 16 + warp_idx * 16;
      int v_col = idx_in_warp / 16 * 8 + n * 16;
      bool is_valid_C = (v_row < curr_iter_len);
      T *src_ptr_C =
          is_valid_C ? v_cache_smem(v_row, v_col) : zero_buffer(0, 0);
      ldsm_t(src_ptr_C, v_frag);
      mma_m16n16k16_bf16bf16bf32(o[n], o_frag, v_frag, o[n]);
    }
    __syncthreads();

    if (kv_idx != num_iterations) {
      last_seq_len = curr_iter_len;
      cp_finished_seq_len += next_iter_len;
      curr_iter_len = next_iter_len;
    }
  } // kv_idx

  for (int n = 0; n < 8; n++) {
    // write the result to osmem, index is 0, 1, 4, 5
    for (int i = 0; i < 2; i++) {
      o_smem[threadIdx.x * 32 + n * 4 + i * 2] = o[n][i * 4];
      o_smem[threadIdx.x * 32 + n * 4 + i * 2 + 1] = o[n][i * 4 + 1];
    }
  }

  // norm m
  if (m != -inf) {
    m *= sm_scale;
  }
  d_smem[threadIdx.x] = d_sum;
  max_smem[threadIdx.x] = m;
  __syncthreads();
  m = -inf;
  d_sum = 1.f;
  // update flashattention metadata across threads
  //  sotre o back to smem and divide by d, 16X16X8X4 -> 16X128 -> 7X128
  if (warp_idx == 0) {
#pragma unroll
    for (uint32_t tidx = 0; tidx < 4; tidx++) {
      // head idx is idx in warp / 4
      // TODO: this is int shmem_idx = idx_in_warp + tidx * 32
      // This is actually mapping to previous threadid
      int shmem_idx = (idx_in_warp / 4) * 4 + tidx * 32 + (idx_in_warp % 4);
      float other_m = max_smem[shmem_idx];
      float other_d = d_smem[shmem_idx];
      // update o,m,d across threads
      float m_prev = m, d_prev = d_sum;
      m = max(m_prev, other_m);
      d_sum = d_prev * expf(m_prev - m) + other_d * expf(other_m - m);
      if (warp_idx == 0) {
        // reduction on K
#pragma unroll
        for (uint32_t n = 0; n < 8; n++) {
#pragma unroll
          for (uint32_t frag_idx = 0; frag_idx < 2; frag_idx++) {
            float o_new1 = o_smem[shmem_idx * 32 + n * 4 + frag_idx * 2];
            float o_new2 = o_smem[shmem_idx * 32 + n * 4 + frag_idx * 2 + 1];
            o[n][frag_idx * 4] = o[n][frag_idx * 4] * expf(m_prev - m) +
                                 o_new1 * expf(other_m - m);
            o[n][frag_idx * 4 + 1] = o[n][frag_idx * 4 + 1] * expf(m_prev - m) +
                                     o_new2 * expf(other_m - m);
          }
        }
      }
    }
  }
  __syncthreads();

  // update the o and m and d on other warps
  // print each head
#pragma unroll
  for (int n = 0; n < 8; n++) {
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
      if (warp_idx == 0) {
        int row = idx_in_warp / 4 + 8 * (i % 2);
        int col = (idx_in_warp % 4) * 2 + 8 * (i / 2) + n * 16;
        if (row < NUM_Q_HEADS) {
          output_smem.at(row, col) = bfloat16(o[n][i * 2] / d_sum);
          output_smem.at(row, col + 1) = bfloat16(o[n][i * 2 + 1] / d_sum);
        }
      }
    }
  }
  __syncthreads();

// update KV cache
#pragma unroll
  for (int i = threadIdx.x; i < 128; i += NUM_THREADS) {
    // offset
    int row = seq_len - 1;
    int col = i;
    k_cache_dmem.at(row, col) = k_cache_smem.at(last_seq_len - 1, col);
  }

#pragma unroll
  for (int i = threadIdx.x; i < 128; i += NUM_THREADS) {
    // offset
    int row = seq_len - 1;
    int col = i;
    v_cache_dmem.at(row, col) = v_cache_smem.at(last_seq_len - 1, col);
  }

// write output to device memory
#pragma unroll
  for (int i = threadIdx.x; i < (NUM_Q_HEADS * 128); i += NUM_THREADS) {
    // offset
    int row = i / 128;
    int col = (i % 128);
    output_dmem.at(row, col) = output_smem.at(row, col);
  }
}

} // namespace kernel
