
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
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"
namespace kernel {

// kernel Input: 9X128, K_Cache: 4KX128, V_Cache:4KX128
// Load Q = 8 X 128, K = 1 X 128, V = 1 X 128
// load K into K_Cache, V into V_cache
template <typename T, size_t SEQ_LEN>
__device__ __forceinline__ void
    single_batch_decoding_kernel(void const *qkv_ptr,
                                 void *k_cache_ptr,
                                 void *v_cache_ptr,
                                 void *output_ptr) {
  constexpr int chunk_size = 16 / sizeof(T);
  constexpr size_t MAX_SEQ_LEN = 512;
  constexpr size_t KV_CHUNK_SIZE = 64;
  constexpr int NUM_Q_HEADS = 7;
  constexpr int NUM_K_HEADS = 1;
  constexpr int NUM_V_HEADS = 1;
  constexpr int HEAD_DIM = 128;

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  size_t num_iterations = (SEQ_LEN + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(SEQ_LEN, KV_CHUNK_SIZE);

  const __restrict__ T *d_q = static_cast<T const *>(qkv_ptr);
  const __restrict__ T *d_k =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * NUM_Q_HEADS;
  const __restrict__ T *d_v =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * (NUM_Q_HEADS + NUM_K_HEADS);
  T __restrict__ *d_k_cache = static_cast<T *>(k_cache_ptr);
  T __restrict__ *d_v_cache = static_cast<T *>(v_cache_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  dmem_row_const<T, 7, 128, 128> q_dmem(d_q);
  dmem_row_const<T, 128, 1, 1> k_dmem(d_k);
  dmem_row_const<T, 1, 128, 128> v_dmem(d_v);
  dmem_row<T, 128, SEQ_LEN, SEQ_LEN> k_cache_dmem(d_k_cache);
  dmem_row<T, 128, SEQ_LEN, SEQ_LEN> v_cache_dmem(d_v_cache);
  dmem_row<T, 7, 128, 128> output_dmem(d_output);

  extern __shared__ char smem[];

  // copy input
  T *shared_q = (T *)(smem + 128);
  // T *shared_q_buffer = (T *)(smem + 1920);
  // copy weight
  T *shared_k = (T *)(smem + 1920);
  T *shared_k_buffer = (T *)(smem + 18304);

  T *shared_v = (T *)(smem + 34688);
  T *shared_v_buffer = (T *)(smem + 51072);
  // intermidiate
  T *mm_output = (T *)(smem + 128);
  T *shared_output = (T *)(smem + 128);
  T *zero_buf = (T *)(smem);

  // flashattn metadata
  float *d_smem = (float *)(smem + 67456);
  float *max_smem = (float *)(smem + 67968);
  // float *o_smem = (float *)(smem + 68480);
  // define the swizzle mode

  // zero buffer
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);

  smem_row<T, 3, 3, 3, 7, 128, 128> q_smem(shared_q);
  // smem_row<T, 3, 3, 3, 7, 128, 128> q_smem_buffer(shared_q_buffer);

  //(16, 128) per stage
  smem_row<T, 3, 3, 3, 128, KV_CHUNK_SIZE, KV_CHUNK_SIZE> k_cache_smem(
      shared_k);
  smem_row<T, 3, 3, 3, 128, KV_CHUNK_SIZE, KV_CHUNK_SIZE> k_cache_smem_buffer(
      shared_k_buffer);

  smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128> v_cache_smem(shared_v);
  smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128> v_cache_smem_buffer(
      shared_v_buffer);

  smem_row<T, 3, 3, 3, 7, 128, 128> output_smem(shared_output);

// load first Q, K, V
#pragma unroll
  for (int i = threadIdx.x; i < 112; i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    load_smem(q_smem(row, col), q_dmem(row, col));
  }

  // metadata for flashattention
  float o[2][8];
#pragma unroll
  for (int n = 0; n < 2; n++) {
#pragma unroll
    for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
      o[n][frag_idx] = 0.0f;
    }
  }
  float d = 1.f;
  float m = -inf;

  // 16 * 128
#pragma unroll
  for (int i = threadIdx.x; i < (128 * (curr_iter_len / 8)); i += NUM_THREADS) {
    // offset
    int row = i / (curr_iter_len / 8);
    int col = (i % (curr_iter_len / 8)) * 8;
    if (col == curr_iter_len - 1) {
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
    if (row == curr_iter_len - 1) {
      // from qkv
      load_smem(v_cache_smem_buffer(row, col), v_dmem(0, col));
    } else {
      // from cache
      load_smem(v_cache_smem_buffer(row, col), v_cache_dmem(row, col));
    }
  }
  cp_async_fence();

  float s_frag[8];
#pragma unroll
  for (int frag_idx = 0; frag_idx < 8; frag_idx++) {
    s_frag[frag_idx] = 0.0f;
  }

  // KV iteration
  //  N = 64 per iter
  for (uint32_t kv_idx = 0; kv_idx < num_iterations; kv_idx += 1) {
    // load next k, v
    int next_iter_len = kv_idx + 1 < num_iterations
                            ? std::min(SEQ_LEN, (kv_idx + 2) * KV_CHUNK_SIZE) -
                                  (kv_idx + 1) * KV_CHUNK_SIZE
                            : -1;

    if (kv_idx + 1 != num_iterations) {
#pragma unroll
      for (int i = threadIdx.x; i < (128 * (next_iter_len / 8));
           i += NUM_THREADS) {
        // offset
        int row = i / (next_iter_len / 8);
        int col = (i % (next_iter_len / 8)) * 8;
        if (col == next_iter_len - 1) {
          load_smem(k_cache_smem(row, col), k_dmem(0, col));
        } else {
          load_smem(k_cache_smem(row, col), k_cache_dmem(row, col));
        }
      }
#pragma unroll
      for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
        // offset
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row == next_iter_len - 1) {
          load_smem(v_cache_smem(row, col), v_dmem(0, col));
        } else {
          load_smem(v_cache_smem(row, col), v_cache_dmem(row, col));
        }
      }
    }
    cp_async_fence();
    cp_async_wait<1>();
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
    uint32_t a_frag[4], b_frag[4], v_frag[4];
    // MNK = 7, 64, 128, tiledMMA 7, 64, 16, thread layout 1,4,1
    int m_row = idx_in_warp % 16;
    // int n_row = idx_in_warp % 16 + warp_idx * 16;
    int n_col = warp_idx * 16 + idx_in_warp / 16 * 8;
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      int m_col = k * 16 + idx_in_warp / 16 * 8;
      // int n_col = idx_in_warp / 16 * 8 + k * 16;
      int n_row = idx_in_warp % 16 + k * 16;
      bool is_valid = (m_row < NUM_Q_HEADS);
      T *src_ptr = is_valid ? q_smem(m_row, m_col) : zero_buffer(0, 0);

      ldsm(src_ptr, &a_frag[0]);
      ldsm_t(k_cache_smem(n_row, n_col), &b_frag[0]);
      mma_m16n16k16_bf16bf16bf32(s_frag, a_frag, b_frag, s_frag);
    }
    __syncthreads();
    // printf("frag.x %d, %f, %f, %f, %f, %f, %f, %f, %f\n",
    //        threadIdx.x,
    //        s_frag[0],
    //        s_frag[1],
    //        s_frag[2],
    //        s_frag[3],
    //        s_frag[4],
    //        s_frag[5],
    //        s_frag[6],
    //        s_frag[7]);

    // each thread contains 4 values that belong to a head
    // head0 (T0, T1 T2 T3), (T32 T33 T34 T35)... = 16 threads = 64 values= 1 *
    // 16 * 4, not accumed head1 (T4 T5 T6 T7)   (T5 T6 T7 T8) firt get a local
    // max, we need frag 0,1 and 4,5, 23 and 78 are in the next 8X16, which is
    // not valid

    float m_prev = m;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // update M, apply mask when length doesn't match the padding length 16
        int idx = i * 4 + j;
        s_frag[idx] = (n_col < curr_iter_len) ? s_frag[idx] : -inf;
        m = max(s_frag[idx], m);
      }
    }
    // update m, d, o
    float o_scale = ptx_exp2(m_prev - m);
    __syncthreads();

    d *= o_scale;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int idx = i * 4 + j;
        s_frag[idx] = ptx_exp2(s_frag[idx] - m);
        d += s_frag[idx];
      }
    }
// update o
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[n][i] *= o_scale;
      }
    }
    // o is 16 * 64, v is 64 X 128 -> 16 * 128
    uint32_t o_frag[4];
    convert_f32_to_bf16_uint32(s_frag, o_frag);
    __syncthreads();

    for (int n = 0; n < 2; n++) {
      for (int k = 0; k < 4; k++) {
        int v_row = idx_in_warp % 16 + k * 16;
        int v_col = warp_idx * 16 + idx_in_warp / 16 * 8 + n * 64;
        ldsm(v_cache_smem(v_row, v_col), v_frag);
        mma_m16n16k16_bf16bf16bf32(o[n], o_frag, v_frag, o[n]);
      }
    }
    __syncthreads();

    // s_frag = 8  * 128, assume it is 1 * 16, num_n=4 = 16X16X4 = 64X16
    // write d/m/o to shared mem
    d_smem[threadIdx.x] = d;
    max_smem[threadIdx.x] = m;
    // reset m d o
    __syncthreads();

// each heads has 16 threads
#pragma unroll
    for (uint32_t tidx = 0; tidx < 16; tidx++) {
      // head idx is idx in warp / 4
      int shmem_idx = (idx_in_warp / 4) + (tidx / 4) * 32 + (tidx % 4);
      float other_m = max_smem[shmem_idx];
      float other_d = d_smem[shmem_idx];

      // update o,m,d across threads
      float m_prev = 0, d_prev = d;
      // m = (other_m == -inf) ? m : max(m_prev, other_m);
      m = max(m_prev, other_m);
      d = d_prev * ptx_exp2(m_prev - m) + other_d * ptx_exp2(other_m - m);
#pragma unroll
      for (uint32_t n = 0; n < 2; n++) {
#pragma unroll
        for (uint32_t frag_idx = 0; frag_idx < 8; frag_idx++) {
          o[n][frag_idx] = o[n][frag_idx] * ptx_exp2(m_prev - m);
        }
      }
    }
    curr_iter_len = next_iter_len;
  }

  // sotre o back to smem and divide by d, 16X16X8X4 -> 16X128 -> 7X128
#pragma unroll
  for (int n = 0; n < 2; n++) {
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
      int row = idx_in_warp / 4 + 8 * (i % 2);
      int col = (idx_in_warp % 4) * 2 + 16 * warp_idx + 8 * (i / 2) + n * 64;
      if (row >= NUM_Q_HEADS) {
        continue;
      }
      output_smem.at(row, col) = bfloat16(o[n][i * 2] / d);
      output_smem.at(row, col + 1) = bfloat16(o[n][i * 2 + 1] / d);
    }
  }
  __syncthreads();

// update KV cache
#pragma unroll
  for (int i = threadIdx.x; i < 128; i += NUM_THREADS) {
    // offset
    int col = SEQ_LEN - 1;
    int row = i;
    k_cache_dmem.at(row, col) = k_cache_smem.at(row, col);
  }
#pragma unroll
  for (int i = threadIdx.x; i < 128; i += NUM_THREADS) {
    // offset
    int row = SEQ_LEN - 1;
    int col = i;
    v_cache_dmem.at(row, col) = v_cache_smem.at(row, col);
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
