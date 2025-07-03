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

  size_t num_iterations = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(seq_len, KV_CHUNK_SIZE);
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
  constexpr size_t K_NORM_SUM_OFFSET = Q_NORM_SUM_OFFSET + HEAD_DIM * sizeof(float);
  
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


  // To be continued...


}

} // namespace kernel
