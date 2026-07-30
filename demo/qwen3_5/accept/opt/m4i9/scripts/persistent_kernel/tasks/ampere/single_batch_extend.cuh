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
          int WEIGHT_STRIDE,
          int OUTPUT_STRIDE,
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
                               float k_eps,
                               void *q_norm_debug_ptr = nullptr,
                               void *k_norm_debug_ptr = nullptr) {
  // print norm weight

  // constexpr int chunk_size = 16 / sizeof(T);
  constexpr size_t MAX_SEQ_LEN = 512;
  constexpr size_t KV_CHUNK_SIZE = 64;
  float const sm_scale = (1.f / sqrt((float)HEAD_DIM));

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  size_t total_seq_len = seq_len + EXTEND_NUM;

  constexpr int TOTAL_Q_VEC_NUM = NUM_Q_HEADS * (EXTEND_NUM + 1);
  constexpr int NUM_Q_TOKEN_DIM_ITER = (TOTAL_Q_VEC_NUM + 16 - 1) / 16;
  size_t num_iterations = (total_seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(total_seq_len, KV_CHUNK_SIZE);
  int cp_finished_seq_len = curr_iter_len;
  int last_seq_len = curr_iter_len;

  // These are just the first line of the new qkvs
  __restrict__ T const *d_q = static_cast<T const *>(qkv_ptr);
  __restrict__ T const *d_k =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * NUM_Q_HEADS;
  __restrict__ T const *d_v =
      static_cast<T const *>(qkv_ptr) + HEAD_DIM * (NUM_Q_HEADS + NUM_KV_HEADS);
  constexpr int GLOBAL_QKVS_OFFSET =
      HEAD_DIM * (NUM_Q_HEADS + NUM_KV_HEADS + NUM_KV_HEADS) * 8;
  T __restrict__ *d_k_cache = static_cast<T *>(k_cache_ptr);
  T __restrict__ *d_v_cache = static_cast<T *>(v_cache_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  // Debug output tensors (optional)
  T *d_q_norm_debug = static_cast<T *>(q_norm_debug_ptr);
  T *d_k_norm_debug = static_cast<T *>(k_norm_debug_ptr);

  // second & third parameter in the template is actually not used
  dmem_row_const<T, NUM_Q_HEADS, NUM_Q_HEADS * 128, GLOBAL_QKVS_OFFSET> q_dmem(
      d_q);
  dmem_row_const<T, 1, HEAD_DIM, GLOBAL_QKVS_OFFSET> k_dmem(d_k);
  dmem_row_const<T, 1, HEAD_DIM, GLOBAL_QKVS_OFFSET> v_dmem(d_v);
  dmem_row<T, MAX_SEQ_LEN, HEAD_DIM, WEIGHT_STRIDE> k_cache_dmem(d_k_cache);
  dmem_row<T, MAX_SEQ_LEN, HEAD_DIM, WEIGHT_STRIDE> v_cache_dmem(d_v_cache);
  dmem_row<T, TOTAL_Q_VEC_NUM, NUM_Q_HEADS * HEAD_DIM, OUTPUT_STRIDE>
      output_dmem(d_output); // [NUM_Q_HEADS * (EXTEND_NUM + 1) * 128 * 2B]

  // Debug tensor layouts (if enabled)
  // dmem_row<T, NUM_Q_HEADS, 128, (EXTEND_NUM + 1) * 128>
  // q_norm_debug_dmem(d_q_norm_debug); // [NUM_Q_HEADS, EXTEND_NUM+1, HEAD_DIM]
  dmem_row<T, (EXTEND_NUM + 1) * NUM_Q_HEADS, 128, 128> q_norm_debug_dmem(
      d_q_norm_debug); // [NUM_Q_HEADS, EXTEND_NUM+1, HEAD_DIM]
  dmem_row<T, EXTEND_NUM + 1, 128, 128> k_norm_debug_dmem(
      d_k_norm_debug); // [EXTEND_NUM+1, HEAD_DIM]

  extern __shared__ char smem[];

  constexpr size_t SHARED_Q_OFFSET = 128;

  constexpr size_t SHARED_K_OFFSET =
      SHARED_Q_OFFSET + TOTAL_Q_VEC_NUM * HEAD_DIM * sizeof(T);
  constexpr size_t SHARED_K_BUFFER_OFFSET =
      SHARED_K_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  constexpr size_t SHARED_V_OFFSET =
      SHARED_K_BUFFER_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  constexpr size_t SHARED_V_BUFFER_OFFSET =
      SHARED_V_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);

  constexpr size_t D_OFFSET =
      SHARED_V_BUFFER_OFFSET + KV_CHUNK_SIZE * HEAD_DIM * sizeof(T);
  constexpr size_t MAX_OFFSET =
      D_OFFSET + NUM_Q_TOKEN_DIM_ITER * 2 * NUM_THREADS * sizeof(float);
  constexpr size_t O_OFFSET =
      MAX_OFFSET + NUM_Q_TOKEN_DIM_ITER * 2 * NUM_THREADS * sizeof(float);

  constexpr size_t Q_NORM_SUM_OFFSET =
      O_OFFSET + NUM_Q_TOKEN_DIM_ITER * 2 * NUM_THREADS * 4 /*0 1 4 5 for t0*/ *
                     8 /* 8 iter along hidden dim */ *
                     sizeof(float); // TODO: check
  constexpr size_t K_NORM_SUM_OFFSET =
      Q_NORM_SUM_OFFSET + NUM_WARPS * sizeof(float);

  constexpr size_t TOTAL_SHARED_MEM_SIZE =
      K_NORM_SUM_OFFSET + NUM_WARPS * sizeof(float);

  constexpr size_t SHARED_OUTPUT_OFFSET = 128;
  constexpr size_t ZERO_BUFFER_OFFSET = 0;

  // copy input
  T *shared_q = (T *)(smem + SHARED_Q_OFFSET); // 1792 bytes (4 * 6 * 128 * 2B)
  // copy weight
  T *shared_k = (T *)(smem + SHARED_K_OFFSET); // 16384 bytes (64 * 128 * 2B)
  T *shared_k_buffer =
      (T *)(smem + SHARED_K_BUFFER_OFFSET); // 16384 bytes (64 * 128 * 2B)

  T *shared_v = (T *)(smem + SHARED_V_OFFSET); // 16384 bytes (64 * 128 * 2B)
  T *shared_v_buffer =
      (T *)(smem + SHARED_V_BUFFER_OFFSET); // 16384 bytes (64 * 128 * 2B)
  // intermidiate
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET); // reuse shared_q
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);        // 16 bytes (8 * 2B)

  // flashattn metadata
  float *d_smem = (float *)(smem + D_OFFSET);     // 512 bytes (128 * 4B)
  float *max_smem = (float *)(smem + MAX_OFFSET); // 512 bytes (128 * 4B)
  float *o_smem = (float *)(smem + O_OFFSET);     // 16384 bytes (128 * 32 * 4B)

  float *qnorm_sum = (float *)(smem + Q_NORM_SUM_OFFSET); // 16 bytes (4 * 4B)
  float *knorm_sum = (float *)(smem + K_NORM_SUM_OFFSET); // 16 bytes (4 * 4B)
  // define the swizzle mode

  // zero buffer
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);

  using QSmem = smem_row<T, 3, 3, 3, TOTAL_Q_VEC_NUM, 128, 128>;
  using KSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using VSmem = smem_row<T, 3, 3, 3, KV_CHUNK_SIZE, 128, 128>;
  using OSmem = smem_row<T, 3, 3, 3, TOTAL_Q_VEC_NUM, 128, 128>;
  QSmem q_smem(shared_q);

  KSmem k_cache_smem(shared_k);               // 16384 bytes (64 * 128 * 2B)
  KSmem k_cache_smem_buffer(shared_k_buffer); // 16384 bytes (64 * 128 * 2B)
  VSmem v_cache_smem(shared_v);               // 16384 bytes (64 * 128 * 2B)
  VSmem v_cache_smem_buffer(shared_v_buffer); // 16384 bytes (64 * 128 * 2B)
  OSmem output_smem(shared_output);           // [4 * 128 * 2B]

  // smem_row<T, 3, 3, 3, NUM_Q_HEADS, 128, 128> output_smem(shared_output);

  // todo, add a chunk assigned function
  vec_zero_t<T, 8>::fill_zero(zero_buf);

// load first Q, K, V
#pragma unroll
  for (int i = threadIdx.x; i < TOTAL_Q_VEC_NUM * (HEAD_DIM / 8);
       i += NUM_THREADS) {
    // offset in shared memory
    int q_smem_row = i / 16;
    int q_smem_col = (i % 16) * 8;

    int q_dmem_row = i / (NUM_Q_HEADS * (HEAD_DIM / 8));
    int q_dmem_col = (i % (NUM_Q_HEADS * (HEAD_DIM / 8))) * 8;

    load_smem(q_smem(q_smem_row, q_smem_col), q_dmem(q_dmem_row, q_dmem_col));
  }

#pragma unroll
  for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    if (row >= seq_len - 1) { // last and extended tokens
      // from qk
      int token_idx = row - (seq_len - 1);
      load_smem(k_cache_smem_buffer(row, col), k_dmem(token_idx, col));
    } else {
      // from cache
      load_smem(k_cache_smem_buffer(row, col), k_cache_dmem(row, col));
    }
  }

// V data loading: extract V from each token's QKV data for new tokens
#pragma unroll
  for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
    // offset
    int row = i / 16;
    int col = (i % 16) * 8;
    if (row >= seq_len - 1) { // last and extended tokens
      // from qkv
      int token_idx = row - (seq_len - 1);
      load_smem(v_cache_smem_buffer(row, col), v_dmem(token_idx, col));
    } else {
      load_smem(v_cache_smem_buffer(row, col), v_cache_dmem(row, col));
    }
  }
  cp_async_fence();

  // metadata for flashattention
  float o[NUM_Q_TOKEN_DIM_ITER][8][8];
#pragma unroll
  for (int q_head_i = 0; q_head_i < NUM_Q_TOKEN_DIM_ITER; q_head_i++) {
#pragma unroll
    for (int n = 0; n < 8; n++) {
      clear_8_floats(o[q_head_i][n]);
    }
  }
  // float d_sum = 1.f;
  float d_sum[2 * NUM_Q_TOKEN_DIM_ITER];
#pragma unroll
  for (int i = 0; i < 2 * NUM_Q_TOKEN_DIM_ITER; i++) {
    d_sum[i] = 1.f;
  }
  // float m = -inf;
  float m[2 * NUM_Q_TOKEN_DIM_ITER];
#pragma unroll
  for (int i = 0; i < 2 * NUM_Q_TOKEN_DIM_ITER; i++) {
    // TODO: Chunk value assignment
    m[i] = -inf;
  }

  /*
  KV iteration
  */

  //  N = 64 per iter
  for (uint32_t kv_idx = 0; kv_idx < num_iterations; kv_idx += 1) {
    // load next k, v
    int next_iter_len =
        kv_idx + 1 < num_iterations
            ? static_cast<int>(
                  std::min(total_seq_len, (kv_idx + 2) * KV_CHUNK_SIZE) -
                  (kv_idx + 1) * KV_CHUNK_SIZE)
            : -1;

    // Current chunk range: [kv_idx * KV_CHUNK_SIZE, kv_idx * KV_CHUNK_SIZE +
    // curr_iter_len) New tokens range: [seq_len-1, seq_len+EXTEND_NUM)
    // (includes original last token + EXTEND_NUM new tokens) Calculate
    // intersection and write back if any
    int chunk_start = kv_idx * KV_CHUNK_SIZE;
    int chunk_end = chunk_start + curr_iter_len;
    int new_tokens_start = seq_len - 1;
    int new_tokens_end =
        seq_len + EXTEND_NUM; // includes all EXTEND_NUM+1 tokens

    // These are used for norm and write back judgement
    int cur_chunk_new_kv_start = max(chunk_start, new_tokens_start);
    int cur_chunk_new_kv_end = min(chunk_end, new_tokens_end);

    // async load next k, v
    if (kv_idx + 1 != num_iterations) {
#pragma unroll
      for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
        // offset
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row + cp_finished_seq_len >= seq_len - 1) {
          // from qkv
          int token_idx = row + cp_finished_seq_len - (seq_len - 1);
          load_smem(k_cache_smem(row, col), k_dmem(token_idx, col));
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
          // from qkv
          int token_idx = row + cp_finished_seq_len - (seq_len - 1);
          load_smem(v_cache_smem(row, col), v_dmem(token_idx, col));
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
      // Q norm - only execute in the first chunk (kv_idx == 0) since all Q
      // tokens are loaded at once
      if (kv_idx == 0) {
        rms_norm<T, QSmem, NUM_Q_HEADS, HEAD_DIM>(
            q_smem,
            static_cast<T const *>(qnorm_weight_ptr),
            qnorm_sum,
            q_eps,
            EXTEND_NUM + 1 /*window_size*/,
            0, // token_offset = 0 for Q tokens
            rotary_emd,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM);
      }
      // K norm - only execute for chunks that contain new K tokens
      if (cur_chunk_new_kv_start < cur_chunk_new_kv_end) {
        for (int kv_pos = cur_chunk_new_kv_start; kv_pos < cur_chunk_new_kv_end;
             kv_pos++) {

          int token_offset_in_chunk = kv_pos - chunk_start;

          rms_norm<T, KSmem, NUM_KV_HEADS, HEAD_DIM>(
              k_cache_smem,
              static_cast<T const *>(knorm_weight_ptr),
              knorm_sum,
              k_eps,
              1 /*window_size*/,
              token_offset_in_chunk,
              rotary_emd,
              static_cast<T const *>(cos_ptr) + kv_pos * HEAD_DIM,
              static_cast<T const *>(sin_ptr) + kv_pos * HEAD_DIM);
        }
      }
    } else {
      if (rotary_emd && kv_idx == 0) {
        // q rope
        rotary_embedding<T, QSmem, NUM_Q_HEADS, EXTEND_NUM + 1, HEAD_DIM>(
            q_smem,
            static_cast<T const *>(cos_ptr) + (seq_len - 1) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len - 1) * HEAD_DIM,
            0);
      } else if (rotary_emd && cur_chunk_new_kv_start < cur_chunk_new_kv_end) {
        for (int kv_pos = cur_chunk_new_kv_start; kv_pos < cur_chunk_new_kv_end;
             kv_pos++) {
          int token_offset_in_chunk = kv_pos - chunk_start;
          // k rope
          rotary_embedding<T, KSmem, NUM_KV_HEADS, 1, HEAD_DIM>(
              k_cache_smem,
              static_cast<T const *>(cos_ptr) + kv_pos * HEAD_DIM,
              static_cast<T const *>(sin_ptr) + kv_pos * HEAD_DIM,
              token_offset_in_chunk);
        }
      }
    }

    __syncthreads();

    // MMA
    float s_frag[NUM_Q_TOKEN_DIM_ITER][8];
#pragma unroll
    for (int i = 0; i < NUM_Q_TOKEN_DIM_ITER; i++) {
      clear_8_floats(s_frag[i]);
    }

    uint32_t a_frag[4], b_frag[4], v_frag[4];

    // QK^T
    //  MNK = 4 * (extend_num + 1), 64, 128, tiledMMA 16, 64, 16, thread layout
    //  () int n_col = warp_idx * 16 + idx_in_warp / 16 * 8;
    int n_row = (idx_in_warp / 16) * 8 + (idx_in_warp % 8) + warp_idx * 16;

#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {

      int m_col = k * 16 + idx_in_warp / 16 * 8;
      int n_col = ((idx_in_warp % 16) / 8) * 8 + k * 16;

      bool is_valid_B = (n_row < curr_iter_len);
      T *src_ptr_B =
          is_valid_B ? k_cache_smem(n_row, n_col) : zero_buffer(0, 0);
      ldsm(src_ptr_B, &b_frag[0]);
#pragma unroll
      for (int q_head_i = 0; q_head_i < NUM_Q_TOKEN_DIM_ITER; q_head_i++) {

        int m_row = idx_in_warp % 16 + q_head_i * 16;

        bool is_valid_A = (m_row < TOTAL_Q_VEC_NUM);
        T *src_ptr_A = is_valid_A ? q_smem(m_row, m_col) : zero_buffer(0, 0);
        ldsm(src_ptr_A, &a_frag[0]);

        mma_m16n16k16_bf16bf16bf32(
            s_frag[q_head_i], a_frag, b_frag, s_frag[q_head_i]);
      }
    }
    __syncthreads();

#pragma unroll
    for (int q_head_i = 0; q_head_i < NUM_Q_TOKEN_DIM_ITER; q_head_i++) {
#pragma unroll
      for (int l = 0; l < 2; l++) { // upper 8 or lower 8
        // get local max
        float m_prev = m[q_head_i * 2 + l];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            // update M, apply mask when length doesn't match the padding length
            // 16
            int idx = l * 2 + i * 4 + j; // 0 1 4 5 / 2 3 6 7
            // int row = idx_in_warp / 4;
            int col = (idx_in_warp % 4) * 2 + i * 8 + j +
                      warp_idx * 16; // [16, 64]'s col

            // Apply causal mask
            int q_row = idx_in_warp / 4 + q_head_i * 16 +
                        l * 8; // Q token row in the tensor
            int q_token_idx =
                q_row / NUM_Q_HEADS; // Which Q token (0 to EXTEND_NUM)
            int q_token_pos =
                seq_len - 1 + q_token_idx; // Absolute position of Q token
            int k_cache_pos =
                kv_idx * KV_CHUNK_SIZE + col; // Absolute position of K token

            // Apply both padding mask and causal mask
            bool is_valid =
                (col < curr_iter_len) && (k_cache_pos <= q_token_pos);
            s_frag[q_head_i][idx] = is_valid ? s_frag[q_head_i][idx] : -inf;
            m[q_head_i * 2 + l] =
                max(s_frag[q_head_i][idx], m[q_head_i * 2 + l]);
          }
        }
        // get global max across 4 threads
        m[q_head_i * 2 + l] =
            max(m[q_head_i * 2 + l], shfl_xor_sync(m[q_head_i * 2 + l], 0x2));
        m[q_head_i * 2 + l] =
            max(m[q_head_i * 2 + l], shfl_xor_sync(m[q_head_i * 2 + l], 0x1));

        // update m, d, o
        // float o_scale = expf(m_prev * sm_scale - m * sm_scale);
        float o_scale =
            expf(m_prev * sm_scale - m[q_head_i * 2 + l] * sm_scale);
        float d_local = 0.f;
        d_sum[q_head_i * 2 + l] *= o_scale;

// update across local 4 threads
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            int idx = l * 2 + i * 4 + j;
            // s_frag[idx] = expf(s_frag[idx] * sm_scale - m * sm_scale);
            int col = (idx_in_warp % 4) * 2 + i * 8 + j + warp_idx * 16;
            int row = idx_in_warp / 4 + q_head_i * 16 + l * 8;
            // 0 means exp(-inf)

            // Apply causal mask in softmax computation too
            int q_token_idx_softmax =
                row / NUM_Q_HEADS; // Which Q token (0 to EXTEND_NUM)
            int q_token_pos_softmax =
                seq_len - 1 +
                q_token_idx_softmax; // Absolute position of Q token
            int k_cache_pos_softmax =
                kv_idx * KV_CHUNK_SIZE + col; // Absolute position of K token
            bool is_valid_softmax =
                (col < curr_iter_len) && (row < TOTAL_Q_VEC_NUM) &&
                (k_cache_pos_softmax <= q_token_pos_softmax);

            s_frag[q_head_i][idx] =
                is_valid_softmax ? expf(s_frag[q_head_i][idx] * sm_scale -
                                        m[q_head_i * 2 + l] * sm_scale)
                                 : 0;
            d_local += s_frag[q_head_i][idx];
          }
        }

// update o
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          o[q_head_i][n][0 + 2 * l] *= o_scale;
          o[q_head_i][n][1 + 2 * l] *= o_scale;
          o[q_head_i][n][4 + 2 * l] *= o_scale;
          o[q_head_i][n][5 + 2 * l] *= o_scale;
        }
        // sum the d across 4threads
        d_local += shfl_xor_sync(d_local, 0X1);
        d_local += shfl_xor_sync(d_local, 0X2);
        d_sum[q_head_i * 2 + l] += d_local;
      } // l

      // //QK^T * V
      uint32_t o_frag[4];
      convert_f32_to_bf16_uint32(s_frag[q_head_i], o_frag);

      for (int n = 0; n < 8; n++) {
        int v_row = idx_in_warp % 16 + warp_idx * 16;
        int v_col = idx_in_warp / 16 * 8 + n * 16;
        bool is_valid_C = (v_row < curr_iter_len);
        T *src_ptr_C =
            is_valid_C ? v_cache_smem(v_row, v_col) : zero_buffer(0, 0);
        ldsm_t(src_ptr_C, v_frag);
        mma_m16n16k16_bf16bf16bf32(
            o[q_head_i][n], o_frag, v_frag, o[q_head_i][n]);
      }
      __syncthreads();

    } // q_head_i

    // Write back new K and V tokens in the current chunk to KV cache
    if (cur_chunk_new_kv_start < cur_chunk_new_kv_end) {
#pragma unroll
      for (int kv_cache_row = cur_chunk_new_kv_start;
           kv_cache_row < cur_chunk_new_kv_end;
           kv_cache_row++) {
#pragma unroll
        for (int i = threadIdx.x; i < 128; i += NUM_THREADS) {
          int col = i;
          int smem_row =
              kv_cache_row -
              chunk_start; // relative position in the current chunk's smem
          k_cache_dmem.at(kv_cache_row, col) = k_cache_smem.at(smem_row, col);
          v_cache_dmem.at(kv_cache_row, col) = v_cache_smem.at(smem_row, col);
        }
      }
    }

    if (kv_idx != num_iterations) {
      last_seq_len = curr_iter_len;
      cp_finished_seq_len += next_iter_len;
      curr_iter_len = next_iter_len;
    }
  } // kv_idx

#pragma unroll
  for (int q_head_i = 0; q_head_i < NUM_Q_TOKEN_DIM_ITER; q_head_i++) {
#pragma unroll
    for (int l = 0; l < 2; l++) { // (0 1 4 5) or (2 3 6 7)
#pragma unroll
      for (int n = 0; n < 8; n++) { // 8 blocks along HEAD_DIM
// write the result to osmem, index is 0, 1, 4, 5 / 2, 3, 6, 7
#pragma unroll
        for (int i = 0; i < 2; i++) {  // (0 1) or (4 5)
          int reg_idx = l * 2 + i * 4; // 0, 1, 4, 5 / 2, 3, 6, 7
          int osmem_offset = q_head_i * 2 * NUM_THREADS * 32 +
                             l * NUM_THREADS * 32 + threadIdx.x * 32 + n * 4 +
                             i * 2;
          // 0&1 / 4&5 / 2&3 / 6&7
          o_smem[osmem_offset] = o[q_head_i][n][reg_idx];
          o_smem[osmem_offset + 1] = o[q_head_i][n][reg_idx + 1];
        }
      }
      if (m[q_head_i * 2 + l] != -inf) {
        m[q_head_i * 2 + l] *= sm_scale;
      }
      d_smem[threadIdx.x] = d_sum[q_head_i * 2 + l];
      max_smem[threadIdx.x] = m[q_head_i * 2 + l];
      __syncthreads();
      m[q_head_i * 2 + l] = -inf;
      d_sum[q_head_i * 2 + l] = 1.f;

      if (warp_idx == 0) {
// Reduce across 4 warps
#pragma unroll
        for (uint32_t warp_id = 0; warp_id < 4; warp_id++) {
          // head idx is idx in warp / 4
          // TODO: I think this is int shmem_idx = idx_in_warp + warp_id * 32
          // This is actually mapping to the related threads in different warps
          int shmem_idx =
              (idx_in_warp / 4) * 4 + warp_id * 32 + (idx_in_warp % 4);
          float other_m = max_smem[shmem_idx];
          float other_d = d_smem[shmem_idx];
          // update o,m,d across threads
          float m_prev = m[q_head_i * 2 + l], d_prev = d_sum[q_head_i * 2 + l];
          m[q_head_i * 2 + l] = max(m_prev, other_m);
          d_sum[q_head_i * 2 + l] =
              d_prev * expf(m_prev - m[q_head_i * 2 + l]) +
              other_d * expf(other_m - m[q_head_i * 2 + l]);

// reduction on K
#pragma unroll
          for (uint32_t n = 0; n < 8; n++) {
#pragma unroll
            for (uint32_t frag_idx = 0; frag_idx < 2; frag_idx++) {
              int osmem_offset = q_head_i * 2 * NUM_THREADS * 32 +
                                 l * NUM_THREADS * 32 + shmem_idx * 32 + n * 4 +
                                 frag_idx * 2;
              int reg_idx = frag_idx * 4 + l * 2; // 01 -> 45 --> 23 -> 67
              float o_new1 = o_smem[osmem_offset];
              float o_new2 = o_smem[osmem_offset + 1];
              o[q_head_i][n][reg_idx] =
                  o[q_head_i][n][reg_idx] * expf(m_prev - m[q_head_i * 2 + l]) +
                  o_new1 * expf(other_m - m[q_head_i * 2 + l]);
              o[q_head_i][n][reg_idx + 1] =
                  o[q_head_i][n][reg_idx + 1] *
                      expf(m_prev - m[q_head_i * 2 + l]) +
                  o_new2 * expf(other_m - m[q_head_i * 2 + l]);
            } // frag_idx
          }   // n
        }     // warp_id
      }       // only warp_idx == 0
      __syncthreads();

    } // l

    if (warp_idx == 0) {
// update the o and m and d on other warps
// print each head
#pragma unroll
      for (int n = 0; n < 8; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          // if (warp_idx == 0) {
          int row = idx_in_warp / 4 + 8 * (i % 2) + q_head_i * 16;
          int col = (idx_in_warp % 4) * 2 + 8 * (i / 2) + n * 16;
          // i   row   col
          // 0   0     0
          // 1   8     0
          // 2   0     8
          // 3   8     8
          if (row < TOTAL_Q_VEC_NUM) {
            output_smem.at(row, col) =
                bfloat16(o[q_head_i][n][i * 2] / d_sum[q_head_i * 2 + i % 2]);
            output_smem.at(row, col + 1) = bfloat16(
                o[q_head_i][n][i * 2 + 1] / d_sum[q_head_i * 2 + i % 2]);
          } else {
            output_smem.at(row, col) = bfloat16(0.0f);
            output_smem.at(row, col + 1) = bfloat16(0.0f);
          }
        }
      }
    }
    __syncthreads();
  } // q_head_i

// write output to device memory
#pragma unroll
  for (int i = threadIdx.x; i < (TOTAL_Q_VEC_NUM * HEAD_DIM);
       i += NUM_THREADS) {
    // offset
    int smem_row = i / HEAD_DIM;
    int smem_col = i % HEAD_DIM;

    int dmem_row = i / (NUM_Q_HEADS * HEAD_DIM);
    int dmem_col = (i % (NUM_Q_HEADS * HEAD_DIM));
    output_dmem.at(dmem_row, dmem_col) = output_smem.at(smem_row, smem_col);
  }
}

} // namespace kernel