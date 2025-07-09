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

// Multi-token decoding kernel
// Input layout: [Q_token0, K_token0, V_token0, Q_token1, K_token1, V_token1, ...]
// Each token has NUM_Q_HEADS Q heads, NUM_KV_HEADS K heads, NUM_KV_HEADS V heads
template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int WEIGHT_STRIDE,
          int NUM_TOKENS>
__device__ __forceinline__ void
    single_batch_multitoken_decoding_kernel(void const *qkv_ptr,
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
  constexpr size_t MAX_SEQ_LEN = 512;
  constexpr size_t KV_CHUNK_SIZE = 64;
  float const sm_scale = (1.f / sqrt((float)HEAD_DIM));

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  // Ensure we don't exceed max sequence length
  if (seq_len + NUM_TOKENS - 1 >= MAX_SEQ_LEN) {
    return;  // Early exit to prevent buffer overflow
  }

  size_t num_iterations = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  int curr_iter_len = std::min(seq_len, KV_CHUNK_SIZE);
  int cp_finished_seq_len = curr_iter_len;
  int last_seq_len = curr_iter_len;

  const __restrict__ T *d_qkv = static_cast<T const *>(qkv_ptr);
  T __restrict__ *d_k_cache = static_cast<T *>(k_cache_ptr);
  T __restrict__ *d_v_cache = static_cast<T *>(v_cache_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  // Memory layouts for multi-token access
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> k_cache_dmem(d_k_cache);
  dmem_row<T, MAX_SEQ_LEN, 128, WEIGHT_STRIDE> v_cache_dmem(d_v_cache);
  
  extern __shared__ char smem[];

  // Shared memory layout
  T *shared_q = (T *)(smem + 128);
  T *shared_k = (T *)(smem + 1920);
  T *shared_k_buffer = (T *)(smem + 18304);
  T *shared_v = (T *)(smem + 34688);
  T *shared_v_buffer = (T *)(smem + 51072);
  T *shared_output = (T *)(smem + 128);
  T *zero_buf = (T *)(smem);

  // flashattn metadata - need more space for multi-token
  float *d_smem = (float *)(smem + 67456);
  float *max_smem = (float *)(smem + 67968);
  float *o_smem = (float *)(smem + 68480);

  float *qnorm_sum = (float *)(smem + 84864);
  float *knorm_sum = (float *)(smem + 84880);

  // Shared memory abstractions
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);
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

  // Initialize zero buffer
  for (int i = 0; i < 8; i++) {
    zero_buffer.at(i) = (bfloat16)0.0f;
  }

  // Process each token
  for (int token_idx = 0; token_idx < NUM_TOKENS; token_idx++) {
    // Calculate offsets for current token in input layout
    // Layout: [Q_heads, K_heads, V_heads] for each token
    int heads_per_token = NUM_Q_HEADS + NUM_KV_HEADS + NUM_KV_HEADS;
    int token_offset = token_idx * heads_per_token * HEAD_DIM;
    
    const T *d_q = d_qkv + token_offset;
    const T *d_k = d_q + NUM_Q_HEADS * HEAD_DIM;
    const T *d_v = d_k + NUM_KV_HEADS * HEAD_DIM;

    // Create memory views for current token
    dmem_row_const<T, NUM_Q_HEADS, 128, 128> q_dmem(d_q);
    dmem_row_const<T, NUM_KV_HEADS, 128, 128> k_dmem(d_k);
    dmem_row_const<T, NUM_KV_HEADS, 128, 128> v_dmem(d_v);
    dmem_row<T, NUM_Q_HEADS, 128, 128> output_dmem(d_output + token_idx * NUM_Q_HEADS * HEAD_DIM);

    // Load Q for current token
    #pragma unroll
    for (int i = threadIdx.x; i < NUM_Q_HEADS * (HEAD_DIM / 8); i += NUM_THREADS) {
      int row = i / 16;
      int col = (i % 16) * 8;
      load_smem(q_smem(row, col), q_dmem(row, col));
    }

    // Initialize flash attention metadata
    float o[8][8];
    #pragma unroll
    for (int n = 0; n < 8; n++) {
      clear_8_floats(o[n]);
    }
    float d_sum = 1.f;
    float m = -inf;

    // Load first KV chunk
    #pragma unroll
    for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
      int row = i / 16;
      int col = (i % 16) * 8;
      if (row == seq_len - 1 + token_idx) {
        // Load from current token's K
        load_smem(k_cache_smem_buffer(row, col), k_dmem(0, col));
      } else {
        // Load from cache
        load_smem(k_cache_smem_buffer(row, col), k_cache_dmem(row, col));
      }
    }

    #pragma unroll
    for (int i = threadIdx.x; i < (curr_iter_len * 16); i += NUM_THREADS) {
      int row = i / 16;
      int col = (i % 16) * 8;
      if (row == seq_len - 1 + token_idx) {
        // Load from current token's V
        load_smem(v_cache_smem_buffer(row, col), v_dmem(0, col));
      } else {
        // Load from cache
        load_smem(v_cache_smem_buffer(row, col), v_cache_dmem(row, col));
      }
    }
    cp_async_fence();

    // KV iteration
    for (uint32_t kv_idx = 0; kv_idx < num_iterations; kv_idx += 1) {
      int next_iter_len =
          kv_idx + 1 < num_iterations
              ? static_cast<int>(std::min(seq_len + token_idx + 1, 
                                         (kv_idx + 2) * KV_CHUNK_SIZE) -
                                 (kv_idx + 1) * KV_CHUNK_SIZE)
              : -1;

      if (kv_idx + 1 != num_iterations) {
        #pragma unroll
        for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
          int row = i / 16;
          int col = (i % 16) * 8;
          int global_row = cp_finished_seq_len + row;
          if (global_row == seq_len - 1 + token_idx) {
            load_smem(k_cache_smem(row, col), k_dmem(0, col));
          } else {
            load_smem(k_cache_smem(row, col), k_cache_dmem(global_row, col));
          }
        }
        #pragma unroll
        for (int i = threadIdx.x; i < (next_iter_len * 16); i += NUM_THREADS) {
          int row = i / 16;
          int col = (i % 16) * 8;
          int global_row = cp_finished_seq_len + row;
          if (global_row == seq_len - 1 + token_idx) {
            load_smem(v_cache_smem(row, col), v_dmem(0, col));
          } else {
            load_smem(v_cache_smem(row, col), v_cache_dmem(global_row, col));
          }
        }
        cp_async_fence();
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }

      // Swap buffers
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

      // Apply normalization if needed
      if (qk_norm && kv_idx == 0) {
        rms_norm<T, QSmem, NUM_Q_HEADS, HEAD_DIM>(
            q_smem,
            static_cast<T const *>(qnorm_weight_ptr),
            qnorm_sum,
            q_eps,
            0,
            rotary_emd,
            static_cast<T const *>(cos_ptr) + (seq_len + token_idx) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len + token_idx) * HEAD_DIM);
      }

      if (qk_norm && kv_idx == num_iterations - 1) {
        rms_norm<T, KSmem, NUM_KV_HEADS, HEAD_DIM>(
            k_cache_smem,
            static_cast<T const *>(knorm_weight_ptr),
            knorm_sum,
            k_eps,
            curr_iter_len - 1,
            rotary_emd,
            static_cast<T const *>(cos_ptr) + (seq_len + token_idx) * HEAD_DIM,
            static_cast<T const *>(sin_ptr) + (seq_len + token_idx) * HEAD_DIM);
      }
      __syncthreads();

      // Compute attention scores
      float s_frag[8];
      clear_8_floats(s_frag);

      uint32_t a_frag[4], b_frag[4], v_frag[4];

      // QK^T computation
      int m_row = idx_in_warp % 16;
      int n_row = (idx_in_warp / 16) * 8 + (idx_in_warp % 8) + warp_idx * 16;

      #pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        int m_col = k * 16 + idx_in_warp / 16 * 8;
        int n_col = ((idx_in_warp % 16) / 8) * 8 + k * 16;
        bool is_valid_A = (m_row < NUM_Q_HEADS);
        T *src_ptr_A = is_valid_A ? q_smem(m_row, m_col) : zero_buffer(0, 0);
        ldsm(src_ptr_A, &a_frag[0]);
        bool is_valid_B = (n_row < curr_iter_len);
        T *src_ptr_B = is_valid_B ? k_cache_smem(n_row, n_col) : zero_buffer(0, 0);
        ldsm(src_ptr_B, &b_frag[0]);
        mma_m16n16k16_bf16bf16bf32(s_frag, a_frag, b_frag, s_frag);
      }
      __syncthreads();

      // Update flash attention state
      float m_prev = m;

      // Get local max
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          int idx = i * 4 + j;
          int col = (idx_in_warp % 4) * 2 + i * 8 + j + warp_idx * 16;
          s_frag[idx] = (col < curr_iter_len) ? s_frag[idx] : -inf;
          m = max(s_frag[idx], m);
        }
      }

      // Get global max across threads
      m = max(m, shfl_xor_sync(m, 0x2));
      m = max(m, shfl_xor_sync(m, 0x1));

      // Update scaling factors
      float o_scale = expf(m_prev * sm_scale - m * sm_scale);
      float d_local = 0.f;
      d_sum *= o_scale;

      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          int idx = i * 4 + j;
          int col = (idx_in_warp % 4) * 2 + i * 8 + j + warp_idx * 16;
          int row = idx_in_warp / 4;
          s_frag[idx] = ((col < curr_iter_len) && (row < NUM_Q_HEADS))
                            ? expf(s_frag[idx] * sm_scale - m * sm_scale)
                            : 0;
          d_local += s_frag[idx];
        }
      }

      // Update output accumulator
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          o[n][i] *= o_scale;
        }
      }

      // Sum d across threads
      d_local += shfl_xor_sync(d_local, 0X1);
      d_local += shfl_xor_sync(d_local, 0X2);
      d_sum += d_local;

      // Compute attention output
      uint32_t o_frag[4];
      convert_f32_to_bf16_uint32(s_frag, o_frag);

      for (int n = 0; n < 8; n++) {
        int v_row = idx_in_warp % 16 + warp_idx * 16;
        int v_col = idx_in_warp / 16 * 8 + n * 16;
        bool is_valid_C = (v_row < curr_iter_len);
        T *src_ptr_C = is_valid_C ? v_cache_smem(v_row, v_col) : zero_buffer(0, 0);
        ldsm_t(src_ptr_C, v_frag);
        mma_m16n16k16_bf16bf16bf32(o[n], o_frag, v_frag, o[n]);
      }
      __syncthreads();

      if (kv_idx != num_iterations) {
        last_seq_len = curr_iter_len;
        cp_finished_seq_len += next_iter_len;
        curr_iter_len = next_iter_len;
      }
    }

    // Store intermediate results
    for (int n = 0; n < 8; n++) {
      for (int i = 0; i < 2; i++) {
        o_smem[threadIdx.x * 32 + n * 4 + i * 2] = o[n][i * 4];
        o_smem[threadIdx.x * 32 + n * 4 + i * 2 + 1] = o[n][i * 4 + 1];
      }
    }

    if (m != -inf) {
      m *= sm_scale;
    }
    d_smem[threadIdx.x] = d_sum;
    max_smem[threadIdx.x] = m;
    __syncthreads();

    // Final reduction across warps
    m = -inf;
    d_sum = 1.f;
    
    if (warp_idx == 0) {
      #pragma unroll
      for (uint32_t tidx = 0; tidx < 4; tidx++) {
        int shmem_idx = (idx_in_warp / 4) * 4 + tidx * 32 + (idx_in_warp % 4);
        float other_m = max_smem[shmem_idx];
        float other_d = d_smem[shmem_idx];
        float m_prev = m, d_prev = d_sum;
        m = max(m_prev, other_m);
        d_sum = d_prev * expf(m_prev - m) + other_d * expf(other_m - m);
        
        if (warp_idx == 0) {
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

    // Write normalized output
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

    // Update KV cache for current token
    #pragma unroll
    for (int i = threadIdx.x; i < (NUM_KV_HEADS * HEAD_DIM); i += NUM_THREADS) {
      int head = i / HEAD_DIM;
      int col = i % HEAD_DIM;
      k_cache_dmem.at(seq_len - 1 + token_idx, col) = k_dmem.at(head, col);
    }

    #pragma unroll
    for (int i = threadIdx.x; i < (NUM_KV_HEADS * HEAD_DIM); i += NUM_THREADS) {
      int head = i / HEAD_DIM;
      int col = i % HEAD_DIM;
      v_cache_dmem.at(seq_len - 1 + token_idx, col) = v_dmem.at(head, col);
    }

    // Write output to device memory
    #pragma unroll
    for (int i = threadIdx.x; i < (NUM_Q_HEADS * HEAD_DIM); i += NUM_THREADS) {
      int row = i / HEAD_DIM;
      int col = i % HEAD_DIM;
      output_dmem.at(row, col) = output_smem.at(row, col);
    }
    
    __syncthreads();
  }
}

} // namespace kernel