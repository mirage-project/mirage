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
#include "tasks/common/common_header.cuh"

namespace kernel {

// ============================================================================
// DFlash non-causal block attention (correctness-first).
//
// One task = one request. The B block-query tokens attend NON-CAUSALLY to
// [context (ctx_len) ++ block (B)] keys. Optional sliding window limits each
// query to keys within `sliding_window` of its absolute position.
//
// Inputs are already projected / normed / roped:
//   q : [B, NUM_Q_HEADS, HEAD_DIM]   (block queries; q_norm + RoPE applied)
//   k : [T, NUM_KV_HEADS, HEAD_DIM]  (T = ctx_len + B; k_norm + RoPE applied;
//                                     context rows first, then this block)
//   v : [T, NUM_KV_HEADS, HEAD_DIM]  (raw v_proj output)
//   out: [B, NUM_Q_HEADS, HEAD_DIM]
//
// Absolute positions: context key j -> j (0..ctx_len-1); block key/query i ->
// ctx_len + i. So key j position == j, query i position == ctx_len + i.
//
// Layout: one warp computes one (query, q_head) pair; each of the 32 lanes owns
// HEAD_DIM/32 dims. Online (flash) softmax over all T keys, read from global.
// GQA: kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS).
// ============================================================================
template <typename T, int NUM_Q_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int B>
__device__ __forceinline__ void dflash_attention_sm100(void const *q_ptr,
                                                       void const *k_ptr,
                                                       void const *v_ptr,
                                                       void *output_ptr,
                                                       int total_kv,
                                                       int sliding_window) {
  static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
  constexpr int DPT = HEAD_DIM / 32; // dims per lane
  constexpr int NUM_QO_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
  constexpr int WARPS = NUM_THREADS / 32;

  // Only the NUM_THREADS consumer threads do work (runtime launches more).
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  T const *q = static_cast<T const *>(q_ptr);
  T const *k = static_cast<T const *>(k_ptr);
  T const *v = static_cast<T const *>(v_ptr);
  T *out = static_cast<T *>(output_ptr);

  int const T_kv = total_kv; // ctx_len + B
  int const ctx_len = T_kv - B;
  float const scale = rsqrtf(static_cast<float>(HEAD_DIM));

  int const warp = threadIdx.x / 32;
  int const lane = threadIdx.x % 32;
  int const total_pairs = B * NUM_Q_HEADS;

  // each warp grabs (query,head) pairs in a strided loop
  for (int pair = warp; pair < total_pairs; pair += WARPS) {
    int const qi = pair / NUM_Q_HEADS; // query row 0..B-1
    int const h = pair % NUM_Q_HEADS;  // q head 0..NUM_Q_HEADS-1
    int const kvh = h / NUM_QO_PER_KV; // kv head
    int const q_pos = ctx_len + qi;

    // this lane's dims of q for (qi,h)
    float q_reg[DPT];
    T const *q_row = q + ((qi * NUM_Q_HEADS) + h) * HEAD_DIM;
#pragma unroll
    for (int e = 0; e < DPT; ++e) {
      q_reg[e] = static_cast<float>(q_row[lane * DPT + e]);
    }

    float m_i = -inf; // running max (repo constant, worker_config.h)
    float l_i = 0.0f; // running denom
    float acc[DPT];
#pragma unroll
    for (int e = 0; e < DPT; ++e) {
      acc[e] = 0.0f;
    }

    for (int j = 0; j < T_kv; ++j) {
      // sliding window mask (non-causal): |q_pos - key_pos| >= window -> skip
      if (sliding_window > 0) {
        int d = q_pos - j;
        d = d < 0 ? -d : d;
        if (d >= sliding_window) {
          continue;
        }
      }
      T const *k_row = k + ((j * NUM_KV_HEADS) + kvh) * HEAD_DIM;
      float partial = 0.0f;
#pragma unroll
      for (int e = 0; e < DPT; ++e) {
        partial += q_reg[e] * static_cast<float>(k_row[lane * DPT + e]);
      }
      // warp reduce -> full dot product on all lanes
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, off);
      }
      float score = partial * scale;

      // online softmax update
      float m_new = fmaxf(m_i, score);
      float corr = __expf(m_i - m_new);
      float p = __expf(score - m_new);
      l_i = l_i * corr + p;
      T const *v_row = v + ((j * NUM_KV_HEADS) + kvh) * HEAD_DIM;
#pragma unroll
      for (int e = 0; e < DPT; ++e) {
        acc[e] = acc[e] * corr + p * static_cast<float>(v_row[lane * DPT + e]);
      }
      m_i = m_new;
    }

    float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    T *o_row = out + ((qi * NUM_Q_HEADS) + h) * HEAD_DIM;
#pragma unroll
    for (int e = 0; e < DPT; ++e) {
      o_row[lane * DPT + e] = static_cast<T>(acc[e] * inv);
    }
  }
}

} // namespace kernel
