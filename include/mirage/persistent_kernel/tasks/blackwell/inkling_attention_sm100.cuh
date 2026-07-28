/* Copyright 2026 CMU
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

#include <cutlass/arch/barrier.h>

// ============ Inkling GQA decode attention (InklingAttention) ==============
//
// Single new token at position P = ctx_len. Reference math (eager path):
//   scaling   = 1 / HEAD_DIM                (q/k are per-head RMS-normed)
//   bias[h,d] = (r_h @ proj)[d], d = P - j clamped to [0, EXTENT), 0 outside
//   tau       = 1 + alpha * ln(max((P+1)/n_floor, 1))   (global layers only;
//               multiplies q AND bias, i.e. the whole pre-softmax score)
//   s_j       = tau * (dot(q_h, k_j) * scaling + bias[h, P-j])
//   sliding window (local layers): only keys with 0 <= P-j < SW attend.
//
// The per-(head,distance) bias table [NUM_Q_HEADS, EXTENT] is precomputed
// each step by a tiny GEMM (r [NQ,d_rel] @ proj [d_rel,EXTENT]) upstream.
//
// Inputs (per task; grid.x = G partitions kv heads, imap offsets pointers):
//   q      [1, NUM_Q_HEADS_t * D]  bf16 (per-head q_norm applied upstream)
//   ctx_k  [MAX_CTX, NUM_KV_HEADS_t * D] bf16 (k_norm+sconv applied on store)
//   ctx_v  [MAX_CTX, NUM_KV_HEADS_t * D] bf16
//   blk_k  [1, NUM_KV_HEADS_t * D] bf16 (this step's key, post k_norm)
//   blk_v  [1, NUM_KV_HEADS_t * D] bf16
//   bias   [NUM_Q_HEADS_t, EXTENT] bf16 (output of the bias GEMM)
//   step   [1] int32 = ctx_len (= P, number of cached tokens)
//   out    [1, NUM_Q_HEADS_t * D] bf16
//
// Strides are FULL row widths (the runtime offsets base pointers per task).
//
// Layout: 4 warps stripe the key range; each warp keeps an online softmax
// per q head with the V accumulator distributed over lanes (4 dims/lane).
// Cross-warp merge via shared memory at the end of each kv-head group.

namespace kernel {

template <typename T,
          int NUM_Q_HEADS,    // q heads for THIS task
          int NUM_KV_HEADS,   // kv heads for THIS task
          int HEAD_DIM,       // 128
          int EXTENT,         // rel_extent (512 local / 1024 global)
          int SLIDING_WINDOW, // 0 = global
          int Q_STRIDE,
          int KV_STRIDE,
          int O_STRIDE,
          int BIAS_STRIDE // full bias row width (== EXTENT)
          >
__device__ __forceinline__ void
    inkling_attention_task_impl(void const *q_ptr,
                                void const *ctx_k_ptr,
                                void const *ctx_v_ptr,
                                void const *blk_k_ptr,
                                void const *blk_v_ptr,
                                void const *bias_ptr,
                                void const *step_ptr,
                                void *output_ptr,
                                float log_scaling_alpha, // 0 = no log scaling
                                int log_scaling_n_floor) {
  constexpr int NUM_QO_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
  static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0, "");
  static_assert(HEAD_DIM % 128 == 0 || HEAD_DIM == 128, "");
  constexpr int ELTS_PER_LANE = HEAD_DIM / 32; // 4 for D=128
  constexpr int NUM_WARPS = 4;
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr float inv_scale = 1.0f / float(HEAD_DIM);

  constexpr int WORKER_SYNC_BARRIER_ID = 6;
  cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, WORKER_SYNC_BARRIER_ID);
  // Only NUM_THREADS worker threads participate (runtime may launch more;
  // the extras never touch smem nor arrive at the named barrier).
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  T const *__restrict__ q = static_cast<T const *>(q_ptr);
  T const *__restrict__ ctx_k = static_cast<T const *>(ctx_k_ptr);
  T const *__restrict__ ctx_v = static_cast<T const *>(ctx_v_ptr);
  T const *__restrict__ blk_k = static_cast<T const *>(blk_k_ptr);
  T const *__restrict__ blk_v = static_cast<T const *>(blk_v_ptr);
  T const *__restrict__ bias = static_cast<T const *>(bias_ptr);
  int const ctx_len = *static_cast<int const *>(step_ptr);
  T *__restrict__ out = static_cast<T *>(output_ptr);

  int const warp_idx = threadIdx.x >> 5;
  int const lane_idx = threadIdx.x & 31;

  int const P = ctx_len; // position of the new token
  float tau = 1.0f;
  if (log_scaling_alpha != 0.0f) {
    float ratio = float(P + 1) / float(log_scaling_n_floor);
    tau = 1.0f + log_scaling_alpha * logf(fmaxf(ratio, 1.0f));
  }

  // key range (causal, inclusive of self at j = P)
  int j_start = 0;
  if (SLIDING_WINDOW > 0) {
    j_start = P - SLIDING_WINDOW + 1;
    j_start = j_start < 0 ? 0 : j_start;
  }

  // smem for cross-warp merge: per warp, per head: m, l, acc[HEAD_DIM]
  // Must match the declaration in mla_prefill_sm100.cuh — extern __shared__
  // symbols with the same name must agree across the whole megakernel TU.
  extern __shared__ __align__(128) uint8_t smem_raw[];
  float *s_acc = reinterpret_cast<float *>(smem_raw);
  // [NUM_WARPS][NUM_QO_PER_KV][HEAD_DIM]
  float *s_m = s_acc + NUM_WARPS * NUM_QO_PER_KV * HEAD_DIM;
  float *s_l = s_m + NUM_WARPS * NUM_QO_PER_KV;

  for (int g = 0; g < NUM_KV_HEADS; g++) {
    int const kv_col = g * HEAD_DIM;

    // preload q fragments: lane holds elems [lane*E, lane*E+E) of each head
    float q_frag[NUM_QO_PER_KV][ELTS_PER_LANE];
#pragma unroll
    for (int h = 0; h < NUM_QO_PER_KV; h++) {
      int const q_col = (g * NUM_QO_PER_KV + h) * HEAD_DIM;
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        q_frag[h][e] = float(q[q_col + lane_idx * ELTS_PER_LANE + e]);
      }
    }

    float m_local[NUM_QO_PER_KV];
    float l_local[NUM_QO_PER_KV];
    float o_local[NUM_QO_PER_KV][ELTS_PER_LANE];
#pragma unroll
    for (int h = 0; h < NUM_QO_PER_KV; h++) {
      m_local[h] = -1e30f;
      l_local[h] = 0.0f;
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        o_local[h][e] = 0.0f;
      }
    }

    // stripe keys over warps: j = j_start + warp, step NUM_WARPS; j <= P
    for (int j = j_start + warp_idx; j <= P; j += NUM_WARPS) {
      T const *k_row;
      T const *v_row;
      if (j < ctx_len) {
        k_row = ctx_k + size_t(j) * KV_STRIDE + kv_col;
        v_row = ctx_v + size_t(j) * KV_STRIDE + kv_col;
      } else {
        k_row = blk_k + kv_col;
        v_row = blk_v + kv_col;
      }
      float k_frag[ELTS_PER_LANE];
      float v_frag[ELTS_PER_LANE];
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        k_frag[e] = float(k_row[lane_idx * ELTS_PER_LANE + e]);
        v_frag[e] = float(v_row[lane_idx * ELTS_PER_LANE + e]);
      }
      int const d = P - j; // in [0, P], < SW when windowed
#pragma unroll
      for (int h = 0; h < NUM_QO_PER_KV; h++) {
        float partial = 0.0f;
#pragma unroll
        for (int e = 0; e < ELTS_PER_LANE; e++) {
          partial += q_frag[h][e] * k_frag[e];
        }
        // warp allreduce
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          partial += __shfl_xor_sync(0xffffffff, partial, off);
        }
        float b = (d < EXTENT)
                      ? float(bias[(g * NUM_QO_PER_KV + h) * BIAS_STRIDE + d])
                      : 0.0f;
        float s = tau * (partial * inv_scale + b);
        // online softmax update
        float m_new = fmaxf(m_local[h], s);
        float resc = expf(m_local[h] - m_new);
        float p = expf(s - m_new);
        l_local[h] = l_local[h] * resc + p;
#pragma unroll
        for (int e = 0; e < ELTS_PER_LANE; e++) {
          o_local[h][e] = o_local[h][e] * resc + p * v_frag[e];
        }
        m_local[h] = m_new;
      }
    }

    // ---- cross-warp merge ----
#pragma unroll
    for (int h = 0; h < NUM_QO_PER_KV; h++) {
      if (lane_idx == 0) {
        s_m[warp_idx * NUM_QO_PER_KV + h] = m_local[h];
        s_l[warp_idx * NUM_QO_PER_KV + h] = l_local[h];
      }
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        s_acc[(warp_idx * NUM_QO_PER_KV + h) * HEAD_DIM +
              lane_idx * ELTS_PER_LANE + e] = o_local[h][e];
      }
    }
    wg_barrier.arrive_and_wait();

    // each warp finalizes NUM_QO_PER_KV/... let warp 0..3 split heads
    for (int h = warp_idx; h < NUM_QO_PER_KV; h += NUM_WARPS) {
      float m_star = -1e30f;
#pragma unroll
      for (int w = 0; w < NUM_WARPS; w++) {
        m_star = fmaxf(m_star, s_m[w * NUM_QO_PER_KV + h]);
      }
      float l_sum = 0.0f;
      float scale_w[NUM_WARPS];
#pragma unroll
      for (int w = 0; w < NUM_WARPS; w++) {
        scale_w[w] = expf(s_m[w * NUM_QO_PER_KV + h] - m_star);
        l_sum += scale_w[w] * s_l[w * NUM_QO_PER_KV + h];
      }
      float const inv_l = 1.0f / l_sum;
      int const o_col = (g * NUM_QO_PER_KV + h) * HEAD_DIM;
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
          acc += scale_w[w] * s_acc[(w * NUM_QO_PER_KV + h) * HEAD_DIM +
                                    lane_idx * ELTS_PER_LANE + e];
        }
        out[o_col + lane_idx * ELTS_PER_LANE + e] = T(acc * inv_l);
      }
    }
    wg_barrier.arrive_and_wait(); // reuse smem for next kv head
  }
}

} // namespace kernel
