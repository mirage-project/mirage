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

// ================= GLM-4.x MoE Router (Glm4MoeTopkRouter) ===================
//
// Reference math (zai-org/GLM-4.6, transformers modeling_glm4_moe.py,
// n_group == 1 so group limiting is a no-op):
//   scores = sigmoid(logits)                         // [rows, NUM_ROUTED]
//   topk_idx = top-K(scores + e_score_correction_bias)
//   w = scores.gather(topk_idx)                      // UNBIASED scores
//   w = w / (sum(w) + 1e-20) * routed_scaling_factor // norm_topk_prob
//
// The shared expert (n_shared_experts == 1, same intermediate size as the
// routed experts) is folded into the expert weight tensor as expert
// NUM_ROUTED, always selected at slot TOPK with weight exactly 1.0 (the HF
// reference adds shared_experts(x) unweighted), so the standard
// moe_w13 / silu_mul / moe_w2 / mul_sum_add pipeline runs unchanged.
//
// The DeepSeek topk_sigmoid_sm100 kernel cannot be reused here: its
// vectorized layout statically requires a power-of-2 expert count, and GLM
// has 160 routed experts. Decode row counts are tiny, so this is the same
// scalar one-thread-per-row scheme as inkling_moe_router_sm100.
//
// Outputs (same ABI as topk_sigmoid_sm100 / inkling_moe_router):
//   weights_ptr  : float [num_rows, TOPK + N_SHARED]
//   indices_ptr  : int   [NUM_TOTAL, num_rows], k_idx+1 if selected, else 0
//   active_ptr   : int   [NUM_TOTAL + 1], compacted active ids; count at [end]
//
// The logits buffer is zeroed after reading (split-k gate linear reuse).
// Run as a single task: grid (1,1,1).

namespace kernel {

template <typename T,
          int NUM_ROUTED,    // 160
          int N_SHARED,      // 1
          int TOPK,          // 8
          int LOGITS_STRIDE  // row stride of logits buffer (>= NUM_ROUTED;
                             // may be padded for the gate linear)
          >
__device__ __forceinline__ void
    glm_moe_router_task_impl(void *logits_ptr, // [num_rows, LOGITS_STRIDE]
                             void const *bias_ptr, // float [NUM_ROUTED]
                             void *weights_ptr,    // float [num_rows, K+S]
                             void *indices_ptr,    // int [NUM_TOTAL, num_rows]
                             void *active_ptr,     // int [NUM_TOTAL + 1]
                             int num_rows,
                             float routed_scaling_factor) {
  constexpr int NUM_TOTAL = NUM_ROUTED + N_SHARED;
  constexpr int K_OUT = TOPK + N_SHARED;
  static_assert(LOGITS_STRIDE >= NUM_ROUTED, "");

  T *logits = static_cast<T *>(logits_ptr);
  float const *bias = static_cast<float const *>(bias_ptr);
  float *weights = static_cast<float *>(weights_ptr);
  int *indices = static_cast<int *>(indices_ptr);
  int *active = static_cast<int *>(active_ptr);

  // ---- Phase 0: clear routing structures ----
  for (int e = threadIdx.x; e < NUM_TOTAL; e += blockDim.x) {
    for (int r = 0; r < num_rows; ++r) {
      indices[e * num_rows + r] = 0;
    }
    active[e] = -1;
  }
  if (threadIdx.x == 0) {
    active[NUM_TOTAL] = 0;
  }
  __syncthreads();

  // ---- Phase 1: one thread per row (decode rows are few) ----
  for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
    T *lrow = logits + row * LOGITS_STRIDE;
    float sig[NUM_ROUTED];    // unbiased sigmoid scores (final weights)
    float choice[NUM_ROUTED]; // sigmoid + bias (selection only)
#pragma unroll 4
    for (int e = 0; e < NUM_ROUTED; ++e) {
      float logit = float(lrow[e]);
      lrow[e] = T(0); // reset for split-k gate linear
      sig[e] = 1.0f / (1.0f + expf(-logit));
      choice[e] = sig[e] + bias[e];
    }

    int sel[K_OUT];
    float w[K_OUT];
    float denom = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
      int best = -1;
      float best_v = -1e30f;
      for (int e = 0; e < NUM_ROUTED; ++e) {
        if (choice[e] > best_v) {
          best_v = choice[e];
          best = e;
        }
      }
      sel[k] = best;
      w[k] = sig[best];
      denom += sig[best];
      choice[best] = -1e30f;
    }
    float const scale = routed_scaling_factor / (denom + 1e-20f);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      w[k] *= scale;
    }
    // shared experts: always selected, weight exactly 1.0 (unscaled)
#pragma unroll
    for (int s = 0; s < N_SHARED; ++s) {
      sel[TOPK + s] = NUM_ROUTED + s;
      w[TOPK + s] = 1.0f;
    }

#pragma unroll
    for (int k = 0; k < K_OUT; ++k) {
      weights[row * K_OUT + k] = w[k];
      int const e = sel[k];
      indices[e * num_rows + row] = k + 1;
      active[e] = e; // mark (benign write race: all writers store e)
    }
  }
  __syncthreads();

  // ---- Phase 2: compact active expert ids ----
  // Snapshot marks into registers BEFORE any compaction write so a write to
  // active[pos] cannot clobber a mark that has not been read yet.
  constexpr int MAX_OWNED = (NUM_TOTAL + 127) / 128 + 1;
  int owned[MAX_OWNED];
  int n_owned = 0;
  for (int e = threadIdx.x; e < NUM_TOTAL; e += blockDim.x) {
    if (active[e] >= 0) {
      owned[n_owned++] = e;
    }
  }
  __syncthreads();
  for (int i = 0; i < n_owned; ++i) {
    int pos = atomicAdd(active + NUM_TOTAL, 1);
    active[pos] = owned[i];
  }
}

} // namespace kernel
