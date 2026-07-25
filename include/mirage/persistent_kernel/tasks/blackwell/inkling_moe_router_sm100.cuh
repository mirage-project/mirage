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

// ================= Inkling MoE Router (InklingTopkRouter) ===================
//
// Reference math (thinkingmachines/Inkling, transformers modular_inkling.py):
//   logits = x @ W.T                    // [rows, 258] = 256 routed + 2 shared
//   scores_for_choice = sigmoid(logits[:, :256]) + e_score_correction_bias
//   topk_idx = top-6(scores_for_choice)             // NO group limiting
//   topk_logits = [logits[topk_idx], logits[256], logits[257]]
//   lp = logsigmoid(topk_logits)
//   w = exp(lp - logsumexp(lp)) * route_scale(8.0) * global_scale
//   -> 6 routed weights + 2 shared "gammas"
//
// MPK mapping: shared experts are folded into the expert weight tensor as
// experts 256 and 257 (gamma commutes with down_proj; weights are applied
// after down_proj in both routed and shared paths). This kernel therefore
// emits top-(K+S) selections over NUM_TOTAL experts so the standard
// moe_w13 / silu_mul / moe_w2 / mul_sum_add pipeline runs unchanged.
//
// Outputs (same ABI as topk_sigmoid_sm100):
//   weights_ptr  : float [num_rows, TOPK + N_SHARED]
//   indices_ptr  : int   [NUM_TOTAL, num_rows], k_idx+1 if selected, else 0
//   active_ptr   : int   [NUM_TOTAL + 1], compacted active ids; count at [end]
//
// The logits buffer is zeroed after reading (split-k gate linear reuse).
// Run as a single task: grid (1,1,1).

namespace kernel {

__device__ __forceinline__ float inkling_logsigmoid(float x) {
  // logsigmoid(x) = -softplus(-x)
  float nx = -x;
  float sp = (nx > 20.0f) ? nx : log1pf(expf(nx));
  return -sp;
}

template <typename T,
          int NUM_ROUTED,   // 256
          int N_SHARED,     // 2
          int TOPK,         // 6
          int LOGITS_STRIDE // row stride of logits buffer (>= NUM_TOTAL;
                            // may be padded, e.g. 384, for the gate linear)
          >
__device__ __forceinline__ void
    inkling_moe_router_task_impl(void *logits_ptr, // [num_rows, LOGITS_STRIDE]
                                 void const *bias_ptr,   // float [NUM_ROUTED]
                                 void const *gscale_ptr, // float [1]
                                 void *weights_ptr, // float [num_rows, K+S]
                                 void *indices_ptr, // int [NUM_TOTAL, num_rows]
                                 void *active_ptr,  // int [NUM_TOTAL + 1]
                                 int num_rows,
                                 float route_scale) {
  constexpr int NUM_TOTAL = NUM_ROUTED + N_SHARED;
  constexpr int K_OUT = TOPK + N_SHARED;
  static_assert(LOGITS_STRIDE >= NUM_TOTAL, "");

  T *logits = static_cast<T *>(logits_ptr);
  float const *bias = static_cast<float const *>(bias_ptr);
  float const gscale = *static_cast<float const *>(gscale_ptr);
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
    float l[NUM_TOTAL];
#pragma unroll 4
    for (int e = 0; e < NUM_TOTAL; ++e) {
      l[e] = float(lrow[e]);
      lrow[e] = T(0); // reset for split-k gate linear
    }

    // top-TOPK on sigmoid(l) + bias over routed experts
    float choice[NUM_ROUTED];
#pragma unroll 4
    for (int e = 0; e < NUM_ROUTED; ++e) {
      choice[e] = 1.0f / (1.0f + expf(-l[e])) + bias[e];
    }
    int sel[K_OUT];
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
      choice[best] = -1e30f;
    }
#pragma unroll
    for (int s = 0; s < N_SHARED; ++s) {
      sel[TOPK + s] = NUM_ROUTED + s;
    }

    // weights = softmax(logsigmoid(selected logits)) * route_scale * gscale
    float lp[K_OUT];
    float m = -1e30f;
#pragma unroll
    for (int k = 0; k < K_OUT; ++k) {
      lp[k] = inkling_logsigmoid(l[sel[k]]);
      m = fmaxf(m, lp[k]);
    }
    float denom = 0.0f;
#pragma unroll
    for (int k = 0; k < K_OUT; ++k) {
      lp[k] = expf(lp[k] - m);
      denom += lp[k];
    }
    float const scale = route_scale * gscale / denom;
#pragma unroll
    for (int k = 0; k < K_OUT; ++k) {
      weights[row * K_OUT + k] = lp[k] * scale;
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
