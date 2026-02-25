/* Copyright 2023-2024 CMU
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

#include "cutlass/cutlass.h"
#include <cuda_runtime.h>

namespace mirage {
namespace kernel {

template <typename T, int K>
struct TopKHeap {
  float scores[K];
  int   indices[K];

  __device__ void init() {
    #pragma unroll
    for (int i = 0; i < K; ++i) {
      scores[i]  = -1e20f;
      indices[i] = -1;
    }
  }

  __device__ int find_min_pos() {
    int min_pos = 0;
    #pragma unroll
    for (int i = 1; i < K; i++) {
      if (scores[i] < scores[min_pos]) {
        min_pos = i;
      }
    }
    return min_pos;
  }

  __device__ void insert(float score, int index) {
    int min_pos = find_min_pos();
    if (score > scores[min_pos]) {
      scores[min_pos]  = score;
      indices[min_pos] = index;
    }
  }

  __device__ void sort() {
    for (int i = 1; i < K; i++) {
      float key_score = scores[i];
      int   key_index = indices[i];
      int j = i - 1;
      while (j >= 0 && scores[j] < key_score) {
        scores[j + 1]  = scores[j];
        indices[j + 1] = indices[j];
        j--;
      }
      scores[j + 1]  = key_score;
      indices[j + 1] = key_index;
    }
  }
};

// __device__ version: operates on pre-offset pointers for a single token.
// Called from within the persistent kernel's _execute_task (one task per token).
template <typename T,
          int NUM_EXPERTS,
          int TOPK,
          int WORLD_SIZE,
          bool NORMALIZE = true>
__device__ void moe_routing_device_impl(
    T const *logits_row,          // [NUM_EXPERTS] - pre-offset to this token
    int     *routing_indices_row, // [TOPK] - pre-offset to this token
    T       *routing_weights_row, // [TOPK] - pre-offset to this token
    int     *dispatch_counts,     // [WORLD_SIZE] - shared, atomic update
    int     *expert_counts,       // [NUM_EXPERTS], may be nullptr
    int      experts_per_rank,
    int      rank,
    float    load_balance_factor) {

  const int tid = threadIdx.x;

  __shared__ float s_scores[TOPK];
  __shared__ int   s_indices[TOPK];
  __shared__ int   s_rank_counts[WORLD_SIZE];

  // Phase 1: TopK
  if (tid == 0) {
    TopKHeap<T, TOPK> heap;
    heap.init();
    for (int e = 0; e < NUM_EXPERTS; e++) {
      heap.insert(static_cast<float>(logits_row[e]), e);
    }
    heap.sort();
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      s_scores[k]  = heap.scores[k];
      s_indices[k] = heap.indices[k];
    }
  }
  __syncthreads();

  // Phase 2: Softmax
  if constexpr (NORMALIZE) {
    if (tid == 0) {
      float max_s = s_scores[0];
      #pragma unroll
      for (int k = 1; k < TOPK; k++) {
        if (s_scores[k] > max_s) max_s = s_scores[k];
      }
      float exp_vals[TOPK];
      float sum_exp = 0.0f;
      #pragma unroll
      for (int k = 0; k < TOPK; k++) {
        exp_vals[k] = __expf(s_scores[k] - max_s);
        sum_exp    += exp_vals[k];
      }
      float inv_sum = 1.0f / sum_exp;
      #pragma unroll
      for (int k = 0; k < TOPK; k++) {
        s_scores[k] = exp_vals[k] * inv_sum;
      }
    }
    __syncthreads();
  }

  // Phase 3: Dispatch counts
  if (tid < WORLD_SIZE) {
    s_rank_counts[tid] = 0;
  }
  __syncthreads();

  if (tid == 0) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = s_indices[k];
      int dest_rank = expert_id / experts_per_rank;
      s_rank_counts[dest_rank]++;
      if (expert_counts != nullptr) {
        atomicAdd(&expert_counts[expert_id], 1);
      }
    }
  }
  __syncthreads();

  if (tid < WORLD_SIZE && s_rank_counts[tid] > 0) {
    atomicAdd(&dispatch_counts[tid], s_rank_counts[tid]);
  }
  __syncthreads();

  // Phase 4: Write outputs to pre-offset row pointers
  if (tid < TOPK) {
    routing_indices_row[tid] = s_indices[tid];
    routing_weights_row[tid] = static_cast<T>(s_scores[tid]);
  }
}

// __global__ wrapper: one block per token, uses blockIdx.x as the token index.
// Used for standalone unit tests. Calls moe_routing_device_impl internally.
template <typename T,
          int BATCH_SIZE,
          int NUM_EXPERTS,
          int TOPK,
          int WORLD_SIZE,
          bool NORMALIZE = true>
__global__ void moe_routing_distributed_task_impl(
    T const *router_logits,   // [BATCH_SIZE, NUM_EXPERTS]
    int     *routing_indices, // [BATCH_SIZE, TOPK]
    T       *routing_weights, // [BATCH_SIZE, TOPK]
    int     *dispatch_counts, // [WORLD_SIZE]
    int     *expert_counts,   // [NUM_EXPERTS], may be nullptr
    int      experts_per_rank,
    int      rank,
    float    load_balance_factor) {

  const int token_idx = blockIdx.x;
  moe_routing_device_impl<T, NUM_EXPERTS, TOPK, WORLD_SIZE, NORMALIZE>(
      router_logits   + token_idx * NUM_EXPERTS,
      routing_indices + token_idx * TOPK,
      routing_weights + token_idx * TOPK,
      dispatch_counts,
      expert_counts,
      experts_per_rank,
      rank,
      load_balance_factor);
}

} // namespace kernel
} // namespace mirage
