#pragma once
#include "tasks/common/common_header.cuh"
#include <cutlass/numeric_conversion.h>

// Hash-based MoE routing for DeepSeek V4 early layers.
// Expert indices come from a precomputed table tid2eid[input_ids[b]]
// Routing weights are still computed from
// sqrt(softplus(logits)) gathered at the hash-assigned positions.
//
// One block (256 threads) handles all num_rows tokens.
// threadIdx.x is the token index — no blockIdx usage.

namespace kernel {

__device__ __forceinline__ float _sqrtsoftplus(float x) {
  float sp = (x >= 0.0f) ? (x + log1pf(expf(-x))) : log1pf(expf(x));
  return sqrtf(sp);
}

// NUM_EXPERTS : total number of experts (e.g. 256)
// K           : num experts per token  (e.g. 6)
// ROUTE_SCALE_x1000 : route_scale * 1000 as int (e.g. 1500 for 1.5)
template <typename T, int NUM_EXPERTS, int K, int ROUTE_SCALE_x1000>
__device__ __forceinline__ void sqrtsoftplus_hash_routing_task_impl(
    void *__restrict__ logits_ptr,          // [num_rows, NUM_EXPERTS]  T (BF16)
    void *__restrict__ tid2eid_ptr,         // [vocab_size, K]          INT32
    void *__restrict__ input_ids_ptr,       // [num_rows]               INT64
    void *__restrict__ weights_ptr,         // [num_rows, K]            float
    void *__restrict__ routing_indices_ptr, // [NUM_EXPERTS, num_rows]  INT32
    void *__restrict__ active_experts_ptr,  // [NUM_EXPERTS + 1]        INT32
    int const num_rows) {

  constexpr float route_scale = ROUTE_SCALE_x1000 / 1000.0f;

  T       *logits          = static_cast<T *>(logits_ptr);
  int     *tid2eid         = static_cast<int *>(tid2eid_ptr);
  int64_t *input_ids       = static_cast<int64_t *>(input_ids_ptr);
  float   *weights         = static_cast<float *>(weights_ptr);
  int     *routing_indices = static_cast<int *>(routing_indices_ptr);
  int     *active_experts  = static_cast<int *>(active_experts_ptr);

  // Phase 1: zero routing_indices and reset active_experts marks.
  // Each thread owns one expert slice (NUM_EXPERTS == blockDim.x == 256).
  for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x) {
    for (int b = 0; b < num_rows; b++) {
      routing_indices[e * num_rows + b] = 0;
    }
    active_experts[e] = -1;
  }
  if (threadIdx.x == 0) {
    active_experts[NUM_EXPERTS] = 0; // counter for compact list
  }
  __syncthreads();

  // Phase 2: per-token hash lookup, weight gather, normalize.
  int b = threadIdx.x;
  if (b < num_rows) {
    int64_t word_id = input_ids[b];

    cutlass::NumericConverter<float, T> to_float;

    float w[K];
    float w_sum = 0.0f;

    for (int k = 0; k < K; k++) {
      int expert_id = tid2eid[word_id * K + k];
      float logit = to_float(logits[b * NUM_EXPERTS + expert_id]);
      w[k] = _sqrtsoftplus(logit);
      w_sum += w[k];
    }

    float inv_sum = (w_sum > 0.0f) ? (route_scale / w_sum) : 0.0f;
    for (int k = 0; k < K; k++) {
      weights[b * K + k] = w[k] * inv_sum;
    }

    // Fill routing structures: rank k+1 (1-based) for expert at position k.
    for (int k = 0; k < K; k++) {
      int expert_id = tid2eid[word_id * K + k];
      routing_indices[expert_id * num_rows + b] = k + 1;
      active_experts[expert_id] = expert_id; // idempotent mark
    }
  }
  __syncthreads();

  // Phase 3: compact active expert marks into a dense list at [0..count-1],
  // store count at [NUM_EXPERTS].  Same pattern as topk_softmax_task_impl.
  for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x) {
    if (active_experts[e] >= 0) {
      int pos = atomicAdd(active_experts + NUM_EXPERTS, 1);
      active_experts[pos] = e;
    }
  }
}

} // namespace kernel
