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

#include <cuda_runtime.h>

namespace mirage {
namespace kernel {

/**
 * =============================================================================
 * MOE TOKEN PADDING UTILITIES
 * =============================================================================
 *
 * Handles capacity planning and padding for expert parallelism.
 * Ensures each expert processes a fixed number of tokens for efficiency.
 *
 * CAPACITY PLANNING:
 * - Each expert has max capacity (e.g., 1.25 * avg_tokens_per_expert)
 * - Pad to nearest power-of-2 or multiple for efficient kernels
 * - Handle overflow when expert is over capacity
 *
 * =============================================================================
 */

/**
 * @brief Calculate expert capacity with padding
 *
 * @param num_tokens Total number of tokens
 * @param num_experts Number of experts
 * @param topk Number of experts per token
 * @param capacity_factor Overprovisioning factor (e.g., 1.25)
 * @return Capacity per expert
 */
__host__ __device__ inline int calculate_expert_capacity(
    int num_tokens,
    int num_experts,
    int topk,
    float capacity_factor = 1.25f) {

  // YOUR CODE HERE:
  //
  // 1. Compute average tokens per expert: avg = (num_tokens * topk) / num_experts
  // 2. Apply capacity factor: capacity = avg * capacity_factor
  // 3. Round up to next power of 2 or multiple of 32 for efficiency
  //
  // EXAMPLE:
  // - num_tokens = 1024, num_experts = 8, topk = 2
  // - avg = (1024 * 2) / 8 = 256
  // - capacity = 256 * 1.25 = 320
  // - rounded = 512 (next power of 2)

  int avg_tokens = (num_tokens * topk + num_experts - 1) / num_experts;
  int capacity = static_cast<int>(avg_tokens * capacity_factor);

  // Round up to power of 2
  // YOUR CODE HERE:

  return capacity;
}

/**
 * @brief Pad token buffer to meet capacity requirements
 *
 * TASK FOR YOU:
 * - Fill unused capacity slots with dummy/zero tokens
 * - Ensure expert kernels process uniform workload
 *
 * @tparam T Data type
 * @tparam HIDDEN_DIM Hidden dimension
 */
template <typename T, int HIDDEN_DIM>
__global__ void pad_expert_tokens_kernel(
    T *token_buffer,               // [num_experts, capacity, HIDDEN_DIM]
    int const *actual_counts,      // [num_experts] actual tokens per expert
    int capacity,                  // Capacity per expert
    int num_experts) {

  int expert_id = blockIdx.x;
  if (expert_id >= num_experts) return;

  int actual_count = actual_counts[expert_id];
  int tid = threadIdx.x;

  // YOUR CODE HERE:
  //
  // For tokens from actual_count to capacity:
  // - Set to zero or sentinel value
  // - Each thread handles multiple elements
  //
  // EXAMPLE:
  // - Expert 0 has 200 tokens, capacity = 256
  // - Pad tokens 200-255 with zeros


}

/**
 * @brief Remove padding from expert outputs
 *
 * TASK FOR YOU:
 * - After expert computation, remove dummy tokens
 * - Pack real outputs back into dense format
 *
 * @tparam T Data type
 * @tparam OUTPUT_DIM Output dimension
 */
template <typename T, int OUTPUT_DIM>
__global__ void unpad_expert_outputs_kernel(
    T const *padded_outputs,       // [num_experts, capacity, OUTPUT_DIM]
    T *dense_outputs,              // [total_real_tokens, OUTPUT_DIM]
    int const *actual_counts,      // [num_experts] real tokens per expert
    int const *output_offsets,     // [num_experts] where to write in dense output
    int capacity,
    int num_experts) {

  // YOUR CODE HERE:
  //
  // For each expert:
  // 1. Read actual_counts[expert_id] tokens
  // 2. Copy to dense_outputs starting at output_offsets[expert_id]
  // 3. Skip padding tokens


}

/**
 * @brief Handle capacity overflow with drop or rebalancing
 *
 * When an expert exceeds capacity:
 * - Option 1: Drop tokens (set routing weight to 0)
 * - Option 2: Route to next-best expert
 * - Option 3: Dynamic capacity expansion
 *
 * TASK FOR YOU:
 * - Implement overflow handling strategy
 */
template <typename T>
__device__ bool handle_capacity_overflow(
    int expert_id,
    int *expert_counts,            // Current counts
    int capacity,
    int token_idx,
    T *routing_weights,            // May be modified (set to 0 to drop)
    int *routing_indices) {        // May be modified (reroute)

  // YOUR CODE HERE:
  //
  // SIMPLE STRATEGY: Drop token if expert is full
  // ADVANCED: Try routing to next-best expert


  return true; // Token accepted
}

} // namespace kernel
} // namespace mirage
