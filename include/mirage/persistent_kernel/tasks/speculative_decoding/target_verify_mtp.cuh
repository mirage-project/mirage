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
// MTP Verification Kernels
//
// Three verification modes matching vLLM's rejection_sample_method:
// 1. strict:        Draft token must exactly match target argmax
// 2. probabilistic: P_target(token) > u * P_draft(token), u ~ U(0,1)
// 3. synthetic:     Position-dependent geometric decay acceptance
//
// All kernels process one batch element per thread block (single-threaded
// within the block for the sequential acceptance check).
// ============================================================================

// --- Strict Verification ---
// Compare draft token IDs vs target model's argmax output.
// Accept all tokens up to the first mismatch, plus the target's token at that
// position (bonus token).
//
// Inputs:
//   draft_token_ids:  [NUM_DRAFT_TOKENS]   - draft tokens (from MTP layer)
//   target_token_ids: [NUM_DRAFT_TOKENS+1]  - target model's argmax output
//                     (target[i] = argmax of logits when given tokens up to i)
// Outputs:
//   accepted_count:   [1] (int32)  - number of accepted tokens (0..NUM_DRAFT)
//   output_tokens:    [NUM_DRAFT_TOKENS+1]  - final accepted + bonus tokens
template <int NUM_DRAFT_TOKENS>
__device__ __forceinline__ void
    target_verify_strict_kernel(void const *__restrict__ draft_token_ids_ptr,
                                void const *__restrict__ target_token_ids_ptr,
                                void *__restrict__ accepted_count_ptr,
                                void *__restrict__ output_tokens_ptr) {

  long long const *__restrict__ draft_ids =
      static_cast<long long const *>(draft_token_ids_ptr);
  long long const *__restrict__ target_ids =
      static_cast<long long const *>(target_token_ids_ptr);
  int *__restrict__ accepted_count = static_cast<int *>(accepted_count_ptr);
  long long *__restrict__ output_tokens =
      static_cast<long long *>(output_tokens_ptr);

  int t_id = threadIdx.x;

  __shared__ int accepted_num_smem;

  if (t_id == 0) {
    int accepted = NUM_DRAFT_TOKENS;
    for (int i = 0; i < NUM_DRAFT_TOKENS; i++) {
      // draft_ids[i] is the draft token for position i
      // target_ids[i] is the target model's prediction for position i
      if (draft_ids[i] != target_ids[i]) {
        accepted = i;
        break;
      }
    }
    accepted_num_smem = accepted;
  }
  __syncthreads();

  int final_accepted = accepted_num_smem;

  // Copy accepted tokens + bonus token to output
  // output[0..final_accepted-1] = target_ids[0..final_accepted-1]
  // (these match draft_ids)
  // output[final_accepted] = target_ids[final_accepted] (bonus token)
  if (t_id < final_accepted + 1) {
    output_tokens[t_id] = target_ids[t_id];
  }

  if (t_id == 0) {
    // accepted_count = num accepted + 1 (for bonus token)
    accepted_count[0] = final_accepted + 1;
  }
}

// --- Probabilistic Verification ---
// Accept draft token at position i if:
//   P_target(draft_token_i) > u_i * P_draft(draft_token_i)
// where u_i ~ U(0,1) (from seeds).
// For greedy (temperature=0): compare draft vs target_token_ids directly.
//
// Follows vLLM's design: takes PRE-COMPUTED probabilities (not logits).
// Softmax is computed in a separate kernel before calling this.
// This keeps the verification kernel O(NUM_DRAFT_TOKENS) not O(VOCAB_SIZE).
//
// Inputs:
//   draft_token_ids:   [NUM_DRAFT_TOKENS]
//   target_token_ids:  [NUM_DRAFT_TOKENS+1] (argmax of target model, for greedy
//   path) target_probs:      [NUM_DRAFT_TOKENS] (fp32, P_target(draft_token) at
//   each pos) draft_probs:       [NUM_DRAFT_TOKENS] (fp32, P_draft(draft_token)
//   at each pos) seed:              [1] (uint64)
// Outputs:
//   accepted_count:    [1] (int32)
//   output_tokens:     [NUM_DRAFT_TOKENS+1] (int64)
template <int NUM_DRAFT_TOKENS>
__device__ __forceinline__ void target_verify_probabilistic_kernel(
    void const *__restrict__ draft_token_ids_ptr,
    void const *__restrict__ target_token_ids_ptr,
    void const *__restrict__ target_probs_ptr,
    void const *__restrict__ draft_probs_ptr,
    void const *__restrict__ seed_ptr,
    void *__restrict__ accepted_count_ptr,
    void *__restrict__ output_tokens_ptr) {

  long long const *__restrict__ draft_ids =
      static_cast<long long const *>(draft_token_ids_ptr);
  long long const *__restrict__ target_ids =
      static_cast<long long const *>(target_token_ids_ptr);
  float const *__restrict__ target_probs =
      static_cast<float const *>(target_probs_ptr);
  float const *__restrict__ draft_probs =
      static_cast<float const *>(draft_probs_ptr);
  unsigned long long const *__restrict__ seed =
      static_cast<unsigned long long const *>(seed_ptr);
  int *__restrict__ accepted_count = static_cast<int *>(accepted_count_ptr);
  long long *__restrict__ output_tokens =
      static_cast<long long *>(output_tokens_ptr);

  int t_id = threadIdx.x;

  if (t_id == 0) {
    unsigned long long rng_state = seed[0];
    int accepted = 0;
    bool still_accepting = true;

    for (int i = 0; i < NUM_DRAFT_TOKENS && still_accepting; i++) {
      float p_target = target_probs[i];
      float p_draft = draft_probs[i];

      if (p_draft == 0.0f) {
        // Draft probability is zero — greedy fallback: check if tokens match
        if (draft_ids[i] != target_ids[i]) {
          still_accepting = false;
        } else {
          output_tokens[i] = draft_ids[i];
          accepted++;
        }
      } else {
        // Probabilistic: P_target > u * P_draft
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(rng_state >> 33) /
                  static_cast<float>(1ULL << 31);

        if (p_target > u * p_draft) {
          output_tokens[i] = draft_ids[i];
          accepted++;
        } else {
          still_accepting = false;
        }
      }
    }

    // Bonus token = target model's token at the rejected position
    output_tokens[accepted] = target_ids[accepted];
    accepted_count[0] = accepted + 1;
  }
}

// --- Synthetic Verification ---
// Accept draft token at position i with probability:
//   accept_prob = base_rate * decay^i
// This provides a deterministic acceptance pattern calibrated to
// achieve a target mean acceptance rate.
//
// Inputs:
//   draft_token_ids:  [NUM_DRAFT_TOKENS]
//   target_token_ids: [NUM_DRAFT_TOKENS+1]
//   base_rate:        [1] (fp32) - base acceptance rate
//   decay:            [1] (fp32) - geometric decay factor
//   seed:             [1] (uint64)
// Outputs:
//   accepted_count:   [1] (int32)
//   output_tokens:    [NUM_DRAFT_TOKENS+1] (int64)
template <int NUM_DRAFT_TOKENS>
__device__ __forceinline__ void target_verify_synthetic_kernel(
    void const *__restrict__ draft_token_ids_ptr,
    void const *__restrict__ target_token_ids_ptr,
    void const *__restrict__ base_rate_ptr,
    void const *__restrict__ decay_ptr,
    void const *__restrict__ seed_ptr,
    void *__restrict__ accepted_count_ptr,
    void *__restrict__ output_tokens_ptr) {

  long long const *__restrict__ draft_ids =
      static_cast<long long const *>(draft_token_ids_ptr);
  long long const *__restrict__ target_ids =
      static_cast<long long const *>(target_token_ids_ptr);
  float const *__restrict__ base_rate =
      static_cast<float const *>(base_rate_ptr);
  float const *__restrict__ decay = static_cast<float const *>(decay_ptr);
  unsigned long long const *__restrict__ seed =
      static_cast<unsigned long long const *>(seed_ptr);
  int *__restrict__ accepted_count = static_cast<int *>(accepted_count_ptr);
  long long *__restrict__ output_tokens =
      static_cast<long long *>(output_tokens_ptr);

  int t_id = threadIdx.x;

  if (t_id == 0) {
    float br = base_rate[0];
    float dc = decay[0];
    unsigned long long rng_state = seed[0];
    int accepted = 0;

    float accept_prob = br;
    for (int i = 0; i < NUM_DRAFT_TOKENS; i++) {
      // RNG
      rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
      float u =
          static_cast<float>(rng_state >> 33) / static_cast<float>(1ULL << 31);

      if (u < accept_prob && draft_ids[i] == target_ids[i]) {
        output_tokens[i] = draft_ids[i];
        accepted++;
        accept_prob *= dc; // decay for next position
      } else {
        break;
      }
    }

    // Bonus token: target's prediction at the rejected position
    output_tokens[accepted] = target_ids[accepted];
    accepted_count[0] = accepted + 1;
  }
}

// --- Accept/Commit Kernel ---
// After verification, updates positions and prepares for the next decode round.
//
// Inputs:
//   accepted_count:     [1] (int32) - from verification kernel
//   output_tokens:      [NUM_DRAFT_TOKENS+1] (int64)
//   current_position:   [1] (int32) - current sequence position
// Outputs:
//   new_position:       [1] (int32) - updated position
//   final_output:       [NUM_DRAFT_TOKENS+1] (int64) - tokens to output
//   num_new_tokens:     [1] (int32) - number of new tokens accepted
template <int NUM_DRAFT_TOKENS>
__device__ __forceinline__ void
    mtp_accept_commit_kernel(void const *__restrict__ accepted_count_ptr,
                             void const *__restrict__ output_tokens_ptr,
                             void const *__restrict__ current_position_ptr,
                             void *__restrict__ new_position_ptr,
                             void *__restrict__ final_output_ptr,
                             void *__restrict__ num_new_tokens_ptr) {

  int const *__restrict__ accepted_count =
      static_cast<int const *>(accepted_count_ptr);
  long long const *__restrict__ output_tokens =
      static_cast<long long const *>(output_tokens_ptr);
  int const *__restrict__ current_position =
      static_cast<int const *>(current_position_ptr);
  int *__restrict__ new_position = static_cast<int *>(new_position_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);
  int *__restrict__ num_new_tokens = static_cast<int *>(num_new_tokens_ptr);

  int t_id = threadIdx.x;

  if (t_id == 0) {
    int count = accepted_count[0]; // includes bonus token
    new_position[0] = current_position[0] + count;
    num_new_tokens[0] = count;
  }

  // Copy accepted + bonus tokens to final output
  int count = accepted_count[0];
  if (t_id < count && t_id <= NUM_DRAFT_TOKENS) {
    final_output[t_id] = output_tokens[t_id];
  }
}

} // namespace kernel
