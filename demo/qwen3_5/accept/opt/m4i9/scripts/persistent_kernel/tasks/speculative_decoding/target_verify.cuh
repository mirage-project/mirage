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
#include "../common.h"
namespace kernel {

// simply sequential greedy search
template <int NUM_SPEC_TOKENS>
__device__ __forceinline__ void target_verify_greedy_kernel(
    void const *__restrict__ spec_token_id_ptr,
    void const *__restrict__ target_token_id_ptr,
    void *__restrict__ new_accepted_len_ptr,
    void *__restrict__ tokens_ptr) { // Point to empty slots directly
  long long const *__restrict__ spec_token_id =
      static_cast<long long const *>(spec_token_id_ptr);
  long long const *__restrict__ target_token_id =
      static_cast<long long const *>(target_token_id_ptr);
  int *__restrict__ new_accepted_len = static_cast<int *>(new_accepted_len_ptr);
  long long *__restrict__ new_tokens = static_cast<long long *>(tokens_ptr);

  int t_id = threadIdx.x;
  __shared__ int local_accepted_num_smem;

  if (t_id == 0) {
    int accepted_count = NUM_SPEC_TOKENS;
    for (int i = 0; i < NUM_SPEC_TOKENS; i++) {
      // spec_token_id[0] is original token, [1...NUM_SPEC_TOKENS] are
      // speculative ones.
      if (spec_token_id[i + 1] != target_token_id[i]) {
        accepted_count = i;
        break;
      }
    }
    local_accepted_num_smem = accepted_count;
  }
  __syncthreads();

  int final_accepted_count = local_accepted_num_smem;

  // Copy the newly accepted tokens and the first non-matching target token.
  if (t_id < final_accepted_count + 1) {
    new_tokens[t_id] = target_token_id[t_id];
  }

  if (t_id == NUM_THREADS_PER_WARP) {
    new_accepted_len[0] = final_accepted_count + 1;
  }
}

} // namespace kernel
