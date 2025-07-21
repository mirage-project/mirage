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
namespace kernel {

// Multi-token embedding kernel for EAGLE tree generation
// Input: array of token IDs [token0, token1, ..., tokenN]
// Output: concatenated embeddings [embed0 || embed1 || ... || embedN] with shape [1, num_tokens * OUT_DIM]
template <typename T, int OUT_DIM, int MAX_TOKENS>
__device__ __forceinline__ void
    multi_token_embedding_kernel(void const *__restrict__ token_ids_ptr,
                                void const *__restrict__ embedding_ptr,
                                void *__restrict__ output_ptr,
                                int num_tokens,
                                int embedding_stride) {
  
  // Effective maximum tokens is limited by both MAX_TOKENS and NUM_THREADS
  constexpr int EFFECTIVE_MAX_TOKENS = (MAX_TOKENS < NUM_THREADS) ? MAX_TOKENS : NUM_THREADS;
  
  int64_t const *__restrict__ token_ids = static_cast<int64_t const *>(token_ids_ptr);
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);
  
  // Single check against effective maximum
  if (num_tokens > EFFECTIVE_MAX_TOKENS) {
    if (threadIdx.x == 0) {
      printf("ERROR: multi_token_embedding_kernel: num_tokens (%d) exceeds EFFECTIVE_MAX_TOKENS (%d)\n", 
             num_tokens, EFFECTIVE_MAX_TOKENS);
    }
    return;
  }
  
  // Each thread handles one complete token embedding
  int tid = threadIdx.x;
  
  if (tid < num_tokens) {
    // Each thread processes its assigned token
    int64_t word_idx = token_ids[tid];
    int output_offset = tid * OUT_DIM;
    
    // Copy the entire embedding for this token
    for (int i = 0; i < OUT_DIM; i++) {
      output[output_offset + i] = embedding[word_idx * embedding_stride + i];
    }
  }
}

} // namespace kernel