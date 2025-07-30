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

using bfloat16 = type::bfloat16_t;

template <typename T,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int MAX_TOKENS>
__device__ __forceinline__ void multi_token_linear_kernel(
    void const *input_ptr,      // Points to this token's input slice
    void const *weight_ptr,     // Points to full weight matrix
    void const *residual_ptr,   // Points to this token's residual slice
    void *output_ptr,           // Points to this token's output slice
    int num_tokens,             // Runtime number of tokens
    bool residual = true) 
{
    // Simple and efficient implementation
    // Each thread computes one or more complete dot products
    
    // Cast pointers
    T const *input = static_cast<T const *>(input_ptr);
    T const *weight = static_cast<T const *>(weight_ptr);
    T const *residual_base = residual ? static_cast<T const *>(residual_ptr) : nullptr;
    T *output = static_cast<T *>(output_ptr);
    
    // Shared memory for input vector only
    extern __shared__ char smem[];
    T *shared_input = reinterpret_cast<T *>(smem);
    
    // Load entire input vector into shared memory
    for (int i = threadIdx.x; i < REDUCTION_SIZE; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();
    
    // Each thread computes multiple outputs
    // Use grid-stride loop for better load balancing
    for (int out_idx = threadIdx.x; out_idx < OUTPUT_SIZE; out_idx += blockDim.x) {
        float sum = 0.0f;
        
        // Compute dot product
        // Access weight matrix in column-major order for coalescing
        #pragma unroll 8
        for (int k = 0; k < REDUCTION_SIZE; k++) {
            sum += float(shared_input[k]) * float(weight[k * OUTPUT_SIZE + out_idx]);
        }
        
        // Add residual if enabled
        if (residual) {
            sum += float(residual_base[out_idx]);
        }
        
        // Store result
        output[out_idx] = T(sum);
    }
}

} // namespace kernel