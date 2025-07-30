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
    // Optimized implementation with tiling and better memory access patterns
    constexpr int TILE_K = 128;  // Tile size for reduction dimension
    constexpr int TILE_N = 64;   // Tile size for output dimension
    constexpr int THREADS_PER_BLOCK = 256;  // Typical block size
    
    // Cast pointers
    T const *input = static_cast<T const *>(input_ptr);
    T const *weight = static_cast<T const *>(weight_ptr);
    T const *residual_base = residual ? static_cast<T const *>(residual_ptr) : nullptr;
    T *output = static_cast<T *>(output_ptr);
    
    // Shared memory layout
    extern __shared__ char smem[];
    T *shared_input = reinterpret_cast<T *>(smem);
    T *shared_weight = shared_input + TILE_K;  // Weight tile after input
    
    // Thread configuration
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    // Process output in tiles
    for (int output_tile_start = 0; output_tile_start < OUTPUT_SIZE; output_tile_start += TILE_N) {
        const int output_tile_size = min(TILE_N, OUTPUT_SIZE - output_tile_start);
        
        // Accumulator for this thread's outputs
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Each thread handles up to 4 outputs
        const int outputs_per_thread = (output_tile_size + blockDim.x - 1) / blockDim.x;
        
        // Process reduction dimension in tiles
        for (int k_tile_start = 0; k_tile_start < REDUCTION_SIZE; k_tile_start += TILE_K) {
            const int k_tile_size = min(TILE_K, REDUCTION_SIZE - k_tile_start);
            
            // Cooperative load of input tile into shared memory
            __syncthreads();
            #pragma unroll 4
            for (int i = tid; i < k_tile_size; i += blockDim.x) {
                shared_input[i] = input[k_tile_start + i];
            }
            
            // Cooperative load of weight tile into shared memory
            // Each thread loads one or more elements
            const int weight_elements = k_tile_size * output_tile_size;
            #pragma unroll 2
            for (int i = tid; i < weight_elements; i += blockDim.x) {
                int k_idx = i / output_tile_size;
                int n_idx = i % output_tile_size;
                shared_weight[i] = weight[(k_tile_start + k_idx) * OUTPUT_SIZE + 
                                         output_tile_start + n_idx];
            }
            __syncthreads();
            
            // Compute matrix multiplication for this tile
            #pragma unroll 4
            for (int out_idx = 0; out_idx < outputs_per_thread && 
                 tid * outputs_per_thread + out_idx < output_tile_size; out_idx++) {
                const int n_idx = tid * outputs_per_thread + out_idx;
                
                // Use registers to accumulate partial sums
                float partial_sum = 0.0f;
                
                // Unroll the reduction loop for better ILP
                #pragma unroll 8
                for (int k = 0; k < k_tile_size; k++) {
                    partial_sum += float(shared_input[k]) * 
                                  float(shared_weight[k * output_tile_size + n_idx]);
                }
                
                acc[out_idx] += partial_sum;
            }
        }
        
        // Write results for this output tile
        __syncthreads();
        #pragma unroll 4
        for (int out_idx = 0; out_idx < outputs_per_thread && 
             tid * outputs_per_thread + out_idx < output_tile_size; out_idx++) {
            const int global_out_idx = output_tile_start + tid * outputs_per_thread + out_idx;
            if (global_out_idx < OUTPUT_SIZE) {
                float result = acc[out_idx];
                
                // Add residual if enabled
                if (residual) {
                    result += float(residual_base[global_out_idx]);
                }
                
                // Store result
                output[global_out_idx] = T(result);
            }
        }
    }
}

} // namespace kernel