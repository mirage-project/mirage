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
#include "cutlass/fast_math.h"
#include "mirage/utils/fingerprint_functions.h"

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;

template <typename ElementType>
class ChunkExecutor {
public:

    CUTLASS_DEVICE
    ChunkExecutor(
        ElementType *input_ptr,
        ElementType *output1_ptr,
        ElementType *output2_ptr,
        int3 input_shape,
        int chunk_size,
        int chunk_dim,
        int thread_id,
        int num_threads) {
        // determine the shape of the two output ptrs
        int3 output_shape = {chunk_dim == 0 ? input_shape.x / chunk_size : input_shape.x,
                             chunk_dim == 1 ? input_shape.y / chunk_size : input_shape.y,
                             chunk_dim == 2 ? input_shape.z / chunk_size : input_shape.z};
        int output_num_elements = input_shape.x * input_shape.y * input_shape.z;

        for (int i = 0; i < output_num_elements; i += num_threads) {
            int input_i = i / (input_shape.y * input_shape.z);
            int input_j = (i % (input_shape.y * input_shape.z)) / input_shape.z;
            int input_k = i % input_shape.z;
            if (chunk_dim == 0) {
                if (input_i < output_shape.x) {
                    output1_ptr[i] = input_ptr[i];
                } else { 
                    int i2 = ((input_i - output_shape.x) * (output_shape.y * output_shape.z)) + (input_j * output_shape.z) + input_k;
                    output2_ptr[i2] = input_ptr[i];
                }
            } else if (chunk_dim == 1) {
                if (input_j < output_shape.y) {
                    output1_ptr[i] = input_ptr[i];
                } else {
                    int i2 = (input_i * (output_shape.y * output_shape.z)) + ((input_j - output_shape.y) * output_shape.z) + input_k;
                    output2_ptr[i2] = input_ptr[i];
                }
            } else { // chunk_dim == 2
                if (input_k < output_shape.z) {
                    output1_ptr[i] = input_ptr[i];
                } else {
                    int i2 = (input_i * (output_shape.y * output_shape.z)) + (input_j * output_shape.z) + (input_k - output_shape.z);
                    output2_ptr[i2] = input_ptr[i];
                }
            }
        }
    };
};

class TBChunkFingerprinter {
public:
    CUTLASS_DEVICE
    TBChunkFingerprinter(FPType *input_ptr,
                         FPType *output1_ptr,
                         FPType *output2_ptr,
                         int3 input_shape,
                         int chunk_size,
                         int chunk_dim,
                         int thread_id,
                         int num_threads) {
                            int3 output_shape = {chunk_dim == 0 ? input_shape.x / chunk_size : input_shape.x,
                                chunk_dim == 1 ? input_shape.y / chunk_size : input_shape.y,
                                chunk_dim == 2 ? input_shape.z / chunk_size : input_shape.z};
        int output_num_elements = input_shape.x * input_shape.y * input_shape.z;
        
        FPType one = 1;
        for (int i = 0; i < output_num_elements; i += num_threads) {
            int input_i = i / (input_shape.y * input_shape.z);
            int input_j = (i % (input_shape.y * input_shape.z)) / input_shape.z;
            int input_k = i % input_shape.z;
            if (chunk_dim == 0) {
                if (input_i < output_shape.x) {
                    output1_ptr[i] = compute_mul_fingerprint(input_ptr[i], one);
                } else { 
                    int i2 = ((input_i - output_shape.x) * (output_shape.y * output_shape.z)) + (input_j * output_shape.z) + input_k;
                    output2_ptr[i2] = compute_mul_fingerprint(input_ptr[i], one);
                }
            } else if (chunk_dim == 1) {
                if (input_j < output_shape.y) {
                    output1_ptr[i] = compute_mul_fingerprint(input_ptr[i], one);
                } else {
                    int i2 = (input_i * (output_shape.y * output_shape.z)) + ((input_j - output_shape.y) * output_shape.z) + input_k;
                    output2_ptr[i2] = compute_mul_fingerprint(input_ptr[i], one);
                }
            } else { // chunk_dim == 2
                if (input_k < output_shape.z) {
                    output1_ptr[i] = compute_mul_fingerprint(input_ptr[i], one);
                } else {
                    int i2 = (input_i * (output_shape.y * output_shape.z)) + (input_j * output_shape.z) + (input_k - output_shape.z);
                    output2_ptr[i2] = compute_mul_fingerprint(input_ptr[i], one);
                }
            }
        }
    }
};

}
}