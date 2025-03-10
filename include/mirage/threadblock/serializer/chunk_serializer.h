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

#include <vector_types.h>

namespace mirage {
namespace threadblock {

CUTLASS_HOST_DEVICE
void deserialize_chunk_op_parameters(int const *params, 
                                     int &param_idx, 
                                     int3 &input_shape, 
                                     int &chunk_size, 
                                     int &chunk_dim,
                                     int &input_smem_offset,
                                     int &output1_smem_offset,
                                     int &output2_smem_offset) {
    input_shape.x = params[param_idx++];
    input_shape.y = params[param_idx++];
    input_shape.z = params[param_idx++];

    chunk_size = params[param_idx++];
    chunk_dim = params[param_idx++];

    input_smem_offset = params[param_idx++];
    output1_smem_offset = params[param_idx++];
    output2_smem_offset = params[param_idx++];
}

inline void serialize_chunk_op_parameters(int *params,
                                          int &param_idx,
                                          int3 input_shape,
                                          int chunk_size,
                                          int chunk_dim,
                                          int input_smem_offset,
                                          int output1_smem_offset,
                                          int output2_smem_offset) {
    params[param_idx++] = input_shape.x;
    params[param_idx++] = input_shape.y;
    params[param_idx++] = input_shape.z;

    params[param_idx++] = chunk_size;
    params[param_idx++] = chunk_dim;

    params[param_idx++] = input_smem_offset;
    params[param_idx++] = output1_smem_offset;
    params[param_idx++] = output2_smem_offset;

    assert(param_idx <= NewKernelParams::MAX_NUM_PARAMETERS);
}

}
}