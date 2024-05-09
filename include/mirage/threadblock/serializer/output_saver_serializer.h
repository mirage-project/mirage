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
#include "mirage/layout.h"

namespace mirage {
namespace threadblock {

CUTLASS_HOST_DEVICE
void deserialize_output_saver_parameters(int const *params,
                                         int &param_idx,
                                         int3 &input_matrix_row_offset_block_stride,
                                         int3 &input_matrix_column_offset_block_stride,
                                         int3 &global_offset_block_stride,
                                         int2 &dtensor_matrix_shape,
                                         int2 &stensor_matrix_shape,
                                         mirage::layout::DmemLayout &dlayout,
                                         mirage::layout::SmemLayout &slayout,
                                         int &input_smem_offset,
                                         int &accum_smem_offset) {
  input_matrix_row_offset_block_stride.x = params[param_idx++];
  input_matrix_row_offset_block_stride.y = params[param_idx++];
  input_matrix_row_offset_block_stride.z = params[param_idx++];

  input_matrix_column_offset_block_stride.x = params[param_idx++];
  input_matrix_column_offset_block_stride.y = params[param_idx++];
  input_matrix_column_offset_block_stride.z = params[param_idx++];

  global_offset_block_stride.x = params[param_idx++];
  global_offset_block_stride.y = params[param_idx++];
  global_offset_block_stride.z = params[param_idx++];

  dtensor_matrix_shape.x = params[param_idx++];
  dtensor_matrix_shape.y = params[param_idx++];

  stensor_matrix_shape.x = params[param_idx++];
  stensor_matrix_shape.y = params[param_idx++];

  dlayout = (mirage::layout::DmemLayout) params[param_idx++];
  slayout = (mirage::layout::SmemLayout) params[param_idx++];
  input_smem_offset = params[param_idx++];
  accum_smem_offset = params[param_idx++];
}

inline
void serialize_output_saver_parameters(int *params,
                                       int &param_idx,
                                       int3 input_matrix_row_offset_block_stride,
                                       int3 input_matrix_column_offset_block_stride,
                                       int3 global_offset_block_stride,
                                       int2 dtensor_matrix_shape,
                                       int2 stensor_matrix_shape,
                                       mirage::layout::DmemLayout dlayout,
                                       mirage::layout::SmemLayout slayout,
                                       int input_smem_offset,
                                       int accum_smem_offset) {
  params[param_idx++] = input_matrix_row_offset_block_stride.x;
  params[param_idx++] = input_matrix_row_offset_block_stride.y;
  params[param_idx++] = input_matrix_row_offset_block_stride.z;

  params[param_idx++] = input_matrix_column_offset_block_stride.x;
  params[param_idx++] = input_matrix_column_offset_block_stride.y;
  params[param_idx++] = input_matrix_column_offset_block_stride.z;

  params[param_idx++] = global_offset_block_stride.x;
  params[param_idx++] = global_offset_block_stride.y;
  params[param_idx++] = global_offset_block_stride.z;

  params[param_idx++] = dtensor_matrix_shape.x;
  params[param_idx++] = dtensor_matrix_shape.y;

  params[param_idx++] = stensor_matrix_shape.x;
  params[param_idx++] = stensor_matrix_shape.y;

  params[param_idx++] = dlayout;
  params[param_idx++] = slayout;
  params[param_idx++] = input_smem_offset;
  params[param_idx++] = accum_smem_offset;
  assert(param_idx <= NewKernelParams::MAX_NUM_PARAMETERS);
}

} // namespace threadblock
} // namespace mirage
