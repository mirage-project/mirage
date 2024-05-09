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

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;

template <typename ElementType>
class ElementBinaryExecutor {
public:
  CUTLASS_DEVICE
  ElementBinaryExecutor(mirage::type::TBOperatorType op_type,
                        ElementType *input1_ptr,
                        ElementType *input2_ptr,
                        ElementType *output_ptr,
                        int3 input1_shape,
                        int3 input2_shape,
                        int thread_id,
                        int num_threads) {
    // FIXME: currently we assume broadcast the inner-most dim
    //ElementType *input1_ptr = (ElementType *)(smem_buffer + input1.smem_offset);
    //ElementType *input2_ptr = (ElementType *)(smem_buffer + input2.smem_offset);
    //ElementType *output_ptr = (ElementType *)(smem_buffer + output.smem_offset);
    int3 output_shape = {
        max(input1_shape.x, input2_shape.x),
        max(input1_shape.y, input2_shape.y),
        max(input1_shape.z, input2_shape.z)};
    int output_num_elements = output_shape.x * output_shape.y * output_shape.z;
    int input1_num_elements = input1_shape.x * input1_shape.y * input1_shape.z;
    int input2_num_elements = input2_shape.x * input2_shape.y * input2_shape.z;
    int factor1 = output_num_elements / input1_num_elements;
    int factor2 = output_num_elements / input2_num_elements;
    if (op_type == mirage::type::TB_DIV_OP) {
      for (int i = 0; i < output_num_elements; i += num_threads) {
        output_ptr[i] = input1_ptr[i / factor1] / input2_ptr[i / factor2];
      }
    } else {
      assert(false && "Unsupported operator");
    }
  };
};

class TBElementBinaryFingerPrinter {
public:
  CUTLASS_DEVICE
  TBElementBinaryFingerPrinter(mirage::type::TBOperatorType type,
                               FPType *div_p_lookup_table,
                               FPType *div_q_lookup_table,
                               FPType *input1_ptr,
                               FPType *input2_ptr,
                               FPType *output_ptr,
                               int3 input1_shape,
                               int3 input2_shape,
                               int thread_id,
                               int num_threads) {
    //FPType *output_ptr = (FPType *)(smem_buffer + output.smem_offset);
    //FPType *input1_ptr = (FPType *)(smem_buffer + input1.smem_offset);
    //FPType *input2_ptr = (FPType *)(smem_buffer + input2.smem_offset);
    int input1_dims[3], input2_dims[3], output_dims[3];
    input1_dims[0] = input1_shape.x;
    input1_dims[1] = input1_shape.y;
    input1_dims[2] = input1_shape.z;
    input2_dims[0] = input2_shape.x;
    input2_dims[1] = input2_shape.y;
    input2_dims[2] = input2_shape.z;
    for (int i = 0; i < 3; i++)
      output_dims[i] = max(input1_dims[i], input2_dims[i]);
    int num_elements = output_dims[0] * output_dims[1] * output_dims[2];
    if (type == mirage::type::TB_DIV_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        int idx = i;
        int input1_stride = 1, input1_idx = 0;
        int input2_stride = 1, input2_idx = 0;
        for (int d = 2; d >= 0; d--) {
          input1_idx += (idx % input1_dims[d]) * input1_stride;
          input2_idx += (idx % input2_dims[d]) * input2_stride;
          input1_stride *= input1_dims[d];
          input2_stride *= input2_dims[d];
          idx /= output_dims[d];
        }
        uint32_t x = input1_ptr[input1_idx];
        uint32_t y = input2_ptr[input2_idx];
        uint32_t z =
            (x % FP_P) * div_p_lookup_table[y % FP_P] * FP_Q_MUL_P_MOD_1 +
            (x % FP_Q) * div_q_lookup_table[y % FP_Q] * FP_P_MUL_Q_MOD_1;
        output_ptr[i] = z % FP_PQ;
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

} // namespace threadblock
} // namespace mirage
