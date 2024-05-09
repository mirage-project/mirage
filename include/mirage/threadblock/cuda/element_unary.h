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
class ElementUnaryExecutor {
public:
  CUTLASS_DEVICE
  ElementUnaryExecutor(mirage::type::TBOperatorType op_type,
                       ElementType *base_ptr,
                       int num_elements,
                       int thread_id,
                       int num_threads) {
    //assert(input.smem_offset == output.smem_offset);
    //int num_elements = output.num_elements();
    if (op_type == mirage::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[thread_id] = cutlass::fast_exp(base_ptr[thread_id]);
      }
    }
  }
};

class TBElementUnaryFingerPrinter {
public:
  CUTLASS_DEVICE
  TBElementUnaryFingerPrinter(mirage::type::TBOperatorType type,
                              FPType *exp_lookup_table,
                              FPType *base_ptr,
                              int num_elements,
                              int thread_id,
                              int num_threads) {
    // Assert inplace
    //assert(input.smem_offset == output.smem_offset);
    //FPType *ptr = (FPType *)(smem_buffer + input.smem_offset);
    //int num_elements = output.num_elements();
    if (type == mirage::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        FPType input = base_ptr[i];
        // FPType p_residual = input % FP_P;
        FPType q_residual = input % FP_Q;
        uint32_t result = exp_lookup_table[q_residual];
        result = (result * FP_Q_MUL_P_MOD_1) % FP_PQ;
        base_ptr[i] = result;
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

} // namespace threadblock
} // namespace mirage
