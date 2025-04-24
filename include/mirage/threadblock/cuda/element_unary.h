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

#include <cmath>

namespace mirage {
__constant__ float CLAMP_MIN_MAX_DEVICE[2];
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;
using namespace mirage::utils;

template <typename ElementType>
class ElementUnaryExecutor {
public:
  CUTLASS_DEVICE
  ElementUnaryExecutor(mirage::type::TBOperatorType op_type,
                       ElementType *base_ptr,
                       int num_elements,
                       int thread_id,
                       int num_threads) {
    // assert(input.smem_offset == output.smem_offset);
    // int num_elements = output.num_elements();
    if (op_type == mirage::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = cutlass::fast_exp(base_ptr[i]);
      }
    } else if (op_type == mirage::type::TB_SQUARE_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = base_ptr[i] * base_ptr[i];
      }
    } else if (op_type == mirage::type::TB_SQRT_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = cutlass::fast_sqrt(base_ptr[i]);
      }
    } else if (op_type == mirage::type::TB_SILU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        ElementType x = base_ptr[i];
        base_ptr[i] = x / (1 + cutlass::fast_exp(-x));
      }
    } else if (op_type == mirage::type::TB_GELU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        ElementType x = base_ptr[i];
        // assuming floating point
        base_ptr[i] = (x / 2.0f) * (1.0f + erff(x / sqrtf(2.0f)));
      }
    } else if (op_type == mirage::type::TB_RELU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        ElementType x = base_ptr[i];
        if (x > 0.f) {
          base_ptr[i] = x;
        } else {
          base_ptr[i] = 0.f;
        }
      }
    } else if (op_type == mirage::type::TB_CLAMP_OP) {
      ElementType x = base_ptr[thread_id];
      if (x < CLAMP_MIN_MAX_DEVICE[0]) {
        base_ptr[thread_id] = CLAMP_MIN_MAX_DEVICE[0];
      } else if (x > CLAMP_MIN_MAX_DEVICE[1]) {
        base_ptr[thread_id] = CLAMP_MIN_MAX_DEVICE[1];
      } else {
        base_ptr[thread_id] = x;
      }
    }
  }
};

class TBElementUnaryFingerPrinter {
public:
  CUTLASS_DEVICE
  TBElementUnaryFingerPrinter(mirage::type::TBOperatorType type,
                              FPType *exp_lookup_table,
                              FPType *sqrt_p_lookup_table,
                              FPType *sqrt_q_lookup_table,
                              FPType *base_ptr,
                              int num_elements,
                              int thread_id,
                              int num_threads) {
    // Assert inplace
    // assert(input.smem_offset == output.smem_offset);
    // FPType *ptr = (FPType *)(smem_buffer + input.smem_offset);
    // int num_elements = output.num_elements();
    if (type == mirage::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_exp_fingerprint(base_ptr[i], exp_lookup_table);
      }
    } else if (type == mirage::type::TB_SQUARE_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_square_fingerprint(base_ptr[i]);
      }
    } else if (type == mirage::type::TB_SQRT_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_sqrt_fingerprint(
            base_ptr[i], sqrt_p_lookup_table, sqrt_q_lookup_table);
      }
    } else if (type == mirage::type::TB_SILU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_silu_fingerprint(base_ptr[i], exp_lookup_table);
      }
    } else if (type == mirage::type::TB_RELU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_relu_fingerprint(base_ptr[i]);
      }
    } else if (type == mirage::type::TB_CLAMP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_clamp_fingerprint(base_ptr[i]);
      }
    } else if (type == mirage::type::TB_GELU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_gelu_fingerprint(base_ptr[i], exp_lookup_table);
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

} // namespace threadblock
} // namespace mirage
