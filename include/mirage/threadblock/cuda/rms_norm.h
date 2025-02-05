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

#include "mirage/utils/fingerprint_functions.h"

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;

class TBRmsNormFingerPrinter {
public:
  CUTLASS_DEVICE
  TBRmsNormFingerPrinter(FPType *input_ptr,
                         FPType *output_ptr,
                         FPType *div_p_lookup_table,
                         FPType *div_q_lookup_table,
                         FPType *sqrt_p_lookup_table,
                         FPType *sqrt_q_lookup_table,
                         int output_num_elements,
                         int norm_size,
                         int thread_id,
                         int num_threads) {
    int num_samples = output_num_elements / norm_size;
    assert(output_num_elements == num_samples * norm_size);
    for (int idx = thread_id; idx < num_samples; idx += num_threads) {
      FPType square_sum = 0;
      for (int k = 0; k < norm_size; k++) {
        int pos = idx * norm_size + k;
        FPType x = input_ptr[pos];
        x = compute_mul_fingerprint(x, x);
        square_sum = compute_add_fingerprint(square_sum, x);
      }
      // Compute rooted mean square
      FPType rms = 0;
      {
        FPType x = square_sum;
        FPType n = norm_size % FP_PQ;
        // Compute z = x / n;
        FPType z = compute_div_fingerprint(
            x, n, div_p_lookup_table, div_q_lookup_table);
        // Perform sqrt for root-mean-square
        rms = compute_sqrt_fingerprint(
            z, sqrt_p_lookup_table, sqrt_q_lookup_table);
      }
      for (int k = 0; k < norm_size; k++) {
        int pos = idx * norm_size + k;
        FPType x = input_ptr[pos];
        // Compute z = x / rms
        FPType z = compute_div_fingerprint(
            x, rms, div_p_lookup_table, div_q_lookup_table);
        output_ptr[pos] = z;
      }
    }
  }
};

} // namespace threadblock
} // namespace mirage
