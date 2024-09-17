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

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;
using namespace mirage::config;

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
      uint32_t square_sum = 0;
      for (int k = 0; k < norm_size; k++) {
        int pos = idx * norm_size + k;
	uint32_t x = input_ptr[pos] % FP_PQ;
	x = (x * x) % FP_PQ;
	square_sum = (square_sum + x) % FP_PQ;
      }
      // Compute rooted mean square
      uint32_t rms = 0;
      {
        uint32_t x = square_sum;
        uint32_t n = norm_size;
        // Compute z = x / n;
        uint32_t z =
            (x % FP_P) * div_p_lookup_table[n % FP_P] * FP_Q_MUL_P_MOD_1 +
            (x % FP_Q) * div_q_lookup_table[n % FP_Q] * FP_P_MUL_Q_MOD_1;
        // Perform sqrt for root-mean-square
        rms = sqrt_p_lookup_table[z % FP_P] * FP_Q_MUL_P_MOD_1 +
              sqrt_q_lookup_table[z % FP_Q] * FP_P_MUL_Q_MOD_1;
      }
      for (int k = 0; k < norm_size; k++) {
        int pos = idx * norm_size + k;
	uint32_t x = input_ptr[pos] % FP_PQ;
	// Compute z = x / rms
	uint32_t z =
            (x % FP_P) * div_p_lookup_table[rms % FP_P] * FP_Q_MUL_P_MOD_1 +
            (x % FP_Q) * div_q_lookup_table[rms % FP_Q] * FP_P_MUL_Q_MOD_1;
        output_ptr[pos] = z % FP_PQ;
      }
    }
  }

};

} // namespace threadblock
} // namespace mirage
