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
using namespace mirage::config;

class TBForloopAccumFingerprinter {
public:
  CUTLASS_DEVICE
  TBForloopAccumFingerprinter(mirage::type::FPType *input_ptr,
                              mirage::type::FPType *output_ptr,
                              int num_elements,
                              bool reset_output,
                              int thread_id,
                              int num_threads) {
    // mirage::type::FPType *input_ptr =
    //     (mirage::type::FPType *)(input.smem_offset + smem_buffer);
    // mirage::type::FPType *output_ptr =
    //     (mirage::type::FPType *)(output.smem_offset + smem_buffer);
    // int num_elements = (int)input.num_elements();
    // int num_elements = stensor_matrix_shape.x * stensor_matrix_shape.y;
    if (reset_output) {
      for (int idx = thread_id; idx < num_elements; idx += num_threads) {
        output_ptr[idx] = input_ptr[idx];
      }
      // if (thread_id == 0) {
      //  printf("Accumu(0): block(%d %d %d) output(%d) input(%d)\n",
      //         blockIdx.x,
      //         blockIdx.y,
      //         blockIdx.z,
      //         output_ptr[thread_id],
      //         input_ptr[thread_id]);
      //}
    } else {
      for (int idx = thread_id; idx < num_elements; idx += num_threads) {
        uint32_t value = input_ptr[idx];
        // if (thread_id == 0) {
        //  printf("Accumu(1): block(%d %d %d) output_old(%d) input(%d) "
        //         "output_new(%d)\n",
        //         blockIdx.x,
        //         blockIdx.y,
        //         blockIdx.z,
        //         output_ptr[thread_id],
        //         input_ptr[thread_id],
        //         (value + output_ptr[idx]) % FP_PQ);
        //}
        output_ptr[idx] = (value + output_ptr[idx]) % FP_PQ;
      }
    }
  }
};


} // namespace threadblock
} // namespace mirage

