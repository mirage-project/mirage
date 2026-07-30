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

#ifdef MIRAGE_FINGERPRINT_USE_CUDA

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/static_switch.h"

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;

class TBConcatFingerprinter {
public:
  CUTLASS_DEVICE
  TBConcatFingerprinter(FPType *A_ptr,
                        FPType *B_ptr,
                        FPType *output_ptr,
                        int output_num_elements,
                        int A_concat_dim_size,
                        int B_concat_dim_size,
                        int inner_size,
                        int thread_id,
                        int num_threads) {
    for (int i = thread_id; i < output_num_elements; i += num_threads) {
      int inner_idx = i % inner_size;
      int concat_dim_idx =
          (i / inner_size) % (A_concat_dim_size + B_concat_dim_size);
      int outer_idx =
          (i / inner_size) / (A_concat_dim_size + B_concat_dim_size);
      if (concat_dim_idx < A_concat_dim_size) {
        int A_idx = inner_idx + concat_dim_idx * inner_size +
                    outer_idx * inner_size * A_concat_dim_size;
        output_ptr[i] = A_ptr[A_idx];
      } else {
        int B_idx = inner_idx +
                    (concat_dim_idx - A_concat_dim_size) * inner_size +
                    outer_idx * inner_size * B_concat_dim_size;
        output_ptr[i] = B_ptr[B_idx];
      }
    }
  };
};

} // namespace threadblock
} // namespace mirage

#endif // MIRAGE_FINGERPRINT_USE_CUDA
