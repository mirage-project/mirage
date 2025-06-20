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

#include "mirage/threadblock/smem_tensor.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/static_switch.h"

namespace mirage {
namespace threadblock {

class TBMatmulFingerprinter {
public:
  CUTLASS_DEVICE
  TBMatmulFingerprinter(FPType *A_ptr,
                        FPType *B_ptr,
                        FPType *C_ptr,
                        int a_m_size,
                        int c_n_size,
                        int a_k_size,
                        int thread_id,
                        int num_threads) {
    // Note that we assume all tensors are in row-major layouts
    // when computing fingerprints
    // FPType *A_ptr = (FPType *)(smem_buffer + A.smem_offset);
    // FPType *B_ptr = (FPType *)(smem_buffer + B.smem_offset);
    // FPType *C_ptr = (FPType *)(smem_buffer + C.smem_offset);
    // int num_batches = 1;
    // for (int i = 0; i < C.num_dims - 2; i++) {
    //  num_batches *= C.dim[i];
    //}
    // Do not support batch matmul in TB
    // assert(num_batches == 1);
    int num_elements = a_m_size * c_n_size;
    // int c_n_size = C.dim[C.num_dims - 1];
    // int a_k_size = A.dim[A.num_dims - 1];
    int b_n_size = c_n_size;
    for (int i = thread_id; i < num_elements; i += num_threads) {
      FPType result = 0;
      int m = i / c_n_size;
      int n = i % c_n_size;
      for (int k = 0; k < a_k_size; k++) {
        FPType a = A_ptr[m * a_k_size + k];
        FPType b = B_ptr[k * b_n_size + n];
        FPType ab = compute_mul_fingerprint(a, b);
        result = compute_add_fingerprint(result, ab);
      }
      C_ptr[i] = result;
    } // for i
  }
};

} // namespace threadblock
} // namespace mirage

#endif // MIRAGE_FINGERPRINT_USE_CUDA
