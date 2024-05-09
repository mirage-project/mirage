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

#include "mirage/kernel/device_tensor.h"
#include "mirage/utils/cuda_helper.h"

namespace mirage {
namespace kernel {

bool DTensor::has_same_fingerprint(DTensor const &ref) const {
  if (data_type != ref.data_type) {
    return false;
  }
  if (layout != ref.layout) {
    return false;
  }
  if (num_dims != ref.num_dims) {
    return false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dim[i] != ref.dim[i]) {
      return false;
    }
  }
  mirage::type::FPType *A = (mirage::type::FPType *)malloc(fingerprint_size());
  mirage::type::FPType *B = (mirage::type::FPType *)malloc(fingerprint_size());
  checkCUDA(cudaMemcpy(A, fp_ptr, fingerprint_size(), cudaMemcpyDeviceToHost));
  checkCUDA(
      cudaMemcpy(B, ref.fp_ptr, fingerprint_size(), cudaMemcpyDeviceToHost));
  int num_elements = (int)this->num_elements();
  for (int i = 0; i < num_elements; i++) {
    if (A[i] != B[i]) {
      free(A);
      free(B);
      return false;
    }
  }
  free(A);
  free(B);
  return true;
}

} // namespace kernel
} // namespace mirage
