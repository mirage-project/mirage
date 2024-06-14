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

#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/cuda_helper.h"

namespace mirage {
namespace kernel {

mirage::cpu::CTensor DTensor::copy_fingerprint_to_ctensor() const {
  // Assume a single GPU for now
  assert(owner_op->kgraph->gpu_dim.x == 1);
  mirage::cpu::CTensor ctensor;
  ctensor.data_type = data_type;
  ctensor.layout = mirage::layout::dmemlayout_to_cmemlayout(layout);
  ctensor.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++) {
    ctensor.dim[i] = dim[i];
  }
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  // FIXME: memory leakage since we do not free this fingerprint buffer
  // This is ok for now since we only copy the input mugraph's fingerprint
  // to ctensor
  ctensor.fp_ptr = (mirage::type::FPType *)malloc(fingerprint_size());
  checkCUDA(cudaMemcpy(ctensor.fp_ptr,
                       dmm->base_ptr[0] + fp_offset,
                       fingerprint_size(),
                       cudaMemcpyDeviceToHost));
  return ctensor;
}

bool DTensor::has_same_fingerprint(mirage::cpu::CTensor const &ref) const {
  if (data_type != ref.data_type) {
    return false;
  }
  if (mirage::layout::dmemlayout_to_cmemlayout(layout) != ref.layout) {
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
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  mirage::type::FPType *A = (mirage::type::FPType *)malloc(fingerprint_size());
  checkCUDA(cudaMemcpy(A,
                       dmm->base_ptr[0] + fp_offset,
                       fingerprint_size(),
                       cudaMemcpyDeviceToHost));
  int num_elements = (int)this->num_elements();
  for (int i = 0; i < num_elements; i++) {
    if (A[i] != ref.fp_ptr[i]) {
      free(A);
      return false;
    }
  }
  free(A);
  return true;
}

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
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  mirage::type::FPType *A = (mirage::type::FPType *)malloc(fingerprint_size());
  mirage::type::FPType *B = (mirage::type::FPType *)malloc(fingerprint_size());
  checkCUDA(cudaMemcpy(A,
                       dmm->base_ptr[0] + fp_offset,
                       fingerprint_size(),
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(B,
                       dmm->base_ptr[0] + ref.fp_offset,
                       fingerprint_size(),
                       cudaMemcpyDeviceToHost));
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
