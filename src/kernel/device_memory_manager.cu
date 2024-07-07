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
#include "mirage/utils/cuda_helper.h"

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;

DeviceMemoryManager *DeviceMemoryManager::singleton = nullptr;

DeviceMemoryManager::DeviceMemoryManager(int _num_gpus) : num_gpus(_num_gpus) {
  off_t offset = 0;
  // fingerprint related fields
  // exponential lookup table
  exp_lookup_table_offset = offset;
  // make future tensors 16 bytes aligned
  offset += (sizeof(FPType) * FP_Q + 15) / 16 * 16;
  // check PQ relations
  assert(FP_Q < FP_P);
  assert((FP_P - 1) % FP_Q == 0);
  FPType exp_table[FP_Q];
  exp_table[0] = 1;
  for (int i = 1; i < FP_Q; i++) {
    exp_table[i] = (exp_table[i - 1] * FP_EXP_BASE) % FP_P;
  }
  assert((exp_table[FP_Q - 1] * FP_EXP_BASE) % FP_P == 1);
  // division p lookup table
  div_p_lookup_table_offset = offset;
  // make future tensors 16 bytes aligned
  offset += (sizeof(FPType) * FP_P + 15) / 16 * 16;
  FPType div_p_table[FP_P];
  for (int i = 0; i < FP_P; i++) {
    div_p_table[i] = 1;
    for (int j = 1; j < FP_P; j++) {
      if ((i * j) % FP_P == 1) {
        div_p_table[i] = j;
      }
    }
    if (i > 1) {
      assert(div_p_table[i] != 1);
    }
  }
  // division q lookup table
  div_q_lookup_table_offset = offset;
  // make future tensors 16 bytes aligned
  offset += (sizeof(FPType) * FP_Q + 15) / 16 * 16;
  FPType div_q_table[FP_Q];
  for (int i = 0; i < FP_Q; i++) {
    div_q_table[i] = 1;
    for (int j = 1; j < FP_Q; j++) {
      if ((i * j) % FP_Q == 1) {
        div_q_table[i] = j;
      }
    }
    if (i > 1) {
      assert(div_q_table[i] != 1);
    }
  }
  for (int i = 0; i < num_gpus; i++) {
    checkCUDA(cudaSetDevice(i));
    checkCUDA(cudaStreamCreate(&stream[i]));
    // Note that we allocate all fingerprint buffers
    // on the 0-th GPU to avoid inter-GPU communication
    // for computing fingerprints
    size_t allocated_size = mirage::config::MAX_DMEM_DATA_SIZE + offset;
    if (i == 0) {
      allocated_size += mirage::config::MAX_DMEM_FP_SIZE * num_gpus;
    }
    checkCUDA(cudaMalloc(&alloc_base_ptr[i], allocated_size));
    checkCUDA(cublasCreate(&blas[i]));
    checkCUDA(cublasSetMathMode(blas[i], CUBLAS_TENSOR_OP_MATH));
    // Copy exp_table, div_p_table, and div_q_table from DRAM to device memory
    cudaMemcpy(alloc_base_ptr[i] + exp_lookup_table_offset,
               exp_table,
               sizeof(FPType) * FP_Q,
               cudaMemcpyHostToDevice);
    cudaMemcpy(alloc_base_ptr[i] + div_p_lookup_table_offset,
               div_p_table,
               sizeof(FPType) * FP_P,
               cudaMemcpyHostToDevice);
    cudaMemcpy(alloc_base_ptr[i] + div_q_lookup_table_offset,
               div_q_table,
               sizeof(FPType) * FP_Q,
               cudaMemcpyHostToDevice);
    data_base_ptr[i] = alloc_base_ptr[i] + offset;
    if (i == 0) {
      for (int k = 0; k < num_gpus; k++) {
        fp_base_ptr[i] = data_base_ptr[i] + mirage::config::MAX_DMEM_DATA_SIZE +
                         mirage::config::MAX_DMEM_FP_SIZE * ((size_t)k);
      }
    }
  }
}

DeviceMemoryManager::~DeviceMemoryManager() {
  for (int i = 0; i < num_gpus; i++) {
    checkCUDA(cudaFree(alloc_base_ptr[i]));
    checkCUDA(cudaStreamDestroy(stream[i]));
    checkCUDA(cublasDestroy(blas[i]));
  }
}

#ifdef DEADCODE
bool DeviceMemoryManager::allocate(DTensor &tensor, bool allocate_fingerprint) {
  // assert that the start of the tensor is 16 bytes aligned
  assert(offset % 16 == 0);
  char *ret_ptr = base_ptr + offset;
  size_t tensor_size = tensor.data_size();
  // make tensor_size a multiplier of 16
  tensor_size = (tensor_size + 15) / 16 * 16;
  offset += tensor_size;
  tensor.data_offset = ret_ptr - base_ptr;
  allocated_tensors.push_back(std::make_pair(tensor.data_offset, tensor_size));

  if (allocate_fingerprint) {
    assert(offset % 16 == 0);
    ret_ptr = base_ptr + offset;
    size_t tensor_size = tensor.fingerprint_size();
    tensor_size = (tensor_size + 15) / 16 * 16;
    offset += tensor_size;
    tensor.fp_offset = ret_ptr - base_ptr;
    allocated_tensors.push_back(std::make_pair(tensor.fp_offset, tensor_size));
  }
  // Assert that we haven't used more than what we pre-allocated
  assert(offset <= total_size);

  return true;
}

bool DeviceMemoryManager::free(DTensor &tensor) {
  // Currently assume that tensors are freed in the reverse order
  // so ptr must be the last tensor we have created
  // Note that a non-negative fp_offset means that we have
  // allocated memory for its fingerprint
  if (tensor.fp_offset >= 0) {
    assert(allocated_tensors.size() > 0);
    assert(allocated_tensors.back().first == tensor.fp_offset);
    offset -= allocated_tensors.back().second;
    allocated_tensors.pop_back();
  }
  assert(allocated_tensors.size() > 0);
  assert(allocated_tensors.back().first == tensor.data_offset);
  offset -= allocated_tensors.back().second;
  allocated_tensors.pop_back();
  return true;
}
#endif

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    singleton = new DeviceMemoryManager(1 /*num_gpus*/);
  }
  return singleton;
}

} // namespace kernel
} // namespace mirage
