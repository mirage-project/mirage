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

#ifdef MIRAGE_FINGERPRINT_USE_CUDA
DeviceMemoryManager::DeviceMemoryManager(int _num_gpus, int _gpu_id)
    : num_devices(_num_gpus), gpu_id(_gpu_id) {
  // fingerprint related fields
  checkCUDA(cudaSetDevice(gpu_id));
  printf("Mirage::DeviceMemoryManager: gpu_id(%d) num_gpus(%d)",
         gpu_id,
         num_devices);
  // Part 1: exponential lookup table
  // make future tensors 16 bytes aligned
  checkCUDA(
      cudaMalloc(&exp_lookup_table, (sizeof(FPType) * FP_Q + 15) / 16 * 16));
  // check PQ relations
  assert(FP_Q < FP_P);
  assert((FP_P - 1) % FP_Q == 0);
  FPType exp_table[FP_Q];
  exp_table[0] = 1;
  for (int i = 1; i < FP_Q; i++) {
    exp_table[i] = (exp_table[i - 1] * FP_EXP_BASE) % FP_P;
  }
  assert((exp_table[FP_Q - 1] * FP_EXP_BASE) % FP_P == 1);
  checkCUDA(cudaMemcpy(exp_lookup_table,
                       exp_table,
                       sizeof(FPType) * FP_Q,
                       cudaMemcpyHostToDevice));
  // Part 2: division p lookup table
  // make future tensors 16 bytes aligned
  checkCUDA(
      cudaMalloc(&div_p_lookup_table, (sizeof(FPType) * FP_P + 15) / 16 * 16));
  FPType div_p_table[FP_P];
  for (uint32_t i = 0; i < FP_P; i++) {
    div_p_table[i] = 1;
    for (uint32_t j = 1; j < FP_P; j++) {
      if ((i * j) % FP_P == 1) {
        div_p_table[i] = j;
      }
    }
    if (i > 1) {
      assert(div_p_table[i] != 1);
    }
  }
  checkCUDA(cudaMemcpy(div_p_lookup_table,
                       div_p_table,
                       sizeof(FPType) * FP_P,
                       cudaMemcpyHostToDevice));
  // Part 3: division q lookup table
  // make future tensors 16 bytes aligned
  checkCUDA(
      cudaMalloc(&div_q_lookup_table, (sizeof(FPType) * FP_Q + 15) / 16 * 16));
  FPType div_q_table[FP_Q];
  for (uint32_t i = 0; i < FP_Q; i++) {
    div_q_table[i] = 1;
    for (uint32_t j = 1; j < FP_Q; j++) {
      if ((i * j) % FP_Q == 1) {
        div_q_table[i] = j;
      }
    }
    if (i > 1) {
      assert(div_q_table[i] != 1);
    }
  }
  checkCUDA(cudaMemcpy(div_q_lookup_table,
                       div_q_table,
                       sizeof(FPType) * FP_Q,
                       cudaMemcpyHostToDevice));
  // Part 4: sqrt p lookup table
  // make future tensors 16 bytes aligned
  checkCUDA(
      cudaMalloc(&sqrt_p_lookup_table, (sizeof(FPType) * FP_P + 15) / 16 * 16));
  // Solving the congruence b=x^2 mod p using the following formulas:
  // if p == 3 mod 4, then x = b^{(p+1)/4} is a solution
  assert(FP_P % 4 == 3);
  FPType sqrt_p_table[FP_P];
  for (uint32_t i = 0; i < FP_P; i++) {
    sqrt_p_table[i] = 1;
    for (uint32_t j = 0; j < (FP_P + 1) / 4; j++) {
      sqrt_p_table[i] = (sqrt_p_table[i] * i) % FP_P;
    }
    // assert((sqrt_p_table[i] * sqrt_p_table[i]) % FP_P == i);
  }
  checkCUDA(cudaMemcpy(sqrt_p_lookup_table,
                       sqrt_p_table,
                       sizeof(FPType) * FP_P,
                       cudaMemcpyHostToDevice));
  // Part 5: sqrt q lookup table
  // make future tensors 16 bytes aligned
  checkCUDA(
      cudaMalloc(&sqrt_q_lookup_table, (sizeof(FPType) * FP_Q + 15) / 16 * 16));
  assert(FP_Q % 4 == 3);
  FPType sqrt_q_table[FP_Q];
  for (uint32_t i = 0; i < FP_Q; i++) {
    sqrt_q_table[i] = 1;
    for (uint32_t j = 0; j < (FP_Q + 1) / 4; j++) {
      sqrt_q_table[i] = (sqrt_q_table[i] * i) % FP_Q;
    }
    // assert((sqrt_q_table[i] * sqrt_q_table[i]) % FP_Q == i);
  }
  checkCUDA(cudaMemcpy(sqrt_q_lookup_table,
                       sqrt_q_table,
                       sizeof(FPType) * FP_Q,
                       cudaMemcpyHostToDevice));
  // data and fingerprints
  for (int i = 0; i < num_devices; i++) {
    // Note that we allocate all fingerprint buffers
    // on the 0-th GPU to avoid inter-GPU communication
    // for computing fingerprints
    // In addition, we allocate an extra space for storing
    // stensors' fingerprints in the device memory
    if (i == 0) {
      for (int k = 0; k < num_devices; k++) {
        checkCUDA(
            cudaMalloc(&fp_base_ptr[k], mirage::config::MAX_DMEM_FP_SIZE));
      }
      checkCUDA(
          cudaMalloc(&stensor_fp_base_ptr,
                     mirage::config::MAX_SMEM_FP_SIZE *
                         mirage::config::MAX_NUM_THREADBLOCKS_PER_KERNEL));
    }
  }
}

DeviceMemoryManager::~DeviceMemoryManager() {
  for (int i = 0; i < num_devices; i++) {
    if (i == 0) {
      checkCUDA(cudaFree(exp_lookup_table));
      checkCUDA(cudaFree(div_p_lookup_table));
      checkCUDA(cudaFree(div_q_lookup_table));
      checkCUDA(cudaFree(sqrt_p_lookup_table));
      checkCUDA(cudaFree(sqrt_q_lookup_table));
      for (int k = 0; k < num_devices; k++) {
        checkCUDA(cudaFree(fp_base_ptr[i]));
      }
      checkCUDA(cudaFree(stensor_fp_base_ptr));
    }
  }
}

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    int num_devices;
    checkCUDA(cudaGetDeviceCount(&num_devices));
    singleton = new DeviceMemoryManager(1 /*num_devices*/, 0 /*device_id*/);
  }
  return singleton;
}

/*static*/
void DeviceMemoryManager::set_gpu_device_id(int gpu_id) {
  // set_gpu_device_id must be called before creating DeviceMemoryManager
  assert(singleton == nullptr);
  int num_devices;
  checkCUDA(cudaGetDeviceCount(&num_devices));
  singleton = new DeviceMemoryManager(1 /*num_devices*/, gpu_id /*gpu_id*/);
}

void cython_set_gpu_device_id(int gpu_id) {
  DeviceMemoryManager::set_gpu_device_id(gpu_id);
}
#endif // MIRAGE_FINGERPRINT_USE_CUDA

} // namespace kernel
} // namespace mirage
