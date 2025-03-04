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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/matmul.h"
#include "mirage/kernel/operator.h"
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
namespace mirage {
namespace kernel {

class DeviceMemoryManager {
public:
  static DeviceMemoryManager *singleton;
  DeviceMemoryManager(int device_id, int num_gpus);
  ~DeviceMemoryManager(void);

public:
  static DeviceMemoryManager *get_instance();
  static void set_gpu_device_id(int gpu_id);

public:
  int num_gpus, gpu_id;
  // fingerprint related fields
  mirage::type::FPType *exp_lookup_table;
  mirage::type::FPType *div_p_lookup_table;
  mirage::type::FPType *div_q_lookup_table;
  mirage::type::FPType *sqrt_p_lookup_table;
  mirage::type::FPType *sqrt_q_lookup_table;
  // fields for managing the preallocated cuda buffer
  // Note that data_base_ptr[i]
  // points to buffers on the i-th GPU,
  // while all fp_base_ptrs refer
  // to buffers on the 0-th GPU since we compute
  // fingerprint on a single device to avoid inter-GPU
  // communication
  char *data_base_ptr[mirage::config::MAX_NUM_GPUS];
  char *fp_base_ptr[mirage::config::MAX_NUM_GPUS];
  char *stensor_fp_base_ptr;

public:
  cudaStream_t stream[mirage::config::MAX_NUM_GPUS];
  cublasHandle_t blas[mirage::config::MAX_NUM_GPUS];
  // cudnnHandle_t cudnn;
};

void cython_set_gpu_device_id(int gpu_id);

} // namespace kernel
} // namespace mirage
