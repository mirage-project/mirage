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
  DeviceMemoryManager(int num_gpus);
  ~DeviceMemoryManager(void);
  // bool allocate(DTensor &tensor, bool allocate_fingerprint = true);
  // bool free(DTensor &tensor);

public:
  static DeviceMemoryManager *get_instance();

public:
  // off_t offset;
  // size_t per_gpu_memory_size;
  // std::vector<std::pair<int64_t, size_t>> allocated_tensors;
  // fingerprint related fields
  // mirage::type::FPType *exp_lookup_table;
  // mirage::type::FPType *div_p_lookup_table;
  // mirage::type::FPType *div_q_lookup_table;
  int num_gpus;
  off_t exp_lookup_table_offset;
  off_t div_p_lookup_table_offset;
  off_t div_q_lookup_table_offset;
  // fields for managing the preallocated cuda buffer
  // Note that both alloc_base_ptr[i] and data_base_ptr[i]
  // point to buffers on the i-th GPU,
  // while all fp_base_ptrs refer
  // to buffers on the 0-th GPU since we compute
  // fingerprint on a single device to avoid inter-GPU
  // communication
  char *alloc_base_ptr[mirage::config::MAX_NUM_GPUS];
  char *data_base_ptr[mirage::config::MAX_NUM_GPUS];
  char *fp_base_ptr[mirage::config::MAX_NUM_GPUS];

public:
  cudaStream_t stream[mirage::config::MAX_NUM_GPUS];
  cublasHandle_t blas[mirage::config::MAX_NUM_GPUS];
  // cudnnHandle_t cudnn;
};

} // namespace kernel
} // namespace mirage
