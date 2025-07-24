/* Copyright 2025 CMU
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

#include "mirage/config.h"
#include "mirage/persistent_kernel/runtime_header.h"

namespace mirage {
namespace runtime {

struct IODesc {
  enum IOType {
    TorchTensor,
    FusedTorchTensor,
    CUDAMallocTensor,
    NVSHMEMMallocTensor
  };
  IODesc(IOType _type,
         std::string _name,
         mirage::kernel::DTensor const &_tensor,
         void *_torch_data_ptr = nullptr);
  IOType type;
  std::string name;
  TensorDesc tensor;
  // Only used for torch tensor
  void *torch_data_ptr;
  // Only used for fused tensors
  int num_groups;
  std::vector<IODesc> sub_descs;
};

struct TaskGraphResult {
  std::string cuda_code;
  std::string json_file;
};

class RuntimeTensorManager {
public:
  static RuntimeTensorManager *singleton;
  RuntimeTensorManager();
public:
  static RuntimeTensorManager *get_instance();
  void new_cuda_tensor(std::vector<int> const &dims,
                       std::vector<size_t> const &strides,
                       mirage::type::DataType data_type,
                       char const *name);
  void new_nvshmem_tensor(std::vector<int> const &dims,
                          std::vector<size_t> const &strides,
                          mirage::type::DataType data_type,
                          char const *name);
  std::unordered_map<std::string, size_t> cuda_tensors;
  std::unordered_map<std::string, size_t> nvshmem_tensors;
};

void cython_new_cuda_tensor(
    std::vector<int> const &dims,
    std::vector<size_t> const &strides,
    mirage::type::DataType data_type,
    char const *name);

void cython_new_nvshmem_tensor(
    std::vector<int> const &dims,
    std::vector<size_t> const &strides,
    mirage::type::DataType data_type,
    char const *name);

} // namespace runtime
} // namespace mirage
