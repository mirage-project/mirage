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

#include "mirage/utils/cuda_helper.h"

namespace mirage {
namespace utils {

cudaDataType_t to_cuda_datatype(mirage::type::DataType type) {
  switch (type) {
    case mirage::type::DT_FLOAT16:
      return CUDA_R_16F;
    case mirage::type::DT_FLOAT32:
      return CUDA_R_32F;
    case mirage::type::DT_DOUBLE:
      return CUDA_R_64F;
    default:
      assert(false && "Unspoorted cuda data type");
  }
  return CUDA_R_16F;
}

size_t get_max_shared_mem() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProps;
  cudaGetDeviceProperties(&deviceProps, device);
  return deviceProps.sharedMemPerBlock;
}

} // namespace utils
} // namespace mirage
