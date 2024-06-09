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
#include "mirage/utils/containers.h"

namespace mirage {
namespace kernel {

DeviceMemoryOffsetManager::DeviceMemoryOffsetManager() : offset(0) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  total_size = dmm->total_size;
}

bool DeviceMemoryOffsetManager::allocate(DTensor &tensor,
                                          bool allocate_fingerprint) {
  assert(offset % 16 == 0);
  size_t data_size = tensor.data_size();
  data_size = (data_size + 15) / 16 * 16;
  if (offset + data_size > total_size) {
    return false;
  }
  tensor.data_offset = offset;
  offset += data_size;
  allocated_tensors.push_back({tensor.data_offset, data_size});

  if (allocate_fingerprint) {
    assert(offset % 16 == 0);
    size_t fingerprint_size = tensor.fingerprint_size();
    fingerprint_size = (fingerprint_size + 15) / 16 * 16;
    if (offset + fingerprint_size > total_size) {
      return false;
    }
    tensor.fp_offset = offset;
    offset += fingerprint_size;
    allocated_tensors.push_back({tensor.fp_offset, fingerprint_size});
  }

  return true;
}

bool DeviceMemoryOffsetManager::free(DTensor &tensor) {
  if (tensor.fp_offset != -1) {
    assert(allocated_tensors.size() > 0);
    assert(allocated_tensors.back().first == tensor.fp_offset);
    offset -= allocated_tensors.back().second;
    allocated_tensors.pop_back();
    tensor.fp_offset = -1;
  }

  assert(tensor.data_offset != -1);
  assert(allocated_tensors.size() > 0);
  assert(allocated_tensors.back().first == tensor.data_offset);
  offset -= allocated_tensors.back().second;
  allocated_tensors.pop_back();
  tensor.data_offset = -1;

  return true;
}

} // namespace kernel
} // namespace mirage
