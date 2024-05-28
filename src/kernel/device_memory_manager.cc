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

DeviceMemoryManagerWrapper::DeviceMemoryManagerWrapper() : offset(0) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  total_size = dmm->total_size;
}
DeviceMemoryManagerWrapper::~DeviceMemoryManagerWrapper() {}

bool DeviceMemoryManagerWrapper::allocate(DTensor const &tensor,
                                          bool allocate_fingerprint) {
  assert(offset % 16 == 0);
  size_t data_size = tensor.data_size();
  data_size = (data_size + 15) / 16 * 16;
  if (offset + data_size > total_size) {
    return false;
  }
  offset += data_size;
  guid2data_size[tensor.guid] = data_size;

  if (allocate_fingerprint) {
    assert(offset % 16 == 0);
    size_t fingerprint_size = tensor.fingerprint_size();
    fingerprint_size = (fingerprint_size + 15) / 16 * 16;
    if (offset + fingerprint_size > total_size) {
      return false;
    }
    offset += fingerprint_size;
    guid2fp_size[tensor.guid] = fingerprint_size;
  }

  return true;
}

bool DeviceMemoryManagerWrapper::free(DTensor const &tensor) {
  assert(offset % 16 == 0);
  assert(contains_key(guid2data_size, tensor.guid));
  size_t data_size = guid2data_size.at(tensor.guid);
  assert(offset >= data_size);
  offset -= data_size;
  if (contains_key(guid2data_ptr, tensor.guid)) {
    DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
    dmm->free(guid2data_ptr.at(tensor.guid));
  }

  if (contains_key(guid2fp_size, tensor.guid)) {
    assert(offset % 16 == 0);
    size_t fingerprint_size = guid2fp_size.at(tensor.guid);
    assert(offset >= fingerprint_size);
    offset -= fingerprint_size;
    if (contains_key(guid2fp_ptr, tensor.guid)) {
      DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
      dmm->free(guid2fp_ptr.at(tensor.guid));
    }
  }

  return true;
}

bool DeviceMemoryManagerWrapper::free_physical_memory(DTensor const &tensor) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();

  if (contains_key(guid2fp_ptr, tensor.guid)) {
    dmm->free(guid2fp_ptr.at(tensor.guid));
  }

  if (contains_key(guid2data_ptr, tensor.guid)) {
    dmm->free(guid2data_ptr.at(tensor.guid));
  }

  return true;
}

void *DeviceMemoryManagerWrapper::get_data_ptr(size_t guid) {
  if (contains_key(guid2data_ptr, guid)) {
    return guid2data_ptr.at(guid);
  }

  assert(contains_key(guid2data_size, guid));
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  void *data_ptr = nullptr;
  dmm->allocate(guid2data_size.at(guid), data_ptr);
  guid2data_ptr[guid] = data_ptr;
  return data_ptr;
}

type::FPType *DeviceMemoryManagerWrapper::get_fp_ptr(size_t guid) {
  if (contains_key(guid2fp_ptr, guid)) {
    return guid2fp_ptr.at(guid);
  }

  assert(contains_key(guid2fp_ptr, guid));
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  void *fp_ptr = nullptr;
  dmm->allocate(guid2fp_size.at(guid), fp_ptr);
  guid2fp_ptr[guid] = (type::FPType *)fp_ptr;
  return (type::FPType *)fp_ptr;
}

} // namespace kernel
} // namespace mirage
