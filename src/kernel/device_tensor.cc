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
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/hash_utils.h"
#include <functional>

namespace mirage {
namespace kernel {

/*static*/ const DTensor DTensor::EMPTY_TENSOR = {/*zero-initialization*/};

DTensor::DTensor() {
  data_type = mirage::type::DT_UNKNOWN;
  layout = mirage::layout::DmemUnknownLayout;
  num_dims = 0;
  for (int i = 0; i < mirage::config::MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    // stride[i] = 0;
  }
  owner_op = nullptr;
  owner_ts_idx = -1000;
  data_offset = -1000;
  fp_offset = -1000;
}

size_t DTensor::get_owner_independent_hash() const {
  size_t ret = std::hash<int>()((data_type));
  hash_combine(ret, layout);
  hash_combine(ret, num_dims);
  for (int i = 0; i < num_dims; i++) {
    hash_combine(ret, dim[i]);
  }
  return ret;
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU
cpu::CTensor DTensor::copy_fingerprint_to_ctensor() const {
  cpu::CTensor ctensor;
  ctensor.data_type = data_type;
  ctensor.layout = layout::dmemlayout_to_cmemlayout(layout);
  ctensor.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++) {
    ctensor.dim[i] = dim[i];
  }
  kernel::DeviceMemoryManager *dmm =
      kernel::DeviceMemoryManager::get_instance();
  for (size_t device_id = 0; device_id < owner_op->kgraph->gpu_dim.x;
       ++device_id) {
    ctensor.fp_ptr[device_id] = new type::FPType[ctensor.num_elements()];
    size_t size_of_fp = ctensor.num_elements() * sizeof(type::FPType);
    memcpy(ctensor.fp_ptr[device_id],
           dmm->fp_base_ptr[device_id] + fp_offset,
           size_of_fp);
  }
  return ctensor;
}

bool DTensor::has_same_fingerprint(cpu::CTensor const &ref) const {
  if (data_type != ref.data_type) {
    return false;
  }
  if (layout::dmemlayout_to_cmemlayout(layout) != ref.layout) {
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

  size_t size_of_fp = num_elements() * sizeof(type::FPType);

  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();

  for (size_t device_id = 0; device_id < owner_op->kgraph->gpu_dim.x;
       ++device_id) {
    char *fp_ptr =
        reinterpret_cast<char *>(dmm->fp_base_ptr[device_id] + fp_offset);
    char *ref_fp_ptr = reinterpret_cast<char *>(ref.fp_ptr[device_id]);

    if (memcmp(fp_ptr, ref_fp_ptr, size_of_fp) != 0) {
      return false;
    }
  }

  return true;
}
#endif

std::atomic<int64_t> DTensor::next_guid = 10000000;

} // namespace kernel
} // namespace mirage

namespace std {

size_t hash<mirage::kernel::DTensor>::operator()(
    mirage::kernel::DTensor const &tensor) const {
  size_t ret = hash<int>()((tensor.data_type));
  hash_combine(ret, tensor.layout);
  hash_combine(ret, tensor.num_dims);
  for (int i = 0; i < tensor.num_dims; i++) {
    hash_combine(ret, tensor.dim[i]);
    // hash_combine(ret, tensor.stride[i]);
  }
  hash_combine(ret, tensor.owner_op);
  hash_combine(ret, tensor.owner_ts_idx);
  hash_combine(ret, tensor.data_offset);
  return ret;
}

} // namespace std
