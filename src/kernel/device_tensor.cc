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
