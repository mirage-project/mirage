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
#include "mirage/utils/hash_utils.h"
#include <functional>

namespace mirage {
namespace kernel {

DTensor::DTensor() {
  data_type = mirage::type::DT_UNKNOWN;
  layout = mirage::layout::DmemUnknownLayout;
  num_dims = 0;
  for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    // stride[i] = 0;
  }
  owner_op = nullptr;
  owner_ts_idx = -1000;
  data_ptr = nullptr;
}

/*
DTensorShape DTensor::get_shape() const {
  DTensorShape shape;
  shape.data_type = data_type;
  shape.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++) {
    shape.dim[i] = dim[i];
    shape.stride[i] = stride[i];
  }
  return shape;
}
*/

int DTensor::next_guid = 20000;

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
  hash_combine(ret, tensor.data_ptr);
  return ret;
}

} // namespace std
