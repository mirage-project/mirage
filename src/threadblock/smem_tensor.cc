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

#include "mirage/threadblock/smem_tensor.h"
#include "mirage/utils/hash_utils.h"
#include <functional>

namespace mirage {
namespace threadblock {
int STensor::next_guid = 10000;
} // namespace threadblock
} // namespace mirage

namespace std {

size_t hash<mirage::threadblock::STensor>::operator()(
    mirage::threadblock::STensor const &tensor) const {
  size_t ret = hash<int>()((tensor.data_type));
  hash_combine(ret, tensor.layout);
  hash_combine(ret, tensor.num_dims);
  for (int i = 0; i < tensor.num_dims; i++) {
    hash_combine(ret, tensor.dim[i]);
    // hash_combine(ret, tensor.stride[i]);
  }
  hash_combine(ret, tensor.owner_op);
  hash_combine(ret, tensor.owner_ts_idx);
  hash_combine(ret, tensor.smem_offset);
  return ret;
}

} // namespace std
