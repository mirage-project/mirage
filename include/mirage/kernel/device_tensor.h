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

#include "mirage/layout.h"
#include "mirage/type.h"
#include "mirage/utils/json_utils.h"
#include <cstddef>
#include <functional>

namespace mirage {
namespace kernel {

#define MAX_TENSOR_DIMS 4

class KNOperator;

struct alignas(16) DTensor {
  DTensor(void);
  inline bool operator==(DTensor const &b) const {
    if (data_type != b.data_type) {
      return false;
    }
    if (layout != b.layout) {
      return false;
    }
    if (num_dims != b.num_dims) {
      return false;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return false;
      }
      // if (stride[i] != b.stride[i]) {
      //   return false;
      // }
    }
    if (owner_op != b.owner_op) {
      return false;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return false;
    }
    assert(data_ptr == b.data_ptr);
    return true;
  }
  inline bool operator!=(DTensor const &b) const {
    if (data_type != b.data_type) {
      return true;
    }
    if (layout != b.layout) {
      return true;
    }
    if (num_dims != b.num_dims) {
      return true;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return true;
      }
      // if (stride[i] != b.stride[i]) {
      //   return true;
      // }
    }
    if (owner_op != b.owner_op) {
      return true;
    }
    if (owner_ts_idx != b.owner_ts_idx) {
      return true;
    }
    assert(data_ptr == b.data_ptr);
    return false;
  }

  inline size_t num_elements() const {
    size_t num = 1;
    for (int i = 0; i < num_dims; i++) {
      num *= dim[i];
    }
    return num;
  }

  inline size_t data_size() const {
    using namespace mirage::type;
    size_t data_type_size = get_datatype_size(data_type);
    return num_elements() * data_type_size;
  }

  inline size_t fingerprint_size() const {
    using namespace mirage::type;
    size_t data_type_size = sizeof(FPType);
    return num_elements() * data_type_size;
  }

  bool has_same_fingerprint(DTensor const &ref) const;
  mirage::type::DataType data_type;
  mirage::layout::DmemLayout layout;
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  size_t guid;
  // int stride[MAX_TENSOR_DIMS];
  //  DTensor fields
  KNOperator *owner_op;
  int owner_ts_idx;
  // pointer to data
  void *data_ptr;
  // pointer to fingerprint
  mirage::type::FPType *fp_ptr;

  static int next_guid;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    DTensor, data_type, layout, num_dims, dim, guid)

} // namespace kernel
} // namespace mirage

namespace std {
template <>
struct hash<mirage::kernel::DTensor> {
  size_t operator()(mirage::kernel::DTensor const &) const;
};
}; // namespace std
