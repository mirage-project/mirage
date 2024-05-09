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
#include "cutlass/cutlass.h"
#include <cstddef>
#include <functional>

namespace mirage {
namespace threadblock {

#define MAX_TENSOR_DIMS 4

class TBOperator;

struct alignas(16) STensor {
  CUTLASS_HOST_DEVICE
  STensor(void) {
    data_type = mirage::type::DT_UNKNOWN;
    num_dims = 0;
    for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
      dim[i] = 0;
      // stride[i] = 0;
    }
    owner_op = nullptr;
    owner_ts_idx = -1000;
    smem_offset = 128;
    // accum_output = false;
  }

  CUTLASS_HOST_DEVICE
  bool operator==(STensor const &b) const {
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
    if (smem_offset != b.smem_offset) {
      return false;
    }
    return true;
  }

  CUTLASS_HOST_DEVICE
  bool operator!=(STensor const &b) const {
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
    if (smem_offset != b.smem_offset) {
      return true;
    }
    return false;
  }

  size_t size() const {
    size_t num_elements = 1;
    using namespace mirage::type;
    size_t data_type_size = 1;
    switch (data_type) {
      case DT_INT8: {
        data_type_size = 1;
        break;
      }
      case DT_BFLOAT16:
      case DT_FLOAT16: {
        data_type_size = 2;
        break;
      }
      case DT_FLOAT32: {
        data_type_size = 4;
        break;
      }
      case DT_UNKNOWN:
      default:
        assert(false);
    }
    for (int i = 0; i < num_dims; i++) {
      num_elements *= dim[i];
    }
    return num_elements * data_type_size;
  }

  CUTLASS_HOST_DEVICE
  size_t num_elements() const {
    if (num_dims == 4) {
      return dim[0] * dim[1] * dim[2] * dim[3];
    } else if (num_dims == 3) {
      return dim[0] * dim[1] * dim[2];
    } else if (num_dims == 2) {
      return dim[0] * dim[1];
    } else {
      return dim[0];
    }
  }

  mirage::type::DataType data_type;
  mirage::layout::SmemLayout layout;
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  int guid;
  // int stride[MAX_TENSOR_DIMS];
  //  STensor fields
  TBOperator *owner_op;
  int owner_ts_idx;
  int smem_offset;
  // a flag indicating whether we should accumulate output
  // This flag is false by default and should only be set
  // by an output saver to indicate its input should
  // be accumulated
  // bool accum_output;

  static int next_guid;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    STensor, data_type, layout, num_dims, dim, smem_offset, guid)

} // namespace threadblock
} // namespace mirage

namespace std {
template <>
struct hash<mirage::threadblock::STensor> {
  size_t operator()(mirage::threadblock::STensor const &) const;
};
}; // namespace std
