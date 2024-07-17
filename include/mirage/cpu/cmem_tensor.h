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

#include "cutlass/cutlass.h"
#include "mirage/config.h"
#include "mirage/layout.h"
#include "mirage/type.h"
#include <cstddef>

namespace mirage {
namespace cpu {

constexpr int MAX_TENSOR_DIMS = 4;

struct CTensor {
  CTensor(void) {
    data_type = mirage::type::DT_UNKNOWN;
    num_dims = 0;
    for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
      dim[i] = 0;
    }
    for (int i = 0; i < mirage::config::MAX_NUM_GPUS; i++) {
      fp_ptr[i] = nullptr;
    }
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
  mirage::layout::CmemLayout layout;
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  mirage::type::FPType *fp_ptr[mirage::config::MAX_NUM_GPUS];
};

} // namespace cpu
} // namespace mirage
