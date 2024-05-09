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

#include "mirage/type.h"

namespace mirage {
namespace type {

size_t get_datatype_size(DataType type) {
  switch (type) {
    case DT_INT8:
      return 1;
    case DT_BFLOAT16:
    case DT_FLOAT16:
      return 2;
    case DT_FLOAT32:
      return 4;
    case DT_DOUBLE:
      return 8;
    case DT_UNKNOWN:
    default:
      assert(false);
  }
}

} // namespace type
} // namespace mirage
