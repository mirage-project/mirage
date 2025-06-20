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

#ifdef MIRAGE_BACKEND_USE_CUDA

#include <vector_types.h>

namespace mirage {
namespace threadblock {

CUTLASS_HOST_DEVICE
void deserialize_elementunary_op_parameters(int const *params,
                                            int &param_idx,
                                            int &smem_offset,
                                            int &num_elements) {
  smem_offset = params[param_idx++];
  num_elements = params[param_idx++];
}

inline void serialize_elementunary_op_parameters(int *params,
                                                 int &param_idx,
                                                 int smem_offset,
                                                 int num_elements) {
  params[param_idx++] = smem_offset;
  params[param_idx++] = num_elements;

  assert(param_idx <= NewKernelParams::MAX_NUM_PARAMETERS);
}

} // namespace threadblock
} // namespace mirage

#endif
