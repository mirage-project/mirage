/* Copyright 2025 CMU
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
#include "mirage/runtime/runtime.h"
#include "rms_norm.cuh"

namespace mirage {
namespace runtime {
template <typename Kernel>
__device__ void generic_wrapper_kernel(TensorDesc *inputs,
                                       TensorDesc *outputs,
                                       int4 *tensor_offsets,
                                       int forloop_range) {

  auto params = Kernel::pack_parameters(inputs, outputs, tensor_offsets);
  auto layouts =
      Kernel::create_layouts(inputs, outputs);
  Kernel::execute(params, layouts);
}
}; // namespace runtime
}; // namespace mirage