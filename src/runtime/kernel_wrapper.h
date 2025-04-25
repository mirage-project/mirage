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
#include "attention_part1.cuh"
#include "attention_part2.cuh"
#include "embedding.cuh"
#include "mirage/runtime/runtime.h"
#include "rms_norm.cuh"
#include "silu_mul_linear.cuh"

namespace mirage {
namespace runtime {
template <typename Kernel>
__device__ __forceinline__ void generic_wrapper_kernel(TaskDesc &task_desc,
                                                       int4 *tensor_offsets,
                                                       int forloop_range) {
  Kernel::execute(task_desc, tensor_offsets, forloop_range);
}
}; // namespace runtime
}; // namespace mirage