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

#include "mirage/transpiler/common.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpiler.h"
#include "mirage/transpiler/utils.h"
#include "mirage/threadblock/smem_tensor.h"

namespace mirage {
namespace transpiler {

// Get the swizzle plan for a threadblock graph for Blackwell architecture
void Transpiler::get_threadblock_swizzle_plan_blackwell(
    tb::Graph const &tb_graph, TBSched const &sched) {
  // For now, we'll inherit the same swizzle planning logic from Hopper
  // But in a real implementation, this would be customized for Blackwell's 
  // specific memory access patterns and optimizations

  for (auto& [guid, meta] : stensor_metas) {
    // Note: In Blackwell these parameters might need adjustment
    meta.xor_swizzle_b = 4;
    meta.xor_swizzle_m = 3;
    meta.xor_swizzle_s = 0;

    // For Blackwell, we'd customize this depending on the type of tensor
    // For matmul inputs that will be used by MMA - only check m_input since n_input doesn't exist
    if (meta.m_input) {
      meta.is_xor_swizzled = true;
      // Set innermost_dim to 0 as default
      meta.innermost_dim = 0;
    }
    // For other tensors (like outputs), determine swizzling differently
    else if (meta.is_pipelined_input) {
      meta.is_xor_swizzled = true;
      // Set innermost_dim to 0 as default
      meta.innermost_dim = 0;
    }
    // For all other tensors
    else {
      // For now, don't swizzle other tensors
      meta.is_xor_swizzled = false;
      // Set innermost_dim to 0 as default
      meta.innermost_dim = 0;
    }
  }
}

} // namespace transpiler
} // namespace mirage 