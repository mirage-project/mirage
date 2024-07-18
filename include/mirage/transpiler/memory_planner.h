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

#include <cstddef>
#include <vector>

#include "mirage/transpiler/utils.h"

namespace mirage {
namespace transpiler {

// The memory planner, responsible for planning the start address of each
// DTensor and STensor
// Currently we use a simple allocation strategy. In the future we may
// incorporate more advanced strategies like memory reuse, etc.
template <size_t ALIGN = 16>
class MemoryPlanner {
private:
  size_t cur_addr = 0;

public:
  // size is in bytes
  size_t allocate(size_t size) {
    size_t addr = cur_addr;
    assert(addr % ALIGN == 0);
    cur_addr += size;
    cur_addr = round_to_multiple(cur_addr, ALIGN);
    return addr;
  }
};

} // namespace transpiler
} // namespace mirage
