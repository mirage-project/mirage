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

#include "mirage/transpiler/transpiler.h"

namespace mirage {
namespace transpiler {

// Plan the memory for every DTensor
void Transpiler::plan_dtensor_memory() {
  // The memory planner, responsible for planning the start address of each
  // DTensor
  // Currently we use a simple allocation-only strategy. In the future we may
  // incorporate more advanced strategies like memory reuse, etc.
  // Everything (size, address, alignment, etc.) in MemoryPlanner is in bytes
  static constexpr size_t ALIGN = 128;
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

    // Get the needed size of the buffer
    size_t get_buf_size() {
      return cur_addr;
    }
  };

  MemoryPlanner planner;
  for (kn::DTensor &dtensor : all_dtensors) {
    dguid_t guid = dtensor.guid;
    DTensorMeta &meta = dtensor_metas[guid];
    size_t phy_size =
        meta.num_phy_elems * type::get_datatype_size(dtensor.data_type);
    if (!meta.is_input && !meta.is_output) {
      meta.addr = planner.allocate(phy_size);
    }
  }
  this->d_buf_size = planner.get_buf_size();
}

} // namespace transpiler
} // namespace mirage