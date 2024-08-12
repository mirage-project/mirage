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

#include <unordered_set>

namespace mirage {
namespace transpiler {

// The memory planner, responsible for planning the start address of each
// DTensor and STensor
// Currently we use a simple allocation-only strategy. In the future we may
// incorporate more advanced strategies like memory reuse, etc.
// Everything (size, address, alignment, etc.) in MemoryPlanner is in bytes
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

  // Get the needed size of the buffer
  size_t get_buf_size() {
    return cur_addr;
  }
};

// Plan the memory for every DTensor and STensor
void Transpiler::plan_tensor_memory() {
  // Plan memory for all DTensors
  {
    MemoryPlanner planner;
    for (kn::DTensor &dtensor : all_dtensors) {
      dguid_t guid = dtensor.guid;
      DTensorMeta &meta = dtensor_metas[guid];
      size_t phy_size = meta.num_phy_elems * type::get_datatype_size(dtensor.data_type);
      if (!meta.is_input && !meta.is_output) {
        meta.addr = planner.allocate(phy_size);
      }
    }
    this->d_buf_size = planner.get_buf_size();
  }

  // Plan memory for all STensors
  // TODO(intlsy) Do not allocate memory for STensors that is the accumulator
  // for an output operator with forloop_dim >= 0 (since we do not to
  // accumulator)
  for (kn::KNOperator *const op : this->g->operators) {
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      // Process every KN_CUSTOMIZED_OP separately
      kn::KNCustomizedOp *cur_op = dynamic_cast<kn::KNCustomizedOp *>(op);
      std::unordered_set<sguid_t> processed_sguid;
      MemoryPlanner planner;
      planner.allocate(16); // The lowest 16 bytes are always zero, for matmul
      for (tb::TBOperator *const tb_op : cur_op->bgraph.operators) {
        for (tb::STensor const &stensor :
             Combine(tb_op->input_tensors, tb_op->output_tensors)) {
          sguid_t guid = stensor.guid;
          if (processed_sguid.count(guid)) {
            continue;
          }
          processed_sguid.insert(guid);
          STensorMeta &meta = stensor_metas[guid];
          size_t size = meta.num_phy_elems * type::get_datatype_size(stensor.data_type);
          meta.addr = planner.allocate(size);
        }
      }
      this->custom_op_metas[cur_op].smem_size = planner.get_buf_size();
    }
  }
}

} // namespace transpiler
} // namespace mirage