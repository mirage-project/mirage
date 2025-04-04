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

#include "mirage/runtime.h"

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

Runtime::Runtime()
  : total_num_tasks(1), total_num_events(1), num_graphs(0) {
  all_events[0].num_triggers = 0;
  all_events[0].first_task_id = 0;
  all_events[0].last_task_id = 0;
  all_tasks[0].task_type = TYPE_TERMINATION;
  all_tasks[0].num_inputs = 0;
  all_tasks[0].num_outputs = 0;
}

void Runtime::register_mugraph(mirage::kernel::Graph const& graph, std::vector<type> task_types) {
  assert(operators.size() == task_types.size());
  for (size_t i = 0; i < operators.size(); i++) {
    assert(operators[i]->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
    // Customized op
    kn::KNCustomizedOp const *cur_op =
        dynamic_cast<kn::KNCustomizedOp const *>(operators[i]);
    tb::Graph const &bgraph = cur_op->bgraph;
    int num_inputs = 0, num_outputs = 0;
    for (const auto& op : bgraph.operators) {
      if (op->op_type == mirage::type::TB_INPUT_OP) {
      }
    }
    if (i == 0) {
      assert(bgraph.grid_dim.x == 1);
      assert(bgraph.grid_dim.y == 1);
      assert(bgraph.grid_dim.z == 1);
    } else {
    }
  }
  num_graphs ++;
}

} // namespace runtime
} // namespace mirage
