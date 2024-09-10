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

#include <stdexcept>
#include <unordered_set>

#include "mirage/transpiler/utils.h"

namespace mirage {
namespace transpiler {

// Resolve metadata for DTensors
void Transpiler::resolve_dtensor_meta() {
  // get all_dtensors
  std::unordered_set<dguid_t> processed_dguids;
  for (kernel::KNOperator *const op : this->g->operators) {
    for (kn::DTensor const &dtensor :
         Combine(op->input_tensors, op->output_tensors)) {
      if (processed_dguids.count(dtensor.guid) == 0) {
        processed_dguids.insert(dtensor.guid);
        all_dtensors.push_back(dtensor);
      }
    }
  }

  // Set input label and idx for input DTensors
  int cur_input_idx = 0;
  for (kernel::KNOperator *const op : this->g->operators) {
    if (op->op_type == type::KN_INPUT_OP) {
      kn::DTensor const &tensor = op->output_tensors.at(0);
      dguid_t guid = tensor.guid;
      this->dtensor_metas[guid].is_input = true;
      this->dtensor_metas[guid].input_idx = cur_input_idx;
      cur_input_idx += 1;
    }
  }

  // Set output label and idx for output DTensors
  int cur_output_idx = 0;
  for (auto const &output_dtensor : this->mugraph_output_tensors) {
    dguid_t guid = output_dtensor.guid;
    this->dtensor_metas[guid].is_output = true;
    this->dtensor_metas[guid].output_idx = cur_output_idx;
    cur_output_idx += 1;
  }
}

} // namespace transpiler
} // namespace mirage
