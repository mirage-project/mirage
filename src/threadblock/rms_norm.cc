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

#include "mirage/threadblock/rms_norm.h"
#include "mirage/threadblock/graph.h"
#include <cassert>

namespace mirage {
namespace threadblock {

STensor Graph::rms_norm(STensor const &input) {
  TBOperator *op = create_rms_norm_op(input);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::rms_norm(STensor const *input) {
  TBOperator *op = create_rms_norm_op(*input);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_rms_norm_op(STensor const &input) {
  TBOperator *op = new TBRmsNormOp(this, input);
  // check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
}

TBRmsNormOp::TBRmsNormOp(Graph *_graph, STensor const &input)
    : TBOperator(_graph, mirage::type::TB_RMS_NORM_OP, input) {
  STensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.after_accum = input.after_accum;
  output.smem_offset = bgraph->allocate_fingerprint(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

TBRmsNormOp::~TBRmsNormOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBRmsNormOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
