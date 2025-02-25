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

#include "mirage/threadblock/cluster_accum.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::cluster_accum(STensor const &input,
                             mirage::type::TBOperatorType type) {
  TBOperator *op = create_cluster_accum_op(input, type);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::cluster_accum(STensor const *input,
                              mirage::type::TBOperatorType type) {
  TBOperator *op = create_cluster_accum_op(*input, type);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_cluster_accum_op(STensor const &input,
                                           mirage::type::TBOperatorType type) {
  // // Input stensor must be before accumulation (i.e., inside forloop)
  // if (input.after_accum) {
  //   return nullptr;
  // }
  TBClusterAccumOp *op = new TBClusterAccumOp(this, input, type);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
  return op;
}

TBClusterAccumOp::TBClusterAccumOp(Graph *_graph,
                                   STensor const &input,
                                   mirage::type::TBOperatorType type)
    : TBOperator(_graph, type, input) {
  assert(type >= mirage::type::TB_CLUSTER_ACCUM_NO_RED_OP);
  assert(type < mirage::type::TB_CLUSTER_ACCUM_LAST_OP);
  // assert(!input.after_accum);
  STensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.after_accum = true;
  output.smem_offset = bgraph->allocate_fingerprint(output);
  output_tensors.push_back(output);
}

TBClusterAccumOp::~TBClusterAccumOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBClusterAccumOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
