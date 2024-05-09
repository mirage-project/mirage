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

#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::exp(STensor const &input) {
  TBOperator *op = create_elementunary_op(input, mirage::type::TB_EXP_OP);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_elementunary_op(STensor const &input,
                                          mirage::type::TBOperatorType _type) {
  TBElementUnaryOp *op = new TBElementUnaryOp(this, input, _type);
  return op;
}

TBElementUnaryOp::TBElementUnaryOp(Graph *_graph,
                                   STensor const &input,
                                   mirage::type::TBOperatorType _type)
    : TBOperator(_graph, _type, input) {
  STensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  // Note that we inplace the output by default
  // output.smem_offset = bgraph->allocate(output);
  output.smem_offset = input.smem_offset;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

TBElementUnaryOp::~TBElementUnaryOp() {
  // Don't free since we inplace the output by default
  // bgraph->free(output_tensors);
}

TBElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
