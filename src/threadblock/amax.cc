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

 #include "mirage/threadblock/amax.h"
 #include "mirage/threadblock/graph.h"
 #include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::amax(STensor const &input) {
  TBOperator *op = create_amax_op(input, mirage::type::TB_AMAX_OP);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor* Graph::amax(STensor const *input) {
  TBOperator *op = create_amax_op(*input, mirage::type::TB_AMAX_OP);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator* Graph::create_amax_op(STensor const &input,
                                    mirage::type::TBOperatorType _type) {
  TBAmaxOp *op = new TBAmaxOp(this, input, _type);
  return op;
}

TBAmaxOp::TBAmaxOp(Graph *_graph, 
                           STensor const &input,
                           mirage::type::TBOperatorType _type) 
    : TBOperator(_graph, _type, input) {
  STensor output = input;
  assert(output.layout == mirage::layout::SmemRowMajor);
  // output dim: refer to 
  for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
    output.dim[i] = 1;
  }
  output.owner_op = this;
  output.owner_ts_idx = 0; 
  output.guid = STensor::next_guid++;
  output.after_accum = input.after_accum;     
  output.smem_offset = bgraph->allocate_fingerprint(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

TBAmaxOp::~TBAmaxOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBAmaxOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage