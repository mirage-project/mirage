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

#include "mirage/threadblock/element_binary.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::add(STensor const &input1, STensor const &input2) {
  TBOperator *op =
      create_elementbinary_op(input1, input2, mirage::type::TB_ADD_OP);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor Graph::mul(STensor const &input1, STensor const &input2) {
  TBOperator *op =
      create_elementbinary_op(input1, input2, mirage::type::TB_MUL_OP);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor Graph::div(STensor const &input1, STensor const &input2) {
  TBOperator *op =
      create_elementbinary_op(input1, input2, mirage::type::TB_DIV_OP);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_elementbinary_op(STensor const &input1,
                                           STensor const &input2,
                                           mirage::type::TBOperatorType _type) {
  if (input1.num_dims != input2.num_dims) {
    return nullptr;
  }
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i] && input1.dim[i] > 1 &&
        input2.dim[i] > 1) {
      return nullptr;
    }
  }
  STensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }

  if (smem_offset + output.size() > (off_t)mirage::type::MAX_SMEM_SIZE) {
    return nullptr;
  }

  TBElementBinaryOp *op = new TBElementBinaryOp(this, input1, input2, _type);
  return op;
}

TBElementBinaryOp::TBElementBinaryOp(Graph *_graph,
                                     STensor const &input1,
                                     STensor const &input2,
                                     mirage::type::TBOperatorType _type)
    : TBOperator(_graph, _type, input1, input2) {
  assert(input1.num_dims == input2.num_dims);
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i]) {
      assert(input1.dim[i] == 1 || input2.dim[i] == 1);
    }
  }
  STensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.smem_offset = bgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

TBElementBinaryOp::~TBElementBinaryOp() {
  bgraph->free(output_tensors);
}

TBElementBinaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
