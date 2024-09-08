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

#include "mirage/kernel/element_unary.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

DTensor Graph::exp(DTensor const &input) {
  return elementunary(input, mirage::type::KN_EXP_OP);
}

DTensor *Graph::exp(DTensor const *input) {
  return elementunary(input, mirage::type::KN_EXP_OP);
}

DTensor Graph::square(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SQUARE_OP);
}

DTensor *Graph::square(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SQUARE_OP);
}

DTensor Graph::sqrt(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SQRT_OP);
}

DTensor *Graph::sqrt(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SQRT_OP);
}

DTensor Graph::silu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SILU_OP);
}

DTensor *Graph::silu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SILU_OP);
}

DTensor Graph::elementunary(DTensor const &input,
                            mirage::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary(DTensor const *input,
                             mirage::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(*input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_op(DTensor const &input,
                                          mirage::type::KNOperatorType type) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNElementUnaryOp(this, input, type);
  return op;
}

KNElementUnaryOp::KNElementUnaryOp(Graph *_kgraph,
                                   DTensor const &input,
                                   mirage::type::KNOperatorType type)
    : mirage::kernel::KNOperator(_kgraph, type, input) {
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementUnaryOp::~KNElementUnaryOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace mirage
