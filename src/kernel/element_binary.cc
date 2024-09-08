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

#include "mirage/kernel/element_binary.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

DTensor Graph::add(DTensor const &input1, DTensor const &input2) {
  return elementbinary(input1, input2, mirage::type::KN_ADD_OP);
}

DTensor *Graph::add(DTensor const *input1, DTensor const *input2) {
  return elementbinary(input1, input2, mirage::type::KN_ADD_OP);
}

DTensor Graph::mul(DTensor const &input1, DTensor const &input2) {
  return elementbinary(input1, input2, mirage::type::KN_MUL_OP);
}

DTensor *Graph::mul(DTensor const *input1, DTensor const *input2) {
  return elementbinary(input1, input2, mirage::type::KN_MUL_OP);
}

DTensor Graph::div(DTensor const &input1, DTensor const &input2) {
  return elementbinary(input1, input2, mirage::type::KN_DIV_OP);
}

DTensor *Graph::div(DTensor const *input1, DTensor const *input2) {
  return elementbinary(input1, input2, mirage::type::KN_DIV_OP);
}

DTensor Graph::elementbinary(DTensor const &input1,
                             DTensor const &input2,
                             mirage::type::KNOperatorType type) {
  KNOperator *op = create_elementbinary_op(input1, input2, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementbinary(DTensor const *input1,
                              DTensor const *input2,
                              mirage::type::KNOperatorType type) {
  DTensor output = elementbinary(*input1, *input2, type);
  return &(output.owner_op->output_tensors[0]);
}

KNOperator *Graph::create_elementbinary_op(DTensor const &input1,
                                           DTensor const &input2,
                                           mirage::type::KNOperatorType type) {
  // DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (input1.num_dims != input2.num_dims) {
    return nullptr;
  }
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i] && input1.dim[i] > 1 &&
        input2.dim[i] > 1) {
      return nullptr;
    }
  }
  DTensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }

  if (!can_allocate(output)) {
    return nullptr;
  }

  KNElementBinaryOp *op = new KNElementBinaryOp(this, input1, input2, type);
  return op;
}

KNElementBinaryOp::KNElementBinaryOp(Graph *_kgraph,
                                     DTensor const &input1,
                                     DTensor const &input2,
                                     mirage::type::KNOperatorType type)
    : mirage::kernel::KNOperator(_kgraph, type, input1, input2) {
  assert(input1.num_dims == input2.num_dims);
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i]) {
      assert(input1.dim[i] == 1 || input2.dim[i] == 1);
    }
  }
  DTensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementBinaryOp::~KNElementBinaryOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNElementBinaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace mirage
