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

#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"

namespace mirage {
namespace kernel {

std::vector<size_t> get_default_strides(DTensor const &A) {
  std::vector<size_t> strides(A.num_dims);
  size_t stride = 1;
  for (int i = A.num_dims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= A.dim[i];
  }
  return strides;
}

void Graph::mark_output(DTensor const &A) {
  return mark_output(A, get_default_strides(A));
}

void Graph::mark_output(DTensor const *A) {
  return mark_output(A, get_default_strides(*A));
}

void Graph::mark_output(DTensor const &A, std::vector<size_t> const &strides) {
  KNOperator *op = create_output_op(A, strides);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 0);
}

void Graph::mark_output(DTensor const *A, std::vector<size_t> const &strides) {
  KNOperator *op = create_output_op(*A, strides);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 0);
}

KNOperator *Graph::create_output_op(DTensor const &A,
                                    std::vector<size_t> const &strides) {
  KNOutputOp *op = new KNOutputOp(this, A, strides);
  return op;
}

KNOutputOp::KNOutputOp(Graph *_kgraph,
                       DTensor const &A,
                       std::vector<size_t> const &strides,
                       int3 _output_map)
    : KNOperator(_kgraph, mirage::type::KN_OUTPUT_OP, A),
      output_strides(strides), output_map(_output_map) {}

KNOutputOp::~KNOutputOp() {}

KNOutputOp::operator json() const {
  return json{{"op_type", op_type},
              {"output_strides", output_strides},
              {"output_map", output_map},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace mirage
