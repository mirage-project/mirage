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

#include "mirage/kernel/all_reduce.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>
#include <iostream>

namespace mirage {
namespace kernel {

DTensor Graph::all_reduce(DTensor const &input, bool inplace) {
  KNOperator *op = create_all_reduce_op(input, inplace);
  assert(op != nullptr);
  operators.push_back(op);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::all_reduce(DTensor const *input, bool inplace) {
  KNOperator *op = create_all_reduce_op(*input, inplace);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_all_reduce_op(DTensor const &input, bool inplace) {
  // We do inplace allreduce
  KNAllReduceOp *op = new KNAllReduceOp(this, input, inplace);
  return op;
}

KNAllReduceOp::KNAllReduceOp(Graph *_kgraph,
                             DTensor const &input,
                             bool _inplace)
    : KNOperator(_kgraph, mirage::type::KN_ALLREDUCE_OP, input),
      inplace(_inplace) {
  DTensor output;
  output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  if (inplace) {
    // Assert that the data_offset and fp_offset are
    // identical to these of the input
    assert(output.data_offset == input.data_offset);
    assert(output.fp_offset == input.fp_offset);
  } else {
    kgraph->allocate(output);
  }
  output_tensors.push_back(output);
}

KNAllReduceOp::~KNAllReduceOp() {
  if (!inplace) {
    kgraph->free(output_tensors[0]);
  }
}

KNAllReduceOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"inplace", inplace}};
}

} // namespace kernel
} // namespace mirage
