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

#include "mirage/kernel/reduction.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;

DTensor Graph::reduction(DTensor const &input, int dim, int size) {
  KNOperator *op = create_reduction_op(input, dim, size);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor* Graph::reduction(DTensor const *input, int dim, int size) {
  KNOperator *op = create_reduction_op(*input, dim, size);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *
    Graph::create_reduction_op(DTensor const &input, int dim, int size) {
  if (input.num_dims <= dim) {
    return nullptr;
  }
  if (input.dim[dim] % size != 0) {
    return nullptr;
  }
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (dmm->offset + input.data_size() > dmm->total_size) {
    return nullptr;
  }

  KNReductionOp *op = new KNReductionOp(input, dim, size);
  return op;
}

KNReductionOp::KNReductionOp(DTensor const &input, int dim, int size)
    : KNOperator((KNOperatorType)(KN_REDUCTION_0_OP + dim), input),
      reduction_dim_idx(dim), reduction_dim_size(size) {
  DTensor output = input;
  assert(dim < output.num_dims);
  assert(output.dim[dim] % size == 0);
  output.dim[dim] = size;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNReductionOp::~KNReductionOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i]);
  }
}

KNReductionOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace mirage
