/* Copyright 2023 CMU
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

#include "mirage/kernel/operator.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"

namespace mirage {
namespace kernel {

KNOperator::KNOperator(mirage::type::KNOperatorType _type) : op_type(_type) {}

KNOperator::KNOperator(mirage::type::KNOperatorType _type, DTensor const &A)
    : op_type(_type) {
  input_tensors.push_back(A);
}

KNOperator::KNOperator(mirage::type::KNOperatorType _type,
                       DTensor const &A,
                       DTensor const &B)
    : op_type(_type) {
  input_tensors.push_back(A);
  input_tensors.push_back(B);
}

KNOperator::KNOperator(mirage::type::KNOperatorType _type,
                       std::vector<DTensor> const &inputs)
    : op_type(_type) {
  for (auto const &i : inputs) {
    input_tensors.push_back(i);
  }
}

KNOperator::~KNOperator() {}

DTensor Graph::new_input(std::vector<int> const &dims,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, data_type, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

DTensor* Graph::new_input_ptr(std::vector<int> const &dims,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, data_type, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_input_op(std::vector<int> const &dims,
                                   mirage::type::DataType data_type,
                                   mirage::layout::DmemLayout layout) {
  DTensor tensor;
  tensor.layout = layout;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
    // tensor.stride[i] = (i == tensor.num_dims - 1)
    //                        ? 1
    //                        : tensor.stride[i + 1] * tensor.dim[i + 1];
  }
  tensor.data_type = data_type;

  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (dmm->offset + tensor.data_size() > dmm->total_size) {
    return nullptr;
  }
  KNInputOp *op = new KNInputOp(dims, data_type, layout);
  return op;
}

KNInputOp::KNInputOp(std::vector<int> const &dims,
                     mirage::type::DataType data_type,
                     mirage::layout::DmemLayout layout)
    : KNOperator(mirage::type::KN_INPUT_OP) {
  DTensor tensor;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
    // tensor.stride[i] = (i == tensor.num_dims - 1)
    //                        ? 1
    //                        : tensor.stride[i + 1] * tensor.dim[i + 1];
  }
  tensor.data_type = data_type;
  tensor.layout = layout;
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.guid = DTensor::next_guid++;
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->allocate(tensor);
  output_tensors.push_back(tensor);
}

KNInputOp::~KNInputOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->free(output_tensors[0]);
}

KNInputOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace mirage
