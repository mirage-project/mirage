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

DTensor Graph::new_input(std::vector<int> const &dims,
                         std::vector<size_t> const &strides,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, strides, data_type, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

DTensor *Graph::new_input_ptr(std::vector<int> const &dims,
                              std::vector<size_t> const &strides,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, strides, data_type, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_input_op(std::vector<int> const &dims,
                                   std::vector<size_t> const &strides,
                                   mirage::type::DataType data_type,
                                   mirage::layout::DmemLayout layout) {
  DTensor tensor;
  tensor.layout = layout;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
  }
  tensor.data_type = data_type;

  if (!can_allocate(tensor)) {
    return nullptr;
  }
  KNInputOp *op = new KNInputOp(this, dims, strides, data_type, layout);
  return op;
}

KNInputOp::KNInputOp(Graph *_graph,
                     std::vector<int> const &dims,
                     std::vector<size_t> const &strides,
                     mirage::type::DataType data_type,
                     mirage::layout::DmemLayout layout,
                     int3 _input_map)
    : KNOperator(_graph, mirage::type::KN_INPUT_OP), input_strides(strides),
      input_map(_input_map) {
  assert(dims.size() == strides.size());
  DTensor tensor;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
  }
  tensor.data_type = data_type;
  tensor.layout = layout;
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.guid = DTensor::next_guid++;
  kgraph->allocate(tensor);
  output_tensors.push_back(tensor);
}

KNInputOp::~KNInputOp() {
  kgraph->free(output_tensors[0]);
}

KNInputOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_strides", input_strides},
              {"input_map", input_map},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU
bool KNInputOp::fingerprint(void) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  type::FPType value = 0;
  for (int device_id = 0; device_id < kgraph->gpu_dim.x; ++device_id) {
    type::FPType *fp_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);
    for (size_t i = 0; i < output_tensors[0].num_elements(); ++i) {
      fp_ptr[i] = value;
      value = (value + 1) % config::FP_PQ;
    }
  }
  return true;
}
#endif

} // namespace kernel
} // namespace mirage
