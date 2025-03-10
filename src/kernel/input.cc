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
                         int3 input_map,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, strides, input_map, data_type, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

DTensor *Graph::new_input_ptr(std::vector<int> const &dims,
                              std::vector<size_t> const &strides,
                              int3 input_map,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, strides, input_map, data_type, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

// For the case where the tensor has been divided during construction. Avoid redundant division.
DTensor Graph::new_input_from_constructed(std::vector<int> const &dims,
                         std::vector<size_t> const &strides,
                         int3 input_map,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout) {
  KNOperator *op = create_input_op(dims, strides, input_map, data_type, layout, true);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

KNOperator *Graph::create_input_op(std::vector<int> const &dims,
                                   std::vector<size_t> const &strides,
                                   int3 input_map,
                                   mirage::type::DataType data_type,
                                   mirage::layout::DmemLayout layout,
                                   bool dim_divided) {
  DTensor tensor;
  tensor.layout = layout;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
  }
  tensor.data_type = data_type;

  KNInputOp *op =
      new KNInputOp(this, dims, strides, data_type, layout, input_map, dim_divided);

  if (!can_allocate(op->output_tensors[0])) {
    return nullptr;
  }
  return op;
}

KNInputOp::KNInputOp(Graph *_graph,
                     std::vector<int> const &dims,
                     std::vector<size_t> const &strides,
                     mirage::type::DataType data_type,
                     mirage::layout::DmemLayout layout,
                     int3 _input_map,
                     bool dim_divided)
    : KNOperator(_graph, mirage::type::KN_INPUT_OP), input_strides(strides),
      input_map(_input_map) {
  DTensor tensor;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
  }

  if(!dim_divided) { // The tensor has been divided during construction. Avoid redundant division.
    for (int d = 0; d < 3; d++) {
      int dim_idx = -1;
      int dim_div = 1;
      if (d == 0 && kgraph->gpu_dim.x > 1) {
        dim_idx = input_map.x;
        dim_div = kgraph->gpu_dim.x;
      }
      if (d == 1 && kgraph->gpu_dim.y > 1) {
        dim_idx = input_map.y;
        dim_div = kgraph->gpu_dim.y;
      }
      if (d == 2 && kgraph->gpu_dim.z > 1) {
        dim_idx = input_map.z;
        dim_div = kgraph->gpu_dim.z;
      }

      if (dim_idx >= 0) {
        printf("d=%d:    dim_idx: %d, dim_div: %d, tensor.dim[%d]: %d\n", d, dim_idx, dim_div, dim_idx, tensor.dim[dim_idx]);
        assert(tensor.dim[dim_idx] > 0);
        assert(tensor.dim[dim_idx] % dim_div == 0);
        tensor.dim[dim_idx] /= dim_div;
        printf("After: tensor.dim[%d]: %d\n", dim_idx, tensor.dim[dim_idx]);
      }
    }
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

} // namespace kernel
} // namespace mirage
