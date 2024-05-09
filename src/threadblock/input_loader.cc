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

#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::new_input(mirage::kernel::DTensor const &dtensor,
                         int3 input_map,
                         int forloop_dim,
                         mirage::layout::SmemLayout layout) {
  TBOperator *op = create_input_op(dtensor, input_map, forloop_dim, layout);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_input_op(mirage::kernel::DTensor const &dtensor,
                                   int3 input_map,
                                   int forloop_dim,
                                   mirage::layout::SmemLayout layout) {
  STensor tensor;
  tensor.num_dims = dtensor.num_dims;
  tensor.data_type = dtensor.data_type;
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = dtensor.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && grid_dim.x > 1) {
      dim_idx = input_map.x;
      dim_div = grid_dim.x;
    }
    if (d == 1 && grid_dim.y > 1) {
      dim_idx = input_map.y;
      dim_div = grid_dim.y;
    }
    if (d == 2 && grid_dim.z > 1) {
      dim_idx = input_map.z;
      dim_div = grid_dim.z;
    }
    if (dim_idx >= 0) {
      assert(tensor.dim[dim_idx] > 0);
      assert(tensor.dim[dim_idx] % dim_div == 0);
      tensor.dim[dim_idx] /= dim_div;
    }
  }

  if (forloop_dim >= 0) {
    assert(tensor.dim[forloop_dim] > 0);
    assert(tensor.dim[forloop_dim] % forloop_range == 0);
    tensor.dim[forloop_dim] /= forloop_range;
  }
  // our data loader only supports 2D matrices
  // (i.e., only the last two dims can be larger than 1
  for (int i = 0; i < tensor.num_dims - 2; i++) {
    if (tensor.dim[i] != 1) {
      return nullptr;
    }
    // assert(tensor.dim[i] == 1);
  }

  if (smem_offset + (off_t)tensor.size() > (off_t)mirage::type::MAX_SMEM_SIZE) {
    //printf("smem_offset(%d) tensor.size(%d)\n", smem_offset, tensor.size());
    return nullptr;
  }

  TBInputOp *op = new TBInputOp(this, dtensor, input_map, forloop_dim, layout);
  return op;
}

TBInputOp::TBInputOp(Graph *_graph,
                     mirage::kernel::DTensor const &_dtensor,
                     int3 _input_map,
                     int _forloop_dim,
                     mirage::layout::SmemLayout _layout)
    : TBOperator(_graph, mirage::type::TB_INPUT_OP), dtensor(_dtensor),
      input_map(_input_map), forloop_dim(_forloop_dim) {
  STensor tensor;
  tensor.layout = _layout;
  tensor.num_dims = dtensor.num_dims;
  tensor.data_type = dtensor.data_type;
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = dtensor.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && bgraph->grid_dim.x > 1) {
      dim_idx = input_map.x;
      dim_div = bgraph->grid_dim.x;
    }
    if (d == 1 && bgraph->grid_dim.y > 1) {
      dim_idx = input_map.y;
      dim_div = bgraph->grid_dim.y;
    }
    if (d == 2 && bgraph->grid_dim.z > 1) {
      dim_idx = input_map.z;
      dim_div = bgraph->grid_dim.z;
    }
    if (dim_idx >= 0) {
      assert(tensor.dim[dim_idx] > 0);
      assert(tensor.dim[dim_idx] % dim_div == 0);
      tensor.dim[dim_idx] /= dim_div;
    }
  }

  if (forloop_dim >= 0) {
    assert(tensor.dim[forloop_dim] > 0);
    assert(tensor.dim[forloop_dim] % bgraph->forloop_range == 0);
    tensor.dim[forloop_dim] /= bgraph->forloop_range;
  }

  // Our data loader only supports 2D matrices
  for (int i = 0; i < tensor.num_dims - 2; i++) {
    assert(tensor.dim[i] == 1);
  }
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.guid = STensor::next_guid++;
  tensor.smem_offset = bgraph->allocate(tensor);
  output_tensors.push_back(tensor);
}

TBInputOp::~TBInputOp() {
  bgraph->free(output_tensors[0]);
}

TBInputOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"dtensor", dtensor},
              {"input_map", input_map},
              {"forloop_dim", forloop_dim}};
}

} // namespace threadblock
} // namespace mirage
