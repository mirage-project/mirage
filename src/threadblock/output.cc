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

mirage::kernel::DTensor
    Graph::mark_output(STensor const &stensor,
                       int3 output_map,
                       int output_forloop_dim,
                       mirage::type::TBEpilogueType epilogue) {
  TBOperator *op =
      create_output_op(stensor, output_map, output_forloop_dim, epilogue);
  assert(op != nullptr);
  operators.push_back(op);
  return static_cast<TBOutputOp *>(op)->dtensor;
}

mirage::kernel::DTensor *
    Graph::new_output(STensor const *stensor,
                      int3 output_map,
                      int output_forloop_dim,
                      mirage::type::TBEpilogueType epilogue) {
  TBOperator *op =
      create_output_op(*stensor, output_map, output_forloop_dim, epilogue);
  assert(op != nullptr);
  operators.push_back(op);
  return &(static_cast<TBOutputOp *>(op)->dtensor);
}

TBOperator *Graph::create_output_op(STensor const &stensor,
                                    int3 output_map,
                                    int forloop_dim,
                                    mirage::type::TBEpilogueType epilogue) {
  // TODO(jiazhihao): this check requires an accum operator before output saver
  // we should remove the check when generating distributed kernels
  if (!stensor.after_accum) {
    return nullptr;
  }
  TBOutputOp *op =
      new TBOutputOp(this, stensor, output_map, forloop_dim, epilogue);
  return op;
}

TBOutputOp::TBOutputOp(Graph *_graph,
                       STensor const &input,
                       int3 _output_map,
                       int _forloop_dim,
                       mirage::type::TBEpilogueType _epilogue)
    : TBOperator(_graph, mirage::type::TB_OUTPUT_OP, input),
      output_map(_output_map), forloop_dim(_forloop_dim), epilogue(_epilogue) {
  // Output saver should not have any output stensors
  assert(output_tensors.size() == 0);

  dtensor.num_dims = input.num_dims;
  dtensor.data_type = input.data_type;
  // Currently assume that the output layouts are row-major
  dtensor.layout = mirage::layout::DmemRowMajor;
  for (int i = 0; i < dtensor.num_dims; i++) {
    dtensor.dim[i] = input.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && bgraph->grid_dim.x > 1) {
      dim_idx = output_map.x;
      dim_div = bgraph->grid_dim.x;
    }
    if (d == 1 && bgraph->grid_dim.y > 1) {
      dim_idx = output_map.y;
      dim_div = bgraph->grid_dim.y;
    }
    if (d == 2 && bgraph->grid_dim.z > 1) {
      dim_idx = output_map.z;
      dim_div = bgraph->grid_dim.z;
    }
    if (dim_idx >= 0) {
      assert(dtensor.dim[dim_idx] > 0);
      dtensor.dim[dim_idx] *= dim_div;
    }
  }

  if (forloop_dim >= 0) {
    dtensor.dim[forloop_dim] *= bgraph->forloop_range;
  }
}

TBOutputOp::~TBOutputOp() {
  assert(output_tensors.size() == 0);
  // bgraph->free_fingerprint(output_tensors);
}

TBOutputOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"dtensor", dtensor},
              {"output_map", output_map}};
}

size_t TBOutputOp::get_dtensor_guid() {
  return dtensor.guid;
}

} // namespace threadblock
} // namespace mirage
