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

#include "mirage/threadblock/reduction.h"
#include "mirage/threadblock/graph.h"
#include <cassert>

namespace mirage {
namespace threadblock {

STensor Graph::reduction(STensor const &input, int dim) {
  TBOperator *op = create_reduction_op(input, dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_reduction_op(STensor const &input, int dim) {
  STensor output = input;
  assert(output.num_dims > dim);
  assert(output.layout == mirage::layout::SmemRowMajor);
  output.dim[dim] = 1;
  if (dim < output.num_dims - 2) {
    return nullptr;
  }

  if (smem_offset + (off_t)output.size() > (off_t)mirage::type::MAX_SMEM_SIZE) {
    return nullptr;
  }

  TBOperator *op = new TBReductionOp(this, input, dim, 1 /*size*/);

  return op;
}

STensor Graph::reduction_to_dimx(STensor const &input, int dim) {
  TBOperator *op = create_reduction_to_dimx_op(input, dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_reduction_to_dimx_op(STensor const &input, int dim) {
  STensor output = input;
  assert(output.num_dims > dim);
  assert(output.layout == mirage::layout::SmemRowMajor);
  output.dim[dim] = this->reduction_dimx;

  if (smem_offset + (off_t)output.size() > (off_t)mirage::type::MAX_SMEM_SIZE) {
    return nullptr;
  }

  TBOperator *op =
      new TBReductionOp(this, input, dim, this->reduction_dimx);

  return op;
}

TBReductionOp::TBReductionOp(Graph *bgraph,
                             STensor const &input,
                             int dim,
                             int size)
    : TBOperator(bgraph,
                 size == 1 ? (mirage::type::TBOperatorType)(
                                 mirage::type::TB_REDUCTION_0_OP + dim)
                           : (mirage::type::TBOperatorType)(
                                 mirage::type::TB_REDUCTION_0_TO_DIMX_OP + dim),
                 input),
      reduce_dim(dim), reduce_size(size) {
  // mirage::type::TBOperatorType type = static_cast<mirage::type::TBOperatorType>(
  //     mirage::type::TB_REDUCTION_0_OP + dim);
  // this->op_type = type;
  STensor output = input;
  assert(output.num_dims > reduce_dim);
  assert(output.layout == mirage::layout::SmemRowMajor);
  output.dim[reduce_dim] = reduce_size;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.smem_offset = bgraph->allocate(output);
  output_tensors.push_back(output);
}

TBReductionOp::~TBReductionOp() {
  bgraph->free(output_tensors[0]);
}

TBReductionOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"reduce_dim", reduce_dim},
              {"reduce_size", reduce_size}};
}
} // namespace threadblock
} // namespace mirage
