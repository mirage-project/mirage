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

#include "mirage/threadblock/concat.h"
#include "mirage/threadblock/graph.h"
#include <cassert>

namespace mirage {
namespace threadblock {

STensor Graph::concat(STensor const &A, STensor const &B, int concat_dim) {
  TBOperator *op = create_concat_op(A, B, concat_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::concat(STensor const *A, STensor const *B, int concat_dim) {
  TBOperator *op = create_concat_op(*A, *B, concat_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_concat_op(STensor const &A,
                                    STensor const &B,
                                    int concat_dim) {
  if (A.num_dims != B.num_dims) {
    return nullptr;
  }
  if (A.num_dims <= concat_dim) {
    return nullptr;
  }
  for (int i = 0; i < A.num_dims; i++) {
    if (i != concat_dim && A.dim[i] != B.dim[i]) {
      return nullptr;
    }
  }
  if (A.after_accum != B.after_accum) {
    return nullptr;
  }

  // if (smem_offset + A.size() + B.size() >
  //     (off_t)mirage::config::MAX_SMEM_SIZE) {
  //   return nullptr;
  // }

  TBOperator *op = new TBConcatOp(this, A, B, concat_dim);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
}

TBConcatOp::TBConcatOp(Graph *bgraph,
                       STensor const &A,
                       STensor const &B,
                       int dim)
    : TBOperator(
          bgraph,
          (mirage::type::TBOperatorType)(mirage::type::TB_CONCAT_0_OP + dim),
          A,
          B),
      concat_dim(dim) {
  STensor output = A;
  assert(output.num_dims > concat_dim);
  assert(A.layout == B.layout);
  output.dim[concat_dim] = A.dim[concat_dim] + B.dim[concat_dim];
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.after_accum = A.after_accum;
  output.smem_offset = bgraph->allocate_fingerprint(output);
  output_tensors.push_back(output);
}

TBConcatOp::~TBConcatOp() {
  bgraph->free_fingerprint(output_tensors[0]);
}

TBConcatOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"concat_dim", concat_dim}};
}
} // namespace threadblock
} // namespace mirage
