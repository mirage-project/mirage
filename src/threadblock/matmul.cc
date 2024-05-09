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

#include "mirage/threadblock/matmul.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::matmul(STensor const &A, STensor const &B) {
  TBOperator *op = create_matmul_op(A, B);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_matmul_op(STensor const &A, STensor const &B) {
  if (A.num_dims != B.num_dims) {
    return nullptr;
  }
  if (A.dim[A.num_dims - 1] != B.dim[B.num_dims - 2]) {
    return nullptr;
  }
  for (int i = 0; i < A.num_dims - 2; i++) {
    if (A.dim[i] != 1 || B.dim[i] != 1) {
      return nullptr;
    }
  }

  STensor C;
  C.num_dims = A.num_dims;
  for (int i = 0; i < C.num_dims; i++) {
    C.dim[i] = A.dim[i];
  }
  C.dim[C.num_dims - 1] = B.dim[C.num_dims - 1];
  C.data_type = A.data_type;
  if (smem_offset + (off_t)C.size() > (off_t)mirage::type::MAX_SMEM_SIZE) {
    return nullptr;
  }

  TBMatmulOp *op = new TBMatmulOp(this, A, B);
  return op;
}

TBMatmulOp::TBMatmulOp(Graph *_graph, STensor const &A, STensor const &B)
    : TBOperator(_graph, mirage::type::TB_MATMUL_OP, A, B) {
  STensor C;
  assert(A.num_dims == B.num_dims);
  // Check that this is not a TB-level batch matmul
  for (int i = 0; i < A.num_dims - 2; i++) {
    assert(A.dim[i] == 1);
    assert(B.dim[i] == 1);
  }
  // Currently only support row-major output
  // to be consistent with cutlass
  C.layout = mirage::layout::SmemRowMajor;
  C.num_dims = A.num_dims;
  for (int i = 0; i < C.num_dims; i++) {
    C.dim[i] = A.dim[i];
  }
  C.dim[C.num_dims - 1] = B.dim[C.num_dims - 1];
  C.data_type = A.data_type;
  C.owner_op = this;
  C.owner_ts_idx = 0;
  C.guid = STensor::next_guid++;
  C.smem_offset = bgraph->allocate(C);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

TBMatmulOp::~TBMatmulOp() {
  bgraph->free(output_tensors);
}

TBMatmulOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
