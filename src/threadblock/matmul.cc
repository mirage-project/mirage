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

#include <stdexcept>

namespace mirage {
namespace threadblock {

// create_matmul_op returns nullptr for a shape it cannot build -- mismatched
// dims, a TB-level batch matmul, or (most often) an operand set that exceeds
// MAX_SMEM_SIZE. The assert that used to guard this is compiled out by NDEBUG
// in the Release build, so the nullptr was dereferenced immediately below and
// the whole process segfaulted with no diagnostic. Reported as an exception
// instead; CCore.pxd declares these `except +` so it surfaces in Python.
//
// The other threadblock ops had the same latent segfault and now go through the
// shared check_tb_op() in graph.cc; matmul keeps its own check only because it
// can name the offending operand shapes.
static void check_matmul_op(TBOperator const *op,
                            STensor const &A,
                            STensor const &B) {
  if (op == nullptr) {
    throw std::runtime_error(
        "threadblock matmul: unsupported operand shapes (" +
        std::to_string(A.dim[A.num_dims - 2]) + "x" +
        std::to_string(A.dim[A.num_dims - 1]) + " @ " +
        std::to_string(B.dim[B.num_dims - 2]) + "x" +
        std::to_string(B.dim[B.num_dims - 1]) +
        "); dims may mismatch or the operands may exceed shared memory");
  }
}

STensor Graph::matmul(STensor const &A, STensor const &B) {
  TBOperator *op = create_matmul_op(A, B);
  check_matmul_op(op, A, B);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::matmul(STensor const *A, STensor const *B) {
  TBOperator *op = create_matmul_op(*A, *B);
  check_matmul_op(op, *A, *B);
  operators.push_back(op);
  return &op->output_tensors[0];
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
  if (A.after_accum != B.after_accum) {
    return nullptr;
  }

  TBMatmulOp *op = new TBMatmulOp(this, A, B);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
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
  C.after_accum = A.after_accum;
  C.smem_offset = bgraph->allocate_fingerprint(C);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

TBMatmulOp::~TBMatmulOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBMatmulOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
