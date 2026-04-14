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

#include "mirage/kernel/matmul.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <algorithm>
#include <cassert>
#include <iostream>

namespace mirage {
namespace kernel {

DTensor Graph::matmul(DTensor const &A, DTensor const &B) {
  KNOperator *op = create_matmul_op(A, B);
  assert(op != nullptr);
  operators.push_back(op);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::matmul(DTensor const *A, DTensor const *B) {
  KNOperator *op = create_matmul_op(*A, *B);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_matmul_op(DTensor const &A, DTensor const &B) {
  if (A.num_dims != B.num_dims) {
    return nullptr;
  }
  if (A.dim[A.num_dims - 1] != B.dim[B.num_dims - 2]) {
    return nullptr;
  }
  // Allow broadcast in batch dimensions (one of them must be 1 or they must
  // match), supporting GQA where Q has more heads than K/V.
  for (int i = 0; i < A.num_dims - 2; i++) {
    if (A.dim[i] != B.dim[i] && A.dim[i] != 1 && B.dim[i] != 1) {
      return nullptr;
    }
  }

  DTensor C;
  C.num_dims = A.num_dims;
  for (int i = 0; i < C.num_dims - 2; i++) {
    C.dim[i] = std::max(A.dim[i], B.dim[i]);
  }
  C.dim[C.num_dims - 2] = A.dim[A.num_dims - 2];
  C.dim[C.num_dims - 1] = B.dim[C.num_dims - 1];
  C.data_type = A.data_type;

  if (!can_allocate(C)) {
    return nullptr;
  }

  KNMatmulOp *op = new KNMatmulOp(this, A, B);
  return op;
}

KNMatmulOp::KNMatmulOp(Graph *_kgraph, DTensor const &A, DTensor const &B)
    : KNOperator(_kgraph, mirage::type::KN_MATMUL_OP, A, B) {
  DTensor C;
  assert(A.num_dims == B.num_dims);
  assert(A.dim[A.num_dims - 1] == B.dim[B.num_dims - 2]);
  for (int i = 0; i < A.num_dims - 2; i++) {
    assert(A.dim[i] == B.dim[i] || A.dim[i] == 1 || B.dim[i] == 1);
  }
  // Support broadcast in batch dims (e.g. GQA: Q has more heads than K/V).
  C.num_dims = A.num_dims;
  for (int i = 0; i < C.num_dims - 2; i++) {
    C.dim[i] = std::max(A.dim[i], B.dim[i]);
  }
  C.dim[C.num_dims - 2] = A.dim[A.num_dims - 2];
  C.dim[C.num_dims - 1] = B.dim[C.num_dims - 1];
  C.layout = mirage::layout::DmemRowMajor;
  C.data_type = A.data_type;
  C.owner_op = this;
  C.owner_ts_idx = 0;
  C.guid = DTensor::next_guid++;
  kgraph->allocate(C);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

KNMatmulOp::~KNMatmulOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNMatmulOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

void from_json(json const &j, KNMatmulOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU
bool KNMatmulOp::fingerprint(void) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();

  // Compute the broadcast batch size from the output shape.
  int B = 1;
  for (int i = 0; i < output_tensors[0].num_dims - 2; ++i) {
    B *= output_tensors[0].dim[i];
  }
  int M = input_tensors[0].dim[input_tensors[0].num_dims - 2];
  int N = input_tensors[1].dim[input_tensors[1].num_dims - 1];
  int K = input_tensors[0].dim[input_tensors[0].num_dims - 1];

  // Strides for A and B in terms of matrix tiles (handles broadcast).
  int A_batch = 1;
  for (int i = 0; i < input_tensors[0].num_dims - 2; ++i) {
    A_batch *= input_tensors[0].dim[i];
  }
  int B_batch = 1;
  for (int i = 0; i < input_tensors[1].num_dims - 2; ++i) {
    B_batch *= input_tensors[1].dim[i];
  }
  int A_step = (A_batch == 1) ? 0 : M * K;
  int B_step = (B_batch == 1) ? 0 : K * N;

  for (int device_id = 0; device_id < kgraph->gpu_dim.x; ++device_id) {
    type::FPType *A_base = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[0].fp_offset);
    type::FPType *B_base = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[1].fp_offset);
    type::FPType *C_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);
    for (int b = 0; b < B; ++b) {
      type::FPType *A_ptr = A_base + b * A_step;
      type::FPType *B_ptr = B_base + b * B_step;
      utils::compute_matmul_fingerprint(A_ptr, B_ptr, C_ptr + b * M * N,
                                        1, M, N, K);
    }
  }

  return true;
}
#endif // MIRAGE_FINGERPRINT_USE_CPU

} // namespace kernel
} // namespace mirage
