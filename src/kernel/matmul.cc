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
#include "mirage/utils/hash_utils.h"
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

DTensor* Graph::matmul(DTensor const *A, DTensor const *B) {
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
  for (int i = 0; i < A.num_dims - 2; i++) {
    if (A.dim[i] != B.dim[i]) {
      return nullptr;
    }
  }

  DTensor C;
  C.num_dims = A.num_dims;
  for (int i = 0; i < C.num_dims; i++) {
    C.dim[i] = A.dim[i];
  }
  C.dim[C.num_dims - 1] = B.dim[C.num_dims - 1];
  C.data_type = A.data_type;

  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (dmm->offset + C.data_size() > dmm->total_size) {
    return nullptr;
  }

  KNMatmulOp *op = new KNMatmulOp(A, B);
  return op;
}

KNMatmulOp::KNMatmulOp(DTensor const &A, DTensor const &B)
    : mirage::kernel::KNOperator(mirage::type::KN_MATMUL_OP, A, B) {
  DTensor C;
  assert(A.num_dims == B.num_dims);
  assert(A.dim[A.num_dims - 1] == B.dim[B.num_dims - 2]);
  for (int i = 0; i < A.num_dims - 2; i++) {
    assert(A.dim[i] == B.dim[i]);
  }
  // Currently only support row-major output
  // to be consistent with cutlass
  C.num_dims = A.num_dims;
  for (int i = 0; i < C.num_dims; i++) {
    C.dim[i] = A.dim[i];
  }
  C.dim[C.num_dims - 1] = B.dim[C.num_dims - 1];
  C.layout = mirage::layout::DmemRowMajor;
  C.data_type = A.data_type;
  C.owner_op = this;
  C.owner_ts_idx = 0;
  C.guid = DTensor::next_guid++;
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->allocate(C);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

KNMatmulOp::~KNMatmulOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i]);
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

MatmulKey::MatmulKey(DTensor const &A, DTensor const &B)
    : operand_a(A), operand_b(B) {}

bool MatmulKey::operator==(MatmulKey const &b) const {
  if (b.operand_a != operand_a) {
    return false;
  }
  if (b.operand_b != operand_b) {
    return false;
  }
  return true;
}

} // namespace kernel
} // namespace mirage

namespace std {
size_t hash<mirage::kernel::MatmulKey>::operator()(
    mirage::kernel::MatmulKey const &key) const {
  size_t ret = 0;
  hash_combine(ret, key.operand_a);
  hash_combine(ret, key.operand_b);
  return ret;
}
}; // namespace std
