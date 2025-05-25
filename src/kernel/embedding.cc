/* Copyright 2023-2025 CMU
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

#include "mirage/kernel/embedding.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

DTensor Graph::embedding(DTensor const &input, DTensor const &weight) {
  KNOperator *op = create_embedding_op(input, weight);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::embedding(DTensor const *input, DTensor const *weight) {
  DTensor output = embedding(*input, *weight);
  return &(output.owner_op->output_tensors[0]);
}

KNOperator *Graph::create_embedding_op(DTensor const &input,
                                       DTensor const &weight) {
  if (weight.num_dims != 2) {
    return nullptr;
  }

  if (input.data_type != mirage::type::DT_INT4 &&
      input.data_type != mirage::type::DT_INT8 &&
      input.data_type != mirage::type::DT_UINT16 &&
      input.data_type != mirage::type::DT_INT16) {
    return nullptr;
  }

  DTensor output = input;
  output.data_type = weight.data_type;
  output.dim[output.num_dims - 1] = weight.dim[1];

  if (!can_allocate(output)) {
    return nullptr;
  }

  KNEmbeddingOp *op = new KNEmbeddingOp(this, input, weight);
  return op;
}

KNEmbeddingOp::KNEmbeddingOp(Graph *_kgraph,
                             DTensor const &input,
                             DTensor const &weight)
    : mirage::kernel::KNOperator(
          _kgraph, mirage::type::KN_EMBEDDING_OP, input, weight) {
  assert(weight.num_dims == 2);
  DTensor output = input;
  output.data_type = weight.data_type;
  output.dim[output.num_dims - 1] = weight.dim[1];
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNEmbeddingOp::~KNEmbeddingOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNEmbeddingOp::operator json() const {
  return json{{"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

bool KNEmbeddingOp::profile(ProfileResult &result) {
  return false;
}

bool KNEmbeddingOp::fingerprint(void) {
  return false;
}

} // namespace kernel
} // namespace mirage
