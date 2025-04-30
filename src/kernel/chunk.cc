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

#include "mirage/kernel/chunk.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

std::vector<DTensor>
    Graph::chunk(DTensor const &input, int chunk_size, int dim) {
  KNOperator *op = create_chunk_op(input, chunk_size, dim);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() > 0);
  return op->output_tensors;
}

int Graph::chunk(DTensor const *input, int chunk_size, int dim) {
  return (int)chunk(*input, chunk_size, dim).size();
}

KNOperator *
    Graph::create_chunk_op(DTensor const &input, int chunk_size, int dim) {
  if (dim < 0 || dim >= input.num_dims || chunk_size <= 0) {
    return nullptr;
  }
  if (input.dim[dim] % chunk_size != 0) {
    return nullptr;
  }
  if (!this->can_allocate(input)) {
    return nullptr;
  }

  KNChunkOp *op = new KNChunkOp(this, input, chunk_size, dim);
  return op;
}

KNChunkOp::KNChunkOp(Graph *_graph,
                     DTensor const &input,
                     int chunk_size,
                     int dim)
    : KNOperator(
          _graph, (type::KNOperatorType)(type::KN_CHUNK_0_OP + dim), input),
      chunk_size(chunk_size), chunk_dim(dim) {
  assert(input.dim[dim] % chunk_size == 0);

  for (size_t i = 0; i < chunk_size; ++i) {
    DTensor output_i = input;
    output_i.dim[dim] /= chunk_size;
    output_i.owner_op = this;
    output_i.owner_ts_idx = i;
    output_i.guid = DTensor::next_guid++;
    kgraph->allocate(output_i);
    output_tensors.push_back(output_i);
  }
}

KNChunkOp::~KNChunkOp() {
  for (auto &output : output_tensors) {
    kgraph->free(output);
  }
}

KNChunkOp::operator json() const {
  return {
      {"op_type", op_type},
      {"input_tensors", input_tensors},
      {"output_tensors", output_tensors},
      {"chunk_size", chunk_size},
      {"chunk_dim", chunk_dim},
  };
}

void from_json(json const &j, KNChunkOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  j.at("chunk_size").get_to(op.chunk_size);
  j.at("chunk_dim").get_to(op.chunk_dim);
}

} // namespace kernel
} // namespace mirage
