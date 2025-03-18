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

#include "mirage/threadblock/chunk.h"
#include "mirage/threadblock/graph.h"
#include <cassert>

namespace mirage {
namespace threadblock {

std::vector<STensor> Graph::chunk(STensor const &input, int chunk_size, int dim) {
    TBOperator *op = create_chunk_op(input, chunk_size, dim);
    assert(op != nullptr);
    operators.push_back(op);
    return op->output_tensors;
}

std::vector<STensor *> Graph::chunk(STensor const *input, int chunk_size, int dim) {
    TBOperator *op = create_chunk_op(*input, chunk_size, dim);
    assert(op != nullptr);
    operators.push_back(op);
    assert(op->output_tensors.size() > 0);
    std::vector<STensor *> res;
    for (auto t : op->output_tensors) {
        res.push_back(&t);
    }
    return res;
}

TBOperator *Graph::create_chunk_op(STensor const &input, int chunk_size, int dim) {
    TBOperator *op = new TBChunkOp(this, input, chunk_size, dim);
    size_t smem_usage = calculate_shared_memory_usage(op);
    if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
        delete op;
        return nullptr;
    } else {
        return op;
    }
}

TBChunkOp::TBChunkOp(Graph *bgraph, STensor const &input, int chunk_size, int dim)
 : TBOperator(bgraph, (type::TBOperatorType)( mirage::type::TB_CHUNK_0_OP + chunk_dim), input), chunk_size(chunk_size), chunk_dim(dim) {
    assert(input.dim[dim] % chunk_size == 0);
    
    for (size_t i = 0; i < chunk_size; i++) {
        STensor output_i = input;
        output_i.dim[dim] /= chunk_size;
        output_i.owner_op = this;
        output_i.owner_ts_idx = i;
        output_i.guid = STensor::next_guid++;
        output_i.smem_offset = bgraph->allocate_fingerprint(output_i);
        output_tensors.push_back(output_i);
    }
}

TBChunkOp::~TBChunkOp() {
    for (size_t i = 0; i < chunk_size; i++) {
        bgraph->free_fingerprint(output_tensors[i]);
    }
}

TBChunkOp::operator json() const {
    return {
        {"op_type", op_type},
        {"input_tensors", input_tensors},
        {"output_tensors", output_tensors},
        {"chunk_size", chunk_size},
        {"chunk_dim", chunk_dim},
    };
}

}
}