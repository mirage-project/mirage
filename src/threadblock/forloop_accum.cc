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

#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::forloop_accum(STensor const &input,
                             mirage::type::TBOperatorType type) {
  TBOperator *op = create_forloop_accum_op(input, type);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::forloop_accum(STensor const *input,
                              mirage::type::TBOperatorType type) {
  TBOperator *op = create_forloop_accum_op(*input, type);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_forloop_accum_op(STensor const &input,
                                           mirage::type::TBOperatorType type) {
  // Input stensor must be before accumulation (i.e., inside forloop)
  if (input.after_accum) {
    return nullptr;
  }
  TBForloopAccumOp *op = new TBForloopAccumOp(this, input, type);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
  return op;
}

STensor Graph::forloop_accum_rescale(STensor const &input,
                                     STensor const &rescale,
                                     mirage::type::TBOperatorType type) {
  TBOperator *op = create_forloop_accum_rescale_op(input, rescale, type);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::forloop_accum_rescale(STensor const *input,
                                      STensor const *rescale,
                                      mirage::type::TBOperatorType type) {
  TBOperator *op = create_forloop_accum_rescale_op(*input, *rescale, type);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *
    Graph::create_forloop_accum_rescale_op(STensor const &input,
                                           STensor const &rescale,
                                           mirage::type::TBOperatorType type) {
  // Input stensor must be before accumulation (i.e., inside forloop)
  if (input.after_accum) {
    return nullptr;
  }
  TBForloopAccumOp *op = new TBForloopAccumOp(this, input, rescale, type);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
  return op;
}

STensor Graph::forloop_accum_max(STensor const &input) {
  TBOperator *op = create_forloop_accum_max_op(input);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::forloop_accum_max(STensor const *input) {
  TBOperator *op = create_forloop_accum_max_op(*input);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_forloop_accum_max_op(STensor const &input) {
  // Input stensor must be before accumulation (i.e., inside forloop)
  if (input.after_accum) {
    return nullptr;
  }
  TBForloopAccumOp *op =
      new TBForloopAccumOp(this, input, mirage::type::TB_FORLOOP_ACCUM_MAX_OP);
  // Check shmem usage
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  } else {
    return op;
  }
  return op;
}

TBForloopAccumOp::TBForloopAccumOp(Graph *_graph,
                                   STensor const &input,
                                   mirage::type::TBOperatorType type)
    : TBOperator(_graph, type, input) {
  assert(type >= mirage::type::TB_FORLOOP_ACCUM_FIRST_OP);
  assert(type < mirage::type::TB_FORLOOP_ACCUM_LAST_OP);
  assert(!input.after_accum);
  STensor output = input;
  switch (type) {
    case mirage::type::TB_FORLOOP_ACCUM_NO_RED_OP:
    case mirage::type::TB_FORLOOP_ACCUM_MAX_OP: {
      // Do nothing
      break;
    }
    case mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
    case mirage::type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case mirage::type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
      // Reduce the last dim to 1
      output.dim[output.num_dims - 1] = 1;
      break;
    }
    case mirage::type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      // Reduce the last dim to reduction_dimx
      output.dim[output.num_dims - 1] = bgraph->reduction_dimx;
      break;
    }
    default: {
      assert(false && "Unhandled forloop accum op");
    }
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.after_accum = true;
  output.smem_offset = bgraph->allocate_fingerprint(output);
  output_tensors.push_back(output);
}

TBForloopAccumOp::TBForloopAccumOp(Graph *_graph,
                                   STensor const &input,
                                   STensor const &rescale,
                                   mirage::type::TBOperatorType type)
    : TBOperator(_graph, type, input, rescale) {
  assert(type == mirage::type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
         type == mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP);
  assert(!input.after_accum);
  STensor output = input;
  switch (type) {
    case mirage::type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP: {
      // Do nothing
      break;
    }
    case mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP: {
      // Reduce the last dim to 1
      output.dim[output.num_dims - 1] = 1;
      break;
    }
    default: {
      assert(false && "Unhandled forloop accum rescale op");
    }
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.after_accum = true;
  output.smem_offset = bgraph->allocate_fingerprint(output);
  output_tensors.push_back(output);
}

TBForloopAccumOp::~TBForloopAccumOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBForloopAccumOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
