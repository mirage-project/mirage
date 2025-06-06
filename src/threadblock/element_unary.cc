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

#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"

namespace mirage {
namespace threadblock {

STensor Graph::exp(STensor const &input) {
  return elementunary(input, mirage::type::TB_EXP_OP);
}

STensor *Graph::exp(STensor const *input) {
  return elementunary(input, mirage::type::TB_EXP_OP);
}

STensor Graph::square(STensor const &input) {
  return elementunary(input, mirage::type::TB_SQUARE_OP);
}

STensor *Graph::square(STensor const *input) {
  return elementunary(input, mirage::type::TB_SQUARE_OP);
}

STensor Graph::sqrt(STensor const &input) {
  return elementunary(input, mirage::type::TB_SQRT_OP);
}

STensor *Graph::sqrt(STensor const *input) {
  return elementunary(input, mirage::type::TB_SQRT_OP);
}

STensor Graph::silu(STensor const &input) {
  return elementunary(input, mirage::type::TB_SILU_OP);
}

STensor *Graph::silu(STensor const *input) {
  return elementunary(input, mirage::type::TB_SILU_OP);
}

STensor Graph::gelu(STensor const &input) {
  return elementunary(input, mirage::type::TB_GELU_OP);
}

STensor *Graph::gelu(STensor const *input) {
  return elementunary(input, mirage::type::TB_GELU_OP);
}

STensor Graph::relu(STensor const &input) {
  return elementunary(input, mirage::type::TB_RELU_OP);
}

STensor *Graph::relu(STensor const *input) {
  return elementunary(input, mirage::type::TB_RELU_OP);
}

STensor Graph::clamp(STensor const &input,
                     float const &min_val,
                     float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

STensor *Graph::clamp(STensor const *input,
                      float const &min_val,
                      float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

STensor Graph::elementunary_clamp(STensor const &input,
                                  float const &min_val,
                                  float const &max_val) {
  TBOperator *op = create_elementunary_clamp_op(input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::elementunary_clamp(STensor const *input,
                                   float const &min_val,
                                   float const &max_val) {
  TBOperator *op = create_elementunary_clamp_op(*input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_elementunary_clamp_op(STensor const &input,
                                                float const &min_val,
                                                float const &max_val) {
  TBElementUnaryOp *op = new TBClampUnaryOp(this, input, min_val, max_val);
  return op;
}

STensor Graph::mul_scalar(STensor const &input, float const &scalar) {
  return elementunary(input, mirage::type::TB_MUL_SCALAR_OP, scalar);
}

STensor *Graph::mul_scalar(STensor const *input, float const &scalar) {
  return elementunary(input, mirage::type::TB_MUL_SCALAR_OP, scalar);
}

STensor Graph::elementunary(STensor const &input,
                            mirage::type::TBOperatorType type,
                            float const &scalar) {
  TBOperator *op = create_elementunary_op(input, type, scalar);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

STensor *Graph::elementunary(STensor const *input,
                             mirage::type::TBOperatorType type,
                             float const &scalar) {
  TBOperator *op = create_elementunary_op(*input, type, scalar);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

TBOperator *Graph::create_elementunary_op(STensor const &input,
                                          mirage::type::TBOperatorType _type,
                                          float const &scalar) {
  TBElementUnaryOp *op = new TBElementUnaryOp(this, input, _type, scalar);
  return op;
}

TBClampUnaryOp::TBClampUnaryOp(Graph *_graph,
                               STensor const &input,
                               float const &min_val,
                               float const &max_val)
    : TBElementUnaryOp(_graph, input, mirage::type::TB_CLAMP_OP, 0.0f),
      min_val(min_val), max_val(max_val) {}

TBElementUnaryOp::TBElementUnaryOp(Graph *_graph,
                                   STensor const &input,
                                   mirage::type::TBOperatorType _type,
                                   float const &scalar)
    : TBOperator(_graph, _type, input), scalar(scalar) {
  STensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = STensor::next_guid++;
  output.after_accum = input.after_accum;
  // Note that we inplace the output by default
  // output.smem_offset = bgraph->allocate(output);
  output.smem_offset = input.smem_offset;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

TBElementUnaryOp::~TBElementUnaryOp() {
  // Don't free since we inplace the output by default
  // bgraph->free(output_tensors);
}

TBElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace threadblock
} // namespace mirage
