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

#include "mirage/kernel/element_unary.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

DTensor Graph::exp(DTensor const &input) {
  return elementunary(input, mirage::type::KN_EXP_OP);
}

DTensor *Graph::exp(DTensor const *input) {
  return elementunary(input, mirage::type::KN_EXP_OP);
}

DTensor Graph::square(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SQUARE_OP);
}

DTensor *Graph::square(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SQUARE_OP);
}

DTensor Graph::sqrt(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SQRT_OP);
}

DTensor *Graph::sqrt(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SQRT_OP);
}

DTensor Graph::silu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_SILU_OP);
}

DTensor *Graph::silu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_SILU_OP);
}

DTensor Graph::gelu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_GELU_OP);
}

DTensor *Graph::gelu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_GELU_OP);
}

DTensor Graph::relu(DTensor const &input) {
  return elementunary(input, mirage::type::KN_RELU_OP);
}

DTensor *Graph::relu(DTensor const *input) {
  return elementunary(input, mirage::type::KN_RELU_OP);
}

DTensor Graph::clamp(DTensor const &input,
                     float const &min_val,
                     float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor *Graph::clamp(DTensor const *input,
                      float const &min_val,
                      float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor Graph::elementunary_clamp(DTensor const &input,
                                  float const &min_val,
                                  float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary_clamp(DTensor const *input,
                                   float const &min_val,
                                   float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(*input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_clamp_op(DTensor const &input,
                                                float const &min_val,
                                                float const &max_val) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNClampUnaryOp(this, input, min_val, max_val);

  return op;
}

DTensor Graph::elementunary(DTensor const &input,
                            mirage::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary(DTensor const *input,
                             mirage::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(*input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_op(DTensor const &input,
                                          mirage::type::KNOperatorType type) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNElementUnaryOp(this, input, type);

  return op;
}

KNClampUnaryOp::KNClampUnaryOp(Graph *_kgraph,
                               DTensor const &input,
                               float min_val,
                               float max_val)
    : KNElementUnaryOp(_kgraph, input, mirage::type::KN_CLAMP_OP),
      min_val(min_val), max_val(max_val) {}

KNElementUnaryOp::KNElementUnaryOp(Graph *_kgraph,
                                   DTensor const &input,
                                   mirage::type::KNOperatorType type)
    : mirage::kernel::KNOperator(_kgraph, type, input) {
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementUnaryOp::~KNElementUnaryOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNElementUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU
bool KNElementUnaryOp::fingerprint(void) {
  kernel::DeviceMemoryManager *dmm =
      kernel::DeviceMemoryManager::get_instance();

  for (int device_id = 0; device_id < dmm->num_devices; ++device_id) {
    type::FPType *input_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[0].fp_offset);
    type::FPType *output_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);

    for (size_t i = 0; i < output_tensors[0].num_elements(); ++i) {
      switch (op_type) {
        case mirage::type::KN_EXP_OP:
          output_ptr[i] =
              utils::compute_exp_fingerprint(input_ptr[i], dmm->exp_lookup_table);
          break;
        case mirage::type::KN_SQUARE_OP:
          output_ptr[i] = utils::compute_square_fingerprint(input_ptr[i]);
          break;
        case mirage::type::KN_SQRT_OP:
          output_ptr[i] = utils::compute_sqrt_fingerprint(
              input_ptr[i],
              dmm->sqrt_p_lookup_table,
              dmm->sqrt_q_lookup_table);
          break;
        case mirage::type::KN_SILU_OP:
          output_ptr[i] =
              utils::compute_silu_fingerprint(input_ptr[i], dmm->exp_lookup_table);
          break;
        case mirage::type::KN_GELU_OP:
          output_ptr[i] =
              utils::compute_gelu_fingerprint(input_ptr[i], dmm->exp_lookup_table);
          break;
        case mirage::type::KN_RELU_OP:
          output_ptr[i] = utils::compute_relu_fingerprint(input_ptr[i]);
          break;
        case mirage::type::KN_CLAMP_OP:
          output_ptr[i] = utils::compute_clamp_fingerprint(input_ptr[i]);
          break;
        default:
          assert(false && "Unsupported kernel unary op for fingerprinting");
      }
    }
  }

  return true;
}
#endif

} // namespace kernel
} // namespace mirage
