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

#include "mirage/kernel/rms_norm.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;

DTensor Graph::rms_norm(DTensor const &input,
                        std::vector<int> const &normalized_shape) {
  KNOperator *op = create_rms_norm_op(input, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::rms_norm(DTensor const *input,
                         std::vector<int> const &normalized_shape) {
  KNOperator *op = create_rms_norm_op(*input, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

DTensor Graph::rms_norm(DTensor const &input,
                        DTensor const &elementwise_affine,
                        std::vector<int> const &normalized_shape) {
  KNOperator *op =
      create_rms_norm_op(input, elementwise_affine, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::rms_norm(DTensor const *input,
                         DTensor const *elementwise_affine,
                         std::vector<int> const &normalized_shape) {
  KNOperator *op =
      create_rms_norm_op(*input, *elementwise_affine, normalized_shape);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *
    Graph::create_rms_norm_op(DTensor const &input,
                              std::vector<int> const &normalized_shape) {
  int num_norm_dims = normalized_shape.size();
  if (num_norm_dims > input.num_dims) {
    return nullptr;
  }
  for (int i = 0; i < num_norm_dims; i++) {
    if (normalized_shape[i] != input.dim[input.num_dims - num_norm_dims + i]) {
      return nullptr;
    }
  }
  if (!this->can_allocate(input)) {
    return nullptr;
  }

  KNRMSNormOp *op = new KNRMSNormOp(this, input, normalized_shape);
  return op;
}

KNOperator *
    Graph::create_rms_norm_op(DTensor const &input,
                              DTensor const &elementwise_affine,
                              std::vector<int> const &normalized_shape) {
  // We currently only allow normalizing the last dimension
  assert(false && "To be implemented");
}

KNRMSNormOp::KNRMSNormOp(Graph *_kgraph,
                         DTensor const &input,
                         std::vector<int> const &normalized_shape)
    : KNOperator(_kgraph, KN_RMS_NORM_OP, input) {
  normalized_size = 1;
  for (size_t i = 0; i < normalized_shape.size(); i++) {
    normalized_size *= normalized_shape[i];
  }
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNRMSNormOp::~KNRMSNormOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNRMSNormOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU
bool KNRMSNormOp::fingerprint(void) {
  int num_samples = output_tensors[0].num_elements() / normalized_size;
  kernel::DeviceMemoryManager *dmm =
      kernel::DeviceMemoryManager::get_instance();

  for (int device_id = 0; device_id < dmm->num_devices; ++device_id) {
    FPType *input_ptr = reinterpret_cast<FPType *>(dmm->fp_base_ptr[device_id] +
                                                   input_tensors[0].fp_offset);
    FPType *output_ptr = reinterpret_cast<FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);
    utils::compute_rms_norm_fingerprint(input_ptr,
                                        output_ptr,
                                        dmm->div_p_lookup_table,
                                        dmm->div_q_lookup_table,
                                        dmm->sqrt_p_lookup_table,
                                        dmm->sqrt_q_lookup_table,
                                        num_samples,
                                        normalized_size);
  }

  return true;
}
#endif

} // namespace kernel
} // namespace mirage
