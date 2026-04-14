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

#include "mirage/kernel/reduction.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;

DTensor Graph::reduction(DTensor const &input, int dim, int size) {
  KNOperator *op = create_reduction_op(input, dim, size);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::reduction(DTensor const *input, int dim, int size) {
  KNOperator *op = create_reduction_op(*input, dim, size);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *
    Graph::create_reduction_op(DTensor const &input, int dim, int size) {
  if (input.num_dims <= dim) {
    return nullptr;
  }
  if (input.dim[dim] % size != 0) {
    return nullptr;
  }
  if (!this->can_allocate(input)) {
    return nullptr;
  }

  KNReductionOp *op = new KNReductionOp(this, input, dim, size);
  return op;
}

KNReductionOp::KNReductionOp(Graph *_kgraph,
                             DTensor const &input,
                             int dim,
                             int size)
    : KNOperator(_kgraph, (KNOperatorType)(KN_REDUCTION_0_OP + dim), input),
      reduction_dim_idx(dim), reduction_dim_size(size) {
  DTensor output = input;
  assert(dim < output.num_dims);
  assert(output.dim[dim] % size == 0);
  output.dim[dim] = size;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNReductionOp::~KNReductionOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNReductionOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU
bool KNReductionOp::fingerprint(void) {
  kernel::DeviceMemoryManager *dmm =
      kernel::DeviceMemoryManager::get_instance();

  int reduction_degree =
      input_tensors[0].dim[reduction_dim_idx] / output_tensors[0].dim[reduction_dim_idx];
  int inner_range = 1;
  for (int i = reduction_dim_idx + 1; i < output_tensors[0].num_dims; ++i) {
    inner_range *= output_tensors[0].dim[i];
  }

  for (int device_id = 0; device_id < dmm->num_devices; ++device_id) {
    type::FPType *input_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[0].fp_offset);
    type::FPType *output_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);

    for (int i = 0; i < output_tensors[0].num_elements(); ++i) {
      int pos = (i / inner_range) * (inner_range * reduction_degree) +
                i % inner_range;
      FPType result = 0;
      for (int k = 0; k < reduction_degree; ++k) {
        result = utils::compute_add_fingerprint(result, input_ptr[pos]);
        pos += inner_range;
      }
      output_ptr[i] = result;
    }
  }

  return true;
}
#endif

} // namespace kernel
} // namespace mirage
