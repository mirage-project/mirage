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

#include "mirage/transpiler/transpiler.h"

#include <stdexcept>
#include <unordered_set>

#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/matmul.h"
#include "mirage/threadblock/reduction.h"
#include "mirage/transpiler/memory_planner.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

// Find the last dimension with stride 1. Return -1 if not found.
static int find_innermost_dim(const size_t strides[], int num_dims) {
  for (int i = num_dims - 1; i >= 0; i--) {
    if (strides[i] == 1) {
      return i;
    }
  }
  return -1;
}

static int find_innermost_dim(std::vector<size_t> const &strides) {
  return find_innermost_dim(strides.data(), strides.size());
}

static size_t get_num_elems_in_16B(type::DataType datatype) {
  size_t elem_size = type::get_datatype_size(datatype);
  return std::max(16 / elem_size, 1ul);
}

// Resolve all metadata for DTensors
void Transpiler::resolve_dtensor_meta() {
  using guid_t = decltype(kn::DTensor::guid);

  // A list of unique (with distinct guid) DTensors
  std::unordered_set<guid_t> processed_guids;
  std::vector<kn::DTensor const *> all_dtensors;
  auto process_dtensor = [&](kn::DTensor const *dtensor) {
    if (processed_guids.count(dtensor->guid)) {
      return;
    }
    processed_guids.insert(dtensor->guid);
    all_dtensors.push_back(dtensor);
  };
  for (kn::KNOperator *const op : this->g->operators) {
    for (kn::DTensor const &dtensor : op->input_tensors) {
      process_dtensor(&dtensor);
    }
    for (kn::DTensor const &dtensor : op->output_tensors) {
      process_dtensor(&dtensor);
    }
  }

  // Get the max guid
  guid_t max_guid = std::numeric_limits<guid_t>::min();
  for (kn::DTensor const *dtensor : all_dtensors) {
    max_guid = std::max(max_guid, dtensor->guid);
  }

  // Enforce the stride of every input tensor
  int cur_input_idx = 0;
  for (kernel::KNOperator *const op : this->g->operators) {
    if (op->op_type == type::KN_INPUT_OP) {
      vector<size_t> const &cur_stride = this->input_strides[cur_input_idx];
      kernel::DTensor const &tensor = op->output_tensors.at(0);
      if (tensor.num_dims != (int)cur_stride.size()) {
        printf("The number of dimensions of the stride of the %dth tensor "
               "(%ld) does not match the tensor's num_dims (%d)\n",
               cur_input_idx,
               cur_stride.size(),
               tensor.num_dims);
        throw std::runtime_error("Stride size mismatch");
      }

      // Currently we only accept row-major layout for DTensors
      int innermost_dim = find_innermost_dim(cur_stride);
      assert(innermost_dim != -1 && "No innermost dim found for input tensor");
      assert(innermost_dim == (int)cur_stride.size() - 1 &&
             "Currently only row-major layout is supported for input tensors");

      guid_t guid = tensor.guid;
      for (int i = 0; i < tensor.num_dims; i++) {
        this->dtensor_metas[guid].strides[i] = cur_stride[i];
      }
      this->dtensor_metas[guid].is_input = true;
      this->dtensor_metas[guid].input_idx = cur_input_idx;

      cur_input_idx += 1;
    }
  }

  // Enforce the stride of the output tensor
  for (kn::DTensor const *dtensor : all_dtensors) {
    if (dtensor->guid == max_guid) {
      vector<size_t> const &cur_stride = this->output_stride;
      if (dtensor->num_dims != (int)cur_stride.size()) {
        printf("The number of dimensions of the stride of the output tensor "
               "(%ld) does not match the tensor's num_dims (%d)\n",
               cur_stride.size(),
               dtensor->num_dims);
        throw std::runtime_error("Stride size mismatch");
      }

      // Currently we only accept row-major layout for DTensors
      int innermost_dim = find_innermost_dim(cur_stride);
      assert(innermost_dim != -1 && "No innermost dim found for output tensor");
      assert(innermost_dim == (int)cur_stride.size() - 1 &&
             "Currently only row-major layout is supported for output tensors");

      guid_t guid = dtensor->guid;
      for (int i = 0; i < dtensor->num_dims; i++) {
        this->dtensor_metas[guid].strides[i] = cur_stride[i];
      }
      this->dtensor_metas[guid].is_output = true;
      break;
    }
  }

  // Calculate the stride & memory planning for intermediate tensors
  // Currently we let their layout to be row major, with strides padded to 16B
  MemoryPlanner memory_planner;
  for (kn::DTensor const *dtensor : all_dtensors) {
    guid_t guid = dtensor->guid;
    DTensorMeta &meta = this->dtensor_metas[guid];
    if (meta.is_input || meta.is_output) {
      continue;
    }
    // An intermediate tensor
    size_t cur_stride = 1;
    size_t num_elems_in_16B = get_num_elems_in_16B(dtensor->data_type);
    for (int i = dtensor->num_dims - 1; i >= 0; i--) {
      size_t cur_dim = (size_t)dtensor->dim[i];
      cur_dim = round_to_multiple(cur_dim, num_elems_in_16B);
      meta.strides[i] = cur_stride;
      cur_stride *= cur_dim;
    }
    size_t size = cur_stride * type::get_datatype_size(dtensor->data_type);
    meta.addr = memory_planner.allocate(size);
  }

  // Calculate the "perfectness" and "innermost dim" of all tensors
  for (kn::DTensor const *dtensor : all_dtensors) {
    assert(this->dtensor_metas.count(dtensor->guid));
    DTensorMeta &meta = this->dtensor_metas[dtensor->guid];
    size_t num_elems_in_16B = get_num_elems_in_16B(dtensor->data_type);
    bool is_perfect = true;
    for (int i = 0; i < dtensor->num_dims; i++) {
      if (meta.strides[i] % num_elems_in_16B != 0 && i != meta.innermost_dim) {
        is_perfect = false;
        break;
      }
    }
    meta.is_layout_perfect = is_perfect;
    meta.innermost_dim = find_innermost_dim(meta.strides, dtensor->num_dims);
  }
}

void Transpiler::resolve_stensor_meta() {
  // We enumerate tb graphs one by one
  using guid_t = decltype(tb::STensor::guid);
  for (kernel::KNOperator *const kn_op : this->g->operators) {
    if (kn_op->op_type != type::KN_CUSTOMIZED_OP) {
      continue;
    }
    kernel::KNCustomizedOp *const cur_kn_op =
        dynamic_cast<kernel::KNCustomizedOp *>(kn_op);
    tb::Graph *cur_tb_g = &(cur_kn_op->bgraph);

    // A list of unique (with distinct guid) STensors in cur_tb_g
    std::unordered_set<guid_t> processed_guids;
    std::vector<tb::STensor const *> all_stensors;
    auto process_stensor = [&](tb::STensor const *stensor) {
      if (processed_guids.count(stensor->guid)) {
        return;
      }
      processed_guids.insert(stensor->guid);
      all_stensors.push_back(stensor);
    };
    for (tb::TBOperator *const op : cur_tb_g->operators) {
      for (tb::STensor const &stensor : op->input_tensors) {
        process_stensor(&stensor);
      }
      for (tb::STensor const &stensor : op->output_tensors) {
        process_stensor(&stensor);
      }
    }

    // Now we are going to resolve all stensors in cur_tb_g
    std::unordered_map<guid_t, STensorMeta> cur_metas;

    // Calculate layouts
    // Refer to https://github.com/mirage-project/mirage/discussions/33
    // for more details about layout resolution

    // Currently, since all DTensors have a row major layout, the best strategy
    // should be, for each stensor, its innermost dim is the last dim, and we
    // swizzle the second last dim if needed
    // In the future, after adding support for non-row-major layout for global
    // tensors, we may use deep first search to find the best layout for
    // stensors
    // TODO(intlsy): Support non-row-major layout for global tensors

    // Swizzle some dimensions if needed
    auto swizzle_second_last_dim = [&](tb::STensor const &stensor) {
      int num_dims = stensor.num_dims;
      cur_metas[stensor.guid].swizzled_dims.push_back(num_dims - 2);
    };
    for (tb::TBOperator *const op : cur_tb_g->operators) {
      switch (op->op_type) {
        case type::TB_INPUT_OP:
        case type::TB_OUTPUT_OP:
        case type::TB_MATMUL_OP: {
          tb::TBMatmulOp *const cur_op = dynamic_cast<tb::TBMatmulOp *>(op);
          if (this->config.target_cc >= GPU_CC::T4) {
            // Leverage ldmatrix on T4+
            swizzle_second_last_dim(cur_op->input_tensors.at(0));
            swizzle_second_last_dim(cur_op->input_tensors.at(1));
          } else {
            // Use UniversalCopy<uint32_t>, still need to swizzle the second
            // last dim
            swizzle_second_last_dim(cur_op->input_tensors.at(0));
            swizzle_second_last_dim(cur_op->input_tensors.at(1));
          }
          if (this->config.target_cc >= GPU_CC::H100) {
            // Leverage stmatrix on H100+
            swizzle_second_last_dim(cur_op->output_tensors.at(0));
          } else {
            // Use UniversalCopy<uint32_t>, still need to swizzle the second
            // last dim
            swizzle_second_last_dim(cur_op->output_tensors.at(0));
          }
          break;
        }
        case type::TB_EXP_OP: {
          // No requirement. We can just iterate along the innermost dimension
          break;
        }
        case type::TB_ADD_OP:
        case type::TB_MUL_OP:
        case type::TB_DIV_OP: {
          // No requirement. NOTE(intlsy) After adding support for column-major
          // global tensors, we may need to deal with the case that two inputs
          // do not have the same layout
          break;
        }
        case type::TB_REDUCTION_0_OP:
        case type::TB_REDUCTION_1_OP:
        case type::TB_REDUCTION_2_OP:
        case type::TB_REDUCTION_0_TO_DIMX_OP:
        case type::TB_REDUCTION_1_TO_DIMX_OP:
        case type::TB_REDUCTION_2_TO_DIMX_OP: {
          int reduce_dim = dynamic_cast<tb::TBReductionOp *>(op)->reduce_dim;
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &output = op->output_tensors.at(0);
          if (reduce_dim != input.num_dims - 1) {
            // No requirement
          } else {
            // Swizzle the second last dim
            swizzle_second_last_dim(input);
            swizzle_second_last_dim(output);
          }
          break;
        }
        case type::TB_CONCAT_0_OP:
        case type::TB_CONCAT_1_OP:
        case type::TB_CONCAT_2_OP: {
          // No requirement
          break;
        }
        case type::TB_CONCAT_THEN_MATMUL_OP: {
          assert(false && "Not supported now");
          break;
        }
        default:
          assert(
              false &&
              ("TB op not supported: " + std::to_string(op->op_type)).c_str());
      }
    }

    // Memory planning
    MemoryPlanner memory_planner;
    for (tb::STensor const *stensor : all_stensors) {
      size_t size = stensor->size();
      cur_metas[stensor->guid].addr = memory_planner.allocate(size);
    }

    // Calculate strides
    for (tb::STensor const *stensor_ptr : all_stensors) {
      STensorMeta &meta = cur_metas[stensor_ptr->guid];
      tb::STensor const &stensor = *stensor_ptr;
      size_t cur_stride = 1;
      for (int i = stensor.num_dims - 1; i >= 0; --i) {
        size_t cur_dim = (size_t)stensor.dim[i];
        cur_dim = round_to_multiple(cur_dim, 8ul);
        meta.strides[i] = cur_stride;
        cur_stride *= cur_dim;
      }
      meta.innermost_dim = find_innermost_dim(meta.strides, stensor.num_dims);
    }

    // Merge cur_metas into this->stensor_metas
    for (auto const &[guid, meta] : cur_metas) {
      assert(!this->stensor_metas.count(guid));
      this->stensor_metas[guid] = meta;
    }
  }
}

} // namespace transpiler
} // namespace mirage
