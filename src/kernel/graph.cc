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

#include "mirage/kernel/graph.h"
#include "mirage/config.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/utils/hash_utils.h"

#include <algorithm>
#include <iostream>

namespace mirage {
namespace kernel {

Graph::Graph(dim3 _gpu_dim) : gpu_dim(_gpu_dim) {
  dmem_data_offset = 0;
  dmem_fp_offset = 0;
}

Graph::~Graph() {
  while (!operators.empty()) {
    delete operators.back();
    operators.pop_back();
  }
}

size_t Graph::pair_hash::operator()(std::pair<int, int> const &p) const {
  size_t h1 = std::hash<int>{}(p.first);
  size_t h2 = std::hash<int>{}(p.second);
  hash_combine(h1, h2);
  return h1;
}

int Graph::get_input_dtensors(DTensor **inputs) {
  int num_inputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == mirage::type::KN_INPUT_OP) {
      assert(op->output_tensors.size() == 1);
      inputs[num_inputs++] = &op->output_tensors[0];
    }
  }
  return num_inputs;
}

int Graph::get_input_dtensor_layout(DTensor const *input, int *strides) {
  for (auto const &op : this->operators) {
    if (op == input->owner_op) {
      assert(op->op_type == mirage::type::KN_INPUT_OP && "input is not an KNInputOp");
      KNInputOp * input_op = static_cast<KNInputOp*>(op);
      int num_dims = (int) input_op->input_strides.size();
      for (int i = 0; i < num_dims; i++) {
        strides[i] = input_op->input_strides[i];
      }
      return num_dims;
    }
  }
  assert(false && "Cannot find input dtensor");
  return 0;
}

bool Graph::can_allocate(DTensor const &tensor,
                         bool allocate_fingerprint) const {
  size_t data_size = ((tensor.data_size() + 15) & ~15);
  if (dmem_data_offset + data_size > mirage::config::MAX_DMEM_DATA_SIZE) {
    return false;
  }
  if (allocate_fingerprint) {
    size_t fp_size = ((tensor.fingerprint_size() + 15) & ~15);
    if (dmem_fp_offset + fp_size > mirage::config::MAX_DMEM_FP_SIZE) {
      return false;
    }
  }
  return true;
}

bool Graph::can_allocate(size_t data_size_in_bytes,
                         size_t fp_size_in_bytes) const {
  if (dmem_data_offset + data_size_in_bytes >
      mirage::config::MAX_DMEM_DATA_SIZE) {
    return false;
  }
  if (dmem_fp_offset + fp_size_in_bytes > mirage::config::MAX_DMEM_FP_SIZE) {
    return false;
  }
  return true;
}

bool Graph::allocate(DTensor &tensor, bool allocate_fingerprint) {
  // assert that the start of the tensor is 16 bytes aligned
  assert(dmem_data_offset % 16 == 0);
  off_t ret = dmem_data_offset;

  size_t aligns_size = ((tensor.data_size() + 15) & ~15);
  dmem_data_offset += aligns_size;

  allocated_data_tensors.push_back(std::make_pair(ret, aligns_size));
  tensor.data_offset = ret;

  if (allocate_fingerprint) {
    assert(dmem_fp_offset % 16 == 0);
    ret = dmem_fp_offset;
    aligns_size = ((tensor.fingerprint_size() + 15) & ~15);
    dmem_fp_offset += aligns_size;
    tensor.fp_offset = ret;
    allocated_fp_tensors.push_back(std::make_pair(ret, aligns_size));
  }

  // Assert that we haven't used more than what we pre-allocated
  // assert(dmem_offset <= dmm->total_size);
  return true;
}

void Graph::free(DTensor &tensor) {
  // Currently assume that tensors are freed in the reverse order
  // so ptr must be the last tensor we have created
  // Note that a non-negative fp_offset means that we have
  // allocated memory for its fingerprint

  if (tensor.fp_offset >= 0) {
    assert(allocated_fp_tensors.size() > 0);
    assert(allocated_fp_tensors.back().first == tensor.fp_offset);
    assert(allocated_fp_tensors.back().second ==
           ((tensor.fingerprint_size() + 15) & ~15));
    dmem_fp_offset -= allocated_fp_tensors.back().second;
    allocated_fp_tensors.pop_back();
    tensor.fp_offset = -1;
  }
  assert(allocated_data_tensors.size() > 0);
  assert(allocated_data_tensors.back().first == tensor.data_offset);
  assert(allocated_data_tensors.back().second ==
         ((tensor.data_size() + 15) & ~15));
  dmem_data_offset -= allocated_data_tensors.back().second;
  allocated_data_tensors.pop_back();
  tensor.data_offset = -1;
}

void to_json(json &j, Graph const &g) {
  for (KNOperator *const op : g.operators) {
    j.push_back(json(*op));
  }
}

void from_json(json const &j, Graph &g) {
  std::unordered_map<size_t, size_t>
      guid_mapping; // from deseralized guid to json guid

  auto get_tensor_from_guid = [&](size_t guid) {
    for (auto const &op : g.operators) {
      for (DTensor const &dtensor : op->output_tensors) {
        if (guid_mapping.at(dtensor.guid) == guid) {
          return dtensor;
        }
      }
    }
    assert(false);
  };

  for (json const &jop : j) {
    type::KNOperatorType op_type;
    jop.at("op_type").get_to(op_type);
    switch (op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        int num_dim, dim[MAX_TENSOR_DIMS];
        type::DataType data_type;
        layout::DmemLayout layout;
        size_t guidO;
        jop.at("output_tensors")[0].at("num_dims").get_to(num_dim);
        jop.at("output_tensors")[0].at("dim").get_to(dim);
        jop.at("output_tensors")[0].at("data_type").get_to(data_type);
        jop.at("output_tensors")[0].at("layout").get_to(layout);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        std::vector<int> dims = to_vector(num_dim, dim);
        // FIXME: the input strides should be obtained from the json file
        // Currently we assume the default strided layout
        std::vector<size_t> strides;
        int num_elements = 1;
        for (size_t i = 0; i < dims.size(); i++) {
          strides.push_back(num_elements);
          num_elements *= dims[dims.size() - 1 - i];
        }
        std::reverse(strides.begin(), strides.end());
        DTensor const &output = g.new_input(dims, strides, data_type, layout);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_MATMUL_OP: {
        size_t guidA, guidB, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("input_tensors")[1].at("guid").get_to(guidB);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.matmul(get_tensor_from_guid(guidA), get_tensor_from_guid(guidB));
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_EXP_OP:
      case type::KNOperatorType::KN_SQUARE_OP:
      case type::KNOperatorType::KN_SQRT_OP:
      case type::KNOperatorType::KN_SILU_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.elementunary(get_tensor_from_guid(guid), op_type);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_DIV_OP:
      case type::KNOperatorType::KN_ADD_OP:
      case type::KNOperatorType::KN_MUL_OP: {
        size_t guidA, guidB, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("input_tensors")[1].at("guid").get_to(guidB);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output = g.elementbinary(
            get_tensor_from_guid(guidA), get_tensor_from_guid(guidB), op_type);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_REDUCTION_0_OP:
      case type::KNOperatorType::KN_REDUCTION_1_OP:
      case type::KNOperatorType::KN_REDUCTION_2_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.reduction(get_tensor_from_guid(guid),
                        op_type - type::KNOperatorType::KN_REDUCTION_0_OP);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        std::vector<DTensor> inputs;
        for (auto const &jinput : jop.at("input_tensors")) {
          size_t guid;
          jinput.at("guid").get_to(guid);
          inputs.push_back(get_tensor_from_guid(guid));
        }
        threadblock::Graph bgraph;
        from_json(jop.at("bgraph"), bgraph);
        for (size_t i = 0; i < bgraph.operators.size(); ++i) {
          if (bgraph.operators[i]->op_type == type::TB_INPUT_OP) {
            static_cast<threadblock::TBInputOp *>(bgraph.operators[i])
                ->dtensor = inputs[i];
          }
        }
        std::vector<DTensor> outputs = g.customized(inputs, bgraph);
        for (size_t i = 0; i < outputs.size(); ++i) {
          size_t guidO;
          jop.at("output_tensors")[i].at("guid").get_to(guidO);
          guid_mapping[outputs[i].guid] = guidO;
        }

        break;
      }
      default:
        assert(false && "Cannot deserialize this operator");
    }
  }
}

} // namespace kernel
} // namespace mirage
