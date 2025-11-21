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
#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/task_register.h"
#include "mirage/utils/hash_utils.h"

#include <algorithm>
#include <iostream>

namespace mirage {
namespace kernel {

Graph::Graph(dim3 _gpu_dim, bool _disable_fingerprint)
    : gpu_dim(_gpu_dim), disable_fingerprint(_disable_fingerprint) {
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

int Graph::get_input_dtensors(DTensor **inputs) const {
  int num_inputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == mirage::type::KN_INPUT_OP) {
      assert(op->output_tensors.size() == 1);
      inputs[num_inputs++] = &op->output_tensors[0];
    }
  }
  return num_inputs;
}

int Graph::get_num_input_dtensors() const {
  int num_inputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == mirage::type::KN_INPUT_OP) {
      num_inputs++;
    }
  }
  return num_inputs;
}

int Graph::get_num_output_dtensors() const {
  int num_outputs = 0;
  for (auto const &op : this->operators) {
    if (op->op_type == mirage::type::KN_OUTPUT_OP) {
      num_outputs++;
    }
  }
  return num_outputs;
}

int Graph::get_input_dtensor_shape_and_stride(DTensor const *input,
                                              int *strides,
                                              int *dims) const {
  for (auto const &op : this->operators) {
    if (op == input->owner_op) {
      assert(op->op_type == mirage::type::KN_INPUT_OP &&
             "input is not an KNInputOp");
      KNInputOp *input_op = static_cast<KNInputOp *>(op);
      int num_dims = (int)input_op->input_strides.size();
      for (int i = 0; i < num_dims; i++) {
        strides[i] = input_op->input_strides[i];
        dims[i] = input->dim[i];
      }
      return num_dims;
    }
  }
  assert(false && "Cannot find input dtensor");
  return 0;
}

bool Graph::can_allocate(DTensor const &tensor,
                         bool allocate_fingerprint) const {
  // We don't need to actually allocate device memory
  // when fingerprint is disabled (e.g., for very large muGraphs)
  if (disable_fingerprint) {
    return true;
  }

  size_t data_size = ((tensor.data_size() + 15) & ~15);
  if (dmem_data_offset + data_size > mirage::config::MAX_DMEM_SIZE) {
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
  // We don't need to actually allocate device memory
  // when fingerprint is disabled (e.g., for very large muGraphs)
  if (disable_fingerprint) {
    return true;
  }

  if (dmem_data_offset + data_size_in_bytes > mirage::config::MAX_DMEM_SIZE) {
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
        int num_dim, dim[mirage::config::MAX_TENSOR_DIMS];
        type::DataType data_type;
        layout::DmemLayout layout;
        std::vector<size_t> input_strides;
        size_t guidO;
        jop.at("output_tensors")[0].at("num_dims").get_to(num_dim);
        jop.at("output_tensors")[0].at("dim").get_to(dim);
        jop.at("input_strides").get_to(input_strides);
        jop.at("output_tensors")[0].at("data_type").get_to(data_type);
        jop.at("output_tensors")[0].at("layout").get_to(layout);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        std::vector<int> dims = to_vector(num_dim, dim);
        DTensor const &output =
            g.new_input(dims, input_strides, data_type, layout);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_OUTPUT_OP: {
        size_t guid;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        std::vector<size_t> output_strides;
        jop.at("output_strides").get_to(output_strides);
        g.mark_output(get_tensor_from_guid(guid), output_strides);
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
      case type::KNOperatorType::KN_SILU_OP:
      case type::KNOperatorType::KN_GELU_OP:
      case type::KNOperatorType::KN_RELU_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.elementunary(get_tensor_from_guid(guid), op_type);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_CLAMP_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.elementunary_clamp(get_tensor_from_guid(guid),
                                 type::CLAMP_MIN_MAX["min_val"],
                                 type::CLAMP_MIN_MAX["max_val"]);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_DIV_OP:
      case type::KNOperatorType::KN_ADD_OP:
      case type::KNOperatorType::KN_MUL_OP:
      case type::KNOperatorType::KN_POW_OP: {
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

size_t Graph::get_owner_independent_hash() const {
  size_t ret = 0;
  hash_combine(ret, gpu_dim);
  for (auto const &op : operators) {
    size_t h = op->get_owner_independent_hash();
    hash_combine(ret, h);
  }
  return ret;
}

// Persistent kernel functions
using namespace mirage::runtime;
void Graph::attach_torch_tensor(DTensor const *input,
                                void *torch_data_ptr,
                                char const *name) {
  io_config.emplace(
      input->guid,
      IODesc(IODesc::TorchTensor, std::string(name), *input, torch_data_ptr));
}

void Graph::attach_cuda_tensor(DTensor const *input, char const *name) {
  io_config.emplace(
      input->guid, IODesc(IODesc::CUDAMallocTensor, std::string(name), *input));
}

void Graph::attach_nvshmem_tensor(DTensor const *input, char const *name) {
  io_config.emplace(
      input->guid,
      IODesc(IODesc::NVSHMEMMallocTensor, std::string(name), *input));
}

DTensor *Graph::fuse_tensors(std::vector<DTensor const *> inputs,
                             int fused_dim,
                             int num_groups,
                             char const *name) {
  // Currently assert that we fuse along the 0-th dim (for weights)
  assert(fused_dim == 0);
  assert(inputs.size() > 0);
  std::vector<int> dims;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    dims.push_back(inputs[0]->dim[i]);
  }
  for (size_t t = 1; t < inputs.size(); t++) {
    dims[0] += inputs[t]->dim[0];
    assert(inputs[0]->num_dims == inputs[t]->num_dims);
    for (int i = 1; i < inputs[t]->num_dims; i++) {
      assert(dims[i] == inputs[t]->dim[i]);
    }
    assert(inputs[0]->data_type == inputs[t]->data_type);
  }
  std::vector<size_t> strides(dims.size(), 1);
  for (int i = inputs[0]->num_dims - 1; i >= 0; i--) {
    if (i == inputs[0]->num_dims - 1) {
      strides[i] = 1;
    } else {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  }
  DTensor *fused =
      new_input_ptr(dims, strides, inputs[0]->data_type, layout::DmemRowMajor);
  IODesc desc(IODesc::FusedTorchTensor, std::string(name), *fused);
  desc.num_groups = num_groups;
  for (size_t t = 0; t < inputs.size(); t++) {
    assert(io_config.find(inputs[t]->guid) != io_config.end());
    IODesc sub_desc = io_config.find(inputs[t]->guid)->second;
    desc.sub_descs.push_back(sub_desc);
    io_config.erase(inputs[t]->guid);
  }
  io_config.emplace(fused->guid, desc);
  return fused;
}

DTensor *Graph::shuffle_tensors(std::vector<DTensor const *> inputs,
                                int shuffled_dim,
                                int num_groups,
                                char const *name) {
  // Currently assert that we shuffle along the 0-th dim (for weights)
  assert(shuffled_dim == 0);
  assert(inputs.size() > 0);
  std::vector<int> dims;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    dims.push_back(inputs[0]->dim[i]);
  }
  for (size_t t = 1; t < inputs.size(); t++) {
    dims[0] += inputs[t]->dim[0];
    assert(inputs[0]->num_dims == inputs[t]->num_dims);
    for (int i = 1; i < inputs[t]->num_dims; i++) {
      assert(dims[i] == inputs[t]->dim[i]);
    }
    assert(inputs[0]->data_type == inputs[t]->data_type);
  }
  std::vector<size_t> strides(dims.size(), 1);
  for (int i = inputs[0]->num_dims - 1; i >= 0; i--) {
    if (i == inputs[0]->num_dims - 1) {
      strides[i] = 1;
    } else {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  }
  DTensor *shuffled =
      new_input_ptr(dims, strides, inputs[0]->data_type, layout::DmemRowMajor);
  IODesc desc(IODesc::ShuffledTorchTensor, std::string(name), *shuffled);
  desc.num_groups = num_groups;
  for (size_t t = 0; t < inputs.size(); t++) {
    assert(io_config.find(inputs[t]->guid) != io_config.end());
    IODesc sub_desc = io_config.find(inputs[t]->guid)->second;
    desc.sub_descs.push_back(sub_desc);
    io_config.erase(inputs[t]->guid);
  }
  io_config.emplace(shuffled->guid, desc);
  return shuffled;
}

void Graph::register_task(char const *task_type, std::vector<int> params) {
  std::string name = std::string(task_type);
  KNOperator const *op = operators.back();
  assert(op->op_type == type::KN_CUSTOMIZED_OP);
  KNCustomizedOp const *customized = static_cast<KNCustomizedOp const *>(op);
  TaskRegister *task_register = TaskRegister::get_instance();
  if (name == "embedding") {
    int variant_id =
        task_register->register_embedding_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING, variant_id);
  } else if (name == "rmsnorm") {
    int variant_id =
        task_register->register_rmsnorm_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM, variant_id);
  } else if (name == "rmsnorm_linear") {
    int variant_id =
        task_register->register_rmsnorm_linear_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(3, 1, TASK_RMS_NORM_LINEAR, variant_id);
  } else if (name == "attention") {
    int variant_id =
        task_register->register_attention_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(7, 1, TASK_ATTENTION_1, variant_id);
  } else if (name == "paged_attention") {
    int variant_id = task_register->register_paged_attention_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(7, 1, TASK_PAGED_ATTENTION_1, variant_id);
  } else if (name == "single_batch_extend_attention") {
    int variant_id = task_register->register_single_batch_extend_attention_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(7, 1, TASK_SINGLE_BATCH_EXTEND_ATTENTION, variant_id);
  } else if (name == "linear") {
    int variant_id = task_register->register_linear_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR, variant_id);
  } else if (name == "linear_with_residual") {
    int variant_id = task_register->register_linear_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] =
        std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL, variant_id);
  } else if (name == "silu_mul") {
    int variant_id =
        task_register->register_silu_mul_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL, variant_id);
  } else if (name == "identity") {
    int variant_id =
        task_register->register_identity_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_IDENTITY, variant_id);
  } else if (name == "silu_mul_linear_with_residual") {
    int variant_id = task_register->register_silu_mul_linear_with_residual_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(3, 1, TASK_SILU_MUL_LINEAR_WITH_RESIDUAL, variant_id);
  } else if (name == "argmax") {
    task_config[op] = std::make_tuple(1, 1, TASK_ARGMAX, 0);
  } else if (name == "argmax_partial") {
    int variant_id =
        task_register->register_argmax_partial_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 2, TASK_ARGMAX_PARTIAL, variant_id);
  } else if (name == "argmax_reduce") {
    int variant_id =
        task_register->register_argmax_reduce_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_ARGMAX_REDUCE, variant_id);
  } else if (name == "allreduce") {
    // `register_reduce_task` will register two tasks, but we only record one
    int variant_id =
        task_register->register_reduce_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_ALLREDUCE, variant_id);
  } else if (name == "find_ngram_partial") {
    int variant_id = task_register->register_find_ngram_partial_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(1, 1, TASK_FIND_NGRAM_PARTIAL, variant_id);
  } else if (name == "find_ngram_global") {
    int variant_id = task_register->register_find_ngram_global_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_FIND_NGRAM_GLOBAL, variant_id);
  } else if (name == "target_verify_greedy") {
    int variant_id = task_register->register_target_verify_greedy_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(2, 1, TASK_TARGET_VERIFY_GREEDY, variant_id);
  }
  // Hopper tasks
  else if (name == "linear_hopper") {
    int variant_id = task_register->register_linear_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_HOPPER, variant_id);
  } else if (name == "linear_with_residual_hopper") {
    int variant_id = task_register->register_linear_hopper_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] =
        std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL_HOPPER, variant_id);
  } else if (name == "paged_attention_hopper") {
    int variant_id = task_register->register_paged_attention_hopper_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(7, 1, TASK_PAGED_ATTENTION_HOPPER, variant_id);
  } else if (name == "rmsnorm_hopper") {
    int variant_id =
        task_register->register_rmsnorm_hopper_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM_HOPPER, variant_id);
  } else if (name == "linear_swapAB_hopper") {
    int variant_id = task_register->register_linear_swapAB_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_LINEAR_SWAPAB_HOPPER, variant_id);
  } else if (name == "linear_swapAB_with_residual_hopper") {
    int variant_id = task_register->register_linear_swapAB_hopper_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] = std::make_tuple(
        3, 1, TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER, variant_id);
  } else if (name == "linear_cutlass_hopper") {
    int variant_id = task_register->register_linear_cutlass_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_LINEAR_CUTLASS_HOPPER, variant_id);
  } else if (name == "linear_cutlass_with_residual_hopper") {
    int variant_id = task_register->register_linear_cutlass_hopper_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] = std::make_tuple(
        3, 1, TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER, variant_id);
  } else if (name == "silu_mul_hopper") {
    int variant_id = task_register->register_silu_mul_hopper_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL_HOPPER, variant_id);
  } else if (name == "embedding_hopper") {
    int variant_id = task_register->register_embedding_hopper_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING_HOPPER, variant_id);
  } else if (name == "moe_w13_linear_sm90") {
    int variant_id = task_register->register_moe_linear_sm90_task(
        customized->bgraph, params, true /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W13_LINEAR_SM90, variant_id);
  } else if (name == "moe_w2_linear_sm90") {
    int variant_id = task_register->register_moe_linear_sm90_task(
        customized->bgraph, params, false /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W2_LINEAR_SM90, variant_id);
  } else if (name == "splitk_linear_swapAB_hopper") {
    int variant_id = task_register->register_splitk_linear_swapAB_hopper_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_SPLITK_LINEAR_SWAPAB_HOPPER, variant_id);
  }
  // SM100 tasks
  else if (name == "linear_sm100") {
    int variant_id = task_register->register_linear_sm100_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] = std::make_tuple(2, 1, TASK_LINEAR_SM100, variant_id);
  } else if (name == "splitk_linear_sm100") {
    int variant_id = task_register->register_splitk_linear_sm100_task(
        customized->bgraph, params, false /*with_residual*/);
    task_config[op] =
        std::make_tuple(2, 1, TASK_SPLITK_LINEAR_SM100, variant_id);
  } else if (name == "linear_with_residual_sm100") {
    int variant_id = task_register->register_linear_sm100_task(
        customized->bgraph, params, true /*with_residual*/);
    task_config[op] =
        std::make_tuple(3, 1, TASK_LINEAR_WITH_RESIDUAL_SM100, variant_id);
  } else if (name == "paged_attention_sm100") {
    int variant_id = task_register->register_paged_attention_sm100_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(7, 1, TASK_ATTN_SM100, variant_id);
  } else if (name == "argmax_partial_sm100") {
    int variant_id = task_register->register_argmax_partial_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(1, 2, TASK_ARGMAX_PARTIAL_SM100, variant_id);
  } else if (name == "argmax_reduce_sm100") {
    int variant_id = task_register->register_argmax_reduce_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(2, 1, TASK_ARGMAX_REDUCE_SM100, variant_id);
  } else if (name == "tensor_init") {
    int variant_id =
        task_register->register_tensor_init_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_TENSOR_INIT, variant_id);
  } else if (name == "moe_topk_softmax_sm100") {
    int variant_id = task_register->register_moe_topk_softmax_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(1, 3, TASK_MOE_TOPK_SOFTMAX_SM100, variant_id);
  } else if (name == "moe_w13_linear_sm100") {
    int variant_id = task_register->register_moe_linear_sm100_task(
        customized->bgraph, params, true /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W13_LINEAR_SM100, variant_id);
  } else if (name == "moe_silu_mul") {
    int variant_id =
        task_register->register_moe_silu_mul_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(1, 1, TASK_SILU_MUL, variant_id);
  } else if (name == "moe_w2_linear_sm100") {
    int variant_id = task_register->register_moe_linear_sm100_task(
        customized->bgraph, params, false /*w13_linear*/);
    task_config[op] =
        std::make_tuple(4, 1, TASK_MOE_W2_LINEAR_SM100, variant_id);
  } else if (name == "moe_mul_sum_add_sm100") {
    int variant_id = task_register->register_moe_mul_sum_add_sm100_task(
        customized->bgraph, params);
    task_config[op] =
        std::make_tuple(3, 1, TASK_MOE_MUL_SUM_ADD_SM100, variant_id);
  } else {
    printf("Unsupported task name: %s\n", name);
    assert(false && "Unsupported task type");
  }
}

} // namespace kernel
} // namespace mirage
