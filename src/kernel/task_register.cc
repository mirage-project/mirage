/* Copyright 2023-2025 CMU
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

#include "mirage/kernel/task_register.h"
#include "mirage/kernel/operator.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

TaskRegister *TaskRegister::singleton = nullptr;

TaskRegister::TaskRegister() {}

TaskRegister *TaskRegister::get_instance() {
  if (singleton == nullptr) {
    singleton = new TaskRegister();
  }
  return singleton;
}

int TaskRegister::register_task_variant(runtime::TaskType type,
                                        std::string const &code) {
  std::vector<std::string> &variants = all_task_variants[type];
  for (size_t i = 0; i < variants.size(); i++) {
    if (variants[i] == code) {
      return (int)(i);
    }
  }
  // Add a new variant
  variants.push_back(code);
  return (int)(variants.size() - 1);
}

int TaskRegister::register_embedding_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  assert(params.size() == 1);
  // params[0]: input source (0: tokens, 1: input_token)
  int batch_size = 0, output_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::embedding_kernel<bfloat16, $, $, $>(",
         batch_size,
         output_size,
         output_stride);
  if (params[0] == 0) {
    code.e("    runtime_config.tokens + runtime_config.step[0], ");
  } else if (params[0] == 1) {
    code.e("    task_desc.inputs[0].base_ptr,");
  }
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr);");
  return register_task_variant(TASK_EMBEDDING, code.to_string());
}

int TaskRegister::register_rmsnorm_linear_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::norm_linear_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.inputs[2].base_ptr,");
  code.e("    1e-6f,");
  code.e("    task_desc.outputs[0].base_ptr);");
  return register_task_variant(TASK_RMS_NORM_LINEAR, code.to_string());
}

int TaskRegister::register_attention_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  assert(params.size() == 4);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::single_batch_decoding_kernel<bfloat16, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         head_dim,
         kv_stride);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.inputs[2].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc.inputs[3].base_ptr,");
  code.e("    task_desc.inputs[4].base_ptr,");
  code.e("    task_desc.inputs[5].base_ptr,");
  code.e("    task_desc.inputs[6].base_ptr,");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTENTION_1, code.to_string());
}

int TaskRegister::register_single_batch_extend_attention_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: extend_num
  // params[5]: output_stride
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int extend_num = params[4];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int output_stride = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::single_batch_extend_kernel<bfloat16, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         head_dim,
         kv_stride,
         output_stride,
         extend_num);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.inputs[2].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc.inputs[3].base_ptr,");
  code.e("    task_desc.inputs[4].base_ptr,");
  code.e("    task_desc.inputs[5].base_ptr,");
  code.e("    task_desc.inputs[6].base_ptr,");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_SINGLE_BATCH_EXTEND_ATTENTION,
                               code.to_string());
}

int TaskRegister::register_silu_mul_linear_with_residual_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1] / 2;
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_linear_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.inputs[2].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.my_gpu_id == 0);");
  return register_task_variant(TASK_SILU_MUL_LINEAR_WITH_RESIDUAL,
                               code.to_string());
}

int TaskRegister::register_linear_with_residual_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::linear_kernel<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.inputs[2].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.my_gpu_id == 0);");
  return register_task_variant(TASK_LINEAR_WITH_RESIDUAL, code.to_string());
}

int TaskRegister::register_argmax_partial_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_elements = input_ops[0]->output_tensors[0].dim[1];
  int num_partial_tasks = output_ops[0]->output_tensors[0].dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    task_desc.outputs[1].base_ptr);");
  return register_task_variant(TASK_ARGMAX_PARTIAL, code.to_string());
}

int TaskRegister::register_argmax_reduce_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params) {
  // params[0]: output size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_kernel<bfloat16, $, $>(", params[0], num_parts);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.step[0],");
  code.e("    runtime_config.tokens);");
  return register_task_variant(TASK_ARGMAX_REDUCE, code.to_string());
}

int TaskRegister::register_find_ngram_partial_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: ngram size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = output_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::find_ngram_partial_kernel<$, $>(", params[0], num_parts);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.step[0] + 1);");

  return register_task_variant(TASK_FIND_NGRAM_PARTIAL, code.to_string());
}

int TaskRegister::register_find_ngram_global_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: ngram size
  // params[1]: spec length
  assert(params.size() == 2);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::find_ngram_global_kernel<$, $, $>(",
         params[0],
         params[1],
         num_parts);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    task_desc.outputs[0].base_ptr,");
  code.e("    runtime_config.step[0]);");
  return register_task_variant(TASK_FIND_NGRAM_GLOBAL, code.to_string());
}

int TaskRegister::register_target_verify_greedy_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_spec_tokens = input_ops[0]->output_tensors[0].dim[1] - 1;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::target_verify_greedy_kernel<$>(", num_spec_tokens);
  code.e("    task_desc.inputs[0].base_ptr,");
  code.e("    task_desc.inputs[1].base_ptr,");
  code.e("    (void*)(runtime_config.new_token_nums),"); // int pointer
  code.e("    (void*)(runtime_config.tokens + runtime_config.step[0] + 1));");
  return register_task_variant(TASK_TARGET_VERIFY_GREEDY, code.to_string());
}

} // namespace runtime
} // namespace mirage
