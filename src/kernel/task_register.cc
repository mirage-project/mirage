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
    code.e("    task_desc->input_ptrs[0],");
  }
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_EMBEDDING, code.to_string());
}

int TaskRegister::register_rmsnorm_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params) {
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
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim = output_ops[0]->output_tensors[0].dim[1];
  // Currently assume that each rmsnorm task processes one token
  assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::rms_norm_impl<bfloat16, $, $>(", batch_size, hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-6f);");
  return register_task_variant(TASK_RMS_NORM, code.to_string());
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0]);");
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTENTION_1, code.to_string());
}

int TaskRegister::register_paged_attention_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
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
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_task_impl<bfloat16, $, $, $, $, "
         "$, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_PAGED_ATTENTION_1, code.to_string());
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_SINGLE_BATCH_EXTEND_ATTENTION,
                               code.to_string());
}

int TaskRegister::register_silu_mul_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, input_stride, output_stride;
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
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SILU_MUL, code.to_string());
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.my_gpu_id == 0);");
  return register_task_variant(TASK_SILU_MUL_LINEAR_WITH_RESIDUAL,
                               code.to_string());
}

int TaskRegister::register_linear_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  if (with_residual) {
    code.e("    task_desc->input_ptrs[2],");
  } else {
    code.e("    nullptr,");
  }
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  if (with_residual) {
    code.e("    runtime_config.my_gpu_id == 0);");
  } else {
    code.e("    false/*residual*/);");
  }
  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL, code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR, code.to_string());
  }
}

int TaskRegister::register_argmax_partial_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: num_partial_tasks
  assert(params.size() == 1);
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
  int num_partial_tasks = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
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
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_kernel<bfloat16, $, $, $>(",
         batch_size,
         params[0],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
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
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    (void*)(runtime_config.new_token_nums),"); // int pointer
  code.e("    (void*)(runtime_config.tokens + runtime_config.step[0] + 1));");
  return register_task_variant(TASK_TARGET_VERIFY_GREEDY, code.to_string());
}

int TaskRegister::register_linear_hopper_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params,
                                              bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
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
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 128;
  int const Kstages = output_size >= 256 ? 3 : 6;
  int const SMEM_M_SIZE = batch_size;
  // int const SMEM_M_SIZE = 64;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = (output_size >= 256)   ? 256
                               : (output_size >= 128) ? 128
                               : (output_size >= 64)  ? 64
                               : (output_size >= 32)  ? 32
                                                      : 16;
  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        B,
        M,
        S,
        batch_size,         /*GMEM_ROW_*/
        output_size,        /*GMEM_COL_*/
        batch_size,         /*SMEM_ROW_*/
        output_tma_cp_size, /*SMEM_COL_*/
        output_stride,      /*GMEM_STRIDE_ROW_*/
        1,                  /*GMEM_STRIDE_COL_*/
        1,                  /*SMEM_REPEAT_ROW_*/
        (output_atom_size + output_tma_cp_size - 1) /
            output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,         /*GMEM_ROW_*/
         output_size,        /*GMEM_COL_*/
         batch_size,         /*SMEM_ROW_*/
         output_tma_cp_size, /*SMEM_COL_*/
         output_stride,      /*GMEM_STRIDE_ROW_*/
         1,                  /*GMEM_STRIDE_COL_*/
         1,                  /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // code.e("printf(\"linear_kernel_hopper start\");");

  code.e("kernel::linear_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
         "TMA_OUT, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         Kstages,
         with_residual ? "TMA_RESIDUAL" : "void",
         output_stride);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_HOPPER, code.to_string());
  }
}
int TaskRegister::register_paged_attention_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if ((int)input_ops.size() < num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Shapes/strides
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int num_q_heads_per_kv = num_q_heads / num_kv_heads;
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int max_tokens = input_ops[0]->dtensor.dim[0];

  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  constexpr int B = 3, M = 3, S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int KV_TILE_SIZE = 64;
  int const qkv_rows = num_q_heads_per_kv + 2;
  int const smem_repeat_col =
      (head_dim + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  int const q_smem_stride = max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE;
  int const kv_smem_stride = KV_TILE_SIZE * TMA_CP_ASYNC_SIZE;
  int const non_cached_kv_smem_stride = max_tokens * TMA_CP_ASYNC_SIZE;
  int const num_pages = (max_seq_len + page_size - 1) / page_size;
  int const num_head_group = qkv_stride / head_dim / (num_q_heads_per_kv + 2);

  code.e("using TMA_Q = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         max_tokens,         /* GMEM_DEPTH */
         qkv_rows,           /* GMEM_ROW   */
         head_dim,           /* GMEM_COL   */
         max_tokens,         /* SMEM_DEPTH */
         num_q_heads_per_kv, /* SMEM_ROW   */
         TMA_CP_ASYNC_SIZE,  /* SMEM_COL   */
         qkv_stride,         /* GMEM_STRIDE_DEPTH */
         head_dim,           /* GMEM_STRIDE_ROW   */
         1,                  /* GMEM_STRIDE_COL   */
         1,                  /* SMEM_REPEAT_ROW   */
         smem_repeat_col,    /* SMEM_REPEAT_COL   */
         q_smem_stride       /* SMEM_STRIDE       */
  );

  code.e("using TMA_KV = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         max_tokens,               /* GMEM_DEPTH */
         qkv_rows,                 /* GMEM_ROW   */
         head_dim,                 /* GMEM_COL   */
         max_tokens,               /* SMEM_DEPTH */
         1,                        /* SMEM_ROW   */
         TMA_CP_ASYNC_SIZE,        /* SMEM_COL   */
         qkv_stride,               /* GMEM_STRIDE_DEPTH */
         head_dim,                 /* GMEM_STRIDE_ROW   */
         1,                        /* GMEM_STRIDE_COL   */
         1,                        /* SMEM_REPEAT_ROW   */
         smem_repeat_col,          /* SMEM_REPEAT_COL   */
         non_cached_kv_smem_stride /* SMEM_STRIDE       */
  );

  code.e("using TMA_PAGED_KV_CACHE = kernel::tma::tma_4d<bfloat16, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         num_pages,                             /* GMEM_OUTERMOST_ */
         page_size,                             /* GMEM_DEPTH   */
         num_head_group,                        /* GMEM_ROW   */
         head_dim,                              /* GMEM_COL   */
         1,                                     /* SMEM_OUTERMOST_ */
         KV_TILE_SIZE,                          /* SMEM_DEPTH   */
         num_q_heads_per_kv,                    /* SMEM_ROW   */
         TMA_CP_ASYNC_SIZE,                     /* SMEM_COL   */
         page_size * head_dim * num_head_group, /* GMEM_STRIDE_OUTERMOST_ */
         page_size * head_dim,                  /* GMEM_STRIDE_DEPTH */
         head_dim,                              /* GMEM_STRIDE_ROW   */
         1,                                     /* GMEM_STRIDE_COL   */
         1,                                     /* SMEM_REPEAT_ROW   */
         smem_repeat_col,                       /* SMEM_REPEAT_COL   */
         kv_smem_stride                         /* SMEM_STRIDE       */
  );

  code.e("using TMA_OUTPUT = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         max_tokens,
         num_q_heads_per_kv * num_head_group,
         head_dim,
         max_tokens,
         num_q_heads_per_kv,
         TMA_CP_ASYNC_SIZE,
         head_dim * num_head_group * num_head_group,
         head_dim,
         1,
         1,
         smem_repeat_col,
         max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE);

  code.e("TMA_Q  tma_q "
         "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));");
  code.e("TMA_KV tma_k "
         "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][1]));");
  code.e("TMA_KV tma_v "
         "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][2]));");

  code.e("TMA_PAGED_KV_CACHE "
         "tma_paged_k_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
         "desc_ptrs[1][0]));");
  code.e("TMA_PAGED_KV_CACHE "
         "tma_paged_v_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
         "desc_ptrs[2][0]));");

  code.e("TMA_OUTPUT "
         "tma_output(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs["
         "0][0]));");

  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, "
         "TMA_Q, TMA_KV, TMA_PAGED_KV_CACHE, "
         "TMA_OUTPUT, $>(",
         num_q_heads_per_kv, /* NUM_QO_HEADS               */
         1,                  /* NUM_KV_HEADS               */
         kv_stride,          /* KV_CACHE_STRIDE            */
         qkv_stride,         /* QKV_STRIDE                 */
         output_size,        /* O_STRIDE (= num_q_heads*head_dim) */
         head_dim,           /* HEAD_DIM                   */
         max_seq_len,        /* MAX_SEQ_LEN                */
         page_size,          /* PAGE_SIZE                  */
         max_tokens          /* MAX_TOKENS                 */
  );
  code.e("    tma_q,");
  code.e("    tma_k,");
  code.e("    tma_v,");
  code.e("    tma_paged_k_cache,");
  code.e("    tma_paged_v_cache,");
  code.e("    tma_output,");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->request_id,");
  code.e("    $,", params[2] > 0); // qk_norm
  code.e("    $,", params[3] > 0); // rope
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->head_group);");

  return register_task_variant(TASK_PAGED_ATTENTION_HOPPER, code.to_string());
}

int TaskRegister::register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
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
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim = output_ops[0]->output_tensors[0].dim[1];

  // Currently assume that each rmsnorm task processes one token
  assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::rms_norm_hopper_impl<bfloat16, $, $>(", batch_size, hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-6f);");
  return register_task_variant(TASK_RMS_NORM_HOPPER, code.to_string());
}

int TaskRegister::register_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
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
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 256;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  // int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16; // batch size padded to
  // 16
  int const SMEM_M_SIZE = 16;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;
  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        0,
        0,
        0,
        batch_size,                      /*GMEM_ROW_*/
        output_size,                     /*GMEM_COL_*/
        batch_size,                      /*SMEM_ROW_*/
        output_tma_cp_size,              /*SMEM_COL_*/
        output_stride,                   /*GMEM_STRIDE_ROW_*/
        1,                               /*GMEM_STRIDE_COL_*/
        1,                               /*SMEM_REPEAT_ROW_*/
        1,                               /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,                      /*GMEM_ROW_*/
         output_size,                     /*GMEM_COL_*/
         batch_size,                      /*SMEM_ROW_*/
         output_tma_cp_size,              /*SMEM_COL_*/
         output_stride,                   /*GMEM_STRIDE_ROW_*/
         1,                               /*GMEM_STRIDE_COL_*/
         1,                               /*SMEM_REPEAT_ROW_*/
         1,                               /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e(
      "kernel::linear_swapAB_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
      "TMA_OUT, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      with_residual ? "TMA_RESIDUAL" : "void",
      output_stride);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_SWAPAB_HOPPER, code.to_string());
  }
}

int TaskRegister::register_linear_cutlass_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;
  constexpr int KSTAGES = 4;

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
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  constexpr int TILE_SIZE = 128;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // NOTE: output_size and batch_size are swapped here
  code.e("auto problem_shape = cute::Shape<cute::Int<$>, cute::Int<$>, "
         "cute::Int<$>>{};",
         output_size,
         batch_size,
         reduction_size);
  // NOTE: output_size and batch_size are swapped here
  code.e("using KernelTraits = kernel::MMAKernelTraits<cutlass::bfloat16_t, $, "
         "$, $, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, "
         "cutlass::layout::RowMajor, cutlass::layout::RowMajor, $, $, $, $, "
         "decltype(problem_shape), $, $>;",
         output_size,
         batch_size,
         reduction_size,
         8,
         64,
         batch_size,
         TILE_SIZE,
         batch_size,
         KSTAGES);
  code.e("using Mainloop = kernel::CollectiveMainloop<KernelTraits>;");
  code.e("using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;");
  // code.e("using StrideA = typename KernelTraits::StrideA;");
  // code.e("using StrideB = typename KernelTraits::StrideB;");
  // code.e("using StrideC = typename KernelTraits::StrideC;");
  // code.e("using StrideD = typename KernelTraits::StrideD;");
  // code.e("StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, "
  //        "{KernelTraits::OUTPUT_SIZE, KernelTraits::REDUCTION_SIZE, 1});");
  // code.e("StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::REDUCTION_SIZE, 1});");
  // code.e("StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});");
  // code.e("StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});");
  // code.e("typename Mainloop::Arguments mainloop_args{");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[1].base_ptr),");
  // code.e("    stride_A,");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[0].base_ptr),");
  // code.e("    stride_B,");
  // code.e("};");
  // code.e("typename Epilogue::Arguments epilogue_args{");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[2].base_ptr),");
  // code.e("    stride_C,");
  // code.e(
  //     "    static_cast<cutlass::bfloat16_t
  //     *>(task_desc.outputs[0].base_ptr),");
  // code.e("    stride_C,");
  // code.e("    {1.0f, 1.0f},");
  // code.e("};");
  // code.e("using MainloopParamsDevice = typename Mainloop::template "
  //        "Params<false>;");
  // code.e("MainloopParamsDevice mainloop_params = "
  //        "Mainloop::to_underlying_arguments<false>(problem_shape, "
  //        "mainloop_args);");
  // code.e("typename Epilogue::Params epilogue_params = "
  //        "Epilogue::to_underlying_arguments(problem_shape, epilogue_args);");

  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;

  code.e("using TMA_B = kernel::tma::tma_2d<cutlass::bfloat16_t, $, $, $, $, "
         "$, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<cutlass::bfloat16_t, $, $, $, $, "
         "$, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");

  code.e("kernel::linear_cutlass_ws_hopper<Mainloop, Epilogue, false, "
         "cutlass::bfloat16_t, $, $, $, TMA_A, TMA_B, "
         "$, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride,
         with_residual);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->input_ptrs[2]");
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_CUTLASS_HOPPER, code.to_string());
  }
}

int TaskRegister::register_silu_mul_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, input_stride, output_stride;
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
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl_hopper<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SILU_MUL_HOPPER, code.to_string());
}

int TaskRegister::register_embedding_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
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
  code.e("kernel::embedding_kernel_hopper<bfloat16, $, $, $>(",
         batch_size,
         output_size,
         output_stride);
  if (params[0] == 0) {
    code.e("    runtime_config.tokens + runtime_config.step[0], ");
  } else if (params[0] == 1) {
    code.e("    task_desc->input_ptrs[0],");
  }
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_EMBEDDING_HOPPER, code.to_string());
}

// SM100 Tasks
int TaskRegister::register_linear_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
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
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,         /*GMEM_ROW_*/
         output_size,        /*GMEM_COL_*/
         MMA_N,              /*SMEM_ROW_*/
         MMA_M,              /*SMEM_COL_*/
         output_stride,      /*GMEM_STRIDE_ROW_*/
         1,                  /*GMEM_STRIDE_COL_*/
         1,                  /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M                 /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // Bias Tensor setup
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), cute::make_stride($, cute::Int<1>{}));",
          batch_size,
          output_size,
          output_stride
  );
  code.e("cute::Tensor mBias = cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>($)), layout_Bias);",
          with_residual ? "task_desc->input_ptrs[2]" : "nullptr"
  );
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         with_residual ? "false" : "true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    mBias,");
  code.e("    tma_out); ");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_SM100,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_SM100, code.to_string());
  }
}

int TaskRegister::register_paged_attention_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
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
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, $, $, $, $, "
         "$, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTN_SM100, code.to_string());
}

int TaskRegister::register_argmax_partial_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: num_partial_tasks
  assert(params.size() == 1);
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
  int num_partial_tasks = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_sm100_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_PARTIAL_SM100, code.to_string());
}

int TaskRegister::register_argmax_reduce_sm100_task(threadblock::Graph const &bgraph,
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
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_sm100_kernel<bfloat16, $, $, $>(",
         batch_size,
         params[0],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_REDUCE_SM100, code.to_string());
}



} // namespace runtime
} // namespace mirage
