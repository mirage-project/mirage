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

static bool graph_input_has_num_dims(threadblock::Graph const &bgraph,
                                     size_t index,
                                     int num_dims) {
  assert(bgraph.operators.size() > index);
  assert(bgraph.operators[index]->op_type == mirage::type::TB_INPUT_OP);
  tb::TBInputOp const *input_op =
      static_cast<tb::TBInputOp const *>(bgraph.operators[index]);
  return input_op->output_tensors[0].num_dims == num_dims;
}

static void emit_deepseek_prefill_flag(mirage::transpiler::CodeKeeper &code) {
  code.e("bool prompt_prefill_ = false;");
  code.e("for (int bi_pf_ = 0; bi_pf_ < MPK_MAX_NUM_BATCHED_REQUESTS; "
         "++bi_pf_) {");
  code.e("  int req_pf_ = runtime_config.request_ids[bi_pf_];");
  code.e("  if (req_pf_ < 0) continue;");
  code.e("  int q_len_pf_ = runtime_config.qo_indptr_buffer[bi_pf_ + 1] - "
         "runtime_config.qo_indptr_buffer[bi_pf_];");
  code.e("  if (runtime_config.step[req_pf_] < "
         "runtime_config.prompt_length[req_pf_] && q_len_pf_ > 8) {");
  code.e("    prompt_prefill_ = true;");
  code.e("    break;");
  code.e("  }");
  code.e("}");
}

static void emit_deepseek_phase_gate(mirage::transpiler::CodeKeeper &code,
                                     int gate_mode) {
  if (gate_mode == 0) {
    return;
  }
  assert(gate_mode == 1 || gate_mode == 2);
  code.e("{");
  emit_deepseek_prefill_flag(code);
  if (gate_mode == 1) {
    code.e("if (!prompt_prefill_) return;");
  } else {
    code.e("if (prompt_prefill_) return;");
  }
  code.e("}");
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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
  // params (optional, default = legacy contiguous):
  //   params[0] = process_dim  (elements per row to normalise; defaults to
  //               the DTensor's last-dim size = contiguous).
  //   params[1] = in_offset_elems   (skip elements at the start of each row).
  //   params[2] = out_offset_elems  (skip elements at the start of each row
  //               in the output; equal to in_offset for in-place).
  // Used by the QKV-a fused path (user #2 part-a, 2026-05-12): when q_a_out
  // and kv_a_out are aliases of a wider qkv_a_out buffer, the per-row offset
  // selects which slice to normalise.
  assert(params.size() == 0 || params.size() == 3);
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
  int hidden_dim_full = output_ops[0]->output_tensors[0].dim[1];
  // Currently assume that each rmsnorm task processes one token
  assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  int process_dim = params.size() == 3 ? params[0] : hidden_dim_full;
  int in_offset = params.size() == 3 ? params[1] : 0;
  int out_offset = params.size() == 3 ? params[2] : 0;
  assert(in_offset + process_dim <= hidden_dim_full);
  assert(out_offset + process_dim <= hidden_dim_full);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::rms_norm_impl<bfloat16, $, $, $, $>(",
         batch_size,
         process_dim,
         in_offset,
         out_offset);
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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
  int max_tokens = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_task_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size,
         max_tokens);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
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
  input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("int num_active_tokens_ = $;", batch_size);
  code.e("#ifndef MPK_TEST_MODE");
  code.e("num_active_tokens_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("#endif");
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    num_active_tokens_);");
  return register_task_variant(TASK_SILU_MUL, code.to_string());
}

int TaskRegister::register_identity_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params) {
  // params: [] (legacy copy) OR [noop_flag] (when 1, emit empty body — the
  //   task graph node still exists for case-3 fork+join shaping but the
  //   kernel does nothing, just `return;`)
  //   OR [noop_flag, gate_decode_q_len_flag] (when gate_decode_q_len_flag==1,
  //   emit a runtime Q_LEN gate at the top of the kernel that returns
  //   immediately if request 0's Q_LEN <= 8. This makes the kpe_sep_v2
  //   phantom-bridge identity a noop on decode iters while still doing the
  //   real BF16 copy on chunked-prefill iters — saving ~16 μs per decode
  //   layer).
  assert(params.size() <= 2);
  bool is_noop = (params.size() >= 1 && params[0] == 1);
  bool gate_decode_q_len = (params.size() >= 2 && params[1] == 1);
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
  // Both input and output tensors should be row major
  assert(input_ops[0]->dtensor.layout == layout::DmemRowMajor);
  assert(output_ops[0]->dtensor.layout == layout::DmemRowMajor);
  // Shape should be guranteed by higher-level APIs

  int outer_dim_size = 1, inner_dim_size, outer_dim_stride, output_size;
  for (int i = 0; i < input_ops[0]->dtensor.num_dims - 1; i++) {
    outer_dim_size *= input_ops[0]->dtensor.dim[i];
  }
  inner_dim_size =
      input_ops[0]->dtensor.dim[input_ops[0]->dtensor.num_dims - 1];
  outer_dim_stride = inner_dim_size;
  output_size = output_ops[0]
                    ->output_tensors[0]
                    .dim[output_ops[0]->output_tensors[0].num_dims - 1];
  // assert(output_size >= bgraph.block_dim.x);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (is_noop) {
    // Empty kernel body — task exists only as a task-graph fork/join
    // shaping node. No data motion, just return.
    code.e("// identity_task no-op variant (graph-shaping only)");
  } else {
    if (gate_decode_q_len) {
      // Runtime Q_LEN gate: skip the BF16 copy entirely on decode iters
      // (Q_LEN <= 8). Used by the kpe_sep_v2 phantom-bridge identity in the
      // chunked-prefill task graph: chunked_prefill itself has a Q_LEN > 8
      // gate, so its kpe_sep_v2 input is never read on decode iters, and
      // letting the buffer keep stale data is harmless.
      code.e("int q_len_id_ = runtime_config.qo_indptr_buffer[1] - "
             "runtime_config.qo_indptr_buffer[0];");
      code.e("if (q_len_id_ <= 8) return;");
    }
    code.e("kernel::identity_task_impl<bfloat16, $, $, $, $>(",
           outer_dim_size,
           inner_dim_size,
           outer_dim_stride,
           output_size);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->output_ptrs[0]);");
  }
  return register_task_variant(TASK_IDENTITY, code.to_string());
}

int TaskRegister::register_identity_2in_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params) {
  // 2-input variant of register_identity_task. Same kernel body (read
  // input_ptrs[0], write output_ptrs[0]) — input_ptrs[1] is a fake-dep
  // handle the kernel never reads. Used to chain a copy task behind a
  // distant producer in the runtime scheduler (e.g. defer the shared-
  // expert gate_up GEMM behind the routed-MoE W13 group GEMM).
  // Params semantics inherit from register_identity_task:
  //   params[] (legacy copy) OR [noop_flag] OR [noop_flag, gate_decode_q_len].
  assert(params.size() <= 2);
  bool is_noop = (params.size() >= 1 && params[0] == 1);
  bool gate_decode_q_len = (params.size() >= 2 && params[1] == 1);
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
  // Both the real input (slot 0) and the output tensor should be row major.
  // input_ops[1] is the fake-dep handle; layout is irrelevant since the
  // kernel never dereferences it.
  assert(input_ops[0]->dtensor.layout == layout::DmemRowMajor);
  assert(output_ops[0]->dtensor.layout == layout::DmemRowMajor);

  int outer_dim_size = 1, inner_dim_size, outer_dim_stride, output_size;
  for (int i = 0; i < input_ops[0]->dtensor.num_dims - 1; i++) {
    outer_dim_size *= input_ops[0]->dtensor.dim[i];
  }
  inner_dim_size =
      input_ops[0]->dtensor.dim[input_ops[0]->dtensor.num_dims - 1];
  outer_dim_stride = inner_dim_size;
  output_size = output_ops[0]
                    ->output_tensors[0]
                    .dim[output_ops[0]->output_tensors[0].num_dims - 1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (is_noop) {
    code.e("// identity_2in_task no-op variant (graph-shaping only)");
  } else {
    if (gate_decode_q_len) {
      code.e("int q_len_id_ = runtime_config.qo_indptr_buffer[1] - "
             "runtime_config.qo_indptr_buffer[0];");
      code.e("if (q_len_id_ <= 8) return;");
    }
    code.e("kernel::identity_task_impl<bfloat16, $, $, $, $>(",
           outer_dim_size,
           inner_dim_size,
           outer_dim_stride,
           output_size);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->output_ptrs[0]);");
  }
  return register_task_variant(TASK_IDENTITY, code.to_string());
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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

int TaskRegister::register_reduction_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  // params[0]: num_gpus
  // params[1]: my_gpu_id
  assert(params.size() == 2);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // input[0]: local data [batch_size, output_size]
  // input[1]: buffer for allgather [num_gpus, batch_size, output_size]
  // output[0]: reduced result [batch_size, output_size]
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
  // For now, the memory partition of the input[0] results in a strided
  // 2D tensor, which cannot be directly transferred by a single nvshmem
  // memput. So we use for loop to iterate over the first dim and transfer each
  // row. If the upperlayer changes this layout, this "for-loop" method can
  // fail. So we assert it here just in case.
  assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
         input_ops[0]->input_map.z == -1);
  // Currently support 2D reduction, buffer has an extra world_size dim
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int output_size = input_ops[0]->output_tensors[0].dim[1];
  // get strides (C20: from dtensor.stride[0] — view-safe; for root tensors
  // this equals owner_op's input_strides[0])
  int input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  int output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
  assert(input_stride == output_stride);
  // Register reduction kernel
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::reduction_kernel<bfloat16, $, $, $, $, $>(",
         params[0],
         params[1],
         batch_size,
         output_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_REDUCE, code.to_string());
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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

  // code.e("using TMA_Q = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $, $,
  // "
  //        "$, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,         /* GMEM_DEPTH */
  //        qkv_rows,           /* GMEM_ROW   */
  //        head_dim,           /* GMEM_COL   */
  //        max_tokens,         /* SMEM_DEPTH */
  //        num_q_heads_per_kv, /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,  /* SMEM_COL   */
  //        qkv_stride,         /* GMEM_STRIDE_DEPTH */
  //        head_dim,           /* GMEM_STRIDE_ROW   */
  //        1,                  /* GMEM_STRIDE_COL   */
  //        1,                  /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,    /* SMEM_REPEAT_COL   */
  //        q_smem_stride       /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_KV = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $,
  // $, "
  //        "$, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,               /* GMEM_DEPTH */
  //        qkv_rows,                 /* GMEM_ROW   */
  //        head_dim,                 /* GMEM_COL   */
  //        max_tokens,               /* SMEM_DEPTH */
  //        1,                        /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,        /* SMEM_COL   */
  //        qkv_stride,               /* GMEM_STRIDE_DEPTH */
  //        head_dim,                 /* GMEM_STRIDE_ROW   */
  //        1,                        /* GMEM_STRIDE_COL   */
  //        1,                        /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,          /* SMEM_REPEAT_COL   */
  //        non_cached_kv_smem_stride /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_PAGED_KV_CACHE = kernel::tma::tma_4d<bfloat16, $, $, $,
  // $, "
  //        "$, $, $, $, $, $, $, $, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        num_pages,                             /* GMEM_OUTERMOST_ */
  //        page_size,                             /* GMEM_DEPTH   */
  //        num_head_group,                        /* GMEM_ROW   */
  //        head_dim,                              /* GMEM_COL   */
  //        1,                                     /* SMEM_OUTERMOST_ */
  //        KV_TILE_SIZE,                          /* SMEM_DEPTH   */
  //        num_q_heads_per_kv,                    /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,                     /* SMEM_COL   */
  //        page_size * head_dim * num_head_group, /* GMEM_STRIDE_OUTERMOST_ */
  //        page_size * head_dim,                  /* GMEM_STRIDE_DEPTH */
  //        head_dim,                              /* GMEM_STRIDE_ROW   */
  //        1,                                     /* GMEM_STRIDE_COL   */
  //        1,                                     /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,                       /* SMEM_REPEAT_COL   */
  //        kv_smem_stride                         /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_OUTPUT = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $,
  // "
  //        "$, $, $, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,
  //        num_q_heads_per_kv * num_head_group,
  //        head_dim,
  //        max_tokens,
  //        num_q_heads_per_kv,
  //        TMA_CP_ASYNC_SIZE,
  //        head_dim * num_head_group * num_head_group,
  //        head_dim,
  //        1,
  //        1,
  //        smem_repeat_col,
  //        max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE);

  // code.e("TMA_Q  tma_q "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));");
  // code.e("TMA_KV tma_k "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][1]));");
  // code.e("TMA_KV tma_v "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][2]));");

  // code.e("TMA_PAGED_KV_CACHE "
  //        "tma_paged_k_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
  //        "desc_ptrs[1][0]));");
  // code.e("TMA_PAGED_KV_CACHE "
  //        "tma_paged_v_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
  //        "desc_ptrs[2][0]));");

  // code.e("TMA_OUTPUT "
  //        "tma_output(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs["
  //        "0][0]));");

  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, $, "
         "$, $, $, $>(",
         num_q_heads_per_kv, /* NUM_QO_HEADS               */
         1,                  /* NUM_KV_HEADS               */
         num_kv_heads,       /* NUM_QO_GROUPS              */
         kv_stride,          /* KV_CACHE_STRIDE            */
         qkv_stride,         /* QKV_STRIDE                 */
         output_size,        /* O_STRIDE (= num_q_heads*head_dim) */
         head_dim,           /* HEAD_DIM                   */
         -1,          /* SEQ_LEN (not used for non-split KV tasks)          */
         max_seq_len, /* MAX_SEQ_LEN                */
         page_size,   /* PAGE_SIZE                  */
         max_tokens,  /* MAX_TOKENS                 */
         "false",     /* PARTITION_KV               */
         1            /* NUM_KV_CHUNKS              */
  );
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0); // qk_norm
  code.e("    $,", params[3] > 0); // rope
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    nullptr,"); // lse, not used for non-split KV tasks
  code.e("    0);");      // kv_idx, not used for non-split KV tasks

  return register_task_variant(TASK_PAGED_ATTENTION_HOPPER, code.to_string());
}

int TaskRegister::register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params (optional, default = legacy contiguous):
  //   params[0] = process_dim    (HIDDEN_DIM the kernel processes per row).
  //   params[1] = in_offset_elems  (skip elements at start of each row).
  //   params[2] = out_offset_elems (same, for output).
  // QKV-a fused path uses these so q_a_layernorm / kv_a_layernorm can run
  // in-place on a wider qkv_a_out buffer.
  assert(params.size() == 0 || params.size() == 3);
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
  int hidden_dim_full = output_ops[0]->output_tensors[0].dim[1];
  // C20 (2026-05-17): use stride[0] (in elements) instead of dim[1] for the
  // row-walk stride. For root tensors stride[0] == dim[1] (row-major
  // default), so non-view callers see no behavior change. For `mpk.narrow`
  // views, dim[1] = slot_width but stride[0] = parent_width — using stride
  // here is what prevents the kernel from overwriting the adjacent slot
  // when stepping to row i+1. compute width (HIDDEN_DIM template param)
  // still derives from dim/process_dim.
  int in_row_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  int out_row_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  // Currently assume that each rmsnorm task processes one token
  // assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  int process_dim = params.size() == 3 ? params[0] : hidden_dim_full;
  int in_offset = params.size() == 3 ? params[1] : 0;
  int out_offset = params.size() == 3 ? params[2] : 0;
  assert(in_offset + process_dim <= hidden_dim_full);
  assert(out_offset + process_dim <= hidden_dim_full);
  int dtensor_batch = output_ops[0]->dtensor.dim[0];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // B34 (2026-05-15): builder may shrink grid.x below mbt so each CTA
  // handles BATCH_SIZE > 1 rows. Skip CTAs whose first row is past the
  // active token count, and clamp the kernel's inner row-loop to the
  // remaining active rows so we don't normalize/overwrite stale bf16.
  // In MPK_TEST_MODE qo_indptr_buffer is uninitialised (zeros), so fall
  // back to the full DTensor batch dim (no skip) to keep unit-tests
  // working — matches the silu_mul test-mode escape hatch.
  code.e("int active_rows_rms_ = $;", dtensor_batch);
  code.e("#ifndef MPK_TEST_MODE");
  code.e("active_rows_rms_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("#endif");
  code.e("int task_first_row_ = "
         "task_desc->task_metadata.request_id * $;",
         batch_size);
  code.e("if (task_first_row_ >= active_rows_rms_) return;");
  code.e("int row_count_cap_ = active_rows_rms_ - task_first_row_;");
  // NUM_THREADS defaults to 256; explicit so we can append IN/OUT_OFFSET.
  // IN_ROW_STRIDE / OUT_ROW_STRIDE = hidden_dim_full so multi-row CTAs walk
  // by the actual physical row width (matters for QKV-a fused slice paths
  // where process_dim < hidden_dim_full).
  code.e("kernel::rms_norm_hopper_impl<bfloat16, $, $, 256, $, $, $, $>(",
         batch_size,
         process_dim,
         in_offset,
         out_offset,
         in_row_stride,
         out_row_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-6f,");
  code.e("    row_count_cap_);");
  return register_task_variant(TASK_RMS_NORM_HOPPER, code.to_string());
}

int TaskRegister::register_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  // assert(params.size() == 0);
  bool rank_with_residual = with_residual;
  if (with_residual) {
    assert(params.size() == 1);
    rank_with_residual = (params[0] == 1);
  } else {
    assert(params.size() == 0);
  }
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 128;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16;
  // int const SMEM_M_SIZE = 16;
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

  if (with_residual && rank_with_residual) {
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
  if (with_residual && rank_with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e(
      "kernel::linear_swapAB_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
      "TMA_OUT, $, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      (with_residual && rank_with_residual) ? "TMA_RESIDUAL" : "void",
      output_stride,
      "false" /*SplitK*/);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual && rank_with_residual) {
    code.e("    &tma_residual,");
    code.e("    runtime_config.my_gpu_id == 0");
  } else {
    code.e("    nullptr,");
    code.e("    false/*residual*/");
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
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
  input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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
  bool rank_with_residual = with_residual;
  if (with_residual) {
    assert(params.size() == 1);
    rank_with_residual = (params[0] == 1);
  } else {
    assert(params.size() == 0);
  }
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,    /*GMEM_ROW_*/
         output_size,   /*GMEM_COL_*/
         MMA_N,         /*SMEM_ROW_*/
         MMA_M,         /*SMEM_COL_*/
         output_stride, /*GMEM_STRIDE_ROW_*/
         1,             /*GMEM_STRIDE_COL_*/
         1,             /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
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
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         (with_residual && rank_with_residual) ? "task_desc->input_ptrs[2]"
                                               : "nullptr");
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, "
         "decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         (with_residual && rank_with_residual) ? "false" : "true",
         /*SplitK=*/"false",
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

int TaskRegister::register_splitk_linear_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  // params: [] (no gate) OR [prefill_gate] (when prefill_gate==1, emit a
  // runtime
  //   Q_LEN gate that returns immediately if request-0's Q_LEN <= 8 — i.e. this
  //   splitk linear runs on PREFILL iters only. Used by the DSv3 router
  //   dual-dispatch: splitk router (prefill) + dsv3_router_gemm GEMV (decode).
  //   Mirrors identity's gate_decode_q_len; default-off => byte-identical for
  //   all other splitk_linear callers (incl. the MTP router gate).
  assert(params.size() <= 1);
  bool prefill_gate = (params.size() >= 1 && params[0] == 1);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0,
      reduction_stride = 0;
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
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->output_tensors[0].dim[1];
  reduction_stride = input_ops[0]->dtensor.dim[1];
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

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
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_stride,  /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_stride,  /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,    /*GMEM_ROW_*/
         output_size,   /*GMEM_COL_*/
         MMA_N,         /*SMEM_ROW_*/
         MMA_M,         /*SMEM_COL_*/
         output_stride, /*GMEM_STRIDE_ROW_*/
         1,             /*GMEM_STRIDE_COL_*/
         1,             /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
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
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         with_residual ? "task_desc->input_ptrs[2]" : "nullptr");
  if (prefill_gate) {
    // Prefill-only gate: skip on DECODE iters (step >= prompt_length for all
    // active requests). The dsv3_router_gemm GEMV companion serves decode; this
    // splitk serves prefill (incl. small final chunks, which step<prompt_length
    // keeps here). step-based (NOT q_len) — chunk-size-independent. Returns
    // before the GEMM work (cheap local TMA objects already constructed are
    // harmless). Mirrors the GEMV's rg_decode_ flag, inverted.
    code.e("bool sk_decode_ = true;");
    code.e("for (int bi_sk_ = 0; bi_sk_ < MPK_MAX_NUM_BATCHED_REQUESTS; "
           "++bi_sk_) {");
    code.e("  int req_sk_ = runtime_config.request_ids[bi_sk_];");
    code.e("  if (req_sk_ < 0) continue;");
    code.e("  if (runtime_config.step[req_sk_] < "
           "runtime_config.prompt_length[req_sk_]) { sk_decode_ = false; "
           "break; }");
    code.e("}");
    code.e("if (sk_decode_) return;");
  }
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, "
         "decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         with_residual ? "false" : "true",
         /*SplitK=*/"true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    mBias,");
  code.e("    tma_out); ");

  return register_task_variant(TASK_SPLITK_LINEAR_SM100, code.to_string());
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
  code.e("kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, $, $, "
         "$, $, "
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
  code.e("    task_desc->task_metadata.request_id,");
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

int TaskRegister::register_argmax_partial_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
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

int TaskRegister::register_argmax_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
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

int TaskRegister::register_sampling_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: seed
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
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int vocab_size = input_ops[0]->output_tensors[0].dim[1];
  int seed = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::sampling_from_logits_kernel<256, 4, bfloat16, int>(");
  code.e("    static_cast<bfloat16*>(task_desc->input_ptrs[0]),");
  code.e("    static_cast<int*>(task_desc->output_ptrs[0]),");
  code.e("    $,", vocab_size);
  code.e("    $,", seed);
  code.e("    0,  // philox_offset");
  code.e("    $);", batch_size);
  return register_task_variant(TASK_SAMPLING_SM100, code.to_string());
}

int TaskRegister::register_tensor_init_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params) {
  // Arity: (1 input, 2 outputs).
  //   input_ops[0]  = linear_input  (dummy dep, not read)
  //   output_ops[0] = linear_output (the tile zeroed by the kernel)
  //   output_ops[1] = linear_input  (dummy dep edge, not written)
  // params: [] (always zero) OR [prefill_gate] (when 1, skip the zero-fill on
  //   DECODE iters: step>=prompt_length for all active reqs). Used by the DSv3
  //   router dual-dispatch so the decode-only GEMV is the SOLE writer of
  //   router_logits at decode (no init-vs-GEMV write race; Codex r2/r3). The
  //   task still dispatches (dep edges intact); only the kernel body returns
  //   early. Default-off => byte-identical for all other tensor_init callers.
  assert(params.size() <= 1);
  bool prefill_gate = (params.size() >= 1 && params[0] == 1);
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
  // The buffer to zero is output_ops[0] (the linear's output).
  assert(output_ops[0]->dtensor.num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int output_size = output_ops[0]->output_tensors[0].dim[1];
  int output_stride = output_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (prefill_gate) {
    // Prefill-only: skip the zero-fill on DECODE (step>=prompt_length for all
    // active reqs), mirroring the splitk router's prefill_gate. Lets the decode
    // GEMV own router_logits without an init-vs-GEMV race.
    code.e("bool ti_decode_ = true;");
    code.e("for (int bi_ti_ = 0; bi_ti_ < MPK_MAX_NUM_BATCHED_REQUESTS; "
           "++bi_ti_) {");
    code.e("  int req_ti_ = runtime_config.request_ids[bi_ti_];");
    code.e("  if (req_ti_ < 0) continue;");
    code.e("  if (runtime_config.step[req_ti_] < "
           "runtime_config.prompt_length[req_ti_]) { ti_decode_ = false; "
           "break; }");
    code.e("}");
    code.e("if (ti_decode_) return;");
  }
  code.e("kernel::tensor_init_zero_sm100_task_impl<$, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_TENSOR_INIT, code.to_string());
}

int TaskRegister::register_elementwise_add_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params (optional):
  //   size 0 (legacy): input_a contiguous with the output's row stride.
  //   size 2: [in_a_stride, in_a_offset] — input_a reads a column slice of
  //           a wider buffer. Used by QKV-a fused diagnostics where
  //           input_a is qkv_a_out (mbt, 2176) and we dump a particular
  //           slice into a smaller (mbt, OUT_SIZE) tensor.
  assert(params.size() == 0 || params.size() == 2);
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
  int output_size = output_ops[0]->output_tensors[0].dim[1];
  int output_stride = output_ops[0]->dtensor.dim[1];
  int in_a_stride = (params.size() == 2) ? params[0] : output_stride;
  int in_a_offset = (params.size() == 2) ? params[1] : 0;
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::elementwise_add_task_impl<cute::bfloat16_t, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         in_a_stride,
         in_a_offset);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_ELEMENTWISE_ADD_SM100, code.to_string());
}

int TaskRegister::register_dsv3_router_gemm_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // DSv3 router-gate CUDA-core GEMV. inputs: [act, weight]; output:
  // router_logits.
  //   act    [M, K]          replicated to every instance (dim_map -1).
  //   weight [N_TOTAL, K]    split dim0 -> grid.x, so the per-task tile is
  //                          [N_PER_CTA, K] and the ptr is pre-sliced.
  //   output [M, N_TOTAL]    split dim1 -> grid.x, tile [M, N_PER_CTA]; the
  //                          kernel writes out[m * N_TOTAL + e] (N_TOTAL
  //                          stride).
  // params: [] (no gate) OR [decode_gate] (when decode_gate==1, emit a runtime
  //   Q_LEN gate that returns immediately if request-0's Q_LEN > 8 — i.e. the
  //   GEMV runs on DECODE iters only, the INVERSE of identity's
  //   gate_decode_q_len. Used by the router dual-dispatch: GEMV (decode) +
  //   splitk router (prefill).
  assert(params.size() <= 1);
  bool decode_gate = (params.size() >= 1 && params[0] == 1);
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
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  // M = active rows, read from the (possibly narrowed) INPUT — small at decode.
  // The OUTPUT may be the FULL router_logits [mbt, N_TOTAL] (non-virtual, so
  // the AnnotatedGraph adds the tensor_init->GEMV WAW edge — avoids the
  // virtual-view ordering race Codex r2 flagged); the kernel writes only rows
  // [0, M).
  int M = input_ops[0]->output_tensors[0].dim[0];
  int K = input_ops[0]->output_tensors[0].dim[1];
  int n_per_cta = input_ops[1]->output_tensors[0].dim[0];
  int n_total = output_ops[0]->dtensor.dim[1];
  assert(input_ops[1]->output_tensors[0].dim[1] == K);
  assert(M <= output_ops[0]->output_tensors[0].dim[0]);
  assert(output_ops[0]->output_tensors[0].dim[1] == n_per_cta);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (decode_gate) {
    // Decode-only gate. A request is DECODING iff step >= prompt_length (it has
    // finished consuming its prompt). This is the chunk-size-independent
    // discriminator (verified persistent_kernel.cuh:291-303: num_new_tokens =
    // prompt_length - step; >0 => prefill chunk of any size 1..mbt). Using
    // step (NOT q_len>8) keeps a small final prefill chunk (q_len<=8,
    // step<prompt_length) on the splitk companion instead of misrouting it into
    // this M=small GEMV. (The dense decode GEMM tolerates a q_len gate because
    // it is M-robust via runtime active_rows; this GEMV bakes M, so it must use
    // the exact phase.) Skip unless ALL active requests are decoding.
    code.e("bool rg_decode_ = true;");
    code.e("for (int bi_rg_ = 0; bi_rg_ < MPK_MAX_NUM_BATCHED_REQUESTS; "
           "++bi_rg_) {");
    code.e("  int req_rg_ = runtime_config.request_ids[bi_rg_];");
    code.e("  if (req_rg_ < 0) continue;");
    code.e("  if (runtime_config.step[req_rg_] < "
           "runtime_config.prompt_length[req_rg_]) { rg_decode_ = false; "
           "break; }");
    code.e("}");
    code.e("if (!rg_decode_) return;");
  }
  code.e("kernel::dsv3_router_gemm_task_impl<cute::bfloat16_t, $, $, $, $>(",
         M,
         n_per_cta,
         K,
         n_total);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_DSV3_ROUTER_GEMM_SM100, code.to_string());
}

int TaskRegister::register_softmax_gather_sm100_task(
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
  // input[0] = logits [batch, vocab], input[1] = token_ids [batch, 1]
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int vocab_size = input_ops[0]->output_tensors[0].dim[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::softmax_gather_task_impl<cute::bfloat16_t, $, $>(",
         batch_size,
         vocab_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_SOFTMAX_GATHER_SM100, code.to_string());
}

int TaskRegister::register_mtp_verify_probabilistic_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1);
  int num_draft_tokens = params[0];
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
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
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::target_verify_probabilistic_kernel<$>(", num_draft_tokens);
  code.e("    task_desc->input_ptrs[0],");   // draft_token_ids
  code.e("    task_desc->input_ptrs[1],");   // target_token_ids
  code.e("    task_desc->input_ptrs[2],");   // target_probs
  code.e("    task_desc->input_ptrs[3],");   // draft_probs
  code.e("    task_desc->input_ptrs[4],");   // seed
  code.e("    task_desc->output_ptrs[0],");  // accepted_count
  code.e("    task_desc->output_ptrs[1]);"); // output_tokens
  return register_task_variant(TASK_MTP_VERIFY_PROBABILISTIC, code.to_string());
}

int TaskRegister::register_mtp_float_scatter_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int batch_size = params[0];
  int num_slots = params[1];
  int slot_idx = params[2];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mtp_float_scatter_kernel<$, $, $>(",
         batch_size,
         num_slots,
         slot_idx);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_MTP_FLOAT_SCATTER, code.to_string());
}

int TaskRegister::register_prob_extract_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 2);
  int max_positions = params[0];
  int num_extract = params[1];
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
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::prob_extract_task_impl<$, $, $>(",
         batch_size,
         max_positions,
         num_extract);
  code.e("    task_desc->input_ptrs[0],");                           // buffer
  code.e("    task_desc->output_ptrs[0],");                          // output
  code.e("    static_cast<int const*>(task_desc->input_ptrs[1]));"); // offset
  return register_task_variant(TASK_PROB_EXTRACT_SM100, code.to_string());
}

int TaskRegister::register_prob_scatter_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1);
  int max_positions = params[0];
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // prob + step_counter
  int num_outputs = 1; // buffer
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::prob_scatter_task_impl<$, $>(", batch_size, max_positions);
  code.e("    task_desc->input_ptrs[0],");                           // prob
  code.e("    task_desc->output_ptrs[0],");                          // buffer
  code.e("    static_cast<int const*>(task_desc->input_ptrs[1]));"); // step
  return register_task_variant(TASK_PROB_SCATTER_SM100, code.to_string());
}

int TaskRegister::register_moe_topk_softmax_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts = 0, num_experts_per_tok = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 3;
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
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  num_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == num_experts);
  // get input stride
  input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::topk_softmax_task_impl<cute::bfloat16_t, $, $, $, $>(",
         /*VPT=*/8,
         /*EXPERTS=*/num_experts,
         /*WARPS_PER_TB=*/8,
         /*BYTES_PER_LDG=*/16);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    nullptr,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  code.e("    $,", num_experts_per_tok);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    0,");
  code.e("    $,", num_experts);
  code.e("    true);");
  return register_task_variant(TASK_MOE_TOPK_SOFTMAX_SM100, code.to_string());
}

int TaskRegister::register_moe_topk_sigmoid_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3 || params.size() == 5 || params.size() == 6);
  int num_groups = params[0];
  int topk_group = params[1];
  float scaling_factor;
  memcpy(&scaling_factor, &params[2], sizeof(float));
  int local_expert_start = params.size() >= 5 ? params[3] : 0;
  int local_expert_end = params.size() >= 5 ? params[4] : -1;
  // PR696: fuse_compaction template arg. 1 (default) = single-CTA path
  // (inline marker-init + ballot compaction; the decode / current behavior).
  // 0 = multi-CTA prefill path (caller pre-inits markers via the marker-init
  // task and runs the compaction task afterward).
  int fuse_compaction = params.size() >= 6 ? params[5] : 1;

  int batch_size = 0, num_experts = 0, num_local_experts = 0,
      num_experts_per_tok = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Validate output shapes
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  num_local_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_local_experts + 1);
  // Validate input shapes
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  num_experts = input_ops[0]->output_tensors[0].dim[1];
  // Validate bias shape
  assert(input_ops[1]->output_tensors[0].num_dims == 1);
  assert(input_ops[1]->output_tensors[0].dim[0] == num_experts);
  if (local_expert_end < 0) {
    local_expert_end = local_expert_start + num_local_experts;
  }
  assert(local_expert_start >= 0);
  assert(local_expert_end == local_expert_start + num_local_experts);
  assert(local_expert_end <= num_experts);

  assert(num_experts % num_groups == 0 &&
         "Number of experts must be divisible by number of groups");
  int experts_per_group = num_experts / num_groups;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // C13 (2026-05-17, REVERTED): tried VPT 8 -> 16 to double ROWS_PER_WARP
  // and halve outer-loop iters. Per-call TOPK_SIGMOID dropped 109 -> 83 μs
  // (-24%), BUT the megakernel's per-token latency REGRESSED 86 -> 122 ms
  // (+42%) — likely register pressure from VPT=16 inflating per-thread
  // arrays (row_chunk[16] + biased_chunk[16]) and globally pinning the
  // megakernel to a lower occupancy via the shared __launch_bounds__.
  // Revert keeps the env hook for opt-in only; default = legacy VPT=8.
  char const *_vpt_env = std::getenv("MPK_DSV3_TOPK_VPT");
  int topk_vpt = _vpt_env ? std::atoi(_vpt_env) : 8;
  if (topk_vpt < 4 || topk_vpt > 32 || (topk_vpt & (topk_vpt - 1)) != 0) {
    topk_vpt = 8;
  }
  code.e("kernel::topk_sigmoid_task_impl<cute::bfloat16_t, $, $, $, $, $, $, "
         "$, $, $, $>(",
         /*VPT=*/topk_vpt,
         /*EXPERTS=*/num_experts,
         /*LOCAL_EXPERTS=*/num_local_experts,
         /*WARPS_PER_TB=*/8,
         /*BYTES_PER_LDG=*/16,
         /*NUM_GROUPS=*/num_groups,
         /*TOPK_GROUP=*/topk_group,
         /*EXPERTS_PER_GROUP=*/experts_per_group,
         /*TOPK_EXPERTS=*/num_experts_per_tok,
         /*FUSE_COMPACTION=*/fuse_compaction);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    nullptr,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    $,", local_expert_start);
  code.e("    $,", local_expert_end);
  code.e("    $f,", scaling_factor);
  // P6 (2026-05-14): bound compute loop to runtime active tokens. The
  // initial "broke correctness" reading was a misdiagnosis — the
  // 19-layer DSv3 baseline already outputs all-zero tokens at
  // profile_start_step=100 (verified by `git stash` baseline test:
  // same all-zero output), so the regression I attributed to P6 was
  // baseline noise. The kernel-side skip is safe: Phase 0 init zeroes
  // the full [0, num_rows) routing range, downstream moe_permute's
  // `slot_1idx > 0` filter treats padded slots as "no routing".
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_MOE_TOPK_SIGMOID_SM100, code.to_string());
}

int TaskRegister::register_moe_linear_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      orig_output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
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
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // get output stride
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[1]);
  orig_output_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // MoE constant:
  int expert_stride = (w13_linear) ? 10 : 8;
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
  // TMA_B for expert weights
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         (num_experts - 1) * orig_output_size + output_size, /*GMEM_ROW_*/
         reduction_size,                                     /*GMEM_COL_*/
         MMA_M,                                              /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,                                  /*SMEM_COL_*/
         reduction_size, /*GMEM_STRIDE_ROW_*/
         1,              /*GMEM_STRIDE_COL_*/
         1,              /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  // Bias Tensor setup
  code.e(
      "cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $, $), "
      "cute::make_stride($, cute::Int<1>{}, $));",
      batch_size,
      output_size,
      num_experts,
      output_stride,
      output_stride * batch_size);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         "nullptr");
  // Topk_indices Tensor setup
  code.e("cute::Layout layout_routing_indices = "
         "cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[2])), layout_routing_indices);");
  // Topk_mask Tensor setup
  code.e("cute::Layout layout_expert_mask = "
         "cute::make_layout(cute::make_shape($), "
         "cute::make_stride(cute::Int<1>{}));",
         num_experts + 1);
  code.e("cute::Tensor mMask = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[3])), layout_expert_mask);");
  // Output Tensor setup
  code.e("cute::Layout layout_output = cute::make_layout(cute::make_shape($, "
         "$, $), "
         "cute::make_stride($, $, cute::Int<1>{}));",
         batch_size,
         num_experts_per_tok,
         output_size,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");
  // Input Tensor setup
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, cute::Int<1>{}, $));",
           batch_size,
           reduction_size,
           num_experts_per_tok,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  code.e("kernel::moe_linear_sm100_task_impl<cute::bfloat16_t, TMA_A, "
         "decltype(mInput), decltype(mBias), decltype(mRoutingIndices), "
         "decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         /*no_bias*/ "true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    mInput,");
  code.e("    mBias,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_SM100, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_SM100, code.to_string());
  }
}

int TaskRegister::register_moe_fp8_sm100_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params,
                                              bool w13_linear) {
  assert(params.size() == 0);
  // Input ordering (6 inputs, 1 output):
  //   [0] input_fp8       [batch, K] or [batch, top_k, K]
  //   [1] input_scale     [batch, K/128] or [batch, top_k, K/128]
  //   [2] weight_fp8      [num_experts, N, K]
  //   [3] weight_scale    [num_experts, N, K/128]
  //   [4] routing_indices [num_experts, batch]
  //   [5] expert_mask     [num_experts+1]
  //   output              [batch, top_k, N]
  int num_inputs = 6;
  int num_outputs = 1;
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0;
  int output_size = 0, orig_output_size = 0, reduction_size = 0,
      output_stride = 0;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Output shape: [batch, top_k, N]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  // output_size is actually N // grid_dim.y
  output_size = output_ops[0]->output_tensors[0].dim[2];

  // Reduction size from input_fp8
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }

  // Weight: [num_experts, N, K]
  assert(input_ops[2]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[2]->output_tensors[0].dim[0];
  assert(input_ops[2]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[2]->output_tensors[0].dim[2] == reduction_size);

  // Routing indices: [num_experts, batch]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[4]->output_tensors[0].dim[1] == batch_size);

  // Mask: [num_experts+1]
  assert(input_ops[5]->output_tensors[0].num_dims == 1);
  assert(input_ops[5]->output_tensors[0].dim[0] == num_experts + 1);

  // Output stride
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[1]);
  orig_output_size = input_ops[2]->dtensor.dim[1];

  int k_scale = reduction_size / 128; // K/128 scale groups

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  // MMA constants (same as BF16 MoE task)
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bK = 128; // FP8: bK=128 for one scale-block per k-tile
  // NUM_AB_STAGE=8 (was 4): at TP=4, moe_w2_fp8 has fp8_k_tile_count=8 +
  // fp8_num_m_tiles=56 + fp8_num_n_tiles=4. With 4 stages the pipeline hung
  // (TP=4 mbt>=40 MoE hang). With 8 stages it passes at 26.9 ms/tok. Matches
  // BF16 moe_linear_sm100's existing pipeline depth. Total smem ≈ 148KB, fits
  // under the 205KB dynamic-smem budget. See project_tp4_moe_hang.md
  // (2026-04-22).
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = MMA_N * num_acc_stages; // 32
  assert(num_tmem_columns <= 512);

  // Expert stride: must match grid_dim.x so each CTA processes a distinct
  // set of experts. With grid_dim=(X, Y, 1), X CTAs handle expert distribution
  // (expert_offset = bid.x, stride = X) and Y CTAs split the N dimension.
  int expert_stride = bgraph.grid_dim.x;

  // TMA for FP8 weight (param_id=2, dtype=uint8_t→UINT8 format, bK=128 tile)
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  code.e("using TMA_Weight = kernel::tma::tma_2d<uint8_t, $, $, $, $, $, "
         "$, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         (num_experts - 1) * orig_output_size + output_size, /*GMEM_ROW_*/
         reduction_size,                                     /*GMEM_COL_*/
         MMA_M,                                              /*SMEM_ROW_*/
         bK,                                                 /*SMEM_COL_*/
         reduction_size, /*GMEM_STRIDE_ROW_*/
         1,              /*GMEM_STRIDE_COL_*/
         1,              /*SMEM_REPEAT_ROW_*/
         1,              /*SMEM_REPEAT_COL_*/
         MMA_M * bK      /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_Weight tma_weight(static_cast<CUtensorMap*>("
         "task_desc->input_tma_desc_ptrs[2][0]));");

  // Input FP8 activation tensor
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    // W2 input: tensor is [batch, topk, K] in memory with strides (topk*K, K,
    // 1). Kernel indexes mInput(batch, topk_idx-1, k_offset), so:
    //   dim 0 = batch (stride topk*K), dim 1 = topk (stride K), dim 2 = K
    //   (stride 1).
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, $, cute::Int<1>{}));",
           batch_size,
           num_experts_per_tok,
           reduction_size,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = cute::make_tensor("
         "cute::make_gmem_ptr(static_cast<uint8_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  // Input scale tensor [batch, K/128] or [batch, top_k, K/128]
  if (w13_linear) {
    code.e("cute::Layout layout_input_scale = cute::make_layout("
           "cute::make_shape($, $), cute::make_stride($, cute::Int<1>{}));",
           batch_size,
           k_scale,
           k_scale);
  } else {
    // W2 input scale: [batch, topk, K/128] with strides (topk*K_scale, K_scale,
    // 1).
    code.e("cute::Layout layout_input_scale = cute::make_layout("
           "cute::make_shape($, $, $), "
           "cute::make_stride($, $, cute::Int<1>{}));",
           batch_size,
           num_experts_per_tok,
           k_scale,
           num_experts_per_tok * k_scale,
           k_scale);
  }
  code.e("cute::Tensor mInputScale = cute::make_tensor("
         "cute::make_gmem_ptr(static_cast<float*>("
         "task_desc->input_ptrs[1])), layout_input_scale);");

  // Weight scale tensor — flat 2D view with strided expert access.
  // When grid_dim.y > 1, the runtime offsets the base pointer per bid.y.
  // Row count = (E-1)*orig_output_size + output_size: expert e's rows start
  // at offset e*orig_output_size, and only output_size rows per expert are
  // accessible from this CTA's base pointer. Same pattern as TMA GMEM_ROW.
  code.e("cute::Layout layout_weight_scale = cute::make_layout("
         "cute::make_shape($, $), cute::make_stride($, cute::Int<1>{}));",
         (num_experts - 1) * orig_output_size + output_size,
         k_scale,
         k_scale);
  code.e("cute::Tensor mWeightScale = cute::make_tensor("
         "cute::make_gmem_ptr(static_cast<float*>("
         "task_desc->input_ptrs[3])), layout_weight_scale);");

  // Routing indices [num_experts, batch]
  code.e("cute::Layout layout_routing_indices = cute::make_layout("
         "cute::make_shape($, $), cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = cute::make_tensor("
         "cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[4])), layout_routing_indices);");

  // Expert mask [num_experts+1]
  code.e("cute::Layout layout_expert_mask = cute::make_layout("
         "cute::make_shape($), cute::make_stride(cute::Int<1>{}));",
         num_experts + 1);
  code.e("cute::Tensor mMask = cute::make_tensor("
         "cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[5])), layout_expert_mask);");

  // Output tensor: kernel indexes mOutput(n_idx, topk_idx-1, m_idx)
  // Shape (batch, topk, output_size) with strides (topk*output_stride,
  // output_stride, 1) so that m_idx (output row) is contiguous within each
  // (batch, topk) slot.
  code.e("cute::Layout layout_output = cute::make_layout("
         "cute::make_shape($, $, $), "
         "cute::make_stride($, $, cute::Int<1>{}));",
         batch_size,
         num_experts_per_tok,
         output_size,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = cute::make_tensor("
         "cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");

  // Kernel call
  code.e("kernel::fp8_moe_group_gemm_sm100_task_impl<TMA_Weight, "
         "decltype(mInput), decltype(mInputScale), decltype(mWeightScale), "
         "decltype(mRoutingIndices), decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_weight,");
  code.e("    mInput,");
  code.e("    mInputScale,");
  code.e("    mWeightScale,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");

  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_FP8_SM100, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_FP8_SM100, code.to_string());
  }
}

int TaskRegister::register_moe_silu_mul_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params) {
  // params: [] (legacy) OR [active_mask_offset, ctas_per_expert, e_local]
  //                       (NEW MoE D3+B11).
  //   active_mask_offset == -1 -> meta input not supplied, no skip.
  //   active_mask_offset >= 0  -> meta is input_ptrs[1], active mask lives
  //                               at meta + active_mask_offset (int32),
  //                               my_expert = bid.x / ctas_per_expert.
  //                               B11: meta + active_mask_offset + e_local
  //                               holds per-expert actual_count (real row
  //                               count, ≤ BM_PADDING) — used to bound the
  //                               silu*mul loop.
  assert(params.size() == 0 || params.size() == 2 || params.size() == 3);
  int active_mask_offset = -1;
  int ctas_per_expert = 0;
  int e_local = 0;
  if (params.size() >= 2) {
    active_mask_offset = params[0];
    ctas_per_expert = params[1];
  }
  if (params.size() >= 3) {
    e_local = params[2];
  }
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = (active_mask_offset >= 0) ? 2 : 1;
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
  // Accept both 3D (batch, topk, intermediate) — OLD MoE path — and 2D
  // (M_total, intermediate) — NEW MoE path. For 2D we treat topk == 1.
  int const out_dims = output_ops[0]->output_tensors[0].num_dims;
  int const in_dims = input_ops[0]->output_tensors[0].num_dims;
  assert(out_dims == in_dims);
  assert(out_dims == 2 || out_dims == 3);
  if (out_dims == 3) {
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
    output_size = output_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[2] == output_size * 2);
  } else {
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    num_experts_per_tok = 1;
    output_size = output_ops[0]->output_tensors[0].dim[1];
    assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  }
  // get input/output strides (C20: stride[N-2] is the row-walk stride
  // regardless of rank; view-safe).
  if (out_dims == 3) {
    input_stride = static_cast<int>(input_ops[0]->dtensor.stride[1]);
    output_stride = static_cast<int>(output_ops[0]->dtensor.stride[1]);
  } else {
    input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
    output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
  }
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // For NEW MoE with active-mask: read mask + actual_count from meta. The
  // mask lives at meta[1, 0:E_LOCAL] (flat offset M_TOTAL+MBT*TOPK), the
  // per-expert actual_count lives immediately after at [E_LOCAL:2*E_LOCAL].
  // Both are written by moe_permute Phase 3.
  // For non-active-mask path (legacy), we still pass the compile-time
  // (num_experts_per_tok * batch_size) as the row bound.
  bool has_active = (active_mask_offset >= 0 && ctas_per_expert > 0);
  if (has_active) {
    // D3 + B11 (2026-05-15): per-CTA short-circuit when this CTA's
    // expert is inactive; otherwise read actual_count to cap the row
    // loop. For decode (active_token=1) each routed expert sees only
    // 1 real row, vs the BM_PADDING (=128) padded layout, so the
    // ROWS_PER_CTA work drops 128× on the active CTA.
    code.e("int const *active_mask_silu_ = "
           "static_cast<int const *>(task_desc->input_ptrs[1]) + $;",
           active_mask_offset);
    // active_mask_silu_[0..E_LOCAL-1]      = active flag
    // active_mask_silu_[E_LOCAL..2E_LOCAL] = actual_count
    code.e("int my_expert_silu_ = task_desc->task_metadata.request_id / $;",
           ctas_per_expert);
    code.e("if (!active_mask_silu_[my_expert_silu_]) return;");
    // actual_count_per_expert lives at meta + active_mask_offset + e_local.
    // For B11 we cap silu*mul's row count to actual_count instead of
    // BM_PADDING (=128). Decode iter: actual_count ~= 1 (8 active
    // experts/iter, 1 routed row each), vs 128 padded rows.
    code.e("int silu_rows_ = active_mask_silu_[$ + my_expert_silu_];", e_local);
    code.e("if (silu_rows_ <= 0) return;");
    code.e("if (silu_rows_ > $) silu_rows_ = $;", batch_size, batch_size);
  }
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  if (has_active) {
    code.e("    silu_rows_);");
  } else {
    code.e("    $);", num_experts_per_tok * batch_size);
  }
  return register_task_variant(TASK_SILU_MUL, code.to_string());
}

int TaskRegister::register_moe_mul_sum_add_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
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
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[2] ==
             input_ops[2]->output_tensors[0].dim[1] &&
         input_ops[0]->output_tensors[0].dim[2] == output_size);
  // get input stride
  input_stride = static_cast<int>(input_ops[0]->dtensor.stride[1]);
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mul_sum_add_sm100_task_impl<cute::bfloat16_t, $, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*NUM_TOPK=*/num_experts_per_tok,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_MOE_MUL_SUM_ADD_SM100, code.to_string());
}

int TaskRegister::register_moe_linear_sm90_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      orig_output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
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
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // get output stride
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[1]);
  orig_output_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // MoE constant:
  int const expert_stride = w13_linear ? 5 : 4;
  // define MMA
  constexpr int MMA_M = 64;
  constexpr int MMA_N = 16;
  constexpr int num_ab_stages = 8;
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  // int const output_tma_cp_size = 128;
  // int const output_atom_size = 128;
  // TMA_B for expert weights
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         //  (num_experts-1) * orig_output_size + output_size, /*GMEM_ROW_*/
         (num_experts)*orig_output_size, /*GMEM_ROW_*/
         reduction_size,                 /*GMEM_COL_*/
         MMA_M,                          /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,              /*SMEM_COL_*/
         reduction_size,                 /*GMEM_STRIDE_ROW_*/
         1,                              /*GMEM_STRIDE_COL_*/
         1,                              /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  // Bias Tensor setup
  code.e(
      "cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $, $), "
      "cute::make_stride($, cute::Int<1>{}, $));",
      batch_size,
      output_size,
      num_experts,
      output_stride,
      output_stride * batch_size);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         "nullptr");
  // Topk_indices Tensor setup
  code.e("cute::Layout layout_routing_indices = "
         "cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[2])), layout_routing_indices);");
  // Topk_mask Tensor setup
  code.e("cute::Layout layout_expert_mask = "
         "cute::make_layout(cute::make_shape($), "
         "cute::make_stride(cute::Int<1>{}));",
         num_experts);
  code.e("cute::Tensor mMask = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[3])), layout_expert_mask);");
  // Output Tensor setup
  code.e("cute::Layout layout_output = cute::make_layout(cute::make_shape($, "
         "$, $), "
         "cute::make_stride($, cute::Int<1>{}, $));",
         batch_size,
         output_size,
         num_experts_per_tok,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");
  // Input Tensor setup
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, cute::Int<1>{}, $));",
           batch_size,
           reduction_size,
           num_experts_per_tok,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  code.e("kernel::moe_linear_sm90_task_impl<cute::bfloat16_t, TMA_A, "
         "decltype(mInput), decltype(mBias), decltype(mRoutingIndices), "
         "decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, "
         "$>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         /*no_bias*/ "true",
         num_ab_stages);
  code.e("    tma_a,");
  code.e("    mInput,");
  code.e("    mBias,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_SM90, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_SM90, code.to_string());
  }
}

int TaskRegister::register_splitk_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  assert(with_residual == false);
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
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16;
  // int const SMEM_M_SIZE = 16;
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
      "TMA_OUT, $, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      with_residual ? "TMA_RESIDUAL" : "void",
      output_stride,
      "true" /*SplitK*/);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  return register_task_variant(TASK_SPLITK_LINEAR_SWAPAB_HOPPER,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
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
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_split_kv_task_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         num_kv_heads,
         kv_stride,
         qkv_stride,
         output_size * num_kv_chunks, // o_stride should consider num_kv_chunks
         head_dim,
         SEQ_LEN_PER_BLOCK,
         max_seq_len,
         page_size,
         max_tokens,
         "true", // PARTITION_KV
         num_kv_chunks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_SM100,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_merge_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  assert(params.size() == 5);
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
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  code.e("kernel::merge_splitkv<bfloat16, $, $, $, $, $, $, "
         "$, $, $>(",
         num_q_heads_per_kv,
         1,
         num_kv_heads,
         head_dim,
         max_tokens,
         true,
         (max_seq_len / SEQ_LEN_PER_BLOCK),
         SEQ_LEN_PER_BLOCK,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_decode_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_heads (e.g. 128)
  // params[1]: d_k (e.g. 576)
  // params[2]: d_v (e.g. 512)
  // params[3]: num_splits
  // params[4]: kv_len (max, not used — runtime kv_len from page table)
  // params[5]: q_len (number of queries per block, for prefill batching;
  //                   default 1 for decode-only)
  assert(params.size() >= 5 && params.size() <= 6);
  int num_heads = params[0];
  int d_k = params[1];
  int d_v = params[2];
  int num_splits = params[3];
  int q_len = (params.size() >= 6) ? params[5] : 1;
  // num_head_groups derived from q_len: each block handles q_len queries × hpb
  // heads. TP-aware: hpb = min(128/q_len, num_heads). In single-GPU
  // (num_heads=128), this equals 128/q_len (original behavior). In TP, caps hpb
  // at local heads. Kernel uses local_num_heads (=num_heads here) for indexing.
  int hpb = std::min(128 / q_len, num_heads);
  while (hpb > 0 && num_heads % hpb != 0) {
    hpb--;
  }
  if (hpb <= 0) {
    hpb = 1;
  }
  int num_head_groups = num_heads / hpb;
  // q_len=1 keeps the legacy mapping where request_id is batch. For q_len>1,
  // grid.y is head group and grid.z is batch (stored in merge_task_offset).
  bool const single_query = (q_len == 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Compute kv_len and q_len from page/qo indptrs at runtime.
  // Q_LEN is dynamic: prefill iters have Q_LEN=mbt new tokens, decode has
  // Q_LEN=1. The TMA descriptor's box height is fixed at compile time
  // (hpb=128/Q_LEN_COMPILE), so runtime Q_LEN must be ≤ compile-time Q_LEN.
  // Kernel only iterates q<Q_LEN_RT. The causal mask uses Q_LEN_RT to compute
  // correct causal_limit per query.
  code.e("{");
  if (single_query) {
    code.e("  int bi_ = task_desc->task_metadata.request_id;");
  } else {
    code.e("  int bi_ = task_desc->task_metadata.merge_task_offset;");
    code.e(
        "  int gi_ = task_desc->task_metadata.request_id;  // head group idx");
  }
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  if (!single_query) {
    code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
    code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
    code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;"); // actual new tokens this iter
    code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
    code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  }
  // Use PR 651 MLA MTP decode kernel (supports Q_LEN=1..mbt prefill batching)
  code.e("  kernel::mla_mtp_decode_sm100_task_impl<false, false>(");
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // Q
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),"); // KV
  code.e(
      "      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),"); // Oa
                                                                      // (bf16)
  code.e("      static_cast<float*>(task_desc->output_ptrs[1]),");    // La
  // DeepSeek V3 MLA softmax_scale = q_head_dim^-0.5 * mscale^2
  //   q_head_dim = 128 (qk_nope) + 64 (qk_rope) = 192  (NOT 576!)
  //   mscale = 0.1 * mscale_all_dim(1.0) * log(scaling_factor=40) + 1.0
  //          ≈ 1.36889; mscale^2 ≈ 1.87385
  //   sm_scale = 1/sqrt(192) * 1.87385 ≈ 0.13525
  // (See modeling_deepseek.py:689-695. d_k=576 is the absorbed latent dim,
  // not the original head dim used to scale the dot product.)
  {
    float const _mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
    float const _sm = (1.0f / sqrtf(192.0f)) * _mscale * _mscale;
    code.e("      $f,", _sm); // softmax scale (DeepSeek V3 YARN-adjusted)
  }
  code.e("      kv_len_,");            // kv_len from runtime
  code.e("      sk_rt_,");             // runtime-effective sk
  code.e("      $,", num_head_groups); // num_head_groups
  if (single_query) {
    code.e("      $,", q_len); // Q_LEN (compile-time 1)
    code.e("      0,");        // gi (head group 0)
  } else {
    code.e("      q_len_rt_,"); // Q_LEN (runtime)
    code.e("      gi_,");       // gi (from request_id)
  }
  code.e("      (int)task_desc->task_metadata.kv_idx,"); // si (split_idx)
  code.e("      bi_,");                                  // bi (batch_idx)
  code.e("      $);", num_heads); // local_num_heads (TP-aware)
  code.e("}");
  return register_task_variant(TASK_MLA_DECODE_SM100, code.to_string());
}

int TaskRegister::register_mla_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_heads (e.g. 128)
  // params[1]: d_v (e.g. 512)
  // params[2]: num_splits
  // params[3]: d_start (start dim index for this task)
  // params[4]: d_count (num dims this task handles)
  // params[5]: q_len (number of queries per block; default 1 for decode)
  assert(params.size() >= 5 && params.size() <= 6);
  int num_heads = params[0];
  int d_v = params[1];
  int num_splits = params[2];
  int d_start = params[3];
  int d_count = params[4];
  int q_len = (params.size() >= 6) ? params[5] : 1;
  // Match decode kernel head_group derivation (TP-aware).
  int hpb = std::min(128 / q_len, num_heads);
  while (hpb > 0 && num_heads % hpb != 0) {
    hpb--;
  }
  if (hpb <= 0) {
    hpb = 1;
  }
  int num_head_groups = num_heads / hpb;
  bool const single_query = (q_len == 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // PR 651 MLA MTP reduce kernel (256 threads for MPK)
  if (!single_query) {
    // Use runtime Q_LEN from qo_indptr so reduce output layout matches what the
    // decode kernel produced for this iter (1 row for decode, mbt rows for
    // prefill).
    code.e("{");
    code.e("  int gi_ = task_desc->task_metadata.request_id;");
    code.e("  int bi_ = task_desc->task_metadata.merge_task_offset;");
    code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
    code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
    code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
    code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
    code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  }
  code.e("kernel::mla_mtp_reduce_sm100_task_impl<256>(");
  code.e(
      "    static_cast<const nv_bfloat16*>(task_desc->input_ptrs[0]),"); // Oa
                                                                         // (bf16)
  code.e("    static_cast<const float*>(task_desc->input_ptrs[1]),");  // La
  code.e("    static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),"); // O
  code.e("    $,", num_splits);                                        // sk
  code.e("    $,", num_head_groups); // num_head_groups
  if (single_query) {
    code.e("    $,", q_len); // Q_LEN (compile-time 1)
  } else {
    code.e("    q_len_rt_,"); // Q_LEN (runtime)
  }
  code.e("    $,", d_start); // dv_base
  if (single_query) {
    code.e("    0,"); // gi (head group 0)
    code.e("    (int)task_desc->task_metadata.request_id,"); // bi
    code.e("    $);", num_heads); // local_num_heads (TP-aware)
  } else {
    code.e("    gi_,");
    code.e("    bi_,");
    code.e("    $);", num_heads); // local_num_heads (TP-aware)
    code.e("}");
  }
  return register_task_variant(TASK_MLA_REDUCE_SM100, code.to_string());
}

int TaskRegister::register_mla_prefill_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_heads (e.g. 128)
  // params[1]: seq_len
  // params[2]: d_ckv (e.g. 512)
  // params[3]: d_kpe (e.g. 64)
  // params[4]: d_v (e.g. 512)
  assert(params.size() == 5);
  int num_heads = params[0];
  int seq_len = params[1];
  int d_ckv = params[2];
  int d_kpe = params[3];
  int d_v = params[4];
  // DeepSeek V3 MLA softmax_scale = q_head_dim^-0.5 * mscale^2 (≈ 0.13525)
  // d_ckv+d_kpe (576) is the absorbed latent dim, NOT the dot-product scale.
  // q_head_dim = 192 (qk_nope=128 + qk_rope=64); mscale from YARN.
  float const _mscale_pf = 0.1f * 1.0f * logf(40.0f) + 1.0f;
  float sm_scale = (1.0f / sqrtf(192.0f)) * _mscale_pf * _mscale_pf;
  (void)d_ckv;
  (void)d_kpe;
  float sm_scale_log2 = sm_scale * 1.44269504089f;

  // MLA prefill: grid = (H, num_q_blocks, B)
  // task_metadata.request_id = batch (bid.z)
  // task_metadata.kv_idx = q_block (bid.y)
  // task_metadata.merge_task_offset = head (bid.x)
  //
  // Inputs: Q_nope [S,H,D_CKV], Q_pe [S,H,D_KPE], CKV [S,D_CKV], KPE [S,D_KPE]
  // Output: O [S,H,D_V]
  // The kernel itself is single-request; we slice each request's Q / KV /
  // output window before calling it.

  (void)seq_len;
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Compute S (total KV length) and Q_LEN (this iteration's chunk length)
  // at runtime. S = pages*page_size + last_page_len, grows as prefill
  // progresses. Q_LEN = qo_indptr[bi+1] - qo_indptr[bi] = num_new_tokens
  // this iteration (can be smaller than mbt).
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int head_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int Q_LEN_ = runtime_config.qo_indptr_buffer[bi_ + 1] - "
         "runtime_config.qo_indptr_buffer[bi_];");
  code.e("  if (req_id_ < 0 || runtime_config.step[req_id_] >= "
         "runtime_config.prompt_length[req_id_] || Q_LEN_ <= 8) return;");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int S_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  auto *q_nope_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[0]) + "
         "qo_fp_ * $;",
         num_heads * d_ckv);
  code.e("  auto *q_pe_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[1]) + "
         "qo_fp_ * $;",
         num_heads * d_kpe);
  code.e("  auto *ckv_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[2]) + "
         "bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_ckv);
  code.e("  auto *kpe_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[3]) + "
         "bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_kpe);
  code.e("  auto *out_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[4]) + "
         "qo_fp_ * $;",
         num_heads * d_v);
  code.e("  kernel::mla_prefill_sm100_task_impl(");
  code.e("      q_nope_ptr_,");
  code.e("      q_pe_ptr_,");
  code.e("      ckv_ptr_,");
  code.e("      kpe_ptr_,");
  // O attached via new_input(store_in_dmem=True) on the Python side (MPK
  // convention shared with mla_decode_layer), so use input_ptrs[4].
  code.e("      out_ptr_,");
  code.e("      S_,");                // runtime S
  code.e("      Q_LEN_,");            // runtime Q_LEN
  code.e("      $,", num_heads);      // H
  code.e("      $f,", sm_scale_log2); // sm_scale_log2
  code.e("      head_,");
  code.e("      task_desc->task_metadata.kv_idx);"); // q_block
  code.e("}");
  return register_task_variant(TASK_MLA_PREFILL_SM100, code.to_string());
}

int TaskRegister::register_mla_prefill_absorbed_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 5);
  int num_heads = params[0];
  int seq_len = params[1];
  int d_ckv = params[2];
  int d_kpe = params[3];
  int d_v = params[4];
  assert(d_ckv == 512);
  assert(d_kpe == 64);
  assert(d_v == 512);
  (void)seq_len;
  float const _mscale_pf = 0.1f * 1.0f * logf(40.0f) + 1.0f;
  float sm_scale = (1.0f / sqrtf(192.0f)) * _mscale_pf * _mscale_pf;
  float sm_scale_log2 = sm_scale * 1.44269504089f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int head_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int Q_LEN_ = runtime_config.qo_indptr_buffer[bi_ + 1] - "
         "runtime_config.qo_indptr_buffer[bi_];");
  code.e("  if (req_id_ < 0 || runtime_config.step[req_id_] >= "
         "runtime_config.prompt_length[req_id_] || Q_LEN_ <= 8) return;");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int S_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  auto *q_ptr_ = static_cast<const nv_bfloat16 *>("
         "task_desc->input_ptrs[0]) + qo_fp_ * $;",
         num_heads * (d_ckv + d_kpe));
  code.e("  auto *kv_ptr_ = static_cast<const nv_bfloat16 *>("
         "task_desc->input_ptrs[1]) + bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_ckv + d_kpe);
  code.e("  auto *out_ptr_ = static_cast<nv_bfloat16 *>("
         "task_desc->output_ptrs[0]) + qo_fp_ * $;",
         num_heads * d_v);
  code.e("  kernel::mla_prefill_sm100_task_impl(");
  code.e("      q_ptr_,");
  code.e("      q_ptr_,");
  code.e("      kv_ptr_,");
  code.e("      kv_ptr_,");
  code.e("      out_ptr_,");
  code.e("      S_,");
  code.e("      Q_LEN_,");
  code.e("      $,", num_heads);
  code.e("      $f,", sm_scale_log2);
  code.e("      head_,");
  code.e("      task_desc->task_metadata.kv_idx,");
  code.e("      $,", num_heads * (d_ckv + d_kpe));
  code.e("      $,", d_ckv + d_kpe);
  code.e("      $,", num_heads * (d_ckv + d_kpe));
  code.e("      $,", d_ckv + d_kpe);
  code.e("      $,", d_ckv + d_kpe);
  code.e("      $,", d_ckv + d_kpe);
  code.e("      $);", d_ckv);
  code.e("}");
  return register_task_variant(TASK_MLA_PREFILL_SM100, code.to_string());
}

int TaskRegister::register_mla_prefill_tp8_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_heads per TP rank (e.g. 16 for TP=8)
  // params[1]: seq_len (must be <= 4096)
  // Inputs: [0] Q_nope [B,S,H,128], [1] Q_pe [B,S,H,64],
  //         [2] K [B,S,192] (nope+rope concat), [3] V [B,S,128]
  // Output: [0] O [B,S,H,128]
  // TMA descriptors for K (input_tma_desc_ptrs[2][0]) and V
  // (input_tma_desc_ptrs[3][0]).
  assert(params.size() == 2);
  int num_heads = params[0];
  int seq_len = params[1];
  float sm_scale = 1.0f / sqrtf(192.0f);
  float sm_scale_log2 = sm_scale * 1.44269504089f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mla_prefill_tp8::mla_prefill_tp8_sm100_task_impl(");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // K TMA
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[3][0]),"); // V TMA
  code.e("    static_cast<const "
         "__nv_bfloat16*>(task_desc->input_ptrs[0]),"); // Qn
  code.e("    static_cast<const "
         "__nv_bfloat16*>(task_desc->input_ptrs[1]),");                  // Qp
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),"); // O
  code.e("    $,", seq_len);                                             // S
  code.e("    $,", num_heads);                                           // H
  code.e("    $f,", sm_scale_log2);                                      // sml2
  code.e("    task_desc->task_metadata.request_id,");         // head (bid.x)
  code.e("    task_desc->task_metadata.kv_idx,");             // q_block (bid.y)
  code.e("    task_desc->task_metadata.merge_task_offset);"); // batch (bid.z)
  return register_task_variant(TASK_MLA_PREFILL_TP8_SM100, code.to_string());
}

int TaskRegister::register_mla_prefill_tp8_chunked_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params: num_heads, q_len, kv_len, q_start [, qfused_mode]
  //   qfused_mode (optional, default 0):
  //     0 = legacy split-buffer layout: input_ptrs[0] = Qn [B, q_len, H, 128],
  //         input_ptrs[1] = Qp [B, q_len, H, 64]. Per-head strides 128 / 64.
  //     1 = fused-Q layout: input_ptrs[0] = Q_fused [B, q_len, H, 192] starting
  //         at nope, input_ptrs[1] = same fused tensor pointer (the kernel
  //         reads Qp from input_ptrs[0] + D_QK_NOPE per-element offset).
  //         Per-head strides 192 / 192. Builder must concatenate
  //         q_b_nope+q_b_pe weights and emit a single FP8 GEMM into a (mbt,
  //         H*192) buffer.
  assert(params.size() == 4 || params.size() == 5);
  int num_heads = params[0];
  int q_len_max = params[1];
  int kv_len_max = params[2];
  int q_start = params[3];
  int qfused_mode = (params.size() == 5) ? params[4] : 0;
  float sm_scale = 1.0f / sqrtf(192.0f);
  float sm_scale_log2 = sm_scale * 1.44269504089f;
  // FuseTensor row-swap layout (2026-05-12 user #2 v2): when qfused_mode=1,
  // weight is rearranged at load time as [all_heads_nope; all_heads_pe]
  // per-rank, so the fused output buffer per row has layout
  //   [head0_nope(128), head1_nope(128), ..., head_{H-1}_nope(128),
  //    head0_pe(64),    head1_pe(64),    ..., head_{H-1}_pe(64)]
  // Per-head strides stay 128 / 64 (matching the LEGACY layout for Qn and
  // Qp slices), but the row stride differs: a row of the fused buffer is
  // H * 192 elements wide, so per-qi advance is H * 192 for BOTH Qn and Qp.
  // Qp_ptr starts at offset H * D_QK_NOPE (= H * 128 elements) from Qn_ptr —
  // the start of the pe region within each row.
  //
  // FP8 block alignment: with the row-swap layout, 128-row FP8 blocks fall
  // cleanly: blocks 0..H-1 are each one head's nope; blocks H..(H+H/2) are
  // 2-heads-per-block pe (same magnitude class). No more nope/pe mixing.
  int qn_head_stride = 128; // D_QK_NOPE — same for both modes
  int qp_head_stride = 64;  // D_QK_ROPE — same for both modes
  // Row stride: legacy split = H * head_stride (default sentinel 0); fused
  // row-swap = H * (D_QK_NOPE + D_QK_ROPE) = H * 192 for BOTH Qn and Qp.
  int qn_row_stride = qfused_mode == 1 ? num_heads * 192 : 0;
  int qp_row_stride = qfused_mode == 1 ? num_heads * 192 : 0;
  // Qp offset within the fused buffer (in bf16 elements). For row-swap:
  // Qp region starts at H * D_QK_NOPE in each row, so Qp_ptr = Qn_ptr + H*128.
  int qp_offset_elems = qfused_mode == 1 ? num_heads * 128 : 0;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("#ifdef MPK_TEST_MODE");
  code.e("kernel::mla_prefill_tp8_chunked::"
         "mla_prefill_tp8_chunked_sm100_task_impl(");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // K_nope
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[3][0]),"); // K_rope
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[4][0]),"); // V
  if (qfused_mode == 1) {
    // Qn = fused base (start of nope region);
    // Qp = fused base + H * D_QK_NOPE (start of pe region within row)
    code.e("    static_cast<const "
           "__nv_bfloat16*>(task_desc->input_ptrs[0]),"); // Qn = fused
    code.e("    static_cast<const "
           "__nv_bfloat16*>(task_desc->input_ptrs[0]) + $,",
           qp_offset_elems); // Qp = Qn + H*128
  } else {
    code.e("    static_cast<const "
           "__nv_bfloat16*>(task_desc->input_ptrs[0]),"); // Qn
    code.e("    static_cast<const "
           "__nv_bfloat16*>(task_desc->input_ptrs[1]),"); // Qp
  }
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),"); // O
  code.e("    $,", q_len_max);
  code.e("    $,", kv_len_max);
  code.e("    $,", q_start);
  code.e("    $,", num_heads);
  code.e("    $f,", sm_scale_log2);
  code.e("    task_desc->task_metadata.request_id,");        // head
  code.e("    task_desc->task_metadata.kv_idx,");            // q_block
  code.e("    task_desc->task_metadata.merge_task_offset,"); // batch
  code.e("    $,", qn_head_stride);
  code.e("    $,", qp_head_stride);
  code.e("    $,", qn_row_stride);
  code.e("    $);", qp_row_stride);
  code.e("#else");
  code.e("{");
  code.e("int bi_ = task_desc->task_metadata.merge_task_offset;");
  code.e("int req_id_ = runtime_config.request_ids[bi_];");
  code.e("if (req_id_ < 0) return;");
  code.e("int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("int Q_LEN_ = runtime_config.qo_indptr_buffer[bi_ + 1] - "
         "runtime_config.qo_indptr_buffer[bi_];");
  code.e("bool prompt_prefill_ = runtime_config.step[req_id_] < "
         "runtime_config.prompt_length[req_id_] && Q_LEN_ > 8;");
  code.e("if (!prompt_prefill_) return;");
  code.e("int q_blocks_ = (Q_LEN_ + 63) / 64;");
  code.e("if (task_desc->task_metadata.kv_idx >= q_blocks_) return;");
  code.e("int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("int KV_LEN_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("int Q_START_ = KV_LEN_ - Q_LEN_;");
  if (qfused_mode == 1) {
    // Row-swap fused buffer: row stride = num_heads * 192;
    // Qp_ region starts at num_heads * D_QK_NOPE within the row.
    code.e("auto *Qn_ = static_cast<const __nv_bfloat16*>("
           "task_desc->input_ptrs[0]) + qo_fp_ * $;",
           num_heads * 192);
    code.e("auto *Qp_ = Qn_ + $;", qp_offset_elems); // = num_heads * 128
  } else {
    code.e("auto *Qn_ = static_cast<const __nv_bfloat16*>("
           "task_desc->input_ptrs[0]) + qo_fp_ * $;",
           num_heads * 128);
    code.e("auto *Qp_ = static_cast<const __nv_bfloat16*>("
           "task_desc->input_ptrs[1]) + qo_fp_ * $;",
           num_heads * 64);
  }
  code.e("auto *O_ = static_cast<__nv_bfloat16*>("
         "task_desc->output_ptrs[0]) + qo_fp_ * $;",
         num_heads * 128);
  code.e("kernel::mla_prefill_tp8_chunked::"
         "mla_prefill_tp8_chunked_sm100_task_impl(");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // K_nope
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[3][0]),"); // K_rope
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[4][0]),"); // V
  code.e("    Qn_,");                                             // Qn
  code.e("    Qp_,");                                             // Qp
  code.e("    O_,");                                              // O
  code.e("    Q_LEN_,");
  code.e("    KV_LEN_,");
  code.e("    Q_START_,");
  code.e("    $,", num_heads);
  code.e("    $f,", sm_scale_log2);
  code.e("    task_desc->task_metadata.request_id,"); // head
  code.e("    task_desc->task_metadata.kv_idx,");     // q_block
  code.e("    0,"); // batch offset already applied via Qn_/Qp_/O_
  code.e("    $,", qn_head_stride);
  code.e("    $,", qp_head_stride);
  code.e("    $,", qn_row_stride);
  code.e("    $);", qp_row_stride);
  code.e("}");
  code.e("#endif");
  return register_task_variant(TASK_MLA_PREFILL_TP8_CHUNKED_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_prefill_tp8_chunked_splitk_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params: num_heads, q_len, kv_len, q_start, num_splits, nqb
  assert(params.size() == 6);
  int num_heads = params[0];
  int q_len = params[1];
  int kv_len = params[2];
  int q_start = params[3];
  int num_splits = params[4];
  int nqb = params[5];
  float sm_scale = 1.0f / sqrtf(192.0f);
  float sm_scale_log2 = sm_scale * 1.44269504089f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mla_prefill_tp8_chunked_splitk::"
         "mla_prefill_tp8_chunked_splitk_sm100_task_impl(");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // K_nope
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[3][0]),"); // K_rope
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[4][0]),"); // V
  code.e("    static_cast<const "
         "__nv_bfloat16*>(task_desc->input_ptrs[0]),"); // Qn
  code.e("    static_cast<const "
         "__nv_bfloat16*>(task_desc->input_ptrs[1]),");          // Qp
  code.e("    static_cast<float*>(task_desc->output_ptrs[0]),"); // partial
  code.e("    $,", q_len);
  code.e("    $,", kv_len);
  code.e("    $,", q_start);
  code.e("    $,", num_heads);
  code.e("    $,", num_splits);
  code.e("    $,", nqb);
  code.e("    $f,", sm_scale_log2);
  code.e("    task_desc->task_metadata.request_id,");         // head
  code.e("    task_desc->task_metadata.kv_idx,");             // packed yidx
  code.e("    task_desc->task_metadata.merge_task_offset);"); // batch
  return register_task_variant(TASK_MLA_PREFILL_TP8_CHUNKED_SPLITK_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_prefill_tp8_chunked_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params: num_heads, q_len, num_splits, nqb
  assert(params.size() == 4);
  int num_heads = params[0];
  int q_len = params[1];
  int num_splits = params[2];
  int nqb = params[3];
  float sm_scale = 1.0f / sqrtf(192.0f);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mla_prefill_tp8_chunked_splitk::"
         "mla_prefill_tp8_chunked_reduce_sm100_task_impl(");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    $,", q_len);
  code.e("    $,", num_heads);
  code.e("    $,", num_splits);
  code.e("    $,", nqb);
  code.e("    $f,", sm_scale);
  code.e("    task_desc->task_metadata.request_id,");         // head
  code.e("    task_desc->task_metadata.kv_idx,");             // q_block
  code.e("    task_desc->task_metadata.merge_task_offset);"); // batch
  return register_task_variant(TASK_MLA_PREFILL_TP8_CHUNKED_REDUCE_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_unified_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_heads local to this TP rank
  // params[1]: max decode q_len for this compiled graph
  // params[2]: max kv_len
  // params[3]: num_splits
  // params[4]: tp_size (1, 2, 4, or 8)
  // params[5]: d_ckv
  // params[6]: d_kpe
  // params[7]: d_v
  assert(params.size() == 8);
  bool const direct_paged_decode_kv = graph_input_has_num_dims(bgraph, 6, 3);
  int num_heads = params[0];
  int q_len = params[1];
  int kv_len = params[2];
  int num_splits = params[3];
  int tp_size = params[4];
  int d_ckv = params[5];
  int d_kpe = params[6];
  int d_v = params[7];
  assert(tp_size == 1 || tp_size == 2 || tp_size == 4 || tp_size == 8);

  int kvt = (kv_len + 128 - 1) / 128;
  int tps = (kvt + num_splits - 1) / num_splits;
  int single_tile = (tps == 1) ? 1 : 0;
  if (std::getenv("MPK_MLA_DISABLE_SINGLE_TILE")) {
    single_tile = 0;
  }
  bool const write_final = (num_splits == 1);

  int q_len_padded = (tp_size == 8) ? ((q_len + 1) & ~1) : q_len;
  int qpg = 1;
  int num_decode_groups = 1;
  if (tp_size == 1) {
    int hpb = num_heads / q_len;
    if (hpb < 1) {
      hpb = 1;
    }
    while (num_heads % hpb != 0) {
      hpb -= 1;
    }
    num_decode_groups = num_heads / hpb;
  } else if (tp_size == 2) {
    qpg = (q_len < 2) ? q_len : 2;
    num_decode_groups = (q_len + qpg - 1) / qpg;
  } else if (tp_size == 4) {
    qpg = (q_len < 4) ? q_len : 4;
    num_decode_groups = (q_len + qpg - 1) / qpg;
  } else {
    qpg = 2;
    num_decode_groups = (q_len_padded + qpg - 1) / qpg;
  }

  float const _mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
  float const sm_scale = (1.0f / sqrtf(192.0f)) * _mscale * _mscale;
  float const sm_scale_log2 = sm_scale * 1.44269504089f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int meta_x_ = task_desc->task_metadata.kv_idx;");
  code.e("  int meta_y_ = task_desc->task_metadata.request_id;");
  code.e("  int meta_z_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int qo_fp_prefill_ = 0;");
  code.e("  int prefill_s_ = 0;");
  code.e("  int prefill_q_len_ = 0;");
  code.e("  int prefill_bi_ptr_ = 0;");
  code.e("  bool prompt_prefill_ = false;");
  code.e("  if (meta_z_ >= 0 && meta_z_ < MPK_MAX_NUM_BATCHED_REQUESTS) {");
  code.e("    prefill_bi_ptr_ = meta_z_;");
  code.e("    qo_fp_prefill_ = runtime_config.qo_indptr_buffer[meta_z_];");
  code.e("    int fp_ = runtime_config.paged_kv_indptr_buffer[meta_z_];");
  code.e("    int lp_ = runtime_config.paged_kv_indptr_buffer[meta_z_ + 1];");
  code.e("    prefill_s_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[meta_z_];");
  code.e("    prefill_q_len_ = runtime_config.qo_indptr_buffer[meta_z_ + 1] - "
         "runtime_config.qo_indptr_buffer[meta_z_];");
  code.e("    int req_id_ = runtime_config.request_ids[meta_z_];");
  code.e("    if (req_id_ >= 0) {");
  code.e("      prompt_prefill_ = "
         "runtime_config.step[req_id_] < "
         "runtime_config.prompt_length[req_id_] && prefill_q_len_ > 8;");
  code.e("    }");
  code.e("  }");
  code.e("  int decode_kv_len_ = 0;");
  code.e("  int decode_q_len_ = 0;");
  code.e("  int decode_sk_rt_ = 1;");
  code.e("  int decode_first_page_pos_ = 0;");
  code.e("  if (meta_y_ >= 0 && meta_y_ < MPK_MAX_NUM_BATCHED_REQUESTS) {");
  code.e("    int fp_ = runtime_config.paged_kv_indptr_buffer[meta_y_];");
  code.e("    int lp_ = runtime_config.paged_kv_indptr_buffer[meta_y_ + 1];");
  code.e("    decode_first_page_pos_ = fp_;");
  code.e("    decode_kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[meta_y_];");
  code.e("    int kvt_rt_ = (decode_kv_len_ + 127) / 128;");
  code.e("    if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e(
      "    decode_sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  code.e("    decode_q_len_ = runtime_config.qo_indptr_buffer[meta_y_ + 1] - "
         "runtime_config.qo_indptr_buffer[meta_y_];");
  code.e("    if (decode_q_len_ < 1) decode_q_len_ = 1;");
  code.e("    if (decode_q_len_ > $) decode_q_len_ = $;", q_len, q_len);
  code.e("  }");
  code.e("  int decode_q_len_padded_ = decode_q_len_ + "
         "((decode_q_len_ & 1) * $);",
         (tp_size == 8) ? 1 : 0);
  code.e("  auto *q_nope_ptr_ = static_cast<const nv_bfloat16 *>("
         "task_desc->input_ptrs[0]) + qo_fp_prefill_ * $;",
         num_heads * d_ckv);
  code.e("  auto *q_pe_ptr_ = static_cast<const nv_bfloat16 *>("
         "task_desc->input_ptrs[1]) + qo_fp_prefill_ * $;",
         num_heads * d_kpe);
  code.e("  auto *ckv_ptr_ = static_cast<const nv_bfloat16 *>("
         "task_desc->input_ptrs[2]) + prefill_bi_ptr_ * "
         "MPK_MAX_SEQ_LENGTH * $;",
         d_ckv);
  code.e("  auto *kpe_ptr_ = static_cast<const nv_bfloat16 *>("
         "task_desc->input_ptrs[3]) + prefill_bi_ptr_ * "
         "MPK_MAX_SEQ_LENGTH * $;",
         d_kpe);
  code.e("  auto *out_ptr_ = static_cast<nv_bfloat16 *>("
         "task_desc->input_ptrs[4]) + qo_fp_prefill_ * $;",
         num_heads * d_v);
  if (single_tile) {
    if (tp_size == 1) {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<true, $, 1>(",
             write_final ? "true" : "false");
    } else if (tp_size == 2) {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<true, $, 2>(",
             write_final ? "true" : "false");
    } else if (tp_size == 4) {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<true, $, 4>(",
             write_final ? "true" : "false");
    } else {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<true, $, 8>(",
             write_final ? "true" : "false");
    }
  } else {
    if (tp_size == 1) {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<false, $, 1>(",
             write_final ? "true" : "false");
    } else if (tp_size == 2) {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<false, $, 2>(",
             write_final ? "true" : "false");
    } else if (tp_size == 4) {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<false, $, 4>(",
             write_final ? "true" : "false");
    } else {
      code.e("  kernel::mla_unified_sm100::"
             "mla_unified_sm100_task_impl<false, $, 8>(",
             write_final ? "true" : "false");
    }
  }
  code.e("      q_nope_ptr_,");
  code.e("      q_pe_ptr_,");
  code.e("      ckv_ptr_,");
  code.e("      kpe_ptr_,");
  code.e("      out_ptr_,");
  code.e("      static_cast<const CUtensorMap *>("
         "task_desc->input_tma_desc_ptrs[5][0]),");
  code.e("      static_cast<const CUtensorMap *>("
         "task_desc->input_tma_desc_ptrs[6][0]),");
  code.e("      static_cast<nv_bfloat16 *>(task_desc->output_ptrs[0]),");
  code.e("      static_cast<float *>(task_desc->output_ptrs[1]),");
  code.e("      prefill_s_,");
  code.e("      decode_kv_len_,");
  code.e("      prefill_q_len_,");
  code.e("      decode_q_len_,");
  code.e("      decode_q_len_padded_,");
  code.e("      $,", num_heads);
  code.e("      $f,", sm_scale_log2);
  code.e("      $f,", sm_scale);
  code.e("      decode_sk_rt_,");
  code.e("      $,", num_decode_groups);
  code.e("      $,", qpg);
  code.e("      false,");
  code.e("      prompt_prefill_,");
  code.e("      $,",
         direct_paged_decode_kv ? "runtime_config.paged_kv_indices_buffer"
                                : "nullptr");
  code.e("      decode_first_page_pos_,");
  code.e("      meta_x_,");
  code.e("      meta_y_,");
  code.e("      meta_z_);");
  code.e("}");
  return register_task_variant(TASK_MLA_UNIFIED_SM100, code.to_string());
}

int TaskRegister::register_mla_mtp_decode_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_head_groups
  // params[1]: q_len
  // params[2]: kv_len
  // params[3]: num_splits (sk)
  assert(params.size() == 4);
  int num_head_groups = params[0];
  int q_len = params[1];
  int kv_len = params[2];
  int num_splits = params[3];
  // Compute single_tile: true when each split handles exactly 1 KV tile
  int kvt = (kv_len + 128 - 1) / 128; // TILE_S = 128
  int tps = (kvt + num_splits - 1) / num_splits;
  int single_tile = (tps == 1) ? 1 : 0;
  bool const write_final = (num_splits == 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.merge_task_offset & 0xffff;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
  code.e("  if (q_len_rt_ > 8) return;");
  code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  // Template dispatch on SINGLE_TILE
  if (single_tile) {
    code.e("  kernel::mla_mtp_decode_sm100_task_impl<true, $>(",
           write_final ? "true" : "false");
  } else {
    code.e("  kernel::mla_mtp_decode_sm100_task_impl<false, $>(",
           write_final ? "true" : "false");
  }
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("    static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),"); // Oa
  code.e("    static_cast<float*>(task_desc->output_ptrs[1]),");       // La
  // DeepSeek V3 MLA softmax_scale = q_head_dim^-0.5 * mscale^2 ≈ 0.13525
  // (NOT 1/sqrt(576) — d_k=576 is the absorbed latent dim, not the dot-product
  // scaling dim. q_head_dim=192=128+64 per modeling_deepseek.py:689-695.)
  {
    float const _mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
    float const _sm = (1.0f / sqrtf(192.0f)) * _mscale * _mscale;
    code.e("    $f,", _sm); // ss
  }
  code.e("    kv_len_,");
  // The task grid is generated with the static num_splits. Keep that same
  // split count in the kernel so block_x metadata decodes to the intended
  // (group, split). Runtime kv_len_ still makes inactive splits take the
  // t0>=t1 path.
  code.e("    $,", num_splits);
  code.e("    $,", num_head_groups);
  code.e("    q_len_rt_,");
  // gi, si, bi from task metadata
  code.e("    task_desc->task_metadata.request_id,"); // gi (head_group)
  code.e("    task_desc->task_metadata.kv_idx,");     // si (split_idx)
  code.e("    bi_);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_SM100, code.to_string());
}

int TaskRegister::register_mla_mtp_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_head_groups
  // params[1]: q_len
  // params[2]: num_splits (sk)
  // params[3]: rd_dv (D_V dims per block)
  assert(params.size() == 4);
  int num_head_groups = params[0];
  int q_len = params[1];
  int num_splits = params[2];
  int rd_dv = params[3];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
  code.e("  if (q_len_rt_ > 8) return;");
  code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  // 256 threads for MPK workers (default template is 512 for standalone)
  code.e("  kernel::mla_mtp_reduce_sm100_task_impl<256>(");
  code.e(
      "    static_cast<const nv_bfloat16*>(task_desc->input_ptrs[0]),"); // Oa
  code.e("    static_cast<const float*>(task_desc->input_ptrs[1]),");    // La
  code.e("    static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");   // O
  // Match the static split count used to size the partial buffers and task
  // grid. Inactive runtime splits contain -inf LSE and reduce away.
  code.e("    $,", num_splits);
  code.e("    $,", num_head_groups);
  code.e("    q_len_rt_,");
  // dv_base, gi, bi from task metadata
  code.e("    task_desc->task_metadata.kv_idx * $,", rd_dv); // dv_base
  code.e("    task_desc->task_metadata.request_id,");        // gi
  code.e("    bi_);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_REDUCE_SM100, code.to_string());
}
int TaskRegister::register_paged_attention_split_kv_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
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
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads, /* NUM_QO_HEADS */
         1,                          /* NUM_KV_HEADS */
         num_kv_heads,               /* NUM_QO_GROUPS */
         kv_stride,                  /* KV_CACHE_STRIDE */
         qkv_stride,                 /* QKV_STRIDE */
         output_size *
             num_kv_chunks, /* O_STRIDE (should consider num_kv_chunks) */
         head_dim,          /* HEAD_DIM */
         SEQ_LEN_PER_BLOCK, /* SEQ_LEN */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         max_tokens,        /* MAX_TOKENS */
         "true",            /* PARTITION_KV */
         num_kv_chunks);    /* NUM_KV_CHUNKS */
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],"); // output_tmp
  code.e("    task_desc->output_ptrs[0],"); // lse
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER,
                               code.to_string());
}

int TaskRegister::register_nvshmem_allgather_strided_put_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_gpus
  // params[1]: my_gpu_id
  assert(params.size() == 2);
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
  // For now, the memory partition of the input[0] results in a strided
  // 2D tensor, which cannot be directly transferred by a single nvshmem
  // memput. So we use for loop to iterate over the first dim and transfer each
  // row. If the upperlayer changes this layout, this "for-loop" method can
  // fail. So we assert it here just in case.
  assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
         input_ops[0]->input_map.z == -1);
  // Currently support 2D reduction, buffer has an extra world_size dim
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int output_size = input_ops[0]->output_tensors[0].dim[1];
  // Row stride (C20: dtensor.stride[0] is view-safe).
  int input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  // For this allgather task, input and output share the same stride
  int output_stride = input_stride;
  // Register nvshmem copy task (allgather)
  mirage::transpiler::CodeKeeper c;
  c.inc_indent();
  c.e("size_t event_index = "
      "get_event_position_index(task_desc->trigger_event);");
  c.inc_indent();
  c.e("int target_gpu_id = "
      "static_cast<int>(get_event_gpu_id(task_desc->trigger_event));");
  c.e("kernel::nvshmem_allgather_strided_put<bfloat16, $, $, $>(",
      batch_size,
      output_size,
      output_stride);
  c.e("  task_desc->output_ptrs[0],");
  c.e("  task_desc->input_ptrs[0],");
  c.e("  &runtime_config.all_event_counters[event_index],");
  c.e("  event_index,");
  c.e("  target_gpu_id,");
  c.e("  runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");

  return register_task_variant(TASK_NVSHMEM_ALLGATHER_STRIDED_PUT,
                               c.to_string());
}

int TaskRegister::register_nvshmem_tile_allreduce_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_gpus
  // params[1]: my_gpu_id
  // params[2] optional: phase gate, 1=prefill, 2=decode.
  assert(params.size() == 2 || params.size() == 3);
  int gate_mode = (params.size() == 3) ? params[2] : 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_outputs = 1;
  int num_inputs = static_cast<int>(bgraph.operators.size()) - num_outputs;
  assert(num_inputs == 1 || num_inputs == 2);
  bool const with_residual = num_inputs == 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
         input_ops[0]->input_map.z == -1);
  if (with_residual) {
    assert(input_ops[1]->input_map.x == 1 && input_ops[1]->input_map.y == -1 &&
           input_ops[1]->input_map.z == -1);
  }
  // Currently support 2D reduction, buffer has an extra world_size dim
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int output_size = input_ops[0]->output_tensors[0].dim[1];
  // Row stride (C20: dtensor.stride[0] is view-safe).
  int input_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  // For this allgather task, input and output share the same stride
  int output_stride = input_stride;
  // Register tile allreduce task
  mirage::transpiler::CodeKeeper c;
  c.inc_indent();
  emit_deepseek_phase_gate(c, gate_mode);
  if (with_residual) {
    c.e("kernel::nvshmem_tile_allreduce_with_residual<__nv_bfloat16, $, $, $>(",
        batch_size,
        output_size,
        output_stride);
    c.e("  task_desc->input_ptrs[0],");
    c.e("  task_desc->input_ptrs[1],");
  } else {
    c.e("kernel::nvshmem_tile_allreduce<__nv_bfloat16, $, $, $>(",
        batch_size,
        output_size,
        output_stride);
    c.e("  task_desc->input_ptrs[0],");
  }
  c.e("  task_desc->output_ptrs[0],");
  c.e("  runtime_config.nvshmem_teams,");
  c.e("  task_desc->task_metadata.task_offset,");
  c.e("  runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_NVSHMEM_TILE_ALLREDUCE, c.to_string());
}

int TaskRegister::register_nvshmem_global_argmax_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_gpus
  // params[1]: my_gpu_id
  // params[2]: vocab_offset
  // params[3]: valid_vocab_size
  // params[4]: partial_chunk_size
  assert(params.size() == 5);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 3;

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
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_partial_tasks = input_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[1]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == num_partial_tasks);
  int valid_vocab_size = params[3];
  int partial_chunk_size = params[4];
  assert(partial_chunk_size > 0);
  assert(valid_vocab_size > 0 &&
         valid_vocab_size <= num_partial_tasks * partial_chunk_size);

  mirage::transpiler::CodeKeeper c;
  c.inc_indent();
  // For chunked prefill, intermediate chunks do not produce user-visible
  // tokens. Skip the cross-rank argmax collective until the active chunk
  // reaches the end of at least one prompt, or until decode has started.
  c.e("{");
  c.e("int active_tokens = "
      "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  c.e("bool should_run_argmax = false;");
  c.e("for (int bi = 0; bi < MPK_MAX_NUM_BATCHED_REQUESTS; ++bi) {");
  c.e("  int req_id = runtime_config.request_ids[bi];");
  c.e("  if (req_id < 0) continue;");
  c.e("  int q_len = runtime_config.qo_indptr_buffer[bi + 1] - "
      "runtime_config.qo_indptr_buffer[bi];");
  c.e("  int step = runtime_config.step[req_id];");
  c.e("  int prompt_len = runtime_config.prompt_length[req_id];");
  c.e("  if (step >= prompt_len || step + q_len >= prompt_len) {");
  c.e("    should_run_argmax = true;");
  c.e("    break;");
  c.e("  }");
  c.e("}");
  c.e("if (!should_run_argmax || active_tokens <= 0) return;");
  c.e("kernel::nvshmem_global_argmax_from_partials_bf16<$, $, $, $, $>(",
      batch_size,
      num_partial_tasks,
      partial_chunk_size,
      valid_vocab_size,
      params[2]);
  c.e("  task_desc->input_ptrs[0],");
  c.e("  task_desc->input_ptrs[1],");
  c.e("  task_desc->output_ptrs[0],");
  c.e("  task_desc->output_ptrs[1],");
  c.e("  task_desc->output_ptrs[2],");
  c.e("  runtime_config.nvshmem_teams,");
  c.e("  task_desc->task_metadata.task_offset,");
  c.e("  active_tokens);");
  c.e("}");
  return register_task_variant(TASK_NVSHMEM_GLOBAL_ARGMAX, c.to_string());
}

int TaskRegister::register_quantize_fp8_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool scale_ue8m0) {
  // Input: bf16 [batch, hidden] or [batch, topk, hidden] (3D flattened)
  // Output: fp8 same shape, scale [..., hidden/group_size]
  // scale_ue8m0=true: packed UE8M0 uint32 scale (for FP8 linear GEMM)
  // scale_ue8m0=false: float32 scale (for MoE group GEMM)
  // active_mode:
  //   0: rows are current qo tokens (default)
  //   1: rows are the current request's full KV length, for prefill cache
  //      decompression over [0, kv_len)
  //   2: qo-token rows, prefill phase only
  //   3: qo-token rows, decode phase only
  // params layout:
  //   size 0: defaults (active_mode=0, no row-slice overrides).
  //   size 1: [active_mode] (legacy).
  //   size 4: [active_mode, hidden_size_override, input_stride_override,
  //            in_offset_elems] — QKV-a fused path uses this to quantize
  //            a slice of a wider buffer.
  //   size 5: [active_mode=5, expert_meta_offset, e_local,
  //            bm_padding, ctas_per_expert] — B15 per-expert
  //            active-rows skip for NEW MoE silu_out quantize.
  assert(params.size() == 0 || params.size() == 1 || params.size() == 4 ||
         params.size() == 5);
  int active_mode = params.empty() ? 0 : params[0];
  // active_mode 4 (B12): no token-indexed skip — process every CTA's
  // ROWS_PER_TASK chunk unconditionally. Used by NEW MoE silu_out
  // quantize where rows are permuted-expert layout, not token index.
  // active_mode 5 (B15): per-expert active-rows cap. Meta is supplied
  // as a 4th tb_graph input; codegen pre-reads active_mask[my_expert]
  // (skip if 0) then actual_count[my_expert] and caps the kernel's
  // ROWS_PER_TASK inner loop.
  assert(active_mode >= 0 && active_mode <= 5);
  bool has_slice_override = (params.size() == 4);
  bool has_expert_active = (params.size() == 5);
  int expert_meta_offset = has_expert_active ? params[1] : -1;
  int expert_e_local = has_expert_active ? params[2] : 0;
  int expert_bm_padding = has_expert_active ? params[3] : 0;
  int expert_ctas_per_expert = has_expert_active ? params[4] : 1;
  int batch_size = 0, hidden_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = has_expert_active ? 2 : 1;
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
  int ndims = input_ops[0]->dtensor.num_dims;
  assert(ndims == 2 || ndims == 3);
  if (ndims == 3) {
    batch_size = input_ops[0]->output_tensors[0].dim[0] *
                 input_ops[0]->output_tensors[0].dim[1];
    hidden_size = input_ops[0]->output_tensors[0].dim[2];
  } else {
    batch_size = input_ops[0]->output_tensors[0].dim[0];
    hidden_size = input_ops[0]->output_tensors[0].dim[1];
  }
  // GLOBAL_STRIDE = stride between rows in linearized layout. C20
  // (2026-05-17): use stride[0] (in elements) instead of dim[1]; for non-view
  // tensors these are equal, but for an `mpk.narrow` view dim[1] is the
  // slot width while stride[0] is the parent's full row width — the
  // latter is what the kernel must walk by to avoid stepping into the
  // adjacent slot.
  int input_stride = (ndims == 3)
                         ? static_cast<int>(input_ops[0]->dtensor.stride[1])
                         : static_cast<int>(input_ops[0]->dtensor.stride[0]);
  int in_offset_elems = 0;
  if (has_slice_override) {
    hidden_size = params[1];
    input_stride = params[2];
    in_offset_elems = params[3];
  }
  int active_row_multiplier =
      (ndims == 3) ? input_ops[0]->output_tensors[0].dim[1] : 1;
  constexpr int GROUP_SIZE = 128;
  int group_tiles = bgraph.grid_dim.x;

  // For UE8M0 path: scale_outer_stride is the stride between packed scale
  // columns in the column-major output layout (= aligned_batch for UE8M0)
  int aligned_batch = scale_ue8m0 ? ((batch_size + 3) / 4) * 4 : 1;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  if (active_mode == 1) {
    code.e("int row_idx_ = task_desc->task_metadata.request_id;");
    code.e("int bi_ = row_idx_ / MPK_MAX_SEQ_LENGTH;");
    code.e("int row_local_ = row_idx_ - bi_ * MPK_MAX_SEQ_LENGTH;");
    code.e("if (bi_ < 0 || bi_ >= MPK_MAX_NUM_BATCHED_REQUESTS) return;");
    code.e("int req_id_ = runtime_config.request_ids[bi_];");
    code.e("if (req_id_ < 0) return;");
    code.e("int q_len_ = runtime_config.qo_indptr_buffer[bi_ + 1] - "
           "runtime_config.qo_indptr_buffer[bi_];");
    code.e("bool prompt_prefill_ = runtime_config.step[req_id_] < "
           "runtime_config.prompt_length[req_id_] && q_len_ > 8;");
    code.e("if (!prompt_prefill_) return;");
    code.e("int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
    code.e("int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
    code.e("int seq_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
           "runtime_config.paged_kv_last_page_len_buffer[bi_];");
    code.e("if (row_local_ < 0 || row_local_ >= seq_len_) return;");
  } else if (active_mode == 4) {
    // process_all_rows: no skip — every CTA quantizes its
    // ROWS_PER_TASK chunk. For NEW MoE silu_out where rows index a
    // permuted-expert layout (M_TOTAL = E_LOCAL × BM_PADDING) and the
    // token-indexed active_rows_ would silently leave most rows
    // uninitialized, feeding stale silu_fp8 to the W2 group GEMM.
  } else if (active_mode == 5) {
    // B15: per-expert active-rows skip. Skip CTA entirely when expert
    // is inactive (active_mask[my_expert]=0). Otherwise read
    // actual_count and pass to kernel as row_count_cap to bound the
    // ROWS_PER_TASK inner loop. CTA→expert mapping: with grid_y =
    // num_workers and BM_PADDING == ROWS_PER_TASK, my_expert =
    // task_metadata.request_id / ctas_per_expert.
    code.e("int const *active_mask_q_ = "
           "static_cast<int const *>(task_desc->input_ptrs[1]) + $;",
           expert_meta_offset);
    code.e("int my_expert_q_ = task_desc->task_metadata.request_id / $;",
           expert_ctas_per_expert);
    code.e("if (!active_mask_q_[my_expert_q_]) return;");
    code.e("int row_count_cap_ = active_mask_q_[$ + my_expert_q_];",
           expert_e_local);
    code.e("if (row_count_cap_ <= 0) return;");
    code.e("if (row_count_cap_ > $) row_count_cap_ = $;",
           expert_bm_padding,
           expert_bm_padding);
  } else {
    code.e("int active_rows_ = "
           "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS] * $;",
           active_row_multiplier);
    if (active_mode == 2 || active_mode == 3) {
      emit_deepseek_prefill_flag(code);
      if (active_mode == 2) {
        code.e("if (!prompt_prefill_) return;");
      } else {
        code.e("if (prompt_prefill_) return;");
      }
    }
    code.e("if (task_desc->task_metadata.request_id >= active_rows_) return;");
  }
  // OUTPUT_STRIDE: when slicing (input_stride > hidden_size), the output
  // buffer is sized for hidden_size per row, so writes must use hidden_size
  // as the row stride. Default (no slice) keeps OUTPUT_STRIDE == input_stride
  // for backward compat with legacy callers where the kernel previously
  // assumed input_stride == output_stride. 2026-05-12 H8 fix.
  int output_stride = has_slice_override ? hidden_size : input_stride;
  // ROWS_PER_TASK: when caller passes grid.y < batch_size, the kernel
  // internally loops over multiple rows per CTA so total launched CTAs
  // stay ≤ num_workers. Default 1 preserves the legacy 1-row-per-CTA
  // behavior when grid.y == batch_size.
  int grid_y_safe = bgraph.grid_dim.y > 0 ? (int)bgraph.grid_dim.y : 1;
  int rows_per_task = (batch_size + grid_y_safe - 1) / grid_y_safe;
  if (rows_per_task < 1) {
    rows_per_task = 1;
  }
  code.e("kernel::per_token_group_quantize_fp8_task_impl<$, $, $, $, $,",
         batch_size,
         hidden_size,
         GROUP_SIZE,
         input_stride,
         group_tiles);
  code.e("    cute::bfloat16_t, __nv_fp8_e4m3, $, $, $>(",
         scale_ue8m0 ? "true" : "false",
         output_stride,
         rows_per_task);
  if (in_offset_elems != 0) {
    // QKV-a fused slice: pre-offset the input pointer so the kernel reads
    // the right column window from a wider buffer.
    code.e("    static_cast<cute::bfloat16_t const*>(task_desc->input_ptrs[0])"
           " + $,",
           in_offset_elems);
  } else {
    code.e("    task_desc->input_ptrs[0],"); // input bf16
  }
  code.e("    task_desc->output_ptrs[0],"); // output fp8
  code.e("    task_desc->output_ptrs[1],"); // output scale
  // scale_outer_stride: for UE8M0 column-major layout [packed_k,
  // aligned_batch], stride between consecutive packed_k entries =
  // aligned_batch. For float32 scale (MoE), scale_outer_stride is unused
  // (float32 writes directly).
  code.e("    1e-10f, -448.0f, 448.0f,");
  code.e("    $,", aligned_batch);
  code.e("    task_desc->task_metadata.request_id,");
  if (active_mode == 5) {
    code.e("    task_desc->task_metadata.kv_idx,");
    code.e("    row_count_cap_);");
  } else {
    code.e("    task_desc->task_metadata.kv_idx);");
  }
  code.e("}");
  return register_task_variant(TASK_QUANTIZE_FP8_SM100, code.to_string());
}

int TaskRegister::register_moe_silu_mul_quantize_fp8_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // C18 (2026-05-17): fused MoE silu·mul + per-token-group FP8 quantize.
  //
  // tb_graph inputs (in order):
  //   [0] input bf16   [M_TOTAL, 2*K_INTER]  (w13_out: gate || up halves)
  //   With ACTIVE_SKIP:
  //   [1] meta int32 [2, ...] (NEW MoE meta; mask + actual_count rows)
  //   Outputs (via store_in_dmem):
  //   [out0] silu_fp8  [M_TOTAL, K_INTER]   FP8
  //   [out1] silu_scale K-outer UE8M0 uint32
  //
  // params (optional):
  //   params[0] = K_INTER override (default = input.dim(1)/2)
  //   params[1] = ROWS_PER_TASK (default = 1)
  //   With ACTIVE_SKIP (params.size()==5):
  //   params[2] = active_mask_offset (int offset into meta)
  //   params[3] = e_local (num local experts)
  //   params[4] = bm_padding (rows per expert in permuted buffer)
  //
  // The fused task writes (M_TOTAL, K_INTER) FP8 + UE8M0 K-outer scale with
  // the SAME layout the standalone `quantize_fp8` task produces — downstream
  // moe_permute / fp8_group_gemm consumers see byte-identical scale data.
  assert(params.size() == 0 || params.size() == 1 || params.size() == 2 ||
         params.size() == 5);
  bool const has_active = (params.size() == 5);

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int const num_inputs = has_active ? 2 : 1;
  int const num_outputs = 2;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int const m_total = input_ops[0]->dtensor.dim[0];
  int const in_row_full = input_ops[0]->dtensor.dim[1];
  int const out_row_full = output_ops[0]->dtensor.dim[1];
  int const k_inter_default = in_row_full / 2;
  int const k_inter = params.size() >= 1 ? params[0] : k_inter_default;
  int const rows_per_task = params.size() >= 2 ? params[1] : 1;
  int const active_mask_offset = has_active ? params[2] : 0;
  int const e_local = has_active ? params[3] : 0;
  int const bm_padding = has_active ? params[4] : 0;
  assert(in_row_full == 2 * k_inter);
  assert(out_row_full == k_inter);

  int const aligned_batch = ((m_total + 3) / 4) * 4;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // Template: K_INTER, GROUP_SIZE, IN_ROW_STRIDE, OUT_ROW_STRIDE, BM_PADDING,
  //           E_LOCAL, T, DST_T, ROWS_PER_TASK, ACTIVE_SKIP
  code.e("kernel::moe_silu_mul_quantize_fp8_task_impl<$, 128, $, $, $, $, "
         "cute::bfloat16_t, __nv_fp8_e4m3, $, $>(",
         k_inter,
         /*IN_ROW_STRIDE=*/in_row_full,
         /*OUT_ROW_STRIDE=*/out_row_full,
         has_active ? bm_padding : 1,
         has_active ? e_local : 1,
         rows_per_task,
         has_active ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");  // w13_out (real input slot)
  code.e("    task_desc->output_ptrs[0],"); // silu_fp8 (output slot 0)
  code.e("    task_desc->output_ptrs[1],"); // silu_scale (output slot 1)
  if (has_active) {
    code.e("    static_cast<int const*>(task_desc->input_ptrs[1]) + $,",
           active_mask_offset);
  } else {
    code.e("    nullptr,");
  }
  code.e("    1e-10f, -448.0f, 448.0f,");
  code.e("    $,", aligned_batch);
  code.e("    task_desc->task_metadata.request_id,"); // task_idx = row_idx
  code.e("    -1);"); // no row_count_cap (caller may still bound it later)
  code.e("}");
  return register_task_variant(TASK_MOE_SILU_MUL_QUANTIZE_FP8_SM100,
                               code.to_string());
}

int TaskRegister::register_fused_rmsnorm_quantize_fp8_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // B37 (2026-05-15): fused RMSNorm + per-token-group FP8 quantize.
  //
  // tb_graph inputs (in order):
  //   [0] input bf16  [M, K_full]              (row-major, possibly wider
  //                                              than process_dim)
  //   [1] weight bf16 [process_dim]            (rms weight)
  //   [2] output_bf16 [M, K_full]              (normalized bf16 output —
  //                                              optional; the kernel still
  //                                              computes it in smem)
  //   [3] output_fp8  [M, process_dim]         (per-row FP8 quantized)
  //   [4] output_scale uint32                  (packed UE8M0 column-major
  //                                              [packed_k, aligned_batch])
  //
  // params (optional, default = legacy contiguous + UE8M0 scale):
  //   params[0] = process_dim     (HIDDEN_DIM the kernel processes per row)
  //   params[1] = in_offset_elems (skip elements at start of each row)
  //   params[2] = out_offset_elems (same, for bf16 output)
  //   params[3] = scale_ue8m0 (1=UE8M0 packed uint32, 0=float32 scale)
  //   params[4] = emit_bf16 (1=write bf16 output, 0=skip the bf16 store)
  //
  // The bf16 output is stored as the 3rd input tensor (store_in_dmem),
  // matching the MPK convention used by mla_decode/mla_prefill — codegen
  // reads `input_ptrs[2]` for the output pointer.
  assert(params.size() == 0 || params.size() == 3 || params.size() == 4 ||
         params.size() == 5);

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int const num_inputs = 2;
  int const num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int dtensor_batch = input_ops[0]->dtensor.dim[0];
  int hidden_dim_full = input_ops[0]->dtensor.dim[1];
  int output_bf16_full = output_ops[0]->dtensor.dim[1];
  int output_fp8_full = output_ops[1]->dtensor.dim[1];

  int process_dim = params.size() >= 3 ? params[0] : hidden_dim_full;
  int in_offset = params.size() >= 3 ? params[1] : 0;
  int out_offset = params.size() >= 3 ? params[2] : 0;
  int scale_ue8m0 = params.size() >= 4 ? params[3] : 1;
  int emit_bf16 = params.size() >= 5 ? params[4] : 1;
  assert(scale_ue8m0 == 0 || scale_ue8m0 == 1);
  assert(emit_bf16 == 0 || emit_bf16 == 1);
  assert(in_offset + process_dim <= hidden_dim_full);
  assert(out_offset + process_dim <= output_bf16_full);

  // BATCH_SIZE per CTA = ceil(batch / grid_x).
  int const grid_x = bgraph.grid_dim.x > 0 ? (int)bgraph.grid_dim.x : 1;
  int batch_size = (dtensor_batch + grid_x - 1) / grid_x;
  if (batch_size < 1) {
    batch_size = 1;
  }

  // UE8M0 column-major scale layout requires the alignment to align with
  // the batch axis dim (see quantize kernel). For the float32 path,
  // scale_outer_stride is unused (writes go to [batch, num_groups] row-
  // major) but we still pass a sane value.
  constexpr int GROUP_SIZE = 128;
  int const aligned_batch =
      scale_ue8m0 ? ((dtensor_batch + 3) / 4) * 4 : dtensor_batch;
  (void)GROUP_SIZE;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // Active-rows gate (B34 convention): skip CTAs whose first row is past
  // the active token count, and clamp the inner loop to the remaining
  // active rows so we don't normalize/overwrite stale bf16.
  code.e("int active_rows_fused_ = $;", dtensor_batch);
  code.e("#ifndef MPK_TEST_MODE");
  code.e("active_rows_fused_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("#endif");
  code.e("int task_first_row_fused_ = "
         "task_desc->task_metadata.request_id * $;",
         batch_size);
  code.e("if (task_first_row_fused_ < 0) task_first_row_fused_ = 0;");
  code.e("if (task_first_row_fused_ >= active_rows_fused_) return;");
  code.e("int row_count_cap_fused_ = "
         "active_rows_fused_ - task_first_row_fused_;");

  // Kernel template params: T, DST_T, BATCH_SIZE, HIDDEN_DIM, GROUP_SIZE,
  // NUM_THREADS, IN_OFFSET, OUT_OFFSET, IN_ROW_STRIDE, OUT_ROW_STRIDE,
  // FP8_ROW_STRIDE, SCALE_UE8M0, EMIT_BF16.
  code.e("kernel::fused_rmsnorm_quantize_fp8_impl<bfloat16, __nv_fp8_e4m3,"
         " $, $, 128, 256, $, $, $, $, $, $, $>(",
         batch_size,
         process_dim,
         in_offset,
         out_offset,
         hidden_dim_full,
         output_bf16_full,
         output_fp8_full,
         scale_ue8m0 ? "true" : "false",
         emit_bf16 ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");  // input bf16
  code.e("    task_desc->input_ptrs[1],");  // weight bf16
  code.e("    task_desc->output_ptrs[0],"); // output bf16
  code.e("    task_desc->output_ptrs[1],"); // output fp8
  code.e("    task_desc->output_ptrs[2],"); // output scale
  code.e("    1e-6f,");                     // rms eps
  code.e("    1e-10f,"); // quantize scale eps (floor for local_max)
  code.e("    -448.0f, 448.0f,");
  code.e("    $,", aligned_batch);
  code.e("    task_desc->task_metadata.request_id,"); // task_idx
  code.e("    row_count_cap_fused_);");
  code.e("}");
  return register_task_variant(TASK_FUSED_RMSNORM_QUANTIZE_FP8_SM100,
                               code.to_string());
}

int TaskRegister::register_linear_fp8_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  // Inputs: input_fp8 [batch, reduction], input_scale [batch, reduction/128],
  //         weight_fp8 [output, reduction], weight_scale [output,
  //         reduction/128], (optional) residual [batch, output]
  // Output: output_bf16 [batch, output]
  bool rank_with_residual = with_residual;
  int gate_mode = 0;
  if (with_residual) {
    assert(params.size() == 1 || params.size() == 2);
    rank_with_residual = (params[0] == 1);
    if (params.size() == 2) {
      gate_mode = params[1];
    }
  } else {
    assert(params.size() == 0 || params.size() == 1);
    if (params.size() == 1) {
      gate_mode = params[0];
    }
  }
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // Inputs: input_fp8, input_scale, weight_fp8, weight_scale, [residual]
  int num_inputs = with_residual ? 5 : 4;
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
  assert(input_ops[0]->dtensor.num_dims == 2); // input_fp8
  reduction_size = input_ops[0]->dtensor.dim[1];
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.inc_indent();
  emit_deepseek_phase_gate(code, gate_mode);

  // Persistent-kernel FP8 GEMM task. Each MPK task is one CTA, so kNumSMs is
  // fixed to 1 and the task walks all internal BLOCK_N tiles in its output
  // shard sequentially.
  code.e("kernel::linear_fp8_sm100_task_impl<");
  code.e("    cute::UMMA::Major::K, cute::UMMA::Major::K,");
  code.e("    128, 128,"); // kGranKA, kGranKB
  code.e("    $, $, $,",
         batch_size,
         output_size,
         reduction_size);      // SHAPE_M, SHAPE_N, SHAPE_K
  code.e("    32, 16, 128,");  // BLOCK_M, BLOCK_N, BLOCK_K
  code.e("    1,");            // kNumGroups
  code.e("    128, 128, 32,"); // kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode
  code.e("    25,"); // kNumStages (fit persistent kernel 207KB smem budget)
  code.e("    128, 128,"); // kNumNonEpilogueThreads, kNumEpilogueThreads
  code.e("    1, false,"); // kNumMulticast, kIsMulticastOnA
  code.e("    1,"); // kNumSMs (persistent kernel: 1 CTA per task, processes all
                    // tiles)
  code.e("    $,",
         (with_residual && rank_with_residual) ? "true"
                                               : "false"); // kWithResidual
  code.e("    mirage::blackwell::linear_fp8_sm100::GemmType::Normal,");
  code.e("    false,"); // kWithAccumulation
  code.e(
      "    cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,");
  code.e("    mirage::blackwell::linear_fp8_sm100::EpilogueIdentity>(");
  code.e("    nullptr,"); // grouped_layout
  code.e(
      "    $, $, $,", batch_size, output_size, reduction_size); // runtime dims
  code.e("    *reinterpret_cast<cute::TmaDescriptor "
         "const*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A
  code.e("    *reinterpret_cast<cute::TmaDescriptor "
         "const*>(task_desc->input_tma_desc_ptrs[2][0]),"); // B
  code.e("    *reinterpret_cast<cute::TmaDescriptor "
         "const*>(task_desc->input_tma_desc_ptrs[1][0]),"); // SFA
  code.e("    *reinterpret_cast<cute::TmaDescriptor "
         "const*>(task_desc->input_tma_desc_ptrs[3][0]),"); // SFB
  if (with_residual && rank_with_residual) {
    code.e("    *reinterpret_cast<cute::TmaDescriptor "
           "const*>(task_desc->input_tma_desc_ptrs[4][0]),"); // residual
  } else {
    code.e("    *reinterpret_cast<cute::TmaDescriptor "
           "const*>(task_desc->output_tma_desc_ptrs[0][0]),"); // dummy
  }
  code.e("    *reinterpret_cast<cute::TmaDescriptor "
         "const*>(task_desc->output_tma_desc_ptrs[0][0]));"); // CD

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_FP8_WITH_RESIDUAL_SM100,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_FP8_SM100, code.to_string());
  }
}

int TaskRegister::register_linear_fp8_swapAB_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  // Inputs (Python-layer order): input_fp8, input_scale, weight_fp8,
  // weight_scale, [residual]
  // Output: output_bf16
  //
  // The kernel internally swaps A<->B (linear_fp8_swapAB_sm100_task_impl):
  // A=weight (M-axis = per-task output_size), B=activation (N-axis = batch). So
  // the codegen routes weight_fp8 -> tma_a, input_fp8 -> tma_b, weight_scale ->
  // tma_sfa, input_scale -> tma_sfb. This is the only place that reorder
  // happens; the runtime task_desc->input_tma_desc_ptrs[] keeps Python-layer
  // order.
  bool rank_with_residual = with_residual;
  int gate_mode = 0;
  if (with_residual) {
    assert(params.size() == 1 || params.size() == 2);
    rank_with_residual = (params[0] == 1);
    if (params.size() == 2) {
      gate_mode = params[1];
    }
  } else {
    assert(params.size() == 0 || params.size() == 1);
    if (params.size() == 1) {
      gate_mode = params[0];
    }
  }
  int batch_size = 0, output_size_per_task = 0, reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 5 : 4;
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
  // STensor (per-task tile) holds the post-grid-split shape.
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size_per_task = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  // Hard constraints from the kernel design (see plan):
  //   - per-task output size must be a multiple of MMA_M=128 (the swapped
  //     A-side tile height; one CTA covers an integer number of these).
  //   - decode-only: batch_size must fit in MMA_N=16 (one shot, no inner
  //     N-walk). Larger M would re-introduce the same serialization the new
  //     kernel was built to avoid.
  assert(
      output_size_per_task % 128 == 0 &&
      "linear_fp8_swapAB_sm100 requires per-task output size divisible by 128");
  assert(batch_size <= 16 &&
         "linear_fp8_swapAB_sm100 is decode-only: BATCH_SIZE must be <= 16");
  assert(reduction_size % 128 == 0 &&
         "linear_fp8_swapAB_sm100 requires K divisible by BLOCK_K=128");

  // Output stride (column dim) in global memory. For the MPK FP8 swapAB
  // kernel we always treat the output as row-major BF16 [BATCH, OUTPUT].
  int output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  // Codegen mirrors register_linear_sm100_task (BF16 MPK) with the only
  // additions being: (a) FP8 element type for A/B, (b) two raw uint32_t*
  // scale pointers passed through after BiasTensor, (c) FP8-tuned
  // BLOCK_K=128 (UMMA_K=32) instead of BLOCK_K=64 (UMMA_K=16).
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  emit_deepseek_phase_gate(code, gate_mode);
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  // FP8 path: TMA copies whole BLOCK_K=128 K-tile per shot (one byte per
  // element, so 128 bytes/row fits in the 128B swizzle).
  constexpr int TMA_CP_ASYNC_SIZE = 128;
  constexpr int TILE_SIZE = 128;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;

  // tma_a = WEIGHT (after swap A-side). FP8 element, [OUTPUT_SIZE,
  // REDUCTION_SIZE], row-major, K-major TMA.
  code.e("using TMA_A = kernel::tma::tma_2d<cutlass::float_e4m3_t, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         output_size_per_task, /*GMEM_ROW_*/
         reduction_size,       /*GMEM_COL_*/
         MMA_M,                /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,    /*SMEM_COL_*/
         reduction_size,       /*GMEM_STRIDE_ROW_*/
         1,                    /*GMEM_STRIDE_COL_*/
         1,                    /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  // tma_b = INPUT (after swap B-side). FP8 element, [BATCH_SIZE,
  // REDUCTION_SIZE], row-major, K-major TMA.
  code.e("using TMA_B = kernel::tma::tma_2d<cutlass::float_e4m3_t, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  // tma_out = OUTPUT BF16 [BATCH_SIZE, OUTPUT_SIZE], row-major.
  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, "
         "$, $, $, $, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,           /*GMEM_ROW_*/
         output_size_per_task, /*GMEM_COL_*/
         MMA_N,                /*SMEM_ROW_*/
         MMA_M,                /*SMEM_COL_*/
         output_stride,        /*GMEM_STRIDE_ROW_*/
         1,                    /*GMEM_STRIDE_COL_*/
         1,                    /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  // Construct typed wrappers from CUtensorMap pointers stashed on TaskDesc.
  // SwapAB wiring: the weight tensor (Python slot 2) becomes the kernel's
  // A; the input tensor (Python slot 0) becomes the kernel's B.
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  // Raw uint32_t* scale pointers. The Python layer's quantize step writes
  // UE8M0-packed scales (4 bytes per K=128 row of A or B). We trust the
  // runtime to point input_ptrs[1] / [3] at those packed buffers.
  // SwapAB wiring: weight_scale (slot 3) is the kernel's A-side scale,
  // input_scale (slot 1) is the kernel's B-side scale.
  code.e("uint32_t const *weight_scale_ptr = "
         "static_cast<uint32_t const*>(task_desc->input_ptrs[3]);");
  code.e("uint32_t const *input_scale_ptr  = "
         "static_cast<uint32_t const*>(task_desc->input_ptrs[1]);");

  // BiasTensor: a CuTe gmem tensor over the residual when present, or a
  // nullptr-backed placeholder when absent. The kernel branches on NOBIAS
  // (compile-time) and never dereferences mBias when NOBIAS=true.
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size_per_task,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         (with_residual && rank_with_residual) ? "task_desc->input_ptrs[4]"
                                               : "nullptr");

  // Non-split: row stride of the packed scale buffer equals the per-row
  // uint32 count, which equals the kernel's compile-time PACKED_SCALE_K.
  // (full_K and per-task K coincide here.) Split-K registration below
  // overrides this.
  int const packed_scale_k = (reduction_size + 511) / 512;

  code.e("kernel::linear_fp8_swapAB_sm100_task_impl<cutlass::float_e4m3_t, "
         "TMA_A, TMA_B, decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, /*SplitK=*/false, $, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size_per_task,
         reduction_size,
         (with_residual && rank_with_residual) ? "false" : "true", // NOBIAS
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    weight_scale_ptr,");
  code.e("    input_scale_ptr,");
  code.e("    /*weight_scale_row_stride=*/$,", packed_scale_k);
  code.e("    /*input_scale_row_stride=*/$,", packed_scale_k);
  code.e("    mBias,");
  code.e("    tma_out);");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_FP8_SWAPAB_WITH_RESIDUAL_SM100,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_FP8_SWAPAB_SM100,
                                 code.to_string());
  }
}

int TaskRegister::register_splitk_linear_fp8_swapAB_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Inputs (Python-layer order): input_fp8, input_scale, weight_fp8,
  // weight_scale. No residual variant for split-K (matches BF16 split-K).
  // Output: output_bf16, pre-zeroed by the caller — kernel uses TMA reduce-add.
  //
  // Grid layout from `linear_splitk_swapAB_fp8_layer`:
  //   grid.x splits OUTPUT (M) → per-task output_size = full_OUT / grid.x
  //   grid.y splits K          → per-task K          = full_K   / grid.y
  // Each CTA at (gx, gy) computes its own (M_shard, K_shard) and reduce-adds
  // into the (gx)-th output slice.
  assert(params.size() == 0);
  int batch_size = 0, output_size_per_task = 0, reduction_size_per_task = 0;
  int reduction_size_full = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
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
  // STensor (per-task tile) reflects the partitioned shape after grid-split.
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size_per_task = output_ops[0]->output_tensors[0].dim[1];
  // Per-task K = the partitioned input's reduction extent.
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  reduction_size_per_task = input_ops[0]->output_tensors[0].dim[1];
  // Full K = the un-partitioned DTensor's reduction extent. Used for the
  // gmem row stride of the packed-scale buffers (one uint32 per 128-K row,
  // packed 4 to a uint32).
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size_full = input_ops[0]->dtensor.dim[1];

  assert(output_size_per_task % 128 == 0 &&
         "splitk_linear_fp8_swapAB_sm100 requires per-task output divisible by "
         "128");
  assert(batch_size <= 16 && "splitk_linear_fp8_swapAB_sm100 is decode-only: "
                             "BATCH_SIZE must be <= 16");
  assert(reduction_size_per_task % 128 == 0 &&
         "splitk_linear_fp8_swapAB_sm100 requires per-task K divisible by "
         "BLOCK_K=128");
  // Stronger constraint: K_per_task must be a multiple of 512 (= BLOCK_K * 4)
  // because UE8M0 scales are packed 4 logical-K per uint32. Picking a
  // split_k_factor that violates this would land slice boundaries inside a
  // packed uint32 and the per-CTA scale-pointer base offset would misalign.
  assert(reduction_size_per_task % 512 == 0 &&
         "splitk_linear_fp8_swapAB_sm100 requires K_per_task divisible by 512 "
         "(split_k_factor must divide full_K / 512 evenly)");
  assert(reduction_size_full % reduction_size_per_task == 0 &&
         "full K must be a multiple of per-task K (uniform split)");

  int output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 128;
  constexpr int TILE_SIZE = 128;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;

  // tma_a = WEIGHT (after swap A-side). Per-task K extent (TBGraph already
  // sliced base_ptr for this CTA's K-shard).
  code.e("using TMA_A = kernel::tma::tma_2d<cutlass::float_e4m3_t, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         output_size_per_task,    /*GMEM_ROW_*/
         reduction_size_per_task, /*GMEM_COL_*/
         MMA_M,                   /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,       /*SMEM_COL_*/
         reduction_size_full,     /*GMEM_STRIDE_ROW_ — full K row stride */
         1,                       /*GMEM_STRIDE_COL_*/
         1,                       /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE,
         MMA_M * TMA_CP_ASYNC_SIZE);
  // tma_b = INPUT (after swap B-side). Per-task K extent.
  code.e("using TMA_B = kernel::tma::tma_2d<cutlass::float_e4m3_t, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,
         reduction_size_per_task,
         MMA_N,
         TMA_CP_ASYNC_SIZE,
         reduction_size_full, /*GMEM_STRIDE_ROW_*/
         1,
         1,
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE,
         MMA_N * TMA_CP_ASYNC_SIZE);
  // tma_out = OUTPUT BF16 [BATCH, OUTPUT_per_task]. Same as non-split — all
  // grid.y CTAs target the same M-shard for reduce-add.
  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, "
         "$, $, $, $, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,
         output_size_per_task,
         MMA_N,
         MMA_M,
         output_stride,
         1,
         1,
         (output_atom_size + output_tma_cp_size - 1) / output_tma_cp_size,
         MMA_N * MMA_M);

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e("uint32_t const *weight_scale_ptr = "
         "static_cast<uint32_t const*>(task_desc->input_ptrs[3]);");
  code.e("uint32_t const *input_scale_ptr  = "
         "static_cast<uint32_t const*>(task_desc->input_ptrs[1]);");

  // NOBIAS path only — no residual variant for split-K.
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size_per_task,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "nullptr)), layout_Bias);");

  // Scale row stride: gmem buffer's row stride = full_packed_K.
  // Per-task PACKED_SCALE_K (kernel template) covers only the K-slice this
  // CTA will read; the runtime row_stride arg lets the kernel index into
  // the K-shard at the right base.
  int const packed_scale_k_full = (reduction_size_full + 511) / 512;

  code.e("kernel::linear_fp8_swapAB_sm100_task_impl<cutlass::float_e4m3_t, "
         "TMA_A, TMA_B, decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, /*NOBIAS=*/true, /*SplitK=*/true, $, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size_per_task,
         reduction_size_per_task,
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    weight_scale_ptr,");
  code.e("    input_scale_ptr,");
  code.e("    /*weight_scale_row_stride=*/$,", packed_scale_k_full);
  code.e("    /*input_scale_row_stride=*/$,", packed_scale_k_full);
  code.e("    mBias,");
  code.e("    tma_out);");

  return register_task_variant(TASK_SPLITK_LINEAR_FP8_SWAPAB_SM100,
                               code.to_string());
}

int TaskRegister::register_linear_fp8_bmm_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Per-head FP8 batched matmul. Each CTA handles one head's slice of
  //     output[n, h, m_lo:m_hi] = input[n, h, :] @ weight[h, m_lo:m_hi, :]^T
  // chosen by (grid.x = M-shard within a head, grid.y = head index).
  //
  // Inputs (Python-layer order, all 3D):
  //   [0] input_fp8     [N, H, D_in]
  //   [1] input_scale   [N, H, packed_K]   (UE8M0 packed, 4 logical scales /
  //   uint32) [2] weight_fp8    [H, D_out, D_in] [3] weight_scale  [H, D_out,
  //   packed_K]
  // Output:
  //   [0] output_bf16   [N, H, D_out]
  //
  // SwapAB wiring is the same as the non-BMM swapAB kernel: weight (slot 2)
  // -> tma_a, input (slot 0) -> tma_b. The only BMM-specific differences are
  // the per-head row strides on the input/output TMAs and on the input_scale
  // row stride — both spanning the H dimension.
  assert(params.size() == 0);

  int num_inputs = 4;
  int num_outputs = 1;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Weight is always 3D [H, D_out, D_in]. Input and output may be 2D
  // (N, H*D_*) or 3D (N, H, D_*) — same byte layout, identical TMA
  // strides. Accept either so callers don't need a reshape kernel.
  assert(input_ops[2]->dtensor.num_dims == 3); // weight 3D required
  int num_heads = input_ops[2]->dtensor.dim[0];
  int D_out_full = input_ops[2]->dtensor.dim[1];
  int reduction_size = input_ops[2]->dtensor.dim[2];

  int const in_dims = input_ops[0]->dtensor.num_dims;
  int const out_dims = output_ops[0]->dtensor.num_dims;
  assert(in_dims == 2 || in_dims == 3);
  assert(out_dims == 2 || out_dims == 3);
  int batch_size = input_ops[0]->dtensor.dim[0];
  if (in_dims == 3) {
    assert(input_ops[0]->dtensor.dim[1] == num_heads);
    assert(input_ops[0]->dtensor.dim[2] == reduction_size);
  } else {
    assert(input_ops[0]->dtensor.dim[1] == num_heads * reduction_size);
  }
  assert(output_ops[0]->dtensor.dim[0] == batch_size);
  if (out_dims == 3) {
    assert(output_ops[0]->dtensor.dim[1] == num_heads);
    assert(output_ops[0]->dtensor.dim[2] == D_out_full);
  } else {
    assert(output_ops[0]->dtensor.dim[1] == num_heads * D_out_full);
  }

  // Per-task M-tile (per-head output shard) and per-CTA head count.
  int grid_x = bgraph.grid_dim.x;
  int grid_y = bgraph.grid_dim.y;
  assert(grid_x >= 1 && grid_y >= 1);
  assert(D_out_full % grid_x == 0 &&
         "linear_fp8_bmm_sm100: D_out must be divisible by grid_dim.x");
  assert(num_heads % grid_y == 0 &&
         "linear_fp8_bmm_sm100: H must be divisible by grid_dim.y");
  int output_size_per_task = D_out_full / grid_x;
  int heads_per_task = num_heads / grid_y;

  // First cut: one head per CTA. The kernel forwards to the existing swapAB
  // GEMM body, which only knows about a single (M, N, K) tile — so multi-head
  // fusion would need an outer loop in linear_fp8_bmm_sm100_task_impl.
  assert(heads_per_task == 1 &&
         "linear_fp8_bmm_sm100 currently supports only H_PER_TASK=1; "
         "set grid_dim.y == H to give each CTA exactly one head.");

  // Constraints inherited from swapAB:
  //   - per-task M (= D_out_per_task) must be a multiple of MMA_M=128.
  //   - decode-only: batch <= MMA_N=16.
  //   - K must be a multiple of BLOCK_K=128.
  assert(output_size_per_task % 128 == 0 &&
         "linear_fp8_bmm_sm100 requires per-task D_out divisible by 128");
  assert(batch_size <= 16 &&
         "linear_fp8_bmm_sm100 is decode-only: BATCH_SIZE must be <= 16");
  assert(reduction_size % 128 == 0 &&
         "linear_fp8_bmm_sm100 requires D_in divisible by 128");

  // Row strides (in elements) for the global gmem tensors. With 3D inputs:
  //   input  [N, H, D_in] -> stride[0] = H * D_in
  //   weight [H, D_out, D_in] -> stride between rows within a head is D_in
  //   output [N, H, D_out] -> stride[0] = H * D_out
  // C20 (2026-05-17): read from dtensor.stride[0] instead of the owner_op's
  // input_strides[0] — view-safe (mpk.narrow inherits parent stride; root
  // tensors have stride populated by input.cc and fixup_legacy_strides).
  int input_row_stride = static_cast<int>(input_ops[0]->dtensor.stride[0]);
  int output_row_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);
  // Fallback if the recorded leading stride happens to be 0 (no source
  // tensor metadata): default to the contiguous packed strides.
  if (input_row_stride == 0) {
    input_row_stride = num_heads * reduction_size;
  }
  if (output_row_stride == 0) {
    output_row_stride = num_heads * D_out_full;
  }

  // Codegen: same MMA shape as the parent swapAB. Only the input/output TMA
  // strides differ (they now span the head dim).
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 128;
  constexpr int TILE_SIZE = 128;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;

  // tma_a = WEIGHT slice [D_out_per_task, D_in], row-major, row stride D_in.
  code.e("using TMA_A = kernel::tma::tma_2d<cutlass::float_e4m3_t, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         output_size_per_task, /*GMEM_ROW_*/
         reduction_size,       /*GMEM_COL_*/
         MMA_M,                /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,    /*SMEM_COL_*/
         reduction_size,       /*GMEM_STRIDE_ROW_*/
         1,                    /*GMEM_STRIDE_COL_*/
         1,                    /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  // tma_b = INPUT slice [N, D_in] for one head, row stride H*D_in (skips
  // past the other heads' per-token slices in the [N, H, D_in] layout).
  code.e("using TMA_B = kernel::tma::tma_2d<cutlass::float_e4m3_t, $, $, $, "
         "$, $, $, $, $, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         input_row_stride,  /*GMEM_STRIDE_ROW_ = H * D_in */
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  // tma_out = OUTPUT slice [N, D_out_per_task], row stride H*D_out.
  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, "
         "$, $, $, $, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,           /*GMEM_ROW_*/
         output_size_per_task, /*GMEM_COL_*/
         MMA_N,                /*SMEM_ROW_*/
         MMA_M,                /*SMEM_COL_*/
         output_row_stride,    /*GMEM_STRIDE_ROW_ = H * D_out */
         1,                    /*GMEM_STRIDE_COL_*/
         1,                    /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
  );
  code.inc_indent();

  // The runtime per-task base pointers in input_tma_desc_ptrs[i][0] already
  // include the head-and-M-tile offset derived from grid coords + the
  // TBGraph partition map, so the kernel just dereferences them as usual.
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  // Raw uint32* scale base pointers (per-task base, head-offset already
  // applied by the runtime). swapAB convention: weight_scale -> A-side.
  code.e("uint32_t const *weight_scale_ptr = "
         "static_cast<uint32_t const*>(task_desc->input_ptrs[3]);");
  code.e("uint32_t const *input_scale_ptr  = "
         "static_cast<uint32_t const*>(task_desc->input_ptrs[1]);");

  // No-bias placeholder: kernel branches on NOBIAS=true at compile time and
  // never dereferences mBias.
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size_per_task,
         output_row_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "nullptr)), layout_Bias);");

  // UE8M0 packed scale row strides. Weight: per-head, contiguous within the
  // head, so packed_K. Input: per-head view of [N, H, packed_K], so the
  // row stride in uint32 elements is H * packed_K.
  int const packed_scale_k = (reduction_size + 511) / 512;
  int const input_scale_row_stride = num_heads * packed_scale_k;

  // -------------------------------------------------------------------------
  // MPK_DSV3_BMM1_CUDA_CORE: env-gated ZERO-TMEM CUDA-core GEMV variant.
  // Same task type / same per-task inputs — just a different codegen body that
  // calls the no-tcgen05/no-TMA/no-mbarrier kernel with RAW per-task pointers.
  // The runtime already offsets every input_ptrs[i] / output_ptrs[0] to this
  // task's (head h = grid.y, shard s = grid.x) box via input_map/output_map
  // (see runtime.cc per-task base_ptr computation), so we pass them DIRECTLY
  // with NO additional h/s offset arithmetic.
  //
  // The cuda_core kernel templates:
  //   OUTPUT_SIZE_PER_TASK = N = output_size_per_task (= 128 for kv-up BMM1)
  //   REDUCTION_SIZE       = K = reduction_size       (= 128)
  //   BATCH_SIZE           = decode active-row tile, capped to <=16 (decode
  //                          active_rows<=8; rows beyond are correctness-dead,
  //                          and a smaller B keeps smem/compute bounded — the
  //                          backing tensor's full mbt is NOT used here).
  //   INPUT_ROW_STRIDE     = input_row_stride (= H*K, batch stride of input)
  //   INPUT_SCALE_STRIDE   = input_scale_row_stride (= H*packed_K, batch stride
  //                          of the per-(batch,head) UE8M0 scale)
  // weight_scale is read with implicit row-stride 1 (= packed_scale_k for
  // K<=512); the runtime already offset weight_scale_ptr to row 0 of this box.
  // output_row_stride = the bf16-element batch row stride of the output view
  // (= H*N_PER_HEAD; for q_nope_abs = q_nope_pe[:,:, :512] parent stride 18432).
  // Gate on VALUE == "1" (not mere presence): the A/B harness passes
  // MPK_DSV3_BMM1_CUDA_CORE=0 for the baseline arm, so `getenv != null` would
  // wrongly fire the gate for the "0" arm too. Match builder.py's == "1".
  // ===========================================================================
  // MPK_BMM1_NOOP framework-floor probe (Codex discriminator #3). When the
  // megakernel is JIT-compiled with -DMPK_BMM1_NOOP (set via
  // MPK_EXTRA_NVCC_DEFINES="-DMPK_BMM1_NOOP"), the LINEAR_FP8_BMM device body
  // does NOTHING except write zeros to its output box and return — skipping all
  // TMA loads / tcgen05 / MMA / scale-decode / output TMA-store. This isolates
  // the per-task dispatch+framework FLOOR (scheduler dispatch + worker
  // task-entry) from the kernel's own load+compute time. The zero-write keeps
  // downstream FP8-quantize from NaN-cascading. The block is gated by the nvcc
  // preprocessor define, NOT a runtime getenv, so the SAME compiled library
  // serves both A/B arms (real vs no-op) selected at megakernel JIT time.
  //
  // Emitted in BOTH the cuda_core and swapAB codegen strings so the gate fires
  // regardless of which device body is otherwise selected. Uses the RAW output
  // pointer task_desc->output_ptrs[0] + batch_size/output_size_per_task/
  // output_row_stride (all the runtime per-task box info), exactly like the
  // cuda_core variant's real output write — so the zero-write box is exact.
  auto emit_bmm1_noop_block = [&](mirage::transpiler::CodeKeeper &c) {
    c.e("#ifdef MPK_BMM1_NOOP");
    c.e("  {");
    c.e("    __nv_bfloat16 *noop_out = "
        "static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]);");
    c.e("    int noop_b = $, noop_n = $, noop_stride = $;",
        batch_size,
        output_size_per_task,
        output_row_stride);
    c.e("    for (int o = threadIdx.x; o < noop_b * noop_n; o += blockDim.x) {");
    c.e("      int bb = o / noop_n, nn = o - bb * noop_n;");
    c.e("      noop_out[(size_t)bb * noop_stride + nn] = __nv_bfloat16(0.0f);");
    c.e("    }");
    c.e("    return;");
    c.e("  }");
    c.e("#endif");
  };

  char const *bmm1_cc_env = std::getenv("MPK_DSV3_BMM1_CUDA_CORE");
  if (bmm1_cc_env != nullptr && std::atoi(bmm1_cc_env) == 1) {
    int const cuda_core_batch = batch_size < 16 ? batch_size : 16;
    emit_bmm1_noop_block(code);
    code.e("kernel::linear_fp8_bmm_cuda_core_sm100_task_impl<$, $, $, $, $>(",
           output_size_per_task, /*OUTPUT_SIZE_PER_TASK=N*/
           reduction_size,       /*REDUCTION_SIZE=K*/
           cuda_core_batch,      /*BATCH_SIZE*/
           input_row_stride,     /*INPUT_ROW_STRIDE=H*K*/
           input_scale_row_stride /*INPUT_SCALE_STRIDE=H*packed_K*/);
    code.e("    static_cast<const __nv_fp8_e4m3*>(task_desc->input_ptrs[2]),");
    code.e("    static_cast<const __nv_fp8_e4m3*>(task_desc->input_ptrs[0]),");
    code.e("    weight_scale_ptr,");
    code.e("    input_scale_ptr,");
    code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
    code.e("    /*output_row_stride=*/$);", output_row_stride);
    return register_task_variant(TASK_LINEAR_FP8_BMM_SM100, code.to_string());
  }

  emit_bmm1_noop_block(code);

  code.e("kernel::linear_fp8_bmm_sm100_task_impl<cutlass::float_e4m3_t, "
         "TMA_A, TMA_B, decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, /*NOBIAS=*/true, $, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size_per_task,
         reduction_size,
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    weight_scale_ptr,");
  code.e("    input_scale_ptr,");
  code.e("    /*weight_scale_row_stride=*/$,", packed_scale_k);
  code.e("    /*input_scale_row_stride=*/$,", input_scale_row_stride);
  code.e("    mBias,");
  code.e("    tma_out);");

  return register_task_variant(TASK_LINEAR_FP8_BMM_SM100, code.to_string());
}

int TaskRegister::register_linear_fp8_bmm_dense_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Per-head FP8 batched matmul wrapping the DENSE block-scaled GEMM body
  // (float32 scales) instead of swapAB (UE8M0). Each CTA handles one head's
  //     output[n, h, :] = input[n, h, :] @ weight[h, :, :]^T
  // chosen by (grid.x = M-shard within a head — first cut requires grid.x=1
  // since per-head D_out=128=BN, grid.y = head index).
  //
  // Inputs (Python-layer order, 3D):
  //   [0] input_fp8     [N, H, D_in]
  //   [1] input_scale   [N, H, nk]          float32, row-major (nk = D_in/128)
  //   [2] weight_fp8    [H, D_out, D_in]
  //   [3] weight_scale  [H, D_out/128, nk]  float32, 128x128-block (D_out=128
  //                                          -> dim1 = 1)
  // Output:
  //   [0] output_bf16   [N, H, D_out]
  //
  // Dense body A/B assignment is the OPPOSITE of swapAB: A = input (param 0,
  // TMA desc slot 0), B = weight (param 2, TMA desc slot 2). The float32
  // scales are raw pointers (sa = input_ptrs[1], sb = input_ptrs[3]). The
  // bf16 output is a raw pointer (output_ptrs[0]).
  assert(params.size() == 0);

  int num_inputs = 4;
  int num_outputs = 1;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Weight is 3D [H, D_out, D_in].
  assert(input_ops[2]->dtensor.num_dims == 3);
  int num_heads = input_ops[2]->dtensor.dim[0];
  int D_out_full = input_ops[2]->dtensor.dim[1];
  int reduction_size = input_ops[2]->dtensor.dim[2];

  // Input is 3D [N, H, D_in]; output is 2D [N, H*D_out] or 3D [N, H, D_out].
  assert(input_ops[0]->dtensor.num_dims == 3 &&
         "linear_fp8_bmm_dense requires 3D input [N, H, D_in] so the per-head "
         "TMA descriptor can read M=dim0, K=dim2, row_stride=stride0.");
  int batch_size = input_ops[0]->dtensor.dim[0];
  assert(input_ops[0]->dtensor.dim[1] == num_heads);
  assert(input_ops[0]->dtensor.dim[2] == reduction_size);

  int const out_dims = output_ops[0]->dtensor.num_dims;
  assert(out_dims == 2 || out_dims == 3);
  assert(output_ops[0]->dtensor.dim[0] == batch_size);
  if (out_dims == 3) {
    assert(output_ops[0]->dtensor.dim[1] == num_heads);
    assert(output_ops[0]->dtensor.dim[2] == D_out_full);
  } else {
    assert(output_ops[0]->dtensor.dim[1] == num_heads * D_out_full);
  }

  // Scales are float32 3D: input_scale [N, H, nk], weight_scale [H, 1, nk].
  int const nk = (reduction_size + 127) / 128;
  assert(input_ops[1]->dtensor.num_dims == 3);
  assert(input_ops[1]->dtensor.dim[0] == batch_size);
  assert(input_ops[1]->dtensor.dim[1] == num_heads);
  assert(input_ops[1]->dtensor.dim[2] == nk &&
         "linear_fp8_bmm_dense: input_scale last dim must be D_in/128 "
         "(float32 1x128-group activation scale)");
  assert(input_ops[3]->dtensor.num_dims == 3);
  assert(input_ops[3]->dtensor.dim[0] == num_heads);
  assert(input_ops[3]->dtensor.dim[2] == nk &&
         "linear_fp8_bmm_dense: weight_scale last dim must be D_in/128 "
         "(float32 128x128-block weight scale)");

  // Grid: grid.x = M-shard within a head (must be 1 for D_out=128=BN),
  // grid.y = head index (must equal H — one head per CTA).
  int grid_x = bgraph.grid_dim.x;
  int grid_y = bgraph.grid_dim.y;
  assert(grid_x == 1 &&
         "linear_fp8_bmm_dense: grid.x must be 1 (per-head D_out=128=BN, the "
         "dense body computes the whole per-head N-tile in one CTA).");
  assert(num_heads % grid_y == 0 &&
         "linear_fp8_bmm_dense: H must be divisible by grid_dim.y");
  int heads_per_task = num_heads / grid_y;
  assert(heads_per_task == 1 &&
         "linear_fp8_bmm_dense currently supports only H_PER_TASK=1; "
         "set grid_dim.y == H to give each CTA exactly one head.");

  // Dense body constraints: BK=128, BN=128. D_out (=N per head) must be a
  // multiple of 128; D_in (=K) must be a multiple of 128. Decode-only M small.
  assert(D_out_full % 128 == 0 &&
         "linear_fp8_bmm_dense requires per-head D_out divisible by 128");
  assert(reduction_size % 128 == 0 &&
         "linear_fp8_bmm_dense requires D_in divisible by 128");

  // Per-head activation-scale row stride: scale is [N, H, nk] row-major, so
  // consecutive M-rows of one head stride by H*nk floats. The per-head base
  // (head h -> +h*nk) is applied by the runtime via the partition map; we
  // only need the row stride here.
  int const sa_row_stride = num_heads * nk;
  // Per-head bf16-output row stride: output is [N, H, D_out] so a row strides
  // by H*D_out. (For 2D [N, H*D_out] the stride is identical.)
  int const C_row_stride = num_heads * D_out_full;

  // BN=128, NS=3, NE=1 (BMM-specific, decode-only). Diagnostic memcheck
  // showed the megakernel-context tcgen05.alloc returns taddr=0 with NE=2
  // (TCA=256 contends with concurrent tcgen05 tasks' TMEM on the same SM,
  // 2×256 = the 512-col SM limit). NE=1 halves the BMM's TMEM ask (TCA=128)
  // so the alloc fits. BMM is M=1 decode-only so NE=1 (no TMEM pipeline)
  // costs no perf — only one MMA call per CTA. SMALLM/MEDIUMM keep NE=2.
  constexpr int BN = 128;
  constexpr int NS = 3;
  constexpr int NE = 1;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // Decode-only active-rows cap, mirroring the dense GEMM variants: the
  // BMM only runs on decode iters, and we clip M to the active token count
  // so non-active output rows keep their prior content.
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;",
         batch_size,
         batch_size);
  code.e("if (runtime_m_ <= 0) return;");
  // Kernel call: <BN, NS, NE>(ta=A(input), tb=B(weight), sa, sb, C, M, N, K,
  //                           sa_row_stride, C_row_stride).
  code.e("kernel::linear_fp8_bmm_dense::linear_fp8_bmm_dense_sm100_task_impl<"
         "$, $, $>(",
         BN,
         NS,
         NE);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A = input
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // B = weight
  code.e("    static_cast<const float*>(task_desc->input_ptrs[1]),");    // sa
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");    // sb
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),"); // C
  code.e("    runtime_m_,");
  code.e("    $,", D_out_full);     // N = per-head D_out
  code.e("    $,", reduction_size); // K = D_in
  code.e("    $,", sa_row_stride);
  code.e("    $);", C_row_stride);
  code.e("}");

  return register_task_variant(TASK_LINEAR_FP8_BMM_DENSE_SM100,
                               code.to_string());
}

// D3 (2026-06-03): fp8-out flavor of the per-head dense BMM. Same A/B/scale
// input plumbing as the bf16 variant, but the epilogue fuses the downstream
// float32-scale per-token-group quantize. Tuple is (4 inputs, 2 outputs):
//   output_ptrs[0] = FP8 buffer  [N, H*D_out] row-major
//   output_ptrs[1] = float32 scale [N, H] row-major (one K-group per head)
// Behind MPK_DSV3_FUSE_EPILOGUE_QUANT (builder wires it only when also under
// MPK_DSV3_BMM_DENSE). The fused FP8 row stride (H*D_out) and scale row stride
// (H) are passed to the kernel; the per-head base pointer comes from the
// runtime partition map (output split on H), same as the bf16 variant.
int TaskRegister::register_linear_fp8_bmm_dense_fp8out_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);

  int num_inputs = 4;
  int num_outputs = 2;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Weight is 3D [H, D_out, D_in].
  assert(input_ops[2]->dtensor.num_dims == 3);
  int num_heads = input_ops[2]->dtensor.dim[0];
  int D_out_full = input_ops[2]->dtensor.dim[1];
  int reduction_size = input_ops[2]->dtensor.dim[2];

  // Input is 3D [N, H, D_in].
  assert(input_ops[0]->dtensor.num_dims == 3 &&
         "linear_fp8_bmm_dense_fp8out requires 3D input [N, H, D_in].");
  int batch_size = input_ops[0]->dtensor.dim[0];
  assert(input_ops[0]->dtensor.dim[1] == num_heads);
  assert(input_ops[0]->dtensor.dim[2] == reduction_size);

  // Output[0] = FP8, 2D [N, H*D_out] or 3D [N, H, D_out].
  int const out_dims = output_ops[0]->dtensor.num_dims;
  assert(out_dims == 2 || out_dims == 3);
  assert(output_ops[0]->dtensor.dim[0] == batch_size);
  if (out_dims == 3) {
    assert(output_ops[0]->dtensor.dim[1] == num_heads);
    assert(output_ops[0]->dtensor.dim[2] == D_out_full);
  } else {
    assert(output_ops[0]->dtensor.dim[1] == num_heads * D_out_full);
  }
  // Output[1] = float32 scale. One K-group per head (D_out=128=BN), so the
  // scale row holds exactly num_heads floats: 2D [N, H] or 3D [N, H, 1].
  int const sc_dims = output_ops[1]->dtensor.num_dims;
  assert(sc_dims == 2 || sc_dims == 3);
  assert(output_ops[1]->dtensor.dim[0] == batch_size);
  if (sc_dims == 2) {
    assert(output_ops[1]->dtensor.dim[1] == num_heads);
  } else {
    assert(output_ops[1]->dtensor.dim[1] == num_heads &&
           output_ops[1]->dtensor.dim[2] == 1);
  }

  // Scales (inputs) are float32 3D: input_scale [N, H, nk], weight_scale
  // [H, 1, nk]. Same layout as the bf16 variant.
  int const nk = (reduction_size + 127) / 128;
  assert(input_ops[1]->dtensor.num_dims == 3);
  assert(input_ops[1]->dtensor.dim[0] == batch_size);
  assert(input_ops[1]->dtensor.dim[1] == num_heads);
  assert(input_ops[1]->dtensor.dim[2] == nk);
  assert(input_ops[3]->dtensor.num_dims == 3);
  assert(input_ops[3]->dtensor.dim[0] == num_heads);
  assert(input_ops[3]->dtensor.dim[2] == nk);

  int grid_x = bgraph.grid_dim.x;
  int grid_y = bgraph.grid_dim.y;
  assert(grid_x == 1 &&
         "linear_fp8_bmm_dense_fp8out: grid.x must be 1 (per-head D_out=128).");
  assert(num_heads % grid_y == 0 &&
         "linear_fp8_bmm_dense_fp8out: H must be divisible by grid_dim.y");
  int heads_per_task = num_heads / grid_y;
  assert(heads_per_task == 1 &&
         "linear_fp8_bmm_dense_fp8out currently supports only H_PER_TASK=1.");

  // Fused epilogue requires the per-head output width to be exactly one
  // K-group (D_out == BN == 128) so each consumer thread owns one group.
  assert(D_out_full == 128 &&
         "linear_fp8_bmm_dense_fp8out requires per-head D_out == 128 (one "
         "K-group per head for the fused float32-scale epilogue).");
  assert(reduction_size % 128 == 0 &&
         "linear_fp8_bmm_dense_fp8out requires D_in divisible by 128");

  // Per-head activation-scale row stride: scale is [N, H, nk] row-major.
  int const sa_row_stride = num_heads * nk;

  constexpr int BN = 128;
  constexpr int NS = 3;
  constexpr int NE = 1;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // Decode-only active-rows cap, mirroring the bf16 variant.
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;",
         batch_size,
         batch_size);
  code.e("if (runtime_m_ <= 0) return;");
  // Kernel call: <BN, NS, NE>(ta=A(input), tb=B(weight), sa, sb, C_fp8,
  //                           C_scale_f32, M, N, K, sa_row_stride, gemm_h).
  code.e("kernel::linear_fp8_bmm_dense::"
         "linear_fp8_bmm_dense_fp8out_sm100_task_impl<$, $, $>(",
         BN,
         NS,
         NE);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A = input
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // B = weight
  code.e("    static_cast<const float*>(task_desc->input_ptrs[1]),");    // sa
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");    // sb
  code.e("    static_cast<__nv_fp8_e4m3*>(task_desc->output_ptrs[0]),"); // FP8
  code.e("    static_cast<float*>(task_desc->output_ptrs[1]),");         // sc
  code.e("    runtime_m_,");
  code.e("    $,", D_out_full);     // N = per-head D_out
  code.e("    $,", reduction_size); // K = D_in
  code.e("    $,", sa_row_stride);
  code.e("    $);", num_heads); // gemm_h => FP8 row stride H*N, scale stride H
  code.e("}");

  return register_task_variant(TASK_LINEAR_FP8_BMM_DENSE_FP8OUT_SM100,
                               code.to_string());
}

static int register_fp8_gemm_dense_variant(TaskRegister *self,
                                           std::vector<int> const &params,
                                           char const *namespace_name,
                                           char const *fn_name,
                                           TaskType task_type,
                                           int BN = 128,
                                           int NS = 3) {
  // params: [M, N, K, num_workers, optional runtime_m_mode]
  // runtime_m_mode=0 (default): use min(compile-time M, active_rows) as
  //   runtime M, where active_rows = qo_indptr_buffer[MAX_NUM_BATCHED_REQUESTS]
  //   = total active tokens in the current iter. This makes the GEMM's
  //   per-row write check (`if (mi < M)`) respect the active-token count,
  //   so decode iters don't overwrite output rows 1..MBT-1 with stale-
  //   FP8-driven garbage when the upstream quantize early-exited for
  //   non-active rows. See scratch/qkva_fusion_bug_FIXED.md for the
  //   multi-iter buffer-poisoning chain that motivated this fix
  //   (2026-05-13).
  // runtime_m_mode=1: gates to prompt-prefill and uses request 0's current
  //   kv_len as runtime M. This is used for ckv -> kv_b_proj decompression.
  // runtime_m_mode=2 (B20, 2026-05-15): prefill-phase gate (Q_LEN > 8) but
  //   keep active_rows as runtime M. Used for the prefill-only branch of
  //   the dual-dispatch O_proj (prefill O_proj reads attn_unabsorbed which
  //   is only valid on prefill iters; decode iters early-exit so the GEMM
  //   doesn't burn ~30 μs on a wasted wave).
  // runtime_m_mode=3 (B20, 2026-05-15): decode-phase gate (Q_LEN <= 8) with
  //   active_rows as runtime M. Used for the decode-only branch of the
  //   dual-dispatch O_proj (decode O_proj reads attn_out which is only
  //   valid on decode iters; prefill iters early-exit).
  assert(params.size() == 4 || params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int runtime_m_mode = (params.size() == 5) ? params[4] : 0;
  assert(runtime_m_mode >= 0 && runtime_m_mode <= 3);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  if (runtime_m_mode == 1) {
    code.e("int req_id_ = runtime_config.request_ids[0];");
    code.e("if (req_id_ < 0) return;");
    code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
           "runtime_config.qo_indptr_buffer[0];");
    code.e("bool prompt_prefill_ = runtime_config.step[req_id_] < "
           "runtime_config.prompt_length[req_id_] && q_len_ > 8;");
    code.e("if (!prompt_prefill_) return;");
    code.e("int fp_ = runtime_config.paged_kv_indptr_buffer[0];");
    code.e("int lp_ = runtime_config.paged_kv_indptr_buffer[1];");
    code.e("int runtime_m_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
           "runtime_config.paged_kv_last_page_len_buffer[0];");
  } else if (runtime_m_mode == 2 || runtime_m_mode == 3) {
    // Phase gate + active_rows cap. Mode 2 = prefill-only, 3 = decode-only.
    code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
           "runtime_config.qo_indptr_buffer[0];");
    if (runtime_m_mode == 2) {
      code.e("if (q_len_ <= 8) return;");
    } else {
      code.e("if (q_len_ > 8) return;");
    }
    code.e("int active_rows_ = "
           "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
    code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
    code.e("if (runtime_m_ <= 0) return;");
  } else {
    // 2026-05-13 ROOT-CAUSE FIX: cap M at active_rows so decode iters
    // (active_rows=1) don't overwrite output rows 1..M-1 with garbage.
    // This makes the GEMM's early-exit semantics symmetric with the
    // upstream quantize task's `request_id >= active_rows` early-exit:
    // the buffer's non-active rows retain their PREFILL content
    // (= correct), and the GEMM doesn't trash them in decode iters.
    code.e("int active_rows_ = "
           "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
    code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
    code.e("if (runtime_m_ <= 0) return;");
  }
  code.e("kernel::$::$<$, $>(", namespace_name, fn_name, BN, NS);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $);", num_workers);
  code.e("}");
  return self->register_task_variant(task_type, code.to_string());
}

int TaskRegister::register_fp8_gemm_dense_smallm_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  // NS-depth lever (2026-06-06): deepen the async K-pipeline NS=3->6 to hide the
  // unhidden weight-TMA latency that dominates these single-wave M=1 decode GEMMs
  // (L6 ledger + reviewer+Codex). NS = pipeline depth ONLY -> numerically IDENTICAL
  // output; same single-CTA-per-tile -> no crash; BN=128/NS=6 SMEM ~198KB < 216KB
  // cap (NS=8@BN128=263KB does NOT fit). Env-gated for a default-off A/B.
  // See experiment_history/NS_depth_lever_plan_20260606.md.
  int const ns = std::getenv("MPK_DSV3_SMALLM_NS6") ? 6 : 3;
  return register_fp8_gemm_dense_variant(
      this,
      params,
      "fp8_gemm_dense_smallm",
      "fp8_gemm_dense_smallm_sm100_task_impl",
      TASK_FP8_GEMM_DENSE_SMALLM_SM100,
      /*BN=*/128,
      /*NS=*/ns);
}

int TaskRegister::register_fp8_gemm_dense_mediumm_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_gemm_dense_variant(
      this,
      params,
      "fp8_gemm_dense_mediumm",
      "fp8_gemm_dense_mediumm_sm100_task_impl",
      TASK_FP8_GEMM_DENSE_MEDIUMM_SM100);
}

// fine-N variants (Gate-1, 2026-06-05): same dense bf16-out body as
// smallm/mediumm (4 inputs, 1 output, runtime_m_ active-rows cap) but emitted
// with BN<128 + a deeper NS pipeline, and the matching tma.cuh B-box=BN case.
// BN32/NS8 for qkv_a/gate_up (N<=4352 → ceil(N/32)<=136, 1 wave); BN64/NS4 for
// o_proj (N=7168 → BN32 would 2-wave). Caller must pass num_workers=ceil(N/BN)
// (see builder _fp8_dense_num_workers) or the GEMM silently multi-waves.
int TaskRegister::register_fp8_gemm_dense_finen_bn32_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_gemm_dense_variant(
      this,
      params,
      "fp8_gemm_dense_finen",
      "fp8_gemm_dense_finen_sm100_task_impl",
      TASK_FP8_GEMM_DENSE_FINEN_BN32_SM100,
      /*BN=*/32,
      /*NS=*/8);
}

int TaskRegister::register_fp8_gemm_dense_finen_bn64_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_gemm_dense_variant(
      this,
      params,
      "fp8_gemm_dense_finen",
      "fp8_gemm_dense_finen_sm100_task_impl",
      TASK_FP8_GEMM_DENSE_FINEN_BN64_SM100,
      /*BN=*/64,
      /*NS=*/4);
}

// D1 (2026-05-17): fp8out variant builder. Same as the bf16 variant but
// the kernel call emits FP8 + packed UE8M0 scale outputs. Task tuple is
// (4 inputs, 2 outputs): output_ptrs[0] = FP8 buffer, output_ptrs[1] =
// packed-scale uint32 buffer. params layout unchanged (M, N, K,
// num_workers, optional runtime_m_mode); `scale_outer_stride` is derived
// from N at codegen time (= N/128 = number of K-groups per row, since
// BN=128 and we statically restrict the fused path to BN=128).
static int
    register_fp8_gemm_dense_fp8out_variant(TaskRegister *self,
                                           std::vector<int> const &params,
                                           char const *namespace_name,
                                           char const *fn_name,
                                           TaskType task_type) {
  assert(params.size() == 4 || params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int runtime_m_mode = (params.size() == 5) ? params[4] : 0;
  assert(runtime_m_mode >= 0 && runtime_m_mode <= 3);
  // BN=128 fixed (per task_impl_tpl<BN=128, NS=3, NE=...>); each consumer
  // thread owns exactly one K-group → scale_outer_stride is the per-row
  // number of K-groups = N / 128.
  assert(N % 128 == 0 &&
         "fp8_gemm_dense_fp8out requires N divisible by 128 (one K-group "
         "per BN tile, per-row scale layout)");
  int scale_outer_stride = N / 128;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  if (runtime_m_mode == 1) {
    code.e("int req_id_ = runtime_config.request_ids[0];");
    code.e("if (req_id_ < 0) return;");
    code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
           "runtime_config.qo_indptr_buffer[0];");
    code.e("bool prompt_prefill_ = runtime_config.step[req_id_] < "
           "runtime_config.prompt_length[req_id_] && q_len_ > 8;");
    code.e("if (!prompt_prefill_) return;");
    code.e("int fp_ = runtime_config.paged_kv_indptr_buffer[0];");
    code.e("int lp_ = runtime_config.paged_kv_indptr_buffer[1];");
    code.e("int runtime_m_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
           "runtime_config.paged_kv_last_page_len_buffer[0];");
  } else if (runtime_m_mode == 2 || runtime_m_mode == 3) {
    code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
           "runtime_config.qo_indptr_buffer[0];");
    if (runtime_m_mode == 2) {
      code.e("if (q_len_ <= 8) return;");
    } else {
      code.e("if (q_len_ > 8) return;");
    }
    code.e("int active_rows_ = "
           "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
    code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
    code.e("if (runtime_m_ <= 0) return;");
  } else {
    code.e("int active_rows_ = "
           "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
    code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
    code.e("if (runtime_m_ <= 0) return;");
  }
  code.e("kernel::$::$<$, $>(", namespace_name, fn_name, 128, 3);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");
  code.e("    static_cast<__nv_fp8_e4m3*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<uint32_t*>(task_desc->output_ptrs[1]),");
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", num_workers);
  code.e("    $);", scale_outer_stride);
  code.e("}");
  return self->register_task_variant(task_type, code.to_string());
}

int TaskRegister::register_fp8_gemm_dense_smallm_fp8out_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_gemm_dense_fp8out_variant(
      this,
      params,
      "fp8_gemm_dense_smallm",
      "fp8_gemm_dense_smallm_fp8out_sm100_task_impl",
      TASK_FP8_GEMM_DENSE_SMALLM_FP8OUT_SM100);
}

int TaskRegister::register_fp8_gemm_dense_mediumm_fp8out_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_gemm_dense_fp8out_variant(
      this,
      params,
      "fp8_gemm_dense_mediumm",
      "fp8_gemm_dense_mediumm_fp8out_sm100_task_impl",
      TASK_FP8_GEMM_DENSE_MEDIUMM_FP8OUT_SM100);
}

// SplitK decode variant. params: [M, N, K, num_workers, SPLIT_K]. Always
// runs with runtime_m_mode=3 (decode-only Q_LEN<=8 + active_rows cap)
// inline; that gate is baked in (no override). Kernel template:
// <BN=128, NS=3, NE=2, SPLIT_K>. NE=2 matches the smallm sweet spot
// (decode active_rows is tiny so wide TMEM staging wastes registers).
//
// Caller MUST pre-zero the output tensor (the Python wrapper prepends a
// tensor_init); the kernel uses red.global.add.bf16x2 PTX atomics to
// accumulate SPLIT_K partials per (m_tile, n_tile).
int TaskRegister::register_fp8_gemm_dense_decode_splitk_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int split_k = params[4];
  // BK=128 in the kernel; nk = K / 128 must be divisible by split_k so
  // the K slice boundaries align with the per-128 scale rows.
  assert(K % (128 * split_k) == 0 &&
         "fp8_gemm_dense_decode_splitk requires K divisible by 128 * SPLIT_K");
  assert(split_k >= 1 && split_k <= 8 &&
         "fp8_gemm_dense_decode_splitk: SPLIT_K in [1, 8]");

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // runtime_m_mode=3 (decode-phase gate + active_rows cap) inline.
  code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
         "runtime_config.qo_indptr_buffer[0];");
  code.e("if (q_len_ > 8) return;");
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
  code.e("if (runtime_m_ <= 0) return;");
  // <BN, NS, NE, SPLIT_K> = <128, 3, 2, split_k>.
  code.e("kernel::fp8_gemm_dense_decode_splitk::fp8_gemm_dense_decode_splitk_"
         "sm100_task_impl<$, $, $, $>(",
         128,
         3,
         2,
         split_k);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $);", num_workers);
  code.e("}");
  return register_task_variant(TASK_FP8_GEMM_DENSE_DECODE_SPLITK_SM100,
                               code.to_string());
}

// Race-free internal split-K dense FP8 GEMM (2026-06-02), targeting the
// qkv_a shape (N=2176, K=7168). Distinct from the decode_splitk variant
// above: instead of the CRASHED red.global.add accumulate, this kernel
// writes exclusive FP32 partial slots, then a gen-tagged atomicAdd
// last-arriver reduces in FP32 + casts once to BF16 (fence.sc.gpu orders
// inter-CTA visibility). Two EXTRA input buffers are threaded in beyond the
// A/B/sa/sb of the base dense variant:
//   input_ptrs[4] = C_partial  (FP32 scratch, SPLIT_K*mm*nn*128*128 floats)
//   input_ptrs[5] = arrive_cnt (uint64, nn = ceil(N/128) entries)
//   output_ptrs[0] = C (BF16 output)
// arrive_cnt is zeroed each iter by a prepended tensor_init (see the Python
// wrapper); the gen-tag bump at the kernel tail is then redundant but
// harmless. params: [M, N, K, num_workers, SPLIT_K]. Kernel template
// <BN=128, NS=3, NE=2, SPLIT_K>. Decode-only gate (runtime_m_mode=3) inline.
int TaskRegister::register_fp8_gemm_dense_qkva_splitk_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int split_k = params[4];
  // BK=128 in the kernel; nk = K / 128 must be divisible by split_k so the K
  // slice boundaries align with the per-128 scale rows.
  assert(K % (128 * split_k) == 0 &&
         "fp8_gemm_dense_qkva_splitk requires K divisible by 128 * SPLIT_K");
  assert(split_k >= 1 && split_k <= 8 &&
         "fp8_gemm_dense_qkva_splitk: SPLIT_K in [1, 8]");

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // runtime_m_mode=3 (decode-phase gate + active_rows cap) inline.
  code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
         "runtime_config.qo_indptr_buffer[0];");
  code.e("if (q_len_ > 8) return;");
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
  code.e("if (runtime_m_ <= 0) return;");
  // <BN, NS, NE, SPLIT_K> = <128, 3, 2, split_k>.
  code.e("kernel::fp8_gemm_dense_qkva_splitk::fp8_gemm_dense_qkva_splitk_"
         "sm100_task_impl<$, $, $, $>(",
         128,
         3,
         2,
         split_k);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),");
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<float*>(task_desc->input_ptrs[4]),");
  code.e("    static_cast<uint64_t*>(task_desc->input_ptrs[5]),");
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $);", num_workers);
  code.e("}");
  return register_task_variant(TASK_FP8_GEMM_DENSE_QKVA_SPLITK_SM100,
                               code.to_string());
}

// External-reduce split-K GEMM. Same compute as the qkva_splitk kernel but
// EXT_REDUCE path: each K-slice CTA writes ONLY its exclusive FP32 partial into
// C_partial (the task OUTPUT) and returns — no arrive_cnt, no in-kernel
// last-arriver. A separate event-gated reduce task (registered below) sums the
// SPLIT_K partials into the BF16 output. ABI: 4 inputs (A,B TMA descriptors +
// sa,sb scales) + 1 output (C_partial FP32). The bf16 C pointer is unused on
// this path (passed nullptr).
int TaskRegister::register_fp8_gemm_dense_qkva_splitk_extred_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int split_k = params[4];
  assert(
      K % (128 * split_k) == 0 &&
      "fp8_gemm_dense_qkva_splitk_extred requires K divisible by 128*SPLIT_K");
  assert(split_k >= 1 && split_k <= 8 &&
         "fp8_gemm_dense_qkva_splitk_extred: SPLIT_K in [1, 8]");

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // runtime_m_mode=3 (decode-phase gate + active_rows cap) inline — identical
  // to the in-kernel-reduce variant so prefill iters early-exit.
  code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
         "runtime_config.qo_indptr_buffer[0];");
  code.e("if (q_len_ > 8) return;");
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
  code.e("if (runtime_m_ <= 0) return;");
  // <BN, NS, NE, SPLIT_K> = <128, 3, 2, split_k>.
  code.e("kernel::fp8_gemm_dense_qkva_splitk::fp8_gemm_dense_qkva_splitk_"
         "extred_sm100_task_impl<$, $, $, $>(",
         128,
         3,
         2,
         split_k);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");     // B
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),"); // sa
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),"); // sb
  code.e("    nullptr,");                                        // C (unused)
  code.e("    static_cast<float*>(task_desc->output_ptrs[0]),"); // C_partial
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $);", num_workers);
  code.e("}");
  return register_task_variant(TASK_FP8_GEMM_DENSE_QKVA_SPLITK_SM100,
                               code.to_string());
}

// Event-gated reduce companion for the external-reduce split-K GEMM. Reads the
// SPLIT_K FP32 partials in C_partial (input_ptrs[0]) and writes the reduced
// BF16 output (output_ptrs[0]). ABI: 1 input (C_partial) + 1 output (C bf16).
// params: [M, N, split_k, num_workers]. Persistent worker loop over output
// tiles (request_id = worker_idx).
int TaskRegister::register_fp8_gemm_dense_splitk_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  // params: [M, N, split_k, num_workers] or [..., ncol]. ncol (default 1) is
  // the per-tile column-split factor: work-unit count = ceil(M/128)*ceil(N/128)
  // *ncol, so more CTAs go active (fills idle SMs at decode M=1) and each does
  // less strided-partial-read work. BN=128 must be divisible by ncol and
  // (128/ncol) a multiple of 16.
  assert(params.size() == 4 || params.size() == 5);
  int M = params[0], N = params[1], split_k = params[2];
  int num_workers = params[3];
  int ncol = (params.size() == 5) ? params[4] : 1;
  assert(split_k >= 1 && split_k <= 8 &&
         "fp8_gemm_dense_splitk_reduce: SPLIT_K in [1, 8]");
  assert(128 % ncol == 0 && (128 / ncol) % 16 == 0 &&
         "fp8_gemm_dense_splitk_reduce: ncol must divide 128 with 128/ncol "
         "%16==0");

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // Decode-phase gate + active_rows cap mirror the GEMM so the reduce runs on
  // exactly the rows the GEMM wrote (and prefill iters early-exit together).
  code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
         "runtime_config.qo_indptr_buffer[0];");
  code.e("if (q_len_ > 8) return;");
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
  code.e("if (runtime_m_ <= 0) return;");
  // <BN, SPLIT_K, NCOL> = <128, split_k, ncol>.
  code.e("kernel::fp8_gemm_dense_splitk_reduce::fp8_gemm_dense_splitk_reduce_"
         "sm100_task_impl<$, $, $>(",
         128,
         split_k,
         ncol);
  code.e(
      "    static_cast<const float*>(task_desc->input_ptrs[0]),"); // C_partial
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),"); // C
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $);", num_workers);
  code.e("}");
  return register_task_variant(TASK_FP8_GEMM_DENSE_SPLITK_REDUCE_SM100,
                               code.to_string());
}

// ferret-produced FP8 split-K dense GEMM with a TMA reduce-add epilogue
// (cp.reduce.async.bulk.tensor.2d into a PRE-ZEROED bf16 output) instead of the
// per-element red.global.add atomics of the crashed decode_splitk variant.
// ABI: 4 inputs (A_fp8 TMA, B_fp8 TMA, sa float*, sb float*) + 1 output (C bf16
// TMA reduce-add). C MUST be tensor_init-zeroed before launch — the kernel
// accumulates into the zero base. The three TMA descriptors (A,B FP8 + C bf16
// reduce-add) are built host-side in create_tma_desc_by_task /
// fill_tma_desc_by_task; here we only pass the raw CUtensorMap* through.
// params: [M, N, K, num_workers, split_k] (mirrors the qkva_splitk task).
int TaskRegister::register_fp8_gemm_dense_splitk_tma_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int split_k = params[4];
  // BK=128 in the kernel; nk = K / 128 must be divisible by split_k so the K
  // slice boundaries align with the per-128 scale rows.
  assert(
      K % (128 * split_k) == 0 &&
      "fp8_gemm_dense_splitk_tma_reduce requires K divisible by 128*SPLIT_K");
  assert(split_k >= 1 && split_k <= 8 &&
         "fp8_gemm_dense_splitk_tma_reduce: SPLIT_K in [1, 8]");

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  // Decode-phase gate + active_rows cap, mirroring the qkva_splitk task so
  // prefill iters early-exit (the prefill path stays on the mediumm GEMM). The
  // pre-zero tensor_init runs unconditionally upstream; this gate skips only
  // the GEMM work on prefill iters (the dual-dispatch companion serves
  // prefill).
  code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
         "runtime_config.qo_indptr_buffer[0];");
  code.e("if (q_len_ > 8) return;");
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
  code.e("if (runtime_m_ <= 0) return;");
  // <BN, NS, NE, SPLIT_K> = <128, 3, 2, split_k>.
  // NE MUST stay 2 (2026-06-05): NE=4 (TCA=NE*MMA_N=64 TMEM cols) IMA-CRASHES at
  // TP=4 under MoE co-residency — empirically confirmed (Test D: backup .cuh @
  // NE=4 also rc=255; NE=2 @ TCA=32 is crash-free). The co-resident MLA-TP4 512-
  // col hold leaves a window where 32 cols fit but 64 don't (NE=3 also rounds to
  // TCA=64 -> same crash). NE=4's MMA<->epilogue overlap is sacrificed for crash-
  // safety; the big win (occupancy + the gflag pre-zero elimination) is NE-indep.
  code.e("kernel::fp8_gemm_dense_splitk_tma_reduce::fp8_gemm_dense_splitk_tma_"
         "reduce_sm100_task_impl<$, $, $, $>(",
         128,
         3,
         2,
         split_k);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");        // B
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),");    // sa
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");    // sb
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),"); // C
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,"); // worker_idx
  code.e("    $,", num_workers);
  code.e(
      "    static_cast<const "
      "CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]));"); // tc (C TMA)
  code.e("}");
  return register_task_variant(TASK_FP8_GEMM_DENSE_SPLITK_TMAREDUCE_SM100,
                               code.to_string());
}

// gflag variant (2026-06-05): same kernel/enum/TMA infra as above, but the
// epilogue eliminates the separate pre-zero tensor_init task via the OPT#1
// store/reduce-add fork — k_slice==0 CTAs tma_store (overwrite) then publish
// gflag[tile]=epoch; k_slice>0 CTAs spin until the epoch lands then reduce-add.
// 5 inputs (A, B, sa, sb, gflag) + 1 output (C); the gflag scratch is input[4]
// (a plain int* with no TMA desc — create_tma_desc_by_task only touches
// inputs 0/1 + output 0). epoch = runtime_config.step[0]+1 (bs=1 slot 0; step
// is mutated only at the iteration boundary by prepare_next_batch, so it is a
// uniform, monotone-per-decode-iter value across all CTAs of one instance — NOT
// step[request_id], which is hijacked to worker_idx=bid.x). HANG-SAFETY: at
// decode mm=ceil(runtime_m/128)=1, so total=nn*split_k; we ASSERT
// total<=num_workers so each worker owns exactly one tile-slice (k0 stores +
// publishes, never spins -> guaranteed progress -> no B36-style multi-tile
// deadlock). Reuses TASK_FP8_GEMM_DENSE_SPLITK_TMAREDUCE_SM100 so request_id /
// create_tma_desc / fill_tma all key on the unchanged enum.
int TaskRegister::register_fp8_gemm_dense_splitk_tma_reduce_gflag_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 5);
  int M = params[0], N = params[1], K = params[2], num_workers = params[3];
  int split_k = params[4];
  assert(K % (128 * split_k) == 0 &&
         "fp8_gemm_dense_splitk_tma_reduce_gflag requires K div by 128*SPLIT_K");
  assert(split_k >= 1 && split_k <= 8 &&
         "fp8_gemm_dense_splitk_tma_reduce_gflag: SPLIT_K in [1, 8]");
  // HANG-SAFETY contract (audited 2026-06-05): decode mm=1 => total=nn*split_k
  // must be <= num_workers so each worker owns one tile-slice. A worker count
  // below this pushes qkv_a into the (deadlock-fragile) multi-tile regime.
  int nn_ = (N + 127) / 128;
  int total_ = nn_ * split_k;
  assert(total_ <= num_workers &&
         "fp8_gemm_dense_splitk_tma_reduce_gflag: nn*split_k must be <= "
         "num_workers (gflag spin-wait is hang-safe only at total<=num_workers)");

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("int q_len_ = runtime_config.qo_indptr_buffer[1] - "
         "runtime_config.qo_indptr_buffer[0];");
  code.e("if (q_len_ > 8) return;");
  code.e("int active_rows_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("int runtime_m_ = active_rows_ < $ ? active_rows_ : $;", M, M);
  code.e("if (runtime_m_ <= 0) return;");
  // Call the 12-arg internal tpl directly (the public 11-arg wrapper hardcodes
  // gflag=nullptr). gflag=input_ptrs[4]; epoch=step[0]+1.
  // NE=2 (NOT 4): NE=4's TCA=64 TMEM cols IMA-crash at MoE co-residency (see the
  // base register fn note); NE=2 (TCA=32) is the crash-free footprint.
  code.e("kernel::fp8_gemm_dense_splitk_tma_reduce::task_impl_low_tmem_tpl<$, "
         "$, $, $>(",
         128,
         3,
         2,
         split_k);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");        // B
  code.e("    static_cast<const float*>(task_desc->input_ptrs[2]),");    // sa
  code.e("    static_cast<const float*>(task_desc->input_ptrs[3]),");    // sb
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),"); // C
  code.e("    runtime_m_,");
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    task_desc->task_metadata.request_id,"); // worker_idx
  code.e("    $,", num_workers);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]),"); // tc (C TMA)
  code.e("    static_cast<int*>(task_desc->input_ptrs[4]),");      // gflag
  code.e("    runtime_config.step[0] + 1);");                      // epoch
  code.e("}");
  return register_task_variant(TASK_FP8_GEMM_DENSE_SPLITK_TMAREDUCE_SM100,
                               code.to_string());
}

// Shared codegen for both group GEMM variants (smallm/largem). Variant
// only changes the namespace + function name + TaskType.
static int register_fp8_group_gemm_variant(TaskRegister *self,
                                           std::vector<int> const &params,
                                           char const *namespace_name,
                                           char const *fn_name,
                                           TaskType task_type) {
  // params: [M_total, N, K, E, num_workers, active_mask_offset]
  // active_mask_offset == -1 means caller did not supply a meta buffer;
  // pass nullptr so the kernel processes every tile (legacy behavior).
  // active_mask_offset >= 0 means input_ptrs[5] is the meta buffer (int32)
  // and the per-expert active mask lives at meta + active_mask_offset.
  assert(params.size() == 6);
  int M_total = params[0], N = params[1], K = params[2], E = params[3];
  int num_workers = params[4];
  int active_mask_offset = params[5];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::$::$(", namespace_name, fn_name);
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),"); // A
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),"); // B
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[2][0]),"); // SFA
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[3][0]),"); // SFB
  code.e("    static_cast<const "
         "CUtensorMap*>(task_desc->output_tma_desc_ptrs[0][0]),");  // D
  code.e("    static_cast<const int*>(task_desc->input_ptrs[4]),"); // m_indices
  if (active_mask_offset >= 0) {
    code.e("    static_cast<const int*>(task_desc->input_ptrs[5]) + "
           "$,",
           active_mask_offset); // active_expert_mask
  } else {
    code.e("    nullptr,"); // no active mask supplied
  }
  code.e("    $,", M_total);
  code.e("    $,", N);
  code.e("    $,", K);
  code.e("    $,", E);
  code.e("    task_desc->task_metadata.request_id,"); // worker_idx
  code.e("    $);", num_workers);
  return self->register_task_variant(task_type, code.to_string());
}

int TaskRegister::register_fp8_group_gemm_smallm_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_group_gemm_variant(
      this,
      params,
      "fp8_group_gemm_smallm",
      "fp8_group_gemm_smallm_sm100_task_impl",
      TASK_FP8_GROUP_GEMM_SMALLM_SM100);
}

int TaskRegister::register_fp8_group_gemm_largem_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  return register_fp8_group_gemm_variant(
      this,
      params,
      "fp8_group_gemm_largem",
      "fp8_group_gemm_largem_sm100_task_impl",
      TASK_FP8_GROUP_GEMM_LARGEM_SM100);
}

// moe_permute_sm100 — see moe_permute_sm100.cuh for the contract.
// Params (compile-time): [K, K_PACKED, MBT, TOPK, E_LOCAL, BM_PADDING]
// Inputs (4): input_fp8 (mbt, K) u8, input_scale (mbt, K_PACKED) u32 UE8M0,
//             topk_weights (mbt, TOPK) f32, routing_indices (E_LOCAL, MBT) i32
// Outputs (3): permuted_fp8 (M_TOTAL, K) u8,
//              permuted_scale (K_PACKED, M_TOTAL) u32 TRANSPOSED,
//              meta (M_TOTAL + MBT*TOPK,) i32 packing
//                 [0       : M_TOTAL]               = permuted_weights (f32
//                 bits) [M_TOTAL : M_TOTAL + MBT*TOPK]    = token_to_permuted
//                 (row+1)
// Static m_indices is set up by the builder via attach_input (constant
// pattern m_indices[r] = r / BM_PADDING) and consumed directly by the
// grouped GEMM — NOT emitted by this task.
// Grid: (E_LOCAL / E_PER_CTA, 1, 1). expert_offset = bid.x = CTA index
// (runtime.cc). The kernel owns experts [bid.x*E_PER_CTA,
// (bid.x+1)*E_PER_CTA). E_PER_CTA = params[6] (default 1; builder gates it
// via MPK_DSV3_PERMUTE_EPC). Passed as a runtime scalar (not a template
// arg) so raising it doesn't trigger a template-instantiation rebuild.
int TaskRegister::register_moe_permute_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 6 || params.size() == 7);
  int K = params[0], K_PACKED = params[1], MBT = params[2];
  int TOPK = params[3], E_LOCAL = params[4], BM_PADDING = params[5];
  int E_PER_CTA = (params.size() == 7) ? params[6] : 1;
  assert(E_PER_CTA >= 1 && E_LOCAL % E_PER_CTA == 0);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_permute_sm100_task_impl<$, $, $, $, $, $>(",
         K,
         K_PACKED,
         MBT,
         TOPK,
         E_LOCAL,
         BM_PADDING);
  code.e("    task_desc->input_ptrs[0],");  // input_fp8
  code.e("    task_desc->input_ptrs[1],");  // input_scale (packed UE8M0)
  code.e("    task_desc->input_ptrs[2],");  // topk_weights
  code.e("    task_desc->input_ptrs[3],");  // routing_indices
  code.e("    task_desc->output_ptrs[0],"); // permuted_fp8
  code.e("    task_desc->output_ptrs[1],"); // permuted_scale (transposed)
  code.e("    task_desc->output_ptrs[2],"); // meta (packed weights+tok2perm)
  code.e("    task_desc->task_metadata.expert_offset,"); // CTA index
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $);", E_PER_CTA); // experts owned by this CTA
  return register_task_variant(TASK_MOE_PERMUTE_SM100, code.to_string());
}

// moe_unpermute_sm100 — see moe_unpermute_sm100.cuh for the contract.
// Params (compile-time): [MBT, TOPK, HIDDEN, M_TOTAL]
// Inputs (3): permuted_output (M_TOTAL, HIDDEN) bf16,
//             meta (M_TOTAL + MBT*TOPK,) i32 (= permuted_weights+token2perm),
//             residual (MBT, HIDDEN) bf16
// Outputs (1): output (MBT, HIDDEN) bf16
// Grid: (ceil(MBT / ROWS_PER_TASK), 1, 1). request_id = bid.x set by runtime.
// B33 (2026-05-15): added ROWS_PER_TASK template — when the wrapper shrinks
// grid_dim.x below MBT, the kernel loops ROWS_PER_TASK = ceil(MBT / grid.x)
// tokens per CTA. Default grid.x == MBT keeps ROWS_PER_TASK == 1 and the
// legacy 1-CTA-per-token shape unchanged.
int TaskRegister::register_moe_unpermute_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int MBT = params[0], TOPK = params[1], HIDDEN = params[2],
      M_TOTAL = params[3];

  // Output stride: pull from the kn-level tensor like moe_mul_sum_add does.
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 3, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    auto *iop = static_cast<tb::TBInputOp *>(op);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(iop);
    } else {
      output_ops.push_back(iop);
    }
  }
  int output_stride = HIDDEN;
  output_stride = static_cast<int>(output_ops[0]->dtensor.stride[0]);

  // B33: rows_per_task = ceil(MBT / grid.x). When the wrapper passes
  // grid.x < MBT the kernel internally loops over multiple tokens per CTA
  // so the total launched CTAs stay ≤ num_workers. Default grid.x == MBT
  // gives rows_per_task == 1 (legacy 1-CTA-per-token contract).
  int grid_x_safe = bgraph.grid_dim.x > 0 ? (int)bgraph.grid_dim.x : 1;
  int rows_per_task = (MBT + grid_x_safe - 1) / grid_x_safe;
  if (rows_per_task < 1) {
    rows_per_task = 1;
  }

  // 2026-05-15 stragglers fix: HIDDEN_SPLIT = grid.y partitions the
  // HIDDEN axis across HIDDEN_SPLIT CTAs per token. Decode case
  // (1 active token) has only HIDDEN_SPLIT CTAs doing actual compute
  // — bumping HIDDEN_SPLIT spreads the 32 μs straggler across more
  // SMs in parallel. Prefill grid balloons to MBT*HIDDEN_SPLIT CTAs
  // but per-CTA work shrinks proportionally.
  int hidden_split = bgraph.grid_dim.y > 0 ? (int)bgraph.grid_dim.y : 1;
  if (hidden_split < 1) {
    hidden_split = 1;
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_unpermute_sm100_task_impl<$, $, $, $, $, $, $>(",
         MBT,
         TOPK,
         HIDDEN,
         M_TOTAL,
         output_stride,
         rows_per_task,
         hidden_split);
  code.e("    task_desc->input_ptrs[0],");  // permuted_output
  code.e("    task_desc->input_ptrs[1],");  // meta
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->task_metadata.kv_idx,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_MOE_UNPERMUTE_SM100, code.to_string());
}

// transpose_scale_sm100 — (M, K_PACKED) uint32 → (K_PACKED, M) uint32.
// Params: [M, K_PACKED]. 1 input, 1 output.
// B13 (2026-05-15): grid_dim.x CTAs stripe M; each CTA handles a
// disjoint chunk of rows. cta_idx / num_ctas passed via task metadata.
int TaskRegister::register_transpose_scale_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 2);
  int M = params[0], K_PACKED = params[1];
  int num_ctas = (int)bgraph.grid_dim.x;
  if (num_ctas < 1) {
    num_ctas = 1;
  }
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::transpose_scale_sm100_task_impl<$, $>(", M, K_PACKED);
  code.e("    task_desc->input_ptrs[0],");            // in (M, K_PACKED)
  code.e("    task_desc->output_ptrs[0],");           // out (K_PACKED, M)
  code.e("    task_desc->task_metadata.request_id,"); // cta_idx = bid.x
  code.e("    $);", num_ctas);
  return register_task_variant(TASK_TRANSPOSE_SCALE_SM100, code.to_string());
}

int TaskRegister::register_assemble_q_decode_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Interleaves (N, H, D_NOPE) + (N, H, D_PE) → (N, H, D_NOPE+D_PE) per head.
  //
  // Inputs (Python-layer order):
  //   [0] q_nope_abs  (N, H, D_NOPE=512) bf16   — BMM output
  //   [1] q_pe        (N, H, D_PE=64)    bf16   — q_b_pe FP8 dense GEMM output
  // Output:
  //   [0] q_nope_pe   (N, H, D_NOPE+D_PE=576) bf16
  //
  // params: [] (default) or [pe_only:int] (1=skip nope copy, used when BMM
  // wrote directly into q_nope_pe[:, :, :D_NOPE] via TMA-stride fuse).
  //
  // grid_dim = (N, 1, 1); partition (0, -1, -1) on all 3 tensors so each
  // CTA processes 1 token. n_active passed to the kernel = STensor.dim[0]
  // = N / grid.x.
  assert(params.size() == 0 || params.size() == 1);
  bool pe_only = (params.size() == 1) && (params[0] == 1);
  int num_inputs = 2, num_outputs = 1;
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Inputs are 3D (N, H, D_NOPE) and (N, H, D_PE). Output may be either
  // 3D (N, H, D_NOPE+D_PE) or 2D (N, H*(D_NOPE+D_PE)) — same byte layout,
  // accept either so callers can keep the existing 2D q_nope_pe buffer.
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  int const out_dims = output_ops[0]->output_tensors[0].num_dims;
  assert(out_dims == 2 || out_dims == 3);
  int n_per_task = output_ops[0]->output_tensors[0].dim[0];
  int D_NOPE = input_ops[0]->output_tensors[0].dim[2];
  int D_PE = input_ops[1]->output_tensors[0].dim[2];
  int H = input_ops[0]->output_tensors[0].dim[1];
  int D_TOTAL;
  if (out_dims == 3) {
    assert(output_ops[0]->output_tensors[0].dim[1] == H);
    D_TOTAL = output_ops[0]->output_tensors[0].dim[2];
  } else {
    assert(output_ops[0]->output_tensors[0].dim[1] == H * (D_NOPE + D_PE));
    D_TOTAL = D_NOPE + D_PE;
  }
  assert(input_ops[0]->output_tensors[0].dim[0] == n_per_task);
  assert(input_ops[1]->output_tensors[0].dim[0] == n_per_task);
  assert(input_ops[1]->output_tensors[0].dim[1] == H);
  assert(D_NOPE + D_PE == D_TOTAL);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Active-row gate: request_id = bid.x = this CTA's token index (set in
  // runtime.cc register_mugraph). The dim_map (0,-1,-1) already offset each
  // CTA's ptr to its token. At decode active_rows=1, only token 0 survives;
  // tokens 1..mbt-1 are padding (their q_nope_pe slots are never read by MLA
  // decode), so early-exiting their CTAs is safe and frees ~127 workers for
  // the concurrent attention-branch GEMMs. Correct for prefill too (keeps
  // CTAs 0..active_rows-1).
  code.e("int active_rows_aq_ = "
         "runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];");
  code.e("if (task_desc->task_metadata.request_id >= active_rows_aq_) return;");
  code.e("kernel::assemble_q_decode_sm100_task_impl<$, $, $, $>(",
         H,
         D_NOPE,
         D_PE,
         pe_only ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $);", n_per_task);
  return register_task_variant(TASK_ASSEMBLE_Q_DECODE_SM100, code.to_string());
}

int TaskRegister::register_mla_kv_gather_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params:
  //   size 3 (legacy): [d_k, d_v, page_size]
  //     c_latent input  is contiguous (mbt, d_v)            -> stride d_v
  //     k_pe     input  is contiguous (mbt, 128 padded)     -> stride 128
  //   size 7 (QKV-a fused): adds
  //     [c_latent_row_stride, c_latent_offset_elems,
  //      k_pe_row_stride,     k_pe_offset_elems]
  //   The two inputs may live at different col-offsets inside a wider buffer.
  assert(params.size() == 3 || params.size() == 7);

  int d_k = params[0];
  int d_v = params[1];
  int page_size = params[2];
  int c_latent_row_stride = (params.size() == 7) ? params[3] : d_v;
  int c_latent_offset = (params.size() == 7) ? params[4] : 0;
  int k_pe_row_stride = (params.size() == 7) ? params[5] : 128;
  int k_pe_offset = (params.size() == 7) ? params[6] : 0;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int S_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  auto *c_latent_new_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[0]) + "
         "qo_fp_ * $ + $;",
         c_latent_row_stride,
         c_latent_offset);
  code.e("  auto *k_pe_new_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[1]) + "
         "qo_fp_ * $ + $;",
         k_pe_row_stride,
         k_pe_offset);
  code.e("  auto *paged_cache_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[2]);");
  code.e("  auto *contiguous_kv_base_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[3]);");
  code.e("  auto *contiguous_kv_ptr_ = "
         "(contiguous_kv_base_ == paged_cache_ptr_) ? contiguous_kv_base_ : "
         "contiguous_kv_base_ + bi_ * S_ * $;",
         d_k);
  code.e("kernel::mla_kv_cache_gather_sm100_task_impl<$, $, $, $, $>(",
         d_k,
         d_v,
         page_size,
         k_pe_row_stride,
         c_latent_row_stride);
  code.e("    c_latent_new_ptr_,");
  code.e("    k_pe_new_ptr_,");
  code.e("    paged_cache_ptr_,"); // paged_cache
  code.e("    contiguous_kv_ptr_,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id);");
  code.e("}");
  return register_task_variant(TASK_MLA_KV_GATHER_SM100, code.to_string());
}

int TaskRegister::register_mla_kv_gather_split_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Split-output variant for the chunked-prefill MLA kernel. Same inputs as
  // the non-split variant plus TWO separate output pointers: ckv_sep and
  // kpe_sep. Layout: ckv_sep [max_seq, D_V=512], kpe_sep [max_seq, D_K-D_V=64].
  // params[0]: d_k, params[1]: d_v, params[2]: page_size
  assert(params.size() == 3);

  int d_k = params[0];
  int d_v = params[1];
  int page_size = params[2];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // k_pe_out in DeepSeek V3 builder is allocated as [mbt, 128] (padded from
  // ROPE_DIM=64 to MMA_M=128 for SM100 linear alignment). Real rope data is
  // in the first 64 cols per row; cols 64..127 are zero padding. Hence per-
  // token stride is K_PE_ROW_STRIDE=128, not ROPE_DIM=64.
  constexpr int k_pe_row_stride = 128;
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  auto *c_latent_new_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[0]) + "
         "qo_fp_ * $;",
         d_v);
  code.e("  auto *k_pe_new_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[1]) + "
         "qo_fp_ * $;",
         k_pe_row_stride);
  code.e("  auto *ckv_sep_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[3]) + "
         "bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_v);
  code.e("  auto *kpe_sep_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[4]) + "
         "bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_k - d_v);
  code.e("kernel::mla_kv_cache_gather_split_sm100_task_impl<$, $, $, $>(",
         d_k,
         d_v,
         page_size,
         k_pe_row_stride);
  code.e("    c_latent_new_ptr_,");
  code.e("    k_pe_new_ptr_,");
  code.e("    task_desc->input_ptrs[2],"); // paged_cache
  // ckv_sep and kpe_sep attached as new_input (store_in_dmem=True) on the
  // Python side — same convention as the non-split variant which treats
  // contiguous_kv as input.
  code.e("    ckv_sep_ptr_,");
  code.e("    kpe_sep_ptr_,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id);");
  code.e("}");
  return register_task_variant(TASK_MLA_KV_GATHER_SPLIT_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_kv_gather_unified_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // Unified variant for the chunked-prefill flow. It appends new KV to the
  // paged cache once. Decode gets a dense concatenated [CKV,KPE] view when
  // needed; prompt prefill gets split CKV/KPE views for kv_b_proj + PR674 MLA.
  //
  // params:
  //   size 3 (legacy): [d_k, d_v, page_size].
  //   size 7 (QKV-a fused): adds [c_latent_row_stride, c_latent_offset_elems,
  //                               k_pe_row_stride, k_pe_offset_elems].
  //   size 8 (gather fan-out, 2026-05-16 C1): also append [num_gather_splits].
  //   When NUM_GATHER_SPLITS > 1, the builder passes grid_dim.y = N_SPLITS, and
  //   each CTA strides seq_pos by N_SPLITS so the formerly-serial gather/append
  //   loops run in parallel across N_SPLITS workers.
  (void)bgraph;
  assert(params.size() == 3 || params.size() == 7 || params.size() == 8);

  int d_k = params[0];
  int d_v = params[1];
  int page_size = params[2];
  int c_latent_row_stride = (params.size() >= 7) ? params[3] : d_v;
  int c_latent_offset = (params.size() >= 7) ? params[4] : 0;
  int k_pe_row_stride = (params.size() >= 7) ? params[5] : 128;
  int k_pe_offset = (params.size() >= 7) ? params[6] : 0;
  int num_gather_splits = (params.size() == 8) ? params[7] : 1;
  assert(num_gather_splits >= 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int S_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  bool prompt_prefill_ = req_id_ >= 0 && "
         "runtime_config.step[req_id_] < "
         "runtime_config.prompt_length[req_id_];");
  code.e("  int q_len_ = runtime_config.qo_indptr_buffer[bi_ + 1] - "
         "runtime_config.qo_indptr_buffer[bi_];");
  code.e("  prompt_prefill_ = prompt_prefill_ && q_len_ > 8;");
  code.e("  auto *c_latent_new_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[0]) + "
         "qo_fp_ * $ + $;",
         c_latent_row_stride,
         c_latent_offset);
  code.e("  auto *k_pe_new_ptr_ = static_cast<const "
         "nv_bfloat16*>(task_desc->input_ptrs[1]) + "
         "qo_fp_ * $ + $;",
         k_pe_row_stride,
         k_pe_offset);
  code.e("  auto *paged_cache_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[2]);");
  code.e("  auto *contiguous_kv_base_ = "
         "static_cast<nv_bfloat16*>(task_desc->input_ptrs[3]);");
  code.e("  auto *contiguous_kv_ptr_ = "
         "(contiguous_kv_base_ == paged_cache_ptr_) ? contiguous_kv_base_ : "
         "contiguous_kv_base_ + bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_k);
  // ckv_sep and kpe_sep are now tracked as task outputs (registration
  // tuple changed from (6, 0) to (4, 2) so downstream consumers get
  // proper dependency edges from the gather). Read pointers from
  // output_ptrs accordingly.
  code.e("  auto *ckv_sep_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]) + "
         "bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_v);
  code.e("  auto *kpe_sep_ptr_ = "
         "static_cast<nv_bfloat16*>(task_desc->output_ptrs[1]) + "
         "bi_ * MPK_MAX_SEQ_LENGTH * $;",
         d_k - d_v);
  code.e(
      "kernel::mla_kv_cache_gather_unified_sm100_task_impl<$, $, $, $, $, $>(",
      d_k,
      d_v,
      page_size,
      k_pe_row_stride,
      c_latent_row_stride,
      num_gather_splits);
  code.e("    c_latent_new_ptr_,");
  code.e("    k_pe_new_ptr_,");
  code.e("    paged_cache_ptr_,"); // paged_cache
  code.e("    contiguous_kv_ptr_,");
  code.e("    ckv_sep_ptr_,");
  code.e("    kpe_sep_ptr_,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    prompt_prefill_,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->task_metadata.kv_idx);");
  code.e("}");
  return register_task_variant(TASK_MLA_KV_GATHER_UNIFIED_SM100,
                               code.to_string());
}

int TaskRegister::register_deepseek_mla_rope_q_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  assert(params.size() == 3);
  int num_heads = params[0];
  int tile_q = params[1];
  int has_split_q = params[2];
  assert(num_heads > 0);
  assert(tile_q > 0);
  assert(has_split_q == 0 || has_split_q == 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::deepseek_mla_rope_sm100_task_impl<$, $, $, true, false>(",
         num_heads,
         tile_q,
         has_split_q ? "true" : "false");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[1]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[1]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[2]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[3]),");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.request_ids,");
  code.e("    runtime_config.step,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->task_metadata.kv_idx,");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_DEEPSEEK_MLA_ROPE_SM100, code.to_string());
}

int TaskRegister::register_deepseek_mla_rope_q_fused_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  // params: [num_heads, tile_q [, phase_gate]]
  //   phase_gate (optional, default 0):
  //     0 = no gate (legacy)
  //     2 = decode-only (skip if Q_LEN > 8) — used for the fused
  //         absorbed-Q ROPE in dual-dispatch. On prefill iters the q_b
  //         decode GEMM early-exits via gate_mode=2, leaving q_nope_pe
  //         with stale data; rotating stale data is wasted work.
  assert(params.size() == 2 || params.size() == 3);
  int num_heads = params[0];
  int tile_q = params[1];
  int phase_gate = (params.size() == 3) ? params[2] : 0;
  assert(num_heads > 0);
  assert(tile_q > 0);
  assert(phase_gate == 0 || phase_gate == 2);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (phase_gate == 2) {
    code.e("{");
    code.e("int q_len_rope_ = runtime_config.qo_indptr_buffer[1] - "
           "runtime_config.qo_indptr_buffer[0];");
    code.e("if (q_len_rope_ > 8) return;");
    code.e("}");
  }
  code.e("kernel::deepseek_mla_rope_sm100_task_impl<$, $, false, true, false>(",
         num_heads,
         tile_q);
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[1]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[2]),");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.request_ids,");
  code.e("    runtime_config.step,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->task_metadata.kv_idx,");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_DEEPSEEK_MLA_ROPE_SM100, code.to_string());
}

int TaskRegister::register_deepseek_mla_rope_q_split_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  // params: [num_heads, tile_q [, qfused_mode [, phase_gate]]]
  //   qfused_mode (optional, default 0):
  //     0 = legacy separate q_pe buffer, row stride = num_heads * 64,
  //         per-head stride 64. q_pe is a standalone (mbt, H*64) tensor.
  //     1 = row-swap fused q_b_prefill_fused (mbt, H*192), pe slice starts
  //         at H*128 within each row. Per-head pe stride = 64. Used when
  //         MPK_DSV3_QB_FUSED=1.
  //   phase_gate (optional, default 0):
  //     0 = no gate (legacy)
  //     1 = prefill-only (skip if Q_LEN <= 8) — used for the unabsorbed-Q
  //         ROPE in dual-dispatch. On decode iters the q_b prefill GEMM
  //         early-exits via gate_mode=1 (and chunked_prefill itself
  //         returns), so rotating the (stale) q_b_prefill_fused buffer
  //         on decode iters is wasted work.
  assert(params.size() == 2 || params.size() == 3 || params.size() == 4);
  int num_heads = params[0];
  int tile_q = params[1];
  int qfused_mode = (params.size() >= 3) ? params[2] : 0;
  int phase_gate = (params.size() >= 4) ? params[3] : 0;
  assert(num_heads > 0);
  assert(tile_q > 0);
  assert(phase_gate == 0 || phase_gate == 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (phase_gate == 1) {
    code.e("{");
    code.e("int q_len_rope_ = runtime_config.qo_indptr_buffer[1] - "
           "runtime_config.qo_indptr_buffer[0];");
    code.e("if (q_len_rope_ <= 8) return;");
    code.e("}");
  }
  if (qfused_mode == 1) {
    // Row-swap fused layout: pe of head h at offset
    //   row * (num_heads * 192) + (num_heads * 128) + h * 64
    int const row_stride = num_heads * 192;
    int const pe_base = num_heads * 128;
    code.e("kernel::deepseek_mla_rope_sm100_task_impl<"
           "$, $, false, true, false, 64, 64, 64, $, $, 64>(",
           num_heads,
           tile_q,
           row_stride,
           pe_base);
  } else {
    code.e("kernel::deepseek_mla_rope_sm100_task_impl<"
           "$, $, false, true, false, 64, 64, 64>(",
           num_heads,
           tile_q);
  }
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[1]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[2]),");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.request_ids,");
  code.e("    runtime_config.step,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->task_metadata.kv_idx,");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_DEEPSEEK_MLA_ROPE_SM100, code.to_string());
}

int TaskRegister::register_deepseek_mla_rope_k_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  (void)bgraph;
  // params: [tile_q [, k_pe_row_stride [, k_pe_offset]]]
  // - Legacy (1 param): standalone k_pe buffer (mbt, 128) → K_PE_STRIDE=128,
  // K_PE_OFFSET=0.
  // - QKV-a fused (3 params): k_pe lives at offset `k_pe_offset` within a
  //   wider (mbt, k_pe_row_stride) buffer. Builder passes e.g. (16, 2176, 2048)
  //   so the kernel rotates qkv_a_out[:, 2048:2112] in place. Defaults keep
  //   legacy callers unaffected.
  assert(params.size() == 1 || params.size() == 3);
  int tile_q = params[0];
  assert(tile_q > 0);
  int k_pe_row_stride = (params.size() == 3) ? params[1] : 128;
  int k_pe_offset = (params.size() == 3) ? params[2] : 0;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Template params (positionally):
  //   NUM_HEADS=1, TILE_Q, HAS_SPLIT_Q=false, DO_Q=false, DO_K=true,
  //   FUSED_HEAD_DIM=576 (unused for K), ROPE_DIM=64, K_PE_STRIDE,
  //   Q_ROW_STRIDE_OVERRIDE=0, Q_PE_BASE_IN_ROW=0, Q_PE_HEAD_STRIDE=0,
  //   K_PE_OFFSET
  code.e("kernel::deepseek_mla_rope_sm100_task_impl<"
         "1, $, false, false, true, 576, 64, $, 0, 0, 0, $>(",
         tile_q,
         k_pe_row_stride,
         k_pe_offset);
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<__nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[1]),");
  code.e("    static_cast<const __nv_bfloat16*>(task_desc->input_ptrs[2]),");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.request_ids,");
  code.e("    runtime_config.step,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->task_metadata.kv_idx,");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_DEEPSEEK_MLA_ROPE_SM100, code.to_string());
}

int TaskRegister::register_mtp_verify_strict_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_draft_tokens (1-7)
  assert(params.size() == 1);
  int num_draft = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::target_verify_strict_kernel<$>(", num_draft);
  code.e("    task_desc->input_ptrs[0],");   // draft_token_ids
  code.e("    task_desc->input_ptrs[1],");   // target_token_ids
  code.e("    task_desc->output_ptrs[0],");  // accepted_count
  code.e("    task_desc->output_ptrs[1]);"); // output_tokens
  return register_task_variant(TASK_MTP_VERIFY_STRICT, code.to_string());
}

int TaskRegister::register_mtp_accept_commit_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_draft_tokens (1-7)
  assert(params.size() == 1);
  int num_draft = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mtp_accept_commit_kernel<$>(", num_draft);
  code.e("    task_desc->input_ptrs[0],");   // accepted_count
  code.e("    task_desc->input_ptrs[1],");   // output_tokens
  code.e("    task_desc->input_ptrs[2],");   // current_position
  code.e("    task_desc->output_ptrs[0],");  // new_position
  code.e("    task_desc->output_ptrs[1],");  // final_output
  code.e("    task_desc->output_ptrs[2]);"); // num_new_tokens
  return register_task_variant(TASK_MTP_ACCEPT_COMMIT, code.to_string());
}

int TaskRegister::register_mtp_token_scatter_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: batch_size, params[1]: num_slots, params[2]: slot_idx
  assert(params.size() == 3);
  int batch_size = params[0];
  int num_slots = params[1];
  int slot_idx = params[2];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mtp_token_scatter_kernel<$, $, $>(",
         batch_size,
         num_slots,
         slot_idx);
  code.e("    task_desc->input_ptrs[0],");   // src: single draft token
  code.e("    task_desc->output_ptrs[0]);"); // dst: all_draft_ids buffer
  return register_task_variant(TASK_MTP_TOKEN_SCATTER, code.to_string());
}

int TaskRegister::register_mtp_prepare_verify_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_draft_tokens, params[1]: max_seq_len
  assert(params.size() == 2);
  int num_draft = params[0];
  int max_seq_len = params[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::mtp_prepare_verify_input_kernel<$, $>(", num_draft, max_seq_len);
  code.e("    task_desc->input_ptrs[0],");  // main_token
  code.e("    task_desc->input_ptrs[1],");  // draft_tokens
  code.e("    task_desc->input_ptrs[2],");  // tokens_buffer
  code.e("    task_desc->input_ptrs[3],");  // step
  code.e("    task_desc->output_ptrs[0],"); // num_new_tokens
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.request_ids,");
  code.e("    task_desc->task_metadata.request_id);"); // request_id (not
                                                       // blockIdx.x)
  return register_task_variant(TASK_MTP_PREPARE_VERIFY, code.to_string());
}

int TaskRegister::register_mtp_build_embed_input_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: batch_size (mbt), params[1]: max_seq_len
  assert(params.size() == 2);
  int batch_size = params[0];
  int max_seq_len = params[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::mtp_build_embed_input_kernel<$, $>(", batch_size, max_seq_len);
  code.e("    task_desc->output_ptrs[0],"); // mtp_input_tokens (output)
  code.e("    runtime_config.tokens,");     // tokens_buffer (global)
  code.e("    task_desc->input_ptrs[0],");  // output_tokens (main argmax)
  code.e("    runtime_config.step,");       // step (global)
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.request_ids,");
  code.e("    task_desc->task_metadata.request_id);");
  return register_task_variant(TASK_MTP_BUILD_EMBED_INPUT, code.to_string());
}

// ============ MLA-MTP TP variants (ferret-derived, no-PDL) ============
//
// Three variants (TP=2/4/8) share structure but differ:
//   - NUM_HEADS hardcoded inside namespace (64/32/16) → not a runtime param
//   - TP=4 splits V across two CTAs via blockIdx.z (z=2 grid)
//   - TP=8 takes Q_LEN_real (Q_LEN is padded to even at the call site)
//
// Each TP has a paired (decode, reduce) task. params layout for decode:
//   [num_groups, q_len, kv_len, num_splits]                 (TP=2)
//   [num_groups, q_len, kv_len, num_splits, v_half]         (TP=4)
//   [num_groups, q_len_padded, kv_len, num_splits, q_len_real]  (TP=8)
// reduce params: [num_groups, q_len, num_splits, rd_dv]

int TaskRegister::register_mla_mtp_decode_tp2_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int num_groups = params[0];
  int q_len = params[1];
  int kv_len = params[2];
  int num_splits = params[3];
  int kvt = (kv_len + 128 - 1) / 128;
  int tps = (kvt + num_splits - 1) / num_splits;
  int single_tile = (tps == 1) ? 1 : 0;
  bool const write_final = (num_splits == 1);
  int qpg = (q_len < 2) ? q_len : 2;
  bool const direct_paged_kv = graph_input_has_num_dims(bgraph, 1, 3);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  // Compute runtime Q_LEN from qo_indptr (dual-dispatch: kernel uses this
  // to apply the Q_LEN>8 early-exit and correct causal masking).
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
  code.e("  if (q_len_rt_ > 8) return;");
  code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  if (single_tile) {
    code.e("  kernel::mla_mtp_tp2::mla_mtp_tp2_main<true, $>(",
           write_final ? "true" : "false");
  } else {
    code.e("  kernel::mla_mtp_tp2::mla_mtp_tp2_main<false, $>(",
           write_final ? "true" : "false");
  }
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("      static_cast<float*>(task_desc->output_ptrs[1]),");
  {
    float const _mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
    float const _sm = (1.0f / sqrtf(192.0f)) * _mscale * _mscale;
    code.e("      $f,", _sm);
  }
  code.e("      kv_len_,");
  // Use the runtime number of active KV tiles (sk_rt_) so the main writes
  // the partial buffer in a compact layout matching what the reduce reads.
  // Matches TP=4 pattern; passing compile-time num_splits made the kernel
  // index past the active si range and read stale partial slots in the
  // reduce — works for kv_len = num_splits * 128 but not for short context.
  code.e("      sk_rt_,");
  code.e("      q_len_rt_,");
  code.e("      $,", qpg);
  code.e("      $,",
         direct_paged_kv ? "runtime_config.paged_kv_indices_buffer"
                         : "nullptr");
  code.e("      fp_,");
  code.e("      task_desc->task_metadata.kv_idx,");
  code.e("      task_desc->task_metadata.request_id);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_TP2_SM100, code.to_string());
}

int TaskRegister::register_mla_mtp_decode_tp2_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int num_groups = params[0];
  int q_len = params[1];
  int num_splits = params[2];
  int rd_dv = params[3];
  int qpg = (q_len < 2) ? q_len : 2;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Dual-dispatch: pass runtime Q_LEN so reduce early-exit mirrors main.
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
  code.e("  if (q_len_rt_ > 8) return;");
  code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  code.e("  kernel::mla_mtp_tp2::mla_mtp_tp2_reduce(");
  code.e("      static_cast<const nv_bfloat16*>(task_desc->input_ptrs[0]),");
  code.e("      static_cast<const float*>(task_desc->input_ptrs[1]),");
  code.e("      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");
  // Match the main task's runtime compact split layout (sk_rt_). Passing
  // compile-time num_splits read stale partial slots beyond sk_rt_; the
  // main now also passes sk_rt_, so the two stay in sync for short context.
  code.e("      sk_rt_,");
  code.e("      $,", num_groups);
  code.e("      q_len_rt_,");
  code.e("      $,", qpg);
  code.e("      task_desc->task_metadata.kv_idx,");
  code.e("      task_desc->task_metadata.request_id,");
  code.e("      bi_);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_mtp_decode_tp4_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int num_groups = params[0];
  int q_len = params[1];
  int kv_len = params[2];
  int num_splits = params[3];
  int kvt = (kv_len + 128 - 1) / 128;
  int tps = (kvt + num_splits - 1) / num_splits;
  int single_tile = (tps == 1) ? 1 : 0;
  bool const write_final = (num_splits == 1);
  int qpg = (q_len < 4) ? q_len : 4;
  bool const direct_paged_kv = graph_input_has_num_dims(bgraph, 1, 3);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  // Dual-dispatch: pass runtime Q_LEN from qo_indptr for early-exit gate.
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
  code.e("  if (q_len_rt_ > 8) return;");
  code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  if (single_tile) {
    code.e("  kernel::mla_mtp_tp4::mla_mtp_tp4_main<true, $>(",
           write_final ? "true" : "false");
  } else {
    code.e("  kernel::mla_mtp_tp4::mla_mtp_tp4_main<false, $>(",
           write_final ? "true" : "false");
  }
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("      static_cast<float*>(task_desc->output_ptrs[1]),");
  {
    float const _mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
    float const _sm = (1.0f / sqrtf(192.0f)) * _mscale * _mscale;
    code.e("      $f,", _sm);
  }
  code.e("      kv_len_,");
  // Use the runtime number of active KV tiles. The graph still contains the
  // static worst-case task count, but tasks beyond sk_rt_ remap to gi >=
  // num_groups and return before touching partial buffers; reduce reads only
  // the compact active split range.
  code.e("      sk_rt_,");
  code.e("      q_len_rt_,");
  code.e("      $,", qpg);
  code.e("      $,",
         direct_paged_kv ? "runtime_config.paged_kv_indices_buffer"
                         : "nullptr");
  code.e("      fp_,");
  // V split is folded into block_x (no z-dim launch in MPK).
  // Python layer multiplies the grid; kernel unpacks the V part.
  code.e("      task_desc->task_metadata.kv_idx,"); // packed block/V-split id
  code.e("      task_desc->task_metadata.request_id);"); // batch
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_TP4_SM100, code.to_string());
}

int TaskRegister::register_mla_mtp_decode_tp4_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int num_groups = params[0];
  int q_len = params[1];
  int num_splits = params[2];
  int rd_dv = params[3];
  int qpg = (q_len < 4) ? q_len : 4;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Dual-dispatch: pass runtime Q_LEN so reduce early-exit mirrors main.
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_rt_ < 1) q_len_rt_ = 1;");
  code.e("  if (q_len_rt_ > 8) return;");
  code.e("  if (q_len_rt_ > $) q_len_rt_ = $;", q_len, q_len);
  code.e("  kernel::mla_mtp_tp4::mla_mtp_tp4_reduce(");
  code.e("      static_cast<const nv_bfloat16*>(task_desc->input_ptrs[0]),");
  code.e("      static_cast<const float*>(task_desc->input_ptrs[1]),");
  code.e("      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");
  // Match the main task's runtime compact split layout.
  code.e("      sk_rt_,");
  code.e("      $,", num_groups);
  code.e("      q_len_rt_,");
  code.e("      $,", qpg);
  code.e("      task_desc->task_metadata.kv_idx,");
  code.e("      task_desc->task_metadata.request_id,");
  code.e("      bi_);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100,
                               code.to_string());
}

int TaskRegister::register_mla_mtp_decode_tp8_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int num_groups = params[0];
  int q_len_padded = params[1];
  int kv_len = params[2];
  int num_splits = params[3];
  int q_len_real = params[4];
  int kvt = (kv_len + 128 - 1) / 128;
  int tps = (kvt + num_splits - 1) / num_splits;
  int single_tile = (tps == 1) ? 1 : 0;
  bool const write_final = (num_splits == 1);
  int qpg = 2;
  bool const direct_paged_kv = graph_input_has_num_dims(bgraph, 1, 3);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.request_id;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  // Dual-dispatch: pass runtime Q_LEN_real; pad to even for Q_LEN_padded.
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_real_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_real_rt_ < 1) q_len_real_rt_ = 1;");
  code.e("  if (q_len_real_rt_ > 8) return;");
  code.e(
      "  if (q_len_real_rt_ > $) q_len_real_rt_ = $;", q_len_real, q_len_real);
  code.e("  int q_len_padded_rt_ = q_len_real_rt_ + (q_len_real_rt_ & 1);");
  if (single_tile) {
    code.e("  kernel::mla_mtp_tp8::mla_mtp_tp8_main<true, $>(",
           write_final ? "true" : "false");
  } else {
    code.e("  kernel::mla_mtp_tp8::mla_mtp_tp8_main<false, $>(",
           write_final ? "true" : "false");
  }
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]),");
  code.e("      static_cast<const "
         "CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0]),");
  code.e("      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");
  code.e("      static_cast<float*>(task_desc->output_ptrs[1]),");
  {
    float const _mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
    float const _sm = (1.0f / sqrtf(192.0f)) * _mscale * _mscale;
    code.e("      $f,", _sm);
  }
  code.e("      kv_len_,");
  // See TP2 MTP decode: task metadata is laid out for static num_splits.
  code.e("      $,", num_splits);
  code.e("      q_len_padded_rt_,");
  code.e("      $,", qpg);
  code.e("      $,",
         direct_paged_kv ? "runtime_config.paged_kv_indices_buffer"
                         : "nullptr");
  code.e("      fp_,");
  code.e("      q_len_real_rt_,");
  code.e("      task_desc->task_metadata.kv_idx,");
  code.e("      task_desc->task_metadata.request_id);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_TP8_SM100, code.to_string());
}

int TaskRegister::register_mla_mtp_decode_tp8_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int num_groups = params[0];
  int q_len_padded = params[1];
  int num_splits = params[2];
  int rd_dv = params[3];
  int qpg = 2;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Dual-dispatch: pass runtime q_len_padded (even-padded runtime Q_LEN).
  code.e("{");
  code.e("  int bi_ = task_desc->task_metadata.merge_task_offset;");
  code.e("  int req_id_ = runtime_config.request_ids[bi_];");
  code.e("  int fp_ = runtime_config.paged_kv_indptr_buffer[bi_];");
  code.e("  int lp_ = runtime_config.paged_kv_indptr_buffer[bi_ + 1];");
  code.e("  int kv_len_ = (lp_ - fp_ - 1) * MPK_PAGE_SIZE + "
         "runtime_config.paged_kv_last_page_len_buffer[bi_];");
  code.e("  int kvt_rt_ = (kv_len_ + 127) / 128;");
  code.e("  if (kvt_rt_ < 1) kvt_rt_ = 1;");
  code.e("  int sk_rt_ = kvt_rt_ < $ ? kvt_rt_ : $;", num_splits, num_splits);
  code.e("  int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];");
  code.e("  int qo_lp_ = runtime_config.qo_indptr_buffer[bi_ + 1];");
  code.e("  int q_len_real_rt_ = qo_lp_ - qo_fp_;");
  code.e("  if (q_len_real_rt_ < 1) q_len_real_rt_ = 1;");
  code.e("  if (q_len_real_rt_ > 8) return;");
  code.e("  if (q_len_real_rt_ > $) q_len_real_rt_ = $;",
         q_len_padded,
         q_len_padded);
  code.e("  int q_len_padded_rt_ = q_len_real_rt_ + (q_len_real_rt_ & 1);");
  code.e("  kernel::mla_mtp_tp8::mla_mtp_tp8_reduce(");
  code.e("      static_cast<const nv_bfloat16*>(task_desc->input_ptrs[0]),");
  code.e("      static_cast<const float*>(task_desc->input_ptrs[1]),");
  code.e("      static_cast<nv_bfloat16*>(task_desc->output_ptrs[0]),");
  // See TP2 MTP reduce: partial layout is static-num_splits based.
  code.e("      $,", num_splits);
  code.e("      $,", num_groups);
  code.e("      q_len_padded_rt_,");
  code.e("      $,", qpg);
  code.e("      task_desc->task_metadata.kv_idx,");
  code.e("      task_desc->task_metadata.request_id,");
  code.e("      bi_);");
  code.e("}");
  return register_task_variant(TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100,
                               code.to_string());
}

} // namespace runtime
} // namespace mirage
