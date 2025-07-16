#include "mirage/transpiler/transpiler.h"

#include "mirage/transpiler/utils.h"
#include "mirage/type.h"

namespace mirage {
namespace transpiler {

// Decide when and how to fuse operators in every threadblock level op
void Transpiler::resolve_tb_fusion() {
  for (kn::KNOperator *const op : g->operators) {
    if (op->op_type != type::KN_CUSTOMIZED_OP) {
      continue;
    }
    kn::KNCustomizedOp *const custom_op =
        dynamic_cast<kn::KNCustomizedOp *>(op);
    tb::Graph const &tb_graph = custom_op->bgraph;

    std::unordered_map<sguid_t, int> num_consumers;
    // Initialization
    for (tb::TBOperator *const op : tb_graph.operators) {
      is_fused_with_prev[op] = false;
      is_fused_with_next[op] = false;
      for (tb::STensor const &stensor :
           Combine(op->input_tensors, op->output_tensors)) {
        num_consumers[stensor.guid] = 0;
      }
    }
    // Count the number of consumers for each tensor
    for (tb::TBOperator *const op : tb_graph.operators) {
      for (tb::STensor const &input : op->input_tensors) {
        num_consumers[input.guid] += 1;
      }
    }
    // Currently we only fuse elementwise unary operators and
    // forloop_accum_no_red with
    // the previous operator, when the output of the previous operator has only
    // one consumer
    for (tb::TBOperator *const op : tb_graph.operators) {
      if ((type::is_threadblock_element_unary(op->op_type)) ||
          (op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP)) {
        tb::STensor const &input0 = op->input_tensors.at(0);
        tb::TBOperator *prev_op = input0.owner_op;
        // Don't fuse with an input op with forloop_dim = -1
        if (prev_op->op_type == type::TB_INPUT_OP &&
            dynamic_cast<tb::TBInputOp *>(prev_op)->forloop_dim == -1) {
          continue;
        }
        // Do not fuse with
        // FORLOOP_ACCUM_NO_RED_OP,FORLOOP_ACCUM_NO_RED_RESCALE_OP, or
        // FORLOOP_ACCUM_MAX_OP since the accum is performed within the loop
        // body while the current operator is outside the loop Do not fuse with
        // input op since we may perform chunked or async input ops Do not fuse
        // with reduction max ops since they have two outputs
        if (num_consumers.at(input0.guid) == 1 &&
            prev_op->op_type != type::TB_FORLOOP_ACCUM_NO_RED_OP &&
            prev_op->op_type != type::TB_INPUT_OP &&
            prev_op->op_type != type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP &&
            prev_op->op_type != type::TB_FORLOOP_ACCUM_MAX_OP &&
            !(prev_op->op_type >= type::TB_REDUCTION_0_MAX_OP &&
              prev_op->op_type <= type::TB_REDUCTION_2_MAX_OP)) {
          is_fused_with_prev[op] = true;
          is_fused_with_next[prev_op] = true;
        }
      }
    }
    // Construct `fusion_chain`
    for (tb::TBOperator *const last_op : tb_graph.operators) {
      if (is_fused_with_next[last_op]) {
        continue;
      }
      // Now op is the tail of a fusion chain
      std::vector<tb::TBOperator const *> fused_ops;
      tb::TBOperator *cur_op = last_op;
      while (true) {
        fused_ops.push_back(cur_op);
        if (is_fused_with_prev[cur_op]) {
          cur_op = cur_op->input_tensors.at(0).owner_op;
        } else {
          break;
        }
      }
      tb::TBOperator *const leading_op = cur_op;
      std::reverse(fused_ops.begin(), fused_ops.end());
      fusion_chain[leading_op] = fused_ops;
      for (tb::TBOperator const *op : fused_ops) {
        chain_leading_op[op] = leading_op;
      }
    }
  }
}

} // namespace transpiler
} // namespace mirage
