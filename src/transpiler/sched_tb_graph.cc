#include "mirage/transpiler/sched_tb_graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/transpiler.h"
#include "mirage/transpiler/utils.h"
#include "mirage/type.h"
#include <algorithm>

namespace mirage {
namespace transpiler {

struct OpMeta {
  int level;
  int fuse_chain_idx; // For grouping fused operators together when sorting
  int pos_in_fuse_chain;
};

// A helper. See code below
std::vector<TBSchedNode>
    ops2sched(vector<std::pair<tb::TBOperator const *, OpMeta>> &ops,
              std::function<bool(tb::TBOperator const *)> const &filter) {
  std::sort(ops.begin(), ops.end(), [&](auto const &a, auto const &b) {
    const OpMeta &meta_a = a.second;
    const OpMeta &meta_b = b.second;
    return meta_a.level != meta_b.level ? meta_a.level < meta_b.level
           : meta_a.fuse_chain_idx != meta_b.fuse_chain_idx
               ? meta_a.fuse_chain_idx < meta_b.fuse_chain_idx
               : meta_a.pos_in_fuse_chain < meta_b.pos_in_fuse_chain;
  });
  std::vector<TBSchedNode> res;
  int last_level = -1;
  int num_ops = ops.size();
  for (int cur_op_idx = 0; cur_op_idx < num_ops;) {
    auto const [op, meta] = ops[cur_op_idx];
    if (!filter(op)) {
      // Skip this operator
      assert(cur_op_idx + 1 == num_ops ||
             ops[cur_op_idx + 1].second.fuse_chain_idx != meta.fuse_chain_idx);
      cur_op_idx++;
      continue;
    }
    if (last_level != -1 && meta.level != last_level) {
      res.push_back({tb_sched_node_t::SYNCTHREADS, {}});
    }
    last_level = meta.level;
    int nxt_op_idx = cur_op_idx + 1;
    while (nxt_op_idx < num_ops &&
           ops[nxt_op_idx].second.fuse_chain_idx == meta.fuse_chain_idx) {
      nxt_op_idx++;
    }
    std::vector<tb::TBOperator const *> fused_ops;
    for (int i = cur_op_idx; i < nxt_op_idx; i++) {
      fused_ops.push_back(ops[i].first);
    }
    res.push_back({tb_sched_node_t::OPERATOR, fused_ops});
    cur_op_idx = nxt_op_idx;
  }
  return res;
}

// See docs/transpiler/transpiler.md for more details
TBSched Transpiler::get_threadblock_schedule(tb::Graph const &tb_graph) {
  // Currently in Mirage, the output tensor of a threadblock level input op must
  // have after_accum = False. We first check against this condition
  for (tb::TBOperator *const op : tb_graph.operators) {
    if (op->op_type == type::TB_INPUT_OP) {
      assert(op->output_tensors[0].after_accum == false);
    }
  }

  TBSched sched;

  // Generate `pre_loop_nodes`
  // Currently it only contains input ops with forloop_dim = -1
  for (tb::TBOperator *const op : tb_graph.operators) {
    if (op->op_type == type::TB_INPUT_OP) {
      tb::TBInputOp *input_op = dynamic_cast<tb::TBInputOp *>(op);
      if (input_op->forloop_dim == -1) {
        assert(is_fused_with_next[op] == false);
        sched.pre_loop_nodes.push_back({tb_sched_node_t::OPERATOR, {op}});
      }
    }
  }

  // Generate `loop_nodes`
  {
    int next_fuse_chain_idx = 0;
    std::unordered_map<tb::TBOperator const *, OpMeta> op2meta;
    // Calculate the level of each operator
    for (tb::TBOperator *const op : tb_graph.operators) {
      if (op->op_type == type::TB_INPUT_OP) {
        op2meta[op] = {0, next_fuse_chain_idx++, 0};
      }
    }
    for (tb::TBOperator *const op : tb_graph.operators) {
      if (op->op_type == type::TB_INPUT_OP) {
        continue;
      }
      if (!op->input_tensors.at(0).after_accum) {
        if (is_fused_with_prev[op]) {
          // If this operator is fused with the previous operator
          tb::TBOperator *prev_op = op->input_tensors.at(0).owner_op;
          assert(op2meta.count(prev_op));
          op2meta[op].level = op2meta[prev_op].level;
          op2meta[op].fuse_chain_idx = op2meta[prev_op].fuse_chain_idx;
          op2meta[op].pos_in_fuse_chain =
              op2meta[prev_op].pos_in_fuse_chain + 1;
        } else {
          int res = 0;
          for (tb::STensor const &input : op->input_tensors) {
            tb::TBOperator *input_op = input.owner_op;
            assert(op2meta.count(input_op));
            res = std::max(res, op2meta[input_op].level);
          }
          op2meta[op] = {res + 1, next_fuse_chain_idx++, 0};
        }
      }
    }
    vector<std::pair<tb::TBOperator const *, OpMeta>> all_ops(op2meta.begin(),
                                                              op2meta.end());
    sched.loop_nodes = ops2sched(all_ops, [&](tb::TBOperator const *op) {
      if (op->op_type != type::TB_INPUT_OP) {
        return true;
      } else {
        tb::TBInputOp const *input_op = dynamic_cast<tb::TBInputOp const *>(op);
        return input_op->forloop_dim != -1;
      }
    });
  }

  {
    int next_fuse_chain_idx = 0;
    std::unordered_map<tb::TBOperator const *, OpMeta> op2meta;
    // Calculate the level of each operator
    for (tb::TBOperator *const op : tb_graph.operators) {
      if (op->op_type == type::TB_INPUT_OP ||
          op->op_type == type::TB_FORLOOP_ACCUM_OP) {
        op2meta[op] = {0, next_fuse_chain_idx++, 0};
      }
    }
    for (tb::TBOperator *const op : tb_graph.operators) {
      if (op->op_type == type::TB_INPUT_OP) {
        continue;
      }
      if (op->input_tensors.at(0).after_accum) {
        if (is_fused_with_prev[op]) {
          // If this operator is fused with the previous operator
          tb::TBOperator *prev_op = op->input_tensors.at(0).owner_op;
          assert(op2meta.count(prev_op));
          op2meta[op].level = op2meta[prev_op].level;
          op2meta[op].fuse_chain_idx = op2meta[prev_op].fuse_chain_idx;
          op2meta[op].pos_in_fuse_chain =
              op2meta[prev_op].pos_in_fuse_chain + 1;
        } else {
          int res = 0;
          for (tb::STensor const &input : op->input_tensors) {
            tb::TBOperator *input_op = input.owner_op;
            assert(op2meta.count(input_op));
            res = std::max(res, op2meta[input_op].level);
          }
          op2meta[op] = {res + 1, next_fuse_chain_idx++, 0};
        }
      }
    }
    vector<std::pair<tb::TBOperator const *, OpMeta>> all_ops(op2meta.begin(),
                                                              op2meta.end());
    sched.post_loop_nodes = ops2sched(all_ops, [&](tb::TBOperator const *op) {
      if (op->op_type == type::TB_INPUT_OP ||
          op->op_type == type::TB_FORLOOP_ACCUM_OP) {
        return false;
      } else {
        return true;
      }
    });
  }

  // Some sanity checks
  auto count_num_operators = [](std::vector<TBSchedNode> const &nodes) {
    size_t res = 0;
    for (const TBSchedNode &node : nodes) {
      if (node.type == tb_sched_node_t::OPERATOR) {
        res += node.ops.size();
      }
    }
    return res;
  };
  assert(count_num_operators(sched.pre_loop_nodes) +
             count_num_operators(sched.loop_nodes) +
             count_num_operators(sched.post_loop_nodes) ==
         tb_graph.operators.size());

  return sched;
}

} // namespace transpiler
} // namespace mirage
