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

  // Metadata for generating the final TBSchedNodeMeta
  bool is_accum_in_reg;
};

// Given all OpMetas for OPs on a fusion chain, generate its TBSchedNodeMeta
static TBSchedNodeMeta get_tb_sched_node_meta(
  vector<std::pair<tb::TBOperator const *, OpMeta>> &ops_on_chain
) {
  // Only place the accumulator fragment in registers when
  // 1. The last operator is a FORLOOP_ACCUM_NO_RED_OP and it is in registers
  // 2. The first operator is a MATMUL_OP and there is no other operator (e.g. exp)
  // In the future may add more rules
  bool is_accum_in_reg = ops_on_chain.back().first->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP &&
                          ops_on_chain.back().second.is_accum_in_reg && 
                          ops_on_chain.at(0).first->op_type == type::TB_MATMUL_OP &&
                          ops_on_chain.size() == 2;
  return {
    is_accum_in_reg
  };
}

// A helper function for generating TBSched
// First, the caller should prepare a list of `OpMetas`. This function will sort
// them to fusion chains, generate TBSchedNoteMetas, and finally generate the
// TBSchedNodes.
static std::vector<TBSchedNode>
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
    auto cur_fusion_chain = vector<std::pair<tb::TBOperator const *, OpMeta>>(
        ops.begin() + cur_op_idx, ops.begin() + nxt_op_idx);
    res.push_back({tb_sched_node_t::OPERATOR, fused_ops, get_tb_sched_node_meta(cur_fusion_chain)});
    cur_op_idx = nxt_op_idx;
  }
  return res;
}

// Get the schedule of a custom threadblock graph
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
    size_t per_thread_accum_numel_tot = 0;
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
      if (op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
        // Decide whether or not to put the forloop accumulator in register files
        size_t accum_numel = op->output_tensors.at(0).num_elements();
        size_t num_thrs = tb_graph.block_dim.x * tb_graph.block_dim.y *
                          tb_graph.block_dim.z;
        size_t per_thr_accum_numel = accum_numel / num_thrs;
        // Use a simple heuristic to decide whether or not to put the forloop
        // accumulator in register files
        // Every thread can have at most 255 32-bit registers (according to
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
        // So we allow accumulators to take up to 192 registers
        if (per_thread_accum_numel_tot + per_thr_accum_numel <= 192) {
          op2meta[op].is_accum_in_reg = true;
          per_thread_accum_numel_tot += per_thr_accum_numel;
        } else {
          op2meta[op].is_accum_in_reg = false;
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
          op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
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
          op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
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
