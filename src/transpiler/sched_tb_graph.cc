#include "mirage/transpiler/sched_tb_graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpiler.h"
#include "mirage/transpiler/utils.h"
#include "mirage/type.h"
#include <algorithm>

namespace mirage {
namespace transpiler {

using std::vector, std::pair;

// Test whether consecutive `chunk_size` elements in layout A are contiguous in
// layout B
//
// See docs/transpiler/transpiler.md for more details
static pair<bool, int>
    can_perform_chunked_copy(tb::STensor const &stensor,
                             STensorMeta const &stensor_meta,
                             kn::DTensor const &dtensor,
                             DTensorMeta const &dtensor_meta) {
  // Check whether all strides of the DTensor are 16B-aligned
  {
    size_t alignment = 16 / type::get_datatype_size(stensor.data_type);
    bool res = true;
    for (int i = 0; i < dtensor.num_dims; ++i) {
      size_t stride = dtensor_meta.strides[i];
      res &= (stride % alignment == 0 || stride == 1);
    }
    if (!res) {
      return {false, 0};
    }
  };

  // Check whether the "real innermost dim" is the same
  auto find_real_innermost_dim =
      [&](int num_dims, int const shape[], size_t const strides[]) -> int {
    for (int i = 0; i < num_dims; ++i) {
      if (strides[i] == 1 && shape[i] != 1) {
        return i;
      }
    }
    // In the case where all dimensions are of size 1
    // return the last dim as the inner most dim
    return num_dims - 1;
  };
  int real_innermost_dtensor = find_real_innermost_dim(
      dtensor.num_dims, dtensor.dim, dtensor_meta.strides);
  int real_innermost_stensor = find_real_innermost_dim(
      stensor.num_dims, stensor.dim, stensor_meta.strides);
  // assert(real_innermost_dtensor != -1);  real_innermost_dtensor can be -1 for
  // input tensors
  assert(real_innermost_stensor != -1);
  return {real_innermost_dtensor == real_innermost_stensor,
          real_innermost_stensor};
}

struct OpChainingMeta {
  int level;
  int fuse_chain_idx; // For grouping fused operators together when sorting
  int pos_in_fuse_chain;
};

struct ChainPiece {
  tb::TBOperator const *op;
  OpChainingMeta chaining_meta;
  TBSchedOpMeta op_meta;
};

// Refine op metadata on a fusion chain
// Sometimes we may want to change opmeta based on the whole fusion chain, for
// example, we may only want to put the accumulator in register files if the
// leading operator of the chain is a matmul. This process is called "refining
// opmeta on chain"
static void refine_opmeta_on_chain(
    vector<pair<tb::TBOperator const *, TBSchedOpMeta>> &chain) {
  // Only place the accumulator fragment in registers when
  // 1. The last operator is a FORLOOP_ACCUM_NO_RED_OP and it is in registers
  // 2. The first operator is a MATMUL_OP and there is no other operator (e.g.
  // exp) In the future may add more rules here
  chain.back().second.is_accum_in_reg &=
      chain.front().first->op_type == type::TB_MATMUL_OP && chain.size() == 2;
}

// A helper function for generating TBSched
// First, the caller should prepare a list of `ChainPiece`s. This function will
// sort them to fusion chains, and generate a list of TBSchedNodes. It also
// inserts `syncthreads` when necessary
static vector<TBSchedNode>
    ops2sched(vector<ChainPiece> &ops,
              std::function<bool(tb::TBOperator const *)> const &filter,
              bool is_in_loop) {
  std::sort(
      ops.begin(), ops.end(), [&](ChainPiece const &a, ChainPiece const &b) {
        OpChainingMeta const &meta_a = a.chaining_meta;
        OpChainingMeta const &meta_b = b.chaining_meta;
        return meta_a.level != meta_b.level ? meta_a.level < meta_b.level
               : meta_a.fuse_chain_idx != meta_b.fuse_chain_idx
                   ? meta_a.fuse_chain_idx < meta_b.fuse_chain_idx
                   : meta_a.pos_in_fuse_chain < meta_b.pos_in_fuse_chain;
      });
  vector<TBSchedNode> res;
  int last_level = -1;
  int num_ops = ops.size();
  for (int cur_op_idx = 0; cur_op_idx < num_ops;) {
    auto const [op, meta, _] = ops[cur_op_idx];
    if (!filter(op)) {
      // Skip this operator
      assert(cur_op_idx + 1 == num_ops ||
             ops[cur_op_idx + 1].chaining_meta.fuse_chain_idx !=
                 meta.fuse_chain_idx);
      cur_op_idx++;
      continue;
    }
    // Update `last_level` and insert `syncthreads` if necessary
    if (last_level != -1 && meta.level != last_level) {
      res.push_back({tb_sched_node_t::SYNCTHREADS, {}});
    }
    last_level = meta.level;
    // Get the current fusion chain
    int nxt_op_idx = cur_op_idx + 1;
    while (nxt_op_idx < num_ops &&
           ops[nxt_op_idx].chaining_meta.fuse_chain_idx ==
               meta.fuse_chain_idx) {
      nxt_op_idx++;
    }
    vector<pair<tb::TBOperator const *, TBSchedOpMeta>> cur_fusion_chain;
    for (int i = cur_op_idx; i < nxt_op_idx; i++) {
      cur_fusion_chain.push_back({ops[i].op, ops[i].op_meta});
    }
    cur_op_idx = nxt_op_idx;
    // Refine opmeta on the chain
    refine_opmeta_on_chain(cur_fusion_chain);
    // Save the chain
    res.push_back({tb_sched_node_t::OPERATOR, cur_fusion_chain});
  }
  return res;
}

// Get the schedule of a custom threadblock graph
// See docs/transpiler/transpiler.md for more details
TBSched Transpiler::get_threadblock_schedule(tb::Graph const &tb_graph) {
  TBSched sched;

  // Get TBSchedOpMeta for all ops individually
  // In this stage, we do not consider the interference on different
  std::unordered_map<tb::TBOperator const *, TBSchedOpMeta> op2op_meta;
  {
    size_t per_thread_accum_numel_tot = 0;
    for (tb::TBOperator *const op : tb_graph.operators) {
      TBSchedOpMeta op_meta;
      if (op->op_type == type::TB_INPUT_OP) {
        // Decide whether or not to use chunked copy and async copy
        assert(is_fused_with_prev[op] == false);
        assert(is_fused_with_next[op] == false);
        tb::TBInputOp const *input_op = dynamic_cast<tb::TBInputOp const *>(op);
        tb::STensor stensor = input_op->output_tensors.at(0);
        kn::DTensor dtensor = input_op->dtensor;
        STensorMeta &stensor_meta = stensor_metas.at(stensor.guid);
        DTensorMeta dtensor_meta = dtensor_metas.at(dtensor.guid);
        int3 imap = input_op->input_map;
        size_t alignment = get_num_elems_in_16B(dtensor.data_type);

        bool is_dtensor_offset_divisible = true;
        for (int dim = 0; dim < 3; ++dim) {
          int div_dim = dim == 0 ? imap.x : dim == 1 ? imap.y : imap.z;
          if (div_dim >= 0) {
            // Dim `div_dim` is divided along `dim`
            int num_tbs = dim == 0   ? tb_graph.grid_dim.x
                          : dim == 1 ? tb_graph.grid_dim.y
                                     : tb_graph.grid_dim.z;
            // Can refer to the offset calculation in `transpiler_tb.cc` for the
            // condition below
            is_dtensor_offset_divisible &=
                num_tbs == 1 || (dtensor.dim[div_dim] / num_tbs *
                                 dtensor_meta.strides[div_dim]) %
                                        alignment ==
                                    0;
          }
        }
        if (input_op->forloop_dim != -1) {
          int forloop_dim = input_op->forloop_dim;
          int forloop_range = dtensor.dim[forloop_dim];
          size_t forloop_dim_stride = dtensor_meta.strides[forloop_dim];
          int tile_side_len = stensor.dim[forloop_dim];
          is_dtensor_offset_divisible &=
              forloop_range == 1 ||
              (tile_side_len * forloop_dim_stride) % alignment == 0;
        }

        std::tie(op_meta.is_chunked_input,
                 op_meta.chunked_input_real_innermost_dim) =
            can_perform_chunked_copy(
                stensor, stensor_meta, dtensor, dtensor_meta);
        op_meta.is_chunked_input &= is_dtensor_offset_divisible;
        op_meta.is_pipelined_input = op_meta.is_chunked_input &&
                                     input_op->forloop_dim != -1 &&
                                     config.target_cc >= GPU_CC::A100;
        stensor_meta.is_pipelined_input = op_meta.is_pipelined_input;
      } else if (op->op_type == type::TB_OUTPUT_OP) {
        // Decide whether or not to use chunked copy
        assert(is_fused_with_prev[op] == false);
        assert(is_fused_with_next[op] == false);
        tb::TBOutputOp const *output_op =
            dynamic_cast<tb::TBOutputOp const *>(op);
        tb::STensor stensor = output_op->input_tensors.at(0);
        kn::DTensor dtensor = output_op->dtensor;
        STensorMeta stensor_meta = stensor_metas.at(stensor.guid);
        DTensorMeta dtensor_meta = dtensor_metas.at(dtensor.guid);
        int3 omap = output_op->output_map;
        size_t alignment = get_num_elems_in_16B(dtensor.data_type);

        bool is_dtensor_offset_divisible = true;
        for (int dim = 0; dim < 3; ++dim) {
          int div_dim = dim == 0 ? omap.x : dim == 1 ? omap.y : omap.z;
          int num_tbs = dim == 0   ? tb_graph.grid_dim.x
                        : dim == 1 ? tb_graph.grid_dim.y
                                   : tb_graph.grid_dim.z;
          if (num_tbs > 1) {
            // The output tensor MUST be divided along this dimension, as stated
            // in the paper
            assert(div_dim >= 0);
            is_dtensor_offset_divisible &= (dtensor.dim[div_dim] / num_tbs *
                                            dtensor_meta.strides[div_dim]) %
                                               alignment ==
                                           0;
          }
        }
        std::tie(op_meta.is_chunked_output,
                 op_meta.chunked_output_real_innermost_dim) =
            can_perform_chunked_copy(
                stensor, stensor_meta, dtensor, dtensor_meta);
        op_meta.is_chunked_output &= is_dtensor_offset_divisible;
      } else if (op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
        // Decide whether or not to put the forloop accumulator in register
        // files
        size_t accum_numel = op->output_tensors.at(0).num_elements();
        size_t num_thrs =
            tb_graph.block_dim.x * tb_graph.block_dim.y * tb_graph.block_dim.z;
        size_t per_thr_accum_numel = accum_numel / num_thrs;
        // Use a simple heuristic to decide whether or not to put the forloop
        // accumulator in register files
        // Every thread can have at most 255 32-bit registers (according to
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
        // So we allow accumulators to take up to 192 registers
        if (per_thread_accum_numel_tot + per_thr_accum_numel <= 192) {
          op_meta.is_accum_in_reg = true;
          per_thread_accum_numel_tot += per_thr_accum_numel;
        } else {
          op_meta.is_accum_in_reg = false;
        }
      } else if (op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
                 op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP) {
        op_meta.is_accum_in_reg = false;
      }
      op2op_meta[op] = op_meta;
    }
  }

  // Generate `pre_loop_nodes`
  // Currently it only contains input ops with forloop_dim = -1
  // TODO(intlsy) If the output of an input operator is never used in the for
  // loop (only used after the for loop), we can safely move it to
  // post_forloop_nodes
  int next_chain_idx = 0;
  {
    vector<ChainPiece> pieces;
    for (tb::TBOperator *const op : tb_graph.operators) {
      tb::TBInputOp *input_op = dynamic_cast<tb::TBInputOp *>(op);
      if (op->op_type == type::TB_INPUT_OP && input_op->forloop_dim == -1) {
        assert(is_fused_with_next[op] == false);
        pieces.push_back({op, {0, next_chain_idx++, 0}, op2op_meta.at(op)});
      }
    }
    sched.pre_loop_nodes = ops2sched(
        pieces, [](tb::TBOperator const *op) { return true; }, false);
  }

  // Generate `loop_nodes`
  {
    std::unordered_map<tb::TBOperator const *, OpChainingMeta> op2chaining_meta;
    // Calculate the level of each operator
    for (tb::TBOperator *const op : tb_graph.operators) {
      if (op->op_type == type::TB_INPUT_OP) {
        op2chaining_meta[op] = {0, next_chain_idx++, 0};
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
          OpChainingMeta prev_chaining_meta = op2chaining_meta.at(prev_op);
          op2chaining_meta[op] = {prev_chaining_meta.level,
                                  prev_chaining_meta.fuse_chain_idx,
                                  prev_chaining_meta.pos_in_fuse_chain + 1};
        } else {
          int res = 0;
          for (tb::STensor const &input : op->input_tensors) {
            tb::TBOperator *input_op = input.owner_op;
            res = std::max(res, op2chaining_meta.at(input_op).level);
          }
          op2chaining_meta[op] = {res + 1, next_chain_idx++, 0};
        }
      }
    }
    vector<ChainPiece> chain_pieces;
    for (auto const &[op, chaining_meta] : op2chaining_meta) {
      chain_pieces.push_back({op, chaining_meta, op2op_meta.at(op)});
    }
    sched.loop_nodes = ops2sched(
        chain_pieces,
        [&](tb::TBOperator const *op) {
          if (op->op_type != type::TB_INPUT_OP) {
            return true;
          } else {
            tb::TBInputOp const *input_op =
                dynamic_cast<tb::TBInputOp const *>(op);
            return input_op->forloop_dim != -1;
          }
        },
        true);
  }

  {
    std::unordered_map<tb::TBOperator const *, OpChainingMeta> op2chaining_meta;
    // Calculate the level of each operator
    for (tb::TBOperator *const op : tb_graph.operators) {
      if (op->op_type == type::TB_INPUT_OP ||
          op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
          op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
          op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP) {
        op2chaining_meta[op] = {0, next_chain_idx++, 0};
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
          OpChainingMeta prev_chaining_meta = op2chaining_meta.at(prev_op);
          op2chaining_meta[op] = {prev_chaining_meta.level,
                                  prev_chaining_meta.fuse_chain_idx,
                                  prev_chaining_meta.pos_in_fuse_chain + 1};
        } else {
          int res = 0;
          for (tb::STensor const &input : op->input_tensors) {
            tb::TBOperator *input_op = input.owner_op;
            assert(op2chaining_meta.find(input_op) != op2chaining_meta.end());
            res = std::max(res, op2chaining_meta.at(input_op).level);
          }
          op2chaining_meta[op] = {res + 1, next_chain_idx++, 0};
        }
      }
    }
    vector<ChainPiece> chain_pieces;
    for (auto const &[op, chaining_meta] : op2chaining_meta) {
      chain_pieces.push_back({op, chaining_meta, op2op_meta.at(op)});
    }
    sched.post_loop_nodes = ops2sched(
        chain_pieces,
        [&](tb::TBOperator const *op) {
          if (op->op_type == type::TB_INPUT_OP ||
              op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
              op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
              op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP) {
            return false;
          } else {
            return true;
          }
        },
        false);
  }

  // Some sanity checks
  auto count_num_operators = [](std::vector<TBSchedNode> const &nodes) {
    size_t res = 0;
    for (TBSchedNode const &node : nodes) {
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
