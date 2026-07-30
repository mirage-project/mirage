#pragma once

#include <vector>

#include "mirage/transpiler/common.h"

namespace mirage {
namespace transpiler {

enum class tb_sched_node_t { OPERATOR, SYNCTHREADS };

// Metadata associated with an OP in a TBSchedNode
struct TBSchedOpMeta {
  // If the op is a TB_FORLOOP_ACCUM_OP...
  // Whether or not to put the forloop accumulator in register files
  bool is_accum_in_reg = false;

  // If the op is a TB_INPUT_OP...
  // Whether or not to use chunked input / software pipelining
  bool is_chunked_input = false;
  int chunked_input_real_innermost_dim = false;
  bool is_pipelined_input = false;

  // If the op is a TB_OUTPUT_OP...
  // Whether or not to use chunked output
  bool is_chunked_output = false;
  int chunked_output_real_innermost_dim = false;
};

// A "node" in the final schedule
// A "schedule" is the execution order and related metadata of all operators
// in a threadblock graph.
class TBSchedNode {
public:
  tb_sched_node_t type;

  // The following fields are only valid if type == OPERATOR
  // We use a vector here since we may perform operator fusion
  // (threadblock-level data reuse)
  std::vector<std::pair<tb::TBOperator const *, TBSchedOpMeta>> ops;
};

// A schedule of a threadblock graph
class TBSched {
public:
  std::vector<TBSchedNode>
      pre_loop_nodes; // Nodes before the for loop, e.g. reading input with
                      // forloop_dim = -1
  std::vector<TBSchedNode> loop_nodes;      // Nodes inside the for loop
  std::vector<TBSchedNode> post_loop_nodes; // Nodes after the for loop
};

} // namespace transpiler
} // namespace mirage
