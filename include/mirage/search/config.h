#pragma once

#include <vector>
#include <vector_types.h>

#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/type.h"

namespace mirage {
namespace search {

using kernel::DTensor;
using kernel::KNOperator;
using threadblock::STensor;
using threadblock::TBOperator;

// int const MAX_NUM_THREADBLOCK_GRAPH_OP = 9; // Outputs not counted
// int const MAX_NUM_KERNEL_GRAPH_OP = 5;
// int const MAX_NUM_THREADBLOCK = 2;
// int const MAX_NUM_THREADBLOCK_INPUT = 3;
// int const MAX_NUM_THREADBLOCK_OUTPUT = 2;
// int const MAX_SEARCH_THREAD = 8;

struct GeneratorConfig {
  int max_num_threadblock_graph_op;
  int max_num_kernel_graph_op;
  int max_num_threadblock_graphs;
  int max_num_threadblock_graph_inputs;
  int max_num_threadblock_graph_outputs;
  int search_thread;

  std::vector<type::KNOperatorType> knop_to_explore;
  std::vector<type::TBOperatorType> tbop_to_explore;
  std::vector<int3> imap_to_explore;
  std::vector<std::vector<int3>> imap_comb_to_explore;
  std::vector<int3> omap_to_explore;
  std::vector<dim3> grid_dim_to_explore;
  std::vector<dim3> block_dim_to_explore;
  std::vector<int> fmap_to_explore;
  std::vector<int> frange_to_explore;
  int reduction_dimx;
  bool enable_attention_specific_optimization;
  bool enable_concat_matmul_transformation;

  void show() const;

  static GeneratorConfig get_default_config();
  // TODO: Remove the following configs and use the heusristic in DimStrategy
  static GeneratorConfig get_attention_default_config();
  static GeneratorConfig get_mlp_default_config();
  static GeneratorConfig get_lora_default_config();
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneratorConfig,
                                   max_num_threadblock_graph_op,
                                   max_num_kernel_graph_op,
                                   max_num_threadblock_graphs,
                                   max_num_threadblock_graph_inputs,
                                   max_num_threadblock_graph_outputs,
                                   search_thread,
                                   knop_to_explore,
                                   tbop_to_explore,
                                   imap_to_explore,
                                   imap_comb_to_explore,
                                   omap_to_explore,
                                   grid_dim_to_explore,
                                   block_dim_to_explore,
                                   fmap_to_explore,
                                   frange_to_explore,
                                   reduction_dimx,
                                   enable_attention_specific_optimization,
                                   enable_concat_matmul_transformation);

} // namespace search
} // namespace mirage
