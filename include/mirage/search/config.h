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

int const MAX_NUM_THREADBLOCK_GRAPH_OP = 7; // Outputs not counted
int const MAX_NUM_KERNEL_GRAPH_OP = 5;
int const MAX_NUM_THREADBLOCK = 2;
int const MAX_NUM_THREADBLOCK_INPUT = 3;
int const MAX_NUM_THREADBLOCK_OUTPUT = 2;

struct GeneratorConfig {
  std::vector<type::KNOperatorType> knop_to_explore;
  std::vector<type::TBOperatorType> tbop_to_explore;
  std::vector<int3> imap_to_explore;
  std::vector<std::vector<int3>> imap_comb_to_explore;
  std::vector<int3> omap_to_explore;
  std::vector<dim3> grid_dim_to_explore;
  std::vector<dim3> block_dim_to_explore;
  std::vector<int> fmap_to_explore;
  std::vector<int> frange_to_explore;
  std::vector<layout::SmemLayout> smem_layout_to_explore;
  int reduction_dimx;

  void print_config() const;

  static GeneratorConfig get_default_config();
  static GeneratorConfig get_attention_default_config();
  static GeneratorConfig get_mlp_default_config();
  static GeneratorConfig get_lora_default_config();
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneratorConfig,
                                   knop_to_explore,
                                   tbop_to_explore,
                                   imap_to_explore,
                                   imap_comb_to_explore,
                                   omap_to_explore,
                                   grid_dim_to_explore,
                                   block_dim_to_explore,
                                   fmap_to_explore,
                                   frange_to_explore,
                                   smem_layout_to_explore,
                                   reduction_dimx)

} // namespace search
} // namespace mirage
