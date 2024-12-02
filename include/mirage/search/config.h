#pragma once

#include <vector>
#include <vector_types.h>

#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/type.h"
#include "mirage/utils/hash_utils.h"

namespace mirage {
namespace search {

using kernel::DTensor;
using kernel::KNOperator;
using threadblock::STensor;
using threadblock::TBOperator;

enum class VerifierType {
  PROBABILISTIC_VERIFIER,
  FORMAL_VERIFIER,
};

struct GeneratorConfig {
  size_t max_num_threadblock_graph_op;
  size_t max_num_kernel_graph_op;
  size_t max_num_threadblock_graphs;
  size_t max_num_threadblock_graph_inputs;
  size_t max_num_threadblock_graph_outputs;
  size_t search_thread;

  VerifierType verifier_type;

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
  bool
      randomized_branches; // Only for developers to tune the search performance
  bool _enable_attention_specific_optimization;
  bool _enable_concat_matmul_transformation;

  void show() const;
  void enable_attention_specific_optimization();
  void enable_concat_matmul_transformation();

  static GeneratorConfig get_default_config();
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
                                   _enable_attention_specific_optimization,
                                   _enable_concat_matmul_transformation);

struct TBGraphConfig {
  dim3 grid_dim, block_dim;
  std::vector<int3> imaps;
  std::vector<int> fmaps;
  int frange;

  bool operator==(TBGraphConfig const &other) const;
  void show() const;
};

} // namespace search
} // namespace mirage

namespace std {

template <>
struct hash<mirage::search::TBGraphConfig> {
  size_t operator()(mirage::search::TBGraphConfig const &config) const;
};

} // namespace std