#pragma once

#include <unordered_map>
#include <unordered_set>

#include "mirage/search/config.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/order.h"
#include "mirage/utils/json_utils.h"

namespace mirage {
namespace search {

template <typename TensorType>
struct SearchContext {
  std::vector<TensorType> all_tensors;
  std::vector<std::shared_ptr<AlgebraicPattern>> algebraic_pattern;
  std::vector<int> num_consumers;
  std::vector<Order> op_order;
  std::vector<std::shared_ptr<AlgebraicPattern>> output_pattern;
};

struct LayerCheckpoint {
  std::unordered_set<type::KNOperatorType> knop_explored;
  std::unordered_set<type::TBOperatorType> tbop_explored;
  std::unordered_set<std::vector<int>> input_idx_explored;
  std::unordered_set<dim3> grid_dim_explored;
  std::unordered_set<dim3> block_dim_explored;
  std::unordered_set<std::vector<int3>> input_map_explored;
  std::unordered_set<std::vector<int>> forloop_dim_explored;
  std::unordered_set<int> forloop_range_explored;
  std::unordered_set<int3> output_map_explored;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LayerCheckpoint,
                                   knop_explored,
                                   tbop_explored,
                                   input_idx_explored,
                                   grid_dim_explored,
                                   block_dim_explored,
                                   input_map_explored,
                                   forloop_dim_explored,
                                   forloop_range_explored,
                                   output_map_explored);

struct Checkpoint {
  kernel::Graph computation_graph;
  json best_graph;
  ProfileResult best_profile_result;
  GeneratorConfig config;
  std::vector<LayerCheckpoint> callstack;
  std::vector<json> generated_graphs;

  int num_total_kernel_graphs;
  int num_total_random_tests;
  int num_valid_kernel_graphs;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Checkpoint,
                                   computation_graph,
                                   best_graph,
                                   best_profile_result,
                                   config,
                                   callstack,
                                   generated_graphs,
                                   num_total_kernel_graphs,
                                   num_total_random_tests,
                                   num_valid_kernel_graphs);

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config,
                       char const *filename);
  KernelGraphGenerator(char const *filename);

  void generate_kernel_graphs();
  void optimize_layout(kernel::Graph &g);
  void save_checkpoint() const;

  kernel::Graph computation_graph;
  json best_graph;
  ProfileResult best_profile_result;

  GeneratorConfig config;
  DimStrategy dim_strategy;

  char const *filename;
  std::vector<json> generated_graphs;

private:
  std::vector<std::shared_ptr<AlgebraicPattern>> final_patterns;
  std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>>
      computation_graph_patterns;

  std::vector<DTensor> output_tensors;

  int num_total_kernel_graphs;
  int num_total_random_tests;
  int num_valid_kernel_graphs;

  std::vector<LayerCheckpoint> callstack;

  void generate_next_tb_operator(
      SearchContext<STensor> &c,
      threadblock::Graph &g,
      std::function<void(int)> const &create_customized_then_next_kn,
      int depth);
  void generate_next_kn_operator(SearchContext<DTensor> &c,
                                 kernel::Graph &g,
                                 int depth);
  void optimize_layout(
      kernel::Graph &g, int op_idx, int ts_idx, int bop_idx, int bts_idx);
  void update_best_graph(kernel::Graph &g);
  bool create_tb_outputs(SearchContext<STensor> &c,
                         threadblock::Graph &g,
                         int3 output_map);

  std::vector<layout::SmemLayout>
      get_valid_output_layout(threadblock::TBOperator const *op, int idx);

  void process_outputs();
  bool check_pattern(std::shared_ptr<AlgebraicPattern> pattern);
  void pattern_eval();
  void fingerprint_eval();
  bool have_same_fingerprint(std::vector<DTensor> const &outputs,
                             std::vector<int> const &match) const;
  bool verify(SearchContext<DTensor> &c, kernel::Graph const &g);
  void recovery_from_checkpoint(Checkpoint const &checkpoint);
};

} // namespace search
} // namespace mirage
