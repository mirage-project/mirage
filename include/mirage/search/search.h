#pragma once

#include <atomic>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "mirage/search/config.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/order.h"
#include "mirage/search/range_propagation/irange.h"
#include "mirage/search/search_context.h"
#include "mirage/search/search_state_manager.h"
#include "mirage/search/verification/verifier.h"
#include "mirage/utils/json_utils.h"

extern "C" {
void get_egraph(char const *expr);
}

namespace mirage {
namespace search {

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config,
                       char const *filename,
                       bool verbose = false);

  void generate_kernel_graphs();

  GeneratorConfig config;
  DimStrategy dim_strategy;

  char const *filename;
  std::vector<json> generated_graphs;
  int num_thread;
  bool verbose;

private:
  // Computation graph-related fields
  std::vector<std::shared_ptr<AbstractExpr>> computation_graph_output_patterns;
  std::vector<std::tuple<std::vector<int>,
                         type::DataType,
                         layout::DmemLayout,
                         std::vector<size_t>>>
      computation_graph_input_attrs;

  // Statistics-related fields
  std::atomic<int> num_total_random_tests;
  std::atomic<int> num_valid_kernel_graphs;
  std::atomic<int> num_total_states;

  // Time
  std::chrono::time_point<std::chrono::steady_clock> start_time;

  // count number of tasks
  std::atomic<int> num_tasks;
  size_t max_depth;

  //
  std::unordered_map<std::string, bool> seen_patterns;

  // Ranges-related fields
  std::vector<std::pair<size_t, IKNRange>> init_ranges;
  std::vector<std::vector<IKNRange>> target_ranges;

  // Verifier
  std::shared_ptr<Verifier> verifier;

  void generate_next_operator(
      SearchContext &c,
      std::function<bool(SearchContext const &)> const &verify,
      std::vector<SerializedSearchContext> &verified,
      size_t depth);

  void preprocess(kernel::Graph const &computation_graph);
  std::vector<bool>
      check_pattern(std::vector<std::shared_ptr<AbstractExpr>> &inputs);
  bool verify(kernel::Graph &g);

  void save_results() const;
  double get_elapsed_time_in_sec() const;
  void show_statistics() const;
};

} // namespace search
} // namespace mirage