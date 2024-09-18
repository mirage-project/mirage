#pragma once

#include <atomic>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "mirage/search/config.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/irange.h"
#include "mirage/search/order.h"
#include "mirage/search/search_context.h"
#include "mirage/search/search_state_manager.h"
#include "mirage/utils/json_utils.h"

namespace mirage {
namespace search {

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config,
                       char const *filename);

  void generate_kernel_graphs();

  GeneratorConfig config;
  DimStrategy dim_strategy;

  char const *filename;
  std::vector<json> generated_graphs;
  int num_thread;

private:
  std::vector<std::shared_ptr<AlgebraicPattern>>
      computation_graph_output_patterns;
  std::vector<cpu::CTensor> computation_graph_output_tensors;
  std::vector<std::tuple<std::vector<int>, type::DataType, layout::DmemLayout>>
      computation_graph_input_attrs;

  std::atomic<int> num_total_kernel_graphs;
  std::atomic<int> num_total_random_tests;
  std::atomic<int> num_valid_kernel_graphs;
  std::atomic<int> num_total_states;

  std::mutex fp_mutex;
  std::mutex generated_graphs_mutex;

  // Ranges-related fields
  std::vector<std::pair<size_t, IKNRange>> init_ranges;
  std::vector<std::vector<IKNRange>> target_ranges;

  void search_from(std::vector<SerializedSearchContext> const &contexts);

  void generate_next_operator(
      SearchContext &c,
      std::function<bool(SearchContext const &)> const &verify,
      std::vector<SerializedSearchContext> &verified);

  bool create_threadblock_outputs(
      SearchContext &c,
      std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> const
          &algebraic_pattern,
      int3 output_map);

  void preprocess(kernel::Graph const &computation_graph);
  bool check_pattern(std::shared_ptr<AlgebraicPattern> pattern);
  bool have_same_fingerprint(std::vector<DTensor> const &outputs,
                             std::vector<int> const &match) const;
  bool verify(kernel::Graph const &g);

  void save_results() const;
};

} // namespace search
} // namespace mirage
