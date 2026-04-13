#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mirage/search/config.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/order.h"
#include "mirage/search/range_propagation/irange.h"
#include "mirage/search/search_context.h"
#include "mirage/search/search_state_manager.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/verification/verifier.h"
#include "mirage/utils/json_utils.h"

namespace mirage {
namespace search {

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config,
                       char const *checkpoint_filename,
                       bool verbose = false);

  // Main entry points
  void search();          // non-symbolic search
  void search_symbolic(); // symbolic search

  // Results (read-only)
  std::vector<json> const &get_generated_graphs() const {
    return generated_graphs;
  }

  // Internal helper exposed for use by free functions in search_concrete.cc /
  // search_symbolic.cc. Not part of the user-facing API.
  bool check_abstract_expr(std::shared_ptr<AbstractExpr const> expr);

  // Legacy public access (for C API and existing callers)
  GeneratorConfig config;
  std::vector<json> generated_graphs;

private:
  DimStrategy dim_strategy;
  std::string checkpoint_filename_;
  bool verbose_;
  int num_threads_;

  // Computation graph info (set by init_from_ref_graph)
  std::vector<std::shared_ptr<AbstractExpr const>>
      computation_graph_output_exprs;
  std::vector<std::tuple<std::vector<int>,
                         type::DataType,
                         layout::DmemLayout,
                         std::vector<size_t>>>
      computation_graph_input_attrs;

  // Statistics
  std::atomic<int> num_total_random_tests;
  std::atomic<int> num_valid_kernel_graphs;
  std::atomic<int> num_total_states;
  std::atomic<int> num_symbolic_graphs;

  // Timing
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::atomic<int> num_tasks;
  size_t multithread_threshold_depth;

  // Ranges
  std::vector<std::pair<size_t, IKNRange>> init_ranges;
  std::vector<std::vector<IKNRange>> target_ranges;

  // Verifier
  std::shared_ptr<Verifier> verifier;

  // --- Non-symbolic search ---
  void enumerate_ops(SearchContext &c,
                     std::function<bool(SearchContext const &)> const &verify,
                     std::vector<SerializedSearchContext> &verified_graphs,
                     size_t search_depth,
                     bool is_a_new_thread_start = false);

  void enumerate_tb_config(
      SearchContext &c,
      std::function<bool(SearchContext const &)> const &verify,
      std::vector<SerializedSearchContext> &verified_graphs,
      size_t search_depth,
      std::vector<int> const &input_tensor_idx);

  void
      emit_tb_graph(SearchContext &c,
                    std::function<bool(SearchContext const &)> const &verify,
                    std::vector<SerializedSearchContext> &verified_graphs,
                    size_t search_depth,
                    std::vector<int> const &input_dtensor_indices_for_tb_graph);

  // --- Symbolic search ---
  void enumerate_symbolic_ops(
      std::shared_ptr<SymbolicKNGraph> kn_graph,
      std::shared_ptr<SymbolicTBGraph> tb_graph,
      std::vector<int> input_dtensor_indices_for_tb_graph,
      SearchLevel level,
      int search_depth,
      bool is_a_new_thread_start = false);

  void enumerate_symbolic_tb_config(std::shared_ptr<SymbolicKNGraph> kn_graph,
                                    std::vector<int> const &input_tensor_idx,
                                    int search_depth);

  void emit_symbolic_tb_graph(
      std::shared_ptr<SymbolicKNGraph> kn_graph,
      std::shared_ptr<SymbolicTBGraph> tb_graph,
      std::vector<int> const &input_dtensor_indices_for_tb_graph,
      int search_depth);

  bool verify_symbolic_graph(SymbolicKNGraph const &symbolic_graph);

  // --- Utilities ---
  void init_from_ref_graph(kernel::Graph const &computation_graph);
  bool verify_concrete_graph(kernel::Graph &g);
  void save_results() const;
  double get_elapsed_time_in_sec() const;
  void show_statistics() const;
};

} // namespace search
} // namespace mirage
