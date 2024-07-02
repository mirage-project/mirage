#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "mirage/search/config.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/order.h"
#include "mirage/utils/json_utils.h"

namespace mirage {
namespace search {

enum class SearchLevel {
  LV_KERNEL,
  LV_THREADBLOCK,
};

struct SearchContext {
  std::shared_ptr<kernel::Graph> kn_graph;
  std::shared_ptr<threadblock::Graph> tb_graph;
  SearchLevel level;

  SearchContext copy() const;

  SearchContext();
  ~SearchContext();
};

void to_json(json &j, SearchContext const &);
void from_json(json const &j, SearchContext &);

struct Checkpoint {
  kernel::Graph computation_graph;
  json best_graph;
  ProfileResult best_profile_result;
  GeneratorConfig config;
  std::vector<SearchContext> search_queue;
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
                                   search_queue,
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
  int num_thread;

  std::chrono::milliseconds timeout;

private:
  std::vector<std::shared_ptr<AlgebraicPattern>> final_patterns;
  std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>>
      computation_graph_patterns;

  std::vector<cpu::CTensor> output_tensors;

  std::atomic<int> num_total_kernel_graphs;
  std::atomic<int> num_total_random_tests;
  std::atomic<int> num_valid_kernel_graphs;

  std::mutex fp_mutex;
  std::mutex generated_graphs_mutex;

  std::queue<SearchContext> search_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::atomic<int> num_active_thread;
  void enqueue(SearchContext const &c);
  bool dequeue(SearchContext &c);
  void launch_thread();

  void generate_next_operator(SearchContext const &c);

  void optimize_layout(
      kernel::Graph &g, int op_idx, int ts_idx, int bop_idx, int bts_idx);
  void update_best_graph(kernel::Graph &g);
  bool create_tb_outputs(
      SearchContext &c,
      std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> const
          &algebraic_pattern,
      int3 output_map);

  std::vector<layout::SmemLayout>
      get_valid_output_layout(threadblock::TBOperator const *op, int idx);

  void process_outputs();
  bool check_pattern(std::shared_ptr<AlgebraicPattern> pattern);
  void fingerprint_eval();
  bool have_same_fingerprint(std::vector<DTensor> const &outputs,
                             std::vector<int> const &match) const;
  bool verify(SearchContext c);
  void recovery_from_checkpoint(Checkpoint const &checkpoint);
};

} // namespace search
} // namespace mirage
