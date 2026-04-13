#include "mirage/search/search.h"
#include "search_helpers.h"

#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/verification/formal_verifier.h"
#include "mirage/search/verification/probabilistic_verifier.h"
#include "mirage/type.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/json_utils.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <thread>

namespace mirage {
namespace search {

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph,
    GeneratorConfig const &config,
    char const *checkpoint_filename,
    bool verbose)
    : config(config), dim_strategy(DimStrategy(config)),
      checkpoint_filename_(checkpoint_filename ? checkpoint_filename : ""),
      verbose_(verbose), num_total_random_tests(0), num_valid_kernel_graphs(0),
      num_total_states(0), num_symbolic_graphs(0), num_tasks(0),
      multithread_threshold_depth(10) {
  unsigned int max_num_threads = std::thread::hardware_concurrency();
  num_threads_ = (config.search_thread > max_num_threads)
                     ? max_num_threads
                     : config.search_thread;

  init_from_ref_graph(computation_graph);
}

void KernelGraphGenerator::search() {

  printf("[Search] Starting search with %d threads...\n", num_threads_);

  start_time = std::chrono::steady_clock::now();
  SearchContext c;
  c.level = SearchLevel::LV_KERNEL;
  c.kn_graph = std::make_shared<kernel::Graph>();

  for (auto const &input_attr : computation_graph_input_attrs) {
    auto [dim, data_type, layout, strides] = input_attr;
    c.kn_graph->new_input(dim, strides, data_type, layout);
  }

  std::vector<SerializedSearchContext> verified_graphs;

  printf("num_thread = %d\n", num_threads_);
#if !defined(MIRAGE_FINGERPRINT_USE_CPU) || defined(MIRAGE_USE_FORMAL_VERIFIER)
#pragma omp parallel num_threads(num_threads_)
#endif
  {
#if !defined(MIRAGE_FINGERPRINT_USE_CPU) || defined(MIRAGE_USE_FORMAL_VERIFIER)
#pragma omp single
#endif
    {
      enumerate_ops(
          c,
          [this](SearchContext const &c) -> bool {
            return c.level == SearchLevel::LV_KERNEL &&
                   this->verify_concrete_graph(*c.kn_graph);
          },
          verified_graphs,
          /*search_depth=*/0,
          /*is_a_new_thread_start=*/true);
    }
  }

  printf("num_tasks = %d tasks\n", num_tasks.load());

  save_results();

  double elapsed = get_elapsed_time_in_sec();
  bool timed_out = config.search_time_limit_sec > 0 &&
                   elapsed >= config.search_time_limit_sec;
  printf("\n");
  printf("[Search] Finished%s. Time elapsed: %fsec\n",
         timed_out ? " (time limit reached)" : "",
         elapsed);
  printf("[Search] Total states explored: %d\n", num_total_states.load());
  printf("[Search] Random tests performed: %d\n",
         num_total_random_tests.load());
  printf("[Search] Valid kernel graphs explored: %d\n",
         num_valid_kernel_graphs.load());
}

void KernelGraphGenerator::search_symbolic() {
  start_time = std::chrono::steady_clock::now();
  std::shared_ptr<SymbolicKNGraph> kn_graph =
      std::make_shared<SymbolicKNGraph>();

  for (auto const &input_attr : computation_graph_input_attrs) {
    auto [dim, data_type, layout, strides] = input_attr;
    kn_graph->add_input(dim, strides, data_type, layout);
  }

#pragma omp parallel num_threads(num_threads_)
  {
#pragma omp single
    {
      enumerate_symbolic_ops(
          kn_graph, nullptr, {}, SearchLevel::LV_KERNEL, 0, true);
    }
  }
  double elapsed = get_elapsed_time_in_sec();
  bool timed_out = config.search_time_limit_sec > 0 &&
                   elapsed >= config.search_time_limit_sec;
  printf("[Search] Symbolic search finished%s. Symbolic graph found: %ld. Time "
         "elapsed: %fsec\n",
         timed_out ? " (time limit reached)" : "",
         generated_graphs.size(),
         elapsed);

  std::ofstream ofs(checkpoint_filename_);
  ofs << json(generated_graphs);
  ofs.close();
}

void KernelGraphGenerator::init_from_ref_graph(
    kernel::Graph const &computation_graph) {
  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      computation_graph_input_attrs.push_back(
          {to_vector(op->output_tensors[0].num_dims, op->output_tensors[0].dim),
           op->output_tensors[0].data_type,
           op->output_tensors[0].layout,
           static_cast<kernel::KNInputOp *>(op)->input_strides});
    }
  }

  std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
      computation_graph_exprs;
  abstract_expr_eval(computation_graph, computation_graph_exprs);

  init_ranges = get_init_ranges(computation_graph);
  target_ranges = get_interact_ranges(init_ranges, computation_graph);
  assert(init_ranges.size() == target_ranges.size());

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      computation_graph_output_exprs.push_back(
          computation_graph_exprs.at(op->input_tensors[0].guid));
    }
  }

  for (auto const &final_expr : computation_graph_output_exprs) {
    initialize_final_expr(final_expr);
  }

  if (config.verifier_type == VerifierType::PROBABILISTIC_VERIFIER) {
    this->verifier = std::make_shared<ProbabilisticVerifier>(computation_graph);
  } else {
    this->verifier = std::make_shared<FormalVerifier>(computation_graph);
  }
}

bool KernelGraphGenerator::check_abstract_expr(
    std::shared_ptr<AbstractExpr const> expr) {
  if (!expr) {
    return false;
  }
  if (subexpr_to_final_expr(expr)) {
    return true;
  }
  return false;
}

void KernelGraphGenerator::save_results() const {
  std::ofstream ofs(checkpoint_filename_);
  ofs << json(generated_graphs);
}

double KernelGraphGenerator::get_elapsed_time_in_sec() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_time)
      .count();
}

void KernelGraphGenerator::show_statistics() const {
  double elapsed_time = get_elapsed_time_in_sec();
  double states_per_second = num_total_states.load() / elapsed_time;
  printf("[Search] States: %d, Random tests: %d, Valid mugraphs: %d, Time: "
         "%lf, States per second: %lf, Threads: %d\r",
         num_total_states.load(),
         num_total_random_tests.load(),
         num_valid_kernel_graphs.load(),
         elapsed_time,
         states_per_second,
         omp_get_num_threads());
}

} // namespace search
} // namespace mirage
