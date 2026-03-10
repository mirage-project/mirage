#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/auto_tuner/auto_tuner.h"
#include "mirage/search/profile.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"
#include "mirage/utils/json_utils.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// NOTE: using-directives in headers are acceptable here because this header is
// only included by example binaries, never by library code.
using namespace mirage;
using namespace mirage::search;
using namespace mirage::kernel;
using namespace mirage::threadblock;

// -----------------------------------------------------------------------------
// Shared profiling and tuning utilities
// -----------------------------------------------------------------------------

// Profile a list of concrete kernel graphs and return the fastest runtime.
// Compiles all graphs in parallel, then runs on GPU sequentially.
inline float profile_best_time(std::vector<json> const &list_graphs) {
  size_t const n = list_graphs.size();
  if (n == 0) return std::numeric_limits<float>::max();

  // Phase 1 — parallel compilation
  std::vector<search::ProfileCompileResult> compiled(n);
  {
    unsigned hw = std::thread::hardware_concurrency();
    size_t num_threads = std::min(n, static_cast<size_t>(hw > 0 ? hw : 8));
    std::atomic<size_t> next_idx{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        while (true) {
          size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
          if (i >= n) break;
          kernel::Graph g;
          from_json(list_graphs[i], g);
          compiled[i] = search::profile_compile(&g);
        }
      });
    }
    for (auto &t : threads) t.join();
  }

  // Phase 2 — sequential GPU profiling
  float best = std::numeric_limits<float>::max();
  for (size_t i = 0; i < n; ++i) {
    if (!compiled[i].is_success) {
      std::cout << "Profile result: " << std::numeric_limits<float>::max() << std::endl;
      std::cout << "Error message:  " << compiled[i].error_message << std::endl;
      continue;
    }
    auto result = search::profile_run(compiled[i]);
    std::cout << "Profile result: " << result.run_time << std::endl;
    std::cout << "Error message:  " << result.error_message << std::endl;
    best = std::min(best, result.run_time);
  }
  return best;
}

// Profile a list of concrete kernel graphs and return the best .so file path.
// Phase 1: compile all graphs in parallel (nvcc is the bottleneck).
// Phase 2: run compiled kernels on GPU sequentially (timing needs exclusive GPU).
inline std::pair<float, std::string> profile_best_with_so(std::vector<json> const &list_graphs) {
  size_t const n = list_graphs.size();
  if (n == 0) return {std::numeric_limits<float>::max(), ""};

  std::cout << "Profiling " << n << " graphs (parallel compile)..." << std::endl;

  // Phase 1 — parallel compilation
  std::vector<search::ProfileCompileResult> compiled(n);
  {
    unsigned hw = std::thread::hardware_concurrency();
    size_t num_threads = std::min(n, static_cast<size_t>(hw > 0 ? hw : 8));
    std::atomic<size_t> next_idx{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        while (true) {
          size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
          if (i >= n) break;
          kernel::Graph g;
          from_json(list_graphs[i], g);
          compiled[i] = search::profile_compile(&g);
        }
      });
    }
    for (auto &t : threads) t.join();
  }
  std::cout << "Compilation done." << std::endl;

  // Phase 2 — sequential GPU profiling
  float best = std::numeric_limits<float>::max();
  std::string best_so;
  for (size_t i = 0; i < n; ++i) {
    if (!compiled[i].is_success) {
      std::cout << "Graph " << i << " compile failed: "
                << compiled[i].error_message << std::endl;
      continue;
    }
    auto result = search::profile_run(compiled[i]);
    std::cout << "Graph " << i << " profile: " << result.run_time << " ms"
              << (result.is_success ? "" : " (FAILED: " + result.error_message + ")")
              << std::endl;
    if (result.is_success && result.run_time < best) {
      best = result.run_time;
      best_so = compiled[i].so_file;
    }
  }
  return {best, best_so};
}

// Auto-tune a list of symbolic kernel graphs and return the fastest runtime.
inline float auto_tune_best_time(std::vector<json> const &list_symbolic_graphs) {
  std::vector<SymbolicKNGraph> symbolic_kn_graphs;
  for (auto const &jgraph : list_symbolic_graphs) {
    SymbolicKNGraph sg;
    from_json(jgraph, sg);
    symbolic_kn_graphs.push_back(sg);
  }
  AutoTuner auto_tuner(AutoTunerConfig{});
  kernel::Graph *tuned = auto_tuner.tune_multi_threaded(symbolic_kn_graphs);
  if (!tuned) return std::numeric_limits<float>::max();
  return search::profile(tuned).run_time;
}

// Auto-tune a list of symbolic kernel graphs and return the best .so file path.
inline std::pair<float, std::string> auto_tune_best_with_so(std::vector<json> const &list_symbolic_graphs) {
  std::vector<SymbolicKNGraph> symbolic_kn_graphs;
  for (auto const &jgraph : list_symbolic_graphs) {
    SymbolicKNGraph sg;
    from_json(jgraph, sg);
    symbolic_kn_graphs.push_back(sg);
  }
  AutoTuner auto_tuner(AutoTunerConfig{});
  kernel::Graph *tuned = auto_tuner.tune_multi_threaded(symbolic_kn_graphs);
  if (!tuned) {
    std::cout << "Auto-tuning failed (no valid graph found)" << std::endl;
    return {std::numeric_limits<float>::max(), ""};
  }
  auto compiled = search::profile_compile(tuned);
  if (!compiled.is_success) {
    std::cout << "Compilation failed: " << compiled.error_message << std::endl;
    return {std::numeric_limits<float>::max(), ""};
  }
  auto result = search::profile_run(compiled);
  return {result.run_time, compiled.so_file};
}

// Dispatch to the appropriate timing method based on whether graphs are symbolic.
inline float get_best_time(std::vector<json> const &graphs, bool use_symbolic) {
  return use_symbolic ? auto_tune_best_time(graphs) : profile_best_time(graphs);
}

// Build a GeneratorConfig for the given search mode and operator type.
inline search::GeneratorConfig get_generator_config(bool use_symbolic_search,
                                                    bool for_attention,
                                                    double time_limit_sec = -1) {
  search::GeneratorConfig config = search::GeneratorConfig::get_default_config();
  if (use_symbolic_search) {
    config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  } else if (for_attention) {
    config.enable_attention_specific_optimization();
  }
  if (time_limit_sec >= 0) {
    config.search_time_limit_sec = time_limit_sec;
  }
  return config;
}

// -----------------------------------------------------------------------------
// Filesystem helpers
// -----------------------------------------------------------------------------

inline void ensure_dir(std::string const &path) {
  std::filesystem::create_directories(path);
}

// -----------------------------------------------------------------------------
// Results file helpers (JSON array, one entry per config)
// -----------------------------------------------------------------------------

inline json load_results(std::string const &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) return json::array();
  json j;
  ifs >> j;
  return j.is_array() ? j : json::array();
}

inline void save_results(std::string const &path, json const &results) {
  std::ofstream ofs(path);
  ofs << results.dump(2);
}

// -----------------------------------------------------------------------------
// Checkpoint helpers
// -----------------------------------------------------------------------------

inline bool checkpoint_exists(std::string const &path) {
  return std::ifstream(path).is_open();
}

inline std::vector<json> load_graphs(std::string const &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) return {};
  json j;
  ifs >> j;
  return std::vector<json>(j.begin(), j.end());
}

// Re-instantiate symbolic graphs with a different set of concrete input shapes.
// Must be called before auto_tune_best_time when loading symbolic graphs from a
// checkpoint (the checkpoint stores shape-independent topology; concrete shapes
// must be applied before the auto-tuner can run).
inline std::vector<json> apply_input_shapes(
    std::vector<json> const &graphs,
    std::vector<std::vector<int>> const &input_shapes) {
  std::vector<json> result;
  for (auto const &g : graphs) {
    SymbolicKNGraph sg;
    from_json(g, sg);
    result.push_back(construct_graph_with_different_input_shapes(sg, input_shapes));
  }
  return result;
}

// Run kernel-graph search (or load from checkpoint if one already exists).
//
// Returns:
//   non-symbolic: concrete KNGraph JSONs  → pass directly to profile_best_time
//   symbolic    : SymbolicKNGraph JSONs
//       - from fresh search : shapes already embedded, pass to auto_tune_best_time
//       - from checkpoint   : caller must call apply_input_shapes first
//
// Use the return value of checkpoint_exists(checkpoint) *before* this call to
// determine whether the result was loaded from a checkpoint.
inline std::vector<json> execute_search(kernel::Graph &ref_graph,
                                        std::string const &checkpoint,
                                        bool use_symbolic,
                                        bool for_attention = false,
                                        double time_limit_sec = -1) {
  if (checkpoint_exists(checkpoint)) {
    return load_graphs(checkpoint);
  }
  search::AbstractExpr::symbolic_expr = use_symbolic;
  search::GeneratorConfig config = get_generator_config(use_symbolic, for_attention, time_limit_sec);
  search::KernelGraphGenerator gen(ref_graph, config, checkpoint.data());
  if (use_symbolic) {
    gen.generate_kernel_graphs_symbolic();
  } else {
    gen.generate_kernel_graphs();
  }
  return gen.generated_graphs;
}
