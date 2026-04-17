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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace mirage;
using namespace mirage::search;
using namespace mirage::kernel;
using namespace mirage::threadblock;

inline float profile_best_time(std::vector<json> const &list_graphs) {
  size_t const n = list_graphs.size();
  if (n == 0) {
    return std::numeric_limits<float>::max();
  }

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
          if (i >= n) {
            break;
          }
          kernel::Graph g;
          from_json(list_graphs[i], g);
          compiled[i] = search::profile_compile(&g);
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  float best = std::numeric_limits<float>::max();
  for (size_t i = 0; i < n; ++i) {
    if (!compiled[i].is_success) {
      std::cout << "Profile result: " << std::numeric_limits<float>::max()
                << std::endl;
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

inline std::pair<float, std::string>
    profile_best_with_so(std::vector<json> const &list_graphs,
                         double *compile_profile_time_out = nullptr) {
  size_t const n = list_graphs.size();
  if (n == 0) {
    return {std::numeric_limits<float>::max(), ""};
  }

  std::cout << "Profiling " << n << " graphs (parallel compile)..."
            << std::endl;

  auto t_compile_start = std::chrono::steady_clock::now();
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
          if (i >= n) {
            break;
          }
          kernel::Graph g;
          from_json(list_graphs[i], g);
          compiled[i] = search::profile_compile(&g);
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }
  auto t_compile_end = std::chrono::steady_clock::now();
  double compile_sec =
      std::chrono::duration<double>(t_compile_end - t_compile_start).count();
  std::cout << "  Compile time: " << compile_sec << " s (" << n << " graphs)"
            << std::endl;

  auto t_profile_start = std::chrono::steady_clock::now();
  float best = std::numeric_limits<float>::max();
  std::string best_so;
  for (size_t i = 0; i < n; ++i) {
    if (!compiled[i].is_success) {
      std::cout << "Graph " << i
                << " compile failed: " << compiled[i].error_message
                << std::endl;
      continue;
    }
    auto result = search::profile_run(compiled[i]);
    std::cout << "Graph " << i << " profile: " << result.run_time << " ms"
              << (result.is_success ? ""
                                    : " (FAILED: " + result.error_message + ")")
              << std::endl;
    if (result.is_success && result.run_time < best) {
      best = result.run_time;
      best_so = compiled[i].so_file;
    }
  }
  auto t_profile_end = std::chrono::steady_clock::now();
  double profile_sec =
      std::chrono::duration<double>(t_profile_end - t_profile_start).count();
  std::cout << "  Profile time: " << profile_sec << " s" << std::endl;
  if (compile_profile_time_out) {
    *compile_profile_time_out = compile_sec + profile_sec;
  }
  return {best, best_so};
}

inline float
    auto_tune_best_time(std::vector<json> const &list_symbolic_graphs) {
  std::vector<SymbolicKNGraph> symbolic_kn_graphs;
  for (auto const &jgraph : list_symbolic_graphs) {
    SymbolicKNGraph sg;
    from_json(jgraph, sg);
    symbolic_kn_graphs.push_back(sg);
  }
  AutoTuner auto_tuner;
  kernel::Graph *tuned = auto_tuner.tune(symbolic_kn_graphs);
  if (!tuned) {
    return std::numeric_limits<float>::max();
  }
  return search::profile(tuned).run_time;
}

inline std::pair<float, std::string>
    auto_tune_best_with_so(std::vector<json> const &list_symbolic_graphs,
                           double *tune_time_out = nullptr) {
  std::vector<SymbolicKNGraph> symbolic_kn_graphs;
  for (auto const &jgraph : list_symbolic_graphs) {
    SymbolicKNGraph sg;
    from_json(jgraph, sg);
    symbolic_kn_graphs.push_back(sg);
  }
  AutoTuner auto_tuner;
  auto t0 = std::chrono::steady_clock::now();
  kernel::Graph *tuned = auto_tuner.tune(symbolic_kn_graphs);
  auto t1 = std::chrono::steady_clock::now();
  double tune_sec = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  Auto-tuning time: " << tune_sec << " s" << std::endl;
  if (tune_time_out) {
    *tune_time_out = tune_sec;
  }
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

inline float get_best_time(std::vector<json> const &graphs, bool use_symbolic) {
  return use_symbolic ? auto_tune_best_time(graphs) : profile_best_time(graphs);
}

struct SymFlags {
  bool grid_dim = false;
  bool frange = false;
  bool imap = false;
  bool fmap = false;
  bool omap = false;

  static SymFlags all() {
    return {true, true, true, true, true};
  }
  static SymFlags none() {
    return {false, false, false, false, false};
  }
};

inline search::GeneratorConfig
    get_generator_config(bool use_symbolic_search,
                         bool for_attention,
                         double time_limit_sec = -1,
                         bool explore_all_mappings = false,
                         bool symbolic_maps = false,
                         SymFlags sym = SymFlags()) {
  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  if (use_symbolic_search) {
    config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  } else if (for_attention) {
    config.enable_attention_specific_optimization();
  }
  if (time_limit_sec >= 0) {
    config.search_time_limit_sec = time_limit_sec;
  }
  config.explore_all_mappings = explore_all_mappings;
  config.symbolic_maps = symbolic_maps;
  if (symbolic_maps) {
    sym = SymFlags::all();
  }
  config.sym_grid_dim = sym.grid_dim;
  config.sym_frange = sym.frange;
  config.sym_imap = sym.imap;
  config.sym_fmap = sym.fmap;
  config.sym_omap = sym.omap;
  return config;
}

inline void ensure_dir(std::string const &path) {
  std::filesystem::create_directories(path);
}

inline json load_results(std::string const &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return json::array();
  }
  json j;
  ifs >> j;
  return j.is_array() ? j : json::array();
}

inline void save_results(std::string const &path, json const &results) {
  std::ofstream ofs(path);
  ofs << results.dump(2);
}

inline bool checkpoint_exists(std::string const &path) {
  return std::ifstream(path).is_open();
}

inline std::vector<json> load_graphs(std::string const &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {};
  }
  json j;
  ifs >> j;
  return std::vector<json>(j.begin(), j.end());
}

// Re-instantiate symbolic graphs with concrete input shapes. Required before
// auto-tuning when graphs were loaded from a checkpoint (the checkpoint stores
// shape-independent topology).
inline std::vector<json>
    apply_input_shapes(std::vector<json> const &graphs,
                       std::vector<std::vector<int>> const &input_shapes) {
  std::vector<json> result;
  for (auto const &g : graphs) {
    SymbolicKNGraph sg;
    from_json(g, sg);
    result.push_back(
        construct_graph_with_different_input_shapes(sg, input_shapes));
  }
  return result;
}

inline std::vector<json> execute_search(kernel::Graph &ref_graph,
                                        std::string const &checkpoint,
                                        bool use_symbolic,
                                        bool for_attention = false,
                                        double time_limit_sec = -1,
                                        bool explore_all_mappings = false,
                                        double *search_time_out = nullptr,
                                        bool symbolic_maps = false,
                                        SymFlags sym = SymFlags()) {
  std::string ckpt = checkpoint;
  if (explore_all_mappings) {
    auto pos = ckpt.rfind(".json");
    if (pos != std::string::npos) {
      ckpt.insert(pos, "_fullspace");
    } else {
      ckpt += "_fullspace";
    }
  }
  if (checkpoint_exists(ckpt)) {
    auto graphs = load_graphs(ckpt);
    std::cout << "  Search: loaded from checkpoint (" << graphs.size()
              << " graphs)" << std::endl;
    if (search_time_out) {
      *search_time_out = -1;
    }
    return graphs;
  }
  search::AbstractExpr::symbolic_expr = use_symbolic;
  search::GeneratorConfig config = get_generator_config(use_symbolic,
                                                        for_attention,
                                                        time_limit_sec,
                                                        explore_all_mappings,
                                                        symbolic_maps,
                                                        sym);
  search::KernelGraphGenerator gen(ref_graph, config, ckpt.data());
  auto t0 = std::chrono::steady_clock::now();
  if (use_symbolic) {
    gen.search_symbolic();
  } else {
    gen.search();
  }
  auto t1 = std::chrono::steady_clock::now();
  double search_sec = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  Search time: " << search_sec << " s ("
            << gen.get_generated_graphs().size() << " graphs, "
            << (use_symbolic ? "symbolic" : "non-symbolic") << ")" << std::endl;
  if (search_time_out) {
    *search_time_out = search_sec;
  }
  return gen.get_generated_graphs();
}
