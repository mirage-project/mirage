#include "common.h"

// RMS-Norm fused with MatMul
//
// Computes: O = matmul(rms_norm(X), W)
//   X: (n, d)  – input activations
//   W: (d, d)  – projection weight
//   O: (n, d)  – output
//
// Usage:
//   ./symbolic_rms_norm                        – sweep all (n, d) configs
//   ./symbolic_rms_norm -d                     – single debug config (n=8, d=4096)
//   ./symbolic_rms_norm --force-nonsym         – re-run non-symbolic search
//   ./symbolic_rms_norm --force-sym            – re-run symbolic search
//   ./symbolic_rms_norm --skip-nonsym          – skip non-symbolic search entirely
//   ./symbolic_rms_norm --skip-sym             – skip symbolic search entirely
//   ./symbolic_rms_norm --sym-checkpoint <f>   – use <f> as symbolic checkpoint
//   ./symbolic_rms_norm --time-limit <sec>     – search time limit (default 3600)
//
// Output files:
//   checkpoints/rms_norm/   – per-config and shared symbolic checkpoints
//   results_rms_norm.json   – best times for each (n, d) x {non-symbolic, symbolic}

struct RmsNormConfig { int n, d; };

static RmsNormConfig const kDebugConfig{8, 4096};

static std::vector<RmsNormConfig> get_configs() {
  std::vector<RmsNormConfig> configs;
  for (int n : {8, 16})
  for (int d : {1024, 2048, 4096})
    configs.push_back({n, d});
  return configs;
}

static void build_ref_graph(kernel::Graph &g, int n, int d) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor D = g.rms_norm(X, {d});
  kernel::DTensor O = g.matmul(D, W);
  g.mark_output(O);
}

static std::string const kCkptDir     = "checkpoints/rms_norm";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_rms_norm.json";

static std::string nonsym_ckpt(RmsNormConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) +
         "_d" + std::to_string(cfg.d) + ".json";
}

// Return the index of the entry matching (n, d) in the results array, or -1.
static int find_result_idx(json const &results, RmsNormConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<RmsNormConfig> const &configs,
                            bool force_nonsym, bool force_sym,
                            bool skip_nonsym = false,
                            bool skip_sym = false,
                            std::string const &sym_ckpt_override = "",
                            double time_limit_sec = -1,
                            bool explore_all_mappings = false,
                            bool search_only = false,
                            bool symbolic_maps = false) {
  ensure_dir(kCkptDir);
  std::string const sym_ckpt =
      sym_ckpt_override.empty() ? kSymCkpt : sym_ckpt_override;
  json results = load_results(kResultsFile);

  // Track the best .so files for verification
  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ----
  if (skip_nonsym) {
    std::cout << "\n=== rms_norm: non-symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== rms_norm: non-symbolic search ===" << std::endl;
  for (auto const &cfg : configs) {
    std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

    int idx = find_result_idx(results, cfg);
    if (!force_nonsym && idx != -1 && results[idx].contains("non_symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    kernel::Graph ref;
    build_ref_graph(ref, cfg.n, cfg.d);
    for (auto const &op : ref.operators) op->fingerprint();

    double ns_search_time = 0;
    std::vector<json> graphs =
        execute_search(ref, nonsym_ckpt(cfg), /*use_symbolic=*/false,
                       /*for_attention=*/false, time_limit_sec, explore_all_mappings,
                       &ns_search_time);
    if (search_only) {
      std::cout << "  --search-only: skipping profiling" << std::endl;
      continue;
    }
    double ns_cp_time = 0;
    auto [best_time, so_path] = profile_best_with_so(graphs, &ns_cp_time);
    std::cout << "  Best time (non-symbolic): " << best_time << " ms" << std::endl;
    std::cout << "  Best .so file: " << so_path << std::endl;

    if (!so_path.empty()) {
      best_nonsym_so = so_path;
    }

    if (idx == -1) {
      results.push_back({{"n", cfg.n}, {"d", cfg.d}});
      idx = (int)results.size() - 1;
    }
    results[idx]["non_symbolic_ms"] = best_time;
    results[idx]["non_symbolic_so"] = so_path;
    results[idx]["non_symbolic_search_s"] = ns_search_time;
    results[idx]["non_symbolic_compile_profile_s"] = ns_cp_time;
    save_results(kResultsFile, results);
  }
  } // end if (!skip_nonsym)

  // ---- Experiment 2: symbolic search (one shared checkpoint) ----
  if (skip_sym) {
    std::cout << "\n=== rms_norm: symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== rms_norm: symbolic search ===" << std::endl;
  double sym_search_time = 0;
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
    for (auto const &op : ref.operators) op->fingerprint();
    execute_search(ref, sym_ckpt, /*use_symbolic=*/true,
                   /*for_attention=*/false, time_limit_sec, explore_all_mappings,
                   &sym_search_time, symbolic_maps);
  }
  if (search_only) {
    std::cout << "  --search-only: skipping per-config tuning" << std::endl;
    return;
  }
  for (auto const &cfg : configs) {
    std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

    int idx = find_result_idx(results, cfg);
    if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    std::vector<json> graphs = load_graphs(sym_ckpt);
    graphs = apply_input_shapes(graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}});
    double sym_tune_time = 0;
    auto [best_time, so_path] = auto_tune_best_with_so(graphs, &sym_tune_time);
    std::cout << "  Best time (symbolic): " << best_time << " ms" << std::endl;
    std::cout << "  Best .so file: " << so_path << std::endl;

    if (!so_path.empty()) {
      best_sym_so = so_path;
    }

    if (idx == -1) {
      results.push_back({{"n", cfg.n}, {"d", cfg.d}});
      idx = (int)results.size() - 1;
    }
    results[idx]["symbolic_ms"] = best_time;
    results[idx]["symbolic_so"] = so_path;
    results[idx]["symbolic_search_s"] = sym_search_time;
    results[idx]["symbolic_tune_s"] = sym_tune_time;
    save_results(kResultsFile, results);
  }
  } // end if (!skip_sym)

  // ---- Output .so file names for correctness checking ----
  std::cout << "\n=== Best Kernel .so Files ===" << std::endl;
  std::cout << "Non-symbolic: " << best_nonsym_so << std::endl;
  std::cout << "Symbolic: " << best_sym_so << std::endl;
  std::cout << "\nTo verify correctness, run:" << std::endl;
  if (!best_nonsym_so.empty() && !best_sym_so.empty()) {
    // Calculate sizes for check_correctness
    auto const &cfg = configs.back();  // Use last config
    size_t input_size = cfg.n * cfg.d;
    size_t weight_size = cfg.d * cfg.d;
    size_t output_size = cfg.n * cfg.d;
    std::cout << "./check_correctness " << best_nonsym_so << " " << best_sym_so
              << " --inputs " << input_size << "," << weight_size
              << " --outputs " << output_size
              << " --buf 0" << std::endl;
  }
}

int main(int argc, char **argv) {
  bool debug        = false;
  bool force_nonsym = false;
  bool force_sym    = false;
  bool skip_nonsym  = false;
  bool skip_sym     = false;
  bool explore_all  = false;
  bool search_only  = false;
  bool sym_maps     = false;
  double time_limit = -1;
  std::string sym_ckpt_override;
  std::string config_str;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-d")                  debug        = true;
    else if (arg == "--force-nonsym") force_nonsym = true;
    else if (arg == "--force-sym")    force_sym    = true;
    else if (arg == "--skip-nonsym")  skip_nonsym  = true;
    else if (arg == "--skip-sym")     skip_sym     = true;
    else if (arg == "--explore-all-maps") explore_all = true;
    else if (arg == "--search-only")     search_only = true;
    else if (arg == "--symbolic-maps")   sym_maps    = true;
    else if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "--config requires n,d argument\n";
        return 1;
      }
      config_str = argv[++i];
    } else if (arg == "--sym-checkpoint") {
      if (i + 1 >= argc) {
        std::cerr << "--sym-checkpoint requires a path argument\n";
        return 1;
      }
      sym_ckpt_override = argv[++i];
    } else if (arg == "--time-limit") {
      if (i + 1 >= argc) {
        std::cerr << "--time-limit requires a value in seconds\n";
        return 1;
      }
      time_limit = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << '\n'
                << "Usage: " << argv[0]
                << " [-d] [--force-nonsym] [--force-sym] [--skip-nonsym]"
                << " [--skip-sym] [--search-only] [--explore-all-maps]"
                << " [--config <n,d>] [--sym-checkpoint <path>]"
                << " [--time-limit <seconds>]\n";
      return 1;
    }
  }
  std::vector<RmsNormConfig> configs;
  if (!config_str.empty()) {
    int n, d;
    if (sscanf(config_str.c_str(), "%d,%d", &n, &d) != 2) {
      std::cerr << "Invalid --config format, expected n,d\n";
      return 1;
    }
    configs.push_back({n, d});
  } else {
    configs = debug ? std::vector<RmsNormConfig>{kDebugConfig} : get_configs();
  }
  run_experiments(configs, force_nonsym, force_sym, skip_nonsym, skip_sym,
                  sym_ckpt_override, time_limit, explore_all, search_only, sym_maps);
  return 0;
}
