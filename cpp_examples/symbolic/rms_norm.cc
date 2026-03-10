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
//   ./symbolic_rms_norm --force-nonsym         – re-run non-symbolic search even if results exist
//   ./symbolic_rms_norm --force-sym            – re-run symbolic search even if results exist
//   ./symbolic_rms_norm --force-nonsym --force-sym  – re-run both
//
// Output files:
//   checkpoints/rms_norm/   – per-config and shared symbolic checkpoints
//   results_rms_norm.json   – best times for each (n, d) x {non-symbolic, symbolic}

struct RmsNormConfig { int n, d; };

static RmsNormConfig const kDebugConfig{8, 4096};

static std::vector<RmsNormConfig> get_configs() {
  std::vector<RmsNormConfig> configs;
  // for (int n : {1, 2, 4, 8, 16})
  // for (int d : {1024, 2048, 4096})
  for (int n : {8})
  for (int d : {4096})
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
                            bool force_nonsym, bool force_sym) {
  ensure_dir(kCkptDir);
  json results = load_results(kResultsFile);
  
  // Track the best .so files for verification
  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ----
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

    std::vector<json> graphs =
        execute_search(ref, nonsym_ckpt(cfg), /*use_symbolic=*/false);
    auto [best_time, so_path] = profile_best_with_so(graphs);
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
    save_results(kResultsFile, results);
  }

  // ---- Experiment 2: symbolic search (one shared checkpoint) ----
  std::cout << "\n=== rms_norm: symbolic search ===" << std::endl;
  // Ensure the symbolic checkpoint exists; run search with debug config if not.
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
    for (auto const &op : ref.operators) op->fingerprint();
    execute_search(ref, kSymCkpt, /*use_symbolic=*/true);
  }
  // Apply the shape-independent checkpoint to every config in the sweep.
  for (auto const &cfg : configs) {
    std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

    int idx = find_result_idx(results, cfg);
    if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    std::vector<json> graphs = load_graphs(kSymCkpt);
    graphs = apply_input_shapes(graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}});
    auto [best_time, so_path] = auto_tune_best_with_so(graphs);
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
    save_results(kResultsFile, results);
  }
  
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
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-d")              debug        = true;
    else if (arg == "--force-nonsym") force_nonsym = true;
    else if (arg == "--force-sym")    force_sym    = true;
    else {
      std::cerr << "Unknown argument: " << arg << "\n"
                << "Usage: " << argv[0]
                << " [-d] [--force-nonsym] [--force-sym]\n";
      return 1;
    }
  }
  std::vector<RmsNormConfig> configs =
      debug ? std::vector<RmsNormConfig>{kDebugConfig} : get_configs();
  run_experiments(configs, force_nonsym, force_sym);
  return 0;
}
