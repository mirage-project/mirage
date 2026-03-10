#include "common.h"

// Chained Matmul with Activation (Two-Layer MLP)
//
// Computes: O = silu(X @ W1) @ W2
//   X:  (n, d)  – input token embeddings
//   W1: (d, d)  – first-layer weight
//   W2: (d, d)  – second-layer weight
//   O:  (n, d)  – output activations
//
// This is a serial matmul chain — the intermediate [n, d] from the first
// matmul feeds into the second.  The search needs to decide if both matmuls
// can share a TB graph (if the intermediate fits in shmem after tiling) or
// must be split into two kernel ops.
//
// Differs from SwiGLU: serial pipeline (fan-in) vs parallel branches (fan-out).
//
// Usage:
//   ./symbolic_two_layer_mlp                        – sweep all (n, d) configs
//   ./symbolic_two_layer_mlp -d                     – single debug config (n=8, d=4096)
//   ./symbolic_two_layer_mlp --force-nonsym         – re-run non-symbolic search
//   ./symbolic_two_layer_mlp --force-sym            – re-run symbolic search
//   ./symbolic_two_layer_mlp --skip-nonsym          – skip non-symbolic search entirely
//   ./symbolic_two_layer_mlp --sym-checkpoint <f>   – use <f> as symbolic checkpoint
//   ./symbolic_two_layer_mlp --time-limit <sec>     – search time limit (default 3600)
//   ./symbolic_two_layer_mlp --max-kn-ops <n>       – max kernel graph ops (default 5)
//
// Output files:
//   checkpoints/two_layer_mlp/  – per-config and shared symbolic checkpoints
//   results_two_layer_mlp.json  – best times for each (n, d) x {non-symbolic, symbolic}

struct TwoLayerMlpConfig { int n, d; };

static TwoLayerMlpConfig const kDebugConfig{8, 4096};

static std::vector<TwoLayerMlpConfig> get_configs() {
  std::vector<TwoLayerMlpConfig> configs;
  for (int n : {8, 16})
  for (int d : {1024, 2048, 4096})
    configs.push_back({n, d});
  return configs;
}

// Reference (unfused) graph:  O = silu(X @ W1) @ W2
static void build_ref_graph(kernel::Graph &g, int n, int d) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W1 = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W2 = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor H    = g.matmul(X, W1);   // (n, d)
  kernel::DTensor H_act = g.silu(H);        // (n, d)
  kernel::DTensor O    = g.matmul(H_act, W2); // (n, d)
  g.mark_output(O);
}

// ---------------------------------------------------------------------------

static std::string const kCkptDir     = "checkpoints/two_layer_mlp";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_two_layer_mlp.json";

static std::string nonsym_ckpt(TwoLayerMlpConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) +
         "_d" + std::to_string(cfg.d) + ".json";
}

static int find_result_idx(json const &results, TwoLayerMlpConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<TwoLayerMlpConfig> const &configs,
                            bool force_nonsym, bool force_sym,
                            bool skip_nonsym = false,
                            bool skip_sym = false,
                            std::string const &sym_ckpt_override = "",
                            double time_limit_sec = -1,
                            int max_kn_ops = -1) {
  ensure_dir(kCkptDir);
  std::string const sym_ckpt =
      sym_ckpt_override.empty() ? kSymCkpt : sym_ckpt_override;
  json results = load_results(kResultsFile);

  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ---------
  if (skip_nonsym) {
    std::cout << "\n=== two_layer_mlp: non-symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== two_layer_mlp: non-symbolic search ===" << std::endl;
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

    search::GeneratorConfig config = get_generator_config(
        /*use_symbolic_search=*/false, /*for_attention=*/false, time_limit_sec);
    if (max_kn_ops > 0) config.max_num_kernel_graph_op = max_kn_ops;
    search::KernelGraphGenerator gen(ref, config, nonsym_ckpt(cfg).data());
    gen.generate_kernel_graphs();

    auto [best_time, so_path] = profile_best_with_so(gen.generated_graphs);
    std::cout << "  Best time (non-symbolic): " << best_time << " ms" << std::endl;
    std::cout << "  Best .so file: " << so_path << std::endl;

    if (!so_path.empty()) best_nonsym_so = so_path;

    if (idx == -1) {
      results.push_back({{"n", cfg.n}, {"d", cfg.d}});
      idx = (int)results.size() - 1;
    }
    results[idx]["non_symbolic_ms"] = best_time;
    results[idx]["non_symbolic_so"] = so_path;
    save_results(kResultsFile, results);
  }
  } // end if (!skip_nonsym)

  // ---- Experiment 2: symbolic search (one shared checkpoint) -------------
  if (skip_sym) {
    std::cout << "\n=== two_layer_mlp: symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== two_layer_mlp: symbolic search ===" << std::endl;
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
    for (auto const &op : ref.operators) op->fingerprint();

    search::AbstractExpr::symbolic_expr = true;
    search::GeneratorConfig config = get_generator_config(
        /*use_symbolic_search=*/true, /*for_attention=*/false, time_limit_sec);
    if (max_kn_ops > 0) config.max_num_kernel_graph_op = max_kn_ops;
    search::KernelGraphGenerator gen(ref, config, sym_ckpt.data());
    if (!checkpoint_exists(sym_ckpt)) {
      gen.generate_kernel_graphs_symbolic();
    }
  }
  {
    std::vector<json> sym_graphs = load_graphs(sym_ckpt);
    for (auto const &cfg : configs) {
      std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

      int idx = find_result_idx(results, cfg);
      if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                  << " ms, skipping" << std::endl;
        continue;
      }

      std::vector<json> graphs = apply_input_shapes(
          sym_graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d}});
      auto [best_time, so_path] = auto_tune_best_with_so(graphs);
      std::cout << "  Best time (symbolic): " << best_time << " ms" << std::endl;
      std::cout << "  Best .so file: " << so_path << std::endl;

      if (!so_path.empty()) best_sym_so = so_path;

      if (idx == -1) {
        results.push_back({{"n", cfg.n}, {"d", cfg.d}});
        idx = (int)results.size() - 1;
      }
      results[idx]["symbolic_ms"] = best_time;
      results[idx]["symbolic_so"] = so_path;
      save_results(kResultsFile, results);
    }
  }
  } // end if (!skip_sym)

  // ---- Summary -----------------------------------------------------------
  std::cout << "\n=== Best Kernel .so Files ===" << std::endl;
  std::cout << "Non-symbolic: " << best_nonsym_so << std::endl;
  std::cout << "Symbolic:     " << best_sym_so << std::endl;
  if (!best_nonsym_so.empty() && !best_sym_so.empty()) {
    auto const &cfg = configs.back();
    size_t x_size  = cfg.n * cfg.d;
    size_t w_size  = cfg.d * cfg.d;
    size_t o_size  = cfg.n * cfg.d;
    std::cout << "\nTo verify correctness, run:" << std::endl;
    std::cout << "./check_correctness " << best_nonsym_so << " " << best_sym_so
              << " --inputs " << x_size << "," << w_size << "," << w_size
              << " --outputs " << o_size
              << " --buf 0" << std::endl;
  }
}

int main(int argc, char **argv) {
  bool debug        = false;
  bool force_nonsym = false;
  bool force_sym    = false;
  bool skip_nonsym  = false;
  bool skip_sym     = false;
  double time_limit = -1;
  int max_kn_ops    = -1;
  std::string sym_ckpt_override;
  std::string config_str;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-d")                  debug        = true;
    else if (arg == "--force-nonsym") force_nonsym = true;
    else if (arg == "--force-sym")    force_sym    = true;
    else if (arg == "--skip-nonsym")  skip_nonsym  = true;
    else if (arg == "--skip-sym")     skip_sym     = true;
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
    } else if (arg == "--max-kn-ops") {
      if (i + 1 >= argc) {
        std::cerr << "--max-kn-ops requires a value\n";
        return 1;
      }
      max_kn_ops = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n"
                << "Usage: " << argv[0]
                << " [-d] [--force-nonsym] [--force-sym] [--skip-nonsym]"
                << " [--skip-sym] [--config <n,d>]"
                << " [--sym-checkpoint <path>] [--time-limit <seconds>]"
                << " [--max-kn-ops <n>]\n";
      return 1;
    }
  }
  std::vector<TwoLayerMlpConfig> configs;
  if (!config_str.empty()) {
    int n, d;
    if (sscanf(config_str.c_str(), "%d,%d", &n, &d) != 2) {
      std::cerr << "Invalid --config format, expected n,d\n";
      return 1;
    }
    configs.push_back({n, d});
  } else {
    configs = debug ? std::vector<TwoLayerMlpConfig>{kDebugConfig} : get_configs();
  }
  run_experiments(configs, force_nonsym, force_sym, skip_nonsym, skip_sym,
                  sym_ckpt_override, time_limit, max_kn_ops);
  return 0;
}
