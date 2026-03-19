#include "common.h"

// Low-Rank Matmul (LoRA adapter path)
//
// Computes: O = (X @ A) @ B
//   X: (n, d)  – input activations
//   A: (d, r)  – LoRA down-projection (low-rank)
//   B: (r, d)  – LoRA up-projection (low-rank)
//   O: (n, d)  – output activations
//
// This is just the LoRA delta path.  The full LoRA (X@W + (X@A)@B) needs
// 4 inputs which exceeds max_num_kernel_graph_op=5 with a customized op.
//
// Why symbolic search wins: highly asymmetric shapes — the intermediate
// [n, r] is tiny (r=16-128) and can fit entirely in shared memory.  The
// search can discover a single TB graph that does both matmuls: forloop
// over d for X@A, then multiply the accumulated [n, r] by B in shared
// memory.  Standard non-symbolic search may struggle with the unusual
// shape ratios.
//
// Usage:
//   ./symbolic_lora                        – sweep all (n, d, r) configs
//   ./symbolic_lora -d                     – single debug config (n=8, d=4096, r=64)
//   ./symbolic_lora --force-nonsym         – re-run non-symbolic search
//   ./symbolic_lora --force-sym            – re-run symbolic search
//   ./symbolic_lora --skip-nonsym          – skip non-symbolic search entirely
//   ./symbolic_lora --sym-checkpoint <f>   – use <f> as symbolic checkpoint
//   ./symbolic_lora --time-limit <sec>     – search time limit (default 3600)
//
// Output files:
//   checkpoints/lora/  – per-config and shared symbolic checkpoints
//   results_lora.json  – best times for each (n, d, r) x {non-symbolic, symbolic}

struct LoRAConfig { int n, d, r; };

static LoRAConfig const kDebugConfig{8, 4096, 64};

static std::vector<LoRAConfig> get_configs() {
  std::vector<LoRAConfig> configs;
  int n = 8;
  int d = 4096;
  for (int r : {16, 64, 128})
    configs.push_back({n, d, r});
  return configs;
}

// Reference (unfused) graph:  O = (X @ A) @ B
static void build_ref_graph(kernel::Graph &g, int n, int d, int r) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor A = g.new_input(
      {d, r}, {(size_t)r, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor B = g.new_input(
      {r, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor H = g.matmul(X, A);   // (n, r)
  kernel::DTensor O = g.matmul(H, B);   // (n, d)
  g.mark_output(O);
}

// ---------------------------------------------------------------------------

static std::string const kCkptDir     = "checkpoints/lora";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_lora.json";

static std::string nonsym_ckpt(LoRAConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) +
         "_d" + std::to_string(cfg.d) +
         "_r" + std::to_string(cfg.r) + ".json";
}

static int find_result_idx(json const &results, LoRAConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d &&
        results[i].value("r", -1) == cfg.r) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<LoRAConfig> const &configs,
                            bool force_nonsym, bool force_sym,
                            bool skip_nonsym = false,
                            bool skip_sym = false,
                            std::string const &sym_ckpt_override = "",
                            double time_limit_sec = -1,
                            bool explore_all_mappings = false,
                            bool search_only = false) {
  ensure_dir(kCkptDir);
  std::string const sym_ckpt =
      sym_ckpt_override.empty() ? kSymCkpt : sym_ckpt_override;
  json results = load_results(kResultsFile);

  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ---------
  if (skip_nonsym) {
    std::cout << "\n=== lora: non-symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== lora: non-symbolic search ===" << std::endl;
  for (auto const &cfg : configs) {
    std::cout << "[n=" << cfg.n << " d=" << cfg.d << " r=" << cfg.r << "]"
              << std::endl;

    int idx = find_result_idx(results, cfg);
    if (!force_nonsym && idx != -1 && results[idx].contains("non_symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    kernel::Graph ref;
    build_ref_graph(ref, cfg.n, cfg.d, cfg.r);
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
    auto [best_time, so_path] = profile_best_with_so(graphs);
    std::cout << "  Best time (non-symbolic): " << best_time << " ms" << std::endl;
    std::cout << "  Best .so file: " << so_path << std::endl;

    if (!so_path.empty()) best_nonsym_so = so_path;

    if (idx == -1) {
      results.push_back({{"n", cfg.n}, {"d", cfg.d}, {"r", cfg.r}});
      idx = (int)results.size() - 1;
    }
    results[idx]["non_symbolic_ms"] = best_time;
    results[idx]["non_symbolic_so"] = so_path;
    results[idx]["non_symbolic_search_s"] = ns_search_time;
    save_results(kResultsFile, results);
  }
  } // end if (!skip_nonsym)

  // ---- Experiment 2: symbolic search (one shared checkpoint) -------------
  if (skip_sym) {
    std::cout << "\n=== lora: symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== lora: symbolic search ===" << std::endl;
  double sym_search_time = 0;
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d, kDebugConfig.r);
    for (auto const &op : ref.operators) op->fingerprint();
    execute_search(ref, sym_ckpt, /*use_symbolic=*/true,
                   /*for_attention=*/false, time_limit_sec, explore_all_mappings,
                   &sym_search_time);
  }
  if (search_only) {
    std::cout << "  --search-only: skipping per-config tuning" << std::endl;
    return;
  }
  for (auto const &cfg : configs) {
    std::cout << "[n=" << cfg.n << " d=" << cfg.d << " r=" << cfg.r << "]"
              << std::endl;

    int idx = find_result_idx(results, cfg);
    if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    std::vector<json> graphs = load_graphs(sym_ckpt);
    graphs = apply_input_shapes(
        graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.r}, {cfg.r, cfg.d}});
    auto [best_time, so_path] = auto_tune_best_with_so(graphs);
    std::cout << "  Best time (symbolic): " << best_time << " ms" << std::endl;
    std::cout << "  Best .so file: " << so_path << std::endl;

    if (!so_path.empty()) best_sym_so = so_path;

    if (idx == -1) {
      results.push_back({{"n", cfg.n}, {"d", cfg.d}, {"r", cfg.r}});
      idx = (int)results.size() - 1;
    }
    results[idx]["symbolic_ms"] = best_time;
    results[idx]["symbolic_so"] = so_path;
    results[idx]["symbolic_search_s"] = sym_search_time;
    save_results(kResultsFile, results);
  }
  } // end if (!skip_sym)

  // ---- Summary -----------------------------------------------------------
  std::cout << "\n=== Best Kernel .so Files ===" << std::endl;
  std::cout << "Non-symbolic: " << best_nonsym_so << std::endl;
  std::cout << "Symbolic:     " << best_sym_so << std::endl;
  if (!best_nonsym_so.empty() && !best_sym_so.empty()) {
    auto const &cfg = configs.back();
    size_t x_size  = cfg.n * cfg.d;
    size_t a_size  = cfg.d * cfg.r;
    size_t b_size  = cfg.r * cfg.d;
    size_t o_size  = cfg.n * cfg.d;
    std::cout << "\nTo verify correctness, run:" << std::endl;
    std::cout << "./check_correctness " << best_nonsym_so << " " << best_sym_so
              << " --inputs " << x_size << "," << a_size << "," << b_size
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
  bool explore_all  = false;
  bool search_only  = false;
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
    else if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "--config requires n,d,r argument\n";
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
      std::cerr << "Unknown argument: " << arg << "\n"
                << "Usage: " << argv[0]
                << " [-d] [--force-nonsym] [--force-sym] [--skip-nonsym]"
                << " [--skip-sym] [--search-only] [--explore-all-maps]"
                << " [--config <n,d,r>] [--sym-checkpoint <path>]"
                << " [--time-limit <seconds>]\n";
      return 1;
    }
  }
  std::vector<LoRAConfig> configs;
  if (!config_str.empty()) {
    int n, d, r;
    if (sscanf(config_str.c_str(), "%d,%d,%d", &n, &d, &r) != 3) {
      std::cerr << "Invalid --config format, expected n,d,r\n";
      return 1;
    }
    configs.push_back({n, d, r});
  } else {
    configs = debug ? std::vector<LoRAConfig>{kDebugConfig} : get_configs();
  }
  run_experiments(configs, force_nonsym, force_sym, skip_nonsym, skip_sym,
                  sym_ckpt_override, time_limit, explore_all, search_only);
  return 0;
}
