#include "common.h"

// SwiGLU Gated MLP
//
// Computes: O = silu(X @ W_gate) * (X @ W_up)
//   X:      (n, d)  – input token embeddings
//   W_gate: (d, d)  – gate-projection weight
//   W_up:   (d, d)  – up-projection weight
//   O:      (n, d)  – output activations
//
// This is the SwiGLU activation used in LLaMA / Mistral / Qwen models.
// The key fusion opportunity: both matmuls share input X, and the silu
// activation on the gate branch is element-wise.  A single TB graph can
// compute both matmuls with a shared forloop over the K dimension, apply
// silu to the gate accumulator, then multiply:
//
//   forloop over K tiles:
//     accum_gate += X_k @ W_gate_k   (NO_RED)
//     accum_up   += X_k @ W_up_k     (NO_RED)
//   output: silu(accum_gate) * accum_up
//
// Differs from rmsnorm_mlp: adds silu activation, removes rms_norm.
//
// Usage:
//   ./symbolic_swiglu                        – sweep all (n, d) configs
//   ./symbolic_swiglu -d                     – single debug config (n=8, d=4096)
//   ./symbolic_swiglu --force-nonsym         – re-run non-symbolic search
//   ./symbolic_swiglu --force-sym            – re-run symbolic search
//   ./symbolic_swiglu --skip-nonsym          – skip non-symbolic search entirely
//   ./symbolic_swiglu --sym-checkpoint <f>   – use <f> as symbolic checkpoint
//   ./symbolic_swiglu --time-limit <sec>     – search time limit (default 3600)
//
// Output files:
//   checkpoints/swiglu/  – per-config and shared symbolic checkpoints
//   results_swiglu.json  – best times for each (n, d) x {non-symbolic, symbolic}

struct SwiGLUConfig { int n, d; };

static SwiGLUConfig const kDebugConfig{8, 4096};

static std::vector<SwiGLUConfig> get_configs() {
  std::vector<SwiGLUConfig> configs;
  for (int n : {8, 16})
  for (int d : {1024, 2048, 4096})
    configs.push_back({n, d});
  return configs;
}

// Reference (unfused) graph:  O = silu(X @ W_gate) * (X @ W_up)
static void build_ref_graph(kernel::Graph &g, int n, int d) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_gate = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_up = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor Gx = g.matmul(X, W_gate);
  kernel::DTensor Gx_act = g.silu(Gx);
  kernel::DTensor U  = g.matmul(X, W_up);
  kernel::DTensor O  = g.mul(Gx_act, U);
  g.mark_output(O);
}

// ---------------------------------------------------------------------------

static std::string const kCkptDir     = "checkpoints/swiglu";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_swiglu.json";

static std::string nonsym_ckpt(SwiGLUConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) +
         "_d" + std::to_string(cfg.d) + ".json";
}

static int find_result_idx(json const &results, SwiGLUConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<SwiGLUConfig> const &configs,
                            bool force_nonsym, bool force_sym,
                            bool skip_nonsym = false,
                            bool skip_sym = false,
                            std::string const &sym_ckpt_override = "",
                            double time_limit_sec = -1) {
  ensure_dir(kCkptDir);
  std::string const sym_ckpt =
      sym_ckpt_override.empty() ? kSymCkpt : sym_ckpt_override;
  json results = load_results(kResultsFile);

  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ---------
  if (skip_nonsym) {
    std::cout << "\n=== swiglu: non-symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== swiglu: non-symbolic search ===" << std::endl;
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
        execute_search(ref, nonsym_ckpt(cfg), /*use_symbolic=*/false,
                       /*for_attention=*/false, time_limit_sec);
    auto [best_time, so_path] = profile_best_with_so(graphs);
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
    std::cout << "\n=== swiglu: symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== swiglu: symbolic search ===" << std::endl;
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
    for (auto const &op : ref.operators) op->fingerprint();
    execute_search(ref, sym_ckpt, /*use_symbolic=*/true,
                   /*for_attention=*/false, time_limit_sec);
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
    graphs = apply_input_shapes(
        graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d}});
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
    } else {
      std::cerr << "Unknown argument: " << arg << "\n"
                << "Usage: " << argv[0]
                << " [-d] [--force-nonsym] [--force-sym] [--skip-nonsym]"
                << " [--skip-sym] [--config <n,d>]"
                << " [--sym-checkpoint <path>] [--time-limit <seconds>]\n";
      return 1;
    }
  }
  std::vector<SwiGLUConfig> configs;
  if (!config_str.empty()) {
    int n, d;
    if (sscanf(config_str.c_str(), "%d,%d", &n, &d) != 2) {
      std::cerr << "Invalid --config format, expected n,d\n";
      return 1;
    }
    configs.push_back({n, d});
  } else {
    configs = debug ? std::vector<SwiGLUConfig>{kDebugConfig} : get_configs();
  }
  run_experiments(configs, force_nonsym, force_sym, skip_nonsym, skip_sym,
                  sym_ckpt_override, time_limit);
  return 0;
}
