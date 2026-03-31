#include "common.h"

// RMS-Norm fused with Linear (rectangular matmul)
//
// Computes: O = rms_norm(X) @ W
//   X: (n, d)       – input activations
//   W: (d, out_dim)  – projection weight (rectangular)
//   O: (n, out_dim)  – output
//
// This is the fused rms_norm + linear projection used in LLM transformer
// layers for QKV projection and gate/up projection. The key difference from
// rms_norm.cc is that W is rectangular (d != out_dim).
//
// End-to-end configs from real models:
//   Chameleon (32-head MHA): X(32,4096) @ W(4096,6144)   – QKV projection
//   Chameleon (MLP):         X(32,4096) @ W(4096,22016)  – gate+up projection
//   LLaMA-3 (GQA decode):   X(8,4096)  @ W(4096,6144)   – QKV projection
//   LLaMA-3 (MLP decode):   X(8,4096)  @ W(4096,28672)  – gate+up projection
//
// Usage:
//   ./symbolic_rms_linear                        – run all configs
//   ./symbolic_rms_linear --config <n,d,out>     – single config
//   ./symbolic_rms_linear --force-nonsym         – re-run non-symbolic search
//   ./symbolic_rms_linear --force-sym            – re-run symbolic search
//   ./symbolic_rms_linear --skip-nonsym          – skip non-symbolic search
//   ./symbolic_rms_linear --skip-sym             – skip symbolic search
//   ./symbolic_rms_linear --symbolic-maps        – use symbolic maps for SSO
//   ./symbolic_rms_linear --time-limit <sec>     – search time limit

struct RmsLinearConfig { int n, d, out_dim; std::string label; };

static std::vector<RmsLinearConfig> get_configs() {
  return {
    {32, 4096, 6144,  "chameleon_qkv"},    // Chameleon QKV: batch*tokens=32, d=4096, out=32*128+2*32*128
    {32, 4096, 22016, "chameleon_mlp"},     // Chameleon MLP: gate+up = 11008*2
    {8,  4096, 6144,  "llama_qkv"},         // LLaMA QKV: batch*tokens=8, out=32*128+2*8*128
    {8,  4096, 28672, "llama_mlp"},         // LLaMA MLP: gate+up = 14336*2
  };
}

static RmsLinearConfig const kDebugConfig{8, 4096, 6144, "debug"};

static void build_ref_graph(kernel::Graph &g, int n, int d, int out_dim) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = g.new_input(
      {d, out_dim}, {(size_t)out_dim, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor D = g.rms_norm(X, {d});
  kernel::DTensor O = g.matmul(D, W);
  g.mark_output(O);
}

static std::string const kCkptDir     = "checkpoints/rms_linear";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_rms_linear.json";

static std::string nonsym_ckpt(RmsLinearConfig const &cfg) {
  return kCkptDir + "/checkpoint_" + cfg.label + ".json";
}

static int find_result_idx(json const &results, RmsLinearConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("label", "") == cfg.label) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<RmsLinearConfig> const &configs,
                            bool force_nonsym, bool force_sym,
                            bool skip_nonsym, bool skip_sym,
                            std::string const &sym_ckpt_override,
                            double time_limit_sec,
                            bool search_only,
                            bool symbolic_maps) {
  ensure_dir(kCkptDir);
  std::string const sym_ckpt =
      sym_ckpt_override.empty() ? kSymCkpt : sym_ckpt_override;
  json results = load_results(kResultsFile);

  std::string best_nonsym_so, best_sym_so;

  // ---- Non-symbolic search (per-config) ----
  if (skip_nonsym) {
    std::cout << "\n=== rms_linear: non-symbolic search (skipped) ===" << std::endl;
  } else {
    std::cout << "\n=== rms_linear: non-symbolic search ===" << std::endl;
    for (auto const &cfg : configs) {
      std::cout << "[" << cfg.label << " n=" << cfg.n
                << " d=" << cfg.d << " out=" << cfg.out_dim << "]" << std::endl;

      int idx = find_result_idx(results, cfg);
      if (!force_nonsym && idx != -1 && results[idx].contains("non_symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                  << " ms, skipping" << std::endl;
        continue;
      }

      kernel::Graph ref;
      build_ref_graph(ref, cfg.n, cfg.d, cfg.out_dim);
      for (auto const &op : ref.operators) op->fingerprint();

      double ns_search_time = 0;
      std::vector<json> graphs =
          execute_search(ref, nonsym_ckpt(cfg), /*use_symbolic=*/false,
                         /*for_attention=*/false, time_limit_sec, false,
                         &ns_search_time);
      if (search_only) {
        std::cout << "  --search-only: skipping profiling" << std::endl;
        continue;
      }
      double ns_cp_time = 0;
      auto [best_time, so_path] = profile_best_with_so(graphs, &ns_cp_time);
      std::cout << "  Best time (non-symbolic): " << best_time << " ms" << std::endl;

      if (!so_path.empty()) best_nonsym_so = so_path;

      if (idx == -1) {
        results.push_back({{"label", cfg.label}, {"n", cfg.n}, {"d", cfg.d}, {"out_dim", cfg.out_dim}});
        idx = (int)results.size() - 1;
      }
      results[idx]["non_symbolic_ms"] = best_time;
      results[idx]["non_symbolic_so"] = so_path;
      results[idx]["non_symbolic_search_s"] = ns_search_time;
      results[idx]["non_symbolic_compile_profile_s"] = ns_cp_time;
      save_results(kResultsFile, results);
    }
  }

  // ---- Symbolic search (one shared checkpoint) ----
  if (skip_sym) {
    std::cout << "\n=== rms_linear: symbolic search (skipped) ===" << std::endl;
  } else {
    std::cout << "\n=== rms_linear: symbolic search ===" << std::endl;
    double sym_search_time = 0;
    {
      kernel::Graph ref;
      build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d, kDebugConfig.out_dim);
      for (auto const &op : ref.operators) op->fingerprint();
      execute_search(ref, sym_ckpt, /*use_symbolic=*/true,
                     /*for_attention=*/false, time_limit_sec, false,
                     &sym_search_time, symbolic_maps);
    }
    if (search_only) {
      std::cout << "  --search-only: skipping per-config tuning" << std::endl;
      return;
    }
    for (auto const &cfg : configs) {
      std::cout << "[" << cfg.label << " n=" << cfg.n
                << " d=" << cfg.d << " out=" << cfg.out_dim << "]" << std::endl;

      int idx = find_result_idx(results, cfg);
      if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                  << " ms, skipping" << std::endl;
        continue;
      }

      std::vector<json> graphs = load_graphs(sym_ckpt);
      graphs = apply_input_shapes(graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.out_dim}});
      double sym_tune_time = 0;
      auto [best_time, so_path] = auto_tune_best_with_so(graphs, &sym_tune_time);
      std::cout << "  Best time (symbolic): " << best_time << " ms" << std::endl;

      if (!so_path.empty()) best_sym_so = so_path;

      if (idx == -1) {
        results.push_back({{"label", cfg.label}, {"n", cfg.n}, {"d", cfg.d}, {"out_dim", cfg.out_dim}});
        idx = (int)results.size() - 1;
      }
      results[idx]["symbolic_ms"] = best_time;
      results[idx]["symbolic_so"] = so_path;
      results[idx]["symbolic_search_s"] = sym_search_time;
      results[idx]["symbolic_tune_s"] = sym_tune_time;
      save_results(kResultsFile, results);
    }
  }

  std::cout << "\n=== Best Kernel .so Files ===" << std::endl;
  std::cout << "Non-symbolic: " << best_nonsym_so << std::endl;
  std::cout << "Symbolic:     " << best_sym_so << std::endl;
}

int main(int argc, char **argv) {
  bool force_nonsym = false;
  bool force_sym    = false;
  bool skip_nonsym  = false;
  bool skip_sym     = false;
  bool search_only  = false;
  bool sym_maps     = false;
  double time_limit = -1;
  std::string sym_ckpt_override;
  std::string config_str;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--force-nonsym")      force_nonsym = true;
    else if (arg == "--force-sym")    force_sym    = true;
    else if (arg == "--skip-nonsym")  skip_nonsym  = true;
    else if (arg == "--skip-sym")     skip_sym     = true;
    else if (arg == "--search-only")  search_only  = true;
    else if (arg == "--symbolic-maps") sym_maps    = true;
    else if (arg == "--config") {
      if (i + 1 >= argc) { std::cerr << "--config requires n,d,out\n"; return 1; }
      config_str = argv[++i];
    } else if (arg == "--sym-checkpoint") {
      if (i + 1 >= argc) { std::cerr << "--sym-checkpoint requires path\n"; return 1; }
      sym_ckpt_override = argv[++i];
    } else if (arg == "--time-limit") {
      if (i + 1 >= argc) { std::cerr << "--time-limit requires seconds\n"; return 1; }
      time_limit = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }
  std::vector<RmsLinearConfig> configs;
  if (!config_str.empty()) {
    int n, d, out;
    if (sscanf(config_str.c_str(), "%d,%d,%d", &n, &d, &out) != 3) {
      std::cerr << "Invalid --config format, expected n,d,out_dim\n";
      return 1;
    }
    configs.push_back({n, d, out, "custom"});
  } else {
    configs = get_configs();
  }
  run_experiments(configs, force_nonsym, force_sym, skip_nonsym, skip_sym,
                  sym_ckpt_override, time_limit, search_only, sym_maps);
  return 0;
}
