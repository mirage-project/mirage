#include "common.h"

// Attention
//
// Computes: O = softmax(Q @ K^T) @ V
//   Q:  (batch, g, head_dim)   g = num_heads * query_seq_len
//   Kt: (batch, head_dim, h)   h = num_heads * kv_seq_len
//   V:  (batch, h, head_dim)
//   O:  (batch, g, head_dim)
//
// Usage:
//   ./symbolic_attention                        – sweep all attention configs
//   ./symbolic_attention -d                     – single debug config
//   ./symbolic_attention --force-nonsym         – re-run non-symbolic search
//   ./symbolic_attention --force-sym            – re-run symbolic search
//   ./symbolic_attention --skip-nonsym          – skip non-symbolic search
//   entirely
//   ./symbolic_attention --skip-sym             – skip symbolic search entirely
//   ./symbolic_attention --sym-checkpoint <f>   – use <f> as symbolic
//   checkpoint
//   ./symbolic_attention --time-limit <sec>     – search time limit (default
//   3600)
//
// Output files:
//   checkpoints/attention/      – per-config and shared symbolic checkpoints
//   results_attention.json      – best times for each config x {non-symbolic,
//   symbolic}

struct AttentionConfig {
  int batch, num_heads, query_seq_len, kv_seq_len, head_dim;
};

static AttentionConfig const kDebugConfig{2, 8, 1, 1024, 128};

static std::vector<AttentionConfig> get_configs() {
  // Real-world shapes: g = num_heads * q_seq, h = num_heads * kv_seq
  // GQA models (LLaMA-7B/Mistral): 8 KV heads, head_dim=128
  // Full-head models (LLaMA-7B): 32 heads, head_dim=128
  // Decode: q_seq=1; Prefill: q_seq=128
  // batch >= 2 needed for parallelization opportunities
  std::vector<AttentionConfig> configs;
  for (int batch : {2, 4}) {
    for (int num_heads : {8, 32}) {
      for (int q_seq : {1, 128}) {
        for (int kv_seq : {128, 1024}) {
          for (int head_dim : {128}) {
            configs.push_back({batch, num_heads, q_seq, kv_seq, head_dim});
          }
        }
      }
    }
  }
  return configs;
}

static std::vector<std::vector<int>>
    get_input_shapes(AttentionConfig const &cfg) {
  int g = cfg.num_heads * cfg.query_seq_len;
  int h = cfg.num_heads * cfg.kv_seq_len;
  return {{cfg.batch, g, cfg.head_dim},
          {cfg.batch, cfg.head_dim, h},
          {cfg.batch, h, cfg.head_dim}};
}

static void build_ref_graph(kernel::Graph &ref, AttentionConfig const &cfg) {
  int g = cfg.num_heads * cfg.query_seq_len;
  int h = cfg.num_heads * cfg.kv_seq_len;
  kernel::DTensor Q =
      ref.new_input({cfg.batch, g, cfg.head_dim},
                    {(size_t)g * cfg.head_dim, (size_t)cfg.head_dim, 1},
                    type::DT_FLOAT16,
                    layout::DmemRowMajor);
  kernel::DTensor Kt = ref.new_input({cfg.batch, cfg.head_dim, h},
                                     {(size_t)h * cfg.head_dim, (size_t)h, 1},
                                     type::DT_FLOAT16,
                                     layout::DmemRowMajor);
  kernel::DTensor V =
      ref.new_input({cfg.batch, h, cfg.head_dim},
                    {(size_t)h * cfg.head_dim, (size_t)cfg.head_dim, 1},
                    type::DT_FLOAT16,
                    layout::DmemRowMajor);
  kernel::DTensor A = ref.matmul(Q, Kt);
  kernel::DTensor E = ref.exp(A);
  kernel::DTensor S = ref.reduction(E, 2);
  kernel::DTensor D = ref.div(E, S);
  kernel::DTensor O = ref.matmul(D, V);
  ref.mark_output(O);
}

static std::string const kCkptDir = "checkpoints/attention";
static std::string const kSymCkpt = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_attention.json";

static std::string nonsym_ckpt(AttentionConfig const &cfg) {
  return kCkptDir + "/checkpoint_batch" + std::to_string(cfg.batch) + "_heads" +
         std::to_string(cfg.num_heads) + "_qseq" +
         std::to_string(cfg.query_seq_len) + "_kvseq" +
         std::to_string(cfg.kv_seq_len) + "_dim" +
         std::to_string(cfg.head_dim) + ".json";
}

static int find_result_idx(json const &results, AttentionConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("batch", -1) == cfg.batch &&
        results[i].value("num_heads", -1) == cfg.num_heads &&
        results[i].value("query_seq_len", -1) == cfg.query_seq_len &&
        results[i].value("kv_seq_len", -1) == cfg.kv_seq_len &&
        results[i].value("head_dim", -1) == cfg.head_dim) {
      return i;
    }
  }
  return -1;
}

static json config_key(AttentionConfig const &cfg) {
  return {{"batch", cfg.batch},
          {"num_heads", cfg.num_heads},
          {"query_seq_len", cfg.query_seq_len},
          {"kv_seq_len", cfg.kv_seq_len},
          {"head_dim", cfg.head_dim}};
}

static void print_config(AttentionConfig const &cfg) {
  std::cout << "[batch=" << cfg.batch << " heads=" << cfg.num_heads
            << " q_seq=" << cfg.query_seq_len << " kv_seq=" << cfg.kv_seq_len
            << " head_dim=" << cfg.head_dim << "]" << std::endl;
}

static void run_experiments(std::vector<AttentionConfig> const &configs,
                            bool force_nonsym,
                            bool force_sym,
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

  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ----
  if (skip_nonsym) {
    std::cout << "\n=== attention: non-symbolic search (skipped) ==="
              << std::endl;
  } else {
    std::cout << "\n=== attention: non-symbolic search ===" << std::endl;
    for (auto const &cfg : configs) {
      print_config(cfg);

      int idx = find_result_idx(results, cfg);
      if (!force_nonsym && idx != -1 &&
          results[idx].contains("non_symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                  << " ms, skipping" << std::endl;
        continue;
      }

      kernel::Graph ref;
      build_ref_graph(ref, cfg);
      for (auto const &op : ref.operators) {
        op->fingerprint();
      }

      double ns_search_time = 0;
      std::vector<json> graphs = execute_search(ref,
                                                nonsym_ckpt(cfg),
                                                /*use_symbolic=*/false,
                                                /*for_attention=*/true,
                                                time_limit_sec,
                                                explore_all_mappings,
                                                &ns_search_time);
      if (search_only) {
        std::cout << "  --search-only: skipping profiling" << std::endl;
        continue;
      }
      double ns_cp_time = 0;
      auto [best_time, so_path] = profile_best_with_so(graphs, &ns_cp_time);
      std::cout << "  Best time (non-symbolic): " << best_time << " ms"
                << std::endl;
      std::cout << "  Best .so file: " << so_path << std::endl;

      if (!so_path.empty()) {
        best_nonsym_so = so_path;
      }

      if (idx == -1) {
        results.push_back(config_key(cfg));
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
    std::cout << "\n=== attention: symbolic search (skipped) ===" << std::endl;
  } else {
    std::cout << "\n=== attention: symbolic search ===" << std::endl;
    double sym_search_time = 0;
    {
      kernel::Graph ref;
      build_ref_graph(ref, kDebugConfig);
      for (auto const &op : ref.operators) {
        op->fingerprint();
      }
      execute_search(ref,
                     sym_ckpt,
                     /*use_symbolic=*/true,
                     /*for_attention=*/true,
                     time_limit_sec,
                     explore_all_mappings,
                     &sym_search_time,
                     symbolic_maps);
    }
    if (search_only) {
      std::cout << "  --search-only: skipping per-config tuning" << std::endl;
      return;
    }
    for (auto const &cfg : configs) {
      print_config(cfg);

      int idx = find_result_idx(results, cfg);
      if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                  << " ms, skipping" << std::endl;
        continue;
      }

      std::vector<json> graphs = load_graphs(sym_ckpt);
      graphs = apply_input_shapes(graphs, get_input_shapes(cfg));
      double sym_tune_time = 0;
      auto [best_time, so_path] =
          auto_tune_best_with_so(graphs, &sym_tune_time);
      std::cout << "  Best time (symbolic): " << best_time << " ms"
                << std::endl;
      std::cout << "  Best .so file: " << so_path << std::endl;

      if (!so_path.empty()) {
        best_sym_so = so_path;
      }

      if (idx == -1) {
        results.push_back(config_key(cfg));
        idx = (int)results.size() - 1;
      }
      results[idx]["symbolic_ms"] = best_time;
      results[idx]["symbolic_so"] = so_path;
      results[idx]["symbolic_search_s"] = sym_search_time;
      results[idx]["symbolic_tune_s"] = sym_tune_time;
      save_results(kResultsFile, results);
    }
  } // end if (!skip_sym)

  // ---- Summary -----------------------------------------------------------
  std::cout << "\n=== Best Kernel .so Files ===" << std::endl;
  std::cout << "Non-symbolic: " << best_nonsym_so << std::endl;
  std::cout << "Symbolic:     " << best_sym_so << std::endl;
}

static void run_ablation(AttentionConfig const &cfg, double time_limit_sec) {
  ensure_dir(kCkptDir);

  struct AblationConfig {
    std::string name;
    SymFlags flags;
  };

  std::vector<AblationConfig> configs = {
      {"all_sym", {true, true, true, true, true}},
      {"no_omap", {true, true, true, true, false}},
      {"no_fmap", {true, true, true, false, true}},
      {"no_imap", {true, true, false, true, true}},
      {"no_omap_fmap", {true, true, true, false, false}},
      {"no_imap_omap", {true, true, false, true, false}},
      {"no_maps", {true, true, false, false, false}},
  };

  int g = cfg.num_heads * cfg.query_seq_len;
  int h = cfg.num_heads * cfg.kv_seq_len;
  kernel::Graph ref;
  build_ref_graph(ref, cfg);
  for (auto const &op : ref.operators) {
    op->fingerprint();
  }

  printf("\n=== Ablation Study: attn batch=%d heads=%d q=%d kv=%d hd=%d ===\n",
         cfg.batch,
         cfg.num_heads,
         cfg.query_seq_len,
         cfg.kv_seq_len,
         cfg.head_dim);
  printf("%-25s %5s %5s %5s %5s %5s  %10s  %10s\n",
         "Config",
         "grid",
         "frnge",
         "imap",
         "fmap",
         "omap",
         "Time(s)",
         "Graphs");

  for (auto const &ac : configs) {
    std::string ckpt = kCkptDir + "/ablation_" + ac.name + ".json";
    std::remove(ckpt.c_str());

    search::AbstractExpr::symbolic_expr = true;
    search::GeneratorConfig gen_config =
        get_generator_config(/*use_symbolic=*/true,
                             /*for_attention=*/true,
                             time_limit_sec,
                             /*explore_all_mappings=*/false,
                             /*symbolic_maps=*/false,
                             ac.flags);
    search::KernelGraphGenerator gen(ref, gen_config, ckpt.data());

    auto t0 = std::chrono::steady_clock::now();
    gen.search_symbolic();
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    printf("%-25s %5s %5s %5s %5s %5s  %10.1f  %10zu\n",
           ac.name.c_str(),
           ac.flags.grid_dim ? "sym" : "enum",
           ac.flags.frange ? "sym" : "enum",
           ac.flags.imap ? "sym" : "enum",
           ac.flags.fmap ? "sym" : "enum",
           ac.flags.omap ? "sym" : "enum",
           elapsed,
           gen.get_generated_graphs().size());

    std::remove(ckpt.c_str());
  }
}

int main(int argc, char **argv) {
  bool debug = false;
  bool force_nonsym = false;
  bool force_sym = false;
  bool skip_nonsym = false;
  bool skip_sym = false;
  bool explore_all = false;
  bool search_only = false;
  bool sym_maps = false;
  bool ablation = false;
  double time_limit = -1;
  std::string sym_ckpt_override;
  std::string config_str;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-d") {
      debug = true;
    } else if (arg == "--force-nonsym") {
      force_nonsym = true;
    } else if (arg == "--force-sym") {
      force_sym = true;
    } else if (arg == "--skip-nonsym") {
      skip_nonsym = true;
    } else if (arg == "--skip-sym") {
      skip_sym = true;
    } else if (arg == "--explore-all-maps") {
      explore_all = true;
    } else if (arg == "--search-only") {
      search_only = true;
    } else if (arg == "--symbolic-maps") {
      sym_maps = true;
    } else if (arg == "--ablation") {
      ablation = true;
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
    } else if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "--config requires batch,heads,q_seq,kv_seq,head_dim\n";
        return 1;
      }
      config_str = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << "\n"
                << "Usage: " << argv[0]
                << " [-d] [--force-nonsym] [--force-sym] [--skip-nonsym]"
                << " [--skip-sym] [--search-only] [--explore-all-maps]"
                << " [--symbolic-maps]"
                << " [--config <batch,heads,q_seq,kv_seq,head_dim>]"
                << " [--sym-checkpoint <path>] [--time-limit <seconds>]\n";
      return 1;
    }
  }
  std::vector<AttentionConfig> configs;
  if (!config_str.empty()) {
    int b, h, q, kv, hd;
    if (sscanf(config_str.c_str(), "%d,%d,%d,%d,%d", &b, &h, &q, &kv, &hd) !=
        5) {
      std::cerr << "Invalid --config format, expected "
                   "batch,heads,q_seq,kv_seq,head_dim\n";
      return 1;
    }
    configs.push_back({b, h, q, kv, hd});
  } else {
    configs =
        debug ? std::vector<AttentionConfig>{kDebugConfig} : get_configs();
  }
  if (ablation) {
    AttentionConfig cfg = configs.empty() ? kDebugConfig : configs[0];
    run_ablation(cfg, time_limit >= 0 ? time_limit : 3600.0);
    return 0;
  }
  run_experiments(configs,
                  force_nonsym,
                  force_sym,
                  skip_nonsym,
                  skip_sym,
                  sym_ckpt_override,
                  time_limit,
                  explore_all,
                  search_only,
                  sym_maps);
  return 0;
}
