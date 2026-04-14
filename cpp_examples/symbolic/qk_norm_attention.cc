#include "common.h"

// O = softmax(rms_norm(Q) @ K^T) @ V

struct AttentionConfig {
  int batch, num_heads, query_seq_len, kv_seq_len, head_dim;
};

static AttentionConfig const kDefaultConfig{2, 8, 1, 1024, 128};

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

  kernel::DTensor Q_norm = ref.rms_norm(Q, {cfg.head_dim});
  kernel::DTensor A = ref.matmul(Q_norm, Kt);
  kernel::DTensor E = ref.exp(A);
  kernel::DTensor S = ref.reduction(E, 2);
  kernel::DTensor D = ref.div(E, S);
  kernel::DTensor O = ref.matmul(D, V);
  ref.mark_output(O);
}

static std::string const kCkptDir = "checkpoints/qk_norm_attention";
static std::string const kSymCkpt = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_qk_norm_attention.json";

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

static void run_experiments(AttentionConfig const &cfg,
                            bool force_nonsym,
                            bool force_sym,
                            bool skip_nonsym,
                            bool skip_sym,
                            std::string const &sym_ckpt_override,
                            double time_limit_sec,
                            bool explore_all_mappings,
                            bool search_only,
                            bool symbolic_maps) {
  ensure_dir(kCkptDir);
  std::string const sym_ckpt =
      sym_ckpt_override.empty() ? kSymCkpt : sym_ckpt_override;
  json results = load_results(kResultsFile);

  std::string best_nonsym_so;
  std::string best_sym_so;

  if (skip_nonsym) {
    std::cout << "\n=== qk_norm_attention: non-symbolic search (skipped) ==="
              << std::endl;
  } else {
    std::cout << "\n=== qk_norm_attention: non-symbolic search ==="
              << std::endl;
    print_config(cfg);

    int idx = find_result_idx(results, cfg);
    if (!force_nonsym && idx != -1 &&
        results[idx].contains("non_symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                << " ms, skipping" << std::endl;
    } else {
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
      if (!search_only) {
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
    }
  }

  if (skip_sym) {
    std::cout << "\n=== qk_norm_attention: symbolic search (skipped) ==="
              << std::endl;
  } else {
    std::cout << "\n=== qk_norm_attention: symbolic search ===" << std::endl;
    double sym_search_time = 0;
    {
      kernel::Graph ref;
      build_ref_graph(ref, cfg);
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
    if (!search_only) {
      print_config(cfg);

      int idx = find_result_idx(results, cfg);
      if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                  << " ms, skipping" << std::endl;
      } else {
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
    }
  }

  std::cout << "\n=== Best Kernel .so Files ===" << std::endl;
  std::cout << "Non-symbolic: " << best_nonsym_so << std::endl;
  std::cout << "Symbolic:     " << best_sym_so << std::endl;
}

int main(int argc, char **argv) {
  AttentionConfig cfg = kDefaultConfig;
  bool force_nonsym = false;
  bool force_sym = false;
  bool skip_nonsym = false;
  bool skip_sym = false;
  bool explore_all = false;
  bool search_only = false;
  bool sym_maps = false;
  double time_limit = -1;
  std::string sym_ckpt_override;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto require = [&](char const *name) {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--batch") {
      cfg.batch = std::atoi(require("--batch"));
    } else if (arg == "--num-heads") {
      cfg.num_heads = std::atoi(require("--num-heads"));
    } else if (arg == "--query-seq-len") {
      cfg.query_seq_len = std::atoi(require("--query-seq-len"));
    } else if (arg == "--kv-seq-len") {
      cfg.kv_seq_len = std::atoi(require("--kv-seq-len"));
    } else if (arg == "--head-dim") {
      cfg.head_dim = std::atoi(require("--head-dim"));
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
    } else if (arg == "--sym-checkpoint") {
      sym_ckpt_override = require("--sym-checkpoint");
    } else if (arg == "--time-limit") {
      time_limit = std::stod(require("--time-limit"));
    } else {
      std::cerr << "Unknown argument: " << arg << '\n'
                << "Usage: " << argv[0]
                << " [--batch <b>] [--num-heads <h>] [--query-seq-len <q>]"
                << " [--kv-seq-len <kv>] [--head-dim <d>]"
                << " [--force-nonsym] [--force-sym] [--skip-nonsym]"
                << " [--skip-sym] [--search-only] [--explore-all-maps]"
                << " [--symbolic-maps]"
                << " [--sym-checkpoint <path>] [--time-limit <seconds>]\n";
      return 1;
    }
  }
  run_experiments(cfg,
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
