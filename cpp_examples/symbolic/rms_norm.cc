#include "common.h"

// O = matmul(rms_norm(X), W)

struct RmsNormConfig {
  int n, d;
};

static RmsNormConfig const kDefaultConfig{8, 4096};

static void build_ref_graph(kernel::Graph &g, int n, int d) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor D = g.rms_norm(X, {d});
  kernel::DTensor O = g.matmul(D, W);
  g.mark_output(O);
}

static std::string const kCkptDir = "checkpoints/rms_norm";
static std::string const kSymCkpt = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_rms_norm.json";

static std::string nonsym_ckpt(RmsNormConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) + "_d" +
         std::to_string(cfg.d) + ".json";
}

static int find_result_idx(json const &results, RmsNormConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(RmsNormConfig const &cfg,
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
    std::cout << "\n=== rms_norm: non-symbolic search (skipped) ==="
              << std::endl;
  } else {
    std::cout << "\n=== rms_norm: non-symbolic search ===" << std::endl;
    std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

    int idx = find_result_idx(results, cfg);
    if (!force_nonsym && idx != -1 &&
        results[idx].contains("non_symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                << " ms, skipping" << std::endl;
    } else {
      kernel::Graph ref;
      build_ref_graph(ref, cfg.n, cfg.d);
      for (auto const &op : ref.operators) {
        op->fingerprint();
      }

      double ns_search_time = 0;
      std::vector<json> graphs = execute_search(ref,
                                                nonsym_ckpt(cfg),
                                                /*use_symbolic=*/false,
                                                /*for_attention=*/false,
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
          results.push_back({{"n", cfg.n}, {"d", cfg.d}});
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
    std::cout << "\n=== rms_norm: symbolic search (skipped) ===" << std::endl;
  } else {
    std::cout << "\n=== rms_norm: symbolic search ===" << std::endl;
    double sym_search_time = 0;
    {
      kernel::Graph ref;
      build_ref_graph(ref, cfg.n, cfg.d);
      for (auto const &op : ref.operators) {
        op->fingerprint();
      }
      execute_search(ref,
                     sym_ckpt,
                     /*use_symbolic=*/true,
                     /*for_attention=*/false,
                     time_limit_sec,
                     explore_all_mappings,
                     &sym_search_time,
                     symbolic_maps);
    }
    if (!search_only) {
      std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

      int idx = find_result_idx(results, cfg);
      if (!force_sym && idx != -1 && results[idx].contains("symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                  << " ms, skipping" << std::endl;
      } else {
        std::vector<json> graphs = load_graphs(sym_ckpt);
        graphs = apply_input_shapes(graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}});
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
          results.push_back({{"n", cfg.n}, {"d", cfg.d}});
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
  std::cout << "Symbolic: " << best_sym_so << std::endl;
  if (!best_nonsym_so.empty() && !best_sym_so.empty()) {
    size_t input_size = cfg.n * cfg.d;
    size_t weight_size = cfg.d * cfg.d;
    size_t output_size = cfg.n * cfg.d;
    std::cout << "\nTo verify correctness, run:" << std::endl;
    std::cout << "./check_correctness " << best_nonsym_so << " " << best_sym_so
              << " --inputs " << input_size << "," << weight_size
              << " --outputs " << output_size << " --buf 0" << std::endl;
  }
}

// Uses fork() + SIGKILL for a hard timeout since verification can block
// indefinitely.
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static void run_ablation(RmsNormConfig const &cfg, double time_limit_sec) {
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

  printf("\n=== Ablation Study: rms_norm n=%d d=%d (timeout=%.0fs) ===\n",
         cfg.n,
         cfg.d,
         time_limit_sec);
  printf("%-25s %5s %5s %5s %5s %5s  %10s\n",
         "Config",
         "grid",
         "frnge",
         "imap",
         "fmap",
         "omap",
         "Time(s)");
  fflush(stdout);

  for (auto const &ac : configs) {
    std::string ckpt = kCkptDir + "/ablation_" + ac.name + ".json";
    std::remove(ckpt.c_str());

    auto wall_start = std::chrono::steady_clock::now();
    pid_t pid = fork();
    if (pid == 0) {
      kernel::Graph ref;
      build_ref_graph(ref, cfg.n, cfg.d);
      for (auto const &op : ref.operators) {
        op->fingerprint();
      }

      search::AbstractExpr::symbolic_expr = true;
      search::GeneratorConfig gen_config =
          get_generator_config(/*use_symbolic=*/true,
                               /*for_attention=*/false,
                               time_limit_sec,
                               /*explore_all_mappings=*/false,
                               /*symbolic_maps=*/false,
                               ac.flags);
      search::KernelGraphGenerator gen(ref, gen_config, ckpt.data());
      gen.search_symbolic();
      _exit(0);
    }
    bool timed_out = false;
    while (true) {
      int status;
      pid_t w = waitpid(pid, &status, WNOHANG);
      if (w > 0) {
        break;
      }
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - wall_start).count();
      if (elapsed > time_limit_sec) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        timed_out = true;
        break;
      }
      usleep(500000);
    }
    auto wall_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(wall_end - wall_start).count();

    printf("%-25s %5s %5s %5s %5s %5s  %10.1f%s\n",
           ac.name.c_str(),
           ac.flags.grid_dim ? "sym" : "enum",
           ac.flags.frange ? "sym" : "enum",
           ac.flags.imap ? "sym" : "enum",
           ac.flags.fmap ? "sym" : "enum",
           ac.flags.omap ? "sym" : "enum",
           elapsed,
           timed_out ? " (timeout)" : "");
    fflush(stdout);

    std::remove(ckpt.c_str());
  }
}

int main(int argc, char **argv) {
  RmsNormConfig cfg = kDefaultConfig;
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
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto require = [&](char const *name) {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--n") {
      cfg.n = std::atoi(require("--n"));
    } else if (arg == "--d") {
      cfg.d = std::atoi(require("--d"));
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
      sym_ckpt_override = require("--sym-checkpoint");
    } else if (arg == "--time-limit") {
      time_limit = std::stod(require("--time-limit"));
    } else {
      std::cerr << "Unknown argument: " << arg << '\n'
                << "Usage: " << argv[0]
                << " [--n <n>] [--d <d>] [--force-nonsym] [--force-sym]"
                << " [--skip-nonsym] [--skip-sym] [--search-only]"
                << " [--explore-all-maps] [--symbolic-maps] [--ablation]"
                << " [--sym-checkpoint <path>] [--time-limit <seconds>]\n";
      return 1;
    }
  }
  if (ablation) {
    run_ablation(cfg, time_limit >= 0 ? time_limit : 3600.0);
    return 0;
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
