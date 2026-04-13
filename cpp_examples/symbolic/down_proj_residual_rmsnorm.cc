#include "common.h"

// Down-proj GEMM + Residual + RMS-Norm
//
// Computes: O = rms_norm(X_in @ W_down + X_res)
//   X_in:   (n, d)  – FFN intermediate activations (after up/gate projections)
//   W_down: (d, d)  – down-projection weight matrix
//   X_res:  (n, d)  – residual connection (pre-FFN token embeddings)
//   O:      (n, d)  – normalized output, ready for the next sub-layer
//
// Usage:
//   ./symbolic_down_proj_residual_rmsnorm                        – sweep all
//   (n, d) configs
//   ./symbolic_down_proj_residual_rmsnorm -d                     – single debug
//   config (n=8, d=4096)
//   ./symbolic_down_proj_residual_rmsnorm --force-nonsym         – re-run
//   non-symbolic search
//   ./symbolic_down_proj_residual_rmsnorm --force-sym            – re-run
//   symbolic search
//   ./symbolic_down_proj_residual_rmsnorm --skip-nonsym          – skip
//   non-symbolic search entirely
//   ./symbolic_down_proj_residual_rmsnorm --skip-sym             – skip
//   symbolic search entirely
//   ./symbolic_down_proj_residual_rmsnorm --sym-checkpoint <f>   – use <f> as
//   symbolic checkpoint
//   ./symbolic_down_proj_residual_rmsnorm --time-limit <sec>     – search time
//   limit (default 3600)
//
// Output files:
//   checkpoints/down_proj/  – per-config and shared symbolic checkpoints
//   results_down_proj.json  – best times for each (n, d) x {non-symbolic,
//   symbolic}

struct DownProjConfig {
  int n, d;
};

static DownProjConfig const kDebugConfig{8, 4096};

static std::vector<DownProjConfig> get_configs() {
  std::vector<DownProjConfig> configs;
  for (int n : {8, 16}) {
    for (int d : {1024, 2048, 4096}) {
      configs.push_back({n, d});
    }
  }
  return configs;
}

static void build_ref_graph(kernel::Graph &g, int n, int d) {
  kernel::DTensor X_in = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_down = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor X_res = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor Y = g.matmul(X_in, W_down);
  kernel::DTensor Z = g.add(Y, X_res);
  kernel::DTensor O = g.rms_norm(Z, {d});
  g.mark_output(O);
}

// Manually constructed fused kernel for the debug config (n=8, d=4096).
//
// Uses two customized threadblock ops:
//
//   Op 1 – fused GEMM: Y = X_in @ W_down
//     64 TBs tile the 4096 output columns; the forloop iterates over K=4096
//     in 64 tiles of 64, exactly matching the rms_norm.cc pattern.
//
//   Op 2 – fused add + RMS-norm: O = rms_norm(Y + X_res)
//     8 TBs (one per row).  forloop_range=1 so each TB loads its complete
//     row (1×4096) in a single pass.  forloop_accum(bZ, NO_RED) promotes
//     the pre-accum tensor bZ to after_accum=true (required for mark_output)
//     while forloop_accum(bZ, RED_LD_RMS) produces rms(Z[i,:]) as a (1,1)
//     scalar.  div then broadcasts it across the 4096 columns.
//
// This is mathematically equivalent to the reference graph and serves as a
// known-correct baseline for comparing against the symbolic-search results.
static void build_fused_graph(kernel::Graph &g) {
  int const n = kDebugConfig.n; // 8
  int const d = kDebugConfig.d; // 4096

  kernel::DTensor X_in = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_down = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor X_res = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

  // ---- Op 1: fused matmul → Y = X_in @ W_down -------------------------
  // 64 TBs over output columns (4096/64 = 64 cols each).
  // Forloop over K=4096 (64 tiles of 64).
  // smem per TB: bX_in(8×64)=1KB + bW_down(64×64)=8KB + bM+bY_acc ≈ 12KB
  kernel::DTensor Y;
  {
    dim3 grid_dim = {64, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim,
                              block_dim,
                              /*forloop_range=*/64,
                              /*reduction_dimx=*/64);
    // bX_in: (8, 64) per iter – all TBs read the same X_in tiles
    threadblock::STensor bX_in = bgraph.new_input(
        X_in, {-1, -1, -1}, /*forloop_dim=*/1, layout::SmemRowMajor);
    // bW_down: (64, 64) per iter – each TB reads its own column block
    threadblock::STensor bW_down = bgraph.new_input(
        W_down, {1, -1, -1}, /*forloop_dim=*/0, layout::SmemRowMajor);
    threadblock::STensor bM = bgraph.matmul(bX_in, bW_down); // (8,64)
    threadblock::STensor bY_acc =
        bgraph.forloop_accum(bM, type::TB_FORLOOP_ACCUM_NO_RED_OP); // (8,64)
    bgraph.mark_output(
        bY_acc, {1, -1, -1}, /*forloop_dim=*/-1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outs = g.customized({X_in, W_down}, bgraph);
    assert(outs.size() == 1);
    Y = outs[0]; // (8, 4096)
  }

  // ---- Op 2: fused add + rms_norm → O = rms_norm(Y + X_res) -----------
  // 8 TBs, one per row.  forloop_range=1 → each TB loads its full row once.
  // smem per TB: bY(1×4096)=8KB + bX_res(8KB) + bZ(8KB) + bZ_acc(8KB)
  //            + bZ_rms(~0) + bO(8KB) ≈ 40KB
  {
    dim3 grid_dim = {8, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim,
                              block_dim,
                              /*forloop_range=*/1,
                              /*reduction_dimx=*/64);
    // Each TB gets 1 row (grid x partitions dim 0: 8/8 = 1 row per TB)
    threadblock::STensor bY =
        bgraph.new_input(Y,
                         {0, -1, -1},
                         /*forloop_dim=*/-1,
                         layout::SmemRowMajor); // (1, 4096)
    threadblock::STensor bX_res =
        bgraph.new_input(X_res,
                         {0, -1, -1},
                         /*forloop_dim=*/-1,
                         layout::SmemRowMajor); // (1, 4096)
    // Z = Y + X_res  (after_accum=false, before forloop_accum)
    threadblock::STensor bZ = bgraph.add(bY, bX_res); // (1,4096)
    // Promote Z to after_accum=true (forloop_range=1 → identity accumulation)
    threadblock::STensor bZ_acc =
        bgraph.forloop_accum(bZ, type::TB_FORLOOP_ACCUM_NO_RED_OP); // (1,4096)
    // rms(Z[i,:]) = sqrt(mean(Z[i,:]^2))  →  shape (1,1)
    threadblock::STensor bZ_rms =
        bgraph.forloop_accum(bZ, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP); // (1,1)
    // O = Z / rms(Z)  – broadcasts (1,1) across 4096 columns
    threadblock::STensor bO = bgraph.div(bZ_acc, bZ_rms); // (1,4096)
    bgraph.mark_output(
        bO, {0, -1, -1}, /*forloop_dim=*/-1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outs = g.customized({Y, X_res}, bgraph);
    assert(outs.size() == 1);
    g.mark_output(outs[0]); // O: (8, 4096)
  }
}

static std::string const kCkptDir = "checkpoints/down_proj";
static std::string const kSymCkpt = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_down_proj.json";

static std::string nonsym_ckpt(DownProjConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) + "_d" +
         std::to_string(cfg.d) + ".json";
}

// Return the index of the entry matching (n, d) in the results array, or -1.
static int find_result_idx(json const &results, DownProjConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<DownProjConfig> const &configs,
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

  // Track the best .so files for verification
  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 0: manually constructed fused kernel (debug config only)
  // ----
  std::cout << "\n=== down_proj_residual_rmsnorm: manually fused kernel ==="
            << std::endl;
  {
    // Build reference (unfused) graph and capture its fingerprint
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
    for (auto const &op : ref.operators) {
      op->fingerprint();
    }
    mirage::cpu::CTensor ref_fp =
        ref.operators.back()->input_tensors[0].copy_fingerprint_to_ctensor();

    // Build the manually fused graph
    kernel::Graph fused;
    build_fused_graph(fused);
    for (auto const &op : fused.operators) {
      op->fingerprint();
    }

    // Verify correctness against the reference fingerprint
    bool correct =
        fused.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp);
    std::cout << "  Fingerprint check: " << (correct ? "PASSED" : "FAILED")
              << std::endl;

    // Profile the fused graph
    auto compiled = search::profile_compile(&fused);
    if (compiled.is_success) {
      auto result = search::profile_run(compiled);
      std::cout << "  Fused kernel time: " << result.run_time << " ms"
                << std::endl;
      std::cout << "  Fused kernel .so:  " << compiled.so_file << std::endl;
    } else {
      std::cout << "  Compilation failed: " << compiled.error_message
                << std::endl;
    }
  }

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ----
  if (skip_nonsym) {
    std::cout
        << "\n=== down_proj_residual_rmsnorm: non-symbolic search (skipped) ==="
        << std::endl;
  } else {
    std::cout << "\n=== down_proj_residual_rmsnorm: non-symbolic search ==="
              << std::endl;
    for (auto const &cfg : configs) {
      std::cout << "[n=" << cfg.n << " d=" << cfg.d << "]" << std::endl;

      int idx = find_result_idx(results, cfg);
      if (!force_nonsym && idx != -1 &&
          results[idx].contains("non_symbolic_ms")) {
        std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                  << " ms, skipping" << std::endl;
        continue;
      }

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
      if (search_only) {
        std::cout << "  --search-only: skipping profiling" << std::endl;
        continue;
      }
      auto [best_time, so_path] = profile_best_with_so(graphs);
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
      save_results(kResultsFile, results);
    }
  } // end if (!skip_nonsym)

  // ---- Experiment 2: symbolic search (one shared checkpoint) ----
  if (skip_sym) {
    std::cout
        << "\n=== down_proj_residual_rmsnorm: symbolic search (skipped) ==="
        << std::endl;
  } else {
    std::cout << "\n=== down_proj_residual_rmsnorm: symbolic search ==="
              << std::endl;
    double sym_search_time = 0;
    {
      kernel::Graph ref;
      build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
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
      graphs = apply_input_shapes(
          graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}, {cfg.n, cfg.d}});
      auto [best_time, so_path] = auto_tune_best_with_so(graphs);
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
    auto const &cfg = configs.back(); // Use last config
    size_t input_size = cfg.n * cfg.d;
    size_t weight_size = cfg.d * cfg.d;
    size_t residual_size = cfg.n * cfg.d;
    size_t output_size = cfg.n * cfg.d;
    std::cout << "./check_correctness " << best_nonsym_so << " " << best_sym_so
              << " --inputs " << input_size << "," << weight_size << ","
              << residual_size << " --outputs " << output_size << " --buf 0"
              << std::endl;
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
    } else if (arg == "--config") {
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
                << " [--skip-sym] [--search-only] [--explore-all-maps]"
                << " [--symbolic-maps] [--config <n,d>]"
                << " [--sym-checkpoint <path>]"
                << " [--time-limit <seconds>]\n";
      return 1;
    }
  }
  std::vector<DownProjConfig> configs;
  if (!config_str.empty()) {
    int n, d;
    if (sscanf(config_str.c_str(), "%d,%d", &n, &d) != 2) {
      std::cerr << "Invalid --config format, expected n,d\n";
      return 1;
    }
    configs.push_back({n, d});
  } else {
    configs = debug ? std::vector<DownProjConfig>{kDebugConfig} : get_configs();
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
