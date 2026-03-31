#include "common.h"

// RMSNorm + MLP (Gated Linear Unit)
//
// Computes: O = (rms_norm(X) @ W_up) * (rms_norm(X) @ W_gate)
//   X:      (n, d)  – input token embeddings
//   W_up:   (d, d)  – up-projection weight
//   W_gate: (d, d)  – gate-projection weight
//   O:      (n, d)  – output activations (GLU without silu, before down-proj)
//
// This is the core of the SwiGLU activation used in LLaMA-style models,
// minus the pointwise silu and the subsequent down-projection.
//
// The key fusion opportunity: both W_up and W_gate share the same rms_norm(X)
// input.  Using the identity  rms_norm(X) @ W = (X @ W) / rms(X),  a single
// kernel can compute both matmuls and the normalisation in one pass:
//
//   forloop over K tiles:
//     accum_up   += X_k @ W_up_k          (NO_RED)
//     accum_gate += X_k @ W_gate_k        (NO_RED)
//     accum_rms  += sum_sq(X_k per row)   (RED_LD_RMS)
//   output: (accum_up / accum_rms) * (accum_gate / accum_rms)
//
// Usage:
//   ./symbolic_rmsnorm_mlp                        – sweep all (n, d) configs
//   ./symbolic_rmsnorm_mlp -d                     – single debug config (n=8, d=4096)
//   ./symbolic_rmsnorm_mlp --force-nonsym         – re-run non-symbolic search
//   ./symbolic_rmsnorm_mlp --force-sym            – re-run symbolic search
//   ./symbolic_rmsnorm_mlp --skip-nonsym          – skip non-symbolic search entirely
//   ./symbolic_rmsnorm_mlp --sym-checkpoint <f>   – use <f> as symbolic checkpoint
//   ./symbolic_rmsnorm_mlp --time-limit <sec>     – search time limit (default 3600)
//
// Output files:
//   checkpoints/rmsnorm_mlp/  – per-config and shared symbolic checkpoints
//   results_rmsnorm_mlp.json  – best times for each (n, d) x {non-symbolic, symbolic}

struct RmsNormMlpConfig { int n, d; };

static RmsNormMlpConfig const kDebugConfig{16, 4096};

static std::vector<RmsNormMlpConfig> get_configs() {
  std::vector<RmsNormMlpConfig> configs;
  for (int n : {8, 16})
  for (int d : {1024, 2048, 4096})
    configs.push_back({n, d});
  return configs;
}

// Reference (unfused) graph:  O = (rms_norm(X) @ W_up) * (rms_norm(X) @ W_gate)
static void build_ref_graph(kernel::Graph &g, int n, int d) {
  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_up = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_gate = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor X_norm = g.rms_norm(X, {d});
  kernel::DTensor U      = g.matmul(X_norm, W_up);
  kernel::DTensor Gx     = g.matmul(X_norm, W_gate);
  kernel::DTensor O      = g.mul(U, Gx);
  g.mark_output(O);
}

// Manually constructed fused kernel for the debug config (n=8, d=4096).
//
// Single customized threadblock op, exactly analogous to rms_norm.cc but
// extended to two simultaneous matmuls:
//
//   grid_dim  = {64, 1, 1}  – 64 TBs, each owns 64 output columns (4096/64)
//   block_dim = {128, 1, 1}
//   forloop_range = 64       – 64 K-tiles of size 64 (4096/64)
//
//   Per forloop iteration k:
//     bX_k      : (8, 64)  – X tile along K, shared by all TBs
//     bW_up_k   : (64, 64) – W_up tile along K, each TB's column block
//     bW_gate_k : (64, 64) – W_gate tile, same column block
//     bM_up     = matmul(bX_k, bW_up_k)    → (8, 64)  partial up-proj
//     bM_gate   = matmul(bX_k, bW_gate_k)  → (8, 64)  partial gate-proj
//     bAccRms  accumulates rms(X[i,:]) via RED_LD_RMS  → (8, 1)
//     bAccUp   accumulates X @ W_up via NO_RED          → (8, 64)
//     bAccGate accumulates X @ W_gate via NO_RED        → (8, 64)
//
//   Post-forloop:
//     bUp   = div(bAccUp,   bAccRms)  → rms_norm(X) @ W_up
//     bGate = div(bAccGate, bAccRms)  → rms_norm(X) @ W_gate
//     bO    = mul(bUp, bGate)         → GLU output  (8, 64)
//
// Smem per TB: bX(1KB) + bW_up(8KB) + bW_gate(8KB) + intermediates ≈ 30KB
static void build_fused_graph(kernel::Graph &g) {
  int const n = kDebugConfig.n;  // 8
  int const d = kDebugConfig.d;  // 4096

  kernel::DTensor X = g.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_up = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_gate = g.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

  {
    dim3 grid_dim  = {64, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim, block_dim,
                              /*forloop_range=*/64, /*reduction_dimx=*/64);

    // bX: (8, 64) per iter – all TBs read the same X tiles (no grid partition)
    threadblock::STensor bX =
        bgraph.new_input(X, {-1, -1, -1}, /*forloop_dim=*/1,
                         layout::SmemRowMajor);
    // bW_up / bW_gate: (64, 64) per iter – each TB reads its own column block
    threadblock::STensor bW_up =
        bgraph.new_input(W_up,   {1, -1, -1}, /*forloop_dim=*/0,
                         layout::SmemRowMajor);
    threadblock::STensor bW_gate =
        bgraph.new_input(W_gate, {1, -1, -1}, /*forloop_dim=*/0,
                         layout::SmemRowMajor);

    // Partial matmul results per K-tile
    threadblock::STensor bM_up   = bgraph.matmul(bX, bW_up);   // (8, 64)
    threadblock::STensor bM_gate = bgraph.matmul(bX, bW_gate); // (8, 64)

    // rms(X[i,:]) – reduces along last dim of each X tile, accumulates across
    // K-tiles.  After 64 iters: sqrt( sum_k sum_j(X_k[i,j]^2) / 4096 )
    threadblock::STensor bAccRms =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP); // (8,1)

    // X @ W_up  and  X @ W_gate  via standard accumulation
    threadblock::STensor bAccUp =
        bgraph.forloop_accum(bM_up,   type::TB_FORLOOP_ACCUM_NO_RED_OP); // (8,64)
    threadblock::STensor bAccGate =
        bgraph.forloop_accum(bM_gate, type::TB_FORLOOP_ACCUM_NO_RED_OP); // (8,64)

    // rms_norm(X) @ W_up   = (X @ W_up)   / rms(X)   (broadcasts (8,1) divisor)
    threadblock::STensor bUp   = bgraph.div(bAccUp,   bAccRms); // (8,64)
    // rms_norm(X) @ W_gate = (X @ W_gate) / rms(X)
    threadblock::STensor bGate = bgraph.div(bAccGate, bAccRms); // (8,64)

    // GLU: element-wise multiply
    threadblock::STensor bO = bgraph.mul(bUp, bGate);           // (8,64)

    bgraph.mark_output(bO, {1, -1, -1}, /*forloop_dim=*/-1,
                       type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outs = g.customized({X, W_up, W_gate}, bgraph);
    assert(outs.size() == 1);
    g.mark_output(outs[0]);  // O: (8, 4096)
  }
}

// ---------------------------------------------------------------------------

static std::string const kCkptDir     = "checkpoints/rmsnorm_mlp";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_rmsnorm_mlp.json";

static std::string nonsym_ckpt(RmsNormMlpConfig const &cfg) {
  return kCkptDir + "/checkpoint_n" + std::to_string(cfg.n) +
         "_d" + std::to_string(cfg.d) + ".json";
}

static int find_result_idx(json const &results, RmsNormMlpConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("n", -1) == cfg.n &&
        results[i].value("d", -1) == cfg.d) {
      return i;
    }
  }
  return -1;
}

static void run_experiments(std::vector<RmsNormMlpConfig> const &configs,
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

  std::string best_nonsym_so;
  std::string best_sym_so;

  // ---- Experiment 0: manually fused kernel (debug config) ----------------
  std::cout << "\n=== rmsnorm_mlp: manually fused kernel ===" << std::endl;
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig.n, kDebugConfig.d);
    for (auto const &op : ref.operators) op->fingerprint();
    mirage::cpu::CTensor ref_fp =
        ref.operators.back()->input_tensors[0].copy_fingerprint_to_ctensor();

    kernel::Graph fused;
    build_fused_graph(fused);
    for (auto const &op : fused.operators) op->fingerprint();

    bool correct =
        fused.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp);
    std::cout << "  Fingerprint check: " << (correct ? "PASSED" : "FAILED")
              << std::endl;

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

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ---------
  if (skip_nonsym) {
    std::cout << "\n=== rmsnorm_mlp: non-symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== rmsnorm_mlp: non-symbolic search ===" << std::endl;
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

    if (!so_path.empty()) best_nonsym_so = so_path;

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

  // ---- Experiment 2: symbolic search (one shared checkpoint) -------------
  if (skip_sym) {
    std::cout << "\n=== rmsnorm_mlp: symbolic search (skipped) ===" << std::endl;
  } else {
  std::cout << "\n=== rmsnorm_mlp: symbolic search ===" << std::endl;
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
    graphs = apply_input_shapes(
        graphs, {{cfg.n, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d}});
    double sym_tune_time = 0;
    auto [best_time, so_path] = auto_tune_best_with_so(graphs, &sym_tune_time);
    std::cout << "  Best time (symbolic): " << best_time << " ms" << std::endl;
    std::cout << "  Best .so file: " << so_path << std::endl;

    if (!so_path.empty()) best_sym_so = so_path;

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

// Ablation study: measure search time with different symbolization levels
static void run_ablation(RmsNormMlpConfig const &cfg, double time_limit_sec) {
  ensure_dir(kCkptDir);

  struct AblationConfig {
    std::string name;
    SymFlags flags;
  };

  // Test from fastest (most symbolic) to slowest (most concrete)
  std::vector<AblationConfig> configs = {
    {"all_sym (full SSO)",         {true,  true,  true,  true,  true}},
    {"+sym_omap",                  {true,  true,  true,  true,  false}},
    {"+sym_fmap",                  {true,  true,  true,  false, false}},
    {"+sym_imap",                  {true,  true,  false, false, false}},
    {"sym_grid+frange only",       {true,  true,  false, false, false}},
    {"sym_grid only",              {true,  false, false, false, false}},
    {"all_concrete (exhaustive)",  {false, false, false, false, false}},
    {"maps_sym, grid_concrete",    {false, false, true,  true,  true}},
  };

  // Deduplicate: configs[2] and [3] are the same (both sym_grid+frange, no maps)
  // Fix: config[2] should be +sym_imap (grid+frange+imap), config[3] is grid+frange only
  configs = {
    {"all_sym",                    {true,  true,  true,  true,  true}},
    {"no_omap",                    {true,  true,  true,  true,  false}},
    {"no_fmap",                    {true,  true,  true,  false, true}},
    {"no_imap",                    {true,  true,  false, true,  true}},
    {"no_omap_fmap",               {true,  true,  true,  false, false}},
    {"no_imap_omap",               {true,  true,  false, true,  false}},
    {"no_maps",                    {true,  true,  false, false, false}},
    {"grid_only",                  {true,  false, false, false, false}},
    {"none",                       {false, false, false, false, false}},
    {"maps_only",                  {false, false, true,  true,  true}},
  };

  kernel::Graph ref;
  build_ref_graph(ref, cfg.n, cfg.d);
  for (auto const &op : ref.operators) op->fingerprint();

  printf("\n=== Ablation Study: rmsnorm_mlp n=%d d=%d ===\n", cfg.n, cfg.d);
  printf("%-25s %5s %5s %5s %5s %5s  %10s  %10s\n",
         "Config", "grid", "frnge", "imap", "fmap", "omap", "Time(s)", "Graphs");

  for (auto const &ac : configs) {
    std::string ckpt = kCkptDir + "/ablation_" + ac.name + ".json";
    // Remove old checkpoint to force fresh search
    std::remove(ckpt.c_str());

    search::AbstractExpr::symbolic_expr = true;
    search::GeneratorConfig gen_config =
        get_generator_config(/*use_symbolic=*/true, /*for_attention=*/false,
                             time_limit_sec, /*explore_all_mappings=*/false,
                             /*symbolic_maps=*/false, ac.flags);
    search::KernelGraphGenerator gen(ref, gen_config, ckpt.data());

    auto t0 = std::chrono::steady_clock::now();
    gen.generate_kernel_graphs_symbolic();
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    printf("%-25s %5s %5s %5s %5s %5s  %10.1f  %10zu\n",
           ac.name.c_str(),
           ac.flags.grid_dim ? "sym" : "enum",
           ac.flags.frange   ? "sym" : "enum",
           ac.flags.imap     ? "sym" : "enum",
           ac.flags.fmap     ? "sym" : "enum",
           ac.flags.omap     ? "sym" : "enum",
           elapsed,
           gen.generated_graphs.size());

    // Clean up ablation checkpoint
    std::remove(ckpt.c_str());
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
  bool ablation     = false;
  double time_limit = -1; // negative = use default (3600s)
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
    else if (arg == "--ablation")        ablation    = true;
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
                << " [--skip-sym] [--search-only] [--explore-all-maps]"
                << " [--symbolic-maps] [--config <n,d>]"
                << " [--sym-checkpoint <path>]"
                << " [--time-limit <seconds>]\n";
      return 1;
    }
  }
  std::vector<RmsNormMlpConfig> configs;
  if (!config_str.empty()) {
    int n, d;
    if (sscanf(config_str.c_str(), "%d,%d", &n, &d) != 2) {
      std::cerr << "Invalid --config format, expected n,d\n";
      return 1;
    }
    configs.push_back({n, d});
  } else {
    configs = debug ? std::vector<RmsNormMlpConfig>{kDebugConfig} : get_configs();
  }
  if (ablation) {
    RmsNormMlpConfig cfg = configs.empty() ? kDebugConfig : configs[0];
    run_ablation(cfg, time_limit >= 0 ? time_limit : 3600.0);
    return 0;
  }
  run_experiments(configs, force_nonsym, force_sym, skip_nonsym, skip_sym,
                  sym_ckpt_override, time_limit, explore_all, search_only,
                  sym_maps);
  return 0;
}
