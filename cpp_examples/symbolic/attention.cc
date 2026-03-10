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
//   ./symbolic_attention        – sweep all attention configs
//   ./symbolic_attention -d     – single debug config for quick testing
//
// Output files:
//   checkpoints/attention/      – per-config and shared symbolic checkpoints
//   results_attention.json      – best times for each config x {non-symbolic, symbolic}

struct AttentionConfig {
  int batch, num_heads, query_seq_len, kv_seq_len, head_dim;
};

static AttentionConfig const kDebugConfig{2, 8, 8, 128, 64};

static std::vector<AttentionConfig> get_configs() {
  std::vector<AttentionConfig> configs;
  for (int batch     : {2, 8})
  for (int num_heads : {8, 16})
  for (int q_seq     : {1, 8, 32})
  for (int kv_seq    : {128, 256, 512, 1024})
  for (int head_dim  : {64, 128})
    configs.push_back({batch, num_heads, q_seq, kv_seq, head_dim});
  return configs;
}

static std::vector<std::vector<int>> get_input_shapes(AttentionConfig const &cfg) {
  int g = cfg.num_heads * cfg.query_seq_len;
  int h = cfg.num_heads * cfg.kv_seq_len;
  return {{cfg.batch, g, cfg.head_dim},
          {cfg.batch, cfg.head_dim, h},
          {cfg.batch, h, cfg.head_dim}};
}

static void build_ref_graph(kernel::Graph &ref, AttentionConfig const &cfg) {
  int g = cfg.num_heads * cfg.query_seq_len;
  int h = cfg.num_heads * cfg.kv_seq_len;
  kernel::DTensor Q = ref.new_input(
      {cfg.batch, g, cfg.head_dim},
      {(size_t)g * cfg.head_dim, (size_t)cfg.head_dim, 1},
      type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor Kt = ref.new_input(
      {cfg.batch, cfg.head_dim, h},
      {(size_t)h * cfg.head_dim, (size_t)h, 1},
      type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor V = ref.new_input(
      {cfg.batch, h, cfg.head_dim},
      {(size_t)h * cfg.head_dim, (size_t)cfg.head_dim, 1},
      type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor A = ref.matmul(Q, Kt);
  kernel::DTensor E = ref.exp(A);
  kernel::DTensor S = ref.reduction(E, 2);
  kernel::DTensor D = ref.div(E, S);
  kernel::DTensor O = ref.matmul(D, V);
  ref.mark_output(O);
}

static std::string const kCkptDir     = "checkpoints/attention";
static std::string const kSymCkpt     = kCkptDir + "/checkpoint_symbolic.json";
static std::string const kResultsFile = "results_attention.json";

static std::string nonsym_ckpt(AttentionConfig const &cfg) {
  return kCkptDir +
         "/checkpoint_batch"  + std::to_string(cfg.batch) +
         "_heads"             + std::to_string(cfg.num_heads) +
         "_qseq"              + std::to_string(cfg.query_seq_len) +
         "_kvseq"             + std::to_string(cfg.kv_seq_len) +
         "_dim"               + std::to_string(cfg.head_dim) +
         ".json";
}

// Return the index of the entry matching this config in the results array, or -1.
static int find_result_idx(json const &results, AttentionConfig const &cfg) {
  for (int i = 0; i < (int)results.size(); ++i) {
    if (results[i].value("batch",          -1) == cfg.batch      &&
        results[i].value("num_heads",      -1) == cfg.num_heads  &&
        results[i].value("query_seq_len",  -1) == cfg.query_seq_len &&
        results[i].value("kv_seq_len",     -1) == cfg.kv_seq_len &&
        results[i].value("head_dim",       -1) == cfg.head_dim) {
      return i;
    }
  }
  return -1;
}

static json config_key(AttentionConfig const &cfg) {
  return {{"batch",         cfg.batch},
          {"num_heads",     cfg.num_heads},
          {"query_seq_len", cfg.query_seq_len},
          {"kv_seq_len",    cfg.kv_seq_len},
          {"head_dim",      cfg.head_dim}};
}

static void print_config(AttentionConfig const &cfg) {
  std::cout << "[batch=" << cfg.batch
            << " heads=" << cfg.num_heads
            << " q_seq=" << cfg.query_seq_len
            << " kv_seq=" << cfg.kv_seq_len
            << " head_dim=" << cfg.head_dim << "]" << std::endl;
}

static void run_experiments(std::vector<AttentionConfig> const &configs) {
  ensure_dir(kCkptDir);
  json results = load_results(kResultsFile);

  // ---- Experiment 1: non-symbolic search (per-config checkpoint) ----
  std::cout << "\n=== attention: non-symbolic search ===" << std::endl;
  for (auto const &cfg : configs) {
    print_config(cfg);

    int idx = find_result_idx(results, cfg);
    if (idx != -1 && results[idx].contains("non_symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["non_symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    kernel::Graph ref;
    build_ref_graph(ref, cfg);
    for (auto const &op : ref.operators) op->fingerprint();

    std::vector<json> graphs =
        execute_search(ref, nonsym_ckpt(cfg), /*use_symbolic=*/false,
                       /*for_attention=*/true);
    float best = profile_best_time(graphs);
    std::cout << "  Best time (non-symbolic): " << best << " ms" << std::endl;

    if (idx == -1) {
      results.push_back(config_key(cfg));
      idx = (int)results.size() - 1;
    }
    results[idx]["non_symbolic_ms"] = best;
    save_results(kResultsFile, results);
  }

  // ---- Experiment 2: symbolic search (one shared checkpoint) ----
  std::cout << "\n=== attention: symbolic search ===" << std::endl;
  // Ensure the symbolic checkpoint exists; run search with debug config if not.
  {
    kernel::Graph ref;
    build_ref_graph(ref, kDebugConfig);
    for (auto const &op : ref.operators) op->fingerprint();
    execute_search(ref, kSymCkpt, /*use_symbolic=*/true, /*for_attention=*/true);
  }
  // Apply the shape-independent checkpoint to every config in the sweep.
  for (auto const &cfg : configs) {
    print_config(cfg);

    int idx = find_result_idx(results, cfg);
    if (idx != -1 && results[idx].contains("symbolic_ms")) {
      std::cout << "  already recorded: " << results[idx]["symbolic_ms"]
                << " ms, skipping" << std::endl;
      continue;
    }

    std::vector<json> graphs = load_graphs(kSymCkpt);
    graphs = apply_input_shapes(graphs, get_input_shapes(cfg));
    float best = auto_tune_best_time(graphs);
    std::cout << "  Best time (symbolic): " << best << " ms" << std::endl;

    if (idx == -1) {
      results.push_back(config_key(cfg));
      idx = (int)results.size() - 1;
    }
    results[idx]["symbolic_ms"] = best;
    save_results(kResultsFile, results);
  }
}

int main(int argc, char **argv) {
  bool debug = (argc > 1 && std::string(argv[1]) == "-d");
  std::vector<AttentionConfig> configs =
      debug ? std::vector<AttentionConfig>{kDebugConfig} : get_configs();
  run_experiments(configs);
  return 0;
}
