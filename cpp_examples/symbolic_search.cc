#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/auto_tuner/auto_tuner.h"
#include "mirage/search/profile.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"
#include "mirage/utils/json_utils.h"

#include <algorithm>
#include <fstream>

using namespace mirage;
using namespace mirage::search;
using namespace mirage::kernel;
using namespace mirage::threadblock;

// -----------------------------------------------------------------------------
// Helpers: profiling and tuning
// -----------------------------------------------------------------------------

static float profile_best_time(std::vector<json> const &list_graphs) {
  float best = std::numeric_limits<float>::max();
  for (auto const &jgraph : list_graphs) {
    kernel::Graph g;
    from_json(jgraph, g);
    auto result = search::profile(&g);
    std::cout << "Profile result: " << result.run_time << std::endl;
    std::cout << "Error message: " << result.error_message << std::endl;
    best = std::min(best, result.run_time);
  }
  return best;
}

static float auto_tune_best_time(std::vector<json> const &list_symbolic_graphs) {
  std::vector<SymbolicKNGraph> symbolic_kn_graphs;
  for (auto const &jgraph : list_symbolic_graphs) {
    SymbolicKNGraph sg;
    from_json(jgraph, sg);
    symbolic_kn_graphs.push_back(sg);
  }
  AutoTuner auto_tuner(AutoTunerConfig{});
  kernel::Graph *tuned = auto_tuner.tune(symbolic_kn_graphs);
  return search::profile(tuned).run_time;
}

static float get_best_time(std::vector<json> const &graphs, bool use_symbolic) {
  return use_symbolic ? auto_tune_best_time(graphs) : profile_best_time(graphs);
}

// -----------------------------------------------------------------------------
// Helpers: generator config and checkpoint load/generate
// -----------------------------------------------------------------------------

static search::GeneratorConfig get_generator_config(bool use_symbolic_search,
                                                   bool for_attention) {
  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  if (use_symbolic_search) {
    config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  } else if (for_attention) {
    config.enable_attention_specific_optimization();
  }
  return config;
}

// -----------------------------------------------------------------------------
// RMS norm test (concrete graph only; no search)
// -----------------------------------------------------------------------------

void test_rms_norm(int n, int d) {
  std::vector<int> dimsX = {n, d};
  std::vector<size_t> stridesX = {(size_t)d, 1};
  std::vector<int> dimsW = {d, d};
  std::vector<size_t> stridesW = {(size_t)d, 1};

  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        dimsX, stridesX, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W = ref_graph.new_input(
        dimsW, stridesW, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor D = ref_graph.rms_norm(X, {X.dim[1]});
    kernel::DTensor O = ref_graph.matmul(D, W);
    ref_graph.mark_output(O);
    for (auto const &op : ref_graph.operators) op->fingerprint();
  }
  mirage::cpu::CTensor ref_fp =
      ref_graph.operators.back()->input_tensors[0].copy_fingerprint_to_ctensor();

  kernel::Graph graph;
  kernel::DTensor X =
      graph.new_input({8, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = graph.new_input({4096, 4096}, {4096, 1}, type::DT_FLOAT16,
                                      layout::DmemRowMajor);
  {
    dim3 grid_dim = {32, 1, 1}, block_dim = {128, 1, 1};
    namespace tb = mirage::threadblock;
    tb::Graph bgraph(grid_dim, block_dim, 16, 64);
    tb::STensor bX = bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
    tb::STensor bW = bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
    tb::STensor bAccX =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    tb::STensor bM = bgraph.matmul(bX, bW);
    tb::STensor bAccM =
        bgraph.forloop_accum(bM, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    tb::STensor bO = bgraph.div(bAccM, bAccX);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outputs = graph.customized({X, W}, bgraph);
    assert(outputs.size() == 1);
    graph.mark_output(outputs[0]);
  }
  for (auto const &op : graph.operators) op->fingerprint();
  assert(graph.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp));

  auto result = search::profile(&graph);
  std::cout << "Profile result: " << result.run_time << std::endl;
  std::cout << "Error message: " << result.error_message << std::endl;
  std::ofstream ofs("rms_norm.cu");
  ofs << result.cuda_code;
  ofs.close();
}

// -----------------------------------------------------------------------------
// Attention test (with optional symbolic search and checkpointing)
// -----------------------------------------------------------------------------

struct AttentionConfig {
  int batch;
  int num_heads;
  int query_seq_len;
  int kv_seq_len;
  int head_dim;
};

std::vector<std::vector<int>> get_input_shapes(AttentionConfig const &config) {
  return {{config.batch, config.num_heads * config.query_seq_len, config.head_dim},
          {config.batch, config.head_dim, config.num_heads * config.kv_seq_len},
          {config.batch, config.num_heads * config.kv_seq_len, config.head_dim}};
}

std::vector<AttentionConfig> get_attention_configs() {
  std::vector<int> batch_list{2, 8};
  std::vector<int> num_heads_list{8, 16};
  std::vector<int> query_seq_len_list{1, 8, 32};
  std::vector<int> kv_seq_len_list{128, 256, 512, 1024};
  std::vector<int> head_dim_list{64, 128};

  std::vector<AttentionConfig> configs;
  for (int batch : batch_list) {
    for (int num_heads : num_heads_list) {
      for (int query_seq_len : query_seq_len_list) {
        for (int kv_seq_len : kv_seq_len_list) {
          for (int head_dim : head_dim_list) {
            configs.push_back(AttentionConfig{batch, num_heads, query_seq_len, kv_seq_len, head_dim});
          }
        }
      }
    }
  }
  return configs;
}

std::string get_checkpoint_name(AttentionConfig const &config, bool use_symbolic_search) {
  if (!use_symbolic_search) {
    return "checkpoint_attention_batch" + std::to_string(config.batch) +
      "_heads" + std::to_string(config.num_heads) +
      "_qseq" + std::to_string(config.query_seq_len) +
      "_kvseq" + std::to_string(config.kv_seq_len) +
      "_dim" + std::to_string(config.head_dim) +
      ".json";
  } else {
    return "checkpoint_attention_symbolic.json";
  }
}

bool is_checkpoint_exists(std::string const &checkpoint_name) {
  std::ifstream ifs(checkpoint_name);
  return ifs.is_open();
}

std::vector<json> load_graphs(std::string const &checkpoint_name) {
  std::ifstream ifs(checkpoint_name);
  if (ifs.is_open()) {
    json j;
    ifs >> j;
    return std::vector<json>(j.begin(), j.end());
  }
  return {};
}

std::vector<json> update_input_shapes_for_symbolic_graphs(std::vector<json> const &graphs, std::vector<std::vector<int>> const &input_shapes) {
  std::vector<json> updated_graphs;
  for (auto const &graph : graphs) {
    SymbolicKNGraph symbolic_kn_graph;
    from_json(graph, symbolic_kn_graph);
    SymbolicKNGraph updated_graph = construct_graph_with_different_input_shapes(symbolic_kn_graph, input_shapes);
    updated_graphs.push_back(updated_graph);
  }
  return updated_graphs;
}

std::vector<json> search_graphs(AttentionConfig const &config, bool use_symbolic_search) {
  int g = config.num_heads * config.query_seq_len;
  int h = config.num_heads * config.kv_seq_len;

  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input(
        {config.batch, g, config.head_dim},
        {(size_t)g * config.head_dim, (size_t)config.head_dim, 1},
        type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor Kt = ref_graph.new_input(
        {config.batch, config.head_dim, h},
        {(size_t)h * config.head_dim, (size_t)h, 1},
        type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor V = ref_graph.new_input(
        {config.batch, h, config.head_dim},
        {(size_t)h * config.head_dim, (size_t)config.head_dim, 1},
        type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor A = ref_graph.matmul(Q, Kt);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2);
    kernel::DTensor D = ref_graph.div(E, S);
    kernel::DTensor O = ref_graph.matmul(D, V);
    ref_graph.mark_output(O);
    for (auto const &op : ref_graph.operators) op->fingerprint();
  }

  search::AbstractExpr::symbolic_expr = use_symbolic_search;
  search::GeneratorConfig search_config =
      get_generator_config(use_symbolic_search, true);

  std::string checkpoint_name = get_checkpoint_name(config, use_symbolic_search);

  std::cout << "[Searching] batch: " << config.batch << ", num_heads: " << config.num_heads
            << ", query_seq_len: " << config.query_seq_len
            << ", kv_seq_len: " << config.kv_seq_len << ", head_dim: " << config.head_dim
            << std::endl;

  search::KernelGraphGenerator gen(ref_graph, search_config, checkpoint_name.data());
  if (use_symbolic_search) {
    gen.generate_kernel_graphs_symbolic();
  } else {
    gen.generate_kernel_graphs();
  }
  return gen.generated_graphs;
}

float test_attention(AttentionConfig const &config, bool use_symbolic_search) {
  std::vector<json> graphs;

  std::string checkpoint_name = get_checkpoint_name(config, use_symbolic_search);
  if (is_checkpoint_exists(checkpoint_name)) {
    graphs = load_graphs(checkpoint_name);
    if (use_symbolic_search) {
      graphs = update_input_shapes_for_symbolic_graphs(graphs, get_input_shapes(config));
    }
  } else {
    graphs = search_graphs(config, use_symbolic_search);
  }

  // Print graphs
  for (auto const &graph : graphs) {
    std::cout << json(graph) << std::endl;
  }

  float best_time = get_best_time(graphs, use_symbolic_search);
  std::cout << "Best time: " << best_time << std::endl;
  return best_time;
}

// -----------------------------------------------------------------------------
// Main: CLI and batch sweep
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  bool use_symbolic_search =
      (argc > 1 && std::string(argv[1]) == "-s");

  std::vector<AttentionConfig> configs = get_attention_configs();

  if (use_symbolic_search) {
    test_attention(AttentionConfig{2, 8, 8, 128, 64}, true);
    return 0;
  }

  return 0;

  std::vector<json> results;
  {
    std::ifstream ifs("results.json");
    if (ifs.good()) {
      try {
        json j;
        ifs >> j;
        if (j.is_array())
          for (auto &el : j) results.push_back(el);
      } catch (...) {}
    }
  }

  for (AttentionConfig const &config : configs) {
    json key = {{"batch", config.batch},
                {"num_heads", config.num_heads},
                {"query_seq_len", config.query_seq_len},
                {"kv_seq_len", config.kv_seq_len},
                {"head_dim", config.head_dim}};
    auto it = std::find_if(
        results.begin(), results.end(),
        [&key](json const &e) {
          return e.contains("key") && e["key"] == key;
        });
    float time;
    if (it != results.end()) {
      time = (*it)["time"].get<float>();
      std::cout << "Config (batch=" << config.batch << ", num_heads="
                << config.num_heads << ", ...) already recorded, skipping"
                << std::endl;
    } else {
      time = test_attention(config, use_symbolic_search);
      results.push_back(json{{"key", key}, {"time", time}});
      std::ofstream ofs("results.json");
      ofs << json(results).dump(2);
      ofs.close();
    }
  }
}
