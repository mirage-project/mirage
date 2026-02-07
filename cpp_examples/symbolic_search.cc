#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/search/profile.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"
#include "mirage/search/verification/formal_verifier.h"

using namespace mirage;

void test_rms_norm(int n, int d, bool use_symbolic_search = false) {
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
    for (auto const &op : ref_graph.operators) {
      op->fingerprint();
    }
  }
  mirage::cpu::CTensor ref_fp = ref_graph.operators.back()
                                    ->input_tensors[0]
                                    .copy_fingerprint_to_ctensor();
  kernel::Graph graph;
  kernel::DTensor X = graph.new_input(
      {8, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = graph.new_input(
      {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

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

  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(
      graph.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp));

  {
    search::ProfileResult result = search::profile(&graph);
    std::cout << "Profile result: " << result.run_time << std::endl;
    std::cout << "Error message: " << result.error_message << std::endl;
    std::ofstream ofs("rms_norm.cu");
    ofs << result.cuda_code;
    ofs.close();
  }

  search::AbstractExpr::symbolic_expr = use_symbolic_search;

  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  if (use_symbolic_search) {
    config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  }
  std::string checkpoint_file_name =
    "checkpoint_rms_"
    + std::to_string(n)
    + "_" + std::to_string(d)
    + (use_symbolic_search ? "_symbolic" : "")
    + ".json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  std::cout << "Generating kernel graphs..." << std::endl;
  if (use_symbolic_search) {
    gen.generate_kernel_graphs_symbolic();
  } else {
    gen.generate_kernel_graphs();
  }

  float best_time = std::numeric_limits<float>::max();
  {
    int i = 0;
    for (auto const &jgraph : gen.generated_graphs) {
      kernel::Graph g;
      from_json(jgraph, g);
      search::ProfileResult result = search::profile(&g);
      std::cout << "Profile result: " << result.run_time << std::endl;
      best_time = std::min(best_time, result.run_time);
      // write code to file
      if (result.error_message.empty()) {
        std::ofstream ofs("rms_norm_" + std::to_string(i) + ".cu");
        ofs << result.cuda_code;
        ofs.close();
      }
      std::cout << "Error message: " << result.error_message << std::endl;
      i++;
    }
  }
  std::cout << "Best time: " << best_time << std::endl;
}

void test_attention(int batch, int num_heads, int query_seq_len, int kv_seq_len, int head_dim, bool use_symbolic_search = false) {
  // Compute concatenated dimensions (num_heads and seq_len are concatenated)
  int g = num_heads * query_seq_len;  // query length (concatenated)
  int h = num_heads * kv_seq_len;     // key/value length (concatenated)
  
  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input({batch, g, head_dim},
                                        {(size_t)g * head_dim, (size_t)head_dim, 1},
                                        type::DT_FLOAT16,
                                        layout::DmemRowMajor);
    kernel::DTensor Kt = ref_graph.new_input({batch, head_dim, h},
                                        {(size_t)h * head_dim, (size_t)h, 1},
                                        type::DT_FLOAT16,
                                        layout::DmemRowMajor);
    kernel::DTensor V = ref_graph.new_input({batch, h, head_dim},
                                        {(size_t)h * head_dim, (size_t)head_dim, 1},
                                        type::DT_FLOAT16,
                                        layout::DmemRowMajor);
    kernel::DTensor A = ref_graph.matmul(Q, Kt);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    kernel::DTensor O = ref_graph.matmul(D, V);
    ref_graph.mark_output(O);
    for (auto const &op : ref_graph.operators) {
      op->fingerprint();
    }
  }
  // mirage::cpu::CTensor ref_fp = ref_graph.operators.back()
  //                                   ->input_tensors[0]
  //                                   .copy_fingerprint_to_ctensor();

  // kernel::Graph graph({1, 1, 1});
  // kernel::DTensor Q = graph.new_input({2, 256, 64},
  //                                     {16384, 64, 1},
  //                                     type::DT_FLOAT16,
  //                                     layout::DmemRowMajor);
  // kernel::DTensor K = graph.new_input({2, 64, 4096},
  //                                     {262144, 4096, 1},
  //                                     type::DT_FLOAT16,
  //                                     layout::DmemColumnMajor);
  // kernel::DTensor V = graph.new_input({2, 4096, 64},
  //                                     {262144, 64, 1},
  //                                     type::DT_FLOAT16,
  //                                     layout::DmemColumnMajor);
  // std::vector<kernel::DTensor> outputs;
  // {
  //   dim3 grid_dim = {2, 8, 8}, block_dim = {128, 1, 1};
  //   int forloop_range = 16, reduction_dimx = 64;
  //   threadblock::Graph bgraph(grid_dim, block_dim, forloop_range, reduction_dimx);
  //   threadblock::STensor bQ = bgraph.new_input(Q, {0, 1, -1}, -1, layout::SmemRowMajor);
  //   threadblock::STensor bK = bgraph.new_input(K, {0, -1, -1}, 2, layout::SmemColumnMajor);
  //   threadblock::STensor bV = bgraph.new_input(V, {0, -1, 2}, 1, layout::SmemRowMajor);
  //   threadblock::STensor bA = bgraph.matmul(bQ, bK);
  //   threadblock::STensor bE = bgraph.exp(bA);
  //   threadblock::STensor bS1 = bgraph.forloop_accum(bE, type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP);
  //   threadblock::STensor bB = bgraph.matmul(bE, bV);
  //   threadblock::STensor bS2 = bgraph.forloop_accum(bB, type::TB_FORLOOP_ACCUM_NO_RED_OP);
  //   threadblock::STensor bO = bgraph.div(bS2, bS1);
  //   bgraph.mark_output(bO, {0, 1, 2}, -1, type::TB_EPILOGUE_NONE);
  //   outputs = graph.customized({Q, K, V}, bgraph);
  //   assert(outputs.size() == 1);
  //   graph.mark_output(outputs[0]);
  // }
  // for (auto const &op : graph.operators) {
  //   op->fingerprint();
  // }
  // assert(
  //     graph.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp));
  
  search::AbstractExpr::symbolic_expr = use_symbolic_search;

  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  if (use_symbolic_search) {
    config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  } else {
    config.enable_attention_specific_optimization();
  }
  std::string checkpoint_file_name =
    "checkpoint_attention_batch"
    + std::to_string(batch)
    + "_heads" + std::to_string(num_heads)
    + "_qseq" + std::to_string(query_seq_len)
    + "_kvseq" + std::to_string(kv_seq_len)
    + "_dim" + std::to_string(head_dim)
    + (use_symbolic_search ? "_symbolic" : "")
    + ".json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  std::cout << "batch: " << batch << ", num_heads: " << num_heads << ", query_seq_len: " << query_seq_len << ", kv_seq_len: " << kv_seq_len << ", head_dim: " << head_dim << std::endl;
  if (use_symbolic_search) {
    gen.generate_kernel_graphs_symbolic();
    std::cout << "num verified symbolic graphs: " << gen.verified_symbolic_graphs.size() << std::endl;
  } else {
    gen.generate_kernel_graphs();
    std::cout << "num verified kernel graphs: " << gen.generated_graphs.size() << std::endl;
  }
  // {
  //   int i = 0;
  //   for (auto const &jgraph : gen.generated_graphs) {
  //     kernel::Graph g;
  //     from_json(jgraph, g);
  //     search::ProfileResult result = search::profile(&g);
  //     std::cout << "Profile result: " << result.run_time << std::endl;
  //     // write code to file
  //     if (result.error_message.empty()) {
  //       std::ofstream ofs("attention_" + std::to_string(i) + ".cu");
  //       ofs << result.cuda_code;
  //       ofs.close();
  //     }
  //     std::cout << "Error message: " << result.error_message << std::endl;
  //     i++;
  //   }
  // }
}

int main(int argc, char **argv) {
  bool use_symbolic_search = false;
  if (argc > 1) {
    if (std::string(argv[1]) == "-s") {
      use_symbolic_search = true;
    } else {
      use_symbolic_search = false;
    }
  }
  // test_rms_norm(8, 4096, use_symbolic_search);
  std::vector<int> batch_list{2, 8};
  std::vector<int> num_heads_list{8, 16};
  std::vector<int> query_seq_len_list{1, 8, 32};
  std::vector<int> kv_seq_len_list{128, 256, 512, 1024};
  std::vector<int> head_dim_list{64, 128};
  for (int batch : batch_list) {
    for (int num_heads : num_heads_list) {
      for (int query_seq_len : query_seq_len_list) {
        for (int kv_seq_len : kv_seq_len_list) {
          for (int head_dim : head_dim_list) {
            test_attention(batch, num_heads, query_seq_len, kv_seq_len, head_dim, use_symbolic_search);
          }
        }
      }
    }
  }
  return 0;
}
