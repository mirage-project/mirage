#include "mirage/kernel/graph.h"
#include "mirage/layout.h"
#include "mirage/search/profile.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"

using namespace mirage;

void test_rms_norm() {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {8, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W = ref_graph.new_input(
        {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
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

  // search::AbstractExpr::symbolic_expr = true;

  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  // config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  std::string checkpoint_file_name = "checkpoint_rms.json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  // gen.generate_kernel_graphs_symbolic();
  gen.generate_kernel_graphs();

  {
    int i = 0;
    for (auto const &jgraph : gen.generated_graphs) {
      kernel::Graph g;
      from_json(jgraph, g);
      search::ProfileResult result = search::profile(&g);
      std::cout << "Profile result: " << result.run_time << std::endl;
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
}

void test_attention(int n, int g, int h, int d) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input({n, g, d},
                                        {(size_t)g * d, (size_t)d, 1},
                                        type::DT_FLOAT16,
                                        layout::DmemRowMajor);
    kernel::DTensor Kt = ref_graph.new_input({n, d, h},
                                        {(size_t)h * d, (size_t)h, 1},
                                        type::DT_FLOAT16,
                                        layout::DmemRowMajor);
    kernel::DTensor V = ref_graph.new_input({n, h, d},
                                        {(size_t)h * d, (size_t)d, 1},
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
  mirage::cpu::CTensor ref_fp = ref_graph.operators.back()
                                    ->input_tensors[0]
                                    .copy_fingerprint_to_ctensor();

  kernel::Graph graph({1, 1, 1});
  kernel::DTensor Q = graph.new_input({2, 256, 64},
                                      {16384, 64, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemRowMajor);
  kernel::DTensor K = graph.new_input({2, 64, 4096},
                                      {262144, 4096, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemColumnMajor);
  kernel::DTensor V = graph.new_input({2, 4096, 64},
                                      {262144, 64, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemColumnMajor);
  std::vector<kernel::DTensor> outputs;
  {
    dim3 grid_dim = {2, 8, 8}, block_dim = {128, 1, 1};
    int forloop_range = 16, reduction_dimx = 64;
    threadblock::Graph bgraph(grid_dim, block_dim, forloop_range, reduction_dimx);
    threadblock::STensor bQ = bgraph.new_input(Q, {0, 1, -1}, -1, layout::SmemRowMajor);
    threadblock::STensor bK = bgraph.new_input(K, {0, -1, -1}, 2, layout::SmemColumnMajor);
    threadblock::STensor bV = bgraph.new_input(V, {0, -1, 2}, 1, layout::SmemRowMajor);
    threadblock::STensor bA = bgraph.matmul(bQ, bK);
    threadblock::STensor bE = bgraph.exp(bA);
    threadblock::STensor bS1 = bgraph.forloop_accum(bE, type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP);
    threadblock::STensor bB = bgraph.matmul(bE, bV);
    threadblock::STensor bS2 = bgraph.forloop_accum(bB, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    threadblock::STensor bO = bgraph.div(bS2, bS1);
    bgraph.mark_output(bO, {0, 1, 2}, -1, type::TB_EPILOGUE_NONE);
    outputs = graph.customized({Q, K, V}, bgraph);
    assert(outputs.size() == 1);
    graph.mark_output(outputs[0]);
  }
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(
      graph.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp));
  
  search::AbstractExpr::symbolic_expr = true;

  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  // config.enable_attention_specific_optimization();
  config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
  std::string checkpoint_file_name = "checkpoint_attention_n" + std::to_string(n) + "_g" + std::to_string(g) + "_h" + std::to_string(h) + "_d" + std::to_string(d) + ".json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  gen.generate_kernel_graphs_symbolic();
}

int main(int argc, char **argv) {
  // test_rms_norm();
  test_attention(2, 256, 4096, 64);
  return 0;
}
