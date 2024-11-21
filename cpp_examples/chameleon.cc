#include "common.h"
#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  // Currently only optimize for these two batch sizes
  int batch_size = miragetest::BATCH_SIZE;
  assert(batch_size == 1 || batch_size == 8);
  kernel::Graph ref_graph({1, 1, 1});
  {
    kernel::DTensor Q = ref_graph.new_input({2 * batch_size, 256, 64},
                                            {16384, 64, 1},
                                            type::DT_FLOAT16,
                                            layout::DmemRowMajor);
    kernel::DTensor K = ref_graph.new_input({2 * batch_size, 64, 4096},
                                            {262144, 4096, 1},
                                            type::DT_FLOAT16,
                                            layout::DmemColumnMajor);
    kernel::DTensor V = ref_graph.new_input({2 * batch_size, 4096, 64},
                                            {262144, 64, 1},
                                            type::DT_FLOAT16,
                                            layout::DmemColumnMajor);
    // Q = ref_graph.rms_norm(Q, {Q.dim[2]});
    V = ref_graph.rms_norm(V, {V.dim[2]});
    kernel::DTensor A = ref_graph.matmul(Q, K);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    kernel::DTensor O = ref_graph.matmul(D, V);
    ref_graph.mark_output(O);
    // ref_graph.all_reduce(O);
    for (auto const &op : ref_graph.operators) {
      op->fingerprint();
    }
    ProfileResult result;
    float total_runtime = 0.0f;
    for (auto const &op : ref_graph.operators) {
      op->profile(result);
      total_runtime = total_runtime + result.run_time;
    }
    printf("[cudnn kernel graph] Total runtime = %.4lfms\n", total_runtime);
  }
  mirage::cpu::CTensor ref_fp = ref_graph.operators.back()
                                    ->input_tensors[0]
                                    .copy_fingerprint_to_ctensor();
  kernel::Graph graph({1, 1, 1});
  kernel::DTensor Q = graph.new_input({2 * batch_size, 256, 64},
                                      {16384, 64, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemRowMajor);
  kernel::DTensor K = graph.new_input({2 * batch_size, 64, 4096},
                                      {262144, 4096, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemColumnMajor);
  kernel::DTensor V = graph.new_input({2 * batch_size, 4096, 64},
                                      {262144, 64, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemColumnMajor);
  std::vector<kernel::DTensor> outputs;
  {
    dim3 grid_dim = {2, 16, 4}, block_dim = {128, 1, 1};
    int forloop_range = 4, reduction_dimx = 64;
    if (batch_size > 1) {
      grid_dim = {16, 8, 2};
    }
    threadblock::Graph bgraph(
        grid_dim, block_dim, forloop_range, reduction_dimx);
    threadblock::STensor bQ =
        bgraph.new_input(Q, {0, -1, 1}, -1, layout::SmemRowMajor);
    threadblock::STensor bK =
        bgraph.new_input(K, {0, 2, -1}, 2, layout::SmemColumnMajor);
    threadblock::STensor bV =
        bgraph.new_input(V, {0, 1, -1}, 1, layout::SmemColumnMajor);
    bV = bgraph.rms_norm(bV);
    threadblock::STensor bA = bgraph.matmul(bQ, bK);
    threadblock::STensor bE = bgraph.exp(bA);
    threadblock::STensor bS = bgraph.matmul(bE, bV);
    threadblock::STensor bO1 =
        bgraph.forloop_accum(bS, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    threadblock::STensor bO2 =
        bgraph.forloop_accum(bE, type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP);
    bgraph.mark_output(bO1, {0, 2, 1}, -1, type::TB_EPILOGUE_NONE);
    bgraph.mark_output(bO2, {0, 2, 1}, -1, type::TB_EPILOGUE_NONE);
    outputs = graph.customized({Q, K, V}, bgraph);
    assert(outputs.size() == 2);
    // kernel::DTensor o1 = graph.reduction(outputs[0], 2 /*dim*/, 64 /*size*/);
    // kernel::DTensor o2 = graph.reduction(outputs[1], 2 /*dim*/);
    // graph.div(o1, o2);
  }
  {
    dim3 grid_dim = {2, 16, 1}, block_dim = {128, 1, 1};
    int forloop_range = 1, reduction_dimx = 64;
    if (batch_size > 1) {
      grid_dim = {16, 8, 1};
    }
    threadblock::Graph bgraph(
        grid_dim, block_dim, forloop_range, reduction_dimx);
    threadblock::STensor bA =
        bgraph.new_input(outputs[0], {0, 1, -1}, -1, layout::SmemRowMajor);
    threadblock::STensor bB =
        bgraph.new_input(outputs[1], {0, 1, -1}, -1, layout::SmemRowMajor);
    threadblock::STensor bRA =
        bgraph.forloop_accum(bA, type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP);
    threadblock::STensor bRB =
        bgraph.forloop_accum(bB, type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP);
    // threadblock::STensor bRA = bgraph.reduction_to_dimx(bA, 2);
    // threadblock::STensor bRB = bgraph.reduction(bB, 2);
    threadblock::STensor bD = bgraph.div(bRA, bRB);
    threadblock::STensor bAcc = bD;
    // threadblock::STensor bAcc = bgraph.forloop_accum(bD,
    // type::TB_FORLOOP_ACCUM_NO_RED_OP);
    bgraph.mark_output(bAcc, {0, 1, -1}, -1, type::TB_EPILOGUE_NONE);
    outputs = graph.customized({outputs[0], outputs[1]}, bgraph);
    assert(outputs.size() == 1);
  }
  graph.mark_output(outputs[0]);
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  // assert(ref_graph.operators.back()->output_tensors[0].has_same_fingerprint(
  //     graph.operators.back()->output_tensors[0]));
  assert(graph.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp));

  ProfileResult result;
  float total_ms = 0.0f;
  for (auto const &op : graph.operators) {
    op->profile(result);
    total_ms = total_ms + result.run_time;
  }
  printf("[2 Block Graphs] Total runtime = %.4lfms\n", total_ms);

  auto st = std::chrono::steady_clock::now();
  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  config.enable_attention_specific_optimization();
  std::string results_filename =
      "results_chameleon_bs" + std::to_string(batch_size) + ".json";
  search::KernelGraphGenerator gen(ref_graph, config, results_filename.data());
  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());
  return 0;
}
