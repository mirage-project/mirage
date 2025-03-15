#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {16, 256}, {256, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W = ref_graph.new_input(
        {256, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.new_input(
        {256, 16}, {16, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor B = ref_graph.new_input(
        {16, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor D = ref_graph.matmul(X, A);
    kernel::DTensor E = ref_graph.matmul(D, B);
    kernel::DTensor C = ref_graph.matmul(X, W);
    kernel::DTensor O = ref_graph.add(C, E);
    ref_graph.mark_output(O);
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

  kernel::Graph graph;
  kernel::DTensor X = graph.new_input(
      {16, 256}, {256, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = graph.new_input(
      {256, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor A = graph.new_input(
      {256, 16}, {16, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor B = graph.new_input(
      {16, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);

  std::vector<kernel::DTensor> outputs;
  {
    dim3 grid_dim = {128, 1, 1}, block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim, block_dim, 2, 64);
    threadblock::STensor bX =
        bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor bW =
        bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
    threadblock::STensor bA =
        bgraph.new_input(A, {-1, -1, -1}, 0, layout::SmemRowMajor);
    threadblock::STensor bB =
        bgraph.new_input(B, {1, -1, -1}, -1, layout::SmemRowMajor);
    threadblock::STensor bD = bgraph.matmul(bX, bA);
    threadblock::STensor bC = bgraph.concat(bX, bD, 1 /*dim*/);
    threadblock::STensor bE = bgraph.concat(bW, bB, 0 /*dim*/);
    threadblock::STensor bO = bgraph.matmul(bC, bE);
    bO = bgraph.forloop_accum(bO, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    outputs = graph.customized({X, W, A, B}, bgraph);
    assert(outputs.size() == 1);
  }
  graph.mark_output(outputs[0]);
  ProfileResult result;
  float total_ms = 0.0f;
  for (auto const &op : graph.operators) {
    op->profile(result);
    total_ms = total_ms + result.run_time;
  }
  printf("[2 Block Graphs] Total runtime = %.4lfms\n", total_ms);
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(graph.operators.back()->input_tensors[0].has_same_fingerprint(ref_fp));
  auto st = std::chrono::steady_clock::now();
  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  config.enable_concat_matmul_transformation();
  search::KernelGraphGenerator gen(ref_graph, config, "checkpoint_lora.json");
  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());

  return 0;
}
