#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {16, 8192}, {8192, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor A = ref_graph.new_input(
        {8192, 8}, {8, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor B = ref_graph.new_input(
        {8, 8192}, {8192, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor D = ref_graph.matmul(X, A);
    kernel::DTensor E = ref_graph.exp(D);
    ref_graph.matmul(E, B);
    // ref_graph.add(X, F);
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
                                    ->output_tensors[0]
                                    .copy_fingerprint_to_ctensor();

  kernel::Graph graph;
  kernel::DTensor X = graph.new_input(
      {16, 8192}, {8192, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor A = graph.new_input(
      {8192, 8}, {8, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor B = graph.new_input(
      {8, 8192}, {8192, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);

  std::vector<kernel::DTensor> outputs;
  {
    dim3 grid_dim = {32, 1, 1}, block_dim = {128, 1, 1};
    namespace tb = mirage::threadblock;
    tb::Graph bgraph(grid_dim, block_dim, 1, 8);
    tb::STensor bX = bgraph.new_input(X, {1, -1, -1}, -1, layout::SmemRowMajor);
    tb::STensor bA = bgraph.new_input(A, {0, -1, -1}, -1, layout::SmemRowMajor);
    tb::STensor bO = bgraph.matmul(bX, bA);
    bO = bgraph.forloop_accum(bO, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    outputs = graph.customized({X, A}, bgraph);
    assert(outputs.size() == 1);
  }
  {
    dim3 grid_dim = {64, 1, 1}, block_dim = {128, 1, 1};
    namespace tb = mirage::threadblock;
    tb::Graph bgraph(grid_dim, block_dim, 1, 8);
    tb::STensor bX =
        bgraph.new_input(outputs[0], {-1, -1, -1}, -1, layout::SmemRowMajor);
    tb::STensor bB = bgraph.new_input(B, {1, -1, -1}, -1, layout::SmemRowMajor);
    tb::STensor bR =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP);
    bB = bgraph.forloop_accum(bB, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    tb::STensor bE = bgraph.exp(bR);
    tb::STensor bO = bgraph.matmul(bE, bB);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    outputs = graph.customized({outputs[0], B}, bgraph);
    assert(outputs.size() == 1);
  }

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
  assert(
      graph.operators.back()->output_tensors[0].has_same_fingerprint(ref_fp));

  auto st = std::chrono::steady_clock::now();
  search::GeneratorConfig config =
      search::GeneratorConfig::get_mlp_default_config();
  config.fmap_to_explore = {-1};
  config.grid_dim_to_explore = {{32, 1, 1}, {64, 1, 1}};
  config.reduction_dimx = 8;
  search::KernelGraphGenerator gen(ref_graph, config, "checkpoint_mlp.json");
  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());

  return 0;
}
