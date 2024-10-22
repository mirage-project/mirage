#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {16, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W1 = ref_graph.new_input(
        {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor W3 = ref_graph.new_input(
        {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor D1 = ref_graph.matmul(X, W1);
    kernel::DTensor D2 = ref_graph.matmul(X, W3);
    D1 = ref_graph.silu(D1);
    ref_graph.mul(D1, D2);
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
      {16, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W1 = graph.new_input(
      {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor W3 = graph.new_input(
      {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemColumnMajor);
  {
    namespace tb = mirage::threadblock;
    dim3 grid_dim = {64, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 64, 64);
    tb::STensor bX = bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
    tb::STensor bW1 =
        bgraph.new_input(W1, {1, -1, -1}, 0, layout::SmemRowMajor);
    tb::STensor bW3 =
        bgraph.new_input(W3, {1, -1, -1}, 0, layout::SmemRowMajor);
    tb::STensor bD1 = bgraph.matmul(bX, bW1);
    tb::STensor bD2 = bgraph.matmul(bX, bW3);
    bD1 = bgraph.forloop_accum(bD1, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    bD2 = bgraph.forloop_accum(bD2, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    bD1 = bgraph.silu(bD1);
    tb::STensor bO = bgraph.mul(bD1, bD2);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outputs =
        graph.customized({X, W1, W3}, bgraph);
    assert(outputs.size() == 1);
  }

  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(
      graph.operators.back()->output_tensors[0].has_same_fingerprint(ref_fp));

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
  std::string checkpoint_file_name = "checkpoint_gated_mlp.json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());

  return 0;
}
