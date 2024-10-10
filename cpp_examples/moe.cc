#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {8, 8, 4096}, {32768, 4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor A = ref_graph.new_input({8, 4096, 4096},
                                            {16777216, 4096, 1},
                                            type::DT_FLOAT16,
                                            layout::DmemColumnMajor);
    kernel::DTensor B = ref_graph.new_input({8, 4096, 4096},
                                            {16777216, 4096, 1},
                                            type::DT_FLOAT16,
                                            layout::DmemColumnMajor);
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
      {8, 8, 4096}, {32768, 4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor A = graph.new_input({8, 4096, 4096},
                                      {16777216, 4096, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemColumnMajor);
  kernel::DTensor B = graph.new_input({8, 4096, 4096},
                                      {16777216, 4096, 1},
                                      type::DT_FLOAT16,
                                      layout::DmemColumnMajor);

  kernel::DTensor output = graph.matmul(X, A);
  {
    namespace tb = mirage::threadblock;
    dim3 grid_dim = {8, 32, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 16, 64);
    tb::STensor bX =
        bgraph.new_input(output, {0, -1, -1}, 2, layout::SmemRowMajor);
    tb::STensor bB =
        bgraph.new_input(B, {0, 2, -1}, 1, layout::SmemColumnMajor);
    tb::STensor bE = bgraph.exp(bX);
    tb::STensor bO = bgraph.matmul(bE, bB);
    bO = bgraph.forloop_accum(bO, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    bgraph.mark_output(bO, {0, 2, -1}, -1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outputs =
        graph.customized({output, B}, bgraph);
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
  search::GeneratorConfig config;
  config.knop_to_explore = {
      type::KN_MATMUL_OP, type::KN_EXP_OP,
      // type::KN_CUSTOMIZED_OP,
  };
  config.tbop_to_explore = {
      type::TB_MATMUL_OP,
      type::TB_EXP_OP,
      type::TB_REDUCTION_1_TO_DIMX_OP,
      type::TB_REDUCTION_2_TO_DIMX_OP,
  };
  config.imap_to_explore = {{
      {0, -1, -1},
      {0, 2, -1},
      {0, 1, -1},
  }};
  config.omap_to_explore = {
      {0, -1, -1},
      {0, 1, -1},
      {0, 2, -1},
  };
  config.grid_dim_to_explore = {{8, 4, 1}, {8, 16, 1}, {8, 32, 1}};
  config.block_dim_to_explore = {{128, 1, 1}};
  config.fmap_to_explore = {-1, 1, 2};
  config.frange_to_explore = {4, 8, 16, 32};
  config.reduction_dimx = 64;
  search::KernelGraphGenerator gen(ref_graph, config, "checkpoint_moe.json");
  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());

  return 0;
}
