#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {8, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W = ref_graph.new_input(
        {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor D = ref_graph.rms_norm(X, {X.dim[1]});
    ref_graph.matmul(D, W);
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
      {8, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = graph.new_input(
      {4096, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

  {
    dim3 grid_dim = {64, 1, 1}, block_dim = {128, 1, 1};
    namespace tb = mirage::threadblock;
    tb::Graph bgraph(grid_dim, block_dim, 64, 64);
    tb::STensor bX = bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
    tb::STensor bW = bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
    tb::STensor bM = bgraph.matmul(bX, bW);
    tb::STensor bAccX =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    tb::STensor bAccM =
        bgraph.forloop_accum(bM, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    tb::STensor bO = bgraph.div(bAccM, bAccX);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outputs = graph.customized({X, W}, bgraph);
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
  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  std::string checkpoint_file_name = "checkpoint_rms.json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  gen.generate_kernel_graphs();

  return 0;
}
