#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X =
        ref_graph.new_input({16, 256}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor W = ref_graph.new_input(
        {256, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.new_input(
        {256, 16}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor B = ref_graph.new_input(
        {16, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor D = ref_graph.matmul(X, A);
    kernel::DTensor E = ref_graph.matmul(D, B);
    kernel::DTensor C = ref_graph.matmul(X, W);
    ref_graph.add(C, E);
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
  kernel::Graph graph;
  kernel::DTensor X =
      graph.new_input({16, 256}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W =
      graph.new_input({256, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor A =
      graph.new_input({256, 16}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor B =
      graph.new_input({16, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);

  std::vector<kernel::DTensor> outputs;
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{0, 0}, {2, 0}}});
    plan.ops.push_back({mirage::type::TB_CONCAT_1_OP, {{0, 0}, {4, 0}}});
    plan.ops.push_back({mirage::type::TB_CONCAT_0_OP, {{1, 0}, {3, 0}}});
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{5, 0}, {6, 0}}});
    plan.input_map.push_back({-1, -1, -1});
    plan.input_map.push_back({1, -1, -1});
    plan.input_map.push_back({-1, -1, -1});
    plan.input_map.push_back({1, -1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor,
        layout::SmemColumnMajor,
        layout::SmemColumnMajor,
        layout::SmemColumnMajor,
    };
    plan.output_map = {1, -1, -1};
    plan.forloop_dim = {1, 0, 0, -1};
    plan.grid_dim = {128, 1, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 2;
    plan.reduction_dimx = 64;
    outputs = graph.customized({X, W, A, B}, plan);
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
  assert(ref_graph.operators.back()->output_tensors[0].has_same_fingerprint(
      graph.operators.back()->output_tensors[0]));

  clock_t st = clock();
  search::GeneratorConfig config =
      search::GeneratorConfig::get_lora_default_config();
  search::KernelGraphGenerator gen(ref_graph, config, "checkpoint_lora.json");
  gen.generate_kernel_graphs();

  clock_t et = clock();

  printf("Search time = %.4lfsec\n", (float)(et - st) / CLOCKS_PER_SEC);

  return 0;
}
