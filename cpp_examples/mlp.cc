#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor X = ref_graph.new_input(
        {16, 8192}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor A = ref_graph.new_input(
        {8192, 8}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor B = ref_graph.new_input(
        {8, 8192}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor D = ref_graph.matmul(X, A);
    kernel::DTensor E = ref_graph.exp(D);
    ref_graph.matmul(E, B);
    //ref_graph.add(X, F);
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
  kernel::DTensor X = graph.new_input(
      {16, 8192}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor A = graph.new_input(
      {8192, 8}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor B = graph.new_input(
      {8, 8192}, type::DT_FLOAT16, layout::DmemColumnMajor);

  std::vector<kernel::DTensor> outputs;
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    plan.input_map.push_back({1, -1, -1});
    plan.input_map.push_back({0, -1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor,
        layout::SmemColumnMajor,
    };
    plan.output_map = {1, -1, -1};
    plan.forloop_dim = {-1, -1};
    plan.grid_dim = {32, 1, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 1;
    plan.reduction_dimx = 8;
    outputs = graph.customized({X, A}, plan);
    assert(outputs.size() == 1);
  }
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_REDUCTION_1_TO_DIMX_OP, {{0, 0}}});
    plan.ops.push_back({mirage::type::TB_EXP_OP, {{2, 0}}});
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{3, 0}, {1, 0}}});
    plan.input_map.push_back({-1, -1, -1});
    plan.input_map.push_back({1, -1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor,
        layout::SmemColumnMajor,
    };
    plan.output_map = {1, -1, -1};
    plan.forloop_dim = {-1, -1};
    plan.grid_dim = {64, 1, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 1;
    plan.reduction_dimx = 8;
    outputs = graph.customized({outputs[0], B}, plan);
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
  search::GeneratorConfig config = search::GeneratorConfig::get_mlp_default_config();
  config.fmap_to_explore = {-1};
  config.grid_dim_to_explore = {{32, 1, 1}, {64, 1, 1}};
  config.reduction_dimx = 8;
  search::KernelGraphGenerator gen(
      ref_graph,
      config,
      "checkpoint_mlp.json");
  gen.generate_kernel_graphs();

  clock_t et = clock();

  printf("Search time = %.4lfsec\n", (float)(et - st) / CLOCKS_PER_SEC);

  return 0;
}
