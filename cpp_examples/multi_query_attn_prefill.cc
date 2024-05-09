#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/search/search.h"
#include "common.h"

using namespace mirage;

int main(int argc, char **argv) {
  // Currently only optimize for these two batch sizes
  int batch_size = miragetest::BATCH_SIZE;
  assert(batch_size == 1 || batch_size == 8);
  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input(
        {batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor K = ref_graph.new_input(
        {batch_size, 64, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor V = ref_graph.new_input(
        {batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.matmul(Q, K);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    ref_graph.matmul(D, V);
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
  kernel::DTensor Q =
      graph.new_input({batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor K =
      graph.new_input({batch_size, 64, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor V =
      graph.new_input({batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
  std::vector<kernel::DTensor> outputs;
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    // plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{3, 0}, {2, 0}}});
    plan.ops.push_back({mirage::type::TB_EXP_OP, {{3, 0}}});
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{4, 0}, {2, 0}}});
    plan.ops.push_back({mirage::type::TB_REDUCTION_2_OP, {{4, 0}}});
    plan.input_map.push_back({0, -1, 1});
    plan.input_map.push_back({0, 2, -1});
    plan.input_map.push_back({0, 1, -1});
    plan.input_smem_layouts = {
        // layout::SmemRowMajor,
        // layout::SmemColumnMajor,
        // layout::SmemColumnMajor,
        layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
        layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
        layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
    };
    plan.output_map = {0, 2, 1};
    plan.forloop_dim = {-1, 2, 1};
    if (batch_size == 1) {
      plan.grid_dim = {1, 16, 16};
    } else {
      plan.grid_dim = {8, 8, 16};
    }
    plan.block_dim = {128, 1, 1};
    if (batch_size == 1) {
      plan.forloop_range = 4;
    } else {
      plan.forloop_range = 8;
    }
    plan.reduction_dimx = 64;
    outputs = graph.customized({Q, K, V}, plan);
    assert(outputs.size() == 2);
    // kernel::DTensor o1 = graph.reduction(outputs[0], 2 /*dim*/, 64 /*size*/);
    // kernel::DTensor o2 = graph.reduction(outputs[1], 2 /*dim*/);
    // graph.div(o1, o2);
  }
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_REDUCTION_2_TO_DIMX_OP, {{0, 0}}});
    plan.ops.push_back({mirage::type::TB_REDUCTION_2_OP, {{1, 0}}});
    plan.ops.push_back({mirage::type::TB_DIV_OP, {{2, 0}, {3, 0}}});
    plan.input_map.push_back({0, 1, -1});
    plan.input_map.push_back({0, 1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor,
        layout::SmemRowMajor,
    };
    plan.output_map = {0, 1, -1};
    plan.forloop_dim = {-1, -1};
    plan.grid_dim = {1, 128, 1};
    if (batch_size == 8) {
      plan.grid_dim = {8, 64, 1};
    }
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 1;
    plan.reduction_dimx = 64;
    outputs = graph.customized({outputs[0], outputs[1]}, plan);
    assert(outputs.size() == 1);
  }
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(ref_graph.operators.back()->output_tensors[0].has_same_fingerprint(
      graph.operators.back()->output_tensors[0]));
  ProfileResult result;
  float total_ms = 0.0f;
  for (auto const &op : graph.operators) {
    op->profile(result);
    total_ms = total_ms + result.run_time;
  }
  printf("[2 Block Graphs] Total runtime = %.4lfms\n", total_ms);

  clock_t st = clock();
  search::GeneratorConfig config = search::GeneratorConfig::get_attention_default_config();
  config.grid_dim_to_explore = {{batch_size, 16, 16}, {batch_size, 8, 16}, {batch_size, 64, 1}, {batch_size, 128, 1}};
  std::string checkpoint_file_name = "checkpoint_multi_query_attn_prefill_bs" +
                                     std::to_string(batch_size) + ".json";
  search::KernelGraphGenerator gen(
      ref_graph,
      config,
      checkpoint_file_name.data());
  gen.generate_kernel_graphs();

  clock_t et = clock();

  printf("Search time = %.4lfsec\n", (float)(et - st) / CLOCKS_PER_SEC);

  return 0;
}
