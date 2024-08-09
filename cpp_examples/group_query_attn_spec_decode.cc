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
    kernel::DTensor Q = ref_graph.new_input(
        {2 * batch_size, 256, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor K = ref_graph.new_input(
        {2 * batch_size, 64, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor V = ref_graph.new_input(
        {2 * batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.matmul(Q, K);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    kernel::DTensor O = ref_graph.matmul(D, V);
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
                                    ->output_tensors[0]
                                    .copy_fingerprint_to_ctensor();
  kernel::Graph graph({1, 1, 1});
  kernel::DTensor Q = graph.new_input(
      {2 * batch_size, 256, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor K = graph.new_input(
      {2 * batch_size, 64, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor V = graph.new_input(
      {2 * batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
  std::vector<kernel::DTensor> outputs;
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    // plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{3, 0}, {2, 0}}});
    plan.ops.push_back({mirage::type::TB_EXP_OP, {{3, 0}}});
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{4, 0}, {2, 0}}});
    plan.ops.push_back({mirage::type::TB_REDUCTION_2_OP, {{4, 0}}});
    plan.ops.push_back({mirage::type::TB_FORLOOP_ACCUM_OP, {{5, 0}}});
    plan.ops.push_back({mirage::type::TB_FORLOOP_ACCUM_OP, {{6, 0}}});
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
    plan.input_forloop_dim = {-1, 2, 1};
    plan.output_map = {0, 2, 1};
    if (batch_size == 1) {
      plan.grid_dim = {2, 16, 4};
    } else {
      plan.grid_dim = {16, 8, 2};
    }
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 4;
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
    plan.ops.push_back({mirage::type::TB_FORLOOP_ACCUM_OP, {{4, 0}}});
    plan.input_map.push_back({0, 1, -1});
    plan.input_map.push_back({0, 1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor,
        layout::SmemRowMajor,
    };
    plan.input_forloop_dim = {-1, -1};
    plan.output_map = {0, 1, -1};
    plan.grid_dim = {2, 16, 1};
    if (batch_size == 8) {
      plan.grid_dim = {16, 8, 1};
    }
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 1;
    plan.reduction_dimx = 64;
    // plan.output_epilogue = mirage::type::TB_EPILOGUE_ALLREDUCE;
    outputs = graph.customized({outputs[0], outputs[1]}, plan);
    assert(outputs.size() == 1);
  }
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  // assert(ref_graph.operators.back()->output_tensors[0].has_same_fingerprint(
  //     graph.operators.back()->output_tensors[0]));
  assert(
      graph.operators.back()->output_tensors[0].has_same_fingerprint(ref_fp));

  ProfileResult result;
  float total_ms = 0.0f;
  for (auto const &op : graph.operators) {
    op->profile(result);
    total_ms = total_ms + result.run_time;
  }
  printf("[2 Block Graphs] Total runtime = %.4lfms\n", total_ms);

  graph.generate_cuda_program("test.cu");
  // return 0;

  auto st = std::chrono::steady_clock::now();
  search::GeneratorConfig config =
      search::GeneratorConfig::get_attention_default_config();
  config.grid_dim_to_explore = {{2 * batch_size, 16, 4},
                                {2 * batch_size, 8, 2},
                                {2 * batch_size, 16, 1},
                                {2 * batch_size, 8, 1}};
  std::string checkpoint_file_name =
      "checkpoint_group_query_attn_spec_decode_bs" +
      std::to_string(batch_size) + ".json";
  search::KernelGraphGenerator gen(
      ref_graph, config, checkpoint_file_name.data());
  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());
  return 0;
}
