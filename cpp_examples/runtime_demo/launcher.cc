#include "mirage/kernel/graph.h"
#include "mirage/runtime/runtime.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;

int main(int argc, char **argv) {
  namespace tb = mirage::threadblock;
  namespace kn = mirage::kernel;
  std::unordered_map<kn::KNCustomizedOp const *, mirage::runtime::TaskType>
      task_types;

  kn::Graph kgraph;
  kn::DTensor X = kgraph.new_input(
      {1, 4096}, {4096, 1}, type::DT_BFLOAT16, layout::DmemRowMajor);
  for (int layer = 0; layer < 1; layer++) {
    // Add RMSLinear
    {
      dim3 grid_dim = {1, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 64, 64);
      kn::DTensor W = kgraph.new_input(
          {4096, 64}, {4096, 1}, type::DT_BFLOAT16, layout::DmemRowMajor);
      tb::STensor bX =
          bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
      tb::STensor bW =
          bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      tb::STensor bM = bgraph.matmul(bX, bW);
      tb::STensor bAccX =
          bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
      tb::STensor bAccM =
          bgraph.forloop_accum(bM, type::TB_FORLOOP_ACCUM_NO_RED_OP);
      tb::STensor bO = bgraph.div(bAccM, bAccX);
      bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
      std::vector<kernel::DTensor> outputs = kgraph.customized({X, W}, bgraph);
      X = outputs[0];
      task_types[static_cast<kn::KNCustomizedOp *>(X.owner_op)] =
          mirage::runtime::TASK_RMS_NORM_LINEAR;
    }
#ifdef DEADCODE
    // Add elementwise
    {
      dim3 grid_dim = {64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 64, 64);
      kn::DTensor Y = kgraph.new_input(
          {1, 4096}, {4096, 1}, type::DT_BFLOAT16, layout::DmemRowMajor);
      tb::STensor bX =
          bgraph.new_input(X, {1, -1, -1}, 1, layout::SmemRowMajor);
      tb::STensor bY =
          bgraph.new_input(Y, {1, -1, -1}, 1, layout::SmemRowMajor);
      tb::STensor bM = bgraph.add(bX, bY);
      tb::STensor bAccM =
          bgraph.forloop_accum(bM, type::TB_FORLOOP_ACCUM_NO_RED_OP);
      bgraph.mark_output(bAccM, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
      std::vector<kernel::DTensor> outputs = kgraph.customized({X, Y}, bgraph);
      X = outputs[0];
    }
#endif
  }

  // Start runtime
  using namespace mirage::runtime;
  Runtime runtime;
  runtime.register_mugraph(kgraph, task_types);
  runtime.launch_persistent_kernel(106 /*num_workers*/, 8 /*num_schedulers*/);
}
