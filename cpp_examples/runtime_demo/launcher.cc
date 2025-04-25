#include "mirage/kernel/graph.h"
#include "mirage/runtime/runtime.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

#include "nvshmem.h"
#include "nvshmemx.h"
#include <mpi.h>

using namespace mirage;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  namespace tb = mirage::threadblock;
  namespace kn = mirage::kernel;
  namespace rt = mirage::runtime;
  int batch_size = 1;
  int max_kv_length = 4096;
  int hidden_size = 3584;
  int num_kv_heads = 4;
  int num_q_heads = 28;
  int num_kv_splits = 16;
  int head_dim = hidden_size / num_q_heads;
  int fused_outdim_1 = head_dim * (num_q_heads + 2 * num_kv_heads);
  int fused_outdim_2 = 2 * hidden_size;

  std::unordered_map<kn::KNOperator const *, std::tuple<int, int, rt::TaskType>>
      task_configs;

  kn::Graph kgraph(dim3{1, 1, 1}, true /*disable_fingerprint*/);
  kn::DTensor X = kgraph.new_input(
      {batch_size, 1}, {1, 1}, type::DT_UINT16, layout::DmemRowMajor);
  kn::DTensor AllReduceBuf =
      kgraph.new_input({batch_size, world_size, hidden_size},
                       {hidden_size * world_size, hidden_size, 1},
                       type::DT_BFLOAT16,
                       layout::DmemRowMajor);

  kn::DTensor Y = kgraph.new_input({batch_size, hidden_size},
                                   {(size_t)hidden_size, 1},
                                   type::DT_BFLOAT16,
                                   layout::DmemRowMajor);
  kn::DTensor AttnIn = kgraph.new_input({batch_size, fused_outdim_1},
                                        {(size_t)fused_outdim_1, 1},
                                        type::DT_BFLOAT16,
                                        layout::DmemRowMajor);
  kn::DTensor MLPMid = kgraph.new_input({batch_size, fused_outdim_2},
                                        {(size_t)fused_outdim_2, 1},
                                        type::DT_BFLOAT16,
                                        layout::DmemRowMajor);
  kn::DTensor MLPOut = kgraph.new_input({batch_size, hidden_size},
                                        {(size_t)hidden_size, 1},
                                        type::DT_BFLOAT16,
                                        layout::DmemRowMajor);
  kn::DTensor AttnOut =
      kgraph.new_input({batch_size, num_kv_splits, num_q_heads * head_dim},
                       {(size_t)num_q_heads * num_kv_splits * head_dim,
                        (size_t)num_kv_splits * head_dim,
                        1},
                       type::DT_BFLOAT16,
                       layout::DmemRowMajor);
  kn::DTensor AttnProjIn = kgraph.new_input({batch_size, hidden_size},
                                            {(size_t)hidden_size, 1},
                                            type::DT_BFLOAT16,
                                            layout::DmemRowMajor);
  kn::DTensor AttnProjOut = kgraph.new_input({batch_size, hidden_size},
                                             {(size_t)hidden_size, 1},
                                             type::DT_BFLOAT16,
                                             layout::DmemRowMajor);
  kn::DTensor AttnAROut = kgraph.new_input({batch_size, hidden_size},
                                           {(size_t)hidden_size, 1},
                                           type::DT_BFLOAT16,
                                           layout::DmemRowMajor);
  // Add Embed
  {
    dim3 grid_dim = {1, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 1, 64);
    kn::DTensor W = kgraph.new_input({32 * 1024, hidden_size},
                                     {(size_t)hidden_size, 1},
                                     type::DT_BFLOAT16,
                                     layout::DmemRowMajor);
    bgraph.new_input(X, {-1, -1, -1}, -1, layout::SmemRowMajor);
    bgraph.new_input(
        W, {-1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    bgraph.new_input(Y, {-1, -1, -1}, -1, layout::SmemRowMajor);
    kgraph.customized({X, W, Y}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(2, 1, rt::TASK_EMBEDDING);
  }
  X = Y;
  for (int layer = 0; layer < 28; layer++) {
    // Add RMS + MatMul
    {
      dim3 grid_dim = {(size_t)fused_outdim_1 / 64, 1, 1},
           block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
      kn::DTensor W = kgraph.new_input({hidden_size, fused_outdim_1},
                                       {(size_t)fused_outdim_2, 1},

                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(AttnIn, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({X, W, AttnIn}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_RMS_NORM_LINEAR);
    }
    // Add attention_1
    {

      kn::DTensor K = kgraph.new_input(
          {batch_size, max_kv_length, num_kv_heads, head_dim},
          {(size_t)num_kv_heads * head_dim, (size_t)head_dim, 1},
          type::DT_BFLOAT16,
          layout::DmemRowMajor);
      kn::DTensor V = kgraph.new_input(
          {batch_size, max_kv_length, num_kv_heads, head_dim},
          {(size_t)num_kv_heads * head_dim, (size_t)head_dim, 1},
          type::DT_BFLOAT16,
          layout::DmemRowMajor);
      dim3 grid_dim = {batch_size, num_kv_heads, num_kv_splits},
           block_dim = {128, 1, 1};
      tb::Graph bgraph(
          grid_dim, block_dim, max_kv_length / num_kv_splits / 64, 64);
      // Note that QKV is concatenated together
      bgraph.new_input(AttnIn, {0, 1, -1}, -1, layout::SmemRowMajor);
      bgraph.new_input(K, {0, 2, 1}, 1, layout::SmemRowMajor);
      bgraph.new_input(V, {0, 2, 1}, 1, layout::SmemRowMajor);
      bgraph.new_input(AttnOut, {0, 2, 1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnIn, K, V, AttnOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_ATTENTION_1);
    }
    // Add attention_2
    {
      dim3 grid_dim = {batch_size, num_q_heads, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1, 64);
      bgraph.new_input(AttnOut, {0, 2, -1}, -1, layout::SmemRowMajor);
      bgraph.new_input(AttnProjIn, {0, 1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnOut, AttnProjIn}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(1, 1, rt::TASK_ATTENTION_2);
    }
    // Add out_projection
    {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
      kn::DTensor W = kgraph.new_input({hidden_size, hidden_size},
                                       {(size_t)hidden_size, 1},
                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      bgraph.new_input(AttnProjIn, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(AttnProjOut, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnProjIn, W, AttnProjOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_MATMUL);
    }
    // Add AllReduce
    if (world_size > 1) {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1, 64);
      bgraph.new_input(AttnProjOut, {1, -1, -1}, -1, layout::SmemRowMajor);
      bgraph.new_input(AllReduceBuf, {2, -1, -1}, -1, layout::SmemRowMajor);
      bgraph.new_input(AttnAROut, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnProjOut, AllReduceBuf, AttnAROut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_ALLREDUCE);
    } else {
      AttnAROut = AttnProjOut;
    }
    // Add RMS + Matmul
    {
      dim3 grid_dim = {fused_outdim_2 / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
      kn::DTensor W = kgraph.new_input({hidden_size, fused_outdim_2},
                                       {(size_t)fused_outdim_2, 1},
                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      bgraph.new_input(AttnAROut, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(MLPMid, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnAROut, W, MLPMid}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_RMS_NORM_LINEAR);
    }
    // silu + Matmul
    {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
      kn::DTensor W = kgraph.new_input({hidden_size, hidden_size},
                                       {(size_t)hidden_size, 1},
                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      bgraph.new_input(MLPMid, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(MLPOut, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({MLPMid, W, MLPOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_SILU_MUL_LINEAR);
    }
    X = MLPOut;
  }

  // Start runtime
  using namespace mirage::runtime;
  Runtime runtime(world_size /*num_gpus*/, rank /*my_gpu_id*/);
  runtime.register_mugraph(kgraph, task_configs);
  runtime.sanity_check();
  runtime.launch_persistent_kernel(106 /*num_workers*/,
                                   6 /*num_local_schedulers*/,
                                   2 /*num_remote_schedulers*/);

  MPI_Finalize();
}
