#include "mirage/kernel/graph.h"
#include "mirage/runtime/runtime.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

#include "nvshmem.h"
#include "nvshmemx.h"
#include <fstream>
#include <iostream>
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
  namespace tp = mirage::type;
  using mirage::runtime::IODesc;
  int vocab_size = 32 * 1024;
  int batch_size = 1;
  int max_kv_length = 4096;
  int hidden_size = 3584;
  int num_kv_heads = 4;
  int num_q_heads = 28;
  int num_kv_splits = 1;
  int head_dim = hidden_size / num_q_heads;
  int fused_outdim_1 = head_dim * (num_q_heads + 2 * num_kv_heads);
  int fused_outdim_2 = 2 * hidden_size;

  std::unordered_map<kn::KNOperator const *, std::tuple<int, int, rt::TaskType>>
      task_configs;
  std::map<tp::GuidType, rt::IODesc> io_configs;

  kn::Graph kgraph(dim3{1, 1, 1}, true /*disable_fingerprint*/);
  kn::DTensor X = kgraph.new_input(
      {batch_size, 1}, {1, 1}, type::DT_UINT16, layout::DmemRowMajor);
  io_configs.emplace(X.guid, IODesc(rt::IODesc::TorchTensor, "input_token", X));

  kn::DTensor AllReduceBuf =
      kgraph.new_input({batch_size, world_size, hidden_size},
                       {hidden_size * world_size, hidden_size, 1},
                       type::DT_BFLOAT16,
                       layout::DmemRowMajor);
  io_configs.emplace(
      AllReduceBuf.guid,
      IODesc(rt::IODesc::NVSHMEMMallocTensor, "all_reduce_buf", AllReduceBuf));

  kn::DTensor Y = kgraph.new_input({batch_size, hidden_size},
                                   {(size_t)hidden_size, 1},
                                   type::DT_BFLOAT16,
                                   layout::DmemRowMajor);
  io_configs.emplace(Y.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "embed_out", Y));
  kn::DTensor AttnIn = kgraph.new_input({batch_size, fused_outdim_1},
                                        {(size_t)fused_outdim_1, 1},
                                        type::DT_BFLOAT16,
                                        layout::DmemRowMajor);
  io_configs.emplace(AttnIn.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "attn_in", AttnIn));
  kn::DTensor MLPMid = kgraph.new_input({batch_size, fused_outdim_2},
                                        {(size_t)fused_outdim_2, 1},
                                        type::DT_BFLOAT16,
                                        layout::DmemRowMajor);
  io_configs.emplace(MLPMid.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "mlp_mid", MLPMid));
  kn::DTensor MLPOut = kgraph.new_input({batch_size, hidden_size},
                                        {(size_t)hidden_size, 1},
                                        type::DT_BFLOAT16,
                                        layout::DmemRowMajor);
  io_configs.emplace(MLPOut.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "mlp_out", MLPOut));
  kn::DTensor AttnOut =
      kgraph.new_input({batch_size, num_q_heads * head_dim},
                       {(size_t)num_q_heads * head_dim, (size_t)head_dim, 1},
                       type::DT_BFLOAT16,
                       layout::DmemRowMajor);
  io_configs.emplace(AttnOut.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "attn_out", MLPOut));
  kn::DTensor AttnProjOut = kgraph.new_input({batch_size, hidden_size},
                                             {(size_t)hidden_size, 1},
                                             type::DT_BFLOAT16,
                                             layout::DmemRowMajor);
  io_configs.emplace(
      AttnProjOut.guid,
      IODesc(rt::IODesc::NVSHMEMMallocTensor, "attn_proj_out", AttnProjOut));
  kn::DTensor AttnAROut = kgraph.new_input({batch_size, hidden_size},
                                           {(size_t)hidden_size, 1},
                                           type::DT_BFLOAT16,
                                           layout::DmemRowMajor);
  io_configs.emplace(
      AttnAROut.guid,
      IODesc(rt::IODesc::NVSHMEMMallocTensor, "attn_allreduce_out", AttnAROut));
  kn::DTensor ArgmaxIn = kgraph.new_input({batch_size, vocab_size},
                                          {(size_t)vocab_size, 1},
                                          type::DT_BFLOAT16,
                                          layout::DmemRowMajor);
  io_configs.emplace(
      ArgmaxIn.guid,
      IODesc(rt::IODesc::CUDAMallocTensor, "argmax_in", ArgmaxIn));
  kn::DTensor ArgmaxOut = kgraph.new_input(
      {batch_size, 1}, {(size_t)1, 1}, type::DT_BFLOAT16, layout::DmemRowMajor);
  io_configs.emplace(
      ArgmaxOut.guid,
      IODesc(rt::IODesc::CUDAMallocTensor, "argmax_out", ArgmaxOut));
  // Add Embed
  {
    dim3 grid_dim = {1, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 1, 64);
    kn::DTensor W = kgraph.new_input({vocab_size, hidden_size},
                                     {(size_t)hidden_size, 1},
                                     type::DT_BFLOAT16,
                                     layout::DmemRowMajor);
    io_configs.emplace(W.guid,
                       IODesc(rt::IODesc::TorchTensor, "embed_tokens", W));
    bgraph.new_input(X, {-1, -1, -1}, -1, layout::SmemRowMajor);
    bgraph.new_input(
        W, {-1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    bgraph.new_input(Y, {-1, -1, -1}, -1, layout::SmemRowMajor);
    kgraph.customized({X, W, Y}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(2, 1, rt::TASK_EMBEDDING);
  }
  X = Y;
  for (int layer = 0; layer < 1; layer++) {
    // Add RMS + MatMul
    {
      dim3 grid_dim = {(size_t)fused_outdim_1 / 64, 1, 1},
           block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
      kn::DTensor W = kgraph.new_input({hidden_size, fused_outdim_1},
                                       {(size_t)fused_outdim_2, 1},
                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      IODesc desc(rt::IODesc::FusedTorchTensor,
                  "layer_" + std::to_string(layer) + "_qkv_proj",
                  W);
      IODesc q_proj(rt::IODesc::TorchTensor,
                    "layer_" + std::to_string(layer) + "_q_proj",
                    W);
      q_proj.tensor.dim[1] = num_q_heads * head_dim;
      q_proj.tensor.stride[0] = q_proj.tensor.dim[1];
      desc.sub_descs.push_back(q_proj);
      IODesc k_proj(rt::IODesc::TorchTensor,
                    "layer_" + std::to_string(layer) + "_k_proj",
                    W);
      k_proj.tensor.dim[1] = num_kv_heads * head_dim;
      k_proj.tensor.stride[0] = k_proj.tensor.dim[1];
      desc.sub_descs.push_back(k_proj);
      IODesc v_proj(rt::IODesc::TorchTensor,
                    "layer_" + std::to_string(layer) + "_v_proj",
                    W);
      v_proj.tensor.dim[1] = num_kv_heads * head_dim;
      v_proj.tensor.stride[0] = v_proj.tensor.dim[1];
      desc.sub_descs.push_back(v_proj);
      assert(q_proj.tensor.dim[1] + k_proj.tensor.dim[1] +
                 v_proj.tensor.dim[1] ==
             desc.tensor.dim[1]);
      io_configs.emplace(W.guid, desc);
      bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(AttnIn, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({X, W, AttnIn}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_RMS_NORM_LINEAR);
    }
    // Add attention
    {
      kn::DTensor K = kgraph.new_input(
          {batch_size, max_kv_length, num_kv_heads, head_dim},
          {(size_t)num_kv_heads * head_dim, (size_t)head_dim, 1},
          type::DT_BFLOAT16,
          layout::DmemRowMajor);
      io_configs.emplace(K.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_k_cache",
                                K));
      kn::DTensor V = kgraph.new_input(
          {batch_size, max_kv_length, num_kv_heads, head_dim},
          {(size_t)num_kv_heads * head_dim, (size_t)head_dim, 1},
          type::DT_BFLOAT16,
          layout::DmemRowMajor);
      io_configs.emplace(V.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_v_cache",
                                V));
      dim3 grid_dim = {batch_size, num_kv_heads, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, max_kv_length / 64, 64);
      // Note that QKV is concatenated together
      bgraph.new_input(AttnIn, {0, 1, -1}, -1, layout::SmemRowMajor);
      bgraph.new_input(K, {0, 2, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(V, {0, 2, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(AttnOut, {0, 1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnIn, K, V, AttnOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_ATTENTION_1);
    }
    // Add out_projection
    {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
      kn::DTensor W = kgraph.new_input({hidden_size, hidden_size},
                                       {(size_t)hidden_size, 1},
                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      io_configs.emplace(W.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_o_proj",
                                W));
      bgraph.new_input(AttnOut, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(AttnProjOut, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({AttnOut, W, AttnProjOut}, bgraph);
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
      IODesc desc(rt::IODesc::FusedTorchTensor,
                  "layer_" + std::to_string(layer) + "_gatedup_proj",
                  W);
      IODesc gate_proj(rt::IODesc::TorchTensor,
                       "layer_" + std::to_string(layer) + "_gate_proj",
                       W);
      gate_proj.tensor.dim[1] = hidden_size;
      gate_proj.tensor.stride[0] = gate_proj.tensor.dim[1];
      desc.sub_descs.push_back(gate_proj);
      IODesc up_proj(rt::IODesc::TorchTensor,
                     "layer_" + std::to_string(layer) + "_up_proj",
                     W);
      up_proj.tensor.dim[1] = hidden_size;
      up_proj.tensor.stride[0] = up_proj.tensor.dim[1];
      desc.sub_descs.push_back(up_proj);
      io_configs.emplace(W.guid, desc);

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
      io_configs.emplace(W.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_down_proj",
                                W));
      bgraph.new_input(MLPMid, {-1, -1, -1}, 1, layout::SmemRowMajor);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
      bgraph.new_input(MLPOut, {1, -1, -1}, -1, layout::SmemRowMajor);
      kgraph.customized({MLPMid, W, MLPOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_SILU_MUL_LINEAR);
    }
    X = MLPOut;
  }
  // Add RMS + Matmul
  {
    dim3 grid_dim = {vocab_size / 64, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
    kn::DTensor W = kgraph.new_input({hidden_size, vocab_size},
                                     {(size_t)vocab_size, 1},
                                     type::DT_BFLOAT16,
                                     layout::DmemRowMajor);
    io_configs.emplace(W.guid, IODesc(rt::IODesc::TorchTensor, "lm_head", W));
    bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
    bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
    bgraph.new_input(ArgmaxIn, {1, -1, -1}, -1, layout::SmemRowMajor);
    kgraph.customized({X, W, ArgmaxIn}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(2, 1, rt::TASK_RMS_NORM_LINEAR);
  }
  // Add argmax
  {
    dim3 grid_dim = {1, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, hidden_size / 64, 64);
    bgraph.new_input(ArgmaxIn, {-1, -1, -1}, -1, layout::SmemRowMajor);
    bgraph.new_input(ArgmaxOut, {-1, -1, -1}, -1, layout::SmemRowMajor);
    kgraph.customized({ArgmaxIn, ArgmaxOut}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(1, 1, rt::TASK_ARGMAX);
  }

  // Start runtime
  using namespace mirage::runtime;
  Runtime runtime(world_size /*num_gpus*/, rank /*my_gpu_id*/);
  runtime.register_mugraph(kgraph, task_configs);
  runtime.sanity_check();
  if (rank == 0) {
    std::string code =
        runtime.print_task_graph(kgraph, task_configs, io_configs);
    std::ofstream outfile("test.cu");
    outfile << code;
    outfile.close();
  }

  MPI_Finalize();
}
