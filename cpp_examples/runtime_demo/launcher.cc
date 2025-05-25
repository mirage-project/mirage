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
  world_size = 1;

  namespace tb = mirage::threadblock;
  namespace kn = mirage::kernel;
  namespace rt = mirage::runtime;
  namespace tp = mirage::type;
  using mirage::runtime::IODesc;
  int vocab_size = 32 * 1024;
  int batch_size = 1;
  int max_kv_length = 4096;
  int hidden_size = 4096;
  int intermediate_size = 12288;
  int num_kv_heads = 8;
  int num_q_heads = 32;
  int num_local_kv_heads = num_kv_heads / world_size;
  int num_local_q_heads = num_q_heads / world_size;
  int num_kv_splits = 1;
  int head_dim = hidden_size / num_q_heads;
  int fused_outdim_1 = head_dim * (num_q_heads + 2 * num_kv_heads);
  int fused_outdim_2 = 2 * intermediate_size;
  int fused_group_size_1, fused_group_size_2;
  int per_task_dim = 32;

  std::unordered_map<kn::KNOperator const *, std::tuple<int, int, rt::TaskType>>
      task_configs;
  std::map<tp::GuidType, rt::IODesc> io_configs;

  kn::Graph kgraph(dim3{1, 1, 1}, true /*disable_fingerprint*/);
  kn::DTensor X = kgraph.new_input(
      {batch_size, 1}, {1, 1}, type::DT_UINT16, layout::DmemRowMajor);
  io_configs.emplace(X.guid, IODesc(rt::IODesc::TorchTensor, "input_token", X));

  kn::DTensor AllReduceBuf =
      kgraph.new_input({world_size, batch_size, hidden_size},
                       {hidden_size * batch_size, hidden_size, 1},
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
  kn::DTensor AttnIn =
      kgraph.new_input({batch_size, fused_outdim_1 / world_size},
                       {(size_t)fused_outdim_1 / world_size, 1},
                       type::DT_BFLOAT16,
                       layout::DmemRowMajor);
  io_configs.emplace(AttnIn.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "attn_in", AttnIn));
  kn::DTensor MLPMid =
      kgraph.new_input({batch_size, fused_outdim_2 / world_size},
                       {(size_t)fused_outdim_2 / world_size, 1},
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
  kn::DTensor MLPFinal = kgraph.new_input({batch_size, hidden_size},
                                          {(size_t)hidden_size, 1},
                                          type::DT_BFLOAT16,
                                          layout::DmemRowMajor);
  io_configs.emplace(
      MLPFinal.guid,
      IODesc(rt::IODesc::CUDAMallocTensor, "mlp_final", MLPFinal));

  kn::DTensor AttnOut = kgraph.new_input(
      {batch_size, num_local_q_heads * head_dim},
      {(size_t)num_local_q_heads * head_dim, (size_t)head_dim, 1},
      type::DT_BFLOAT16,
      layout::DmemRowMajor);
  io_configs.emplace(AttnOut.guid,
                     IODesc(rt::IODesc::CUDAMallocTensor, "attn_out", AttnOut));
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
      dim3 grid_dim = {(size_t)fused_outdim_1 / world_size / per_task_dim,
                       1,
                       1},
           block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1, 64);
      kn::DTensor Wnorm = kgraph.new_input(
          {hidden_size}, {1}, type::DT_BFLOAT16, layout::DmemRowMajor);
      IODesc desc_norm(rt::IODesc::TorchTensor,
                       "layer_" + std::to_string(layer) + "_input_layernorm",
                       Wnorm);
      io_configs.emplace(Wnorm.guid, desc_norm);
      kn::DTensor Wqkv =
          kgraph.new_input({hidden_size, fused_outdim_1 / world_size},
                           {(size_t)fused_outdim_1 / world_size, 1},
                           type::DT_BFLOAT16,
                           layout::DmemRowMajor);
      IODesc desc_qkv(rt::IODesc::FusedTorchTensor,
                      "layer_" + std::to_string(layer) + "_qkv_proj",
                      Wqkv);
      desc_qkv.num_groups = num_local_kv_heads;
      fused_group_size_1 = fused_outdim_1 / num_kv_heads;
      IODesc q_proj(rt::IODesc::TorchTensor,
                    "layer_" + std::to_string(layer) + "_q_proj",
                    Wqkv);
      q_proj.tensor.dim[1] = num_local_q_heads * head_dim;
      q_proj.tensor.stride[0] = q_proj.tensor.dim[1];
      desc_qkv.sub_descs.push_back(q_proj);
      IODesc k_proj(rt::IODesc::TorchTensor,
                    "layer_" + std::to_string(layer) + "_k_proj",
                    Wqkv);
      k_proj.tensor.dim[1] = num_local_kv_heads * head_dim;
      k_proj.tensor.stride[0] = k_proj.tensor.dim[1];
      desc_qkv.sub_descs.push_back(k_proj);
      IODesc v_proj(rt::IODesc::TorchTensor,
                    "layer_" + std::to_string(layer) + "_v_proj",
                    Wqkv);
      v_proj.tensor.dim[1] = num_local_kv_heads * head_dim;
      v_proj.tensor.stride[0] = v_proj.tensor.dim[1];
      desc_qkv.sub_descs.push_back(v_proj);
      assert(q_proj.tensor.dim[1] + k_proj.tensor.dim[1] +
                 v_proj.tensor.dim[1] ==
             desc_qkv.tensor.dim[1]);
      io_configs.emplace(Wqkv.guid, desc_qkv);
      bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(Wnorm, {-1, -1, -1}, 0, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(Wqkv, {1, -1, -1}, 0, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(AttnIn, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized({X, Wnorm, Wqkv, AttnIn}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_RMS_NORM_LINEAR);
    }
#ifdef DEADCODE
    // Add attention
    {
      kn::DTensor K = kgraph.new_input(
          {batch_size, max_kv_length, num_local_kv_heads, head_dim},
          {(size_t)num_local_kv_heads * head_dim, (size_t)head_dim, 1},
          type::DT_BFLOAT16,
          layout::DmemRowMajor);
      io_configs.emplace(K.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_k_cache",
                                K));
      kn::DTensor V = kgraph.new_input(
          {batch_size, max_kv_length, num_local_kv_heads, head_dim},
          {(size_t)num_local_kv_heads * head_dim, (size_t)head_dim, 1},
          type::DT_BFLOAT16,
          layout::DmemRowMajor);
      io_configs.emplace(V.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_v_cache",
                                V));
      dim3 grid_dim = {batch_size, num_local_kv_heads, 1},
           block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1, 64);
      assert(AttnIn.dim[1] / num_local_kv_heads == fused_group_size_1);
      // Note that QKV is concatenated together
      bgraph.new_input(AttnIn, {0, 1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(K, {0, 2, -1}, 1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(V, {0, 2, -1}, 1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(AttnOut, {0, 1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized({AttnIn, K, V, AttnOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_ATTENTION_1);
    }
    // Add out_projection
    {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1/*forloop_dim*/, 64);
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
      // Residual input X
      // IMPORTANT: Note that we need to scale residual input X
      // by the tensor model parallel degree
      bgraph.new_input(X, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(AttnProjOut, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized({AttnOut, W, X, AttnProjOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_MATMUL_WITH_RESIDUAL);
    }
    // Reset residual input as X
    X = AttnProjOut;
    // Add AllReduce
    if (world_size > 1) {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1, 64);
      bgraph.new_input(AttnProjOut, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(AllReduceBuf, {2, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(AttnAROut, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized({AttnProjOut, AllReduceBuf, AttnAROut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_ALLREDUCE);
      X = AttnAROut;
    }
    // Add RMS + Matmul
    {
      dim3 grid_dim = {fused_outdim_2 / world_size / per_task_dim, 1, 1},
           block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1/*forloop_dim*/, 64);
      kn::DTensor Wnorm = kgraph.new_input(
          {hidden_size}, {1}, type::DT_BFLOAT16, layout::DmemRowMajor);
      IODesc desc_norm(rt::IODesc::TorchTensor,
                       "layer_" + std::to_string(layer) +
                           "_post_attn_layernorm",
                       Wnorm);
      io_configs.emplace(Wnorm.guid, desc_norm);

      kn::DTensor Wproj = kgraph.new_input({hidden_size, fused_outdim_2},
                                           {(size_t)fused_outdim_2, 1},
                                           type::DT_BFLOAT16,
                                           layout::DmemRowMajor);
      IODesc desc_proj(rt::IODesc::FusedTorchTensor,
                       "layer_" + std::to_string(layer) + "_gatedup_proj",
                       Wproj);
      desc_proj.num_groups = fused_outdim_2 / world_size / 64;
      fused_group_size_2 = 64;
      IODesc gate_proj(rt::IODesc::TorchTensor,
                       "layer_" + std::to_string(layer) + "_gate_proj",
                       Wproj);
      gate_proj.tensor.dim[1] = intermediate_size;
      gate_proj.tensor.stride[0] = gate_proj.tensor.dim[1];
      desc_proj.sub_descs.push_back(gate_proj);
      IODesc up_proj(rt::IODesc::TorchTensor,
                     "layer_" + std::to_string(layer) + "_up_proj",
                     Wproj);
      up_proj.tensor.dim[1] = intermediate_size;
      up_proj.tensor.stride[0] = up_proj.tensor.dim[1];
      desc_proj.sub_descs.push_back(up_proj);
      io_configs.emplace(Wproj.guid, desc_proj);

      bgraph.new_input(world_size > 1 ? AttnAROut : AttnProjOut,
                       {-1, -1, -1},
                       1,
                       layout::SmemRowMajor,
		       true /*store_in_dmem*/);
      bgraph.new_input(Wnorm, {-1, -1, -1}, 0, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(Wproj, {1, -1, -1}, 0, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(MLPMid, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized(
          {world_size > 1 ? AttnAROut : AttnProjOut, Wnorm, Wproj, MLPMid},
          bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_RMS_NORM_LINEAR);
    }
    // silu + Matmul
    {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1/*forloop_range*/, 64);
      kn::DTensor W = kgraph.new_input({intermediate_size, hidden_size},
                                       {(size_t)hidden_size, 1},
                                       type::DT_BFLOAT16,
                                       layout::DmemRowMajor);
      io_configs.emplace(W.guid,
                         IODesc(rt::IODesc::TorchTensor,
                                "layer_" + std::to_string(layer) + "_down_proj",
                                W));
      // Each forloop iteration handles one group
      assert(MLPMid.dim[1] / bgraph.forloop_range == fused_group_size_2);
      bgraph.new_input(MLPMid, {-1, -1, -1}, 1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor, true /*store_in_dmem*/);
      // Residual input X
      // IMPORTANT: Note that we need to scale residual input X
      // by the tensor model parallel degree
      bgraph.new_input(X, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(MLPOut, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized({MLPMid, W, X, MLPOut}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(3, 1, rt::TASK_SILU_MUL_LINEAR_WITH_RESIDUAL);
    }
    // Reset residual input X
    X = MLPOut;
    // Add AllReduce
    if (world_size > 1) {
      dim3 grid_dim = {hidden_size / 64, 1, 1}, block_dim = {128, 1, 1};
      tb::Graph bgraph(grid_dim, block_dim, 1, 64);
      bgraph.new_input(MLPOut, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(AllReduceBuf, {2, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      bgraph.new_input(MLPFinal, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
      kgraph.customized({MLPOut, AllReduceBuf, MLPFinal}, bgraph);
      task_configs[kgraph.operators.back()] =
          std::make_tuple(2, 1, rt::TASK_ALLREDUCE);
      X = MLPFinal;
    }
#endif
  }
#ifdef DEADCODE
  // Add RMS + Matmul
  {
    dim3 grid_dim = {vocab_size / 32, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 1/*forloop_range*/, 64);
    kn::DTensor W = kgraph.new_input({hidden_size, vocab_size},
                                     {(size_t)vocab_size, 1},
                                     type::DT_BFLOAT16,
                                     layout::DmemRowMajor);
    io_configs.emplace(W.guid, IODesc(rt::IODesc::TorchTensor, "lm_head", W));
    bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor, true /*store_in_dmem*/);
    bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor, true /*store_in_dmem*/);
    bgraph.new_input(ArgmaxIn, {1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    kgraph.customized({X, W, ArgmaxIn}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(2, 1, rt::TASK_RMS_NORM_LINEAR);
  }
  // Add argmax
  {
    dim3 grid_dim = {1, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 1/*forloop_range*/, 64);
    bgraph.new_input(ArgmaxIn, {-1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    bgraph.new_input(ArgmaxOut, {-1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    kgraph.customized({ArgmaxIn, ArgmaxOut}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(1, 1, rt::TASK_ARGMAX);
  }
#else
  {
    dim3 grid_dim = {1, 1, 1}, block_dim = {128, 1, 1};
    tb::Graph bgraph(grid_dim, block_dim, 1/*forloop_range*/, 64);
    bgraph.new_input(AttnIn, {-1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    bgraph.new_input(ArgmaxOut, {-1, -1, -1}, -1, layout::SmemRowMajor, true /*store_in_dmem*/);
    kgraph.customized({AttnIn, ArgmaxOut}, bgraph);
    task_configs[kgraph.operators.back()] =
        std::make_tuple(1, 1, rt::TASK_ARGMAX);
  }
#endif

  // Start runtime
  using namespace mirage::runtime;
  Runtime runtime(world_size /*num_gpus*/, rank /*my_gpu_id*/);
  runtime.register_mugraph(kgraph, task_configs);
  runtime.sanity_check();
  if (rank == 0) {
    std::string code =
        runtime.print_task_graph(kgraph, task_configs, io_configs, true/*use_json_format*/);
    std::ofstream outfile("test.cu");
    outfile << code;
    outfile.close();
  }

  MPI_Finalize();
}
