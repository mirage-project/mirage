// Kernel-wrapper correctness harness for the DeepSeek-V3 TP=4 MLA decode kernel
// (mla_mtp_decode_tp4_sm100.cuh).
//
// Why a kernel-wrapper (not test_mode): test_mode's init_kernel zeros `step`
// and runs a single pass, which is always a prefill (kv_len ~= mbt). It cannot
// drive the decode kernel at kv_len=256. This harness calls the device
// task functions DIRECTLY via thin __global__ shims that forward blockIdx, with
// kv_len passed as an explicit int — full control on one card, no AllReduce, no
// prepare_next_batch.
//
// Pattern lifted from:
//   - sm100_mla/runtime_kernel_wrapper_mla_decode.cu  (torch::extension binding)
//   - sm100_mla_mtp_decode/test_tp4_wrapper.cu         (exact tp4 launch + TMA)
//
// The two device functions are (arg order = ground truth from
// src/kernel/task_register.cc register_mla_mtp_decode_tp4_sm100_task /
// _reduce_sm100_task):
//   mla_mtp_tp4_main<SINGLE_TILE, WRITE_FINAL>(Q_tm, KV_tm, Oa, La, ss,
//       kv_len, sk, Q_LEN, qpg, page_indices, first_page_pos,
//       block_x_packed, block_y)
//   mla_mtp_tp4_reduce(Oa, La, O, sk, num_groups, Q_LEN, qpg,
//       block_x, block_y, block_z)

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>

#include "mirage/persistent_kernel/tasks/blackwell/mla_mtp_decode_tp4_sm100.cuh"

using namespace kernel::mla_mtp_tp4;

// ===== Thin __global__ shims forwarding blockIdx to the __device__ funcs =====
template <bool SINGLE_TILE>
__global__ __launch_bounds__(TB) void shim_main(
    const __grid_constant__ CUtensorMap Q_tm,
    const __grid_constant__ CUtensorMap KV_tm,
    nv_bfloat16 *Oa,
    float *La,
    float ss,
    int kv_len,
    int sk,
    int Q_LEN,
    int qpg,
    int const *page_indices) {
  // Pack the V split (blockIdx.z) into block_x — Mirage MPK has no z-dim.
  // The x dimension may also carry a head-group field when
  // MIRAGE_MLA_TP4_HEAD_GROUPS > 1.
  int block_x_packed = blockIdx.x * V_SPLITS + blockIdx.z;
  mla_mtp_tp4_main<SINGLE_TILE, false>(&Q_tm,
                                       &KV_tm,
                                       Oa,
                                       La,
                                       ss,
                                       kv_len,
                                       sk,
                                       Q_LEN,
                                       qpg,
                                       page_indices,
                                       /*first_page_pos=*/0,
                                       block_x_packed,
                                       blockIdx.y);
}

__global__ __launch_bounds__(RD_TB, 4) void shim_reduce(nv_bfloat16 const *Oa,
                                                        float const *La,
                                                        nv_bfloat16 *O,
                                                        int sk,
                                                        int num_groups,
                                                        int Q_LEN,
                                                        int qpg) {
  mla_mtp_tp4_reduce(Oa,
                     La,
                     O,
                     sk,
                     num_groups,
                     Q_LEN,
                     qpg,
                     blockIdx.x,
                     blockIdx.y,
                     blockIdx.z);
}

static void check_cu(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(err, &s);
    TORCH_CHECK(false, "CUDA driver error: ", s);
  }
}

// Combined init + run + reduce + sync. Stateless: all buffers allocated here so
// the python side only owns Q, KV, attn_out.
//
//   Q:        bf16 [B * q_len * NUM_HEADS, D_K]   (row h = head h, contiguous)
//   KV:       bf16 [B * kv_len, D_K]              (V = first D_V of each row)
//   attn_out: bf16 [B * q_len, NUM_HEADS * D_V]   head-major final output
//   ss:       DeepSeek softmax scale (passed in from python so the scale used by
//             the kernel matches the reference exactly)
void mla_decode_tp4_test(torch::Tensor Q,
                         torch::Tensor KV,
                         torch::Tensor attn_out,
                         int batch_size,
                         int q_len,
                         int kv_len,
                         double softmax_scale) {
  TORCH_CHECK(Q.is_cuda() && KV.is_cuda() && attn_out.is_cuda(),
              "all tensors must be on CUDA");
  TORCH_CHECK(Q.scalar_type() == torch::kBFloat16, "Q must be bf16");
  TORCH_CHECK(KV.scalar_type() == torch::kBFloat16, "KV must be bf16");
  TORCH_CHECK(attn_out.scalar_type() == torch::kBFloat16, "attn_out must be bf16");

  auto dQ = Q.contiguous();
  auto dKV = KV.contiguous();

  int const B = batch_size;
  int const Q_LEN = q_len;
  int const KL = kv_len;
  float const ss = (float)softmax_scale;

  int const qpg = std::min(4, Q_LEN);
  int const num_groups = (Q_LEN + qpg - 1) / qpg;
  int const hpb = HEADS_PER_GROUP; // 32 / HEAD_GROUPS

  // num_splits = number of KV tiles (one tile = TILE_S=128 tokens). This is the
  // builder's max-split decision for the decode path (ceil(kv_len/128)).
  int const kvt = (KL + TILE_S - 1) / TILE_S;
  int const sk = kvt; // sk == ceil(kv_len/128); for kv_len=256 -> 2
  int const tps = (kvt + sk - 1) / sk;
  bool const single_tile = (tps == 1);

  // Partial buffers (LSE-weighted split-K partials, combined by reduce):
  //   partial_blocks = B * num_groups * sk   (matches builder.py:2954-2971)
  //   Oa: [partial_blocks, D_V * 128] bf16    La: [partial_blocks, 128] f32
  int const partial_blocks = B * num_groups * sk;
  auto opts_bf16 = torch::dtype(torch::kBFloat16).device(Q.device());
  auto opts_f32 = torch::dtype(torch::kFloat32).device(Q.device());
  auto Oa = torch::zeros({(long)partial_blocks * D_V * 128}, opts_bf16);
  auto La = torch::zeros({(long)partial_blocks * 128}, opts_f32);

  // ---- Q TMA descriptor (3d, 128B swizzle) ----
  CUtensorMap Qtm;
  {
    uint64_t gd[3] = {64, (uint64_t)B * Q_LEN * NUM_HEADS, (uint64_t)K_ITERS};
    uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)hpb, 1};
    uint32_t es[3] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(&Qtm,
                                    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                    3,
                                    (void *)dQ.data_ptr(),
                                    gd,
                                    gs,
                                    bd,
                                    es,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  // ---- KV TMA descriptor (3d, 128B swizzle) ----
  CUtensorMap KVtm;
  {
    uint64_t gd[3] = {64, (uint64_t)B * KL, (uint64_t)K_ITERS};
    uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)TILE_S, 1};
    uint32_t es[3] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(&KVtm,
                                    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                    3,
                                    (void *)dKV.data_ptr(),
                                    gd,
                                    gs,
                                    bd,
                                    es,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  cudaFuncSetAttribute(shim_main<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SMEM_SIZE);
  cudaFuncSetAttribute(shim_main<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SMEM_SIZE);

  // ---- launch main ----
  {
    dim3 g(num_groups * sk * HEAD_GROUPS, B, V_SPLITS);
    if (single_tile) {
      shim_main<true><<<g, TB, SMEM_SIZE>>>(
          Qtm, KVtm, (nv_bfloat16 *)Oa.data_ptr(), (float *)La.data_ptr(), ss,
          KL, sk, Q_LEN, qpg, /*page_indices=*/nullptr);
    } else {
      shim_main<false><<<g, TB, SMEM_SIZE>>>(
          Qtm, KVtm, (nv_bfloat16 *)Oa.data_ptr(), (float *)La.data_ptr(), ss,
          KL, sk, Q_LEN, qpg, /*page_indices=*/nullptr);
    }
  }

  // ---- launch reduce ----
  {
    dim3 rg((D_V + RD_DV - 1) / RD_DV, num_groups, B);
    shim_reduce<<<rg, RD_TB>>>((nv_bfloat16 *)Oa.data_ptr(),
                               (float *)La.data_ptr(),
                               (nv_bfloat16 *)attn_out.data_ptr(),
                               sk,
                               num_groups,
                               Q_LEN,
                               qpg);
  }

  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch/sync error: ",
              cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_decode_tp4_test",
        &mla_decode_tp4_test,
        "Init+run TP=4 MLA decode main + reduce, sync (for correctness)");
}
