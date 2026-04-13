// Standalone wrapper for mla_mtp_decode_sm100.cuh device functions.
// Separates init (TMA, alloc) from run (kernel launch) for fair benchmarking.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/mla_mtp_decode_sm100.cuh"

using namespace kernel;
using namespace kernel::mla_mtp;

// Thin wrappers forwarding blockIdx to device function params
template <bool SINGLE_TILE>
__global__ __launch_bounds__(TB) void mla_mtp_decode_wrapper(
    const __grid_constant__ CUtensorMap Q_tm,
    const __grid_constant__ CUtensorMap KV_tm,
    nv_bfloat16 *__restrict__ Oa,
    float *__restrict__ La,
    float ss,
    int kv_len,
    int sk,
    int num_head_groups,
    int Q_LEN) {
  int gi = blockIdx.x / sk;
  int si = blockIdx.x % sk;
  int bi = blockIdx.y;
  mla_mtp_decode_sm100_task_impl<SINGLE_TILE>(&Q_tm,
                                              &KV_tm,
                                              Oa,
                                              La,
                                              ss,
                                              kv_len,
                                              sk,
                                              num_head_groups,
                                              Q_LEN,
                                              gi,
                                              si,
                                              bi);
}

__global__ __launch_bounds__(RD_TB) void mla_mtp_reduce_wrapper(
    nv_bfloat16 const *__restrict__ Oa,
    float const *__restrict__ La,
    nv_bfloat16 *__restrict__ O,
    int sk,
    int num_head_groups,
    int Q_LEN) {
  mla_mtp_reduce_sm100_task_impl(Oa,
                                 La,
                                 O,
                                 sk,
                                 num_head_groups,
                                 Q_LEN,
                                 blockIdx.x * RD_DV,
                                 blockIdx.y,
                                 blockIdx.z);
}

// ============ Persistent state ============
static CUtensorMap g_Qtm, g_KVtm;
static torch::Tensor g_Oa, g_La;
static bool g_initialized = false;
static int g_B, g_kv_len, g_sk, g_num_head_groups, g_Q_LEN, g_hpb;
static float g_ss;
static bool g_single_tile;
static constexpr int smem_size = MTP_SMEM_SIZE;

static void check_cu(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(err, &s);
    TORCH_CHECK(false, "CUDA driver error: ", s);
  }
}

void mla_mtp_init(torch::Tensor Q,
                  torch::Tensor KV,
                  torch::Tensor O,
                  int kv_len,
                  float sm_scale,
                  int q_len) {
  g_B = Q.size(0);
  g_Q_LEN = q_len;
  g_kv_len = kv_len;
  g_ss = sm_scale;

  g_hpb = 128 / g_Q_LEN;
  while (NUM_HEADS % g_hpb != 0) {
    g_hpb--;
  }
  g_num_head_groups = NUM_HEADS / g_hpb;

  int max_sk = (kv_len + TILE_S - 1) / TILE_S;
  g_sk = max_sk;

  int kvt = (kv_len + TILE_S - 1) / TILE_S;
  int tps = (kvt + g_sk - 1) / g_sk;
  g_single_tile = (tps == 1);

  int total_blocks = g_B * g_num_head_groups * g_sk;

  auto dQ = Q.view({g_B * g_Q_LEN * NUM_HEADS, D_K}).contiguous();
  auto dKV = KV.view({g_B * kv_len, D_K}).contiguous();

  g_Oa = torch::zeros({(long)total_blocks * D_V * 128},
                      torch::dtype(torch::kBFloat16).device(Q.device()));
  g_La = torch::zeros({(long)total_blocks * 128},
                      torch::dtype(torch::kFloat32).device(Q.device()));

  {
    uint64_t gd[3] = {
        64, (uint64_t)g_B * g_Q_LEN * NUM_HEADS, (uint64_t)K_ITERS};
    uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)g_hpb, 1};
    uint32_t es[3] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(&g_Qtm,
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
  {
    uint64_t gd[3] = {64, (uint64_t)g_B * kv_len, (uint64_t)K_ITERS};
    uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)TILE_S, 1};
    uint32_t es[3] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(&g_KVtm,
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

  cudaFuncSetAttribute(mla_mtp_decode_wrapper<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);
  cudaFuncSetAttribute(mla_mtp_decode_wrapper<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);
  g_initialized = true;
}

void mla_mtp_run(torch::Tensor O) {
  TORCH_CHECK(g_initialized, "Call mla_mtp_init first");

  dim3 grid(g_num_head_groups * g_sk, g_B);
  if (g_single_tile) {
    mla_mtp_decode_wrapper<true>
        <<<grid, TB, smem_size>>>(g_Qtm,
                                  g_KVtm,
                                  (nv_bfloat16 *)g_Oa.data_ptr(),
                                  (float *)g_La.data_ptr(),
                                  g_ss,
                                  g_kv_len,
                                  g_sk,
                                  g_num_head_groups,
                                  g_Q_LEN);
  } else {
    mla_mtp_decode_wrapper<false>
        <<<grid, TB, smem_size>>>(g_Qtm,
                                  g_KVtm,
                                  (nv_bfloat16 *)g_Oa.data_ptr(),
                                  (float *)g_La.data_ptr(),
                                  g_ss,
                                  g_kv_len,
                                  g_sk,
                                  g_num_head_groups,
                                  g_Q_LEN);
  }

  dim3 rg((D_V + RD_DV - 1) / RD_DV, g_num_head_groups, g_B);
  mla_mtp_reduce_wrapper<<<rg, RD_TB>>>((nv_bfloat16 *)g_Oa.data_ptr(),
                                        (float *)g_La.data_ptr(),
                                        (nv_bfloat16 *)O.data_ptr(),
                                        g_sk,
                                        g_num_head_groups,
                                        g_Q_LEN);
}

void mla_mtp_test(torch::Tensor Q,
                  torch::Tensor KV,
                  torch::Tensor O,
                  int kv_len,
                  float sm_scale,
                  int q_len) {
  mla_mtp_init(Q, KV, O, kv_len, sm_scale, q_len);
  mla_mtp_run(O);
  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_mtp_init", &mla_mtp_init);
  m.def("mla_mtp_run", &mla_mtp_run);
  m.def("mla_mtp_test", &mla_mtp_test);
}
