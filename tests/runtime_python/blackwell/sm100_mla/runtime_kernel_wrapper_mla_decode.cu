// Standalone wrapper to test mla_decode_sm100.cuh device function.
// Separates init (TMA, buffers) from run (kernel launch only) for proper
// benchmarking.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/mla_decode_sm100.cuh"

using namespace kernel;

// ============ Wrapper kernels ============

__global__ __launch_bounds__(MLA_TB) void mla_decode_wrapper(
    const __grid_constant__ CUtensorMap Q_tm,
    const __grid_constant__ CUtensorMap KV_tm,
    float *__restrict__ Oa,
    float *__restrict__ La,
    float ss,
    int kv_len,
    int sk) {
  mla_decode_sm100_task_impl(
      &Q_tm, &KV_tm, Oa, La, ss, kv_len, sk, blockIdx.x, blockIdx.y);
}

static constexpr int DV_PER_BLK = 1;
__global__ __launch_bounds__(128) void mla_reduce_wrapper(float const *Oa,
                                                          float const *La,
                                                          nv_bfloat16 *O,
                                                          int sk) {
  int const d = blockIdx.x;
  int const b = blockIdx.y;
  int const h = threadIdx.x;

  float lse_max = -1e30f;
  for (int s = 0; s < sk; s++) {
    lse_max = fmaxf(lse_max, La[(b * sk + s) * MLA_NUM_HEADS + h]);
  }
  float sum_exp = 0.0f;
  for (int s = 0; s < sk; s++) {
    sum_exp += __expf(La[(b * sk + s) * MLA_NUM_HEADS + h] - lse_max);
  }
  float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

  float acc = 0.0f;
  float const *oa_d = Oa + d * MLA_NUM_HEADS + h;
  for (int s = 0; s < sk; s++) {
    float scale =
        __expf(La[(b * sk + s) * MLA_NUM_HEADS + h] - lse_max) * inv_sum;
    acc += scale * oa_d[(b * sk + s) * MLA_D_V * MLA_NUM_HEADS];
  }
  O[(b * MLA_NUM_HEADS + h) * MLA_D_V + d] = __float2bfloat16(acc);
}

// ============ Persistent state (allocated once) ============
static CUtensorMap g_Qtm, g_KVtm;
static torch::Tensor g_Oa, g_La;
static bool g_initialized = false;
static int g_B, g_kv_len, g_sk;
static float g_ss;
static constexpr int smem_size =
    MLA_K_ITERS * MLA_TILE_BYTES + MLA_MAX_STAGES * MLA_TILE_BYTES;

static void check_cu(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(err, &s);
    TORCH_CHECK(false, "CUDA driver error: ", s);
  }
}

// ============ Python API ============

void mla_init(torch::Tensor Q,  // [B, NUM_HEADS, D_K] bf16
              torch::Tensor KV, // [B, kv_len, D_K] bf16
              torch::Tensor O,  // [B, NUM_HEADS, D_V] bf16
              int num_splits,
              float softmax_scale) {
  g_B = Q.size(0);
  g_kv_len = KV.size(1);
  g_sk = num_splits;
  g_ss = softmax_scale;

  auto dQ = Q.view({g_B * MLA_NUM_HEADS, MLA_D_K}).contiguous();
  auto dKV = KV.view({g_B * g_kv_len, MLA_D_K}).contiguous();

  // Allocate intermediate buffers once
  g_Oa = torch::zeros({g_B * g_sk * MLA_D_V * MLA_NUM_HEADS},
                      torch::dtype(torch::kFloat32).device(Q.device()));
  g_La = torch::zeros({g_B * g_sk * MLA_NUM_HEADS},
                      torch::dtype(torch::kFloat32).device(Q.device()));

  // Create TMA descriptors once — identical to mla_host.cu
  {
    uint64_t gd[3] = {64, (uint64_t)g_B * MLA_NUM_HEADS, (uint64_t)MLA_K_ITERS};
    uint64_t gs[2] = {(uint64_t)MLA_D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)MLA_NUM_HEADS, 1};
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
    uint64_t gd[3] = {64, (uint64_t)g_B * g_kv_len, (uint64_t)MLA_K_ITERS};
    uint64_t gs[2] = {(uint64_t)MLA_D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)MLA_TILE_S, 1};
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

  cudaFuncSetAttribute(mla_decode_wrapper,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  g_initialized = true;
}

void mla_run(torch::Tensor O) {
  TORCH_CHECK(g_initialized, "Call mla_init first");

  // Launch decode
  dim3 grid(g_sk, g_B);
  mla_decode_wrapper<<<grid, MLA_TB, smem_size>>>(g_Qtm,
                                                  g_KVtm,
                                                  (float *)g_Oa.data_ptr(),
                                                  (float *)g_La.data_ptr(),
                                                  g_ss,
                                                  g_kv_len,
                                                  g_sk);

  // Launch reduce
  dim3 rg(MLA_D_V / DV_PER_BLK, g_B);
  mla_reduce_wrapper<<<rg, MLA_NUM_HEADS>>>((float *)g_Oa.data_ptr(),
                                            (float *)g_La.data_ptr(),
                                            (nv_bfloat16 *)O.data_ptr(),
                                            g_sk);
}

// Combined init+run for simple correctness testing
void mla_decode_test(torch::Tensor Q,
                     torch::Tensor KV,
                     torch::Tensor O,
                     int num_splits,
                     float softmax_scale) {
  mla_init(Q, KV, O, num_splits, softmax_scale);
  mla_run(O);
  cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_init", &mla_init, "Initialize MLA (TMA descs + buffers)");
  m.def("mla_run", &mla_run, "Run MLA decode+reduce (kernel launch only)");
  m.def("mla_decode_test", &mla_decode_test, "Init+run+sync (for correctness)");
}
