// Standalone wrapper to test mla_prefill_sm100.cuh device function.
// Thin __global__ that forwards blockIdx to __device__ function.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/mla_prefill_sm100.cuh"

using bf16 = __nv_bfloat16;

static constexpr int D_CKV = 512;
static constexpr int D_KPE = 64;
static constexpr int D_V = 512;
static constexpr int BM = 64;
static constexpr int NUM_THREADS = 256;

template <bool REVERSE_BLOCKS>
__global__ __launch_bounds__(NUM_THREADS) void mla_prefill_wrapper(
    bf16 const *__restrict__ Q_nope,
    bf16 const *__restrict__ Q_pe,
    bf16 const *__restrict__ CKV,
    bf16 const *__restrict__ KPE,
    bf16 *__restrict__ O,
    int const S,
    int const H,
    float const sm_scale_log2) {
  int num_q_blocks_total = (S + BM - 1) / BM;
  int q_block =
      REVERSE_BLOCKS ? (num_q_blocks_total - 1 - blockIdx.y) : blockIdx.y;
  kernel::mla_prefill_sm100_task_impl(
      Q_nope, Q_pe, CKV, KPE, O, S, H, sm_scale_log2, blockIdx.x, q_block);
}

// Persistent state
static bool g_initialized = false;
static int g_S, g_H, g_B;
static float g_sm_scale_log2;
static int g_smem_size;

void mla_prefill_init(torch::Tensor Q_nope, // [B, S, H, D_CKV]
                      torch::Tensor Q_pe,   // [B, S, H, D_KPE]
                      torch::Tensor CKV,    // [B, S, D_CKV]
                      torch::Tensor KPE,    // [B, S, D_KPE]
                      torch::Tensor O,      // [B, S, H, D_V]
                      float sm_scale) {
  g_B = Q_nope.size(0);
  g_S = Q_nope.size(1);
  g_H = Q_nope.size(2);
  g_sm_scale_log2 = sm_scale * 1.44269504089f;
  g_smem_size = kernel::mla_prefill::PF_SMEM_SIZE;

  cudaFuncSetAttribute(mla_prefill_wrapper<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       g_smem_size);
  cudaFuncSetAttribute(mla_prefill_wrapper<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       g_smem_size);
  g_initialized = true;
}

void mla_prefill_run(torch::Tensor Q_nope,
                     torch::Tensor Q_pe,
                     torch::Tensor CKV,
                     torch::Tensor KPE,
                     torch::Tensor O) {
  TORCH_CHECK(g_initialized, "Call mla_prefill_init first");

  int num_q_blocks = (g_S + BM - 1) / BM;
  dim3 grid(g_H, num_q_blocks, g_B);
  dim3 block(NUM_THREADS, 1, 1);

  if (g_S <= 2048) {
    mla_prefill_wrapper<true>
        <<<grid, block, g_smem_size>>>((bf16 const *)Q_nope.data_ptr(),
                                       (bf16 const *)Q_pe.data_ptr(),
                                       (bf16 const *)CKV.data_ptr(),
                                       (bf16 const *)KPE.data_ptr(),
                                       (bf16 *)O.data_ptr(),
                                       g_S,
                                       g_H,
                                       g_sm_scale_log2);
  } else {
    mla_prefill_wrapper<false>
        <<<grid, block, g_smem_size>>>((bf16 const *)Q_nope.data_ptr(),
                                       (bf16 const *)Q_pe.data_ptr(),
                                       (bf16 const *)CKV.data_ptr(),
                                       (bf16 const *)KPE.data_ptr(),
                                       (bf16 *)O.data_ptr(),
                                       g_S,
                                       g_H,
                                       g_sm_scale_log2);
  }
}

void mla_prefill_test(torch::Tensor Q_nope,
                      torch::Tensor Q_pe,
                      torch::Tensor CKV,
                      torch::Tensor KPE,
                      torch::Tensor O,
                      float sm_scale) {
  mla_prefill_init(Q_nope, Q_pe, CKV, KPE, O, sm_scale);
  mla_prefill_run(Q_nope, Q_pe, CKV, KPE, O);
  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_prefill_init", &mla_prefill_init);
  m.def("mla_prefill_run", &mla_prefill_run);
  m.def("mla_prefill_test", &mla_prefill_test);
}
