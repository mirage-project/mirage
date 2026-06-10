// Standalone wrapper for mla_prefill_tp8_chunked_sm100.cuh device function.
// A thin __global__ forwards blockIdx and __grid_constant__ TMA descriptors
// to the MPK __device__ task body. Used to validate the device function in
// isolation (correctness vs reference + microbench), independent of MPK
// runtime/scheduling.
//
// Per-head DeepSeek MLA layout:
//   K_nope: [B, kv_len, H, 128]   per-head, 3D TMA
//   K_rope: [B, kv_len, 1,  64]   shared,   2D TMA
//   V:      [B, kv_len, H, 128]   per-head, 3D TMA
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/mla_prefill_tp8_chunked_sm100.cuh"

using bf16 = __nv_bfloat16;
using namespace kernel::mla_prefill_tp8_chunked;

__global__ __launch_bounds__(NT, 2) void mla_prefill_tp8_chunked_wrapper(
    const __grid_constant__ CUtensorMap KN_tm,
    const __grid_constant__ CUtensorMap KR_tm,
    const __grid_constant__ CUtensorMap V_tm,
    bf16 const *__restrict__ Qn,
    bf16 const *__restrict__ Qp,
    bf16 *__restrict__ O,
    int const q_len,
    int const kv_len,
    int const q_start,
    int const H,
    float const sml2) {
  mla_prefill_tp8_chunked_sm100_task_impl(&KN_tm,
                                          &KR_tm,
                                          &V_tm,
                                          Qn,
                                          Qp,
                                          O,
                                          q_len,
                                          kv_len,
                                          q_start,
                                          H,
                                          sml2,
                                          blockIdx.x, // head
                                          blockIdx.y, // qb_in
                                          blockIdx.z  // batch
  );
}

static void check_cu(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(err, &s);
    TORCH_CHECK(false, "CUDA driver error: ", s);
  }
}

// K_nope: 3D, view [kv_len, H, 128] as [kv_len, H*2, 64].
// V:      3D, same shape as K_nope.
static CUtensorMap make_per_head_tma(void *ptr, int kv_len, int H, int d_last) {
  CUtensorMap desc;
  uint64_t gd[3] = {64, (uint64_t)kv_len, (uint64_t)(H * 2)};
  uint64_t gs[2] = {(uint64_t)H * (uint64_t)d_last * sizeof(bf16),
                    64 * sizeof(bf16)};
  uint32_t bd[3] = {64, (uint32_t)BN, 1};
  uint32_t es[3] = {1, 1, 1};
  CUresult err =
      cuTensorMapEncodeTiled(&desc,
                             CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                             3,
                             ptr,
                             gd,
                             gs,
                             bd,
                             es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  check_cu(err);
  return desc;
}

// K_rope: 2D, view [kv_len, 1, 64] as [kv_len, 64].
static CUtensorMap make_kr_tma(void *ptr, int kv_len) {
  CUtensorMap desc;
  uint64_t gd[2] = {64, (uint64_t)kv_len};
  uint64_t gs[1] = {64 * sizeof(bf16)};
  uint32_t bd[2] = {64, (uint32_t)BN};
  uint32_t es[2] = {1, 1};
  CUresult err =
      cuTensorMapEncodeTiled(&desc,
                             CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                             2,
                             ptr,
                             gd,
                             gs,
                             bd,
                             es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  check_cu(err);
  return desc;
}

void mla_prefill_tp8_chunked_test(torch::Tensor Qn,     // [B, q_len, H, 128]
                                  torch::Tensor Qp,     // [B, q_len, H, 64]
                                  torch::Tensor K_nope, // [B, kv_len, H, 128]
                                  torch::Tensor K_rope, // [B, kv_len, 1, 64]
                                  torch::Tensor V,      // [B, kv_len, H, 128]
                                  torch::Tensor O,      // [B, q_len, H, 128]
                                  int64_t q_start,
                                  double sm_scale) {
  int B = Qn.size(0);
  int q_len = Qn.size(1);
  int H = Qn.size(2);
  int kv_len = K_nope.size(1);
  float sml2 = (float)sm_scale * 1.44269504089f;

  CUtensorMap KN_tm =
      make_per_head_tma(K_nope.data_ptr(), kv_len, H, D_QK_NOPE);
  CUtensorMap KR_tm = make_kr_tma(K_rope.data_ptr(), kv_len);
  CUtensorMap V_tm = make_per_head_tma(V.data_ptr(), kv_len, H, D_V);

  cudaFuncSetAttribute(mla_prefill_tp8_chunked_wrapper,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SMEM_SZ);

  int num_q_blocks = (q_len + BM - 1) / BM;
  dim3 grid(H, num_q_blocks, B);
  dim3 block(NT, 1, 1);

  mla_prefill_tp8_chunked_wrapper<<<grid, block, SMEM_SZ>>>(
      KN_tm,
      KR_tm,
      V_tm,
      (bf16 const *)Qn.data_ptr(),
      (bf16 const *)Qp.data_ptr(),
      (bf16 *)O.data_ptr(),
      q_len,
      kv_len,
      (int)q_start,
      H,
      sml2);
  // No cudaDeviceSynchronize here — caller (Python) syncs once via cuda
  // events around the bench loop. Per-iter sync was costing ~10-15 us.
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_prefill_tp8_chunked_test", &mla_prefill_tp8_chunked_test);
}
