// Standalone wrapper for fp8_gemm_dense_sm100.cuh device function.
// A thin __global__ forwards blockIdx.x as worker_idx and gridDim.x as
// num_workers to the MPK __device__ task body. Used to bench the device
// function in isolation, independent of MPK runtime/scheduling.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/fp8_gemm_dense_mediumm_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell/fp8_gemm_dense_smallm_sm100.cuh"

using bf16 = __nv_bfloat16;

template <int BN, int NS>
__global__ __launch_bounds__(256, 1) void fp8_gemm_dense_smallm_wrapper(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    float const *__restrict__ sa,
    float const *__restrict__ sb,
    bf16 *__restrict__ C,
    int const M,
    int const N,
    int const K) {
  kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_sm100_task_impl<BN, NS>(
      &ta, &tb, sa, sb, C, M, N, K, blockIdx.x, gridDim.x);
}

template <int BN, int NS>
__global__ __launch_bounds__(256, 1) void fp8_gemm_dense_mediumm_wrapper(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    float const *__restrict__ sa,
    float const *__restrict__ sb,
    bf16 *__restrict__ C,
    int const M,
    int const N,
    int const K) {
  kernel::fp8_gemm_dense_mediumm::fp8_gemm_dense_mediumm_sm100_task_impl<BN, NS>(
      &ta, &tb, sa, sb, C, M, N, K, blockIdx.x, gridDim.x);
}

static void check_cu(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(err, &s);
    TORCH_CHECK(false, "CUDA driver error: ", s);
  }
}

template <int BN, int NS, typename Kernel>
static void run_fp8_gemm_dense_impl(Kernel k_func,
                                    int smem_size,
                                    void const *A_ptr,
                                    void const *B_ptr,
                                    float const *sa,
                                    float const *sb,
                                    bf16 *C,
                                    int M,
                                    int N,
                                    int K) {
  CUtensorMap ta, tb;
  {
    uint64_t g[2] = {(uint64_t)K, (uint64_t)M};
    uint64_t s[1] = {(uint64_t)K};
    uint32_t b[2] = {128, 128};
    uint32_t e[2] = {1, 1};
    check_cu(cuTensorMapEncodeTiled(&ta,
                                    CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                    2,
                                    (void *)A_ptr,
                                    g,
                                    s,
                                    b,
                                    e,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  {
    uint64_t g[2] = {(uint64_t)K, (uint64_t)N};
    uint64_t s[1] = {(uint64_t)K};
    uint32_t b[2] = {128, (uint32_t)BN};
    uint32_t e[2] = {1, 1};
    check_cu(cuTensorMapEncodeTiled(&tb,
                                    CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                    2,
                                    (void *)B_ptr,
                                    g,
                                    s,
                                    b,
                                    e,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  int total = ((M + 127) / 128) * ((N + BN - 1) / BN);
  int num_waves = (total + num_sms - 1) / num_sms;
  int grid = (total + num_waves - 1) / num_waves;
  grid = std::min(grid, num_sms);

  if (smem_size > 48000) {
    cudaFuncSetAttribute(
        k_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  k_func<<<grid, 256, smem_size>>>(ta, tb, sa, sb, C, M, N, K);
}

void fp8_gemm_dense_smallm(torch::Tensor A,
                           torch::Tensor B,
                           torch::Tensor sa,
                           torch::Tensor sb,
                           torch::Tensor C) {
  int M = A.size(0), K = A.size(1);
  int N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "B K mismatch");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
  constexpr int BN = 128, NS = 4;
  run_fp8_gemm_dense_impl<BN, NS>(
      fp8_gemm_dense_smallm_wrapper<BN, NS>,
      kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_smem_size<BN, NS>(),
      A.data_ptr(),
      B.data_ptr(),
      (float const *)sa.data_ptr(),
      (float const *)sb.data_ptr(),
      (bf16 *)C.data_ptr(),
      M,
      N,
      K);
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch: ", cudaGetErrorString(err));
}

void fp8_gemm_dense_mediumm(torch::Tensor A,
                            torch::Tensor B,
                            torch::Tensor sa,
                            torch::Tensor sb,
                            torch::Tensor C) {
  int M = A.size(0), K = A.size(1);
  int N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "B K mismatch");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
  constexpr int BN = 128, NS = 4;
  run_fp8_gemm_dense_impl<BN, NS>(
      fp8_gemm_dense_mediumm_wrapper<BN, NS>,
      kernel::fp8_gemm_dense_mediumm::fp8_gemm_dense_mediumm_smem_size<BN, NS>(),
      A.data_ptr(),
      B.data_ptr(),
      (float const *)sa.data_ptr(),
      (float const *)sb.data_ptr(),
      (bf16 *)C.data_ptr(),
      M,
      N,
      K);
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemm_dense", &fp8_gemm_dense_smallm);  // back-compat alias
  m.def("fp8_gemm_dense_smallm", &fp8_gemm_dense_smallm);
  m.def("fp8_gemm_dense_mediumm", &fp8_gemm_dense_mediumm);
}
