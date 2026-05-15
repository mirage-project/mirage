// Standalone wrapper for fp8_group_gemm_decode_sm100.cuh device function.
// Mirrors the standalone main() in
// cpp_examples/blackwell_fp8_gemm/fp8_group_gemm_dsv3_decode_sm100.cu but
// goes through the MPK __device__ task body so we exercise the same code
// path the megakernel will run.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_largem_sm100.cuh"
#include "mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_smallm_sm100.cuh"

using bf16 = __nv_bfloat16;

__global__ __launch_bounds__(256, 1) void fp8_group_gemm_smallm_wrapper(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    const __grid_constant__ CUtensorMap tsfa,
    const __grid_constant__ CUtensorMap tsfb,
    const __grid_constant__ CUtensorMap td,
    int const *__restrict__ m_indices,
    int const M_total,
    int const N,
    int const K,
    int const E) {
  kernel::fp8_group_gemm_smallm::fp8_group_gemm_smallm_sm100_task_impl(
      &ta,
      &tb,
      &tsfa,
      &tsfb,
      &td,
      m_indices,
      M_total,
      N,
      K,
      E,
      blockIdx.x,
      gridDim.x);
}

__global__ __launch_bounds__(256, 1) void fp8_group_gemm_largem_wrapper(
    const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,
    const __grid_constant__ CUtensorMap tsfa,
    const __grid_constant__ CUtensorMap tsfb,
    const __grid_constant__ CUtensorMap td,
    int const *__restrict__ m_indices,
    int const M_total,
    int const N,
    int const K,
    int const E) {
  kernel::fp8_group_gemm_largem::fp8_group_gemm_largem_sm100_task_impl(
      &ta,
      &tb,
      &tsfa,
      &tsfb,
      &td,
      m_indices,
      M_total,
      N,
      K,
      E,
      blockIdx.x,
      gridDim.x);
}

static void chk(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(err, &s);
    TORCH_CHECK(false, "CUDA driver: ", s);
  }
}

template <int BN, typename Kernel>
static void run_impl(Kernel k_func,
                     int smem,
                     void const *A_ptr,
                     void const *B_ptr,
                     uint32_t const *sfa_packed,
                     uint32_t const *sfb_packed,
                     bf16 *D,
                     int const *m_indices,
                     int M_total,
                     int N,
                     int K,
                     int E) {
  CUtensorMap ta, tb, tsfa, tsfb, td;
  int nk = (K + 127) / 128;
  int num_sf_k = (nk + 3) / 4;
  // A: [K, M_total] FP8 (uint8 raw)
  {
    uint64_t g[2] = {(uint64_t)K, (uint64_t)M_total};
    uint64_t s[1] = {(uint64_t)K};
    uint32_t b[2] = {128, 128};
    uint32_t e[2] = {1, 1};
    chk(cuTensorMapEncodeTiled(&ta,
                               CU_TENSOR_MAP_DATA_TYPE_UINT8,
                               2,
                               (void *)A_ptr,
                               g,
                               s,
                               b,
                               e,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  // B: [K, E*N] FP8
  {
    uint64_t g[2] = {(uint64_t)K, (uint64_t)E * N};
    uint64_t s[1] = {(uint64_t)K};
    uint32_t b[2] = {128, BN};
    uint32_t e[2] = {1, 1};
    chk(cuTensorMapEncodeTiled(&tb,
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
  // SFA: [M_total, num_sf_k] uint32
  {
    uint64_t g[2] = {(uint64_t)M_total, (uint64_t)num_sf_k};
    uint64_t s[1] = {(uint64_t)M_total * sizeof(uint32_t)};
    uint32_t b[2] = {128, 1};
    uint32_t e[2] = {1, 1};
    chk(cuTensorMapEncodeTiled(&tsfa,
                               CU_TENSOR_MAP_DATA_TYPE_UINT32,
                               2,
                               (void *)sfa_packed,
                               g,
                               s,
                               b,
                               e,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  // SFB: [E*N, num_sf_k] uint32
  {
    uint64_t g[2] = {(uint64_t)E * N, (uint64_t)num_sf_k};
    uint64_t s[1] = {(uint64_t)E * N * sizeof(uint32_t)};
    uint32_t b[2] = {BN, 1};
    uint32_t e[2] = {1, 1};
    chk(cuTensorMapEncodeTiled(&tsfb,
                               CU_TENSOR_MAP_DATA_TYPE_UINT32,
                               2,
                               (void *)sfb_packed,
                               g,
                               s,
                               b,
                               e,
                               CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  // D: [N, M_total] BF16
  {
    uint64_t g[2] = {(uint64_t)N, (uint64_t)M_total};
    uint64_t s[1] = {(uint64_t)N * sizeof(bf16)};
    uint32_t b[2] = {64, 128};
    uint32_t e[2] = {1, 1};
    chk(cuTensorMapEncodeTiled(&td,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               2,
                               (void *)D,
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
  int total_blocks = ((M_total + 127) / 128) * ((N + BN - 1) / BN);
  int grid = std::min(total_blocks, num_sms);
  if (grid <= 0) {
    grid = 1;
  }

  if (smem > 48000) {
    cudaFuncSetAttribute(
        k_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  k_func<<<grid, 256, smem>>>(
      ta, tb, tsfa, tsfb, td, m_indices, M_total, N, K, E);
}

void fp8_group_gemm_decode(
    torch::Tensor A,           // [M_total, K] fp8 e4m3 raw u8
    torch::Tensor B,           // [E, N, K] fp8 e4m3 raw u8
    torch::Tensor sfa_packed,  // [num_sf_k, M_total] uint32 (transposed packed)
    torch::Tensor sfb_packed,  // [num_sf_k, E*N] uint32
    torch::Tensor D,           // [M_total, N] bf16
    torch::Tensor m_indices) { // [M_total] int32
  int M_total = A.size(0), K = A.size(1);
  int E = B.size(0), N = B.size(1);
  TORCH_CHECK(B.size(2) == K, "B K mismatch");
  TORCH_CHECK(D.size(0) == M_total && D.size(1) == N, "D shape");
  TORCH_CHECK(m_indices.size(0) == M_total, "m_indices size");
  int MPE = M_total / E;

  // Variant dispatch: K > 4096 && MPE <= 8 -> smallm (BN=64); else largem
  // (BN=128).
  if (K > 4096 && MPE <= 8) {
    run_impl<64>(
        fp8_group_gemm_smallm_wrapper,
        kernel::fp8_group_gemm_smallm::fp8_group_gemm_smallm_smem_size(),
        A.data_ptr(),
        B.data_ptr(),
        (uint32_t const *)sfa_packed.data_ptr(),
        (uint32_t const *)sfb_packed.data_ptr(),
        (bf16 *)D.data_ptr(),
        (int const *)m_indices.data_ptr(),
        M_total,
        N,
        K,
        E);
  } else {
    run_impl<128>(
        fp8_group_gemm_largem_wrapper,
        kernel::fp8_group_gemm_largem::fp8_group_gemm_largem_smem_size(),
        A.data_ptr(),
        B.data_ptr(),
        (uint32_t const *)sfa_packed.data_ptr(),
        (uint32_t const *)sfb_packed.data_ptr(),
        (bf16 *)D.data_ptr(),
        (int const *)m_indices.data_ptr(),
        M_total,
        N,
        K,
        E);
  }
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_group_gemm_decode", &fp8_group_gemm_decode);
}
