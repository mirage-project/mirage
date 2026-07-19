/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Direct invocation wrapper for fp8_gemm_dense_smallm_sm100_task_impl<128, 3>
// Exercises M=64 and M=128 shapes for the kv_b_v projection (N=4096, K=512).

#include "blackwell/fp8_gemm_dense_smallm_sm100.cuh"
#include "blackwell/per_token_group_quantize_fp8.cuh"
#include <cute/numeric/numeric_types.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

void check_driver_success(CUresult result, char const *what) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  char const *error_name = nullptr;
  char const *error_string = nullptr;
  cuGetErrorName(result, &error_name);
  cuGetErrorString(result, &error_string);
  TORCH_CHECK(false,
              what,
              " failed with ",
              (error_name ? error_name : "unknown"),
              ": ",
              (error_string ? error_string : "unknown"));
}

// Build a 2-D TMA descriptor for an FP8 (uint8) row-major tensor.
// Matches the production encoding in tma.cuh for
// TASK_FP8_GEMM_DENSE_SM100 (smallm flavor):
//   gd  = {K, outer}   (gmem dims, inner-first)
//   gs  = {K}          (gmem stride in bytes; fp8 so K bytes per row)
//   bd  = {128, 128}   (smem box: BK x OUTER_BOX)
//   swizzle = 128B, l2 = NONE, oob = NONE
CUtensorMap make_tma_fp8_2d(void *ptr, int K, int outer) {
  CUtensorMap tm{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(K),
                              static_cast<cuuint64_t>(outer)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(K)}; // bytes
  const cuuint32_t box[2] = {128u, 128u};
  const cuuint32_t elem_strd[2] = {1, 1};
  check_driver_success(
      cuTensorMapEncodeTiled(&tm,
                             CU_TENSOR_MAP_DATA_TYPE_UINT8,
                             2,
                             ptr,
                             dims,
                             strides,
                             box,
                             elem_strd,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(A/B)");
  return tm;
}

// Kernel wrapper: launch one CTA (worker_idx=0, num_workers=1).
__global__ void __launch_bounds__(256)
    fp8_dense_smallm_test_kernel(CUtensorMap const *ta,
                                 CUtensorMap const *tb,
                                 float const *sa,
                                 float const *sb,
                                 __nv_bfloat16 *C,
                                 int M,
                                 int N,
                                 int K) {
  kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_sm100_task_impl<128, 3>(
      ta, tb, sa, sb, C, M, N, K, /*worker_idx=*/0, /*num_workers=*/1);
}

// Multi-CTA wrapper: launch num_workers CTAs, each with worker_idx=blockIdx.x.
// This matches the production launch pattern in the persistent kernel
// (grid = (num_workers, 1, 1)).
__global__ void __launch_bounds__(256)
    fp8_dense_smallm_multi_cta_test_kernel(CUtensorMap const *ta,
                                           CUtensorMap const *tb,
                                           float const *sa,
                                           float const *sb,
                                           __nv_bfloat16 *C,
                                           int M,
                                           int N,
                                           int K,
                                           int num_workers) {
  kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_sm100_task_impl<128, 3>(
      ta,
      tb,
      sa,
      sb,
      C,
      M,
      N,
      K,
      /*worker_idx=*/static_cast<int>(blockIdx.x),
      /*num_workers=*/num_workers);
}

} // namespace

// Python-callable entry point.
// a_fp8:  float8_e4m3fn [M, K]   contiguous
// b_fp8:  float8_e4m3fn [N, K]   contiguous
// sa:     float32        [M, K/128] contiguous (row-major activation scales)
// sb:     float32        [N/128, K/128] contiguous (row-major weight scales)
// output: bfloat16       [M, N]   contiguous (zero-initialised by caller)
void fp8_gemm_dense_smallm_launch(torch::Tensor a_fp8,
                                  torch::Tensor b_fp8,
                                  torch::Tensor sa,
                                  torch::Tensor sb,
                                  torch::Tensor output) {
  TORCH_CHECK(a_fp8.dim() == 2 && b_fp8.dim() == 2,
              "a_fp8 and b_fp8 must be 2D");
  TORCH_CHECK(sa.dim() == 2 && sb.dim() == 2, "sa and sb must be 2D");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(a_fp8.scalar_type() == at::kFloat8_e4m3fn,
              "a_fp8 must be float8_e4m3fn");
  TORCH_CHECK(b_fp8.scalar_type() == at::kFloat8_e4m3fn,
              "b_fp8 must be float8_e4m3fn");
  TORCH_CHECK(sa.scalar_type() == at::kFloat, "sa must be float32");
  TORCH_CHECK(sb.scalar_type() == at::kFloat, "sb must be float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(a_fp8.is_contiguous() && b_fp8.is_contiguous() &&
                  sa.is_contiguous() && sb.is_contiguous() &&
                  output.is_contiguous(),
              "all tensors must be contiguous");

  int const M = static_cast<int>(a_fp8.size(0));
  int const K = static_cast<int>(a_fp8.size(1));
  int const N = static_cast<int>(b_fp8.size(0));

  TORCH_CHECK(b_fp8.size(1) == K, "b_fp8 dim1 must equal K");
  TORCH_CHECK(sa.size(0) == M && sa.size(1) == K / 128,
              "sa shape must be [M, K/128]");
  TORCH_CHECK(sb.size(0) == N / 128 && sb.size(1) == K / 128,
              "sb shape must be [N/128, K/128]");
  TORCH_CHECK(output.size(0) == M && output.size(1) == N,
              "output shape must be [M, N]");
  TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");

  // Build TMA descriptors matching production tma.cuh encoding.
  CUtensorMap ta = make_tma_fp8_2d(a_fp8.data_ptr(), K, M);
  CUtensorMap tb = make_tma_fp8_2d(b_fp8.data_ptr(), K, N);

  // Copy descriptors to device.
  CUtensorMap *d_ta = nullptr, *d_tb = nullptr;
  TORCH_CHECK(cudaMalloc(&d_ta, sizeof(CUtensorMap)) == cudaSuccess,
              "cudaMalloc d_ta failed");
  TORCH_CHECK(cudaMalloc(&d_tb, sizeof(CUtensorMap)) == cudaSuccess,
              "cudaMalloc d_tb failed");
  TORCH_CHECK(
      cudaMemcpy(d_ta, &ta, sizeof(CUtensorMap), cudaMemcpyHostToDevice) ==
          cudaSuccess,
      "cudaMemcpy d_ta failed");
  TORCH_CHECK(
      cudaMemcpy(d_tb, &tb, sizeof(CUtensorMap), cudaMemcpyHostToDevice) ==
          cudaSuccess,
      "cudaMemcpy d_tb failed");

  constexpr int kSmemBytes =
      kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_smem_size<128, 3>();
  TORCH_CHECK(cudaFuncSetAttribute(fp8_dense_smallm_test_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   kSmemBytes) == cudaSuccess,
              "cudaFuncSetAttribute MaxDynamicSharedMemorySize failed");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // cta_group::1 in PTX means "this CTA's own group" — it works with a regular
  // kernel launch (no cluster required). fence.mbarrier_init.release.cluster is
  // also valid with cluster=1x1x1 since it degenerates to a CTA-scoped fence.
  // Use plain cudaLaunchKernel (matching the existing linear_fp8 test wrapper
  // pattern).
  CUtensorMap const *arg_ta = d_ta;
  CUtensorMap const *arg_tb = d_tb;
  float const *arg_sa = static_cast<float const *>(sa.data_ptr());
  float const *arg_sb = static_cast<float const *>(sb.data_ptr());
  __nv_bfloat16 *arg_C = static_cast<__nv_bfloat16 *>(output.data_ptr());
  int arg_M = M;
  int arg_N = N;
  int arg_K = K;

  void *kernel_params[] = {
      &arg_ta, &arg_tb, &arg_sa, &arg_sb, &arg_C, &arg_M, &arg_N, &arg_K};

  cudaError_t launch_err = cudaLaunchKernel(
      reinterpret_cast<void const *>(fp8_dense_smallm_test_kernel),
      dim3(1, 1, 1),
      dim3(256, 1, 1),
      kernel_params,
      kSmemBytes,
      stream);
  TORCH_CHECK(launch_err == cudaSuccess,
              "cudaLaunchKernel (fp8_dense_smallm_test_kernel) failed: ",
              cudaGetErrorString(launch_err));

  auto err = cudaGetLastError();
  cudaFree(d_ta);
  cudaFree(d_tb);
  TORCH_CHECK(err == cudaSuccess,
              "fp8_dense_smallm_test_kernel launch failed: ",
              cudaGetErrorString(err));
}

// Multi-CTA entry point: matches the persistent-kernel launch pattern that
// the production code uses (grid_dim = (num_workers, 1, 1)). Triggers the
// nn-dependent correctness bug at M=128, N=2176, K=7168, num_workers=128.
void fp8_gemm_dense_smallm_multi_cta_launch(torch::Tensor a_fp8,
                                            torch::Tensor b_fp8,
                                            torch::Tensor sa,
                                            torch::Tensor sb,
                                            torch::Tensor output,
                                            int64_t num_workers) {
  TORCH_CHECK(a_fp8.dim() == 2 && b_fp8.dim() == 2,
              "a_fp8 and b_fp8 must be 2D");
  TORCH_CHECK(sa.dim() == 2 && sb.dim() == 2, "sa and sb must be 2D");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(a_fp8.scalar_type() == at::kFloat8_e4m3fn,
              "a_fp8 must be float8_e4m3fn");
  TORCH_CHECK(b_fp8.scalar_type() == at::kFloat8_e4m3fn,
              "b_fp8 must be float8_e4m3fn");
  TORCH_CHECK(sa.scalar_type() == at::kFloat, "sa must be float32");
  TORCH_CHECK(sb.scalar_type() == at::kFloat, "sb must be float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(a_fp8.is_contiguous() && b_fp8.is_contiguous() &&
                  sa.is_contiguous() && sb.is_contiguous() &&
                  output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(num_workers > 0 && num_workers <= 1024,
              "num_workers must be in (0, 1024]");

  int const M = static_cast<int>(a_fp8.size(0));
  int const K = static_cast<int>(a_fp8.size(1));
  int const N = static_cast<int>(b_fp8.size(0));

  TORCH_CHECK(b_fp8.size(1) == K, "b_fp8 dim1 must equal K");
  TORCH_CHECK(sa.size(0) == M && sa.size(1) == K / 128,
              "sa shape must be [M, K/128]");
  TORCH_CHECK(sb.size(0) == N / 128 && sb.size(1) == K / 128,
              "sb shape must be [N/128, K/128]");
  TORCH_CHECK(output.size(0) == M && output.size(1) == N,
              "output shape must be [M, N]");
  TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");

  CUtensorMap ta = make_tma_fp8_2d(a_fp8.data_ptr(), K, M);
  CUtensorMap tb = make_tma_fp8_2d(b_fp8.data_ptr(), K, N);

  CUtensorMap *d_ta = nullptr, *d_tb = nullptr;
  TORCH_CHECK(cudaMalloc(&d_ta, sizeof(CUtensorMap)) == cudaSuccess,
              "cudaMalloc d_ta failed");
  TORCH_CHECK(cudaMalloc(&d_tb, sizeof(CUtensorMap)) == cudaSuccess,
              "cudaMalloc d_tb failed");
  TORCH_CHECK(
      cudaMemcpy(d_ta, &ta, sizeof(CUtensorMap), cudaMemcpyHostToDevice) ==
          cudaSuccess,
      "cudaMemcpy d_ta failed");
  TORCH_CHECK(
      cudaMemcpy(d_tb, &tb, sizeof(CUtensorMap), cudaMemcpyHostToDevice) ==
          cudaSuccess,
      "cudaMemcpy d_tb failed");

  constexpr int kSmemBytes =
      kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_smem_size<128, 3>();
  TORCH_CHECK(cudaFuncSetAttribute(fp8_dense_smallm_multi_cta_test_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   kSmemBytes) == cudaSuccess,
              "cudaFuncSetAttribute MaxDynamicSharedMemorySize failed");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  CUtensorMap const *arg_ta = d_ta;
  CUtensorMap const *arg_tb = d_tb;
  float const *arg_sa = static_cast<float const *>(sa.data_ptr());
  float const *arg_sb = static_cast<float const *>(sb.data_ptr());
  __nv_bfloat16 *arg_C = static_cast<__nv_bfloat16 *>(output.data_ptr());
  int arg_M = M;
  int arg_N = N;
  int arg_K = K;
  int arg_NW = static_cast<int>(num_workers);

  void *kernel_params[] = {&arg_ta,
                           &arg_tb,
                           &arg_sa,
                           &arg_sb,
                           &arg_C,
                           &arg_M,
                           &arg_N,
                           &arg_K,
                           &arg_NW};

  cudaError_t launch_err = cudaLaunchKernel(
      reinterpret_cast<void const *>(fp8_dense_smallm_multi_cta_test_kernel),
      dim3(static_cast<unsigned int>(num_workers), 1, 1),
      dim3(256, 1, 1),
      kernel_params,
      kSmemBytes,
      stream);
  TORCH_CHECK(launch_err == cudaSuccess,
              "cudaLaunchKernel (multi_cta) failed: ",
              cudaGetErrorString(launch_err));

  auto err = cudaGetLastError();
  cudaFree(d_ta);
  cudaFree(d_tb);
  TORCH_CHECK(err == cudaSuccess,
              "fp8_dense_smallm_multi_cta_test_kernel launch failed: ",
              cudaGetErrorString(err));
}

// Kernel wrapper for quantize_fp8 f32-scale + dense GEMM chained.
// Each block of the quantize step quantizes 1 row's 1 group-tile slice.
// Hard-codes M=128, K=7168, GROUP_SIZE=128, GROUP_TILES=4 (matches MPK
// `_fp8_quantize_group_tiles(7168, scale_ue8m0=False) = 4`).
namespace {

__global__ void __launch_bounds__(128)
    quantize_fp8_f32_test_kernel(__nv_bfloat16 const *__restrict__ input,
                                 __nv_fp8_e4m3 *__restrict__ output_q,
                                 float *__restrict__ output_s) {
  // grid = (group_tiles=4, batch=128, 1).  block = (128, 1, 1).
  int const row_idx = blockIdx.y;
  int const group_tile_idx = blockIdx.x;
  kernel::per_token_group_quantize_fp8_task_impl<
      /*BATCH_SIZE=*/128,
      /*HIDDEN_SIZE=*/7168,
      /*GROUP_SIZE=*/128,
      /*GLOBAL_STRIDE=*/7168,
      /*GROUP_TILES=*/4,
      cute::bfloat16_t,
      __nv_fp8_e4m3,
      /*SCALE_UE8M0=*/false>(input,
                             output_q,
                             output_s,
                             /*eps=*/1e-10f,
                             /*min_8bit=*/-448.0f,
                             /*max_8bit=*/448.0f,
                             /*scale_outer_stride=*/0, // unused for f32 scale
                             row_idx,
                             group_tile_idx);
}

} // namespace

// Entry point that runs MPK's quantize_fp8 kernel + the dense GEMM kernel
// back-to-back on the same data, mimicking what happens inside the MPK
// persistent kernel for the QKV-a fused GEMM.
//
// Inputs:
//   a_bf16: (M=128, K=7168) bfloat16
//   b_fp8:  (N, K=7168) float8_e4m3   (pre-quantized weight)
//   sb:     (N/128, K/128) float32    (weight scale)
//   output: (M, N) bfloat16
//   num_workers: GEMM worker count
//
// The wrapper INTERNALLY allocates the FP8 input buffer + scale buffer and
// passes them between the two kernels (same way MPK's _fp8_linear_v2 does).
void quantize_then_gemm_launch(torch::Tensor a_bf16,
                               torch::Tensor b_fp8,
                               torch::Tensor sb,
                               torch::Tensor output,
                               int64_t num_workers) {
  TORCH_CHECK(a_bf16.scalar_type() == at::kBFloat16, "a_bf16 must be bf16");
  TORCH_CHECK(b_fp8.scalar_type() == at::kFloat8_e4m3fn, "b_fp8 must be fp8");
  TORCH_CHECK(sb.scalar_type() == at::kFloat, "sb must be fp32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(a_bf16.dim() == 2 && b_fp8.dim() == 2 && sb.dim() == 2,
              "tensors must be 2D");
  int const M = static_cast<int>(a_bf16.size(0));
  int const K = static_cast<int>(a_bf16.size(1));
  int const N = static_cast<int>(b_fp8.size(0));
  TORCH_CHECK(M == 128 && K == 7168,
              "quantize wrapper assumes M=128, K=7168 (template-hardcoded). "
              "Got M=",
              M,
              ", K=",
              K);
  TORCH_CHECK(b_fp8.size(1) == K, "b_fp8 dim1 must equal K");
  TORCH_CHECK(sb.size(0) == N / 128 && sb.size(1) == K / 128,
              "sb shape mismatch");
  TORCH_CHECK(output.size(0) == M && output.size(1) == N,
              "output shape mismatch");
  TORCH_CHECK(num_workers > 0 && num_workers <= 1024, "bad num_workers");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto opts_u8 =
      torch::TensorOptions().dtype(at::kFloat8_e4m3fn).device(a_bf16.device());
  auto opts_f =
      torch::TensorOptions().dtype(at::kFloat).device(a_bf16.device());
  // FP8 input buffer (M, K) and float32 scale (M, K/128), matching MPK's
  // _fp8_mbt_buffers_for_reduction_f32scale layout.
  torch::Tensor a_fp8 = torch::empty({M, K}, opts_u8);
  torch::Tensor sa = torch::empty({M, K / 128}, opts_f);

  // === Step 1: quantize bf16 → fp8 + f32 scale ===
  dim3 quant_grid(4, 128, 1); // GROUP_TILES=4, BATCH=128
  dim3 quant_block(128, 1, 1);
  __nv_bfloat16 const *q_in =
      static_cast<__nv_bfloat16 const *>(a_bf16.data_ptr());
  __nv_fp8_e4m3 *q_outq = static_cast<__nv_fp8_e4m3 *>(a_fp8.data_ptr());
  float *q_outs = static_cast<float *>(sa.data_ptr());
  void *quant_params[] = {&q_in, &q_outq, &q_outs};
  cudaError_t qerr = cudaLaunchKernel(
      reinterpret_cast<void const *>(quantize_fp8_f32_test_kernel),
      quant_grid,
      quant_block,
      quant_params,
      /*sharedMem=*/0,
      stream);
  TORCH_CHECK(qerr == cudaSuccess,
              "quantize_fp8_f32_test_kernel launch failed: ",
              cudaGetErrorString(qerr));

  // === Step 2: dense GEMM ===
  // Reuse the existing multi_cta launch (which builds TMA descriptors etc.).
  fp8_gemm_dense_smallm_multi_cta_launch(
      a_fp8, b_fp8, sa, sb, output, num_workers);
}

// Standalone test for the OUTPUT_STRIDE fix in the quantize kernel.
// Verifies that quantizing a [128, 1536] slice from a [128, 2176] parent
// buffer writes the FP8 output into a (128, 1536) buffer without
// overflowing past byte 196608 (the H8 bug).
namespace {

__global__ void __launch_bounds__(128)
    quantize_fp8_slice_test_kernel(__nv_bfloat16 const *__restrict__ input,
                                   __nv_fp8_e4m3 *__restrict__ output_q,
                                   float *__restrict__ output_s) {
  int const row_idx = blockIdx.y;
  int const group_tile_idx = blockIdx.x;
  // BATCH=128, HIDDEN=1536, GROUP=128, GLOBAL_STRIDE=2176, GROUP_TILES=1,
  // OUTPUT_STRIDE=1536 (the fix). Mirrors MPK's variant 2 for q_b's quantize.
  kernel::per_token_group_quantize_fp8_task_impl<
      /*BATCH_SIZE=*/128,
      /*HIDDEN_SIZE=*/1536,
      /*GROUP_SIZE=*/128,
      /*GLOBAL_STRIDE=*/2176,
      /*GROUP_TILES=*/1,
      cute::bfloat16_t,
      __nv_fp8_e4m3,
      /*SCALE_UE8M0=*/false,
      /*OUTPUT_STRIDE=*/1536>(input,
                              output_q,
                              output_s,
                              /*eps=*/1e-10f,
                              /*min_8bit=*/-448.0f,
                              /*max_8bit=*/448.0f,
                              /*scale_outer_stride=*/0,
                              row_idx,
                              group_tile_idx);
}

} // namespace

// Quantize a (128, 1536) column slice of a (128, 2176) BF16 buffer to FP8.
// Output FP8 buffer is allocated as (128, 1536) — half the input row stride.
// If OUTPUT_STRIDE is wired correctly, all 128 output rows are written and
// there's no out-of-bounds memory access.
void quantize_fp8_slice_launch(torch::Tensor a_bf16_wide,
                               torch::Tensor a_fp8_out,
                               torch::Tensor scale_out) {
  TORCH_CHECK(a_bf16_wide.dim() == 2 && a_bf16_wide.size(0) == 128 &&
                  a_bf16_wide.size(1) == 2176,
              "a_bf16_wide must be [128, 2176]");
  TORCH_CHECK(a_fp8_out.dim() == 2 && a_fp8_out.size(0) == 128 &&
                  a_fp8_out.size(1) == 1536,
              "a_fp8_out must be [128, 1536]");
  TORCH_CHECK(scale_out.dim() == 2 && scale_out.size(0) == 128 &&
                  scale_out.size(1) == 12,
              "scale_out must be [128, 12]");
  TORCH_CHECK(a_bf16_wide.scalar_type() == at::kBFloat16, "a_bf16 dtype");
  TORCH_CHECK(a_fp8_out.scalar_type() == at::kFloat8_e4m3fn, "fp8 dtype");
  TORCH_CHECK(scale_out.scalar_type() == at::kFloat, "scale dtype");
  TORCH_CHECK(a_bf16_wide.is_contiguous() && a_fp8_out.is_contiguous() &&
                  scale_out.is_contiguous(),
              "tensors must be contiguous");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  __nv_bfloat16 const *in_ptr =
      static_cast<__nv_bfloat16 const *>(a_bf16_wide.data_ptr());
  __nv_fp8_e4m3 *outq = static_cast<__nv_fp8_e4m3 *>(a_fp8_out.data_ptr());
  float *outs = static_cast<float *>(scale_out.data_ptr());
  void *args[] = {&in_ptr, &outq, &outs};
  cudaError_t err = cudaLaunchKernel(
      reinterpret_cast<void const *>(quantize_fp8_slice_test_kernel),
      /*grid=*/dim3(1, 128, 1),
      /*block=*/dim3(128, 1, 1),
      args,
      /*sharedMem=*/0,
      stream);
  TORCH_CHECK(err == cudaSuccess,
              "quantize_fp8_slice_test_kernel launch failed: ",
              cudaGetErrorString(err));
}

// Quantize-only entry point for the 7168-wide case (the QKV-a fusion bug).
// Mirrors MPK's quantize_fp8_layer registration for hidden_size=7168.
//   grid = (group_tiles=4, batch=128, 1), block = (128, 1, 1)
//   Each (block_x=group_tile, block_y=row) writes 14 groups × 128 cols of FP8
//   plus 14 scale values into the row's 56-cell scale slot.
void quantize_fp8_7168_launch(torch::Tensor a_bf16,
                              torch::Tensor a_fp8_out,
                              torch::Tensor scale_out) {
  TORCH_CHECK(a_bf16.dim() == 2 && a_bf16.size(0) == 128 &&
                  a_bf16.size(1) == 7168,
              "a_bf16 must be [128, 7168]");
  TORCH_CHECK(a_fp8_out.dim() == 2 && a_fp8_out.size(0) == 128 &&
                  a_fp8_out.size(1) == 7168,
              "a_fp8_out must be [128, 7168]");
  TORCH_CHECK(scale_out.dim() == 2 && scale_out.size(0) == 128 &&
                  scale_out.size(1) == 56,
              "scale_out must be [128, 56]");
  TORCH_CHECK(a_bf16.scalar_type() == at::kBFloat16, "a_bf16 dtype");
  TORCH_CHECK(a_fp8_out.scalar_type() == at::kFloat8_e4m3fn, "fp8 dtype");
  TORCH_CHECK(scale_out.scalar_type() == at::kFloat, "scale dtype");
  TORCH_CHECK(a_bf16.is_contiguous() && a_fp8_out.is_contiguous() &&
                  scale_out.is_contiguous(),
              "tensors must be contiguous");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  __nv_bfloat16 const *in_ptr =
      static_cast<__nv_bfloat16 const *>(a_bf16.data_ptr());
  __nv_fp8_e4m3 *outq = static_cast<__nv_fp8_e4m3 *>(a_fp8_out.data_ptr());
  float *outs = static_cast<float *>(scale_out.data_ptr());
  void *args[] = {&in_ptr, &outq, &outs};
  cudaError_t err = cudaLaunchKernel(
      reinterpret_cast<void const *>(quantize_fp8_f32_test_kernel),
      /*grid=*/dim3(4, 128, 1),
      /*block=*/dim3(128, 1, 1),
      args,
      /*sharedMem=*/0,
      stream);
  TORCH_CHECK(err == cudaSuccess,
              "quantize_fp8_7168 launch failed: ",
              cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemm_dense_smallm_launch",
        &fp8_gemm_dense_smallm_launch,
        "Direct launch of fp8_gemm_dense_smallm_sm100_task_impl<128,3>");
  m.def("fp8_gemm_dense_smallm_multi_cta_launch",
        &fp8_gemm_dense_smallm_multi_cta_launch,
        "Multi-CTA launch matching the persistent kernel grid pattern.");
  m.def("quantize_then_gemm_launch",
        &quantize_then_gemm_launch,
        "Run MPK's quantize_fp8 kernel + fp8_gemm_dense back-to-back.");
  m.def("quantize_fp8_7168_launch",
        &quantize_fp8_7168_launch,
        "Quantize a [128, 7168] BF16 buffer to FP8 e4m3 + float32 scale. "
        "Standalone variant matching MPK's qkv_a quantize registration.");
  m.def("quantize_fp8_slice_launch",
        &quantize_fp8_slice_launch,
        "Quantize a [128, 1536] slice of a wider [128, 2176] BF16 buffer "
        "with OUTPUT_STRIDE=1536 (H8 fix).");
}
