/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Standalone numeric test wrapper for the per-head dense FP8 BMM
// (linear_fp8_bmm_dense_sm100_task_impl<128, 3, 2>). Launches grid=(1, H, 1):
// each CTA computes one head's GEMM
//     C[:, h, :] = A[:, h, :] @ B[h, :, :]^T
// with per-head TMA descriptors + per-head float32 scale base offsets,
// exactly mirroring the production runtime wiring (per-head base via the
// TBGraph partition map + sa_row_stride = H*nk, C_row_stride = H*N).
//
// Layouts (contiguous, matching the builder/demo dense-BMM repack):
//   A (input)        float8_e4m3 [M, H, K]
//   B (weight)       float8_e4m3 [H, N, K]
//   sa (act scale)   float32     [M, H, K/128]   row-major
//   sb (wt  scale)   float32     [H, 1, K/128]   row-major (N=128 => 1 N-block)
//   C (output)       bfloat16    [M, H, N]

#include "blackwell/linear_fp8_bmm_dense_sm100.cuh"
#include <cute/numeric/numeric_types.hpp>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace {

void check_driver_success(CUresult result, char const *what) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  char const *en = nullptr;
  char const *es = nullptr;
  cuGetErrorName(result, &en);
  cuGetErrorString(result, &es);
  TORCH_CHECK(false, what, " failed: ", (en ? en : "?"), ": ", (es ? es : "?"));
}

// 2-D TMA descriptor over an FP8 (uint8) row-major tensor.
//   gd  = {K, outer}            (inner dim first)
//   gs  = {row_stride_bytes}    (gmem byte stride between consecutive rows)
//   bd  = {128, 128}            (BK x OUTER_BOX)
// `base` is the per-head base pointer (head offset already applied).
CUtensorMap make_tma_fp8_2d(void *base, int K, int outer, uint64_t row_bytes) {
  CUtensorMap tm{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(K),
                              static_cast<cuuint64_t>(outer)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {128u, 128u};
  const cuuint32_t es[2] = {1, 1};
  check_driver_success(
      cuTensorMapEncodeTiled(&tm,
                             CU_TENSOR_MAP_DATA_TYPE_UINT8,
                             2,
                             base,
                             dims,
                             strides,
                             box,
                             es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tm;
}

// One CTA per head. blockIdx.y = head index. d_ta/d_tb are arrays of H
// descriptors (one per head); the float32 scale base pointers + row strides
// reproduce the production runtime offsets.
__global__ void __launch_bounds__(256)
    bmm_dense_test_kernel(CUtensorMap const *d_ta,
                          CUtensorMap const *d_tb,
                          float const *sa_base,
                          float const *sb_base,
                          __nv_bfloat16 *C_base,
                          int M,
                          int N,
                          int K,
                          int H) {
  int const h = blockIdx.y;
  int const nk = (K + 127) / 128;
  // Per-head bases: sa is [M, H, nk] -> +h*nk; sb is [H, 1, nk] -> +h*nk;
  // C is [M, H, N] -> +h*N.
  float const *sa_h = sa_base + (long long)h * nk;
  float const *sb_h = sb_base + (long long)h * nk;
  __nv_bfloat16 *C_h = C_base + (long long)h * N;
  kernel::linear_fp8_bmm_dense::linear_fp8_bmm_dense_sm100_task_impl<128, 3, 2>(
      &d_ta[h],
      &d_tb[h],
      sa_h,
      sb_h,
      C_h,
      M,
      N,
      K,
      /*sa_row_stride=*/H * nk,
      /*C_row_stride=*/H * N);
}

} // namespace

// a_fp8: [M, H, K]  b_fp8: [H, N, K]  sa: [M, H, K/128]  sb: [H, 1, K/128]
// out:   [M, H, N]  (bf16, caller zero-inits)
void linear_fp8_bmm_dense_launch(torch::Tensor a_fp8,
                                 torch::Tensor b_fp8,
                                 torch::Tensor sa,
                                 torch::Tensor sb,
                                 torch::Tensor output) {
  TORCH_CHECK(a_fp8.dim() == 3 && b_fp8.dim() == 3 && output.dim() == 3,
              "a_fp8 [M,H,K], b_fp8 [H,N,K], output [M,H,N] must be 3D");
  TORCH_CHECK(sa.dim() == 3 && sb.dim() == 3, "sa [M,H,nk], sb [H,1,nk] 3D");
  TORCH_CHECK(a_fp8.scalar_type() == at::kFloat8_e4m3fn, "a_fp8 e4m3");
  TORCH_CHECK(b_fp8.scalar_type() == at::kFloat8_e4m3fn, "b_fp8 e4m3");
  TORCH_CHECK(sa.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat,
              "sa/sb float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output bf16");
  TORCH_CHECK(a_fp8.is_contiguous() && b_fp8.is_contiguous() &&
                  sa.is_contiguous() && sb.is_contiguous() &&
                  output.is_contiguous(),
              "contiguous");

  int const M = static_cast<int>(a_fp8.size(0));
  int const H = static_cast<int>(a_fp8.size(1));
  int const K = static_cast<int>(a_fp8.size(2));
  int const N = static_cast<int>(b_fp8.size(1));
  TORCH_CHECK(b_fp8.size(0) == H && b_fp8.size(2) == K, "b_fp8 [H,N,K]");
  TORCH_CHECK(N == 128, "this test fixes per-head N=128 (=BN)");
  TORCH_CHECK(K % 128 == 0, "K multiple of 128");
  TORCH_CHECK(output.size(0) == M && output.size(1) == H && output.size(2) == N,
              "output [M,H,N]");

  // Build H per-head TMA descriptors for A and B.
  // A per head: base = a + h*K (FP8 bytes), gd = {K, M}, row stride = H*K bytes.
  // B per head: base = b + h*N*K, gd = {K, N}, row stride = K bytes.
  uint8_t *a_ptr = static_cast<uint8_t *>(a_fp8.data_ptr());
  uint8_t *b_ptr = static_cast<uint8_t *>(b_fp8.data_ptr());
  std::vector<CUtensorMap> h_ta(H), h_tb(H);
  for (int h = 0; h < H; h++) {
    h_ta[h] = make_tma_fp8_2d(a_ptr + (long long)h * K, K, M,
                              (uint64_t)H * K);
    h_tb[h] = make_tma_fp8_2d(b_ptr + (long long)h * N * K, K, N, (uint64_t)K);
  }
  CUtensorMap *d_ta = nullptr, *d_tb = nullptr;
  TORCH_CHECK(cudaMalloc(&d_ta, H * sizeof(CUtensorMap)) == cudaSuccess, "m1");
  TORCH_CHECK(cudaMalloc(&d_tb, H * sizeof(CUtensorMap)) == cudaSuccess, "m2");
  TORCH_CHECK(cudaMemcpy(d_ta, h_ta.data(), H * sizeof(CUtensorMap),
                         cudaMemcpyHostToDevice) == cudaSuccess,
              "c1");
  TORCH_CHECK(cudaMemcpy(d_tb, h_tb.data(), H * sizeof(CUtensorMap),
                         cudaMemcpyHostToDevice) == cudaSuccess,
              "c2");

  constexpr int kSmem =
      kernel::linear_fp8_bmm_dense::linear_fp8_bmm_dense_smem_size<128, 3, 2>();
  TORCH_CHECK(cudaFuncSetAttribute(bmm_dense_test_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   kSmem) == cudaSuccess,
              "setattr smem");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  CUtensorMap const *arg_ta = d_ta;
  CUtensorMap const *arg_tb = d_tb;
  float const *arg_sa = static_cast<float const *>(sa.data_ptr());
  float const *arg_sb = static_cast<float const *>(sb.data_ptr());
  __nv_bfloat16 *arg_C = static_cast<__nv_bfloat16 *>(output.data_ptr());
  int aM = M, aN = N, aK = K, aH = H;
  void *params[] = {&arg_ta, &arg_tb, &arg_sa, &arg_sb, &arg_C,
                    &aM,     &aN,     &aK,     &aH};
  cudaError_t le = cudaLaunchKernel(
      reinterpret_cast<void const *>(bmm_dense_test_kernel),
      dim3(1, H, 1), dim3(256, 1, 1), params, kSmem, stream);
  TORCH_CHECK(le == cudaSuccess, "launch: ", cudaGetErrorString(le));
  cudaError_t err = cudaStreamSynchronize(stream);
  cudaFree(d_ta);
  cudaFree(d_tb);
  TORCH_CHECK(err == cudaSuccess, "sync: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fp8_bmm_dense_launch", &linear_fp8_bmm_dense_launch,
        "Per-head dense FP8 BMM (grid=(1,H,1)), float32 block scales.");
}
