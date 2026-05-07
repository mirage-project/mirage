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
#include "blackwell/sinkhorn.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// One thread = one 4x4 matrix; grid strides over all tokens.
__global__ __launch_bounds__(256) void sinkhorn_sm100_kernel(
    float const *__restrict__ comb_res_mix,
    float *__restrict__ comb_res_mix_out,
    int num_tokens,
    int repeat,
    float eps) {
  constexpr int token_stride = 16; // 4 * 4
  kernel::sinkhorn_task_impl<token_stride, token_stride>(
      comb_res_mix, comb_res_mix_out, num_tokens, repeat, eps);
}

void launch_sinkhorn(float const *comb_res_mix,
                     float *comb_res_mix_out,
                     int num_tokens,
                     int repeat,
                     float eps,
                     cudaStream_t stream) {
  constexpr int kBlock = 256;
  int const grid = ceil_div(num_tokens, kBlock);
  dim3 const grid_dim(grid, 1, 1);
  dim3 const block_dim(kBlock, 1, 1);
  sinkhorn_sm100_kernel<<<grid_dim, block_dim, 0, stream>>>(
      comb_res_mix, comb_res_mix_out, num_tokens, repeat, eps);
}

} // namespace

void sinkhorn_sm100(torch::Tensor comb_res_mix,
                    torch::Tensor comb_res_mix_out,
                    int repeat,
                    double eps) {
  TORCH_CHECK(comb_res_mix.is_cuda(), "comb_res_mix must be CUDA");
  TORCH_CHECK(comb_res_mix_out.is_cuda(), "comb_res_mix_out must be CUDA");
  TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat,
              "comb_res_mix must be float32");
  TORCH_CHECK(comb_res_mix_out.scalar_type() == at::kFloat,
              "comb_res_mix_out must be float32");
  TORCH_CHECK(comb_res_mix.dim() == 3,
              "comb_res_mix must be [num_tokens, 4, 4]");
  TORCH_CHECK(comb_res_mix_out.sizes() == comb_res_mix.sizes(),
              "comb_res_mix_out must match comb_res_mix shape");
  TORCH_CHECK(comb_res_mix.is_contiguous(), "comb_res_mix must be contiguous");
  TORCH_CHECK(comb_res_mix_out.is_contiguous(),
              "comb_res_mix_out must be contiguous");
  TORCH_CHECK(comb_res_mix.size(1) == 4 && comb_res_mix.size(2) == 4,
              "sinkhorn matrix must be 4x4 (mHC)");
  TORCH_CHECK(repeat >= 1, "repeat must be >= 1");

  int const num_tokens = static_cast<int>(comb_res_mix.size(0));
  float const eps_f = static_cast<float>(eps);
  float const *input_ptr = comb_res_mix.data_ptr<float>();
  float *output_ptr = comb_res_mix_out.data_ptr<float>();
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(comb_res_mix.get_device());

  launch_sinkhorn(input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch error: ",
              cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sinkhorn_sm100",
        &sinkhorn_sm100,
        py::arg("comb_res_mix"),
        py::arg("comb_res_mix_out"),
        py::arg("repeat") = 20,
        py::arg("eps") = 1e-9,
        "mHC Sinkhorn-Knopp forward kernel (4x4)");
}
