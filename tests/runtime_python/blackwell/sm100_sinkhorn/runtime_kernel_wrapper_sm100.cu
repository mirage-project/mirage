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

template <int TOKEN_BLOCK_SIZE, int HIDDEN_SIZE>
__global__ __launch_bounds__(256) void sinkhorn_sm100_kernel(
    float const *__restrict__ comb_res_mix,
    float *__restrict__ comb_res_mix_out,
    int num_tokens,
    int repeat,
    float eps) {
  int const token_begin = blockIdx.x * TOKEN_BLOCK_SIZE;
  int const remaining_tokens = num_tokens - token_begin;
  int const valid_tokens = remaining_tokens < TOKEN_BLOCK_SIZE
                               ? remaining_tokens
                               : TOKEN_BLOCK_SIZE;
  if (valid_tokens <= 0) {
    return;
  }

  int constexpr token_stride = HIDDEN_SIZE * HIDDEN_SIZE;
  kernel::sinkhorn_task_impl<TOKEN_BLOCK_SIZE,
                             HIDDEN_SIZE,
                             token_stride,
                             token_stride>(
      comb_res_mix + token_begin * token_stride,
      comb_res_mix_out + token_begin * token_stride,
      valid_tokens,
      repeat,
      eps);
}

template <int TOKEN_BLOCK_SIZE, int HIDDEN_SIZE>
void launch_sinkhorn(float const *comb_res_mix,
                     float *comb_res_mix_out,
                     int num_tokens,
                     int repeat,
                     float eps,
                     cudaStream_t stream) {
  dim3 const grid_dim(ceil_div(num_tokens, TOKEN_BLOCK_SIZE), 1, 1);
  dim3 const block_dim(256, 1, 1);
  size_t const smem_bytes =
      (TOKEN_BLOCK_SIZE * HIDDEN_SIZE * HIDDEN_SIZE +
       2 * TOKEN_BLOCK_SIZE * HIDDEN_SIZE) *
      sizeof(float);
  sinkhorn_sm100_kernel<TOKEN_BLOCK_SIZE, HIDDEN_SIZE>
      <<<grid_dim, block_dim, smem_bytes, stream>>>(
          comb_res_mix, comb_res_mix_out, num_tokens, repeat, eps);
}

#define DISPATCH_SINKHORN_HIDDEN(TOKEN_BLOCK_SIZE)                            \
  switch (hidden_size) {                                                       \
    case 2:                                                                    \
      launch_sinkhorn<TOKEN_BLOCK_SIZE, 2>(                                    \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    case 4:                                                                    \
      launch_sinkhorn<TOKEN_BLOCK_SIZE, 4>(                                    \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    case 8:                                                                    \
      launch_sinkhorn<TOKEN_BLOCK_SIZE, 8>(                                    \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    case 16:                                                                   \
      launch_sinkhorn<TOKEN_BLOCK_SIZE, 16>(                                   \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    case 32:                                                                   \
      launch_sinkhorn<TOKEN_BLOCK_SIZE, 32>(                                   \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "Unsupported hidden_size=",                                  \
                  hidden_size,                                                 \
                  " (must be one of {2, 4, 8, 16, 32})");                     \
  }

} // namespace

void sinkhorn_sm100(torch::Tensor comb_res_mix,
                    torch::Tensor comb_res_mix_out,
                    int repeat,
                    double eps,
                    int token_block_size) {
  TORCH_CHECK(comb_res_mix.is_cuda(), "comb_res_mix must be CUDA");
  TORCH_CHECK(comb_res_mix_out.is_cuda(), "comb_res_mix_out must be CUDA");
  TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat,
              "comb_res_mix must be float32");
  TORCH_CHECK(comb_res_mix_out.scalar_type() == at::kFloat,
              "comb_res_mix_out must be float32");
  TORCH_CHECK(comb_res_mix.dim() == 3,
              "comb_res_mix must be [num_tokens, hidden_size, hidden_size]");
  TORCH_CHECK(comb_res_mix_out.sizes() == comb_res_mix.sizes(),
              "comb_res_mix_out must match comb_res_mix shape");
  TORCH_CHECK(comb_res_mix.is_contiguous(), "comb_res_mix must be contiguous");
  TORCH_CHECK(comb_res_mix_out.is_contiguous(),
              "comb_res_mix_out must be contiguous");
  TORCH_CHECK(comb_res_mix.size(1) == comb_res_mix.size(2),
              "sinkhorn matrix must be square");
  TORCH_CHECK(repeat >= 1, "repeat must be >= 1");

  int const num_tokens = static_cast<int>(comb_res_mix.size(0));
  int const hidden_size = static_cast<int>(comb_res_mix.size(1));
  float const eps_f = static_cast<float>(eps);
  float const *input_ptr = comb_res_mix.data_ptr<float>();
  float *output_ptr = comb_res_mix_out.data_ptr<float>();
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(comb_res_mix.get_device());

  switch (token_block_size) {
    case 1:
      DISPATCH_SINKHORN_HIDDEN(1)
      break;
    case 2:
      DISPATCH_SINKHORN_HIDDEN(2)
      break;
    case 4:
      DISPATCH_SINKHORN_HIDDEN(4)
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported token_block_size=",
                  token_block_size,
                  " (must be one of {1, 2, 4})");
  }

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
        py::arg("token_block_size") = 1,
        "mHC Sinkhorn-Knopp forward kernel");
}
