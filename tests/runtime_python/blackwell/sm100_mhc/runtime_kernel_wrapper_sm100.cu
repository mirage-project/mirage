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
#include "blackwell/affine_split_activation_sm100.cuh"
#include "blackwell/mul_sum_add_sm100.cuh"
#include "blackwell/mul_sum_add_with_outer_sm100.cuh"
#include "blackwell/sinkhorn.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <torch/extension.h>

using bf16_t = cutlass::bfloat16_t;

namespace {

constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// K2: affine + split + activation
// ============================================================================

template <typename T_in, int N>
__global__ __launch_bounds__(128) void affine_split_activation_kernel(
    T_in const *__restrict__ mixes,
    float const *__restrict__ scale,
    float const *__restrict__ base,
    float *__restrict__ h_pre,
    float *__restrict__ h_post,
    float *__restrict__ h_res_logits,
    int num_tokens) {
  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  constexpr int MIX_HC = N * N + 2 * N;
  kernel::affine_split_activation_sm100_task_impl<T_in, /*BATCH_SIZE=*/1, N>(
      mixes + token * MIX_HC,
      scale,
      base,
      h_pre + token * N,
      h_post + token * N,
      h_res_logits + token * (N * N));
}

template <typename T_in, int N>
void launch_affine_split_activation(T_in const *mixes,
                                    float const *scale,
                                    float const *base,
                                    float *h_pre,
                                    float *h_post,
                                    float *h_res_logits,
                                    int num_tokens,
                                    cudaStream_t stream) {
  dim3 const grid_dim(num_tokens, 1, 1);
  dim3 const block_dim(128, 1, 1);
  affine_split_activation_kernel<T_in, N><<<grid_dim, block_dim, 0, stream>>>(
      mixes, scale, base, h_pre, h_post, h_res_logits, num_tokens);
}

#define DISPATCH_K2_N(T_IN, MIXES, MIXES_DTYPE)                                \
  switch (n) {                                                                 \
    case 2:                                                                    \
      launch_affine_split_activation<T_IN, 2>(                                 \
          static_cast<T_IN const *>(MIXES),                                    \
          scale.data_ptr<float>(),                                             \
          base.data_ptr<float>(),                                              \
          h_pre.data_ptr<float>(),                                             \
          h_post.data_ptr<float>(),                                            \
          h_res_logits.data_ptr<float>(),                                      \
          num_tokens,                                                          \
          stream);                                                             \
      break;                                                                   \
    case 4:                                                                    \
      launch_affine_split_activation<T_IN, 4>(                                 \
          static_cast<T_IN const *>(MIXES),                                    \
          scale.data_ptr<float>(),                                             \
          base.data_ptr<float>(),                                              \
          h_pre.data_ptr<float>(),                                             \
          h_post.data_ptr<float>(),                                            \
          h_res_logits.data_ptr<float>(),                                      \
          num_tokens,                                                          \
          stream);                                                             \
      break;                                                                   \
    case 8:                                                                    \
      launch_affine_split_activation<T_IN, 8>(                                 \
          static_cast<T_IN const *>(MIXES),                                    \
          scale.data_ptr<float>(),                                             \
          base.data_ptr<float>(),                                              \
          h_pre.data_ptr<float>(),                                             \
          h_post.data_ptr<float>(),                                            \
          h_res_logits.data_ptr<float>(),                                      \
          num_tokens,                                                          \
          stream);                                                             \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(                                                             \
          false, "Unsupported n=", n, " (must be one of {2, 4, 8})");          \
  }

void affine_split_activation_sm100(torch::Tensor mixes,
                                   torch::Tensor scale,
                                   torch::Tensor base,
                                   torch::Tensor h_pre,
                                   torch::Tensor h_post,
                                   torch::Tensor h_res_logits,
                                   int n) {
  TORCH_CHECK(mixes.is_cuda(), "mixes must be CUDA");
  TORCH_CHECK(mixes.is_contiguous(), "mixes must be contiguous");
  TORCH_CHECK(mixes.dim() == 2, "mixes must be [num_tokens, mix_hc]");
  int const num_tokens = static_cast<int>(mixes.size(0));
  int const mix_hc = static_cast<int>(mixes.size(1));
  TORCH_CHECK(mix_hc == n * n + 2 * n, "mix_hc must equal n*n + 2*n");

  TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() &&
                  scale.scalar_type() == at::kFloat && scale.numel() == 3,
              "scale must be float32 [3] CUDA contiguous");
  TORCH_CHECK(base.is_cuda() && base.is_contiguous() &&
                  base.scalar_type() == at::kFloat && base.numel() == mix_hc,
              "base must be float32 [mix_hc] CUDA contiguous");
  TORCH_CHECK(h_pre.is_cuda() && h_pre.is_contiguous() &&
                  h_pre.scalar_type() == at::kFloat &&
                  h_pre.sizes() == torch::IntArrayRef({num_tokens, n}),
              "h_pre must be float32 [num_tokens, n] CUDA contiguous");
  TORCH_CHECK(h_post.is_cuda() && h_post.is_contiguous() &&
                  h_post.scalar_type() == at::kFloat &&
                  h_post.sizes() == torch::IntArrayRef({num_tokens, n}),
              "h_post must be float32 [num_tokens, n] CUDA contiguous");
  TORCH_CHECK(h_res_logits.is_cuda() && h_res_logits.is_contiguous() &&
                  h_res_logits.scalar_type() == at::kFloat &&
                  h_res_logits.sizes() ==
                      torch::IntArrayRef({num_tokens, n * n}),
              "h_res_logits must be float32 [num_tokens, n*n] CUDA contiguous");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(mixes.get_device());

  if (mixes.scalar_type() == at::kBFloat16) {
    DISPATCH_K2_N(bf16_t, mixes.data_ptr(), at::kBFloat16)
  } else if (mixes.scalar_type() == at::kFloat) {
    DISPATCH_K2_N(float, mixes.data_ptr(), at::kFloat)
  } else {
    TORCH_CHECK(false, "mixes must be bf16 or float32");
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "K2 launch error: ",
              cudaGetErrorString(err));
}

#undef DISPATCH_K2_N

// ============================================================================
// K4 reuse: mul_sum_add with zero residual
//
// y[token, c] = sum_i weight[token, i] * input[token, i, c]
// (residual = zero buffer)
// ============================================================================

template <typename T, int N, int C>
__global__ __launch_bounds__(256) void mul_sum_add_kernel(
    void const *input_ptr,
    void const *weight_ptr,
    void const *residual_ptr,
    void *output_ptr,
    int num_tokens) {
  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  T const *input = static_cast<T const *>(input_ptr) + token * N * C;
  float const *weight = static_cast<float const *>(weight_ptr) + token * N;
  T const *residual = static_cast<T const *>(residual_ptr) + token * C;
  T *output = static_cast<T *>(output_ptr) + token * C;
  kernel::mul_sum_add_sm100_task_impl<T,
                                      /*BATCH_SIZE=*/1,
                                      /*OUTPUT_SIZE=*/C,
                                      /*NUM_TOPK=*/N,
                                      /*OUTPUT_STRIDE=*/C>(
      input, weight, residual, output);
}

template <typename T, int N>
void launch_mul_sum_add(T const *input,
                        float const *weight,
                        T const *residual,
                        T *output,
                        int num_tokens,
                        int c,
                        cudaStream_t stream) {
  dim3 grid_dim(num_tokens, 1, 1);
  dim3 block_dim(256, 1, 1);
  switch (c) {
    case 128:
      mul_sum_add_kernel<T, N, 128>
          <<<grid_dim, block_dim, 0, stream>>>(
              input, weight, residual, output, num_tokens);
      break;
    case 1024:
      mul_sum_add_kernel<T, N, 1024>
          <<<grid_dim, block_dim, 0, stream>>>(
              input, weight, residual, output, num_tokens);
      break;
    case 4096:
      mul_sum_add_kernel<T, N, 4096>
          <<<grid_dim, block_dim, 0, stream>>>(
              input, weight, residual, output, num_tokens);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported C=",
                  c,
                  " (must be one of {128, 1024, 4096})");
  }
}

void mul_sum_add_sm100(torch::Tensor input,
                       torch::Tensor weight,
                       torch::Tensor residual,
                       torch::Tensor output,
                       int n) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.dim() == 3,
              "input must be bf16 [num_tokens, n, c] CUDA contiguous");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                  weight.scalar_type() == at::kFloat,
              "weight must be float32 [num_tokens, n] CUDA contiguous");
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16,
              "residual must be bf16 [num_tokens, c] CUDA contiguous");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() &&
                  output.scalar_type() == at::kBFloat16,
              "output must be bf16 [num_tokens, c] CUDA contiguous");

  int const num_tokens = static_cast<int>(input.size(0));
  TORCH_CHECK(input.size(1) == n, "input dim 1 must match n");
  int const c = static_cast<int>(input.size(2));
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({num_tokens, n}),
              "weight shape mismatch");
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({num_tokens, c}),
              "residual shape mismatch");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({num_tokens, c}),
              "output shape mismatch");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  bf16_t const *input_ptr =
      reinterpret_cast<bf16_t const *>(input.data_ptr());
  bf16_t const *residual_ptr =
      reinterpret_cast<bf16_t const *>(residual.data_ptr());
  bf16_t *output_ptr =
      reinterpret_cast<bf16_t *>(output.data_ptr());

  switch (n) {
    case 2:
      launch_mul_sum_add<bf16_t, 2>(input_ptr,
                                           weight.data_ptr<float>(),
                                           residual_ptr,
                                           output_ptr,
                                           num_tokens,
                                           c,
                                           stream);
      break;
    case 4:
      launch_mul_sum_add<bf16_t, 4>(input_ptr,
                                           weight.data_ptr<float>(),
                                           residual_ptr,
                                           output_ptr,
                                           num_tokens,
                                           c,
                                           stream);
      break;
    case 8:
      launch_mul_sum_add<bf16_t, 8>(input_ptr,
                                           weight.data_ptr<float>(),
                                           residual_ptr,
                                           output_ptr,
                                           num_tokens,
                                           c,
                                           stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported n=", n);
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "K4 launch error: ",
              cudaGetErrorString(err));
}

// ============================================================================
// K5: mul_sum_add with outer-product residual (post[k] * x[c])
// ============================================================================

template <typename T, int N, int C>
__global__ __launch_bounds__(256) void mul_sum_add_with_outer_kernel(
    void const *residual_ptr,
    void const *x_ptr,
    void const *comb_ptr,
    void const *post_ptr,
    void *output_ptr,
    int num_tokens) {
  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  T const *residual =
      static_cast<T const *>(residual_ptr) + token * N * C;
  T const *x = static_cast<T const *>(x_ptr) + token * C;
  float const *comb =
      static_cast<float const *>(comb_ptr) + token * N * N;
  float const *post = static_cast<float const *>(post_ptr) + token * N;
  T *output = static_cast<T *>(output_ptr) + token * N * C;
  kernel::mul_sum_add_with_outer_sm100_task_impl<T,
                                                 /*BATCH_SIZE=*/1,
                                                 /*OUTPUT_SIZE=*/C,
                                                 /*NUM_TOPK=*/N,
                                                 /*OUTPUT_STRIDE=*/C>(
      residual, x, comb, post, output);
}

template <typename T, int N>
void launch_mul_sum_add_with_outer(T const *residual,
                                   T const *x,
                                   float const *comb,
                                   float const *post,
                                   T *output,
                                   int num_tokens,
                                   int c,
                                   cudaStream_t stream) {
  dim3 grid_dim(num_tokens, 1, 1);
  dim3 block_dim(256, 1, 1);
  switch (c) {
    case 128:
      mul_sum_add_with_outer_kernel<T, N, 128><<<grid_dim, block_dim, 0, stream>>>(
          residual, x, comb, post, output, num_tokens);
      break;
    case 1024:
      mul_sum_add_with_outer_kernel<T, N, 1024><<<grid_dim, block_dim, 0, stream>>>(
          residual, x, comb, post, output, num_tokens);
      break;
    case 4096:
      mul_sum_add_with_outer_kernel<T, N, 4096><<<grid_dim, block_dim, 0, stream>>>(
          residual, x, comb, post, output, num_tokens);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported C=",
                  c,
                  " (must be one of {128, 1024, 4096})");
  }
}

void mul_sum_add_with_outer_sm100(torch::Tensor residual,
                                  torch::Tensor x,
                                  torch::Tensor comb,
                                  torch::Tensor post,
                                  torch::Tensor output,
                                  int n) {
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.dim() == 3,
              "residual must be bf16 [num_tokens, n, c] CUDA contiguous");
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                  x.scalar_type() == at::kBFloat16 && x.dim() == 2,
              "x must be bf16 [num_tokens, c] CUDA contiguous");
  TORCH_CHECK(comb.is_cuda() && comb.is_contiguous() &&
                  comb.scalar_type() == at::kFloat,
              "comb must be float32 [num_tokens, n, n] CUDA contiguous");
  TORCH_CHECK(post.is_cuda() && post.is_contiguous() &&
                  post.scalar_type() == at::kFloat,
              "post must be float32 [num_tokens, n] CUDA contiguous");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() &&
                  output.scalar_type() == at::kBFloat16,
              "output must be bf16 [num_tokens, n, c] CUDA contiguous");

  int const num_tokens = static_cast<int>(residual.size(0));
  TORCH_CHECK(residual.size(1) == n, "residual dim 1 must match n");
  int const c = static_cast<int>(residual.size(2));
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({num_tokens, c}),
              "x shape mismatch");
  TORCH_CHECK(comb.sizes() == torch::IntArrayRef({num_tokens, n, n}),
              "comb shape mismatch");
  TORCH_CHECK(post.sizes() == torch::IntArrayRef({num_tokens, n}),
              "post shape mismatch");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({num_tokens, n, c}),
              "output shape mismatch");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  bf16_t const *residual_ptr =
      reinterpret_cast<bf16_t const *>(residual.data_ptr());
  bf16_t const *x_ptr =
      reinterpret_cast<bf16_t const *>(x.data_ptr());
  bf16_t *output_ptr =
      reinterpret_cast<bf16_t *>(output.data_ptr());

  switch (n) {
    case 2:
      launch_mul_sum_add_with_outer<bf16_t, 2>(residual_ptr,
                                                      x_ptr,
                                                      comb.data_ptr<float>(),
                                                      post.data_ptr<float>(),
                                                      output_ptr,
                                                      num_tokens,
                                                      c,
                                                      stream);
      break;
    case 4:
      launch_mul_sum_add_with_outer<bf16_t, 4>(residual_ptr,
                                                      x_ptr,
                                                      comb.data_ptr<float>(),
                                                      post.data_ptr<float>(),
                                                      output_ptr,
                                                      num_tokens,
                                                      c,
                                                      stream);
      break;
    case 8:
      launch_mul_sum_add_with_outer<bf16_t, 8>(residual_ptr,
                                                      x_ptr,
                                                      comb.data_ptr<float>(),
                                                      post.data_ptr<float>(),
                                                      output_ptr,
                                                      num_tokens,
                                                      c,
                                                      stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported n=", n);
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "K5 launch error: ",
              cudaGetErrorString(err));
}

// ============================================================================
// K3 reuse: sinkhorn (already in sinkhorn.cuh)
// ============================================================================

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

void sinkhorn_sm100(torch::Tensor comb_res_mix,
                    torch::Tensor comb_res_mix_out,
                    int repeat,
                    double eps,
                    int token_block_size) {
  TORCH_CHECK(comb_res_mix.is_cuda() && comb_res_mix.is_contiguous() &&
                  comb_res_mix.scalar_type() == at::kFloat &&
                  comb_res_mix.dim() == 3,
              "comb_res_mix must be float32 [num_tokens, n, n] CUDA contiguous");
  TORCH_CHECK(comb_res_mix.size(1) == comb_res_mix.size(2),
              "sinkhorn matrix must be square");
  TORCH_CHECK(comb_res_mix_out.sizes() == comb_res_mix.sizes() &&
                  comb_res_mix_out.is_cuda() &&
                  comb_res_mix_out.is_contiguous() &&
                  comb_res_mix_out.scalar_type() == at::kFloat,
              "comb_res_mix_out shape/dtype mismatch");
  TORCH_CHECK(repeat >= 1, "repeat must be >= 1");

  int const num_tokens = static_cast<int>(comb_res_mix.size(0));
  int const hidden_size = static_cast<int>(comb_res_mix.size(1));
  float const eps_f = static_cast<float>(eps);
  float const *input_ptr = comb_res_mix.data_ptr<float>();
  float *output_ptr = comb_res_mix_out.data_ptr<float>();
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(comb_res_mix.get_device());

#define SINKHORN_DISPATCH_HIDDEN(TBS)                                          \
  switch (hidden_size) {                                                       \
    case 2:                                                                    \
      launch_sinkhorn<TBS, 2>(                                                 \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    case 4:                                                                    \
      launch_sinkhorn<TBS, 4>(                                                 \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    case 8:                                                                    \
      launch_sinkhorn<TBS, 8>(                                                 \
          input_ptr, output_ptr, num_tokens, repeat, eps_f, stream);           \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported hidden_size=", hidden_size);             \
  }

  switch (token_block_size) {
    case 1:
      SINKHORN_DISPATCH_HIDDEN(1)
      break;
    case 2:
      SINKHORN_DISPATCH_HIDDEN(2)
      break;
    case 4:
      SINKHORN_DISPATCH_HIDDEN(4)
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported token_block_size=",
                  token_block_size,
                  " (must be one of {1, 2, 4})");
  }
#undef SINKHORN_DISPATCH_HIDDEN

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "Sinkhorn launch error: ",
              cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine_split_activation_sm100",
        &affine_split_activation_sm100,
        py::arg("mixes"),
        py::arg("scale"),
        py::arg("base"),
        py::arg("h_pre"),
        py::arg("h_post"),
        py::arg("h_res_logits"),
        py::arg("n"),
        "mHC K2: affine + split + sigmoid/2*sigmoid/identity");

  m.def("mul_sum_add_sm100",
        &mul_sum_add_sm100,
        py::arg("input"),
        py::arg("weight"),
        py::arg("residual"),
        py::arg("output"),
        py::arg("n"),
        "mHC K4 (residual=zeros for plain reduction)");

  m.def("mul_sum_add_with_outer_sm100",
        &mul_sum_add_with_outer_sm100,
        py::arg("residual"),
        py::arg("x"),
        py::arg("comb"),
        py::arg("post"),
        py::arg("output"),
        py::arg("n"),
        "mHC K5: residual mix + post outer-product fused");

  m.def("sinkhorn_sm100",
        &sinkhorn_sm100,
        py::arg("comb_res_mix"),
        py::arg("comb_res_mix_out"),
        py::arg("repeat") = 20,
        py::arg("eps") = 1e-9,
        py::arg("token_block_size") = 1,
        "mHC K3: Sinkhorn-Knopp normalization (reused)");
}
