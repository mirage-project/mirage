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
// Mirror sm100_linear's include order so MPK/CuTe machinery is set up before
// standard CUDA + Torch headers (avoids cute::prefetch / UMMA clashes).
#include "blackwell/task_header.cuh"
#include "hopper/tma_2d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

#include "blackwell/mHC_affine_split_activation.cuh"
#include "blackwell/mHC_mul_sum_add_with_outer.cuh"
#include "blackwell/mHC_rmsnorm.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/bfloat16.h>

using bf16_t = cutlass::bfloat16_t;
using mpk_bf16 = cute::bfloat16_t;

namespace {

constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// K1 (rmsnorm half): per-token RMSNorm with implicit unit weight.
//   y[t, i] = x[t, i] * rsqrt(mean(x[t, :]**2) + eps)
// One block per token. Pair with a torch / linear-mpk matmul to get full K1.
// ============================================================================

template <typename T_in, typename T_out, int HIDDEN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void mHC_rmsnorm_kernel(
    T_in const *__restrict__ x,
    T_out *__restrict__ y,
    int num_tokens,
    float eps) {
  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  kernel::mHC_rmsnorm_task_impl<T_in, T_out, HIDDEN, BLOCK_THREADS>(
      x + token * HIDDEN, y + token * HIDDEN, eps);
}

template <typename T_in, typename T_out, int HIDDEN>
void launch_mHC_rmsnorm(T_in const *x,
                        T_out *y,
                        int num_tokens,
                        float eps,
                        cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 256;
  dim3 const grid_dim(num_tokens, 1, 1);
  dim3 const block_dim(BLOCK_THREADS, 1, 1);
  mHC_rmsnorm_kernel<T_in, T_out, HIDDEN, BLOCK_THREADS>
      <<<grid_dim, block_dim, 0, stream>>>(x, y, num_tokens, eps);
}

#define DISPATCH_K1_HIDDEN(T_IN, T_OUT, IN_PTR, OUT_PTR)                       \
  switch (hidden) {                                                            \
    case 256:                                                                  \
      launch_mHC_rmsnorm<T_IN, T_OUT, 256>(                                    \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    case 512:                                                                  \
      launch_mHC_rmsnorm<T_IN, T_OUT, 512>(                                    \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    case 1024:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 1024>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    case 2048:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 2048>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    case 4096:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 4096>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    case 8192:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 8192>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    case 16384:                                                                \
      launch_mHC_rmsnorm<T_IN, T_OUT, 16384>(                                  \
          IN_PTR, OUT_PTR, num_tokens, eps_f, stream);                         \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "Unsupported hidden=",                                       \
                  hidden,                                                      \
                  " (must be one of {256,512,1024,2048,4096,8192,16384})");    \
  }

void mHC_rmsnorm(torch::Tensor x, torch::Tensor y, double eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
              "x must be 2D [num_tokens, hidden] CUDA contiguous");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.sizes() == x.sizes(),
              "y must match x shape, CUDA contiguous");

  int const num_tokens = static_cast<int>(x.size(0));
  int const hidden = static_cast<int>(x.size(1));
  float const eps_f = static_cast<float>(eps);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

  if (x.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kBFloat16) {
    auto *in_ptr = reinterpret_cast<bf16_t const *>(x.data_ptr());
    auto *out_ptr = reinterpret_cast<bf16_t *>(y.data_ptr());
    DISPATCH_K1_HIDDEN(bf16_t, bf16_t, in_ptr, out_ptr)
  } else if (x.scalar_type() == at::kFloat &&
             y.scalar_type() == at::kBFloat16) {
    auto const *in_ptr = x.data_ptr<float>();
    auto *out_ptr = reinterpret_cast<bf16_t *>(y.data_ptr());
    DISPATCH_K1_HIDDEN(float, bf16_t, in_ptr, out_ptr)
  } else if (x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat) {
    auto const *in_ptr = x.data_ptr<float>();
    auto *out_ptr = y.data_ptr<float>();
    DISPATCH_K1_HIDDEN(float, float, in_ptr, out_ptr)
  } else {
    TORCH_CHECK(false,
                "Unsupported (x,y) dtype combination; supported: "
                "(bf16,bf16), (fp32,bf16), (fp32,fp32)");
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "K1 rmsnorm launch error: ",
              cudaGetErrorString(err));
}

#undef DISPATCH_K1_HIDDEN

// ============================================================================
// K1 (linear half): batched skinny matmul via linear_sm100_mpk_task_impl.
//
// Computes y[bs, OUT_PAD] = x[bs, RED] @ w[OUT_PAD, RED]^T, where the host
// tiles `bs` into MMA_N=16 row chunks (one launch per chunk, fresh TMA descs)
// and pads the output dim to OUT_PAD=128 (a multiple of MMA_M=128). Caller
// presents pre-padded weight + output buffers; the unpadded output slice is
// the user's mix_hc.
// ============================================================================

template <int OUTPUT_SIZE, int REDUCTION_SIZE, int MMA_M, int MMA_N>
__global__ __launch_bounds__(256, 1) void mHC_linear_wrapper(
    void *tma_a_desc_ptr,
    void *tma_b_desc_ptr,
    void *tma_out_desc_ptr) {
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr int OUTPUT_ATOM_REPEAT_COL = 1;

  using TMA_B = kernel::tma::tma_2d<mpk_bf16,
                                    B,
                                    M,
                                    S,
                                    MMA_N,                     /*GMEM_ROW_*/
                                    REDUCTION_SIZE,            /*GMEM_COL_*/
                                    MMA_N,                     /*SMEM_ROW_*/
                                    TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                                    REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                                    1,                         /*GMEM_STRIDE_COL_*/
                                    1,                         /*SMEM_REPEAT_ROW_*/
                                    TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                                    MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                                    true>;
  using TMA_A = kernel::tma::tma_2d<mpk_bf16,
                                    B,
                                    M,
                                    S,
                                    OUTPUT_SIZE,               /*GMEM_ROW_*/
                                    REDUCTION_SIZE,            /*GMEM_COL_*/
                                    MMA_M,                     /*SMEM_ROW_*/
                                    TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                                    REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                                    1,                         /*GMEM_STRIDE_COL_*/
                                    1,                         /*SMEM_REPEAT_ROW_*/
                                    TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                                    MMA_M * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                                    true>;
  using TMA_OUT = kernel::tma::tma_2d<mpk_bf16,
                                      0,
                                      M,
                                      S,
                                      MMA_N,                  /*GMEM_ROW_*/
                                      OUTPUT_SIZE,            /*GMEM_COL_*/
                                      MMA_N,                  /*SMEM_ROW_*/
                                      MMA_M,                  /*SMEM_COL_*/
                                      OUTPUT_SIZE,            /*GMEM_STRIDE_ROW_*/
                                      1,                      /*GMEM_STRIDE_COL_*/
                                      1,                      /*SMEM_REPEAT_ROW_*/
                                      OUTPUT_ATOM_REPEAT_COL, /*SMEM_REPEAT_COL_*/
                                      MMA_N * MMA_M,          /*SMEM_STRIDE_*/
                                      true>;

  TMA_A tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
  TMA_B tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
  TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

  // No-bias path: build a degenerate Bias tensor with the right shape.
  cute::Layout layout_Bias = cute::make_layout(
      cute::make_shape(MMA_N, OUTPUT_SIZE),
      cute::make_stride(OUTPUT_SIZE, cute::Int<1>{}));
  cute::Tensor mBias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<mpk_bf16 *>(nullptr)), layout_Bias);

  kernel::linear_sm100_mpk_task_impl<mpk_bf16,
                                     TMA_A,
                                     TMA_B,
                                     decltype(mBias),
                                     TMA_OUT,
                                     MMA_M,
                                     MMA_N,
                                     /*BATCH_SIZE=*/MMA_N,
                                     OUTPUT_SIZE,
                                     REDUCTION_SIZE,
                                     /*NOBIAS=*/true,
                                     /*SplitK=*/false>(
      tma_a, tma_b, mBias, tma_out);
}

template <int OUT_PAD, int RED>
void launch_mHC_linear(void *input_ptr,
                       void *weight_ptr,
                       void *output_ptr,
                       int bs,
                       cudaStream_t stream) {
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr size_t TILE_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr int smemBytes = 224 * 1024;

  // Weight TMA: shape [OUT_PAD, RED], reused across all batch tiles.
  CUtensorMap host_w_desc;
  uint64_t w_gmem_shape[2] = {static_cast<uint64_t>(OUT_PAD),
                              static_cast<uint64_t>(RED)};
  uint64_t w_gmem_stride[2] = {1, static_cast<uint64_t>(RED)};
  uint32_t w_smem_shape[2] = {static_cast<uint32_t>(MMA_M),
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  mirage::runtime::fill_tma_desc<mpk_bf16, B, M, S, 2>(
      &host_w_desc,
      static_cast<mpk_bf16 *>(weight_ptr),
      w_gmem_shape,
      w_gmem_stride,
      w_smem_shape,
      1,
      TILE_REPEAT_COL);

  CUtensorMap *desc_w_ptr;
  cudaMalloc(&desc_w_ptr, sizeof(CUtensorMap));
  cudaMemcpyAsync(desc_w_ptr,
                  &host_w_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);

  int const num_tiles = ceil_div(bs, MMA_N);
  // Allocate a strip of input + output descriptors so each tile launch points
  // to its own slice of the batch.
  CUtensorMap *desc_i_buf;
  CUtensorMap *desc_o_buf;
  cudaMalloc(&desc_i_buf, num_tiles * sizeof(CUtensorMap));
  cudaMalloc(&desc_o_buf, num_tiles * sizeof(CUtensorMap));

  std::vector<CUtensorMap> host_i_descs(num_tiles);
  std::vector<CUtensorMap> host_o_descs(num_tiles);

  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(RED)};
  uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(OUT_PAD)};
  uint32_t o_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                              static_cast<uint32_t>(MMA_M)};

  auto *in_base = static_cast<mpk_bf16 *>(input_ptr);
  auto *out_base = static_cast<mpk_bf16 *>(output_ptr);

  for (int t = 0; t < num_tiles; ++t) {
    int const row_offset = t * MMA_N;
    int const tile_rows = std::min(MMA_N, bs - row_offset);
    (void)tile_rows; // TMA descriptor uses MMA_N; padding rows tolerated.

    uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(MMA_N),
                                static_cast<uint64_t>(RED)};
    mirage::runtime::fill_tma_desc<mpk_bf16, B, M, S, 2>(
        &host_i_descs[t],
        in_base + static_cast<size_t>(row_offset) * RED,
        i_gmem_shape,
        i_gmem_stride,
        i_smem_shape,
        1,
        TILE_REPEAT_COL);

    uint64_t o_gmem_shape[2] = {static_cast<uint64_t>(MMA_N),
                                static_cast<uint64_t>(OUT_PAD)};
    mirage::runtime::fill_tma_desc<mpk_bf16, 0, M, S, 2>(
        &host_o_descs[t],
        out_base + static_cast<size_t>(row_offset) * OUT_PAD,
        o_gmem_shape,
        o_gmem_stride,
        o_smem_shape,
        1,
        1);
  }

  cudaMemcpyAsync(desc_i_buf,
                  host_i_descs.data(),
                  num_tiles * sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(desc_o_buf,
                  host_o_descs.data(),
                  num_tiles * sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);

  auto *kernel_ptr =
      &mHC_linear_wrapper<OUT_PAD, RED, MMA_M, MMA_N>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  cutlass::ClusterLaunchParams params = {
      grid_dim, block_dim, cluster_dim, smemBytes, stream};

  for (int t = 0; t < num_tiles; ++t) {
    cutlass::launch_kernel_on_cluster(params,
                                      (void const *)kernel_ptr,
                                      desc_w_ptr,
                                      desc_i_buf + t,
                                      desc_o_buf + t);
  }

  // Free TMA descriptors after the stream consumes them.
  cudaStreamSynchronize(stream);
  cudaFree(desc_w_ptr);
  cudaFree(desc_i_buf);
  cudaFree(desc_o_buf);
}

void mHC_linear(torch::Tensor input,
                torch::Tensor weight_padded,
                torch::Tensor output_padded) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
                  input.scalar_type() == at::kBFloat16,
              "input must be bf16 [bs, reduction] CUDA contiguous");
  TORCH_CHECK(weight_padded.is_cuda() && weight_padded.is_contiguous() &&
                  weight_padded.dim() == 2 &&
                  weight_padded.scalar_type() == at::kBFloat16,
              "weight_padded must be bf16 [128, reduction] CUDA contiguous");
  TORCH_CHECK(output_padded.is_cuda() && output_padded.is_contiguous() &&
                  output_padded.dim() == 2 &&
                  output_padded.scalar_type() == at::kBFloat16,
              "output_padded must be bf16 [bs, 128] CUDA contiguous");

  int const bs = static_cast<int>(input.size(0));
  int const red = static_cast<int>(input.size(1));
  TORCH_CHECK(weight_padded.size(0) == 128,
              "weight_padded rows must equal 128 (MMA_M); pad mix_hc to 128");
  TORCH_CHECK(weight_padded.size(1) == red, "weight reduction must match input");
  TORCH_CHECK(output_padded.size(0) == bs &&
                  output_padded.size(1) == 128,
              "output_padded must be [bs, 128]");
  TORCH_CHECK(bs % 16 == 0, "bs must be a multiple of 16 (MMA_N)");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  void *in_ptr = input.data_ptr();
  void *w_ptr = weight_padded.data_ptr();
  void *out_ptr = output_padded.data_ptr();

  switch (red) {
    case 768:
      launch_mHC_linear<128, 768>(in_ptr, w_ptr, out_ptr, bs, stream);
      break;
    case 1024:
      launch_mHC_linear<128, 1024>(in_ptr, w_ptr, out_ptr, bs, stream);
      break;
    case 2048:
      launch_mHC_linear<128, 2048>(in_ptr, w_ptr, out_ptr, bs, stream);
      break;
    case 4096:
      launch_mHC_linear<128, 4096>(in_ptr, w_ptr, out_ptr, bs, stream);
      break;
    case 8192:
      launch_mHC_linear<128, 8192>(in_ptr, w_ptr, out_ptr, bs, stream);
      break;
    case 16384:
      launch_mHC_linear<128, 16384>(in_ptr, w_ptr, out_ptr, bs, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported reduction=",
                  red,
                  " (must be one of {768,1024,2048,4096,8192,16384})");
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "K1 linear launch error: ",
              cudaGetErrorString(err));
}

// ============================================================================
// K2: affine + split + activation
// ============================================================================

template <typename T_in, int N>
__global__ __launch_bounds__(128) void mHC_affine_split_activation_kernel(
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
  kernel::mHC_affine_split_activation_task_impl<T_in, /*BATCH_SIZE=*/1, N>(
      mixes + token * MIX_HC,
      scale,
      base,
      h_pre + token * N,
      h_post + token * N,
      h_res_logits + token * (N * N));
}

template <typename T_in, int N>
void launch_mHC_affine_split_activation(T_in const *mixes,
                                    float const *scale,
                                    float const *base,
                                    float *h_pre,
                                    float *h_post,
                                    float *h_res_logits,
                                    int num_tokens,
                                    cudaStream_t stream) {
  dim3 const grid_dim(num_tokens, 1, 1);
  dim3 const block_dim(128, 1, 1);
  mHC_affine_split_activation_kernel<T_in, N>
      <<<grid_dim, block_dim, 0, stream>>>(
          mixes, scale, base, h_pre, h_post, h_res_logits, num_tokens);
}

#define DISPATCH_K2_N(T_IN, MIXES, MIXES_DTYPE)                                \
  switch (n) {                                                                 \
    case 2:                                                                    \
      launch_mHC_affine_split_activation<T_IN, 2>(                             \
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
      launch_mHC_affine_split_activation<T_IN, 4>(                             \
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
      launch_mHC_affine_split_activation<T_IN, 8>(                             \
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

void mHC_affine_split_activation(torch::Tensor mixes,
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
__global__ __launch_bounds__(256) void mHC_mul_sum_add_with_outer_kernel(
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
  kernel::mHC_mul_sum_add_with_outer_task_impl<T,
                                                 /*BATCH_SIZE=*/1,
                                                 /*OUTPUT_SIZE=*/C,
                                                 /*NUM_TOPK=*/N,
                                                 /*OUTPUT_STRIDE=*/C>(
      residual, x, comb, post, output);
}

template <typename T, int N>
void launch_mHC_mul_sum_add_with_outer(T const *residual,
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
      mHC_mul_sum_add_with_outer_kernel<T, N, 128><<<grid_dim, block_dim, 0, stream>>>(
          residual, x, comb, post, output, num_tokens);
      break;
    case 1024:
      mHC_mul_sum_add_with_outer_kernel<T, N, 1024><<<grid_dim, block_dim, 0, stream>>>(
          residual, x, comb, post, output, num_tokens);
      break;
    case 4096:
      mHC_mul_sum_add_with_outer_kernel<T, N, 4096><<<grid_dim, block_dim, 0, stream>>>(
          residual, x, comb, post, output, num_tokens);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported C=",
                  c,
                  " (must be one of {128, 1024, 4096})");
  }
}

void mHC_mul_sum_add_with_outer(torch::Tensor residual,
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
      launch_mHC_mul_sum_add_with_outer<bf16_t, 2>(residual_ptr,
                                                      x_ptr,
                                                      comb.data_ptr<float>(),
                                                      post.data_ptr<float>(),
                                                      output_ptr,
                                                      num_tokens,
                                                      c,
                                                      stream);
      break;
    case 4:
      launch_mHC_mul_sum_add_with_outer<bf16_t, 4>(residual_ptr,
                                                      x_ptr,
                                                      comb.data_ptr<float>(),
                                                      post.data_ptr<float>(),
                                                      output_ptr,
                                                      num_tokens,
                                                      c,
                                                      stream);
      break;
    case 8:
      launch_mHC_mul_sum_add_with_outer<bf16_t, 8>(residual_ptr,
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
  m.def("mHC_rmsnorm",
        &mHC_rmsnorm,
        py::arg("x"),
        py::arg("y"),
        py::arg("eps") = 1e-6,
        "mHC K1 (rmsnorm half): per-token RMSNorm with implicit unit weight");

  m.def("mHC_linear",
        &mHC_linear,
        py::arg("input"),
        py::arg("weight_padded"),
        py::arg("output_padded"),
        "mHC K1 (linear half): batched skinny matmul via linear_sm100_mpk; "
        "weight padded to [128, RED], output padded to [bs, 128]; "
        "user must slice unpadded mix_hc cols afterwards");

  m.def("mHC_affine_split_activation",
        &mHC_affine_split_activation,
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
        "mHC K4 (residual=zeros for plain reduction; shared with sm100_moe)");

  m.def("mHC_mul_sum_add_with_outer",
        &mHC_mul_sum_add_with_outer,
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
