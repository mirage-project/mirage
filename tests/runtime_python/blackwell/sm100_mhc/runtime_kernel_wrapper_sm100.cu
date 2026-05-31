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

#include <cooperative_groups.h>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

#include "blackwell/mHC_hc_pre_megakernel.cuh"
#include "blackwell/mHC_hc_pre_post_fused.cuh"
#include "blackwell/mHC_linear.cuh"
#include "blackwell/mHC_mul_sum_add_with_outer.cuh"
#include "blackwell/mHC_rmsnorm.cuh"
#include "blackwell/sinkhorn.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/bfloat16.h>

using bf16_t = cutlass::bfloat16_t;
using mpk_bf16 = cute::bfloat16_t;

namespace {

constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// Default to B200's SM count. Caller can override per-call.
constexpr int kDefaultNumCTAs = 148;
constexpr int kBlockThreads = 256;

// `num_ctas == 0` means "use the device SM count". Cached after first query.
int resolve_num_ctas(int num_ctas, int device) {
  if (num_ctas > 0) {
    return num_ctas;
  }
  static int cached_sm_count = -1;
  if (cached_sm_count < 0) {
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    cached_sm_count = sm_count > 0 ? sm_count : kDefaultNumCTAs;
  }
  return cached_sm_count;
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
  // Grid-stride over tokens so a fixed number of CTAs (set by caller, e.g.
  // device SM count) handles arbitrary workloads.
  for (int64_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    kernel::mHC_rmsnorm_task_impl<T_in, T_out, HIDDEN, BLOCK_THREADS>(
        x + token * HIDDEN, y + token * HIDDEN, eps);
  }
}

template <typename T_in, typename T_out, int HIDDEN>
void launch_mHC_rmsnorm(T_in const *x,
                        T_out *y,
                        int num_tokens,
                        float eps,
                        int num_ctas,
                        cudaStream_t stream) {
  // Cap grid at num_tokens so we don't launch idle CTAs for tiny workloads.
  int const grid = num_ctas < num_tokens ? num_ctas : num_tokens;
  dim3 const grid_dim(grid, 1, 1);
  dim3 const block_dim(kBlockThreads, 1, 1);
  mHC_rmsnorm_kernel<T_in, T_out, HIDDEN, kBlockThreads>
      <<<grid_dim, block_dim, 0, stream>>>(x, y, num_tokens, eps);
}

#define DISPATCH_K1_HIDDEN(T_IN, T_OUT, IN_PTR, OUT_PTR)                       \
  switch (hidden) {                                                            \
    case 256:                                                                  \
      launch_mHC_rmsnorm<T_IN, T_OUT, 256>(                                    \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    case 512:                                                                  \
      launch_mHC_rmsnorm<T_IN, T_OUT, 512>(                                    \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    case 1024:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 1024>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    case 2048:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 2048>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    case 4096:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 4096>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    case 8192:                                                                 \
      launch_mHC_rmsnorm<T_IN, T_OUT, 8192>(                                   \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    case 16384:                                                                \
      launch_mHC_rmsnorm<T_IN, T_OUT, 16384>(                                  \
          IN_PTR, OUT_PTR, num_tokens, eps_f, num_ctas, stream);               \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "Unsupported hidden=",                                       \
                  hidden,                                                      \
                  " (must be one of {256,512,1024,2048,4096,8192,16384})");    \
  }

void mHC_rmsnorm(torch::Tensor x,
                 torch::Tensor y,
                 double eps,
                 int num_ctas_arg) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
              "x must be 2D [num_tokens, hidden] CUDA contiguous");
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.sizes() == x.sizes(),
              "y must match x shape, CUDA contiguous");

  int const num_tokens = static_cast<int>(x.size(0));
  int const hidden = static_cast<int>(x.size(1));
  float const eps_f = static_cast<float>(eps);
  int const num_ctas = resolve_num_ctas(num_ctas_arg, x.get_device());
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
  TORCH_CHECK(
      err == cudaSuccess, "K1 rmsnorm launch error: ", cudaGetErrorString(err));
}

#undef DISPATCH_K1_HIDDEN

// ============================================================================
// K1 (linear half): tcgen05 + TMA + TMEM bf16 GEMM via mHC_linear.cuh.
//   y[bs, OUT_PAD] = x[bs, K] @ w[OUT_PAD, K]^T  (OUT_PAD = MMA_M = 128)
// One CTA per batch tile (MMA_N rows). Single launch with grid_dim = bs/MMA_N.
// Caller pads weight to [128, K], slices first n_actual cols of output.
// ============================================================================

template <int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE>
__global__ __launch_bounds__(256, 1) void mHC_linear_wrapper(
    void *tma_a_desc_ptr, void *tma_b_desc_ptr, void *tma_out_desc_ptr) {
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr int OUTPUT_ATOM_REPEAT_COL = 1;

  using TMA_B =
      kernel::tma::tma_2d<mpk_bf16,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                /*GMEM_ROW_*/
                          REDUCTION_SIZE,            /*GMEM_COL_*/
                          MMA_N,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,         /*SMEM_COL_*/
                          REDUCTION_SIZE,            /*GMEM_STRIDE_ROW_*/
                          1,                         /*GMEM_STRIDE_COL_*/
                          1,                         /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,   /*SMEM_REPEAT_COL_*/
                          MMA_N * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<mpk_bf16,
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
  using TMA_OUT =
      kernel::tma::tma_2d<mpk_bf16,
                          0,
                          M,
                          S,
                          BATCH_SIZE,             /*GMEM_ROW_*/
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

  kernel::mHC_linear_task_impl<mpk_bf16,
                               TMA_A,
                               TMA_B,
                               TMA_OUT,
                               MMA_M,
                               MMA_N,
                               BATCH_SIZE,
                               OUTPUT_SIZE,
                               REDUCTION_SIZE>(tma_a, tma_b, tma_out);
}

template <int BATCH_SIZE, int RED>
void launch_mHC_linear(void *input_ptr,
                       void *weight_ptr,
                       void *output_ptr,
                       int num_ctas,
                       cudaStream_t stream) {
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int OUT_PAD = 128;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr size_t TILE_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr int smemBytes = 224 * 1024;

  static_assert(BATCH_SIZE % MMA_N == 0,
                "BATCH_SIZE must be a multiple of MMA_N=16");

  // One TMA descriptor each for the full A, B, and Out tensors (no host-side
  // tiling). All CTAs in the grid share these descriptors.
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

  CUtensorMap host_i_desc;
  uint64_t i_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                              static_cast<uint64_t>(RED)};
  uint64_t i_gmem_stride[2] = {1, static_cast<uint64_t>(RED)};
  uint32_t i_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  mirage::runtime::fill_tma_desc<mpk_bf16, B, M, S, 2>(
      &host_i_desc,
      static_cast<mpk_bf16 *>(input_ptr),
      i_gmem_shape,
      i_gmem_stride,
      i_smem_shape,
      1,
      TILE_REPEAT_COL);

  CUtensorMap host_o_desc;
  uint64_t o_gmem_shape[2] = {static_cast<uint64_t>(BATCH_SIZE),
                              static_cast<uint64_t>(OUT_PAD)};
  uint64_t o_gmem_stride[2] = {1, static_cast<uint64_t>(OUT_PAD)};
  uint32_t o_smem_shape[2] = {static_cast<uint32_t>(MMA_N),
                              static_cast<uint32_t>(MMA_M)};
  mirage::runtime::fill_tma_desc<mpk_bf16, 0, M, S, 2>(
      &host_o_desc,
      static_cast<mpk_bf16 *>(output_ptr),
      o_gmem_shape,
      o_gmem_stride,
      o_smem_shape,
      1,
      1);

  // Persistent descriptor buffer (allocated once, reused). Avoids ~50 µs of
  // cudaMalloc/cudaFree per call.
  static CUtensorMap *desc_buf = nullptr;
  if (desc_buf == nullptr) {
    cudaMalloc(&desc_buf, 3 * sizeof(CUtensorMap));
  }
  CUtensorMap *desc_w_ptr = desc_buf + 0;
  CUtensorMap *desc_i_ptr = desc_buf + 1;
  CUtensorMap *desc_o_ptr = desc_buf + 2;
  cudaMemcpyAsync(desc_w_ptr,
                  &host_w_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(desc_i_ptr,
                  &host_i_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(desc_o_ptr,
                  &host_o_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);

  auto *kernel_ptr =
      &mHC_linear_wrapper<OUT_PAD, RED, MMA_M, MMA_N, BATCH_SIZE>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  // Grid-strided over n_tiles. Cap at the work count so we don't launch
  // empty CTAs for small batches.
  constexpr int kNumNTiles = BATCH_SIZE / MMA_N;
  int const grid = num_ctas < kNumNTiles ? num_ctas : kNumNTiles;
  dim3 grid_dim(grid, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  cutlass::ClusterLaunchParams params = {
      grid_dim, block_dim, cluster_dim, smemBytes, stream};

  cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, desc_w_ptr, desc_i_ptr, desc_o_ptr);
}

#define DISPATCH_MHC_LINEAR_BS_K(BS)                                           \
  switch (red) {                                                               \
    case 256:                                                                  \
      launch_mHC_linear<BS, 256>(in_ptr, w_ptr, out_ptr, num_ctas, stream);    \
      break;                                                                   \
    case 512:                                                                  \
      launch_mHC_linear<BS, 512>(in_ptr, w_ptr, out_ptr, num_ctas, stream);    \
      break;                                                                   \
    case 1024:                                                                 \
      launch_mHC_linear<BS, 1024>(in_ptr, w_ptr, out_ptr, num_ctas, stream);   \
      break;                                                                   \
    case 2048:                                                                 \
      launch_mHC_linear<BS, 2048>(in_ptr, w_ptr, out_ptr, num_ctas, stream);   \
      break;                                                                   \
    case 4096:                                                                 \
      launch_mHC_linear<BS, 4096>(in_ptr, w_ptr, out_ptr, num_ctas, stream);   \
      break;                                                                   \
    case 8192:                                                                 \
      launch_mHC_linear<BS, 8192>(in_ptr, w_ptr, out_ptr, num_ctas, stream);   \
      break;                                                                   \
    case 16384:                                                                \
      launch_mHC_linear<BS, 16384>(in_ptr, w_ptr, out_ptr, num_ctas, stream);  \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "Unsupported K=",                                            \
                  red,                                                         \
                  " (must be one of {256,512,1024,2048,4096,8192,16384})");    \
  }

void mHC_linear(torch::Tensor input,
                torch::Tensor weight_padded,
                torch::Tensor output_padded,
                int num_ctas_arg) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
                  input.scalar_type() == at::kBFloat16,
              "input must be bf16 [bs, K] CUDA contiguous");
  TORCH_CHECK(weight_padded.is_cuda() && weight_padded.is_contiguous() &&
                  weight_padded.dim() == 2 &&
                  weight_padded.scalar_type() == at::kBFloat16,
              "weight_padded must be bf16 [128, K] CUDA contiguous");
  TORCH_CHECK(output_padded.is_cuda() && output_padded.is_contiguous() &&
                  output_padded.dim() == 2 &&
                  output_padded.scalar_type() == at::kBFloat16,
              "output_padded must be bf16 [bs, 128] CUDA contiguous");

  int const bs = static_cast<int>(input.size(0));
  int const red = static_cast<int>(input.size(1));
  TORCH_CHECK(weight_padded.size(0) == 128,
              "weight_padded rows must equal 128 (MMA_M); pad mix_hc to 128");
  TORCH_CHECK(weight_padded.size(1) == red, "weight K must match input K");
  TORCH_CHECK(output_padded.size(0) == bs && output_padded.size(1) == 128,
              "output_padded must be [bs, 128]");
  TORCH_CHECK(bs % 16 == 0, "bs must be a multiple of MMA_N=16");

  int const num_ctas = resolve_num_ctas(num_ctas_arg, input.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  void *in_ptr = input.data_ptr();
  void *w_ptr = weight_padded.data_ptr();
  void *out_ptr = output_padded.data_ptr();

  switch (bs) {
    case 16:
      DISPATCH_MHC_LINEAR_BS_K(16);
      break;
    case 64:
      DISPATCH_MHC_LINEAR_BS_K(64);
      break;
    case 256:
      DISPATCH_MHC_LINEAR_BS_K(256);
      break;
    case 1024:
      DISPATCH_MHC_LINEAR_BS_K(1024);
      break;
    case 4096:
      DISPATCH_MHC_LINEAR_BS_K(4096);
      break;
    case 8192:
      DISPATCH_MHC_LINEAR_BS_K(8192);
      break;
    case 16384:
      DISPATCH_MHC_LINEAR_BS_K(16384);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported bs=",
                  bs,
                  " (must be one of {16,64,256,1024,4096,8192,16384})");
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess, "K1 linear launch error: ", cudaGetErrorString(err));
}

#undef DISPATCH_MHC_LINEAR_BS_K

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
  for (int64_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    T const *residual = static_cast<T const *>(residual_ptr) + token * N * C;
    T const *x = static_cast<T const *>(x_ptr) + token * C;
    float const *comb = static_cast<float const *>(comb_ptr) + token * N * N;
    float const *post = static_cast<float const *>(post_ptr) + token * N;
    T *output = static_cast<T *>(output_ptr) + token * N * C;
    kernel::mHC_mul_sum_add_with_outer_task_impl<T,
                                                 /*BATCH_SIZE=*/1,
                                                 /*OUTPUT_SIZE=*/C,
                                                 /*NUM_TOPK=*/N,
                                                 /*OUTPUT_STRIDE=*/C>(
        residual, x, comb, post, output);
  }
}

template <typename T, int N>
void launch_mHC_mul_sum_add_with_outer(T const *residual,
                                       T const *x,
                                       float const *comb,
                                       float const *post,
                                       T *output,
                                       int num_tokens,
                                       int c,
                                       int num_ctas,
                                       cudaStream_t stream) {
  int const grid = num_ctas < num_tokens ? num_ctas : num_tokens;
  dim3 grid_dim(grid, 1, 1);
  dim3 block_dim(kBlockThreads, 1, 1);
  switch (c) {
    case 128:
      mHC_mul_sum_add_with_outer_kernel<T, N, 128>
          <<<grid_dim, block_dim, 0, stream>>>(
              residual, x, comb, post, output, num_tokens);
      break;
    case 1024:
      mHC_mul_sum_add_with_outer_kernel<T, N, 1024>
          <<<grid_dim, block_dim, 0, stream>>>(
              residual, x, comb, post, output, num_tokens);
      break;
    case 4096:
      mHC_mul_sum_add_with_outer_kernel<T, N, 4096>
          <<<grid_dim, block_dim, 0, stream>>>(
              residual, x, comb, post, output, num_tokens);
      break;
    default:
      TORCH_CHECK(
          false, "Unsupported C=", c, " (must be one of {128, 1024, 4096})");
  }
}

void mHC_mul_sum_add_with_outer(torch::Tensor residual,
                                torch::Tensor x,
                                torch::Tensor comb,
                                torch::Tensor post,
                                torch::Tensor output,
                                int n,
                                int num_ctas_arg) {
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

  int const num_ctas = resolve_num_ctas(num_ctas_arg, residual.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  bf16_t const *residual_ptr =
      reinterpret_cast<bf16_t const *>(residual.data_ptr());
  bf16_t const *x_ptr = reinterpret_cast<bf16_t const *>(x.data_ptr());
  bf16_t *output_ptr = reinterpret_cast<bf16_t *>(output.data_ptr());

  switch (n) {
    case 2:
      launch_mHC_mul_sum_add_with_outer<bf16_t, 2>(residual_ptr,
                                                   x_ptr,
                                                   comb.data_ptr<float>(),
                                                   post.data_ptr<float>(),
                                                   output_ptr,
                                                   num_tokens,
                                                   c,
                                                   num_ctas,
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
                                                   num_ctas,
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
                                                   num_ctas,
                                                   stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported n=", n);
  }

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "K5 launch error: ", cudaGetErrorString(err));
}

// ============================================================================
// K3 standalone: sinkhorn (4x4)
// ============================================================================

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

void sinkhorn_sm100(torch::Tensor comb_res_mix,
                    torch::Tensor comb_res_mix_out,
                    int repeat,
                    double eps,
                    int num_ctas_arg) {
  TORCH_CHECK(
      comb_res_mix.is_cuda() && comb_res_mix.is_contiguous() &&
          comb_res_mix.scalar_type() == at::kFloat && comb_res_mix.dim() == 3,
      "comb_res_mix must be float32 [num_tokens, 4, 4] CUDA contiguous");
  TORCH_CHECK(comb_res_mix.size(1) == 4 && comb_res_mix.size(2) == 4,
              "sinkhorn matrix must be 4x4 (mHC)");
  TORCH_CHECK(comb_res_mix_out.sizes() == comb_res_mix.sizes() &&
                  comb_res_mix_out.is_cuda() &&
                  comb_res_mix_out.is_contiguous() &&
                  comb_res_mix_out.scalar_type() == at::kFloat,
              "comb_res_mix_out shape/dtype mismatch");
  TORCH_CHECK(repeat >= 1, "repeat must be >= 1");

  int const num_tokens = static_cast<int>(comb_res_mix.size(0));
  float const eps_f = static_cast<float>(eps);
  float const *input_ptr = comb_res_mix.data_ptr<float>();
  float *output_ptr = comb_res_mix_out.data_ptr<float>();
  int const num_ctas =
      resolve_num_ctas(num_ctas_arg, comb_res_mix.get_device());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(comb_res_mix.get_device());

  int const tokens_per_cta_floor = ceil_div(num_tokens, kBlockThreads);
  int const grid =
      tokens_per_cta_floor < num_ctas ? tokens_per_cta_floor : num_ctas;
  dim3 const grid_dim(grid > 0 ? grid : 1, 1, 1);
  dim3 const block_dim(kBlockThreads, 1, 1);
  sinkhorn_sm100_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input_ptr, output_ptr, num_tokens, repeat, eps_f);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess, "Sinkhorn launch error: ", cudaGetErrorString(err));
}

// ============================================================================
// v3: persistent megakernel fusing rmsnorm + linear + tail (K2+K3+K4) into
// one launch. Single grid (148/128 CTAs, 256 threads). Three sequential
// stages with gmem-atomic barriers between them.
// ============================================================================

template <int OUTPUT_SIZE,    // = 128 (linear output pad / MMA_M)
          int REDUCTION_SIZE, // = K = n*C (linear reduction)
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,     // = bs (compile-time)
          int RMSNORM_HIDDEN, // = REDUCTION_SIZE (rmsnorm input dim)
          int TAIL_C,         // = c (per-head dim)
          int TOKENS_PER_CTA> // = tail's batched-token count k
__global__ __launch_bounds__(256, 1) void mHC_hc_pre_v3_kernel(
    // Stage 1 (rmsnorm) input
    void const *__restrict__ x_fp32,
    // Stage 1 output / Stage 2 input scratch
    void *__restrict__ x_norm_bf16,
    // Stage 2 (linear) TMA descriptors -- A=weight, B=x_norm, OUT=mixes_pad
    void *tma_a_desc_ptr,
    void *tma_b_desc_ptr,
    void *tma_out_desc_ptr,
    // Stage 2 output / Stage 3 input scratch
    void const *__restrict__ mixes_pad,
    // Stage 3 (tail) inputs
    void const *__restrict__ scale_ptr,
    void const *__restrict__ base_ptr,
    void const *__restrict__ x_orig_bf16,
    // Stage 3 outputs
    void *__restrict__ f_pre,
    void *__restrict__ h_post_out,
    void *__restrict__ comb_out,
    // Misc
    int num_tokens,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    float rmsnorm_eps) {
  // ---- Stage 1: rmsnorm ----
  for (int64_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    kernel::mHC_rmsnorm_task_impl<float, mpk_bf16, RMSNORM_HIDDEN, 256>(
        static_cast<float const *>(x_fp32) + token * RMSNORM_HIDDEN,
        static_cast<mpk_bf16 *>(x_norm_bf16) + token * RMSNORM_HIDDEN,
        rmsnorm_eps);
  }

  cooperative_groups::this_grid().sync();

  // ---- Stage 2: linear ----
  //
  // Reuses the exact TMA type machinery as the standalone mHC_linear
  // wrapper. Each CTA loops grid-strided over n_tiles inside the task impl.
  {
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
                                      BATCH_SIZE,
                                      REDUCTION_SIZE,
                                      MMA_N,
                                      TMA_CP_ASYNC_SIZE,
                                      REDUCTION_SIZE,
                                      1,
                                      1,
                                      TMA_CP_ASYNC_REPEAT_COL,
                                      MMA_N * TMA_CP_ASYNC_SIZE,
                                      true>;
    using TMA_A = kernel::tma::tma_2d<mpk_bf16,
                                      B,
                                      M,
                                      S,
                                      OUTPUT_SIZE,
                                      REDUCTION_SIZE,
                                      MMA_M,
                                      TMA_CP_ASYNC_SIZE,
                                      REDUCTION_SIZE,
                                      1,
                                      1,
                                      TMA_CP_ASYNC_REPEAT_COL,
                                      MMA_M * TMA_CP_ASYNC_SIZE,
                                      true>;
    using TMA_OUT = kernel::tma::tma_2d<mpk_bf16,
                                        0,
                                        M,
                                        S,
                                        BATCH_SIZE,
                                        OUTPUT_SIZE,
                                        MMA_N,
                                        MMA_M,
                                        OUTPUT_SIZE,
                                        1,
                                        1,
                                        OUTPUT_ATOM_REPEAT_COL,
                                        MMA_N * MMA_M,
                                        true>;
    TMA_A tma_a(static_cast<CUtensorMap *>(tma_a_desc_ptr));
    TMA_B tma_b(static_cast<CUtensorMap *>(tma_b_desc_ptr));
    TMA_OUT tma_out(static_cast<CUtensorMap *>(tma_out_desc_ptr));

    // Linear uses default NUM_AB_STAGE=8 since the tail stage will reuse
    // this same dynamic smem region after the grid barrier.
    kernel::mHC_linear_task_impl<mpk_bf16,
                                 TMA_A,
                                 TMA_B,
                                 TMA_OUT,
                                 MMA_M,
                                 MMA_N,
                                 BATCH_SIZE,
                                 OUTPUT_SIZE,
                                 REDUCTION_SIZE>(tma_a, tma_b, tma_out);
  }

  cooperative_groups::this_grid().sync();

  // ---- Stage 3: tail (K2 + K3 + K4) ----
  //
  // Reads mixes_pad with row stride = 128 (no slice copy needed).
  // Reuses the same dynamic smem region as the linear stage; linear's
  // PipedSharedStorage is dead after the barrier.
  extern __shared__ char shared_memory_v3[];
  kernel::mHC_hc_pre_tail_fused_v2_dyn_smem_task_impl<
      mpk_bf16,
      /*N=*/4,
      TAIL_C,
      TOKENS_PER_CTA,
      /*BLOCK_THREADS=*/256,
      /*MIX_STRIDE=*/OUTPUT_SIZE>(mixes_pad,
                                  scale_ptr,
                                  base_ptr,
                                  x_orig_bf16,
                                  f_pre,
                                  h_post_out,
                                  comb_out,
                                  sinkhorn_repeat,
                                  sinkhorn_eps,
                                  num_tokens,
                                  shared_memory_v3);
}

void mHC_hc_pre_v3(torch::Tensor x_fp32,
                   torch::Tensor x_norm_scratch,
                   torch::Tensor weight_padded,
                   torch::Tensor mixes_pad_scratch,
                   torch::Tensor scale,
                   torch::Tensor base,
                   torch::Tensor x_orig,
                   torch::Tensor f_pre,
                   torch::Tensor h_post,
                   torch::Tensor comb,
                   int n,
                   int c,
                   int sinkhorn_repeat,
                   double sinkhorn_eps,
                   double rmsnorm_eps,
                   int num_ctas_arg,
                   int tokens_per_cta) {
  TORCH_CHECK(n == 4, "v3 hardcoded to n=4");
  TORCH_CHECK(x_fp32.is_cuda() && x_fp32.is_contiguous() &&
                  x_fp32.scalar_type() == at::kFloat && x_fp32.dim() == 2,
              "x_fp32 must be float32 [bs, K] CUDA contiguous");
  TORCH_CHECK(x_norm_scratch.sizes() == x_fp32.sizes() &&
                  x_norm_scratch.is_cuda() && x_norm_scratch.is_contiguous() &&
                  x_norm_scratch.scalar_type() == at::kBFloat16,
              "x_norm_scratch must be bf16 matching x_fp32 shape");
  TORCH_CHECK(weight_padded.is_cuda() && weight_padded.is_contiguous() &&
                  weight_padded.dim() == 2 &&
                  weight_padded.scalar_type() == at::kBFloat16 &&
                  weight_padded.size(0) == 128,
              "weight_padded must be bf16 [128, K] CUDA contiguous");
  TORCH_CHECK(mixes_pad_scratch.is_cuda() &&
                  mixes_pad_scratch.is_contiguous() &&
                  mixes_pad_scratch.dim() == 2 &&
                  mixes_pad_scratch.scalar_type() == at::kBFloat16 &&
                  mixes_pad_scratch.size(1) == 128,
              "mixes_pad_scratch must be bf16 [bs, 128]");

  int const bs = static_cast<int>(x_fp32.size(0));
  int const k = static_cast<int>(x_fp32.size(1));
  TORCH_CHECK(weight_padded.size(1) == k, "weight K must match input K");
  TORCH_CHECK(mixes_pad_scratch.size(0) == bs, "mixes_pad bs mismatch");
  TORCH_CHECK(bs % 16 == 0, "bs must be a multiple of MMA_N=16");
  TORCH_CHECK(tokens_per_cta == 32 || tokens_per_cta == 64 ||
                  tokens_per_cta == 128,
              "tokens_per_cta must be 32, 64, or 128");

  int const num_ctas = resolve_num_ctas(num_ctas_arg, x_fp32.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_fp32.get_device());

  // ---- TMA descriptors for stage 2 (linear) ----
  constexpr int B_ = 3, M_ = 3, S_ = 3;
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int OUT_PAD = 128;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr size_t TILE_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  // Same as standalone linear: 224 KB dynamic smem. v3's tail stage now
  // reuses the same region (linear's data is dead after the grid barrier),
  // so we get linear's full 8-stage AB pipeline back without exceeding the
  // 227 KB B200 per-block opt-in cap.
  constexpr int smemBytes = 224 * 1024;

  CUtensorMap host_w_desc, host_i_desc, host_o_desc;
  uint64_t w_gmem_shape[2] = {OUT_PAD, (uint64_t)k};
  uint64_t w_gmem_stride[2] = {1, (uint64_t)k};
  uint32_t w_smem_shape[2] = {MMA_M, TMA_CP_ASYNC_SIZE};
  mirage::runtime::fill_tma_desc<mpk_bf16, B_, M_, S_, 2>(
      &host_w_desc,
      static_cast<mpk_bf16 *>(weight_padded.data_ptr()),
      w_gmem_shape,
      w_gmem_stride,
      w_smem_shape,
      1,
      TILE_REPEAT_COL);

  uint64_t i_gmem_shape[2] = {(uint64_t)bs, (uint64_t)k};
  uint64_t i_gmem_stride[2] = {1, (uint64_t)k};
  uint32_t i_smem_shape[2] = {MMA_N, TMA_CP_ASYNC_SIZE};
  mirage::runtime::fill_tma_desc<mpk_bf16, B_, M_, S_, 2>(
      &host_i_desc,
      static_cast<mpk_bf16 *>(x_norm_scratch.data_ptr()),
      i_gmem_shape,
      i_gmem_stride,
      i_smem_shape,
      1,
      TILE_REPEAT_COL);

  uint64_t o_gmem_shape[2] = {(uint64_t)bs, OUT_PAD};
  uint64_t o_gmem_stride[2] = {1, OUT_PAD};
  uint32_t o_smem_shape[2] = {MMA_N, MMA_M};
  mirage::runtime::fill_tma_desc<mpk_bf16, 0, M_, S_, 2>(
      &host_o_desc,
      static_cast<mpk_bf16 *>(mixes_pad_scratch.data_ptr()),
      o_gmem_shape,
      o_gmem_stride,
      o_smem_shape,
      1,
      1);

  // Persistent device-side scratch: 3 TMA descs (barrier counters dropped;
  // grid sync now uses cooperative_groups::this_grid().sync()).
  static CUtensorMap *desc_buf = nullptr;
  if (desc_buf == nullptr) {
    cudaMalloc(&desc_buf, 3 * sizeof(CUtensorMap));
  }
  cudaMemcpyAsync(desc_buf + 0,
                  &host_w_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(desc_buf + 1,
                  &host_i_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(desc_buf + 2,
                  &host_o_desc,
                  sizeof(CUtensorMap),
                  cudaMemcpyHostToDevice,
                  stream);

  // Grid: same as standalone linear (n_tiles can exceed num_ctas; task
  // impl does grid-stride). Cap at num_ctas.
  int const linear_n_tiles = bs / MMA_N;
  int const grid_count = num_ctas < linear_n_tiles ? num_ctas : linear_n_tiles;
  dim3 grid_dim(grid_count, 1, 1);
  dim3 block_dim(256, 1, 1);

  float sk_eps_f = static_cast<float>(sinkhorn_eps);
  float rms_eps_f = static_cast<float>(rmsnorm_eps);

#define LAUNCH_V3(BS, K, C, TPC)                                               \
  do {                                                                         \
    auto *kernel_ptr =                                                         \
        &mHC_hc_pre_v3_kernel<OUT_PAD, K, MMA_M, MMA_N, BS, K, C, TPC>;        \
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(                                     \
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));  \
    void *x_fp32_p = const_cast<void *>(x_fp32.data_ptr());                    \
    void *x_norm_p = const_cast<void *>(x_norm_scratch.data_ptr());            \
    void *desc_a = desc_buf + 0;                                               \
    void *desc_b = desc_buf + 1;                                               \
    void *desc_o = desc_buf + 2;                                               \
    void *mixes_p = const_cast<void *>(mixes_pad_scratch.data_ptr());          \
    void *scale_p = const_cast<float *>(scale.data_ptr<float>());              \
    void *base_p = const_cast<float *>(base.data_ptr<float>());                \
    void *x_orig_p = const_cast<void *>(x_orig.data_ptr());                    \
    void *f_pre_p = const_cast<void *>(f_pre.data_ptr());                      \
    void *h_post_p = const_cast<float *>(h_post.data_ptr<float>());            \
    void *comb_p = const_cast<float *>(comb.data_ptr<float>());                \
    int bs_arg = bs;                                                           \
    int sk_repeat_arg = sinkhorn_repeat;                                       \
    void *kargs[] = {&x_fp32_p,                                                \
                     &x_norm_p,                                                \
                     &desc_a,                                                  \
                     &desc_b,                                                  \
                     &desc_o,                                                  \
                     &mixes_p,                                                 \
                     &scale_p,                                                 \
                     &base_p,                                                  \
                     &x_orig_p,                                                \
                     &f_pre_p,                                                 \
                     &h_post_p,                                                \
                     &comb_p,                                                  \
                     &bs_arg,                                                  \
                     &sk_repeat_arg,                                           \
                     &sk_eps_f,                                                \
                     &rms_eps_f};                                              \
    CUTE_CHECK_ERROR(cudaLaunchCooperativeKernel((void const *)kernel_ptr,     \
                                                 grid_dim,                     \
                                                 block_dim,                    \
                                                 kargs,                        \
                                                 smemBytes,                    \
                                                 stream));                     \
  } while (0)

#define DISPATCH_V3_C(BS_, K_)                                                 \
  switch (c) {                                                                 \
    case 128:                                                                  \
      switch (tokens_per_cta) {                                                \
        case 32:                                                               \
          LAUNCH_V3(BS_, K_, 128, 32);                                         \
          break;                                                               \
        case 64:                                                               \
          LAUNCH_V3(BS_, K_, 128, 64);                                         \
          break;                                                               \
        case 128:                                                              \
          LAUNCH_V3(BS_, K_, 128, 128);                                        \
          break;                                                               \
      }                                                                        \
      break;                                                                   \
    case 1024:                                                                 \
      switch (tokens_per_cta) {                                                \
        case 32:                                                               \
          LAUNCH_V3(BS_, K_, 1024, 32);                                        \
          break;                                                               \
        case 64:                                                               \
          LAUNCH_V3(BS_, K_, 1024, 64);                                        \
          break;                                                               \
        case 128:                                                              \
          LAUNCH_V3(BS_, K_, 1024, 128);                                       \
          break;                                                               \
      }                                                                        \
      break;                                                                   \
    case 4096:                                                                 \
      switch (tokens_per_cta) {                                                \
        case 32:                                                               \
          LAUNCH_V3(BS_, K_, 4096, 32);                                        \
          break;                                                               \
        case 64:                                                               \
          LAUNCH_V3(BS_, K_, 4096, 64);                                        \
          break;                                                               \
        case 128:                                                              \
          LAUNCH_V3(BS_, K_, 4096, 128);                                       \
          break;                                                               \
      }                                                                        \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported c=", c);                                 \
  }

#define DISPATCH_V3_BS_K(BS_)                                                  \
  switch (k) {                                                                 \
    case 1024:                                                                 \
      DISPATCH_V3_C(BS_, 1024);                                                \
      break;                                                                   \
    case 4096:                                                                 \
      DISPATCH_V3_C(BS_, 4096);                                                \
      break;                                                                   \
    case 16384:                                                                \
      DISPATCH_V3_C(BS_, 16384);                                               \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported K=", k);                                 \
  }

  switch (bs) {
    case 1024:
      DISPATCH_V3_BS_K(1024);
      break;
    case 4096:
      DISPATCH_V3_BS_K(4096);
      break;
    case 8192:
      DISPATCH_V3_BS_K(8192);
      break;
    default:
      TORCH_CHECK(false, "Unsupported bs=", bs);
  }

#undef DISPATCH_V3_BS_K
#undef DISPATCH_V3_C
#undef LAUNCH_V3

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "v3 megakernel launch error: ",
              cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // num_ctas=0 means "use device SM count". Caller can pin to 128 / 148 / etc.
  m.def("mHC_rmsnorm",
        &mHC_rmsnorm,
        py::arg("x"),
        py::arg("y"),
        py::arg("eps") = 1e-6,
        py::arg("num_ctas") = 0,
        "mHC K1 (rmsnorm half): per-token RMSNorm with implicit unit weight");

  m.def("mHC_linear",
        &mHC_linear,
        py::arg("input"),
        py::arg("weight_padded"),
        py::arg("output_padded"),
        py::arg("num_ctas") = 0,
        "mHC K1 (linear half): tcgen05+TMA+TMEM bf16 GEMM (MPK-fusable); "
        "weight padded to [128, K], output padded to [bs, 128]; "
        "user slices first n_actual cols afterwards");

  m.def("mHC_mul_sum_add_with_outer",
        &mHC_mul_sum_add_with_outer,
        py::arg("residual"),
        py::arg("x"),
        py::arg("comb"),
        py::arg("post"),
        py::arg("output"),
        py::arg("n"),
        py::arg("num_ctas") = 0,
        "mHC K5: residual mix + post outer-product fused");

  m.def("sinkhorn_sm100",
        &sinkhorn_sm100,
        py::arg("comb_res_mix"),
        py::arg("comb_res_mix_out"),
        py::arg("repeat") = 20,
        py::arg("eps") = 1e-9,
        py::arg("num_ctas") = 0,
        "mHC K3: Sinkhorn-Knopp normalization (4x4)");

  m.def("mHC_hc_pre_v3",
        &mHC_hc_pre_v3,
        py::arg("x_fp32"),
        py::arg("x_norm_scratch"),
        py::arg("weight_padded"),
        py::arg("mixes_pad_scratch"),
        py::arg("scale"),
        py::arg("base"),
        py::arg("x_orig"),
        py::arg("f_pre"),
        py::arg("h_post"),
        py::arg("comb"),
        py::arg("n"),
        py::arg("c"),
        py::arg("sinkhorn_repeat") = 20,
        py::arg("sinkhorn_eps") = 1e-9,
        py::arg("rmsnorm_eps") = 1e-6,
        py::arg("num_ctas") = 0,
        py::arg("tokens_per_cta") = 32,
        "v3 persistent megakernel: rmsnorm + linear + tail fused, single "
        "launch.");
}
