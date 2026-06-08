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

#include "blackwell/mHC_post.cuh"
#include "blackwell/mHC_post_pre.cuh"
#include "blackwell/mHC_pre.cuh"
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
// mHC_post: HC post + residual fusion with outer-product residual
//   y[k, c] = post[k] * x[c] + sum_i comb[i, k] * residual[i, c]
// ============================================================================

// Each block owns TOKENS_PER_BLK tokens, its threads split into that many
// contiguous sub-groups (one per token) so all threads stay busy at small C.
template <typename T, int N, int C, int TOKENS_PER_BLK>
__global__ __launch_bounds__(256) void mHC_post_kernel(void const *residual_ptr,
                                                       void const *x_ptr,
                                                       void const *comb_ptr,
                                                       void const *post_ptr,
                                                       void *output_ptr,
                                                       int num_tokens) {
  int const threads_per_token = blockDim.x / TOKENS_PER_BLK;
  int const group = threadIdx.x / threads_per_token; // which token slot
  int const lane = threadIdx.x % threads_per_token;  // index within slot
  for (int64_t tile = blockIdx.x; tile * TOKENS_PER_BLK < num_tokens;
       tile += gridDim.x) {
    int64_t token = tile * TOKENS_PER_BLK + group;
    if (token >= num_tokens) {
      continue;
    }
    T const *residual = static_cast<T const *>(residual_ptr) + token * N * C;
    T const *x = static_cast<T const *>(x_ptr) + token * C;
    float const *comb = static_cast<float const *>(comb_ptr) + token * N * N;
    float const *post = static_cast<float const *>(post_ptr) + token * N;
    T *output = static_cast<T *>(output_ptr) + token * N * C;
    kernel::mHC_post_task_impl<T,
                               /*BATCH_SIZE=*/1,
                               /*OUTPUT_SIZE=*/C,
                               /*NUM_TOPK=*/N,
                               /*OUTPUT_STRIDE=*/C>(
        residual, x, comb, post, output, lane, threads_per_token);
  }
}

template <typename T, int N>
void launch_mHC_post(T const *residual,
                     T const *x,
                     float const *comb,
                     float const *post,
                     T *output,
                     int num_tokens,
                     int c,
                     int num_ctas,
                     cudaStream_t stream) {
  // Vectorized by 8 (uint4): work unit is a channel-vec, c_vec = C/8.
  (void)num_ctas;
  constexpr int VEC = 8;
  int const c_vec = c / VEC;

#define LAUNCH_POST(C_, TPB)                                                   \
  do {                                                                         \
    int tpt = c_vec < (256 / (TPB)) ? c_vec : (256 / (TPB));                   \
    tpt = ((tpt + 31) / 32) * 32;                                              \
    if (tpt < 32)                                                              \
      tpt = 32;                                                                \
    int const block_threads = tpt * (TPB);                                     \
    int const tiles = (num_tokens + (TPB)-1) / (TPB);                          \
    dim3 grid_dim(tiles, 1, 1);                                                \
    dim3 block_dim(block_threads, 1, 1);                                       \
    mHC_post_kernel<T, N, C_, TPB><<<grid_dim, block_dim, 0, stream>>>(        \
        residual, x, comb, post, output, num_tokens);                          \
  } while (0)

  switch (c) {
    case 128:
      // 16 vecs/token; 8 tokens/block -> 128 threads, fully utilized.
      LAUNCH_POST(128, 8);
      break;
    case 1024:
      // 128 vecs/token; 2 tokens/block -> 256 threads.
      LAUNCH_POST(1024, 2);
      break;
    case 4096:
      // 512 vecs/token; 1 token/block -> 256 threads.
      LAUNCH_POST(4096, 1);
      break;
    case 7168:
      // 896 vecs/token; 1 token/block -> 256 threads (DeepSeek V4 pro).
      LAUNCH_POST(7168, 1);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported C=",
                  c,
                  " (must be one of {128, 1024, 4096, 7168})");
  }
#undef LAUNCH_POST
}

void mHC_post(torch::Tensor residual,
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
      launch_mHC_post<bf16_t, 2>(residual_ptr,
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
      launch_mHC_post<bf16_t, 4>(residual_ptr,
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
      launch_mHC_post<bf16_t, 8>(residual_ptr,
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
  TORCH_CHECK(
      err == cudaSuccess, "mHC_post launch error: ", cudaGetErrorString(err));
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
// mHC_post_pre_v2: CUDA-core fused post + prenorm-GEMM (vLLM mhc_fused style)
// followed by the split-k reduction and the existing k2 tail.
//
//   [fused] post + GEMM + sqrsum  (register new_r, no round-trip, split-k)
//   [reduce] fold SPLIT_K partials -> mixes_pad (bf16) + sqrsum
//   [k2]    RMS-fold + affines + sinkhorn + weighted sum -> f_pre/h_post/comb
//
// Unlike the tcgen05 cooperative kernel (mHC_post_pre), the GEMM is a
// thread-level FMA loop, so smem stays tiny (full occupancy) and split-k over
// the hidden dim fills the grid at low token counts -- the structure vLLM uses
// to avoid both the pad-to-128 waste and the occupancy cliff.
// ============================================================================

// Forward decl: the k2 tail kernel is defined later (shared with mHC_pre_k2).
template <int N, int C, int RMS_HIDDEN, int TOKENS_PER_CTA>
__global__ void mHC_pre_k2_kernel(void const *__restrict__ mixes_pad,
                                  void const *__restrict__ sqrsum,
                                  void const *__restrict__ scale_ptr,
                                  void const *__restrict__ base_ptr,
                                  void const *__restrict__ x_orig_bf16,
                                  void *__restrict__ f_pre,
                                  void *__restrict__ h_post_out,
                                  void *__restrict__ comb_out,
                                  int num_tokens,
                                  int sinkhorn_repeat,
                                  float sinkhorn_eps,
                                  float rms_eps);

template <int N, int C, int MIX_HC, int BLOCK_THREADS, int SPLIT_K>
__global__ __launch_bounds__(BLOCK_THREADS) void mHC_post_pre_k1_kernel(
    void const *__restrict__ residual,
    void const *__restrict__ x,
    void const *__restrict__ comb,
    void const *__restrict__ post,
    void const *__restrict__ fn,
    void *__restrict__ residual_out,
    float *__restrict__ out_partial,
    float *__restrict__ sqr_partial,
    int num_tokens) {
  int const token = blockIdx.x;
  int const i_ks = blockIdx.y;
  if (token >= num_tokens) {
    return;
  }
  kernel::
      mHC_post_pre_k1_task_impl<mpk_bf16, N, C, MIX_HC, BLOCK_THREADS, SPLIT_K>(
          static_cast<mpk_bf16 const *>(residual),
          static_cast<mpk_bf16 const *>(x),
          static_cast<float const *>(comb),
          static_cast<float const *>(post),
          static_cast<float const *>(fn),
          static_cast<mpk_bf16 *>(residual_out),
          out_partial,
          sqr_partial,
          num_tokens,
          token,
          i_ks);
}

template <int N, int MIX_HC, int MIX_PAD, int SPLIT_K>
__global__ void
    mHC_post_pre_k1_reduce_kernel(float const *__restrict__ out_partial,
                                  float const *__restrict__ sqr_partial,
                                  void *__restrict__ mixes_pad,
                                  float *__restrict__ sqrsum,
                                  int num_tokens) {
  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  kernel::mHC_post_pre_k1_reduce_impl<N, MIX_HC, MIX_PAD, SPLIT_K>(
      out_partial, sqr_partial, mixes_pad, sqrsum, num_tokens, token);
}

void mHC_post_pre_v2(torch::Tensor residual_in,
                     torch::Tensor x_in,
                     torch::Tensor comb_in,
                     torch::Tensor post_in,
                     torch::Tensor fn,            // [MIX_HC, N, C] fp32 weight
                     torch::Tensor residual_next, // [tokens, N, C] bf16
                     torch::Tensor out_partial,   // [SPLIT_K, tokens, MIX_HC]
                     torch::Tensor sqr_partial,   // [SPLIT_K, tokens]
                     torch::Tensor mixes_pad,     // [tokens, 128] bf16 scratch
                     torch::Tensor sqrsum,        // [tokens] fp32 scratch
                     torch::Tensor scale,
                     torch::Tensor base,
                     torch::Tensor f_pre,
                     torch::Tensor h_post,
                     torch::Tensor comb,
                     int n,
                     int c,
                     int split_k,
                     int sinkhorn_repeat,
                     double sinkhorn_eps,
                     double rms_eps,
                     int tokens_per_cta) {
  TORCH_CHECK(n == 4, "post_pre_v2 hardcoded to n=4");
  constexpr int MIX_PAD = 128;
  int const num_tokens = static_cast<int>(residual_in.size(0));
  int const mix_hc = n * n + 2 * n;
  TORCH_CHECK(residual_in.is_cuda() && residual_in.is_contiguous() &&
                  residual_in.scalar_type() == at::kBFloat16 &&
                  residual_in.dim() == 3 && residual_in.size(1) == n &&
                  residual_in.size(2) == c,
              "residual_in must be bf16 [tokens, n, c]");
  TORCH_CHECK(x_in.is_cuda() && x_in.is_contiguous() &&
                  x_in.scalar_type() == at::kBFloat16 &&
                  x_in.sizes() == torch::IntArrayRef({num_tokens, c}),
              "x_in must be bf16 [tokens, c]");
  TORCH_CHECK(comb_in.is_cuda() && comb_in.is_contiguous() &&
                  comb_in.scalar_type() == at::kFloat &&
                  comb_in.sizes() == torch::IntArrayRef({num_tokens, n, n}),
              "comb_in must be float32 [tokens, n, n]");
  TORCH_CHECK(post_in.is_cuda() && post_in.is_contiguous() &&
                  post_in.scalar_type() == at::kFloat &&
                  post_in.sizes() == torch::IntArrayRef({num_tokens, n}),
              "post_in must be float32 [tokens, n]");
  TORCH_CHECK(fn.is_cuda() && fn.is_contiguous() &&
                  fn.scalar_type() == at::kFloat &&
                  fn.sizes() == torch::IntArrayRef({mix_hc, n, c}),
              "fn must be float32 [mix_hc, n, c]");
  TORCH_CHECK(residual_next.is_cuda() && residual_next.is_contiguous() &&
                  residual_next.scalar_type() == at::kBFloat16 &&
                  residual_next.dim() == 3 && residual_next.size(1) == n &&
                  residual_next.size(2) == c,
              "residual_next must be bf16 [tokens, n, c]");
  TORCH_CHECK(out_partial.is_cuda() && out_partial.is_contiguous() &&
                  out_partial.scalar_type() == at::kFloat &&
                  out_partial.sizes() ==
                      torch::IntArrayRef({split_k, num_tokens, mix_hc}),
              "out_partial must be float32 [split_k, tokens, mix_hc]");
  TORCH_CHECK(sqr_partial.is_cuda() && sqr_partial.is_contiguous() &&
                  sqr_partial.scalar_type() == at::kFloat &&
                  sqr_partial.sizes() ==
                      torch::IntArrayRef({split_k, num_tokens}),
              "sqr_partial must be float32 [split_k, tokens]");
  TORCH_CHECK(mixes_pad.is_cuda() && mixes_pad.is_contiguous() &&
                  mixes_pad.scalar_type() == at::kBFloat16 &&
                  mixes_pad.sizes() ==
                      torch::IntArrayRef({num_tokens, MIX_PAD}),
              "mixes_pad must be bf16 [tokens, 128]");
  TORCH_CHECK(sqrsum.is_cuda() && sqrsum.is_contiguous() &&
                  sqrsum.scalar_type() == at::kFloat && sqrsum.dim() == 1 &&
                  sqrsum.size(0) == num_tokens,
              "sqrsum must be float32 [tokens]");
  TORCH_CHECK(c % split_k == 0, "c must be divisible by split_k");
  TORCH_CHECK(tokens_per_cta == 32 || tokens_per_cta == 64 ||
                  tokens_per_cta == 128,
              "tokens_per_cta must be 32, 64, or 128");

  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(residual_in.get_device());
  constexpr int BT = 128;

  void *res_p = const_cast<void *>(residual_in.data_ptr());
  void *x_p = const_cast<void *>(x_in.data_ptr());
  void *comb_p = const_cast<float *>(comb_in.data_ptr<float>());
  void *post_p = const_cast<float *>(post_in.data_ptr<float>());
  void *fn_p = const_cast<float *>(fn.data_ptr<float>());
  void *res_next_p = const_cast<void *>(residual_next.data_ptr());
  float *outp_p = out_partial.data_ptr<float>();
  float *sqrp_p = sqr_partial.data_ptr<float>();
  void *mixes_p = const_cast<void *>(mixes_pad.data_ptr());
  float *sqrsum_p = sqrsum.data_ptr<float>();

  dim3 fused_grid(num_tokens, split_k, 1);
  dim3 fused_block(BT, 1, 1);

#define LAUNCH_FUSED(C_, SK_)                                                  \
  mHC_post_pre_k1_kernel<4, C_, 24, BT, SK_>                                   \
      <<<fused_grid, fused_block, 0, stream>>>(res_p,                          \
                                               x_p,                            \
                                               comb_p,                         \
                                               post_p,                         \
                                               fn_p,                           \
                                               res_next_p,                     \
                                               outp_p,                         \
                                               sqrp_p,                         \
                                               num_tokens);                    \
  mHC_post_pre_k1_reduce_kernel<4, 24, MIX_PAD, SK_>                           \
      <<<dim3(num_tokens, 1, 1), dim3(32, 1, 1), 0, stream>>>(                 \
          outp_p, sqrp_p, mixes_p, sqrsum_p, num_tokens)

#define DISPATCH_FUSED_SK(C_)                                                  \
  switch (split_k) {                                                           \
    case 1:                                                                    \
      LAUNCH_FUSED(C_, 1);                                                     \
      break;                                                                   \
    case 2:                                                                    \
      LAUNCH_FUSED(C_, 2);                                                     \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_FUSED(C_, 4);                                                     \
      break;                                                                   \
    case 8:                                                                    \
      LAUNCH_FUSED(C_, 8);                                                     \
      break;                                                                   \
    case 16:                                                                   \
      LAUNCH_FUSED(C_, 16);                                                    \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported split_k=", split_k);                     \
  }

  switch (c) {
    case 128:
      DISPATCH_FUSED_SK(128);
      break;
    case 1024:
      DISPATCH_FUSED_SK(1024);
      break;
    case 4096:
      DISPATCH_FUSED_SK(4096);
      break;
    case 7168:
      DISPATCH_FUSED_SK(7168);
      break;
    default:
      TORCH_CHECK(false, "Unsupported c=", c);
  }
#undef DISPATCH_FUSED_SK
#undef LAUNCH_FUSED

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_post_pre_v2 fused launch error: ",
              cudaGetErrorString(err));

  // ---- k2 tail (existing kernel) ----
  size_t const k2_smem =
      tokens_per_cta * (4 * n + n * n + n * n + n) * sizeof(float) + 1024;
  int const tiles = ceil_div(num_tokens, tokens_per_cta);
  int const grid = kDefaultNumCTAs < tiles ? kDefaultNumCTAs : tiles;
  dim3 k2_grid(grid > 0 ? grid : 1, 1, 1);
  dim3 k2_block(256, 1, 1);
  void *scale_p = const_cast<float *>(scale.data_ptr<float>());
  void *base_p = const_cast<float *>(base.data_ptr<float>());
  void *x_orig_p = const_cast<void *>(residual_next.data_ptr());
  void *f_pre_p = const_cast<void *>(f_pre.data_ptr());
  void *h_post_p = const_cast<float *>(h_post.data_ptr<float>());
  void *comb_p2 = const_cast<float *>(comb.data_ptr<float>());
  float sk_eps_f = static_cast<float>(sinkhorn_eps);
  float rms_eps_f = static_cast<float>(rms_eps);

#define LAUNCH_V2_K2(C_, RH_, TPC_)                                            \
  do {                                                                         \
    auto *kp = &mHC_pre_k2_kernel<4, C_, RH_, TPC_>;                           \
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(                                     \
        kp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)k2_smem));       \
    kp<<<k2_grid, k2_block, k2_smem, stream>>>(mixes_p,                        \
                                               sqrsum_p,                       \
                                               scale_p,                        \
                                               base_p,                         \
                                               x_orig_p,                       \
                                               f_pre_p,                        \
                                               h_post_p,                       \
                                               comb_p2,                        \
                                               num_tokens,                     \
                                               sinkhorn_repeat,                \
                                               sk_eps_f,                       \
                                               rms_eps_f);                     \
  } while (0)

#define DISPATCH_V2_K2_TPC(C_, RH_)                                            \
  switch (tokens_per_cta) {                                                    \
    case 32:                                                                   \
      LAUNCH_V2_K2(C_, RH_, 32);                                               \
      break;                                                                   \
    case 64:                                                                   \
      LAUNCH_V2_K2(C_, RH_, 64);                                               \
      break;                                                                   \
    case 128:                                                                  \
      LAUNCH_V2_K2(C_, RH_, 128);                                              \
      break;                                                                   \
  }

  switch (c) {
    case 128:
      DISPATCH_V2_K2_TPC(128, 4 * 128);
      break;
    case 1024:
      DISPATCH_V2_K2_TPC(1024, 4 * 1024);
      break;
    case 4096:
      DISPATCH_V2_K2_TPC(4096, 4 * 4096);
      break;
    case 7168:
      DISPATCH_V2_K2_TPC(7168, 4 * 7168);
      break;
    default:
      TORCH_CHECK(false, "Unsupported c=", c);
  }
#undef DISPATCH_V2_K2_TPC
#undef LAUNCH_V2_K2

  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_post_pre_v2 k2 launch error: ",
              cudaGetErrorString(err));
}

// ============================================================================
// mHC_pre_k1_cuda_core: CUDA-core variant of pre_k1 (prenorm GEMM + sqrsum), no
// tensor cores. Same outputs as the tcgen05 mHC_pre_k1 (mixes_pad bf16 +
// sqrsum) so it drop-in feeds the k2 tail. Wins over tcgen05 at low token
// count, where the MMA/TMA fixed setup cost dominates; tcgen05 wins at high T.
// split_k over the reduction fills the grid when tokens are few.
// ============================================================================

template <int MIX_HC,
          int K,
          int BLOCK_THREADS,
          int SPLIT_K,
          int MIX_PAD = 128,
          int TPB = 1>
__global__ __launch_bounds__(BLOCK_THREADS) void mHC_pre_k1_cuda_core_kernel(
    void const *__restrict__ residual,
    float const *__restrict__ fn,
    float *__restrict__ out_partial, // used when SPLIT_K>1
    float *__restrict__ sqr_partial, // used when SPLIT_K>1
    void *__restrict__ mixes_pad,    // used when SPLIT_K==1 (direct write)
    float *__restrict__ sqrsum,      // used when SPLIT_K==1
    int num_tokens) {
  int const token0 = blockIdx.x * TPB; // first token of this CTA's group
  int const i_ks = blockIdx.y;
  if (token0 >= num_tokens) {
    return;
  }
  kernel::mHC_pre_k1_cuda_core_task_impl<mpk_bf16,
                                         MIX_HC,
                                         K,
                                         BLOCK_THREADS,
                                         SPLIT_K,
                                         MIX_PAD,
                                         TPB>(
      static_cast<mpk_bf16 const *>(residual),
      static_cast<float const *>(fn),
      out_partial,
      sqr_partial,
      mixes_pad,
      sqrsum,
      num_tokens,
      token0,
      i_ks);
}

template <int MIX_HC, int MIX_PAD, int SPLIT_K>
__global__ void
    mHC_pre_k1_cuda_core_reduce_kernel(float const *__restrict__ out_partial,
                                       float const *__restrict__ sqr_partial,
                                       void *__restrict__ mixes_pad,
                                       float *__restrict__ sqrsum,
                                       int num_tokens) {
  int const token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  kernel::mHC_pre_k1_cuda_core_reduce_impl<MIX_HC, MIX_PAD, SPLIT_K>(
      out_partial, sqr_partial, mixes_pad, sqrsum, num_tokens, token);
}

// residual [tokens, K] bf16, fn [MIX_HC, K] fp32 -> mixes_pad [tokens,128] bf16
// + sqrsum [tokens] fp32. out_partial/sqr_partial are [split_k, tokens, *]
// scratch. n is hc_mult (4 -> MIX_HC=24).
void mHC_pre_k1_cuda_core(torch::Tensor residual,
                          torch::Tensor fn,
                          torch::Tensor out_partial,
                          torch::Tensor sqr_partial,
                          torch::Tensor mixes_pad,
                          torch::Tensor sqrsum,
                          int n,
                          int split_k) {
  TORCH_CHECK(n == 4, "pre_k1_cuda_core hardcoded to n=4");
  constexpr int MIX_PAD = 128;
  int const mix_hc = n * n + 2 * n; // 24
  int const num_tokens = static_cast<int>(residual.size(0));
  int const K = static_cast<int>(residual.size(1));
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.dim() == 2,
              "residual must be bf16 [tokens, K]");
  TORCH_CHECK(fn.is_cuda() && fn.is_contiguous() &&
                  fn.scalar_type() == at::kFloat && fn.dim() == 2 &&
                  fn.size(0) == mix_hc && fn.size(1) == K,
              "fn must be float32 [mix_hc, K]");
  TORCH_CHECK(out_partial.is_cuda() && out_partial.is_contiguous() &&
                  out_partial.scalar_type() == at::kFloat &&
                  out_partial.sizes() ==
                      torch::IntArrayRef({split_k, num_tokens, mix_hc}),
              "out_partial must be float32 [split_k, tokens, mix_hc]");
  TORCH_CHECK(sqr_partial.is_cuda() && sqr_partial.is_contiguous() &&
                  sqr_partial.scalar_type() == at::kFloat &&
                  sqr_partial.sizes() ==
                      torch::IntArrayRef({split_k, num_tokens}),
              "sqr_partial must be float32 [split_k, tokens]");
  TORCH_CHECK(mixes_pad.is_cuda() && mixes_pad.is_contiguous() &&
                  mixes_pad.scalar_type() == at::kBFloat16 &&
                  mixes_pad.sizes() ==
                      torch::IntArrayRef({num_tokens, MIX_PAD}),
              "mixes_pad must be bf16 [tokens, 128]");
  TORCH_CHECK(sqrsum.is_cuda() && sqrsum.is_contiguous() &&
                  sqrsum.scalar_type() == at::kFloat && sqrsum.dim() == 1 &&
                  sqrsum.size(0) == num_tokens,
              "sqrsum must be float32 [tokens]");
  TORCH_CHECK(K % split_k == 0, "K must be divisible by split_k");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  constexpr int BT = 128;

  void *res_p = const_cast<void *>(residual.data_ptr());
  float *fn_p = const_cast<float *>(fn.data_ptr<float>());
  float *outp_p = out_partial.data_ptr<float>();
  float *sqrp_p = sqr_partial.data_ptr<float>();
  void *mixes_p = const_cast<void *>(mixes_pad.data_ptr());
  float *sqrsum_p = sqrsum.data_ptr<float>();

  dim3 block(BT, 1, 1);
  // TPB amortizes the fn weight reload across tokens (the L1 bottleneck at high
  // T); pick it only when there are plenty of token-groups to still fill the
  // grid. split_k fills the grid at low T. They're opposed, so use TPB>1 only
  // when split_k==1 and tokens are abundant.
  int tpb = 1;
  if (split_k == 1) {
    if (num_tokens >= 4096) {
      tpb = 4;
    } else if (num_tokens >= 1024) {
      tpb = 2;
    }
  }

#define LAUNCH_PRE_K1_CUDA(K_, SK_, TPB_)                                      \
  do {                                                                         \
    dim3 grid((num_tokens + (TPB_)-1) / (TPB_), SK_, 1);                       \
    mHC_pre_k1_cuda_core_kernel<24, K_, BT, SK_, MIX_PAD, TPB_>                \
        <<<grid, block, 0, stream>>>(                                          \
            res_p, fn_p, outp_p, sqrp_p, mixes_p, sqrsum_p, num_tokens);       \
    /* SPLIT_K==1 writes mixes_pad+sqrsum directly in the GEMM epilogue, so    \
       the separate reduce launch is only needed for split-k. */               \
    if ((SK_) > 1) {                                                           \
      mHC_pre_k1_cuda_core_reduce_kernel<24, MIX_PAD, SK_>                     \
          <<<dim3(num_tokens, 1, 1), dim3(32, 1, 1), 0, stream>>>(             \
              outp_p, sqrp_p, mixes_p, sqrsum_p, num_tokens);                  \
    }                                                                          \
  } while (0)

#define LAUNCH_PRE_K1_CUDA_TPB(K_, SK_)                                        \
  switch (tpb) {                                                               \
    case 1:                                                                    \
      LAUNCH_PRE_K1_CUDA(K_, SK_, 1);                                          \
      break;                                                                   \
    case 2:                                                                    \
      LAUNCH_PRE_K1_CUDA(K_, SK_, 2);                                          \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_PRE_K1_CUDA(K_, SK_, 4);                                          \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "bad tpb");                                           \
  }

#define DISPATCH_PRE_K1_CUDA_SK(K_)                                            \
  switch (split_k) {                                                           \
    case 1:                                                                    \
      LAUNCH_PRE_K1_CUDA_TPB(K_, 1);                                           \
      break;                                                                   \
    case 2:                                                                    \
      LAUNCH_PRE_K1_CUDA(K_, 2, 1);                                            \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_PRE_K1_CUDA(K_, 4, 1);                                            \
      break;                                                                   \
    case 8:                                                                    \
      LAUNCH_PRE_K1_CUDA(K_, 8, 1);                                            \
      break;                                                                   \
    case 16:                                                                   \
      LAUNCH_PRE_K1_CUDA(K_, 16, 1);                                           \
      break;                                                                   \
    case 32:                                                                   \
      LAUNCH_PRE_K1_CUDA(K_, 32, 1);                                           \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported split_k=", split_k);                     \
  }

  switch (K) {
    case 512: // c=128
      DISPATCH_PRE_K1_CUDA_SK(512);
      break;
    case 4096: // c=1024
      DISPATCH_PRE_K1_CUDA_SK(4096);
      break;
    case 16384: // c=4096
      DISPATCH_PRE_K1_CUDA_SK(16384);
      break;
    case 28672: // c=7168 (DeepSeek V4 pro)
      DISPATCH_PRE_K1_CUDA_SK(28672);
      break;
    default:
      TORCH_CHECK(false, "Unsupported K=", K);
  }
#undef DISPATCH_PRE_K1_CUDA_SK
#undef LAUNCH_PRE_K1_CUDA_TPB
#undef LAUNCH_PRE_K1_CUDA

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_pre_k1_cuda_core launch error: ",
              cudaGetErrorString(err));
}

// ============================================================================
// mHC_pre_k1_tensor_core: raw-PTX tcgen05 pre_k1 (no CUTLASS/CuTe). Same
// outputs as the cutlass mHC_pre_k1 (mixes_pad bf16 [tokens,128] + sqrsum).
// bf16 operands, kind::f16 MMA, hand-written TMA/TMEM/mbarrier.
// ============================================================================

template <int K,
          int OUT_PAD,
          int BLOCK_N,
          int BLOCK_K,
          int MIX_HC,
          int NUM_STAGES>
void launch_pre_k1_tensor_core(void *residual_ptr,
                               void *weight_ptr,
                               void *mixes_pad_ptr,
                               void *sqrsum_ptr,
                               int batch,
                               cudaStream_t stream) {
  CUtensorMap A_tmap{}, B_tmap{};
  // A = weight fn [OUT_PAD, K]; B = residual [batch, K]. 128B swizzle pins the
  // TMA box's contiguous-K dim to 64 bf16; the kernel issues BLOCK_K/64 such
  // loads per stage, so the descriptor box-K is always 64 (NOT BLOCK_K).
  ::init_2d_bf16_tmap(&A_tmap, weight_ptr, OUT_PAD, K, 64, OUT_PAD);
  ::init_2d_bf16_tmap(&B_tmap, residual_ptr, batch, K, 64, BLOCK_N);

  constexpr int A_bytes = OUT_PAD * BLOCK_K * 2;
  constexpr int B_bytes = BLOCK_N * BLOCK_K * 2;
  constexpr int smem_bytes = (A_bytes + B_bytes) * NUM_STAGES;

  auto *kp =
      &kernel::pre_k1_tensor_core::mHC_pre_k1_tensor_core_kernel<K,
                                                                 OUT_PAD,
                                                                 BLOCK_N,
                                                                 BLOCK_K,
                                                                 MIX_HC,
                                                                 NUM_STAGES>;
  // Opt in to >48KB dynamic smem. Use >= : at exactly 48KB the default cap is
  // already saturated by driver reserve, so the launch needs the opt-in too.
  if (smem_bytes >= 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  dim3 grid((batch + BLOCK_N - 1) / BLOCK_N, 1, 1);
  dim3 block(OUT_PAD + 2 * 32, 1, 1);
  kp<<<grid, block, smem_bytes, stream>>>(
      A_tmap,
      B_tmap,
      reinterpret_cast<__nv_bfloat16 const *>(residual_ptr),
      reinterpret_cast<__nv_bfloat16 *>(mixes_pad_ptr),
      static_cast<float *>(sqrsum_ptr),
      batch);
}

// Tunable tile configs, indexed by `cfg`. Each is (BLOCK_N, BLOCK_K,
// NUM_STAGES). cfg=-1 selects the per-(K,batch) tuned default below.
//   0: 16/64/4   1: 32/64/4   2: 32/128/3  3: 64/64/4
//   4: 64/128/3  5: 128/64/3  6: 128/128/2 7: 16/128/4
//   8: 32/256/2  9: 64/256/2
void mHC_pre_k1_tensor_core(torch::Tensor residual,
                            torch::Tensor weight_padded,
                            torch::Tensor mixes_pad,
                            torch::Tensor sqrsum,
                            int n,
                            int cfg) {
  TORCH_CHECK(n == 4, "pre_k1_tensor_core hardcoded to n=4");
  constexpr int OUT_PAD = 128;
  int const batch = static_cast<int>(residual.size(0));
  int const K = static_cast<int>(residual.size(1));
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.dim() == 2,
              "residual must be bf16 [batch, K]");
  TORCH_CHECK(weight_padded.is_cuda() && weight_padded.is_contiguous() &&
                  weight_padded.scalar_type() == at::kBFloat16 &&
                  weight_padded.size(0) == OUT_PAD &&
                  weight_padded.size(1) == K,
              "weight_padded must be bf16 [128, K]");
  TORCH_CHECK(mixes_pad.is_cuda() && mixes_pad.is_contiguous() &&
                  mixes_pad.scalar_type() == at::kBFloat16 &&
                  mixes_pad.sizes() == torch::IntArrayRef({batch, OUT_PAD}),
              "mixes_pad must be bf16 [batch, 128]");
  TORCH_CHECK(sqrsum.is_cuda() && sqrsum.is_contiguous() &&
                  sqrsum.scalar_type() == at::kFloat && sqrsum.size(0) == batch,
              "sqrsum must be float32 [batch]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  void *res_p = const_cast<void *>(residual.data_ptr());
  void *w_p = const_cast<void *>(weight_padded.data_ptr());
  void *mx_p = const_cast<void *>(mixes_pad.data_ptr());
  void *sq_p = const_cast<float *>(sqrsum.data_ptr<float>());

  // NUM_STAGES and OUTPUT_PAD are FIXED (NS=2, OUT_PAD=128). The sweep grid is
  // BLOCK_N x BLOCK_K = {16,32,64,128} x {64,128,256}; cfg = bn_idx*3 + bk_idx.
  // BN=128/BK=256 (cfg 11) exceeds the 224 KB smem cap at NS=2, so it's elided.
  constexpr int SWEEP_NS = 2;
  int sel = cfg;
  if (sel < 0) {
    // Tuned default (full BN x BK sweep, B200): small N tile fills the GPU and
    // large BLOCK_K cuts pipeline iterations. cfg 2 = BN16/BK256 wins for
    // t<=1024 across all c. At t>=4096 the larger K-tile's smem pressure starts
    // to bite, so drop to BK=128 (cfg 1 = BN16/BK128).
    sel = (batch >= 4096) ? 1 /*BN16/BK128*/ : 2 /*BN16/BK256*/;
  }

  // (K_, BN, BK, NS) instantiations. cfg 0..10 use SWEEP_NS=2; cfg 11+ probe
  // LARGER tiles that only fit by trading stages: BK=512 needs NS=1 (the A
  // weight tile dominates smem), and BK=256/NS=3 spends the extra budget on a
  // deeper pipeline instead.
#define LAUNCH_CFG(K_, BN, BK)                                                 \
  launch_pre_k1_tensor_core<K_, OUT_PAD, BN, BK, 24, SWEEP_NS>(                \
      res_p, w_p, mx_p, sq_p, batch, stream)
#define LAUNCH_CFG_NS(K_, BN, BK, NS)                                          \
  launch_pre_k1_tensor_core<K_, OUT_PAD, BN, BK, 24, NS>(                      \
      res_p, w_p, mx_p, sq_p, batch, stream)
  // Configs 0..10 (BK<=256) are valid for every supported K. The large-BK
  // probes (BK in {512,1024}) require K divisible by that BK, so they're only
  // instantiated in the LARGE_K branch (K>=4096); LARGE=0 omits them.
#define DISPATCH_CFG(K_, LARGE)                                                \
  switch (sel) {                                                               \
    case 0:                                                                    \
      LAUNCH_CFG(K_, 16, 64);                                                  \
      break;                                                                   \
    case 1:                                                                    \
      LAUNCH_CFG(K_, 16, 128);                                                 \
      break;                                                                   \
    case 2:                                                                    \
      LAUNCH_CFG(K_, 16, 256);                                                 \
      break;                                                                   \
    case 3:                                                                    \
      LAUNCH_CFG(K_, 32, 64);                                                  \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_CFG(K_, 32, 128);                                                 \
      break;                                                                   \
    case 5:                                                                    \
      LAUNCH_CFG(K_, 32, 256);                                                 \
      break;                                                                   \
    case 6:                                                                    \
      LAUNCH_CFG(K_, 64, 64);                                                  \
      break;                                                                   \
    case 7:                                                                    \
      LAUNCH_CFG(K_, 64, 128);                                                 \
      break;                                                                   \
    case 8:                                                                    \
      LAUNCH_CFG(K_, 64, 256);                                                 \
      break;                                                                   \
    case 9:                                                                    \
      LAUNCH_CFG(K_, 128, 64);                                                 \
      break;                                                                   \
    case 10:                                                                   \
      LAUNCH_CFG(K_, 128, 128);                                                \
      break;                                                                   \
      DISPATCH_LARGE_##LARGE(K_) default                                       \
          : TORCH_CHECK(false, "Unsupported cfg=", sel, " for K=", K);         \
  }
#define DISPATCH_LARGE_0(K_)
// BK=512 fits only at NS=1 (A weight tile dominates); BK=1024 is over the
// 224KB cap even at NS=1, so 512 is the largest BK we can run.
#define DISPATCH_LARGE_1(K_)                                                   \
  case 11:                                                                     \
    LAUNCH_CFG_NS(K_, 16, 512, 1);                                             \
    break;                                                                     \
  case 12:                                                                     \
    LAUNCH_CFG_NS(K_, 32, 512, 1);                                             \
    break;                                                                     \
  case 13:                                                                     \
    LAUNCH_CFG_NS(K_, 16, 256, 3);                                             \
    break;
  switch (K) {
    case 512:
      DISPATCH_CFG(512, 0);
      break;
    case 4096:
      DISPATCH_CFG(4096, 1);
      break;
    case 16384:
      DISPATCH_CFG(16384, 1);
      break;
    case 28672:
      DISPATCH_CFG(28672, 1);
      break;
    default:
      TORCH_CHECK(false, "Unsupported K=", K);
  }
#undef DISPATCH_CFG
#undef DISPATCH_LARGE_0
#undef DISPATCH_LARGE_1
#undef LAUNCH_CFG
#undef LAUNCH_CFG_NS
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_pre_k1_tensor_core launch error: ",
              cudaGetErrorString(err));
}

template <int N, int C, int RMS_HIDDEN, int TOKENS_PER_CTA>
__global__ __launch_bounds__(256) void mHC_pre_k2_kernel(
    void const *__restrict__ mixes_pad,
    void const *__restrict__ sqrsum,
    void const *__restrict__ scale_ptr,
    void const *__restrict__ base_ptr,
    void const *__restrict__ x_orig_bf16,
    void *__restrict__ f_pre,
    void *__restrict__ h_post_out,
    void *__restrict__ comb_out,
    int num_tokens,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    float rms_eps) {
  extern __shared__ char smem_k2[];
  kernel::mHC_pre_k2_task_impl<mpk_bf16,
                               N,
                               C,
                               RMS_HIDDEN,
                               TOKENS_PER_CTA,
                               /*BLOCK_THREADS=*/256,
                               /*MIX_STRIDE=*/128>(mixes_pad,
                                                   sqrsum,
                                                   scale_ptr,
                                                   base_ptr,
                                                   x_orig_bf16,
                                                   f_pre,
                                                   h_post_out,
                                                   comb_out,
                                                   sinkhorn_repeat,
                                                   sinkhorn_eps,
                                                   rms_eps,
                                                   num_tokens,
                                                   smem_k2);
}

// Low-t k2: one CTA per token (grid = num_tokens) to fill the SMs at small
// batch, where the 32-tokens/CTA default leaves only ceil(t/32) blocks idle.
template <int N, int C, int RMS_HIDDEN>
__global__ __launch_bounds__(256) void mHC_pre_k2_lowt_kernel(
    void const *__restrict__ mixes_pad,
    void const *__restrict__ sqrsum,
    void const *__restrict__ scale_ptr,
    void const *__restrict__ base_ptr,
    void const *__restrict__ x_orig_bf16,
    void *__restrict__ f_pre,
    void *__restrict__ h_post_out,
    void *__restrict__ comb_out,
    int num_tokens,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    float rms_eps) {
  kernel::mHC_pre_k2_lowt_task_impl<mpk_bf16,
                                    N,
                                    C,
                                    RMS_HIDDEN,
                                    /*BLOCK_THREADS=*/256,
                                    /*MIX_STRIDE=*/128>(mixes_pad,
                                                        sqrsum,
                                                        scale_ptr,
                                                        base_ptr,
                                                        x_orig_bf16,
                                                        f_pre,
                                                        h_post_out,
                                                        comb_out,
                                                        sinkhorn_repeat,
                                                        sinkhorn_eps,
                                                        rms_eps,
                                                        num_tokens);
}

// Fused low-t k2: reduces the k1 GEMM's split-k partials INLINE then runs the
// tail, folding the separate reduce launch into k2 (3 launches -> 2 at low t).
// mixes_ptr/sqrsum_ptr point at out_partial / sqr_partial.
template <int N, int C, int RMS_HIDDEN, int SPLIT_K>
__global__ __launch_bounds__(256) void mHC_pre_k2_lowt_fused_kernel(
    void const *__restrict__ out_partial,
    void const *__restrict__ sqr_partial,
    void const *__restrict__ scale_ptr,
    void const *__restrict__ base_ptr,
    void const *__restrict__ x_orig_bf16,
    void *__restrict__ f_pre,
    void *__restrict__ h_post_out,
    void *__restrict__ comb_out,
    int num_tokens,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    float rms_eps) {
  kernel::mHC_pre_k2_lowt_task_impl<mpk_bf16,
                                    N,
                                    C,
                                    RMS_HIDDEN,
                                    /*BLOCK_THREADS=*/256,
                                    /*MIX_STRIDE=*/128,
                                    /*RDSPLIT_K=*/SPLIT_K>(out_partial,
                                                           sqr_partial,
                                                           scale_ptr,
                                                           base_ptr,
                                                           x_orig_bf16,
                                                           f_pre,
                                                           h_post_out,
                                                           comb_out,
                                                           sinkhorn_repeat,
                                                           sinkhorn_eps,
                                                           rms_eps,
                                                           num_tokens);
}

void mHC_pre_k2(torch::Tensor mixes_pad,
                torch::Tensor sqrsum,
                torch::Tensor scale,
                torch::Tensor base,
                torch::Tensor x_orig,
                torch::Tensor f_pre,
                torch::Tensor h_post,
                torch::Tensor comb,
                int n,
                int c,
                int rms_hidden,
                int sinkhorn_repeat,
                double sinkhorn_eps,
                double rms_eps,
                int num_ctas_arg,
                int tokens_per_cta) {
  TORCH_CHECK(n == 4, "pre K2 hardcoded to n=4");
  TORCH_CHECK(mixes_pad.is_cuda() && mixes_pad.is_contiguous() &&
                  mixes_pad.dim() == 2 &&
                  mixes_pad.scalar_type() == at::kBFloat16 &&
                  mixes_pad.size(1) == 128,
              "mixes_pad must be bf16 [bs, 128] CUDA contiguous");
  TORCH_CHECK(sqrsum.is_cuda() && sqrsum.is_contiguous() &&
                  sqrsum.scalar_type() == at::kFloat && sqrsum.dim() == 1,
              "sqrsum must be float32 [bs] CUDA contiguous");
  TORCH_CHECK(tokens_per_cta == 32 || tokens_per_cta == 64 ||
                  tokens_per_cta == 128,
              "tokens_per_cta must be 32, 64, or 128");

  int const num_tokens = static_cast<int>(mixes_pad.size(0));
  TORCH_CHECK(sqrsum.size(0) == num_tokens, "sqrsum bs mismatch");

  int const num_ctas = resolve_num_ctas(num_ctas_arg, mixes_pad.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(mixes_pad.get_device());

  // smem: TOKENS_PER_CTA * (4*N + N*N + N*N + N) floats for the tail buffers,
  // plus the small rms_scale[] array. 128 covers the largest tile.
  size_t const smemBytes =
      tokens_per_cta * (4 * n + n * n + n * n + n) * sizeof(float) + 1024;

  void *mixes_p = const_cast<void *>(mixes_pad.data_ptr());
  void *sq_p = const_cast<float *>(sqrsum.data_ptr<float>());
  void *scale_p = const_cast<float *>(scale.data_ptr<float>());
  void *base_p = const_cast<float *>(base.data_ptr<float>());
  void *x_orig_p = const_cast<void *>(x_orig.data_ptr());
  void *f_pre_p = const_cast<void *>(f_pre.data_ptr());
  void *h_post_p = const_cast<float *>(h_post.data_ptr<float>());
  void *comb_p = const_cast<float *>(comb.data_ptr<float>());
  float sk_eps_f = static_cast<float>(sinkhorn_eps);
  float rms_eps_f = static_cast<float>(rms_eps);

  int const tiles = ceil_div(num_tokens, tokens_per_cta);
  int const grid = num_ctas < tiles ? num_ctas : tiles;
  dim3 grid_dim(grid > 0 ? grid : 1, 1, 1);
  dim3 block_dim(256, 1, 1);

#define LAUNCH_PRE_K2(C_, RH_, TPC_)                                           \
  do {                                                                         \
    auto *kp = &mHC_pre_k2_kernel<4, C_, RH_, TPC_>;                           \
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(                                     \
        kp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));     \
    kp<<<grid_dim, block_dim, smemBytes, stream>>>(mixes_p,                    \
                                                   sq_p,                       \
                                                   scale_p,                    \
                                                   base_p,                     \
                                                   x_orig_p,                   \
                                                   f_pre_p,                    \
                                                   h_post_p,                   \
                                                   comb_p,                     \
                                                   num_tokens,                 \
                                                   sinkhorn_repeat,            \
                                                   sk_eps_f,                   \
                                                   rms_eps_f);                 \
  } while (0)

#define DISPATCH_PRE_K2_TPC(C_, RH_)                                           \
  switch (tokens_per_cta) {                                                    \
    case 32:                                                                   \
      LAUNCH_PRE_K2(C_, RH_, 32);                                              \
      break;                                                                   \
    case 64:                                                                   \
      LAUNCH_PRE_K2(C_, RH_, 64);                                              \
      break;                                                                   \
    case 128:                                                                  \
      LAUNCH_PRE_K2(C_, RH_, 128);                                             \
      break;                                                                   \
  }

  // Low-t path: the default packs 32 tokens/CTA, so a batch under ~32*SMs
  // leaves the GPU under-filled. Below the threshold, route to the one-CTA-per-
  // token kernel (grid = num_tokens) which fills the SMs. Threshold = 32 * a
  // typical SM count, so we only switch when the default grid is clearly small.
  int const k2_lowt_thresh = 32 * resolve_num_ctas(0, mixes_pad.get_device());
  bool const use_lowt = (num_tokens < k2_lowt_thresh);

#define LAUNCH_PRE_K2_LOWT(C_, RH_)                                            \
  do {                                                                         \
    dim3 lg(num_tokens, 1, 1);                                                 \
    mHC_pre_k2_lowt_kernel<4, C_, RH_>                                         \
        <<<lg, block_dim, 0, stream>>>(mixes_p,                                \
                                       sq_p,                                   \
                                       scale_p,                                \
                                       base_p,                                 \
                                       x_orig_p,                               \
                                       f_pre_p,                                \
                                       h_post_p,                               \
                                       comb_p,                                 \
                                       num_tokens,                             \
                                       sinkhorn_repeat,                        \
                                       sk_eps_f,                               \
                                       rms_eps_f);                             \
  } while (0)

#define DISPATCH_PRE_K2_C(C_, RH_)                                             \
  do {                                                                         \
    if (use_lowt) {                                                            \
      LAUNCH_PRE_K2_LOWT(C_, RH_);                                             \
    } else {                                                                   \
      DISPATCH_PRE_K2_TPC(C_, RH_);                                            \
    }                                                                          \
  } while (0)

  // RMS_HIDDEN is the residual reduction dim K = n*c.
  TORCH_CHECK(rms_hidden == n * c, "rms_hidden must equal n*c");
  switch (c) {
    case 128:
      DISPATCH_PRE_K2_C(128, 4 * 128);
      break;
    case 1024:
      DISPATCH_PRE_K2_C(1024, 4 * 1024);
      break;
    case 4096:
      DISPATCH_PRE_K2_C(4096, 4 * 4096);
      break;
    case 7168:
      DISPATCH_PRE_K2_C(7168, 4 * 7168);
      break;
    default:
      TORCH_CHECK(false, "Unsupported c=", c);
  }

#undef DISPATCH_PRE_K2_C
#undef LAUNCH_PRE_K2_LOWT
#undef DISPATCH_PRE_K2_TPC
#undef LAUNCH_PRE_K2

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess, "mhc_pre_k2 launch error: ", cudaGetErrorString(err));
}

// ============================================================================
// mHC_pre_k1: heuristic dispatch between the CUDA-core and raw-PTX tcgen05
// implementations. Takes the raw-PTX-style inputs (residual [tokens,K] bf16 +
// weight_padded [128,K] bf16); when it routes to the CUDA-core path it derives
// the fp32 fn[mix_hc,K] + split-k scratch internally.
//
// Heuristic (from the B200 sweep): tensor cores have a large fixed MMA/TMA/TMEM
// setup cost, so they only win once there are enough tokens to amortize it.
//   * tokens < CUDA_T_THRESH -> CUDA-core (FFMA + split-k): wins at decode.
//   * otherwise              -> raw-PTX tcgen05: wins at prefill.
// The threshold ~256 is where the sweep showed the crossover across c.
// ============================================================================
static int pick_cuda_split_k(int K, int num_tokens) {
  // Two competing needs: fill the grid (tokens*split_k >= ~SM count) and keep
  // each CTA's serial reduction long enough (K/split_k >= ~1024 elems). B200
  // sweep optimum: ~512-1024 total CTAs, never below 1024 elems/CTA, cap 32.
  int grid_target =
      (num_tokens >= 1024) ? 1 : (1024 + num_tokens - 1) / num_tokens;
  int const work_cap = (K >= 1024) ? (K / 1024) : 1; // don't go below ~1024/CTA
  if (grid_target > work_cap) {
    grid_target = work_cap;
  }
  if (grid_target > 32) {
    grid_target = 32;
  }
  int sk = 1;
  for (int cand : {1, 2, 4, 8, 16, 32}) {
    if (K % cand == 0 && cand <= grid_target) {
      sk = cand;
    }
  }
  return sk;
}

void mHC_pre_k1(torch::Tensor residual,
                torch::Tensor weight_padded,
                torch::Tensor mixes_pad,
                torch::Tensor sqrsum,
                int n) {
  TORCH_CHECK(n == 4, "pre_k1 hardcoded to n=4");
  int const mix_hc = n * n + 2 * n;
  int const num_tokens = static_cast<int>(residual.size(0));
  int const K = static_cast<int>(residual.size(1));

  // Crossover threshold: below it, CUDA-core; at/above it, tcgen05 raw-PTX.
  constexpr int CUDA_T_THRESH = 256;
  if (num_tokens < CUDA_T_THRESH) {
    // CUDA-core path: needs fp32 fn[mix_hc,K] + split-k scratch.
    auto fn =
        weight_padded.slice(0, 0, mix_hc).to(torch::kFloat32).contiguous();
    int const split_k = pick_cuda_split_k(K, num_tokens);
    auto opts_f =
        torch::TensorOptions().dtype(torch::kFloat32).device(residual.device());
    auto out_partial = torch::empty({split_k, num_tokens, mix_hc}, opts_f);
    auto sqr_partial = torch::empty({split_k, num_tokens}, opts_f);
    mHC_pre_k1_cuda_core(
        residual, fn, out_partial, sqr_partial, mixes_pad, sqrsum, n, split_k);
  } else {
    mHC_pre_k1_tensor_core(
        residual, weight_padded, mixes_pad, sqrsum, n, /*cfg=*/-1);
  }
}

// ============================================================================
// mHC_pre: the full prenorm pipeline (k1 GEMM + k2 tail) in as few launches as
// possible. For the low-t cuda_core path this fuses the GEMM's split-k reduce
// into the k2 tail -> 2 launches (GEMM + fused-k2) instead of 3 (GEMM + reduce
// + k2), removing the per-launch overhead that dominates at decode. For the
// high-t tensor_core path it's the standard k1 + k2.
// ============================================================================
template <int N, int C, int RMS_HIDDEN, int SPLIT_K>
static void launch_pre_fused_k2(void *outp,
                                void *sqrp,
                                void *scale,
                                void *base,
                                void *x_orig,
                                void *f_pre,
                                void *h_post,
                                void *comb,
                                int num_tokens,
                                int sinkhorn_repeat,
                                float sk_eps,
                                float rms_eps,
                                cudaStream_t stream) {
  dim3 g(num_tokens, 1, 1), b(256, 1, 1);
  mHC_pre_k2_lowt_fused_kernel<N, C, RMS_HIDDEN, SPLIT_K>
      <<<g, b, 0, stream>>>(outp,
                            sqrp,
                            scale,
                            base,
                            x_orig,
                            f_pre,
                            h_post,
                            comb,
                            num_tokens,
                            sinkhorn_repeat,
                            sk_eps,
                            rms_eps);
}

void mHC_pre(torch::Tensor residual,
             torch::Tensor weight_padded,
             torch::Tensor x_orig,
             torch::Tensor scale,
             torch::Tensor base,
             torch::Tensor f_pre,
             torch::Tensor h_post,
             torch::Tensor comb,
             torch::Tensor mixes_pad,
             torch::Tensor sqrsum,
             int n,
             int c,
             int sinkhorn_repeat,
             double sinkhorn_eps,
             double rms_eps,
             int tokens_per_cta) {
  TORCH_CHECK(n == 4, "mHC_pre hardcoded to n=4");
  int const mix_hc = n * n + 2 * n;
  int const num_tokens = static_cast<int>(residual.size(0));
  int const K = static_cast<int>(residual.size(1));
  TORCH_CHECK(K == n * c, "K must equal n*c");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  float const sk_eps = static_cast<float>(sinkhorn_eps);
  float const rms_eps_f = static_cast<float>(rms_eps);

  constexpr int CUDA_T_THRESH = 256;
  bool const fused = (num_tokens < CUDA_T_THRESH);

  if (fused) {
    // --- low-t: cuda_core GEMM (writes partials) + fused-reduce-k2 ---
    auto fn =
        weight_padded.slice(0, 0, mix_hc).to(torch::kFloat32).contiguous();
    int const split_k = pick_cuda_split_k(K, num_tokens);
    auto opts_f =
        torch::TensorOptions().dtype(torch::kFloat32).device(residual.device());
    auto out_partial = torch::empty({split_k, num_tokens, mix_hc}, opts_f);
    auto sqr_partial = torch::empty({split_k, num_tokens}, opts_f);
    // GEMM only -- write partials (split_k>=1 always writes partials here so
    // the fused k2 can reduce them; we never call the standalone reduce
    // kernel).
    void *res_p = const_cast<void *>(residual.data_ptr());
    float *fn_p = fn.data_ptr<float>();
    float *outp_p = out_partial.data_ptr<float>();
    float *sqrp_p = sqr_partial.data_ptr<float>();
    void *mx_p = const_cast<void *>(mixes_pad.data_ptr()); // unused when SK>1
    float *ss_p = sqrsum.data_ptr<float>();                // unused when SK>1
    constexpr int BT = 128;

#define PRE_GEMM(K_, SK_)                                                      \
  do {                                                                         \
    dim3 grid((num_tokens + 0), SK_, 1);                                       \
    mHC_pre_k1_cuda_core_kernel<24, K_, BT, SK_, 128, 1>                       \
        <<<grid, dim3(BT, 1, 1), 0, stream>>>(                                 \
            res_p, fn_p, outp_p, sqrp_p, mx_p, ss_p, num_tokens);              \
  } while (0)
#define PRE_FUSEDK2(C_, RH_, SK_)                                              \
  launch_pre_fused_k2<4, C_, RH_, SK_>(                                        \
      outp_p,                                                                  \
      sqrp_p,                                                                  \
      const_cast<float *>(scale.data_ptr<float>()),                            \
      const_cast<float *>(base.data_ptr<float>()),                             \
      const_cast<void *>(x_orig.data_ptr()),                                   \
      const_cast<void *>(f_pre.data_ptr()),                                    \
      const_cast<float *>(h_post.data_ptr<float>()),                           \
      const_cast<float *>(comb.data_ptr<float>()),                             \
      num_tokens,                                                              \
      sinkhorn_repeat,                                                         \
      sk_eps,                                                                  \
      rms_eps_f,                                                               \
      stream)

#define PRE_DISPATCH_SK(K_, C_, RH_)                                           \
  switch (split_k) {                                                           \
    case 1:                                                                    \
      PRE_GEMM(K_, 1);                                                         \
      PRE_FUSEDK2(C_, RH_, 1);                                                 \
      break;                                                                   \
    case 2:                                                                    \
      PRE_GEMM(K_, 2);                                                         \
      PRE_FUSEDK2(C_, RH_, 2);                                                 \
      break;                                                                   \
    case 4:                                                                    \
      PRE_GEMM(K_, 4);                                                         \
      PRE_FUSEDK2(C_, RH_, 4);                                                 \
      break;                                                                   \
    case 8:                                                                    \
      PRE_GEMM(K_, 8);                                                         \
      PRE_FUSEDK2(C_, RH_, 8);                                                 \
      break;                                                                   \
    case 16:                                                                   \
      PRE_GEMM(K_, 16);                                                        \
      PRE_FUSEDK2(C_, RH_, 16);                                                \
      break;                                                                   \
    case 32:                                                                   \
      PRE_GEMM(K_, 32);                                                        \
      PRE_FUSEDK2(C_, RH_, 32);                                                \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "bad split_k=", split_k);                             \
  }

    switch (c) {
      case 1024:
        PRE_DISPATCH_SK(4096, 1024, 4 * 1024);
        break;
      case 4096:
        PRE_DISPATCH_SK(16384, 4096, 4 * 4096);
        break;
      case 7168:
        PRE_DISPATCH_SK(28672, 7168, 4 * 7168);
        break;
      default:
        TORCH_CHECK(false, "Unsupported c=", c);
    }
#undef PRE_DISPATCH_SK
#undef PRE_FUSEDK2
#undef PRE_GEMM
  } else {
    // --- high-t: standard tensor_core k1 + k2 ---
    mHC_pre_k1_tensor_core(residual,
                           weight_padded,
                           mixes_pad,
                           sqrsum,
                           n,
                           /*cfg=*/-1);
    mHC_pre_k2(mixes_pad,
               sqrsum,
               scale,
               base,
               x_orig,
               f_pre,
               h_post,
               comb,
               n,
               c,
               K,
               sinkhorn_repeat,
               sinkhorn_eps,
               rms_eps,
               0,
               tokens_per_cta);
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess, "mHC_pre launch error: ", cudaGetErrorString(err));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // num_ctas=0 means "use device SM count". Caller can pin to 128 / 148 / etc.
  m.def("mHC_post",
        &mHC_post,
        py::arg("residual"),
        py::arg("x"),
        py::arg("comb"),
        py::arg("post"),
        py::arg("output"),
        py::arg("n"),
        py::arg("num_ctas") = 0,
        "mHC post: y[k,c] = post[k]*x[c] + sum_i comb[i,k]*residual[i,c] "
        "(comb NOT transposed; matches torch hc_post)");

  m.def(
      "mHC_pre_k1_tensor_core",
      &mHC_pre_k1_tensor_core,
      py::arg("residual"),
      py::arg("weight_padded"),
      py::arg("mixes_pad"),
      py::arg("sqrsum"),
      py::arg("n"),
      py::arg("cfg") = -1,
      "Raw-PTX tcgen05 pre_k1 (no CUTLASS/CuTe): mixes = residual @ fn.T + "
      "sqrsum, bf16 kind::f16 MMA. Same outputs as mHC_pre_k1. cfg=-1 uses "
      "the tuned tile config; 0..9 force a specific (BLOCK_N,BLOCK_K,STAGES).");

  m.def("mHC_pre_k1_cuda_core",
        &mHC_pre_k1_cuda_core,
        py::arg("residual"),
        py::arg("fn"),
        py::arg("out_partial"),
        py::arg("sqr_partial"),
        py::arg("mixes_pad"),
        py::arg("sqrsum"),
        py::arg("n"),
        py::arg("split_k") = 1,
        "CUDA-core pre_k1: mixes = residual @ fn.T + per-token sqrsum (no "
        "tensor cores, split-k). Same outputs as the tcgen05 mHC_pre_k1; wins "
        "at low token count.");

  m.def("mHC_post_pre_v2",
        &mHC_post_pre_v2,
        py::arg("residual_in"),
        py::arg("x_in"),
        py::arg("comb_in"),
        py::arg("post_in"),
        py::arg("fn"),
        py::arg("residual_next"),
        py::arg("out_partial"),
        py::arg("sqr_partial"),
        py::arg("mixes_pad"),
        py::arg("sqrsum"),
        py::arg("scale"),
        py::arg("base"),
        py::arg("f_pre"),
        py::arg("h_post"),
        py::arg("comb"),
        py::arg("n"),
        py::arg("c"),
        py::arg("split_k") = 1,
        py::arg("sinkhorn_repeat") = 20,
        py::arg("sinkhorn_eps") = 1e-9,
        py::arg("rms_eps") = 1e-6,
        py::arg("tokens_per_cta") = 32,
        "CUDA-core fused post + prenorm-GEMM (vLLM mhc_fused style) + split-k "
        "reduce + k2 tail. Outputs next layer f_pre / h_post / comb.");

  m.def("sinkhorn_sm100",
        &sinkhorn_sm100,
        py::arg("comb_res_mix"),
        py::arg("comb_res_mix_out"),
        py::arg("repeat") = 20,
        py::arg("eps") = 1e-9,
        py::arg("num_ctas") = 0,
        "mHC K3: Sinkhorn-Knopp normalization (4x4)");

  m.def(
      "mHC_pre_k1",
      &mHC_pre_k1,
      py::arg("residual"),
      py::arg("weight_padded"),
      py::arg("mixes_pad"),
      py::arg("sqrsum"),
      py::arg("n"),
      "mHC pre K1: mixes = residual @ fn.T + per-token sqrsum. Heuristically "
      "dispatches to the CUDA-core impl (tokens<256, decode) or the raw-PTX "
      "tcgen05 impl (tokens>=256, prefill). residual bf16 [tokens,K], weight "
      "padded to [128,K], mixes_pad bf16 [tokens,128], sqrsum fp32 [tokens].");

  m.def("mHC_pre",
        &mHC_pre,
        py::arg("residual"),
        py::arg("weight_padded"),
        py::arg("x_orig"),
        py::arg("scale"),
        py::arg("base"),
        py::arg("f_pre"),
        py::arg("h_post"),
        py::arg("comb"),
        py::arg("mixes_pad"),
        py::arg("sqrsum"),
        py::arg("n"),
        py::arg("c"),
        py::arg("sinkhorn_repeat") = 20,
        py::arg("sinkhorn_eps") = 1e-9,
        py::arg("rms_eps") = 1e-6,
        py::arg("tokens_per_cta") = 32,
        "Full mHC pre (k1 GEMM + k2 tail) in minimal launches. Low t (<256): "
        "cuda_core GEMM + fused-reduce-k2 = 2 launches. High t: tensor_core k1 "
        "+ k2. Outputs f_pre/h_post/comb (mixes_pad/sqrsum are scratch).");

  m.def("mHC_pre_k2",
        &mHC_pre_k2,
        py::arg("mixes_pad"),
        py::arg("sqrsum"),
        py::arg("scale"),
        py::arg("base"),
        py::arg("x_orig"),
        py::arg("f_pre"),
        py::arg("h_post"),
        py::arg("comb"),
        py::arg("n"),
        py::arg("c"),
        py::arg("rms_hidden"),
        py::arg("sinkhorn_repeat") = 20,
        py::arg("sinkhorn_eps") = 1e-9,
        py::arg("rms_eps") = 1e-6,
        py::arg("num_ctas") = 0,
        py::arg("tokens_per_cta") = 32,
        "mHC pre K2 (vLLM split): rms-scale gemm output (via sqrsum) + "
        "pre/post/comb mix (sinkhorn) + pre-weighted residual sum.");
}
