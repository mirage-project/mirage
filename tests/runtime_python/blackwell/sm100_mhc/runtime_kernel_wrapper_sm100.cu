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
#include "runtime_header.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>

#include <cooperative_groups.h>

#include "blackwell/mHC_post.cuh"
#include "blackwell/mHC_post_pre.cuh"
#include "blackwell/mHC_pre.cuh"
#include "blackwell/sinkhorn.cuh"
#include <ATen/cuda/CUDAContext.h>

using bf16_t = __nv_bfloat16;

#define MHC_CUDA_CHECK(e)                                                      \
  do {                                                                         \
    cudaError_t _err = (e);                                                    \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ",                           \
                cudaGetErrorString(_err));                                     \
  } while (0)

namespace {

constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

static void check_tensor(const torch::Tensor &t,
                         at::ScalarType dt,
                         c10::IntArrayRef shape,
                         const char *name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt &&
                  t.sizes() == shape,
              name, " must be ", dt, " ", shape,
              " CUDA contiguous, got ", t.sizes(), " ", t.scalar_type());
}

template <typename T, int N, int C, int TOKENS_PER_BLK>
__global__ __launch_bounds__(256) void mHC_post_kernel(void const *residual_ptr,
                                                       void const *x_ptr,
                                                       void const *comb_ptr,
                                                       void const *post_ptr,
                                                       void *output_ptr,
                                                       int num_tokens) {
  int const threads_per_token = blockDim.x / TOKENS_PER_BLK;
  int const group = threadIdx.x / threads_per_token;
  int const lane = threadIdx.x % threads_per_token;
  for (int64_t tile = blockIdx.x; tile * TOKENS_PER_BLK < num_tokens; tile += gridDim.x) {
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
                               1,
                               C,
                               N,
                               C>(
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
      LAUNCH_POST(128, 8);
      break;
    case 1024:
      LAUNCH_POST(1024, 2);
      break;
    case 4096:
      LAUNCH_POST(4096, 1);
      break;
    case 7168:
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
              int n) {
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.dim() == 3 && residual.size(1) == n,
              "residual must be bf16 [num_tokens, n, c] CUDA contiguous");

  int const num_tokens = static_cast<int>(residual.size(0));
  int const c = static_cast<int>(residual.size(2));

  check_tensor(x, at::kBFloat16, {num_tokens, c}, "x");
  check_tensor(comb, at::kFloat, {num_tokens, n, n}, "comb");
  check_tensor(post, at::kFloat, {num_tokens, n}, "post");
  check_tensor(output, at::kBFloat16, {num_tokens, n, c}, "output");

  int num_ctas = 0;
  cudaDeviceGetAttribute(&num_ctas, cudaDevAttrMultiProcessorCount, residual.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  bf16_t const *residual_ptr = reinterpret_cast<bf16_t const *>(residual.data_ptr());
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

__global__ __launch_bounds__(256) void sinkhorn_sm100_kernel(
    float const *__restrict__ comb_res_mix,
    float *__restrict__ comb_res_mix_out,
    int num_tokens,
    int repeat,
    float eps) {
  constexpr int token_stride = 16;
  kernel::sinkhorn_task_impl<token_stride, token_stride>(
      comb_res_mix, comb_res_mix_out, num_tokens, repeat, eps);
}

void sinkhorn_sm100(torch::Tensor comb_res_mix,
                    torch::Tensor comb_res_mix_out,
                    int repeat,
                    double eps) {
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

  constexpr int THREADS_PER_BLOCK = 256;
  int const num_tokens = static_cast<int>(comb_res_mix.size(0));
  float const eps_f = static_cast<float>(eps);
  float const *input_ptr = comb_res_mix.data_ptr<float>();
  float *output_ptr = comb_res_mix_out.data_ptr<float>();
  int num_ctas = 0;
  cudaDeviceGetAttribute(&num_ctas, cudaDevAttrMultiProcessorCount,
                         comb_res_mix.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(comb_res_mix.get_device());

  int const tokens_per_cta_floor = ceil_div(num_tokens, THREADS_PER_BLOCK);
  int const grid = tokens_per_cta_floor < num_ctas ? tokens_per_cta_floor : num_ctas;
  dim3 const grid_dim(grid > 0 ? grid : 1, 1, 1);
  dim3 const block_dim(THREADS_PER_BLOCK, 1, 1);
  sinkhorn_sm100_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input_ptr, output_ptr, num_tokens, repeat, eps_f);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess, "Sinkhorn launch error: ", cudaGetErrorString(err));
}

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

template <int N, int C, int MIX_HC, int BLOCK_THREADS, int SPLIT_K,
          int TPB = 1, int TILE_N = MIX_HC>
__global__ __launch_bounds__(BLOCK_THREADS) void mHC_post_pre_k1_kernel(
    void const *__restrict__ residual,
    void const *__restrict__ x,
    void const *__restrict__ comb,
    void const *__restrict__ post,
    void const *__restrict__ fn,
    void *__restrict__ residual_out,
    float *__restrict__ out_partial,
    float *__restrict__ sqr_partial,
    void *__restrict__ mixes_pad,
    float *__restrict__ sqrsum,
    int num_tokens) {
  int const token0 = blockIdx.x * TPB;
  int const i_ks = blockIdx.y;
  int const i_nt = blockIdx.z;
  if (token0 >= num_tokens) {
    return;
  }
  kernel::mHC_post_pre_k1_task_impl<bf16_t,
                                    N,
                                    C,
                                    MIX_HC,
                                    BLOCK_THREADS,
                                    SPLIT_K,
                                    128,
                                    TPB,
                                    TILE_N>(
      static_cast<bf16_t const *>(residual),
      static_cast<bf16_t const *>(x),
      static_cast<float const *>(comb),
      static_cast<float const *>(post),
      static_cast<__nv_bfloat16 const *>(fn),
      static_cast<bf16_t *>(residual_out),
      out_partial,
      sqr_partial,
      mixes_pad,
      sqrsum,
      num_tokens,
      token0,
      i_ks,
      i_nt);
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
  kernel::mHC_pre_k2_lowt_task_impl<bf16_t,
                                    N,
                                    C,
                                    RMS_HIDDEN,
                                    256,
                                    128,
                                    0>(mixes_pad,
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
  kernel::mHC_pre_k2_lowt_task_impl<bf16_t,
                                    N,
                                    C,
                                    RMS_HIDDEN,
                                    256,
                                    128,
                                    SPLIT_K>(out_partial,
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

void mHC_post_pre_v2(torch::Tensor residual_in,
                     torch::Tensor x_in,
                     torch::Tensor comb_in,
                     torch::Tensor post_in,
                     torch::Tensor fn,
                     torch::Tensor residual_next,
                     torch::Tensor out_partial,
                     torch::Tensor sqr_partial,
                     torch::Tensor mixes_pad,
                     torch::Tensor sqrsum,
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
                     int tokens_per_cta,
                     int tile_n) {
  TORCH_CHECK(n == 4, "post_pre_v2 hardcoded to n=4");
  constexpr int MIX_PAD = 128;
  int const num_tokens = static_cast<int>(residual_in.size(0));
  int const mix_hc = n * n + 2 * n;
  check_tensor(residual_in, at::kBFloat16, {num_tokens, n, c}, "residual_in");
  check_tensor(x_in, at::kBFloat16, {num_tokens, c}, "x_in");
  check_tensor(comb_in, at::kFloat, {num_tokens, n, n}, "comb_in");
  check_tensor(post_in, at::kFloat, {num_tokens, n}, "post_in");
  check_tensor(fn, at::kBFloat16, {mix_hc, n, c}, "fn");
  check_tensor(
      residual_next, at::kBFloat16, {num_tokens, n, c}, "residual_next");
  check_tensor(
      out_partial, at::kFloat, {split_k, num_tokens, mix_hc}, "out_partial");
  check_tensor(sqr_partial, at::kFloat, {split_k, num_tokens}, "sqr_partial");
  check_tensor(mixes_pad, at::kBFloat16, {num_tokens, MIX_PAD}, "mixes_pad");
  check_tensor(sqrsum, at::kFloat, {num_tokens}, "sqrsum");
  TORCH_CHECK(c % split_k == 0, "c must be divisible by split_k");
  TORCH_CHECK(tokens_per_cta == 32 || tokens_per_cta == 64 ||
                  tokens_per_cta == 128,
              "tokens_per_cta must be 32, 64, or 128");

  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(residual_in.get_device());
  void *res_p = const_cast<void *>(residual_in.data_ptr());
  void *x_p = const_cast<void *>(x_in.data_ptr());
  void *comb_p = const_cast<float *>(comb_in.data_ptr<float>());
  void *post_p = const_cast<float *>(post_in.data_ptr<float>());
  void *fn_p = const_cast<void *>(fn.data_ptr());
  void *res_next_p = const_cast<void *>(residual_next.data_ptr());
  float *outp_p = out_partial.data_ptr<float>();
  float *sqrp_p = sqr_partial.data_ptr<float>();
  void *mixes_p = const_cast<void *>(mixes_pad.data_ptr());
  float *sqrsum_p = sqrsum.data_ptr<float>();

  int tpb;
  if (split_k == 1 && num_tokens >= 512) tpb = (c > 4096) ? 4 : 2;
  else tpb = 1;

  TORCH_CHECK(mix_hc % tile_n == 0, "tile_n must divide mix_hc");

#define LAUNCH_FUSED(C_, SK_, TPB_, BT_, TN_)                                  \
  do {                                                                         \
    dim3 _fg((num_tokens + (TPB_)-1) / (TPB_), (SK_), (24 / (TN_)));           \
    dim3 _fb((BT_), 1, 1);                                                     \
    mHC_post_pre_k1_kernel<4, C_, 24, BT_, SK_, TPB_, TN_>                      \
        <<<_fg, _fb, 0, stream>>>(res_p,                                       \
                                  x_p,                                         \
                                  comb_p,                                      \
                                  post_p,                                      \
                                  fn_p,                                        \
                                  res_next_p,                                  \
                                  outp_p,                                      \
                                  sqrp_p,                                      \
                                  mixes_p,                                     \
                                  sqrsum_p,                                    \
                                  num_tokens);                                 \
    if ((SK_) > 1) {                                                           \
      mHC_post_pre_k1_reduce_kernel<4, 24, MIX_PAD, SK_>                       \
          <<<dim3(num_tokens, 1, 1), dim3(32, 1, 1), 0, stream>>>(             \
              outp_p, sqrp_p, mixes_p, sqrsum_p, num_tokens);                  \
    }                                                                          \
  } while (0)

#define LAUNCH_FUSED_TN(C_, SK_, TPB_, BT_)                                    \
  switch (tile_n) {                                                            \
    case 24: LAUNCH_FUSED(C_, SK_, TPB_, BT_, 24); break;                      \
    case 6:  LAUNCH_FUSED(C_, SK_, TPB_, BT_, 6);  break;                      \
    case 1:  LAUNCH_FUSED(C_, SK_, TPB_, BT_, 1);  break;                      \
    default: TORCH_CHECK(false, "Unsupported tile_n=", tile_n,                 \
                         " (instantiated: 1, 6, 24)");                         \
  }

#define LAUNCH_FUSED_TPB(C_, SK_)                                              \
  switch (tpb) {                                                               \
    case 1:                                                                    \
      LAUNCH_FUSED_TN(C_, SK_, 1, 256);                                        \
      break;                                                                   \
    case 2:                                                                    \
      LAUNCH_FUSED_TN(C_, SK_, 2, 256);                                        \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_FUSED_TN(C_, SK_, 4, 256);                                        \
      break;                                                                   \
    case 8:                                                                    \
      LAUNCH_FUSED_TN(C_, SK_, 8, 256);                                        \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "bad tpb");                                           \
  }

#define DISPATCH_FUSED_SK(C_)                                                  \
  switch (split_k) {                                                           \
    case 1:                                                                    \
      LAUNCH_FUSED_TPB(C_, 1);                                                 \
      break;                                                                   \
    case 2:                                                                    \
      LAUNCH_FUSED_TN(C_, 2, 1, 256);                                          \
      break;                                                                   \
    case 4:                                                                    \
      LAUNCH_FUSED_TN(C_, 4, 1, 256);                                          \
      break;                                                                   \
    case 8:                                                                    \
      LAUNCH_FUSED_TN(C_, 8, 1, 256);                                          \
      break;                                                                   \
    case 16:                                                                   \
      LAUNCH_FUSED_TN(C_, 16, 1, 128);                                         \
      break;                                                                   \
    case 32:                                                                   \
      LAUNCH_FUSED_TN(C_, 32, 1, 64);                                          \
      break;                                                                   \
    case 64:                                                                   \
      LAUNCH_FUSED_TN(C_, 64, 1, 64);                                          \
      break;                                                                   \
    case 128:                                                                  \
      LAUNCH_FUSED_TN(C_, 128, 1, 64);                                         \
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
#undef LAUNCH_FUSED_TPB
#undef LAUNCH_FUSED_TN
#undef LAUNCH_FUSED

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_post_pre_v2 fused launch error: ",
              cudaGetErrorString(err));

  int num_ctas = 0;
  cudaDeviceGetAttribute(&num_ctas, cudaDevAttrMultiProcessorCount, residual_in.get_device());
  int const k2_lowt_thresh = 32 * num_ctas;
  bool const use_lowt_k2 = (num_tokens < k2_lowt_thresh);

  size_t const k2_smem =
      tokens_per_cta * (4 * n + n * n + n * n + n) * sizeof(float) + 1024;
  int const tiles = ceil_div(num_tokens, tokens_per_cta);
  int const grid = num_ctas < tiles ? num_ctas : tiles;
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

#define LAUNCH_V2_K2_LOWT(C_, RH_, SK_)                                        \
  do {                                                                         \
    dim3 _lg(num_tokens, 1, 1);                                                \
    if ((SK_) <= 1) {                                                          \
      mHC_pre_k2_lowt_kernel<4, C_, RH_>                                       \
          <<<_lg, k2_block, 0, stream>>>(mixes_p,                              \
                                         sqrsum_p,                             \
                                         scale_p,                              \
                                         base_p,                               \
                                         x_orig_p,                             \
                                         f_pre_p,                              \
                                         h_post_p,                             \
                                         comb_p2,                              \
                                         num_tokens,                           \
                                         sinkhorn_repeat,                      \
                                         sk_eps_f,                             \
                                         rms_eps_f);                           \
    } else if ((SK_) <= 8) {                                                   \
      mHC_pre_k2_lowt_fused_kernel<4, C_, RH_, SK_>                            \
          <<<_lg, k2_block, 0, stream>>>(outp_p,                               \
                                         sqrp_p,                               \
                                         scale_p,                              \
                                         base_p,                               \
                                         x_orig_p,                             \
                                         f_pre_p,                              \
                                         h_post_p,                             \
                                         comb_p2,                              \
                                         num_tokens,                           \
                                         sinkhorn_repeat,                      \
                                         sk_eps_f,                             \
                                         rms_eps_f);                           \
    } else {                                                                   \
      mHC_pre_k2_lowt_kernel<4, C_, RH_>                                       \
          <<<_lg, k2_block, 0, stream>>>(mixes_p,                              \
                                         sqrsum_p,                             \
                                         scale_p,                              \
                                         base_p,                               \
                                         x_orig_p,                             \
                                         f_pre_p,                              \
                                         h_post_p,                             \
                                         comb_p2,                              \
                                         num_tokens,                           \
                                         sinkhorn_repeat,                      \
                                         sk_eps_f,                             \
                                         rms_eps_f);                           \
    }                                                                          \
  } while (0)

#define LAUNCH_V2_K2(C_, RH_, TPC_)                                            \
  do {                                                                         \
    auto *kp = &mHC_pre_k2_kernel<4, C_, RH_, TPC_>;                           \
    MHC_CUDA_CHECK(cudaFuncSetAttribute(                                     \
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

#define DISPATCH_V2_K2(C_, RH_)                                                \
  do {                                                                         \
    if (use_lowt_k2) {                                                         \
      switch (split_k) {                                                       \
        case 1:  LAUNCH_V2_K2_LOWT(C_, RH_, 1);  break;                       \
        case 2:  LAUNCH_V2_K2_LOWT(C_, RH_, 2);  break;                       \
        case 4:  LAUNCH_V2_K2_LOWT(C_, RH_, 4);  break;                       \
        case 8:  LAUNCH_V2_K2_LOWT(C_, RH_, 8);  break;                       \
        case 16: LAUNCH_V2_K2_LOWT(C_, RH_, 16); break;                       \
        case 32: LAUNCH_V2_K2_LOWT(C_, RH_, 32); break;                       \
        case 64: LAUNCH_V2_K2_LOWT(C_, RH_, 64); break;                       \
        case 128: LAUNCH_V2_K2_LOWT(C_, RH_, 128); break;                     \
        default: TORCH_CHECK(false, "Unsupported split_k=", split_k);         \
      }                                                                        \
    } else {                                                                   \
      DISPATCH_V2_K2_TPC(C_, RH_);                                             \
    }                                                                          \
  } while (0)

  switch (c) {
    case 128:
      DISPATCH_V2_K2(128, 4 * 128);
      break;
    case 1024:
      DISPATCH_V2_K2(1024, 4 * 1024);
      break;
    case 4096:
      DISPATCH_V2_K2(4096, 4 * 4096);
      break;
    case 7168:
      DISPATCH_V2_K2(7168, 4 * 7168);
      break;
    default:
      TORCH_CHECK(false, "Unsupported c=", c);
  }
#undef DISPATCH_V2_K2
#undef DISPATCH_V2_K2_TPC
#undef LAUNCH_V2_K2
#undef LAUNCH_V2_K2_LOWT

  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_post_pre_v2 k2 launch error: ",
              cudaGetErrorString(err));
}

template <int MIX_HC,
          int K,
          int BLOCK_THREADS,
          int SPLIT_K,
          int MIX_PAD = 128,
          int TPB = 1>
__global__ __launch_bounds__(BLOCK_THREADS) void mHC_pre_k1_cuda_core_kernel(
    void const *__restrict__ residual,
    __nv_bfloat16 const *__restrict__ fn,
    float *__restrict__ out_partial,
    float *__restrict__ sqr_partial,
    void *__restrict__ mixes_pad,
    float *__restrict__ sqrsum,
    int num_tokens) {
  int const token0 = blockIdx.x * TPB;
  int const i_ks = blockIdx.y;
  if (token0 >= num_tokens) {
    return;
  }
  kernel::mHC_pre_k1_cuda_core_task_impl<bf16_t,
                                         MIX_HC,
                                         K,
                                         BLOCK_THREADS,
                                         SPLIT_K,
                                         MIX_PAD,
                                         TPB>(
      static_cast<bf16_t const *>(residual),
      fn,
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

void mHC_pre_k1_decode(torch::Tensor residual,
                       torch::Tensor fn,
                       torch::Tensor out_partial,
                       torch::Tensor sqr_partial,
                       torch::Tensor mixes_pad,
                       torch::Tensor sqrsum,
                       int n,
                       int split_k) {
  TORCH_CHECK(n == 4, "pre_k1_cuda_core hardcoded to n=4");
  constexpr int MIX_PAD = 128;
  int const mix_hc = n * n + 2 * n;
  int const num_tokens = static_cast<int>(residual.size(0));
  int const K = static_cast<int>(residual.size(1));
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.dim() == 2,
              "residual must be bf16 [tokens, K]");
  check_tensor(fn, at::kBFloat16, {mix_hc, K}, "fn");
  check_tensor(out_partial, at::kFloat, {split_k, num_tokens, mix_hc}, "out_partial");
  check_tensor(sqr_partial, at::kFloat, {split_k, num_tokens}, "sqr_partial");
  check_tensor(mixes_pad, at::kBFloat16, {num_tokens, MIX_PAD}, "mixes_pad");
  check_tensor(sqrsum, at::kFloat, {num_tokens}, "sqrsum");
  TORCH_CHECK(K % split_k == 0, "K must be divisible by split_k");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  constexpr int BT = 128;

  void *res_p = const_cast<void *>(residual.data_ptr());
  __nv_bfloat16 *fn_p = reinterpret_cast<__nv_bfloat16 *>(fn.data_ptr());
  float *outp_p = out_partial.data_ptr<float>();
  float *sqrp_p = sqr_partial.data_ptr<float>();
  void *mixes_p = const_cast<void *>(mixes_pad.data_ptr());
  float *sqrsum_p = sqrsum.data_ptr<float>();

  dim3 block(BT, 1, 1);
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
    case 512:
      DISPATCH_PRE_K1_CUDA_SK(512);
      break;
    case 4096:
      DISPATCH_PRE_K1_CUDA_SK(4096);
      break;
    case 16384:
      DISPATCH_PRE_K1_CUDA_SK(16384);
      break;
    case 28672:
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
              "mHC_pre_k1_decode launch error: ",
              cudaGetErrorString(err));
}

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
  if (smem_bytes >= 48 * 1024) {
    MHC_CUDA_CHECK(cudaFuncSetAttribute(
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

void mHC_pre_k1_prefill(torch::Tensor residual,
                        torch::Tensor weight_padded,
                        torch::Tensor mixes_pad,
                        torch::Tensor sqrsum,
                        int n) {
  TORCH_CHECK(n == 4, "pre_k1_prefill hardcoded to n=4");
  constexpr int OUT_PAD = 128;
  int const batch = static_cast<int>(residual.size(0));
  int const K = static_cast<int>(residual.size(1));
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                  residual.scalar_type() == at::kBFloat16 &&
                  residual.dim() == 2,
              "residual must be bf16 [batch, K]");
  check_tensor(weight_padded, at::kBFloat16, {OUT_PAD, K}, "weight_padded");
  check_tensor(mixes_pad, at::kBFloat16, {batch, OUT_PAD}, "mixes_pad");
  check_tensor(sqrsum, at::kFloat, {batch}, "sqrsum");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  void *res_p = const_cast<void *>(residual.data_ptr());
  void *w_p = const_cast<void *>(weight_padded.data_ptr());
  void *mx_p = const_cast<void *>(mixes_pad.data_ptr());
  void *sq_p = const_cast<float *>(sqrsum.data_ptr<float>());

  constexpr int BN = 16, NS = 2;
  int const BK = (batch >= 4096) ? 128 : 256;
#define LAUNCH_PREFILL(K_)                                                     \
  do {                                                                         \
    if (BK == 128)                                                             \
      launch_pre_k1_tensor_core<K_, OUT_PAD, BN, 128, 24, NS>(                 \
          res_p, w_p, mx_p, sq_p, batch, stream);                             \
    else                                                                       \
      launch_pre_k1_tensor_core<K_, OUT_PAD, BN, 256, 24, NS>(                 \
          res_p, w_p, mx_p, sq_p, batch, stream);                             \
  } while (0)
  switch (K) {
    case 512:
      LAUNCH_PREFILL(512);
      break;
    case 4096:
      LAUNCH_PREFILL(4096);
      break;
    case 16384:
      LAUNCH_PREFILL(16384);
      break;
    case 28672:
      LAUNCH_PREFILL(28672);
      break;
    default:
      TORCH_CHECK(false, "Unsupported K=", K);
  }
#undef LAUNCH_PREFILL
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "mHC_pre_k1_prefill launch error: ",
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
  kernel::mHC_pre_k2_task_impl<bf16_t,
                               N,
                               C,
                               RMS_HIDDEN,
                               TOKENS_PER_CTA,
                               256,
                               128>(mixes_pad,
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
                int tokens_per_cta,
                int force_path) {
  TORCH_CHECK(n == 4, "pre K2 hardcoded to n=4");
  TORCH_CHECK(mixes_pad.is_cuda() &&
              mixes_pad.is_contiguous() &&
              mixes_pad.dim() == 2 &&
              mixes_pad.scalar_type() == at::kBFloat16 &&
              mixes_pad.size(1) == 128,
              "mixes_pad must be bf16 [bs, 128] CUDA contiguous");
  TORCH_CHECK(tokens_per_cta == 32 ||
              tokens_per_cta == 64 ||
              tokens_per_cta == 128,
              "tokens_per_cta must be 32, 64, or 128");

  int const num_tokens = static_cast<int>(mixes_pad.size(0));
  check_tensor(sqrsum, at::kFloat, {num_tokens}, "sqrsum");

  int num_ctas = 0;
  cudaDeviceGetAttribute(&num_ctas, cudaDevAttrMultiProcessorCount,
                         mixes_pad.get_device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(mixes_pad.get_device());

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
    MHC_CUDA_CHECK(cudaFuncSetAttribute(                                     \
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

  int const k2_lowt_thresh = 32 * num_ctas;
  // force_path: 0 = auto (token-count heuristic), 1 = lowt (1 token/CTA),
  // 2 = batched (tokens_per_cta tokens/CTA). For benchmarking the crossover.
  bool const use_lowt = (force_path == 1) ? true
                        : (force_path == 2) ? false
                                            : (num_tokens < k2_lowt_thresh);

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

static int pick_cuda_split_k(int K, int num_tokens) {
  int grid_target =
      (num_tokens >= 1024) ? 1 : (1024 + num_tokens - 1) / num_tokens;
  int const work_cap = (K >= 1024) ? (K / 1024) : 1;
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

  constexpr int CUDA_T_THRESH = 256;
  if (num_tokens < CUDA_T_THRESH) {
    auto fn =
        weight_padded.slice(0, 0, mix_hc).to(torch::kBFloat16).contiguous();
    int const split_k = pick_cuda_split_k(K, num_tokens);
    auto opts_f =
        torch::TensorOptions().dtype(torch::kFloat32).device(residual.device());
    auto out_partial = torch::empty({split_k, num_tokens, mix_hc}, opts_f);
    auto sqr_partial = torch::empty({split_k, num_tokens}, opts_f);
    mHC_pre_k1_decode(
        residual, fn, out_partial, sqr_partial, mixes_pad, sqrsum, n, split_k);
  } else {
    mHC_pre_k1_prefill(residual, weight_padded, mixes_pad, sqrsum, n);
  }
}

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
    auto fn =
        weight_padded.slice(0, 0, mix_hc).to(torch::kBFloat16).contiguous();
    int const split_k = pick_cuda_split_k(K, num_tokens);
    auto opts_f =
        torch::TensorOptions().dtype(torch::kFloat32).device(residual.device());
    auto out_partial = torch::empty({split_k, num_tokens, mix_hc}, opts_f);
    auto sqr_partial = torch::empty({split_k, num_tokens}, opts_f);
    void *res_p = const_cast<void *>(residual.data_ptr());
    __nv_bfloat16 *fn_p = reinterpret_cast<__nv_bfloat16 *>(fn.data_ptr());
    float *outp_p = out_partial.data_ptr<float>();
    float *sqrp_p = sqr_partial.data_ptr<float>();
    void *mx_p = const_cast<void *>(mixes_pad.data_ptr());
    float *ss_p = sqrsum.data_ptr<float>();
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
    mHC_pre_k1_prefill(residual, weight_padded, mixes_pad, sqrsum, n);
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
               tokens_per_cta,
               /*force_path=*/0);
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess, "mHC_pre launch error: ", cudaGetErrorString(err));
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mHC_post",
        &mHC_post,
        py::arg("residual"),
        py::arg("x"),
        py::arg("comb"),
        py::arg("post"),
        py::arg("output"),
        py::arg("n"),
        "mHC post: y[k,c] = post[k]*x[c] + sum_i comb[i,k]*residual[i,c] "
        "(comb NOT transposed; matches torch hc_post)");

  m.def(
      "mHC_pre_k1_prefill",
      &mHC_pre_k1_prefill,
      py::arg("residual"),
      py::arg("weight_padded"),
      py::arg("mixes_pad"),
      py::arg("sqrsum"),
      py::arg("n"),
      "Prefill pre_k1 (raw-PTX tcgen05, no CUTLASS/CuTe): mixes = residual @ "
      "fn.T + sqrsum, bf16 kind::f16 MMA. Same outputs as mHC_pre_k1.");

  m.def("mHC_pre_k1_decode",
        &mHC_pre_k1_decode,
        py::arg("residual"),
        py::arg("fn"),
        py::arg("out_partial"),
        py::arg("sqr_partial"),
        py::arg("mixes_pad"),
        py::arg("sqrsum"),
        py::arg("n"),
        py::arg("split_k") = 1,
        "Decode pre_k1 (CUDA-core): mixes = residual @ fn.T + per-token sqrsum "
        "(no tensor cores, split-k). Same outputs as the tcgen05 mHC_pre_k1; "
        "wins at low token count.");

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
        py::arg("tile_n") = 24,
        "CUDA-core fused post + prenorm-GEMM (vLLM mhc_fused style) + split-k "
        "reduce + k2 tail. Outputs next layer f_pre / h_post / comb. "
        "tile_n: outputs computed per CTA, must be 1, 6, or 24 (see tile_n_for).");

  m.def("sinkhorn_sm100",
        &sinkhorn_sm100,
        py::arg("comb_res_mix"),
        py::arg("comb_res_mix_out"),
        py::arg("repeat") = 20,
        py::arg("eps") = 1e-9,
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
        py::arg("tokens_per_cta") = 32,
        py::arg("force_path") = 0,
        "mHC pre K2 (vLLM split): rms-scale gemm output (via sqrsum) + "
        "pre/post/comb mix (sinkhorn) + pre-weighted residual sum. "
        "force_path: 0=auto, 1=lowt (1 tok/CTA), 2=batched (tokens_per_cta/CTA).");
}
