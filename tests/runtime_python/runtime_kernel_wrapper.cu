#include "argmax.cuh"
#include "bfloat16.h"
#include "linear.cuh"
#include "norm.cuh"
#include "norm_linear.cuh"
#include "paged_attention.cuh"
#include "silu_mul_linear.cuh"
#include "single_batch_decoding.cuh"
#include "single_batch_gqa.cuh"
#include "embedding.cuh"
#include "prompt_lookup.cuh"
#include "target_verify.cuh"
#include "bfloat16.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

// using kernel::argmax_kernel;
using kernel::linear_kernel;
using kernel::norm_linear_task_impl;
using kernel::paged_attention_task_impl;
using kernel::silu_mul_linear_task_impl;
using kernel::single_batch_decoding_kernel;
using kernel::single_batch_gqa_kernel;
using kernel::embedding_kernel;
using kernel::find_ngram_partial_kernel;
using kernel::find_ngram_global_kernel;
using kernel::argmax_partial_kernel;
using kernel::argmax_reduce_kernel;
using kernel::target_verify_greedy_kernel;
using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void single_batch_gqa_kernel_wrapper(void const *qkv_ptr,
                                                void *k_cache_ptr,
                                                void *v_cache_ptr,
                                                void *output_ptr,
                                                size_t seq_len,
                                                bool qk_norm,
                                                bool rotary_embed,
                                                void const *qnorm_weight_ptr,
                                                void const *knorm_weight_ptr,
                                                void const *cos_ptr,
                                                void const *sin_ptr,
                                                float q_eps,
                                                float k_eps) {
  single_batch_gqa_kernel<T, 4>(qkv_ptr,
                                k_cache_ptr,
                                v_cache_ptr,
                                output_ptr,
                                seq_len,
                                qk_norm,
                                rotary_embed,
                                qnorm_weight_ptr,
                                knorm_weight_ptr,
                                cos_ptr,
                                sin_ptr,
                                q_eps,
                                k_eps);
}

void single_batch_gqa(
    torch::Tensor qkv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor output,
    size_t seq_len,
    bool qk_norm,
    bool rotary_embed,
    torch::optional<torch::Tensor> qnorm_weight = torch::nullopt,
    torch::optional<torch::Tensor> knorm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  void const *qkv_ptr = qkv.data_ptr();
  void *k_cache_ptr = k_cache.data_ptr();
  void *v_cache_ptr = v_cache.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 88888;

  void const *qnorm_weight_ptr = qk_norm ? qnorm_weight->data_ptr() : nullptr;
  void const *knorm_weight_ptr = qk_norm ? knorm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rotary_embed ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rotary_embed ? sin->data_ptr() : nullptr;

  cudaFuncSetAttribute(single_batch_gqa_kernel_wrapper<bfloat16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  single_batch_gqa_kernel_wrapper<bfloat16>
      <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
                                           k_cache_ptr,
                                           v_cache_ptr,
                                           output_ptr,
                                           seq_len,
                                           qk_norm,
                                           rotary_embed,
                                           qnorm_weight_ptr,
                                           knorm_weight_ptr,
                                           cos_ptr,
                                           sin_ptr,
                                           q_eps,
                                           k_eps);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Single Batch Decoding

template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int WEIGHT_STRIDE>
__global__ void single_batch_decoding_wrapper(void const *qkv_ptr,
                                              void *k_cache_ptr,
                                              void *v_cache_ptr,
                                              void *output_ptr,
                                              size_t seq_len,
                                              bool qk_norm,
                                              bool rotary_emd,
                                              void const *qnorm_weight_ptr,
                                              void const *knorm_weight_ptr,
                                              void const *cos_ptr,
                                              void const *sin_ptr,
                                              float q_eps,
                                              float k_eps) {
  single_batch_decoding_kernel<T,
                               NUM_Q_HEADS,
                               NUM_KV_HEADS,
                               HEAD_DIM,
                               WEIGHT_STRIDE>(qkv_ptr,
                                              k_cache_ptr,
                                              v_cache_ptr,
                                              output_ptr,
                                              seq_len,
                                              qk_norm,
                                              rotary_emd,
                                              qnorm_weight_ptr,
                                              knorm_weight_ptr,
                                              cos_ptr,
                                              sin_ptr,
                                              q_eps,
                                              k_eps);
}

void single_batch_decoding(
    torch::Tensor qkv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor output,
    size_t seq_len,
    bool qk_norm,
    bool rotary_emd,
    torch::optional<torch::Tensor> qnorm_weight = torch::nullopt,
    torch::optional<torch::Tensor> knorm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  void const *qkv_ptr = qkv.data_ptr();
  void *k_cache_ptr = k_cache.data_ptr();
  void *v_cache_ptr = v_cache.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 88888;

  void const *qnorm_weight_ptr = qk_norm ? qnorm_weight->data_ptr() : nullptr;
  void const *knorm_weight_ptr = qk_norm ? knorm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rotary_emd ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rotary_emd ? sin->data_ptr() : nullptr;

  cudaFuncSetAttribute(single_batch_decoding_wrapper<bfloat16, 4, 1, 128, 128>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  single_batch_decoding_wrapper<bfloat16, 4, 1, 128, 128>
      <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
                                           k_cache_ptr,
                                           v_cache_ptr,
                                           output_ptr,
                                           seq_len,
                                           qk_norm,
                                           rotary_emd,
                                           qnorm_weight_ptr,
                                           knorm_weight_ptr,
                                           cos_ptr,
                                           sin_ptr,
                                           q_eps,
                                           k_eps);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Paged Attention

template <typename T,
          int NUM_Q_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int KV_STRIDE>
__global__ void paged_attention_wrapper(void const *qkv_ptr,
                                        void *paged_k_cache_ptr,
                                        void *paged_v_cache_ptr,
                                        void *output_ptr,
                                        void const *paged_kv_indices_buffer_ptr,
                                        size_t seq_len,
                                        bool qk_norm,
                                        bool rope,
                                        void const *q_norm_weight_ptr,
                                        void const *k_norm_weight_ptr,
                                        void const *cos_ptr,
                                        void const *sin_ptr,
                                        float q_eps,
                                        float k_eps) {
  paged_attention_task_impl<T,
                            NUM_Q_PER_KV,
                            HEAD_DIM,
                            PAGE_SIZE,
                            MAX_SEQ_LEN,
                            KV_STRIDE>(qkv_ptr,
                                       paged_k_cache_ptr,
                                       paged_v_cache_ptr,
                                       output_ptr,
                                       paged_kv_indices_buffer_ptr,
                                       seq_len,
                                       qk_norm,
                                       rope,
                                       q_norm_weight_ptr,
                                       k_norm_weight_ptr,
                                       cos_ptr,
                                       sin_ptr,
                                       q_eps,
                                       k_eps);
}

template <typename T,
          int NUM_Q_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int KV_STRIDE>
void launch_paged_attention(void const *qkv_ptr,
                            void *paged_k_cache_ptr,
                            void *paged_v_cache_ptr,
                            void *output_ptr,
                            void const *paged_kv_indices_buffer_ptr,
                            size_t seq_len,
                            bool qk_norm,
                            bool rope,
                            void const *q_norm_weight_ptr,
                            void const *k_norm_weight_ptr,
                            void const *cos_ptr,
                            void const *sin_ptr,
                            float q_eps,
                            float k_eps) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 112640;

  cudaFuncSetAttribute(paged_attention_wrapper<T,
                                               NUM_Q_PER_KV,
                                               HEAD_DIM,
                                               PAGE_SIZE,
                                               MAX_SEQ_LEN,
                                               KV_STRIDE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  paged_attention_wrapper<T,
                          NUM_Q_PER_KV,
                          HEAD_DIM,
                          PAGE_SIZE,
                          MAX_SEQ_LEN,
                          KV_STRIDE>
      <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
                                           paged_k_cache_ptr,
                                           paged_v_cache_ptr,
                                           output_ptr,
                                           paged_kv_indices_buffer_ptr,
                                           seq_len,
                                           qk_norm,
                                           rope,
                                           q_norm_weight_ptr,
                                           k_norm_weight_ptr,
                                           cos_ptr,
                                           sin_ptr,
                                           q_eps,
                                           k_eps);
}

void paged_attention(
    torch::Tensor qkv,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor output,
    torch::Tensor paged_kv_indices_buffer,
    size_t seq_len,
    bool qk_norm,
    bool rope,
    torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  void const *qkv_ptr = qkv.data_ptr();
  void *paged_k_cache_ptr = paged_k_cache.data_ptr();
  void *paged_v_cache_ptr = paged_v_cache.data_ptr();
  void *output_ptr = output.data_ptr();
  void const *paged_kv_indices_buffer_ptr = paged_kv_indices_buffer.data_ptr();

  void const *q_norm_weight_ptr = qk_norm ? q_norm_weight->data_ptr() : nullptr;
  void const *k_norm_weight_ptr = qk_norm ? k_norm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rope ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rope ? sin->data_ptr() : nullptr;

  launch_paged_attention<bfloat16, 4, 128, 64, 512, 128>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      paged_kv_indices_buffer_ptr,
      seq_len,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// RMSNorm Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void norm_linear_kernel_wrapper(void const *input_ptr,
                                           void const *norm_weight_ptr,
                                           void const *weight_ptr,
                                           float eps,
                                           void *output_ptr) {
  norm_linear_task_impl<T,
                        BATCH_SIZE,
                        OUTPUT_SIZE,
                        REDUCTION_SIZE,
                        OUTPUT_SIZE>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_norm_linear(void const *input_ptr,
                        void const *norm_weight_ptr,
                        void const *weight_ptr,
                        float eps,
                        void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 1024 * 99;

  cudaFuncSetAttribute(
      norm_linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  norm_linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

void norm_linear(torch::Tensor input,
                 torch::Tensor norm_weight,
                 torch::Tensor weight,
                 float eps,
                 torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *norm_weight_ptr = norm_weight.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (output.size(1)) {
    case 16:
      launch_norm_linear<bfloat16, 1, 16, 4096>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
      break;
    case 32:
      launch_norm_linear<bfloat16, 1, 32, 4096>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
      break;
    case 64:
      launch_norm_linear<bfloat16, 1, 64, 4096>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
      break;
    case 256:
      launch_norm_linear<bfloat16, 1, 256, 4096>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
      break;
    case 1600:
      launch_norm_linear<bfloat16, 1, 1600, 4096>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
      break;
    default:
      printf("Unsupported output size in test: %zu\n", output.size(1));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Window RMSNorm Linear

template <typename T, int BATCH_SIZE, int WINDOW_SIZE, int HEAD_DIM>
__global__ void window_rms_norm_kernel_wrapper(void const *input_ptr,
                                               void const *weight_ptr,
                                               float eps,
                                               void *output_ptr,
                                               bool rotary_emd = false,
                                               void const *cos_ptr = nullptr,
                                               void const *sin_ptr = nullptr) {
  constexpr size_t q_num = BATCH_SIZE * WINDOW_SIZE;
  using Smem = kernel::smem_row<T, 3, 3, 3, q_num, 128, 128>;

  extern __shared__ char smem[];
  T *smem_ptr = reinterpret_cast<T *>(smem);
  Smem input_smem(smem_ptr);

  // TODO(Wenqin): q_num * HEAD_DIM is a conservative number.
  float *reduce_smem = reinterpret_cast<float *>(smem) + q_num * HEAD_DIM;

  T const *d_input = static_cast<T const *>(input_ptr);
  T *d_output = const_cast<T *>(static_cast<T const *>(output_ptr));

  kernel::dmem_row_const<T, q_num, 128, 128> input_dmem(d_input);
  kernel::dmem_row<T, q_num, 128, 128> output_dmem(d_output);

  for (int i = threadIdx.x; i < q_num * (HEAD_DIM / 8); i += NUM_THREADS) {
    int row = i / 16;
    int col = (i % 16) * 8;
    kernel::load_smem(input_smem(row, col), input_dmem(row, col));
  }
  kernel::cp_async_fence();
  kernel::cp_async_wait<0>();

  // some thread didn't issue any cp.async ldgsts inst, so we should ask them
  // to wait.
  __syncthreads();

  T const *norm_weight_ptr = static_cast<T const *>(weight_ptr);
  T const *rope_cos_ptr = static_cast<T const *>(cos_ptr);
  T const *rope_sin_ptr = static_cast<T const *>(sin_ptr);

  kernel::window_rms_norm<T, Smem, 1, WINDOW_SIZE, HEAD_DIM>(input_smem,
                                                             norm_weight_ptr,
                                                             reduce_smem,
                                                             eps,
                                                             rotary_emd,
                                                             rope_cos_ptr,
                                                             rope_sin_ptr);

  __syncthreads();

  for (int i = threadIdx.x; i < q_num * (HEAD_DIM / 8); i += NUM_THREADS) {
    // write back
    int row = i / 16;
    int col = (i % 16) * 8;
    for (int j = 0; j < 8; j++) {
      int col_j = col + j;
      *output_dmem(row, col_j) = *input_smem(row, col_j);
    }
  }
}

#define WINDOW_RMSNORM_LINEAR_LAUNCHER(HEAD_DIM, WINDOW_SIZE)                  \
  launch_window_rms_norm<bfloat16, 1, WINDOW_SIZE, HEAD_DIM>(                  \
      input_ptr, weight_ptr, eps, output_ptr, rotary_emd, cos_ptr, sin_ptr);

#define DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(HEAD_DIM)                   \
  switch (window_size) {                                                       \
    case 1:                                                                    \
      WINDOW_RMSNORM_LINEAR_LAUNCHER(HEAD_DIM, 1);                             \
      break;                                                                   \
    case 2:                                                                    \
      WINDOW_RMSNORM_LINEAR_LAUNCHER(HEAD_DIM, 2);                             \
      break;                                                                   \
    case 3:                                                                    \
      WINDOW_RMSNORM_LINEAR_LAUNCHER(HEAD_DIM, 3);                             \
      break;                                                                   \
    case 4:                                                                    \
      WINDOW_RMSNORM_LINEAR_LAUNCHER(HEAD_DIM, 4);                             \
      break;                                                                   \
    default:                                                                   \
      printf("Unsupported window size in test: %zu\n", window_size);           \
      break;                                                                   \
  }

#define DISPATCH_WINDOW_RMSNORM_LINEAR_HEAD_DIM()                              \
  switch (head_dim) {                                                          \
    case 16:                                                                   \
      DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(16);                          \
      break;                                                                   \
    case 32:                                                                   \
      DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(32);                          \
      break;                                                                   \
    case 64:                                                                   \
      DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(64);                          \
      break;                                                                   \
    case 128:                                                                  \
      DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(128);                         \
      break;                                                                   \
    case 256:                                                                  \
      DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(256);                         \
      break;                                                                   \
    case 1600:                                                                 \
      DISPATCH_WINDOW_RMSNORM_LINEAR_WINDOW_SIZE(1600);                        \
      break;                                                                   \
    default:                                                                   \
      printf("Unsupported head dim in test: %zu\n", head_dim);                 \
      break;                                                                   \
  }

#define WINDOW_RMSNORM_LINEAR() DISPATCH_WINDOW_RMSNORM_LINEAR_HEAD_DIM()

template <typename T, int BATCH_SIZE, int WINDOW_SIZE, int HEAD_DIM>
void launch_window_rms_norm(void const *input_ptr,
                            void const *weight_ptr,
                            float eps,
                            void *output_ptr,
                            bool rotary_emd = false,
                            void const *cos_ptr = nullptr,
                            void const *sin_ptr = nullptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 1024 * 99;

  cudaFuncSetAttribute(
      window_rms_norm_kernel_wrapper<T, BATCH_SIZE, WINDOW_SIZE, HEAD_DIM>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  window_rms_norm_kernel_wrapper<T, BATCH_SIZE, WINDOW_SIZE, HEAD_DIM>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, eps, output_ptr, rotary_emd, cos_ptr, sin_ptr);
}

void window_rms_norm(
    torch::Tensor input, // shape [batch, window_size, head_dim]
    torch::Tensor weight,
    float eps,
    torch::Tensor output,
    bool rotary_emd = false,
    torch::optional<torch::Tensor> cos = c10::nullopt,
    torch::optional<torch::Tensor> sin = c10::nullopt) {
  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *cos_ptr = cos ? cos->data_ptr() : nullptr;
  void const *sin_ptr = sin ? sin->data_ptr() : nullptr;
  void *output_ptr = output.data_ptr();
  size_t head_dim = output.size(2);
  size_t window_size = output.size(1);
  auto dtype = input.dtype().toScalarType();

  DISPATCH_WINDOW_RMSNORM_LINEAR_HEAD_DIM();

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA window_norm_linear kernel launch error: %s\n",
           cudaGetErrorString(err));
  }
}

// SiLU MUL Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void silu_mul_linear_kernel_wrapper(void const *input_ptr,
                                               void const *weight_ptr,
                                               void const *bias_ptr,
                                               void *output_ptr) {
  silu_mul_linear_task_impl<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, weight_ptr, bias_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_silu_mul_linear(void const *input_ptr,
                            void const *weight_ptr,
                            void const *bias_ptr,
                            void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 112640;

  cudaFuncSetAttribute(silu_mul_linear_kernel_wrapper<T,
                                                      BATCH_SIZE,
                                                      OUTPUT_SIZE,
                                                      REDUCTION_SIZE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  silu_mul_linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, bias_ptr, output_ptr);
}

void silu_mul_linear(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *bias_ptr = bias.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (output.size(1)) {
    case 16:
      launch_silu_mul_linear<bfloat16, 1, 16, 12288>(
          input_ptr, weight_ptr, bias_ptr, output_ptr);
      break;
    case 32:
      launch_silu_mul_linear<bfloat16, 1, 32, 12288>(
          input_ptr, weight_ptr, bias_ptr, output_ptr);
      break;
    case 64:
      launch_silu_mul_linear<bfloat16, 1, 64, 12288>(
          input_ptr, weight_ptr, bias_ptr, output_ptr);
      break;
    default:
      printf("Unsupported output size in test: %zu\n", output.size(1));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear(void const *input_ptr,
                   void const *weight_ptr,
                   void const *residual_ptr,
                   void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 112640;

  cudaFuncSetAttribute(
      linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

void linear(torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor residual,
            torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (output.size(1)) {
    case 16:
      launch_linear<bfloat16, 1, 16, 4096>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
      break;
    case 32:
      launch_linear<bfloat16, 1, 32, 4096>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
      break;
    case 64:
      launch_linear<bfloat16, 1, 64, 4096>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
      break;
    default:
      printf("Unsupported output size in test: %zu\n", output.size(1));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Embedding Kernel
template <typename T, int CHUNK_SIZE, int OUTPUT_DIM_SIZE>
__global__ void embedding_kernel_wrapper(void const *input_ptr,
                                         void const *embedding_ptr,
                                         void *output_ptr) {
  int input_offset = blockIdx.x;
  int64_t const *__restrict__ input = static_cast<int64_t const *>(input_ptr) + input_offset;
  int embedding_offset = blockIdx.y * CHUNK_SIZE;
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr) + embedding_offset;
  int output_offset = blockIdx.y * CHUNK_SIZE + blockIdx.x * OUTPUT_DIM_SIZE;
  T *__restrict__ output = static_cast<T *>(output_ptr) + output_offset;

  if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0) {
    printf("input_offset: %d, embedding_offset: %d, output_offset: %d\n", input_offset, embedding_offset, output_offset);
  }
  embedding_kernel<T, CHUNK_SIZE, OUTPUT_DIM_SIZE>(
      input, embedding, output);
  // if (blockIdx.x == 1 && blockIdx.y == 1) {
  //   printf("input: %d, embedding: %d, output: %d\n", input, embedding, output);
  // }
}

void embedding(torch::Tensor input,
               torch::Tensor weight,
               torch::Tensor output) {

  dim3 grid_dim(input.size(1), output.size(1) / 128, 1);
  printf("grid_dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
  dim3 block_dim(128, 1, 1);

  embedding_kernel_wrapper<float, 128, 4096><<<grid_dim, block_dim>>>(
      input.data_ptr(),
      weight.data_ptr(),
      output.data_ptr());

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error in embedding: %s\n",
           cudaGetErrorString(err));
  }
}

// Prompt Lookup Kernel
template <int NGRAM_SIZE, int NUM_WORKERS>
__global__ void find_ngram_partial_kernel_wrapper(long long const *__restrict__ input_ptr,
                                                  long long *__restrict__ output_id_ptr,
                                                  int input_token_num) {
  // Each block gets a pointer to its unique output slot.
  long long *block_output_ptr = output_id_ptr + blockIdx.x;
  find_ngram_partial_kernel<NGRAM_SIZE, NUM_WORKERS>(input_ptr, block_output_ptr, input_token_num);
}

template <int NGRAM_SIZE, int SPEC_LENGTH, int NUM_PARTIAL_TASKS>
__global__ void find_ngram_global_kernel_wrapper(long long const *__restrict__ input_array,
                                                 long long const *__restrict__ tokens_ptr,
                                                 long long *__restrict__ output_result,
                                                 int step) {
  find_ngram_global_kernel<NGRAM_SIZE, SPEC_LENGTH, NUM_PARTIAL_TASKS>(input_array, tokens_ptr, output_result, step);
}

void prompt_lookup(torch::Tensor all_tokens,
                   int prompt_len,
                   int ngram_size,
                   int spec_length,
                   torch::Tensor final_output) {
  
  constexpr int NUM_WORKERS = 96; // Corresponds to grid size
  dim3 partial_grid_dim(NUM_WORKERS, 1, 1);
  dim3 partial_block_dim(128, 1, 1);
  
  auto partial_output_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor partial_output = torch::full({NUM_WORKERS}, INT_MAX, partial_output_options);
  
  if (ngram_size == 3) {
    find_ngram_partial_kernel_wrapper<3, NUM_WORKERS><<<partial_grid_dim, partial_block_dim>>>(
        static_cast<long long const *>(all_tokens.data_ptr()),
        static_cast<long long *>(partial_output.data_ptr()),
        prompt_len);
  } else {
    throw std::runtime_error("Unsupported ngram_size for prompt_lookup test");
  }

  dim3 global_grid_dim(1, 1, 1);
  dim3 global_block_dim(128, 1, 1);

  if (ngram_size == 3 && spec_length == 5) {
     find_ngram_global_kernel_wrapper<3, 5, NUM_WORKERS><<<global_grid_dim, global_block_dim>>>(
        static_cast<long long const *>(partial_output.data_ptr()),
        static_cast<long long const *>(all_tokens.data_ptr()),
        static_cast<long long *>(final_output.data_ptr()),
        prompt_len);
  } else {
     throw std::runtime_error("Unsupported ngram_size/spec_length for prompt_lookup test");
  }
  
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error in prompt_lookup: %s\n",
           cudaGetErrorString(err));
  }
}

// Verify Kernel
template <int NUM_SPEC_TOKENS>
__global__ void target_verify_greedy_kernel_wrapper(
                                 void const *__restrict__ spec_token_id_ptr,
                                 void const *__restrict__ target_token_id_ptr, 
                                 void *__restrict__ final_output_ptr,
                                 void *__restrict__ tokens_ptr) {
    target_verify_greedy_kernel<NUM_SPEC_TOKENS>(
        spec_token_id_ptr, target_token_id_ptr, final_output_ptr, tokens_ptr);
}

void verify(torch::Tensor spec_tokens,
            torch::Tensor target_tokens,
            torch::Tensor accepted_len,
            torch::Tensor new_tokens) {
    
    constexpr int NUM_SPEC_TOKENS = 5; // Must match python test
    if (spec_tokens.size(0) != NUM_SPEC_TOKENS + 1 ||
        target_tokens.size(0) < NUM_SPEC_TOKENS ||
        accepted_len.size(0) != 1 ||
        new_tokens.size(0) != NUM_SPEC_TOKENS + 1)
         {
        throw std::runtime_error("Invalid tensor shape for verify test");
    }

    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(128, 1, 1);

    target_verify_greedy_kernel_wrapper<NUM_SPEC_TOKENS><<<grid_dim, block_dim>>>(
        spec_tokens.data_ptr(),
        target_tokens.data_ptr(),
        accepted_len.data_ptr(),
        new_tokens.data_ptr()
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error in verify: %s\n", cudaGetErrorString(err));
    }
}

// Argmax Kernel
template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__global__ void argmax_partial_kernel_wrapper(void const *__restrict__ input_ptr,
                                              void *__restrict__ output_val_ptr,
                                              void *__restrict__ output_idx_ptr,
                                              int vocab_size) {
  int batch_idx = blockIdx.y;
  int chunk_idx = blockIdx.x;

  T const *row_input_ptr = static_cast<T const *>(input_ptr) + batch_idx * BATCH_SIZE * vocab_size + chunk_idx * CHUNK_SIZE;
  T *row_output_val_ptr = static_cast<T *>(output_val_ptr) + batch_idx * BATCH_SIZE * NUM_PARTIAL_TASKS + chunk_idx;
  long long *row_output_idx_ptr = static_cast<long long *>(output_idx_ptr) + batch_idx * BATCH_SIZE * NUM_PARTIAL_TASKS + chunk_idx;

  argmax_partial_kernel<T, BATCH_SIZE, CHUNK_SIZE, NUM_PARTIAL_TASKS>(
      row_input_ptr, row_output_val_ptr, row_output_idx_ptr);
}

template <typename T, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__global__ void argmax_reduce_kernel_wrapper(void const *__restrict__ input_val_ptr,
                                             void const *__restrict__ input_idx_ptr,
                                             void *__restrict__ final_output_ptr,
                                             int step,
                                             long long *tokens) {
  int row_idx = blockIdx.y;
  T const *row_input_val_ptr = static_cast<T const *>(input_val_ptr) + row_idx * NUM_PARTIAL_TASKS;
  long long const *row_input_idx_ptr = static_cast<long long const *>(input_idx_ptr) + row_idx * NUM_PARTIAL_TASKS;
  long long *row_output_ptr = static_cast<long long *>(final_output_ptr) + row_idx;

  argmax_reduce_kernel<T, CHUNK_SIZE, NUM_PARTIAL_TASKS>(
      row_input_val_ptr, row_input_idx_ptr, row_output_ptr, step, tokens);
}

void argmax(
  torch::Tensor input, 
  torch::Tensor final_output,
  torch::Tensor partial_idx,
  torch::Tensor partial_val) {
  // long long n_row = input.size(0);
  // long long vocab_size = input.size(1);
  constexpr long long n_row = 6;
  constexpr long long vocab_size = 153600;
  
  constexpr int TOTAL_TASKS = 96;
  long long chunk_size = vocab_size / TOTAL_TASKS;
  if (vocab_size % TOTAL_TASKS != 0) {
      throw std::runtime_error("vocab_size must be divisible by NUM_PARTIAL_TASKS");
  }

  constexpr int BATCH_NUM = 3;
  constexpr int BATCH_SIZE = 2;
  constexpr int TASK_PER_BATCH = TOTAL_TASKS / BATCH_NUM; // 32
  constexpr int CHUNK_SIZE = vocab_size / TASK_PER_BATCH; // 4800

  dim3 partial_grid_dim(TOTAL_TASKS / BATCH_NUM, BATCH_NUM, 1);

  // Create intermediate tensors for partial results
  // auto options_val = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  auto options_idx = torch::TensorOptions().dtype(torch::kInt64).device(input.device());
  // torch::Tensor partial_val = torch::empty({n_row, TASK_PER_BATCH}, options_val);
  // torch::Tensor partial_idx = torch::empty({n_row, TASK_PER_BATCH}, options_idx);
  torch::Tensor tokens = torch::empty({1}, options_idx);

  // Launch partial kernel
  dim3 block_dim(128, 1, 1);
  if (BATCH_SIZE == 2) {

    argmax_partial_kernel_wrapper<bfloat16, BATCH_SIZE, CHUNK_SIZE, TASK_PER_BATCH>
      <<<partial_grid_dim, block_dim>>>(
          input.data_ptr(),
          partial_val.data_ptr(),
          partial_idx.data_ptr(),
          vocab_size);
  }

  // Launch reduce kernel
  dim3 reduce_grid_dim(1, n_row, 1);
  argmax_reduce_kernel_wrapper<bfloat16, CHUNK_SIZE, TASK_PER_BATCH>
      <<<reduce_grid_dim, block_dim>>>(
          partial_val.data_ptr(),
          partial_idx.data_ptr(),
          final_output.data_ptr(),
          0,
          static_cast<long long *>(tokens.data_ptr()));
  
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error in argmax: %s\n", cudaGetErrorString(err));
  }
}

// pybind11 bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prompt_lookup", &prompt_lookup, "Prompt lookup kernel");
  m.def("embedding", &embedding, "Embedding kernel");
  m.def("linear", &linear, "Linear kernel");
  m.def("argmax", &argmax, "Argmax kernel");
  m.def("verify", &verify, "Target verification kernel");
  m.def("norm_linear", &norm_linear, "RMSNorm Linear kernel");
  m.def("silu_mul_linear", &silu_mul_linear, "SILU MUL Linear kernel");
  m.def("single_batch_decoding",
        &single_batch_decoding,
        py::arg("qkv"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("output"),
        py::arg("seq_len"),
        py::arg("qk_norm"),
        py::arg("rotary_embed"),
        py::arg("qnorm_weight") = py::none(),
        py::arg("knorm_weight") = py::none(),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("q_eps") = 0.0f,
        py::arg("k_eps") = 0.0f);
  m.def("paged_attention", &paged_attention, "Paged Attention");
  m.def("window_rms_norm", &window_rms_norm, "Window RMSNorm");
}