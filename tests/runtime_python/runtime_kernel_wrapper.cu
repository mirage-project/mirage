#include "argmax.cuh"
#include "bfloat16.h"
#include "linear.cuh"
#include "multitoken_paged_attention.cuh"
#include "norm.cuh"
#include "norm_linear.cuh"
#include "paged_attention.cuh"
#include "silu_mul_linear.cuh"
#include "single_batch_decoding.cuh"
#include "single_batch_gqa.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

// using kernel::argmax_kernel;
using kernel::linear_kernel;
using kernel::multitoken_paged_attention_task_impl;
using kernel::norm_linear_task_impl;
using kernel::paged_attention_task_impl;
using kernel::silu_mul_linear_task_impl;
using kernel::single_batch_decoding_kernel;
using kernel::single_batch_gqa_kernel;
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

// Multitoken Paged Attention

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int MAX_TOKENS = 1>
__global__ void multitoken_paged_attention_wrapper(
    void const *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps) {
  multitoken_paged_attention_task_impl<T,
                                       NUM_QO_HEADS,
                                       NUM_KV_HEADS,
                                       HEAD_DIM,
                                       PAGE_SIZE,
                                       MAX_SEQ_LEN,
                                       MAX_TOKENS>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
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
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int MAX_TOKENS = 1>
void launch_multitoken_paged_attention(
    void const *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
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

  cudaFuncSetAttribute(multitoken_paged_attention_wrapper<T,
                                                          NUM_QO_HEADS,
                                                          NUM_KV_HEADS,
                                                          HEAD_DIM,
                                                          PAGE_SIZE,
                                                          MAX_SEQ_LEN,
                                                          MAX_TOKENS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  multitoken_paged_attention_wrapper<T,
                                     NUM_QO_HEADS,
                                     NUM_KV_HEADS,
                                     HEAD_DIM,
                                     PAGE_SIZE,
                                     MAX_SEQ_LEN,
                                     MAX_TOKENS>
      <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
                                           paged_k_cache_ptr,
                                           paged_v_cache_ptr,
                                           output_ptr,
                                           qo_indptr_buffer_ptr,
                                           paged_kv_indptr_buffer_ptr,
                                           paged_kv_indices_buffer_ptr,
                                           paged_kv_last_page_len_buffer_ptr,
                                           request_id,
                                           qk_norm,
                                           rope,
                                           q_norm_weight_ptr,
                                           k_norm_weight_ptr,
                                           cos_ptr,
                                           sin_ptr,
                                           q_eps,
                                           k_eps);
}

void multitoken_paged_attention(
    torch::Tensor qkv,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor output,
    torch::Tensor qo_indptr_buffer,
    torch::Tensor paged_kv_indptr_buffer,
    torch::Tensor paged_kv_indices_buffer,
    torch::Tensor paged_kv_last_page_len_buffer,
    int request_id,
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
  int const *qo_indptr_buffer_ptr = qo_indptr_buffer.data_ptr<int>();
  int const *paged_kv_indptr_buffer_ptr =
      paged_kv_indptr_buffer.data_ptr<int>();
  int const *paged_kv_indices_buffer_ptr =
      paged_kv_indices_buffer.data_ptr<int>();
  int const *paged_kv_last_page_len_buffer_ptr =
      paged_kv_last_page_len_buffer.data_ptr<int>();

  void const *q_norm_weight_ptr = qk_norm ? q_norm_weight->data_ptr() : nullptr;
  void const *k_norm_weight_ptr = qk_norm ? k_norm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rope ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rope ? sin->data_ptr() : nullptr;

  launch_multitoken_paged_attention<bfloat16, 4, 1, 128, 64, 512, 4>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
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

// Argmax

// template <typename T>
// __global__ void argmax_kernel_wrapper(void const *input_ptr, void
// *output_ptr) {
//   argmax_kernel<T, 1, 32768>(input_ptr, output_ptr);
// }

// template <typename T>
// void launch_argmax(void const *input_ptr, void *output_ptr) {
//   dim3 grid_dim(1, 1, 1);
//   dim3 block_dim(128, 1, 1);
//   size_t smem_size = 36666;

//   cudaFuncSetAttribute(argmax_kernel_wrapper<T>,
//                        cudaFuncAttributeMaxDynamicSharedMemorySize,
//                        smem_size);

//   argmax_kernel_wrapper<T>
//       <<<grid_dim, block_dim, smem_size>>>(input_ptr, output_ptr);
// }

// void argmax(torch::Tensor input, torch::Tensor output) {

//   void const *input_ptr = input.data_ptr();
//   void *output_ptr = output.data_ptr();

//   launch_argmax<bfloat16>(input_ptr, output_ptr);

//   cudaError_t err = cudaDeviceSynchronize();
//   if (err != cudaSuccess) {
//     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
//   }
// }

// pybind11 bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear, "Linear kernel");
  // m.def("argmax", &argmax, "argmax kernel");
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
  m.def("multitoken_paged_attention",
        &multitoken_paged_attention,
        "Multitoken Paged Attention");
  m.def("window_rms_norm", &window_rms_norm, "Window RMSNorm");
}