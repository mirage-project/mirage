
#include "bfloat16.h"
#include "multitoken_paged_attention_32_64_split_kv.cuh"
#include "merge_splitkv.cuh"
#include "single_batch_decoding.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

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
  kernel::single_batch_decoding_kernel<T,
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
  size_t smem_size = 67456;

  void const *qnorm_weight_ptr = qk_norm ? qnorm_weight->data_ptr() : nullptr;
  void const *knorm_weight_ptr = qk_norm ? knorm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rotary_emd ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rotary_emd ? sin->data_ptr() : nullptr;

  cudaFuncSetAttribute(single_batch_decoding_wrapper<bfloat16, 4, 1, 128, 128>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int WARMUP_RUNS = 16;
  constexpr int BENCHMARK_RUNS = 1000;

  printf("=== Multitoken Paged Attention Kernel Performance Profiling ===\n");

  for (int i = 0; i < WARMUP_RUNS; i++) {
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
  }
  cudaDeviceSynchronize();

  printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  float *iteration_times = new float[BENCHMARK_RUNS];
  float total_time_ms = 0.0f;
  cudaEventRecord(start);
  for (int i = 0; i < BENCHMARK_RUNS; i++) {

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
    cudaDeviceSynchronize();
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time_ms, start, stop);

  float avg_time_ms = total_time_ms / BENCHMARK_RUNS;

  printf("  Average: %.3f ms\n", avg_time_ms);

  printf("===============================\n");

  delete[] iteration_times;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

////////////////////////////////////////////////////////////

// Multitoken Paged Attention

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 16,
          bool PARTITION_KV = true>
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
    float k_eps,
    void *o_tmp,
    void *lse_tmp) {

  float* lse_base = static_cast<float*>(lse_tmp);
  float* lse_block = lse_base + NUM_QO_HEADS * MAX_TOKENS * blockIdx.x;

  T* o_tmp_base = static_cast<T*>(o_tmp);
  T* o_tmp_block = o_tmp_base + NUM_QO_HEADS * MAX_TOKENS * HEAD_DIM * blockIdx.x;
  kernel::multitoken_paged_attention_task_impl_32_64_split_kv<
      T,
      NUM_QO_HEADS,
      NUM_KV_HEADS,
      KV_CACHE_STRIDE,
      QKV_STRIDE,
      O_STRIDE,
      HEAD_DIM,
      256,
      MAX_SEQ_LEN,
      PAGE_SIZE,
      MAX_TOKENS,
      PARTITION_KV>(qkv_ptr,
                    paged_k_cache_ptr,
                    paged_v_cache_ptr,
                    reinterpret_cast<void*>(o_tmp_block),
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
                    k_eps,
                    reinterpret_cast<void*>(lse_block),
                    blockIdx.x);
}


template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 16,
          bool PARTITION_KV = true>
__global__ void merge_splitkv_wrapper(
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    void *o_tmp,
    void *lse_tmp,
    void *output_ptr) {
      kernel::merge_splitkv<T, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_TOKENS, PARTITION_KV, (MAX_SEQ_LEN / 256), 256, PAGE_SIZE>(lse_tmp, o_tmp, qo_indptr_buffer_ptr, paged_kv_indptr_buffer_ptr, paged_kv_last_page_len_buffer_ptr, request_id, output_ptr);               
    }

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 16,
          bool PARTITION_KV = true>
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
    float k_eps,
    void *o_tmp_ptr,
   void *lse_tmp_ptr) {

  dim3 grid_dim(PARTITION_KV ? MAX_SEQ_LEN / 256 : 1, 1, 1);
  dim3 block_dim(128, 1, 1);

  printf("luanch blocks %d\n", MAX_SEQ_LEN / 256);

  size_t smem_size = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;

  cudaFuncSetAttribute(
      multitoken_paged_attention_wrapper<T,
                                         NUM_QO_HEADS,
                                         NUM_KV_HEADS,
                                         KV_CACHE_STRIDE,
                                         QKV_STRIDE,
                                         O_STRIDE,
                                         HEAD_DIM,
                                         MAX_SEQ_LEN,
                                         PAGE_SIZE,
                                         MAX_TOKENS,
                                        PARTITION_KV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  multitoken_paged_attention_wrapper<T,
                                     NUM_QO_HEADS,
                                     NUM_KV_HEADS,
                                     KV_CACHE_STRIDE,
                                     QKV_STRIDE,
                                     O_STRIDE,
                                     HEAD_DIM,
                                     MAX_SEQ_LEN,
                                     PAGE_SIZE,
                                     MAX_TOKENS,
                                     PARTITION_KV>
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
                                           k_eps,
                                          o_tmp_ptr,
                                        lse_tmp_ptr);

  cudaDeviceSynchronize();


  cudaFuncSetAttribute(
    merge_splitkv_wrapper<T,
                          NUM_QO_HEADS,
                          NUM_KV_HEADS,
                          KV_CACHE_STRIDE,
                          QKV_STRIDE,
                          O_STRIDE,
                          HEAD_DIM,
                          MAX_SEQ_LEN,
                          PAGE_SIZE,
                          MAX_TOKENS,
                          PARTITION_KV>,
  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  merge_splitkv_wrapper<T,
                        NUM_QO_HEADS,
                        NUM_KV_HEADS,
                        KV_CACHE_STRIDE,
                        QKV_STRIDE,
                        O_STRIDE,
                        HEAD_DIM,
                        MAX_SEQ_LEN,
                        PAGE_SIZE,
                        MAX_TOKENS,
                        PARTITION_KV>
                        <<<dim3(1,1,1), block_dim, smem_size>>>(qo_indptr_buffer_ptr, paged_kv_indptr_buffer_ptr, paged_kv_last_page_len_buffer_ptr, request_id, o_tmp_ptr, lse_tmp_ptr, output_ptr);

  cudaDeviceSynchronize();

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // constexpr int WARMUP_RUNS = 0;
  // constexpr int BENCHMARK_RUNS = 1;

  // printf("=== Multitoken Paged Attention Kernel Performance Profiling
  // ===\n");

  // for (int i = 0; i < WARMUP_RUNS; i++) {
  // multitoken_paged_attention_wrapper<T,
  //                                    NUM_QO_HEADS,
  //                                    NUM_KV_HEADS,
  //                                    KV_CACHE_STRIDE,
  //                                    QKV_STRIDE,
  //                                    O_STRIDE,
  //                                    HEAD_DIM,
  //                                    MAX_SEQ_LEN,
  //                                    PAGE_SIZE,
  //                                    MAX_TOKENS>
  //     <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
  //                                          paged_k_cache_ptr,
  //                                          paged_v_cache_ptr,
  //                                          output_ptr,
  //                                          qo_indptr_buffer_ptr,
  //                                          paged_kv_indptr_buffer_ptr,
  //                                          paged_kv_indices_buffer_ptr,
  //                                          paged_kv_last_page_len_buffer_ptr,
  //                                          request_id,
  //                                          qk_norm,
  //                                          rope,
  //                                          q_norm_weight_ptr,
  //                                          k_norm_weight_ptr,
  //                                          cos_ptr,
  //                                          sin_ptr,
  //                                          q_eps,
  //                                          k_eps);
  // }
  // cudaDeviceSynchronize();

  // printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  // float *iteration_times = new float[BENCHMARK_RUNS];
  // float total_time_ms = 0.0f;
  // cudaEventRecord(start);
  // for (int i = 0; i < BENCHMARK_RUNS; i++) {

  //   multitoken_paged_attention_wrapper<T,
  //                                      NUM_QO_HEADS,
  //                                      NUM_KV_HEADS,
  //                                      KV_CACHE_STRIDE,
  //                                      QKV_STRIDE,
  //                                      O_STRIDE,
  //                                      HEAD_DIM,
  //                                      MAX_SEQ_LEN,
  //                                      PAGE_SIZE,
  //                                      MAX_TOKENS>
  //       <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
  //                                            paged_k_cache_ptr,
  //                                            paged_v_cache_ptr,
  //                                            output_ptr,
  //                                            qo_indptr_buffer_ptr,
  //                                            paged_kv_indptr_buffer_ptr,
  //                                            paged_kv_indices_buffer_ptr,
  //                                            paged_kv_last_page_len_buffer_ptr,
  //                                            request_id,
  //                                            qk_norm,
  //                                            rope,
  //                                            q_norm_weight_ptr,
  //                                            k_norm_weight_ptr,
  //                                            cos_ptr,
  //                                            sin_ptr,
  //                                            q_eps,
  //                                            k_eps);
  // }

  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);

  // cudaEventElapsedTime(&total_time_ms, start, stop);

  // float avg_time_ms = total_time_ms / BENCHMARK_RUNS;

  // printf("\n=== Multitoken Paged Attention Performance Results ===\n");
  // printf("Configuration:\n");
  // printf("  NUM_QO_HEADS=%d, NUM_KV_HEADS=%d, HEAD_DIM=%d\n",
  //        NUM_QO_HEADS,
  //        NUM_KV_HEADS,
  //        HEAD_DIM);
  // printf("  PAGE_SIZE=%d, MAX_SEQ_LEN=%d, MAX_TOKENS=%d\n",
  //        PAGE_SIZE,
  //        MAX_SEQ_LEN,
  //        MAX_TOKENS);
  // printf("  Average: %.3f ms\n", avg_time_ms);

  // printf("===============================\n");

  // delete[] iteration_times;
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
}

template<typename T,
         int NUM_QO_HEADS,
         int NUM_KV_HEADS,
         int KV_CACHE_STRIDE,
         int QKV_STRIDE,
         int O_STRIDE,
         int HEAD_DIM,
         int PAGE_SIZE,
         int MAX_TOKENS>
void launch_multitoken_paged_attention_dispatch(
    void const* qkv_ptr,
    void* paged_k_cache_ptr,
    void* paged_v_cache_ptr,
    void* output_ptr,
    int const* qo_indptr_buffer_ptr,
    int const* paged_kv_indptr_buffer_ptr,
    int const* paged_kv_indices_buffer_ptr,
    int const* paged_kv_last_page_len_buffer_ptr,
    int request_id,
    bool qk_norm,
    bool rope,
    void const* q_norm_weight_ptr,
    void const* k_norm_weight_ptr,
    void const* cos_ptr,
    void const* sin_ptr,
    float q_eps,
    float k_eps,
    void* o_tmp_ptr,
    void* lse_tmp_ptr,
  int seq_len) {

  switch (seq_len) {
    case 256:
      launch_multitoken_paged_attention<T,
                                        NUM_QO_HEADS,
                                        NUM_KV_HEADS,
                                        KV_CACHE_STRIDE,
                                        QKV_STRIDE,
                                        O_STRIDE,
                                        HEAD_DIM,
                                        256,
                                        PAGE_SIZE,
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
          k_eps,
          o_tmp_ptr,
          lse_tmp_ptr);
      break;

    case 512:
      launch_multitoken_paged_attention<T,
                                        NUM_QO_HEADS,
                                        NUM_KV_HEADS,
                                        KV_CACHE_STRIDE,
                                        QKV_STRIDE,
                                        O_STRIDE,
                                        HEAD_DIM,
                                        512,
                                        PAGE_SIZE,
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
          k_eps,
          o_tmp_ptr,
          lse_tmp_ptr);
      break;

      case 1024:
      launch_multitoken_paged_attention<T,
                                        NUM_QO_HEADS,
                                        NUM_KV_HEADS,
                                        KV_CACHE_STRIDE,
                                        QKV_STRIDE,
                                        O_STRIDE,
                                        HEAD_DIM,
                                        1024,
                                        PAGE_SIZE,
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
          k_eps,
          o_tmp_ptr,
          lse_tmp_ptr);
      break;

      case 2048:
      launch_multitoken_paged_attention<T,
                                        NUM_QO_HEADS,
                                        NUM_KV_HEADS,
                                        KV_CACHE_STRIDE,
                                        QKV_STRIDE,
                                        O_STRIDE,
                                        HEAD_DIM,
                                        2048,
                                        PAGE_SIZE,
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
          k_eps,
          o_tmp_ptr,
          lse_tmp_ptr);
      break;

      case 4096:
      launch_multitoken_paged_attention<T,
                                        NUM_QO_HEADS,
                                        NUM_KV_HEADS,
                                        KV_CACHE_STRIDE,
                                        QKV_STRIDE,
                                        O_STRIDE,
                                        HEAD_DIM,
                                        4096,
                                        PAGE_SIZE,
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
          k_eps,
          o_tmp_ptr,
          lse_tmp_ptr);
      break;

      case 8192:
      launch_multitoken_paged_attention<T,
                                        NUM_QO_HEADS,
                                        NUM_KV_HEADS,
                                        KV_CACHE_STRIDE,
                                        QKV_STRIDE,
                                        O_STRIDE,
                                        HEAD_DIM,
                                        8192,
                                        PAGE_SIZE,
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
          k_eps,
          o_tmp_ptr,
          lse_tmp_ptr);
      break;

    default:
      TORCH_CHECK(false, "Unsupported seq_len = ", seq_len);
  }
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
    torch::Tensor o_tmp,
    torch::Tensor lse_tmp,
    torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f,
    int seq_len = 512) {
  void const *qkv_ptr = qkv.data_ptr();
  void *paged_k_cache_ptr = paged_k_cache.data_ptr();
  void *paged_v_cache_ptr = paged_v_cache.data_ptr();
  void *output_ptr = output.data_ptr();

  void *o_tmp_ptr = o_tmp.data_ptr();
  void *lse_tmp_ptr = lse_tmp.data_ptr();
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
  int const qo_heads = 4;
  int const kv_heads = 1;
  int const head_dim = 128;
  int const qkv_stride = (qo_heads + 2 * kv_heads) * head_dim;
  assert(qkv_stride == qkv.stride(0));
  int const kv_stride = head_dim * kv_heads;
  assert(kv_stride == paged_k_cache.stride(1));
  int const o_stride = head_dim * qo_heads;
  int const page_size = 128;
  int const max_seq_len = 512;

  int const max_tokens = 8;

  launch_multitoken_paged_attention_dispatch<bfloat16,
                                             qo_heads,
                                             kv_heads,
                                             kv_stride,
                                             qkv_stride,
                                             o_stride,
                                             head_dim,
                                             page_size,
                                             max_tokens>(
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
      k_eps,
      o_tmp_ptr,
      lse_tmp_ptr,
    seq_len);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// pybind11 bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("multitoken_paged_attention_partition_kv",
  //       &multitoken_paged_attention,
  //       "Multitoken Paged Attention");
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
  m.def("multitoken_paged_attention",
        &multitoken_paged_attention,
        "Multitoken Paged Attention");
}
