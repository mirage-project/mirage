#include "runtime_header.h"
#include "tasks/ampere/multitoken_paged_attention_4_16.cuh"
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

constexpr int QO_HEADS = 4;
constexpr int KV_HEADS = 1;
constexpr int HEAD_DIM = 128;
constexpr int QKV_STRIDE = (QO_HEADS + 2 * KV_HEADS) * HEAD_DIM;
constexpr int O_STRIDE = QO_HEADS * HEAD_DIM;
constexpr int MAX_SEQ_LEN = 512;
constexpr int PAGE_SIZE = 64;

__global__ void paged_attention_4_16_wrapper(void const *qkv,
                                             void *k_cache,
                                             void *v_cache,
                                             void *output,
                                             int const *qo_indptr,
                                             int const *kv_indptr,
                                             int const *kv_indices,
                                             int const *kv_last_page_len,
                                             int16_t request_id) {
  kernel::multitoken_paged_attention_task_impl_4_16<bfloat16,
                                                    QO_HEADS,
                                                    KV_HEADS,
                                                    KV_HEADS * HEAD_DIM,
                                                    QKV_STRIDE,
                                                    O_STRIDE,
                                                    HEAD_DIM,
                                                    MAX_SEQ_LEN,
                                                    PAGE_SIZE>(
      qkv, k_cache, v_cache, output, qo_indptr, kv_indptr, kv_indices,
      kv_last_page_len, request_id, false, false, nullptr, nullptr, nullptr,
      nullptr, 0.0f, 0.0f);
}

void paged_attention_4_16(torch::Tensor qkv,
                          torch::Tensor k_cache,
                          torch::Tensor v_cache,
                          torch::Tensor output,
                          torch::Tensor qo_indptr,
                          torch::Tensor kv_indptr,
                          torch::Tensor kv_indices,
                          torch::Tensor kv_last_page_len,
                          int64_t request_id) {
  TORCH_CHECK(qkv.stride(0) == QKV_STRIDE);
  TORCH_CHECK(k_cache.size(1) == PAGE_SIZE);
  size_t smem_size = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(paged_attention_4_16_wrapper,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);
  paged_attention_4_16_wrapper<<<1, 128, smem_size>>>(
      qkv.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), output.data_ptr(),
      qo_indptr.data_ptr<int>(), kv_indptr.data_ptr<int>(),
      kv_indices.data_ptr<int>(), kv_last_page_len.data_ptr<int>(),
      static_cast<int16_t>(request_id));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_4_16", &paged_attention_4_16);
}
