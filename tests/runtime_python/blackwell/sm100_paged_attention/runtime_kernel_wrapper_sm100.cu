/* Copyright 2025 CMU
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
// Direct launcher for the SM100 paged-attention task, used to reach states
// the megakernel's test mode cannot build: it takes the page table and the
// qo/kv indptrs verbatim, so a request can carry a CACHED PREFIX
// (num_tokens < seq_len). That is what the sliding window's leading-tile skip
// depends on, and under test mode the scheduler always admits a fresh request
// and schedules a whole prefill, giving num_tokens == seq_len.
#include "blackwell/task_header.cuh"
#include "common/bfloat16.h"
#include "runtime_header.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

// GPT-OSS attention geometry: GQA 8:1, 64-wide heads, 64-token pages.
constexpr int NUM_QO_PER_KV = 8;
constexpr int NUM_KV_HEADS = 1;
constexpr int HEAD_DIM = 64;
constexpr int PAGE_SIZE = 64;
constexpr int MAX_SEQ_LEN = 256;
constexpr int MAX_TOKENS = 8;
constexpr int QKV_STRIDE = (NUM_QO_PER_KV + 2 * NUM_KV_HEADS) * HEAD_DIM;
constexpr int O_STRIDE = NUM_QO_PER_KV * HEAD_DIM;
constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;

template <int WINDOW_SIZE>
__global__ void
    paged_attention_sm100_wrapper(void const *qkv_ptr,
                                  void *paged_k_cache_ptr,
                                  void *paged_v_cache_ptr,
                                  void *output_ptr,
                                  int const *qo_indptr_buffer_ptr,
                                  int const *paged_kv_indptr_buffer_ptr,
                                  int const *paged_kv_indices_buffer_ptr,
                                  int const *paged_kv_last_page_len_buffer_ptr,
                                  void const *q_norm_weight_ptr,
                                  void const *k_norm_weight_ptr,
                                  void const *cos_ptr,
                                  void const *sin_ptr) {
  kernel::multitoken_paged_attention_sm100_task_impl<bfloat16,
                                                     NUM_QO_PER_KV,
                                                     NUM_KV_HEADS,
                                                     KV_CACHE_STRIDE,
                                                     QKV_STRIDE,
                                                     O_STRIDE,
                                                     HEAD_DIM,
                                                     MAX_SEQ_LEN,
                                                     PAGE_SIZE,
                                                     0,
                                                     0,
                                                     MAX_TOKENS,
                                                     HEAD_DIM,
                                                     WINDOW_SIZE>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      /*request_id*/ 0,
      /*qk_norm*/ false,
      /*rope*/ true,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      1e-6f,
      1e-6f);
}

template <int WINDOW_SIZE>
static void launch(torch::Tensor qkv,
                   torch::Tensor k_cache,
                   torch::Tensor v_cache,
                   torch::Tensor output,
                   torch::Tensor qo_indptr,
                   torch::Tensor kv_indptr,
                   torch::Tensor kv_indices,
                   torch::Tensor kv_last_page_len,
                   torch::Tensor q_norm,
                   torch::Tensor k_norm,
                   torch::Tensor cos,
                   torch::Tensor sin) {
  size_t smem_size = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(paged_attention_sm100_wrapper<WINDOW_SIZE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);
  paged_attention_sm100_wrapper<WINDOW_SIZE>
      <<<dim3(1, 1, 1), dim3(128, 1, 1), smem_size>>>(
          qkv.data_ptr(),
          k_cache.data_ptr(),
          v_cache.data_ptr(),
          output.data_ptr(),
          qo_indptr.data_ptr<int>(),
          kv_indptr.data_ptr<int>(),
          kv_indices.data_ptr<int>(),
          kv_last_page_len.data_ptr<int>(),
          q_norm.data_ptr(),
          k_norm.data_ptr(),
          cos.data_ptr(),
          sin.data_ptr());
}

// WINDOW_SIZE is a template parameter, so the dispatch enumerates the windows
// the test uses: 0 (full causal), 96 (skips an odd number of KV tiles) and
// 32 (an even number). The parity matters -- the kernel's double-buffer phase
// is counted from the first tile it visits.
static void paged_attention_sm100(torch::Tensor qkv,
                                  torch::Tensor k_cache,
                                  torch::Tensor v_cache,
                                  torch::Tensor output,
                                  torch::Tensor qo_indptr,
                                  torch::Tensor kv_indptr,
                                  torch::Tensor kv_indices,
                                  torch::Tensor kv_last_page_len,
                                  torch::Tensor q_norm,
                                  torch::Tensor k_norm,
                                  torch::Tensor cos,
                                  torch::Tensor sin,
                                  int64_t window_size) {
#define DISPATCH(W)                                                            \
  case W:                                                                      \
    launch<W>(qkv,                                                             \
              k_cache,                                                         \
              v_cache,                                                         \
              output,                                                          \
              qo_indptr,                                                       \
              kv_indptr,                                                       \
              kv_indices,                                                      \
              kv_last_page_len,                                                \
              q_norm,                                                          \
              k_norm,                                                          \
              cos,                                                             \
              sin);                                                            \
    break;
  switch (window_size) {
    DISPATCH(0)
    DISPATCH(32)
    DISPATCH(96)
    default:
      TORCH_CHECK(false, "window_size ", window_size, " is not instantiated");
  }
#undef DISPATCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_sm100",
        &paged_attention_sm100,
        "Paged attention SM100 with an explicit page table");
}
