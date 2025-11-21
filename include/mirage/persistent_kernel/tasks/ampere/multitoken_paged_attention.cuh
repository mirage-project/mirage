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
#include "multitoken_paged_attention_32_64.cuh"
#include "multitoken_paged_attention_4_16.cuh"

namespace kernel {
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8>
__device__ __forceinline__ void multitoken_paged_attention_task_impl(
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
  if constexpr ((MAX_TOKENS * NUM_QO_HEADS) <= 16) {
    multitoken_paged_attention_task_impl_4_16<T,
                                              NUM_QO_HEADS,
                                              NUM_KV_HEADS,
                                              KV_CACHE_STRIDE,
                                              QKV_STRIDE,
                                              O_STRIDE,
                                              HEAD_DIM,
                                              MAX_SEQ_LEN,
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
        k_eps);
  } else if ((MAX_TOKENS * NUM_QO_HEADS) <= 64) {
    multitoken_paged_attention_task_impl_32_64<T,
                                               NUM_QO_HEADS,
                                               NUM_KV_HEADS,
                                               KV_CACHE_STRIDE,
                                               QKV_STRIDE,
                                               O_STRIDE,
                                               HEAD_DIM,
                                               MAX_SEQ_LEN,
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
        k_eps);
  }
}
} // namespace kernel