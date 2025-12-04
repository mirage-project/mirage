/* Copyright 2025 Mirage Team
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
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/blackwell/attention/decode_attention_sm100.cuh"
#include "tasks/blackwell/attention/prefill_attention_sm100.cuh"
namespace kernel {

template <typename T,
          class QTensor,
          typename TMA_K,
          typename TMA_V,
          class OTensor,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM_QK,
          int HEAD_DIM_V,
          int PAGE_SIZE,
          bool CAUSAL=true,
          int NUM_BUCKETS=128,
          int MIN_KV_CHUNK_SIZE=512,
          int P_Q_TILE_SIZE=128,
          int P_KV_TILE_SIZE=128,
          int D_Q_TILE_SIZE=16,
          int D_KV_TILE_SIZE=128>
__device__ __forceinline__ void
    mixed_attention_sm100_task_impl(
        QTensor mQ,
        const TMA_K &tma_k,
        const TMA_V &tma_v,
        OTensor mO,
        int const *decode_work_indptr,
        int const *prefill_work_indptr,
        int const *worker_batch_indices,
        int const *worker_kv_head_indices,
        int const *worker_packed_qo_indices,
        int const *worker_kv_start,
        int const *worker_kv_end,
        int const *paged_kv_indices_buffer_ptr,
        int const *paged_kv_indptr_buffer_ptr,
        int const *paged_kv_last_page_len_buffer_ptr,
        int worker_idx) {

    // Prefill attention works
    prefill_attn_sm100<
        T, 
        QTensor,
        TMA_K,
        TMA_V,
        OTensor,
        NUM_QO_HEADS, 
        NUM_KV_HEADS, 
        HEAD_DIM_QK, 
        HEAD_DIM_V,
        PAGE_SIZE,
        P_Q_TILE_SIZE,
        P_KV_TILE_SIZE
    >(
        mQ,
        tma_k,
        tma_v,
        mO,
        prefill_work_indptr, 
        worker_batch_indices,
        worker_kv_head_indices,
        worker_packed_qo_indices,
        worker_kv_start,
        worker_kv_end,
        paged_kv_indices_buffer_ptr,
        paged_kv_indptr_buffer_ptr,
        paged_kv_last_page_len_buffer_ptr,
        worker_idx
    );
    
    __syncthreads();
    // Decode attention works
  
} // mixed_attention_sm100_task_impl

} // namespace kernel
