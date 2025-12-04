/* Copyright 2025 CMU
 * Copyright (c) 2025 by FlashInfer team.
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


 /* input params required for SM100 attention plan implementation
 * qo_indptr_buffer_ptr: pointer to the indptr buffer for QO, number of query tokens per batch
 * kv_indptr_buffer_ptr: pointer to the indptr buffer for KV, number of key/value tokens per batch
 * 
 */

/* output params required for SM100 attention plan implementation
 * prefill_qo_tile_indices_buffer_ptr: pointer to the indices buffer for prefill QO tiles 
 * kv_indptr_buffer_ptr: pointer to the indptr buffer for KV, number of key/value tokens per batch
 * 
 */

#pragma once
#include <cub/block/block_scan.cuh>
#include "tasks/common/common_header.cuh"
namespace kernel {

union alignas(8) CostIndex {
  struct {
    int bucket_idx;
    float cost;
  };
  long long packed;
};

__device__ __forceinline__ CostIndex min(CostIndex a, CostIndex b) {
  return a.cost < b.cost || (a.cost == b.cost && a.bucket_idx < b.bucket_idx) ? a : b;
}

__device__ __forceinline__ CostIndex get_min_cost_index(CostIndex* warp_min_cost,
                                                        CostIndex cost_index, int num_buckets) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    CostIndex other;
    other.packed = __shfl_xor_sync(0xffffffff, cost_index.packed, offset);
    cost_index = min(cost_index, other);
  }
  if (static_cast<int>(threadIdx.x) % 32 == 0) {
    warp_min_cost[static_cast<int>(threadIdx.x) / 32] = cost_index;
  }
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < 32) {
    cost_index = static_cast<int>(threadIdx.x) * 32 < num_buckets
                     ? warp_min_cost[threadIdx.x]
                     : CostIndex{static_cast<int>(threadIdx.x) * 32,
                                 cuda::std::numeric_limits<float>::infinity()};
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      CostIndex other;
      other.packed = __shfl_xor_sync(0xffffffff, cost_index.packed, offset);
      cost_index = min(cost_index, other);
    }
    if (static_cast<int>(threadIdx.x) == 0) {
      warp_min_cost[0] = cost_index;
    }
  }
  __syncthreads();
  return warp_min_cost[0];
}

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int PAGE_SIZE,
          bool CAUSAL=true,
          int NUM_BUCKETS=128,
          int MIN_KV_CHUNK_SIZE=512,
          int P_Q_TILE_SIZE=128,
          int P_KV_TILE_SIZE=128,
          int D_Q_TILE_SIZE=16,
          int D_KV_TILE_SIZE=128>
__device__ __forceinline__ void
    attention_plan_sm100_task_impl(
        // input params
        int const *qo_indptr_buffer_ptr,
        int const *paged_kv_indptr_buffer_ptr,
        int const *paged_kv_indices_buffer_ptr,
        int const *paged_kv_last_page_len_buffer_ptr,
        int batch_size,
        // output params
        int *prefill_work_indptr,
        int *decode_work_indptr,
        int *worker_batch_indices,
        int *worker_kv_head_indices,
        int *worker_packed_qo_indices,
        int *worker_kv_start,
        int *worker_kv_end,
        int *worker_partial_indptr,
        int *merge_indptr,
        int *merge_o_indices
) {
    static_assert(NUM_QO_HEADS % NUM_KV_HEADS == 0, "NUM_QO_HEADS must be multiple of NUM_KV_HEADS");
    
    constexpr int GQA_GROUP_SIZE = NUM_QO_HEADS / NUM_KV_HEADS;

    // step 1: calculate avg kv_chunk_length per work (can be optimized with smem reduction by binding batch_size dimension to threads)
    int prefill_total_kv_len = 0;
    int decode_total_kv_len = 0;
    for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx){
        int qo_len = qo_indptr_buffer_ptr[batch_idx + 1] - qo_indptr_buffer_ptr[batch_idx];
        int packed_qo_len = qo_len * GQA_GROUP_SIZE;
        int num_pages = paged_kv_indptr_buffer_ptr[batch_idx + 1] - paged_kv_indptr_buffer_ptr[batch_idx];
        int kv_len = (num_pages - 1) * PAGE_SIZE +
                        paged_kv_last_page_len_buffer_ptr[batch_idx];
        int processed_packed_qo_len = 0;
        while (packed_qo_len > 0){
            if (packed_qo_len <= D_Q_TILE_SIZE){
                // decode tile
                decode_total_kv_len += CAUSAL ? kv_len - processed_packed_qo_len / GQA_GROUP_SIZE : kv_len;
                packed_qo_len -= D_Q_TILE_SIZE;
                processed_packed_qo_len += D_Q_TILE_SIZE;
            } else {
                // prefill tile
                prefill_total_kv_len += CAUSAL ? kv_len - processed_packed_qo_len / GQA_GROUP_SIZE : kv_len;
                packed_qo_len -= P_Q_TILE_SIZE;
                processed_packed_qo_len += P_Q_TILE_SIZE;
            }
        }
    }

    // heuristic to estimate avg kv_chunk_length per work, prefill workload is more expensive so we give it a higher weight
    int const prefill_cost = 2;
    int const decode_cost = 1;

    // dispatch works based on the overall cost (estimated by kv length x q_tile_size)
    int total_cost = prefill_total_kv_len * prefill_cost + decode_total_kv_len * decode_cost;
    int avg_cost_per_work = (total_cost + NUM_BUCKETS - 1) / NUM_BUCKETS;
    int prefill_avg_kv_len_per_work =
        (avg_cost_per_work + prefill_cost - 1) / prefill_cost;
    int decode_avg_kv_len_per_work =
        (avg_cost_per_work + decode_cost - 1) / decode_cost;
    prefill_avg_kv_len_per_work = (prefill_avg_kv_len_per_work + MIN_KV_CHUNK_SIZE - 1) / MIN_KV_CHUNK_SIZE * MIN_KV_CHUNK_SIZE; // align to MIN_KV_CHUNK_SIZE
    decode_avg_kv_len_per_work = (decode_avg_kv_len_per_work + MIN_KV_CHUNK_SIZE - 1) / MIN_KV_CHUNK_SIZE * MIN_KV_CHUNK_SIZE; // align to MIN_KV_CHUNK_SIZE

    // step 2: calculate work number per bucket
    __shared__ CostIndex warp_min_cost[32];
    using BlockScan = cub::BlockScan<int, MAX_BUCKET_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    CostIndex thread_local_cost_index = {static_cast<int>(threadIdx.x), 0.f};
    int decode_thread_local_work_counter = 0;
    int merge_work_thread_local_work_counter = 0;
    if (static_cast<int>(threadIdx.x) >= NUM_BUCKETS) {
        thread_local_cost_index.cost = cuda::std::numeric_limits<float>::infinity();
    }

    for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx){
        int qo_len = qo_indptr_buffer_ptr[batch_idx + 1] - qo_indptr_buffer_ptr[batch_idx];
        int packed_qo_len = qo_len * GQA_GROUP_SIZE;
        int num_pages = paged_kv_indptr_buffer_ptr[batch_idx + 1] - paged_kv_indptr_buffer_ptr[batch_idx];
        int kv_len = (num_pages - 1) * PAGE_SIZE +
                        paged_kv_last_page_len_buffer_ptr[batch_idx];
        int processed_packed_qo_len = 0;
        while (packed_qo_len > 0){
            if (packed_qo_len <= D_Q_TILE_SIZE){
                // decode tile
                int remaining_kv_len = CAUSAL ? kv_len - processed_packed_qo_len / GQA_GROUP_SIZE : kv_len;
                while (remaining_kv_len > 0){
                    int work_kv_chunk_size = min(remaining_kv_len, decode_avg_kv_len_per_work);
                    for (int kv_head_idx = 0; kv_head_idx < NUM_KV_HEADS; ++kv_head_idx){
                        auto min_cost_index =
                            get_min_cost_index(warp_min_cost, thread_local_cost_index, NUM_BUCKETS);
                        int bucket_idx = min_cost_index.bucket_idx;
                        if (bucket_idx == threadIdx.x){
                            thread_local_cost_index.cost += work_kv_chunk_size * decode_cost;
                            decode_thread_local_work_counter += 1;
                        }
                    }
                    remaining_kv_len -= work_kv_chunk_size;
                }
                packed_qo_len -= D_Q_TILE_SIZE;
                processed_packed_qo_len += D_Q_TILE_SIZE;
            } else {
                // prefill tile
                int remaining_kv_len = CAUSAL ? kv_len - processed_packed_qo_len / GQA_GROUP_SIZE : kv_len;
                while (remaining_kv_len > 0){
                    int work_kv_chunk_size = min(remaining_kv_len, prefill_avg_kv_len_per_work);
                    for (int kv_head_idx = 0; kv_head_idx < NUM_KV_HEADS; ++kv_head_idx){
                        auto min_cost_index =
                            get_min_cost_index(warp_min_cost, thread_local_cost_index, NUM_BUCKETS);
                        int bucket_idx = min_cost_index.bucket_idx;
                        if (bucket_idx == threadIdx.x){
                            thread_local_cost_index.cost += work_kv_chunk_size * prefill_cost;
                            prefill_thread_local_work_counter += 1;
                        }
                    }
                    remaining_kv_len -= work_kv_chunk_size;
                }
                packed_qo_len -= P_Q_TILE_SIZE;
                processed_packed_qo_len += P_Q_TILE_SIZE;
            }
        }
    } 
    __syncthreads();
    int prefill_thread_local_work_indptr = 0;
    int decode_thread_local_work_indptr = 0;
    BlockScan(temp_storage).ExclusiveSum(prefill_thread_local_work_counter, prefill_thread_local_work_indptr);
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < NUM_BUCKETS) {
        prefill_work_indptr[threadIdx.x] = prefill_thread_local_work_indptr;
    }
    if (static_cast<int>(threadIdx.x) + 1 == NUM_BUCKETS) {
        prefill_work_indptr[NUM_BUCKETS] = prefill_thread_local_work_indptr + prefill_thread_local_work_counter;
    }
    BlockScan(temp_storage).ExclusiveSum(decode_thread_local_work_counter, decode_thread_local_work_indptr);
    __syncthreads();
    decode_thread_local_work_indptr += prefill_work_indptr[NUM_BUCKETS]; // decode work indptr follows prefill work indptr
    if (static_cast<int>(threadIdx.x) < NUM_BUCKETS) {
        decode_work_indptr[threadIdx.x] = decode_thread_local_work_indptr;
    }
    if (static_cast<int>(threadIdx.x) + 1 == NUM_BUCKETS) {
        decode_work_indptr[NUM_BUCKETS] = decode_thread_local_work_indptr + decode_thread_local_work_counter;
    }

    // step 3: fill in work meta info per bucket
    int prefill_thread_local_work_counter = 0;
    int decode_thread_local_work_counter = 0;
    if (static_cast<int>(threadIdx.x) >= NUM_BUCKETS) {
        thread_local_cost_index.cost = cuda::std::numeric_limits<float>::infinity();
    } else {
        thread_local_cost_index.cost = 0.f;
    }

    int partial_o_nnz = 0;
    int split_qo_count = 0;
    int split_kv_tile_count = 0;

    if (threadIdx.x == 0){
        merge_indptr[split_qo_count] = split_kv_tile_count;
    }

    for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx){
        int qo_len = qo_indptr_buffer_ptr[batch_idx + 1] - qo_indptr_buffer_ptr[batch_idx];
        int packed_qo_len = qo_len * GQA_GROUP_SIZE;
        int num_pages = paged_kv_indptr_buffer_ptr[batch_idx + 1] - paged_kv_indptr_buffer_ptr[batch_idx];
        int kv_len = (num_pages - 1) * PAGE_SIZE +
                        paged_kv_last_page_len_buffer_ptr[batch_idx];
        int processed_packed_qo_len = 0;
        while (packed_qo_len > 0){
            if (packed_qo_len <= D_Q_TILE_SIZE){
                // decode tile
                int remaining_kv_len = CAUSAL ? kv_len - processed_packed_qo_len / GQA_GROUP_SIZE : kv_len;
                int kv_start = 0;
                bool split_kv = remaining_kv_len > decode_avg_kv_len_per_work;
                int num_kv_tiles = split_kv ? 
                                    (remaining_kv_len + D_KV_TILE_SIZE - 1) / D_KV_TILE_SIZE : 1;
                int row_tile_size = min(D_Q_TILE_SIZE, packed_qo_len);
                while (remaining_kv_len > 0){
                    int work_kv_chunk_size = min(remaining_kv_len, decode_avg_kv_len_per_work);
                    for (int kv_head_idx = 0; kv_head_idx < NUM_KV_HEADS; ++kv_head_idx){
                        auto min_cost_index =
                            get_min_cost_index(warp_min_cost, thread_local_cost_index, NUM_BUCKETS);
                        int bucket_idx = min_cost_index.bucket_idx;
                        if (bucket_idx == threadIdx.x){
                            thread_local_cost_index.cost += work_kv_chunk_size * decode_cost;
                            // add meta info for this work
                            worker_batch_indices[decode_thread_local_work_indptr + decode_thread_local_work_counter] =
                                batch_idx;
                            worker_kv_head_indices[decode_thread_local_work_indptr + decode_thread_local_work_counter] =
                                kv_head_idx;
                            worker_packed_qo_indices[decode_thread_local_work_indptr + decode_thread_local_work_counter] =
                                (packed_qo_len - 1) / D_Q_TILE_SIZE;
                            worker_kv_start[decode_thread_local_work_indptr + decode_thread_local_work_counter] =
                                kv_start;
                            worker_kv_end[decode_thread_local_work_indptr + decode_thread_local_work_counter] =
                                kv_start + work_kv_chunk_size;
                            worker_partial_indptr[decode_thread_local_work_indptr + decode_thread_local_work_counter] =
                                partial_o_nnz;
                            decode_thread_local_work_counter += 1;
                        }
                    }
                    remaining_kv_len -= work_kv_chunk_size;
                    kv_start += work_kv_chunk_size;
                }
                if (split_kv && threadIdx.x == 0){
                    for (int row = 0; row < row_tile_size; ++row){
                        split_kv_tile_count += num_kv_tiles;
                        split_qo_count += 1;
                        merge_indptr[split_qo_count] = split_kv_tile_count;
                        int q = ((packed_qo_len / D_Q_TILE_SIZE) * D_Q_TILE_SIZE + row) / GQA_GROUP_SIZE;
                        int r = ((packed_qo_len / D_Q_TILE_SIZE) * D_Q_TILE_SIZE + row) % GQA_GROUP_SIZE;
                        merge_o_indices[split_qo_count - 1] = (qo_indptr_buffer_ptr[batch_idx] + q) * NUM_QO_HEADS * GQA_GROUP_SIZE + r;
                    }
                    partial_o_nnz += row_tile_size * num_kv_tiles;
                }
                packed_qo_len -= D_Q_TILE_SIZE;
                processed_packed_qo_len += D_Q_TILE_SIZE;
            } else {
                // prefill tile
                int remaining_kv_len = CAUSAL ? kv_len - processed_packed_qo_len / GQA_GROUP_SIZE : kv_len;
                int kv_start = 0;
                bool split_kv = remaining_kv_len > prefill_avg_kv_len_per_work;
                int num_kv_tiles = split_kv ? 
                                    (remaining_kv_len + P_KV_TILE_SIZE - 1) / P_KV_TILE_SIZE : 1;
                int row_tile_size = min(P_Q_TILE_SIZE, packed_qo_len);
                while (remaining_kv_len > 0){
                    int work_kv_chunk_size = min(remaining_kv_len, prefill_avg_kv_len_per_work);
                    for (int kv_head_idx = 0; kv_head_idx < NUM_KV_HEADS; ++kv_head_idx){
                        auto min_cost_index =
                            get_min_cost_index(warp_min_cost, thread_local_cost_index, NUM_BUCKETS);
                        int bucket_idx = min_cost_index.bucket_idx;
                        if (bucket_idx == threadIdx.x){
                            thread_local_cost_index.cost += work_kv_chunk_size * prefill_cost;
                            // add meta info for this work
                            worker_batch_indices[prefill_thread_local_work_indptr + prefill_thread_local_work_counter] =
                                batch_idx;
                            worker_kv_head_indices[prefill_thread_local_work_indptr + prefill_thread_local_work_counter] =
                                kv_head_idx;
                            worker_packed_qo_indices[prefill_thread_local_work_indptr + prefill_thread_local_work_counter] =
                                (packed_qo_len - 1) / P_Q_TILE_SIZE;
                            worker_kv_start[prefill_thread_local_work_indptr + prefill_thread_local_work_counter] =
                                kv_start;
                            worker_kv_end[prefill_thread_local_work_indptr + prefill_thread_local_work_counter] =
                                kv_start + work_kv_chunk_size;
                            worker_partial_indptr[prefill_thread_local_work_indptr + prefill_thread_local_work_counter] =
                                partial_o_nnz;
                            prefill_thread_local_work_counter += 1;
                        }
                    }
                    remaining_kv_len -= work_kv_chunk_size;
                    kv_start += work_kv_chunk_size;
                }
                if (split_kv && threadIdx.x == 0){
                    for (int row = 0; row < row_tile_size; ++row){
                        split_kv_tile_count += num_kv_tiles;
                        split_qo_count += 1;
                        merge_indptr[split_qo_count] = split_kv_tile_count;
                        int q = ((packed_qo_len / P_Q_TILE_SIZE) * P_Q_TILE_SIZE + row) / GQA_GROUP_SIZE;
                        int r = ((packed_qo_len / P_Q_TILE_SIZE) * P_Q_TILE_SIZE + row) % GQA_GROUP_SIZE;
                        merge_o_indices[split_qo_count - 1] = (qo_indptr_buffer_ptr[batch_idx] + q) * NUM_QO_HEADS * GQA_GROUP_SIZE + r;
                    }
                    partial_o_nnz += row_tile_size * num_kv_tiles;
                }
                packed_qo_len -= P_Q_TILE_SIZE;
                processed_packed_qo_len += P_Q_TILE_SIZE;
            }
        }
    } 
  
} // attention_plan_sm100_task_impl

} // namespace kernel