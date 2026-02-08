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

// this kernel merges the result of one KV head chunk back to the full KV
// cache，
// it taks the output of multitoken_paged_attention_task_impl_32_64_split_kv and
// the log exp sum as input,
namespace kernel {

template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int HEAD_DIM,
          int MAX_TOKENS = 8,
          bool PARTITION_KV = true,
          int NUM_KV_CHUNKS = 1,
          int KV_CHUNK_SIZE = 256,
          int PAGE_SIZE = 4096>
__device__ __forceinline__ void
    merge_splitkv(void const *lse,
                  void const *o,
                  int const *qo_indptr_buffer_ptr,
                  int const *paged_kv_indptr_buffer_ptr,
                  int const *paged_kv_last_page_len_buffer_ptr,
                  int16_t request_id,
                  void *output,
                  int merge_task_offset) {
  if (threadIdx.x >= 128) {
    return;
  }
  T const *o_ptr = reinterpret_cast<T const *>(o);
  T *output_ptr = reinterpret_cast<T *>(output);
  float const *lse_ptr = reinterpret_cast<float const *>(lse);
  // constexpr int GLOBAL_ITERS_M = (NUM_QO_HEADS_PER_KV + 64 - 1) / 64;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int seq_len = (num_pages - 1) * PAGE_SIZE +
                paged_kv_last_page_len_buffer_ptr[request_id];

  int num_chunks = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;

  // size of o and output is NUM_QO_HEADS_PER_KV * MAX_TOKENS * 128
  // let each thread process one
  //  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS_PER_KV / NUM_KV_HEADS;
  //  constexpr int NUM_Q = MAX_TOKENS * NUM_QO_PER_KV;
  //  constexpr int GLOBAL_ITERS_M = (NUM_Q + 64 - 1) / 64;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  // Exit the current task is number of query tokens is zero
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  constexpr int THREADS_PER_TOKEN = 16; // let 16 threads process one head
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;

  int thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int group_id = threadIdx.x / THREADS_PER_TOKEN;
  int head_partition = thread_in_group;

  // let 16 threads to process one head_dim
#pragma unroll 1
  for (int tok = group_id; tok < num_tokens * NUM_QO_HEADS_PER_KV;
       tok += num_groups) {

    int token_idx = tok / NUM_QO_HEADS_PER_KV;
    int head_idx = tok % NUM_QO_HEADS_PER_KV;

#pragma unroll 1
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float m_global = -inf;
      float d_global = 1.f;
      float o_global = 0.f;
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        // process 8 tokens
        float m_prev = m_global,
              d_prev = d_global; // save previous values
        // int lse_offset = kv_idx * (MAX_TOKENS * NUM_QO_HEADS_PER_KV) + tok;
        // int o_offset = (kv_idx * (MAX_TOKENS * NUM_QO_HEADS_PER_KV) + tok) *
        // HEAD_DIM +  head_partition * VAL_PER_THREAD + i;

        int lse_offset =
            head_idx + kv_idx * NUM_QO_HEADS_PER_KV +
            token_idx * NUM_QO_GROUPS * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        // int lse_offset = merge_task_offset * NUM_QO_HEADS_PER_KV + head_idx +
        // kv_idx * NUM_QO_HEADS_PER_KV + token_idx * NUM_QO_GROUPS *
        // NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        int o_offset =
            lse_offset * HEAD_DIM + head_partition * VAL_PER_THREAD + i;

        float other_m = lse_ptr[lse_offset], other_d = 1;
        m_global = max(m_prev, other_m);
        d_global = d_prev * ptx_exp2(m_prev - m_global) +
                   other_d * ptx_exp2(other_m - m_global);
        // accumulate o
        float other_o = (float)o_ptr[o_offset];

        o_global = o_global * ptx_exp2(m_prev - m_global) +
                   other_o * ptx_exp2(other_m - m_global);
      }
      output_ptr[token_idx * NUM_QO_GROUPS * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 head_idx * HEAD_DIM + head_partition * VAL_PER_THREAD + i] =
          (T)__fdividef(o_global, d_global);
    }
  }
}

} // namespace kernel