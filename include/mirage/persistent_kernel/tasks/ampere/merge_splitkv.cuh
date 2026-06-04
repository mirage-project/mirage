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
          int PAGE_SIZE = 4096,
          int PREFILL_THRESHOLD = -1,
          int DIRECT_UNSPLIT_PREFILL = 0>
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
  int request_num_chunks = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  if constexpr (PREFILL_THRESHOLD >= 0) {
    int const packed_qo = num_tokens * NUM_QO_HEADS_PER_KV;
    if (packed_qo <= PREFILL_THRESHOLD) {
      return;
    }
    if constexpr (DIRECT_UNSPLIT_PREFILL) {
      if (seq_len <= KV_CHUNK_SIZE) {
        return;
      }
    }
    if (seq_len <= KV_CHUNK_SIZE) {
      request_num_chunks = 1;
    }
  }
  request_num_chunks = min(request_num_chunks, NUM_KV_CHUNKS);
  int const history_len = seq_len - num_tokens;

  constexpr int THREADS_PER_TOKEN = 16; // let 16 threads process one head
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;
  static_assert(HEAD_DIM % THREADS_PER_TOKEN == 0);

  int thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int group_id = threadIdx.x / THREADS_PER_TOKEN;
  int head_partition = thread_in_group;
  int warp_lane = threadIdx.x & 31;
  unsigned group_mask = (warp_lane < THREADS_PER_TOKEN) ? 0x0000ffffu
                                                        : 0xffff0000u;
  extern __shared__ char smem[];
  float *s_chunk_weights = reinterpret_cast<float *>(smem);

  // let 16 threads to process one head_dim
#pragma unroll 1
  for (int tok = group_id; tok < num_tokens * NUM_QO_HEADS_PER_KV;
       tok += num_groups) {

    int token_idx = tok / NUM_QO_HEADS_PER_KV;
    int head_idx = tok % NUM_QO_HEADS_PER_KV;
    int token_seq_len = history_len + token_idx + 1;
    int num_chunks = min(NUM_KV_CHUNKS,
                         max(1, (token_seq_len + KV_CHUNK_SIZE - 1) /
                                    KV_CHUNK_SIZE));

    if (thread_in_group == 0) {
      float m_global = -inf;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        int const partial_group_offset =
            (kv_idx * NUM_QO_GROUPS + merge_task_offset) *
            NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            partial_group_offset + head_idx +
            token_idx * NUM_QO_GROUPS * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        float other_m = lse_ptr[lse_offset];
        m_global = max(m_global, other_m);
      }

      float d_global = 0.f;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        int const partial_group_offset =
            (kv_idx * NUM_QO_GROUPS + merge_task_offset) *
            NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            partial_group_offset + head_idx +
            token_idx * NUM_QO_GROUPS * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        float other_m = lse_ptr[lse_offset];
        float weight =
            (other_m == -inf || m_global == -inf) ? 0.f
                                                  : ptx_exp2(other_m - m_global);
        s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx] = weight;
        d_global += weight;
      }

      if (d_global > 0.f) {
        float inv_d = __fdividef(1.f, d_global);
#pragma unroll 1
        for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
          s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx] *= inv_d;
        }
      }
    }
    __syncwarp(group_mask);

#pragma unroll 1
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float o_global = 0.f;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        int partial_group_offset =
            (kv_idx * NUM_QO_GROUPS + merge_task_offset) *
            NUM_QO_HEADS_PER_KV;
        int lse_offset =
            partial_group_offset + head_idx +
            token_idx * NUM_QO_GROUPS * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        int o_offset =
            lse_offset * HEAD_DIM + head_partition * VAL_PER_THREAD + i;

        float weight = s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx];
        if (weight != 0.f) {
          float other_o = (float)o_ptr[o_offset];
          o_global += other_o * weight;
        }
      }
      output_ptr[token_idx * NUM_QO_GROUPS * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 merge_task_offset * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 head_idx * HEAD_DIM + head_partition * VAL_PER_THREAD + i] =
          (T)o_global;
    }
    __syncwarp(group_mask);
  }
}

} // namespace kernel

namespace kernel {

template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int MAX_TOKENS,
          int NUM_KV_CHUNKS = 1,
          int KV_CHUNK_SIZE = 256,
          int PAGE_SIZE = 4096,
          int PREFILL_THRESHOLD = -1,
          int DIRECT_UNSPLIT_PREFILL = 0,
          int MERGE_COUNTER_OFFSET = 0,
          int SPLIT_FLAGS_OFFSET = 0>
__device__ __forceinline__ void
    merge_splitkv_planned_persistent(void const *lse,
                                     void const *o,
                                     int *plan_buffer,
                                     int const *qo_indptr_buffer_ptr,
                                     int const *paged_kv_indptr_buffer_ptr,
                                     int const *paged_kv_last_page_len_buffer_ptr,
                                     void *output) {
  T const *o_ptr = reinterpret_cast<T const *>(o);
  T *output_ptr = reinterpret_cast<T *>(output);
  float const *lse_ptr = reinterpret_cast<float const *>(lse);
  int *merge_counter = plan_buffer + MERGE_COUNTER_OFFSET;
  // Decode occupancy-gate decision written by the planner (single source of
  // truth). bit0 = decode split on. When SPLIT_FLAGS_OFFSET is 0 (legacy codegen
  // that does not pass the slot) default to on, matching the plain is_unsplit
  // skip.
  int const split_flags =
      (SPLIT_FLAGS_OFFSET > 0) ? plan_buffer[SPLIT_FLAGS_OFFSET] : 1;
  bool const decode_split_on = (split_flags & 1) != 0;

  constexpr int THREADS_PER_TOKEN = 16;
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int TOTAL_QO_HEADS = NUM_KV_HEADS * NUM_QO_HEADS_PER_KV;
  static_assert(HEAD_DIM % THREADS_PER_TOKEN == 0);
  int const thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int const group_id = threadIdx.x / THREADS_PER_TOKEN;
  int const head_partition = thread_in_group;
  int const warp_lane = threadIdx.x & 31;
  int const src_lane = (warp_lane < THREADS_PER_TOKEN) ? 0 : THREADS_PER_TOKEN;
  unsigned const group_mask = (warp_lane < THREADS_PER_TOKEN) ? 0x0000ffffu
                                                              : 0xffff0000u;
  extern __shared__ char smem[];
  float *s_chunk_weights = reinterpret_cast<float *>(smem);

  int const total_tokens = qo_indptr_buffer_ptr[MPK_MAX_NUM_BATCHED_REQUESTS];
  int const total_rows = total_tokens * TOTAL_QO_HEADS;

  while (true) {
    int row = 0;
    if (thread_in_group == 0) {
      row = atomicAdd(merge_counter, 1);
    }
    row = __shfl_sync(group_mask, row, src_lane);
    if (row >= total_rows) {
      break;
    }

    int const token_pos = row / TOTAL_QO_HEADS;
    int const flat_head = row - token_pos * TOTAL_QO_HEADS;
    int const kv_head = flat_head / NUM_QO_HEADS_PER_KV;
    int const head_idx = flat_head - kv_head * NUM_QO_HEADS_PER_KV;

    int request_id = 0;
#pragma unroll 1
    for (int r = 0; r < MPK_MAX_NUM_BATCHED_REQUESTS; r++) {
      if (token_pos < qo_indptr_buffer_ptr[r + 1]) {
        request_id = r;
        break;
      }
    }
    int const first_token_pos = qo_indptr_buffer_ptr[request_id];
    int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
    int const num_tokens = last_token_pos - first_token_pos;
    int const token_idx = token_pos - first_token_pos;

    int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
    int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
    int const num_pages = last_page_pos - first_page_pos;
    int const seq_len = (num_pages - 1) * PAGE_SIZE +
                        paged_kv_last_page_len_buffer_ptr[request_id];

    int request_num_chunks = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
    bool skip = num_tokens <= 0;
    if constexpr (PREFILL_THRESHOLD >= 0) {
      int const packed_qo = num_tokens * NUM_QO_HEADS_PER_KV;
      bool const is_decode = packed_qo <= PREFILL_THRESHOLD;
      bool const is_unsplit = seq_len <= KV_CHUNK_SIZE;
      if (is_decode) {
        // A decode request produced partials (and so needs merging) only when it
        // was actually split: long enough to span >1 chunk (!is_unsplit) AND the
        // runtime occupancy gate left splitting on (decode_split_on). Any other
        // case wrote straight to final (consumer kv_tile < 0) and must be
        // skipped. Using the runtime gate flag — not just is_unsplit — is what
        // keeps a long-context decode correct when the gate closes and leaves it
        // unsplit.
        skip = skip || is_unsplit || !decode_split_on;
      } else if constexpr (DIRECT_UNSPLIT_PREFILL) {
        // Prefill always splits when seq > chunk, so an unsplit (seq <= chunk)
        // prefill request was written straight to final and must be skipped.
        skip = skip || is_unsplit;
      }
      if (is_unsplit) {
        request_num_chunks = 1;
      }
    }
    request_num_chunks = min(request_num_chunks, NUM_KV_CHUNKS);
    if (skip) {
      continue;
    }
    int const history_len = seq_len - num_tokens;
    int token_seq_len = history_len + token_idx + 1;
    int num_chunks = min(NUM_KV_CHUNKS,
                         max(1, (token_seq_len + KV_CHUNK_SIZE - 1) /
                                    KV_CHUNK_SIZE));

    if (thread_in_group == 0) {
      float m_global = -inf;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
        int const partial_group_offset =
            (kv_idx * NUM_KV_HEADS + kv_head) * NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            token_pos * NUM_KV_CHUNKS * NUM_KV_HEADS * NUM_QO_HEADS_PER_KV +
            partial_group_offset + head_idx;
        float other_m = lse_ptr[lse_offset];
        m_global = max(m_global, other_m);
      }

      float d_global = 0.f;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
        int const partial_group_offset =
            (kv_idx * NUM_KV_HEADS + kv_head) * NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            token_pos * NUM_KV_CHUNKS * NUM_KV_HEADS * NUM_QO_HEADS_PER_KV +
            partial_group_offset + head_idx;
        float other_m = lse_ptr[lse_offset];
        float weight =
            (other_m == -inf || m_global == -inf) ? 0.f
                                                  : ptx_exp2(other_m - m_global);
        s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx] = weight;
        d_global += weight;
      }

      if (d_global > 0.f) {
        float inv_d = __fdividef(1.f, d_global);
#pragma unroll 1
        for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
          s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx] *= inv_d;
        }
      }
    }
    __syncwarp(group_mask);

#pragma unroll 1
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float o_global = 0.f;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
        int const partial_group_offset =
            (kv_idx * NUM_KV_HEADS + kv_head) * NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            token_pos * NUM_KV_CHUNKS * NUM_KV_HEADS * NUM_QO_HEADS_PER_KV +
            partial_group_offset + head_idx;
        int const o_offset =
            lse_offset * HEAD_DIM + head_partition * VAL_PER_THREAD + i;
        float weight = s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx];
        if (weight != 0.f) {
          float other_o = static_cast<float>(o_ptr[o_offset]);
          o_global += other_o * weight;
        }
      }
      output_ptr[token_pos * NUM_KV_HEADS * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 kv_head * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 head_idx * HEAD_DIM + head_partition * VAL_PER_THREAD + i] =
          (T)o_global;
    }
    __syncwarp(group_mask);
  }
}

} // namespace kernel

namespace kernel {

template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int MAX_TOKENS,
          int NUM_MERGE_SPLITS,
          int NUM_KV_CHUNKS = 1,
          int KV_CHUNK_SIZE = 256,
          int PAGE_SIZE = 4096,
          int PREFILL_THRESHOLD = -1,
          int DIRECT_UNSPLIT_PREFILL = 0>
__device__ __forceinline__ void
    merge_splitkv_persistent(void const *lse,
                             void const *o,
                             int const *qo_indptr_buffer_ptr,
                             int const *paged_kv_indptr_buffer_ptr,
                             int const *paged_kv_last_page_len_buffer_ptr,
                             int16_t request_id,
                             int merge_task_offset,
                             int merge_split_idx,
                             void *output) {
  T const *o_ptr = reinterpret_cast<T const *>(o);
  T *output_ptr = reinterpret_cast<T *>(output);
  float const *lse_ptr = reinterpret_cast<float const *>(lse);
  constexpr int THREADS_PER_TOKEN = 16;
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;
  static_assert(HEAD_DIM % THREADS_PER_TOKEN == 0);
  int const thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int const group_id = threadIdx.x / THREADS_PER_TOKEN;
  int const head_partition = thread_in_group;
  int const warp_lane = threadIdx.x & 31;
  unsigned const group_mask = (warp_lane < THREADS_PER_TOKEN) ? 0x0000ffffu
                                                              : 0xffff0000u;
  extern __shared__ char smem[];
  float *s_chunk_weights = reinterpret_cast<float *>(smem);

  int const kv_head = merge_task_offset;
  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  int const num_tokens = last_token_pos - first_token_pos;
  if (num_tokens <= 0) {
    return;
  }

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const seq_len = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[request_id];

  int request_num_chunks = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
  if constexpr (PREFILL_THRESHOLD >= 0) {
    int const packed_qo = num_tokens * NUM_QO_HEADS_PER_KV;
    if (packed_qo <= PREFILL_THRESHOLD) {
      return;
    }
    if constexpr (DIRECT_UNSPLIT_PREFILL) {
      if (seq_len <= KV_CHUNK_SIZE) {
        return;
      }
    }
    if (seq_len <= KV_CHUNK_SIZE) {
      request_num_chunks = 1;
    }
  }
  request_num_chunks = min(request_num_chunks, NUM_KV_CHUNKS);
  int const history_len = seq_len - num_tokens;

  for (int tok = merge_split_idx * num_groups + group_id;
       tok < num_tokens * NUM_QO_HEADS_PER_KV;
       tok += NUM_MERGE_SPLITS * num_groups) {
    int const token_idx = tok / NUM_QO_HEADS_PER_KV;
    int const head_idx = tok - token_idx * NUM_QO_HEADS_PER_KV;
    int token_seq_len = history_len + token_idx + 1;
    int num_chunks = min(NUM_KV_CHUNKS,
                         max(1, (token_seq_len + KV_CHUNK_SIZE - 1) /
                                    KV_CHUNK_SIZE));

    if (thread_in_group == 0) {
      float m_global = -inf;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
        int const partial_group_offset =
            (kv_idx * NUM_KV_HEADS + kv_head) * NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            (first_token_pos + token_idx) * NUM_KV_CHUNKS *
                NUM_KV_HEADS * NUM_QO_HEADS_PER_KV +
            partial_group_offset + head_idx;
        float other_m = lse_ptr[lse_offset];
        m_global = max(m_global, other_m);
      }

      float d_global = 0.f;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
        int const partial_group_offset =
            (kv_idx * NUM_KV_HEADS + kv_head) * NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            (first_token_pos + token_idx) * NUM_KV_CHUNKS *
                NUM_KV_HEADS * NUM_QO_HEADS_PER_KV +
            partial_group_offset + head_idx;
        float other_m = lse_ptr[lse_offset];
        float weight =
            (other_m == -inf || m_global == -inf) ? 0.f
                                                  : ptx_exp2(other_m - m_global);
        s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx] = weight;
        d_global += weight;
      }

      if (d_global > 0.f) {
        float inv_d = __fdividef(1.f, d_global);
#pragma unroll 1
        for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
          s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx] *= inv_d;
        }
      }
    }
    __syncwarp(group_mask);

#pragma unroll 1
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float o_global = 0.f;
#pragma unroll 1
      for (int kv_idx = 0; kv_idx < num_chunks; kv_idx++) {
        int const partial_group_offset =
            (kv_idx * NUM_KV_HEADS + kv_head) * NUM_QO_HEADS_PER_KV;
        int const lse_offset =
            (first_token_pos + token_idx) * NUM_KV_CHUNKS *
                NUM_KV_HEADS * NUM_QO_HEADS_PER_KV +
            partial_group_offset + head_idx;
        int const o_offset =
            lse_offset * HEAD_DIM + head_partition * VAL_PER_THREAD + i;
        float weight = s_chunk_weights[group_id * NUM_KV_CHUNKS + kv_idx];
        if (weight != 0.f) {
          float other_o = static_cast<float>(o_ptr[o_offset]);
          o_global += other_o * weight;
        }
      }
      output_ptr[(first_token_pos + token_idx) * NUM_KV_HEADS *
                     NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 kv_head * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 head_idx * HEAD_DIM + head_partition * VAL_PER_THREAD + i] =
          (T)o_global;
    }
    __syncwarp(group_mask);
  }
}

} // namespace kernel
