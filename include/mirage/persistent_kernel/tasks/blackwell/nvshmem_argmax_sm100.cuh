/* Copyright 2026 CMU
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

#ifdef USE_NVSHMEM

namespace kernel {

template <int BATCH_SIZE,
          int NUM_PARTIAL_TASKS,
          int PARTIAL_CHUNK_SIZE,
          int VALID_VOCAB_SIZE,
          int VOCAB_OFFSET>
__device__ __forceinline__ void
    nvshmem_global_argmax_from_partials_bf16(
        void const *__restrict__ partial_value_ptr,
        void const *__restrict__ partial_index_ptr,
        void *__restrict__ scratch_value_ptr,
        void *__restrict__ scratch_index_ptr,
        void *__restrict__ output_ptr,
        void *__restrict__ _teams,
        int task_offset,
        int active_tokens) {
  bfloat16 const *__restrict__ partial_value =
      static_cast<bfloat16 const *>(partial_value_ptr);
  long long const *__restrict__ partial_index =
      static_cast<long long const *>(partial_index_ptr);
  float *__restrict__ scratch_value = static_cast<float *>(scratch_value_ptr);
  long long *__restrict__ scratch_index =
      static_cast<long long *>(scratch_index_ptr);
  long long *__restrict__ output = static_cast<long long *>(output_ptr);

  nvshmem_team_t *teams = reinterpret_cast<nvshmem_team_t *>(_teams);
  nvshmem_team_t team = teams[task_offset];
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int const team_size = teami->size;
  int const team_rank = teami->my_pe;
  int const num_active_rows = max(0, min(active_tokens, BATCH_SIZE));
  int const tidx = threadIdx.x;

#pragma unroll
  for (int batch_idx = 0; batch_idx < num_active_rows; batch_idx++) {
    float local_max = -inf;
    long long local_idx = -1;
    bfloat16 const *value_row =
        partial_value + batch_idx * NUM_PARTIAL_TASKS;
    long long const *index_row =
        partial_index + batch_idx * NUM_PARTIAL_TASKS;
    for (int i = tidx; i < NUM_PARTIAL_TASKS; i += NUM_THREADS) {
      long long relative_idx = index_row[i];
      long long candidate_idx =
          static_cast<long long>(i) * PARTIAL_CHUNK_SIZE + relative_idx;
      if (relative_idx >= 0 && candidate_idx < VALID_VOCAB_SIZE) {
        float val = static_cast<float>(value_row[i]);
        if (val > local_max) {
          local_max = val;
          local_idx = candidate_idx;
        }
      }
    }

    block_reduce_max_idx_sm100<float>(local_max, local_idx);

    if (tidx == 0) {
      scratch_value[team_rank * BATCH_SIZE + batch_idx] = local_max;
      scratch_index[team_rank * BATCH_SIZE + batch_idx] =
          local_idx >= 0 ? static_cast<long long>(VOCAB_OFFSET) + local_idx : -1;
    }
  }

  __syncthreads();
  __threadfence();
  mpkar_sync_block(team);

  if (tidx == 0) {
    for (int batch_idx = 0; batch_idx < num_active_rows; batch_idx++) {
      float best_val = -inf;
      long long best_idx = -1;
      for (int peer_idx = 0; peer_idx < team_size; peer_idx++) {
        int peer = mpkar_team_translate_pe(teami, peer_idx);
        float const *peer_values = static_cast<float const *>(
            mpkar_peer_ptr(scratch_value, peer));
        long long const *peer_indices = static_cast<long long const *>(
            mpkar_peer_ptr(scratch_index, peer));
        float val = peer_values[peer_idx * BATCH_SIZE + batch_idx];
        long long idx = peer_indices[peer_idx * BATCH_SIZE + batch_idx];
        if (val > best_val) {
          best_val = val;
          best_idx = idx;
        }
      }
      output[batch_idx] = best_idx;
    }
  }

  __syncthreads();
  __threadfence();
  mpkar_sync_block(team);
}

} // namespace kernel

#endif // USE_NVSHMEM
