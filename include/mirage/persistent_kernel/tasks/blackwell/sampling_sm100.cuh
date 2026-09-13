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

/*
 * Temperature / top-k / top-p sampling, split into a partial task over vocab
 * chunks and a reduce task, mirroring argmax_partial_sm100 / argmax_reduce.
 *
 * The partial task keeps the TOPK_MAX largest scaled logits of its chunk plus
 * the chunk's max and its exp-sum, so the reduce task can rebuild the exact
 * softmax normalizer over the whole vocabulary without a second pass over the
 * logits. The reduce task pops candidates in descending order, stops once the
 * top-k count or the top-p mass is reached, and draws from the surviving set
 * with the Gumbel-Max trick (argmax of logit/T + Gumbel noise is an exact
 * categorical draw, so no renormalization is needed).
 *
 * The union of the per-chunk top-TOPK_MAX lists contains the global
 * top-TOPK_MAX, so the result is exact whenever the kept set is that large or
 * smaller; TOPK_MAX therefore acts as an upper bound on top_k and on the
 * nucleus size.
 */

#pragma once
#include "runtime_header.h"
#include "tasks/common/utils.cuh"

#include <curand_kernel.h>
#include <cutlass/arch/barrier.h>

namespace kernel {

int constexpr SAMPLING_MAX_WARPS = 32;
// Named barriers owned by these kernels; argmax_sm100 uses 6.
int constexpr SAMPLING_BAR_ID = 7;

__device__ __forceinline__ void warp_reduce_max_idx_sampling(float &val,
                                                             int &idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_down_sync(0xffffffff, val, offset);
    int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// Block-wide max with index. Unlike argmax's helper the result is broadcast to
// every thread, so control flow that depends on it stays uniform.
__device__ __forceinline__ void block_reduce_max_idx_sampling(
    float &val, int &idx, float *scratch_val, int *scratch_idx) {
  cutlass::arch::NamedBarrier bar(NUM_THREADS, SAMPLING_BAR_ID);
  int const num_warps = NUM_THREADS / NUM_THREADS_PER_WARP;
  // warp_id() shuffles across the whole warp, so it has to be evaluated while
  // the warp is still converged.
  int const my_lane_id = lane_id();
  int const my_warp_id = warp_id();

  warp_reduce_max_idx_sampling(val, idx);
  if (my_lane_id == 0) {
    scratch_val[my_warp_id] = val;
    scratch_idx[my_warp_id] = idx;
  }
  bar.arrive_and_wait();

  float best_val = -INFINITY;
  int best_idx = -1;
  for (int w = 0; w < num_warps; w++) {
    if (scratch_val[w] > best_val) {
      best_val = scratch_val[w];
      best_idx = scratch_idx[w];
    }
  }
  // Keep the scratch alive until every thread has read it.
  bar.arrive_and_wait();
  val = best_val;
  idx = best_idx;
}

__device__ __forceinline__ float block_reduce_sum_sampling(float val,
                                                           float *scratch) {
  cutlass::arch::NamedBarrier bar(NUM_THREADS, SAMPLING_BAR_ID);
  int const num_warps = NUM_THREADS / NUM_THREADS_PER_WARP;
  int const my_lane_id = lane_id();
  int const my_warp_id = warp_id();

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  if (my_lane_id == 0) {
    scratch[my_warp_id] = val;
  }
  bar.arrive_and_wait();

  float total = 0.0f;
  for (int w = 0; w < num_warps; w++) {
    total += scratch[w];
  }
  bar.arrive_and_wait();
  return total;
}

// Gumbel noise for one vocabulary position. Keying the Philox subsequence on
// the token id (not on the candidate slot) keeps a token's noise independent
// of how the vocabulary happened to be chunked.
__device__ __forceinline__ float sampling_gumbel_noise(uint64_t philox_seed,
                                                       uint64_t philox_offset,
                                                       uint64_t token_id) {
  constexpr float kEPSILON = 1e-20f;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, token_id, philox_offset, &state);
  float u = curand_uniform(&state);
  return -__logf(-__logf(u + kEPSILON) + kEPSILON);
}

// Map a logits row to the decode step of the request that owns it. Falls back
// to `fallback_offset + batch_idx` when request metadata is unset (test_mode).
__device__ __forceinline__ unsigned long long
    sampling_philox_offset_for_row(int batch_idx,
                                   int const *steps,
                                   int const *request_ids,
                                   int const *qo_indptr,
                                   unsigned long long fallback_offset) {
  for (int r = 0; r < MPK_MAX_NUM_BATCHED_REQUESTS; ++r) {
    int const start = qo_indptr[r];
    int const end = qo_indptr[r + 1];
    if (batch_idx >= start && batch_idx < end) {
      int const rid = request_ids[r];
      if (rid >= 0) {
        return (unsigned long long)steps[rid] * MPK_MAX_NUM_BATCHED_TOKENS +
               (unsigned long long)batch_idx;
      }
      break;
    }
  }
  return fallback_offset + (unsigned long long)batch_idx;
}

/*
 * Per-chunk stage. Writes, for its chunk of the vocabulary:
 *   output_val[0 .. TOPK_MAX-1] : the TOPK_MAX largest logits, scaled by
 *                                 1/temperature, in descending order
 *   output_val[TOPK_MAX]        : the chunk max (same scaling)
 *   output_val[TOPK_MAX + 1]    : sum of exp(scaled logit - chunk max)
 *   output_idx[0 .. TOPK_MAX-1] : the matching vocabulary ids, -1 when unused
 *
 * Positions at or after VOCAB_SIZE are lm_head padding and are excluded, as in
 * argmax_partial_sm100 (#751/#752/#755).
 */
template <typename T,
          int BATCH_SIZE,
          int CHUNK_SIZE,
          int NUM_PARTIAL_TASKS,
          int TOPK_MAX,
          int VOCAB_SIZE = CHUNK_SIZE *NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    sampling_partial_sm100_kernel(void const *__restrict__ input_ptr,
                                  void *__restrict__ output_val_ptr,
                                  void *__restrict__ output_idx_ptr,
                                  int num_active_tokens,
                                  float inv_temperature,
                                  int chunk_start) {
  static_assert(TOPK_MAX > 0 && TOPK_MAX <= NUM_THREADS,
                "TOPK_MAX must fit one candidate per thread in the reduce "
                "stage's Gumbel draw");
  // Whole chunk lives in smem. Qwen3 with 128 workers → CHUNK=1200 floats.
  static_assert(sizeof(float) * CHUNK_SIZE +
                        sizeof(float) * SAMPLING_MAX_WARPS +
                        sizeof(int) * SAMPLING_MAX_WARPS + 256 <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "sampling_partial smem exceeds the megakernel dynamic budget");

  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  float *__restrict__ output_val = static_cast<float *>(output_val_ptr);
  long long *__restrict__ output_idx = static_cast<long long *>(output_idx_ptr);

  int const val_stride = NUM_PARTIAL_TASKS * (TOPK_MAX + 2);
  int const idx_stride = NUM_PARTIAL_TASKS * TOPK_MAX;
  int const valid_len = max(0, min(CHUNK_SIZE, VOCAB_SIZE - chunk_start));

  extern __shared__ char smem[];
  char *base = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
  float *scratch_val = reinterpret_cast<float *>(base);
  int *scratch_idx = reinterpret_cast<int *>(scratch_val + SAMPLING_MAX_WARPS);
  float *chunk = reinterpret_cast<float *>(scratch_idx + SAMPLING_MAX_WARPS);

  cutlass::arch::NamedBarrier bar(NUM_THREADS, SAMPLING_BAR_ID);
  int const tidx = threadIdx.x;
  if (tidx >= NUM_THREADS) {
    return;
  }

  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    for (int i = tidx; i < CHUNK_SIZE; i += NUM_THREADS) {
      chunk[i] =
          i < valid_len
              ? static_cast<float>(
                    input[i + batch_idx * CHUNK_SIZE * NUM_PARTIAL_TASKS]) *
                    inv_temperature
              : -INFINITY;
    }
    bar.arrive_and_wait();

    float chunk_max = -INFINITY;
    int chunk_max_pos = -1;
    for (int i = tidx; i < CHUNK_SIZE; i += NUM_THREADS) {
      if (chunk[i] > chunk_max) {
        chunk_max = chunk[i];
        chunk_max_pos = i;
      }
    }
    block_reduce_max_idx_sampling(
        chunk_max, chunk_max_pos, scratch_val, scratch_idx);

    float sum_exp = 0.0f;
    if (chunk_max_pos >= 0) {
      for (int i = tidx; i < valid_len; i += NUM_THREADS) {
        sum_exp += __expf(chunk[i] - chunk_max);
      }
    }
    sum_exp = block_reduce_sum_sampling(sum_exp, scratch_val);

    if (tidx == 0) {
      output_val[batch_idx * val_stride + TOPK_MAX] = chunk_max;
      output_val[batch_idx * val_stride + TOPK_MAX + 1] =
          chunk_max_pos >= 0 ? sum_exp : 0.0f;
    }

    for (int j = 0; j < TOPK_MAX; j++) {
      float val = -INFINITY;
      int pos = -1;
      for (int i = tidx; i < CHUNK_SIZE; i += NUM_THREADS) {
        if (chunk[i] > val) {
          val = chunk[i];
          pos = i;
        }
      }
      block_reduce_max_idx_sampling(val, pos, scratch_val, scratch_idx);

      if (tidx == 0) {
        output_val[batch_idx * val_stride + j] = val;
        output_idx[batch_idx * idx_stride + j] =
            pos >= 0 ? static_cast<long long>(chunk_start + pos) : -1LL;
        if (pos >= 0) {
          chunk[pos] = -INFINITY;
        }
      }
      bar.arrive_and_wait();
    }
  }
}

/*
 * Reduce stage. Merges the per-chunk candidates, applies top-k and top-p, and
 * draws one token per active row.
 *
 * `greedy` (temperature <= 0) skips the noise and returns the argmax, which
 * keeps a single compiled graph usable for both deterministic and sampled
 * decoding.
 */
template <typename T, int BATCH_SIZE, int NUM_PARTIAL_TASKS, int TOPK_MAX>
__device__ __forceinline__ void
    sampling_reduce_sm100_kernel(void const *__restrict__ input_val_ptr,
                                 void const *__restrict__ input_idx_ptr,
                                 void *__restrict__ final_output_ptr,
                                 int num_active_tokens,
                                 float top_p,
                                 int top_k,
                                 bool greedy,
                                 unsigned long long philox_seed,
                                 unsigned long long philox_fallback_offset,
                                 int const *steps,
                                 int const *request_ids,
                                 int const *qo_indptr) {
  int constexpr NUM_CANDIDATES = NUM_PARTIAL_TASKS * TOPK_MAX;
  static_assert(TOPK_MAX <= NUM_THREADS,
                "one thread per surviving candidate is assumed below");
  static_assert(sizeof(float) * (2 * NUM_CANDIDATES + TOPK_MAX) +
                        sizeof(int) * (NUM_CANDIDATES + TOPK_MAX) +
                        sizeof(float) * SAMPLING_MAX_WARPS +
                        sizeof(int) * SAMPLING_MAX_WARPS + 256 <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "sampling_reduce smem exceeds the megakernel dynamic budget");

  float const *__restrict__ partial_val =
      static_cast<float const *>(input_val_ptr);
  long long const *__restrict__ partial_idx =
      static_cast<long long const *>(input_idx_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);

  int const val_stride = NUM_PARTIAL_TASKS * (TOPK_MAX + 2);
  int const idx_stride = NUM_PARTIAL_TASKS * TOPK_MAX;

  extern __shared__ char smem[];
  char *base = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
  float *scratch_val = reinterpret_cast<float *>(base);
  int *scratch_idx = reinterpret_cast<int *>(scratch_val + SAMPLING_MAX_WARPS);
  float *cand_val = reinterpret_cast<float *>(scratch_idx + SAMPLING_MAX_WARPS);
  int *cand_idx = reinterpret_cast<int *>(cand_val + NUM_CANDIDATES);
  float *sel_val = reinterpret_cast<float *>(cand_idx + NUM_CANDIDATES);
  int *sel_idx = reinterpret_cast<int *>(sel_val + TOPK_MAX);

  cutlass::arch::NamedBarrier bar(NUM_THREADS, SAMPLING_BAR_ID);
  int const tidx = threadIdx.x;
  if (tidx >= NUM_THREADS) {
    return;
  }

  int const k_limit = (top_k > 0 && top_k < TOPK_MAX) ? top_k : TOPK_MAX;

  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    for (int i = tidx; i < NUM_CANDIDATES; i += NUM_THREADS) {
      int const chunk = i / TOPK_MAX;
      int const slot = i % TOPK_MAX;
      cand_val[i] =
          partial_val[batch_idx * val_stride + chunk * (TOPK_MAX + 2) + slot];
      cand_idx[i] = static_cast<int>(partial_idx[batch_idx * idx_stride + i]);
    }

    // Rebuild the exact softmax normalizer from the per-chunk (max, sum) pairs.
    float global_max = -INFINITY;
    int global_max_pos = -1;
    for (int c = tidx; c < NUM_PARTIAL_TASKS; c += NUM_THREADS) {
      float const chunk_max =
          partial_val[batch_idx * val_stride + c * (TOPK_MAX + 2) + TOPK_MAX];
      if (chunk_max > global_max) {
        global_max = chunk_max;
        global_max_pos = c;
      }
    }
    block_reduce_max_idx_sampling(
        global_max, global_max_pos, scratch_val, scratch_idx);

    float total = 0.0f;
    for (int c = tidx; c < NUM_PARTIAL_TASKS; c += NUM_THREADS) {
      float const chunk_max =
          partial_val[batch_idx * val_stride + c * (TOPK_MAX + 2) + TOPK_MAX];
      float const chunk_sum = partial_val[batch_idx * val_stride +
                                          c * (TOPK_MAX + 2) + TOPK_MAX + 1];
      if (chunk_max > -INFINITY) {
        total += chunk_sum * __expf(chunk_max - global_max);
      }
    }
    total = block_reduce_sum_sampling(total, scratch_val);

    // Pop candidates in descending order until top-k or the top-p mass is hit.
    // Every thread sees the same reduction result, so the loop bounds stay
    // uniform across the block.
    float const mass_target = top_p < 1.0f ? top_p * total : INFINITY;
    float mass = 0.0f;
    int num_selected = 0;
    for (int j = 0; j < k_limit; j++) {
      float val = -INFINITY;
      int pos = -1;
      for (int i = tidx; i < NUM_CANDIDATES; i += NUM_THREADS) {
        if (cand_val[i] > val) {
          val = cand_val[i];
          pos = i;
        }
      }
      block_reduce_max_idx_sampling(val, pos, scratch_val, scratch_idx);
      if (pos < 0 || val == -INFINITY) {
        break;
      }
      if (tidx == 0) {
        sel_val[j] = val;
        sel_idx[j] = cand_idx[pos];
        cand_val[pos] = -INFINITY;
      }
      mass += __expf(val - global_max);
      num_selected = j + 1;
      bar.arrive_and_wait();
      if (mass >= mass_target) {
        break;
      }
    }

    int winner = num_selected > 0 ? 0 : -1;
    if (!greedy && num_selected > 0) {
      unsigned long long const row_offset = sampling_philox_offset_for_row(
          batch_idx, steps, request_ids, qo_indptr, philox_fallback_offset);
      float noisy = -INFINITY;
      int pos = -1;
      if (tidx < num_selected) {
        noisy = sel_val[tidx] +
                sampling_gumbel_noise(philox_seed,
                                      row_offset,
                                      static_cast<uint64_t>(sel_idx[tidx]));
        pos = tidx;
      }
      block_reduce_max_idx_sampling(noisy, pos, scratch_val, scratch_idx);
      winner = pos;
    }

    if (tidx == 0) {
      final_output[batch_idx] =
          winner >= 0 ? static_cast<long long>(sel_idx[winner]) : -1LL;
    }
    bar.arrive_and_wait();
  }
}

} // namespace kernel
