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
#pragma once
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

// topk_reduce includes
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

// mirage includes
#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"

// ====================== TopK Sigmoid (Group-Aware) ==========================
//
// DeepSeek V3 group-aware sigmoid routing:
//   1. sigmoid(logits) -> scores
//   2. scores_biased = scores + e_score_correction_bias
//   3. Group scores: top-2 per group, sum
//   4. Select top-K groups
//   5. Mask non-selected groups, top-K experts from remainder
//   6. Gather original (unbiased) sigmoid scores for selected experts
//   7. Normalize and scale
//
// Thread layout (256 experts, bf16):
//   VPT=8, THREADS_PER_ROW=32 (full warp), ROWS_PER_WARP=1, 8 warps
//   Thread t holds experts [t*8, t*8+7]
//   Group g (32 experts) maps to threads [g*4, g*4+3]

namespace kernel {

static constexpr int WARP_SIZE_SIGMOID = 32;

// Helper: merge two sorted-descending pairs (a1>=a2, b1>=b2) into top-2
__device__ __forceinline__ void
    merge_top2(float &t1, float &t2, float o1, float o2) {
  if (o1 > t1) {
    t2 = max(t1, o2);
    t1 = o1;
  } else {
    t2 = max(t2, o1);
  }
}

template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int LOCAL_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG,
          int NUM_GROUPS,
          int TOPK_GROUP,
          int EXPERTS_PER_GROUP,
          int TOPK_EXPERTS>
__device__ __forceinline__ void topk_sigmoid_task_impl(
    void *__restrict__ input_ptr, // [num_rows, NUM_EXPERTS]
    void *__restrict__ bias_ptr,  // [NUM_EXPERTS] float
    bool const *__restrict__ finished,
    void *__restrict__ output_ptr, // [num_rows, TOPK_EXPERTS]
    int const num_rows,
    void *__restrict__ mpk_routing_indices_ptr,   // [LOCAL_EXPERTS, num_rows]
    void *__restrict__ mpk_active_expert_ids_ptr, // [LOCAL_EXPERTS + 1]
    int const start_expert,
    int const end_expert,
    float const routed_scaling_factor) {

  // Pointers
  T *input = static_cast<T *>(input_ptr);
  float const *bias = static_cast<float const *>(bias_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *mpk_routing_indices = static_cast<int *>(mpk_routing_indices_ptr);
  int *mpk_active_expert_ids = static_cast<int *>(mpk_active_expert_ids_ptr);

  // ---- Phase 0: Initialize routing structures ----
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    int const local_expert = expert - start_expert;
    if (mpk_routing_indices != nullptr) {
      for (int row = 0; row < num_rows; ++row) {
        mpk_routing_indices[local_expert * num_rows + row] = 0;
      }
    }
    if (mpk_active_expert_ids != nullptr) {
      mpk_active_expert_ids[local_expert] = -1;
    }
  }
  if (threadIdx.x == 0 && mpk_active_expert_ids != nullptr) {
    mpk_active_expert_ids[LOCAL_EXPERTS] = 0;
  }
  __syncthreads();

  // Compile-time checks
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT % ELTS_PER_LDG == 0,
                "VPT must be multiple of ELTS_PER_LDG");
  static_assert(WARP_SIZE_SIGMOID % THREADS_PER_ROW == 0,
                "THREADS_PER_ROW must divide warp size");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE_SIGMOID,
                "THREADS_PER_ROW can be at most warp size");

  // Group mapping
  static constexpr int THREADS_PER_GROUP = EXPERTS_PER_GROUP / VPT;
  static_assert(EXPERTS_PER_GROUP % VPT == 0,
                "EXPERTS_PER_GROUP must be divisible by VPT");
  static_assert(NUM_GROUPS * EXPERTS_PER_GROUP == NUM_EXPERTS,
                "NUM_GROUPS * EXPERTS_PER_GROUP must equal NUM_EXPERTS");

  // Work partitioning
  static constexpr int ELTS_PER_WARP = WARP_SIZE_SIGMOID * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "ELTS_PER_ROW must divide ELTS_PER_WARP");

  int const warp_idx = threadIdx.x / WARP_SIZE_SIGMOID;
  int const lane_idx = threadIdx.x % WARP_SIZE_SIGMOID;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;

  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  int const thread_row = warp_base_row + thread_row_in_warp;
  uint32_t const warp_mask = (num_rows % 2 == 1 && thread_row == num_rows - 1)
                                 ? 0x0000ffff
                                 : 0xffffffff;

  if (thread_row < num_rows) {

    bool const row_is_active = finished ? !finished[thread_row] : true;

    // ---- Phase 1: Load logits, apply sigmoid, load bias ----
    T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    int const thread_group_idx = lane_idx % THREADS_PER_ROW;
    int const first_elt_read_by_thread =
        thread_group_idx * (BYTES_PER_LDG / sizeof(T));
    T *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;
    T row_chunk_temp[VPT];
    AccessType *row_chunk_vec_ptr =
        reinterpret_cast<AccessType *>(&row_chunk_temp);
    AccessType *vec_thread_read_ptr =
        reinterpret_cast<AccessType *>(thread_read_ptr);

    // Vectorized loads
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    cutlass::NumericConverter<float, T> converter;

    // Compute sigmoid and biased scores
    float row_chunk[VPT];    // unbiased sigmoid scores (for final weights)
    float biased_chunk[VPT]; // sigmoid + bias (for selection)

    int const bias_offset = thread_group_idx * VPT;
    for (int ii = 0; ii < VPT; ++ii) {
      float logit = converter(row_chunk_temp[ii]);
      row_chunk_temp[ii] = static_cast<T>(0); // reset for split-k
      float sig = 1.0f / (1.0f + expf(-logit));
      row_chunk[ii] = sig;
      biased_chunk[ii] = sig + bias[bias_offset + ii];
    }

    // Write back zeros (same as softmax kernel, for split-k gate linear)
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      vec_thread_read_ptr[ii * THREADS_PER_ROW] = row_chunk_vec_ptr[ii];
    }

    // ---- Phase 2: Group top-2 reduction ----
    // Each thread computes local top-2 of its VPT biased values
    float local_top1 = biased_chunk[0];
    float local_top2 = -1e30f;
    for (int ii = 1; ii < VPT; ++ii) {
      float val = biased_chunk[ii];
      if (val > local_top1) {
        local_top2 = local_top1;
        local_top1 = val;
      } else if (val > local_top2) {
        local_top2 = val;
      }
    }

    // Reduce top-2 across THREADS_PER_GROUP threads within the group
    for (int mask = THREADS_PER_GROUP / 2; mask > 0; mask /= 2) {
      float other_top1 =
          __shfl_xor_sync(warp_mask, local_top1, mask, THREADS_PER_ROW);
      float other_top2 =
          __shfl_xor_sync(warp_mask, local_top2, mask, THREADS_PER_ROW);
      merge_top2(local_top1, local_top2, other_top1, other_top2);
    }
    float group_score = local_top1 + local_top2;

    // ---- Phase 3: Broadcast group scores and select top-K groups ----
    float all_group_scores[NUM_GROUPS];
    for (int g = 0; g < NUM_GROUPS; ++g) {
      int source_lane = g * THREADS_PER_GROUP;
      all_group_scores[g] =
          __shfl_sync(warp_mask, group_score, source_lane, THREADS_PER_ROW);
    }

    // Iterative top-K group selection
    bool group_selected[NUM_GROUPS];
    for (int g = 0; g < NUM_GROUPS; ++g) {
      group_selected[g] = false;
    }
    for (int ki = 0; ki < TOPK_GROUP; ++ki) {
      int best_g = 0;
      float best_s = -1e30f;
      for (int g = 0; g < NUM_GROUPS; ++g) {
        if (!group_selected[g] && all_group_scores[g] > best_s) {
          best_s = all_group_scores[g];
          best_g = g;
        }
      }
      group_selected[best_g] = true;
    }

    // ---- Phase 4: Mask non-selected groups ----
    int my_group = thread_group_idx / THREADS_PER_GROUP;
    if (!group_selected[my_group]) {
      for (int ii = 0; ii < VPT; ++ii) {
        biased_chunk[ii] = -10000.f;
      }
    }

    // ---- Phase 5: Top-K expert selection (same loop as softmax) ----
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float weight_sum = 0.f;

    for (int k_idx = 0; k_idx < TOPK_EXPERTS; ++k_idx) {
      // Find local argmax on biased_chunk
      float max_val = biased_chunk[0];
      int expert = start_col;
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = biased_chunk[ldg * ELTS_PER_LDG + ii];
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

      // Argmax reduce across subgroup
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max =
            __shfl_xor_sync(warp_mask, max_val, mask, THREADS_PER_ROW);
        int other_expert =
            __shfl_xor_sync(warp_mask, expert, mask, THREADS_PER_ROW);
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Gather original (unbiased) sigmoid score from owning thread
      int owning_thread = expert / VPT;
      int local_idx = expert % VPT;
      float my_score = 0.f;
      if (thread_group_idx == owning_thread) {
        my_score = row_chunk[local_idx];
      }
      float orig_score =
          __shfl_sync(warp_mask, my_score, owning_thread, THREADS_PER_ROW);

      // Write output and routing indices (one thread per subgroup writes)
      if (thread_group_idx == 0) {
        bool const node_uses_expert =
            expert >= start_expert && expert < end_expert;
        bool const should_process_row = row_is_active && node_uses_expert;
        int const out_idx = TOPK_EXPERTS * thread_row + k_idx;
        output[out_idx] = orig_score;
        weight_sum += orig_score;

        if (should_process_row && mpk_routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          mpk_routing_indices[local_expert * num_rows + thread_row] = k_idx + 1;
          if (mpk_active_expert_ids != nullptr) {
            mpk_active_expert_ids[local_expert] = local_expert;
          }
        }
      }

      // Blank out the winning value for next iteration
      if (k_idx + 1 < TOPK_EXPERTS) {
        int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        int const thread_to_clear_in_group =
            (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
        if (thread_group_idx == thread_to_clear_in_group) {
          int const offset_for_expert = expert % ELTS_PER_LDG;
          biased_chunk[ldg_group_for_expert * ELTS_PER_LDG +
                       offset_for_expert] = -10000.f;
        }
      }
    }

    // ---- Phase 6: Normalize and scale ----
    if (thread_group_idx == 0) {
      float inv_sum = 1.0f / (weight_sum + 1e-20f);
      for (int k_idx = 0; k_idx < TOPK_EXPERTS; ++k_idx) {
        int const out_idx = TOPK_EXPERTS * thread_row + k_idx;
        output[out_idx] = output[out_idx] * inv_sum * routed_scaling_factor;
      }
    }
  }
  __syncthreads();

  // ---- Phase 7: Compact active expert IDs ----
  if (mpk_active_expert_ids != nullptr) {
    for (int expert = start_expert + threadIdx.x; expert < end_expert;
         expert += blockDim.x) {
      int const local_expert = expert - start_expert;
      int const mark = mpk_active_expert_ids[local_expert];
      if (mark >= 0) {
        int const pos = atomicAdd(mpk_active_expert_ids + LOCAL_EXPERTS, 1);
        mpk_active_expert_ids[pos] = local_expert;
      }
    }
  }
}

} // namespace kernel
