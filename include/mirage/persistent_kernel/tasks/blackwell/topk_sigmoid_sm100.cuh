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

// Warp-ballot prefix compaction of the active-expert marker array, factored
// so it can run (a) inline at the end of the single-CTA topk kernel
// (FUSE_COMPACTION=true) or (b) as a standalone kernel after a multi-CTA topk
// pass (FUSE_COMPACTION=false) — in the latter case the kernel-launch boundary
// is the global barrier that orders all marker writes before this read.
//
// Contract: mpk_active_expert_ids[0..LOCAL_EXPERTS) holds a marker per local
// expert (>=0 if active, -1 otherwise); slot [LOCAL_EXPERTS] receives the
// active count. Only warp 0 participates. Race-free even when nearly all
// experts are active (prefill): after processing chunk k the compacted write
// cursor `count` is <= 32*(k+1) = the first index chunk (k+1) will read, so
// writes never clobber a not-yet-read marker.
template <int LOCAL_EXPERTS>
__device__ __forceinline__ void
    compact_active_experts_ballot(int *mpk_active_expert_ids) {
  if (threadIdx.x < WARP_SIZE_SIGMOID) {
    int const lane = threadIdx.x;
    int count = 0;
    for (int chunk_base = 0; chunk_base < LOCAL_EXPERTS;
         chunk_base += WARP_SIZE_SIGMOID) {
      int const local_expert = chunk_base + lane;
      int const mark = (local_expert < LOCAL_EXPERTS)
                           ? mpk_active_expert_ids[local_expert]
                           : -1;
      unsigned const ballot = __ballot_sync(0xffffffff, mark >= 0);
      int const my_offset = __popc(ballot & ((1u << lane) - 1));
      if (mark >= 0) {
        mpk_active_expert_ids[count + my_offset] = local_expert;
      }
      count += __popc(ballot);
    }
    if (lane == 0) {
      mpk_active_expert_ids[LOCAL_EXPERTS] = count;
    }
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
          int TOPK_EXPERTS,
          bool FUSE_COMPACTION = true>
__device__ __forceinline__ void topk_sigmoid_task_impl(
    void *__restrict__ input_ptr, // [num_rows, NUM_EXPERTS]
    void *__restrict__ bias_ptr,  // [NUM_EXPERTS] float
    bool const *__restrict__ finished,
    void *__restrict__ output_ptr, // [num_rows, TOPK_EXPERTS]
    int const num_rows,            // compile-time MBT, used as stride for the
                                   // (LOCAL_EXPERTS, MBT) routing buffer.
    void *__restrict__ mpk_routing_indices_ptr,   // [LOCAL_EXPERTS, num_rows]
    void *__restrict__ mpk_active_expert_ids_ptr, // [LOCAL_EXPERTS + 1]
    int const start_expert,
    int const end_expert,
    float const routed_scaling_factor,
    int const num_active_rows) { // runtime active token count
                                 // (<= num_rows); compute (Phase 1+)
                                 // only iterates over [0, num_active_rows).
                                 // Phase 0 init still zeros the full
                                 // [0, num_rows) range so padded slots
                                 // in routing_indices stay 0 for the
                                 // downstream moe_permute scan. For
                                 // decode this drops the compute from
                                 // ~108 μs (full mbt=128) to ~1 μs.

  // Pointers
  T *input = static_cast<T *>(input_ptr);
  float const *bias = static_cast<float const *>(bias_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *mpk_routing_indices = static_cast<int *>(mpk_routing_indices_ptr);
  int *mpk_active_expert_ids = static_cast<int *>(mpk_active_expert_ids_ptr);

  // ---- Phase 0: Initialize routing structures ----
  // Init only [0, num_active_rows) of the routing buffer: moe_permute
  // (D5) only scans up to num_active_rows, so rows >= num_active_rows
  // are never read. Skipping their init drops decode's 128*128=16384
  // zero-stores to 128*1=128. The mpk_active_expert_ids vector still
  // gets the full LOCAL_EXPERTS reset because Phase 7's compaction
  // reads `mark[local_expert]` for every local expert.
  //
  // FUSE_COMPACTION=true (single-CTA, the decode / default path): the one CTA
  // owns the entire marker array + routing buffer, so it does the upfront
  // init here exactly as before.
  // FUSE_COMPACTION=false (multi-CTA prefill path): the marker array is
  // pre-init'd to -1 (counter 0) by the caller BEFORE launch — doing it here
  // in every CTA would race the Phase-5 marker SETs of peer CTAs (no global
  // barrier between CTAs), silently dropping active experts. Each CTA instead
  // zeroes only its own row-chunk's slice of the routing buffer inside the
  // loop below, so CTAs never touch each other's rows.
  if (FUSE_COMPACTION) {
    int const init_rows =
        (num_active_rows < num_rows) ? num_active_rows : num_rows;
    for (int expert = start_expert + threadIdx.x; expert < end_expert;
         expert += blockDim.x) {
      int const local_expert = expert - start_expert;
      if (mpk_routing_indices != nullptr) {
        for (int row = 0; row < init_rows; ++row) {
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
  // ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP — number of rows one full
  // pass of the kernel can cover. For DSv3 (NUM_EXPERTS=256, VPT=8) this is
  // 8 * 1 = 8, but `num_rows` is max_num_batched_tokens which can be >8 for
  // prefill (e.g., mbt=128). Without the outer loop below, rows 8..num_rows-1
  // would silently keep mpk_routing_indices=0 (Phase-0 init), causing the
  // group GEMM (`if (topk_idx_n > 0)` skip) to drop them — a silent MoE
  // prefill precision bug. Loop one chunk-of-ROWS_PER_CTA at a time.
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
  // Compute (Phases 1+) iterates only over active rows. Padded rows in
  // [num_active_rows, num_rows) keep the zero-init from Phase 0 — that
  // makes downstream moe_permute's `slot_1idx > 0` check correctly
  // treat them as "no routing".
  //
  // Row-chunk loop. MEGAKERNEL CONTEXT WARNING: inside the persistent
  // megakernel, blockIdx.x is the executing WORKER's physical CTA id (and
  // gridDim.x the worker count) — NOT a per-task grid coordinate. The fused
  // single-CTA variant must therefore never key its chunking on
  // blockIdx/gridDim: a task landing on worker w would silently process only
  // rows [w*ROWS_PER_CTA, w*ROWS_PER_CTA+ROWS_PER_CTA) — at decode
  // (num_active_rows=1) that means EMPTY routing whenever w != 0.
  // FUSE_COMPACTION=false keeps the grid-stride for STANDALONE
  // multi-CTA launches only; before any megakernel use it must derive a
  // virtual chunk index from task metadata instead (builder wiring for the
  // multi-CTA prefill path is still pending).
  int const chunk_base = FUSE_COMPACTION ? 0 : blockIdx.x * ROWS_PER_CTA;
  int const chunk_stride =
      FUSE_COMPACTION ? ROWS_PER_CTA : gridDim.x * ROWS_PER_CTA;
  for (int row_base = chunk_base; row_base < num_active_rows;
       row_base += chunk_stride) {
    // Multi-CTA path only: zero THIS chunk's slice of the routing buffer
    // (rows [row_base, row_hi) across all local experts) before computing it.
    // Single-CTA path did this upfront in Phase 0; here it must be per-chunk
    // so peer CTAs never race on rows they don't own. Compiled out when
    // FUSE_COMPACTION (the upfront Phase-0 zero already covered every row).
    if (!FUSE_COMPACTION && mpk_routing_indices != nullptr) {
      int const row_hi = min(row_base + ROWS_PER_CTA, num_active_rows);
      for (int le = 0; le < LOCAL_EXPERTS; ++le) {
        for (int row = row_base + threadIdx.x; row < row_hi;
             row += blockDim.x) {
          mpk_routing_indices[le * num_rows + row] = 0;
        }
      }
      __syncthreads();
    }
    int const thread_row = row_base + warp_base_row + thread_row_in_warp;
    // Warp mask: special case is for the THREADS_PER_ROW=16 / 2-rows-per-
    // warp config where the last (odd) row's upper half-warp needs masking.
    // It must be compile-time gated on ROWS_PER_WARP==2: with full-warp rows
    // (THREADS_PER_ROW=32, ROWS_PER_WARP=1, the DSv3 config) every row spans
    // all 32 lanes, so a half mask is ALWAYS wrong — and an odd compile-time
    // `num_rows` (e.g. MBT=1 in test mode) would otherwise spuriously produce
    // 0x0000ffff for the last row and break the __shfl_sync reductions (UB:
    // lanes 16..31 execute the shuffle while outside the mask).
    uint32_t const warp_mask =
        (ROWS_PER_WARP == 2 && num_rows % 2 == 1 && thread_row == num_rows - 1)
            ? 0x0000ffff
            : 0xffffffff;

    if (thread_row < num_active_rows) {

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
          output[out_idx] = should_process_row ? orig_score : 0.0f;
          if (row_is_active) {
            weight_sum += orig_score;
          }

          if (should_process_row && mpk_routing_indices != nullptr) {
            int const local_expert = expert - start_expert;
            mpk_routing_indices[local_expert * num_rows + thread_row] =
                k_idx + 1;
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
  } // end for(row_base) — close ROWS_PER_CTA loop
  __syncthreads();

  // ---- Phase 7: Compact active expert IDs ----
  // FUSE_COMPACTION=true (single-CTA): run the warp-ballot compaction inline
  // now — the preceding __syncthreads guarantees every Phase-5 marker SET is
  // visible. FUSE_COMPACTION=false (multi-CTA): skipped here; the caller runs
  // compact_active_experts_ballot in a SEPARATE kernel after this one returns,
  // because compaction must observe markers written by ALL CTAs and only the
  // kernel-launch boundary provides that global barrier.
  if (FUSE_COMPACTION && mpk_active_expert_ids != nullptr) {
    compact_active_experts_ballot<LOCAL_EXPERTS>(mpk_active_expert_ids);
  }
  // MPK signals task completion from thread 0; publish routing zeros
  // and compacted mask writes made by all CTA threads before consumers run.
  asm volatile("membar.gl;" ::: "memory");
}

} // namespace kernel
