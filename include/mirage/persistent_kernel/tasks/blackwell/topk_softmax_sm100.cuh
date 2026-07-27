#pragma once
#include <cstdio>
#include <iostream>

// Use Thrust to handle host/device allocations
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

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

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k. 3) This implementation assumes 8 warps are being used.
*/

namespace kernel {

static constexpr int WARP_SIZE = 32;

// ====================== Fused TopK softmax kernel
// =============================== This kernel fuses the softmax, max and argmax
// into a single kernel. Block size is strictly 256 (8 warps): dim3
// block(WARP_SIZE*WARPS_PER_CTA, 1, 1)
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __forceinline__ void topk_softmax_task_impl(
    void *__restrict__ input_ptr, // [num_rows, NUM_EXPERTS]
    bool const *__restrict__ finished,
    void *__restrict__ output_ptr, // [num_rows, k]
    int const num_rows,
    int const k,
    void *__restrict__ mpk_routing_indices_ptr, // [NUM_EXPERTS, num_rows] laid
                                                // out as expert-major: expert *
                                                // num_rows
                                                // + row
    void *__restrict__ mpk_active_expert_ids_ptr, // [NUM_EXPERTS + 1] last
                                                  // element stores num active
                                                  // experts
    int const start_expert,
    int const end_expert,
    bool const renormalize,
    // Qwen3.5 (M2-I7 / probe P5): HF's Qwen3_5MoeTopKRouter ends with
    // `router_top_value.to(router_logits.dtype)`, i.e. the renormalized weights
    // that reach the combine are bf16, not fp32 (oracle
    // moe*.topk_renorm_weights). Measured effect of the difference at the
    // combine boundary: 1.6e-3 frob-rel, the same order as the combine's own
    // bf16 output-rounding floor (p5_router_semantics.json section E).
    // DeepSeek-V3's reference keeps fp32 weights, so this defaults OFF and the
    // generated code is unchanged for every existing caller.
    bool const round_weights_to_output_dtype = false,
    // M3-I8: how many of `num_rows` carry a LIVE token this iteration.
    //
    // `num_rows` is the COMPILE-TIME `max_num_batched_tokens` (16 for the
    // Qwen3.5 build), while `prepare_next_batch` packs only
    // `qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` live tokens into rows
    // [0, live) and leaves rows [live, num_rows) holding the previous
    // iteration's residue (attention/GDN write per REQUEST slot, so nothing
    // refreshes them). Those padding rows are still routed: each contributes
    // its own top-k marks, and every expert they touch becomes an ACTIVATED
    // group that the grouped GEMM then streams weights for and discards.
    // Measured by M3-I1: 56.4 activated groups per layer at bs1, where a
    // single top-8 token needs 8.
    //
    // Gating the MARKING (`mpk_routing_indices` / `mpk_active_expert_ids`)
    // right-sizes the group set. It deliberately does NOT gate the row read
    // (the input-buffer zeroing that lets a split-k gate linear accumulate
    // must still cover every row) and does NOT gate the top-k weight write, so
    // only the two grouped-GEMM consumers see any difference.
    //
    // <= 0, or a value the task cannot honour, means "no gating" -- the
    // pre-M3-I8 behaviour -- so a caller that has no live-row count (test
    // mode, a single-layer harness) is unaffected.
    int const num_active_rows = -1) {
  // Pointers
  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *mpk_routing_indices = static_cast<int *>(mpk_routing_indices_ptr);
  int *mpk_active_expert_ids = static_cast<int *>(mpk_active_expert_ids_ptr);
  // initialize routing indices to 0; active-id marks to -1; count to 0
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (mpk_routing_indices != nullptr) {
      for (int row = 0; row < num_rows; ++row) {
        mpk_routing_indices[expert * num_rows + row] = 0;
      }
    }
    if (mpk_active_expert_ids != nullptr) {
      mpk_active_expert_ids[expert - start_expert] = -1;
    }
  }
  // Thread 0 always exists, unlike threadIdx.x == NUM_EXPERTS when
  // NUM_EXPERTS == blockDim.x == 256.
  if (threadIdx.x == 0 && mpk_active_expert_ids != nullptr) {
    mpk_active_expert_ids[NUM_EXPERTS] = 0;
  }
  __syncthreads();
  // Compile-time checks
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW =
      ELTS_PER_ROW / VPT; // subgroup size in a warp
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");
  static_assert(THREADS_PER_ROW == WARP_SIZE ||
                    THREADS_PER_ROW == WARP_SIZE / 2,
                "This kernel only supports THREADS_PER_ROW of 16 or 32");

  // Work partitioning
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP =
      ELTS_PER_WARP / ELTS_PER_ROW; // rows each warp processes
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  int const warp_idx = threadIdx.x / WARP_SIZE;
  int const lane_idx = threadIdx.x % WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;

  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;

  // Rows one pass of this block covers. `thread_row` is derived from threadIdx
  // alone, so before M3-I5b the kernel simply DROPPED every row past this
  // bound (`if (thread_row < num_rows)` with no loop around it) -- the silent
  // 16-row router cap M2-I9 root-caused. The row-tile loop below repeats the
  // same pass at row_tile_base = 0, ROWS_PER_CTA, 2*ROWS_PER_CTA, ... so any
  // num_rows is routed.
  //
  // BIT-EXACT FOR num_rows <= ROWS_PER_CTA BY CONSTRUCTION:
  //   (1) the trip count is ceil(num_rows / ROWS_PER_CTA) = 1, so
  //       row_tile_base is identically 0 and `thread_row` evaluates to the
  //       pre-change expression `warp_base_row + thread_row_in_warp`;
  //   (2) the loop body is the pre-change body verbatim (only re-indented) --
  //       same operations, same operands, same order, same shuffle masks;
  //   (3) `num_rows` arrives as an integer LITERAL from the generated call
  //       (task_register.cc emits `batch_size`), so the loop constant-folds
  //       away entirely and the emitted SASS is unchanged.
  // Above the old cap, nothing is reduced, accumulated or communicated ACROSS
  // tiles: every reduction (max, sum, argmax) is a shuffle inside one row's
  // sub-group, and each tile writes a DISJOINT row slice of `output` /
  // `mpk_routing_indices`. The only cross-tile state is the idempotent
  // `mpk_active_expert_ids[local_expert] = local_expert` mark, which is
  // order-independent by construction; it is compacted after the loop, behind
  // the same single __syncthreads() as before.
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Rows at or above this index are padding for THIS iteration; they are read
  // (and zeroed) like every other row, but they do not activate expert groups.
  int const live_rows = (num_active_rows > 0 && num_active_rows < num_rows)
                            ? num_active_rows
                            : num_rows;

  for (int row_tile_base = 0; row_tile_base < num_rows;
       row_tile_base += ROWS_PER_CTA) {
    int const thread_row = row_tile_base + warp_base_row + thread_row_in_warp;
    uint32_t warp_mask = 0xffffffffu;
    if constexpr (THREADS_PER_ROW != WARP_SIZE) {
      constexpr uint32_t subgroup_mask = (1u << THREADS_PER_ROW) - 1u;
      // The final warp can contain a single live sub-group when num_rows is
      // odd. Restrict the shuffle mask to that sub-group instead of
      // hard-coding the lower 16 lanes. Expressed in `thread_row` / `num_rows`
      // only, so it is the LAST TILE's partial warp that gets restricted.
      if ((num_rows % ROWS_PER_WARP) != 0 && thread_row == num_rows - 1) {
        warp_mask = subgroup_mask << (thread_row_in_warp * THREADS_PER_ROW);
      }
    }

    if (thread_row < num_rows) {

      bool const row_is_active =
          (finished ? !finished[thread_row] : true) && (thread_row < live_rows);

      // Compute per-thread read pointers
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

      // Vectorized loads across the row
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
      }

      cutlass::NumericConverter<float, T> converter;

      float row_chunk[VPT];
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = converter(row_chunk_temp[ii]);
        row_chunk_temp[ii] =
            static_cast<T>(0); // reset input buffer to 0 for split-k gate linear
      }

      // reset input buffer to 0 for split-k gate linear
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        vec_thread_read_ptr[ii * THREADS_PER_ROW] = row_chunk_vec_ptr[ii];
      }

      // Max reduction within subgroup
      float thread_max = row_chunk[0];
      for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
      }
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other =
            __shfl_xor_sync(warp_mask, thread_max, mask, THREADS_PER_ROW);
        thread_max = max(thread_max, other);
      }

      // Softmax numerator and sum within subgroup
      float row_sum = 0.f;
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
      }
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(warp_mask, row_sum, mask, THREADS_PER_ROW);
      }

      float const inv_row_sum = 1.f / row_sum;
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * inv_row_sum;
      }

      // Fused Top-K selection within subgroup
      int start_col = first_elt_read_by_thread;
      static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
      float row_sum_for_renormalize = 0.f;

      for (int k_idx = 0; k_idx < k; ++k_idx) {
        float max_val = row_chunk[0];
        int expert = start_col;
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
             ++ldg, col += COLS_PER_GROUP_LDG) {
          for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
            float val = row_chunk[ldg * ELTS_PER_LDG + ii];
            if (val > max_val) {
              max_val = val;
              expert = col + ii;
            }
          }
        }

        // Argmax reduce across subgroup with index tie-breaker (prefer lower
        // index)
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

        // Write out the selected top-k value/index (one thread per subgroup
        // writes)
        if (thread_group_idx == 0) {
          bool const node_uses_expert =
              expert >= start_expert && expert < end_expert;
          bool const should_process_row = row_is_active && node_uses_expert;
          int const out_idx = k * thread_row + k_idx;
          output[out_idx] = max_val;
          // indices[out_idx] =
          //     should_process_row ? (expert - start_expert) : NUM_EXPERTS;
          row_sum_for_renormalize += max_val;
          // Optionally populate MPK routing structures
          if (should_process_row && mpk_routing_indices != nullptr) {
            int const local_expert = expert - start_expert;
            // Write 1-based rank into routing indices; stride by num_rows per
            // expert
            mpk_routing_indices[local_expert * num_rows + thread_row] = k_idx + 1;
            // Sparse mark expert as active; idempotent without atomics
            if (mpk_active_expert_ids != nullptr) {
              mpk_active_expert_ids[local_expert] = local_expert;
            }
          }
        }

        // Blank out the winning value for the next iteration
        if (k_idx + 1 < k) {
          int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
          int const thread_to_clear_in_group =
              (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
          if (thread_group_idx == thread_to_clear_in_group) {
            int const offset_for_expert = expert % ELTS_PER_LDG;
            row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
                -10000.f;
          }
        }
      }

      // Optional renormalization of top-k weights
      if (renormalize && thread_group_idx == 0) {
        cutlass::NumericConverter<T, float> to_output_dtype;
        float inv = 1.f / row_sum_for_renormalize;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
          int const out_idx = k * thread_row + k_idx;
          float w = output[out_idx] * inv;
          output[out_idx] =
              round_weights_to_output_dtype ? converter(to_output_dtype(w)) : w;
        }
      }
    }
  }
  __syncthreads();
  // ---- Compact the marks into a DENSE, ASCENDING list and count (M3-I5c) ----
  //
  // The pre-M3-I5c body was an in-place read-then-scatter with no barrier
  // between a thread's read of its own mark and other threads' compacted
  // writes:
  //
  //     int const mark = mpk_active_expert_ids[local_expert];   // read slot j
  //     if (mark >= 0) {
  //       int const pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1);
  //       mpk_active_expert_ids[pos] = expert;                  // write slot pos
  //     }
  //
  // Compacted entries land in slots [0, n_active), which ALIAS the marks of
  // experts [0, n_active), and nothing orders thread j's read of slot j against
  // another thread's write of slot j. Two independent defects:
  //
  //  (1) RACE, present even at blockDim.x == NUM_EXPERTS (one pass per thread).
  //      Every scatter stores a non-negative id, so the corruption is one-sided:
  //      an INACTIVE expert j whose slot was overwritten passes `mark >= 0` and
  //      appends ITSELF (`expert`, note: not `mark`) to the list. The set gains
  //      phantom experts and the count inflates; an active expert is never lost,
  //      because no scatter ever stores a negative value. With enough phantoms
  //      `pos` reaches NUM_EXPERTS and clobbers the counter itself.
  //  (2) GUARANTEED miscount when blockDim.x < NUM_EXPERTS and the grid-stride
  //      loop makes more than one pass: a thread's own pass-p scatter can land
  //      on a slot it reads in a later pass. That is arithmetic, not a
  //      scheduling accident (found by M3-I9b). No shipped graph launches this
  //      router with blockDim.x < NUM_EXPERTS today, so the single-pass shape
  //      was hiding defect (2) entirely and merely thinning defect (1).
  //
  // Replacement: a barrier-separated PREFIX-COUNT compaction, one tile of
  // blockDim.x experts at a time, carrying the running base in a register.
  //
  //     base_t = #active in [0, t*B)              (block-uniform, in-register)
  //     rank   = base_t + #{ j in [t*B, local_expert) : mark[j] >= 0 }
  //     mpk_active_expert_ids[rank] = expert
  //
  // Race-freedom, with no assumption about warp size, blockDim.x, NUM_EXPERTS,
  // the number of active experts, or how many row tiles produced the marks:
  //   * tile t READS only slots [t*B, min((t+1)*B, n_local)) -- its own marks;
  //   * every write of tile t targets a slot < base_{t+1} <= min((t+1)*B,
  //     n_local), because at most one active expert exists per slot. So tile t
  //     can never touch a LATER tile's marks; and tile t+1's writes cannot race
  //     tile t's reads either, since a thread reaching tile t+1's barrier
  //     implies every thread already passed tile t's barrier and therefore
  //     finished tile t's reads;
  //   * the one __syncthreads() inside the tile separates that tile's reads from
  //     that tile's writes -- the only remaining overlap;
  //   * that barrier is reached by every thread: `mpk_active_expert_ids`,
  //     `start_expert`, `end_expert` and `blockDim.x` are block-uniform, so the
  //     trip count is uniform and no thread can skip it.
  //   * the count slot (NUM_EXPERTS) is written once, by thread 0, after the
  //     loop; compacted entries only ever occupy slots < n_local <= NUM_EXPERTS,
  //     so that store needs no barrier of its own.
  //
  // DETERMINISM: `rank` is a pure prefix count over the mark array, so the list
  // comes out strictly ascending in expert id under every schedule. The
  // atomicAdd -- the only source of run-to-run permutation in this task -- is
  // gone. The SET is unchanged (a mark is written iff some row's top-k selected
  // that expert, exactly as before), which is what the grouped-GEMM consumers
  // key on.
  //
  // COST: n_local L1-resident broadcast loads per thread (256 at the shipped
  // shape). This task is 0.085% of measured per-step worker time (564 us of
  // 665 ms, demo/qwen3_5/accept/opt/pertask_by_bs.csv). No shared memory is
  // used, so the megakernel's smem budget and occupancy are untouched.
  if (mpk_active_expert_ids != nullptr) {
    int const num_local_experts = end_expert - start_expert;
    int const block_size = static_cast<int>(blockDim.x);
    int base = 0; // #active strictly below this tile; same in every thread
    for (int tile_base = 0; tile_base < num_local_experts;
         tile_base += block_size) {
      int const tile_end = (tile_base + block_size < num_local_experts)
                               ? (tile_base + block_size)
                               : num_local_experts;
      int const local_expert = tile_base + static_cast<int>(threadIdx.x);
      bool is_active = false;
      int rank_in_tile = 0;
      int tile_count = 0;
      for (int j = tile_base; j < tile_end; ++j) {
        if (mpk_active_expert_ids[j] >= 0) {
          ++tile_count;
          if (j < local_expert) {
            ++rank_in_tile;
          }
          if (j == local_expert) {
            is_active = true;
          }
        }
      }
      __syncthreads(); // every read of this tile's marks precedes every write
      if (is_active) {
        mpk_active_expert_ids[base + rank_in_tile] =
            start_expert + local_expert;
      }
      base += tile_count;
    }
    if (threadIdx.x == 0) {
      mpk_active_expert_ids[NUM_EXPERTS] = base;
    }
  }
}

namespace detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                "");
  static constexpr int
      VECs_PER_THREAD = (EXPERTS / (ELTS_PER_LDG * WARP_SIZE)) > 0
                            ? (EXPERTS / (ELTS_PER_LDG * WARP_SIZE))
                            : 1;
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / (EXPERTS / VPT);
};
} // namespace detail

} // namespace kernel
