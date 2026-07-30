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
// ---------------------------------------------------------------------------
// M4-I6 (2026-07-30): body replaced by the ferret MoE-router winner, tag v013
// ("boundary-warp converged padding compute") of ferret/workspace5, whose
// frozen `golden` block was byte-identical to the M3-I5b/M3-I5c/M3-I8 body this
// replaces (413 lines, verified before import). Standalone: 141.7 / 150.5 /
// 145.6 / 170.7 / 170.3 % of the FlashInfer TRT-LLM routing kernel's throughput
// at N_LIVE 1/2/4/8/16 (min_ratio 1.417).
//
// The v-numbered comments below name the change each block carries. The one
// SEMANTIC change, called out because it is the only place the bytes move:
// rows at or above `num_active_rows` (PADDING rows) now write deterministic
// ZEROS into their `output` (top-k weight) slots instead of the softmax values
// golden computed from their residue logits. Everything else -- live rows'
// weights, the whole of `mpk_routing_indices`, the whole of
// `mpk_active_expert_ids`, and the zeroed input buffer -- is bit-identical to
// the body this replaces, on every instantiation. See the v007 block for the
// consumer audit; `live_rows == num_rows` whenever `num_active_rows` is -1, so
// every caller that does not opt into M3-I8 gating (test mode, single-layer
// harnesses, the 128-expert Qwen3-30B-A3B instantiation) keeps golden's bytes
// exactly.
// ---------------------------------------------------------------------------
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

  // (v004) PROLOGUE: issue tile-0's vectorized logits load BEFORE the
  // routing-indices zero-init loop and its __syncthreads(), so the cold-DRAM
  // round trip overlaps the init stores and the barrier wait instead of
  // sitting fully exposed after them. Bit-exact: input_ptr does not alias
  // the two output tensors, the same bytes are read from the same addresses
  // by the same lanes, and each lane's zeroing store of its own chunk
  // happens after its own convert exactly as before (program order per
  // lane). Later row tiles (num_rows > ROWS_PER_CTA instantiations) reload
  // inside the loop as before, so the shared template stays general.
  int const thread_group_idx = lane_idx % THREADS_PER_ROW;
  int const first_elt_read_by_thread =
      thread_group_idx * (BYTES_PER_LDG / sizeof(T));
  using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;
  T row_chunk_temp[VPT];
  AccessType *row_chunk_vec_ptr =
      reinterpret_cast<AccessType *>(&row_chunk_temp);
  // (v008) live_rows hoisted above the prologue load (was defined just
  // before the row-tile loop) so the load gate below can use it. Rows at
  // or above this index are padding for THIS iteration; since v007/v008
  // they are neither computed NOR read -- only their zeroing store-back
  // remains (the load-bearing split-K gate-linear invariant needs the
  // FINAL bytes to be 0, not a read-then-write round trip).
  int const live_rows = (num_active_rows > 0 && num_active_rows < num_rows)
                            ? num_active_rows
                            : num_rows;
  {
    int const tile0_row = warp_base_row + thread_row_in_warp;
    // (v008) padding rows skip the global LOAD entirely (gate tightened
    // from num_rows to live_rows); their row_chunk_temp is zero-filled at
    // the store-back site inside the row loop instead. live_rows ==
    // num_rows in ungated instantiations, so this constant-folds back to
    // the old gate there.
    if (tile0_row < live_rows) {
      AccessType *tile0_read_ptr = reinterpret_cast<AccessType *>(
          input + tile0_row * NUM_EXPERTS + first_elt_read_by_thread);
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = tile0_read_ptr[ii * THREADS_PER_ROW];
      }
    }
  }
  // initialize routing indices to 0; active-id marks to -1; count to 0
  // (v003) Each expert's row stripe [expert*num_rows, (expert+1)*num_rows)
  // is contiguous int32; zero it with 128-bit stores when num_rows is a
  // multiple of 4 (stripe start is then 16B-aligned for any 16B-aligned
  // base -- checked once). Pure zero-fill: written bytes are identical to
  // the scalar loop, so this is trivially bit-exact; scalar fallback keeps
  // the shared template general for any num_rows/base.
  bool const vec_zero_ok =
      ((num_rows & 3) == 0) &&
      ((reinterpret_cast<uintptr_t>(mpk_routing_indices_ptr) & 15) == 0);
  // (v006) Shared-memory shadow bitmask of active experts, one bit per local
  // expert. Set by an atomicOr at the SAME guarded sites as the existing
  // global mpk_active_expert_ids mark stores (which are kept, unchanged, for
  // bit-exact final array contents); read ONLY by the compaction below,
  // replacing its NUM_EXPERTS sequential global-memory active-test loads per
  // thread (the word IS the ballot).
  //
  // STORAGE CLASS: static `__shared__`, as in the benchmarked v013 kernel, NOT
  // the megakernel's dynamic arena. gdn_recurrent_sm100.cuh:635-637 records the
  // opposite convention ("shared memory comes from the megakernel's dynamic
  // arena; the standalone kernel used static __shared__, which would blow the
  // worker's static budget"), and M4-I6 tried the arena first -- it FAILS,
  // loudly and immediately, because this task has THREE standalone launchers
  // that pass zero dynamic shared memory:
  //   sm100_moe/runtime_kernel_wrapper_sm100.cu:101              <<<g, b, 0>>>
  //   sm100_moe_sigmoid/runtime_kernel_wrapper_sm100.cu          (sibling)
  //   sm100_moe_block_qwen35/runtime_kernel_wrapper_moe_block.cu (oracle test)
  // With `extern __shared__` all sixteen gate-1 cells died with
  // cudaErrorIllegalAddress. Taking the arena would make the task's contract
  // "callers must supply >= 32 B of dynamic smem", enforced only by an illegal
  // access -- so the arena convention is right for the big consumers (GDN's
  // kilobytes) and wrong here.
  //
  // The static cost is bounded and inside an existing allowance: 32 bytes at 256
  // experts, 16 at 128, against WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 6
  // KiB (runtime_header.h:30), which is subtracted from
  // MAX_DYNAMIC_SHARED_MEMORY_SIZE precisely so the worker and its inlined task
  // bodies can hold static smem. Static task smem is already shipped elsewhere
  // (softmax_gather_sm100.cuh:62,88; mla_decode_sm100.cuh:92-93). M4-I6 gate 2
  // reads the generated megakernel TU's smem/register/spill lines before and
  // after to keep that claim measured rather than asserted.
  //
  // Visibility needs no new barrier: the zeroing here precedes the tile-0
  // init-visibility __syncthreads (either site), every atomicOr precedes the
  // pre-compaction __syncthreads, which is CTA-scope ordering enough for warp
  // 0's reads.
  constexpr int NUM_MASK_WORDS = (NUM_EXPERTS + 31) / 32;
  __shared__ uint32_t active_bits[NUM_MASK_WORDS];
  // (v010) ALL init zero/reset work moved OFF warp 0 (threads 32..255,
  // stride blockDim.x-32): warp 0's critical chain (gating load -> softmax
  // -> top-k, ~2.8us at bs1) previously executed its 32-stripe share of
  // the init loop serially BEFORE that chain; DIAG_NOINIT measured the
  // loop at ~0.29us exposed (bs1-8). Repartitioning hides the init under
  // warp 0's chain. Bit-exact: identical zero/reset bytes, only the
  // writing threads change, and the SAME tile-0 CTA-wide __syncthreads
  // still orders every init store before every rank/mark write (the
  // happens-before edge is per-CTA, not per-thread). Warp-uniform guard;
  // blockDim.x is strictly 256 per the task contract (see header comment).
  int const init_t = (int)threadIdx.x - 32;
  if (init_t >= 0) {
    for (int w = init_t; w < NUM_MASK_WORDS; w += blockDim.x - 32) {
      active_bits[w] = 0u;
    }
  }
  if (init_t >= 0) {
    for (int expert = start_expert + init_t; expert < end_expert;
         expert += blockDim.x - 32) {
      if (mpk_routing_indices != nullptr) {
        int *stripe = mpk_routing_indices + expert * num_rows;
        if (vec_zero_ok) {
          int4 *stripe4 = reinterpret_cast<int4 *>(stripe);
          for (int row4 = 0; row4 < num_rows / 4; ++row4) {
            stripe4[row4] = make_int4(0, 0, 0, 0);
          }
        } else {
          for (int row = 0; row < num_rows; ++row) {
            stripe[row] = 0;
          }
        }
      }
    }
    // (v011) Mark reset split OUT of the stride-224 expert loop above into
    // its own int4-vectorized loop: 256 scalar 4-byte stores (one per
    // expert-iteration) become 64 16-byte stores, shrinking warps 1-7's
    // remaining instruction count (post-v010 the pre-write barrier is gated
    // by these warps' own completion, not warp 0's chain). Bit-exact: the
    // mark array receives the identical -1 bytes over the identical
    // [0, n_local) range; only store width and writing-thread mapping
    // change, and the same tile-0 CTA-wide __syncthreads still orders every
    // init store before every rank/mark write. Runtime-guarded scalar
    // fallback keeps unaligned / non-multiple-of-4 instantiations correct.
    if (mpk_active_expert_ids != nullptr) {
      int const n_local_init = end_expert - start_expert;
      bool const vec_mark_ok =
          ((n_local_init & 3) == 0) &&
          ((reinterpret_cast<uintptr_t>(mpk_active_expert_ids) & 15) == 0);
      if (vec_mark_ok) {
        int4 *mark4 = reinterpret_cast<int4 *>(mpk_active_expert_ids);
        for (int q = init_t; q < n_local_init / 4; q += blockDim.x - 32) {
          mark4[q] = make_int4(-1, -1, -1, -1);
        }
      } else {
        for (int e = init_t; e < n_local_init; e += blockDim.x - 32) {
          mpk_active_expert_ids[e] = -1;
        }
      }
    }
  }
  // (v010) count reset moved off warp 0 too (thread 32 always exists at
  // blockDim.x == 256; same barrier-ordering argument as above).
  if (threadIdx.x == 32 && mpk_active_expert_ids != nullptr) {
    mpk_active_expert_ids[NUM_EXPERTS] = 0;
  }
  // (v005) NO __syncthreads() here. The barrier that makes the init stores
  // above visible moved INSIDE the row-tile loop, to sit AFTER tile 0's
  // softmax/top-k compute and right BEFORE the (now deferred) rank/mark
  // writes -- see the tile loop. The init-store drain + CTA-wide barrier
  // stall then overlaps the row compute (which never touches the two
  // routing arrays) instead of serializing ahead of it. The happens-before
  // edge (every init zero/reset precedes every rank/mark write) is
  // preserved exactly: one full-CTA barrier still separates them.

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

  // (v008) live_rows now defined above the prologue load -- see there.

  // (v003; hoisted to function scope at v005) register cap for the buffered
  // top-k values and (v005) the deferred (expert, rank) mark writes.
  // Production k=8 constant-folds both buffered paths in.
  constexpr int TOPK_REG_CAP = 16;
  // (v005) When one subgroup lane can own each k round (k <=
  // THREADS_PER_ROW), DEFER the mpk_routing_indices rank writes and
  // mpk_active_expert_ids marks to after the relocated init-visibility
  // barrier below, and DISTRIBUTE them: during round k_idx the lane with
  // thread_group_idx == k_idx captures the (subgroup-uniform) winner into
  // a private scalar, and after the barrier each such lane issues its own
  // single rank+mark store -- k parallel writer lanes per row instead of
  // one leader lane serially issuing k stores (the serial variant's
  // exposure scaled with N_LIVE and cost +1.1-1.6us at N_LIVE=8/16; see
  // a006 in progress.md). The fallback (k > THREADS_PER_ROW) keeps the
  // original inline writes and takes the barrier BEFORE the k-loop
  // instead, so the shared template stays general. Both barrier sites are
  // CTA-uniform (row_tile_base and k are uniform), exactly one executes,
  // on tile 0 only, and k is a call-site literal in production
  // (task_register.cc emits 8), so the branch constant-folds away there.
  bool const defer_marks = (k <= THREADS_PER_ROW);

  for (int row_tile_base = 0; row_tile_base < num_rows;
       row_tile_base += ROWS_PER_CTA) {
    int const thread_row = row_tile_base + warp_base_row + thread_row_in_warp;
    // (v005) fallback path: k too large to buffer -> marks are written
    // inline inside the k-loop, so init visibility is needed up front.
    if (row_tile_base == 0 && !defer_marks) {
      __syncthreads();
    }
    // (v005) per-lane deferred-winner scalar (lane j holds round j's
    // winner) + row_is_active, hoisted to tile scope so the deferred write
    // block below can see them.
    int my_expert = 0;
    bool row_is_active = false;
    uint32_t warp_mask = 0xffffffffu;
    if constexpr (THREADS_PER_ROW != WARP_SIZE) {
      constexpr uint32_t subgroup_mask = (1u << THREADS_PER_ROW) - 1u;
      // (v007) Every sub-group uses its OWN mask unconditionally. The
      // padding-row compute skip below means a warp can now hold one LIVE
      // sub-group and one PADDING sub-group that never reaches the shuffle
      // chain; a full-warp mask would name never-arriving lanes in
      // __shfl_xor_sync (UB -- hang/garbage under ITS, and unreproducible
      // co-residency drift in the megakernel). Sub-group-scoped masks make
      // each row's shuffles self-contained regardless of the neighbor
      // row's liveness, and subsume the old odd-num_rows tail-warp
      // restriction (golden keeps that original conditional form).
      warp_mask = subgroup_mask << (thread_row_in_warp * THREADS_PER_ROW);
    }
    // (v013) BOUNDARY-WARP CONVERGED PADDING COMPUTE: when live_rows
    // splits a warp (production: only n_live=1 -- warp 0 holds live row 0
    // in lanes 0-15 and padding row 1 in lanes 16-31; every other n_live
    // lands the boundary on a warp edge), the v007 padding skip makes the
    // two sub-groups DIVERGE, so the warp serializes the live chain THEN
    // the padding short path -- pure added critical-path time on the one
    // warp that matters. Measured: bs1 consistently SLOWER than bs2
    // (2.880-2.912 vs 2.816us) despite strictly less live work. Fix: the
    // padding sub-group of a warp that ALSO holds a live row runs the
    // SAME full compute chain, CONVERGED, on deterministic all-zero
    // logits (registers only -- k <= TOPK_REG_CAP means every per-round
    // value lands in topk_vals[], never in global memory), then writes
    // the exact same zero weight bytes the v007 skip path wrote. Marks/
    // rank writes stay gated by row_is_active (false here), the deferred
    // my_expert capture is dead for inactive rows, and full-padding warps
    // (no live row) keep the v007 skip -- so every output byte on every
    // path is unchanged; only the boundary warp's control flow converges.
    bool const padding_converge =
        (row_tile_base + warp_base_row) < live_rows && (k <= TOPK_REG_CAP);

    if (thread_row < num_rows) {

      row_is_active =
          (finished ? !finished[thread_row] : true) && (thread_row < live_rows);

      // Compute per-thread read pointers (registers/constants hoisted to the
      // prologue at v004)
      T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
      T *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
      AccessType *vec_thread_read_ptr =
          reinterpret_cast<AccessType *>(thread_read_ptr);

      // Vectorized loads across the row: tile 0 was loaded by the prologue;
      // later tiles (only reached when num_rows > ROWS_PER_CTA) reload here.
      // (v008) padding rows skip the reload too -- same gate as the
      // prologue load.
      if (row_tile_base != 0 && thread_row < live_rows) {
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
          row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
      }

      cutlass::NumericConverter<float, T> converter;

      float row_chunk[VPT];
      if (thread_row < live_rows) {
        for (int ii = 0; ii < VPT; ++ii) {
          row_chunk[ii] = converter(row_chunk_temp[ii]);
          row_chunk_temp[ii] =
              static_cast<T>(0); // reset input buffer to 0 for split-k gate linear
        }
      } else {
        // (v008) padding row: nothing was loaded, so nothing to convert --
        // just materialize the zeros the unconditional store-back below
        // writes. The split-K gate-linear invariant needs the FINAL input
        // bytes to be 0; it never required reading them first. row_chunk
        // stays uninitialized here and is never read (every use below sits
        // under the same thread_row < live_rows gate).
        for (int ii = 0; ii < VPT; ++ii) {
          row_chunk_temp[ii] = static_cast<T>(0);
        }
        // (v013) boundary-warp padding lanes run the converged chain on
        // deterministic all-zero logits (results discarded -- zeros are
        // written below regardless); register fill only.
        if (padding_converge) {
          for (int ii = 0; ii < VPT; ++ii) {
            row_chunk[ii] = 0.f;
          }
        }
      }

      // reset input buffer to 0 for split-k gate linear (UNCONDITIONAL for
      // all rows < num_rows -- the load-bearing zeroing invariant)
      // (v012) cache-streaming store hint (__stcs -> st.global.cs): these
      // bytes are never re-read by this kernel invocation, so evict-first
      // avoids holding the ~8KB gating buffer in L1/L2 at normal priority.
      // Same bytes, same addresses, same ordering (the hint changes cache
      // residency policy only, not memory-consistency semantics) -- the
      // downstream split-K gate-linear reader still sees the zeros.
      // Compile-time width dispatch; non-16/8/4-byte AccessType falls back
      // to the plain store.
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        AccessType *dst = &vec_thread_read_ptr[ii * THREADS_PER_ROW];
        AccessType const *src = &row_chunk_vec_ptr[ii];
        if constexpr (sizeof(AccessType) == 16) {
          __stcs(reinterpret_cast<int4 *>(dst),
                 *reinterpret_cast<int4 const *>(src));
        } else if constexpr (sizeof(AccessType) == 8) {
          __stcs(reinterpret_cast<int2 *>(dst),
                 *reinterpret_cast<int2 const *>(src));
        } else if constexpr (sizeof(AccessType) == 4) {
          __stcs(reinterpret_cast<int *>(dst),
                 *reinterpret_cast<int const *>(src));
        } else {
          *dst = *src;
        }
      }

      // (v007) PADDING-ROW COMPUTE SKIP: rows >= live_rows get their input
      // read+zeroed above exactly like live rows (the load-bearing zeroing
      // invariant, topk_softmax_sm100.cuh:100-107, is untouched), but skip
      // the whole max-reduce/softmax/top-k/renormalize chain. Audited safe
      // (2026-07-29, recorded in progress.md "Padding-row observability
      // audit"): topk_weights has exactly ONE downstream consumer,
      // mul_sum_add_sm100.cuh:43, whose padding-row products land only in
      // discarded padding rows of moe_out (builder.py:88-135 cross-row
      // isolation, M2 AC-3); the routing-index/mark writes were ALREADY
      // liveness-gated in golden. Padding rows write deterministic ZEROS
      // to their weight slots (else-branch below) instead of golden's
      // computed-but-unobserved softmax values, so downstream padding
      // moe_out stays finite and replay-deterministic. This DELIBERATELY
      // diverges from golden bytes on padding rows only -- the harness
      // gate is relaxed to live-rows-bit-exact + padding-rows-exactly-zero
      // for topk_weights ONLY (all other outputs remain full-compare).
      // live_rows == num_rows when num_active_rows is -1/num_rows, so the
      // skip constant-folds away in ungated instantiations.
      if (thread_row < live_rows || padding_converge) {
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
        // (v003) Buffer the row's top-k values in registers across the k-loop
        // instead of writing raw values to global `output` each round and
        // re-reading them for renormalization: an fp32 store+load round trip
        // is lossless, so the deferred single write below produces the exact
        // same bits from the exact same arithmetic. Fallback to the original
        // write-through path for k beyond the cap keeps the template general
        // (production k=8 constant-folds the buffered path).
        float topk_vals[TOPK_REG_CAP];
        bool const buffer_topk = (k <= TOPK_REG_CAP);

        // (v009) SORTED-LANE POP top-k: the golden k-loop re-scans all VPT
        // local values EVERY round (the 16-element scan sits on the critical
        // chain between consecutive rounds; a011's diag ladder attributed
        // ~1.5-1.7us of the bs1 wall to this loop). Instead: pack each
        // element ONCE into the v004 order-preserving 64-bit key (identical
        // map: sortable-float high word, 0xffffffff-col low word -> unsigned
        // key order == golden's value>/tie-lower-col semantics, keys globally
        // UNIQUE because col is), bitonic-sort the lane's 16 keys ONCE
        // (static indices -> register-resident; a010's proven network), keep
        // a DESCENDING register list, and each round offer only the lane's
        // current head. Cross-lane winner = max over per-lane heads = max
        // over all unconsumed keys (each lane's list pops in descending
        // order) = exactly golden's round winner, round by round; the winner
        // lane (key == wkey, unique) shifts its list down one (static
        // unrolled shift, register moves only -- no dynamic indexing, per
        // the a009 local-memory lesson). Epilogue statements per round are
        // verbatim golden's. Serial fallback below keeps the template
        // general (k > VPT or other VPT/THREADS_PER_ROW instantiations).
        bool fast_done = false;
        if constexpr (VPT == 16 && THREADS_PER_ROW == 16) {
          if (k <= VPT) {
            unsigned long long kk[VPT];
  #pragma unroll
            for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
  #pragma unroll
              for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                int const e = ldg * ELTS_PER_LDG + ii;
                int const col = start_col + ldg * COLS_PER_GROUP_LDG + ii;
                unsigned int const vb = __float_as_uint(row_chunk[e]);
                unsigned int const vs =
                    vb ^ (((unsigned int)((int)vb >> 31)) | 0x80000000u);
                kk[e] = ((unsigned long long)vs << 32) |
                        (unsigned long long)(0xffffffffu - (unsigned int)col);
              }
            }
            // per-lane bitonic sort-16, ASCENDING (all indices static)
  #pragma unroll
            for (int size = 2; size <= VPT; size <<= 1) {
  #pragma unroll
              for (int stride = size >> 1; stride > 0; stride >>= 1) {
  #pragma unroll
                for (int i = 0; i < VPT; ++i) {
                  int const j = i ^ stride;
                  if (j > i) {
                    bool const up = ((i & size) == 0);
                    unsigned long long const a = kk[i], b = kk[j];
                    bool const sw = up ? (a > b) : (a < b);
                    kk[i] = sw ? b : a;
                    kk[j] = sw ? a : b;
                  }
                }
              }
            }
            // descending view for head-popping
            unsigned long long desc[VPT];
  #pragma unroll
            for (int i = 0; i < VPT; ++i) {
              desc[i] = kk[VPT - 1 - i];
            }
            // Per-round emission, verbatim golden statements (shared by the
            // paired and tail rounds below).
            auto emit_round = [&](unsigned long long wkey, int k_idx) {
              float max_val;
              int expert;
              {
                unsigned int const win_sortable = (unsigned int)(wkey >> 32);
                unsigned int const win_bits =
                    win_sortable ^
                    ((win_sortable & 0x80000000u) ? 0x80000000u : 0xffffffffu);
                max_val = __uint_as_float(win_bits);
                expert = (int)(0xffffffffu - (unsigned int)(wkey & 0xffffffffu));
              }

              // ---- epilogue: verbatim golden per-round statements ----
              if (thread_group_idx == 0) {
                bool const node_uses_expert =
                    expert >= start_expert && expert < end_expert;
                bool const should_process_row = row_is_active && node_uses_expert;
                int const out_idx = k * thread_row + k_idx;
                if (buffer_topk) {
                  topk_vals[k_idx] = max_val;
                } else {
                  output[out_idx] = max_val;
                }
                row_sum_for_renormalize += max_val;
                if (!defer_marks && should_process_row &&
                    mpk_routing_indices != nullptr) {
                  int const local_expert = expert - start_expert;
                  mpk_routing_indices[local_expert * num_rows + thread_row] =
                      k_idx + 1;
                  if (mpk_active_expert_ids != nullptr) {
                    mpk_active_expert_ids[local_expert] = local_expert;
                    atomicOr(&active_bits[local_expert >> 5],
                             1u << (local_expert & 31));
                  }
                }
              }
              if (defer_marks && thread_group_idx == k_idx) {
                my_expert = expert;
              }
            };

            // (v009) ROUND PAIRING: one 4-level butterfly computes the global
            // TOP-2 of the current heads multiset, where each lane offers its
            // sorted top-2 (h1 >= h2). Correctness: the global #1/#2 of all
            // unconsumed keys always lie in the union of per-lane top-2s (a
            // lane's 3rd key cannot be global #2 without its first two being
            // #1 and #2), and golden's rounds 2r/2r+1 winners ARE the global
            // #1/#2 of the unconsumed multiset -- so (m1, m2) equal golden's
            // consecutive round winners exactly, and emission order (m1 then
            // m2) preserves golden's per-round statement order bit-for-bit.
            // This halves the sequential cross-lane shuffle-level count vs
            // one-reduce-per-round (32 -> 16 levels for k=8) while keeping
            // the same total shuffle instruction count. Pop accounting per
            // lane: h1 consumed iff h1 ∈ {m1,m2}; h2 consumed iff h2 == m2
            // (h2 == m2 implies h1 == m1: if h1 were < m1 yet > m2 = h2, m2
            // would have been h1 -- contradiction; keys unique).
            unsigned long long h1 = desc[0], h2 = desc[1];
            int k_base = 0;
            for (; k_base + 1 < k; k_base += 2) {
              unsigned long long a1 = h1, a2 = h2;
              for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
                unsigned long long const b1 =
                    __shfl_xor_sync(warp_mask, a1, mask, THREADS_PER_ROW);
                unsigned long long const b2 =
                    __shfl_xor_sync(warp_mask, a2, mask, THREADS_PER_ROW);
                bool const a_wins = a1 > b1;
                unsigned long long const s_if_a = a2 > b1 ? a2 : b1;
                unsigned long long const s_if_b = b2 > a1 ? b2 : a1;
                a1 = a_wins ? a1 : b1;
                a2 = a_wins ? s_if_a : s_if_b;
              }
              emit_round(a1, k_base);
              emit_round(a2, k_base + 1);
              if (k_base + 2 < k) {
                int const pops =
                    ((h1 == a1 || h1 == a2) ? 1 : 0) + ((h2 == a2) ? 1 : 0);
                if (pops >= 1) {
  #pragma unroll
                  for (int i = 0; i < VPT - 1; ++i) {
                    desc[i] = desc[i + 1];
                  }
                }
                if (pops == 2) {
  #pragma unroll
                  for (int i = 0; i < VPT - 1; ++i) {
                    desc[i] = desc[i + 1];
                  }
                }
                h1 = desc[0];
                h2 = desc[1];
              }
            }
            // odd-k tail: one classic single-max round
            if (k_base < k) {
              unsigned long long wkey = h1;
              for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
                unsigned long long const other =
                    __shfl_xor_sync(warp_mask, wkey, mask, THREADS_PER_ROW);
                wkey = other > wkey ? other : wkey;
              }
              emit_round(wkey, k_base);
            }
            fast_done = true;
          }
        }

        if (!fast_done) {
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
          // index). (v004) Packed 64-bit key: high word = monotone
          // float->uint map of the value (order-preserving for every finite
          // float incl. the -10000.f blank sentinel; softmax values are
          // positive finite, no NaN/-0.0 possible), low word = 0xffffffff -
          // index so unsigned max breaks value ties toward the LOWER index --
          // exactly golden's (>, ==&&idx<) semantics, so the winner (value
          // bits AND index) is identical every round.
          unsigned int const val_bits = __float_as_uint(max_val);
          unsigned int const val_sortable =
              val_bits ^ (((unsigned int)((int)val_bits >> 31)) | 0x80000000u);
          unsigned long long key =
              ((unsigned long long)val_sortable << 32) |
              (unsigned long long)(0xffffffffu - (unsigned int)expert);
          for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            unsigned long long const other =
                __shfl_xor_sync(warp_mask, key, mask, THREADS_PER_ROW);
            key = other > key ? other : key;
          }
          {
            unsigned int const win_sortable = (unsigned int)(key >> 32);
            unsigned int const win_bits =
                win_sortable ^
                ((win_sortable & 0x80000000u) ? 0x80000000u : 0xffffffffu);
            max_val = __uint_as_float(win_bits);
            expert = (int)(0xffffffffu - (unsigned int)(key & 0xffffffffu));
          }

          // Write out the selected top-k value/index (one thread per subgroup
          // writes)
          if (thread_group_idx == 0) {
            bool const node_uses_expert =
                expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;
            int const out_idx = k * thread_row + k_idx;
            if (buffer_topk) {
              topk_vals[k_idx] = max_val;
            } else {
              output[out_idx] = max_val;
            }
            // indices[out_idx] =
            //     should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            row_sum_for_renormalize += max_val;
            // Optionally populate MPK routing structures.
            // (v005) deferred path: nothing to do here (round k_idx's writer
            // lane captured the winner below); inline fallback unchanged.
            if (!defer_marks && should_process_row && mpk_routing_indices != nullptr) {
              int const local_expert = expert - start_expert;
              // Write 1-based rank into routing indices; stride by num_rows per
              // expert
              mpk_routing_indices[local_expert * num_rows + thread_row] = k_idx + 1;
              // Sparse mark expert as active; idempotent without atomics
              if (mpk_active_expert_ids != nullptr) {
                mpk_active_expert_ids[local_expert] = local_expert;
                // (v006) shadow bit for the smem-read compaction
                atomicOr(&active_bits[local_expert >> 5],
                         1u << (local_expert & 31));
              }
            }
          }

          // (v005) round-owner capture for the deferred distributed write:
          // `expert` is subgroup-uniform here (it came out of the full
          // shuffle reduce), so lane k_idx's copy equals the leader's.
          if (defer_marks && thread_group_idx == k_idx) {
            my_expert = expert;
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
        } // !fast_done

        // Optional renormalization + the (v003) deferred single write of the
        // buffered top-k values. Final memory contents are bit-identical to
        // the golden write-then-rewrite sequence in every (renormalize,
        // round_weights, buffer_topk) combination.
        if (thread_group_idx == 0 && thread_row >= live_rows) {
          // (v013) boundary-warp padding sub-group leader: identical
          // deterministic zero weight bytes as the v007 skip path below;
          // the converged chain's computed values are discarded.
          bool const vec_ok = ((k & 3) == 0) &&
              ((reinterpret_cast<uintptr_t>(output) & 15) == 0);
          if (vec_ok) {
            float4 *out4 = reinterpret_cast<float4 *>(output + k * thread_row);
            for (int q = 0; q < k / 4; ++q) {
              out4[q] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
          } else {
            for (int k_idx = 0; k_idx < k; ++k_idx) {
              output[k * thread_row + k_idx] = 0.f;
            }
          }
        } else if (thread_group_idx == 0) {
          if (buffer_topk) {
            if (renormalize) {
              cutlass::NumericConverter<T, float> to_output_dtype;
              float inv = 1.f / row_sum_for_renormalize;
              // (v004) 128-bit stores of the row's contiguous k weights when
              // k%4==0 and the base is 16B-aligned (k*thread_row*4 is then a
              // multiple of 16 too). Per-element value computation unchanged
              // -- only store width differs, so bytes are identical; scalar
              // fallback keeps the template general.
              bool const vec_store_ok =
                  ((k & 3) == 0) &&
                  ((reinterpret_cast<uintptr_t>(output) & 15) == 0);
              if (vec_store_ok) {
                float4 *out4 =
                    reinterpret_cast<float4 *>(output + k * thread_row);
                for (int q = 0; q < k / 4; ++q) {
                  float4 v;
                  float const w0 = topk_vals[q * 4 + 0] * inv;
                  float const w1 = topk_vals[q * 4 + 1] * inv;
                  float const w2 = topk_vals[q * 4 + 2] * inv;
                  float const w3 = topk_vals[q * 4 + 3] * inv;
                  v.x = round_weights_to_output_dtype
                            ? converter(to_output_dtype(w0)) : w0;
                  v.y = round_weights_to_output_dtype
                            ? converter(to_output_dtype(w1)) : w1;
                  v.z = round_weights_to_output_dtype
                            ? converter(to_output_dtype(w2)) : w2;
                  v.w = round_weights_to_output_dtype
                            ? converter(to_output_dtype(w3)) : w3;
                  out4[q] = v;
                }
              } else {
                for (int k_idx = 0; k_idx < k; ++k_idx) {
                  float const w = topk_vals[k_idx] * inv;
                  output[k * thread_row + k_idx] =
                      round_weights_to_output_dtype
                          ? converter(to_output_dtype(w))
                          : w;
                }
              }
            } else {
              for (int k_idx = 0; k_idx < k; ++k_idx) {
                output[k * thread_row + k_idx] = topk_vals[k_idx];
              }
            }
          } else if (renormalize) {
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
      } else if (thread_group_idx == 0) {
        // (v007) padding row: deterministic zero weights (see block comment
        // above); same vectorization guard as the live-row store path.
        bool const vec_ok = ((k & 3) == 0) &&
            ((reinterpret_cast<uintptr_t>(output) & 15) == 0);
        if (vec_ok) {
          float4 *out4 = reinterpret_cast<float4 *>(output + k * thread_row);
          for (int q = 0; q < k / 4; ++q) {
            out4[q] = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        } else {
          for (int k_idx = 0; k_idx < k; ++k_idx) {
            output[k * thread_row + k_idx] = 0.f;
          }
        }
      }
    }

    // (v005) Relocated init-visibility barrier: sits AFTER tile 0's whole
    // softmax/top-k compute (which touches neither routing array), so the
    // init-store drain + barrier stall overlaps ~the full row compute
    // instead of preceding it. CTA-uniform condition; tiles > 0 need no
    // barrier (tile 0's barrier already ordered init before them).
    if (row_tile_base == 0 && defer_marks) {
      __syncthreads();
    }
    // (v005) Deferred rank/mark writes, DISTRIBUTED: lane j (j < k) of
    // each live row's subgroup writes round j's rank and mark -- same
    // values, same addresses, same gating (row_is_active &&
    // node-uses-expert && non-null pointers) as the original inline
    // writes; only which thread issues them and where in program order
    // changed. No other thread reads these until the pre-compaction
    // barrier below; distinct (expert,row) pairs write disjoint rank
    // slots (one row selects each expert at most once across rounds, so
    // lane j's slot is unique within the row and rows are disjoint by
    // address); the expert mark write is idempotent (always the same
    // value) even when several lanes/rows mark the same expert -- so the
    // final memory image is bit-identical and schedule-independent.
    if (defer_marks && thread_row < num_rows && row_is_active &&
        thread_group_idx < k && mpk_routing_indices != nullptr) {
      int const expert = my_expert;
      if (expert >= start_expert && expert < end_expert) {
        int const local_expert = expert - start_expert;
        mpk_routing_indices[local_expert * num_rows + thread_row] =
            thread_group_idx + 1;
        if (mpk_active_expert_ids != nullptr) {
          mpk_active_expert_ids[local_expert] = local_expert;
          // (v006) shadow bit for the smem-read compaction
          atomicOr(&active_bits[local_expert >> 5], 1u << (local_expert & 31));
        }
      }
    }
  }
  __syncthreads();
  // ---- Compact the marks into a DENSE, ASCENDING list and count ----
  //
  // THE CONTRACT THIS BLOCK OWES ITS CONSUMERS (M3-I5c, retained verbatim in
  // intent across the M4-I6 rewrite -- read it before touching anything here):
  // the compacted list must be DENSE, STRICTLY ASCENDING in expert id, and
  // produced with NO atomicAdd. The shipped pre-M3-I5c body was an in-place
  // read-then-scatter that took the slot from `atomicAdd(mpk_active_expert_ids
  // + NUM_EXPERTS, 1)`, and it had two independent defects: (1) compacted
  // entries land in slots [0, n_active) which ALIAS the marks of experts
  // [0, n_active), with nothing ordering thread j's read of slot j against
  // another thread's write of it -- an INACTIVE expert whose slot was
  // overwritten passes `mark >= 0` and appends ITSELF, inflating the set and
  // the count, and with enough phantoms the position reaches NUM_EXPERTS and
  // clobbers the counter; (2) a GUARANTEED miscount whenever blockDim.x <
  // NUM_EXPERTS makes the grid-stride loop take more than one pass, because a
  // thread's earlier scatter can land on a slot it reads later -- arithmetic,
  // not a scheduling accident (M3-I9b). The atomicAdd was also the ONLY source
  // of run-to-run permutation in this task, so removing it is what makes the
  // ORDER deterministic and not merely the set. M3-I5c validated the
  // replacement at 800/800 order-clean + racecheck 0 hazards; any future
  // rewrite of this block owes the same evidence, plus the >=200-run same-input
  // replay stress named below.
  //
  // (v002) Warp-0-only ballot/popcount compaction, replacing the M3-I5c
  // per-thread serial tile scan (256 gmem loads + compare per thread, all
  // 256 threads). Computes the IDENTICAL result: rank is the same exclusive
  // prefix count of active marks below each slot, evaluated warp-chunk by
  // warp-chunk in ascending expert order, so the output list has the same
  // SET and the same strictly-ascending ORDER, position by position, and
  // the count slot gets the same total.
  //
  // Race-freedom (chunk-granular version of M3-I5c's own argument, with no
  // assumption about NUM_EXPERTS or the active count):
  //   * only warp 0 touches the array after the preceding __syncthreads()
  //     (which orders every row-loop mark before these reads);
  //   * within a chunk, __ballot_sync is a warp-wide sync point: all 32
  //     lane reads of chunk c complete before any lane's write of chunk c;
  //   * writes of chunk c land at slots [base_c, base_{c+1}) and
  //     base_{c+1} <= chunk_base + WARP_SIZE (at most one active per slot),
  //     i.e. strictly below every LATER chunk's read region, so no write
  //     ever races a later read;
  //   * the count slot (NUM_EXPERTS) is written once, by lane 0, after the
  //     loop; compacted entries only occupy slots < n_local <= NUM_EXPERTS.
  //
  // DETERMINISM: rank is a pure integer function of the mark array and the
  // lane index -- no atomics, no schedule dependence. Verified by the
  // harness's >=200-run same-input replay stress (task.yaml constraint for
  // any compaction-algorithm change, mirroring M3-I5c's C4 check).
  //
  // Generality: works for any num_local_experts (ceil(n/32) chunks) and any
  // blockDim.x >= WARP_SIZE, so the shared template's other instantiations
  // (e.g. 128 experts) fall through the same logic, not a 256-only path.
  // (v006) The active-test now reads the shared-memory shadow bitmask
  // instead of re-loading mpk_active_expert_ids[j] from (cold) global
  // memory: the bitmask word for a 32-expert chunk IS the chunk's ballot
  // result, so the __ballot_sync disappears too. The bit for local expert
  // e is set iff the SAME guarded condition that stored the global mark
  // held (identical guard, adjacent atomicOr), and init cleared all words
  // before the tile-0 barrier, so bit e == (mpk_active_expert_ids[e] >= 0)
  // exactly -- same active SET, same exclusive-prefix rank arithmetic,
  // hence the same dense ascending list and count, position by position.
  // Race-freedom is now trivial: the compaction's writes can no longer
  // alias its read source at all (reads are smem, writes are global), so
  // the v002 chunk-ordering argument is subsumed. Determinism: rank is a
  // pure function of the bitmask and lane index; the harness's >=200-run
  // same-input replay stress covers this compaction-algorithm change per
  // the task.yaml constraint.
  if (mpk_active_expert_ids != nullptr) {
    int const num_local_experts = end_expert - start_expert;
    if (threadIdx.x < WARP_SIZE) {
      int const lane = static_cast<int>(threadIdx.x);
      int base = 0; // #active strictly below this chunk; warp-uniform
      for (int chunk_base = 0; chunk_base < num_local_experts;
           chunk_base += WARP_SIZE) {
        int const j = chunk_base + lane;
        // Writers only set bits for valid local experts, so no tail mask
        // is needed on the word itself; the j bound still gates the write.
        uint32_t const active_mask = active_bits[chunk_base >> 5];
        bool const is_active =
            (j < num_local_experts) && ((active_mask >> lane) & 1u);
        if (is_active) {
          int const rank =
              base + __popc(active_mask & ((1u << lane) - 1u));
          mpk_active_expert_ids[rank] = start_expert + j;
        }
        base += __popc(active_mask);
      }
      if (lane == 0) {
        mpk_active_expert_ids[NUM_EXPERTS] = base;
      }
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
