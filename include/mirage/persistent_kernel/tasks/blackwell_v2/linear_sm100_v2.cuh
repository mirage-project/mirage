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

// v2 linear for decode, built on the Channel primitives in channel.cuh.
//
// Synchronization and storage are separate:
//   Channel  - mbarriers only (full/empty edges). Its producer/consumer cursors
//              own the stage index, which keeps the four role functions in
//              step.
//   SmemRing - per-stage SMEM offsets, and optionally the page ids for
//              cross-task page release.
// Shapes, SMEM/SEM ordinals, and PTX wrappers all live in kernel::linear
// (linear_spec.h + linear_device.cuh).
//
// W and A share one empty edge (mma_mbar): the mma's single tcgen05.commit
// per K-iteration frees both, so only the W cursor waits on it and A just
// tracks the stage alongside it.
//
// Each role re-inits its async edges at the start of a task to drop stray
// arrivals left on a reused ring slot by the previous occupant. Without that
// the kernel deadlocks once slots are recycled across tasks.
//
// tiles_per_task must be 1: a partial last task (num_tiles % tpt != 0) has
// different barrier accounting and deadlocks on slot reuse.

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/channel.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_device.cuh"

namespace kernel {
namespace linear_v2 {

// Constants, SMEM/SEM ordinals, and PTX wrappers come from kernel::linear.
using ::kernel::linear::A_SIZE;
using ::kernel::linear::BLOCK_K;
using ::kernel::linear::BLOCK_M;
using ::kernel::linear::BLOCK_N;
using ::kernel::linear::CH_A;
using ::kernel::linear::CH_W;
using ::kernel::linear::CHANNELS;
using ::kernel::linear::elect_sync;
using ::kernel::linear::I_DESC;
using ::kernel::linear::L2_EVICT_FIRST;
using ::kernel::linear::L2_EVICT_LAST;
using ::kernel::linear::L2_EVICT_NORMAL;
using ::kernel::linear::mbarrier_arrive;
using ::kernel::linear::mbarrier_arrive_expect_tx;
using ::kernel::linear::mbarrier_wait;
using ::kernel::linear::MMA_K;
using ::kernel::linear::NUM_STAGES;
using ::kernel::linear::SEM_A_TMA_BASE;
using ::kernel::linear::SEM_COMPUTE_DONE;
using ::kernel::linear::SEM_EPILOGUE_BASE;
using ::kernel::linear::SEM_MAINLOOP_BASE;
using ::kernel::linear::SEM_MMA_BASE;
using ::kernel::linear::SEM_TMEM_READY;
using ::kernel::linear::SEM_W_TMA_BASE;
using ::kernel::linear::SMEM_DESC;
using ::kernel::linear::tcgen05_commit;
using ::kernel::linear::tcgen05_mma;
using ::kernel::linear::tma_3d_load_l2;
using ::kernel::linear::W_SIZE;
using ::kernel::linear::WARP_SIZE;
using ::kernel::linear::warp_uniform;

using mpk::ch::By;

// Channels carry only mbarriers:
//   WChan/AChan: full = TMA-arrived for that stage; empty = the shared
//   mma_mbar. AccChan: TMEM-backed and double-buffered (2 slots).
//   mainloop_stage cycles
//            mod 2, alternating TMEM columns taddr+0 / taddr+BLOCK_N so tile
//            t+1's MMA overlaps the compute reading tile t.
using WChan = mpk::ch::Channel<NUM_STAGES, By::Tma, By::Mma>;
using AChan = mpk::ch::Channel<NUM_STAGES, By::Tma, By::Mma>;
using AccChan = mpk::ch::TmemChannel<2, By::Mma, By::Warp>;

// A W stage is 2 dedicated contiguous pages. With CROSS_TASK_PAGES on, the W
// ring runs the per-stage page lifecycle (loader acquires, mma releases
// page by page) so the next task's loader can TMA into pages this task has
// already finished with. A stages are sub-page and packed, so per-stage page
// control isn't safe for them: the A ring carries no pages, and A's pages (plus
// scratch, which shares one) are freed in a parallel end-of-task sweep over the
// pages no ring owns. PAGES_PER_SLOT=0 compiles the page methods away.
using ::kernel::linear::CROSS_TASK_PAGES;
using WRing = mpk::ch::SmemRing<NUM_STAGES, CROSS_TASK_PAGES ? 2 : 0>;
using ARing = mpk::ch::SmemRing<NUM_STAGES>;

// ── Build channels (mbars) + rings (storage) from kernel::linear addresses ──
__device__ __forceinline__ void
    make_wa(int smem,
            int dyn_sem_base,
            mirage::runtime::TaskDesc const *task_desc,
            WChan &Wc,
            AChan &Ac,
            WRing &Wr,
            ARing &Ar) {
  // Edge wiring is read from CHANNELS (constexpr-folded). The static_asserts in
  // linear_spec.h pin these to the SEM_*_BASE ordinals.
  constexpr int w_full = CHANNELS[CH_W].full_sem_base;
  constexpr int w_empty = CHANNELS[CH_W].empty_sem_base;
  constexpr int a_full = CHANNELS[CH_A].full_sem_base;
  constexpr int a_sh = CHANNELS[CH_A].shares_empty_with; // 0 -> share W empty
  Wc.full = dyn_sem_base + w_full * 8;
  Wc.empty = dyn_sem_base + w_empty * 8;
  Ac.full = dyn_sem_base + a_full * 8;
  Ac.empty = (a_sh >= 0) ? Wc.empty // SHARED
                         : dyn_sem_base + CHANNELS[CH_A].empty_sem_base * 8;

  // Per-stage SMEM offsets. W also records its 2 physical page ids so the ring
  // owns their cross-task lifecycle. A is storage-only.
  for (int s = 0; s < NUM_STAGES; s++) {
    Wr.slot_offsets[s] =
        smem + task_desc->smem_region_offset(::kernel::linear::REGION_W_0 + s);
    Ar.slot_offsets[s] =
        smem + task_desc->smem_region_offset(::kernel::linear::REGION_A_0 + s);
    if constexpr (CROSS_TASK_PAGES) {
      int const w_page0 =
          task_desc->smem_region_page(::kernel::linear::REGION_W_0 + s);
      Wr.pages[s][0] = w_page0;
      Wr.pages[s][1] = w_page0 + 1; // W_SIZE spans 2 contiguous pages
    }
  }
}

// TmemChannel stores only barrier addresses + cols_per_slot. taddr lives on
// the cursor (set via TmemProducer::set_taddr / TmemConsumer::set_taddr after
// the mma's tcgen05.alloc publishes it).
__device__ __forceinline__ AccChan make_acc_channel(int dyn_sem_base) {
  return AccChan{
      /*cols_per_slot =*/BLOCK_N,
      /*full          =*/dyn_sem_base + SEM_MAINLOOP_BASE * 8,
      /*empty         =*/dyn_sem_base + SEM_EPILOGUE_BASE * 8,
  };
}

// ── Derived shape, computed once per role from identical inputs ────────────
// Every role used to recompute these — easy to drift if one diverges. One
// function, called the same way from all three roles, makes drift impossible.
struct TaskCtx {
  int num_spatial_tiles; // grid_m (= N_real / BLOCK_M)
  int num_tiles;         // num_spatial_tiles * SPLIT_K
  int tiles;             // tiles_to_process this task (≤ TILES_PER_TASK)
  int iters;             // iters_per_slice (= K / BLOCK_K / SPLIT_K)

  __device__ bool bounds_fail(int tile_idx) const {
    return tile_idx >= num_tiles;
  }
};

template <int SPLIT_K, int TILES_PER_TASK>
__device__ __forceinline__ TaskCtx ctx_from(int N_real, int K, int tile_idx) {
  TaskCtx c;
  c.num_spatial_tiles = N_real / BLOCK_M;
  c.num_tiles = c.num_spatial_tiles * SPLIT_K;
  int left = c.num_tiles - tile_idx;
  c.tiles = (TILES_PER_TASK < left) ? TILES_PER_TASK : left;
  c.iters = (K / BLOCK_K) / SPLIT_K;
  return c;
}

// ── PTX wrappers (same emitted instructions as the inline forms in v2) ─────

// MMA inner loop over BLOCK_K. Emits exactly the same tcgen05.mma sequence as
// v2's hand-rolled k2/k1 nested loops. The `accumulate` flag controls only the
// FIRST tcgen05_mma's enable_d predicate; all subsequent inner MMAs always
// accumulate (matching v2). `i != 0` is equivalent to v2's pass-through of `i`
// because tcgen05_mma's wrapper does `setp.ne.b32 p, %4, 0` — any nonzero value
// produces the same predicate.
__device__ __forceinline__ void
    mma_k_block(int tmem, int W_smem, int A_smem, bool accumulate) {
  uint64_t a_desc = SMEM_DESC | (uint64_t)((uint32_t)W_smem >> 4);
  uint64_t b_desc = SMEM_DESC | (uint64_t)((uint32_t)A_smem >> 4);

  // First 64-byte K chunk, first MMA — enable_d picks zero-vs-accumulate.
  tcgen05_mma(tmem, a_desc, b_desc, I_DESC, accumulate ? 1 : 0);

  // Rest of the first 64-byte K chunk — always accumulate.
  for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
    a_desc += (32 >> 4);
    b_desc += (32 >> 4);
    tcgen05_mma(tmem, a_desc, b_desc, I_DESC, 1);
  }

  // Remaining 64-byte K chunks.
  for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
    uint64_t a2 =
        SMEM_DESC | (uint64_t)(((uint32_t)W_smem + k1 * BLOCK_M * 128) >> 4);
    uint64_t b2 =
        SMEM_DESC | (uint64_t)(((uint32_t)A_smem + k1 * BLOCK_N * 128) >> 4);
    for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
      tcgen05_mma(tmem, a2, b2, I_DESC, 1);
      a2 += (32 >> 4);
      b2 += (32 >> 4);
    }
  }
}

// 16-register TMEM read (the giant inline asm). The `t_addr` layout is
// `(warp_id * 32 << 16) | t_col` — caller computes this, same as v2.
__device__ __forceinline__ void tcgen05_ld_16(float (&out)[16], int t_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=f"(out[0]),
                 "=f"(out[1]),
                 "=f"(out[2]),
                 "=f"(out[3]),
                 "=f"(out[4]),
                 "=f"(out[5]),
                 "=f"(out[6]),
                 "=f"(out[7]),
                 "=f"(out[8]),
                 "=f"(out[9]),
                 "=f"(out[10]),
                 "=f"(out[11]),
                 "=f"(out[12]),
                 "=f"(out[13]),
                 "=f"(out[14]),
                 "=f"(out[15])
               : "r"(t_addr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// Global stores with L1::no_allocate hint (matches v2's epilogue stores).
__device__ __forceinline__ void st_bf16(nv_bfloat16 *dst, nv_bfloat16 v) {
  asm volatile("st.global.L1::no_allocate.b16 [%0], %1;" ::"l"(dst),
               "h"(*(uint16_t *)&v)
               : "memory");
}

__device__ __forceinline__ void st_f32(float *dst, float v) {
  asm volatile("st.global.L1::no_allocate.b32 [%0], %1;" ::"l"(dst), "f"(v)
               : "memory");
}

// Prefetch a CUtensorMap descriptor into L1 (so the first TMA on it doesn't
// stall on descriptor fetch). Same instruction as v2's inline asm.
__device__ __forceinline__ void prefetch_tensormap(void const *tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap));
}

// Acquire fence following a batch of mbarrier.init writes. Matches v2.
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// Per-thread fence required between thread-local register reads and tcgen05
// operations that consume them. Same instruction v2 uses.
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// ═══════════════════════════════════════════════════════════════════════════
// Loader role (warp 4, elected lane only) — TMA loop + start-of-task re-init.
// ═══════════════════════════════════════════════════════════════════════════
template <int SPLIT_K = 1, int W_L2_HINT = 0, int TILES_PER_TASK = 1>
__device__ __noinline__ void
    linear_loader_task(mirage::runtime::TaskDesc const *task_desc,
                       mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                       mirage::runtime::RuntimeConfig const &runtime_config,
                       CUtensorMap const *W_tmap_ptr,
                       CUtensorMap const *A_tmap_ptr,
                       int N_real,
                       int K,
                       int tile_idx,
                       int instruction_index,
                       int iter_num,
                       int dyn_sem_base) {
  if (!elect_sync()) {
    return;
  }

  prefetch_tensormap(W_tmap_ptr);
  prefetch_tensormap(A_tmap_ptr);

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

  // Shape — one place, used by all three roles.
  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  if (c.bounds_fail(tile_idx)) {
    return;
  }

  MPK_V2_PROF_SNAPSHOT()

  constexpr uint64_t W_HINT =
      (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;

  // ── Channels (sync) + rings (storage) + cursors ─────────────────────────
  WChan Wc;
  AChan Ac;
  WRing Wr;
  ARing Ar;
  make_wa(smem, dyn_sem_base, task_desc, Wc, Ac, Wr, Ar);

  // Re-init the loader's async edges per each channel's reinit_*_by policy.
  ::kernel::linear::reinit_for_role(::kernel::linear::Role::Loader,
                                    dyn_sem_base);

  mpk::ch::Producer<WChan> pW{Wc};
  mpk::ch::Producer<AChan> pA{Ac};
  // Both ph start at 1 (pre-empty). pW.ph mirrors v2's `mma_phase`.

  bool dep_done = false;

  for (int t = 0; t < c.tiles; t++) {
    int const cur_tile_idx = tile_idx + t;
    int const cur_spatial_idx = cur_tile_idx % c.num_spatial_tiles;
    int const cur_k_slice = cur_tile_idx / c.num_spatial_tiles;
    int const cur_k_start = cur_k_slice * c.iters;
    int const cur_off_m = cur_spatial_idx * BLOCK_M;

    for (int i = 0; i < c.iters; i++) {
      int const iter_k = cur_k_start + i;
      int const z_coord = iter_k * (BLOCK_K / 64);

      // Cross-task page acquire on FIRST touch of each W stage (first
      // NUM_STAGES global iters each visit a distinct stage once): wait the
      // prior task's release of this stage's W pages before TMA-ing into them —
      // this overlaps THIS task's W loads with the PRIOR task's compute.
      // Steady-state reuse after that rides the in-task empty edge below.
      if constexpr (CROSS_TASK_PAGES) {
        if (t * c.iters + i < NUM_STAGES) {
          Wr.acquire(pW.st,
                     runtime_smem,
                     instruction_index,
                     mirage::runtime_v2::runtime_wait_page_ready);
        }
      }

      // Wait shared empty (mma_mbar[stage]) — one wait per iter, covers both.
      // (timed-wait on the FIRST ring lap only: that is where the cold-start
      // exposure lives; timing every K-iter measurably slowed the kernel.)
      MPK_V2_TIMED_WAIT_IF(t * c.iters + i < NUM_STAGES,
                           V2_PROF_GROUP_LOADER_STALL,
                           V2_PROF_MMA_EMPTY_WAIT,
                           pW.wait_free());
      int const W_smem = Wr.slot_addr(pW.st); // storage addr from the ring

      // W TMA — v2 order: cp.async.bulk first, expect_tx after.
      tma_3d_load_l2(
          W_smem, W_tmap_ptr, 0, cur_off_m, z_coord, pW.full_mbar(), W_HINT);
      mbarrier_arrive_expect_tx(pW.full_mbar(), W_SIZE);

      // Cross-SM dep wait once (gates A — matches v2's prefetch pattern).
      if (!dep_done) {
        mirage::runtime_v2::wait_task_dependency(
            runtime_config, task_desc, iter_num);
        dep_done = true;
      }

      // A TMA — A shares the empty edge (already waited via pW), so no separate
      // wait. Storage addr comes from A's ring at A's cursor stage.
      int const A_smem = Ar.slot_addr(pA.st);
      tma_3d_load_l2(
          A_smem, A_tmap_ptr, 0, 0, z_coord, pA.full_mbar(), L2_EVICT_LAST);
      mbarrier_arrive_expect_tx(pA.full_mbar(), A_SIZE);

      // Advance both cursors in lockstep.
      pW.commit_tma();
      pA.commit_tma();
    }
  }
  // The loader doesn't wait out its in-flight TMAs at task end: blocking on the
  // final mma_mbar there would deadlock against the next task reusing the slot.
  // Stale arrivals are cleared by the start-of-task re-init instead.
}

// ═══════════════════════════════════════════════════════════════════════════
// Mma role (warp 5, all 32 lanes — alloc/dealloc are sync.aligned).
// ═══════════════════════════════════════════════════════════════════════════
template <int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void
    linear_mma_task(mirage::runtime::TaskDesc const *task_desc,
                    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
                    int N_real,
                    int K,
                    int tile_idx,
                    int dyn_sem_base) {
  int const lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  // NOTE page protocol: this early return skips the blanket page-free below,
  // so a task that actually bounds-fails would desync page_finished parity
  // and deadlock the next task on this slot. Unreachable at tiles_per_task=1
  // (task count == tile count, see header USAGE note); must be revisited if
  // tiles_per_task>1 is ever fixed.
  if (c.bounds_fail(tile_idx)) {
    return;
  }

  MPK_V2_PROF_SNAPSHOT()

  // Re-init the mma's async edges per the CHANNELS/ONESHOT reinit policy.
  if (lane_id == 0) {
    ::kernel::linear::reinit_for_role(::kernel::linear::Role::Mma,
                                      dyn_sem_base);
  }
  __syncwarp();

  // ── TMEM alloc + publish (v2-identical) ─────────────────────────────────
  int const scratch_smem_addr =
      smem + task_desc->smem_region_offset(::kernel::linear::REGION_SCRATCH);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          scratch_smem_addr),
      "r"(BLOCK_N * 2));
  int const taddr = *reinterpret_cast<int *>(
      smem_ptr +
      task_desc->smem_region_offset(::kernel::linear::REGION_SCRATCH));
  if (lane_id == 0) {
    mbarrier_arrive(dyn_sem_base + SEM_TMEM_READY * 8);
  }

  // ── Channels (sync) + rings (storage) + cursors ─────────────────────────
  WChan Wc;
  AChan Ac;
  WRing Wr;
  ARing Ar;
  make_wa(smem, dyn_sem_base, task_desc, Wc, Ac, Wr, Ar);
  mpk::ch::Consumer<WChan> cW{Wc};
  mpk::ch::Consumer<AChan> cA{Ac};
  // Both ph start at 0 — mirrors v2's `tma_phase = 0`.

  AccChan Acc = make_acc_channel(dyn_sem_base);
  mpk::ch::TmemProducer<AccChan> pAcc{Acc};
  pAcc.set_taddr(taddr);
  // pAcc.ph starts at 1 — mirrors v2's `epilogue_phase = 1` (pre-empty).

  int const total_g = c.tiles * c.iters;
  if (elect_sync()) {
    for (int t = 0; t < c.tiles; t++) {
      // Wait epilogue_mbar (TMEM column free); returns column for tile t.
      // (timed-waits below: elected lane == the mma-stall track writer.)
      int tmem;
      MPK_V2_TIMED_WAIT_IF(true,
                           V2_PROF_GROUP_MMA_STALL,
                           V2_PROF_EPILOGUE_WAIT,
                           tmem = pAcc.wait_free());

      for (int i = 0; i < c.iters; i++) {
        // Time only the W wait (representative; A shares the edge). Plain cA
        // wait stays a separate statement so the unprofiled expansion is
        // exactly `cW.wait_full(); cA.wait_full();` — textually identical to
        // baseline (sm100 is sensitive to branches around tcgen05 waits;
        // see sm100_branch_ima).
        MPK_V2_TIMED_WAIT_IF(t * c.iters + i < NUM_STAGES,
                             V2_PROF_GROUP_MMA_STALL,
                             V2_PROF_W_TMA_WAIT,
                             cW.wait_full()); // W_tma_mbar[stage]
        cA.wait_full();                       // A_tma_mbar[stage]
        int const W_smem = Wr.slot_addr(cW.st);
        int const A_smem = Ar.slot_addr(cA.st);

        tcgen05_fence_after_thread_sync();

        // Same descriptor math + tcgen05_mma sequence as v2. `i != 0` produces
        // identical PTX to v2's pass-through of `i` (setp.ne treats any nonzero
        // as accumulate).
        mma_k_block(tmem, W_smem, A_smem, /*accumulate=*/i != 0);

        int const stg = cW.st; // stage consumed this iter
        // Release SHARED mma_mbar ONCE per iter — matches v2's single commit.
        cW.release_mma();
        cA.advance();

        // Free this W stage's (dedicated) pages at its LAST use → the next
        // task's loader can TMA its weights into them while we finish later
        // stages. The last NUM_STAGES global iters visit each stage once.
        if constexpr (CROSS_TASK_PAGES) {
          if (t * c.iters + i >= total_g - NUM_STAGES) {
            Wr.release(
                stg, runtime_smem, mirage::runtime_v2::runtime_finish_page);
          }
        }
      }

      // Signal mainloop_mbar — async tcgen05.commit; compute waits this.
      pAcc.commit_mma();
    }
    // W stages never visited (only when total_g < NUM_STAGES) are freed here so
    // every W page is released exactly once. Empty (zero cost) when
    // total_g >= NUM_STAGES, which holds for every real linear; disjoint from
    // the in-loop releases above, so no double-free.
    if constexpr (CROSS_TASK_PAGES) {
      for (int s = total_g; s < NUM_STAGES; s++) {
        Wr.release(s, runtime_smem, mirage::runtime_v2::runtime_finish_page);
      }
    }
  }

  // Reconverge before freeing pages. The MMA loop ran only on the elected
  // lane; under Volta+ ITS the other lanes are not rejoined at the if-block
  // exit, so without this they could arrive page_finished while the elected
  // lane's MMA is still reading those pages — letting the next task's loader
  // TMA into them mid-MMA.
  __syncwarp();

  // Task-end page free, PARALLEL across lanes: each lane frees its own page
  // unless the W ring already freed it per-stage. When cross-task is off, the W
  // ring owns nothing (owns()==false) → this frees all 14 = baseline. A's pages
  // and scratch's (which shares one) are freed here at task end, which is safe
  // for scratch's whole-task lifetime.
  if (lane_id < MAX_SMEM_PAGES_PER_TASK && !Wr.owns(lane_id)) {
    mirage::runtime_v2::runtime_finish_page(runtime_smem, lane_id, 1);
  }
  __syncwarp();

  // Wait compute_done — one-shot, kept raw (matches v2).
  if (lane_id == 0) {
    MPK_V2_TIMED_WAIT_IF(true,
                         V2_PROF_GROUP_MMA_STALL,
                         V2_PROF_COMPUTE_DONE_WAIT,
                         mbarrier_wait(dyn_sem_base + SEM_COMPUTE_DONE * 8, 0));
  }
  __syncwarp();

  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(BLOCK_N * 2));
}

// ═══════════════════════════════════════════════════════════════════════════
// Consumer role (warps 0–3, 128 threads).
// ═══════════════════════════════════════════════════════════════════════════
template <bool HAS_RESIDUAL,
          int M_REAL = 16,
          int SPLIT_K = 1,
          int TILES_PER_TASK = 1>
__device__ __noinline__ void
    linear_compute_task(mirage::runtime::TaskDesc const *task_desc,
                        nv_bfloat16 *C_ptr,
                        nv_bfloat16 const *res_ptr,
                        int N_real,
                        int K,
                        int tile_idx,
                        float *workspace,
                        int dyn_sem_base) {
  int const warp_id = warp_uniform(threadIdx.x / WARP_SIZE);
  int const lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];

  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  if (c.bounds_fail(tile_idx)) {
    return;
  }

  MPK_V2_PROF_SNAPSHOT()

  // Wait mma's TMEM-addr publish — one-shot, kept raw (matches v2).
  // (timed-wait on thread 0 only — all four compute warps' lane 0 wait,
  // but the compute-stall track has a single designated writer.)
  if (lane_id == 0) {
    MPK_V2_TIMED_WAIT_IF(threadIdx.x == 0,
                         V2_PROF_GROUP_COMPUTE_STALL,
                         V2_PROF_TMEM_READY_WAIT,
                         mbarrier_wait(dyn_sem_base + SEM_TMEM_READY * 8, 0));
  }
  __syncwarp();
  int const taddr = *reinterpret_cast<int *>(
      smem_ptr +
      task_desc->smem_region_offset(::kernel::linear::REGION_SCRATCH));

  AccChan Acc = make_acc_channel(dyn_sem_base);
  mpk::ch::TmemConsumer<AccChan> cAcc{Acc};
  cAcc.set_taddr(taddr);
  // cAcc.ph starts at 0 — mirrors v2's `mainloop_phase = 0`.

  for (int t = 0; t < c.tiles; t++) {
    int const cur_tile_idx = tile_idx + t;
    int const cur_spatial_idx = cur_tile_idx % c.num_spatial_tiles;
    int const cur_k_slice = cur_tile_idx / c.num_spatial_tiles;
    int const bid_m = cur_spatial_idx;

    // Wait MMA done for this tile; cursor gives the TMEM column. All 128
    // threads wait; only thread 0 (the compute-stall writer) times it.
    int t_col;
    MPK_V2_TIMED_WAIT_IF(threadIdx.x == 0,
                         V2_PROF_GROUP_COMPUTE_STALL,
                         V2_PROF_MAINLOOP_WAIT,
                         t_col = cAcc.wait_full());
#ifdef MPK_ENABLE_PROFILING
    // RECONVERGE: in profiling builds the thread-0 timing branch diverges
    // warp 0 (Volta+ ITS doesn't rejoin at the merge) and the tcgen05.ld
    // below needs a converged warp. Unprofiled builds have no branch (the
    // macro is a bare wait), so no reconverge is needed.
    __syncwarp();
#endif

    tcgen05_fence_after_thread_sync();

    int const n_real = bid_m * BLOCK_M + warp_id * 32 + lane_id;
    if (n_real < N_real) {
      int const t_addr = (warp_id * 32 << 16) + t_col;

      float f[16];
      tcgen05_ld_16(f, t_addr);
      tcgen05_wait_ld();

      if constexpr (SPLIT_K == 1) {
        // Precision-clamp round-trip: bf16-quantize GEMM output before adding
        // residual (in float), then bf16-quantize again at store. v2 does this
        // exactly; do NOT collapse the round-trip — semantics change.
        if constexpr (HAS_RESIDUAL) {
#pragma unroll
          for (int m = 0; m < M_REAL; m++) {
            nv_bfloat16 gemm_bf16 = __float2bfloat16(f[m]);
            f[m] = __bfloat162float(gemm_bf16) +
                   __bfloat162float(res_ptr[m * N_real + n_real]);
          }
        }
#pragma unroll
        for (int m = 0; m < M_REAL; m++) {
          st_bf16(C_ptr + m * N_real + n_real, __float2bfloat16(f[m]));
        }
      } else {
        float *ws_base = workspace + cur_k_slice * M_REAL * N_real;
#pragma unroll
        for (int m = 0; m < M_REAL; m++) {
          st_f32(ws_base + m * N_real + n_real, f[m]);
        }
      }
    }

    // Release epilogue_mbar (128-thread sync arrival).
    cAcc.release_warp();
  }

  // Signal compute_done (128 threads, sync) — one-shot, kept raw.
  mbarrier_arrive(dyn_sem_base + SEM_COMPUTE_DONE * 8);
}

} // namespace linear_v2
} // namespace kernel
