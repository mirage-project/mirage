/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

// MPK v3 — linear (Qwen3 decode), Channel-based.
//
// Constants, SMEM region ordinals, SEM ordinals, and PTX wrappers all come
// from the shared `kernel::linear` namespace (linear_spec.h + linear_device.cuh)
// — the single source of truth. This file has NO dependency on linear_v2.
//
// Operational-sequence invariants (kept so behavior is well-defined and was
// validated byte-for-byte against the original v2 baseline during bring-up):
//     * SMEM layout: REGION_W_0..5 / REGION_A_0..5 / REGION_SCRATCH from
//       kernel::linear (linear_spec.h).
//     * Semaphore ordinals: SEM_W_TMA_BASE / SEM_A_TMA_BASE / SEM_MMA_BASE /
//       SEM_MAINLOOP_BASE / SEM_EPILOGUE_BASE / SEM_TMEM_READY /
//       SEM_CONSUMER_DONE from kernel::linear. mma_mbar is a single SHARED
//       empty edge for W and A (not split).
//     * PTX wrappers (tma_3d_load_l2, tcgen05_mma/commit, mbarrier_wait/
//       arrive/expect_tx) from kernel::linear (linear_device.cuh).
//     * mbar order: cp.async.bulk first, then expect_tx.
//     * LOAD-BEARING re-init at task start (mma/W_tma/A_tma in loader;
//       mainloop/epilogue/consumer_done in launcher) clears stray async
//       arrivals on reused ring slots. The end-of-loader drain was tried and
//       REMOVED 2026-05-30 (it caused an intermittent cross-op hang).
//
// What Channel buys here:
//   * Typed barrier addressing — pW.full_mbar() vs W_tma_mbar_base + s*8.
//   * Centralized phase tracking inside cursors — pW.ph (mma_phase, init 1,
//     flip on stage wrap); cW.ph (tma_phase, init 0); pAcc.ph (epilogue_phase,
//     init 1); cAcc.ph (mainloop_phase, init 0).
//
// Design (original "sync ≠ storage" split):
//   * Channel  = mbarriers only (full/empty).  Cursors (Producer/Consumer)
//     own the stage index `st` — the single source that keeps the four role
//     functions from desyncing. Cursors carry NO storage.
//   * SmemRing = per-stage SMEM offsets (+ optional page IDs for Phase-E
//     per-stage page release). The role indexes it with the cursor's stage:
//     `Wr.slot_addr(pW.st)`.
//
// What we hand-manage (doesn't fit a clean 1 Channel : 1 ring):
//   * W and A SHARE one empty edge (v2's mma_mbar). The launcher emits ONE
//     tcgen05.commit per K-iter that frees both. We model that by giving both
//     W and A channels the same `empty` mbar; only pW calls wait_free (one
//     wait/iter); A's storage addr comes from Ar.slot_addr(pA.st); pA advances
//     via commit_tma. Launcher: cW.release_mma emits the single commit;
//     cA.advance() (release-less) keeps the A cursor in lockstep.
//
// This file is now behaviorally v2-equivalent (re-inits + same op sequence),
// expressed via the Channel/SmemRing abstraction. No drain. Page release is a
// task-end blanket (byte-identical to v2); SmemRing's PAGES_PER_SLOT=0 keeps
// release_pages() a no-op. Phase E flips that on.
//
// USAGE: call with tiles_per_task=1. tiles_per_task>1 is a latent bug —
// partial-tile tasks (num_tiles % tpt != 0) leave inconsistent barrier state
// that deadlocks on cross-op slot reuse (the 2026-05-30 lm_head hang). tpt=1
// is also faster for decode shapes (better SM occupancy).

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/channel.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_device.cuh"

namespace kernel {
namespace linear_v3 {

// ── Single source of truth: kernel::linear (linear_spec.h + linear_device.cuh).
// v3 no longer depends on linear_sm100_v2.cuh — constants, ordinals, and PTX
// wrappers all come from kernel::linear, so v2 can be deleted independently.
using ::kernel::linear::WARP_SIZE;
using ::kernel::linear::BLOCK_M;
using ::kernel::linear::BLOCK_N;
using ::kernel::linear::BLOCK_K;
using ::kernel::linear::MMA_K;
using ::kernel::linear::NUM_STAGES;
using ::kernel::linear::W_SIZE;
using ::kernel::linear::A_SIZE;
using ::kernel::linear::SEM_W_TMA_BASE;
using ::kernel::linear::SEM_A_TMA_BASE;
using ::kernel::linear::SEM_MMA_BASE;
using ::kernel::linear::SEM_MAINLOOP_BASE;
using ::kernel::linear::SEM_EPILOGUE_BASE;
using ::kernel::linear::SEM_TMEM_READY;
using ::kernel::linear::SEM_CONSUMER_DONE;
using ::kernel::linear::CHANNELS;
using ::kernel::linear::CH_W;
using ::kernel::linear::CH_A;
using ::kernel::linear::elect_sync;
using ::kernel::linear::warp_uniform;
using ::kernel::linear::mbarrier_wait;
using ::kernel::linear::mbarrier_arrive_expect_tx;
using ::kernel::linear::mbarrier_arrive;
using ::kernel::linear::tcgen05_commit;
using ::kernel::linear::tcgen05_mma;
using ::kernel::linear::tma_3d_load_l2;
using ::kernel::linear::SMEM_DESC;
using ::kernel::linear::I_DESC;
using ::kernel::linear::L2_EVICT_FIRST;
using ::kernel::linear::L2_EVICT_LAST;
using ::kernel::linear::L2_EVICT_NORMAL;

using mpk::ch::By;

// ── Channel + ring type aliases (original design: sync ≠ storage) ───────────
// Channels carry ONLY mbarriers:
//   WChan/AChan: full = per-stream TMA-arrived; empty = SHARED mma_mbar.
//   AccChan: TMEM-backed, SLOTS=2 — DOUBLE-BUFFERED. mainloop_stage cycles %2,
//            alternating TMEM columns taddr+0 / taddr+BLOCK_N so tile t+1's MMA
//            overlaps the consumer's read of tile t. mainloop_mbar/epilogue_mbar
//            each have 2 slots (controller init + launcher re-init touch both).
//            cols_per_slot=BLOCK_N → 2*16=32 cols = the alloc.
// SmemRings carry storage (per-stage SMEM offsets). PAGES_PER_SLOT=0 for now:
//   page release stays a task-end blanket (byte-identical to v2). Flip to the
//   real page counts + call ring.release_pages() in the consumer to enable
//   Phase-E per-stage cross-task overlap.
using WChan   = mpk::ch::Channel    <NUM_STAGES, By::Tma, By::Mma>;
using AChan   = mpk::ch::Channel    <NUM_STAGES, By::Tma, By::Mma>;
using AccChan = mpk::ch::TmemChannel<2,          By::Mma, By::Warp>;

// W stage = W_SIZE/PAGE = 2 dedicated contiguous pages. When CROSS_TASK_PAGES
// is on, the W ring owns the per-stage cross-task page lifecycle (loader
// acquires / launcher releases page-by-page, so task N+1's loader TMAs into
// pages task N frees while N still computes its later stages). A stages are
// sub-page and packed (multiple A regions per physical page), so per-stage page
// control is unsafe for A — ARing stays 0 (its pages ride the task-end blanket).
using ::kernel::linear::CROSS_TASK_PAGES;
// W owns its 2 dedicated pages/stage and runs the cross-task page lifecycle
// (release per stage for overlap, acquire on first touch). A is storage-only:
// its pages — and scratch's, which shares one — are freed at task end by a
// PARALLEL sweep over the pages NO ring owns (each lane its own page). That
// keeps the frees parallel (not serialized on one lane) and needs no per-task
// counter. PAGES_PER_SLOT=0 when cross-task is off → the page methods vanish.
using WRing = mpk::ch::SmemRing<NUM_STAGES, CROSS_TASK_PAGES ? 2 : 0>;
using ARing = mpk::ch::SmemRing<NUM_STAGES>;

// ── Build channels (mbars) + rings (storage) from kernel::linear addresses ──
__device__ __forceinline__ void
make_wa(int smem, int dyn_sem_base,
        mirage::runtime::TaskDesc const *task_desc,
        WChan &Wc, AChan &Ac, WRing &Wr, ARing &Ar) {
  // Phase 3: edge wiring synthesized from CHANNELS (constexpr-folded; the
  // static_asserts in linear_spec.h guarantee these equal the old SEM_*_BASE
  // literals, so this is byte-identical to the hardcoded version).
  constexpr int w_full  = CHANNELS[CH_W].full_sem_base;
  constexpr int w_empty = CHANNELS[CH_W].empty_sem_base;
  constexpr int a_full  = CHANNELS[CH_A].full_sem_base;
  constexpr int a_sh    = CHANNELS[CH_A].shares_empty_with;  // 0 -> share W empty
  Wc.full  = dyn_sem_base + w_full  * 8;
  Wc.empty = dyn_sem_base + w_empty * 8;
  Ac.full  = dyn_sem_base + a_full  * 8;
  Ac.empty = (a_sh >= 0) ? Wc.empty                                  // SHARED
                         : dyn_sem_base + CHANNELS[CH_A].empty_sem_base * 8;

  // Per-stage SMEM offsets. W also records its 2 physical page ids so the ring
  // owns their cross-task lifecycle. A is storage-only.
  for (int s = 0; s < NUM_STAGES; s++) {
    Wr.slot_offsets[s] = smem + task_desc->smem_region_offset(
                                  ::kernel::linear::REGION_W_0 + s);
    Ar.slot_offsets[s] = smem + task_desc->smem_region_offset(
                                  ::kernel::linear::REGION_A_0 + s);
    if constexpr (CROSS_TASK_PAGES) {
      const int w_page0 = task_desc->smem_region_page(
                                    ::kernel::linear::REGION_W_0 + s);
      Wr.pages[s][0] = w_page0;
      Wr.pages[s][1] = w_page0 + 1;   // W_SIZE spans 2 contiguous pages
    }
  }
}

// TmemChannel stores only barrier addresses + cols_per_slot. taddr lives on
// the cursor (set via TmemProducer::set_taddr / TmemConsumer::set_taddr after
// the launcher's tcgen05.alloc publishes it).
__device__ __forceinline__ AccChan
make_acc_channel(int dyn_sem_base) {
  return AccChan{
      /*cols_per_slot =*/ BLOCK_N,
      /*full          =*/ dyn_sem_base + SEM_MAINLOOP_BASE * 8,
      /*empty         =*/ dyn_sem_base + SEM_EPILOGUE_BASE * 8,
  };
}

// ── Derived shape, computed once per role from identical inputs ────────────
// Every role used to recompute these — easy to drift if one diverges. One
// function, called the same way from all three roles, makes drift impossible.
struct TaskCtx {
  int num_spatial_tiles;   // grid_m (= N_real / BLOCK_M)
  int num_tiles;           // num_spatial_tiles * SPLIT_K
  int tiles;               // tiles_to_process this task (≤ TILES_PER_TASK)
  int iters;               // iters_per_slice (= K / BLOCK_K / SPLIT_K)

  __device__ bool bounds_fail(int tile_idx) const {
    return tile_idx >= num_tiles;
  }
};

template <int SPLIT_K, int TILES_PER_TASK>
__device__ __forceinline__ TaskCtx
ctx_from(int N_real, int K, int tile_idx) {
  TaskCtx c;
  c.num_spatial_tiles = N_real / BLOCK_M;
  c.num_tiles         = c.num_spatial_tiles * SPLIT_K;
  int left            = c.num_tiles - tile_idx;
  c.tiles             = (TILES_PER_TASK < left) ? TILES_PER_TASK : left;
  c.iters             = (K / BLOCK_K) / SPLIT_K;
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
    uint64_t a2 = SMEM_DESC |
                  (uint64_t)(((uint32_t)W_smem + k1 * BLOCK_M * 128) >> 4);
    uint64_t b2 = SMEM_DESC |
                  (uint64_t)(((uint32_t)A_smem + k1 * BLOCK_N * 128) >> 4);
    for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
      tcgen05_mma(tmem, a2, b2, I_DESC, 1);
      a2 += (32 >> 4);
      b2 += (32 >> 4);
    }
  }
}

// 16-register TMEM read (the giant inline asm). The `t_addr` layout is
// `(warp_id * 32 << 16) | t_col` — caller computes this, same as v2.
__device__ __forceinline__ void
tcgen05_ld_16(float (&out)[16], int t_addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(out[0]),  "=f"(out[1]),  "=f"(out[2]),  "=f"(out[3]),
        "=f"(out[4]),  "=f"(out[5]),  "=f"(out[6]),  "=f"(out[7]),
        "=f"(out[8]),  "=f"(out[9]),  "=f"(out[10]), "=f"(out[11]),
        "=f"(out[12]), "=f"(out[13]), "=f"(out[14]), "=f"(out[15])
      : "r"(t_addr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// Global stores with L1::no_allocate hint (matches v2's epilogue stores).
__device__ __forceinline__ void
st_bf16(nv_bfloat16 *dst, nv_bfloat16 v) {
  asm volatile("st.global.L1::no_allocate.b16 [%0], %1;"
               :: "l"(dst), "h"(*(uint16_t *)&v)
               : "memory");
}

__device__ __forceinline__ void
st_f32(float *dst, float v) {
  asm volatile("st.global.L1::no_allocate.b32 [%0], %1;"
               :: "l"(dst), "f"(v)
               : "memory");
}

// Prefetch a CUtensorMap descriptor into L1 (so the first TMA on it doesn't
// stall on descriptor fetch). Same instruction as v2's inline asm.
__device__ __forceinline__ void
prefetch_tensormap(void const *tmap) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(tmap));
}

// Acquire fence following a batch of mbarrier.init writes. Matches v2.
__device__ __forceinline__ void
mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// Per-thread fence required between thread-local register reads and tcgen05
// operations that consume them. Same instruction v2 uses.
__device__ __forceinline__ void
tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// ═══════════════════════════════════════════════════════════════════════════
// Loader role (warp 4, elected lane only) — TMA loop + start-of-task re-init.
// ═══════════════════════════════════════════════════════════════════════════
template <int SPLIT_K = 1, int W_L2_HINT = 0, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_loader_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    mirage::runtime::RuntimeConfig const &runtime_config,
    CUtensorMap const *W_tmap_ptr,
    CUtensorMap const *A_tmap_ptr,
    int N_real, int K,
    int tile_idx,
    int instruction_index,
    int iter_num,
    int dyn_sem_base) {
  if (!elect_sync()) return;

  prefetch_tensormap(W_tmap_ptr);
  prefetch_tensormap(A_tmap_ptr);

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

  // Shape — one place, used by all three roles.
  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  if (c.bounds_fail(tile_idx)) return;

  MPK_V2_PROF_SNAPSHOT()

  constexpr uint64_t W_HINT =
      (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;

  // ── Channels (sync) + rings (storage) + cursors ─────────────────────────
  WChan Wc; AChan Ac; WRing Wr; ARing Ar;
  make_wa(smem, dyn_sem_base, task_desc, Wc, Ac, Wr, Ar);

  // Loader re-init from CHANNELS reinit_*_by policy (table-driven, Phase 2b).
  ::kernel::linear::reinit_for_role(::kernel::linear::Role::Loader, dyn_sem_base);

  mpk::ch::Producer<WChan> pW{Wc};
  mpk::ch::Producer<AChan> pA{Ac};
  // Both ph start at 1 (pre-empty). pW.ph mirrors v2's `mma_phase`.

  bool dep_done = false;

  for (int t = 0; t < c.tiles; t++) {
    const int cur_tile_idx    = tile_idx + t;
    const int cur_spatial_idx = cur_tile_idx % c.num_spatial_tiles;
    const int cur_k_slice     = cur_tile_idx / c.num_spatial_tiles;
    const int cur_k_start     = cur_k_slice * c.iters;
    const int cur_off_m       = cur_spatial_idx * BLOCK_M;

    for (int i = 0; i < c.iters; i++) {
      const int iter_k  = cur_k_start + i;
      const int z_coord = iter_k * (BLOCK_K / 64);

      // Cross-task page acquire on FIRST touch of each W stage (first NUM_STAGES
      // global iters each visit a distinct stage once): wait the prior task's
      // release of this stage's W pages before TMA-ing into them — this overlaps
      // THIS task's W loads with the PRIOR task's compute. Steady-state reuse
      // after that rides the in-task empty edge below.
      if constexpr (CROSS_TASK_PAGES) {
        if (t * c.iters + i < NUM_STAGES)
          Wr.acquire(pW.st, runtime_smem, instruction_index,
                     mirage::runtime_v2::runtime_wait_page_ready);
      }

      // Wait shared empty (mma_mbar[stage]) — one wait per iter, covers both.
      // (timed-wait on the FIRST ring lap only: that is where the cold-start
      // exposure lives; timing every K-iter measurably slowed the kernel.)
#ifdef MPK_ENABLE_PROFILING
      if (t * c.iters + i < NUM_STAGES) {
        MPK_V2_TIMED_WAIT(V2_PROF_GROUP_LOADER_PHASE, V2_PROF_MMA_EMPTY_WAIT,
                          pW.wait_free());
      } else {
        pW.wait_free();
      }
#else
      pW.wait_free();
#endif
      const int W_smem = Wr.slot_addr(pW.st);   // storage addr from the ring

      // W TMA — v2 order: cp.async.bulk first, expect_tx after.
      tma_3d_load_l2(W_smem, W_tmap_ptr, 0, cur_off_m, z_coord,
                     pW.full_mbar(), W_HINT);
      mbarrier_arrive_expect_tx(pW.full_mbar(), W_SIZE);

      // Cross-SM dep wait once (gates A — matches v2's prefetch pattern).
      if (!dep_done) {
        mirage::runtime_v2::wait_task_dependency(
            runtime_config, task_desc, iter_num);
        dep_done = true;
      }

      // A TMA — A shares the empty edge (already waited via pW), so no separate
      // wait. Storage addr comes from A's ring at A's cursor stage.
      const int A_smem = Ar.slot_addr(pA.st);
      tma_3d_load_l2(A_smem, A_tmap_ptr, 0, 0, z_coord,
                     pA.full_mbar(), L2_EVICT_LAST);
      mbarrier_arrive_expect_tx(pA.full_mbar(), A_SIZE);

      // Advance both cursors in lockstep; track commit count for drain.
      pW.commit_tma();   // advance() + ++n_commits
      pA.commit_tma();
    }
  }

  // ── NO DRAIN ─────────────────────────────────────────────────────────────
  // The end-of-loader pW.drain() was REMOVED 2026-05-30: it caused
  // an intermittent cross-op deadlock (the loader blocking on the launcher's
  // final mma_mbar — the SHARED W/A empty edge — commits at task end tangles
  // with cross-op ring-slot reuse). The load-bearing stale-arrival defense is
  // the start-of-task re-init of mma/W_tma/A_tma (loader) + mainloop/epilogue/
  // consumer_done (launcher), NOT the drain. Removing it: 14/14 demo runs
  // clean (was ~1/3 hang), tokens correct, latency unchanged. The earlier
  // belief that the drain was the structural fix was wrong; the re-inits are.
}

// ═══════════════════════════════════════════════════════════════════════════
// Launcher role (warp 5, all 32 lanes — alloc/dealloc are sync.aligned).
// ═══════════════════════════════════════════════════════════════════════════
template <int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_launcher_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    int N_real, int K, int tile_idx, int dyn_sem_base) {
  const int lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  // NOTE page protocol: this early return skips the blanket page-free below,
  // so a task that actually bounds-fails would desync page_finished parity
  // and deadlock the next task on this slot. Unreachable at tiles_per_task=1
  // (task count == tile count, see header USAGE note); must be revisited if
  // tiles_per_task>1 is ever fixed.
  if (c.bounds_fail(tile_idx)) return;

  MPK_V2_PROF_SNAPSHOT()

  // Launcher re-init from CHANNELS/ONESHOT reinit_*_by policy (table-driven, Phase 2b).
  if (lane_id == 0) {
    ::kernel::linear::reinit_for_role(::kernel::linear::Role::Launcher,
                                      dyn_sem_base);
  }
  __syncwarp();

  // ── TMEM alloc + publish (v2-identical) ─────────────────────────────────
  const int scratch_smem_addr = smem + task_desc->smem_region_offset(
                                  ::kernel::linear::REGION_SCRATCH);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(scratch_smem_addr), "r"(BLOCK_N * 2));
  const int taddr = *reinterpret_cast<int *>(
      smem_ptr + task_desc->smem_region_offset(
          ::kernel::linear::REGION_SCRATCH));
  if (lane_id == 0) {
    mbarrier_arrive(dyn_sem_base + SEM_TMEM_READY * 8);
  }

  // ── Channels (sync) + rings (storage) + cursors ─────────────────────────
  WChan Wc; AChan Ac; WRing Wr; ARing Ar;
  make_wa(smem, dyn_sem_base, task_desc, Wc, Ac, Wr, Ar);
  mpk::ch::Consumer<WChan> cW{Wc};
  mpk::ch::Consumer<AChan> cA{Ac};
  // Both ph start at 0 — mirrors v2's `tma_phase = 0`.

  AccChan Acc = make_acc_channel(dyn_sem_base);
  mpk::ch::TmemProducer<AccChan> pAcc{Acc};
  pAcc.set_taddr(taddr);
  // pAcc.ph starts at 1 — mirrors v2's `epilogue_phase = 1` (pre-empty).

  const int total_g = c.tiles * c.iters;
  if (elect_sync()) {
    for (int t = 0; t < c.tiles; t++) {
      // Wait epilogue_mbar (TMEM column free); returns column for tile t.
      // (timed-waits below: elected lane == the launcher-phase track writer.)
#ifdef MPK_ENABLE_PROFILING
      int tmem;
      MPK_V2_TIMED_WAIT(V2_PROF_GROUP_LAUNCHER_PHASE, V2_PROF_EPILOGUE_WAIT,
                        tmem = pAcc.wait_free());
#else
      const int tmem = pAcc.wait_free();
#endif

      for (int i = 0; i < c.iters; i++) {
#ifdef MPK_ENABLE_PROFILING
        // timed-wait on the first ring lap only (see loader note above).
        // NOTE: this construct must NOT exist in unprofiled builds — an
        // if/else with identical arms here produced an illegal memory
        // access on sm_100a (codegen sensitivity around branch +
        // mbarrier-wait + tcgen05; bisected 2026-06-06).
        if (t * c.iters + i < NUM_STAGES) {
          MPK_V2_TIMED_WAIT(V2_PROF_GROUP_LAUNCHER_PHASE, V2_PROF_W_TMA_WAIT,
                            (cW.wait_full(), cA.wait_full()));
        } else {
          cW.wait_full();
          cA.wait_full();
        }
#else
        cW.wait_full();                       // W_tma_mbar[stage]
        cA.wait_full();                       // A_tma_mbar[stage]
#endif
        const int W_smem = Wr.slot_addr(cW.st);
        const int A_smem = Ar.slot_addr(cA.st);

        tcgen05_fence_after_thread_sync();

        // Same descriptor math + tcgen05_mma sequence as v2. `i != 0` produces
        // identical PTX to v2's pass-through of `i` (setp.ne treats any nonzero
        // as accumulate).
        mma_k_block(tmem, W_smem, A_smem, /*accumulate=*/ i != 0);

        const int stg = cW.st;                // stage consumed this iter
        // Release SHARED mma_mbar ONCE per iter — matches v2's single commit.
        cW.release_mma();
        cA.advance();

        // Free this W stage's (dedicated) pages at its LAST use → the next
        // task's loader can TMA its weights into them while we finish later
        // stages. The last NUM_STAGES global iters visit each stage once.
        if constexpr (CROSS_TASK_PAGES) {
          if (t * c.iters + i >= total_g - NUM_STAGES)
            Wr.release(stg, runtime_smem, mirage::runtime_v2::runtime_finish_page);
        }
      }

      // Signal mainloop_mbar — async tcgen05.commit; consumer waits this.
      pAcc.commit_mma();
    }
    // W stages never visited (only when total_g < NUM_STAGES) are freed here so
    // every W page is released exactly once. Empty (zero cost) when
    // total_g >= NUM_STAGES, which holds for every real linear; disjoint from
    // the in-loop releases above, so no double-free.
    if constexpr (CROSS_TASK_PAGES) {
      for (int s = total_g; s < NUM_STAGES; s++)
        Wr.release(s, runtime_smem, mirage::runtime_v2::runtime_finish_page);
    }
  }

  // RECONVERGE before freeing pages. The MMA loop above runs only on the
  // elected lane; on Volta+ independent thread scheduling the non-elected lanes
  // are NOT reconverged at the end of the if-block, so without this __syncwarp
  // they arrive page_finished for their pages while the elected lane's MMA is
  // still reading W/A from those pages — the next tasks' loaders then race far
  // ahead on the page pipeline (and can TMA into pages mid-MMA). THE verified
  // fix for the intermittent page-parity deadlock: baseline hangs ~1/12 runs;
  // this __syncwarp alone is 40/40 clean (the controller-side proxy fence
  // alone was NOT sufficient — still hung).
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

  // Wait consumer_done — one-shot, kept raw (matches v2).
  if (lane_id == 0) {
#ifdef MPK_ENABLE_PROFILING
    MPK_V2_TIMED_WAIT(V2_PROF_GROUP_LAUNCHER_PHASE,
                      V2_PROF_CONSUMER_DONE_WAIT,
                      mbarrier_wait(dyn_sem_base + SEM_CONSUMER_DONE * 8, 0));
#else
    mbarrier_wait(dyn_sem_base + SEM_CONSUMER_DONE * 8, 0);
#endif
  }
  __syncwarp();

  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :: "r"(taddr), "r"(BLOCK_N * 2));
}

// ═══════════════════════════════════════════════════════════════════════════
// Consumer role (warps 0–3, 128 threads).
// ═══════════════════════════════════════════════════════════════════════════
template <bool HAS_RESIDUAL, int M_REAL = 16,
          int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_consumer_task(
    mirage::runtime::TaskDesc const *task_desc,
    nv_bfloat16       *C_ptr,
    nv_bfloat16 const *res_ptr,
    int N_real, int K,
    int tile_idx,
    float *workspace,
    int dyn_sem_base) {
  const int warp_id = warp_uniform(threadIdx.x / WARP_SIZE);
  const int lane_id = threadIdx.x & 31;

  extern __shared__ __align__(1024) char smem_ptr[];

  const TaskCtx c = ctx_from<SPLIT_K, TILES_PER_TASK>(N_real, K, tile_idx);
  if (c.bounds_fail(tile_idx)) return;

  MPK_V2_PROF_SNAPSHOT()

  // Wait launcher's TMEM-addr publish — one-shot, kept raw (matches v2).
  // (timed-wait on thread 0 only — all four consumer warps' lane 0 wait,
  // but the consumer-phase track has a single designated writer.)
  if (lane_id == 0) {
#ifdef MPK_ENABLE_PROFILING
    if (threadIdx.x == 0) {
      MPK_V2_TIMED_WAIT(V2_PROF_GROUP_CONSUMER_PHASE,
                        V2_PROF_TMEM_READY_WAIT,
                        mbarrier_wait(dyn_sem_base + SEM_TMEM_READY * 8, 0));
    } else {
      mbarrier_wait(dyn_sem_base + SEM_TMEM_READY * 8, 0);
    }
#else
    mbarrier_wait(dyn_sem_base + SEM_TMEM_READY * 8, 0);
#endif
  }
  __syncwarp();
  const int taddr = *reinterpret_cast<int *>(
      smem_ptr + task_desc->smem_region_offset(
          ::kernel::linear::REGION_SCRATCH));

  AccChan Acc = make_acc_channel(dyn_sem_base);
  mpk::ch::TmemConsumer<AccChan> cAcc{Acc};
  cAcc.set_taddr(taddr);
  // cAcc.ph starts at 0 — mirrors v2's `mainloop_phase = 0`.

  for (int t = 0; t < c.tiles; t++) {
    const int cur_tile_idx    = tile_idx + t;
    const int cur_spatial_idx = cur_tile_idx % c.num_spatial_tiles;
    const int cur_k_slice     = cur_tile_idx / c.num_spatial_tiles;
    const int bid_m           = cur_spatial_idx;

    // Wait MMA done for this tile; cursor gives the TMEM column. All 128
    // threads wait; only thread 0 (the consumer-phase writer) times it.
#ifdef MPK_ENABLE_PROFILING
    // All 128 threads wait; thread 0 (the consumer-phase writer) times it.
    int t_col;
    if (threadIdx.x == 0) {
      MPK_V2_TIMED_WAIT(V2_PROF_GROUP_CONSUMER_PHASE, V2_PROF_MAINLOOP_WAIT,
                        t_col = cAcc.wait_full());
    } else {
      t_col = cAcc.wait_full();
    }
    // RECONVERGE: the branch diverges warp 0 (Volta+ ITS does not rejoin at
    // the merge) and tcgen05.ld.sync.aligned below needs a converged warp.
    __syncwarp();
#else
    const int t_col = cAcc.wait_full();
#endif

    tcgen05_fence_after_thread_sync();

    const int n_real = bid_m * BLOCK_M + warp_id * 32 + lane_id;
    if (n_real < N_real) {
      const int t_addr = (warp_id * 32 << 16) + t_col;

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

  // Signal consumer_done (128 threads, sync) — one-shot, kept raw.
  mbarrier_arrive(dyn_sem_base + SEM_CONSUMER_DONE * 8);
}

} // namespace linear_v3
} // namespace kernel
