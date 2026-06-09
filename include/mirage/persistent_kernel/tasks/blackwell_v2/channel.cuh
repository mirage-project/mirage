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

// Typed producer/consumer rings between warps. Synchronization and storage are
// separate primitives that compose:
//
//   Channel     - sync only: DEPTH stages of full[]/empty[] mbarriers.
//   SmemRing    - storage: per-stage SMEM offsets, and optionally page ids and
//                 per-stage page release for cross-task overlap.
//   TmemChannel - TMEM-backed accumulator. Its storage is linear in the stage
//                 (taddr + st*cols), so it keeps addressing inline instead of
//                 pairing with a ring.
//
// The producer/consumer cursors hold the per-warp stage, phase, and commit
// count. The cursor is the sole owner of the stage index `st`, which is what
// keeps the four role functions in step. A Channel cursor carries no storage;
// the role gets the SMEM address from the paired ring: ring.slot_addr(c.st).
//
//   - Each side is tagged sync (By::Warp) or async (By::Tma / By::Mma); the tag
//     records who arrives the barrier.
//   - Producer::drain() waits out any outstanding async empty arrivals before
//     teardown. linear doesn't use it (it re-inits at task start instead); it's
//     here for channels that do.
//
// mbarriers are addressed as SMEM byte addresses (int), stride 8, exactly like
// the existing v2 dynamic_semaphores convention (dyn_sem_base + ordinal*8).
// SmemRing slot offsets are per-stage, caller-populated, each 1024-aligned for
// TMA 128B-swizzle correctness.

#pragma once

#include "mirage/persistent_kernel/tasks/blackwell_v2/sm100_ptx.cuh"

namespace mpk {
namespace ch {

using ::kernel::sm100_ptx::mbar_init;
using ::kernel::sm100_ptx::mbar_wait;
using ::kernel::sm100_ptx::mbar_tx;
using ::kernel::sm100_ptx::tcgen05_commit;

// plain synchronous arrival (not in sm100_ptx)
__device__ __forceinline__ void mbar_arrive(int addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(addr)
               : "memory");
}

// 3D TMA load — bytes land in SMEM @ dst, arrives mbar on completion.
__device__ __forceinline__ void
cp_async_bulk_tensor_3d(int dst, void const *tmap, int x, int y, int z,
                        int mbar, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global."
               "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
               "[%0], [%1, {%2, %3, %4}], [%5], %6;"
               :
               : "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(mbar),
                 "l"(hint)
               : "memory");
}

enum class By { Warp, Tma, Mma };

// ────────────────────────────────────────────────────────────────────────────
// Channel — SYNCHRONIZATION primitive (mbarriers only). DEPTH stages of
// full[] / empty[] mbarrier pairs. Does NOT own storage — pair it with a
// SmemRing (below) or a TmemChannel for actual data.
//
// Separating sync from storage lets one Channel optionally talk to multiple
// storage views (e.g. a paired W+A pipeline sharing an empty edge) and lets
// page-lifetime concerns (which pages back which slot) live in SmemRing
// without leaking into the sync primitive.
// ────────────────────────────────────────────────────────────────────────────
template <int DEPTH, By PROD, By CONS>
struct Channel {
  int full;                  // SMEM byte addr of full[0]  (stride 8)
  int empty;                 // SMEM byte addr of empty[0] (stride 8)

  static constexpr int  depth      = DEPTH;
  static constexpr bool prod_async = (PROD != By::Warp);
  static constexpr bool cons_async = (CONS != By::Warp);

  __device__ int full_mbar (int s) const { return full  + s * 8; }
  __device__ int empty_mbar(int s) const { return empty + s * 8; }

  // controller warp, once per task. empty_arrivals = #arrivers of the empty
  // edge (1 for an MMA-commit release, 4*32 for a 128-thread warp release).
  __device__ void init(int empty_arrivals) const {
    for (int s = 0; s < DEPTH; s++) {
      mbar_init(full_mbar(s),  1);
      mbar_init(empty_mbar(s), empty_arrivals);
    }
  }

  // Re-init only the FULL mbars. Called by the producer at task-start to
  // clear any stale arrivals left on full[] by a prior occupant of this ring
  // slot (e.g. a late-landing TMA byte-delivery or tcgen05.commit). Producer
  // is the arriver of full[], so it owns clearing it. The matching consumer
  // hasn't issued any wait on these mbars yet (the producer's first arrive
  // is what unblocks the consumer's first wait), so the re-init is race-free.
  __device__ void reinit_full() const {
    for (int s = 0; s < DEPTH; s++) mbar_init(full_mbar(s), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }

  // Re-init only the EMPTY mbars (consumer's outbound).
  __device__ void reinit_empty(int empty_arrivals) const {
    for (int s = 0; s < DEPTH; s++) mbar_init(empty_mbar(s), empty_arrivals);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
};

// ---- Producer cursor (held by the warp that fills the channel) -------------
// The cursor owns the stage index `st` (single source of truth — prevents the
// four role functions from desyncing). It carries NO storage. To get the SMEM
// write address for the current stage, the role indexes a paired SmemRing:
//     ring.slot_addr(p.st)
// `st` is read-only from outside; only the cursor advances it.
template <class Ch>
struct Producer {
  Ch  ch;
  int st = 0;
  int ph = 1;          // pre-empty: slots start free; first DEPTH wait_frees pass
  int n_commits = 0;   // total commits since init; drain() uses this

  __device__ int full_mbar() const { return ch.full_mbar(st); }

  // Wait until the current slot is free to overwrite. Returns nothing — the
  // role looks up the SMEM address from the SmemRing at this stage.
  __device__ void wait_free() { mbar_wait(ch.empty_mbar(st), ph); }

  __device__ void advance() { if (++st == Ch::depth) { st = 0; ph ^= 1; } }

  // commit variants — pick by how the producer fills the slot.
  __device__ void commit_warp() { mbar_arrive   (ch.full_mbar(st)); advance(); ++n_commits; }
  __device__ void commit_mma()  { tcgen05_commit(ch.full_mbar(st)); advance(); ++n_commits; }
  // commit_tma(): the cp.async.bulk the role issued already carries full_mbar,
  // so the TMA engine arrives it asynchronously; we only advance the cursor.
  __device__ void commit_tma()  { advance(); ++n_commits; }

  // DRAIN: wait every outstanding empty arrival that this producer's loop did
  // not consume. Pre-empty consumes the first `depth` wait_frees, the loop
  // consumes (n_commits - depth) real releases when n_commits ≥ depth, leaving
  // exactly `depth` un-acquired releases at the end. When n_commits < depth
  // (small task), only `n_commits` releases ever happen — drain waits exactly
  // that many. Auto-correct for both cases.
  //
  // Works for both async and sync consumer releases:
  //   * cons_async: confirms async arrivals LANDED before the ring slot
  //     recycles — this is the stale-arrival hang fix.
  //   * cons_sync : ensures all consumer reads completed (e.g. TMEM dealloc
  //     is safe after this point).
  __device__ void drain() {
    int n = n_commits < Ch::depth ? n_commits : Ch::depth;
    for (int d = 0; d < n; d++) {
      mbar_wait(ch.empty_mbar(st), ph);
      advance();
    }
  }
};

// ---- Consumer cursor (held by the warp(s) that drain the channel) ----------
// Like Producer: owns `st`, carries no storage. Role gets the SMEM read
// address via ring.slot_addr(c.st), and frees the stage's pages
// via ring.release_pages(c.st, rt) at the release point.
template <class Ch>
struct Consumer {
  Ch  ch;
  int st = 0;
  int ph = 0;

  __device__ int full_mbar() const { return ch.full_mbar(st); }

  // Wait until the current slot has data. Returns nothing — role indexes the
  // SmemRing for the read address.
  __device__ void wait_full() { mbar_wait(ch.full_mbar(st), ph); }

  __device__ void advance() { if (++st == Ch::depth) { st = 0; ph ^= 1; } }

  __device__ void release_warp() { mbar_arrive   (ch.empty_mbar(st)); advance(); }
  __device__ void release_mma()  { tcgen05_commit(ch.empty_mbar(st)); advance(); }
};

// ────────────────────────────────────────────────────────────────────────────
// SmemRing — STORAGE primitive. Owns the per-stage SMEM offsets and
// (optionally) the physical page IDs backing each stage. Pairs with a Channel:
// the Channel synchronizes, the SmemRing addresses. slot_offsets / pages are
// caller-populated (from the planner via task_desc->smem_region_offset / page
// metadata) — no contiguous layout is assumed.
//
// SmemRing also owns the cross-task PAGE lifecycle for the
// pages this ring owns. Beyond the in-task full/empty edges (data-ready /
// slot-free), a physical page has an owner that spans tasks: task N+1 may not
// write it until task N frees it. The ring owns that edge for ITS pages:
//   * release(s)  — free stage s's pages; a page is freed only when its LAST
//                   owning stage releases it (refcounted, so packed sub-page
//                   stages that share a page are handled — partial release is a
//                   no-op until the page is fully done).
//   * acquire(s)  — wait the prior task's free of stage s's pages before writing.
//   * owns(pg)    — does this ring own physical page pg (so a task-end sweep can
//                   free the pages NO ring owns — scratch/unused — in parallel,
//                   each lane its own page, with no broadcast).
// PAGES_PER_SLOT==0 → storage-only ring; all page methods compile to nothing
// (owns()==false), so the task-end sweep frees every page = baseline behavior.
// Runtime-independent: the task passes the finish/wait functors.
template <int DEPTH, int PAGES_PER_SLOT = 0>
struct SmemRing {
  int slot_offsets[DEPTH];
  int pages[DEPTH][PAGES_PER_SLOT > 0 ? PAGES_PER_SLOT : 1];  // physical page ids

  static constexpr int depth          = DEPTH;
  static constexpr int pages_per_slot = PAGES_PER_SLOT;

  __device__ int slot_addr(int s) const { return slot_offsets[s]; }

  // Does this ring own physical page pg? (for the task-end sweep over pages no
  // ring owns). PAGES_PER_SLOT==0 → owns nothing.
  __device__ bool owns(int pg) const {
    if constexpr (PAGES_PER_SLOT > 0)
      for (int s = 0; s < DEPTH; s++)
        #pragma unroll
        for (int p = 0; p < PAGES_PER_SLOT; p++)
          if (pages[s][p] == pg) return true;
    return false;
  }

  // Free stage s's pages (this ring owns dedicated pages per stage → just free
  // them; no refcount needed). acquire(s) waits the prior task's free first.
  template <class RT, class FinishPageFn>
  __device__ void release(int s, RT *rt, FinishPageFn finish_page) const {
    if constexpr (PAGES_PER_SLOT > 0)
      #pragma unroll
      for (int p = 0; p < PAGES_PER_SLOT; p++) finish_page(rt, pages[s][p], 1);
  }

  template <class RT, class WaitPageFn>
  __device__ void acquire(int s, RT *rt, int instruction_index,
                          WaitPageFn wait_page_ready) const {
    if constexpr (PAGES_PER_SLOT > 0)
      #pragma unroll
      for (int p = 0; p < PAGES_PER_SLOT; p++)
        wait_page_ready(rt, pages[s][p], instruction_index);
  }
};

// ────────────────────────────────────────────────────────────────────────────
// TmemChannel — like Channel but the data buffer lives in TMEM, not SMEM.
// Mbarriers still live in SMEM. The cursor returns TMEM column addresses
// (taddr + st * cols_per_slot). taddr is set per-task after tcgen05.alloc.
// This removes the v3 leak where tasks read pAcc.stage() and did the column
// arithmetic themselves.
// ────────────────────────────────────────────────────────────────────────────
template <int SLOTS, By PROD, By CONS>
struct TmemChannel {
  int cols_per_slot;   // = TMEM columns per slot (e.g. BLOCK_N=16)
  int full;            // SMEM byte addr of full[0]  (stride 8)
  int empty;           // SMEM byte addr of empty[0] (stride 8)

  static constexpr int  slots      = SLOTS;
  static constexpr bool prod_async = (PROD != By::Warp);
  static constexpr bool cons_async = (CONS != By::Warp);

  __device__ int full_mbar (int s) const { return full  + s * 8; }
  __device__ int empty_mbar(int s) const { return empty + s * 8; }

  __device__ void init(int empty_arrivals) const {
    for (int s = 0; s < SLOTS; s++) {
      mbar_init(full_mbar(s),  1);
      mbar_init(empty_mbar(s), empty_arrivals);
    }
  }

  // Re-init only the FULL mbars (TmemProducer's outbound).
  __device__ void reinit_full() const {
    for (int s = 0; s < SLOTS; s++) mbar_init(full_mbar(s), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }

  // Re-init only the EMPTY mbars (TmemConsumer's outbound).
  __device__ void reinit_empty(int empty_arrivals) const {
    for (int s = 0; s < SLOTS; s++) mbar_init(empty_mbar(s), empty_arrivals);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
};

template <class TCh>
struct TmemProducer {
  TCh ch;
  int taddr = 0;       // TMEM base column — set after tcgen05.alloc
  int st = 0;
  int ph = 1;
  int n_commits = 0;

  __device__ void set_taddr(int t) { taddr = t; }

  __device__ int wait_free() {
    mbar_wait(ch.empty_mbar(st), ph);
    return taddr + st * ch.cols_per_slot;
  }
  __device__ int full_mbar() const { return ch.full_mbar(st); }
  __device__ void advance() { if (++st == TCh::slots) { st = 0; ph ^= 1; } }

  __device__ void commit_mma()  { tcgen05_commit(ch.full_mbar(st)); advance(); ++n_commits; }
  __device__ void commit_warp() { mbar_arrive   (ch.full_mbar(st)); advance(); ++n_commits; }

  __device__ void drain() {
    int n = n_commits < TCh::slots ? n_commits : TCh::slots;
    for (int d = 0; d < n; d++) {
      mbar_wait(ch.empty_mbar(st), ph);
      advance();
    }
  }
};

template <class TCh>
struct TmemConsumer {
  TCh ch;
  int taddr = 0;
  int st = 0;
  int ph = 0;

  __device__ void set_taddr(int t) { taddr = t; }

  __device__ int wait_full() {
    mbar_wait(ch.full_mbar(st), ph);
    return taddr + st * ch.cols_per_slot;
  }
  __device__ void advance() { if (++st == TCh::slots) { st = 0; ph ^= 1; } }

  __device__ void release_warp() { mbar_arrive   (ch.empty_mbar(st)); advance(); }
  __device__ void release_mma()  { tcgen05_commit(ch.empty_mbar(st)); advance(); }
};

} // namespace ch
} // namespace mpk
