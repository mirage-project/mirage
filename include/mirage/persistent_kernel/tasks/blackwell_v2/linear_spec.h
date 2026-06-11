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

// The linear task's shape spec: tile constants, SMEM region ordinals,
// op-private semaphore ordinals, host-safe MMA descriptor constants, and the
// planner feed (make_smem_info). It is host-includable (no __device__ PTX) so
// task_register.cc can call make_smem_info(), and the device side
// (linear_ptx.cuh / linear_sm100_v2.cuh) includes it too, so host and device
// share one set of constants. Declare these here, not in the .cuh.

#pragma once

#include <cstdint>

#include "mirage/kernel/task_register.h" // mirage::runtime::TaskSmemInfo
#include <string>

namespace kernel {
namespace linear {

// ── Tile shape ──────────────────────────────────────────────────────────────
inline constexpr int WARP_SIZE = 32;
inline constexpr int BLOCK_M = 128;
inline constexpr int BLOCK_N = 16;
inline constexpr int BLOCK_K = 128;
inline constexpr int MMA_K = 16;
inline constexpr int NUM_STAGES = 6;

// Cross-task page pipeline (loader acquires / mma releases W pages per
// stage so task N+1's loader can TMA into pages task N frees — see
// channel.cuh SmemRing acquire_pages/release_pages). The MECHANISM is complete
// and verified (correct, deadlock-free), but it only PAYS OFF with two
// co-requisites that don't exist yet: (1) per-task footprint <= half the
// 14-page pool (NUM_STAGES<=3) so two linear tasks coexist, and (2) the SMEM
// planner double-buffering page assignments so consecutive tasks get DISJOINT
// pages. Without those, N and N+1 share all 14 pages and acquire/release only
// adds sync (measured ~+12%). Default OFF (task-end blanket release, baseline
// perf); flip to true once (1)+(2) land.
inline constexpr bool CROSS_TASK_PAGES = false;

inline constexpr int t_size_bytes = 2;                          // bf16
inline constexpr int W_SIZE = BLOCK_M * BLOCK_K * t_size_bytes; // 32768
inline constexpr int A_SIZE = BLOCK_N * BLOCK_K * t_size_bytes; //  4096

// ── SMEM region ordinals (must match make_smem_info push order below) ───────
inline constexpr int REGION_W_0 = 0;
inline constexpr int REGION_A_0 = NUM_STAGES;          // 6
inline constexpr int REGION_SCRATCH = 2 * NUM_STAGES;  // 12
inline constexpr int NUM_REGIONS = 2 * NUM_STAGES + 1; // 13
inline constexpr int SCRATCH_BYTES = 16;

// ── Op-private semaphore ordinals (relative to dyn_sem_base / SEM_OP_BASE) ──
//   [+0  ..+5 ]  W_tma_mbar      (count=1,            loader→mma)
//   [+6  ..+11]  A_tma_mbar      (count=1,            loader→mma)
//   [+12 ..+17]  mma_mbar        (count=1,            mma→loader, shared W/A
//   empty)
//   [+18 ..+19]  mainloop_mbar   (count=1,            mma→compute)
//   [+20 ..+21]  epilogue_mbar   (count=4*WARP_SIZE,  compute→mma)
//   [+22      ]  tmem_ready      (count=1,            mma→compute)
//   [+23      ]  compute_done   (count=4*WARP_SIZE,  compute→mma)
inline constexpr int SEM_W_TMA_BASE = 0;
inline constexpr int SEM_A_TMA_BASE = NUM_STAGES;            // 6
inline constexpr int SEM_MMA_BASE = 2 * NUM_STAGES;          // 12
inline constexpr int SEM_MAINLOOP_BASE = 3 * NUM_STAGES;     // 18
inline constexpr int SEM_EPILOGUE_BASE = 3 * NUM_STAGES + 2; // 20
inline constexpr int SEM_TMEM_READY = 3 * NUM_STAGES + 4;    // 22
inline constexpr int SEM_COMPUTE_DONE = 3 * NUM_STAGES + 5;  // 23
inline constexpr int NUM_OP_SEMS = 3 * NUM_STAGES + 6;       // 24

// ── MMA descriptors (host-safe constexpr; no PTX) ───────────────────────────
inline constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3'FFFFULL) >> 4ULL;
}
inline constexpr uint64_t SMEM_DESC =
    (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
inline constexpr uint32_t I_DESC = (1U << 4U) | (1U << 7U) | (1U << 10U) |
                                   ((uint32_t)BLOCK_N >> 3U << 17U) |
                                   ((uint32_t)BLOCK_M >> 4U << 24U);

// ── L2 cache-policy hints ───────────────────────────────────────────────────
inline constexpr uint64_t L2_EVICT_FIRST = 0x12F0000000000000ULL;
inline constexpr uint64_t L2_EVICT_LAST = 0x14F0000000000000ULL;
inline constexpr uint64_t L2_EVICT_NORMAL = 0x16F0000000000000ULL;

// ── Capacity check ──────────────────────────────────────────────────────────
inline constexpr int total_smem_bytes() {
  return NUM_STAGES * (W_SIZE + A_SIZE) + SCRATCH_BYTES;
}
inline constexpr int PLANNER_CAPACITY_BYTES = 225 * 1024 - 6 * 1024; // 224256
static_assert(total_smem_bytes() <= PLANNER_CAPACITY_BYTES,
              "linear SMEM footprint exceeds planner CAPACITY_BYTES");

// ── Planner feed (host). Region push order defines the REGION_* ordinals. ───
inline ::mirage::runtime::TaskSmemInfo make_smem_info() {
  ::mirage::runtime::TaskSmemInfo info{total_smem_bytes(),
                                       /*alignment=*/1024,
                                       {}};
  // 6 W regions: each exactly 2 pages, contiguous, NOT packable.
  // release_step = s+1: stage s is consumed (and thus freeable) before stage
  // s+1 in the pipeline. Distinct per stage so the planner's release-order
  // chaining (v2_smem_planner._release_order) is non-degenerate. This is a
  // planner-side ordering hint only — the device runtime reads
  // physical_page_start and any assignment is a valid permutation, so it
  // changes WHICH pages back a stage, not behavior. (Real per-stage release at
  // runtime is step two.)
  for (int s = 0; s < NUM_STAGES; s++) {
    info.regions.push_back({"linear_W_" + std::to_string(s),
                            W_SIZE,
                            /*alignment=*/1024,
                            /*page_count=*/-1,
                            /*can_pack=*/false,
                            /*release_step=*/s + 1,
                            /*contiguous=*/true});
  }
  // 6 A regions: each 4 KB (sub-page), packable → planner shares pages.
  // A stage s is consumed alongside W stage s → same release_step (s+1).
  for (int s = 0; s < NUM_STAGES; s++) {
    info.regions.push_back({"linear_A_" + std::to_string(s),
                            A_SIZE,
                            /*alignment=*/1024,
                            /*page_count=*/-1,
                            /*can_pack=*/true,
                            /*release_step=*/s + 1,
                            /*contiguous=*/true});
  }
  // Linear-private cross-warp scratch (TMEM allocation address). 16 B,
  // packable. Lives the whole task → released last (highest release_step).
  info.regions.push_back({"linear_scratch",
                          SCRATCH_BYTES,
                          /*alignment=*/16,
                          /*page_count=*/-1,
                          /*can_pack=*/true,
                          /*release_step=*/NUM_STAGES + 1,
                          /*contiguous=*/true});
  return info;
}

// ── Channel / dataflow declaration (host-readable) ──────────────────────────
// The linear op's producer→compute pipeline, declared as DATA the planner /
// codegen can read — instead of being wired only in device code (make_wa).
//
// CONSUMED by three places (the table is live, not documentation):
//   * linear_ptx.cuh linear_init()      — dispatcher's structural mbar init
//   * linear_ptx.cuh reinit_for_role()  — per-role start-of-task re-init of
//     async edges (clears prior-slot strays; placement per reinit_*_by)
//   * linear_sm100_v2.cuh make_wa()        — wires the Channel cursors' full/
//     empty addresses (incl. A sharing W's mma empty edge)
// Future: derive cross-task page-release timing from the release edge.
//
// `Arrival` mirrors mpk::ch::By (Warp = sync warp arrival; Tma/Mma = async
// engine arrival, i.e. the edges that need a start-of-task re-init).
enum class Arrival { Warp, Tma, Mma };
enum class Role { Loader, Mma, Consumer, None };

struct ChannelSpec {
  char const *name;
  int depth; // ring stages (SMEM) or TMEM slots
  Role producer;
  Arrival producer_kind; // fills full[] (how it arrives)
  Role compute;
  Arrival compute_kind;    // frees empty[] = the release edge
  int full_sem_base;       // SEM ordinal of full[0]
  int empty_sem_base;      // SEM ordinal of empty[0] (release edge)
  int full_arrivals;       // mbar_init count for full[]
  int empty_arrivals;      // mbar_init count for empty[]
  int storage_region_base; // REGION_* base, or -1 if TMEM-backed
  int shares_empty_with;   // index of channel sharing this empty edge, or -1
  bool tmem;               // storage in TMEM (not an SMEM ring)
  // Start-of-task re-init owner per edge (the role that touches the edge FIRST,
  // clearing prior-slot async strays). Role::None = no role re-init (dispatcher
  // init suffices). Encodes the proven re-init placement as data.
  Role reinit_full_by;
  Role reinit_empty_by;
};

inline constexpr ChannelSpec CHANNELS[] = {
    // W: loader TMA-fills full (W_tma); mma MMA-frees the SHARED mma edge.
    //   loader re-inits both W.full and the shared mma empty (it waits mma
    //   first via pre-empty).
    {"W",
     NUM_STAGES,
     Role::Loader,
     Arrival::Tma,
     Role::Mma,
     Arrival::Mma,
     SEM_W_TMA_BASE,
     SEM_MMA_BASE,
     /*full=*/1,
     /*empty=*/1,
     REGION_W_0,
     /*shares_empty_with=*/-1,
     /*tmem=*/false,
     /*reinit_full_by=*/Role::Loader,
     /*reinit_empty_by=*/Role::Loader},
    // A: same producer/compute; shares W's mma_mbar empty edge (index 0), so
    //   its empty re-init is covered by W's loader re-init (reinit_empty=None).
    {"A",
     NUM_STAGES,
     Role::Loader,
     Arrival::Tma,
     Role::Mma,
     Arrival::Mma,
     SEM_A_TMA_BASE,
     SEM_MMA_BASE,
     1,
     1,
     REGION_A_0,
     /*shares_empty_with=*/0,
     false,
     /*reinit_full_by=*/Role::Loader,
     /*reinit_empty_by=*/Role::None},
    // Acc: mma MMA-commits full (mainloop); 128 compute threads free
    // empty (epilogue). TMEM-backed, double-buffered (2 slots). Mma
    // re-inits both (it produces full and waits epilogue).
    {"Acc",
     2,
     Role::Mma,
     Arrival::Mma,
     Role::Consumer,
     Arrival::Warp,
     SEM_MAINLOOP_BASE,
     SEM_EPILOGUE_BASE,
     1,
     4 * WARP_SIZE,
     /*storage_region_base=*/-1,
     /*shares_empty_with=*/-1,
     /*tmem=*/true,
     /*reinit_full_by=*/Role::Mma,
     /*reinit_empty_by=*/Role::Mma},
};
inline constexpr int NUM_CHANNELS = sizeof(CHANNELS) / sizeof(CHANNELS[0]);

// Channel indices (for device wiring that needs a specific channel).
inline constexpr int CH_W = 0, CH_A = 1, CH_ACC = 2;

// One-shot handshake semaphores (not ring channels): single arrive→wait.
struct OneShotSem {
  char const *name;
  int sem;
  int arrivals;
  Role reinit_by; // role that re-inits at task start (None = dispatcher only)
};
inline constexpr OneShotSem ONESHOT_SEMS[] = {
    // tmem_ready: compute waits it at task start (concurrent with mma),
    // so NO role re-init — re-initing it would race the compute's wait.
    {"tmem_ready", SEM_TMEM_READY, 1, Role::None}, // mma→compute
    // compute_done: mma waits it; re-init by mma before it publishes
    // tmem_ready (ordered before any compute arrival). LOAD-BEARING.
    {"compute_done", SEM_COMPUTE_DONE, 4 * WARP_SIZE, Role::Mma}, // compute→mma
};
inline constexpr int NUM_ONESHOT_SEMS =
    sizeof(ONESHOT_SEMS) / sizeof(ONESHOT_SEMS[0]);

// ── Consistency: the declared graph must reproduce exactly the op-private SEM
// budget. Catches drift if NUM_STAGES / ordinals / counts change without the
// table being updated (the layout-duplication footgun this whole table fixes).
inline constexpr int declared_sem_count() {
  int n = 0;
  for (auto const &c : CHANNELS) {
    n += c.depth; // full[] mbars
    if (c.shares_empty_with < 0) {
      n += c.depth; // empty[] (unless shared)
    }
  }
  n += NUM_ONESHOT_SEMS;
  return n;
}
static_assert(declared_sem_count() == NUM_OP_SEMS,
              "declared channel graph must cover exactly NUM_OP_SEMS sems");
static_assert(CHANNELS[1].empty_sem_base == CHANNELS[0].empty_sem_base,
              "A must share W's mma_mbar empty edge");
static_assert(CHANNELS[0].full_sem_base == SEM_W_TMA_BASE &&
                  CHANNELS[1].full_sem_base == SEM_A_TMA_BASE &&
                  CHANNELS[2].full_sem_base == SEM_MAINLOOP_BASE &&
                  CHANNELS[2].empty_sem_base == SEM_EPILOGUE_BASE,
              "channel ordinals must match the SEM_* constants");
// storage_region_base is descriptive only — make_wa() addresses storage via
// the REGION_* ordinals directly. Assert the table can't drift from them.
static_assert(CHANNELS[0].storage_region_base == REGION_W_0 &&
                  CHANNELS[1].storage_region_base == REGION_A_0 &&
                  CHANNELS[2].storage_region_base == -1,
              "channel storage bases must match the REGION_* constants");

} // namespace linear
} // namespace kernel
