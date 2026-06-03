/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the linear_sm100_v2 SMEM region declaration
// fed to the planner.
//
// Phase 3.2/3.3: linear declares TWO regions per pipeline stage — one for
// the weight tile W (32 KB, exactly 2 pages) and one for the activation
// tile A (4 KB, packable). 6 stages × (W + A) = 12 W-pages + 2 packed
// A-pages = 14 pages, exactly NUM_PAGES.
//
// Why split W and A instead of one combined 36864-byte region per stage?
// A combined region rounds up to 3 pages (49152 B), wasting 12288 B per
// stage. 6 × 3 = 18 pages exceeds the 14-page budget. The W/A split keeps
// the same release-step semantics (both buffers retire together at the
// same MMA-consumed point) without the per-stage page padding.
//
// linear_sm100_v2.cuh addresses each buffer via
//   task_desc->smem_region_offset(REGION_W_0 + stage)   for W
//   task_desc->smem_region_offset(REGION_A_0 + stage)   for A
//
// release_step is uniform across stages right now (== 1, "release at task
// end"). Phase 3.5 will tighten this so the planner can reuse a stage's
// pages as soon as MMA has consumed it, enabling cross-task overlap.
//
// Mbarriers live in RuntimeSMEM::dynamic_semaphores (Phase 3.1), not in
// any per-stage region.
//
// Host-safe.

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace linear_sm100_v2 {

// These constants mirror the constexpr values in linear_sm100_v2.cuh
// (kernel::linear_v2). Keep them in sync if the .cuh changes — the cross-check
// is by hand because the .cuh is device-only and not includable from here.
inline constexpr int BLOCK_M    = 128;
inline constexpr int BLOCK_N    = 16;
inline constexpr int BLOCK_K    = 128;
inline constexpr int NUM_STAGES = 6;

inline constexpr int t_size_bytes = 2;  // bf16
inline constexpr int A_BYTES      = BLOCK_N * BLOCK_K * t_size_bytes;  // 4096
inline constexpr int W_BYTES      = BLOCK_M * BLOCK_K * t_size_bytes;  // 32768

// Region ordinals. Must match the push_back order in make_smem_info() and
// the smem_region_offset(...) calls in linear_sm100_v2.cuh.
inline constexpr int REGION_W_0    = 0;
inline constexpr int REGION_A_0    = NUM_STAGES;            // 6
inline constexpr int REGION_SCRATCH = 2 * NUM_STAGES;       // 12
inline constexpr int NUM_REGIONS    = 2 * NUM_STAGES + 1;   // 13

// Scratch region: holds the per-task tmem allocation address that the
// launcher role publishes for the consumer role to read. Tiny (one int).
// Packable so it shares a page with the (sub-page) A regions.
inline constexpr int SCRATCH_BYTES = 16;  // pad to 16B for alignment safety

inline constexpr int total_smem_bytes() {
  return NUM_STAGES * (W_BYTES + A_BYTES) + SCRATCH_BYTES;
}

// Capacity check: 6 W's (2 pages each) + 6 A's (packable, 2 pages total) =
// 14 pages exactly. Keep PLANNER_CAPACITY_BYTES in sync with
// python/mirage/mpk/v2_smem_planner.py:CAPACITY_BYTES.
inline constexpr int PLANNER_CAPACITY_BYTES = 225 * 1024 - 6 * 1024;  // 224256
static_assert(total_smem_bytes() <= PLANNER_CAPACITY_BYTES,
              "linear_sm100_v2 SMEM footprint exceeds the planner's "
              "CAPACITY_BYTES; either shrink the kernel or reduce NUM_STAGES");

inline ::mirage::runtime::TaskSmemInfo make_smem_info() {
  ::mirage::runtime::TaskSmemInfo info{total_smem_bytes(),
                                       /*alignment=*/1024,
                                       {}};
  // 6 W regions: each exactly 2 pages, must be contiguous, NOT packable.
  for (int s = 0; s < NUM_STAGES; s++) {
    info.regions.push_back({"linear_W_" + std::to_string(s),
                            W_BYTES,
                            /*alignment=*/1024,
                            /*page_count=*/-1,
                            /*can_pack=*/false,
                            /*release_step=*/1,
                            /*contiguous=*/true});
  }
  // 6 A regions: each 4 KB (sub-page), packable so the planner shares a
  // page across multiple A's. 16 KB / 4 KB = 4 A's per page → 2 pages.
  for (int s = 0; s < NUM_STAGES; s++) {
    info.regions.push_back({"linear_A_" + std::to_string(s),
                            A_BYTES,
                            /*alignment=*/1024,
                            /*page_count=*/-1,
                            /*can_pack=*/true,
                            /*release_step=*/1,
                            /*contiguous=*/true});
  }
  // Linear-private cross-warp scratch (currently: TMEM allocation address).
  // 16 B, packable, lands in the half-empty A page.
  info.regions.push_back({"linear_scratch",
                          SCRATCH_BYTES,
                          /*alignment=*/16,
                          /*page_count=*/-1,
                          /*can_pack=*/true,
                          /*release_step=*/1,
                          /*contiguous=*/true});
  return info;
}

} // namespace linear_sm100_v2
} // namespace kernel
