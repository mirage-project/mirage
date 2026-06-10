/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the rmsnorm_v2 SMEM region layout.
//
// Both the host-side task registration (src/kernel/task_register.cc) and the
// device-side typed buffer view (RmsNormBuffers in rmsnorm_v2.cuh) include
// this header. The region names, order, sizes, alignments, and release_steps
// declared here are what the planner sees and what the device addresses by
// ordinal via task_desc->smem_region_offset(N).
//
// Constraints:
//   - Host-safe: this header must NOT include any device-only construct
//     (no __device__, no PTX asm). It is included by a regular .cc file.
//   - The order of regions[] is the contract for the smem_region_offset
//     ordinals used in the device-side RmsNormBuffers constructor.

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace rmsnorm_v2 {

// Region ordinals — index into the regions[] make_smem_info() builds and the
// smem_region_offset(...) calls in RmsNormBuffers.
inline constexpr int REGION_INPUT  = 0;
inline constexpr int REGION_WEIGHT = 1;
inline constexpr int REGION_OUTPUT = 2;
inline constexpr int REGION_REDUCE = 3;
inline constexpr int NUM_REGIONS   = 4;

inline constexpr int ALIGN = 1024;  // keeps TMA-swizzle (128B) safe

// Raw (unpadded) per-region byte sizes. These are the single source for the
// region sizes: the device SmemBuffer<> views in RmsNormBuffers compute from
// the same two functions, so there's no second copy of the formula to drift.
inline constexpr int raw_buffer_bytes(int t_size_bytes, int hidden_dim) {
  return t_size_bytes * hidden_dim;  // input / weight / output
}
inline constexpr int raw_reduce_bytes(int num_threads) {
  return (int)sizeof(float) * num_threads;
}

inline constexpr int round_up(int n, int align) {
  return (n + align - 1) & ~(align - 1);
}

// release_step is uniform across all four regions: the task holds every buffer
// until the final store. Finer granularity would need splitting the kernel.
inline ::mirage::runtime::TaskSmemInfo
make_smem_info(int t_size_bytes, int hidden_dim, int num_threads) {
  int const buf    = round_up(raw_buffer_bytes(t_size_bytes, hidden_dim), ALIGN);
  int const reduce = round_up(raw_reduce_bytes(num_threads), ALIGN);

  ::mirage::runtime::TaskSmemInfo info{3 * buf + reduce, ALIGN, {}};
  info.regions.push_back({"input",  buf,    ALIGN, -1, /*can_pack=*/true, /*release_step=*/2, /*contiguous=*/true});
  info.regions.push_back({"weight", buf,    ALIGN, -1, true, 2, true});
  info.regions.push_back({"output", buf,    ALIGN, -1, true, 2, true});
  info.regions.push_back({"reduce", reduce, ALIGN, -1, true, 2, true});
  return info;
}

} // namespace rmsnorm_v2
} // namespace kernel
