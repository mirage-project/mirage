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

// Region ordinals. Must match the push_back order in make_smem_info() and
// the smem_region_offset(...) calls in RmsNormBuffers.
inline constexpr int REGION_INPUT  = 0;
inline constexpr int REGION_WEIGHT = 1;
inline constexpr int REGION_OUTPUT = 2;
inline constexpr int REGION_REDUCE = 3;
inline constexpr int NUM_REGIONS   = 4;

// SmemBuffer pads each buffer to ALIGN=1024 to keep TMA-swizzle safe; this
// helper mirrors that rounding so the planner reserves the same byte count
// the device-side typed view will consume.
inline constexpr int round_up_1024(int n) {
  return (n + 1023) & ~1023;
}

inline constexpr int input_region_bytes(int t_size_bytes, int hidden_dim) {
  return round_up_1024(t_size_bytes * hidden_dim);
}

inline constexpr int reduce_region_bytes(int num_threads) {
  // sizeof(float) * num_threads, padded to ALIGN=1024.
  return round_up_1024(4 * num_threads);
}

inline constexpr int total_smem_bytes(int t_size_bytes,
                                      int hidden_dim,
                                      int num_threads) {
  return 3 * input_region_bytes(t_size_bytes, hidden_dim) +
         reduce_region_bytes(num_threads);
}

// Build the planner-facing TaskSmemInfo. release_step is uniform across all
// four regions because the rmsnorm task body holds every buffer until the
// final store; a finer-grained breakdown would require splitting the kernel.
inline ::mirage::runtime::TaskSmemInfo
make_smem_info(int t_size_bytes, int hidden_dim, int num_threads) {
  int const buf_bytes    = input_region_bytes(t_size_bytes, hidden_dim);
  int const reduce_bytes = reduce_region_bytes(num_threads);
  int const total_bytes  = 3 * buf_bytes + reduce_bytes;

  ::mirage::runtime::TaskSmemInfo info{total_bytes, /*alignment=*/1024, {}};
  info.regions.push_back(
      {"input",  buf_bytes,    1024, -1, /*can_pack=*/true,  /*release_step=*/2, /*contiguous=*/true});
  info.regions.push_back(
      {"weight", buf_bytes,    1024, -1, true, 2, true});
  info.regions.push_back(
      {"output", buf_bytes,    1024, -1, true, 2, true});
  info.regions.push_back(
      {"reduce", reduce_bytes, 1024, -1, true, 2, true});
  return info;
}

} // namespace rmsnorm_v2
} // namespace kernel
