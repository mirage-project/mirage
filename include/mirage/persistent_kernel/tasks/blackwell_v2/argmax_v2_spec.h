/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the argmax_sm100_v2 SMEM region layout.
// Shared by both TASK_ARGMAX_PARTIAL_SM100_V2 and TASK_ARGMAX_REDUCE_SM100_V2
// because the two reuse the same ArgmaxBuffers<T> typed view.
//
// Host-safe: must NOT pull in any device-only construct.

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace argmax_v2 {

// Region ordinals. Must match the smem_region_offset(N) calls in
// ArgmaxBuffers in argmax_sm100.cuh.
inline constexpr int REGION_IDX  = 0;
inline constexpr int REGION_VAL  = 1;
inline constexpr int NUM_REGIONS = 2;

// Mirrors ArgmaxBuffers<T>::IdxBuf / ValBuf padding:
//   IdxBuf  = SmemBuffer<sizeof(long long) * 32, 128>  → padded to 128
//   ValBuf  = SmemBuffer<sizeof(T)         * 32, 16>   → padded to 16
inline constexpr int idx_region_bytes() {
  return (8 * 32 + 127) & ~127;  // = 256
}

inline constexpr int val_region_bytes(int t_size_bytes) {
  return (t_size_bytes * 32 + 15) & ~15;  // bf16 → 64
}

inline ::mirage::runtime::TaskSmemInfo
make_smem_info(int t_size_bytes) {
  int const idx_padded = idx_region_bytes();
  int const val_padded = val_region_bytes(t_size_bytes);
  ::mirage::runtime::TaskSmemInfo info{idx_padded + val_padded, /*alignment=*/128, {}};
  // Both buffers are scratch alive for the entire reduce — same release_step.
  info.regions.push_back(
      {"idx", idx_padded, 128, -1, /*can_pack=*/true,  /*release_step=*/1, /*contiguous=*/true});
  info.regions.push_back(
      {"val", val_padded, 16,  -1, true, 1, true});
  return info;
}

} // namespace argmax_v2
} // namespace kernel
