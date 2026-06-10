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

// Region ordinals — index into the regions[] make_smem_info() builds and the
// smem_region_offset(N) calls in ArgmaxBuffers (argmax_sm100.cuh).
inline constexpr int REGION_IDX  = 0;
inline constexpr int REGION_VAL  = 1;
inline constexpr int NUM_REGIONS = 2;

inline constexpr int IDX_ALIGN = 128;
inline constexpr int VAL_ALIGN = 16;

// Raw (unpadded) per-region byte sizes — the single source for both the device
// ArgmaxBuffers<T> view and make_smem_info(). Scratch holds 32 entries, one per
// warp (max 32 warps per block).
inline constexpr int raw_idx_bytes() { return (int)sizeof(long long) * 32; }
inline constexpr int raw_val_bytes(int t_size_bytes) { return t_size_bytes * 32; }

inline constexpr int round_up(int n, int align) {
  return (n + align - 1) & ~(align - 1);
}

inline ::mirage::runtime::TaskSmemInfo
make_smem_info(int t_size_bytes) {
  int const idx = round_up(raw_idx_bytes(), IDX_ALIGN);
  int const val = round_up(raw_val_bytes(t_size_bytes), VAL_ALIGN);
  ::mirage::runtime::TaskSmemInfo info{idx + val, IDX_ALIGN, {}};
  // Both buffers are scratch held for the whole reduce — same release_step.
  info.regions.push_back({"idx", idx, IDX_ALIGN, -1, /*can_pack=*/true, /*release_step=*/1, /*contiguous=*/true});
  info.regions.push_back({"val", val, VAL_ALIGN, -1, true, 1, true});
  return info;
}

} // namespace argmax_v2
} // namespace kernel
