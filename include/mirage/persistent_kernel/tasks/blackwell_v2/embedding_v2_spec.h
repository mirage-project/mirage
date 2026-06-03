/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// embedding_v2 is GMEM-only — no SMEM regions. The spec exists only so the
// task_register call site reads from the same place every other v2 task does.
//
// Host-safe.

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace embedding_v2 {

inline constexpr int NUM_REGIONS = 0;

inline ::mirage::runtime::TaskSmemInfo make_smem_info() {
  return ::mirage::runtime::TaskSmemInfo{/*size=*/0, /*alignment=*/1, {}};
}

} // namespace embedding_v2
} // namespace kernel
