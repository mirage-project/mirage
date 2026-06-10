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
#pragma once

#include "mirage/persistent_kernel/runtime_header.h"

namespace kernel {
namespace v2_task {

// Per-role entry points for a v2 task. A task subclasses TaskInterface and
// overrides the roles it implements (`using consumer = MyConsumer;` etc.);
// roles it omits fall back to a no-op. Codegen emits Task::<role>::run() calls.
//
// SMEM layout, semaphores, and the planner feed live in each task's host-safe
// <task>_spec.h (make_smem_info), which is the single source the host-side
// task registration reads.

struct NoopControl {
  __device__ __forceinline__ static int release_lid(int query) { return query; }

  template <typename... Args>
  __device__ __forceinline__ static int init_semaphores(Args...) {
    return 0;
  }
};

struct NoopRole {
  template <typename... Args>
  __device__ __forceinline__ static void run(Args...) {}
};

struct TaskInterface {
  using control = NoopControl;
  using loader = NoopRole;
  using launcher = NoopRole;
  using consumer = NoopRole;
  using storer = NoopRole;
};

} // namespace v2_task
} // namespace kernel
