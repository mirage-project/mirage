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

enum class Role {
  Control,
  Loader,
  Launcher,
  Consumer,
  Storer,
};

inline constexpr int GROUP_UNUSED = -1;
inline constexpr int GROUP_TASK_START = 0;
inline constexpr int GROUP_TASK_END = 1;

struct SmemRegionSpec {
  int region_type = 0;
  char const *debug_name = "";
  int size = 0;
  int alignment = 1;
  int page_count = -1;
  bool can_pack = false;
  bool contiguous = true;
  int acquire_group = GROUP_TASK_START;
  int release_group = GROUP_TASK_END;
  Role owner_role = Role::Consumer;
};

struct ReleaseGroupSpec {
  int group_id = GROUP_UNUSED;
  char const *debug_name = "unused";
};

struct SemaphoreSpec {
  int semaphore_id = 0;
  char const *debug_name = "";
  Role producer_role = Role::Control;
  Role consumer_role = Role::Consumer;
  int initial_count = 0;
  int arrive_count = 1;
};

struct RoleSpec {
  Role role = Role::Consumer;
  int first_warp = 0;
  int num_warps = 0;
};

struct TaskSpecView {
  mirage::runtime::TaskType task_type = mirage::runtime::TASK_TERMINATE;
  char const *debug_name = "";
  int total_smem_size = 0;
  int alignment = 1;

  SmemRegionSpec const *smem_regions = nullptr;
  int num_smem_regions = 0;

  ReleaseGroupSpec const *release_groups = nullptr;
  int num_release_groups = 0;

  SemaphoreSpec const *semaphores = nullptr;
  int num_semaphores = 0;

  RoleSpec const *roles = nullptr;
  int num_roles = 0;
};

struct NoopControl {
  __device__ __forceinline__ static int release_lid(int query) {
    return query;
  }

  template <typename... Args>
  __device__ __forceinline__ static int init_semaphores(Args...) {
    return 0;
  }
};

struct NoopRole {
  template <typename... Args>
  __device__ __forceinline__ static void run(Args...) {}
};

struct TaskInterfaceDefaults {
  using control = NoopControl;
  using loader = NoopRole;
  using launcher = NoopRole;
  using consumer = NoopRole;
  using storer = NoopRole;
};

template <typename Derived>
struct TaskInterface : public TaskInterfaceDefaults {
  static constexpr TaskSpecView task_spec() {
    return Derived::spec();
  }
};

struct NoopTask : public TaskInterface<NoopTask> {
  static constexpr TaskSpecView spec() {
    return TaskSpecView{
        mirage::runtime::TASK_TERMINATE,
        "noop",
        0,
        1,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
    };
  }
};

} // namespace v2_task
} // namespace kernel
