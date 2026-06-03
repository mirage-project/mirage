/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include "mirage/kernel/v2_role_codegen.h"

#include <cassert>

namespace mirage {
namespace kernel {

namespace {

namespace rt = mirage::runtime;
namespace tr = mirage::transpiler;

enum class V2Role {
  InitSemaphores,
  Loader,
  Launcher,
  Consumer,
  Storer,
};

std::string const &role_body(rt::TaskRoleVariantCode const &code,
                             V2Role role) {
  switch (role) {
    case V2Role::InitSemaphores:
      return code.init_semaphores;
    case V2Role::Loader:
      return code.loader;
    case V2Role::Launcher:
      return code.launcher;
    case V2Role::Consumer:
      return code.consumer;
    case V2Role::Storer:
      return code.storer;
  }
  return code.consumer;
}

// Phase 3.5: page-lifecycle prefix that runs at the start of every loader
// body. Each lane of warp 4 handles one physical page: wait for the prior
// task to release it; for pages this task doesn't use, finish them right
// away (the "claim+release ASAP" pattern from MegaKernels' NoOp/matvec).
// Pages this task does use are left for the consumer suffix to finish.
// Phase 3.5: page-lifecycle prefix at start of every loader body
// (MegaKernels NoOp/matvec pattern, lane-parallel). For each physical page:
//   - lane K waits for the previous task's release of page K
//   - if THIS task does not use page K, lane K arrives page K right away
//     (the "claim+release ASAP" pattern — frees pages the task doesn't
//     touch so the next task's loader can re-TMA into them sooner)
// Pages this task uses are released later by the consumer suffix instead.
// Net: every page gets exactly one arrive per task.
char const *kLoaderPagePrefix =
    "{\n"
    "  int const _lane = threadIdx.x & 31;\n"
    "  if (_lane < MAX_SMEM_PAGES_PER_TASK) {\n"
    "    runtime_wait_page_ready(runtime_smem, _lane, instruction_index);\n"
    "    if (!task_uses_page(task_desc, _lane)) {\n"
    "      runtime_finish_page(runtime_smem, _lane, 1);\n"
    "    }\n"
    "  }\n"
    "  __syncwarp();\n"
    "}\n";

// Phase 3.5: page-lifecycle suffix that runs at the end of every consumer
// body. Releases the pages this task uses (the ones the loader prefix did
// NOT release). Tasks that do their own per-stage release (e.g. linear in
// 3.5b) opt out via auto_consumer_finish=false.
//
// Single-threaded: consumer has up to 4 warps × 32 = 128 threads, but
// page_finished mbarriers have count=1 so we want exactly one arrive per
// task per page. Thread 0 issues all arrives.
//
// Iterates physical pages — NOT regions — because the planner packs
// multiple sub-page regions into the same physical page (e.g. linear's
// six 4-KB A regions land on two pages, four+two). A region-based loop
// would arrive page X once per packed region, multi-flipping parity.
// Phase 3.5: page-lifecycle suffix at end of every consumer body. Lane-
// parallel match for the loader prefix: lane K of consumer warp 0 arrives
// page K iff this task uses page K (the loader prefix already arrived
// pages this task doesn't use). Together they guarantee one arrive per
// page per task without the single-thread serialization that blocked
// consumer warp 0 in the original implementation.
char const *kConsumerPageSuffix =
    "{\n"
    "  if (threadIdx.x < MAX_SMEM_PAGES_PER_TASK &&\n"
    "      task_uses_page(task_desc, threadIdx.x)) {\n"
    "    runtime_finish_page(runtime_smem, threadIdx.x, 1);\n"
    "  }\n"
    "  __syncwarp();\n"
    "}\n";

bool has_role_body(std::vector<rt::TaskRoleVariantCode> const &variants,
                   V2Role role) {
  for (rt::TaskRoleVariantCode const &variant : variants) {
    if (!role_body(variant, role).empty()) {
      return true;
    }
    // Phase 3.5: even an empty user body means we will emit a synthetic
    // loader (just the page-lifecycle prefix) for any task that opted in.
    if (role == V2Role::Loader && variant.auto_loader_page_lifecycle) {
      return true;
    }
  }
  return false;
}

void emit_role_cases(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register,
    V2Role role) {
  for (auto const &task : task_register.all_v2_task_role_variants) {
    if (!has_role_body(task.second, role)) {
      continue;
    }
    auto name_it = task_type_to_name.find(task.first);
    assert(name_it != task_type_to_name.end());
    code.e("case $:", name_it->second);
    bool first_variant = true;
    for (size_t variant_id = 0; variant_id < task.second.size();
         variant_id++) {
      rt::TaskRoleVariantCode const &variant = task.second[variant_id];
      std::string const &body = role_body(variant, role);
      // Phase 3.5: the loader case may need to emit a body even when the
      // user-provided body is empty, to carry the auto page-lifecycle
      // prefix. Other roles only emit if they have user content (or, for
      // consumer, if they have user content; the auto suffix piggybacks
      // on the user body, it does not synthesize one on its own).
      bool const auto_loader_prefix =
          (role == V2Role::Loader) && variant.auto_loader_page_lifecycle;
      bool const auto_consumer_suffix =
          (role == V2Role::Consumer) && variant.auto_consumer_finish &&
          !body.empty();
      if (body.empty() && !auto_loader_prefix) {
        continue;
      }
      std::string const cond = first_variant ? "if" : "else if";
      code.e("  $ (task_desc->variant_id == $) {", cond, variant_id);
      if (auto_loader_prefix) {
        code.e("$", kLoaderPagePrefix);
      }
      if (!body.empty()) {
        code.e("$", body);
      }
      if (auto_consumer_suffix) {
        code.e("$", kConsumerPageSuffix);
      }
      code.e("}");
      first_variant = false;
    }
    code.e("  break;");
  }
}

void emit_role_dispatcher(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register,
    char const *function_name,
    V2Role role) {
  code.e("__device__ __forceinline__ void");
  code.e("$(TaskDesc const *task_desc,", function_name);
  code.e("  RuntimeConfig const &runtime_config,");
  code.e("  RuntimeSMEM *runtime_smem,");
  code.e("  int instruction_index,");
  code.e("  int iter_num) {");
  code.e("(void)runtime_config;");
  code.e("(void)runtime_smem;");
  code.e("(void)instruction_index;");
  code.e("(void)iter_num;");
  code.e("switch (task_desc->task_type) {");
  emit_role_cases(code, task_type_to_name, task_register, role);
  code.e("default:");
  code.e("  break;");
  code.e("}");
  code.e("}");
}

} // namespace

void generate_v2_role_dispatch_code(
    tr::CodeKeeper &code,
    std::map<rt::TaskType, std::string> const &task_type_to_name,
    rt::TaskRegister const &task_register) {
  // The dispatchers reference v2-only types (RuntimeSMEM, etc.) and helpers.
  // Skip them entirely in v1 builds — v1 has its own dispatch path.
  code.e("#ifdef USE_RUNTIME_V2");
  code.e("namespace mirage {");
  code.e("namespace runtime_v2 {");
  code.e("using namespace mirage::runtime;");
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_init_semaphores_v2",
                       V2Role::InitSemaphores);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_loader_task_v2",
                       V2Role::Loader);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_launcher_task_v2",
                       V2Role::Launcher);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_consumer_task_v2",
                       V2Role::Consumer);
  emit_role_dispatcher(code,
                       task_type_to_name,
                       task_register,
                       "_execute_storer_task_v2",
                       V2Role::Storer);
  code.e("} // namespace runtime_v2");
  code.e("} // namespace mirage");
  code.e("#endif // USE_RUNTIME_V2");
}

} // namespace kernel
} // namespace mirage
