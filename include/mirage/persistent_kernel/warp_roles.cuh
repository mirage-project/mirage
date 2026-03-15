/* Copyright 2025 CMU
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

// Warp-specialized persistent kernel roles for Blackwell (SM100).
// 12 warps = 384 threads total:
//   Warps 0-7   (threads 0-255):   Consumer  — all existing task code
//   Warp 8      (threads 256-287):  Launcher  — future MMA coordination
//   Warp 9      (threads 288-319):  Loader    — TMA prefetch
//   Warp 10     (threads 320-351):  Storer    — GMEM barrier_arrive
//   Warp 11     (threads 352-383):  Controller— instruction pipeline, barrier_wait

#if defined(MIRAGE_GRACE_BLACKWELL)

#include <cutlass/arch/barrier.h>
#include "tasks/hopper/barrier.cuh"

// ── Role Warp Indices ───────────────────────────────────────────────────────
constexpr int ROLE_CONSUMER_WARPS_BEGIN = 0;
constexpr int ROLE_CONSUMER_WARPS_END   = 8;  // exclusive
constexpr int ROLE_LAUNCHER_WARP        = 8;
constexpr int ROLE_LOADER_WARP          = 9;
constexpr int ROLE_STORER_WARP          = 10;
constexpr int ROLE_CONTROLLER_WARP      = 11;

constexpr int TOTAL_NUM_WARPS           = 12;
constexpr int TOTAL_NUM_THREADS         = TOTAL_NUM_WARPS * 32;  // 384
constexpr int CONSUMER_WARP_COUNT       = ROLE_CONSUMER_WARPS_END - ROLE_CONSUMER_WARPS_BEGIN; // 8
// MPK_CONSUMER_NUM_THREADS, MPK_CONSUMER_SYNC (int), and MPK_CONSUMER_SYNC() (macro)
// are defined in tasks/common/worker_config.h

// ── Instruction Pipeline ────────────────────────────────────────────────────
constexpr int PIPELINE_STAGES = 2;

struct __align__(128) InstructionState {
  mirage::runtime::TaskDesc task;  // loaded by controller from GMEM
  int trigger_event_idx;          // GMEM barrier to signal after completion (-1 = none)
};

// Wait parity for mbarrier try_wait.parity:
// try_wait.parity succeeds when (current_phase & 1) != phaseParity.
// mbarrier starts at phase 0. After k-th completion, phase = k.
// For the k-th use of slot (0-indexed), to wait for completion (phase k+1):
//   pass parity = k & 1, so wait succeeds when phase parity != (k & 1).
__device__ __forceinline__ int phase_bit(int task_idx) {
  int k = task_idx / PIPELINE_STAGES;  // k-th use of this slot
  return k & 1;
}

// ── SMEM flag for consumer→storer handshake ─────────────────────────────────
// Volatile SMEM load with acquire semantics (block-scope).
__device__ __forceinline__ int ld_acquire_smem_s32(int const *addr) {
  int val;
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("ld.acquire.cta.shared.b32 %0, [%1];\n"
               : "=r"(val) : "r"(smem_ptr) : "memory");
  return val;
}

// Volatile SMEM store with release semantics (block-scope).
__device__ __forceinline__ void st_release_smem_s32(int *addr, int val) {
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("st.release.cta.shared.b32 [%0], %1;\n"
               :: "r"(smem_ptr), "r"(val) : "memory");
}

// ── Pipeline shared memory layout ──────────────────────────────────────────
// These are declared in static_worker.cuh as __shared__ variables.
// This struct groups them for documentation; actual allocation is via __shared__.
struct PipelineSmem {
  InstructionState instructions[PIPELINE_STAGES];
  kernel::Barrier  instruction_arrived[PIPELINE_STAGES];   // controller → all
  kernel::Barrier  instruction_finished[PIPELINE_STAGES];  // storer → controller
  int              compute_done_flag[PIPELINE_STAGES];     // consumer → storer
};

#endif // MIRAGE_GRACE_BLACKWELL
