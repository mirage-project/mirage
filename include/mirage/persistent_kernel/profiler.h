/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdint.h>
namespace tb {

__device__ __forceinline__ uint32_t get_block_idx() {
  return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ uint32_t get_num_blocks() {
  return gridDim.x * gridDim.y * gridDim.z;
}

__device__ __forceinline__ uint32_t get_thread_idx() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

/*
     13 bits      8 bits        9 bits         2 bits
    [31-19]      [18-11]       [10-2]         [1-0]
   [event no] [block group] [event type] [begin/end/instant]
*/
constexpr uint32_t EVENT_IDX_SHIFT = 2;
constexpr uint32_t BLOCK_GROUP_IDX_SHIFT = 11;
// top 8 bits of the tag represents the nth event of the same type
constexpr uint32_t EVENT_NO_SHIFT = 19;

constexpr uint32_t EVENT_BEGIN = 0x0;
constexpr uint32_t EVENT_END = 0x1;
constexpr uint32_t EVENT_INSTANT = 0x2;

__device__ __forceinline__ void sleep_cycles(uint32_t cycles) {
  uint32_t start = 0, now = 0;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(start));
  do {
    asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(now));
  } while ((now - start) < cycles);
}

__device__ __forceinline__ uint32_t encode_tag(uint32_t block_group_idx,
                                               uint32_t event_idx,
                                               uint32_t event_type) {
  return (block_group_idx << BLOCK_GROUP_IDX_SHIFT) |
         (event_idx << EVENT_IDX_SHIFT) | event_type;
}

__device__ __forceinline__ uint32_t make_event_tag_start(uint32_t base_tag,
                                                         uint32_t event_id,
                                                         uint32_t event_no) {
  return base_tag | (event_id << EVENT_IDX_SHIFT) |
         (event_no << EVENT_NO_SHIFT) | EVENT_BEGIN;
}

__device__ __forceinline__ uint32_t make_event_tag_end(uint32_t base_tag,
                                                       uint32_t event_id,
                                                       uint32_t event_no) {
  return base_tag | (event_id << EVENT_IDX_SHIFT) |
         (event_no << EVENT_NO_SHIFT) | EVENT_END;
}

__device__ __forceinline__ uint32_t make_event_tag_instant(uint32_t base_tag,
                                                           uint32_t event_id,
                                                           uint32_t event_no) {
  return base_tag | (event_id << EVENT_IDX_SHIFT) |
         (event_no << EVENT_NO_SHIFT) | EVENT_INSTANT;
}

__device__ __forceinline__ uint32_t get_timestamp() {
  uint32_t volatile ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}

struct ProfilerEntry {
  union {
    struct {
      uint32_t nblocks;
      uint32_t ngroups;
    };
    struct {
      uint32_t tag;
      uint32_t delta_time;
    };
    uint64_t raw;
  };
};

#define TB_SLEEP_MS(ms) tb::sleep_cycles((ms)*1000000)
#define TB_SLEEP_US(us) tb::sleep_cycles((us)*1000)

#define PROFILER_CLOSURE_PARAMS_DECL                                           \
  volatile tb::ProfilerEntry entry;                                            \
  uint64_t *profiler_write_ptr;                                                \
  uint64_t *profiler_write_end;                                                \
  uint32_t profiler_write_stride;                                              \
  uint32_t profiler_entry_tag_base;                                            \
  bool profiler_write_thread_predicate;

// End of the caller's profiler buffer. `persistent_kernel.py` emits
// -DMPK_PROFILER_BUFFER_ENTRIES=<numel> whenever profiling is on; without it
// the macros keep their historical unbounded behaviour.
//
// This matters because a run whose events exceed the buffer used to keep
// writing PAST the tensor. It was previously unreachable in practice - every
// profiled MODE_OFFLINE run was truncated to ~2 steps by the bug fixed in
// persistent_kernel.cuh - so a full-length profiled run is exactly the case
// that needs the bound. Overflow now DROPS events (the CSV exporter's
// dangling-BEGIN check still reports it) instead of corrupting memory.
#ifdef MPK_PROFILER_BUFFER_ENTRIES
#define MPK_PROFILER_BUFFER_END(buf) ((buf) + (MPK_PROFILER_BUFFER_ENTRIES))
#else
#define MPK_PROFILER_BUFFER_END(buf) (static_cast<uint64_t *>(nullptr))
#endif

#define PROFILER_CAN_WRITE                                                     \
  (profiler_write_thread_predicate &&                                          \
   (profiler_write_end == nullptr || profiler_write_ptr < profiler_write_end))

// #define PROFILER_PARAMS_DECL uint64_t *profiler_buffer;

// Generalized init: the caller supplies the block's index inside a GLOBAL
// block-index space and the size of that space, instead of both being taken
// from the launching grid.
//
// This exists because the persistent runtime can run its workers and its
// schedulers as two SEPARATE kernel launches (`split_worker_scheduler`). With
// per-launch `get_block_idx()`/`get_num_blocks()` both launches number their
// blocks from 0, so worker block b and scheduler block b write their events to
// the same slots (`1 + b`, stride `num_blocks`) and share one tag namespace,
// and the header records whichever launch's block 0 initialized first. That
// corrupted rows in the CSV export and made the Perfetto export raise
// `KeyError: (80, 0)` because its `tid_map` only covers `range(nblocks)`
// (demo/qwen3_5/accept/probes/runtime/p9_methodology.md, step 2, bugs 1-2).
#define PROFILER_INIT_GLOBAL(profiler_buffer,                                  \
                             group_idx,                                        \
                             num_groups,                                       \
                             write_thread_predicate,                           \
                             global_block_idx,                                 \
                             global_num_blocks)                                \
  if ((global_block_idx) == 0 && tb::get_thread_idx() == 0) {                  \
    entry.nblocks = (global_num_blocks);                                       \
    entry.ngroups = num_groups;                                                \
    profiler_buffer[0] = entry.raw;                                            \
  }                                                                            \
  profiler_write_ptr =                                                         \
      profiler_buffer + 1 + (global_block_idx)*num_groups + group_idx;         \
  profiler_write_end = MPK_PROFILER_BUFFER_END(profiler_buffer);               \
  profiler_write_stride = (global_num_blocks)*num_groups;                      \
  profiler_entry_tag_base =                                                    \
      tb::encode_tag((global_block_idx)*num_groups + group_idx, 0, 0);         \
  profiler_write_thread_predicate = write_thread_predicate;

#define PROFILER_INIT(                                                         \
    profiler_buffer, group_idx, num_groups, write_thread_predicate)            \
  PROFILER_INIT_GLOBAL(profiler_buffer,                                        \
                       group_idx,                                              \
                       num_groups,                                             \
                       write_thread_predicate,                                 \
                       tb::get_block_idx(),                                    \
                       tb::get_num_blocks())

#define PROFILER_EVENT_START(event, event_no)                                  \
  if (PROFILER_CAN_WRITE) {                                                    \
    entry.tag =                                                                \
        tb::make_event_tag_start(profiler_entry_tag_base, event, event_no);    \
    entry.delta_time = tb::get_timestamp();                                    \
    *profiler_write_ptr = entry.raw;                                           \
    profiler_write_ptr += profiler_write_stride;                               \
  }                                                                            \
  __threadfence_block();

#define PROFILER_EVENT_END(event, event_no)                                    \
  __threadfence_block();                                                       \
  if (PROFILER_CAN_WRITE) {                                                    \
    entry.tag =                                                                \
        tb::make_event_tag_end(profiler_entry_tag_base, event, event_no);      \
    entry.delta_time = tb::get_timestamp();                                    \
    *profiler_write_ptr = entry.raw;                                           \
    profiler_write_ptr += profiler_write_stride;                               \
  }

#define PROFILER_EVENT_INSTANT(event, event_no)                                \
  __threadfence_block();                                                       \
  if (PROFILER_CAN_WRITE) {                                                    \
    entry.tag =                                                                \
        tb::make_event_tag_instant(profiler_entry_tag_base, event, event_no);  \
    entry.delta_time = tb::get_timestamp();                                    \
    *profiler_write_ptr = entry.raw;                                           \
  }                                                                            \
  __threadfence_block();

} // namespace tb
