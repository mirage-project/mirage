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
  uint32_t profiler_write_stride;                                              \
  uint32_t profiler_entry_tag_base;                                            \
  bool profiler_write_thread_predicate;

// #define PROFILER_PARAMS_DECL uint64_t *profiler_buffer;

#define PROFILER_INIT(                                                         \
    profiler_buffer, group_idx, num_groups, write_thread_predicate)            \
  if (tb::get_block_idx() == 0 && tb::get_thread_idx() == 0) {                 \
    entry.nblocks = tb::get_num_blocks();                                      \
    entry.ngroups = num_groups;                                                \
    profiler_buffer[0] = entry.raw;                                            \
  }                                                                            \
  profiler_write_ptr =                                                         \
      profiler_buffer + 1 + tb::get_block_idx() * num_groups + group_idx;      \
  profiler_write_stride = tb::get_num_blocks() * num_groups;                   \
  profiler_entry_tag_base =                                                    \
      tb::encode_tag(tb::get_block_idx() * num_groups + group_idx, 0, 0);      \
  profiler_write_thread_predicate = write_thread_predicate;

#define PROFILER_EVENT_START(event, event_no)                                  \
  if (profiler_write_thread_predicate) {                                       \
    entry.tag =                                                                \
        tb::make_event_tag_start(profiler_entry_tag_base, event, event_no);    \
    entry.delta_time = tb::get_timestamp();                                    \
    *profiler_write_ptr = entry.raw;                                           \
    profiler_write_ptr += profiler_write_stride;                               \
  }                                                                            \
  __threadfence_block();

#define PROFILER_EVENT_END(event, event_no)                                    \
  __threadfence_block();                                                       \
  if (profiler_write_thread_predicate) {                                       \
    entry.tag =                                                                \
        tb::make_event_tag_end(profiler_entry_tag_base, event, event_no);      \
    entry.delta_time = tb::get_timestamp();                                    \
    *profiler_write_ptr = entry.raw;                                           \
    profiler_write_ptr += profiler_write_stride;                               \
  }

#define PROFILER_EVENT_INSTANT(event, event_no)                                \
  __threadfence_block();                                                       \
  if (profiler_write_thread_predicate) {                                       \
    entry.tag =                                                                \
        tb::make_event_tag_instant(profiler_entry_tag_base, event, event_no);  \
    entry.delta_time = tb::get_timestamp();                                    \
    *profiler_write_ptr = entry.raw;                                           \
  }                                                                            \
  __threadfence_block();

} // namespace tb
