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

#include <cstdint>
#include <string>
#include <utility>


namespace mirage {
namespace transpiler {

class Profiler {
public:
  static std::pair<std::string, std::string> get_profiling_ptr();
};

// constants
#define PROFILER_CONSTANTS_DECL \
  static constexpr uint32_t EVENT_IDX_SHIFT = 2; \
  static constexpr uint32_t BLOCK_IDX_SHIFT = 14; \
  static constexpr uint32_t EVENT_BEGIN = 0x0; \
  static constexpr uint32_t EVENT_END = 0x1;


// helper functions 
#define PROFILER_HELPER_FUNCTIONS_DECL \
  __device__ __forceinline__ uint32_t get_block_idx() { \
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; \
  } \
  \
  __device__ __forceinline__ uint32_t get_num_blocks() { \
    return gridDim.x * gridDim.y * gridDim.z; \
  } \
  \
  __device__ __forceinline__ uint32_t get_thread_idx() { \
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; \
  } \
  \
  __device__ __forceinline__ uint32_t encode_tag(uint32_t block_idx, uint32_t event_idx, \
                                               uint32_t event_type) { \
    return (block_idx << BLOCK_IDX_SHIFT) | (event_idx << EVENT_IDX_SHIFT) | event_type; \
  } \
  \
  __device__ __forceinline__ uint32_t get_timestamp() { \
    volatile uint32_t ret; \
    asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret)); \
    return ret; \
  }

// ProfilerEntry structure
#define PROFILER_ENTRY_DECL \
  struct ProfilerEntry { \
    union { \
      struct { \
        uint32_t nblocks; \
        uint32_t ngroups; \
      }; \
      struct { \
        uint32_t tag; \
        uint32_t delta_time; \
      }; \
      uint64_t raw; \
    }; \
  };

#ifdef MIRAGE_ENABLE_PROFILER

#define PROFILER_ADDITIONAL_FUNC_PARAMS , uint64_t* profiler_buffer_ptr
#define PROFILER_ADDITIONAL_FUNC_PARAMS_ARGS , profiler_buffer_ptr

#define PROFILER_DEVICE_BUFFER_PTR_SETTER \
  { \
    uint64_t* host_ptr = static_cast<uint64_t*>(profiler_buffer); \
    cudaMemcpyToSymbol(profiler_buffer_ptr, &host_ptr, sizeof(uint64_t*)); \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
      printf("CUDA error in cudaMemcpyToSymbol: %s\n", cudaGetErrorString(error)); \
    } \
  }

#define PROFILER_INCLUDE_ALL_DECL \
  PROFILER_CONSTANTS_DECL \
  PROFILER_HELPER_FUNCTIONS_DECL \
  PROFILER_ENTRY_DECL \
  PROFILER_DEVICE_BUFFER_PTR_DECL


#define PROFILER_WRITE_PARAMS_DECL \
  uint64_t* profiler_write_ptr;      \
  uint32_t profiler_write_stride;    \
  uint32_t profiler_entry_tag_base;  \
  bool profiler_write_thread_predicate;

#define PROFILER_DEVICE_BUFFER_PTR_DECL __device__ uint64_t* profiler_buffer_ptr;

#define PROFILER_INIT(profiler_buffer_ptr,                     \
                      write_thread_predicate)                                                   \
  volatile ProfilerEntry entry;                                                                 \
  if (get_thread_idx() == 0) {                                          \
    entry.nblocks = get_num_blocks();                                                           \
    profiler_buffer_ptr[0] = entry.raw;                                                      \
  }                                                                                             \
  profiler_write_ptr =                                                                  \
      profiler_buffer_ptr + 1 + get_block_idx();                    \
  profiler_write_stride = get_num_blocks();                                \
  profiler_entry_tag_base = encode_tag(get_block_idx(), 0, 0); \
  profiler_write_thread_predicate = write_thread_predicate; \



#define PROFILER_EVENT_START(event)                                                  \
  if (profiler_write_thread_predicate) {                                              \
    entry.tag =                                                                               \
        profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_BEGIN; \
    entry.delta_time = get_timestamp();                                                       \
    *profiler_write_ptr = entry.raw;                                                  \
    profiler_write_ptr += profiler_write_stride;                              \
  }                                                                                           \
  __threadfence_block();

#define PROFILER_EVENT_END(event)                                                  \
  __threadfence_block();                                                                    \
  if (profiler_write_thread_predicate) {                                            \
    entry.tag =                                                                             \
        profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_END; \
    entry.delta_time = get_timestamp();                                                     \
    *profiler_write_ptr = entry.raw;                                                \
    profiler_write_ptr += profiler_write_stride;                            \
  }

#else

#define PROFILER_ADDITIONAL_FUNC_PARAMS
#define PROFILER_ADDITIONAL_FUNC_PARAMS_ARGS

#define PROFILER_ADDITIONAL_PARAMS_SETTER

#define PROFILER_INCLUDE_ALL_DECL
#define PROFILER_WRITE_PARAMS_DECL
#define PROFILER_INIT(profiler_buffer_ptr, write_thread_predicate)
#define PROFILER_EVENT_START(event)
#define PROFILER_EVENT_END(event)
#define PROFILER_DEVICE_BUFFER_PTR_SETTER

#endif


} // namespace transpiler
} // namespace mirage
