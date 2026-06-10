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

// v2 RMS-norm task, adapted from ampere/rmsnorm.cuh. Only the first NUM_THREADS
// (128) participate so it can run inside v2's 512-thread block, and it syncs on
// named barrier 2 (0 is block-wide, 1 belongs to the linear task). Inputs and
// output are raw GMEM pointers; it stages input and weight into SMEM via
// cp.async, computes in place, and writes back to GMEM.

#pragma once
#include "../common/common_header.cuh"
#include "../common/worker_config.h"
#include "mirage/persistent_kernel/smem_buffer.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/rmsnorm_v2_spec.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/task_interface.cuh"

namespace kernel {
namespace rmsnorm_v2 {

// Named barrier for this task (128 threads participate).
__device__ inline void task_sync() {
    asm volatile("bar.sync 2, %0;" :: "n"(NUM_THREADS));
}

// SMEM buffer layout for rmsnorm_task<T, BATCH_SIZE, HIDDEN_DIM>.
// Buffer offsets within the task's SMEM region are constexpr (sequential
// per-task layout). PADDED_BYTES rounds each buffer to 1024 for TMA-swizzle
// safety. v2 page-managed tasks address shared memory through page regions,
// not a task-wide dynamic-SMEM offset.
//
// Phase semantics for rmsnorm:
//   0 = TMA load (input/weight), 1 = compute (read input/weight, write output/reduce),
//   2 = store (read output)
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
struct RmsNormBuffers {
    using B = mirage::runtime_v2::SmemBuffer<raw_buffer_bytes(sizeof(T), HIDDEN_DIM), ALIGN>;
    using BReduce = mirage::runtime_v2::SmemBuffer<raw_reduce_bytes(NUM_THREADS), ALIGN>;

    B input;
    B weight;
    B output;
    BReduce reduce;

    __device__ explicit RmsNormBuffers(
        char *base, mirage::runtime::TaskDesc const *task_desc)
        : input (base + task_desc->smem_region_offset(REGION_INPUT)),
          weight(base + task_desc->smem_region_offset(REGION_WEIGHT)),
          output(base + task_desc->smem_region_offset(REGION_OUTPUT)),
          reduce(base + task_desc->smem_region_offset(REGION_REDUCE)) {}
};

// Cross-check: the device-side typed view consumes exactly the regions the
// planner reserves via make_smem_info(...) in rmsnorm_v2_spec.h. Both the
// number of buffers in RmsNormBuffers and the ordinals used above must match
// NUM_REGIONS / REGION_* declared there. If you add a buffer, update both.
static_assert(NUM_REGIONS == 4, "RmsNormBuffers and make_smem_info must agree");

template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __noinline__ void rms_norm_task(
    mirage::runtime::TaskDesc const *task_desc,
	    void const *input_ptr,
	    void const *weight_ptr,
	    void *output_ptr,
	    float eps,
	    void *runtime_smem
	) {
    // v2 runtime launches with 512 threads; only first NUM_THREADS participate.
    if (threadIdx.x >= NUM_THREADS) return;

    static_assert(BATCH_SIZE == 1);
    extern __shared__ char smem[];
    static_assert(HIDDEN_DIM % NUM_THREADS == 0);
    constexpr int ELTS_PER_THREAD = HIDDEN_DIM / NUM_THREADS;
    constexpr int BYTES_PER_THREAD = ELTS_PER_THREAD * sizeof(T);
    constexpr int BYTES_PER_CP = []() {
        if constexpr (BYTES_PER_THREAD % 16 == 0) return 16;
        else if constexpr (BYTES_PER_THREAD % 8 == 0) return 8;
        else { static_assert(BYTES_PER_THREAD % 4 == 0); return 4; }
    }();
    constexpr int CHUNK_SIZE = BYTES_PER_CP / sizeof(T);
    constexpr int TILE_SIZE = NUM_THREADS * CHUNK_SIZE;
    static_assert(HIDDEN_DIM % TILE_SIZE == 0);
    constexpr int NUM_TILES = HIDDEN_DIM / TILE_SIZE;
    constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * HIDDEN_DIM / CHUNK_SIZE;

    T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
    T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
    T *__restrict__ d_output = static_cast<T *>(output_ptr);

    // Buffer layout: typed view over planner-provided SMEM page regions.
    using Buffers = RmsNormBuffers<T, BATCH_SIZE, HIDDEN_DIM>;
	    Buffers bufs(smem, task_desc);
	    T *shared_input_buffer  = bufs.input.template ptr<T>();
	    T *shared_weight_buffer = bufs.weight.template ptr<T>();
	    T *shared_output_buffer = bufs.output.template ptr<T>();
	    float *reduce_smem      = bufs.reduce.template ptr<float>();

    // Warm up input tile for the first atoms
    {
        load_smem<T, BYTES_PER_CP>(shared_input_buffer + threadIdx.x * CHUNK_SIZE,
                                    d_input + threadIdx.x * CHUNK_SIZE);
        load_smem<T, BYTES_PER_CP>(shared_weight_buffer + threadIdx.x * CHUNK_SIZE,
                                    d_weight + threadIdx.x * CHUNK_SIZE);
        cp_async_fence();
    }

    float sum = 0.0f;
#pragma unroll
    for (int for_idx = 0; for_idx < NUM_TILES; for_idx++) {
        if (for_idx + 1 < NUM_TILES) {
            load_smem<T, BYTES_PER_CP>(
                shared_input_buffer + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE,
                d_input + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
            load_smem<T, BYTES_PER_CP>(
                shared_weight_buffer + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE,
                d_weight + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
            cp_async_fence();
            cp_async_wait<1>();
        } else if (for_idx + 1 == NUM_TILES) {
            cp_async_wait<0>();
        }
        task_sync();
#pragma unroll
        for (int i = threadIdx.x; i < TILE_SIZE; i += NUM_THREADS) {
            float val = (float)shared_input_buffer[for_idx * TILE_SIZE + i];
            sum += val * val;
        }
    }

#pragma unroll
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        sum += shfl_xor_sync(sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        reduce_smem[threadIdx.x / 32] = sum;
    }
    task_sync();
    sum = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
        sum += shfl_xor_sync(sum, offset);
    }
    if (threadIdx.x == 0) {
        reduce_smem[0] = sum;
    }
    task_sync();

    float rms_rcp = rsqrt(reduce_smem[0] / float(HIDDEN_DIM) + eps);

#pragma unroll
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
        float val = (float)shared_input_buffer[i];
        float w = (float)shared_weight_buffer[i];
        val *= rms_rcp * w;
        shared_output_buffer[i] = (T)val;
    }
    task_sync();
#pragma unroll
	    for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
	        if constexpr (BYTES_PER_CP == 16) {
            *((__uint128_t *)((void *)&d_output[i * CHUNK_SIZE])) =
                *((__uint128_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
        } else if constexpr (BYTES_PER_CP == 8) {
            *((uint64_t *)((void *)&d_output[i * CHUNK_SIZE])) =
                *((uint64_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
        } else {
            *((uint32_t *)((void *)&d_output[i * CHUNK_SIZE])) =
                *((uint32_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
	        }
	    }
}

struct RmsNormLoader {
  __device__ __forceinline__ static void
  run(mirage::runtime::TaskDesc const *,
      mirage::runtime::RuntimeConfig const &,
      void *,
      int) {}
};

struct RmsNormMma {
  __device__ __forceinline__ static void
  run(mirage::runtime::TaskDesc const *,
      mirage::runtime::RuntimeConfig const &,
      void *,
      int) {}
};

struct RmsNormCompute {
  template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
  __device__ __forceinline__ static void
  run(mirage::runtime::TaskDesc const *task_desc,
      mirage::runtime::RuntimeConfig const &runtime_config,
      void *runtime_smem,
      int) {
    rms_norm_task<T, BATCH_SIZE, HIDDEN_DIM>(
        task_desc,
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->output_ptrs[0],
        1e-6f,
        runtime_smem);
  }
};

struct RmsNormStorer {
  __device__ __forceinline__ static void
  run(mirage::runtime::TaskDesc const *,
      mirage::runtime::RuntimeConfig const &,
      void *,
      int) {}
};

struct RmsNormTask : public ::kernel::v2_task::TaskInterface {
  using loader = RmsNormLoader;
  using mma = RmsNormMma;
  using compute = RmsNormCompute;
  using storer = RmsNormStorer;
};

} // namespace rmsnorm_v2
} // namespace kernel
