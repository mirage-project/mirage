// v2 RMS-norm task. Adapted from ampere/rmsnorm.cuh (rms_norm_impl).
//
// Changes from source:
//   - Thread guard: only first NUM_THREADS (128) participate, so it can be
//     launched inside v2's 512-thread runtime block.
//   - __syncthreads() → bar.sync 2, NUM_THREADS (named barrier, 128 threads).
//     Barrier 0 is implicit block-wide __syncthreads; barrier 1 is used by
//     linear_task; use barrier 2 for norm.
//
// Input / output: raw GMEM pointers. Task loads input and weight via cp.async
// to SMEM, computes in place, and writes output back to GMEM.

#pragma once
#include "../common/common_header.cuh"
#include "../common/worker_config.h"

namespace kernel {
namespace rmsnorm_v2 {

// Named barrier for this task (128 threads participate).
__device__ inline void task_sync() {
    asm volatile("bar.sync 2, %0;" :: "n"(NUM_THREADS));
}

template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __noinline__ void rms_norm_task(
    void const *input_ptr,
    void const *weight_ptr,
    void *output_ptr,
    float eps
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

    constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET = sizeof(T) * HIDDEN_DIM;
    constexpr size_t SHARED_OUTPUT_BUFFER_OFFSET =
        SHARED_WEIGHT_BUFFER_OFFSET + sizeof(T) * HIDDEN_DIM;
    constexpr size_t REDUCE_BUFFER_OFFSET =
        SHARED_OUTPUT_BUFFER_OFFSET + sizeof(T) * HIDDEN_DIM;
    T *shared_input_buffer  = (T *)(smem);
    T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);
    T *shared_output_buffer = (T *)(smem + SHARED_OUTPUT_BUFFER_OFFSET);
    float *reduce_smem = reinterpret_cast<float *>(smem + REDUCE_BUFFER_OFFSET);

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

} // namespace rmsnorm_v2
} // namespace kernel
