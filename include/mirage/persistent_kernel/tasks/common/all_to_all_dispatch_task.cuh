/* Copyright 2023-2025 CMU
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

#include "cutlass/cutlass.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// nvshmem.h and nvshmemx.h are intentionally NOT included here.
// They must be included by the parent compilation unit (.cu file) BEFORE
// this header, so that device-side NVSHMEM APIs are available to the
// #if USE_NVSHMEM blocks below without contaminating other translation units.

namespace mirage {
namespace kernel {

// __device__ version: MPK-compatible single-block dispatch.
// Called from within the persistent kernel's _execute_task (one task per
// worker thread block).  All tokens are processed by a single block —
// no cooperative launch or grid.sync() required.
//
// recv_buf  : NVSHMEM symmetric buffer of shape
//             [WORLD_SIZE * BATCH_SIZE * TOPK, HIDDEN_DIM].
//             Source PE `r` exclusively writes to the slice starting at
//             r * BATCH_SIZE * TOPK, so different source PEs never collide.
//
// sync_sigs : NVSHMEM symmetric uint64_t[WORLD_SIZE] signal array.
//             After all puts, thread 0 writes sync_sigs[rank] = 1 on every PE
//             via nvshmem_uint64_p so the combine phase can signal-wait.
//
template <typename T,
          int BATCH_SIZE,
          int HIDDEN_DIM,
          int TOPK,
          int WORLD_SIZE>
__device__ void all_to_all_dispatch_device_impl(
    T const    *input_tokens,      // [BATCH_SIZE, HIDDEN_DIM]
    int const  *routing_indices,   // [BATCH_SIZE, TOPK]
    T const    *routing_weights,   // [BATCH_SIZE, TOPK] (unused)
    T          *recv_buf,          // NVSHMEM symmetric
    int        *send_counts,       // [WORLD_SIZE] — output
    int        *send_offsets,      // [WORLD_SIZE] — scratch/output
    int         num_experts,
    int         experts_per_rank,
    int         rank,
    uint64_t   *sync_sigs) {       // NVSHMEM symmetric [WORLD_SIZE]

  const int tid = threadIdx.x;

  __shared__ int local_counts[WORLD_SIZE];
  __shared__ int send_base[WORLD_SIZE];

  // ── PHASE 1: COUNT ─────────────────────────────────────────────────────────
  if (tid < WORLD_SIZE) local_counts[tid] = 0;
  __syncthreads();

  for (int t = tid; t < BATCH_SIZE; t += blockDim.x) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = routing_indices[t * TOPK + k];
      int dest_rank = expert_id / experts_per_rank;
      atomicAdd(&local_counts[dest_rank], 1);
    }
  }
  __syncthreads();

  // ── PHASE 2: COMPUTE OFFSETS ───────────────────────────────────────────────
  if (tid < WORLD_SIZE) send_counts[tid] = local_counts[tid];
  __syncthreads();

  if (tid == 0) {
    int cumulative = 0;
    for (int r = 0; r < WORLD_SIZE; r++) {
      send_offsets[r] = cumulative;
      cumulative += send_counts[r];
    }
  }
  __syncthreads();

  // Snapshot initial offsets before Phase 3 consumes them with atomicAdd.
  if (tid < WORLD_SIZE) send_base[tid] = send_offsets[tid];
  __syncthreads();

  // ── PHASE 3: DISPATCH ─────────────────────────────────────────────────────
  for (int t = tid; t < BATCH_SIZE; t += blockDim.x) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = routing_indices[t * TOPK + k];
      int dest_rank = expert_id / experts_per_rank;

      int write_pos = atomicAdd(&send_offsets[dest_rank], 1);
      int local_pos = write_pos - send_base[dest_rank];
      // Each source PE has its own section in dest's recv_buf — no overlap.
      int recv_pos  = rank * BATCH_SIZE * TOPK + local_pos;

      const T *src_ptr  = input_tokens + t * HIDDEN_DIM;
      T       *dst_ptr  = recv_buf + recv_pos * HIDDEN_DIM;

      constexpr int ELEMS_PER_VEC = 16 / sizeof(T);
      constexpr int NUM_VECS      = HIDDEN_DIM / ELEMS_PER_VEC;
      constexpr int REMAINDER     = HIDDEN_DIM % ELEMS_PER_VEC;

      if (dest_rank == rank) {
        // Local copy: vectorised for bandwidth.
        #pragma unroll
        for (int v = 0; v < NUM_VECS; v++) {
          uint4 data = *reinterpret_cast<const uint4 *>(src_ptr + v * ELEMS_PER_VEC);
          *reinterpret_cast<uint4 *>(dst_ptr + v * ELEMS_PER_VEC) = data;
        }
        if constexpr (REMAINDER > 0) {
          #pragma unroll
          for (int r = 0; r < REMAINDER; r++) {
            dst_ptr[NUM_VECS * ELEMS_PER_VEC + r] = src_ptr[NUM_VECS * ELEMS_PER_VEC + r];
          }
        }
      } else {
#if USE_NVSHMEM
        constexpr size_t COPY_BYTES = HIDDEN_DIM * sizeof(T);
        nvshmem_putmem_nbi(dst_ptr, src_ptr, COPY_BYTES, dest_rank);
#endif
      }
    }
  }
  __syncthreads();

  // ── PHASE 4: FENCE + SIGNAL ───────────────────────────────────────────────
  if (tid == 0) {
#if USE_NVSHMEM
    nvshmem_quiet();
    for (int pe = 0; pe < WORLD_SIZE; pe++) {
      nvshmem_uint64_p(sync_sigs + rank, 1ULL, pe);
    }
#else
    __threadfence_system();
    ((volatile uint64_t *)sync_sigs)[rank] = 1ULL;
#endif
  }
}

// __global__ wrapper for standalone tests (NOT for MPK persistent kernel).
// Uses cooperative_groups::grid.sync() for multi-block coordination.
//
// routing_weights  : kept in signature for API compatibility; unused here.
// grid_counter     : reserved (cooperative-launch scratch); may be nullptr.
//
template <typename T,
          int BATCH_SIZE,
          int HIDDEN_DIM,
          int TOPK,
          int WORLD_SIZE>
__global__ void all_to_all_dispatch_task_impl(
    T const    *input_tokens,      // [BATCH_SIZE, HIDDEN_DIM]
    int const  *routing_indices,   // [BATCH_SIZE, TOPK]
    T const    *routing_weights,   // [BATCH_SIZE, TOPK] (unused)
    T          *recv_buf,          // NVSHMEM symmetric
    int        *send_counts,       // [WORLD_SIZE] — output
    int        *send_offsets,      // [WORLD_SIZE] — scratch/output
    int         num_experts,
    int         experts_per_rank,
    int         rank,
    int        *grid_counter,      // reserved
    uint64_t   *sync_sigs) {       // NVSHMEM symmetric [WORLD_SIZE]

  using namespace cooperative_groups;
  auto grid = this_grid();

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ int local_counts[WORLD_SIZE];
  // Initial send_offsets snapshot; used to compute per-source slot in recv_buf.
  __shared__ int send_base[WORLD_SIZE];

  // ── PHASE 1: COUNT ─────────────────────────────────────────────────────────
  if (tid < WORLD_SIZE) local_counts[tid] = 0;
  __syncthreads();

  const int batch_per_block = BATCH_SIZE / gridDim.x;
  const int start_token     = bid * batch_per_block;
  const int end_token       = start_token + batch_per_block;

  for (int t = start_token + tid; t < end_token; t += blockDim.x) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = routing_indices[t * TOPK + k];
      int dest_rank = expert_id / experts_per_rank;
      atomicAdd(&local_counts[dest_rank], 1);
    }
  }
  __syncthreads();

  // ── PHASE 2: GLOBAL OFFSETS ────────────────────────────────────────────────
  if (tid < WORLD_SIZE) atomicAdd(&send_counts[tid], local_counts[tid]);
  grid.sync();

  if (bid == 0 && tid == 0) {
    int cumulative = 0;
    for (int r = 0; r < WORLD_SIZE; r++) {
      send_offsets[r] = cumulative;
      cumulative += send_counts[r];
    }
  }
  grid.sync();

  // Snapshot initial offsets before Phase 3 consumes them with atomicAdd.
  if (tid < WORLD_SIZE) send_base[tid] = send_offsets[tid];
  __syncthreads();

  // ── PHASE 3: DISPATCH ─────────────────────────────────────────────────────
  for (int t = start_token + tid; t < end_token; t += blockDim.x) {
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
      int expert_id = routing_indices[t * TOPK + k];
      int dest_rank = expert_id / experts_per_rank;

      int write_pos = atomicAdd(&send_offsets[dest_rank], 1);
      int local_pos = write_pos - send_base[dest_rank];
      // Each source PE has its own section in dest's recv_buf — no overlap.
      int recv_pos  = rank * BATCH_SIZE * TOPK + local_pos;

      const T *src_ptr  = input_tokens + t * HIDDEN_DIM;
      T       *dst_ptr  = recv_buf + recv_pos * HIDDEN_DIM;

      constexpr size_t       COPY_BYTES    = HIDDEN_DIM * sizeof(T);
      constexpr int          ELEMS_PER_VEC = 16 / sizeof(T);
      constexpr int          NUM_VECS      = HIDDEN_DIM / ELEMS_PER_VEC;
      constexpr int          REMAINDER     = HIDDEN_DIM % ELEMS_PER_VEC;

      if (dest_rank == rank) {
        // Local copy: vectorised for bandwidth.
        #pragma unroll
        for (int v = 0; v < NUM_VECS; v++) {
          uint4 data = *reinterpret_cast<const uint4 *>(src_ptr + v * ELEMS_PER_VEC);
          *reinterpret_cast<uint4 *>(dst_ptr + v * ELEMS_PER_VEC) = data;
        }
        if constexpr (REMAINDER > 0) {
          #pragma unroll
          for (int r = 0; r < REMAINDER; r++) {
            dst_ptr[NUM_VECS * ELEMS_PER_VEC + r] = src_ptr[NUM_VECS * ELEMS_PER_VEC + r];
          }
        }
      } else {
#if USE_NVSHMEM
        // Non-blocking NVSHMEM put: write this PE's token slice to remote PE.
        nvshmem_putmem_nbi(dst_ptr, src_ptr, COPY_BYTES, dest_rank);
#endif
      }
    }
  }
  __syncthreads();

  // ── PHASE 4: FENCE + SIGNAL ───────────────────────────────────────────────
  grid.sync();

  if (bid == 0 && tid == 0) {
#if USE_NVSHMEM
    // Ensure all NVSHMEM puts have been delivered to their destinations.
    nvshmem_quiet();
    // Broadcast sync_sigs[rank] = 1 to every PE so combines can signal-wait.
    // nvshmem_uint64_p uses the point-to-point AMO path (no collective headers
    // needed) and is defined in device/nvshmem_defines.h.
    for (int pe = 0; pe < WORLD_SIZE; pe++) {
      nvshmem_uint64_p(sync_sigs + rank, 1ULL, pe);
    }
#else
    __threadfence_system();
    ((volatile uint64_t *)sync_sigs)[rank] = 1ULL;
#endif
  }
}

} // namespace kernel
} // namespace mirage
