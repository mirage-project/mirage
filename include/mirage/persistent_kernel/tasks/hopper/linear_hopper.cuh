
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
#include "../common.h"
#include "../dmem_layout.cuh"
#include "../element_binary.cuh"
#include "../element_unary.cuh"
#include "../reduction.cuh"
#include "../smem_layout.cuh"
#include "../utils.cuh"
#include "smem_layout_tma.cuh"
#include "tma.cuh"
#include "utils.cuh"
#include "wgmma.cuh"
namespace kernel {

using namespace tma;
using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int Kstages,
          typename TMA_A,
          typename TMA_B,
          typename TMA_RESIDUAL,
          typename TMA_OUT,
          int OUTPUT_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void
    linear_kernel_hopper(const TMA_A &tma_a,
                         const TMA_B &tma_b,
                         const TMA_RESIDUAL &tma_residual,
                         const TMA_OUT &tma_out) {

  constexpr int chunk_size = 16 / sizeof(T);
  constexpr int TILE_SIZE =
      REDUCTION_SIZE < TMA_A::SMEM_COL * TMA_A::SMEM_REPEAT_COL
          ? REDUCTION_SIZE
          : TMA_A::SMEM_COL * TMA_A::SMEM_REPEAT_COL;
  constexpr int THREADS_PER_WARPGROUP = 128;
  constexpr int CONSUMER_WARPGROUPS = 1;
  constexpr int PRODUCER_WARPGROUPS = 1;
  constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;

  // The actual tma instructions are issued for each 64 cols when swizzle<3,3,3>
  // is used large tile size is wrapped, but
  constexpr int INPUT_TMA_TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = INPUT_TMA_TILE_SIZE;
  constexpr int OUTPUT_TMA_TILE_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;

  constexpr int TMA_TRANS_BYTES_A = sizeof(T) * BATCH_SIZE * TILE_SIZE;
  constexpr int TMA_TRANS_BYTES_B = sizeof(T) * TILE_SIZE * OUTPUT_SIZE;
  constexpr int TMA_TRANS_BYTES_RESIDUAL = sizeof(T) * BATCH_SIZE * OUTPUT_SIZE;

  // using SM90_64x64x16_F32BF16BF16
  constexpr int num_n = OUTPUT_SIZE / 64;
  constexpr int num_m = BATCH_SIZE / 64;
  constexpr int num_k = REDUCTION_SIZE / TILE_SIZE;

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;
  int warpgroup_id = warp_idx / WARPGROUP_WARPS;

  extern __shared__ char smem[];

  constexpr size_t ZERO_BUFFER_OFFSET = 0;

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET = ZERO_BUFFER_OFFSET + 0;

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET + sizeof(T) * Kstages * BATCH_SIZE * TILE_SIZE;

  constexpr size_t SHARED_RESIDUAL_BUFFER_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * Kstages * TILE_SIZE * OUTPUT_SIZE;

  constexpr size_t SHARED_MM_OUTPUT_BUFFER_OFFSET =
      SHARED_RESIDUAL_BUFFER_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_SIZE;

  constexpr size_t SHARED_INPUT_BARRIER_OFFSET =
      (SHARED_MM_OUTPUT_BUFFER_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_SIZE +
       7) /
      8 * 8;

  constexpr size_t SHARED_WEIGHT_BARRIER_OFFSET =
      (SHARED_INPUT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t SHARED_RESIDUAL_BARRIER_OFFSET =
      (SHARED_WEIGHT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t SHARED_COMPUTE_DONE_OFFSET =
      (SHARED_RESIDUAL_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  // copy input
  T *shared_input = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  // copy weight
  T *shared_weight = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);
  // residual
  T *shared_residual = (T *)(smem + SHARED_RESIDUAL_BUFFER_OFFSET);
  // output
  T *mm_output = (T *)(smem + SHARED_MM_OUTPUT_BUFFER_OFFSET);

  // define the swizzle mode
  using InputSmem = smem_tma<T,
                             B,
                             M,
                             S,
                             BATCH_SIZE,
                             INPUT_TMA_TILE_SIZE,
                             TILE_SIZE / INPUT_TMA_TILE_SIZE>;
  InputSmem input_smem(shared_input);
  InputSmem input_smem_buffer(shared_input);

  using WeightSmem = smem_tma<T,
                              B,
                              M,
                              S,
                              OUTPUT_SIZE,
                              WEIGHT_TMA_TILE_SIZE,
                              TILE_SIZE / WEIGHT_TMA_TILE_SIZE>;
  WeightSmem input_weight_smem(shared_weight);
  WeightSmem input_weight_smem_buffer(shared_weight);

  using ResidualSmem = smem_tma<T,
                                0,
                                0,
                                0,
                                BATCH_SIZE,
                                OUTPUT_TMA_TILE_SIZE,
                                OUTPUT_SIZE / OUTPUT_TMA_TILE_SIZE>;
  ResidualSmem residual_smem(shared_residual);

  using A_DESC = wgmma::mma_descriptor<InputSmem>;
  using B_DESC = wgmma::mma_descriptor<WeightSmem>;

  using OutputSmem = smem_tma<T,
                              0,
                              0,
                              0,
                              BATCH_SIZE,
                              OUTPUT_TMA_TILE_SIZE,
                              OUTPUT_SIZE / OUTPUT_TMA_TILE_SIZE>;
  OutputSmem mm_output_smem(mm_output);
  float s_frag[32];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    clear_8_floats(s_frag + i * 8);
  }

  // define barries
  Barrier *input_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_INPUT_BARRIER_OFFSET);
  Barrier *weight_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_WEIGHT_BARRIER_OFFSET);
  Barrier *residual_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_RESIDUAL_BARRIER_OFFSET);
  Barrier *compute_done =
      reinterpret_cast<Barrier *>(smem + SHARED_COMPUTE_DONE_OFFSET);

  // init the barriers and launch the first group of copy
  if (threadIdx.x == 0) {
    for (int i = 0; i < Kstages; i++) {
      initialize_barrier(input_barrier[i], 1);
      initialize_barrier(weight_barrier[i], 1);
      initialize_barrier(compute_done[i], 1);
    }
    initialize_barrier(residual_barrier[0], 1);
  }

  __syncthreads();

  // warp specialization data movement warpgroup
  if (warpgroup_id == NUM_WARPGROUPS - 1) {

    // wg_decrease_regs<32>();
    if (lane_id() == 0 && warp_idx == (NUM_WARPGROUPS * WARPGROUP_WARPS - 4)) {
      int prefetch = (Kstages < num_k) ? Kstages : num_k;
      for (int i = 0; i < prefetch; i++) {
        int slot = i % Kstages;
        int tma_coords_A[2] = {i * TILE_SIZE, 0};
        int tma_coords_B[2] = {i * TILE_SIZE, 0};

        input_smem_buffer.set_ptr(shared_input +
                                  slot * TMA_TRANS_BYTES_A / sizeof(T));
        set_barrier_transaction_bytes(input_barrier[slot], TMA_TRANS_BYTES_A);
        tma_a.tma_cp_async(
            input_barrier[slot], input_smem_buffer(0, 0), tma_coords_A);

        input_weight_smem_buffer.set_ptr(shared_weight +
                                         slot * TMA_TRANS_BYTES_B / sizeof(T));
        set_barrier_transaction_bytes(weight_barrier[slot], TMA_TRANS_BYTES_B);
        tma_b.tma_cp_async(
            weight_barrier[slot], input_weight_smem_buffer(0, 0), tma_coords_B);
      }

      // launch tma for residual
      set_barrier_transaction_bytes(residual_barrier[0],
                                    TMA_TRANS_BYTES_RESIDUAL);
      tma_residual.tma_cp_async(
          residual_barrier[0], residual_smem(0, 0), {0, 0});

      for (int i = prefetch; i < num_k; i++) {
        int slot = i % Kstages;
        int phase = (i / Kstages) % 2;
        wait(compute_done[slot], phase ^ 1);

        int tma_coords_A[2] = {i * TILE_SIZE, 0};
        int tma_coords_B[2] = {i * TILE_SIZE, 0};

        input_smem_buffer.set_ptr(shared_input +
                                  slot * TMA_TRANS_BYTES_A / sizeof(T));
        set_barrier_transaction_bytes(input_barrier[slot], TMA_TRANS_BYTES_A);
        tma_a.tma_cp_async(
            input_barrier[slot], input_smem_buffer(0, 0), tma_coords_A);

        input_weight_smem_buffer.set_ptr(shared_weight +
                                         slot * TMA_TRANS_BYTES_B / sizeof(T));
        set_barrier_transaction_bytes(weight_barrier[slot], TMA_TRANS_BYTES_B);
        tma_b.tma_cp_async(
            weight_barrier[slot], input_weight_smem_buffer(0, 0), tma_coords_B);
      }
    }
  } else {
    // warp specialization compute warpgroup
    // wg_increase_regs<160>();
    for (int i = 0; i < num_k; i++) {
      int slot = i % Kstages;
      int phase = (i / Kstages) % 2;
      // wait input, weight
      wait(input_barrier[slot], phase);
      wait(weight_barrier[slot], phase);

      input_smem.set_ptr(shared_input + (slot)*TMA_TRANS_BYTES_A / sizeof(T));
      input_weight_smem.set_ptr(shared_weight +
                                (slot)*TMA_TRANS_BYTES_B / sizeof(T));

      //  if (threadIdx.x == 0) {
      //    printf("i: %d\n", i);
      //    printf("input_smem ptr: %p\n", input_smem(0, 0));
      //    printf("input_weight_smem ptr: %p\n", input_weight_smem(0, 0));
      //    printf("input_smem\n");
      //    for (int j = 0; j < BATCH_SIZE; j++) {
      //      for (int k = 0; k < TILE_SIZE; k++) {
      //        printf("%f ", (float)input_smem.at(j, k));
      //      }
      //      printf("\n");
      //    }
      //  printf("input_weight_smem\n");
      //  for (int j = 0; j < TILE_SIZE; j++) {
      //    for (int k = 0; k < OUTPUT_SIZE; k++) {
      //      printf("%f ", (float)input_weight_smem.at(j, k));
      //    }
      //    printf("\n");
      //  }
      //  }

      A_DESC a_desc(input_smem(0, 0));
      B_DESC b_desc(input_weight_smem(0, 0));

      //   wgmma::warpgroup_fence_fragment(s_frag);
      wgmma::warpgroup_arrive();
      // wgmma
      wgmma::mma<bfloat16,
                 64,
                 OUTPUT_SIZE,
                 16,
                 InputSmem,
                 WeightSmem,
                 A_DESC,
                 B_DESC,
                 false,
                 false>(s_frag, a_desc, b_desc);
      wgmma::mma_commit_group();
      wgmma::mma_async_wait();
      //   wgmma::warpgroup_fence_fragment(s_frag);

      // flip compute done
      if (idx_in_warp == 0 && warp_idx % 4 == 0) {
        arrive(compute_done[slot], 1);
      }
    }
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16:~:text=The%20layout%20of%20the%20fragments%20held%20by%20different%20threads%20is%20shown%20in%20Figure%20149.
    // write back to shared memory

    // asm volatile("" ::: "memory");

#pragma unroll 1
    for (uint32_t i = 0; i < (OUTPUT_SIZE / 4); i++) {
      int row = (warp_idx % 4) * 16 + (i % 2) * 8 + idx_in_warp / 4;
      int col = (i / 2) * 8 + (idx_in_warp % 4) * 2;
      mm_output_smem.at(row, col) =
          bfloat16(s_frag[i * 2]) + residual_smem.at(row, col);
      mm_output_smem.at(row, col + 1) =
          bfloat16(s_frag[i * 2 + 1]) + residual_smem.at(row, col + 1);
    }

    // make sure generic proxy's modification to smem is visible to tma store
    // async proxy this is intra-thread sync
    async_proxy_fence();

    // this is inter-thread sync
    // wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);

    // copy back to dmem
    if (warp_idx % 4 == 0 && lane_id() == 0) {
      tma_out.tma_store_async(mm_output_smem(0, 0), {0, 0});
      store_commit_group();
    }
  }
}

} // namespace kernel