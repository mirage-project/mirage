
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
          typename TMA_OUT,
          typename TMA_RESIDUAL = void,
          int OUTPUT_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void
    linear_kernel_swapAB_hopper(const TMA_A &tma_a,
                                const TMA_B &tma_b,
                                const TMA_OUT &tma_out,
                                const TMA_RESIDUAL *tma_residual = nullptr,
                                void *output_ptr = nullptr) {

  constexpr int TILE_SIZE =
      REDUCTION_SIZE < TMA_A::SMEM_COL * TMA_A::SMEM_REPEAT_COL
          ? REDUCTION_SIZE
          : TMA_A::SMEM_COL * TMA_A::SMEM_REPEAT_COL;
  constexpr int CONSUMER_WARPGROUPS = 1;
  constexpr int PRODUCER_WARPGROUPS = 1;
  constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
  constexpr int THREADS_PER_WARPGROUP = 128;

  // The actual tma instructions are issued for each 64 cols when swizzle<3,3,3>
  // is used large tile size is wrapped, but
  constexpr int INPUT_TMA_TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = INPUT_TMA_TILE_SIZE;
  constexpr int OUTPUT_TMA_TILE_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;
  constexpr bool HAS_RESIDUAL = !std::is_void<TMA_RESIDUAL>::value;

  // NOTE(Yu): Assume batch size is smaller than 16, and padding the batch size to 16
  static_assert(BATCH_SIZE <= 16, "Batch size must be smaller or equal to 16 in swapAB");
  constexpr int SMEM_M_SIZE = 16;

  constexpr int TMA_TRANS_BYTES_A = sizeof(T) * TILE_SIZE * OUTPUT_ATOM_SIZE;
  constexpr int TMA_TRANS_BYTES_B = sizeof(T) * BATCH_SIZE * TILE_SIZE;
  constexpr int TMA_TRANS_BYTES_RESIDUAL =
      HAS_RESIDUAL ? sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE : 0;

  // using SM90_64x64x16_F32BF16BF16
  constexpr int NUM_ITER_N =
      (OUTPUT_SIZE + OUTPUT_ATOM_SIZE - 1) / OUTPUT_ATOM_SIZE;
  constexpr int NUM_ITER_K = (REDUCTION_SIZE + TILE_SIZE - 1) / TILE_SIZE;

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;
  int warpgroup_id = warp_idx / WARPGROUP_WARPS;

  extern __shared__ char smem_ptr[];
  uintptr_t smem = (reinterpret_cast<uintptr_t>(smem_ptr) + 1023) / 1024 * 1024;

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET = 0;

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      (SHARED_INPUT_BUFFER_OFFSET +
       sizeof(T) * Kstages * SMEM_M_SIZE * TILE_SIZE + 1023) /
      1024 * 1024;

  constexpr size_t SHARED_RESIDUAL_BUFFER_OFFSET =
      (SHARED_WEIGHT_BUFFER_OFFSET +
       sizeof(T) * Kstages * TILE_SIZE * OUTPUT_ATOM_SIZE + 1023) /
      1024 * 1024;

  constexpr size_t SHARED_MM_OUTPUT_BUFFER_OFFSET =
      HAS_RESIDUAL
          ? (SHARED_RESIDUAL_BUFFER_OFFSET +
             sizeof(T) * SMEM_M_SIZE * OUTPUT_ATOM_SIZE * Kstages + 1023) /
                1024 * 1024
          : SHARED_RESIDUAL_BUFFER_OFFSET;

  constexpr size_t SHARED_INPUT_BARRIER_OFFSET =
      (SHARED_MM_OUTPUT_BUFFER_OFFSET +
       sizeof(T) * SMEM_M_SIZE * OUTPUT_ATOM_SIZE * Kstages + 15) /
      16 * 16;

  constexpr size_t SHARED_WEIGHT_BARRIER_OFFSET =
      (SHARED_INPUT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t SHARED_RESIDUAL_BARRIER_OFFSET =
      (SHARED_WEIGHT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t SHARED_COMPUTE_DONE_OFFSET =
      HAS_RESIDUAL ? (SHARED_RESIDUAL_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8
                   : SHARED_RESIDUAL_BARRIER_OFFSET;

  constexpr size_t SHARED_RESIDUAL_DONE_OFFSET =
      (SHARED_COMPUTE_DONE_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t TOTAL_SHARED_MEMORY =
      SHARED_RESIDUAL_DONE_OFFSET + 8 * Kstages;

  static_assert(TOTAL_SHARED_MEMORY <= 224 * 1024);

  // copy input
  T *shared_input = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  // copy weight
  T *shared_weight = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);
  // residual
  T *shared_residual =
      HAS_RESIDUAL ? (T *)(smem + SHARED_RESIDUAL_BUFFER_OFFSET) : nullptr;
  // output
  T *mm_output = (T *)(smem + SHARED_MM_OUTPUT_BUFFER_OFFSET);

  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE>;
  OutputDmem output_dmem(d_output);

  // define the swizzle mode
  using InputSmem = smem_tma<T,
                             B,
                             M,
                             S,
                             SMEM_M_SIZE,
                             INPUT_TMA_TILE_SIZE,
                             TILE_SIZE / INPUT_TMA_TILE_SIZE>;
  InputSmem input_smem(shared_input);

  using WeightSmem = smem_tma<T,
                              B,
                              M,
                              S,
                              OUTPUT_ATOM_SIZE,
                              WEIGHT_TMA_TILE_SIZE,
                              TILE_SIZE / WEIGHT_TMA_TILE_SIZE>;
  WeightSmem input_weight_smem(shared_weight);

  using ResidualSmem = smem_tma<T,
                                B,
                                M,
                                S,
                                SMEM_M_SIZE,
                                OUTPUT_TMA_TILE_SIZE,
                                OUTPUT_ATOM_SIZE / OUTPUT_TMA_TILE_SIZE>;
  ResidualSmem residual_smem(shared_residual);

  using A_DESC = wgmma::mma_descriptor<WeightSmem>;
  using B_DESC = wgmma::mma_descriptor<InputSmem>;

  // using OutputSmem = smem_tma<T,
  //                             B,
  //                             M,
  //                             S,
  //                             OUTPUT_SIZE, // or OUTPUT_ATOM_SIZE?
  //                             SMEM_M_SIZE, 
  //                             1>;
  using OutputSmem = smem_tma<T,
                  B,
                  M,
                  S,
                  SMEM_M_SIZE,
                  OUTPUT_ATOM_SIZE,
                  1>;
  OutputSmem mm_output_smem(mm_output);

  // define barries
  Barrier *input_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_INPUT_BARRIER_OFFSET);
  Barrier *weight_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_WEIGHT_BARRIER_OFFSET);
  Barrier *residual_barrier =
      HAS_RESIDUAL
          ? reinterpret_cast<Barrier *>(smem + SHARED_RESIDUAL_BARRIER_OFFSET)
          : nullptr;
  Barrier *compute_done =
      reinterpret_cast<Barrier *>(smem + SHARED_COMPUTE_DONE_OFFSET);
  Barrier *residual_done =
      HAS_RESIDUAL
          ? reinterpret_cast<Barrier *>(smem + SHARED_RESIDUAL_DONE_OFFSET)
          : nullptr;

  // init the barriers and launch the first group of copy
  if (threadIdx.x == 0) {
    for (int i = 0; i < Kstages; i++) {
      initialize_barrier(input_barrier[i], 1);
      initialize_barrier(weight_barrier[i], 1);
      initialize_barrier(compute_done[i], 1);
      if constexpr (HAS_RESIDUAL) {
        initialize_barrier(residual_barrier[i], 1);
        initialize_barrier(residual_done[i], 1);
      }
    }
  }

  __syncthreads();

  // warp specialization data movement warpgroup
  if (warpgroup_id == NUM_WARPGROUPS - 1) {

    wg_decrease_regs<32>();
    if (lane_id() == 0 && warp_idx == (NUM_WARPGROUPS * WARPGROUP_WARPS - 4)) {
      for (int output_atom_idx = 0; output_atom_idx < NUM_ITER_N;
           output_atom_idx++) {
        // launch tma for residual
        int slot_residual = output_atom_idx % Kstages;
        int phase_residual = (output_atom_idx / Kstages) % 2;
        if constexpr (HAS_RESIDUAL) {
          wait(residual_done[slot_residual], phase_residual ^ 1);
        }

        if constexpr (HAS_RESIDUAL) {
          residual_smem.set_ptr(shared_residual + slot_residual *
                                                      TMA_TRANS_BYTES_RESIDUAL /
                                                      sizeof(T));
          set_barrier_transaction_bytes(residual_barrier[slot_residual],
                                        TMA_TRANS_BYTES_RESIDUAL);
          tma_residual->tma_cp_async(residual_barrier[slot_residual],
                                     residual_smem(0, 0),
                                     {output_atom_idx * OUTPUT_ATOM_SIZE, 0});
        }

        for (int i = 0; i < NUM_ITER_K; i++) {
          int slot = (output_atom_idx * NUM_ITER_K + i) % Kstages;
          int phase = ((output_atom_idx * NUM_ITER_K + i) / Kstages) % 2;
          wait(compute_done[slot], phase ^ 1);

          int tma_coords_A[2] = {i * TILE_SIZE,
                                 output_atom_idx * OUTPUT_ATOM_SIZE};
          int tma_coords_B[2] = {i * TILE_SIZE, 0};

          input_weight_smem.set_ptr(shared_weight +
                                    slot * OUTPUT_ATOM_SIZE * TILE_SIZE);
          input_smem.set_ptr(shared_input +
                             slot * SMEM_M_SIZE * TILE_SIZE);

          set_barrier_transaction_bytes(weight_barrier[slot],
                                        TMA_TRANS_BYTES_A);
          set_barrier_transaction_bytes(input_barrier[slot], TMA_TRANS_BYTES_B);

          tma_a.tma_cp_async(
              weight_barrier[slot], input_weight_smem(0, 0), tma_coords_A);
          tma_b.tma_cp_async(
              input_barrier[slot], input_smem(0, 0), tma_coords_B);
        }
      }
    }
  } else {
    // warp specialization compute warpgroup
    wg_increase_regs<160>();
    float s_frag[SMEM_M_SIZE / 2];
    for (int output_atom_idx = 0; output_atom_idx < NUM_ITER_N;
         output_atom_idx++) {
#pragma unroll
      for (int i = 0; i < SMEM_M_SIZE / 16; i++) {
        clear_8_floats(s_frag + i * 8);
      }

      for (int i = 0; i < NUM_ITER_K; i++) {
        int slot = (output_atom_idx * NUM_ITER_K + i) % Kstages;
        int phase = ((output_atom_idx * NUM_ITER_K + i) / Kstages) % 2;
        // wait input, weight
        wait(input_barrier[slot], phase);
        wait(weight_barrier[slot], phase);
#if 0
        if (threadIdx.x == 0) {
         printf("i: %d\n", i);
         printf("input_smem ptr: %p\n", input_smem(0, 0));
         printf("input_weight_smem ptr: %p\n", input_weight_smem(0, 0));
         printf("input_smem\n");
         for (int j = 0; j < BATCH_SIZE; j++) {
           for (int k = 0; k < TILE_SIZE; k++) {
             printf("%f ", (float)input_smem.at(j, k));
           }
           printf("\n");
         }
         printf("input_weight_smem\n");
        for (int j = 0; j < OUTPUT_ATOM_SIZE; j++) {
         for (int k = 0; k < TILE_SIZE; k++) {
             printf("%f ", (float)input_weight_smem.at(j, k));
           }
           printf("\n");
         }
       }
#endif

        input_weight_smem.set_ptr(shared_weight +
                                  (slot)*OUTPUT_ATOM_SIZE * TILE_SIZE);
        input_smem.set_ptr(shared_input + (slot)*SMEM_M_SIZE * TILE_SIZE);
        A_DESC a_desc(input_weight_smem(0, 0));
        B_DESC b_desc(input_smem(0, 0));

        wgmma::warpgroup_fence_fragment(s_frag);
        wgmma::warpgroup_arrive();
        // wgmma
        wgmma::mma<bfloat16,
                   64, // output atom size fixed to 64
                   16, // Assume batch size is not larger than 16
                   16,
                   WeightSmem,
                   InputSmem,
                   A_DESC,
                   B_DESC,
                   false,
                   false>(s_frag, a_desc, b_desc);
        wgmma::mma_commit_group();
        wgmma::mma_async_wait();
        wgmma::warpgroup_fence_fragment(s_frag);

        // flip compute done
        if (idx_in_warp == 0 && warp_idx % 4 == 0) {
          arrive(compute_done[slot], 1);
        }
      }

      int slot_residual;
      int phase_residual;
      if constexpr (HAS_RESIDUAL) {
        slot_residual = output_atom_idx % Kstages;
        phase_residual = output_atom_idx / Kstages % 2;
        wait(residual_barrier[slot_residual], phase_residual);
        residual_smem.set_ptr(shared_residual + slot_residual *
                                                    TMA_TRANS_BYTES_RESIDUAL /
                                                    sizeof(T));
      }

      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16:~:text=The%20layout%20of%20the%20fragments%20held%20by%20different%20threads%20is%20shown%20in%20Figure%20149.
      // write back to shared memory
      store_async_wait<Kstages - 1>();
      int slot_output = output_atom_idx % Kstages;
      mm_output_smem.set_ptr(mm_output +
                             slot_output * BATCH_SIZE * OUTPUT_ATOM_SIZE);
#pragma unroll 1
      for (uint32_t i = 0; i < (SMEM_M_SIZE / 4); i++) {
        int row = (warp_idx % 4) * 16 + (i % 2) * 8 + idx_in_warp / 4;
        int col = (i / 2) * 8 + (idx_in_warp % 4) * 2;

        // if (1) {
          //   printf("threadIdx.x: %d, i: %d, warp_idx: %d, idx_in_warp: %d, (row, col): (%d, %d), (float)s_frag[%d] = %f\n", threadIdx.x, output_atom_idx, warp_idx, idx_in_warp, row, col, i * 2, (float)(bfloat16(s_frag[i * 2])));
          //   printf("threadIdx.x: %d, i: %d, warp_idx: %d, idx_in_warp: %d, (row, col): (%d, %d), (float)s_frag[%d] = %f\n", threadIdx.x, output_atom_idx, warp_idx, idx_in_warp, row, col, i * 2 + 1, (float)(bfloat16(s_frag[i * 2 + 1])));
          // }
          if constexpr (HAS_RESIDUAL) {
            // mm_output_smem.at(row, col) =
            // bfloat16(s_frag[i * 2]) + residual_smem.at(col, row);
            // mm_output_smem.at(row, col + 1) =
            // bfloat16(s_frag[i * 2 + 1]) + residual_smem.at(col + 1, row);
              mm_output_smem.at(col, row) =
              bfloat16(s_frag[i * 2]) + residual_smem.at(col, row);
              mm_output_smem.at(col + 1, row) =
              bfloat16(s_frag[i * 2 + 1]) + residual_smem.at(col + 1, row);
          } else {
            mm_output_smem.at(col, row) = bfloat16(s_frag[i * 2]);
            mm_output_smem.at(col + 1, row) = bfloat16(s_frag[i * 2 + 1]);
          }
#if 0
          if (1) {
            printf("has residual: %d, residual_smem.at(%d, %d) = %f\n", HAS_RESIDUAL, row, col, (float)residual_smem.at(row, col));
            printf("has residual: %d, residual_smem.at(%d, %d) = %f\n", HAS_RESIDUAL, row, col + 1, (float)residual_smem.at(row, col + 1));
            printf("threadIdx.x: %d, i: %d, (row, col): (%d, %d), (float)s_frag[%d] = %f\n", threadIdx.x, i, row, col, i * 2, (float)(bfloat16(s_frag[i * 2])));
            printf("threadIdx.x: %d, i: %d, (row, col): (%d, %d), (float)s_frag[%d] = %f\n", threadIdx.x, i, row, col, i * 2 + 1, (float)(bfloat16(s_frag[i * 2 + 1])));
          }
#endif
      }

      // make sure generic proxy's modification to smem is visible to tma store
      // async proxy this is intra-thread sync
      async_proxy_fence();

      // this is inter-thread sync
      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(1);
#if 0
      if (threadIdx.x == 0) {

        printf("mm_output_smem\n");
        for (int j = 0; j < BATCH_SIZE; j++) {
          for (int k = 0; k < OUTPUT_SIZE; k++) {
            printf("%f ", (float)mm_output_smem.at(k, j));
          }
          printf("\n");
        }
      }
#endif
      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(1);

      // copy back to dmem
      if (warp_idx % 4 == 0 && lane_id() == 0) {
      //   tma_out.tma_store_async(mm_output_smem(0, 0),
      //                           {output_atom_idx * OUTPUT_ATOM_SIZE, 0});
      //   store_commit_group();
        if constexpr (HAS_RESIDUAL) {
          arrive(residual_done[slot_residual], 1);
        }
      }
      // Final writeback: store accumulated output (residual already included if
      // any)

      constexpr int CHUNK_SIZE = 16 / sizeof(T);
      constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
      constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
      constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

    #pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
        int row = i / CHUNKS_PER_ROW_C;
        int col = (i % CHUNKS_PER_ROW_C) << log2_CHUNK_SIZE;
        // printf("threadIdx.x: %d, row = %d, col = %d, output_dmem_ptr = %p, 128 bit aligned = %d\n", threadIdx.x, row, col, ((void *)&output_dmem.at(col, row + output_atom_idx * OUTPUT_ATOM_SIZE)), ((uintptr_t)&output_dmem.at(col, row + output_atom_idx * OUTPUT_ATOM_SIZE)) % 16 == 0);
        *((__uint128_t *)((void *)&output_dmem.at(row, col + output_atom_idx * OUTPUT_ATOM_SIZE))) = *((__uint128_t *)((void *)&mm_output_smem.at(row, col)));
      }


// #pragma unroll
//       for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE * SMEM_M_SIZE; i += NUM_THREADS) {
//         int src_row = i / SMEM_M_SIZE;
//         int src_col = (i % SMEM_M_SIZE);
//         output_dmem.at(src_col, src_row+output_atom_idx * OUTPUT_ATOM_SIZE) = mm_output_smem.at(src_row, src_col);
//       }
    }
  }
  // store_async_wait<0>();
  // __syncthreads();
}

} // namespace kernel