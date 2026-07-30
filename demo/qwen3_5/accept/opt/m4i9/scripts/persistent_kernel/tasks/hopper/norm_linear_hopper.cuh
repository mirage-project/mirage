
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
#include "barrier.cuh"
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
          typename TMA_INPUT,
          typename TMA_NORM_WEIGHT,
          typename TMA_LINEAR_WEIGHT,
          typename TMA_OUT,
          int OUTPUT_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void
    norm_linear_kernel_hopper(const TMA_INPUT &tma_input,
                              const TMA_NORM_WEIGHT &tma_norm_weight,
                              const TMA_LINEAR_WEIGHT &tma_linear_weight,
                              const TMA_OUT &tma_out,
                              float eps) {

  constexpr int chunk_size = 16 / sizeof(T);
  constexpr int TILE_SIZE = 64;
  constexpr int THREADS_PER_WARPGROUP = 128;
  constexpr int CONSUMER_WARPGROUPS = 1;
  constexpr int PRODUCER_WARPGROUPS = 1;
  constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;

  constexpr int TMA_TRANS_BYTES_INPUT = sizeof(T) * BATCH_SIZE * TILE_SIZE;
  constexpr int TMA_TRANS_BYTES_NORM_WEIGHT =
      sizeof(T) * BATCH_SIZE * TILE_SIZE;
  constexpr int TMA_TRANS_BYTES_LINEAR_WEIGHT =
      sizeof(T) * TILE_SIZE * OUTPUT_SIZE;

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

  constexpr size_t SHARED_NORM_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET + sizeof(T) * Kstages * BATCH_SIZE * TILE_SIZE;

  constexpr size_t SHARED_LINEAR_WEIGHT_BUFFER_OFFSET =
      SHARED_NORM_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * Kstages * BATCH_SIZE * TILE_SIZE;

  constexpr size_t SHARED_MUL_OUTPUT_BUFFER_OFFSET =
      SHARED_LINEAR_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * Kstages * TILE_SIZE * OUTPUT_SIZE;

  constexpr size_t SHARED_ELEMENTARY_UNARY_OUTPUT_OFFSET =
      SHARED_MUL_OUTPUT_BUFFER_OFFSET + sizeof(T) * BATCH_SIZE * TILE_SIZE;

  constexpr size_t SHARED_MM_OUTPUT_OFFSET =
      SHARED_ELEMENTARY_UNARY_OUTPUT_OFFSET +
      sizeof(T) * BATCH_SIZE * OUTPUT_SIZE;

  constexpr size_t SHARED_REDUCTION_OUTPUT_OFFSET =
      SHARED_MM_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_SIZE;

  constexpr size_t SHARED_OUTPUT_OFFSET =
      SHARED_REDUCTION_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE;

  constexpr size_t SHARED_INPUT_BARRIER_OFFSET =
      (SHARED_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_SIZE + 7) / 8 * 8;

  constexpr size_t SHARED_NORM_WEIGHT_BARRIER_OFFSET =
      (SHARED_INPUT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t SHARED_LINEAR_WEIGHT_BARRIER_OFFSET =
      (SHARED_NORM_WEIGHT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  constexpr size_t SHARED_COMPUTE_DONE_OFFSET =
      (SHARED_LINEAR_WEIGHT_BARRIER_OFFSET + 8 * Kstages + 7) / 8 * 8;

  // input
  T *shared_input = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  // norm weight
  T *shared_norm_weight = (T *)(smem + SHARED_NORM_WEIGHT_BUFFER_OFFSET);
  // linear weight
  T *shared_linear_weight = (T *)(smem + SHARED_LINEAR_WEIGHT_BUFFER_OFFSET);
  // mul output
  T *shared_mul_output = (T *)(smem + SHARED_MUL_OUTPUT_BUFFER_OFFSET);
  // element unary output
  T *shared_element_unary_output =
      (T *)(smem + SHARED_ELEMENTARY_UNARY_OUTPUT_OFFSET);
  // reduction output
  T *shared_reduction_output = (T *)(smem + SHARED_REDUCTION_OUTPUT_OFFSET);
  // output
  T *mm_output = (T *)(smem + SHARED_MM_OUTPUT_OFFSET);

  // define the swizzle mode
  using InputSmem = smem_row<T, B, M, S, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  InputSmem input_smem(shared_input);
  InputSmem input_smem_buffer(shared_input);
  InputSmem mul_output_smem(shared_mul_output);
  InputSmem element_unary_smem(shared_element_unary_output);

  using NormWeightSmem = smem_row<T, B, M, S, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  NormWeightSmem norm_weight_smem(shared_norm_weight);
  NormWeightSmem norm_weight_smem_buffer(shared_norm_weight);

  using LinearWeightSmem =
      smem_col<T, B, M, S, TILE_SIZE, OUTPUT_SIZE, TILE_SIZE>;
  LinearWeightSmem linear_weight_smem(shared_linear_weight);
  LinearWeightSmem linear_weight_smem_buffer(shared_linear_weight);

  using A_DESC = wgmma::mma_descriptor<InputSmem>;
  using B_DESC = wgmma::mma_descriptor<LinearWeightSmem>;

  using MmaOutputSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  MmaOutputSmem mm_output_smem(mm_output);

  using ReductionOutputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, 1, 1>;
  ReductionOutputSmem reduction_output_smem(shared_reduction_output);

  float s_frag[32];

#pragma unroll
  for (int i = 0; i < 4; i++) {
    clear_8_floats(s_frag + i * 8);
  }

  clear_smem_buffer<T, BATCH_SIZE * TILE_SIZE>(shared_element_unary_output);

  // define barriers
  Barrier *input_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_INPUT_BARRIER_OFFSET);
  Barrier *norm_weight_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_NORM_WEIGHT_BARRIER_OFFSET);
  Barrier *linear_weight_barrier =
      reinterpret_cast<Barrier *>(smem + SHARED_LINEAR_WEIGHT_BARRIER_OFFSET);
  Barrier *compute_done =
      reinterpret_cast<Barrier *>(smem + SHARED_COMPUTE_DONE_OFFSET);

  // init the barriers and launch the first group of copy
  if (threadIdx.x == 0) {
    for (int i = 0; i < Kstages; i++) {
      initialize_barrier(input_barrier[i], 1);
      initialize_barrier(norm_weight_barrier[i], 1);
      initialize_barrier(linear_weight_barrier[i], 1);
      initialize_barrier(compute_done[i], 1);
    }
  }

  __syncthreads();

  // warp specialization data movement warpgroup
  if (warpgroup_id == NUM_WARPGROUPS - 1) {

    // wg_decrease_regs<32>();
    if (lane_id() == 0 && warp_idx == (NUM_WARPGROUPS * WARPGROUP_WARPS - 4)) {
      int prefetch = (Kstages < num_k) ? Kstages : num_k;
      for (int i = 0; i < prefetch; i++) {
        if (threadIdx.x == 128) {
          // printf("prefetch: %d\n", i);
        }
        int slot = i % Kstages;
        int tma_coords_in[2] = {i * TILE_SIZE, 0};
        int tma_coords_w[2] = {i * TILE_SIZE, 0};

        input_smem_buffer.set_ptr(shared_input +
                                  slot * TMA_TRANS_BYTES_INPUT / sizeof(T));
        set_barrier_transaction_bytes(input_barrier[slot],
                                      TMA_TRANS_BYTES_INPUT);
        tma_input.tma_cp_async(
            input_barrier[slot], input_smem_buffer(0, 0), tma_coords_in);

        norm_weight_smem_buffer.set_ptr(shared_norm_weight +
                                        slot * TMA_TRANS_BYTES_NORM_WEIGHT /
                                            sizeof(T));
        set_barrier_transaction_bytes(norm_weight_barrier[slot],
                                      TMA_TRANS_BYTES_NORM_WEIGHT);
        tma_norm_weight.tma_cp_async(norm_weight_barrier[slot],
                                     norm_weight_smem_buffer(0, 0),
                                     tma_coords_in);

        linear_weight_smem_buffer.set_ptr(shared_linear_weight +
                                          slot * TMA_TRANS_BYTES_LINEAR_WEIGHT /
                                              sizeof(T));
        set_barrier_transaction_bytes(linear_weight_barrier[slot],
                                      TMA_TRANS_BYTES_LINEAR_WEIGHT);
        tma_linear_weight.tma_cp_async(linear_weight_barrier[slot],
                                       linear_weight_smem_buffer(0, 0),
                                       tma_coords_w);
      }

      for (int i = prefetch; i < num_k; i++) {
        int slot = i % Kstages;
        int phase = (i / Kstages) % 2;
        wait(compute_done[slot], phase ^ 1);

        int tma_coords_in[2] = {i * TILE_SIZE, 0};
        int tma_coords_w[2] = {i * TILE_SIZE, 0};

        input_smem_buffer.set_ptr(shared_input +
                                  slot * TMA_TRANS_BYTES_INPUT / sizeof(T));
        set_barrier_transaction_bytes(input_barrier[slot],
                                      TMA_TRANS_BYTES_INPUT);
        tma_input.tma_cp_async(
            input_barrier[slot], input_smem_buffer(0, 0), tma_coords_in);

        norm_weight_smem_buffer.set_ptr(shared_norm_weight +
                                        slot * TMA_TRANS_BYTES_NORM_WEIGHT /
                                            sizeof(T));
        set_barrier_transaction_bytes(norm_weight_barrier[slot],
                                      TMA_TRANS_BYTES_NORM_WEIGHT);
        tma_norm_weight.tma_cp_async(norm_weight_barrier[slot],
                                     norm_weight_smem_buffer(0, 0),
                                     tma_coords_in);

        linear_weight_smem_buffer.set_ptr(shared_linear_weight +
                                          slot * TMA_TRANS_BYTES_LINEAR_WEIGHT /
                                              sizeof(T));
        set_barrier_transaction_bytes(linear_weight_barrier[slot],
                                      TMA_TRANS_BYTES_LINEAR_WEIGHT);
        tma_linear_weight.tma_cp_async(linear_weight_barrier[slot],
                                       linear_weight_smem_buffer(0, 0),
                                       tma_coords_w);
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
      wait(norm_weight_barrier[slot], phase);

      input_smem.set_ptr(shared_input +
                         (slot)*TMA_TRANS_BYTES_INPUT / sizeof(T));
      norm_weight_smem.set_ptr(shared_norm_weight +
                               (slot)*TMA_TRANS_BYTES_NORM_WEIGHT / sizeof(T));
      linear_weight_smem.set_ptr(shared_linear_weight +
                                 (slot)*TMA_TRANS_BYTES_LINEAR_WEIGHT /
                                     sizeof(T));

      // elementwise mul
      mul(mul_output_smem, input_smem, norm_weight_smem);

      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);

      // mma
      wait(linear_weight_barrier[slot], phase);

      A_DESC a_desc(mul_output_smem(0, 0));
      B_DESC b_desc(linear_weight_smem(0, 0));

      //   wgmma::warpgroup_fence_fragment(s_frag);
      wgmma::warpgroup_arrive();
      // wgmma
      wgmma::mma<bfloat16,
                 64,
                 OUTPUT_SIZE,
                 16,
                 InputSmem,
                 LinearWeightSmem,
                 A_DESC,
                 B_DESC,
                 false,
                 false>(s_frag, a_desc, b_desc);
      wgmma::mma_commit_group();

      float const scalars[] = {0.0f, 1.0f / float(REDUCTION_SIZE)};
      perform_element_unary_chain_kernel<true,
                                         decltype(element_unary_smem),
                                         decltype(input_smem),
                                         ElementUnaryOpType::SQUARE,
                                         ElementUnaryOpType::MULSCALAR>(
          element_unary_smem, input_smem, scalars);

      wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);

      wgmma::mma_async_wait();
      //   wgmma::warpgroup_fence_fragment(s_frag);

      // flip compute done
      if (idx_in_warp == 0 && warp_idx % 4 == 0) {
        arrive(compute_done[slot], 1);
      }
    }
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16:~:text=The%20layout%20of%20the%20fragments%20held%20by%20different%20threads%20is%20shown%20in%20Figure%20149.
    // write back to shared memory

    float const scalars[] = {eps, 0.0f};
    reduction_sum_col<T,
                      decltype(reduction_output_smem),
                      decltype(element_unary_smem),
                      ElementUnaryOpType::ADDSCALAR,
                      ElementUnaryOpType::SQRT>(
        reduction_output_smem, element_unary_smem, scalars);
    wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);

#pragma unroll 1
    for (uint32_t i = 0; i < (OUTPUT_SIZE / 4); i++) {
      int row = (warp_idx % 4) * 16 + (i % 2) * 8 + idx_in_warp / 4;
      int col = (i / 2) * 8 + (idx_in_warp % 4) * 2;
      mm_output_smem.at(row, col) = bfloat16(s_frag[i * 2]);
      mm_output_smem.at(row, col + 1) = bfloat16(s_frag[i * 2 + 1]);
    }

    wg_sync<THREADS_PER_WARPGROUP * CONSUMER_WARPGROUPS>(8);

    // divide
    div_col(mm_output_smem, mm_output_smem, reduction_output_smem);

    // make sure generic proxy's modification to smem is visible to tma store
    // async proxy this is intra-thread sync
    async_proxy_fence();

    // copy back to dmem
    if (warp_idx % 4 == 0 && lane_id() == 0) {
      tma_out.tma_store_async(mm_output_smem(0, 0), {0, 0});
      store_commit_group();
    }
    store_async_wait<0>();
  }
}

} // namespace kernel