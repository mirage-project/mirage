
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
#include "common.h"
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"
#include <cstdint>
namespace kernel {

using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int K_PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              bool residual = true) {
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE;
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_C = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      OUTPUT_ATOM_SIZE / 16 <= 4 ? OUTPUT_ATOM_SIZE / 16 : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;

  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = OUTPUT_ATOM_SIZE / NUM_WARPS_N / 16;
  constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual =
      residual ? static_cast<T const *>(residual_ptr) : nullptr;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem =
      dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem =
      dmem_row_const<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  // sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE;
  // sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_RESIDUAL_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t MM_INTERMEDIATE_OFFSET =
      SHARED_RESIDUAL_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      MM_INTERMEDIATE_OFFSET +
      sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // residual
  T *shared_residual =
      residual ? (T *)(smem + SHARED_RESIDUAL_OFFSET) : nullptr;

  // intermediate
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem =
      smem_row<T, 0, 0, 0, K_PIPE_MAX * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using WeightBufferSmem =
      smem_col<T, 3, 3, 3, TILE_SIZE, K_PIPE_MAX * OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using OutputSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  using MatMulIntermediateSmem = smem_row<T,
                                          0,
                                          0,
                                          0,
                                          NUM_WARPS_K * BATCH_SIZE,
                                          OUTPUT_ATOM_SIZE,
                                          OUTPUT_ATOM_SIZE>;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputSmem residual_smem(shared_residual);

  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  OutputSmem output_smem(shared_output);

  for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++,
           d_weight += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
           d_residual = residual ? d_residual + OUTPUT_ATOM_SIZE : nullptr,
           d_output += OUTPUT_ATOM_SIZE) {
    weight_dmem.set_ptr(d_weight);
    residual_dmem.set_ptr(d_residual);
    output_dmem.set_ptr(d_output);

    InputBufferSmem input_buffer_smem(shared_input_buffer);
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);

    if (residual) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_C; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_C;
        int col = (i & (CHUNKS_PER_ROW_C - 1)) << log2_CHUNK_SIZE;
        load_smem(residual_smem(row, col), residual_dmem(row, col));
      }
    }

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        int dst_row = src_row + ((k_pipe + 1) * BATCH_SIZE);
        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
        load_smem(input_buffer_smem(dst_row, dst_col),
                  input_dmem(src_row, src_col));
      }
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        int src_col = i >> log2_CHUNKS_PER_COL_B;
        int dst_col =
            src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));
        load_smem(weight_buffer_smem(dst_row, dst_col),
                  weight_dmem(src_row, src_col));
      }
      cp_async_fence();
    }

    // accumulator
    float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // copy
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int row = i >> log2_CHUNKS_PER_ROW_A;
          int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int src_col = dst_col + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(row, dst_col), input_dmem(row, src_col));
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          int col = i >> log2_CHUNKS_PER_COL_B;
          load_smem(weight_buffer_smem(dst_row, col),
                    weight_dmem(src_row, col));
        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }

      // rotate the buffers
      input_buffer_smem.set_ptr(shared_input_buffer +
                                BATCH_SIZE * TILE_SIZE *
                                    ((for_idx + 1) % K_PIPE_MAX));
      input_smem.set_ptr(shared_input_buffer +
                         BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_buffer_smem.set_ptr(shared_weight_buffer +
                                 TILE_SIZE * OUTPUT_ATOM_SIZE *
                                     ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer +
                          TILE_SIZE * OUTPUT_ATOM_SIZE *
                              ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();

      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            T *src_ptr =
                is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm(weight_smem(n_row, n_col), b_frag);
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }
      __syncthreads();
    }

    // write back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                bfloat16(s_frag[m][n][(i << 1)]);
            mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
    __syncthreads();

    if (NUM_WARPS_K > 1) {
      reduction_sum_row<decltype(output_smem), decltype(mm_intermediate_smem)>(
          output_smem, mm_intermediate_smem);
      __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < BATCH_SIZE; row++) {
#pragma unroll
      for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
        T val = NUM_WARPS_K > 1 ? output_smem.at(row, i)
                                : mm_intermediate_smem.at(row, i);
        output_dmem.at(row, i) =
            residual ? val + residual_smem.at(row, i) : val;
      }
    }
    if (output_atom_idx + 1 < NUM_OUTPUT_ATOMS) {
      __syncthreads();
    }
  }
}



template <typename T,   // e.g., fp16
        //   typename FP8, // e.g., uint8_t
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          typename FP8 = __uint8_t, // e.g., uint8_t
          int K_PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel_fp8_weight(  
    void const *input_ptr,             // fp16 input
    void const *weight_fp8_ptr,        // fp8 weight
    void const *weight_scale_ptr,      // fp16 scale (per group)
    void const *residual_ptr,          // fp16 residual
    void *output_ptr,                  // fp16 output
    bool residual = true) {
      
      // Add near the top of the function (constants + helpers)
      constexpr int WB_K = 128;  // weight_block_size along K
      constexpr int WB_N = 128;  // weight_block_size along N

      // Number of scale blocks
      constexpr int K_BLOCKS = ceil_div_constexpr(REDUCTION_SIZE, WB_K);
      constexpr int N_BLOCKS = ceil_div_constexpr(OUTPUT_SIZE,    WB_N);

  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE;
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_C = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // FP8 specific
  constexpr int CHUNK_SIZE_W = 16 / sizeof(FP8); // for FP8 load chunks
  constexpr int NUM_CHUNKS_B_FP8 = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE_W;
  constexpr int CHUNKS_PER_COL_B_FP8 = TILE_SIZE / CHUNK_SIZE_W;
  constexpr int log2_CHUNK_SIZE_W = log2_constexpr(CHUNK_SIZE_W);
  constexpr int log2_CHUNKS_PER_COL_B_FP8 = log2_constexpr(CHUNKS_PER_COL_B_FP8);


  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      OUTPUT_ATOM_SIZE / 16 <= 4 ? OUTPUT_ATOM_SIZE / 16 : 4;
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;

  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = OUTPUT_ATOM_SIZE / NUM_WARPS_N / 16;
  constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  // Weight pointer is FP8
  FP8 const *__restrict__ d_weight_fp8 = static_cast<FP8 const *>(weight_fp8_ptr);
  T const *__restrict__ d_weight_scale = static_cast<T const *>(weight_scale_ptr);
  T const *__restrict__ d_residual =
      residual ? static_cast<T const *>(residual_ptr) : nullptr;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmemFP8 =
      dmem_col_const<FP8, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;

  using ResidualDmem =
      dmem_row_const<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>;

    using ScaleDmem =
      dmem_col_const<T, (REDUCTION_SIZE + WB_K - 1) / WB_K, (OUTPUT_SIZE + WB_N - 1) / WB_N, (REDUCTION_SIZE + WB_K - 1) / WB_K>;
  ScaleDmem weight_scale_dmem(d_weight_scale);


  InputDmem input_dmem(d_input);
  WeightDmemFP8 weight_dmem_fp8(d_weight_fp8);  // <-- FP8 weight DMEM
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  // sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * K_PIPE_MAX * BATCH_SIZE * TILE_SIZE;
  // sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_RESIDUAL_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(FP8) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t MM_INTERMEDIATE_OFFSET =
      SHARED_RESIDUAL_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      MM_INTERMEDIATE_OFFSET +
      sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SCALE_OFFSET =
      SHARED_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  // alignas(16) // Ensure alignment for FP8
  FP8 *shared_weight_buffer = (FP8 *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);
  // 0819 n check
  // residual
  T *shared_residual =
      residual ? (T *)(smem + SHARED_RESIDUAL_OFFSET) : nullptr;

  // intermediate
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  T *scale = (T *)(smem + SCALE_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem =
      smem_row<T, 0, 0, 0, K_PIPE_MAX * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem =
      smem_col<FP8, 3, 4, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using WeightBufferSmem =
      smem_col<FP8, 3, 4, 3, TILE_SIZE, K_PIPE_MAX * OUTPUT_ATOM_SIZE, TILE_SIZE>;
  using OutputSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
  using MatMulIntermediateSmem = smem_row<T,
                                          0,
                                          0,
                                          0,
                                          NUM_WARPS_K * BATCH_SIZE,
                                          OUTPUT_ATOM_SIZE,
                                          OUTPUT_ATOM_SIZE>;

  // using ScaleSmem =
  //     smem_col<T, 0, 0, 0, (REDUCTION_SIZE + WB_K - 1) / WB_K,
  //              (OUTPUT_ATOM_SIZE + WB_N - 1) / WB_N, (REDUCTION_SIZE + WB_K - 1) / WB_K>;
  using ScaleSmem =
      smem_col<T, 0, 0, 0, (REDUCTION_SIZE + WB_K - 1) / WB_K,
               (OUTPUT_SIZE + WB_N - 1) / WB_N, (REDUCTION_SIZE + WB_K - 1) / WB_K>;
  
  __shared__ T current_scale;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputSmem residual_smem(shared_residual);

  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  OutputSmem output_smem(shared_output);

  ScaleSmem weight_scale_smem(scale);

  // load the weight scale into shared memory
  constexpr int SCALE_ROWS = (REDUCTION_SIZE + WB_K - 1) / WB_K;
  // constexpr int SCALE_COLS = (OUTPUT_ATOM_SIZE + WB_N - 1) / WB_N;
  constexpr int SCALE_COLS = (OUTPUT_SIZE + WB_N - 1) / WB_N;

  // constexpr int thread_count = SCALE_ROWS * SCALE_COLS * sizeof(T) / 128;
  constexpr int thread_count = SCALE_ROWS * SCALE_COLS * sizeof(T) / 16; // 16 bytes per thread

  // this is where the real problem is! if you comment this out, it will show all 0 output! looks like
  // the scale is not loaded correctly into shared memory
  if (threadIdx.x < thread_count) {
    constexpr int col_num = 16 / sizeof(T); // 16 bytes per T
    int idx = threadIdx.x * col_num;
    int row = idx % SCALE_ROWS;
    int col = idx / SCALE_ROWS;
    load_smem(weight_scale_smem(row, col),
              weight_scale_dmem(row, col));
  }

  for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++,
           d_weight_fp8 += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
           d_residual = residual ? d_residual + OUTPUT_ATOM_SIZE : nullptr,
           d_output += OUTPUT_ATOM_SIZE) {
    weight_dmem_fp8.set_ptr(d_weight_fp8);
    residual_dmem.set_ptr(d_residual);
    output_dmem.set_ptr(d_output);

    InputBufferSmem input_buffer_smem(shared_input_buffer);
    // alignas(16) // Ensure alignment for FP8
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);

    if (residual) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_C; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_C;
        int col = (i & (CHUNKS_PER_ROW_C - 1)) << log2_CHUNK_SIZE;
        load_smem(residual_smem(row, col), residual_dmem(row, col));
      }
    }

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
      // note that it's < K_PIPE_MAX - 1, because the last k_pipe is handled separately!!!
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        // i / 2^x where x s.t. 2^x is the closest to CHUNKS_PER_ROW_A = (TILE_SIZE / CHUNK_SIZE) 
        int dst_row = src_row + ((k_pipe + 1) * BATCH_SIZE);
        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
        load_smem(input_buffer_smem(dst_row, dst_col),
                  input_dmem(src_row, src_col));
      }
      
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B_FP8; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B_FP8 - 1)) << log2_CHUNK_SIZE_W;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        int src_col = i >> log2_CHUNKS_PER_COL_B_FP8;
        int dst_col =
            src_col + ((k_pipe + 1) << log2_constexpr(OUTPUT_ATOM_SIZE));

        // need to debug it!
        load_smem<FP8>(weight_buffer_smem(dst_row, dst_col),
                static_cast<const FP8 *>(weight_dmem_fp8(src_row, src_col)));
      }
      cp_async_fence();
    }

    // accumulator
    alignas(16) float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      if (threadIdx.x == 0) {
        // calculate the current scale
        int row_scale = ((for_idx) * TILE_SIZE) / WB_K;
        int col_scale = (output_atom_idx * OUTPUT_ATOM_SIZE) / WB_N;
        current_scale = weight_scale_smem.at(row_scale, col_scale);
      }

      // copy
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int row = i >> log2_CHUNKS_PER_ROW_A;
          int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int src_col = dst_col + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(row, dst_col), input_dmem(row, src_col));
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B_FP8; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B_FP8 - 1)) << log2_CHUNK_SIZE_W;
          // we assume that CHUNKS_PER_COL_B_FP8 - 1 is the power of 2, so we can use bitwise operations, which is equivalent to taking the lower bits
          // of the index
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          int col = i >> log2_CHUNKS_PER_COL_B_FP8;
          // Load FP8 weights from GMEM into registers
          load_smem<FP8>(weight_buffer_smem(dst_row, col),
                    weight_dmem_fp8(src_row, col));

        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }

      // rotate the buffers
      input_buffer_smem.set_ptr(shared_input_buffer +
                                BATCH_SIZE * TILE_SIZE *
                                    ((for_idx + 1) % K_PIPE_MAX));
      input_smem.set_ptr(shared_input_buffer +
                         BATCH_SIZE * TILE_SIZE * ((for_idx + 1) % K_PIPE_MAX));
      weight_buffer_smem.set_ptr(shared_weight_buffer +
                                 TILE_SIZE * OUTPUT_ATOM_SIZE *
                                     ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer +
                          TILE_SIZE * OUTPUT_ATOM_SIZE *
                              ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();
      uint32_t a_frag[4], b_frag[4];

      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int base_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4);
          int j = lane_idx & 0x3;
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int base_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4);
            T *src_ptr =
                is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            int dr = (j & 0x1) * 8;
            uint64_t temp =0;
            int dcp = lane_idx >> 1;
            int dc2 = lane_idx & 0xF;

            FP8 q_weight[8];
            load_row_8x8b( weight_smem(base_row + dr, base_col + dcp) , *reinterpret_cast<uint64_t*>(q_weight)); 
            b_frag[0] = fp8x2_to_bf16_v2(reinterpret_cast<uint16_t*>(q_weight)[0]);
            (reinterpret_cast<T *>(b_frag))[0] *= current_scale;
            (reinterpret_cast<T *>(b_frag))[1] *= current_scale;
            b_frag[1] = fp8x2_to_bf16_v2(reinterpret_cast<uint16_t*>(q_weight)[1]);
            (reinterpret_cast<T *>(b_frag))[2] *= current_scale;
            (reinterpret_cast<T *>(b_frag))[3] *= current_scale;
            b_frag[2] = fp8x2_to_bf16_v2(reinterpret_cast<uint16_t*>(q_weight)[2]);
            (reinterpret_cast<T *>(b_frag))[4] *= current_scale;
            (reinterpret_cast<T *>(b_frag))[5] *= current_scale;
            b_frag[3] = fp8x2_to_bf16_v2(reinterpret_cast<uint16_t*>(q_weight)[3]);
            (reinterpret_cast<T *>(b_frag))[6] *= current_scale;
            (reinterpret_cast<T *>(b_frag))[7] *= current_scale;
            
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
            // mma_m16n16k16_fp8fp8bf32(
            //     s_frag[m][n], a_frag, b_frag, s_frag[m][n]); // this one is slightly slower than the above bf16 version
          }
        }
      }
      __syncthreads();
    }

    // write back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                bfloat16(s_frag[m][n][(i << 1)]);
            mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
    __syncthreads();

    if (NUM_WARPS_K > 1) {
      reduction_sum_row<decltype(output_smem), decltype(mm_intermediate_smem)>(
          output_smem, mm_intermediate_smem);
      __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < BATCH_SIZE; row++) {
#pragma unroll
      for (int i = threadIdx.x; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
        T val = NUM_WARPS_K > 1 ? output_smem.at(row, i)
                                : mm_intermediate_smem.at(row, i);
        output_dmem.at(row, i) =
            residual ? val + residual_smem.at(row, i) : val;
      }
    }
    if (output_atom_idx + 1 < NUM_OUTPUT_ATOMS) {
      __syncthreads();
    }
  }
}

} // namespace kernel
