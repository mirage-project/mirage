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
namespace kernel {

using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 2>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int log2_OUTPUT_ATOM_SIZE = log2_constexpr(OUTPUT_ATOM_SIZE);
  constexpr int NUM_OUTPUT_ATOMS =
      (OUTPUT_SIZE + OUTPUT_ATOM_SIZE - 1) / OUTPUT_ATOM_SIZE;

  constexpr int TILE_SIZE = 128;
  constexpr int log2_TILE_SIZE = log2_constexpr(TILE_SIZE);
  constexpr int SMEM_MAX_BANDWIDTH = 128 / sizeof(T);
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int TOTAL_WEIGHT_BLOCKS_TO_LOAD =
      FORLOOP_RANGE * NUM_OUTPUT_ATOMS; // For global pipe loading
  constexpr int WEIGHT_PIPE_MAX = PIPE_MAX < TOTAL_WEIGHT_BLOCKS_TO_LOAD
                                      ? PIPE_MAX
                                      : TOTAL_WEIGHT_BLOCKS_TO_LOAD;
  constexpr int INPUT_PIPE_MAX = WEIGHT_PIPE_MAX;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);
  constexpr int log2_CHUNKS_PER_ROW_C = log2_constexpr(CHUNKS_PER_ROW_C);

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARPS_N =
      4; // We always use NUM_WARPS_K = 1 and NUM_WARPS_N = 4
  constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;
  // Do not support split K for now
  static_assert(NUM_WARPS_K == 1);

  // TODO: support NUM_ITERS_M > 1, i.e., BATCH_SIZE > 16
  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N =
      (OUTPUT_ATOM_SIZE + NUM_WARPS_N * 16 - 1) / (NUM_WARPS_N * 16);
  constexpr int NUM_ITERS_K =
      (TILE_SIZE + NUM_WARPS_K * 16 - 1) / (NUM_WARPS_K * 16);

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = lane_id();

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  // CANNOT perform residual when redisual_ptr is nullptr
  if (residual_ptr == nullptr) {
    assert(!residual);
  }

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem =
      dmem_col_const<T, TILE_SIZE, OUTPUT_ATOM_SIZE, REDUCTION_SIZE>;
  using ResidualDmem = dmem_row_const<T, BATCH_SIZE, OUTPUT_SIZE, O_STRIDE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_SIZE, O_STRIDE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  ResidualDmem residual_dmem(d_residual);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // STensors' offsets
  constexpr size_t ZERO_BUFFER_OFFSET = 0;
  // sizeof(T) * 8

  constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
      ZERO_BUFFER_OFFSET + sizeof(T) * 64;
  // sizeof(T) * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET +
      sizeof(T) * BATCH_SIZE * INPUT_PIPE_MAX * TILE_SIZE;
  // sizeof(T) * TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      // MM_INTERMEDIATE_OFFSET +
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_SIZE

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row_2dcol<T,
                                   3,
                                   3,
                                   3,
                                   BATCH_SIZE * INPUT_PIPE_MAX,
                                   SMEM_MAX_BANDWIDTH,
                                   TILE_SIZE / SMEM_MAX_BANDWIDTH>;
  using WeightSmem = smem_col_2drow<T,
                                    3,
                                    3,
                                    3,
                                    SMEM_MAX_BANDWIDTH,
                                    TILE_SIZE / SMEM_MAX_BANDWIDTH,
                                    WEIGHT_PIPE_MAX * OUTPUT_ATOM_SIZE>;
  using OutputFullSmem =
      smem_row<T, 3, 3, 3, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputFullSmem output_smem(shared_output);

  // Initialize output_smem: if residual is provided, preload it; otherwise zero
#pragma unroll
  for (int i = threadIdx.x; i < BATCH_SIZE * OUTPUT_SIZE / CHUNK_SIZE;
       i += NUM_THREADS) {
    int row = i / (OUTPUT_SIZE / CHUNK_SIZE);
    int col = (i % (OUTPUT_SIZE / CHUNK_SIZE)) << log2_CHUNK_SIZE;
    // TODO: use ignore-src in load_smem to avoid if-else
    if (residual) {
      load_smem(output_smem(row, col), residual_dmem(row, col));
    } else {
      *((__uint128_t *)((void *)&output_smem.at(row, col))) = 0ul;
    }
  }

  // Warm up weight and input tiles for the first WEIGHT_PIPE_MAX - 1 atoms
  int global_pipe_idx = 0;
  // #pragma unroll 0
  for (; global_pipe_idx < WEIGHT_PIPE_MAX - 1; ++global_pipe_idx) {
    int src_stage_offset = (global_pipe_idx % NUM_OUTPUT_ATOMS)
                           << log2_OUTPUT_ATOM_SIZE;
    int buffer_stage_offset = (global_pipe_idx % WEIGHT_PIPE_MAX)
                              << log2_OUTPUT_ATOM_SIZE;
    int global_pipe_row = global_pipe_idx / NUM_OUTPUT_ATOMS;
    int global_pipe_offset = global_pipe_row << log2_TILE_SIZE;
    int input_pipe_offset = (global_pipe_row % INPUT_PIPE_MAX) * BATCH_SIZE;

    // int buffer_stage = global_pipe_idx % WEIGHT_PIPE_MAX;
    if (global_pipe_idx % NUM_OUTPUT_ATOMS == 0) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int src_row = i >> log2_CHUNKS_PER_ROW_A;
        int dst_row = src_row + input_pipe_offset;

        int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        int src_col = dst_col + global_pipe_offset;
        load_smem(input_smem(dst_row, dst_col), input_dmem(src_row, src_col));
      }
    }
#pragma unroll
    for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
      int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
      int src_row = dst_row + global_pipe_offset;

      int col_within = i >> log2_CHUNKS_PER_COL_B;
      int src_col = src_stage_offset + col_within;
      int dst_col = buffer_stage_offset + col_within;

      load_smem(weight_smem(dst_row, dst_col), weight_dmem(src_row, src_col));
    }
    cp_async_fence();
  }

  // Outer loop over K tiles; inner loop over output atoms
  // accumulator
  float s_frag[NUM_OUTPUT_ATOMS][NUM_ITERS_M][NUM_ITERS_N][8];
#pragma unroll
  for (uint32_t output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++) {
#pragma unroll
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[output_atom_idx][m][n]);
      }
    }
  }
#pragma unroll 2
  for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
    int cur_input_stage = for_idx % INPUT_PIPE_MAX;

    // Loop over output atoms for this K-slice
#pragma unroll
    for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
         ++output_atom_idx, ++global_pipe_idx) {
      int src_stage_offset = (global_pipe_idx % NUM_OUTPUT_ATOMS)
                             << log2_OUTPUT_ATOM_SIZE;
      int buffer_stage_offset = (global_pipe_idx % WEIGHT_PIPE_MAX)
                                << log2_OUTPUT_ATOM_SIZE;
      int global_pipe_row = global_pipe_idx / NUM_OUTPUT_ATOMS;
      int global_pipe_offset = global_pipe_row << log2_TILE_SIZE;
      int input_pipe_offset = (global_pipe_row % INPUT_PIPE_MAX) * BATCH_SIZE;

      // Prefetch next weight atom into ring buffer stage_write
      if (global_pipe_idx < TOTAL_WEIGHT_BLOCKS_TO_LOAD) {
        // Load input tile at the first output atom
        if (global_pipe_idx % NUM_OUTPUT_ATOMS == 0) {
#pragma unroll
          for (int i = threadIdx.x; i < NUM_CHUNKS_A;
               i += NUM_THREADS) { // 1 time
            int src_row = i >> log2_CHUNKS_PER_ROW_A;
            int dst_row = src_row + input_pipe_offset;

            int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
            int src_col = dst_col + global_pipe_offset;

            load_smem(input_smem(dst_row, dst_col),
                      input_dmem(src_row, src_col));
          }
        }
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int src_row = dst_row + global_pipe_offset;

          int col_within = i >> log2_CHUNKS_PER_COL_B;
          int src_col = src_stage_offset + col_within;
          int dst_col = buffer_stage_offset + col_within;

          load_smem(weight_smem(dst_row, dst_col),
                    weight_dmem(src_row, src_col));
        }
        cp_async_fence();
        cp_async_wait<WEIGHT_PIPE_MAX - 1>();
      } else if (global_pipe_idx == TOTAL_WEIGHT_BLOCKS_TO_LOAD) {
        cp_async_wait<0>();
      }
      __syncthreads();

      // MMA using the loaded input and weight tiles
      uint32_t a_frag[4], b_frag[4];
#pragma unroll
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF) + (m << 4);
        bool is_input_valid = (m_row < num_active_tokens);
        int smem_row = m_row + cur_input_stage * BATCH_SIZE;
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
          bool is_weight_valid = (n_col < OUTPUT_ATOM_SIZE);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            int weight_stage_read =
                (for_idx * NUM_OUTPUT_ATOMS + output_atom_idx) %
                WEIGHT_PIPE_MAX;

            // Do not use ternary operator here, it will cause the
            // compiler to generate branch among threads
            T *valid_input_ptr = input_smem(smem_row, m_col);
            T *invalid_input_ptr = zero_buffer(0, 0);
            T *input_ptr = is_input_valid ? valid_input_ptr : invalid_input_ptr;

            T *valid_weight_ptr = weight_smem(
                n_row, weight_stage_read * OUTPUT_ATOM_SIZE + n_col);
            T *invalid_weight_ptr = zero_buffer(0, 0);
            T *weight_ptr =
                is_weight_valid ? valid_weight_ptr : invalid_weight_ptr;

            ldsm(input_ptr, a_frag);
            ldsm(weight_ptr, b_frag);
            mma_m16n16k16_bf16bf16bf32(s_frag[output_atom_idx][m][n],
                                       a_frag,
                                       b_frag,
                                       s_frag[output_atom_idx][m][n]);
          }
        }
      }
      __syncthreads();
    }
  }
  // Accumulate this atom's contribution into the full output_smem at offset
#pragma unroll
  for (uint32_t output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++) {
#pragma unroll
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          int col_within = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                           ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
          int col = col_within + output_atom_idx * OUTPUT_ATOM_SIZE;
          if (row_in_warp < num_active_tokens &&
              col_within < OUTPUT_ATOM_SIZE) {
            output_smem.at(row_in_warp, col) +=
                bfloat16(s_frag[output_atom_idx][m][n][(i << 1)]);
            output_smem.at(row_in_warp, col + 1) +=
                bfloat16(s_frag[output_atom_idx][m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
  }
  __syncthreads();

  // Final writeback: store accumulated output (residual already included if
  // any)
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
    int row = i / CHUNKS_PER_ROW_C;
    int col = (i % CHUNKS_PER_ROW_C) << log2_CHUNK_SIZE;
    *((__uint128_t *)((void *)&output_dmem.at(row, col))) =
        *((__uint128_t *)((void *)&output_smem.at(row, col)));
  }
}

} // namespace kernel