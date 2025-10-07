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
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
class LinearTask {
public:
  static constexpr int NUM_INPUT = 2;
  static constexpr int NUM_WEIGHT = 1;
  static constexpr int NUM_OUTPUT = 1;

  struct Params {
    void const *input_ptr;
    void const *weight_ptr;
    void const *residual_ptr;
    void *output_ptr;
    int num_active_tokens;
    bool residual;
  };

  using Config = Task<LinearTask>::Config;

  __device__ static constexpr Config config() {
    constexpr int TILE_SIZE = 128;
    constexpr int INPUT_SIZE =
        BATCH_SIZE * INPUT_PIPE_MAX * TILE_SIZE * sizeof(T);
    constexpr int WEIGHT_SIZE =
        TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_SIZE sizeof(T);
    constexpr int OUTPUT_SIZE_BYTES = BATCH_SIZE * OUTPUT_SIZE * sizeof(T);

    constexpr int in_pages = (INPUT_SIZE_BYTES + PAGE_SIZE - 1) / PAGE_SIZE;
    constexpr int weight_pages =
        (WEIGHT_SIZE_BYTES + PAGE_SIZE - 1) / PAGE_SIZE;
    constexpr int output_pages =
        (OUTPUT_SIZE_BYTES + PAGE_SIZE - 1) / PAGE_SIZE;
    return Config{
        Config::InputPages{in_pages, in_pages}, // 2 inputs: same footprint
        Config::WeightPages{weight_pages},
        output_pages,
        TensorLifetime::IN_OUTPUT,
        TensorLifetime::COMPUTE,
        TensorLifetime::IN_OUTPUT,
        false,
    };
  }

  __device__ static int prefetch(Params &params,
                                 TensorDesc &input_desc,
                                 TensorDesc &weight_desc,
                                 TensorDesc &output_desc,
                                 TensorDesc &temp_desc) {

    T *shared_input = (T *)input_desc.ptr;
    T *shared_weight = (T *)weight_desc.ptr;
    T *shared_output = (T *)output_desc.ptr;
    T *zero_buf = (T *)temp_desc.ptr;

    // set pointers for smem and dmem

    // Initialize output_smem: if residual is provided, preload it; otherwise
    // zero
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
  }

  __device__ static void mainloop(Params &params,
                                  TensorDesc &input_desc,
                                  TensorDesc &weight_desc,
                                  TensorDesc &output_desc,
                                  TensorDesc &temp_desc,
                                  int &previous_copy_done_flag) {
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
          // can start to prefetch next task, update a shared memory variable
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
              T *input_ptr =
                  is_input_valid ? valid_input_ptr : invalid_input_ptr;

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

// Final writeback: store accumulated output (residual already included if any)
#pragma unroll
    for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
      int row = i / CHUNKS_PER_ROW_C;
      int col = (i % CHUNKS_PER_ROW_C) << log2_CHUNK_SIZE;
      *((__uint128_t *)((void *)&output_dmem.at(row, col))) =
          *((__uint128_t *)((void *)&output_smem.at(row, col)));
    }
  }

  __device__ static void epilogue(Params &params, TensorDesc &output_desc) {
    // No-op
  }
};