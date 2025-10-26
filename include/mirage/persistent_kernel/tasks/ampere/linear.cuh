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
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "tasks/common/common_header.cuh"
namespace kernel {

using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 3>
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

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE; // 128
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE; // 1024
  constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE; // 16
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
  // constexpr int NUM_ITERS_K =
  //     (TILE_SIZE + NUM_WARPS_K * 16 - 1) / (NUM_WARPS_K * 16);
  constexpr int NUM_ITERS_K = 8;

  constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
  constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int warp_row = warp_idx >> log2_NUM_WARPS_N;
  int warp_col = warp_idx & (NUM_WARPS_N - 1);
  int lane_idx = threadIdx.x & 0x1f;

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  // CANNOT perform residual when redisual_ptr is nullptr
  // if (residual_ptr == nullptr) {
  //   assert(!residual);
  // }

  // int bid = blockIdx.x;
  // d_weight += OUTPUT_SIZE * REDUCTION_SIZE * bid;
  // d_residual += OUTPUT_SIZE * bid;
  // d_output += OUTPUT_SIZE * bid;


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
      sizeof(T) * BATCH_SIZE * WEIGHT_PIPE_MAX * TILE_SIZE;
  // sizeof(T) * TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      // MM_INTERMEDIATE_OFFSET +
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_ATOM_SIZE;
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
                                   BATCH_SIZE,
                                   TILE_SIZE,
                                   INPUT_PIPE_MAX>;
  using WeightSmem = smem_col_2drow<T,
                                    3,
                                    3,
                                    3,
                                    TILE_SIZE,
                                    OUTPUT_ATOM_SIZE,
                                    WEIGHT_PIPE_MAX>;
  using OutputFullSmem =
      smem_row<T, 3, 3, 3, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;

  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputFullSmem output_smem(shared_output);


#pragma unroll
  for (uint32_t output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
       output_atom_idx++) {
    // loop among OUTPUT_ATOM_SIZE
    float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];

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

    // initialize registers
#pragma unroll
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    int itile_to_read = 0;
    int ismem_read_stage = 0;
    int ismem_write_stage = 0;

    // Warm up weight and input tiles for the first WEIGHT_PIPE_MAX - 1 atoms
#pragma unroll
    for (int istage = 0; istage < WEIGHT_PIPE_MAX - 1; ++istage) {
      int src_stage_offset = istage << log2_TILE_SIZE;
      // we don't need module for WEIGHT_PIPE_MAX here, because we just load WEIGHT_PIPE_MAX - 1 pipe.

      if (output_atom_idx == 0) {
        // we only load input when we just enter a new loop for new iteration for atoms.
#pragma unroll
        for (int chu = 0; chu < NUM_CHUNKS_A / NUM_THREADS; chu ++) {
          int tid = threadIdx.x;
          int threadCol = (tid & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
          constexpr int ROWS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_ROW_A; // 8

          // int dst_row = threadRow;
          // int src_row = dst_row + src_stage_offset;
          int dst_col = threadCol;
          int src_col = dst_col + src_stage_offset;

          int row_within = threadRow + chu * ROWS_PER_ITERATION;
          int src_row = row_within;
          int dst_row = row_within;

          load_smem(input_smem(dst_row, dst_col, istage),
                    input_dmem(src_row, src_col));
        }
// #pragma unroll
//           for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
//             int src_row = i >> log2_CHUNKS_PER_ROW_A;
//             int dst_row = src_row;

//             int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
//             int src_col = dst_col + src_stage_offset;
//             load_smem(input_smem(dst_row, dst_col, istage), input_dmem(src_row, src_col));
//           }
        }
#pragma unroll
      for (int chu = 0; chu < NUM_CHUNKS_B / NUM_THREADS; chu ++) {
        int tid = threadIdx.x;
        int threadRow = (tid & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int threadCol = tid >> log2_CHUNKS_PER_COL_B;
        constexpr int COLS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_COL_B; // 8

        int dst_row = threadRow;
        int src_row = dst_row + src_stage_offset;

        int col_within = threadCol + chu * COLS_PER_ITERATION;
        int src_col = (output_atom_idx << log2_OUTPUT_ATOM_SIZE) + col_within;
        int dst_col = col_within;

        load_smem(weight_smem(dst_row, dst_col, istage),
                  weight_dmem(src_row, src_col));
      }
      cp_async_fence();

      ++itile_to_read;
      ++ismem_write_stage;
    } // warm up for PIPE_MAX - 1

    // wait for first warm up pipeline cp.async finished
    constexpr int PIPE_INSIDE_TILE = 2;
    uint32_t a_frag[PIPE_INSIDE_TILE][4], b_frag[PIPE_INSIDE_TILE][4];
    // uint32_t a_frag0[4], a_frag1[4], b_frag0[4], b_frag1[4];
    cp_async_wait<WEIGHT_PIPE_MAX - 2>();
    __syncthreads();

    int first_m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + ((lane_idx >> 4) << 3);
    int first_n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (((lane_idx & 0xF) >> 3) << 3);
    // int first_weight_stage_read =
    //       (for_idx * NUM_OUTPUT_ATOMS + output_atom_idx) %
    //       WEIGHT_PIPE_MAX;
    int first_smem_row = (lane_idx & 0xF);
    bool first_is_input_valid = (first_smem_row < num_active_tokens);
    T* first_valid_input_ptr = input_smem(first_smem_row, first_m_col, 0);
    T* first_invalid_input_ptr = zero_buffer(0, 0);
    int first_n_col = (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
    bool first_is_weight_valid = (first_n_col < OUTPUT_ATOM_SIZE);
    T* first_input_ptr = first_is_input_valid ? first_valid_input_ptr : first_invalid_input_ptr;
    T* first_valid_weight_ptr = weight_smem(first_n_row, first_n_col, 0);
    T* first_invalid_weight_ptr = zero_buffer(0, 0);
    T* first_weight_ptr =
          first_is_weight_valid ? first_valid_weight_ptr : first_invalid_weight_ptr;

    // NOTE: if we just comment out the two below ldsm, we will see 20% perf gain!!! why?

    ldsm(first_input_ptr, a_frag[0]);
    ldsm(first_weight_ptr, b_frag[0]);
    // ldsm(first_input_ptr, a_frag0);
    // ldsm(first_weight_ptr, b_frag0);

    T* input_ptr = first_input_ptr;
    T* weight_ptr = first_weight_ptr;
#pragma unroll 1
    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // Loop over output atoms for this K-slice
#pragma unroll
      for (int k = 0; k < NUM_ITERS_K; k++) {
        int k_next = (k + 1) % NUM_ITERS_K;

        if(k == 0) {
          // loading next tile (itile_to_read) when k is 0 and itile_to_read small than FORLOOP_RANGE.
          if(itile_to_read < FORLOOP_RANGE) {
            int src_stage_offset = itile_to_read << log2_TILE_SIZE;
            // Prefetch next weight atom into ring buffer stage_write
            // Load input tile at the first output atom
            if (output_atom_idx  == 0) {
#pragma unroll
              for (int chu = 0; chu < NUM_CHUNKS_A / NUM_THREADS; chu ++) {
                int tid = threadIdx.x;
                int threadCol = (tid & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
                int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
                constexpr int ROWS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_ROW_A; // 8

                // int dst_row = threadRow;
                // int src_row = dst_row + src_stage_offset;
                int dst_col = threadCol;
                int src_col = dst_col + src_stage_offset;

                int row_within = threadRow + chu * ROWS_PER_ITERATION;
                int src_row = row_within;
                int dst_row = row_within;

                load_smem(input_smem(dst_row, dst_col, ismem_write_stage),
                          input_dmem(src_row, src_col));
              }
// #pragma unroll
//               for (int i = threadIdx.x; i < NUM_CHUNKS_A;
//                   i += NUM_THREADS) {
//                 int src_row = i >> log2_CHUNKS_PER_ROW_A;
//                 int dst_row = src_row;

//                 int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
//                 int src_col = dst_col + src_stage_offset;

//                 load_smem(input_smem(dst_row, dst_col, ismem_write_stage),
//                           input_dmem(src_row, src_col));
//               }
            }
#pragma unroll
            for (int chu = 0; chu < NUM_CHUNKS_B / NUM_THREADS; chu ++) {
              int tid = threadIdx.x;
              int threadRow = (tid & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
              int threadCol = tid >> log2_CHUNKS_PER_COL_B;
              constexpr int COLS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_COL_B; // 8

              int dst_row = threadRow;
              int src_row = dst_row + src_stage_offset;

              int col_within = threadCol + chu * COLS_PER_ITERATION;
              int src_col = (output_atom_idx << log2_OUTPUT_ATOM_SIZE) + col_within;
              int dst_col = col_within;

              load_smem(weight_smem(dst_row, dst_col, ismem_write_stage),
                        weight_dmem(src_row, src_col));
            }
            itile_to_read ++;
            ismem_write_stage = (ismem_write_stage + 1) % WEIGHT_PIPE_MAX;
          } // itile_to_read < FORLOOP_RANGE, which means we should load next tile
          cp_async_fence();
        } // k == 0 for load next tile


        // size_t input_inside_tile_offset = ((k_next << 4) / SMEM_MAX_BANDWIDTH) * (8*64) + ((k_next << 4) % SMEM_MAX_BANDWIDTH);
        // size_t weight_inside_tile_offset = ((k_next << 4) / SMEM_MAX_BANDWIDTH) * (64*64) + ((k_next << 4) % SMEM_MAX_BANDWIDTH);
        if (k == NUM_ITERS_K - 1) {
          // wait cp.async because we will load next tile data in to regs when k == NUM_ITERS_K - 1.
          if(FORLOOP_RANGE - for_idx > 2) {
              cp_async_wait<WEIGHT_PIPE_MAX - 2>();
            } else {
              cp_async_wait<0>();
            }
            __syncthreads();

            // int tmp_ismem_read_stage = ismem_read_stage;
            ismem_read_stage = (ismem_read_stage + 1) % WEIGHT_PIPE_MAX;
            // input_ptr += (ismem_read_stage - tmp_ismem_read_stage) * (8 * 128);
            // weight_ptr += (ismem_read_stage - tmp_ismem_read_stage) * (64 * 128);
        } // k == NUM_ITERS_K - 1

        static_assert(NUM_ITERS_M == 1);
        // If we use NUM_ITERS_M and NUM_ITERS_N inside NUM_ITERS_K, the loop for NUM_ITERS_K couldn't be unrolled in nvcc which hurts performance.
// #pragma unroll
//         for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
          int m = 0;
          int m_row = (lane_idx & 0xF) + (m << 4);
          bool is_input_valid = (m_row < num_active_tokens);
          // move below code to later for advance stage
          static_assert(NUM_ITERS_N == 1);
// #pragma unroll
//           for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
            int n = 0;
            int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                        ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
            bool is_weight_valid = (n_col < OUTPUT_ATOM_SIZE);

            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);

            int smem_row = m_row;
            // Do not use ternary operator here, it will cause the
            // compiler to generate branch among threads
            T *valid_input_ptr = input_smem(smem_row, m_col, ismem_read_stage);
            T *invalid_input_ptr = zero_buffer(0, 0);
            // T *input_ptr = is_input_valid ? valid_input_ptr : invalid_input_ptr;
            T *input_ptr = valid_input_ptr;

            T *valid_weight_ptr = weight_smem(
                n_row, n_col, ismem_read_stage);
            T *invalid_weight_ptr = zero_buffer(0, 0);
            // T *weight_ptr =
            //     is_weight_valid ? valid_weight_ptr : invalid_weight_ptr;
            T *weight_ptr = valid_weight_ptr;
            

            // ldsm(input_ptr + input_inside_tile_offset , a_frag[(k + 1) % PIPE_INSIDE_TILE]);
            // ldsm(weight_ptr + weight_inside_tile_offset, b_frag[(k + 1) % PIPE_INSIDE_TILE]);
            // mma_m16n16k16_bf16bf16bf32(s_frag[m][n],
            //                            a_frag[k % PIPE_INSIDE_TILE],
            //                            b_frag[k % PIPE_INSIDE_TILE],
            //                            s_frag[m][n]);

            ldsm(input_ptr , a_frag[(k + 1) % PIPE_INSIDE_TILE]);
            ldsm(weight_ptr, b_frag[(k + 1) % PIPE_INSIDE_TILE]);
            mma_m16n16k16_bf16bf16bf32(s_frag[m][n],
                                       a_frag[k % PIPE_INSIDE_TILE],
                                       b_frag[k % PIPE_INSIDE_TILE],
                                       s_frag[m][n]);
          // } // loop for NUM_ITERS_N, it may not be 1
        // } // loop for NUM_ITERS_M, it should always be 1, no sense loop
      } // loop for NUM_ITERS_K
    } // loop for FORLOOP_RANGE

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
                bfloat16(s_frag[m][n][(i << 1)]);
            output_smem.at(row_in_warp, col + 1) +=
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      } // loop for NUM_ITERS_N, it may not be 1
    } // loop for NUM_ITERS_M, it should always be 1, no sense loop

    __syncthreads();
  } // loop among OUTPUT_ATOM_SIZE

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
