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
          int PIPE_MAX = 3>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual) {
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  constexpr int NUM_OUTPUT_ATOMS = OUTPUT_SIZE / OUTPUT_ATOM_SIZE;
  constexpr int TILE_SIZE = 128;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int WEIGHT_PIPE_MAX =
      PIPE_MAX < NUM_OUTPUT_ATOMS ? PIPE_MAX : NUM_OUTPUT_ATOMS;

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
      ZERO_BUFFER_OFFSET + sizeof(T) * 8;
  // sizeof(T) * BATCH_SIZE * TILE_SIZE

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
      SHARED_INPUT_BUFFER_OFFSET + sizeof(T) * BATCH_SIZE * TILE_SIZE;
  // sizeof(T) * TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_ATOM_SIZE

  constexpr size_t MM_INTERMEDIATE_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * TILE_SIZE * WEIGHT_PIPE_MAX * OUTPUT_ATOM_SIZE;
  // sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE

  constexpr size_t SHARED_OUTPUT_OFFSET =
      MM_INTERMEDIATE_OFFSET +
      sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
  // sizeof(T) * BATCH_SIZE * OUTPUT_SIZE

  // zero buffer
  T *zero_buf = (T *)(smem + ZERO_BUFFER_OFFSET);
  vec_zero_t<T, 8>::fill_zero(zero_buf);

  // copy
  T *shared_input_buffer = (T *)(smem + SHARED_INPUT_BUFFER_OFFSET);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);

  // intermediate
  T *mm_intermediate = (T *)(smem + MM_INTERMEDIATE_OFFSET);

  // output
  T *shared_output = (T *)(smem + SHARED_OUTPUT_OFFSET);

  // define the swizzle mode
  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 3, 3, 3, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem = smem_col<T,
                              3,
                              3,
                              3,
                              TILE_SIZE,
                              WEIGHT_PIPE_MAX * OUTPUT_ATOM_SIZE,
                              TILE_SIZE>;
  using OutputFullSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  using OutputAtomViewSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_SIZE>;
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

  OutputFullSmem output_smem(shared_output);
  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  // Initialize output_smem: if residual is provided, preload it; otherwise zero

#pragma unroll
  for (int i = threadIdx.x; i < BATCH_SIZE * OUTPUT_SIZE / CHUNK_SIZE;
       i += NUM_THREADS) {
    int row = i / (OUTPUT_SIZE / CHUNK_SIZE);
    int col = (i % (OUTPUT_SIZE / CHUNK_SIZE)) << log2_CHUNK_SIZE;
    if (residual) {
      load_smem(output_smem(row, col), residual_dmem(row, col));
    } else {
      *((__uint128_t *)((void *)&output_smem.at(row, col))) = 0ul;
    }
  }

  // Outer loop over K tiles; inner loop over output atoms
  for (int for_idx = 0; for_idx < FORLOOP_RANGE;
       for_idx++, d_weight += TILE_SIZE) {

    weight_dmem.set_ptr(d_weight);

    // Load input tile for this K-slice once into single-plane smem
#pragma unroll
    for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
      int row = i >> log2_CHUNKS_PER_ROW_A;
      int dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
      int src_col = dst_col + (for_idx << log2_constexpr(TILE_SIZE));
      load_smem(input_smem(row, dst_col), input_dmem(row, src_col));
    }
    cp_async_fence();

    // Warm up weight tiles for the first WEIGHT_PIPE_MAX - 1 atoms
    for (int w_pipe = 0; w_pipe < WEIGHT_PIPE_MAX - 1; ++w_pipe) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int src_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;

        int src_stage = w_pipe;
        int buffer_stage = (w_pipe + 1) % WEIGHT_PIPE_MAX;

        int col_within = i >> log2_CHUNKS_PER_COL_B;
        int src_col = src_stage * OUTPUT_ATOM_SIZE + col_within;
        int dst_col = buffer_stage * OUTPUT_ATOM_SIZE + col_within;

        load_smem(weight_smem(dst_row, dst_col), weight_dmem(src_row, src_col));
      }
      cp_async_fence();
    }

    // Loop over output atoms for this K-slice
    for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
         output_atom_idx++) {
      // Prefetch next weight atom into ring buffer stage_write
      if (output_atom_idx + WEIGHT_PIPE_MAX - 1 < NUM_OUTPUT_ATOMS) {
#pragma unroll
        for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int src_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;

          int src_stage = output_atom_idx + WEIGHT_PIPE_MAX - 1;
          int buffer_stage =
              (output_atom_idx + WEIGHT_PIPE_MAX) % WEIGHT_PIPE_MAX;

          int col_within = i >> log2_CHUNKS_PER_COL_B;
          int src_col = src_stage * OUTPUT_ATOM_SIZE + col_within;
          int dst_col = buffer_stage * OUTPUT_ATOM_SIZE + col_within;

          load_smem(weight_smem(dst_row, dst_col),
                    weight_dmem(src_row, src_col));
        }
        cp_async_fence();
        cp_async_wait<WEIGHT_PIPE_MAX - 1>();
      } else if (output_atom_idx + WEIGHT_PIPE_MAX - 1 == NUM_OUTPUT_ATOMS) {
        cp_async_wait<0>();
      }
      __syncthreads();

      // accumulator
      float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          clear_8_floats(s_frag[m][n]);
        }
      }

      // MMA using the loaded input and weight tiles
      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_valid = (m_row < num_active_tokens);
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
            int weight_stage_read = (output_atom_idx + 1) % WEIGHT_PIPE_MAX;
            T *src_ptr =
                is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm(weight_smem(n_row,
                             weight_stage_read * OUTPUT_ATOM_SIZE + n_col),
                 b_frag);
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }
      __syncthreads();

      // write back to shared intermediate
      // TODO: mm_intermediate_smem can be removed when NUM_WARPS_K == 1
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
          for (uint32_t i = 0; i < 4; i++) {
            int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
            if (row_in_warp < num_active_tokens) {
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

      // Accumulate this atom's contribution into the full output_smem at offset
      int atom_col_offset = output_atom_idx * OUTPUT_ATOM_SIZE;
      if (NUM_WARPS_K > 1) {
        // Reduce across K-warps and accumulate into output_smem slice
        OutputAtomViewSmem out_slice(shared_output + atom_col_offset);
        reduction_sum_row_add<OutputAtomViewSmem,
                              decltype(mm_intermediate_smem)>(
            out_slice, mm_intermediate_smem);
      } else {
        // Directly accumulate without reduction across warps
        for (int idx = threadIdx.x; idx < BATCH_SIZE * OUTPUT_ATOM_SIZE;
             idx += NUM_THREADS) {
          int row = idx / OUTPUT_ATOM_SIZE;
          if (row < num_active_tokens) {
            int col = idx % OUTPUT_ATOM_SIZE;
            float prev = float(output_smem.at(row, atom_col_offset + col));
            float addv = float(mm_intermediate_smem.at(row, col));
            output_smem.at(row, atom_col_offset + col) = bfloat16(prev + addv);
          }
        }
      }
      __syncthreads();
    }
  }

  // Final writeback: store accumulated output (residual already included if
  // any)
  // TODO: loop over BATCH_SIZE * OUTPUT_SIZE
#pragma unroll
  for (int row = 0; row < num_active_tokens; row++) {
#pragma unroll
    for (int i = threadIdx.x; i < OUTPUT_SIZE; i += NUM_THREADS) {
      output_dmem.at(row, i) = output_smem.at(row, i);
    }
  }
}

} // namespace kernel