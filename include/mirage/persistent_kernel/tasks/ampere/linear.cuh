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

#define DEBUG 0

#if DEBUG
#define DCHECK(condition)                                                      \
  if ((condition) == 0) {                                                      \
    printf("Dcheck failed at %s:%d\n", __FILE__, __LINE__);                    \
  }
#else
#define DCHECK(condition)
#endif // DEBUG

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
  constexpr int OUTPUT_ATOM_SIZE = OUTPUT_SIZE <= 64 ? OUTPUT_SIZE : 64;
  constexpr int log2_OUTPUT_ATOM_SIZE = log2_constexpr(OUTPUT_ATOM_SIZE);

  constexpr int TILE_SIZE = 128;
  constexpr int log2_TILE_SIZE = log2_constexpr(TILE_SIZE);
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int ADJUSTED_PIPE_MAX =
      PIPE_MAX < FORLOOP_RANGE ? PIPE_MAX : FORLOOP_RANGE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_OUTPUT = BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_C = OUTPUT_ATOM_SIZE / CHUNK_SIZE;

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
      (OUTPUT_SIZE + OUTPUT_ATOM_SIZE - 1) / OUTPUT_ATOM_SIZE;
  constexpr int NUM_ITERS_K =
      (TILE_SIZE + NUM_WARPS_K * 16 - 1) / (NUM_WARPS_K * 16);
  // constexpr int NUM_ITERS_K = 8;

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
      sizeof(T) * BATCH_SIZE * ADJUSTED_PIPE_MAX * TILE_SIZE;

  constexpr size_t SHARED_OUTPUT_OFFSET =
      // MM_INTERMEDIATE_OFFSET +
      SHARED_WEIGHT_BUFFER_OFFSET +
      sizeof(T) * TILE_SIZE * ADJUSTED_PIPE_MAX * OUTPUT_ATOM_SIZE;

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
  using InputSmem =
      smem_row_2dcol<T, 3, 3, 3, BATCH_SIZE, TILE_SIZE, ADJUSTED_PIPE_MAX>;
  using WeightSmem = smem_col_2drow<T,
                                    3,
                                    3,
                                    3,
                                    TILE_SIZE,
                                    OUTPUT_ATOM_SIZE,
                                    ADJUSTED_PIPE_MAX>;
  using OutputFullSmem =
      smem_row<T, 3, 3, 3, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;

  // we no longger need zero buffer, but we could keep it to make sure shared
  // memory was aligned.
  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input_buffer);
  WeightSmem weight_smem(shared_weight_buffer);

  OutputFullSmem output_smem(shared_output);

#pragma unroll
  for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
    // If we use NUM_ITERS_M and NUM_ITERS_N inside NUM_ITERS_K, the
    // loop for NUM_ITERS_K couldn't be unrolled in nvcc which hurts
    // performance.
#pragma unroll
    for (uint32_t nn = 0; nn < NUM_ITERS_N; nn++) {
      float s_frag[8];

      // should we sync here? if NUM_ITERS_N > 1, I suppose we should do it,
      // because we will write output_smem later, but it may be still used in
      // some warp which are still write to gmem.
      if (NUM_ITERS_N > 1) {
        __syncthreads();
      }
      // Initialize output_smem: if residual is provided, preload it; otherwise
      // zero
#pragma unroll
      for (int i = threadIdx.x; i < BATCH_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
           i += NUM_THREADS) {
        int row = i / (OUTPUT_ATOM_SIZE / CHUNK_SIZE);
        int dst_col = (i % (OUTPUT_ATOM_SIZE / CHUNK_SIZE)) << log2_CHUNK_SIZE;
        int src_col = dst_col + (nn << log2_OUTPUT_ATOM_SIZE);
        // TODO: use ignore-src in load_smem to avoid if-else
        if (residual) {
          load_smem(output_smem(row, dst_col), residual_dmem(row, src_col));
        } else {
          *((__uint128_t *)((void *)&output_smem.at(row, dst_col))) = 0ul;
        }
      }

      // initialize registers
#pragma unroll
      for (uint32_t r = 0; r < 8; r++) {
        s_frag[r] = 0;
      }

      int ismem_read_stage = 0;
      int ismem_write_stage = 0;

      // Warm up weight and input tiles for the first ADJUSTED_PIPE_MAX - 1
      // tile.
#pragma unroll
      for (int istage = 0; istage < ADJUSTED_PIPE_MAX - 1; ++istage) {
        // we don't need module for ADJUSTED_PIPE_MAX here, because we just load
        // ADJUSTED_PIPE_MAX - 1 pipe.
        int src_stage_offset = istage << log2_TILE_SIZE;

#pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS_A / NUM_THREADS; chunk++) {
          int tid = threadIdx.x;
          int threadCol = (tid & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
          constexpr int ROWS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_ROW_A;

          int dst_col = threadCol;
          int src_col = dst_col + src_stage_offset;

          int row_within = threadRow + chunk * ROWS_PER_ITERATION;
          int src_row = row_within;
          int dst_row = row_within;

          load_smem(input_smem(dst_row, dst_col, istage),
                    input_dmem(src_row, src_col));
        }
#pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS_B / NUM_THREADS; chunk++) {
          int tid = threadIdx.x;
          int threadRow = (tid & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int threadCol = tid >> log2_CHUNKS_PER_COL_B;
          constexpr int COLS_PER_ITERATION = NUM_THREADS / CHUNKS_PER_COL_B;

          int dst_row = threadRow;
          int src_row = dst_row + src_stage_offset;

          int col_within = threadCol + chunk * COLS_PER_ITERATION;
          int src_col = (nn << log2_OUTPUT_ATOM_SIZE) + col_within;
          int dst_col = col_within;

          load_smem(weight_smem(dst_row, dst_col, istage),
                    weight_dmem(src_row, src_col));
        }
        cp_async_fence();

        ++ismem_write_stage;
      } // warm up for ADJUSTED_PIPE_MAX - 1

      constexpr int PIPE_INSIDE_TILE = 2;
      uint32_t a_frag[PIPE_INSIDE_TILE][4], b_frag[PIPE_INSIDE_TILE][4];
      // wait for first warm up pipeline cp.async finished
      cp_async_wait<ADJUSTED_PIPE_MAX - 2>();
      __syncthreads();

      int warmup_m_col =
          (warp_row << (4 + log2_NUM_ITERS_K)) + ((lane_idx >> 4) << 3);
      int warmup_n_row =
          (warp_row << (4 + log2_NUM_ITERS_K)) + (((lane_idx & 0xF) >> 3) << 3);
      int warmup_smem_row = (lane_idx & 0xF);
      int warmup_n_col =
          (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
      T *warmup_input_ptr = input_smem(warmup_smem_row, warmup_m_col, 0);
      DCHECK(warmup_n_col < OUTPUT_ATOM_SIZE);
      T *warmup_weight_ptr = weight_smem(warmup_n_row, warmup_n_col, 0);

      ldsm(warmup_input_ptr, a_frag[0]);
      ldsm(warmup_weight_ptr, b_frag[0]);

#pragma unroll 1
      for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
#pragma unroll
        for (int k = 0; k < NUM_ITERS_K; k++) {
          // TODO(Wenqin): use pointer advance for the pointer for input and
          // weight shared memory instead of address calculation for
          // input_smem and weight_smem, because in each iteration in the K
          // dim for the OUTER_ROW/COL, they just advanced a compile-time know
          // offset, and it seems the CUTLASS version just use some ADD inst to
          // do it.
          int k_next = (k + 1) % NUM_ITERS_K;

          if (k == 0) {
            // loading next tile (for_idx + ADJUSTED_PIPE_MAX - 1) when k is 0.
            if (for_idx + ADJUSTED_PIPE_MAX - 1 < FORLOOP_RANGE) {
              int src_stage_offset = (for_idx + ADJUSTED_PIPE_MAX - 1)
                                     << log2_TILE_SIZE;
              // Prefetch next weight tile into ring buffer stage_write
              // Load input tile at the first output tile
#pragma unroll
              for (int chunk = 0; chunk < NUM_CHUNKS_A / NUM_THREADS; chunk++) {
                // we don't need to hoist the threadCol and threadRow,,
                // accorrding to experiment, the nvcc could hoist these const.
                int tid = threadIdx.x;
                int threadCol = (tid & (CHUNKS_PER_ROW_A - 1))
                                << log2_CHUNK_SIZE;
                int threadRow = tid >> log2_CHUNKS_PER_ROW_A;
                constexpr int ROWS_PER_ITERATION =
                    NUM_THREADS / CHUNKS_PER_ROW_A; // 8

                int dst_col = threadCol;
                int src_col = dst_col + src_stage_offset;

                int row_within = threadRow + chunk * ROWS_PER_ITERATION;
                int src_row = row_within;
                int dst_row = row_within;

                load_smem(input_smem(dst_row, dst_col, ismem_write_stage),
                          input_dmem(src_row, src_col));
              }
#pragma unroll
              for (int chunk = 0; chunk < NUM_CHUNKS_B / NUM_THREADS; chunk++) {
                int tid = threadIdx.x;
                int threadRow = (tid & (CHUNKS_PER_COL_B - 1))
                                << log2_CHUNK_SIZE;
                int threadCol = tid >> log2_CHUNKS_PER_COL_B;
                constexpr int COLS_PER_ITERATION =
                    NUM_THREADS / CHUNKS_PER_COL_B; // 8

                int dst_row = threadRow;
                int src_row = dst_row + src_stage_offset;

                int col_within = threadCol + chunk * COLS_PER_ITERATION;
                int src_col = (nn << log2_OUTPUT_ATOM_SIZE) + col_within;
                int dst_col = col_within;

                load_smem(weight_smem(dst_row, dst_col, ismem_write_stage),
                          weight_dmem(src_row, src_col));
              }
              ismem_write_stage = (ismem_write_stage + 1) % ADJUSTED_PIPE_MAX;
            }
            cp_async_fence();
          } // k == 0 for load next tile

          if (k == NUM_ITERS_K - 1) {
            // wait cp.async because we will load next tile data in to regs
            // when k == NUM_ITERS_K - 1.
            if (FORLOOP_RANGE - for_idx > 2) {
              cp_async_wait<ADJUSTED_PIPE_MAX - 2>();
            } else {
              cp_async_wait<0>();
            }
            __syncthreads();

            // TODO(Wenqin): The comment out code below here is what we could
            // do for just use ADD for input and weight shared memory pointer.
            // int tmp_ismem_read_stage = ismem_read_stage;
            ismem_read_stage = (ismem_read_stage + 1) % ADJUSTED_PIPE_MAX;
            // input_ptr += (ismem_read_stage - tmp_ismem_read_stage) * (8 *
            // 128); weight_ptr += (ismem_read_stage - tmp_ismem_read_stage) *
            // (64 * 128);
          } // k == NUM_ITERS_K - 1

          static_assert(NUM_ITERS_M == 1);

          int m_row = (lane_idx & 0xF) + (m << 4);
          int n_col =
              (warp_col << 4) + ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
          DCHECK(n_col < OUTPUT_ATOM_SIZE);

          int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                      ((lane_idx >> 4) << 3);
          int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k_next << 4) +
                      (((lane_idx & 0xF) >> 3) << 3);

          int smem_row = m_row;
          T *valid_input_ptr = input_smem(smem_row, m_col, ismem_read_stage);
          // we don't need to check for is_input_valid, because we will use
          // num_active_tokens for the output, we will just pick valid output.
          T *input_ptr = valid_input_ptr;

          T *valid_weight_ptr = weight_smem(n_row, n_col, ismem_read_stage);
          T *weight_ptr = valid_weight_ptr;

          ldsm(input_ptr, a_frag[(k + 1) % PIPE_INSIDE_TILE]);
          ldsm(weight_ptr, b_frag[(k + 1) % PIPE_INSIDE_TILE]);
          mma_m16n16k16_bf16bf16bf32(s_frag,
                                     a_frag[k % PIPE_INSIDE_TILE],
                                     b_frag[k % PIPE_INSIDE_TILE],
                                     s_frag);

        } // loop for NUM_ITERS_K
      }   // loop for FORLOOP_RANGE

#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
        int col_within =
            (warp_col << 4) + ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
        int col = col_within;
        DCHECK(col_within < OUTPUT_ATOM_SIZE);
        if (row_in_warp < num_active_tokens) {
          // TODO: try st.matrix here?
          output_smem.at(row_in_warp, col) += bfloat16(s_frag[(i << 1)]);
          output_smem.at(row_in_warp, col + 1) +=
              bfloat16(s_frag[(i << 1) | 0x1]);
        }
      }
      __syncthreads();

      // Final writeback: store accumulated output (residual already included if
      // any)
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
        int row = i / CHUNKS_PER_ROW_C;
        int src_col = (i % CHUNKS_PER_ROW_C) << log2_CHUNK_SIZE;
        int dst_col = src_col + (nn << log2_OUTPUT_ATOM_SIZE);
        *((__uint128_t *)((void *)&output_dmem.at(row, dst_col))) =
            *((__uint128_t *)((void *)&output_smem.at(row, src_col)));
      }
    } // loop for NUM_ITERS_N, it may not be 1
  }   // loop for NUM_ITERS_M, it should always be 1, no sense loop
}

} // namespace kernel
