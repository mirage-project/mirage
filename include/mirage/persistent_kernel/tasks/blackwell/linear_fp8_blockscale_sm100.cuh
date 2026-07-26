/* Copyright 2026 CMU
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
#include "../common/common_header.cuh"
#include <cstdint>

// Dense FP8 GEMM that consumes the checkpoint's PRESERVED float32 block scales.
//
// Why a separate kernel from linear_fp8_sm100.cuh: that one is a DeepGEMM port
// built on the SM100 block-scaled UMMA (`tcgen05.mma.kind::mxf8f6f4`), whose
// scale-factor operands are hardware-typed UE8M0 (`cutlass::float_ue8m0_t`,
// linear_fp8_sm100.cuh:409-412). Exponent-only scales cannot represent the
// checkpoint's float32 `weight_scale_inv`, so that path first rewrites the
// weights under fresh per-row power-of-two scales
// (DeepSeekV3Builder._requantize_fp8_for_ue8m0). This kernel instead applies
// the float32 scales OUTSIDE the MMA, at the 128-K-tile boundary:
//
//   out[m,n] = sum_kt  a_scale[m,kt] * b_scale[n/128,kt]
//                    * sum_{k in kt} A_q[m,k] * B_q[n,k]
//
// The inner sum is a plain (unscaled) FP8 MMA with FP32 accumulation; the
// per-tile product `a_scale * b_scale` is folded into an FP32 register
// accumulator once per K tile. That is the "promotion" semantics class of
// vLLM's CutlassFp8BlockScaledMMKernel (docs/qwen35/vllm-graph.md 3.4-3.5) and
// keeps the checkpoint's 128x128 block values bit-exact: no power-of-two
// rounding and no per-row collapse (docs/qwen35/v1-architecture.md 6.2).
//
// Scale layouts consumed here (both float32, both exactly as produced
// upstream):
//   a_scale: [BATCH_SIZE, REDUCTION_SIZE/128] row-major -- the fp32-scale
//            variant of per_token_group_quantize_fp8_task_impl (SCALE_UE8M0 =
//            false writes `[batch, num_groups]`, per_token_group_quantize_fp8
//            .cuh:124-127), i.e. the same primitive vLLM runs with
//            use_ue8m0=False.
//   b_scale: [OUTPUT_SIZE/128, REDUCTION_SIZE/128] row-major -- the
//            checkpoint's `weight_scale_inv` slice for this task's output rows,
//            attached without any transform.

namespace kernel {

namespace linear_fp8_blockscale {

constexpr int GROUP_K = 128;  // scale group along K (checkpoint block size)
constexpr int BLOCK_N = 128;  // scale block along N (checkpoint block size)
constexpr int NUM_STAGES = 2; // cp.async double buffering
// Shared-memory tiles are byte-addressed E4M3. The 16-byte row padding makes
// the MMA fragment reads (one uint32 per thread at row lane/4, byte offset
// 4*(lane%4)) hit 32 distinct banks instead of 4.
constexpr int SMEM_ROW_STRIDE = GROUP_K + 16;

// One M tile per pass, sized to the batch so decode (B <= 16) issues exactly
// one 16-row MMA row-block instead of padding out to 64.
constexpr int tile_m(int batch_size) {
  return batch_size <= 16 ? 16 : (batch_size <= 32 ? 32 : 64);
}

// Dynamic shared memory the task needs: A and B staging tiles, double buffered.
constexpr int smem_bytes(int batch_size) {
  return NUM_STAGES * (tile_m(batch_size) + BLOCK_N) * SMEM_ROW_STRIDE;
}

} // namespace linear_fp8_blockscale

// D += A * B for one m16n8k32 E4M3 tile, FP32 accumulate. A/B fragments are
// 4/2 packed uint32 registers, D is 4 floats; the operand layouts are the
// standard ones documented for this instruction (validated against a scalar
// reference on sm_100a before this kernel was written).
__device__ __forceinline__ void
    mma_m16n8k32_e4m3_f32(float *D, uint32_t const *A, uint32_t const *B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// out[BATCH_SIZE, OUTPUT_SIZE] = input_fp8 @ weight_fp8^T (+ residual), with
// the float32 block scales applied per 128-element K tile.
//
// OUTPUT_SIZE is this TASK's slice of the projection (the grid splits the
// weight's row dimension), and both scale tensors are sliced the same way, so
// b_scale here is [OUTPUT_SIZE/128, REDUCTION_SIZE/128].
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE,
          bool WITH_RESIDUAL>
__device__ __forceinline__ void
    linear_fp8_blockscale_task_impl(void const *__restrict__ input_fp8_ptr,
                                    void const *__restrict__ input_scale_ptr,
                                    void const *__restrict__ weight_fp8_ptr,
                                    void const *__restrict__ weight_scale_ptr,
                                    void const *__restrict__ residual_ptr,
                                    void *__restrict__ output_ptr) {
  using namespace linear_fp8_blockscale;
  constexpr int MMA_K = 32; // K per mma.sync instruction
  constexpr int NUM_K_SUBTILES = GROUP_K / MMA_K;
  constexpr int NUM_K_TILES = REDUCTION_SIZE / GROUP_K;
  constexpr int NUM_N_BLOCKS = OUTPUT_SIZE / BLOCK_N;

  static_assert(REDUCTION_SIZE % GROUP_K == 0,
                "FP8 block scales require K to be a multiple of 128");
  static_assert(OUTPUT_SIZE % BLOCK_N == 0,
                "FP8 block scales require the per-task N to be a multiple "
                "of 128 (one checkpoint scale block)");
  static_assert(WORKER_NUM_THREADS == 256,
                "linear_fp8_blockscale_sm100 assumes the SM100 worker's "
                "256-thread block");
  static_assert(smem_bytes(BATCH_SIZE) <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "linear_fp8_blockscale_sm100 exceeds the worker smem budget");

  constexpr int TILE_M = tile_m(BATCH_SIZE);
  constexpr int NUM_M_TILES = (BATCH_SIZE + TILE_M - 1) / TILE_M;
  constexpr int NUM_TASK_WARPS = WORKER_NUM_THREADS / NUM_THREADS_PER_WARP;
  constexpr int WARPS_M = TILE_M / 16;
  constexpr int WARPS_N = NUM_TASK_WARPS / WARPS_M;
  constexpr int N_PER_WARP = BLOCK_N / WARPS_N;
  constexpr int MMA_N_ITERS = N_PER_WARP / 8;

  constexpr int SMEM_A_BYTES = TILE_M * SMEM_ROW_STRIDE;
  constexpr int SMEM_B_BYTES = BLOCK_N * SMEM_ROW_STRIDE;
  constexpr int BYTES_PER_CP = 16;
  constexpr int CHUNKS_PER_ROW = GROUP_K / BYTES_PER_CP;
  constexpr int A_CHUNKS = TILE_M * CHUNKS_PER_ROW;
  constexpr int B_CHUNKS = BLOCK_N * CHUNKS_PER_ROW;

  uint8_t const *__restrict__ d_input =
      static_cast<uint8_t const *>(input_fp8_ptr);
  uint8_t const *__restrict__ d_weight =
      static_cast<uint8_t const *>(weight_fp8_ptr);
  float const *__restrict__ d_input_scale =
      static_cast<float const *>(input_scale_ptr);
  float const *__restrict__ d_weight_scale =
      static_cast<float const *>(weight_scale_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // 16-byte alignment is required by cp.async's 16-byte destination form.
  extern __shared__ __align__(16) uint8_t smem_blockscale[];
  uint8_t *smem_a = smem_blockscale;
  uint8_t *smem_b = smem_a + NUM_STAGES * SMEM_A_BYTES;

  int const tid = threadIdx.x;
  int const lane = lane_id();
  int const warp = tid / NUM_THREADS_PER_WARP;
  int const warp_m = warp / WARPS_N;
  int const warp_n = warp % WARPS_N;
  // Fragment coordinates shared by A, B and D: the operand row/column a lane
  // owns is lane/4, the k byte offset it owns is 4*(lane%4).
  int const frag_row = lane >> 2;
  int const frag_k = (lane & 3) * 4;

  // Stage a K tile of A (rows m0..m0+TILE_M) and B (rows n0..n0+BLOCK_N).
  // Rows past the batch are zero-filled by cp.async's src-size predicate, so
  // they contribute nothing to the MMA.
  auto load_k_tile = [&](int stage, int m0, int n0, int kt) {
    uint8_t *stage_a = smem_a + stage * SMEM_A_BYTES;
    uint8_t *stage_b = smem_b + stage * SMEM_B_BYTES;
    size_t const k_off = (size_t)kt * GROUP_K;
    for (int c = tid; c < A_CHUNKS; c += WORKER_NUM_THREADS) {
      int const r = c / CHUNKS_PER_ROW;
      int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
      int const g_row = m0 + r;
      // Rows past the batch copy 0 bytes (cp.async zero-fills); their source
      // address is clamped so it always stays inside the tensor.
      int const src_row = g_row < BATCH_SIZE ? g_row : BATCH_SIZE - 1;
      load_smem_with_predict<uint8_t, BYTES_PER_CP>(
          stage_a + r * SMEM_ROW_STRIDE + x,
          d_input + (size_t)src_row * REDUCTION_SIZE + k_off + x,
          g_row < BATCH_SIZE);
    }
    for (int c = tid; c < B_CHUNKS; c += WORKER_NUM_THREADS) {
      int const r = c / CHUNKS_PER_ROW;
      int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
      load_smem<uint8_t, BYTES_PER_CP>(
          stage_b + r * SMEM_ROW_STRIDE + x,
          d_weight + (size_t)(n0 + r) * REDUCTION_SIZE + k_off + x);
    }
  };

  for (int nb = 0; nb < NUM_N_BLOCKS; ++nb) {
    int const n0 = nb * BLOCK_N;
    for (int mt = 0; mt < NUM_M_TILES; ++mt) {
      int const m0 = mt * TILE_M;
      int const row0 = m0 + warp_m * 16 + frag_row;
      int const row1 = row0 + 8;

      // Promoted (fully scaled) FP32 accumulator for this output tile.
      float acc[MMA_N_ITERS][4];
#pragma unroll
      for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[ni][i] = 0.0f;
        }
      }

      load_k_tile(0, m0, n0, 0);
      cp_async_fence();

      for (int kt = 0; kt < NUM_K_TILES; ++kt) {
        if (kt + 1 < NUM_K_TILES) {
          load_k_tile((kt + 1) & 1, m0, n0, kt + 1);
          cp_async_fence();
          cp_async_wait<1>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();

        uint8_t const *stage_a = smem_a + (kt & 1) * SMEM_A_BYTES;
        uint8_t const *stage_b = smem_b + (kt & 1) * SMEM_B_BYTES;

        // Unscaled FP8 MMA over the 128-element K tile.
        float partial[MMA_N_ITERS][4];
#pragma unroll
        for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            partial[ni][i] = 0.0f;
          }
        }
#pragma unroll
        for (int ks = 0; ks < NUM_K_SUBTILES; ++ks) {
          int const k_byte = ks * MMA_K + frag_k;
          uint32_t a_frag[4];
          a_frag[0] = *reinterpret_cast<uint32_t const *>(
              stage_a + (warp_m * 16 + frag_row) * SMEM_ROW_STRIDE + k_byte);
          a_frag[1] = *reinterpret_cast<uint32_t const *>(
              stage_a + (warp_m * 16 + frag_row + 8) * SMEM_ROW_STRIDE +
              k_byte);
          a_frag[2] = *reinterpret_cast<uint32_t const *>(
              stage_a + (warp_m * 16 + frag_row) * SMEM_ROW_STRIDE + k_byte +
              16);
          a_frag[3] = *reinterpret_cast<uint32_t const *>(
              stage_a + (warp_m * 16 + frag_row + 8) * SMEM_ROW_STRIDE +
              k_byte + 16);
#pragma unroll
          for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
            int const col = warp_n * N_PER_WARP + ni * 8 + frag_row;
            uint32_t b_frag[2];
            b_frag[0] = *reinterpret_cast<uint32_t const *>(
                stage_b + col * SMEM_ROW_STRIDE + k_byte);
            b_frag[1] = *reinterpret_cast<uint32_t const *>(
                stage_b + col * SMEM_ROW_STRIDE + k_byte + 16);
            mma_m16n8k32_e4m3_f32(partial[ni], a_frag, b_frag);
          }
        }

        // Promotion: fold this K tile's float32 scales into the accumulator.
        // Rows past the batch read no scale (their data is zero anyway).
        float const b_scale = d_weight_scale[nb * NUM_K_TILES + kt];
        float const s0 =
            (row0 < BATCH_SIZE)
                ? d_input_scale[(size_t)row0 * NUM_K_TILES + kt] * b_scale
                : 0.0f;
        float const s1 =
            (row1 < BATCH_SIZE)
                ? d_input_scale[(size_t)row1 * NUM_K_TILES + kt] * b_scale
                : 0.0f;
#pragma unroll
        for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
          acc[ni][0] += partial[ni][0] * s0;
          acc[ni][1] += partial[ni][1] * s0;
          acc[ni][2] += partial[ni][2] * s1;
          acc[ni][3] += partial[ni][3] * s1;
        }
        __syncthreads();
      }

#pragma unroll
      for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
        int const col = n0 + warp_n * N_PER_WARP + ni * 8 + (lane & 3) * 2;
        if (row0 < BATCH_SIZE) {
          float v0 = acc[ni][0];
          float v1 = acc[ni][1];
          if constexpr (WITH_RESIDUAL) {
            v0 += float(d_residual[(size_t)row0 * O_STRIDE + col]);
            v1 += float(d_residual[(size_t)row0 * O_STRIDE + col + 1]);
          }
          d_output[(size_t)row0 * O_STRIDE + col] = T(v0);
          d_output[(size_t)row0 * O_STRIDE + col + 1] = T(v1);
        }
        if (row1 < BATCH_SIZE) {
          float v2 = acc[ni][2];
          float v3 = acc[ni][3];
          if constexpr (WITH_RESIDUAL) {
            v2 += float(d_residual[(size_t)row1 * O_STRIDE + col]);
            v3 += float(d_residual[(size_t)row1 * O_STRIDE + col + 1]);
          }
          d_output[(size_t)row1 * O_STRIDE + col] = T(v2);
          d_output[(size_t)row1 * O_STRIDE + col + 1] = T(v3);
        }
      }
      // The next M tile / N block reuses both staging buffers.
      __syncthreads();
    }
  }
}

} // namespace kernel
