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
// The dense preserved-block-scale kernel: this file reuses its `mma_m16n8k32
// _e4m3_f32` wrapper so both kernels are provably the same MMA instruction.
#include "linear_fp8_blockscale_sm100.cuh"
#include <cstdint>

// Grouped (MoE) FP8 GEMM that consumes the checkpoint's PRESERVED float32
// block scales -- the routed-expert mirror of linear_fp8_blockscale_sm100.cuh.
//
// Why it exists: fp8_group_gemm_sm100.cuh is built on the SM100 block-scaled
// UMMA (`tcgen05.mma.kind::mxf8f6f4.block_scale`), whose scale operands are
// hardware-typed UE8M0. Its scale warp therefore converts BOTH scale operands
// with
//
//     ue8m0 = (__float_as_uint(sf_val) >> 23) & 0xFF     [:1379, :1394]
//
// which keeps only the float32 EXPONENT FIELD: an exact power-of-two
// TRUNCATION toward zero, applied scale = sf / m for the mantissa m in [1,2).
// Probe P2 (demo/qwen3_5/accept/probes/fp8/p2_verdict.json) measured what that
// costs on this checkpoint's real expert scales; this kernel is the named
// fail-closed fallback of docs/qwen35/v1-architecture.md 6.2.
//
// Semantics (identical promotion class to the dense fp32-scale kernel and to
// vLLM's CutlassFp8BlockScaledMMKernel):
//
//   out[m, slot, n] = sum_kt  a_scale[row(m,slot), kt] * b_scale[e, n/128, kt]
//                           * sum_{k in kt} A_q[row(m,slot), k] * B_q[e, n, k]
//
// The inner sum is an unscaled FP8 `mma.sync` with FP32 accumulation; the
// per-tile product a_scale*b_scale is folded into an FP32 accumulator once per
// 128-element K tile. No power-of-two rounding, no per-row collapse.
//
// Tensors (all exactly as produced upstream, no transform):
//   input_fp8    W13_LINEAR : [BATCH, K]              e4m3
//                otherwise  : [BATCH, NUM_TOPK, K]    e4m3
//   input_scale  W13_LINEAR : [BATCH, K/128]          float32
//                otherwise  : [BATCH, NUM_TOPK, K/128]float32
//                (the fp32-scale variant of per_token_group_quantize_fp8, i.e.
//                 quantize_fp8_layer(scale_ue8m0=False))
//   weight_fp8   [NUM_EXPERTS, ORIG_OUTPUT_SIZE, K]   e4m3, this task's N
//                slice selected by the caller's pointer offset
//   weight_scale [NUM_EXPERTS, ORIG_OUTPUT_SIZE/128, K/128] float32 -- THE
//                CHECKPOINT'S `weight_scale_inv`, UNEXPANDED. The grouped
//                UE8M0 path needs builder-side repeat_interleave(128) to a
//                per-row [E*N, K/128] tensor; this kernel indexes the block
//                directly, so that 128x expansion (and its 128x scale-memory
//                traffic) disappears.
//   routing      [NUM_EXPERTS, BATCH] int32, value = topk slot + 1, 0 = not
//                routed (same convention as fp8_group_gemm_sm100.cuh)
//   mask         [NUM_EXPERTS + 1] int32: mask[i] = i-th activated expert id,
//                mask[NUM_EXPERTS] = number of activated experts
//   output       [BATCH, NUM_TOPK, ORIG_OUTPUT_SIZE] bf16; only routed
//                (token, slot) pairs are written, so the caller zero-inits
//                (the MoE builder already does via tensor_init_layer).

namespace kernel {

namespace moe_fp8_blockscale {

constexpr int GROUP_K = 128;  // scale group along K (checkpoint block size)
constexpr int BLOCK_N = 128;  // scale block along N (checkpoint block size)
constexpr int NUM_STAGES = 2; // cp.async double buffering
// Byte-addressed E4M3 smem tiles; the 16-byte row padding spreads the MMA
// fragment reads (one uint32 per thread at row lane/4, byte offset 4*(lane%4))
// over 32 banks instead of 4.
constexpr int SMEM_ROW_STRIDE = GROUP_K + 16;

// One M tile per pass. A routed expert can own at most BATCH rows, and decode
// runs BATCH <= 16, so this is a single 16-row MMA row-block in practice.
constexpr int tile_m(int max_rows) {
  return max_rows <= 16 ? 16 : (max_rows <= 32 ? 32 : 64);
}

// Rows the gather list must hold: the M loop always runs whole TILE_M tiles.
constexpr int rows_capacity(int max_rows) {
  return ((max_rows + tile_m(max_rows) - 1) / tile_m(max_rows)) *
         tile_m(max_rows);
}

// Dynamic shared memory: A/B staging tiles (double buffered) plus the
// per-expert (token, slot) gather list and its row count.
constexpr int smem_bytes(int max_rows) {
  return NUM_STAGES * (tile_m(max_rows) + BLOCK_N) * SMEM_ROW_STRIDE +
         (2 * rows_capacity(max_rows) + 1) * (int)sizeof(int) + 16;
}

} // namespace moe_fp8_blockscale

// OUTPUT_SIZE is this TASK's slice of the expert's output rows (the grid may
// split N); ORIG_OUTPUT_SIZE is the full per-expert row count, i.e. the stride
// between experts in weight_fp8 / weight_scale / output.
template <typename T,
          int BATCH_SIZE,
          int NUM_TOPK,
          int NUM_EXPERTS,
          int OUTPUT_SIZE,
          int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool W13_LINEAR>
__device__ __forceinline__ void
    moe_fp8_blockscale_task_impl(void const *__restrict__ input_fp8_ptr,
                                 void const *__restrict__ input_scale_ptr,
                                 void const *__restrict__ weight_fp8_ptr,
                                 void const *__restrict__ weight_scale_ptr,
                                 void const *__restrict__ routing_ptr,
                                 void const *__restrict__ mask_ptr,
                                 void *__restrict__ output_ptr,
                                 int expert_offset,
                                 int expert_stride) {
  using namespace moe_fp8_blockscale;
  constexpr int MMA_K = 32; // K per mma.sync instruction
  constexpr int NUM_K_SUBTILES = GROUP_K / MMA_K;
  constexpr int NUM_K_TILES = REDUCTION_SIZE / GROUP_K;
  constexpr int NUM_N_BLOCKS = OUTPUT_SIZE / BLOCK_N;
  constexpr int SCALE_ROWS_PER_EXPERT = ORIG_OUTPUT_SIZE / BLOCK_N;

  static_assert(REDUCTION_SIZE % GROUP_K == 0,
                "FP8 block scales require K to be a multiple of 128");
  static_assert(OUTPUT_SIZE % BLOCK_N == 0,
                "FP8 block scales require the per-task N to be a multiple of "
                "128 (one checkpoint scale block)");
  static_assert(ORIG_OUTPUT_SIZE % BLOCK_N == 0,
                "the expert's full N must be a whole number of scale blocks");
  static_assert(WORKER_NUM_THREADS == 256,
                "moe_fp8_blockscale_sm100 assumes the SM100 worker's "
                "256-thread block");
  static_assert(smem_bytes(BATCH_SIZE) <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "moe_fp8_blockscale_sm100 exceeds the worker smem budget");

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
  int32_t const *__restrict__ d_routing =
      static_cast<int32_t const *>(routing_ptr);
  int32_t const *__restrict__ d_mask = static_cast<int32_t const *>(mask_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // 16-byte alignment is required by cp.async's 16-byte destination form.
  extern __shared__ __align__(16) uint8_t smem_moe_blockscale[];
  uint8_t *smem_a = smem_moe_blockscale;
  uint8_t *smem_b = smem_a + NUM_STAGES * SMEM_A_BYTES;
  constexpr int MAX_ROWS = TILE_M * NUM_M_TILES;
  int *smem_rows = reinterpret_cast<int *>(smem_b + NUM_STAGES * SMEM_B_BYTES);
  int *smem_tok = smem_rows;                  // gathered token index per A row
  int *smem_slot = smem_rows + MAX_ROWS;      // its topk slot
  int *smem_count = smem_rows + 2 * MAX_ROWS; // how many rows were gathered

  int const tid = threadIdx.x;
  int const lane = lane_id();
  int const warp = tid / NUM_THREADS_PER_WARP;
  int const warp_m = warp / WARPS_N;
  int const warp_n = warp % WARPS_N;
  // Fragment coordinates shared by A, B and D: the operand row/column a lane
  // owns is lane/4, the k byte offset it owns is 4*(lane%4).
  int const frag_row = lane >> 2;
  int const frag_k = (lane & 3) * 4;

  int const num_activated = d_mask[NUM_EXPERTS];

  for (int ae = expert_offset; ae < num_activated; ae += expert_stride) {
    int const expert = d_mask[ae];

    // ---- Gather this expert's routed rows (token, slot) ----
    // routing[expert, token] is the topk slot + 1, 0 when not routed.
    __syncthreads();
    if (tid == 0) {
      int n = 0;
      for (int t = 0; t < BATCH_SIZE; ++t) {
        int const slot = d_routing[(size_t)expert * BATCH_SIZE + t];
        if (slot > 0 && n < MAX_ROWS) {
          smem_tok[n] = t;
          smem_slot[n] = slot - 1;
          ++n;
        }
      }
      // Pad the tail so the A-tile gather can index any row unconditionally.
      for (int r = n; r < MAX_ROWS; ++r) {
        smem_tok[r] = 0;
        smem_slot[r] = 0;
      }
      *smem_count = n;
    }
    __syncthreads();
    int const num_rows = *smem_count;
    if (num_rows == 0) {
      continue;
    }

    // Stage a K tile of A (gathered rows m0..m0+TILE_M) and B (weight rows
    // n0..n0+BLOCK_N of this expert). Rows past num_rows copy 0 bytes
    // (cp.async zero-fills), so they contribute nothing to the MMA.
    auto load_k_tile = [&](int stage, int m0, int n0, int kt) {
      uint8_t *stage_a = smem_a + stage * SMEM_A_BYTES;
      uint8_t *stage_b = smem_b + stage * SMEM_B_BYTES;
      size_t const k_off = (size_t)kt * GROUP_K;
      for (int c = tid; c < A_CHUNKS; c += WORKER_NUM_THREADS) {
        int const r = c / CHUNKS_PER_ROW;
        int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
        int const gathered = m0 + r;
        bool const valid = gathered < num_rows;
        int const idx = valid ? gathered : 0;
        size_t src_row;
        if constexpr (W13_LINEAR) {
          src_row = (size_t)smem_tok[idx];
        } else {
          src_row = (size_t)smem_tok[idx] * NUM_TOPK + smem_slot[idx];
        }
        load_smem_with_predict<uint8_t, BYTES_PER_CP>(
            stage_a + r * SMEM_ROW_STRIDE + x,
            d_input + src_row * REDUCTION_SIZE + k_off + x,
            valid);
      }
      for (int c = tid; c < B_CHUNKS; c += WORKER_NUM_THREADS) {
        int const r = c / CHUNKS_PER_ROW;
        int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
        size_t const w_row =
            (size_t)expert * ORIG_OUTPUT_SIZE + (size_t)(n0 + r);
        load_smem<uint8_t, BYTES_PER_CP>(stage_b + r * SMEM_ROW_STRIDE + x,
                                         d_weight + w_row * REDUCTION_SIZE +
                                             k_off + x);
      }
    };

    for (int nb = 0; nb < NUM_N_BLOCKS; ++nb) {
      int const n0 = nb * BLOCK_N;
      // The checkpoint block scale for these 128 weight rows, all K tiles.
      float const *b_scale_row =
          d_weight_scale +
          ((size_t)expert * SCALE_ROWS_PER_EXPERT + nb) * NUM_K_TILES;

      for (int mt = 0; mt < NUM_M_TILES; ++mt) {
        int const m0 = mt * TILE_M;
        if (m0 >= num_rows) {
          break;
        }
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
          float const b_scale = b_scale_row[kt];
          float s0 = 0.0f, s1 = 0.0f;
          if (row0 < num_rows) {
            size_t sr;
            if constexpr (W13_LINEAR) {
              sr = (size_t)smem_tok[row0];
            } else {
              sr = (size_t)smem_tok[row0] * NUM_TOPK + smem_slot[row0];
            }
            s0 = d_input_scale[sr * NUM_K_TILES + kt] * b_scale;
          }
          if (row1 < num_rows) {
            size_t sr;
            if constexpr (W13_LINEAR) {
              sr = (size_t)smem_tok[row1];
            } else {
              sr = (size_t)smem_tok[row1] * NUM_TOPK + smem_slot[row1];
            }
            s1 = d_input_scale[sr * NUM_K_TILES + kt] * b_scale;
          }
#pragma unroll
          for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
            acc[ni][0] += partial[ni][0] * s0;
            acc[ni][1] += partial[ni][1] * s0;
            acc[ni][2] += partial[ni][2] * s1;
            acc[ni][3] += partial[ni][3] * s1;
          }
          __syncthreads();
        }

        // Scatter to output[token, slot, n0 + col].
#pragma unroll
        for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
          int const col = n0 + warp_n * N_PER_WARP + ni * 8 + (lane & 3) * 2;
          if (row0 < num_rows) {
            size_t const orow =
                ((size_t)smem_tok[row0] * NUM_TOPK + smem_slot[row0]) *
                ORIG_OUTPUT_SIZE;
            d_output[orow + col] = T(acc[ni][0]);
            d_output[orow + col + 1] = T(acc[ni][1]);
          }
          if (row1 < num_rows) {
            size_t const orow =
                ((size_t)smem_tok[row1] * NUM_TOPK + smem_slot[row1]) *
                ORIG_OUTPUT_SIZE;
            d_output[orow + col] = T(acc[ni][2]);
            d_output[orow + col + 1] = T(acc[ni][3]);
          }
        }
        // The next M tile / N block reuses both staging buffers.
        __syncthreads();
      }
    }
  }
}

} // namespace kernel
