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

// ===========================================================================
// M4-I7: the ferret `moe-fp8-grouped-vllm-beat` winner (workspace3 tag v012,
// c8b5b24, min_ratio 0.801 over 10 configs) integrated behind a compile-time
// dispatcher, exactly as M4-I2 did for the dense sibling.
//
// The pre-M4-I7 body is preserved BYTE-FOR-BYTE as
// `kernel::golden::moe_fp8_blockscale_task_impl` (sha256 of the frozen region
// is asserted by opt/m4i7/scripts/check_golden.py). It remains the fallback
// for every shape the fast paths do not cover -- in particular PREFILL, where
// BATCH_SIZE is the full `max_num_batched_tokens`.
//
// WHAT THE FAST PATHS CHANGE, and why each is value-neutral:
//
//  (a) WORK-ITEM FLATTENING. The golden body walks `ae` (activated experts,
//      strided by `expert_stride`) and, inside each, all `NUM_N_BLOCKS` column
//      blocks. The fast body walks a single flattened space
//      `wi in [expert_offset, num_activated * NUM_N_BLOCKS)` step
//      `expert_stride`, with `ae = wi / NUM_N_BLOCKS`, `nb = wi % NUM_N_BLOCKS`.
//      Every (expert, block) pair is still covered exactly once for any
//      `expert_offset in [0, expert_stride)`, each pair writes a disjoint set
//      of output elements, and the per-column K accumulation order is
//      untouched -- so the result is BIT-IDENTICAL.
//      This is the load-bearing MPK-dispatch decision, and it is the mirror of
//      M4-I2's (f): `task_register.cc` emits `expert_stride = grid.x = 128`
//      tasks per (layer, stage, N-split) but only `num_activated` of them ever
//      had work, so at bs1 roughly 8-13 of 128 emitted tasks were live and the
//      other ~115 exited immediately. Flattening spreads the SAME work over
//      `num_activated * NUM_N_BLOCKS` of those already-emitted tasks. Unlike
//      raising `moe_n_splits` (M4-I5's width lever, measured x1.11 at bs1 and a
//      regression at bs16), this costs NO extra dead-task dispatch and does not
//      shrink the N tile.
//  (b) A WIDER FETCH. `PATH 1` stages 4 adjacent K tiles per buffer and pulls
//      each weight row as one 512 B `cp.async.bulk`; `PATH 0` is the golden
//      fetch (one K tile per stage, 16 B `cp.async`) with the measured w13-only
//      `#pragma unroll`. `PATH 2` (8 K tiles, 1 KiB rows, TILE_N=64) is built
//      and bit-exact but NOT shipped -- it measured -1.7% at bs1 and -10.5% at
//      bs16 in MPK; the dispatcher below carries the sweep. All three feed the
//      SAME `mma.sync.m16n8k32.e4m3` and apply the SAME per-K-tile fp32 scale
//      product, in ascending K order.
//  (c) A ballot-compaction routing gather on w2 only (bit-identical smem
//      contents: same ascending-token compaction, different writer lane).
//
// PRESERVED-FP32 BLOCK SCALES (the whole reason this file exists) are UNTOUCHED
// on every path: `b_scale_row[kt]` is read straight from the checkpoint's
// `weight_scale_inv` as float32 and multiplied by the float32 activation scale
// once per 128-element K tile, into an FP32 accumulator. No ue8m0 truncation,
// no requantisation, no per-row collapse. A TILE_N=64 work item reads the row
// of its CONTAINING 128-column scale block (`n0 / BLOCK_N`), so the values a
// column sees do not depend on the tiling.
//
// THE PER-TASK N SLICE IS UNCHANGED. `OUTPUT_SIZE % BLOCK_N == 0` still holds
// (static_assert below), so `weight_scale`'s grid split stays the exact
// `dim(1) * 128 == output_size` division `task_register.cc` already asserts --
// none of M4-I2's row-replication machinery is needed here, and the
// integer-division scale-slice hazard cannot be reached. TILE_N=64 subdivides
// only INSIDE a task, below the scale block.
// ===========================================================================


// =========================================================================
// GOLDEN -- the pre-M4-I7 body, FROZEN. Byte-for-byte identical to
// moe_fp8_blockscale_sm100.cuh before this change (region sha256 below,
// re-checked by opt/m4i7/scripts/check_golden.py). NEVER EDIT.
//   region sha256: 298aa9c455f4e7885f9ef86af45259a8448a6f427b9c0b70987f180de12033e3
// =========================================================================
namespace golden {

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

} // namespace golden


// =========================================================================
// FAST -- ferret workspace3 v012 (`git show v012:kernel.cu`, namespace `cand`),
// verbatim apart from the namespace rename recorded in
// opt/m4i7/scripts/gen_header.py and a tightened smem static_assert. Its own
// dispatcher is REPLACED below (see (a) in the header note).
// =========================================================================

namespace moe_fp8_blockscale_fast {

constexpr int GROUP_K = 128;  // scale group along K (checkpoint block size)
constexpr int BLOCK_N = 128;  // scale block along N (checkpoint block size)
constexpr int NUM_STAGES = 2; // double buffering (stage granularity varies)
// Three fetch paths, chosen at RUNTIME per launch (total_work vs #SMs), each
// identified by a compile-time PATH id whose smem footprint the device reads
// back from %dynamic_smem_size:
//  - PATH 0, LEGACY (oversubscribed grids, bs4+): v004's one-k-tile stages +
//    16B cp.async, 41.6KB smem -> 4-5 resident CTAs/SM, which large grids
//    need (SUPER_K=4 measured -20..-45% there; SUPER_K=2 -9..-23%).
//  - PATH 1, BULK-128 (sub-one-wave grids at TILE_N=128: w13_bs2, w2_bs1):
//    stage = 4 adjacent k-tiles, each B row fetched as ONE 512B
//    cp.async.bulk. Lifts the per-CTA fetch ceiling ~11-22% (a007: 128B
//    bulk LOSES to 16B cp.async; 512B wins). ~152KB smem -> 1 CTA/SM,
//    free when the grid fits one wave (and always true in-MPK).
//  - PATH 2, BULK-64 (grids that fit one wave even at TILE_N=64: w13_bs1):
//    64-col work items + 8 k-tiles per stage -> per-CTA weight bytes HALVE
//    (128KB vs 256KB through the ~20GB/s per-CTA pipe, the w13_bs1 binder)
//    while each bulk row grows to 1KB (better per-op efficiency per
//    machine.md's byte-width crossover). ~166KB smem, 1 CTA/SM.
constexpr int path_super_k(int p) { return p == 2 ? 8 : (p == 1 ? 4 : 1); }
constexpr int path_tile_n(int p) { return p == 2 ? 64 : BLOCK_N; }

constexpr int stage_tiles(int p) { return path_super_k(p); }
constexpr int stage_k(int p) { return stage_tiles(p) * GROUP_K; }
constexpr int row_stride(int p) { return stage_k(p) + 16; }

constexpr int tile_m(int max_rows) {
  return max_rows <= 16 ? 16 : (max_rows <= 32 ? 32 : 64);
}

constexpr int rows_capacity(int max_rows) {
  return ((max_rows + tile_m(max_rows) - 1) / tile_m(max_rows)) *
         tile_m(max_rows);
}

constexpr int smem_bytes(int max_rows, int path) {
  return NUM_STAGES * (tile_m(max_rows) + path_tile_n(path)) *
             row_stride(path) +
         (2 * rows_capacity(max_rows) + 1) * (int)sizeof(int) + 16 +
         32; // align slack + NUM_STAGES x 8B mbarrier (bulk paths only)
}

static_assert(smem_bytes(16, 2) > smem_bytes(16, 1) &&
                  smem_bytes(16, 1) > smem_bytes(16, 0),
              "path smem footprints must be strictly ordered so the device "
              "%dynamic_smem_size dispatch is unambiguous");

// K-clamped smem accounting: the impl clamps STAGE_TILES to the whole K
// (see moe_impl_path), so on w2 (K=512, 4 k-tiles) PATH2's 8-tile stage
// really lays out only 4 tiles -> ~84.7KB, not the 166.6KB upper bound.
// Allocating the clamped size admits TWO resident CTAs/SM on w2, which is
// what makes a 2-wave TILE_N=64 grid profitable there (a011: byte-halving
// LOSES at 1 CTA/SM; a012: wins when residency absorbs the second wave).
// On w13 (16 k-tiles) the clamp is a no-op and these equal smem_bytes().
constexpr int stage_tiles_k(int p, int red_k) {
  return stage_tiles(p) < red_k / GROUP_K ? stage_tiles(p) : red_k / GROUP_K;
}
constexpr int smem_bytes_k(int max_rows, int path, int red_k) {
  return NUM_STAGES * (tile_m(max_rows) + path_tile_n(path)) *
             (stage_tiles_k(path, red_k) * GROUP_K + 16) +
         (2 * rows_capacity(max_rows) + 1) * (int)sizeof(int) + 16 + 32;
}
static_assert(smem_bytes_k(16, 2, 2048) == smem_bytes(16, 2) &&
                  smem_bytes_k(16, 1, 512) == smem_bytes(16, 1) &&
                  smem_bytes_k(16, 0, 512) == smem_bytes(16, 0),
              "clamp must be a no-op wherever the stage already fits K");
static_assert(smem_bytes_k(16, 2, 512) > smem_bytes(16, 0) &&
                  smem_bytes_k(16, 2, 512) < smem_bytes(16, 1),
              "w2's clamped PATH2 layout must stay distinguishable from the "
              "other paths' allocations for the dyn-smem dispatch");

} // namespace moe_fp8_blockscale_fast

namespace moe_fp8_blockscale_fast {

// .cg (bypass-L1) variants for read-once streamed rows on the cand fetch
// paths: every A row and every legacy-path B row is consumed exactly once
// from smem, never re-read through L1, so .ca's L1 allocation only evicts
// other data. Golden keeps the original .ca helpers above (frozen).
// CP_ASYNC_CG_L2_HINT lets the L2 prefetch-size hint be swept from the
// nvcc command line (-DCP_ASYNC_CG_L2_HINT='"L2::256B"').
#ifndef CP_ASYNC_CG_L2_HINT
#define CP_ASYNC_CG_L2_HINT "L2::128B"
#endif

template <typename T, int BYTES = 16>
__device__ __forceinline__ void load_smem_cg(T *smem_ptr, T const *gmem_ptr) {
#ifdef CP_ASYNC_SM80_ENABLED
  static_assert(BYTES == 16, "cp.async.cg supports only 16-byte copies");
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global." CP_ASYNC_CG_L2_HINT
               " [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr),
               "n"(BYTES),
               "r"(BYTES));
#endif
}

template <typename T, int BYTES = 16>
__device__ __forceinline__ void
    load_smem_with_predict_cg(T *smem_ptr, T const *gmem_ptr, bool pred) {
#ifdef CP_ASYNC_SM80_ENABLED
  static_assert(BYTES == 16, "cp.async.cg supports only 16-byte copies");
  int src_in_bytes = pred ? BYTES : 0;
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global." CP_ASYNC_CG_L2_HINT
               " [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr),
               "n"(BYTES),
               "r"(src_in_bytes));
#endif
}


// ---- 1-D cp.async.bulk + mbarrier helpers (bulk-path B-tile fetch) ----
__device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t count) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t *bar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *bar,
                                                          uint32_t tx_bytes) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(addr),
      "r"(tx_bytes)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *bar,
                                                     uint32_t phase) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra LAB_WAIT;\n\t"
      "}" ::"r"(addr),
      "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void cp_async_bulk_g2s(void *dst_smem,
                                                  void const *src_gmem,
                                                  uint32_t bytes,
                                                  uint64_t *bar) {
  uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], %2, [%3];" ::"r"(dst_addr),
               "l"(src_gmem),
               "r"(bytes),
               "r"(bar_addr)
               : "memory");
}

} // namespace moe_fp8_blockscale_fast

template <typename T,
          int BATCH_SIZE,
          int NUM_TOPK,
          int NUM_EXPERTS,
          int OUTPUT_SIZE,
          int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool W13_LINEAR,
          int PATH>
// __noinline__: each PATH body compiles as an isolated function, so adding a
// path can never perturb another path's codegen inside the shared dispatch
// kernel (inlining all paths into one body measurably regressed the legacy
// path ~20% on w13). Called once per task invocation -- call cost is noise.
__device__ __noinline__ void
    moe_impl_path(void const *__restrict__ input_fp8_ptr,
                  void const *__restrict__ input_scale_ptr,
                  void const *__restrict__ weight_fp8_ptr,
                  void const *__restrict__ weight_scale_ptr,
                  void const *__restrict__ routing_ptr,
                  void const *__restrict__ mask_ptr,
                  void *__restrict__ output_ptr,
                  int expert_offset,
                  int expert_stride) {
  using namespace moe_fp8_blockscale_fast;
  constexpr bool USE_BULK = PATH >= 1;
  constexpr int MMA_K = 32; // K per mma.sync instruction
  constexpr int NUM_K_SUBTILES = GROUP_K / MMA_K;
  constexpr int NUM_K_TILES = REDUCTION_SIZE / GROUP_K;
  // Clamp the stage depth to the whole K so every (family, PATH) template
  // instantiation is well-formed even where the host never launches it
  // (e.g. PATH 2 on w2's K=512 clamps 8 -> 4 k-tiles/stage); the namespace
  // smem_bytes() is an upper bound on the clamped layout, so the launch
  // allocation always covers the in-kernel offsets.
  constexpr int STAGE_TILES_RAW = stage_tiles(PATH);
  constexpr int STAGE_TILES =
      STAGE_TILES_RAW < NUM_K_TILES ? STAGE_TILES_RAW : NUM_K_TILES;
  constexpr int STAGE_K = STAGE_TILES * GROUP_K; // K bytes per stage row
  constexpr int SMEM_ROW_STRIDE = STAGE_K + 16;
  constexpr int NUM_SUPER_TILES = NUM_K_TILES / STAGE_TILES;
  constexpr int TILE_N = path_tile_n(PATH); // N cols per work item
  constexpr int NUM_N_BLOCKS = OUTPUT_SIZE / TILE_N;
  constexpr int SCALE_ROWS_PER_EXPERT = ORIG_OUTPUT_SIZE / BLOCK_N;

  static_assert(NUM_K_TILES % STAGE_TILES == 0,
                "K must be a whole number of stages");
  static_assert(REDUCTION_SIZE % GROUP_K == 0,
                "FP8 block scales require K to be a multiple of 128");
  static_assert(OUTPUT_SIZE % BLOCK_N == 0,
                "FP8 block scales require the per-task N to be a multiple of "
                "128 (one checkpoint scale block)");
  static_assert(OUTPUT_SIZE % TILE_N == 0,
                "the per-task N must be a whole number of work-item tiles");
  static_assert(BLOCK_N % TILE_N == 0,
                "TILE_N must evenly subdivide one checkpoint scale block so "
                "each work item reads exactly one b_scale row");
  static_assert(ORIG_OUTPUT_SIZE % BLOCK_N == 0,
                "the expert's full N must be a whole number of scale blocks");
  static_assert(WORKER_NUM_THREADS == 256,
                "moe_fp8_blockscale_sm100 assumes the SM100 worker's "
                "256-thread block");
  static_assert(smem_bytes_k(BATCH_SIZE, PATH, REDUCTION_SIZE) <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "moe_fp8_blockscale_sm100 fast path exceeds the worker smem "
                "budget");

  constexpr int TILE_M = tile_m(BATCH_SIZE);
  constexpr int NUM_M_TILES = (BATCH_SIZE + TILE_M - 1) / TILE_M;
  constexpr int NUM_TASK_WARPS = WORKER_NUM_THREADS / NUM_THREADS_PER_WARP;
  constexpr int WARPS_M = TILE_M / 16;
  constexpr int WARPS_N = NUM_TASK_WARPS / WARPS_M;
  constexpr int N_PER_WARP = TILE_N / WARPS_N;
  constexpr int MMA_N_ITERS = N_PER_WARP / 8;
  static_assert(N_PER_WARP % 8 == 0,
                "each warp must cover a whole number of 8-col MMA blocks");

  constexpr int SMEM_A_BYTES = TILE_M * SMEM_ROW_STRIDE;
  constexpr int SMEM_B_BYTES = TILE_N * SMEM_ROW_STRIDE;
  constexpr int BYTES_PER_CP = 16;
  constexpr int CHUNKS_PER_ROW = STAGE_K / BYTES_PER_CP;
  constexpr int A_CHUNKS = TILE_M * CHUNKS_PER_ROW;
  constexpr int B_CHUNKS = TILE_N * CHUNKS_PER_ROW;

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

  extern __shared__ __align__(16) uint8_t smem_moe_blockscale[];
  uint8_t *smem_a = smem_moe_blockscale;
  uint8_t *smem_b = smem_a + NUM_STAGES * SMEM_A_BYTES;
  constexpr int MAX_ROWS = TILE_M * NUM_M_TILES;
  int *smem_rows = reinterpret_cast<int *>(smem_b + NUM_STAGES * SMEM_B_BYTES);
  int *smem_tok = smem_rows;                  // gathered token index per A row
  int *smem_slot = smem_rows + MAX_ROWS;      // its topk slot
  int *smem_count = smem_rows + 2 * MAX_ROWS; // how many rows were gathered
  uint64_t *smem_bar = reinterpret_cast<uint64_t *>(
      (reinterpret_cast<uintptr_t>(smem_count + 1) + 15) & ~(uintptr_t)15);

  int const tid = threadIdx.x;
  int const lane = lane_id();
  int const warp = tid / NUM_THREADS_PER_WARP;
  int const warp_m = warp / WARPS_N;
  int const warp_n = warp % WARPS_N;
  int const frag_row = lane >> 2;
  int const frag_k = (lane & 3) * 4;

  int const num_activated = d_mask[NUM_EXPERTS];
  // Work unit = one (activated expert, 128-col N block) pair, not one whole
  // expert: at N_LIVE 1-16 only ~8-103 experts activate, so a per-expert
  // decomposition leaves most CTAs idle while each active CTA serially walks
  // all NUM_N_BLOCKS. Output columns are disjoint across work items and the
  // per-column K accumulation order is unchanged, so bit-exactness holds.
  int const total_work = num_activated * NUM_N_BLOCKS;

  if constexpr (USE_BULK) {
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < NUM_STAGES; ++s) {
        mbarrier_init(&smem_bar[s], 1);
      }
      fence_proxy_async_shared();
    }
    // The hoisted stage-0 B fetch (below) touches the barriers BEFORE the
    // first work item's gather sync, so consumers must observe the init
    // here, at entry. (Uniform: every thread of the CTA reaches this.)
    __syncthreads();
  }
  uint32_t bar_phase0 = 0, bar_phase1 = 0;

  for (int wi = expert_offset; wi < total_work; wi += expert_stride) {
    int const ae = wi / NUM_N_BLOCKS;
    int const nb = wi % NUM_N_BLOCKS;
    int const expert = d_mask[ae];
    int const n0 = nb * TILE_N;

    // One stage = STAGE_TILES adjacent k-tiles, split in two halves so the
    // gather-INDEPENDENT B (weight) fetch can be issued first:
    //   load_stage_b -- B rows; address depends only on (expert, n0). Bulk
    //     path: one STAGE_K-byte contiguous cp.async.bulk per weight row
    //     (k-tiles are adjacent along K in gmem); legacy path: 16B cp.async
    //     chunks.
    //   load_stage_a -- A rows (few, gathered, predicated, 16B cp.async);
    //     needs smem_tok/smem_slot, so it can only start after the gather.
    auto load_stage_b = [&](int stage, int st) {
      uint8_t *stage_b = smem_b + stage * SMEM_B_BYTES;
      size_t const k_off = (size_t)st * STAGE_K;
      if constexpr (USE_BULK) {
        // tid 0's expect_tx may land after other lanes' complete_tx; the
        // mbarrier tx-count is signed, so the phase still completes only
        // when all BLOCK_N * STAGE_K bytes have arrived.
        if (tid == 0) {
          mbarrier_arrive_expect_tx(&smem_bar[stage],
                                    (uint32_t)(TILE_N * STAGE_K));
        }
        for (int r = tid; r < TILE_N; r += WORKER_NUM_THREADS) {
          size_t const w_row =
              (size_t)expert * ORIG_OUTPUT_SIZE + (size_t)(n0 + r);
          cp_async_bulk_g2s(stage_b + r * SMEM_ROW_STRIDE,
                            d_weight + w_row * REDUCTION_SIZE + k_off,
                            (uint32_t)STAGE_K,
                            &smem_bar[stage]);
        }
      } else if constexpr (W13_LINEAR) {
        // w13 (K=2048) only: explicit unroll of the chunk-issue loop is a
        // measured +6.5-7.8% on the PATH0 configs; the SAME pragma on the
        // w2 (K=512) instantiation measurably REGRESSED w2_bs4/8/16 by
        // 6-14%, so w2 keeps the v010 codegen byte-for-byte (branch below).
#pragma unroll
        for (int c = tid; c < B_CHUNKS; c += WORKER_NUM_THREADS) {
          int const r = c / CHUNKS_PER_ROW;
          int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
          size_t const w_row =
              (size_t)expert * ORIG_OUTPUT_SIZE + (size_t)(n0 + r);
          load_smem_cg<uint8_t, BYTES_PER_CP>(stage_b + r * SMEM_ROW_STRIDE +
                                                  x,
                                              d_weight +
                                                  w_row * REDUCTION_SIZE +
                                                  k_off + x);
        }
      } else {
        for (int c = tid; c < B_CHUNKS; c += WORKER_NUM_THREADS) {
          int const r = c / CHUNKS_PER_ROW;
          int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
          size_t const w_row =
              (size_t)expert * ORIG_OUTPUT_SIZE + (size_t)(n0 + r);
          load_smem_cg<uint8_t, BYTES_PER_CP>(stage_b + r * SMEM_ROW_STRIDE +
                                                  x,
                                              d_weight +
                                                  w_row * REDUCTION_SIZE +
                                                  k_off + x);
        }
      }
    };
    auto load_stage_a = [&](int stage, int m0, int st, int nrows) {
      uint8_t *stage_a = smem_a + stage * SMEM_A_BYTES;
      size_t const k_off = (size_t)st * STAGE_K;
      if constexpr (W13_LINEAR) {
        // Unroll gated to w13 for the same reason as load_stage_b above.
#pragma unroll
        for (int c = tid; c < A_CHUNKS; c += WORKER_NUM_THREADS) {
          int const r = c / CHUNKS_PER_ROW;
          int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
          int const gathered = m0 + r;
          bool const valid = gathered < nrows;
          int const idx = valid ? gathered : 0;
          size_t const src_row = (size_t)smem_tok[idx];
          load_smem_with_predict_cg<uint8_t, BYTES_PER_CP>(
              stage_a + r * SMEM_ROW_STRIDE + x,
              d_input + src_row * REDUCTION_SIZE + k_off + x,
              valid);
        }
      } else {
        for (int c = tid; c < A_CHUNKS; c += WORKER_NUM_THREADS) {
          int const r = c / CHUNKS_PER_ROW;
          int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
          int const gathered = m0 + r;
          bool const valid = gathered < nrows;
          int const idx = valid ? gathered : 0;
          size_t const src_row =
              (size_t)smem_tok[idx] * NUM_TOPK + smem_slot[idx];
          load_smem_with_predict_cg<uint8_t, BYTES_PER_CP>(
              stage_a + r * SMEM_ROW_STRIDE + x,
              d_input + src_row * REDUCTION_SIZE + k_off + x,
              valid);
        }
      }
    };

    // HOIST: start the stage-0 B fetch's DRAM round trip before (and
    // overlapping) the routing gather's own DRAM round trip. The two are
    // independent, and v008 chained them serially, putting the gather
    // latency on every work item's critical path (the diagnosed w2 binder:
    // ncu shows w2 latency-bound in gather/routing, not fetch-bound).
    // Writing stage-0 B smem here is safe: the previous work item's last
    // compute reads of it drained before that item's post-compute
    // __syncthreads.
    load_stage_b(0, 0);
    if constexpr (!USE_BULK) {
      cp_async_fence(); // own commit group; drained by wait<1>/wait<0> below
    }

    // ---- Gather this expert's routed rows (token, slot) ----
    // Two compile-time gather variants (bit-identical smem contents either
    // way -- same ascending-t compaction, only which thread writes changed,
    // so the bit-exact gate is unaffected):
    //  * w2 (!W13_LINEAR, K=512): warp-parallel ballot compaction. The short
    //    K makes the old tid==0 serial 16-iteration walk a measurable
    //    fraction of each work item (profiler: w2_bs1 latency-bound in
    //    gather/routing); ballot cuts it to one coalesced 64B load. Measured
    //    -3..-6% on every w2 config.
    //  * w13 (K=2048): keep the serial tid==0 walk. The ballot variant
    //    measurably REGRESSED every w13 config (+2..+10%, reproducible on an
    //    idle GPU) -- the gather is noise next to the 256KB weight fetch
    //    there, and the extra live registers/convergence in the main loop's
    //    codegen cost more than the walk. Separate template instantiation =>
    //    w13 keeps its exact prior codegen.
    // (No barrier here: the previous item's readers of smem_tok/smem_slot/
    // smem_count all drained at its end-of-m-tile __syncthreads (or the
    // num_rows==0 path's own barrier), and on the first item nothing has
    // read them yet -- the gather warp may start its d_routing DRAM read
    // immediately instead of waiting for the whole CTA.)
    if constexpr (W13_LINEAR) {
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
        for (int r = n; r < MAX_ROWS; ++r) {
          smem_tok[r] = 0;
          smem_slot[r] = 0;
        }
        *smem_count = n;
      }
    } else {
      if (warp == 0) {
        static_assert(BATCH_SIZE <= NUM_THREADS_PER_WARP,
                      "ballot gather assumes one token per lane");
        int slot = 0;
        if (lane < BATCH_SIZE) {
          slot = d_routing[(size_t)expert * BATCH_SIZE + lane];
        }
        uint32_t const routed = __ballot_sync(0xffffffffu, slot > 0);
        int const pos = __popc(routed & ((1u << lane) - 1u));
        if (slot > 0) {
          smem_tok[pos] = lane;
          smem_slot[pos] = slot - 1;
        }
        int const n = __popc(routed);
        for (int r = n + lane; r < MAX_ROWS; r += NUM_THREADS_PER_WARP) {
          smem_tok[r] = 0;
          smem_slot[r] = 0;
        }
        if (lane == 0) {
          *smem_count = n;
        }
      }
    }
    __syncthreads();
    int const num_rows = *smem_count;
    if (num_rows == 0) {
      // Defensive only (an activated expert always has >=1 routed row).
      // The hoisted stage-0 B fetch was already issued: consume its
      // completion so phase/group accounting stays aligned for the next
      // work item.
      if constexpr (USE_BULK) {
        mbarrier_wait_parity(&smem_bar[0], bar_phase0);
        bar_phase0 ^= 1u;
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();
      continue;
    }

    {
      // One checkpoint scale block spans BLOCK_N=128 output cols; a TILE_N=64
      // work item reads the row of its CONTAINING block (n0/BLOCK_N == nb/2),
      // identical scale values per column as the 128-wide decomposition.
      float const *b_scale_row =
          d_weight_scale +
          ((size_t)expert * SCALE_ROWS_PER_EXPERT + (n0 / BLOCK_N)) *
              NUM_K_TILES;

      for (int mt = 0; mt < NUM_M_TILES; ++mt) {
        int const m0 = mt * TILE_M;
        if (m0 >= num_rows) {
          break;
        }
        int const row0 = m0 + warp_m * 16 + frag_row;
        int const row1 = row0 + 8;

        float acc[MMA_N_ITERS][4];
#pragma unroll
        for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            acc[ni][i] = 0.0f;
          }
        }

        if (mt > 0) {
          // Stage-0 B was clobbered by the previous m-tile's K loop (only
          // reachable when NUM_M_TILES > 1, never for BATCH_SIZE=16):
          // re-issue it. mt==0 uses the hoisted pre-gather fetch.
          load_stage_b(0, 0);
        }
        load_stage_a(0, m0, 0, num_rows);
        cp_async_fence();

        for (int st = 0; st < NUM_SUPER_TILES; ++st) {
          if (st + 1 < NUM_SUPER_TILES) {
            load_stage_b((st + 1) & 1, st + 1);
            load_stage_a((st + 1) & 1, m0, st + 1, num_rows);
            cp_async_fence();
            cp_async_wait<1>(); // stage st's cp.async ops (A; legacy also B)
          } else {
            cp_async_wait<0>();
          }
          if constexpr (USE_BULK) {
            // B tile of stage st (bulk path): parity flips once per use of
            // the stage's mbarrier; every thread tracks the same uniform
            // sequence.
            if (st & 1) {
              mbarrier_wait_parity(&smem_bar[1], bar_phase1);
              bar_phase1 ^= 1u;
            } else {
              mbarrier_wait_parity(&smem_bar[0], bar_phase0);
              bar_phase0 ^= 1u;
            }
          }
          __syncthreads();

          uint8_t const *stage_a = smem_a + (st & 1) * SMEM_A_BYTES;
          uint8_t const *stage_b = smem_b + (st & 1) * SMEM_B_BYTES;

          // Per-kt scale promotion runs in ascending kt order inside the
          // stage -- the fp32 accumulate sequence is IDENTICAL to the
          // golden kernel's one-k-tile-per-stage loop, so bit-exactness
          // holds.
#pragma unroll
          for (int kl = 0; kl < STAGE_TILES; ++kl) {
            int const kt = st * STAGE_TILES + kl;

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
              int const k_byte = kl * GROUP_K + ks * MMA_K + frag_k;
              uint32_t a_frag[4];
              a_frag[0] = *reinterpret_cast<uint32_t const *>(
                  stage_a + (warp_m * 16 + frag_row) * SMEM_ROW_STRIDE +
                  k_byte);
              a_frag[1] = *reinterpret_cast<uint32_t const *>(
                  stage_a + (warp_m * 16 + frag_row + 8) * SMEM_ROW_STRIDE +
                  k_byte);
              a_frag[2] = *reinterpret_cast<uint32_t const *>(
                  stage_a + (warp_m * 16 + frag_row) * SMEM_ROW_STRIDE +
                  k_byte + 16);
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
          }
          // Interior stages only: protects buffer (st&1) from the NEXT
          // iteration's load of stage st+2. After the LAST stage the next
          // smem write (next m-tile / next item, always stage-0 buffers)
          // can only be issued by a thread that passed the end-of-m-tile
          // __syncthreads below, which itself requires every thread to have
          // finished this stage's compute -- so the last-stage barrier is
          // redundant and only adds a full-CTA sync to the critical path.
          if (st + 1 < NUM_SUPER_TILES) {
            __syncthreads();
          }
        }

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
        __syncthreads();
      }
    }
  }

  if constexpr (USE_BULK) {
    // MPK folds this task into a persistent megakernel that reuses smem
    // across tasks: invalidate the mbarriers so the next task's init is
    // well-defined. The barrier keeps stragglers still polling try_wait on
    // the final phase from racing the inval.
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < NUM_STAGES; ++s) {
        mbarrier_inval(&smem_bar[s]);
      }
    }
  }
}


namespace moe_fp8_blockscale_fast {

// ---- admissibility, all compile-time -----------------------------------
// A PATH is admissible for an instantiation when every static_assert inside
// moe_impl_path would hold for it. The dispatcher below static_asserts that
// SOME path is reachable for every instantiation it claims, so an inadmissible
// shape fails the BUILD rather than a numeric check at run time (M4-I2's
// lesson: do not lean on `if constexpr` branch-discarding to suppress a
// static_assert -- nvcc may still parse the discarded branch).
constexpr int num_k_tiles(int red_k) {
  return red_k / GROUP_K;
}
constexpr int warps_m(int max_rows) {
  return tile_m(max_rows) / 16;
}
constexpr int warps_n(int max_rows) {
  return (WORKER_NUM_THREADS / NUM_THREADS_PER_WARP) / warps_m(max_rows);
}

constexpr bool path_admissible(int max_rows,
                               int path,
                               int out_n,
                               int red_k,
                               bool w13) {
  return red_k % GROUP_K == 0 && out_n % BLOCK_N == 0 &&
         out_n % path_tile_n(path) == 0 &&
         BLOCK_N % path_tile_n(path) == 0 &&
         num_k_tiles(red_k) % stage_tiles_k(path, red_k) == 0 &&
         (path_tile_n(path) / warps_n(max_rows)) % 8 == 0 &&
         smem_bytes_k(max_rows, path, red_k) <=
             mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE &&
         // the w2 ballot gather puts one token per lane
         (w13 || max_rows <= NUM_THREADS_PER_WARP);
}

// M4-I2's lesson, applied structurally: nvcc is not guaranteed to leave a
// DISCARDED `if constexpr` branch uninstantiated (it kept parsing one while
// recovering from an earlier diagnostic and surfaced its static_assert). So the
// PATH template argument is SANITISED here rather than guarded at the call
// site -- a spuriously instantiated branch compiles as PATH 0, and the safety
// rests on the dispatcher's reachability static_assert, which no compiler's
// instantiation eagerness can affect.
constexpr int safe_path(int max_rows,
                        int p,
                        int out_n,
                        int red_k,
                        bool w13) {
  return path_admissible(max_rows, p, out_n, red_k, w13) ? p : 0;
}

// The ferret run validated exactly BATCH_SIZE == 16 with 1..16 live rows -- the
// shipped decode geometry (`max_num_batched_tokens = 16`). PREFILL instantiates
// the same template with the full batched-token count; it stays on the golden
// path, which is the proven one (goal.md AC-5 depends on prefill).
constexpr int FAST_MAX_BATCH = 16;

constexpr bool fast_path_ok(int max_rows, int out_n, int red_k, bool w13) {
#ifdef MPK_MOE_BLOCKSCALE_BASELINE
  return false; // A/B arm A: pin the pre-M4-I7 kernel from one tree
#else
  return max_rows <= FAST_MAX_BATCH &&
         (path_admissible(max_rows, 0, out_n, red_k, w13) ||
          path_admissible(max_rows, 1, out_n, red_k, w13) ||
          path_admissible(max_rows, 2, out_n, red_k, w13));
#endif
}

// The GOLDEN path needs whole 128-column blocks and it is the only fallback, so
// a shape the fast paths reject must be one the golden path can run.
constexpr bool golden_can_run(int out_n, int red_k) {
  return out_n % BLOCK_N == 0 && red_k % GROUP_K == 0;
}

// The dynamic smem a launcher must provide for this instantiation: the max over
// the golden layout and every admissible fast layout. In the megakernel every
// worker already owns MAX_DYNAMIC_SHARED_MEMORY_SIZE, so this matters only to
// out-of-megakernel harnesses (the pybind test wrapper). The dispatcher ALSO
// checks %dynamic_smem_size at run time and degrades to the narrow path rather
// than reading past a short allocation, so getting this wrong is slow, not
// silently wrong.
constexpr int launch_smem_bytes(int max_rows, int out_n, int red_k, bool w13) {
  int need = golden::moe_fp8_blockscale::smem_bytes(max_rows);
  for (int p = 0; p <= 2; ++p) {
    if (fast_path_ok(max_rows, out_n, red_k, w13) &&
        path_admissible(max_rows, p, out_n, red_k, w13)) {
      int const b = smem_bytes_k(max_rows, p, red_k);
      need = b > need ? b : need;
    }
  }
  return need;
}

} // namespace moe_fp8_blockscale_fast

// The MPK-facing entry point: a compile-time dispatcher over `fast_path_ok`,
// then a runtime choice among the admissible fetch paths.
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
  using namespace moe_fp8_blockscale_fast;
  constexpr bool FAST =
      fast_path_ok(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
  static_assert(FAST || golden_can_run(OUTPUT_SIZE, REDUCTION_SIZE),
                "no admissible path for this instantiation: the fast paths "
                "rejected it and the golden path needs whole 128-column scale "
                "blocks with K a multiple of 128");

  if constexpr (!FAST) {
    golden::moe_fp8_blockscale_task_impl<T,
                                        BATCH_SIZE,
                                        NUM_TOPK,
                                        NUM_EXPERTS,
                                        OUTPUT_SIZE,
                                        ORIG_OUTPUT_SIZE,
                                        REDUCTION_SIZE,
                                        W13_LINEAR>(input_fp8_ptr,
                                                    input_scale_ptr,
                                                    weight_fp8_ptr,
                                                    weight_scale_ptr,
                                                    routing_ptr,
                                                    mask_ptr,
                                                    output_ptr,
                                                    expert_offset,
                                                    expert_stride);
  } else {
    constexpr bool OK0 =
        path_admissible(BATCH_SIZE, 0, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
    constexpr bool OK1 =
        path_admissible(BATCH_SIZE, 1, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
    constexpr bool OK2 =
        path_admissible(BATCH_SIZE, 2, OUTPUT_SIZE, REDUCTION_SIZE, W13_LINEAR);
    static_assert(OK0, "the legacy fetch path must always be admissible when "
                       "the fast body runs -- it is the in-body fallback");

#define MPK_MOE_RUN_PATH(P)                                                    \
  moe_impl_path<T,                                                             \
                BATCH_SIZE,                                                    \
                NUM_TOPK,                                                      \
                NUM_EXPERTS,                                                   \
                OUTPUT_SIZE,                                                   \
                ORIG_OUTPUT_SIZE,                                              \
                REDUCTION_SIZE,                                                \
                W13_LINEAR,                                                    \
                safe_path(BATCH_SIZE,                                          \
                          (P),                                                 \
                          OUTPUT_SIZE,                                         \
                          REDUCTION_SIZE,                                      \
                          W13_LINEAR)>(input_fp8_ptr,                          \
                     input_scale_ptr,                                          \
                     weight_fp8_ptr,                                           \
                     weight_scale_ptr,                                         \
                     routing_ptr,                                              \
                     mask_ptr,                                                 \
                     output_ptr,                                               \
                     expert_offset,                                            \
                     expert_stride)

#if defined(MPK_MOE_PATH_POLICY)
    // Sweep/diagnostic pin. Falls back to the legacy path where the pinned one
    // is inadmissible, so every instantiation still builds.
    constexpr int PIN = MPK_MOE_PATH_POLICY;
    static_assert(PIN >= 0 && PIN <= 2, "MPK_MOE_PATH_POLICY must be 0, 1 or 2");
    if constexpr (PIN == 2 && OK2) {
      MPK_MOE_RUN_PATH(2);
    } else if constexpr (PIN == 1 && OK1) {
      MPK_MOE_RUN_PATH(1);
    } else {
      MPK_MOE_RUN_PATH(0);
    }
#else
    // ---- the shipped rule, and it is MEASURED, not reasoned ---------------
    // In the ferret harness one CTA ran one work item, so "does the grid fit one
    // wave" (work items vs %nsmid) decided whether the wide-smem paths'
    // 1-CTA/SM residency was free. IN MPK THAT DENOMINATOR IS WRONG: there is
    // exactly one persistent worker per SM, each owning the WHOLE dynamic smem
    // budget, so residency is fixed at 1 CTA/SM whichever path runs and the wide
    // layouts cost nothing. Which left the question open, so it was swept with
    // MPK_MOE_PATH_POLICY -- three pinned arms, 3 reps, bs1 and bs16, arms
    // interleaved in one GPU claim (opt/m4i7/tables/path_policy.txt):
    //
    //            bs1 median ms     bs16 median ms
    //   PATH 0        826.2             3452.0
    //   PATH 1        823.9             3371.2     <-- best at both
    //   PATH 2        837.8             3724.1     <-- loses, badly at bs16
    //
    // TWO RESULTS. (1) PATH 1 does dominate PATH 0 in MPK, as predicted: the
    // ferret run only ever preferred PATH 0 to protect 4-5 CTAs/SM of residency,
    // which does not exist here. The margin is +0.3% at bs1 and +2.4% at bs16.
    // (2) PATH 2 (TILE_N=64) LOSES at every measured batch size, -1.7% at bs1
    // and -10.5% at bs16. Its whole premise was halving per-CTA weight bytes to
    // recruit a second wave of CTAs; in MPK the task count is fixed by the graph
    // and the flattened work space already saturates it, so halving the tile only
    // doubles the per-item gathers, A re-fetches and epilogues for the same MMAs.
    //
    // So the rule is simply: PATH 1 when admissible, else PATH 0. No runtime
    // mask read, no branch on the task's critical path. PATH 2 stays REACHABLE
    // (and bit-exact -- Gate 1 covers it) only through MPK_MOE_PATH_POLICY=2, so
    // the sweep can be repeated if the geometry changes; e.g. a larger
    // moe_n_splits would shrink OUTPUT_SIZE and change the trade.
    //
    // FAIL-CLOSED against a SHORT allocation. In the megakernel this is always
    // MAX_DYNAMIC_SHARED_MEMORY_SIZE, but a standalone launcher can hand the
    // task less (the pybind wrapper used to size its launch off the golden
    // layout alone), and PATH 1's 152 KiB layout on a 56 KiB allocation would
    // write past the arena. Reading %dynamic_smem_size makes the device follow
    // the allocation exactly -- CTA-uniform, one register read, and it degrades
    // to PATH 0 rather than corrupting memory.
    uint32_t dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if constexpr (OK1) {
      if (dyn_smem >= (uint32_t)smem_bytes_k(BATCH_SIZE, 1, REDUCTION_SIZE)) {
        MPK_MOE_RUN_PATH(1);
        return;
      }
    }
    MPK_MOE_RUN_PATH(0);
#endif
#undef MPK_MOE_RUN_PATH
  }
}


} // namespace kernel
