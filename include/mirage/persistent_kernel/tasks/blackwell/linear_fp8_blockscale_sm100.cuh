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
//
// ===========================================================================
// TWO PATHS: `_golden` (the original, above) and `_fast` (the ferret winner).
// ===========================================================================
//
// `linear_fp8_blockscale_task_impl` is now a COMPILE-TIME DISPATCHER. It calls
// `linear_fp8_blockscale_task_impl_fast` -- ported from the ferret
// `dense-fp8-blockscale` winner, workspace4 tag v011 (fed45b8) -- whenever that
// path's preconditions hold, and otherwise the ORIGINAL
// `linear_fp8_blockscale_task_impl_golden`, whose body is preserved
// BYTE-FOR-BYTE (only the function name changed).
//
// The ferret loop gated every iteration on a host-fp32 reference with the
// IDENTICAL per-128-K-tile promotion order and required max-abs-diff == 0
// (bit-exact) on all 30 shape/M configs, so the fast path is bit-exact against
// the golden path BY CONSTRUCTION; every transformation below moves the same
// bytes through the same fp32 expressions in the same order. v011 measured
// min_ratio 1.011 against vLLM's `cutlass_3x_gemm_fp8_blockwise` over the 30
// configs (worst outproj_M2 101.1%, best gdnz_M4 128.1%).
//
// WHAT THE FAST PATH CHANGES, and why each is value-neutral:
//
//   (a) The activation tile is staged ONCE for the WHOLE K extent (TILE_M x K,
//       <= 64 KiB at K=4096) instead of once per 128-wide K tile. Same bytes,
//       read once instead of NUM_K_TILES times.
//   (b) The fp32 scale panels ([BATCH_SIZE, K/128] activation + this task's
//       [K/128] weight slice) and the bf16 residual tile are staged into shared
//       memory in A's commit group. The golden path issued DEPENDENT GLOBAL
//       loads for the scales inside the K loop, squarely on the inter-tile
//       critical path and cold in MPK steady state. Identical values, identical
//       multiply order.
//   (c) A deep per-warp cp.async B ring (up to full prefetch of the task's
//       whole B extent) replaces the 2-stage double buffer. The ~31 GB/s
//       per-CTA streaming "cap" was RING-DEPTH-limited, not hardware.
//   (d) B staging is PER-WARP: each warp copies only the N_PER_WARP-row slice
//       it alone consumes, so it waits on its own commit groups and
//       `__syncwarp()`s. The K loop takes NO block-wide barrier at all.
//   (e) `ldmatrix` replaces four discrete LDS.32 per fragment; K tiles are
//       processed in PAIRS with their MMA issue streams interleaved and their
//       fragment registers double-buffered. Promotion stays strictly ordered
//       (kt2 then kt2+1), so the across-tile accumulate order is untouched.
//   (f) OUTPUT_SIZE may be a SUB-multiple of the 128-row scale block
//       (16/32/64): the projection is split into MORE tasks each streaming
//       FEWER weight rows. This is the load-bearing MPK-dispatch decision --
//       see below.
//
// (f) IS THE STRUCTURAL DIFFERENCE, and it is a BUILDER-side change. MPK runs
// ONE persistent worker CTA per SM and one task at a time, so a projection
// dispatched as N/128 tasks occupies only N/128 of the 148 SMs: gdn out_proj
// and attn o_proj (N=2048) reach 16. Slicing to OUTPUT_SIZE 16 makes that 128.
// The measured effect is large: the ferret run's score went 0.727 (slice 128)
// -> 0.862 the moment per-shape slicing landed, and the deep rings that took it
// to 1.011 only FIT in the worker's shared-memory budget at a narrow slice.
//
// A sub-block slice lies inside exactly ONE checkpoint scale block, so the
// caller must pass the CONTAINING block-row's `weight_scale` pointer. MPK
// splits an input by integer division (`runtime.cc`: block_size =
// dim[input_map.x] / grid_dim.x), which would silently return 0 for a
// [N/128, K/128] scale under a grid of N/16 -- every task would then read scale
// row 0. The builder therefore attaches the scale ROW-REPLICATED to one row per
// task (`Qwen35Builder._fp8_block_scale`), which is bit-identical data and lets
// the ordinary grid split hand each task its containing row;
// `linear_fp8_blockscale_layer` asserts the shape so a mis-wired caller fails
// closed instead of computing with the wrong scales.

namespace kernel {

namespace linear_fp8_blockscale {

constexpr int GROUP_K = 128;  // scale group along K (checkpoint block size)
constexpr int BLOCK_N = 128;  // scale block along N (checkpoint block size)
constexpr int NUM_STAGES = 2; // cp.async double buffering (golden path)
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

// ---------------------------------------------------------------------------
// Fast-path (ferret v011) policy. Every function below is a verbatim port.
// ---------------------------------------------------------------------------

// Per-task N extent actually tiled per pass. OUTPUT_SIZE may be a SUB-multiple
// of the 128-row scale block (16/32/64): narrow slices launch MORE tasks each
// streaming FEWER weight rows. The task's slice must lie inside ONE checkpoint
// scale block (the caller passes the containing block-row's weight_scale ptr).
constexpr int task_block_n(int output_size) {
  return output_size < BLOCK_N ? output_size : BLOCK_N;
}

// cp.async B-tile ring depth, per K extent. Prefetch depth is stages-1 and
// must stay <= the K-tile count (down: K=512 -> 4 tiles), or the
// wait<stages-2> group arithmetic under-waits and silently corrupts.
// DEEP rings at narrow slices: the long-standing ~31GB/s per-CTA streaming
// "cap" was RING-DEPTH-LIMITED, not hardware -- BW*latency ~= 25KB in flight
// is exactly what a 6-deep ring holds at slice 32/64 (6 x 4.6/9.2KB); the
// depth-7 null result predates N-slicing (slice 128, already >25KB in flight,
// genuinely capped). K=2048 (16 tiles) gets depth 17 = the whole B extent
// prefetched at the prologue (in-loop refills all empty); K=4096 gets 29, the
// max fitting the worker budget: 29*4608(B) + 65792(A) + 1024(res) +
// 2176(scales) = 202624 <= MAX_DYNAMIC_SHARED_MEMORY_SIZE (205824).
constexpr int num_stages(int reduction_size, int output_size) {
  // Full prefetch (tiles+1 stages) wherever smem allows: K=512 -> 5,
  // K=2048 -> 17, K=4096 -> 33 at slice 16 (144.5KB) but capped at 29 at
  // slice 32 (202624B, the budget max). Wide slices (128-row tiles) keep the
  // shallow depth-6 ring -- a deep ring at 18432B/tile blows the budget.
  return task_block_n(output_size) >= 128
             ? ((reduction_size / 128) >= 8 ? 6 : 4)
             : (reduction_size / 128) >= 32
                   ? (output_size <= 16 ? 33 : 29)
                   : ((reduction_size / 128) >= 8 ? 17 : 5);
}

// The activation tile is staged ONCE for the whole K extent (TILE_M x K is
// at most 64KB at K=4096) instead of per K tile; K+16 keeps the same
// 16-bytes-per-128 padding phase as SMEM_ROW_STRIDE, so fragment reads stay
// bank-conflict-free.
constexpr int a_row_stride(int reduction_size) {
  return reduction_size + 16;
}

// Fast-path dynamic shared memory: whole-K A tile + B staging ring + bf16
// residual staging (reserved unconditionally; loaded only when WITH_RESIDUAL so
// the epilogue never serializes a global read) + fp32 scale staging
// ([batch, K/128] activation panel + [K/128] weight slice), so the K loop's
// promotion never reads cold global memory.
constexpr int smem_bytes(int batch_size, int reduction_size, int output_size) {
  return tile_m(batch_size) * a_row_stride(reduction_size) +
         num_stages(reduction_size, output_size) * task_block_n(output_size) *
             SMEM_ROW_STRIDE +
         tile_m(batch_size) * task_block_n(output_size) * 2 +
         (batch_size + 1) * (reduction_size / GROUP_K) * 4;
}

// Whether the v011 fast path is admissible for this instantiation. Every clause
// is a static_assert inside `_fast`; evaluating them here lets the dispatcher
// discard the call with `if constexpr` so an inadmissible shape compiles the
// golden path instead of failing the build.
constexpr bool fast_path_ok(int batch_size,
                            int output_size,
                            int reduction_size) {
#ifdef MPK_FP8_DENSE_BASELINE
  // M4-I2's A/B arm: pin the pre-M4-I2 golden path so both arms come from one
  // tree. The builder pins the matching slice-128 grid off the same env var.
  return false;
#else
  return
      // one 16-row MMA row-block per warp: the per-warp B ring assumes every
      // warp owns a disjoint N slice (WARPS_M == 1).
      tile_m(batch_size) == 16 &&
      // paired K-tile interleave needs an even tile count; the scale staging
      // needs K/128 float32s to pack into whole 16-byte chunks.
      reduction_size % GROUP_K == 0 && (reduction_size / GROUP_K) % 2 == 0 &&
      ((reduction_size / GROUP_K) * 4) % 16 == 0 &&
      // N is whole 128-row scale blocks, or a >= 16 sub-multiple of one.
      (output_size % BLOCK_N == 0 ||
       (BLOCK_N % output_size == 0 && output_size >= 16)) &&
      // each active warp needs >= one n8 MMA, and B fragments load via paired
      // ldmatrix.x4 (even MMA_N_ITERS) or a single ldmatrix.x2 (== 1).
      task_block_n(output_size) >= 16 &&
      // ring arithmetic + shared-memory budget.
      num_stages(reduction_size, output_size) >= 4 &&
      num_stages(reduction_size, output_size) - 1 <=
          reduction_size / GROUP_K &&
      smem_bytes(batch_size, reduction_size, output_size) <=
          mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
#endif
}

// OUTPUT_SIZE sanitised for the golden path, which requires whole 128-row scale
// blocks. A NO-OP on every instantiation where the golden path is actually
// reachable -- the dispatcher static_asserts that -- and it exists only so that
// a toolchain which instantiates a DISCARDED `if constexpr` branch (nvcc does,
// under some flag combinations, while recovering from an earlier diagnostic)
// still compiles instead of reporting the golden path's assert for a slice the
// golden path was never going to run. Depending on branch discarding to suppress
// a static_assert is not portable; the reachability assert is what carries the
// safety, and it does not depend on discarding at all.
constexpr int golden_output_size(int output_size) {
  return output_size % BLOCK_N == 0 ? output_size : BLOCK_N;
}

// Dynamic shared memory the DISPATCHED task actually needs. External launchers
// (the kernel-wrapper tests) must size their arena with this, not either path's
// own figure. MPK itself always gives a worker the full arena.
constexpr int task_smem_bytes(int batch_size,
                              int reduction_size,
                              int output_size) {
  return fast_path_ok(batch_size, output_size, reduction_size)
             ? smem_bytes(batch_size, reduction_size, output_size)
             : smem_bytes(batch_size);
}

// cp.async with the .cg (L1-bypass) cache hint, vendored here rather than added
// to tasks/common/copy_sm80.cuh so no other task's code generation moves: 16B
// cp.async.ca allocates L1 lines and chokes on L1 MSHRs, .cg streams via L2
// only. Identical bytes land in shared memory either way.
template <int BYTES = 16>
__device__ __forceinline__ void load_smem_cg(void *smem_ptr,
                                             void const *gmem_ptr) {
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "cp.async only supports 4, 8, or 16 bytes");
  uint32_t const smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(BYTES),
               "r"(BYTES));
}

template <int BYTES = 16>
__device__ __forceinline__ void
    load_smem_cg_predict(void *smem_ptr, void const *gmem_ptr, bool pred) {
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "cp.async only supports 4, 8, or 16 bytes");
  int const src_in_bytes = pred ? BYTES : 0;
  uint32_t const smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(BYTES),
               "r"(src_in_bytes));
}

// One ldmatrix.x4: four 8-row x 16-byte matrices, lanes 0-7/8-15/16-23/24-31
// supply the row addresses of matrices 0..3. Register r_i of lane l receives
// the 4 bytes at (row l/4, byte 4*(l%4)) of matrix i -- exactly the
// mma.m16n8k32 fragment layout, so this replaces four discrete LDS.32 per
// fragment with one instruction, moving identical bytes (bit-exact). Same
// instruction as common/copy_sm80.cuh's `ldsm`, restated on a void const* so
// the port stays literally the banked kernel.
__device__ __forceinline__ void ldsm_x4(void const *smem_ptr, uint32_t *R) {
  uint32_t const addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(addr));
}

// One ldmatrix.x2: two 8-row x 16-byte matrices (lanes 0-7 / 8-15 supply the
// row addresses; lanes 16-31's are ignored). Used for the MMA_N_ITERS == 1
// (per-warp n8) B fragment: matrix 0 = the warp's 8 B rows at k byte +0,
// matrix 1 = the same rows at +16 -- exactly b_frag[0]/b_frag[1] of one
// m16n8k32 B operand, same bytes the x4 pair path moves (bit-exact).
__device__ __forceinline__ void ldsm_x2(void const *smem_ptr, uint32_t *R) {
  uint32_t const addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(addr));
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
//
// THE GOLDEN PATH. Body preserved BYTE-FOR-BYTE from the pre-M4-I2 kernel; the
// only edit is the `_golden` suffix on the name. It is what the ferret winner
// was gated bit-exact against, and it remains the path MPK compiles for any
// instantiation `fast_path_ok` rejects (batch > 16, odd K-tile counts, a slice
// whose ring does not fit shared memory).
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE,
          bool WITH_RESIDUAL>
__device__ __forceinline__ void linear_fp8_blockscale_task_impl_golden(
    void const *__restrict__ input_fp8_ptr,
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

// ===========================================================================
// THE FAST PATH -- ferret `dense-fp8-blockscale` winner, workspace4 tag v011
// (fed45b8). Ported verbatim; see the file header for the transformation list
// and the bit-exactness argument. The only edits against the banked kernel are
// mechanical: its vendored `load_smem`/`load_smem_with_predict` (which carry the
// .cg hint the Mirage helpers do not) are called here as `load_smem_cg` /
// `load_smem_cg_predict`.
// ===========================================================================
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE,
          bool WITH_RESIDUAL>
__device__ __forceinline__ void linear_fp8_blockscale_task_impl_fast(
    void const *__restrict__ input_fp8_ptr,
    void const *__restrict__ input_scale_ptr,
    void const *__restrict__ weight_fp8_ptr,
    void const *__restrict__ weight_scale_ptr,
    void const *__restrict__ residual_ptr,
    void *__restrict__ output_ptr) {
  using namespace linear_fp8_blockscale;
  constexpr int MMA_K = 32; // K per mma.sync instruction
  constexpr int NUM_K_SUBTILES = GROUP_K / MMA_K;
  constexpr int NUM_K_TILES = REDUCTION_SIZE / GROUP_K;
  constexpr int TASK_BLOCK_N = task_block_n(OUTPUT_SIZE);
  constexpr int NUM_N_BLOCKS = OUTPUT_SIZE / TASK_BLOCK_N;
  constexpr int NUM_STAGES = num_stages(REDUCTION_SIZE, OUTPUT_SIZE);

  static_assert(REDUCTION_SIZE % GROUP_K == 0,
                "FP8 block scales require K to be a multiple of 128");
  static_assert(OUTPUT_SIZE % BLOCK_N == 0 ||
                    (BLOCK_N % OUTPUT_SIZE == 0 && OUTPUT_SIZE >= 16),
                "per-task N must be a multiple of the 128-row scale block, "
                "or a sub-multiple >= 16 lying inside ONE block (the caller "
                "then passes the containing block-row's weight_scale ptr)");
  static_assert(WORKER_NUM_THREADS == 256,
                "linear_fp8_blockscale_sm100 assumes the SM100 worker's "
                "256-thread block");
  static_assert(smem_bytes(BATCH_SIZE, REDUCTION_SIZE, OUTPUT_SIZE) <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "linear_fp8_blockscale_sm100 exceeds the worker smem budget");
  static_assert(NUM_STAGES - 1 <= REDUCTION_SIZE / GROUP_K,
                "prefetch depth must not exceed the K-tile count");

  constexpr int TILE_M = tile_m(BATCH_SIZE);
  constexpr int NUM_M_TILES = (BATCH_SIZE + TILE_M - 1) / TILE_M;
  constexpr int NUM_TASK_WARPS = WORKER_NUM_THREADS / NUM_THREADS_PER_WARP;
  constexpr int WARPS_M = TILE_M / 16;
  // Narrow slices (TASK_BLOCK_N < 128) run with fewer ACTIVE compute warps
  // (each warp owns an n8-or-wider disjoint B row slice); the remaining
  // warps still help with the block-cooperative A/scale prologue, then idle
  // at the epilogue barrier. Slope scales with the per-CTA B bytes, which is
  // the point of slicing.
  constexpr int WARPS_N = (NUM_TASK_WARPS / WARPS_M) < (TASK_BLOCK_N / 8)
                              ? (NUM_TASK_WARPS / WARPS_M)
                              : (TASK_BLOCK_N / 8);
  constexpr int ACTIVE_WARPS = WARPS_M * WARPS_N;
  constexpr int N_PER_WARP = TASK_BLOCK_N / WARPS_N;
  constexpr int MMA_N_ITERS = N_PER_WARP / 8;
  static_assert(MMA_N_ITERS >= 1, "each active warp needs >= one n8 MMA");
  static_assert(WARPS_M == 1,
                "per-warp B ring assumes every warp owns a disjoint "
                "N_PER_WARP-row B slice (decode batch <= 16 -> TILE_M 16)");

  constexpr int A_ROW_STRIDE = a_row_stride(REDUCTION_SIZE);
  constexpr int SMEM_A_TOTAL = TILE_M * A_ROW_STRIDE;
  constexpr int SMEM_B_BYTES = TASK_BLOCK_N * SMEM_ROW_STRIDE;
  constexpr int BYTES_PER_CP = 16;
  constexpr int CHUNKS_PER_ROW = GROUP_K / BYTES_PER_CP;
  constexpr int A_CHUNKS = TILE_M * (REDUCTION_SIZE / BYTES_PER_CP);

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
  uint8_t *smem_a = smem_blockscale; // whole-K activation tile, staged once
  uint8_t *smem_b = smem_a + SMEM_A_TOTAL; // NUM_STAGES-deep B ring
  // bf16 residual staging tile (loaded with A's commit group when
  // WITH_RESIDUAL, so it's resident long before the epilogue).
  T *smem_res = reinterpret_cast<T *>(smem_b + NUM_STAGES * SMEM_B_BYTES);
  // fp32 scale staging, loaded with A's commit group: the whole
  // [BATCH_SIZE, K/128] activation-scale panel plus this task's [K/128]
  // weight-scale slice. The K loop's promotion previously issued dependent
  // GLOBAL loads per K tile (cold after the L2 flush / in MPK steady state)
  // squarely on the inter-tile critical path; staged here they are ~20-cycle
  // smem reads. Values and multiply order are identical -- bit-exact.
  float *smem_ascale = reinterpret_cast<float *>(
      reinterpret_cast<uint8_t *>(smem_res) + TILE_M * TASK_BLOCK_N * 2);
  float *smem_bscale = smem_ascale + BATCH_SIZE * NUM_K_TILES;

  int const tid = threadIdx.x;
  int const lane = lane_id();
  int const warp = tid / NUM_THREADS_PER_WARP;
  bool const active = warp < ACTIVE_WARPS;
  int const warp_m = (warp / WARPS_N) % WARPS_M;
  int const warp_n = warp % WARPS_N;
  // Fragment coordinates shared by A, B and D: the operand row/column a lane
  // owns is lane/4, the k byte offset it owns is 4*(lane%4).
  int const frag_row = lane >> 2;

  // Stage the WHOLE activation extent (rows m0..m0+TILE_M x all of K) once.
  // Rows past the batch are zero-filled by cp.async's src-size predicate, so
  // they contribute nothing to the MMA.
  auto load_a_full = [&](int m0, int n0) {
    constexpr int A_CHUNKS_PER_ROW = REDUCTION_SIZE / BYTES_PER_CP;
    for (int c = tid; c < A_CHUNKS; c += WORKER_NUM_THREADS) {
      int const r = c / A_CHUNKS_PER_ROW;
      int const x = (c % A_CHUNKS_PER_ROW) * BYTES_PER_CP;
      int const g_row = m0 + r;
      // Rows past the batch copy 0 bytes (cp.async zero-fills); their source
      // address is clamped so it always stays inside the tensor.
      int const src_row = g_row < BATCH_SIZE ? g_row : BATCH_SIZE - 1;
      load_smem_cg_predict<BYTES_PER_CP>(
          smem_a + r * A_ROW_STRIDE + x,
          d_input + (size_t)src_row * REDUCTION_SIZE + x,
          g_row < BATCH_SIZE);
    }
    // Scale staging (same commit group as A). Both panels are contiguous
    // and KT*4 is a multiple of 16 for every K >= 512, so plain 16-byte
    // chunks cover them exactly.
    static_assert((NUM_K_TILES * 4) % BYTES_PER_CP == 0,
                  "scale staging assumes K/128 float32s pack into 16B chunks");
    {
      constexpr int AS_CHUNKS = BATCH_SIZE * NUM_K_TILES * 4 / BYTES_PER_CP;
      uint8_t const *src = reinterpret_cast<uint8_t const *>(d_input_scale);
      for (int c = tid; c < AS_CHUNKS; c += WORKER_NUM_THREADS) {
        load_smem_cg<BYTES_PER_CP>(
            reinterpret_cast<uint8_t *>(smem_ascale) + c * BYTES_PER_CP,
            src + c * BYTES_PER_CP);
      }
      constexpr int BS_CHUNKS = NUM_K_TILES * 4 / BYTES_PER_CP;
      uint8_t const *bsrc = reinterpret_cast<uint8_t const *>(
          d_weight_scale + (size_t)(n0 / BLOCK_N) * NUM_K_TILES);
      for (int c = tid; c < BS_CHUNKS; c += WORKER_NUM_THREADS) {
        load_smem_cg<BYTES_PER_CP>(
            reinterpret_cast<uint8_t *>(smem_bscale) + c * BYTES_PER_CP,
            bsrc + c * BYTES_PER_CP);
      }
    }
    if constexpr (WITH_RESIDUAL) {
      // Stage this tile's bf16 residual alongside A (same commit group):
      // 16B = 8 bf16, TASK_BLOCK_N cols = TASK_BLOCK_N*2/16 chunks per row.
      constexpr int RES_CHUNKS_PER_ROW = TASK_BLOCK_N * 2 / BYTES_PER_CP;
      constexpr int RES_CHUNKS = TILE_M * RES_CHUNKS_PER_ROW;
      for (int c = tid; c < RES_CHUNKS; c += WORKER_NUM_THREADS) {
        int const r = c / RES_CHUNKS_PER_ROW;
        int const x = (c % RES_CHUNKS_PER_ROW) * BYTES_PER_CP;
        int const g_row = m0 + r;
        int const src_row = g_row < BATCH_SIZE ? g_row : BATCH_SIZE - 1;
        load_smem_cg_predict<BYTES_PER_CP>(
            reinterpret_cast<uint8_t *>(smem_res) + r * (TASK_BLOCK_N * 2) + x,
            reinterpret_cast<uint8_t const *>(
                d_residual + (size_t)src_row * O_STRIDE + n0) +
                x,
            g_row < BATCH_SIZE);
      }
    }
  };
  // Stage one B K-tile into a ring slot -- PER-WARP: each warp issues
  // cp.asyncs only for the N_PER_WARP-row slice it alone consumes (fragment
  // reads index warp_n * N_PER_WARP + ...), so B completion is a per-warp
  // dependency: the warp waits on its own commit groups and __syncwarp()s.
  // No block-wide barrier is needed anywhere in the K loop, and no warp ever
  // stalls on another warp's slowest cp.async.
  constexpr int B_CHUNKS_PER_WARP = N_PER_WARP * CHUNKS_PER_ROW;
  auto load_b_tile = [&](int stage, int n0, int kt) {
    if (!active) {
      return; // idle warps (narrow slices) issue no B copies
    }
    uint8_t *stage_b = smem_b + stage * SMEM_B_BYTES;
    size_t const k_off = (size_t)kt * GROUP_K;
    int const r_base = warp_n * N_PER_WARP;
    for (int c = lane; c < B_CHUNKS_PER_WARP; c += NUM_THREADS_PER_WARP) {
      int const r = r_base + c / CHUNKS_PER_ROW;
      int const x = (c % CHUNKS_PER_ROW) * BYTES_PER_CP;
      load_smem_cg<BYTES_PER_CP>(
          stage_b + r * SMEM_ROW_STRIDE + x,
          d_weight + (size_t)(n0 + r) * REDUCTION_SIZE + k_off + x);
    }
  };

  for (int nb = 0; nb < NUM_N_BLOCKS; ++nb) {
    int const n0 = nb * TASK_BLOCK_N;
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

      // Pipeline prologue: the whole-K A tile first (its commit group is the
      // OLDEST, so the first wait below retires it before any fragment
      // read), then the first NUM_STAGES-1 B tiles. Short-K shapes commit
      // empty groups so the in-loop wait arithmetic stays exact.
      load_a_full(m0, n0);
      cp_async_fence();
#pragma unroll
      for (int s = 0; s < NUM_STAGES - 1; ++s) {
        if (s < NUM_K_TILES) {
          load_b_tile(s, n0, s);
        }
        cp_async_fence();
      }
      // Retire the block-cooperative A(+residual) group ONCE up front (B
      // stage groups stay in flight) and make it visible block-wide. B is
      // warp-private from here on, so the K loop below never takes a
      // block-wide barrier again.
      cp_async_wait<NUM_STAGES - 1>();
      __syncthreads();

      // Narrow slices: warps past ACTIVE_WARPS have no B slice / no output
      // columns -- they skip straight to the tile barrier below.
      if (active) {
        // K tiles are processed in PAIRS with their unscaled-MMA issue
        // streams interleaved: at narrow slices (MMA_N_ITERS == 1) a single
        // tile has ONE serial fp32 MMA dependency chain per warp, and with B
        // streaming de-bottlenecked by N-slicing that chain is the exposed
        // per-tile serial cost. Pairing doubles the independent chains.
        // PROMOTION stays strictly ordered (kt2 then kt2+1, each with the
        // identical per-tile subtile order) -- the across-tile multiply-
        // accumulate order of the numerics contract is untouched, bit-exact.
        static_assert(NUM_K_TILES % 2 == 0,
                      "paired K-tile interleave needs an even tile count");
        static_assert(NUM_STAGES >= 4,
                      "pair loop waits on the two oldest of NUM_STAGES-1 "
                      "outstanding B groups");
        for (int kt2 = 0; kt2 < NUM_K_TILES; kt2 += 2) {
          // NUM_STAGES-1 commit groups are outstanding here (two committed
          // per pair below, empty through the drain), so wait<NUM_STAGES-3>
          // completes THIS warp's slices of B tiles kt2 AND kt2+1;
          // __syncwarp() publishes lanes' copies warp-wide.
          cp_async_wait<NUM_STAGES - 3>();
          __syncwarp();

          uint8_t const *stage_b0 = smem_b + (kt2 % NUM_STAGES) * SMEM_B_BYTES;
          uint8_t const *stage_b1 =
              smem_b + ((kt2 + 1) % NUM_STAGES) * SMEM_B_BYTES;

          // Unscaled FP8 MMA partials, one set per tile of the pair.
          float partial[2][MMA_N_ITERS][4];
#pragma unroll
          for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                partial[t][ni][i] = 0.0f;
              }
            }
          }
          static_assert(MMA_N_ITERS == 1 || MMA_N_ITERS % 2 == 0,
                        "B fragments load via paired ldmatrix.x4 (even "
                        "MMA_N_ITERS) or a single ldmatrix.x2 (== 1)");
          int const j8 = lane & 7;
          int const sel = lane >> 3; // which of the 4 ldmatrix matrices
          int const a_row = warp_m * 16 + j8 + (sel & 1) * 8;
          // Double-buffered fragment registers over the interleaved stream:
          // step s covers subtile s>>1 of tile s&1; step s+1's ldmatrix
          // loads issue BEFORE step s's mma, hiding the shared-memory
          // fragment latency behind tensor-core work. Same instructions on
          // the same bytes -- bit-exact.
          uint32_t a_frag[2][4];
          uint32_t b_frag[2][(MMA_N_ITERS + 1) / 2][4];
          auto load_frags = [&](int s, int buf) {
            int const t = s & 1;
            int const ks = s >> 1;
            int const cb = (kt2 + t) * GROUP_K + ks * MMA_K + (sel >> 1) * 16;
            ldsm_x4(smem_a + a_row * A_ROW_STRIDE + cb, a_frag[buf]);
            uint8_t const *stage = t ? stage_b1 : stage_b0;
            if constexpr (MMA_N_ITERS % 2 == 0) {
#pragma unroll
              for (int ni = 0; ni < MMA_N_ITERS; ni += 2) {
                int const brow =
                    warp_n * N_PER_WARP + (ni + (sel >> 1)) * 8 + j8;
                int const bcb = ks * MMA_K + (sel & 1) * 16;
                ldsm_x4(stage + brow * SMEM_ROW_STRIDE + bcb,
                        b_frag[buf][ni / 2]);
              }
            } else {
              int const brow = warp_n * N_PER_WARP + j8;
              int const bcb = ks * MMA_K + (sel & 1) * 16;
              ldsm_x2(stage + brow * SMEM_ROW_STRIDE + bcb, b_frag[buf][0]);
            }
          };
          load_frags(0, 0);
#pragma unroll
          for (int s = 0; s < 2 * NUM_K_SUBTILES; ++s) {
            int const cur = s & 1;
            if (s + 1 < 2 * NUM_K_SUBTILES) {
              load_frags(s + 1, cur ^ 1);
            }
            float(*p)[4] = partial[s & 1];
            if constexpr (MMA_N_ITERS % 2 == 0) {
#pragma unroll
              for (int ni = 0; ni < MMA_N_ITERS; ni += 2) {
                mma_m16n8k32_e4m3_f32(p[ni], a_frag[cur], b_frag[cur][ni / 2]);
                mma_m16n8k32_e4m3_f32(
                    p[ni + 1], a_frag[cur], b_frag[cur][ni / 2] + 2);
              }
            } else {
              mma_m16n8k32_e4m3_f32(p[0], a_frag[cur], b_frag[cur][0]);
            }
          }

          // Refill the two consumed slots AFTER the pair's ldmatrix reads
          // (warp program order is the WAR guard -- the second refill's slot
          // aliases slot kt2%NUM_STAGES when NUM_STAGES is even). Always
          // commit both groups (empty near the tail) to keep the wait<>
          // group count invariant.
          if (kt2 + NUM_STAGES - 1 < NUM_K_TILES) {
            load_b_tile(
                (kt2 + NUM_STAGES - 1) % NUM_STAGES, n0, kt2 + NUM_STAGES - 1);
          }
          cp_async_fence();
          if (kt2 + NUM_STAGES < NUM_K_TILES) {
            load_b_tile((kt2 + NUM_STAGES) % NUM_STAGES, n0, kt2 + NUM_STAGES);
          }
          cp_async_fence();

          // Promotion: fold each tile's float32 scales into the accumulator,
          // kt2 STRICTLY BEFORE kt2+1 -- the same across-tile order as the
          // one-tile-at-a-time loop. Rows past the batch read no scale
          // (their data is zero anyway). Scales come from the smem panels.
#pragma unroll
          for (int t = 0; t < 2; ++t) {
            int const kt = kt2 + t;
            float const b_scale = smem_bscale[kt];
            float const s0 =
                (row0 < BATCH_SIZE)
                    ? smem_ascale[row0 * NUM_K_TILES + kt] * b_scale
                    : 0.0f;
            float const s1 =
                (row1 < BATCH_SIZE)
                    ? smem_ascale[row1 * NUM_K_TILES + kt] * b_scale
                    : 0.0f;
#pragma unroll
            for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
              acc[ni][0] += partial[t][ni][0] * s0;
              acc[ni][1] += partial[t][ni][1] * s0;
              acc[ni][2] += partial[t][ni][2] * s1;
              acc[ni][3] += partial[t][ni][3] * s1;
            }
          }
          // No trailing sync: the per-warp wait at the top of the next pair
          // plus warp program order is the WAR guard before this warp's
          // slice of any ring slot is refilled.
        }

#pragma unroll
        for (int ni = 0; ni < MMA_N_ITERS; ++ni) {
          int const lcol = warp_n * N_PER_WARP + ni * 8 + (lane & 3) * 2;
          int const col = n0 + lcol;
          if (row0 < BATCH_SIZE) {
            float v0 = acc[ni][0];
            float v1 = acc[ni][1];
            if constexpr (WITH_RESIDUAL) {
              // Residual was staged into smem with A's commit group.
              v0 += float(smem_res[(row0 - m0) * TASK_BLOCK_N + lcol]);
              v1 += float(smem_res[(row0 - m0) * TASK_BLOCK_N + lcol + 1]);
            }
            d_output[(size_t)row0 * O_STRIDE + col] = T(v0);
            d_output[(size_t)row0 * O_STRIDE + col + 1] = T(v1);
          }
          if (row1 < BATCH_SIZE) {
            float v2 = acc[ni][2];
            float v3 = acc[ni][3];
            if constexpr (WITH_RESIDUAL) {
              v2 += float(smem_res[(row1 - m0) * TASK_BLOCK_N + lcol]);
              v3 += float(smem_res[(row1 - m0) * TASK_BLOCK_N + lcol + 1]);
            }
            d_output[(size_t)row1 * O_STRIDE + col] = T(v2);
            d_output[(size_t)row1 * O_STRIDE + col + 1] = T(v3);
          }
        }
      } // if (active)
      // The next M tile / N block reuses both staging buffers.
      __syncthreads();
    }
  }
}

// The entry point task_register.cc emits. Compile-time dispatch only: the
// discarded branch is never instantiated, so an instantiation the fast path
// cannot serve (batch > 16, odd K-tile count, a ring that does not fit shared
// memory) compiles the golden path instead of failing its static_asserts.
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
  // THE SAFETY PROPERTY, asserted where no compiler's instantiation eagerness
  // can affect it: every instantiation has an admissible path. If the fast path
  // rejects this (batch > 16, odd K-tile count, a ring that does not fit shared
  // memory) then the golden path must be able to run it, and the golden path
  // needs whole 128-row scale blocks. A builder that asks for a sub-block slice
  // at, say, batch 64 fails the BUILD here rather than silently computing 128
  // rows' worth of output for a 64-row slice.
  static_assert(linear_fp8_blockscale::fast_path_ok(
                    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE) ||
                    OUTPUT_SIZE % linear_fp8_blockscale::BLOCK_N == 0,
                "no admissible linear_fp8_blockscale path for this "
                "instantiation: the ferret fast path rejected it and the golden "
                "path requires a per-task N of whole 128-row scale blocks");
  if constexpr (linear_fp8_blockscale::fast_path_ok(
                    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)) {
    linear_fp8_blockscale_task_impl_fast<T,
                                         BATCH_SIZE,
                                         OUTPUT_SIZE,
                                         REDUCTION_SIZE,
                                         O_STRIDE,
                                         WITH_RESIDUAL>(input_fp8_ptr,
                                                        input_scale_ptr,
                                                        weight_fp8_ptr,
                                                        weight_scale_ptr,
                                                        residual_ptr,
                                                        output_ptr);
  } else {
    // golden_output_size() is the identity on every reachable instantiation
    // (guaranteed by the static_assert above); see its comment.
    linear_fp8_blockscale_task_impl_golden<
        T,
        BATCH_SIZE,
        linear_fp8_blockscale::golden_output_size(OUTPUT_SIZE),
        REDUCTION_SIZE,
        O_STRIDE,
        WITH_RESIDUAL>(input_fp8_ptr,
                       input_scale_ptr,
                       weight_fp8_ptr,
                       weight_scale_ptr,
                       residual_ptr,
                       output_ptr);
  }
}

} // namespace kernel
