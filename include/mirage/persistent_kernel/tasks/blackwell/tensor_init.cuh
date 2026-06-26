/* Copyright 2025 Mirage Team
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
#include "tasks/common/common_header.cuh"
namespace kernel {

// Vectorized zero-fill for the splitk-linear output tile.
//
// Caller wires this as a (1 input, 2 outputs) MPK task whose grid_dim and
// per-tensor input_maps mirror the downstream splitk linear:
//   * output_ptrs[0] -> the linear's output buffer (this kernel zeroes it)
//   * output_ptrs[1] -> the linear's input (untouched; pure dep edge)
//   * input_ptrs[0]  -> the linear's input (untouched; pure dep edge)
//
// Stores are 16-byte (int4) — alignment is guaranteed by the splitk linear's
// own constraint that per-task OUTPUT_SIZE and full OUTPUT_STRIDE are
// multiples of 128 bf16 (= 256 bytes).
template <int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    tensor_init_zero_sm100_task_impl(void *target_ptr) {
  using bf16 = cute::bfloat16_t;
  static_assert(OUTPUT_SIZE % 8 == 0,
                "tensor_init: OUTPUT_SIZE must be multiple of 8 (16B vec)");
  bf16 *base = static_cast<bf16 *>(target_ptr);
  constexpr int VEC = 8; // 8 bf16 = 16 bytes
  constexpr int VEC_PER_ROW = OUTPUT_SIZE / VEC;
  int4 const zero = {0, 0, 0, 0};
#pragma unroll
  for (int row = 0; row < BATCH_SIZE; ++row) {
    int4 *row_ptr = reinterpret_cast<int4 *>(base + row * OUTPUT_STRIDE);
    for (int i = threadIdx.x; i < VEC_PER_ROW; i += blockDim.x) {
      row_ptr[i] = zero;
    }
  }
} // tensor_init_zero_sm100_task_impl

// POISON-fill variant — DIAGNOSTIC ONLY (MPK_DSV3_POISON_TENSORINIT_AFTER_STEP0).
// Writes a sentinel bit pattern (bf16 NaN 0x7FC0 in every 16-bit lane) instead
// of zero. Used to safety-test the SKIP_TENSORINIT_AFTER_STEP0 lever: on a
// decode step>=1 we write poison into the ACTIVATION regions the skip would have
// left untouched; if the skip premise holds (every activation region is
// overwritten-before-read / zeroed-inside-the-kernel) the poison never reaches
// the logits and the output is clean; if any region is read-before-write on some
// step, the poison (NaN / huge garbage int) propagates -> deterministic NaN /
// garbage output, which is IMMUNE to the FP-atomicAdd run-to-run nondeterminism
// that makes the token-identity gate inconclusive.
//
// SKIP_HEAD_BYTES: leave the first SKIP_HEAD_BYTES of the buffer UNTOUCHED. This
// is mandatory for the attn-block megakernel scratch: its first 16 bytes hold
// the sense-reversing grid-barrier counter/gen (ATTN_BARRIER_BYTES=8 + pad to
// 16). That counter is SELF-MAINTAINED across steps (the last arriver resets it
// to 0); the SKIP lever PRESERVES that self-maintained 0, so poison must NOT
// clobber it (clobbering it would hang the barrier deterministically — a FALSE
// POSITIVE that is unrelated to whether the activation regions are stale-read-
// safe). SKIP_HEAD_BYTES must be a multiple of 16 (16B-vec store granularity).
template <int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE,
          int SKIP_HEAD_BYTES = 0>
__device__ __forceinline__ void
    tensor_init_poison_sm100_task_impl(void *target_ptr) {
  using bf16 = cute::bfloat16_t;
  static_assert(OUTPUT_SIZE % 8 == 0,
                "tensor_init: OUTPUT_SIZE must be multiple of 8 (16B vec)");
  static_assert(SKIP_HEAD_BYTES % 16 == 0,
                "tensor_init poison: SKIP_HEAD_BYTES must be multiple of 16");
  bf16 *base = static_cast<bf16 *>(target_ptr);
  constexpr int VEC = 8; // 8 bf16 = 16 bytes
  constexpr int VEC_PER_ROW = OUTPUT_SIZE / VEC;
  constexpr int VEC_SKIP = SKIP_HEAD_BYTES / 16; // 16B vecs to preserve at head
  // 0x7FC0 = bf16 quiet-NaN; packed twice per 32-bit lane.
  int const lane = 0x7FC07FC0;
  int4 const poison = {lane, lane, lane, lane};
#pragma unroll
  for (int row = 0; row < BATCH_SIZE; ++row) {
    int4 *row_ptr = reinterpret_cast<int4 *>(base + row * OUTPUT_STRIDE);
    int const start = (row == 0) ? VEC_SKIP : 0;
    for (int i = start + threadIdx.x; i < VEC_PER_ROW; i += blockDim.x) {
      row_ptr[i] = poison;
    }
  }
} // tensor_init_poison_sm100_task_impl

} // namespace kernel
