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
#include "tasks/common/common_header.cuh"

// Qwen3.5 shared-expert gate + residual fold:
//
//     r'[b, :] = residual[b, :] + sigmoid(x[b, :] . w_sg^T) * shared[b, :]
//
// `x` is the PRE-MLP hidden state (the same tensor the router reads), `w_sg` is
// the `[1, hidden]` unquantized `shared_expert_gate` weight, and `shared` is
// the shared expert's post-`down_proj` output. The scalar gate is applied AFTER
// the down projection, not inside the MLP (vllm-graph.md 2.3.3; transformers
// `Qwen3_5MoeSparseMoeBlock.forward`).
//
// It exists because a `linear_layer` at N = 1 is degenerate (mpk-gaps.md Gap
// 8): one output element cannot fill a GEMM tile, and the gate must then be
// broadcast back across all `hidden` columns anyway. Folding the GEMV, the
// sigmoid, the broadcast multiply and the residual add into one task also lets
// the result be handed straight to `moe_mul_sum_add_layer`'s single `residual`
// argument, which is what DeepSeek-V3's builder does for its (ungated) shared
// expert (persistent_kernel.py `moe_mul_sum_add_layer`).
//
// CAST POSITIONS are load-bearing and follow the HF reference, not the
// mathematically-natural fp32 chain (docs/qwen35/v1-architecture.md 2.4 #12,
// oracle `moe*.shared_gate_logit` / `moe*.shared_gate_sigmoid`, both bf16):
//   1. the GEMV accumulates in fp32 and is rounded to bf16   -> logit
//   2. sigmoid is evaluated in fp32 and rounded to bf16      -> gate
//      (torch computes bf16 sigmoid via an fp32 opmath and casts back)
//   3. the multiply-add runs in fp32 and rounds once         -> r'
// Step 2's rounding is what makes the task insensitive to the megakernel's
// `-use_fast_math` rewrite of `expf`: a ~2-fp32-ULP difference in the sigmoid
// is four orders of magnitude below one bf16 LSB.
//
// Grid: (num_tasks, 1, 1) partitioning the BATCH dimension. The gate needs the
// whole `x` row, so the hidden dimension must NOT be split across tasks.
// Block: (256, 1, 1) on Blackwell.

namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int HIDDEN_SIZE,
          int X_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void sigmoid_gate_mul_add_task_impl(
    void const *x_ptr,        // [BATCH_SIZE, HIDDEN_SIZE]  pre-MLP hidden state
    void const *gate_w_ptr,   // [1, HIDDEN_SIZE]           shared_expert_gate
    void const *shared_ptr,   // [BATCH_SIZE, OUTPUT_SIZE]  shared expert output
    void const *residual_ptr, // [BATCH_SIZE, OUTPUT_SIZE]  layer residual
    void *output_ptr) {       // [BATCH_SIZE, OUTPUT_SIZE]
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  T const *__restrict__ d_w = static_cast<T const *>(gate_w_ptr);
  T const *__restrict__ d_shared = static_cast<T const *>(shared_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // One float per warp for the GEMV cross-warp reduction, plus one slot for the
  // broadcast gate. blockDim.x <= 256 => at most 8 warps, so the gate lives at
  // slot 8 and the whole buffer is 9 floats.
  constexpr int MAX_WARPS = 8;
  constexpr int GATE_SLOT = MAX_WARPS;
  extern __shared__ char smem[];
  float *reduce_buf = reinterpret_cast<float *>(smem);

  int const lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int const warp = threadIdx.x / NUM_THREADS_PER_WARP;
  int const num_warps =
      (blockDim.x + NUM_THREADS_PER_WARP - 1) / NUM_THREADS_PER_WARP;

  for (int row = 0; row < BATCH_SIZE; ++row) {
    // ---- 1. gate GEMV: logit = x[row, :] . w_sg, fp32 accumulate ----
    float partial = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += blockDim.x) {
      partial += float(d_x[row * X_STRIDE + i]) * float(d_w[i]);
    }
#pragma unroll
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    __syncthreads(); // reduce_buf is reused across rows
    if (lane == 0) {
      reduce_buf[warp] = partial;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float logit = 0.0f;
      for (int w = 0; w < num_warps; ++w) {
        logit += reduce_buf[w];
      }
      // ---- 2. HF cast positions: bf16 logit, then bf16 sigmoid ----
      float logit_rounded = float(T(logit));
      float gate = 1.0f / (1.0f + expf(-logit_rounded));
      reduce_buf[GATE_SLOT] = float(T(gate));
    }
    __syncthreads();
    float const gate = reduce_buf[GATE_SLOT];

    // ---- 3. broadcast multiply + residual add, one rounding ----
    for (int i = threadIdx.x; i < OUTPUT_SIZE; i += blockDim.x) {
      float const s = float(d_shared[row * O_STRIDE + i]);
      float const r = float(d_residual[row * O_STRIDE + i]);
      d_output[row * O_STRIDE + i] = T(r + gate * s);
    }
  }
}

} // namespace kernel
