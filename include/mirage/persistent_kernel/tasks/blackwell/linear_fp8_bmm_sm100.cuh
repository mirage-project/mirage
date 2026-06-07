/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// FP8 batched matmul (BMM) for Blackwell SM100. Each CTA computes the
// per-head GEMM
//     output[n, h, m_lo:m_hi] = input[n, h, :] @ weight[h, m_lo:m_hi, :]^T
// for a single head h chosen by grid.y, and an M-shard (m_lo, m_hi) chosen
// by grid.x. H is exposed as an extra workload-split dimension on top of
// the existing swapAB MMA_M=128 split.
//
// Grid contract (set in the Python layer):
//   grid_dim = (D_OUT / 128, H / H_PER_TASK, 1)
//   block_dim = (256, 1, 1)
// First cut: H_PER_TASK = 1 (one head per CTA). The kernel body is the
// existing swapAB UMMA pipeline — we reach the per-head slice through
// per-task TMA descriptors that the runtime constructs from the TBGraph
// partition map (input split on dim H, weight split on dim H + dim D_OUT,
// output split on dim H + dim D_OUT). Future H_PER_TASK > 1 work would
// add an outer head loop here.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "linear_fp8_swapAB_sm100.cuh"

namespace kernel {

template <typename T_,
          typename TMA_A,
          typename TMA_B,
          class BiasTensor,
          typename TMA_OUT,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE_PER_TASK,
          int REDUCTION_SIZE,
          bool NOBIAS,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __forceinline__ void
    linear_fp8_bmm_sm100_task_impl(TMA_A const &tma_a,
                                   TMA_B const &tma_b,
                                   uint32_t const *weight_scale_ptr,
                                   uint32_t const *input_scale_ptr,
                                   int weight_scale_row_stride,
                                   int input_scale_row_stride,
                                   BiasTensor mBias,
                                   TMA_OUT const &tma_out) {
  // The swapAB kernel's body is agnostic to whether the per-CTA tile comes
  // from a flat [OUT, K] / [BATCH, K] matrix or from a per-head slice of a
  // larger [H, OUT, K] / [BATCH, H, K] tensor — the TMA descriptors fully
  // encode the gmem strides, so per-head row stride (H*K for input, H*OUT
  // for output, K for weight) is supplied via the TMA template parameters
  // instantiated in register_linear_fp8_bmm_sm100_task.
  linear_fp8_swapAB_sm100_task_impl<T_,
                                    TMA_A,
                                    TMA_B,
                                    BiasTensor,
                                    TMA_OUT,
                                    MMA_M,
                                    MMA_N,
                                    BATCH_SIZE,
                                    OUTPUT_SIZE_PER_TASK,
                                    REDUCTION_SIZE,
                                    NOBIAS,
                                    /*SplitK=*/false,
                                    NUM_AB_STAGE,
                                    NUM_ACC_STAGE,
                                    NUM_C_STAGE>(tma_a,
                                                 tma_b,
                                                 weight_scale_ptr,
                                                 input_scale_ptr,
                                                 weight_scale_row_stride,
                                                 input_scale_row_stride,
                                                 mBias,
                                                 tma_out);
}

// ============================================================================
// linear_fp8_bmm_cuda_core_sm100_task_impl — ZERO-TMEM CUDA-core alternative
// (ferret workspace5, MPK_DSV3_BMM1_CUDA_CORE gate). Computes the per-head
// kv-up BMM1 with scalar float accumulation and NO tcgen05 / NO TMEM / NO TMA
// / NO mbarrier — eliminating the per-task tensor-core setup overhead that the
// swapAB path pays. LOSES standalone (~0.6x ref), the in-MPK A/B is the test.
//
// Numerically equivalent to the swapAB path (modulo float-vs-tensorcore accum
// order): both decode UE8M0 byte b as 2^(b-127) and form acc = dot(input_fp8,
// weight_fp8) * input_dequant * weight_dequant.
//
// MEMORY: this version carves its scratch from the worker's DYNAMIC shared
// pool (`extern __shared__ char shared_memory[]`), NOT function-scope static
// __shared__ — function-scope __shared__ would be counted as STATIC smem in
// the worker __global__ (summed across all reachable task bodies regardless of
// dispatch), which on top of the megakernel's 214KB dynamic request overflows
// the SM100 ~228KB per-block cap and breaks the launch. The dynamic carve
// (~27KB at N=128,K=128,B=16) fits comfortably inside the 214KB pool.
//
// ABI (raw pointers, already per-task (head h, shard s) offset by the MPK
//      runtime via input_map/output_map — the caller passes input_ptrs[]/
//      output_ptrs[0] DIRECTLY, with NO additional h/s offsetting):
//   weight        : per-task base, row-major [N=OUTPUT_SIZE_PER_TASK][K] FP8
//   input         : this-head base; batch row b at b*INPUT_ROW_STRIDE, K
//                   contiguous FP8 (INPUT_ROW_STRIDE = NUM_HEADS*K)
//   weight_scale  : [N] uint32 UE8M0 (low byte), stride 1
//   input_scale   : per-batch uint32 UE8M0, batch b at b*INPUT_SCALE_STRIDE
//                   (INPUT_SCALE_STRIDE = NUM_HEADS)
//   output        : this-task base (column offset h*N_PER_HEAD + s*N baked in
//                   by the runtime); element (b,n) at b*output_row_stride + n
//   output_row_stride : BF16-element batch row stride (= H * N_PER_HEAD = 18432
//                       for the DSv3 q_nope_pe [mbt,H,576] parent view)
//
// BATCH_SIZE is compiled to the DECODE active-row tile (16), NOT the backing
// tensor's mbt (128): at decode active_rows<=8<16 carry real data, rows 16..127
// of q_nope_abs are correctness-dead (assemble_q_decode + MLA decode are both
// active-row gated). This matches the swapAB path's effective behavior (it too
// leaves rows beyond active as don't-care) while keeping smem/compute bounded.
// ============================================================================
template <int OUTPUT_SIZE_PER_TASK,
          int REDUCTION_SIZE,
          int BATCH_SIZE,
          int INPUT_ROW_STRIDE,
          int INPUT_SCALE_STRIDE>
__device__ __noinline__ void linear_fp8_bmm_cuda_core_sm100_task_impl(
    const __nv_fp8_e4m3 *__restrict__ weight,
    const __nv_fp8_e4m3 *__restrict__ input,
    const uint32_t *__restrict__ weight_scale,
    const uint32_t *__restrict__ input_scale,
    __nv_bfloat16 *__restrict__ output,
    int output_row_stride) {
  constexpr int N = OUTPUT_SIZE_PER_TASK;
  constexpr int K = REDUCTION_SIZE;
  constexpr int B = BATCH_SIZE;
  constexpr int WROW_B = K + 16; // padded FP8 weight row stride (no bank conflict)
  constexpr int U4 = K / 16;     // uint4 (16 FP8) chunks per K-row

  // Dynamic shared-memory carve (replaces the delivered function-scope static
  // __shared__). Layout: [smem_w (uint8, N*WROW_B)] [smem_in (float, B*K)]
  // [smem_sfw (float, N)] [smem_sfi (float, B)]. Align the float region to 16B.
  extern __shared__ char shared_memory[];
  uint8_t *smem_w = reinterpret_cast<uint8_t *>(shared_memory);
  size_t off = static_cast<size_t>(N) * WROW_B;
  off = (off + 15) & ~static_cast<size_t>(15); // align to 16B for float reads
  float *smem_in = reinterpret_cast<float *>(shared_memory + off);
  off += static_cast<size_t>(B) * K * sizeof(float);
  float *smem_sfw = reinterpret_cast<float *>(shared_memory + off);
  off += static_cast<size_t>(N) * sizeof(float);
  float *smem_sfi = reinterpret_cast<float *>(shared_memory + off);

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // ── COLD-DATA DOUBLET PROBE (env-gated -DMPK_BMM1_DOUBLET) ───────────────
  // Run the weight-load(+input+scales)+compute body TWICE in one task:
  // rep0 reads the 2MB weight COLD from HBM; rep1 reads it WARM (L2/TLB-hot
  // from rep0). %clock64-bracket each rep; printf cold=rep0, warm=rep1 from
  // block 0. warm<<cold ⇒ the in-MPK skinny-GEMM penalty is cold-first-touch
  // (data/TLB) → a weight-prefetch is a safe in-MPK lever. warm≈cold ⇒ the cost
  // is NOT cold-ness (fundamental compute/issue). Idempotent (same inputs →
  // same output; rep1 just overwrites). No mbarrier/tcgen05 → safe to repeat.
#ifdef MPK_BMM1_DOUBLET
  const int _nrep = 2;
#else
  const int _nrep = 1;
#endif
  long long _t[3];
  for (int _rep = 0; _rep < _nrep; ++_rep) {
#ifdef MPK_BMM1_DOUBLET
    __syncthreads();
    if (tid == 0)
      _t[_rep] = clock64();
    __syncthreads();
#endif
    // weight -> FP8 smem, padded rows, vectorized uint4.
    for (int u4 = tid; u4 < N * U4; u4 += nthreads) {
      int n = u4 / U4, kb = (u4 - n * U4) * 16;
      *reinterpret_cast<uint4 *>(smem_w + n * WROW_B + kb) =
          *reinterpret_cast<const uint4 *>(weight + (size_t)n * K + kb);
    }
    // input -> float smem (per-head base, batch stride INPUT_ROW_STRIDE).
    for (int u4 = tid; u4 < B * U4; u4 += nthreads) {
      int b = u4 / U4, kb = (u4 - b * U4) * 16;
      uint4 v = *reinterpret_cast<const uint4 *>(
          input + (size_t)b * INPUT_ROW_STRIDE + kb);
      const __nv_fp8_e4m3 *fb = reinterpret_cast<const __nv_fp8_e4m3 *>(&v);
      float *dst = smem_in + b * K + kb;
#pragma unroll
      for (int j = 0; j < 16; ++j)
        dst[j] = float(fb[j]);
    }
    if (tid < N)
      smem_sfw[tid] = exp2f(float(weight_scale[tid] & 0xFFu) - 127.0f);
    if (tid < B)
      smem_sfi[tid] =
          exp2f(float(input_scale[tid * INPUT_SCALE_STRIDE] & 0xFFu) - 127.0f);
    __syncthreads();

    // compute: each thread owns output elements o = b*N + n, full-K dot.
    for (int o = tid; o < B * N; o += nthreads) {
      int b = o / N, n = o - b * N;
      const float *irow = smem_in + b * K;
      const uint8_t *wrow = smem_w + n * WROW_B;
      float acc = 0.f;
#pragma unroll
      for (int k4 = 0; k4 < U4; ++k4) {
        uint4 wv = *reinterpret_cast<const uint4 *>(wrow + k4 * 16);
        const __nv_fp8_e4m3 *wb = reinterpret_cast<const __nv_fp8_e4m3 *>(&wv);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          acc += irow[k4 * 16 + j] * float(wb[j]);
      }
      acc *= smem_sfi[b] * smem_sfw[n];
      output[(size_t)b * output_row_stride + n] = __nv_bfloat16(acc);
    }
  }
#ifdef MPK_BMM1_DOUBLET
  __syncthreads();
  if (tid == 0)
    _t[2] = clock64();
  // Print only from one CTA (block 0) to avoid 128× spam; a few prints is fine.
  if (tid == 0 && blockIdx.x == 0)
    printf("[BMM1_DOUBLET] cold=%lld warm=%lld (clk) ratio=%.2f\n",
           _t[1] - _t[0], _t[2] - _t[1],
           (double)(_t[2] - _t[1]) / (double)(_t[1] - _t[0] + 1));
#endif
}

} // namespace kernel
