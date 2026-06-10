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
} // namespace kernel
