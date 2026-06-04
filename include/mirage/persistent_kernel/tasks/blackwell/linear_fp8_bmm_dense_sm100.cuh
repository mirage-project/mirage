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

// Per-head FP8 batched matmul (BMM) for Blackwell SM100, wrapping the DENSE
// block-scaled GEMM body (fp8_gemm_dense_sm100_common.cuh) instead of the
// swapAB body. Each CTA computes one head's GEMM (head chosen by grid.y):
//     output[n, h, :] = input[n, h, :] @ weight[h, :, :]^T
//
// Why the dense body instead of linear_fp8_bmm_sm100 (swapAB)?  The dense
// family uses float32 128-K-aligned block scales (sa: [M, K/128] 1x128-group
// activation scale, sb: [N/128, K/128] 128x128-block weight scale). swapAB
// uses UE8M0 scales packed at 512-K granularity, which CANNOT split a small
// per-head K=512. The dense float32 layout is split-K-friendly (when the
// kernel team lands dense split-K), so BMM2's K=512 per-head reduction can be
// parallelized. No immediate perf win — this is a forward-compatible,
// correctness-equivalent re-encoding of the same math.
//
// A/B assignment is the OPPOSITE of swapAB: here A = activation (input),
// B = weight, exactly as fp8_gemm_dense_qout_common::task_impl_tpl expects
// (ta_ptr = A[M,K], tb_ptr = B[N,K]).
//
// Per-head slicing comes from per-task TMA descriptors + per-head scale base
// pointers that the runtime constructs from the TBGraph partition map
// (input split on dim H, weight split on dim H + D_out, output split on dim
// H + D_out) — identical to linear_fp8_bmm_sm100. The only BMM-specific twist
// the dense body needs is the activation-scale row stride: the activation
// scale is [M, H, nk] row-major, so consecutive M-rows of one head stride by
// H*nk, not nk. That stride is passed via `sa_row_stride`.
//
// Grid contract (set in the Python layer):
//   grid_dim  = (D_OUT / 128, H / H_PER_TASK, 1)   (first cut: H_PER_TASK = 1)
//   block_dim = (256, 1, 1)
// Each CTA handles exactly one head, so it forwards worker_idx=0,
// num_workers=1 to the dense body (one head = one full (M/128)x(N/128) tile
// set computed by a single persistent task).

#pragma once

#include "fp8_gemm_dense_qout_sm100_common.cuh"

namespace kernel {
namespace linear_fp8_bmm_dense {

template <int BN, int NS, int NE>
__device__ __noinline__ void
    linear_fp8_bmm_dense_sm100_task_impl(CUtensorMap const *ta_ptr,
                                         CUtensorMap const *tb_ptr,
                                         float const *__restrict__ sa,
                                         float const *__restrict__ sb,
                                         __nv_bfloat16 *__restrict__ C,
                                         int const M,
                                         int const N,
                                         int const K,
                                         int const sa_row_stride,
                                         int const C_row_stride) {
  // One head per CTA: the per-task TMA descriptors and per-head scale base
  // pointers already encode the head offset, so this CTA computes the entire
  // per-head tile by itself (worker_idx=0, num_workers=1).
  fp8_gemm_dense_qout_common::task_impl_tpl<BN, NS, NE>(ta_ptr,
                                                   tb_ptr,
                                                   sa,
                                                   sb,
                                                   C,
                                                   M,
                                                   N,
                                                   K,
                                                   /*worker_idx=*/0,
                                                   /*num_workers=*/1,
                                                   /*C_fp8=*/nullptr,
                                                   /*C_scale=*/nullptr,
                                                   /*scale_outer_stride=*/0,
                                                   sa_row_stride,
                                                   C_row_stride);
}

template <int BN, int NS, int NE>
__host__ __device__ inline constexpr int linear_fp8_bmm_dense_smem_size() {
  return fp8_gemm_dense_qout_common::smem_size_tpl<BN, NS, NE>();
}

} // namespace linear_fp8_bmm_dense
} // namespace kernel
