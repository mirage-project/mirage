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

// linear_fp8_bmm_dense_sm100: TP=8 decode V-absorption BMM (M=1, N=128, K=512,
// 16 local heads; grid=(1,16,1), block=(256,1,1), one output tile per CTA).

#pragma once

// The BMM body runs __noinline__ behind the worker dispatch boundary so it gets
// its own register budget under -rdc=true rather than spilling the worker frame.
// The single-token decode build (MPK_DSV3_FORCEINLINE, mbt == 1) forceinlines it
// into the worker frame instead.
#ifndef MPK_DSV3_TASK_INLINE
#ifdef MPK_DSV3_FORCEINLINE
#define MPK_DSV3_TASK_INLINE __forceinline__
#else
#define MPK_DSV3_TASK_INLINE __noinline__
#endif
#endif

#include "fp8_gemm_dense_qout_sm100_common.cuh"

namespace kernel {
namespace linear_fp8_bmm_dense {

template <int BN, int NS, int NE>
__device__ MPK_DSV3_TASK_INLINE void
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
  fp8_gemm_dense_qout_common::task_impl_tpl<BN, NS, NE>(
      ta_ptr,
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
