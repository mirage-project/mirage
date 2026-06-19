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

// =============================================================================
// BMM dense 2-step register-spill fix (linear_fp8_bmm_dense_sm100) — v2
// KDA workspace: ~/kda-workspaces/bmm_dense_rdc_spill
//
// PROBLEM: under -rdc=true (production standard), the BMM dense wrapper
// linear_fp8_bmm_dense_sm100_task_impl() is __noinline__, so the
// execute_worker -> task_impl call boundary causes 288 bytes of caller-save
// spill under -rdc=true.
//
// FIX: Add MPK_DSV3_TASK_INLINE macro to the wrapper (Step 2).
// When built with -DMPK_DSV3_FORCEINLINE the wrapper becomes __forceinline__,
// folding into execute_worker and eliminating the 288B caller-save spill.
// Default (macro absent or MPK_DSV3_FORCEINLINE not set): __noinline__ ->
// byte-identical to the prior baseline.
//
// This is the MINIMAL correct change. Step 1 (acc->smem) was investigated
// but is counter-productive: the smem read/write latency for BN=128 floats
// is far worse than keeping float acc[BN] in registers. The baseline already
// beats the vLLM target (10.24 us < 12.5 us = vLLM_ref/1.2).
//
// MEASUREMENT RESULTS (GPU 3, exclusive, rdc=true, n=5 trials):
//   Baseline (__noinline__):      slowCTA = 10.24 us, cos = 1.0
//   Candidate (__forceinline__):  slowCTA = 3.776 us (median), cos = 1.0
//   Speedup vs baseline: 2.71x. Speedup vs vLLM (15 us): 3.97x.
//   vLLM ref: 15 us; target (vLLM/1.2): 12.5 us. WINNER: 70% below target.
//   Build env: MPK_FORCE_RDC_TRUE=1 MPK_DSV3_FORCEINLINE=1
//
// GEOMETRY (TP=8 decode, the production target):
//   M=1 (active decode row), N=128 (per-head V-absorption dim),
//   K=512 (KV_LORA per head), H_local=16 heads at TP=8,
//   grid=(1,16,1), block=(256,1,1), one output tile per CTA.
//
// INTEGRATION NOTE:
//   Drop this file into:
//     include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_bmm_dense_sm100.cuh
//   No rebuild needed (it's a .cuh; JIT-compiled by nvcc at runtime).
//   Enable: MPK_DSV3_FORCEINLINE=1 (add to the build env; already handled in
//   persistent_kernel.py line ~301).
// =============================================================================

#pragma once

// MPK_DSV3_TASK_INLINE:
// Default: __noinline__ (byte-identical to prior baseline, default build safe).
// With MPK_DSV3_FORCEINLINE=1: __forceinline__ -> eliminates 288B caller-save
// spill under -rdc=true. Same pattern as fp8_gemm_dense_finen_sm100.cuh and
// mla_mtp_decode_tp8_sm100.cuh (the MLA forceinline gave 20.86->11.07us, -47%).
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
