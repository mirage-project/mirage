/* Copyright 2025 CMU
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

// Dense FP8 block-scaled GEMM, small-M variant (M ≤ 512 sweet spot).
// NE=2 TMEM stages. Body is in fp8_gemm_dense_sm100_common.cuh.

#pragma once

#include "fp8_gemm_dense_sm100_common.cuh"

namespace kernel {
namespace fp8_gemm_dense_smallm {

template <int BN, int NS>
__device__ __noinline__ void
    fp8_gemm_dense_smallm_sm100_task_impl(CUtensorMap const *ta_ptr,
                                          CUtensorMap const *tb_ptr,
                                          float const *__restrict__ sa,
                                          float const *__restrict__ sb,
                                          __nv_bfloat16 *__restrict__ C,
                                          int const M,
                                          int const N,
                                          int const K,
                                          int const worker_idx,
                                          int const num_workers) {
  fp8_gemm_dense_common::task_impl_tpl<BN, NS, /*NE=*/2>(
      ta_ptr, tb_ptr, sa, sb, C, M, N, K, worker_idx, num_workers);
}

// D1 (2026-05-17): variant that fuses per-128-col-group UE8M0 quantize into
// the consumer epilogue. Output is FP8 + packed UE8M0 scale instead of bf16.
// Eliminates the standalone per_token_group_quantize_fp8 task that today
// runs immediately downstream on the q_b_nope BMM-decode path.
//
// NE=1 (2026-06-05, NOT 2): this fp8out GEMM (q_b_nope, M=1 decode) is the
// SAME TMEM-contention-fragile shape as the BMM (see task_register.cc:6237) —
// at NE=2, TCA=NE*BN=256 cols and two concurrent tcgen05 tasks on one SM hit
// the 512-col limit, so tcgen05.alloc can return taddr=0 → "out-of-range"
// in tcgen05.mma. It survived at NE=2 only by schedule luck; enabling fine-N
// (MPK_DSV3_FINEN=1) shifts the wave timing and pushes q_b into the bad
// contention window (compute-sanitizer pinned the crash here, common.cuh:315).
// NE=1 (TCA=128) halves the ask so the alloc always fits, and is NUMERICALLY
// IDENTICAL (NE is only the MMA↔epilogue pipeline depth) + free at M=1
// decode (one MMA/CTA, no pipeline to fill) — exactly the BMM's resolution.
template <int BN, int NS>
__device__ __noinline__ void fp8_gemm_dense_smallm_fp8out_sm100_task_impl(
    CUtensorMap const *ta_ptr,
    CUtensorMap const *tb_ptr,
    float const *__restrict__ sa,
    float const *__restrict__ sb,
    __nv_fp8_e4m3 *__restrict__ C_fp8,
    uint32_t *__restrict__ C_scale,
    int const M,
    int const N,
    int const K,
    int const worker_idx,
    int const num_workers,
    int const scale_outer_stride) {
  fp8_gemm_dense_common::task_impl_tpl<BN,
                                       NS,
                                       /*NE=*/1,
                                       /*EPILOGUE_QUANTIZE_FP8=*/true>(
      ta_ptr,
      tb_ptr,
      sa,
      sb,
      /*C=*/nullptr,
      M,
      N,
      K,
      worker_idx,
      num_workers,
      C_fp8,
      C_scale,
      scale_outer_stride);
}

template <int BN, int NS>
__host__ __device__ inline constexpr int fp8_gemm_dense_smallm_smem_size() {
  return fp8_gemm_dense_common::smem_size_tpl<BN, NS, /*NE=*/2>();
}

} // namespace fp8_gemm_dense_smallm
} // namespace kernel
