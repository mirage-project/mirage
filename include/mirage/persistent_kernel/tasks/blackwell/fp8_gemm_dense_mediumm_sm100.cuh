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

// Dense FP8 block-scaled GEMM, medium-M variant (M=512..2048 sweet spot).
// NE=4 TMEM stages (vs 2 in smallm) — more MMA↔epilogue overlap. Body is in
// fp8_gemm_dense_sm100_common.cuh. Beats DeepGEMM 1.14-3.82x at M=512..2048;
// trails DeepGEMM at M ≥ 4096 (use deep_gemm.fp8_gemm_nt for that range).
// Adapted from cpp_examples/blackwell_fp8_gemm/fp8_gemm_dense_tp8_sm100.cu
// (v002, ferret-generated).

#pragma once

#include "fp8_gemm_dense_sm100_common.cuh"

namespace kernel {
namespace fp8_gemm_dense_mediumm {

template <int BN, int NS>
__device__ __noinline__ void
    fp8_gemm_dense_mediumm_sm100_task_impl(CUtensorMap const *ta_ptr,
                                           CUtensorMap const *tb_ptr,
                                           float const *__restrict__ sa,
                                           float const *__restrict__ sb,
                                           __nv_bfloat16 *__restrict__ C,
                                           int const M,
                                           int const N,
                                           int const K,
                                           int const worker_idx,
                                           int const num_workers) {
  fp8_gemm_dense_common::task_impl_tpl<BN, NS, /*NE=*/4>(
      ta_ptr, tb_ptr, sa, sb, C, M, N, K, worker_idx, num_workers);
}

// D1 (2026-05-17): see smallm header — same epilogue-fused UE8M0 quantize
// variant, NE=4 TMEM staging.
template <int BN, int NS>
__device__ __noinline__ void
    fp8_gemm_dense_mediumm_fp8out_sm100_task_impl(
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
                                       /*NE=*/4,
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
__host__ __device__ inline constexpr int fp8_gemm_dense_mediumm_smem_size() {
  return fp8_gemm_dense_common::smem_size_tpl<BN, NS, /*NE=*/4>();
}

} // namespace fp8_gemm_dense_mediumm
} // namespace kernel
