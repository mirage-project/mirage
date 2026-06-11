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

// EPILOGUE_QUANTIZE_FP8 ("fp8out") variants of the dense
// block-scaled GEMM. Split out of fp8_gemm_dense_{smallm,mediumm}_sm100.cuh
// (PR #707 review) so the proven bf16 kernels stay byte-identical to the
// fine-tuned baseline. The experimental epilogue-quantize task_impl_tpl
// lives in fp8_gemm_dense_qout_sm100_common.cuh (namespace
// fp8_gemm_dense_qout_common); the bf16 path is unchanged.
//
// Wrapper names / namespaces are unchanged so the task_register codegen
// (kernel::fp8_gemm_dense_{smallm,mediumm}::..._fp8out_sm100_task_impl)
// keeps resolving.

#pragma once

#include "fp8_gemm_dense_qout_sm100_common.cuh"

namespace kernel {
namespace fp8_gemm_dense_smallm {

// Variant that fuses per-128-col-group UE8M0 quantize into
// the consumer epilogue. Output is FP8 + packed UE8M0 scale instead of bf16.
// Eliminates the standalone per_token_group_quantize_fp8 task that today
// runs immediately downstream on the q_b_nope BMM-decode path.
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
  fp8_gemm_dense_qout_common::task_impl_tpl<BN,
                                            NS,
                                            /*NE=*/2,
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

} // namespace fp8_gemm_dense_smallm

namespace fp8_gemm_dense_mediumm {

// See smallm header — same epilogue-fused UE8M0 quantize
// variant, NE=4 TMEM staging.
template <int BN, int NS>
__device__ __noinline__ void fp8_gemm_dense_mediumm_fp8out_sm100_task_impl(
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
  fp8_gemm_dense_qout_common::task_impl_tpl<BN,
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

} // namespace fp8_gemm_dense_mediumm
} // namespace kernel
