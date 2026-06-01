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

// Grouped FP8 GEMM, largem variant — BN=128, NS=6. Default for everything
// outside the smallm niche (K<=4096 OR MPE>8). Body in
// fp8_group_gemm_sm100_common.cuh.

#pragma once

#include "fp8_group_gemm_sm100_common.cuh"
#include "fp8_group_gemm_largem_compact_sm100.cuh" // ferret ws8 compact-dispatch

namespace kernel {
namespace fp8_group_gemm_largem {

constexpr int BN = 128;
constexpr int NS = 6;

__device__ __noinline__ void fp8_group_gemm_largem_sm100_task_impl(
    CUtensorMap const *ta_ptr,
    CUtensorMap const *tb_ptr,
    CUtensorMap const *tsfa_ptr,
    CUtensorMap const *tsfb_ptr,
    CUtensorMap const *td_ptr,
    int const *__restrict__ m_indices,
    int const *__restrict__ active_expert_mask,
    int const M_total,
    int const N,
    int const K,
    int const E,
    int const worker_idx,
    int const num_workers) {
  // COMPACT-DISPATCH (ferret ws8, 2026-06-01): loop only ACTIVE experts instead
  // of all E_local*nn tile slots — at decode ~97% of tiles are idle skips. Drop-
  // in for the common task_impl_tpl<128,6>: identical runtime signature + BN/NS
  // config, same smem, and active_expert_mask==nullptr falls back to all-experts
  // (== legacy behavior, so prefill / no-mask paths are unchanged). Standalone
  // ~1.55x at decode_4active. Measuring the in-MPK per-MoE-layer Δ; the baseline
  // arm is recovered by reverting this file via git.
  fp8_group_gemm_largem_compact::fp8_group_gemm_largem_compact_task_impl(
      ta_ptr,
      tb_ptr,
      tsfa_ptr,
      tsfb_ptr,
      td_ptr,
      m_indices,
      active_expert_mask,
      M_total,
      N,
      K,
      E,
      worker_idx,
      num_workers);
}

inline constexpr int fp8_group_gemm_largem_smem_size() {
  // max() so neither the common nor the compact path can OOB the dynamic smem
  // budget (they are equal in practice; belt-and-suspenders for the JIT path).
  int a = fp8_group_gemm_common::smem_size_tpl<BN, NS>();
  int b = fp8_group_gemm_largem_compact::fp8_group_gemm_largem_compact_smem_size();
  return a > b ? a : b;
}

} // namespace fp8_group_gemm_largem
} // namespace kernel
