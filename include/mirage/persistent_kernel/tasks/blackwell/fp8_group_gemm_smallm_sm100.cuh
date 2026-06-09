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

// Grouped FP8 GEMM, smallm variant — BN=64, NS=8. Best for K>4096 && MPE<=8
// (gate_up_M{1,4,8}). Body in fp8_group_gemm_sm100_common.cuh.

#pragma once

#include "fp8_group_gemm_sm100_common.cuh"

namespace kernel {
namespace fp8_group_gemm_smallm {

constexpr int BN = 64;
constexpr int NS = 8;

__device__ __noinline__ void fp8_group_gemm_smallm_sm100_task_impl(
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
  fp8_group_gemm_common::task_impl_tpl<BN, NS>(ta_ptr,
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

inline constexpr int fp8_group_gemm_smallm_smem_size() {
  return fp8_group_gemm_common::smem_size_tpl<BN, NS>();
}

} // namespace fp8_group_gemm_smallm
} // namespace kernel
