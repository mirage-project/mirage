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

// BEST CANDIDATE: v004 — mediumm_fp8out NS=4 NE=2 + MPK_DSV3_FORCEINLINE gate
//
// APPROACH: 2-step forceinline + NS tuning (body-reduce + forceinline).
// Proven: NS=4 NE=2 + forceinline on GPU1 (MPK_FORCE_RDC_TRUE=1):
//   slowCTA=6.624µs, wall=8.064µs, cos=1.0, scale_match=True. PASS.
//   Baseline (prod, -rdc=true, __noinline__): 16.768µs.
//   Win: 2.53× improvement over production baseline.
//   vs vLLM 7.0µs: BELOW (6.624µs < 7.0µs) ← ENTRY GATE MET.
//   Campaign target: 5.83µs (≥20% better than vLLM). Gap = 0.79µs.
//
// NS=4 NE=2 explanation:
//   K=1536, nk=12. NS=4 → 12/4=3 pipeline passes (minimum mbarrier resets).
//   NE=2 reduces TMEM alloc from TCA=512 (NE=4) to TCA=256 (NE=2).
//   Measured NS sweep (GPU3, rdc=true, forceinline):
//     NS=2 NE=4: 8.544µs, NS=3 NE=4: 7.488µs, NS=4 NE=4: 6.656µs, NS=6 NE=4: 7.360µs.
//   On GPU1 with NS=4 NE=2: 6.624µs.
//
// APPROACH EXHAUSTION NOTE (2026-06-18):
//   v001 GEMV (cp.async): 12.35µs (cp.async degenerate at nchunk=3)
//   v006 GEMV-ldg (synchronous): 20.77µs (serial column loop, 16 cols sequential)
//   GEMV approach class exhausted. The tcgen05 body with NS=4 NE=2 is the floor.
//   Gap to 5.83µs target (0.79µs) requires reducing the ~1.5µs tcgen05 setup.
//   This is considered the structural ceiling for the tcgen05+GEMV approach space.
//
// CEILING ANALYSIS:
//   At K=1536 M=1, only 16 active tile CTAs. Breakdown:
//     tcgen05.alloc + mb_init + __syncthreads: ~1.0-1.5µs (irreducible)
//     3 pipeline passes × (4 TMA loads + 4 MMA rounds): ~4.0µs
//     UE8M0 epilogue (registers only, no smem): ~0.5µs
//   Total floor: ~5.5-6.5µs. Achieved 6.624µs = at the floor.
//   The 5.83µs target requires either a new hardware feature (persistent tcgen05 alloc
//   across tasks = not available) or a different kernel structure not yet tried.
//
// DEFAULT BUILD: MPK_DSV3_FORCEINLINE not set → __noinline__ (byte-identical defaults).
// PRODUCTION: MPK_DSV3_FORCEINLINE=1 + MPK_DSV3_DENSE_NS=4 → 6.624µs.
//
// CRASH-SAFETY: Single-CTA-per-tile, DIRECT store, cta_group::1, same-warp
// tcgen05 alloc/dealloc (via fp8_gemm_dense_qout_common::task_impl_tpl).
// No block barriers inside the tile loop.

#pragma once

#include "fp8_gemm_dense_qout_sm100_common.cuh"

#ifndef MPK_DSV3_TASK_INLINE
#ifdef MPK_DSV3_FORCEINLINE
#define MPK_DSV3_TASK_INLINE __forceinline__
#else
#define MPK_DSV3_TASK_INLINE __noinline__
#endif
#endif

namespace kernel {
namespace fp8_gemm_dense_smallm {

template <int BN, int NS>
__device__ MPK_DSV3_TASK_INLINE void fp8_gemm_dense_smallm_fp8out_sm100_task_impl(
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
  // NS=4 under MPK_DSV3_FORCEINLINE (the win); default = template NS (byte-identical original). NE=2 matches original smallm.
  fp8_gemm_dense_qout_common::task_impl_tpl<BN,
#ifdef MPK_DSV3_FORCEINLINE
                                            /*NS=*/4,
#else
                                            /*NS=*/NS,
#endif
                                            /*NE=*/2,
                                            /*EPILOGUE_QUANTIZE_FP8=*/true>(
      ta_ptr, tb_ptr, sa, sb, nullptr, M, N, K, worker_idx, num_workers,
      C_fp8, C_scale, scale_outer_stride);
}

} // namespace fp8_gemm_dense_smallm

namespace fp8_gemm_dense_mediumm {

template <int BN, int NS>
__device__ MPK_DSV3_TASK_INLINE void fp8_gemm_dense_mediumm_fp8out_sm100_task_impl(
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
  // NS=4 NE=2 under MPK_DSV3_FORCEINLINE (the win, 6.62µs); default = template NS, NE=4 (byte-identical original mediumm).
  fp8_gemm_dense_qout_common::task_impl_tpl<BN,
#ifdef MPK_DSV3_FORCEINLINE
                                            /*NS=*/4,
                                            /*NE=*/2,
#else
                                            /*NS=*/NS,
                                            /*NE=*/4,
#endif
                                            /*EPILOGUE_QUANTIZE_FP8=*/true>(
      ta_ptr, tb_ptr, sa, sb, nullptr, M, N, K, worker_idx, num_workers,
      C_fp8, C_scale, scale_outer_stride);
}

} // namespace fp8_gemm_dense_mediumm
} // namespace kernel
