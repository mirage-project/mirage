/* Copyright 2025 Mirage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once
#include "../common/utils.cuh"
#include "../common/worker_config.h"
namespace kernel {

// ============================================================================
// assemble_q_decode_sm100
// ============================================================================
//
// Tiny helper task that interleaves the BMM-absorbed Q-nope buffer with the
// Q-pe buffer into the per-head [nope_512|pe_64] layout that the MLA decode
// TMA expects.
//
//   q_nope_abs: (N, H, D_NOPE=512) bf16  ← output of linear_fp8_bmm_sm100
//   q_pe:       (N, H, D_PE=64)    bf16  ← output of q_b_pe FP8 dense GEMM
//   q_nope_pe:  (N, H, D_NOPE+D_PE=576) bf16  ← MLA decode Q input
//
// One CTA per token. Each CTA writes H * 576 bf16 = 18432 elements (DSv3 TP=1)
// or 4608 elements (DSv3 TP=4 with H_local=32). 128 threads * 144 elements per
// thread = trivial wallclock at TP=4; up to ~576 elements/thread at TP=1.
//

template <int H, int D_NOPE, int D_PE>
__device__ __forceinline__ void assemble_q_decode_sm100_task_impl(
    void const *q_nope_abs_ptr,
    void const *q_pe_ptr,
    void *q_nope_pe_ptr,
    int n_active) {
  constexpr int D_TOTAL = D_NOPE + D_PE;
  nv_bfloat16 const *__restrict__ nope_in =
      static_cast<nv_bfloat16 const *>(q_nope_abs_ptr);
  nv_bfloat16 const *__restrict__ pe_in =
      static_cast<nv_bfloat16 const *>(q_pe_ptr);
  nv_bfloat16 *__restrict__ out =
      static_cast<nv_bfloat16 *>(q_nope_pe_ptr);

  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  int const total_per_token = H * D_TOTAL;
#pragma unroll 1
  for (int t = 0; t < n_active; ++t) {
    for (int i = tid; i < total_per_token; i += nthreads) {
      int h = i / D_TOTAL;
      int d = i % D_TOTAL;
      nv_bfloat16 v;
      if (d < D_NOPE) {
        v = nope_in[t * H * D_NOPE + h * D_NOPE + d];
      } else {
        v = pe_in[t * H * D_PE + h * D_PE + (d - D_NOPE)];
      }
      out[t * H * D_TOTAL + h * D_TOTAL + d] = v;
    }
  }
}

} // namespace kernel
