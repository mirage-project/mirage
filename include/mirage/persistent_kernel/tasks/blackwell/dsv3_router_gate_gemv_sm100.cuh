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

// BF16 router-gate GEMV: logits[M, N] = hidden[M, K] @ W_gate[N, K]^T
//   M in {1, 4}, K = 7168, N = 256. BF16 in/out, FP32 accum.
// Raw-pointer ABI (not CUtensorMap): task_impl(hidden, W_gate, logits, M, N, K,
//   worker_idx, num_workers); direct uint4 coalesced loads, no
//   TMA/tcgen05/TMEM.
// Grid: worker_idx/num_workers passed explicitly; at 136 workers
// EPC=ceil(N/136),
//   trailing workers with no assigned expert guard on e >= N.
//   The MPK dispatcher MUST set blockDim.x = GEMV_BLOCK = 512.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {
namespace dsv3_router_gate_gemv {

// ---------------------------------------------------------------------------
// Constants (tuned at v002: GRID_N=256, BLOCK=512 is the optimal config)
// ---------------------------------------------------------------------------

static constexpr int GEMV_BLOCK = 512; // threads per CTA (16 warps)
static constexpr int GEMV_N = 256;     // total number of experts
static constexpr int GEMV_K = 7168;    // hidden dimension
static constexpr int GEMV_MAX_M = 4; // maximum supported M (static wred sizing)

// ---------------------------------------------------------------------------
// Main task function
//
// Signature: raw-pointer BF16 GEMV ABI (task.yaml §DROP-IN ABI NOTE).
//
// hidden     — BF16 device pointer for hidden[M, K], row-major.
// W_gate     — BF16 device pointer for W_gate[N, K], row-major (each row is
//              one expert's weight vector; transposed on multiply).
// logits     — BF16 output pointer for logits[M, N], row-major.
// M          — number of active token rows (1 or 4 in production decode).
// N          — number of experts (256 for DeepSeek-V3).
// K          — hidden dimension (7168 for DeepSeek-V3).
// worker_idx — this CTA's persistent-kernel worker index.
// num_workers— total number of persistent-kernel workers (= gridDim.x in
//              standalone at GRID_N=256; MPK runtime value at ~136).
// ---------------------------------------------------------------------------
__device__ __noinline__ void
    dsv3_router_gate_gemv_task_impl(__nv_bfloat16 const *__restrict__ hidden,
                                    __nv_bfloat16 const *__restrict__ W_gate,
                                    __nv_bfloat16 *__restrict__ logits,
                                    int const M,
                                    int const N,
                                    int const K,
                                    int const worker_idx,
                                    int const num_workers) {
  using bf16 = __nv_bfloat16;

  // EPC = experts per CTA (ceiling divide so all N experts are covered).
  // At GRID_N=256, num_workers=256: EPC=1, TPE=512, WPE=16.
  // At num_workers=136 (MPK): EPC=2, TPE=256, WPE=8 (first ~120 workers);
  //   trailing workers guard e >= N below.
  int const EPC = (N + num_workers - 1) / num_workers;
  int const TPE = blockDim.x / EPC; // threads per expert
  int const WPE = TPE / 32;         // warps per expert (>= 1)
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  int const sub = tid / TPE;    // expert index within this CTA (0..EPC-1)
  int const subtid = tid % TPE; // thread index inside sub-group (0..TPE-1)
  int const e = worker_idx * EPC + sub; // global expert index

  // Static shared memory for warp-level partial sums.
  // Layout: wred[warp_idx * GEMV_MAX_M + m]. Sized for GEMV_BLOCK warps *
  // MAX_M. At BLOCK=512: 16 warps * 4 rows = 64 floats = 256 bytes.
  __shared__ float wred[(GEMV_BLOCK / 32) * GEMV_MAX_M];

  if (e >= N) {
    return; // guard trailing workers when N % num_workers != 0
  }

  bf16 const *wrow = W_gate + (size_t)e * K;

  // --- K-reduction with uint4 vectorized BF16 loads --------------------------
  // Each sub-group of TPE threads partitions K into K/8 uint4 loads distributed
  // round-robin across subtid. Each uint4 = 8 BF16 elements.
  // The W_gate row is loaded ONCE and reused for all M hidden rows, amortising
  // the HBM bottleneck for M > 1.

  float acc[GEMV_MAX_M];
#pragma unroll
  for (int m = 0; m < GEMV_MAX_M; ++m) {
    acc[m] = 0.f;
  }

  union {
    uint4 v;
    bf16 h[8];
  } wb, hb;

  for (int base = subtid * 8; base < K; base += TPE * 8) {
    wb.v = *reinterpret_cast<uint4 const *>(wrow + base);
    for (int m = 0; m < M; ++m) {
      hb.v = *reinterpret_cast<uint4 const *>(hidden + (size_t)m * K + base);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        acc[m] += __bfloat162float(hb.h[t]) * __bfloat162float(wb.h[t]);
      }
    }
  }

  // --- Warp-level reduction (lane 0 holds the warp partial sum)
  // ---------------
  for (int m = 0; m < M; ++m) {
    float a = acc[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      a += __shfl_down_sync(0xffffffffu, a, o);
    }
    if (lane == 0) {
      wred[warp * GEMV_MAX_M + m] = a;
    }
  }
  __syncthreads();

  // --- Cross-warp reduction: the first warp of each expert sub-group collects
  //     WPE partial sums (one per warp in the sub-group) via shuffle-reduce.
  if ((warp % WPE) == 0) {
    for (int m = 0; m < M; ++m) {
      float v = (lane < WPE) ? wred[(warp + lane) * GEMV_MAX_M + m] : 0.f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
      }
      if (lane == 0) {
        logits[(size_t)m * N + e] = __float2bfloat16(v);
      }
    }
  }
}

} // namespace dsv3_router_gate_gemv
} // namespace kernel
