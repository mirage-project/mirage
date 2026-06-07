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
#pragma once
#include "tasks/common/common_header.cuh"

// DeepSeek-V3 router-gate GEMV (decode, M<=16). Ported from the proven
// TensorRT-LLM / vLLM / SGLang CUDA-core router kernel:
//   TRT-LLM cpp/.../dsv3MinLatencyKernels/dsv3RouterGemm.cu
//   vLLM    csrc/moe/dsv3_router_gemm_bf16_out.cu
//
// The router gate is a skinny GEMM: out[M, N_TOTAL] = act[M, K] @ wt[N_TOTAL, K]^T
// with N_TOTAL = NUM_EXPERTS (256), K = HIDDEN (7168), M = decode tokens (<=16).
// A tensor-core GEMM wastes ~all its MMA at M=1 (the in-tree split-K swap-AB
// router is ~12.6us, ~4x vLLM's ~3us). This CUDA-core GEMV is memory-bound and
// crash-free by construction: NO tensor core / tcgen05 / TMA / cross-CTA reduce
// (so it cannot hit the multi-rank split-K tcgen05-concurrency crash, INDEX
// #183/#186). Each task instance (CTA) owns N_PER_CTA expert columns and does
// the full-K reduction in-CTA (128 threads split K via vectorized loads + a
// warp-butterfly + cross-warp smem reduction).
//
// Pointer convention (set by the runtime via dim_maps, mirroring swap-AB):
//   act_ptr : [M, K]          activation, REPLICATED to every instance (stride K).
//   wt_ptr  : [N_PER_CTA, K]  this instance's expert weight rows (weight dim0 ->
//                             grid.x, so the ptr is pre-sliced; stride K).
//   out_ptr : base pre-offset to this instance's column block of [M, N_TOTAL];
//             write out[m * N_TOTAL + e] for e in [0, N_PER_CTA).

namespace kernel {

template <typename T, int M, int N_PER_CTA, int K, int N_TOTAL>
__device__ __forceinline__ void
    dsv3_router_gemm_task_impl(void const *act_ptr,
                               void const *wt_ptr,
                               void *out_ptr) {
  constexpr int VPT = 16 / sizeof(T); // 8 bf16 per 16B (uint4) vector load
  static_assert(K % VPT == 0, "K must be divisible by the vector width");
  static_assert(N_PER_CTA >= 1, "N_PER_CTA must be >= 1");

  T const *__restrict__ act = static_cast<T const *>(act_ptr);
  T const *__restrict__ wt = static_cast<T const *>(wt_ptr);
  T *__restrict__ out = static_cast<T *>(out_ptr);

  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  int const laneId = tid & 31;
  int const warpId = tid >> 5;
  int const nWarps = (nthreads + 31) >> 5;

  // Per-(token, local-expert) accumulator held in registers.
  float acc[M][N_PER_CTA];
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int e = 0; e < N_PER_CTA; ++e) {
      acc[m][e] = 0.0f;
    }
  }

  // Strided K-loop: thread `tid` covers VPT-element chunks at tid*VPT,
  // +nthreads*VPT, ... Robust to any blockDim.x (idle threads contribute 0).
  for (int k = tid * VPT; k < K; k += nthreads * VPT) {
    float wf[N_PER_CTA][VPT];
#pragma unroll
    for (int e = 0; e < N_PER_CTA; ++e) {
      uint4 wv = *reinterpret_cast<uint4 const *>(wt + e * K + k);
      T const *wb = reinterpret_cast<T const *>(&wv);
#pragma unroll
      for (int j = 0; j < VPT; ++j) {
        wf[e][j] = static_cast<float>(wb[j]);
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      uint4 av = *reinterpret_cast<uint4 const *>(act + m * K + k);
      T const *ab = reinterpret_cast<T const *>(&av);
      float af[VPT];
#pragma unroll
      for (int j = 0; j < VPT; ++j) {
        af[j] = static_cast<float>(ab[j]);
      }
#pragma unroll
      for (int e = 0; e < N_PER_CTA; ++e) {
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
          acc[m][e] += af[j] * wf[e][j];
        }
      }
    }
  }

  // Cross-thread reduction: warp butterfly then cross-warp via smem.
  // Static smem stays small (<= M*N_PER_CTA*8 floats); for M=16,N_PER_CTA=2 that
  // is 1 KiB, well under the per-worker static budget.
  __shared__ float sm[M * N_PER_CTA][8];
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int e = 0; e < N_PER_CTA; ++e) {
      float v = acc[m][e];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
      }
      if (laneId == 0) {
        sm[m * N_PER_CTA + e][warpId] = v;
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int e = 0; e < N_PER_CTA; ++e) {
        float s = 0.0f;
        for (int w = 0; w < nWarps; ++w) {
          s += sm[m * N_PER_CTA + e][w];
        }
        out[m * N_TOTAL + e] = static_cast<T>(s);
      }
    }
  }
}

} // namespace kernel
