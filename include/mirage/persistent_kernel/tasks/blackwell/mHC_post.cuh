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
#pragma once
#include "tasks/common/common_header.cuh"
#include <cuda_bf16.h>

// mHC post: HC Post + Residual Fusion.
//
// y[k, c] = post[k] * x[c] + sum_i comb[i, k] * residual[i, c]
//
// `comb` is the Sinkhorn-normalized combination matrix from hc_pre. The
// contraction sums over the FIRST (row) index i -- i.e. column k of comb --
// matching the torch model's hc_post (NOT transposed). Output channel k is
// formed from all input residual channels i weighted by comb[i, k].
//
// Reads: residual[NUM_TOPK, OUTPUT_SIZE], x[OUTPUT_SIZE], post[NUM_TOPK],
//        comb[NUM_TOPK*NUM_TOPK].
// Writes: y[NUM_TOPK, OUTPUT_SIZE].
//
// Per output row k, the residual is the rank-1 outer product post[k] * x[c]
// (computed inline) rather than a precomputed dense buffer; avoids the (n*C)
// intermediate read/write.

namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int NUM_TOPK,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void
    mHC_post_task_impl(void const *residual_ptr,
                       void const *x_ptr,
                       void const *comb_ptr,
                       void const *post_ptr,
                       void *output_ptr,
                       // Thread-group remap: `tid`/`nthreads` default to the
                       // whole block, but a multi-token kernel passes the
                       // sub-group's lane and size so several tokens share one
                       // block. Caller offsets the pointers to its token.
                       int tid = -1,
                       int nthreads = 0) {
  int const t_id = (tid >= 0) ? tid : (int)threadIdx.x;
  int const n_threads = (nthreads > 0) ? nthreads : (int)blockDim.x;
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  float const *__restrict__ d_comb = static_cast<float const *>(comb_ptr);
  float const *__restrict__ d_post = static_cast<float const *>(post_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // Channel-major + vectorized: each thread processes VEC=8 contiguous channels
  // per step via 128-bit (uint4) loads/stores, computing all NUM_TOPK outputs
  // from a single load of each residual vec. `residual` is read exactly once
  // (the old k-major loop re-read it NUM_TOPK x); comb/post are hoisted to
  // registers per row. VEC*sizeof(bf16) = 16 B -> one LDG.E.128 / STG.E.128.
  constexpr int VEC = 8; // 8 bf16 = 16 B = uint4
  static_assert(OUTPUT_SIZE % VEC == 0, "OUTPUT_SIZE must be a multiple of 8");
  int const c_vec_count = OUTPUT_SIZE / VEC;

  for (int row_idx = 0; row_idx < BATCH_SIZE; ++row_idx) {
    float comb_reg[NUM_TOPK * NUM_TOPK];
    float post_reg[NUM_TOPK];
#pragma unroll
    for (int e = 0; e < NUM_TOPK * NUM_TOPK; ++e) {
      comb_reg[e] = d_comb[row_idx * NUM_TOPK * NUM_TOPK + e];
    }
#pragma unroll
    for (int k = 0; k < NUM_TOPK; ++k) {
      post_reg[k] = d_post[row_idx * NUM_TOPK + k];
    }

    T const *__restrict__ res_row =
        d_residual + (int64_t)row_idx * OUTPUT_STRIDE * NUM_TOPK;
    T const *__restrict__ x_row = d_x + (int64_t)row_idx * OUTPUT_STRIDE;
    T *__restrict__ out_row =
        d_output + (int64_t)row_idx * OUTPUT_STRIDE * NUM_TOPK;

    for (int v = t_id; v < c_vec_count; v += n_threads) {
      // Load NUM_TOPK residual vecs (8 channels each) + the x vec, once.
      uint4 r_raw[NUM_TOPK];
#pragma unroll
      for (int t = 0; t < NUM_TOPK; ++t) {
        r_raw[t] = reinterpret_cast<uint4 const *>(res_row + t * OUTPUT_STRIDE)[v];
      }
      uint4 x_raw = reinterpret_cast<uint4 const *>(x_row)[v];

      // Unpack to fp32: r_f[t][e], x_f[e] for the VEC lanes.
      float r_f[NUM_TOPK][VEC];
      float x_f[VEC];
#pragma unroll
      for (int t = 0; t < NUM_TOPK; ++t) {
        __nv_bfloat162 const *b =
            reinterpret_cast<__nv_bfloat162 const *>(&r_raw[t]);
#pragma unroll
        for (int e2 = 0; e2 < VEC / 2; ++e2) {
          float2 f = __bfloat1622float2(b[e2]);
          r_f[t][2 * e2 + 0] = f.x;
          r_f[t][2 * e2 + 1] = f.y;
        }
      }
      {
        __nv_bfloat162 const *bx =
            reinterpret_cast<__nv_bfloat162 const *>(&x_raw);
#pragma unroll
        for (int e2 = 0; e2 < VEC / 2; ++e2) {
          float2 f = __bfloat1622float2(bx[e2]);
          x_f[2 * e2 + 0] = f.x;
          x_f[2 * e2 + 1] = f.y;
        }
      }

      // Compute all NUM_TOPK outputs for the VEC lanes; pack + store as uint4.
#pragma unroll
      for (int k = 0; k < NUM_TOPK; ++k) {
        float out_f[VEC];
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          float s = post_reg[k] * x_f[e];
#pragma unroll
          for (int t = 0; t < NUM_TOPK; ++t) {
            // comb[t, k] (row t, col k): untransposed torch hc_post convention.
            s += r_f[t][e] * comb_reg[t * NUM_TOPK + k];
          }
          out_f[e] = s;
        }
        uint4 packed;
        __nv_bfloat162 *p = reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
        for (int e2 = 0; e2 < VEC / 2; ++e2) {
          p[e2] = __floats2bfloat162_rn(out_f[2 * e2 + 0], out_f[2 * e2 + 1]);
        }
        reinterpret_cast<uint4 *>(out_row + k * OUTPUT_STRIDE)[v] = packed;
      }
    }
  }
}

} // namespace kernel
