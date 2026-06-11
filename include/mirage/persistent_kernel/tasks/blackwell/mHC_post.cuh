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
                       int tid = -1,
                       int nthreads = 0) {
  int const t_id = (tid >= 0) ? tid : (int)threadIdx.x;
  int const n_threads = (nthreads > 0) ? nthreads : (int)blockDim.x;
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T const *__restrict__ d_x = static_cast<T const *>(x_ptr);
  float const *__restrict__ d_comb = static_cast<float const *>(comb_ptr);
  float const *__restrict__ d_post = static_cast<float const *>(post_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  constexpr int VEC = 8;
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
      uint4 r_raw[NUM_TOPK];
#pragma unroll
      for (int t = 0; t < NUM_TOPK; ++t) {
        r_raw[t] =
            reinterpret_cast<uint4 const *>(res_row + t * OUTPUT_STRIDE)[v];
      }
      uint4 x_raw = reinterpret_cast<uint4 const *>(x_row)[v];

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

#pragma unroll
      for (int k = 0; k < NUM_TOPK; ++k) {
        float out_f[VEC];
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          float s = post_reg[k] * x_f[e];
#pragma unroll
          for (int t = 0; t < NUM_TOPK; ++t) {
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

}
