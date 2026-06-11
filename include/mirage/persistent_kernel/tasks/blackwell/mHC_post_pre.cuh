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

#include "mHC_pre.cuh"
#include <cuda_bf16.h>

namespace kernel {

template <typename T,
          int N,
          int C,
          int MIX_HC,
          int BLOCK_THREADS,
          int SPLIT_K,
          int MIX_PAD = 128,
          int TPB = 1,
          int TILE_N = MIX_HC>
__device__ __forceinline__ void mHC_post_pre_k1_task_impl(
    T const *__restrict__ residual,
    T const *__restrict__ x,
    float const *__restrict__ comb,
    float const *__restrict__ post,
    __nv_bfloat16 const *__restrict__ fn,
    T *__restrict__ residual_out,
    float *__restrict__ out_partial,
    float *__restrict__ sqr_partial,
    void *__restrict__ mixes_pad,
    float *__restrict__ sqrsum,
    int num_tokens,
    int token0,
    int i_ks,
    int i_nt) {
  constexpr bool DIRECT = (SPLIT_K == 1);
  constexpr int C_PER_SPLIT = C / SPLIT_K;
  static_assert(C % SPLIT_K == 0, "C must be divisible by SPLIT_K");
  static_assert(MIX_HC % TILE_N == 0, "MIX_HC must be divisible by TILE_N");
  bool const owns_side = (i_nt == 0);
  int const o_base = i_nt * TILE_N;
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp_id = tid >> 5;
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;

  int const ntok = (num_tokens - token0) < TPB ? (num_tokens - token0) : TPB;

  float pm[TPB][N];
  float cm[TPB][N][N];
#pragma unroll
  for (int t = 0; t < TPB; ++t) {
    if (t < ntok) {
      int const tok = token0 + t;
#pragma unroll
      for (int j = 0; j < N; ++j) {
        pm[t][j] = post[tok * N + j];
      }
#pragma unroll
      for (int kk = 0; kk < N; ++kk) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
          cm[t][kk][j] = comb[tok * N * N + kk * N + j];
        }
      }
    }
  }

  float acc[TPB][TILE_N];
  float sqr[TPB];
#pragma unroll
  for (int t = 0; t < TPB; ++t) {
    sqr[t] = 0.0f;
#pragma unroll
    for (int o = 0; o < TILE_N; ++o) {
      acc[t][o] = 0.0f;
    }
  }

  int const h_base = i_ks * C_PER_SPLIT;

  for (int it = tid; it < C_PER_SPLIT; it += BLOCK_THREADS) {
    int const h = h_base + it;

    float fn_oj[TILE_N][N];
#pragma unroll
    for (int o = 0; o < TILE_N; ++o) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        fn_oj[o][j] = __bfloat162float(fn[((int64_t)(o_base + o) * N + j) * C + h]);
      }
    }

#pragma unroll
    for (int t = 0; t < TPB; ++t) {
      if (t < ntok) {
        int const tok = token0 + t;
        T const *res_t = residual + (int64_t)tok * N * C;
        T const *x_t = x + (int64_t)tok * C;
        T *res_out_t = residual_out + (int64_t)tok * N * C;

        float const xv = __bfloat162float(x_t[h]);
        float rk[N];
#pragma unroll
        for (int kk = 0; kk < N; ++kk) {
          rk[kk] = __bfloat162float(res_t[kk * C + h]);
        }
        float new_r[N];
#pragma unroll
        for (int j = 0; j < N; ++j) {
          float v = pm[t][j] * xv;
#pragma unroll
          for (int kk = 0; kk < N; ++kk) {
            v += cm[t][kk][j] * rk[kk];
          }
          new_r[j] = v;
          if (owns_side) {
            res_out_t[j * C + h] = __float2bfloat16(v);
            sqr[t] += v * v;
          }
        }
#pragma unroll
        for (int o = 0; o < TILE_N; ++o) {
          float s = 0.0f;
#pragma unroll
          for (int j = 0; j < N; ++j) {
            s += fn_oj[o][j] * new_r[j];
          }
          acc[t][o] += s;
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TPB; ++t) {
#pragma unroll
    for (int o = 0; o < TILE_N; ++o) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc[t][o] += __shfl_xor_sync(0xffffffff, acc[t][o], off);
      }
    }
    if (owns_side) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        sqr[t] += __shfl_xor_sync(0xffffffff, sqr[t], off);
      }
    }
  }

  __shared__ float s_acc[TPB][NUM_WARPS][TILE_N];
  __shared__ float s_sqr[TPB][NUM_WARPS];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
#pragma unroll
      for (int o = 0; o < TILE_N; ++o) {
        s_acc[t][warp_id][o] = acc[t][o];
      }
      if (owns_side) {
        s_sqr[t][warp_id] = sqr[t];
      }
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    __nv_bfloat16 *mixes = static_cast<__nv_bfloat16 *>(mixes_pad);
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
      if (t >= ntok) {
        continue;
      }
      int const token = token0 + t;
      if (lane < TILE_N) {
        int const o = o_base + lane;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          v += s_acc[t][w][lane];
        }
        if (DIRECT) {
          mixes[(int64_t)token * MIX_PAD + o] = __float2bfloat16(v);
        } else {
          out_partial[((int64_t)i_ks * num_tokens + token) * MIX_HC + o] = v;
        }
      }
      if (owns_side && lane == 0) {
        float v2 = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          v2 += s_sqr[t][w];
        }
        if (DIRECT) {
          sqrsum[token] = v2;
        } else {
          sqr_partial[(int64_t)i_ks * num_tokens + token] = v2;
        }
      }
    }
  }
}

template <int N, int MIX_HC, int MIX_PAD, int SPLIT_K>
__device__ __forceinline__ void mHC_post_pre_k1_reduce_impl(
    float const *__restrict__ out_partial,
    float const *__restrict__ sqr_partial,
    void *__restrict__ mixes_pad,
    float *__restrict__ sqrsum,
    int num_tokens,
    int token) {
  __nv_bfloat16 *mixes = static_cast<__nv_bfloat16 *>(mixes_pad);
  int const o = threadIdx.x;
  if (o < MIX_HC) {
    float v = 0.0f;
#pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
      v += out_partial[((int64_t)s * num_tokens + token) * MIX_HC + o];
    }
    mixes[(int64_t)token * MIX_PAD + o] = __float2bfloat16(v);
  }
  if (o == 0) {
    float sq = 0.0f;
#pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
      sq += sqr_partial[(int64_t)s * num_tokens + token];
    }
    sqrsum[token] = sq;
  }
}

template <typename T_in,
          int N,
          int C,
          int RMS_HIDDEN,
          int TOKENS_PER_CTA = 32,
          int BLOCK_THREADS = 256,
          int MIX_STRIDE = 0>
__device__ __forceinline__ void
    mHC_post_pre_k2_task_impl(void const *mixes_ptr,
                              void const *sqrsum_ptr,
                              void const *scale_ptr,
                              void const *base_ptr,
                              void const *x_ptr,
                              void *f_pre_ptr,
                              void *h_post_out_ptr,
                              void *comb_out_ptr,
                              int sinkhorn_repeat,
                              float sinkhorn_eps,
                              float rms_eps,
                              int num_tokens,
                              char *dyn_smem,
                              int token_base_override = -1) {
  mHC_pre_k2_task_impl<T_in,
                       N,
                       C,
                       RMS_HIDDEN,
                       TOKENS_PER_CTA,
                       BLOCK_THREADS,
                       MIX_STRIDE>(mixes_ptr,
                                   sqrsum_ptr,
                                   scale_ptr,
                                   base_ptr,
                                   x_ptr,
                                   f_pre_ptr,
                                   h_post_out_ptr,
                                   comb_out_ptr,
                                   sinkhorn_repeat,
                                   sinkhorn_eps,
                                   rms_eps,
                                   num_tokens,
                                   dyn_smem,
                                   token_base_override);
}

}
