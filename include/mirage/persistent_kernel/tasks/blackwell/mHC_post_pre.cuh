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

// mHC fused post -> next-layer pre (CUDA-core), in two kernels:
//
//   mHC_post_pre_k1 : hc_post of the current layer + the next layer's prenorm
//                     GEMM + sqrsum, fused so the post output never round-trips
//                     through gmem. Mirrors vLLM's mhc_fused_tilelang.
//                       new_r[j,h]        = post[j]*x[h] + sum_k
//                       comb[k,j]*res[k,h] residual_out[j,h] = new_r[j,h] (next
//                       residual) sqrsum[t]        += sum new_r^2 (RMS denom)
//                       mixes[t,o]       += sum_jh fn[o,j,h]*new_r (= y @ fn.T)
//                     plus a split-k reduce folding partials ->
//                     mixes_pad+sqrsum.
//
//   mHC_post_pre_k2 : the pre tail (RMS-fold + pre/post sigmoid affines +
//                     sinkhorn(4x4) + pre_mix-weighted residual sum) ->
//                     f_pre / h_post / comb for the next layer. Forwards to the
//                     shared mHC_pre_k2 implementation (identical math).
//
// k1's GEMM is a thread-level FMA loop: each thread owns a contiguous hidden
// slice (h in [0,C)) for one token, computes new_r for all N heads in registers
// and contracts against fn. mix_hc (=24) is too narrow for tensor cores, so
// this matches vLLM's CUDA-core choice, avoids the pad-to-128 waste, keeps smem
// tiny, and uses split-k over the hidden dim to fill the grid at low token
// counts.
//
// These are two separate kernel launches (k1 writes mixes_pad+sqrsum to gmem;
// k2 consumes them), since the GEMM and the sinkhorn tail have different grid /
// thread shapes.

#include "blackwell/mHC_pre.cuh"
#include <cuda_bf16.h>

namespace kernel {

// ---- Stage k1: post + prenorm GEMM + sqrsum ----
// TPB tokens share one block; fn[o,j,h] is loaded once and reused across all
// TPB tokens, dividing the dominant fn L2->SM traffic by TPB. Mirrors the
// pre_k1_cuda_core TPB design.
template <typename T,
          int N,      // hc_mult (4)
          int C,      // hidden_size per head
          int MIX_HC, // N*N + 2*N (24 for N=4)
          int BLOCK_THREADS,
          int SPLIT_K,
          int MIX_PAD = 128,
          int TPB = 1> // tokens processed per CTA
__device__ __forceinline__ void mHC_post_pre_k1_task_impl(
    T const *__restrict__ residual,       // [tokens, N, C]
    T const *__restrict__ x,              // [tokens, C]
    float const *__restrict__ comb,       // [tokens, N, N]
    float const *__restrict__ post,       // [tokens, N]
    __nv_bfloat16 const *__restrict__ fn, // [MIX_HC, N, C]  (weight_t, bf16)
    T *__restrict__ residual_out,         // [tokens, N, C]
    float *__restrict__ out_partial,      // [SPLIT_K, tokens, MIX_HC] (SK>1)
    float *__restrict__ sqr_partial,      // [SPLIT_K, tokens]         (SK>1)
    void *__restrict__ mixes_pad,         // [tokens, MIX_PAD] bf16    (SK==1)
    float *__restrict__ sqrsum,           // [tokens]                  (SK==1)
    int num_tokens,
    int token0, // first token of this CTA's TPB-group
    int i_ks) {
  constexpr bool DIRECT = (SPLIT_K == 1);
  constexpr int C_PER_SPLIT = C / SPLIT_K;
  static_assert(C % SPLIT_K == 0, "C must be divisible by SPLIT_K");
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp_id = tid >> 5;
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;

  int const ntok = (num_tokens - token0) < TPB ? (num_tokens - token0) : TPB;

  // Hoist post / comb for all TPB tokens into registers.
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

  // Per-token accumulators. fn loaded once per h, reused across all TPB tokens.
  float acc[TPB][MIX_HC];
  float sqr[TPB];
#pragma unroll
  for (int t = 0; t < TPB; ++t) {
    sqr[t] = 0.0f;
#pragma unroll
    for (int o = 0; o < MIX_HC; ++o) {
      acc[t][o] = 0.0f;
    }
  }

  int const h_base = i_ks * C_PER_SPLIT;

  for (int it = tid; it < C_PER_SPLIT; it += BLOCK_THREADS) {
    int const h = h_base + it;

    // Load fn[o,j,h] once for this h; will be reused by all TPB tokens.
    // fn layout: [MIX_HC, N, C] flattened -> fn[o*N*C + j*C + h].
    float fn_oj[MIX_HC][N];
#pragma unroll
    for (int o = 0; o < MIX_HC; ++o) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        fn_oj[o][j] = __bfloat162float(fn[((int64_t)o * N + j) * C + h]);
      }
    }

    // Per-token: compute new_r, write residual_out, accumulate sqr and mixes.
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
      if (t < ntok) {
        int const tok = token0 + t;
        T const *res_t = residual + (int64_t)tok * N * C;
        T const *x_t = x + (int64_t)tok * C;
        T *res_out_t = residual_out + (int64_t)tok * N * C;

        float const xv = static_cast<float>(x_t[h]);
        float rk[N];
#pragma unroll
        for (int kk = 0; kk < N; ++kk) {
          rk[kk] = static_cast<float>(res_t[kk * C + h]);
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
          res_out_t[j * C + h] = static_cast<T>(v);
          sqr[t] += v * v;
        }
#pragma unroll
        for (int o = 0; o < MIX_HC; ++o) {
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

  // Warp-reduce each token's accumulators.
#pragma unroll
  for (int t = 0; t < TPB; ++t) {
#pragma unroll
    for (int o = 0; o < MIX_HC; ++o) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc[t][o] += __shfl_xor_sync(0xffffffff, acc[t][o], off);
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sqr[t] += __shfl_xor_sync(0xffffffff, sqr[t], off);
    }
  }

  __shared__ float s_acc[TPB][NUM_WARPS][MIX_HC];
  __shared__ float s_sqr[TPB][NUM_WARPS];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TPB; ++t) {
#pragma unroll
      for (int o = 0; o < MIX_HC; ++o) {
        s_acc[t][warp_id][o] = acc[t][o];
      }
      s_sqr[t][warp_id] = sqr[t];
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
      if (lane < MIX_HC) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
          v += s_acc[t][w][lane];
        }
        if (DIRECT) {
          mixes[(int64_t)token * MIX_PAD + lane] = __float2bfloat16(v);
        } else {
          out_partial[((int64_t)i_ks * num_tokens + token) * MIX_HC + lane] = v;
        }
      }
      if (lane == 0) {
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

// k1 split-k reduce: fold SPLIT_K partials -> mixes_pad (bf16, padded to
// MIX_PAD cols) + sqrsum. Launched as one block per token with >= MIX_HC
// threads.
template <int N, int MIX_HC, int MIX_PAD, int SPLIT_K>
__device__ __forceinline__ void mHC_post_pre_k1_reduce_impl(
    float const *__restrict__ out_partial, // [SPLIT_K, tokens, MIX_HC]
    float const *__restrict__ sqr_partial, // [SPLIT_K, tokens]
    void *__restrict__ mixes_pad,          // [tokens, MIX_PAD] bf16
    float *__restrict__ sqrsum,            // [tokens]
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

// ---- Stage k2: pre tail (RMS-fold + affines + sinkhorn + weighted sum) ----
// The math is identical to the standalone pre tail, so this forwards to the
// shared mHC_pre_k2 implementation. Consumes k1's mixes_pad + sqrsum and the
// next-layer residual (residual_out from k1), producing f_pre/h_post/comb.
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

} // namespace kernel
