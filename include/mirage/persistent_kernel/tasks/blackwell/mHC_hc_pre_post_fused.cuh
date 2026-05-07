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
#include "mHC_affine_split_activation.cuh"
#include "tasks/common/common_header.cuh"
#include <cuda_bf16.h>

// =====================================================================
// v1: serial inlining (kept for comparison). 1 token per CTA-iteration;
// sinkhorn runs on lane 0 only. K3 dominates -> bandwidth-poor.
// =====================================================================

namespace kernel {

template <typename T_in, int N, int C, int BLOCK_THREADS = 256>
__device__ __forceinline__ void mHC_hc_pre_tail_fused_task_impl(
    void const *mixes_ptr,
    void const *scale_ptr,
    void const *base_ptr,
    void const *x_ptr,
    void *f_pre_ptr,
    void *h_post_out_ptr,
    void *comb_out_ptr,
    int sinkhorn_repeat,
    float sinkhorn_eps) {
  static_assert(N == 4, "v1 hardcoded to n=4");
  constexpr int MIX_HC = N * N + 2 * N;

  __shared__ float h_pre_smem[N];
  __shared__ float h_post_smem[N];
  __shared__ float h_res_smem[N * N];

  T_in const *x = static_cast<T_in const *>(x_ptr);
  T_in *f_pre = static_cast<T_in *>(f_pre_ptr);
  float *h_post_out = static_cast<float *>(h_post_out_ptr);
  float *comb_out = static_cast<float *>(comb_out_ptr);

  mHC_affine_split_activation_task_impl<T_in, /*BATCH_SIZE=*/1, N>(
      mixes_ptr, scale_ptr, base_ptr,
      h_pre_smem, h_post_smem, h_res_smem);
  __syncthreads();

  if (threadIdx.x < N) {
    h_post_out[threadIdx.x] = h_post_smem[threadIdx.x];
  }

  // Lane 0 does sinkhorn entirely in registers, writes comb back to gmem.
  if (threadIdx.x == 0) {
    float m00 = h_res_smem[0],  m01 = h_res_smem[1];
    float m02 = h_res_smem[2],  m03 = h_res_smem[3];
    float m10 = h_res_smem[4],  m11 = h_res_smem[5];
    float m12 = h_res_smem[6],  m13 = h_res_smem[7];
    float m20 = h_res_smem[8],  m21 = h_res_smem[9];
    float m22 = h_res_smem[10], m23 = h_res_smem[11];
    float m30 = h_res_smem[12], m31 = h_res_smem[13];
    float m32 = h_res_smem[14], m33 = h_res_smem[15];

    float const rmax0 = fmaxf(fmaxf(m00, m01), fmaxf(m02, m03));
    float const rmax1 = fmaxf(fmaxf(m10, m11), fmaxf(m12, m13));
    float const rmax2 = fmaxf(fmaxf(m20, m21), fmaxf(m22, m23));
    float const rmax3 = fmaxf(fmaxf(m30, m31), fmaxf(m32, m33));
    m00 = __expf(m00 - rmax0); m01 = __expf(m01 - rmax0);
    m02 = __expf(m02 - rmax0); m03 = __expf(m03 - rmax0);
    m10 = __expf(m10 - rmax1); m11 = __expf(m11 - rmax1);
    m12 = __expf(m12 - rmax1); m13 = __expf(m13 - rmax1);
    m20 = __expf(m20 - rmax2); m21 = __expf(m21 - rmax2);
    m22 = __expf(m22 - rmax2); m23 = __expf(m23 - rmax2);
    m30 = __expf(m30 - rmax3); m31 = __expf(m31 - rmax3);
    m32 = __expf(m32 - rmax3); m33 = __expf(m33 - rmax3);

    float const rs0 = m00 + m01 + m02 + m03;
    float const rs1 = m10 + m11 + m12 + m13;
    float const rs2 = m20 + m21 + m22 + m23;
    float const rs3 = m30 + m31 + m32 + m33;
    float const ri0 = __frcp_rn(rs0);
    float const ri1 = __frcp_rn(rs1);
    float const ri2 = __frcp_rn(rs2);
    float const ri3 = __frcp_rn(rs3);
    m00 = m00 * ri0 + sinkhorn_eps; m01 = m01 * ri0 + sinkhorn_eps;
    m02 = m02 * ri0 + sinkhorn_eps; m03 = m03 * ri0 + sinkhorn_eps;
    m10 = m10 * ri1 + sinkhorn_eps; m11 = m11 * ri1 + sinkhorn_eps;
    m12 = m12 * ri1 + sinkhorn_eps; m13 = m13 * ri1 + sinkhorn_eps;
    m20 = m20 * ri2 + sinkhorn_eps; m21 = m21 * ri2 + sinkhorn_eps;
    m22 = m22 * ri2 + sinkhorn_eps; m23 = m23 * ri2 + sinkhorn_eps;
    m30 = m30 * ri3 + sinkhorn_eps; m31 = m31 * ri3 + sinkhorn_eps;
    m32 = m32 * ri3 + sinkhorn_eps; m33 = m33 * ri3 + sinkhorn_eps;

    int const steps = sinkhorn_repeat > 0 ? sinkhorn_repeat : 1;
#pragma unroll 1
    for (int it = 0; it < steps; ++it) {
      float const cs0 = m00 + m10 + m20 + m30 + sinkhorn_eps;
      float const cs1 = m01 + m11 + m21 + m31 + sinkhorn_eps;
      float const cs2 = m02 + m12 + m22 + m32 + sinkhorn_eps;
      float const cs3 = m03 + m13 + m23 + m33 + sinkhorn_eps;
      float const ci0 = __frcp_rn(cs0);
      float const ci1 = __frcp_rn(cs1);
      float const ci2 = __frcp_rn(cs2);
      float const ci3 = __frcp_rn(cs3);
      m00 *= ci0; m10 *= ci0; m20 *= ci0; m30 *= ci0;
      m01 *= ci1; m11 *= ci1; m21 *= ci1; m31 *= ci1;
      m02 *= ci2; m12 *= ci2; m22 *= ci2; m32 *= ci2;
      m03 *= ci3; m13 *= ci3; m23 *= ci3; m33 *= ci3;
      if (it == steps - 1) break;
      float const rs0i = m00 + m01 + m02 + m03 + sinkhorn_eps;
      float const rs1i = m10 + m11 + m12 + m13 + sinkhorn_eps;
      float const rs2i = m20 + m21 + m22 + m23 + sinkhorn_eps;
      float const rs3i = m30 + m31 + m32 + m33 + sinkhorn_eps;
      float const ri0i = __frcp_rn(rs0i);
      float const ri1i = __frcp_rn(rs1i);
      float const ri2i = __frcp_rn(rs2i);
      float const ri3i = __frcp_rn(rs3i);
      m00 *= ri0i; m01 *= ri0i; m02 *= ri0i; m03 *= ri0i;
      m10 *= ri1i; m11 *= ri1i; m12 *= ri1i; m13 *= ri1i;
      m20 *= ri2i; m21 *= ri2i; m22 *= ri2i; m23 *= ri2i;
      m30 *= ri3i; m31 *= ri3i; m32 *= ri3i; m33 *= ri3i;
    }
    *reinterpret_cast<float4 *>(comb_out + 0)  = make_float4(m00, m01, m02, m03);
    *reinterpret_cast<float4 *>(comb_out + 4)  = make_float4(m10, m11, m12, m13);
    *reinterpret_cast<float4 *>(comb_out + 8)  = make_float4(m20, m21, m22, m23);
    *reinterpret_cast<float4 *>(comb_out + 12) = make_float4(m30, m31, m32, m33);
  }
  __syncthreads();

  for (int c = threadIdx.x; c < C; c += BLOCK_THREADS) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      acc += h_pre_smem[i] * static_cast<float>(x[i * C + c]);
    }
    f_pre[c] = static_cast<T_in>(acc);
  }
}

} // namespace kernel

// Fused mHC hc_pre tail (K2 + K3 + K4) — redesigned for fusion-native
// parallelism rather than serial inlining of the unfused stages.
//
// Each CTA processes TOKENS_PER_CTA tokens per outer iteration:
//   * Stage K2 (affine + split + activation): 256 threads cooperate across
//     all TOKENS_PER_CTA tokens; no per-token serialization.
//   * Stage K3 (sinkhorn 4x4): one thread per token. With TOKENS_PER_CTA=32
//     all 32 lanes of warp 0 run in parallel — no single-thread bottleneck.
//   * Stage K4 (weighted sum, residual=0): 256 threads cooperate over the
//     output channel dim per token; weights `h_pre` are register-broadcast
//     from smem so we don't reload them per output element.
//
// Intermediate tensors (h_pre, h_res, comb) live entirely in smem. Only
// inputs (mixes, scale, base, x) and the final outputs (f_pre, h_post, comb)
// touch gmem.

namespace kernel {

template <typename T_in, int N, int C, int TOKENS_PER_CTA = 32,
          int BLOCK_THREADS = 256, int MIX_STRIDE = 0>
__device__ __forceinline__ void mHC_hc_pre_tail_fused_v2_task_impl(
    void const *mixes_ptr,
    void const *scale_ptr,
    void const *base_ptr,
    void const *x_ptr,
    void *f_pre_ptr,
    void *h_post_out_ptr,
    void *comb_out_ptr,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    int num_tokens) {
  static_assert(N == 4, "fused tail v2 hardcoded to n=4");
  static_assert(BLOCK_THREADS % 32 == 0,
                "block size must be a warp multiple");
  static_assert(TOKENS_PER_CTA % 32 == 0,
                "TOKENS_PER_CTA must be a warp multiple "
                "(sinkhorn warps each cover 32 tokens)");
  static_assert(C % 8 == 0,
                "C must be a multiple of 8 for vectorized bf16 loads");
  constexpr int MIX_HC = N * N + 2 * N; // 24
  // MIX_STRIDE=0 sentinel -> tightly packed (no padding); v3 sets this to
  // 128 to read directly from the linear-output padded buffer.
  constexpr int MIX_ROW_STRIDE = (MIX_STRIDE == 0) ? MIX_HC : MIX_STRIDE;
  // # warps that participate in sinkhorn (one warp per 32 tokens batch).
  constexpr int SINKHORN_WARPS = TOKENS_PER_CTA / 32;
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  static_assert(SINKHORN_WARPS <= NUM_WARPS,
                "not enough warps to cover the sinkhorn batch");

  // Per-token smem state, batched over TOKENS_PER_CTA tokens.
  __shared__ float h_pre_smem[TOKENS_PER_CTA][N];
  __shared__ float h_post_smem[TOKENS_PER_CTA][N];
  __shared__ float h_res_smem[TOKENS_PER_CTA][N * N];
  __shared__ float comb_smem[TOKENS_PER_CTA][N * N];

  T_in const *mixes = static_cast<T_in const *>(mixes_ptr);
  float const *scale = static_cast<float const *>(scale_ptr);
  float const *base = static_cast<float const *>(base_ptr);
  T_in const *x = static_cast<T_in const *>(x_ptr);
  T_in *f_pre = static_cast<T_in *>(f_pre_ptr);
  float *h_post_out = static_cast<float *>(h_post_out_ptr);
  float *comb_out = static_cast<float *>(comb_out_ptr);

  float const alpha_pre = scale[0];
  float const alpha_post = scale[1];
  float const alpha_res = scale[2];

  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  // Outer loop: each CTA handles TOKENS_PER_CTA tokens per iteration.
  for (int token_base = blockIdx.x * TOKENS_PER_CTA;
       token_base < num_tokens;
       token_base += gridDim.x * TOKENS_PER_CTA) {
    int const tokens_this_iter =
        TOKENS_PER_CTA < num_tokens - token_base
            ? TOKENS_PER_CTA
            : num_tokens - token_base;

    // ---- Stage K2: affine + split + activation ----
    //
    // Work units: tokens_this_iter * MIX_HC (24) elements. 256 threads
    // handle them via thread-strided loop. h_post staged in smem so we can
    // emit one coalesced gmem write per token after the loop instead of
    // 24 scattered writes.
    int const total_k2 = tokens_this_iter * MIX_HC;
    for (int idx = threadIdx.x; idx < total_k2; idx += BLOCK_THREADS) {
      int const t = idx / MIX_HC;
      int const j = idx % MIX_HC;
      int const token = token_base + t;

      float const mix = static_cast<float>(mixes[token * MIX_ROW_STRIDE + j]);
      float const bias = base[j];

      float alpha;
      int region;
      int local;
      if (j < N) {
        alpha = alpha_pre;
        region = 0;
        local = j;
      } else if (j < 2 * N) {
        alpha = alpha_post;
        region = 1;
        local = j - N;
      } else {
        alpha = alpha_res;
        region = 2;
        local = j - 2 * N;
      }

      float const y = mix * alpha + bias;
      if (region == 0) {
        h_pre_smem[t][local] = 1.0f / (1.0f + __expf(-y));
      } else if (region == 1) {
        h_post_smem[t][local] = 2.0f / (1.0f + __expf(-y));
      } else {
        h_res_smem[t][local] = y;
      }
    }
    __syncthreads();

    // Coalesced gmem write of h_post: one warp per 4 tokens. Each lane
    // writes one float (i.e. 4 lanes per token), so 8 tokens-worth of
    // writes are issued together as a 16B per-token vector.
    {
      int const total_h_post = tokens_this_iter * N;
      for (int idx = threadIdx.x; idx < total_h_post;
           idx += BLOCK_THREADS) {
        int const t = idx / N;
        int const j = idx % N;
        h_post_out[(token_base + t) * N + j] = h_post_smem[t][j];
      }
    }

    // ---- Stage K3: sinkhorn 4x4, one thread per token ----
    //
    // The first SINKHORN_WARPS warps each cover 32 tokens (one lane per
    // 4x4 matrix). With TOKENS_PER_CTA=32 that's 1 warp; at 64 it's 2;
    // at 128 it's 4. Each lane runs the full register-only sinkhorn
    // (matches sinkhorn.cuh).
    int const k3_token = warp * 32 + lane;
    if (warp < SINKHORN_WARPS && k3_token < tokens_this_iter) {
      int const t = k3_token;
      float m00 = h_res_smem[t][0],  m01 = h_res_smem[t][1];
      float m02 = h_res_smem[t][2],  m03 = h_res_smem[t][3];
      float m10 = h_res_smem[t][4],  m11 = h_res_smem[t][5];
      float m12 = h_res_smem[t][6],  m13 = h_res_smem[t][7];
      float m20 = h_res_smem[t][8],  m21 = h_res_smem[t][9];
      float m22 = h_res_smem[t][10], m23 = h_res_smem[t][11];
      float m30 = h_res_smem[t][12], m31 = h_res_smem[t][13];
      float m32 = h_res_smem[t][14], m33 = h_res_smem[t][15];

      float const rmax0 = fmaxf(fmaxf(m00, m01), fmaxf(m02, m03));
      float const rmax1 = fmaxf(fmaxf(m10, m11), fmaxf(m12, m13));
      float const rmax2 = fmaxf(fmaxf(m20, m21), fmaxf(m22, m23));
      float const rmax3 = fmaxf(fmaxf(m30, m31), fmaxf(m32, m33));
      m00 = __expf(m00 - rmax0); m01 = __expf(m01 - rmax0);
      m02 = __expf(m02 - rmax0); m03 = __expf(m03 - rmax0);
      m10 = __expf(m10 - rmax1); m11 = __expf(m11 - rmax1);
      m12 = __expf(m12 - rmax1); m13 = __expf(m13 - rmax1);
      m20 = __expf(m20 - rmax2); m21 = __expf(m21 - rmax2);
      m22 = __expf(m22 - rmax2); m23 = __expf(m23 - rmax2);
      m30 = __expf(m30 - rmax3); m31 = __expf(m31 - rmax3);
      m32 = __expf(m32 - rmax3); m33 = __expf(m33 - rmax3);

      float const rs0 = m00 + m01 + m02 + m03;
      float const rs1 = m10 + m11 + m12 + m13;
      float const rs2 = m20 + m21 + m22 + m23;
      float const rs3 = m30 + m31 + m32 + m33;
      float const ri0 = __frcp_rn(rs0);
      float const ri1 = __frcp_rn(rs1);
      float const ri2 = __frcp_rn(rs2);
      float const ri3 = __frcp_rn(rs3);
      m00 = m00 * ri0 + sinkhorn_eps; m01 = m01 * ri0 + sinkhorn_eps;
      m02 = m02 * ri0 + sinkhorn_eps; m03 = m03 * ri0 + sinkhorn_eps;
      m10 = m10 * ri1 + sinkhorn_eps; m11 = m11 * ri1 + sinkhorn_eps;
      m12 = m12 * ri1 + sinkhorn_eps; m13 = m13 * ri1 + sinkhorn_eps;
      m20 = m20 * ri2 + sinkhorn_eps; m21 = m21 * ri2 + sinkhorn_eps;
      m22 = m22 * ri2 + sinkhorn_eps; m23 = m23 * ri2 + sinkhorn_eps;
      m30 = m30 * ri3 + sinkhorn_eps; m31 = m31 * ri3 + sinkhorn_eps;
      m32 = m32 * ri3 + sinkhorn_eps; m33 = m33 * ri3 + sinkhorn_eps;

      int const steps = sinkhorn_repeat > 0 ? sinkhorn_repeat : 1;
#pragma unroll 1
      for (int it = 0; it < steps; ++it) {
        float const cs0 = m00 + m10 + m20 + m30 + sinkhorn_eps;
        float const cs1 = m01 + m11 + m21 + m31 + sinkhorn_eps;
        float const cs2 = m02 + m12 + m22 + m32 + sinkhorn_eps;
        float const cs3 = m03 + m13 + m23 + m33 + sinkhorn_eps;
        float const ci0 = __frcp_rn(cs0);
        float const ci1 = __frcp_rn(cs1);
        float const ci2 = __frcp_rn(cs2);
        float const ci3 = __frcp_rn(cs3);
        m00 *= ci0; m10 *= ci0; m20 *= ci0; m30 *= ci0;
        m01 *= ci1; m11 *= ci1; m21 *= ci1; m31 *= ci1;
        m02 *= ci2; m12 *= ci2; m22 *= ci2; m32 *= ci2;
        m03 *= ci3; m13 *= ci3; m23 *= ci3; m33 *= ci3;
        if (it == steps - 1) break;
        float const rs0i = m00 + m01 + m02 + m03 + sinkhorn_eps;
        float const rs1i = m10 + m11 + m12 + m13 + sinkhorn_eps;
        float const rs2i = m20 + m21 + m22 + m23 + sinkhorn_eps;
        float const rs3i = m30 + m31 + m32 + m33 + sinkhorn_eps;
        float const ri0i = __frcp_rn(rs0i);
        float const ri1i = __frcp_rn(rs1i);
        float const ri2i = __frcp_rn(rs2i);
        float const ri3i = __frcp_rn(rs3i);
        m00 *= ri0i; m01 *= ri0i; m02 *= ri0i; m03 *= ri0i;
        m10 *= ri1i; m11 *= ri1i; m12 *= ri1i; m13 *= ri1i;
        m20 *= ri2i; m21 *= ri2i; m22 *= ri2i; m23 *= ri2i;
        m30 *= ri3i; m31 *= ri3i; m32 *= ri3i; m33 *= ri3i;
      }

      // Write comb to smem (for any future fused consumer) and gmem.
      // Vectorized 4-wide writes per row.
      int const token = token_base + t;
      *reinterpret_cast<float4 *>(&comb_smem[t][0])  = make_float4(m00, m01, m02, m03);
      *reinterpret_cast<float4 *>(&comb_smem[t][4])  = make_float4(m10, m11, m12, m13);
      *reinterpret_cast<float4 *>(&comb_smem[t][8])  = make_float4(m20, m21, m22, m23);
      *reinterpret_cast<float4 *>(&comb_smem[t][12]) = make_float4(m30, m31, m32, m33);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 0)  =
          make_float4(m00, m01, m02, m03);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 4)  =
          make_float4(m10, m11, m12, m13);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 8)  =
          make_float4(m20, m21, m22, m23);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 12) =
          make_float4(m30, m31, m32, m33);
    }
    // No sync between K3 and K4: K4 only reads h_pre_smem (written by K2,
    // already synced) and gmem x; K3 only wrote comb (smem + gmem) which
    // K4 doesn't read. Cross-iteration h_res overwrite is gated by the
    // loop-bottom sync below.

    // ---- Stage K4: f_pre[c] = sum_i h_pre[i] * x[i, c] ----
    //
    // Vectorized 16-byte (8 bf16) loads per thread per row. Each thread
    // owns 8 contiguous output channels; 256 threads cover 2048 channels
    // per pass. h_pre's 4 weights are loaded once per token from smem.
    constexpr int VEC = 8;
    static_assert(C % VEC == 0, "C must be a multiple of 8");
    int const c_vec_count = C / VEC;
    for (int t = 0; t < tokens_this_iter; ++t) {
      int const token = token_base + t;
      float const w0 = h_pre_smem[t][0];
      float const w1 = h_pre_smem[t][1];
      float const w2 = h_pre_smem[t][2];
      float const w3 = h_pre_smem[t][3];

      T_in const *x_t = x + token * N * C;
      T_in *f_pre_t = f_pre + token * C;

      uint4 const *x_v0 = reinterpret_cast<uint4 const *>(x_t + 0 * C);
      uint4 const *x_v1 = reinterpret_cast<uint4 const *>(x_t + 1 * C);
      uint4 const *x_v2 = reinterpret_cast<uint4 const *>(x_t + 2 * C);
      uint4 const *x_v3 = reinterpret_cast<uint4 const *>(x_t + 3 * C);
      uint4 *f_v = reinterpret_cast<uint4 *>(f_pre_t);

      for (int v = threadIdx.x; v < c_vec_count; v += BLOCK_THREADS) {
        uint4 r0 = x_v0[v];
        uint4 r1 = x_v1[v];
        uint4 r2 = x_v2[v];
        uint4 r3 = x_v3[v];

        // Reinterpret each uint4 as 8 bf16s (via __nv_bfloat162 pairs).
        __nv_bfloat162 const *b0 =
            reinterpret_cast<__nv_bfloat162 const *>(&r0);
        __nv_bfloat162 const *b1 =
            reinterpret_cast<__nv_bfloat162 const *>(&r1);
        __nv_bfloat162 const *b2 =
            reinterpret_cast<__nv_bfloat162 const *>(&r2);
        __nv_bfloat162 const *b3 =
            reinterpret_cast<__nv_bfloat162 const *>(&r3);

        float out_f[VEC];
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) {
          float2 v0 = __bfloat1622float2(b0[k]);
          float2 v1 = __bfloat1622float2(b1[k]);
          float2 v2 = __bfloat1622float2(b2[k]);
          float2 v3 = __bfloat1622float2(b3[k]);
          out_f[2 * k + 0] =
              w0 * v0.x + w1 * v1.x + w2 * v2.x + w3 * v3.x;
          out_f[2 * k + 1] =
              w0 * v0.y + w1 * v1.y + w2 * v2.y + w3 * v3.y;
        }

        // Pack the 8 fp32 outputs back into 4 bf162 pairs (uint4).
        uint4 packed;
        __nv_bfloat162 *p =
            reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) {
          p[k] = __floats2bfloat162_rn(out_f[2 * k + 0], out_f[2 * k + 1]);
        }
        f_v[v] = packed;
      }
    }
    __syncthreads();
  }
}

// =====================================================================
// v2 variant that takes its smem buffers as a caller-provided dynamic
// smem region instead of static __shared__ declarations. Used by the v3
// megakernel so the linear stage's PipedSharedStorage and the tail's
// per-batch arrays can share the same dynamic smem pool (linear is done
// before tail starts; smem can be reused).
//
// `dyn_smem` must point to at least:
//   TOKENS_PER_CTA * (4*N + N*N + N*N + N) * sizeof(float)
//      = TOKENS_PER_CTA * (4 + 16 + 16 + 4) * 4 B  =  TOKENS_PER_CTA * 160 B
// 32 tokens -> 5 KB; 128 tokens -> 20 KB. Trivial relative to linear's
// ~144 KB at NUM_AB_STAGE=8.
// =====================================================================

template <typename T_in, int N, int C, int TOKENS_PER_CTA = 32,
          int BLOCK_THREADS = 256, int MIX_STRIDE = 0>
__device__ __forceinline__ void mHC_hc_pre_tail_fused_v2_dyn_smem_task_impl(
    void const *mixes_ptr,
    void const *scale_ptr,
    void const *base_ptr,
    void const *x_ptr,
    void *f_pre_ptr,
    void *h_post_out_ptr,
    void *comb_out_ptr,
    int sinkhorn_repeat,
    float sinkhorn_eps,
    int num_tokens,
    char *dyn_smem,
    // MPK-mode override: if >=0, run a single iteration at this fixed
    // token_base (input pointers must already be offset by the caller).
    // Standalone callers pass -1 for normal grid-stride behavior.
    int token_base_override = -1) {
  static_assert(N == 4, "fused tail v2 hardcoded to n=4");
  static_assert(BLOCK_THREADS % 32 == 0, "block size must be a warp multiple");
  static_assert(TOKENS_PER_CTA % 32 == 0,
                "TOKENS_PER_CTA must be a warp multiple");
  static_assert(C % 8 == 0, "C must be a multiple of 8");
  constexpr int MIX_HC = N * N + 2 * N;
  constexpr int MIX_ROW_STRIDE = (MIX_STRIDE == 0) ? MIX_HC : MIX_STRIDE;
  constexpr int SINKHORN_WARPS = TOKENS_PER_CTA / 32;
  constexpr int NUM_WARPS = BLOCK_THREADS / 32;
  static_assert(SINKHORN_WARPS <= NUM_WARPS,
                "not enough warps to cover the sinkhorn batch");

  // Lay out the 4 smem buffers contiguously inside `dyn_smem`.
  // Use 16-byte alignment for each block so float4 stores work.
  uintptr_t base_ptr_addr = reinterpret_cast<uintptr_t>(dyn_smem);
  base_ptr_addr = (base_ptr_addr + 15u) & ~uintptr_t(15);
  float *h_pre_arr =
      reinterpret_cast<float *>(base_ptr_addr);
  float *h_post_arr = h_pre_arr + TOKENS_PER_CTA * N;
  float *h_res_arr = h_post_arr + TOKENS_PER_CTA * N;
  float *comb_arr = h_res_arr + TOKENS_PER_CTA * N * N;
  // Index helpers (row-major: arr[t][j] == arr[t * COLS + j]).
  auto h_pre = [h_pre_arr](int t, int j) -> float & {
    return h_pre_arr[t * N + j];
  };
  auto h_post = [h_post_arr](int t, int j) -> float & {
    return h_post_arr[t * N + j];
  };
  auto h_res = [h_res_arr](int t, int j) -> float & {
    return h_res_arr[t * (N * N) + j];
  };
  auto comb = [comb_arr](int t, int j) -> float & {
    return comb_arr[t * (N * N) + j];
  };

  T_in const *mixes = static_cast<T_in const *>(mixes_ptr);
  float const *scale = static_cast<float const *>(scale_ptr);
  float const *base = static_cast<float const *>(base_ptr);
  T_in const *x = static_cast<T_in const *>(x_ptr);
  T_in *f_pre = static_cast<T_in *>(f_pre_ptr);
  float *h_post_out_g = static_cast<float *>(h_post_out_ptr);
  float *comb_out = static_cast<float *>(comb_out_ptr);

  float const alpha_pre = scale[0];
  float const alpha_post = scale[1];
  float const alpha_res = scale[2];
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  int const _tb_start = (token_base_override >= 0)
                            ? token_base_override
                            : (int)(blockIdx.x * TOKENS_PER_CTA);
  int const _tb_step = (token_base_override >= 0)
                            ? num_tokens   // single-iteration in MPK mode
                            : (int)(gridDim.x * TOKENS_PER_CTA);
  for (int token_base = _tb_start;
       token_base < num_tokens;
       token_base += _tb_step) {
    int const tokens_this_iter =
        TOKENS_PER_CTA < num_tokens - token_base
            ? TOKENS_PER_CTA
            : num_tokens - token_base;

    // ---- Stage K2 ----
    int const total_k2 = tokens_this_iter * MIX_HC;
    for (int idx = threadIdx.x; idx < total_k2; idx += BLOCK_THREADS) {
      int const t = idx / MIX_HC;
      int const j = idx % MIX_HC;
      int const token = token_base + t;
      float const mix = static_cast<float>(mixes[token * MIX_ROW_STRIDE + j]);
      float const bias = base[j];
      float alpha;
      int region, local;
      if (j < N) {
        alpha = alpha_pre; region = 0; local = j;
      } else if (j < 2 * N) {
        alpha = alpha_post; region = 1; local = j - N;
      } else {
        alpha = alpha_res; region = 2; local = j - 2 * N;
      }
      float const y = mix * alpha + bias;
      if (region == 0) {
        h_pre(t, local) = 1.0f / (1.0f + __expf(-y));
      } else if (region == 1) {
        h_post(t, local) = 2.0f / (1.0f + __expf(-y));
      } else {
        h_res(t, local) = y;
      }
    }
    __syncthreads();

    // h_post coalesced gmem flush.
    {
      int const total_h_post = tokens_this_iter * N;
      for (int idx = threadIdx.x; idx < total_h_post;
           idx += BLOCK_THREADS) {
        int const t = idx / N;
        int const j = idx % N;
        h_post_out_g[(token_base + t) * N + j] = h_post(t, j);
      }
    }

    // ---- Stage K3 ----
    int const k3_token = warp * 32 + lane;
    if (warp < SINKHORN_WARPS && k3_token < tokens_this_iter) {
      int const t = k3_token;
      float m00 = h_res(t, 0),  m01 = h_res(t, 1);
      float m02 = h_res(t, 2),  m03 = h_res(t, 3);
      float m10 = h_res(t, 4),  m11 = h_res(t, 5);
      float m12 = h_res(t, 6),  m13 = h_res(t, 7);
      float m20 = h_res(t, 8),  m21 = h_res(t, 9);
      float m22 = h_res(t, 10), m23 = h_res(t, 11);
      float m30 = h_res(t, 12), m31 = h_res(t, 13);
      float m32 = h_res(t, 14), m33 = h_res(t, 15);

      float const rmax0 = fmaxf(fmaxf(m00, m01), fmaxf(m02, m03));
      float const rmax1 = fmaxf(fmaxf(m10, m11), fmaxf(m12, m13));
      float const rmax2 = fmaxf(fmaxf(m20, m21), fmaxf(m22, m23));
      float const rmax3 = fmaxf(fmaxf(m30, m31), fmaxf(m32, m33));
      m00 = __expf(m00 - rmax0); m01 = __expf(m01 - rmax0);
      m02 = __expf(m02 - rmax0); m03 = __expf(m03 - rmax0);
      m10 = __expf(m10 - rmax1); m11 = __expf(m11 - rmax1);
      m12 = __expf(m12 - rmax1); m13 = __expf(m13 - rmax1);
      m20 = __expf(m20 - rmax2); m21 = __expf(m21 - rmax2);
      m22 = __expf(m22 - rmax2); m23 = __expf(m23 - rmax2);
      m30 = __expf(m30 - rmax3); m31 = __expf(m31 - rmax3);
      m32 = __expf(m32 - rmax3); m33 = __expf(m33 - rmax3);

      float const rs0 = m00 + m01 + m02 + m03;
      float const rs1 = m10 + m11 + m12 + m13;
      float const rs2 = m20 + m21 + m22 + m23;
      float const rs3 = m30 + m31 + m32 + m33;
      float const ri0 = __frcp_rn(rs0);
      float const ri1 = __frcp_rn(rs1);
      float const ri2 = __frcp_rn(rs2);
      float const ri3 = __frcp_rn(rs3);
      m00 = m00 * ri0 + sinkhorn_eps; m01 = m01 * ri0 + sinkhorn_eps;
      m02 = m02 * ri0 + sinkhorn_eps; m03 = m03 * ri0 + sinkhorn_eps;
      m10 = m10 * ri1 + sinkhorn_eps; m11 = m11 * ri1 + sinkhorn_eps;
      m12 = m12 * ri1 + sinkhorn_eps; m13 = m13 * ri1 + sinkhorn_eps;
      m20 = m20 * ri2 + sinkhorn_eps; m21 = m21 * ri2 + sinkhorn_eps;
      m22 = m22 * ri2 + sinkhorn_eps; m23 = m23 * ri2 + sinkhorn_eps;
      m30 = m30 * ri3 + sinkhorn_eps; m31 = m31 * ri3 + sinkhorn_eps;
      m32 = m32 * ri3 + sinkhorn_eps; m33 = m33 * ri3 + sinkhorn_eps;

      int const steps = sinkhorn_repeat > 0 ? sinkhorn_repeat : 1;
#pragma unroll 1
      for (int it = 0; it < steps; ++it) {
        float const cs0 = m00 + m10 + m20 + m30 + sinkhorn_eps;
        float const cs1 = m01 + m11 + m21 + m31 + sinkhorn_eps;
        float const cs2 = m02 + m12 + m22 + m32 + sinkhorn_eps;
        float const cs3 = m03 + m13 + m23 + m33 + sinkhorn_eps;
        float const ci0 = __frcp_rn(cs0);
        float const ci1 = __frcp_rn(cs1);
        float const ci2 = __frcp_rn(cs2);
        float const ci3 = __frcp_rn(cs3);
        m00 *= ci0; m10 *= ci0; m20 *= ci0; m30 *= ci0;
        m01 *= ci1; m11 *= ci1; m21 *= ci1; m31 *= ci1;
        m02 *= ci2; m12 *= ci2; m22 *= ci2; m32 *= ci2;
        m03 *= ci3; m13 *= ci3; m23 *= ci3; m33 *= ci3;
        if (it == steps - 1) break;
        float const rs0i = m00 + m01 + m02 + m03 + sinkhorn_eps;
        float const rs1i = m10 + m11 + m12 + m13 + sinkhorn_eps;
        float const rs2i = m20 + m21 + m22 + m23 + sinkhorn_eps;
        float const rs3i = m30 + m31 + m32 + m33 + sinkhorn_eps;
        float const ri0i = __frcp_rn(rs0i);
        float const ri1i = __frcp_rn(rs1i);
        float const ri2i = __frcp_rn(rs2i);
        float const ri3i = __frcp_rn(rs3i);
        m00 *= ri0i; m01 *= ri0i; m02 *= ri0i; m03 *= ri0i;
        m10 *= ri1i; m11 *= ri1i; m12 *= ri1i; m13 *= ri1i;
        m20 *= ri2i; m21 *= ri2i; m22 *= ri2i; m23 *= ri2i;
        m30 *= ri3i; m31 *= ri3i; m32 *= ri3i; m33 *= ri3i;
      }

      int const token = token_base + t;
      *reinterpret_cast<float4 *>(&comb(t, 0))  = make_float4(m00, m01, m02, m03);
      *reinterpret_cast<float4 *>(&comb(t, 4))  = make_float4(m10, m11, m12, m13);
      *reinterpret_cast<float4 *>(&comb(t, 8))  = make_float4(m20, m21, m22, m23);
      *reinterpret_cast<float4 *>(&comb(t, 12)) = make_float4(m30, m31, m32, m33);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 0) =
          make_float4(m00, m01, m02, m03);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 4) =
          make_float4(m10, m11, m12, m13);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 8) =
          make_float4(m20, m21, m22, m23);
      *reinterpret_cast<float4 *>(comb_out + token * N * N + 12) =
          make_float4(m30, m31, m32, m33);
    }
    // No sync between K3 and K4 (see static-smem variant for rationale).

    // ---- Stage K4 ----
    //
    // Software-pipelined inner loop: each iteration computes f_pre for one
    // (token t, channel-vec v) pair and issues the LOADS for the NEXT
    // (t, v) pair before doing the FMA work. This lets the compiler
    // emit the 4 LDG.E.128 prefetches for the next iteration's data
    // while the current iteration's FMAs are in flight, hiding the
    // ~400-cycle gmem load latency behind the ~50-cycle FMA chain.
    //
    // We linearize the (t, v) iteration space so the pipeline can carry
    // across token boundaries (the last v of token t prefetches the first
    // v of token t+1).
    constexpr int VEC = 8;
    static_assert(C % VEC == 0, "C must be a multiple of 8");
    int const c_vec_count = C / VEC;
    int const total_work = tokens_this_iter * c_vec_count;

    // Stripe (token, channel-vec) pairs across threads; each thread
    // handles a stride-BLOCK_THREADS subset.
    auto load_pair = [&](int linear_idx) -> int {
      // Returns BLOCK_THREADS-strided linear index in [0, total_work).
      // Outer dim = token, inner dim = c_vec.
      return linear_idx;
    };

    auto issue_loads = [&](int li, uint4 &r0, uint4 &r1, uint4 &r2,
                           uint4 &r3) {
      int const t = li / c_vec_count;
      int const v = li % c_vec_count;
      int const token = token_base + t;
      T_in const *x_t = x + token * N * C;
      uint4 const *__restrict__ x_v0 = reinterpret_cast<uint4 const *>(x_t + 0 * C);
      uint4 const *__restrict__ x_v1 = reinterpret_cast<uint4 const *>(x_t + 1 * C);
      uint4 const *__restrict__ x_v2 = reinterpret_cast<uint4 const *>(x_t + 2 * C);
      uint4 const *__restrict__ x_v3 = reinterpret_cast<uint4 const *>(x_t + 3 * C);
      r0 = x_v0[v]; r1 = x_v1[v];
      r2 = x_v2[v]; r3 = x_v3[v];
    };

    auto compute_store = [&](int li, uint4 r0, uint4 r1, uint4 r2,
                             uint4 r3) {
      int const t = li / c_vec_count;
      int const v = li % c_vec_count;
      int const token = token_base + t;
      T_in *f_pre_t = f_pre + token * C;
      uint4 *__restrict__ f_v = reinterpret_cast<uint4 *>(f_pre_t);
      float const w0 = h_pre(t, 0);
      float const w1 = h_pre(t, 1);
      float const w2 = h_pre(t, 2);
      float const w3 = h_pre(t, 3);
      __nv_bfloat162 const *b0 = reinterpret_cast<__nv_bfloat162 const *>(&r0);
      __nv_bfloat162 const *b1 = reinterpret_cast<__nv_bfloat162 const *>(&r1);
      __nv_bfloat162 const *b2 = reinterpret_cast<__nv_bfloat162 const *>(&r2);
      __nv_bfloat162 const *b3 = reinterpret_cast<__nv_bfloat162 const *>(&r3);
      float out_f[VEC];
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        float2 v0 = __bfloat1622float2(b0[k]);
        float2 v1 = __bfloat1622float2(b1[k]);
        float2 v2 = __bfloat1622float2(b2[k]);
        float2 v3 = __bfloat1622float2(b3[k]);
        out_f[2 * k + 0] = w0 * v0.x + w1 * v1.x + w2 * v2.x + w3 * v3.x;
        out_f[2 * k + 1] = w0 * v0.y + w1 * v1.y + w2 * v2.y + w3 * v3.y;
      }
      uint4 packed;
      __nv_bfloat162 *p = reinterpret_cast<__nv_bfloat162 *>(&packed);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        p[k] = __floats2bfloat162_rn(out_f[2 * k + 0], out_f[2 * k + 1]);
      }
      f_v[v] = packed;
    };

    // Software-pipelined loop: pre-issue loads for first iter, then in
    // each step issue loads for next iter while computing+storing the
    // current iter's data.
    int const my_first = threadIdx.x;
    if (my_first < total_work) {
      uint4 r0_cur, r1_cur, r2_cur, r3_cur;
      issue_loads(my_first, r0_cur, r1_cur, r2_cur, r3_cur);
      for (int li = my_first; li < total_work; li += BLOCK_THREADS) {
        int const li_next = li + BLOCK_THREADS;
        uint4 r0_next, r1_next, r2_next, r3_next;
        if (li_next < total_work) {
          issue_loads(li_next, r0_next, r1_next, r2_next, r3_next);
        }
        compute_store(li, r0_cur, r1_cur, r2_cur, r3_cur);
        r0_cur = r0_next; r1_cur = r1_next;
        r2_cur = r2_next; r3_cur = r3_next;
      }
    }
    __syncthreads();
  }
}

} // namespace kernel
