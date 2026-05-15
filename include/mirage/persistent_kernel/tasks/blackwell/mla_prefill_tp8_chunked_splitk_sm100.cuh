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

// Split-K + Reduce companion to mla_prefill_tp8_chunked_sm100.cuh.
// Used when chunk * H * B is too small to fill the GPU with the BM=64 main
// path: each block does only 1/num_splits of the KV range, writing per-row
// partial output (D_V floats + m + d) to a global float buffer; the reduce
// kernel then combines the partials into the final bf16 O.
//
// Same DeepSeek V3 unabsorbed dimensions as the main kernel:
//   K_nope per-head (3D TMA, head*2+half), K_rope shared (2D TMA),
//   V per-head (3D TMA, head*2+half).

#pragma once

#include "mla_prefill_tp8_chunked_sm100.cuh"

namespace kernel {
namespace mla_prefill_tp8_chunked_splitk {

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

using kernel::mla_prefill_tp8_chunked::BM;
using kernel::mla_prefill_tp8_chunked::BN;
using kernel::mla_prefill_tp8_chunked::cdiv;
using kernel::mla_prefill_tp8_chunked::cpa;
using kernel::mla_prefill_tp8_chunked::cpa_commit;
using kernel::mla_prefill_tp8_chunked::cpa_wait;
using kernel::mla_prefill_tp8_chunked::D_QK;
using kernel::mla_prefill_tp8_chunked::D_QK_NOPE;
using kernel::mla_prefill_tp8_chunked::D_QK_ROPE;
using kernel::mla_prefill_tp8_chunked::D_V;
using kernel::mla_prefill_tp8_chunked::do_mask_softmax;
using kernel::mla_prefill_tp8_chunked::do_pv_half;
using kernel::mla_prefill_tp8_chunked::do_qk_half;
using kernel::mla_prefill_tp8_chunked::HALF_N;
using kernel::mla_prefill_tp8_chunked::KN0_OFF;
using kernel::mla_prefill_tp8_chunked::KN1_OFF;
using kernel::mla_prefill_tp8_chunked::KP_OFF;
using kernel::mla_prefill_tp8_chunked::mbar_init_1;
using kernel::mla_prefill_tp8_chunked::mbar_tx;
using kernel::mla_prefill_tp8_chunked::mbar_wait_1;
using kernel::mla_prefill_tp8_chunked::MBK_OFF;
using kernel::mla_prefill_tp8_chunked::MBV_OFF;
using kernel::mla_prefill_tp8_chunked::NMDV;
using kernel::mla_prefill_tp8_chunked::NT;
using kernel::mla_prefill_tp8_chunked::Q_NOPE_SZ;
using kernel::mla_prefill_tp8_chunked::SMEM_SZ;
using kernel::mla_prefill_tp8_chunked::swz;
using kernel::mla_prefill_tp8_chunked::tma2d;
using kernel::mla_prefill_tp8_chunked::tma3d;
using kernel::mla_prefill_tp8_chunked::TMA_BLK;
using kernel::mla_prefill_tp8_chunked::V0_OFF;
using kernel::mla_prefill_tp8_chunked::V1_OFF;

// Named barrier 5 (chunked main uses 4).
__device__ __forceinline__ void splitk_sync() {
  asm volatile("bar.sync 5, %0;" ::"n"(NT));
}

__device__ __noinline__ void mla_prefill_tp8_chunked_splitk_sm100_task_impl(
    CUtensorMap const *KN_tm_ptr,
    CUtensorMap const *KR_tm_ptr,
    CUtensorMap const *V_tm_ptr,
    bf16 const *__restrict__ Qn,
    bf16 const *__restrict__ Qp,
    float *__restrict__ partial,
    int const q_len,
    int const kv_len,
    int const q_start,
    int const H,
    int const num_splits,
    int const nqb,
    float const sml2,
    int const head, // bid.x
    int const yidx, // bid.y; encodes (qb_rev, split_id)
    int const bat   // bid.z
) {
  if (threadIdx.x >= NT) {
    return;
  }
  int const split_id = yidx % num_splits;
  int const qb_rev = yidx / num_splits;
  int const qb = nqb - 1 - qb_rev;
  int const qs = qb * BM;
  int const tid = threadIdx.x;
  int const wid = tid / 32;
  int const lid = tid % 32;
  long long const bqn = (long long)bat * q_len * H * D_QK_NOPE;
  long long const bqp = (long long)bat * q_len * H * D_QK_ROPE;

  extern __shared__ __align__(1024) uint8_t sm_raw_splitk[];
  int sb = __cvta_generic_to_shared(sm_raw_splitk);
  sb = (sb + 1023) & ~1023;
  int qn_s = sb, qp_s = sb + Q_NOPE_SZ;
  int kn0 = sb + KN0_OFF;
  int kn1 = sb + KN1_OFF;
  int kps = sb + KP_OFF;
  int v0s = sb + V0_OFF;
  int v1s = sb + V1_OFF;
  int mbk = sb + MBK_OFF;
  int mbv = sb + MBV_OFF;
  if (tid == 0) {
    mbar_init_1(mbk, 1);
    mbar_init_1(mbv, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  splitk_sync();

  // Load Q.
  {
    constexpr int SN = D_QK_NOPE * 2, SP = D_QK_ROPE * 2;
    for (int i = tid; i < BM * (D_QK_NOPE / 8); i += NT) {
      int r = i / (D_QK_NOPE / 8), c = i % (D_QK_NOPE / 8), qi = qs + r;
      int a = qn_s + swz<SN>(r, c);
      if (qi < q_len) {
        cpa(a,
            Qn + bqn + (long long)qi * H * D_QK_NOPE +
                (long long)head * D_QK_NOPE + c * 8);
      } else {
        asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n" ::"r"(a));
      }
    }
    for (int i = tid; i < BM * (D_QK_ROPE / 8); i += NT) {
      int r = i / (D_QK_ROPE / 8), c = i % (D_QK_ROPE / 8), qi = qs + r;
      int a = qp_s + swz<SP>(r, c);
      if (qi < q_len) {
        cpa(a,
            Qp + bqp + (long long)qi * H * D_QK_ROPE +
                (long long)head * D_QK_ROPE + c * 8);
      } else {
        asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};\n" ::"r"(a));
      }
    }
    cpa_commit();
    cpa_wait<0>();
    splitk_sync();
  }
  int qnl = qn_s + swz<(D_QK_NOPE * 2)>(wid * 16 + (lid % 16), lid / 16);
  int qpl = qp_s + swz<(D_QK_ROPE * 2)>(wid * 16 + (lid % 16), lid / 16);
  int const kr_swz = (lid % 8) + (lid / 16) * 8;
  int const kc_swz = (lid % 16) / 8;

  float of[NMDV][8];
#pragma unroll
  for (int i = 0; i < NMDV; i++) {
    for (int j = 0; j < 8; j++) {
      of[i][j] = 0.f;
    }
  }
  float ms[2] = {-INFINITY, -INFINITY};
  float ds[2] = {1.f, 1.f};
  float sf0[HALF_N][8], sf1[HALF_N][8];

  int const kvend = min(kv_len, q_start + qs + BM);
  int const total_tiles = cdiv(kvend, BN);
  int const tiles_per_split = cdiv(total_tiles, num_splits);
  int const t_start = split_id * tiles_per_split;
  int const t_end = min(t_start + tiles_per_split, total_tiles);
  int const nt = t_end - t_start;
  int mphk = 0, mphv = 0;

  auto tld_k = [&](int kvb) {
    if (tid == 0) {
      mbar_tx(mbk, 3 * TMA_BLK);
      tma3d(KN_tm_ptr, kn0, mbk, 0, kvb, head * 2 + 0);
      tma3d(KN_tm_ptr, kn1, mbk, 0, kvb, head * 2 + 1);
      tma2d(KR_tm_ptr, kps, mbk, 0, kvb);
    }
  };
  auto tld_v = [&](int kvb) {
    if (tid == 0) {
      mbar_tx(mbv, 2 * TMA_BLK);
      tma3d(V_tm_ptr, v0s, mbv, 0, kvb, head * 2 + 0);
      tma3d(V_tm_ptr, v1s, mbv, 0, kvb, head * 2 + 1);
    }
  };
  if (nt > 0) {
    tld_k(t_start * BN);
    tld_v(t_start * BN);
  }

#pragma unroll 1
  for (int t = t_start; t < t_end; t++) {
    int kvb = t * BN;
    mbar_wait_1(mbk, mphk);
    mphk ^= 1;
    do_qk_half(sf0, 0, qpl, qnl, kps, kn0, kn1, kr_swz, kc_swz, lid);
    do_qk_half(sf1, HALF_N, qpl, qnl, kps, kn0, kn1, kr_swz, kc_swz, lid);
    splitk_sync();
    if (t + 1 < t_end) {
      tld_k((t + 1) * BN);
    }
    do_mask_softmax(
        sf0, 0, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
    mbar_wait_1(mbv, mphv);
    mphv ^= 1;
    do_pv_half(sf0, 0, v0s, v1s, lid, ds, of);
    do_mask_softmax(
        sf1, HALF_N, kvb, q_start, qs, kv_len, wid, lid, sml2, ms, ds, of);
    do_pv_half(sf1, HALF_N, v0s, v1s, lid, ds, of);
    splitk_sync();
    if (t + 1 < t_end) {
      tld_v((t + 1) * BN);
    }
  }

  // Write partial: stride D_V+4 per row (extra 4 = m, d, _, _).
  // Layout: partial[split_id][bat][qb][head][row][D_V+4]. The standalone
  // computes pbase as (split_id+bat)*stride_bat — accidentally OK for B=1
  // (bat=0) but wrong for B>1. We reproduce the same layout exactly.
  long long const stride_row = D_V + 4;
  long long const stride_head = (long long)BM * stride_row;
  long long const stride_qb = (long long)H * stride_head;
  long long const stride_bat = (long long)nqb * stride_qb;
  long long const pbase =
      (long long)split_id * stride_bat + (long long)bat * stride_bat +
      (long long)qb * stride_qb + (long long)head * stride_head;
  int g = lid / 4, t2 = lid % 4;
  {
    int row = wid * 16 + g;
    if (qs + row < q_len) {
      long long roff = pbase + (long long)row * stride_row;
#pragma unroll
      for (int md = 0; md < NMDV; md++) {
        int db = md * 16;
        *(float2 *)&partial[roff + db + 2 * t2] =
            make_float2(of[md][0], of[md][1]);
        *(float2 *)&partial[roff + db + 2 * t2 + 8] =
            make_float2(of[md][4], of[md][5]);
      }
      if (t2 == 0) {
        partial[roff + D_V] = ms[0];
        partial[roff + D_V + 1] = ds[0];
      }
    }
  }
  {
    int row = wid * 16 + g + 8;
    if (qs + row < q_len) {
      long long roff = pbase + (long long)row * stride_row;
#pragma unroll
      for (int md = 0; md < NMDV; md++) {
        int db = md * 16;
        *(float2 *)&partial[roff + db + 2 * t2] =
            make_float2(of[md][2], of[md][3]);
        *(float2 *)&partial[roff + db + 2 * t2 + 8] =
            make_float2(of[md][6], of[md][7]);
      }
      if (t2 == 0) {
        partial[roff + D_V] = ms[1];
        partial[roff + D_V + 1] = ds[1];
      }
    }
  }
}

// REDUCE: combines partial outputs across num_splits, writes final bf16.
// Grid: (H, nqb, B). 256 threads = 64 rows × 4 threads/row.
__device__ __noinline__ void mla_prefill_tp8_chunked_reduce_sm100_task_impl(
    float const *__restrict__ partial,
    bf16 *__restrict__ O,
    int const q_len,
    int const H,
    int const num_splits,
    int const nqb,
    float const sm_scale,
    int const head, // bid.x
    int const qb,   // bid.y
    int const bat   // bid.z
) {
  if (threadIdx.x >= 256) {
    return;
  }
  int const qs = qb * BM;
  int const row = threadIdx.x / 4;
  int const col_group = threadIdx.x % 4;
  long long const stride_row = D_V + 4;
  long long const stride_head = (long long)BM * stride_row;
  long long const stride_qb = (long long)H * stride_head;
  long long const stride_bat = (long long)nqb * stride_qb;
  if (qs + row >= q_len) {
    return;
  }

  float m_global = -INFINITY, d_global = 0.f;
  float o_local[32];
#pragma unroll
  for (int d = 0; d < 32; d++) {
    o_local[d] = 0.f;
  }
  int const d_start = col_group * 32;

  for (int s = 0; s < num_splits; s++) {
    long long roff = (long long)s * stride_bat + (long long)bat * stride_bat +
                     (long long)qb * stride_qb + (long long)head * stride_head +
                     (long long)row * stride_row;
    float m_s = -INFINITY, d_s = 0.f;
    if (col_group == 0) {
      m_s = partial[roff + D_V];
      d_s = partial[roff + D_V + 1];
    }
    m_s = __shfl_sync(0xffffffff, m_s, (threadIdx.x & ~3));
    d_s = __shfl_sync(0xffffffff, d_s, (threadIdx.x & ~3));
    if (m_s == -INFINITY) {
      continue;
    }
    float vals[32];
#pragma unroll
    for (int d = 0; d < 32; d += 4) {
      float4 v = *(float4 const *)&partial[roff + d_start + d];
      vals[d] = v.x;
      vals[d + 1] = v.y;
      vals[d + 2] = v.z;
      vals[d + 3] = v.w;
    }
    if (m_s > m_global) {
      float scale = expf((m_global - m_s) * sm_scale);
      d_global = d_global * scale + d_s;
#pragma unroll
      for (int d = 0; d < 32; d++) {
        o_local[d] = o_local[d] * scale + vals[d];
      }
      m_global = m_s;
    } else {
      float scale = expf((m_s - m_global) * sm_scale);
      d_global += d_s * scale;
#pragma unroll
      for (int d = 0; d < 32; d++) {
        o_local[d] += vals[d] * scale;
      }
    }
  }
  float dr = (m_global != -INFINITY) ? (1.0f / d_global) : 0.f;
  long long ooff = (long long)bat * q_len * H * D_V +
                   (long long)(qs + row) * H * D_V + (long long)head * D_V;
#pragma unroll
  for (int d = 0; d < 32; d += 4) {
    bf16_2 lo = __float22bfloat162_rn(
        make_float2(o_local[d] * dr, o_local[d + 1] * dr));
    bf16_2 hi = __float22bfloat162_rn(
        make_float2(o_local[d + 2] * dr, o_local[d + 3] * dr));
    int2 packed;
    packed.x = *(int *)&lo;
    packed.y = *(int *)&hi;
    *(int2 *)&O[ooff + d_start + d] = packed;
  }
}

} // namespace mla_prefill_tp8_chunked_splitk
} // namespace kernel
