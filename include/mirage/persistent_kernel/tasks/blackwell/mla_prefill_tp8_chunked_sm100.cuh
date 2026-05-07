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

// MLA Chunked Prefill (TP=8) — true unabsorbed, per-head K/V.
// Source: mla_chunked_prefill_tp8_unabsorbed_perhead.cu (BM=64 main variant).
//
// DeepSeek V3 MLA dimensions (post kv_b_proj decompression, per TP=8 rank):
//   Q_nope: [B, q_len, H=16, 128]   per-head
//   Q_rope: [B, q_len, H=16,  64]   per-head
//   K_nope: [B, kv_len, H=16, 128]  per-head (from kv_b_proj)
//   K_rope: [B, kv_len, 1,    64]   shared across heads
//   V:      [B, kv_len, H=16, 128]  per-head (from kv_b_proj)
//   O:      [B, q_len,  H=16, 128]  per-head
//
// QK = Q_nope @ K_nope^T + Q_rope @ K_rope^T (separate MMA passes for nope+rope).
// kv_b_proj decompression is done OUTSIDE this kernel; caller provides
// already-decompressed per-head K_nope and V.
//
// MPK adaptation pattern (per kernel_adaptation_guide):
//   __global__         → __device__ __noinline__
//   blockIdx.{x,y,z}   → function parameters (head, qb_in, bat)
//   __grid_constant__  → CUtensorMap const* (heap-allocated by runtime)
//   __syncthreads()    → bar.sync 4, NT  (NT=128 of MPK's 256-thread block)
//   Thread guard       → if (threadIdx.x >= 128) return;
//   1024-byte SMEM align: round sb up to 1024 so 128B-swizzle TMA stores
//   land on swizzle-aligned offsets even when MPK's static-shared prefix
//   leaves `extern __shared__` only 128-aligned.

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {
namespace mla_prefill_tp8_chunked {

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

static constexpr int D_QK_NOPE = 128;
static constexpr int D_QK_ROPE = 64;
static constexpr int D_QK = 192;
static constexpr int D_V = 128;
static constexpr int BM = 64;
static constexpr int BN = 128;
static constexpr int NT = 128;
static constexpr int MK = 16;
static constexpr int HALF_N = BN / 16 / 2;  // 4
static constexpr int NMDK = D_QK_NOPE / MK; // 8
static constexpr int NMRK = D_QK_ROPE / MK; // 4
static constexpr int NMDV = D_V / 16;       // 8

// SMEM: Qn + Qp + KN0/KN1/KP/V0/V1 + 2 mbarriers (mbk, mbv)
static constexpr int Q_NOPE_SZ = BM * D_QK_NOPE * 2; // 16 KB
static constexpr int Q_PE_SZ = BM * D_QK_ROPE * 2;   //  8 KB
static constexpr int TMA_BLK = BN * 64 * 2;          // 16 KB
static constexpr int KN0_OFF = Q_NOPE_SZ + Q_PE_SZ;
static constexpr int KN1_OFF = KN0_OFF + TMA_BLK;
static constexpr int KP_OFF = KN1_OFF + TMA_BLK;
static constexpr int V0_OFF = KP_OFF + TMA_BLK;
static constexpr int V1_OFF = V0_OFF + TMA_BLK;
static constexpr int MBK_OFF = V1_OFF + TMA_BLK;
static constexpr int MBV_OFF = MBK_OFF + 16;
// +1024 so the 1024-byte round-up of sb inside the task body never overruns
// the cudaFuncAttributeMaxDynamicSharedMemorySize budget the caller set.
static constexpr int SMEM_SZ = MBV_OFF + 16 + 1024;

template <int S>
__device__ __forceinline__ int swz(int r, int c) {
  if constexpr (S >= 128) {
    c ^= (r % 8) / (128 / S > 1 ? 128 / S : 1);
  }
  return r * S + c * 16;
}
__device__ __forceinline__ void ldm4(uint32_t r[4], int a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t r[4], int a) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(a));
}
__device__ __forceinline__ void
    hmma(uint32_t const A[4], uint32_t const B[2], float C[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}
__device__ __forceinline__ void
    hmma0(uint32_t const A[4], uint32_t const B[2], float C[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
               : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[0]),
                 "r"(B[1]),
                 "f"(0.f),
                 "f"(0.f),
                 "f"(0.f),
                 "f"(0.f));
}
__device__ __forceinline__ void
    hmma16(uint32_t const A[4], uint32_t const B[4], float C[8]) {
  hmma(A, &B[0], &C[0]);
  hmma(A, &B[2], &C[4]);
}
__device__ __forceinline__ void
    hmma16_0(uint32_t const A[4], uint32_t const B[4], float C[8]) {
  hmma0(A, &B[0], &C[0]);
  hmma0(A, &B[2], &C[4]);
}
__device__ __forceinline__ void rowsum(float *d, uint32_t *s) {
  asm volatile("{mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,_,%1,_},{%2,%3,%4,%5},{%6,%7},{%0,0.,%1,0.};}\n"
               : "+f"(d[0]), "+f"(d[1])
               : "r"(s[0]),
                 "r"(s[1]),
                 "r"(s[2]),
                 "r"(s[3]),
                 "r"(1065369472u),
                 "r"(1065369472u));
}
__device__ __forceinline__ void cpa(int d, void const *s) {
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::"r"(d), "l"(s));
}
__device__ __forceinline__ void cpa_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cpa_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ float sxor(float v, int m) {
  return __shfl_xor_sync(0xffffffff, v, m);
}
__device__ __forceinline__ float ex2(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0,%1;\n" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t f2b(float a, float b) {
  bf16_2 v = __float22bfloat162_rn(make_float2(a, b));
  return *(uint32_t *)&v;
}
__host__ __device__ __forceinline__ int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ void
    tma4d(CUtensorMap const *d, int sa, int mb,
          int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_"
               "tx::bytes [%0],[%1,{%2,%3,%4,%5}],[%6];" ::"r"(sa),
               "l"((uint64_t)d),
               "r"(c0),
               "r"(c1),
               "r"(c2),
               "r"(c3),
               "r"(mb)
               : "memory");
}
__device__ __forceinline__ void
    tma2d(CUtensorMap const *d, int sa, int mb, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
               "tx::bytes [%0],[%1,{%2,%3}],[%4];" ::"r"(sa),
               "l"((uint64_t)d),
               "r"(c0),
               "r"(c1),
               "r"(mb)
               : "memory");
}
__device__ __forceinline__ void mbar_init_1(int a, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_tx(int a, int b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(a),
               "r"(b));
}
__device__ __forceinline__ void mbar_wait_1(int a, int p) {
  asm volatile("{.reg .pred P;\nW: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
               "P,[%0],%1,0x989680;\n@P bra D;\n bra W;\nD:}" ::"r"(a),
               "r"(p));
}

// Named barrier 4 (128 threads).
__device__ __forceinline__ void task_sync() {
  asm volatile("bar.sync 4, %0;" ::"n"(NT));
}

__device__ __forceinline__ void do_qk_half(float sf[][8],
                                           int noff,
                                           int qpl,
                                           int qnl,
                                           int kps,
                                           int kn0,
                                           int kn1,
                                           int kr,
                                           int kc,
                                           int lid) {
  constexpr int S128 = 128;
  int kpl = kps + swz<S128>(kr, kc);
  int kn0l = kn0 + swz<S128>(kr, kc);
  int kn1l = kn1 + swz<S128>(kr, kc);
#pragma unroll
  for (int mk = 0; mk < NMRK; mk++) {
    {
      uint32_t qr[4];
      ldm4(qr, qpl ^ (mk * 32));
#pragma unroll
      for (int nl = 0; nl < HALF_N; nl++) {
        uint32_t k2[4];
        ldm4(k2, (kpl + (noff + nl) * 16 * S128) ^ (mk * 32));
        if (mk == 0) {
          hmma16_0(qr, k2, sf[nl]);
        } else {
          hmma16(qr, k2, sf[nl]);
        }
      }
    }
    {
      uint32_t qr[4];
      ldm4(qr, qnl ^ (mk * 32));
#pragma unroll
      for (int nl = 0; nl < HALF_N; nl++) {
        uint32_t k2[4];
        ldm4(k2, (kn0l + (noff + nl) * 16 * S128) ^ (mk * 32));
        hmma16(qr, k2, sf[nl]);
      }
    }
  }
#pragma unroll
  for (int mk = NMRK; mk < NMDK; mk++) {
    uint32_t qr[4];
    ldm4(qr, qnl ^ (mk * 32));
#pragma unroll
    for (int nl = 0; nl < HALF_N; nl++) {
      uint32_t k2[4];
      ldm4(k2, (kn1l + (noff + nl) * 16 * S128) ^ ((mk - 4) * 32));
      hmma16(qr, k2, sf[nl]);
    }
  }
}

__device__ __forceinline__ void do_mask_softmax(float sf[][8],
                                                int noff,
                                                int kvb,
                                                int q_start,
                                                int qs,
                                                int kv_len,
                                                int wid,
                                                int lid,
                                                float sml2,
                                                float ms[2],
                                                float ds[2],
                                                float of[][8]) {
  if (kvb + BN > q_start + qs) {
    int qrb = q_start + qs + wid * 16;
#pragma unroll
    for (int nl = 0; nl < HALF_N; nl++) {
#pragma unroll
      for (int ri = 0; ri < 8; ri++) {
        int rit = ((ri & 2) == 0) ? (lid / 4) : (lid / 4 + 8);
        int kvc = 2 * (lid % 4) + ((ri & 4) ? 8 : 0) + (ri & 1);
        int qp = qrb + rit, kvp = kvb + (noff + nl) * 16 + kvc;
        if (!((kvp <= qp) && (kvp < kv_len))) {
          sf[nl][ri] = -INFINITY;
        }
      }
    }
  }
  float mp[2] = {ms[0], ms[1]};
#pragma unroll
  for (int j = 0; j < 2; j++) {
#pragma unroll
    for (int nl = 0; nl < HALF_N; nl++) {
      float lm = fmaxf(fmaxf(sf[nl][j * 2], sf[nl][j * 2 + 1]),
                       fmaxf(sf[nl][j * 2 + 4], sf[nl][j * 2 + 5]));
      ms[j] = fmaxf(ms[j], lm);
    }
    ms[j] = fmaxf(ms[j], sxor(ms[j], 0x2));
    ms[j] = fmaxf(ms[j], sxor(ms[j], 0x1));
    float nms = -(ms[j] * sml2);
    float sc = ex2(__fmaf_rn(mp[j], sml2, nms));
    ds[j] *= sc;
    if (mp[j] != ms[j]) {
#pragma unroll
      for (int md = 0; md < NMDV; md++) {
        of[md][j * 2 + 0] *= sc;
        of[md][j * 2 + 1] *= sc;
        of[md][j * 2 + 4] *= sc;
        of[md][j * 2 + 5] *= sc;
      }
    }
#pragma unroll
    for (int nl = 0; nl < HALF_N; nl++) {
      sf[nl][j * 2 + 0] = ex2(__fmaf_rn(sf[nl][j * 2 + 0], sml2, nms));
      sf[nl][j * 2 + 1] = ex2(__fmaf_rn(sf[nl][j * 2 + 1], sml2, nms));
      sf[nl][j * 2 + 4] = ex2(__fmaf_rn(sf[nl][j * 2 + 4], sml2, nms));
      sf[nl][j * 2 + 5] = ex2(__fmaf_rn(sf[nl][j * 2 + 5], sml2, nms));
    }
  }
}

__device__ __forceinline__ void do_pv_half(float sf[][8],
                                           int noff,
                                           int v0s,
                                           int v1s,
                                           int lid,
                                           float ds[2],
                                           float of[][8]) {
  constexpr int S128 = 128;
  uint32_t pf[HALF_N][4];
#pragma unroll
  for (int nl = 0; nl < HALF_N; nl++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      pf[nl][i] = f2b(sf[nl][i * 2], sf[nl][i * 2 + 1]);
    }
    rowsum(ds, pf[nl]);
  }
  int vr0 = lid % 16, vcb = lid / 16;
#pragma unroll
  for (int mkv = 0; mkv < HALF_N; mkv++) {
#pragma unroll
    for (int md = 0; md < NMDV; md++) {
      uint32_t vf[4];
      int vs_base = (md < 4) ? v0s : v1s;
      int md_local = (md < 4) ? md : (md - 4);
      ldm4t(vf,
            vs_base + swz<S128>(vr0 + (noff + mkv) * 16, vcb + md_local * 2));
      hmma16(pf[mkv], vf, of[md]);
    }
  }
}

__device__ __forceinline__ void
    finalize_o(float of[][8], float ms[2], float ds[2]) {
  float dr[2];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    if (ms[j] != -INFINITY) {
      asm volatile("rcp.approx.ftz.f32 %0,%1;" : "=f"(dr[j]) : "f"(ds[j]));
    } else {
      dr[j] = 0.f;
    }
  }
#pragma unroll
  for (int md = 0; md < NMDV; md++) {
#pragma unroll
    for (int ri = 0; ri < 8; ri++) {
      of[md][ri] *= dr[(ri % 4) / 2];
    }
  }
}

__device__ __forceinline__ void write_o(float of[][8],
                                        bf16 *O,
                                        long long bo,
                                        int qs,
                                        int q_len,
                                        int H,
                                        int head,
                                        int wid,
                                        int lid) {
  int g = lid / 4, t2 = lid % 4;
#pragma unroll
  for (int md = 0; md < NMDV; md++) {
    int db = md * 16, qp = qs + wid * 16 + g;
    if (qp < q_len) {
      long long off = bo + (long long)qp * H * D_V + (long long)head * D_V + db;
      *(bf16_2 *)&O[off + 2 * t2] =
          __float22bfloat162_rn(make_float2(of[md][0], of[md][1]));
      *(bf16_2 *)&O[off + 2 * t2 + 8] =
          __float22bfloat162_rn(make_float2(of[md][4], of[md][5]));
    }
    qp = qs + wid * 16 + g + 8;
    if (qp < q_len) {
      long long off = bo + (long long)qp * H * D_V + (long long)head * D_V + db;
      *(bf16_2 *)&O[off + 2 * t2] =
          __float22bfloat162_rn(make_float2(of[md][2], of[md][3]));
      *(bf16_2 *)&O[off + 2 * t2 + 8] =
          __float22bfloat162_rn(make_float2(of[md][6], of[md][7]));
    }
  }
}

__device__ __noinline__ void mla_prefill_tp8_chunked_sm100_task_impl(
    CUtensorMap const *KN_tm_ptr, // K_nope, per-head [B, kv_len, H, 128]
    CUtensorMap const *KR_tm_ptr, // K_rope, shared  [B, kv_len, 1,  64]
    CUtensorMap const *V_tm_ptr,  // V,      per-head [B, kv_len, H, 128]
    bf16 const *__restrict__ Qn,  // [B, q_len, H, 128]
    bf16 const *__restrict__ Qp,  // [B, q_len, H,  64]
    bf16 *__restrict__ O,         // [B, q_len, H, 128]
    int const q_len,
    int const kv_len,
    int const q_start,
    int const H,
    float const sml2,
    int const head,  // bid.x
    int const qb_in, // bid.y
    int const bat    // bid.z
) {
  if (threadIdx.x >= NT) {
    return;
  }
  const int qb = cdiv(q_len, BM) - 1 - qb_in;
  const int qs = qb * BM;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const long long bqn = (long long)bat * q_len * H * D_QK_NOPE;
  const long long bqp = (long long)bat * q_len * H * D_QK_ROPE;
  const long long bo = (long long)bat * q_len * H * D_V;

  extern __shared__ __align__(1024) uint8_t sm_raw_chunk[];
  int sb = __cvta_generic_to_shared(sm_raw_chunk);
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
  task_sync();
  // Load Q via cp.async.
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
    task_sync();
  }
  int qnl = qn_s + swz<(D_QK_NOPE * 2)>(wid * 16 + (lid % 16), lid / 16);
  int qpl = qp_s + swz<(D_QK_ROPE * 2)>(wid * 16 + (lid % 16), lid / 16);
  const int kr_swz = (lid % 8) + (lid / 16) * 8;
  const int kc_swz = (lid % 16) / 8;

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
  int kvend = min(kv_len, q_start + qs + BM);
  int nt = cdiv(kvend, BN);
  int mphk = 0, mphv = 0;

  // K_nope and V are loaded with 4D TMA (dim layout [BK=64, kv_len, 2 halves,
  // H heads]) so the kernel can read interleaved kv_b_proj output [kv_len, H,
  // qk_nope+v_head] as a strided view: K_nope at offset 0 with head stride
  // 256 elements, V at offset +qk_nope_head_dim (=128) with same head stride.
  // K_rope is shared across heads and stays 2D.
  auto tld_k = [&](int kvb) {
    if (tid == 0) {
      mbar_tx(mbk, 3 * TMA_BLK);
      tma4d(KN_tm_ptr, kn0, mbk, 0, kvb, 0, head);
      tma4d(KN_tm_ptr, kn1, mbk, 0, kvb, 1, head);
      tma2d(KR_tm_ptr, kps, mbk, 0, kvb);
    }
  };
  auto tld_v = [&](int kvb) {
    if (tid == 0) {
      mbar_tx(mbv, 2 * TMA_BLK);
      tma4d(V_tm_ptr, v0s, mbv, 0, kvb, 0, head);
      tma4d(V_tm_ptr, v1s, mbv, 0, kvb, 1, head);
    }
  };
  if (nt > 0) {
    tld_k(0);
    tld_v(0);
  }

#pragma unroll 1
  for (int t = 0; t < nt; t++) {
    int kvb = t * BN;
    mbar_wait_1(mbk, mphk);
    mphk ^= 1;
    do_qk_half(sf0, 0, qpl, qnl, kps, kn0, kn1, kr_swz, kc_swz, lid);
    do_qk_half(sf1, HALF_N, qpl, qnl, kps, kn0, kn1, kr_swz, kc_swz, lid);
    task_sync();
    if (t + 1 < nt) {
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
    task_sync();
    if (t + 1 < nt) {
      tld_v((t + 1) * BN);
    }
  }
  finalize_o(of, ms, ds);
  write_o(of, O, bo, qs, q_len, H, head, wid, lid);
}

} // namespace mla_prefill_tp8_chunked
} // namespace kernel
