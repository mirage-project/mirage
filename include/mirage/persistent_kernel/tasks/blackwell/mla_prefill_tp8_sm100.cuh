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

// MLA Prefill (TP=8) for DeepSeek V3 on B200 — UNABSORBED, TMA KV loads.
// Adapted from src/kernel/mla_prefill_tp8.cu (standalone kernel).
//
// MPK changes from the standalone:
//   __global__  → __device__ __noinline__
//   blockIdx.{x,y,z}         → function parameters (head, q_block, batch)
//   __grid_constant__ TMA    → CUtensorMap const* parameters
//   __syncthreads()          → bar.sync 3, 128 (only 128 threads participate
//                              out of MPK's worker thread count)
//   Thread guard at entry    → if (threadIdx.x >= 128) return;
//   Removed __launch_bounds__, launcher function, main().
//
// Each task handles one (head, q_block, batch) tuple. Grid = (H, num_q_blocks, B).
// TP=8 means NUM_HEADS per rank is 128/8 = 16; the kernel doesn't encode that —
// it just iterates over H passed as a parameter (H = 16 for TP=8).

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {
namespace mla_prefill_tp8 {

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
static constexpr int HALF_N = BN / 16 / 2;    // 4
static constexpr int NMDK = D_QK_NOPE / MK;   // 8
static constexpr int NMRK = D_QK_ROPE / MK;   // 4
static constexpr int NMDV = D_V / 16;         // 8

// SMEM: Q_nope + Q_pe + 5 TMA blocks of [BN,64] with 128B swizzle + mbar
static constexpr int Q_NOPE_SZ = BM * D_QK_NOPE * 2; // 16 KB
static constexpr int Q_PE_SZ = BM * D_QK_ROPE * 2;   // 8 KB
static constexpr int TMA_BLK = BN * 64 * 2;          // 16 KB per [BN,64] block
// Layout: Q_nope | Q_pe | KN0 | KN1 | KP | V0 | V1 | mbar
static constexpr int KN0_OFF = Q_NOPE_SZ + Q_PE_SZ;
static constexpr int KN1_OFF = KN0_OFF + TMA_BLK;
static constexpr int KP_OFF = KN1_OFF + TMA_BLK;
static constexpr int V0_OFF = KP_OFF + TMA_BLK;
static constexpr int V1_OFF = V0_OFF + TMA_BLK;
static constexpr int MBAR_OFF = V1_OFF + TMA_BLK;
// +1024 so the 1024-byte round-up of sb inside the task body never runs past
// the caller's cudaFuncAttributeMaxDynamicSharedMemorySize budget.
static constexpr int SMEM_SZ = MBAR_OFF + 16 + 1024;

// Swizzle for 128B: row stride S bytes, 16B chunks XOR'd by (r%8)
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
    hmma(const uint32_t A[4], const uint32_t B[2], float C[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[0]),
                 "r"(B[1]));
}
__device__ __forceinline__ void
    hmma0(const uint32_t A[4], const uint32_t B[2], float C[4]) {
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
    hmma16(const uint32_t A[4], const uint32_t B[4], float C[8]) {
  hmma(A, &B[0], &C[0]);
  hmma(A, &B[2], &C[4]);
}
__device__ __forceinline__ void
    hmma16_0(const uint32_t A[4], const uint32_t B[4], float C[8]) {
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
__device__ __forceinline__ void cpa(int d, const void *s) {
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
    tma3d(CUtensorMap const *d, int sa, int mb, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_"
               "tx::bytes [%0],[%1,{%2,%3,%4}],[%5];" ::"r"(sa),
               "l"((uint64_t)d),
               "r"(c0),
               "r"(c1),
               "r"(c2),
               "r"(mb)
               : "memory");
}
__device__ __forceinline__ void mbar_init_1(int a, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" ::"r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_tx(int a, int b) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(a), "r"(b));
}
__device__ __forceinline__ void mbar_wait_1(int a, int p) {
  asm volatile("{.reg .pred P;\nW: "
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
               "P,[%0],%1,0x989680;\n@P bra D;\n bra W;\nD:}" ::"r"(a),
               "r"(p));
}

// Named barrier 3 (128 threads) for this task — barriers 0-2 are used by
// other v2 tasks; see the kernel_adaptation_guide memory.
__device__ __forceinline__ void task_sync() {
  asm volatile("bar.sync 3, %0;" ::"n"(NT));
}

__device__ __noinline__ void mla_prefill_tp8_sm100_task_impl(
    CUtensorMap const *K_tm_ptr,
    CUtensorMap const *V_tm_ptr,
    bf16 const *__restrict__ Qn, // [B, S, H, D_QK_NOPE]
    bf16 const *__restrict__ Qp, // [B, S, H, D_QK_ROPE]
    bf16 *__restrict__ O,        // [B, S, H, D_V]
    int const S,
    int const H,
    float const sml2, // sm_scale * log2(e)
    int const head,    // was blockIdx.x
    int const qb_in,   // was blockIdx.y input (mapped to qb = cdiv(S,BM)-1-qb_in)
    int const bat      // was blockIdx.z
) {
  // MPK worker may launch with more threads; only NT participate.
  if (threadIdx.x >= NT) {
    return;
  }

  const int qb = cdiv(S, BM) - 1 - qb_in;
  const int qs = qb * BM;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const long long bqn = (long long)bat * S * H * D_QK_NOPE;
  const long long bqp = (long long)bat * S * H * D_QK_ROPE;
  const long long bo = (long long)bat * S * H * D_V;
  // 128B swizzle TMA requires SMEM destination aligned to the 1024-byte
  // swizzle tile. In MPK the dynamic SMEM starts after ~7 KB of static
  // shared (TaskDesc buffers), leaving the extern __shared__ base only
  // 128-aligned. Round sb up to 1024 so the tile destinations land on
  // swizzle-aligned offsets. This costs <= 1 KB of SMEM.
  extern __shared__ __align__(1024) uint8_t sm_raw_tp8[];
  int sb = __cvta_generic_to_shared(sm_raw_tp8);
  sb = (sb + 1023) & ~1023;
  int qn_s = sb, qp_s = sb + Q_NOPE_SZ;
  int kn0 = sb + KN0_OFF;
  int kn1 = sb + KN1_OFF;
  int kps = sb + KP_OFF;
  int v0s = sb + V0_OFF;
  int v1s = sb + V1_OFF;
  int mbs = sb + MBAR_OFF;
  if (tid == 0) {
    mbar_init_1(mbs, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  task_sync();
  // Load Q via cp.async
  {
    constexpr int SN = D_QK_NOPE * 2, SP = D_QK_ROPE * 2;
    for (int i = tid; i < BM * (D_QK_NOPE / 8); i += NT) {
      int r = i / (D_QK_NOPE / 8), c = i % (D_QK_NOPE / 8), qi = qs + r;
      int a = qn_s + swz<SN>(r, c);
      if (qi < S) {
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
      if (qi < S) {
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
  constexpr int SNB = D_QK_NOPE * 2, SPB = D_QK_ROPE * 2, S128 = 128;
  int qnl = qn_s + swz<SNB>(wid * 16 + (lid % 16), lid / 16);
  int qpl = qp_s + swz<(D_QK_ROPE * 2)>(wid * 16 + (lid % 16), lid / 16);
  const int kr = (lid % 8) + (lid / 16) * 8, kc = (lid % 16) / 8;

  float of[NMDV][8];
#pragma unroll
  for (int i = 0; i < NMDV; i++) {
    for (int j = 0; j < 8; j++) {
      of[i][j] = 0.f;
    }
  }
  float ms[2] = {-INFINITY, -INFINITY};
  float ds[2] = {1.f, 1.f};
  float sf[HALF_N][8];
  int kvend = min(S, qs + BM), nt = cdiv(kvend, BN);
  int mph = 0;

  auto tld = [&](int kvb) {
    if (tid == 0) {
      mbar_tx(mbs, 5 * TMA_BLK);
      tma3d(K_tm_ptr, kn0, mbs, 0, kvb, 0);
      tma3d(K_tm_ptr, kn1, mbs, 0, kvb, 1);
      tma3d(K_tm_ptr, kps, mbs, 0, kvb, 2);
      tma3d(V_tm_ptr, v0s, mbs, 0, kvb, 0);
      tma3d(V_tm_ptr, v1s, mbs, 0, kvb, 1);
    }
  };
  if (nt > 0) {
    tld(0);
  }

#pragma unroll 1
  for (int t = 0; t < nt; t++) {
    int kvb = t * BN;
    mbar_wait_1(mbs, mph);
    mph ^= 1;
    task_sync();

#pragma unroll
    for (int half = 0; half < 2; half++) {
      int noff = half * HALF_N;
      // QK
      {
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
      // Causal mask
      if (kvb + BN > qs) {
        int qrb = qs + wid * 16;
#pragma unroll
        for (int nl = 0; nl < HALF_N; nl++) {
#pragma unroll
          for (int ri = 0; ri < 8; ri++) {
            int rit = ((ri & 2) == 0) ? (lid / 4) : (lid / 4 + 8);
            int kvc = 2 * (lid % 4) + ((ri & 4) ? 8 : 0) + (ri & 1);
            int qp = qrb + rit, kvp = kvb + (noff + nl) * 16 + kvc;
            if (!((kvp <= qp) && (kvp < S))) {
              sf[nl][ri] = -INFINITY;
            }
          }
        }
      }
      // Softmax
      {
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
#pragma unroll
          for (int md = 0; md < NMDV; md++) {
            of[md][j * 2 + 0] *= sc;
            of[md][j * 2 + 1] *= sc;
            of[md][j * 2 + 4] *= sc;
            of[md][j * 2 + 5] *= sc;
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
      // bf16 + rowsum
      uint32_t pf[HALF_N][4];
#pragma unroll
      for (int nl = 0; nl < HALF_N; nl++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          pf[nl][i] = f2b(sf[nl][i * 2], sf[nl][i * 2 + 1]);
        }
        rowsum(ds, pf[nl]);
      }
      // PV
      {
        int vr0 = lid % 16, vcb = lid / 16;
#pragma unroll
        for (int mkv = 0; mkv < HALF_N; mkv++) {
#pragma unroll
          for (int md = 0; md < NMDV; md++) {
            uint32_t vf[4];
            int vs_base = (md < 4) ? v0s : v1s;
            int md_local = (md < 4) ? md : (md - 4);
            ldm4t(vf, vs_base + swz<S128>(vr0 + (noff + mkv) * 16, vcb + md_local * 2));
            hmma16(pf[mkv], vf, of[md]);
          }
        }
      }
    }
    task_sync();
    if (t + 1 < nt) {
      tld((t + 1) * BN);
    }
  }
  // Normalize
  {
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
  // Write O
  {
    int g = lid / 4, t2 = lid % 4;
#pragma unroll
    for (int md = 0; md < NMDV; md++) {
      int db = md * 16, qp = qs + wid * 16 + g;
      if (qp < S) {
        long long off = bo + (long long)qp * H * D_V + (long long)head * D_V + db;
        *(bf16_2 *)&O[off + 2 * t2] =
            __float22bfloat162_rn(make_float2(of[md][0], of[md][1]));
        *(bf16_2 *)&O[off + 2 * t2 + 8] =
            __float22bfloat162_rn(make_float2(of[md][4], of[md][5]));
      }
      qp = qs + wid * 16 + g + 8;
      if (qp < S) {
        long long off = bo + (long long)qp * H * D_V + (long long)head * D_V + db;
        *(bf16_2 *)&O[off + 2 * t2] =
            __float22bfloat162_rn(make_float2(of[md][2], of[md][3]));
        *(bf16_2 *)&O[off + 2 * t2 + 8] =
            __float22bfloat162_rn(make_float2(of[md][6], of[md][7]));
      }
    }
  }
}

} // namespace mla_prefill_tp8
} // namespace kernel
