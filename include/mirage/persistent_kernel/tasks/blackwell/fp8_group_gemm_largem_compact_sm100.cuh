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

// Grouped FP8 GEMM, compact-dispatch variant (W2 MoE down-projection).
//
// Motivation: MPK's incumbent kernel (fp8_group_gemm_largem_sm100) iterates
// all E_local*nn tile slots per dispatch, checking active_expert_mask per tile.
// At DSv3 TP=4 EP=2 decode (4/128 active experts), 54/56 iterations per
// worker are cheap skips, wasting ~6.75 us per call. This variant builds a
// compact active-expert list at kernel start (warp-ballot deterministic scan,
// ~1 us) then loops only over num_active*nn tiles, eliminating all skips.
//
// Performance (GPU B200, BN=128, NS=6, M=16384, K=1024, N=7168, 128 workers):
//   decode_4active:  18.46 us vs incumbent 28.67 us → 1.55x
//   decode_16active: 36.90 us vs incumbent 45.09 us → 1.22x
//   decode_32active: 61.47 us vs incumbent 67.74 us → 1.10x
//
// KERNEL_RESULT {"decode_4active": 407.0728, "decode_16active": 814.8517, "decode_32active": 978.1615}
// KERNEL_RESULT_REFERENCE {"decode_4active": 262.1440, "decode_16active": 666.8020, "decode_32active": 887.5995}

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace kernel {
namespace fp8_group_gemm_largem_compact {

constexpr int BN = 128;
constexpr int NS = 6;

// ────────────────────────── Device helpers (local to this TU) ─────────────────
namespace detail {

__device__ __forceinline__ uint32_t elect_one_sync_impl() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}
__device__ __forceinline__ void mb_init_impl(int a, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(c));
}
__device__ __forceinline__ void mb_wait_impl(int a, int p) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LW:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DN;\n\t"
      "bra.uni LW;\n\t"
      "DN:\n\t"
      "}" ::"r"(a),
      "r"(p),
      "r"(0x989680));
}
__device__ __forceinline__ void mb_arrive_impl(int a) {
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a)
      : "memory");
}
__device__ __forceinline__ void mb_arrive_tx_impl(int a, int s) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
          a),
      "r"(s)
      : "memory");
}
__device__ __forceinline__ void
tma_ld_impl(int d, const void *t, int x, int y, int m) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];" ::"r"(d),
      "l"(t),
      "r"(x),
      "r"(y),
      "r"(m)
      : "memory");
}
__device__ __forceinline__ constexpr uint64_t denc_impl(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}
__device__ __forceinline__ uint64_t mkdesc_impl(int a) {
  return denc_impl(a) | (denc_impl(1024) << 32ULL) | (1ULL << 46ULL) |
         (2ULL << 61ULL);
}
__device__ __forceinline__ uint64_t mkdesc_sf_impl(int a) {
  return denc_impl(a) | (denc_impl(128) << 32ULL) | (1ULL << 46ULL);
}
__device__ __forceinline__ uint32_t ld_shared_u32_impl(const void *p) {
  uint32_t r;
  asm volatile(
      "ld.shared.u32 %0, [%1];"
      : "=r"(r)
      : "r"((uint32_t)__cvta_generic_to_shared(p)));
  return r;
}
__device__ __forceinline__ void st_shared_u32_impl(void *p, uint32_t v) {
  asm volatile(
      "st.shared.u32 [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(p)),
      "r"(v));
}

} // namespace detail

// ────────────────────────── Smem size helper ──────────────────────────────────
inline constexpr int fp8_group_gemm_largem_compact_smem_size() {
  constexpr int BM = 128, BK = 128;
  constexpr int NE = 2;
  constexpr int STORE_BN = 64, NUM_TMA_ST = 1;
  constexpr int SCD_TOT = NUM_TMA_ST * BM * STORE_BN * 2;
  constexpr int SA = BM * BK, SB = BN * BK;
  constexpr int SFA_SIZE = 128 * 4, SFB_SIZE = BN * 4;
  int base = SCD_TOT + NS * (SA + SB + SFA_SIZE + SFB_SIZE);
  base = (base + 7) & ~7;
  base += (NS * 3 + NE * 2) * 8 + 8;
  base = (base + 1023) & ~1023;
  return base;
}

// ────────────────────────── Main device function ──────────────────────────────
__device__ __noinline__ void fp8_group_gemm_largem_compact_task_impl(
    CUtensorMap const *ta_ptr,
    CUtensorMap const *tb_ptr,
    CUtensorMap const *tsfa_ptr,
    CUtensorMap const *tsfb_ptr,
    CUtensorMap const *td_ptr,
    int const *__restrict__ m_indices,
    int const *__restrict__ active_expert_mask,
    int const M_total,
    int const N,
    int const K,
    int const E,
    int const worker_idx,
    int const num_workers) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  constexpr int BM = 128, BK = 128, UK = 32;
  constexpr int SF_PER_LOAD = 4;
  constexpr int NE = 2;

  const int tid = threadIdx.x, wid = tid / 32;
  const uint32_t lid = tid % 32;
  const int nm = (M_total + BM - 1) / BM;
  const int nn = (N + BN - 1) / BN;
  const int nk = (K + BK - 1) / BK;
  const int total = nm * nn;

  extern __shared__ __align__(1024) uint8_t sm[];
  __shared__ int s_compact[128]; // sorted compact list of active expert IDs
  __shared__ int s_num_active;
  __shared__ int s_warp_count[4];
  __shared__ int s_warp_base[4];

  constexpr int STORE_BN = 64;
  constexpr int NUM_TMA_ST = 1;
  constexpr int SCD_STAGE = BM * STORE_BN * 2;
  constexpr int SCD_TOT = SCD_STAGE * NUM_TMA_ST;
  constexpr int SA = BM * BK, SB = BN * BK;
  constexpr int SFA_SIZE = 128 * 4;
  constexpr int SFB_SIZE = BN * 4;

  auto sCD = [&](int s) -> uint8_t * { return sm + s * SCD_STAGE; };
  auto sA = [&](int s) -> uint8_t * { return sm + SCD_TOT + s * SA; };
  auto sB = [&](int s) -> uint8_t * {
    return sm + SCD_TOT + NS * SA + s * SB;
  };
  auto sSFA = [&](int s) -> uint32_t * {
    return (uint32_t *)(sm + SCD_TOT + NS * (SA + SB) + s * SFA_SIZE);
  };
  auto sSFB = [&](int s) -> uint32_t * {
    return (uint32_t *)(sm + SCD_TOT + NS * (SA + SB) + NS * SFA_SIZE +
                        s * SFB_SIZE);
  };

  int bar_base = SCD_TOT + NS * (SA + SB + SFA_SIZE + SFB_SIZE);
  bar_base = (bar_base + 7) & ~7;
  auto bars = reinterpret_cast<uint64_t *>(sm + bar_base);
  int bf = __cvta_generic_to_shared(bars);
  int be = bf + NS * 8;
  int bsf = be + NS * 8;
  int btf = bsf + NS * 8;
  int bte = btf + NE * 8;
  auto tp = reinterpret_cast<uint32_t *>(bars + NS * 3 + NE * 2);

  constexpr int SF_BLOCK_N = ((BN + 127) / 128) * 128;
  constexpr int TMEM_SFA_COLS = 128 / 32;
  constexpr int TMEM_SFB_COLS = SF_BLOCK_N / 32;
  constexpr int TMEM_SFA = NE * BN;
  constexpr int TMEM_SFB = TMEM_SFA + TMEM_SFA_COLS;
  constexpr int TMEM_TOTAL = NE * BN + TMEM_SFA_COLS + TMEM_SFB_COLS;
  constexpr int TCA = TMEM_TOTAL <= 32    ? 32
                      : TMEM_TOTAL <= 64  ? 64
                      : TMEM_TOTAL <= 128 ? 128
                      : TMEM_TOTAL <= 256 ? 256
                                          : 512;

  // ── Init: prefetch tensormaps, init mbarriers, alloc TMEM ──
  if (wid == 0 && detail::elect_one_sync_impl()) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(ta_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tb_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tsfa_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tsfb_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(td_ptr));
  }
  if (wid == 1 && detail::elect_one_sync_impl()) {
    for (int i = 0; i < NS; i++) {
      detail::mb_init_impl(bf + i * 8, 1);
      detail::mb_init_impl(be + i * 8, 1);
      detail::mb_init_impl(bsf + i * 8, 32);
    }
    for (int i = 0; i < NE; i++) {
      detail::mb_init_impl(btf + i * 8, 1);
      detail::mb_init_impl(bte + i * 8, 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (wid == 2) {
    int a = __cvta_generic_to_shared(tp);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            a),
        "r"(TCA));
  }
  __syncthreads();
  const uint32_t taddr = *tp;

  // ── COMPACT PROLOGUE: build deterministic sorted active-expert list ──
  // Uses warp-ballot to produce a CONSISTENT ordering across all blocks.
  // atomicAdd scatter was non-deterministic → different blocks used different
  // orderings → duplicate / missing tiles. Fix: warp-ballot prefix scan.
  {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (warp_id < 4) {
      int expert_id = warp_id * 32 + lane_id;
      int is_active = (active_expert_mask != nullptr)
                          ? active_expert_mask[expert_id]
                          : 1;
      uint32_t ballot = __ballot_sync(0xFFFFFFFF, is_active);
      int local_count = __popc(ballot);
      if (lane_id == 0)
        s_warp_count[warp_id] = local_count;
    }
    __syncthreads();

    if (tid == 0) {
      int tot = 0;
      for (int w = 0; w < 4; w++) {
        s_warp_base[w] = tot;
        tot += s_warp_count[w];
      }
      s_num_active = tot;
    }
    __syncthreads();

    if (tid / 32 < 4) {
      int warp_id2 = tid / 32;
      int lane_id2 = tid % 32;
      int expert_id = warp_id2 * 32 + lane_id2;
      int is_active = (active_expert_mask != nullptr)
                          ? active_expert_mask[expert_id]
                          : 1;
      uint32_t ballot = __ballot_sync(0xFFFFFFFF, is_active);
      if (is_active) {
        int local_rank = __popc(ballot & ((1u << lane_id2) - 1u));
        int global_rank = s_warp_base[warp_id2] + local_rank;
        s_compact[global_rank] = expert_id;
      }
    }
    __syncthreads();
  }

  const int num_active = s_num_active;

#ifdef MPK_GG_DUMP_NUMACTIVE
  // DIAGNOSTIC (env-gated via MPK_EXTRA_NVCC_DEFINES="-DMPK_GG_DUMP_NUMACTIVE"):
  // settle the "group-GEMM 2us is too fast" suspicion. Prints, once per
  // group-GEMM task instance that worker 0 participates in, the runtime active-
  // expert count read from active_expert_mask (popcount). K distinguishes
  // W13 (K=7168) from W2 (K=1024). num_active≈4 at bs=1 decode + slowCTA≈FLOP
  // floor => the kernel does REAL whole-expert work (2us was a P50 artifact);
  // num_active==0 => null-output bug (garbage active-mask binding).
  if (worker_idx == 0 && tid == 0) {
    printf("[GG_DUMP] N=%d K=%d E=%d M_total=%d num_active=%d total_tiles=%d\n",
           N, K, E, M_total, num_active, total);
  }
#endif

  if (wid < 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
  }

  constexpr uint32_t base_idesc =
      ((uint32_t)(BN / 8) << 17) | (1u << 23) | ((uint32_t)(BM / 128) << 27);

  // Resolve iteration index to (bm, bn) tile coordinates.
  // All warps use the same s_compact[] so tile assignments are consistent.
  auto resolve = [&](int iter, int &bm, int &bn) -> int {
    int bidx = iter * num_workers + worker_idx;
    if (bidx >= num_active * nn)
      return 1; // break
    int ae = bidx / nn;
    bn = bidx % nn;
    bm = s_compact[ae]; // deterministic expert ID from sorted compact list
    return 0;
  };

  // ── WARP 0: TMA LOAD ──────────────────────────────────────────────────────
  if (wid == 0 && detail::elect_one_sync_impl()) {
    int gki = 0;
    for (int iter = 0;;iter++) {
      int bm, bn;
      int r = resolve(iter, bm, bn);
      if (r == 1)
        break;
      // m_indices[bm*BM] gives the true expert_id in MPK's permuted layout.
      // In the dense-padded layout used here, expert_id == bm.
      // For full MPK compatibility: use __ldg(m_indices + bm * BM) if needed.
      int om = bm * BM;
      int expert_id = (m_indices != nullptr) ? __ldg(m_indices + bm * BM) : bm;
      int on_b = expert_id * N + bn * BN;
      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int ph = (gki / NS) & 1;
        detail::mb_wait_impl(be + s * 8, ph ^ 1);
        int mb = bf + s * 8;
        int as_ = __cvta_generic_to_shared(sA(s));
        int bs_ = __cvta_generic_to_shared(sB(s));
        detail::tma_ld_impl(as_, ta_ptr, ki * BK, om, mb);
        detail::tma_ld_impl(bs_, tb_ptr, ki * BK, on_b, mb);
        int tx = SA + SB;
        if (ki % SF_PER_LOAD == 0) {
          int sfas_ = __cvta_generic_to_shared(sSFA(s));
          int sfbs_ = __cvta_generic_to_shared(sSFB(s));
          int sf_k = ki / SF_PER_LOAD;
          detail::tma_ld_impl(sfas_, tsfa_ptr, om, sf_k, mb);
          detail::tma_ld_impl(sfbs_, tsfb_ptr, on_b, sf_k, mb);
          tx += SFA_SIZE + SFB_SIZE;
        }
        detail::mb_arrive_tx_impl(mb, tx);
      }
    }
  }
  // ── WARP 2: UTCCP TRANSPOSE ───────────────────────────────────────────────
  else if (wid == 2) {
    auto utccp_transpose = [&](uint32_t *ptr) {
      uint32_t v[4];
#pragma unroll
      for (int i = 0; i < 4; i++)
        v[i] = detail::ld_shared_u32_impl(ptr + (i ^ (lid >> 3)) * 32 + lid);
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; i++)
        detail::st_shared_u32_impl(ptr + lid * 4 + (i ^ (lid >> 3)), v[i]);
    };
    int gki = 0;
    for (int iter = 0;;iter++) {
      int bm, bn;
      int r = resolve(iter, bm, bn);
      if (r == 1)
        break;
      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int ph = (gki / NS) & 1;
        detail::mb_wait_impl(bf + s * 8, ph);
        if (ki % SF_PER_LOAD == 0) {
          utccp_transpose(sSFA(s));
          for (int b = 0; b < SF_BLOCK_N; b += 128) {
            utccp_transpose(sSFB(s) + b);
          }
          asm volatile("fence.proxy.async.shared::cta;");
        }
        detail::mb_arrive_impl(bsf + s * 8);
      }
    }
  }
  // ── WARP 1: MMA ISSUE ─────────────────────────────────────────────────────
  else if (wid == 1 && detail::elect_one_sync_impl()) {
    int gki = 0, work = 0;
    for (int iter = 0;;iter++) {
      int bm, bn;
      int r = resolve(iter, bm, bn);
      if (r == 1)
        break;
      int accum_idx = work % NE;
      int accum_ph = (work / NE) & 1;
      detail::mb_wait_impl(bte + accum_idx * 8, accum_ph ^ 1);
      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int ph = (gki / NS) & 1;
        detail::mb_wait_impl(bsf + s * 8, ph);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (ki % SF_PER_LOAD == 0) {
          int sfas_ = __cvta_generic_to_shared(sSFA(s));
          uint64_t sfa_desc = detail::mkdesc_sf_impl(sfas_);
          asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                           TMEM_SFA),
                       "l"(sfa_desc));
          for (int b = 0; b < SF_BLOCK_N / 128; b++) {
            int sfbs_ = __cvta_generic_to_shared(sSFB(s) + b * 128);
            uint64_t sfb_desc = detail::mkdesc_sf_impl(sfbs_);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    TMEM_SFB + b * 4),
                "l"(sfb_desc));
          }
        }
        uint32_t sf_id = ki % SF_PER_LOAD;
        uint32_t idesc = base_idesc | (sf_id << 4) | (sf_id << 29);
        int as_ = __cvta_generic_to_shared(sA(s));
        int bs_ = __cvta_generic_to_shared(sB(s));
        uint32_t tc = taddr + accum_idx * BN;
        for (int k = 0; k < BK / UK; k++) {
          uint64_t ad = detail::mkdesc_impl(as_ + k * UK);
          uint64_t bd = detail::mkdesc_impl(bs_ + k * UK);
          uint32_t en = (ki > 0 || k > 0) ? 1u : 0u;
          asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, "
              "%2, %3, [%5], [%6], p;\n\t"
              "}\n" ::"r"(tc),
              "l"(ad),
              "l"(bd),
              "r"(idesc),
              "r"(en),
              "r"(TMEM_SFA),
              "r"(TMEM_SFB));
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster."
            "b64 [%0];" ::"r"(be + s * 8)
            : "memory");
        if (ki == nk - 1) {
          asm volatile("tcgen05.fence::before_thread_sync;");
          asm volatile(
              "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
              "cluster.b64 [%0];" ::"r"(btf + accum_idx * 8)
              : "memory");
        }
      }
      work++;
    }
  }
  // ── WARPS 4-7: EPILOGUE (TMEM → SMEM → TMA STORE) ────────────────────────
  else if (wid >= 4) {
    const int ew = wid - 4;
    uint32_t tma_st = 0;
    int work = 0;
    for (int iter = 0;;iter++) {
      int bm, bn;
      int r = resolve(iter, bm, bn);
      if (r == 1)
        break;
      int om = bm * BM;
      int on = bn * BN;
      int accum_idx = work % NE;
      int accum_ph = (work / NE) & 1;
      detail::mb_wait_impl(btf + accum_idx * 8, accum_ph);
      asm volatile("tcgen05.fence::after_thread_sync;");
      constexpr int NUM_N_ST = BN / STORE_BN;
#pragma unroll
      for (int si = 0; si < NUM_N_ST; si++) {
        if (ew == 0)
          asm volatile(
              "cp.async.bulk.wait_group.read %0;" ::"n"(NUM_TMA_ST - 1)
              : "memory");
        asm volatile("bar.sync 6, 128;");
#pragma unroll
        for (int i = 0; i < 8; i++) {
          uint32_t row = lid, col = i ^ (row & 7u);
          uint32_t tc = accum_idx * BN + si * STORE_BN + i * 8;
          uint32_t so = ew * 32 * 128 + row * 128 + col * 16;
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
          asm volatile(
              "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
              "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
              : "=r"(v0),
                "=r"(v1),
                "=r"(v2),
                "=r"(v3),
                "=r"(v4),
                "=r"(v5),
                "=r"(v6),
                "=r"(v7)
              : "r"(tc));
          asm volatile("tcgen05.wait::ld.sync.aligned;");
          uint32_t b0, b1, b2, b3;
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "r"(v0), "r"(v1));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "r"(v2), "r"(v3));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "r"(v4), "r"(v5));
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "r"(v6), "r"(v7));
          uint32_t sa =
              __cvta_generic_to_shared(sCD(tma_st) + so);
          asm volatile(
              "st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sa),
              "r"(b0),
              "r"(b1),
              "r"(b2),
              "r"(b3)
              : "memory");
        }
        if (si == NUM_N_ST - 1) {
          asm volatile("tcgen05.fence::before_thread_sync;");
          detail::mb_arrive_impl(bte + accum_idx * 8);
        }
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("bar.sync 6, 128;");
        if (ew == 0 && detail::elect_one_sync_impl()) {
          uint64_t dd = reinterpret_cast<uint64_t>(td_ptr);
          uint32_t sp = __cvta_generic_to_shared(sCD(tma_st));
          asm volatile(
              "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, "
              "{%2, %3}], [%1];" ::"l"(dd),
              "r"(sp),
              "r"(on + si * STORE_BN),
              "r"(om)
              : "memory");
          asm volatile("cp.async.bulk.commit_group;");
        }
        tma_st = (tma_st + 1) % NUM_TMA_ST;
      }
      work++;
    }
    if (ew == 0)
      asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(0) : "memory");
    if (ew == 1)
      asm volatile(
          "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
          "r"(TCA));
  }
#endif // __CUDA_ARCH__ >= 1000
}

} // namespace fp8_group_gemm_largem_compact
} // namespace kernel
