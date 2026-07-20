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

// Shared body for grouped FP8 block-scaled GEMM tasks (smallm + largem).
// Adapted from
// cpp_examples/blackwell_fp8_gemm/fp8_group_gemm_dsv3_decode_sm100.cu (v007,
// ferret-generated). Beats current DeepGEMM 1.05-1.29x across MPE 1-1024 on
// E=32 + gate_up (K=7168 N=4096) / down (K=2048 N=7168).
//
// Two variants share this body, differing only in (BN, NS):
//   smallm: BN=64,  NS=8  (used when K>4096 && MPE<=8 — gate_up_M{1,4,8})
//   largem: BN=128, NS=6  (everything else — most shapes)
//
// Architecture: tcgen05.mma.kind::mxf8f6f4.block_scale (fused hardware dequant
// with UE8M0 scales), 4 warp roles (TMA load / UTCCP transpose / MMA issue /
// epilogue with TMA store), setmaxnreg register reallocation.
//
// Inputs (all FP8 e4m3 except scales). All shapes use PyTorch convention
// (outermost first, innermost last):
//   A_fp8    [M_total, K]              row-major, K innermost
//   B_fp8    [E, N, K]                 per-expert weights, K innermost
//                                      (flattens to [E*N, K] for the kernel)
//   sfa      [num_sf_k, M_total]       packed UE8M0 uint32 (4 UE8M0/uint32);
//                                      M_total innermost — transposed vs. the
//                                      natural [M_total, num_sf_k] producer
//                                      layout (see transpose_scale_sm100)
//   sfb      [num_sf_k, E*N]           packed UE8M0 uint32 (4 UE8M0/uint32);
//                                      E*N innermost — same transposition
//   m_indices[M_total]                 int32, expert id per row (rows in
//                                      [bm*BM, (bm+1)*BM) must share expert)
//   D_bf16   [M_total, N]              bf16 output, N innermost
//
// MPK adaptation: each task instance handles a slice of (bm, bn) tiles via
// persistent loop strided by num_workers. Register layer with
// grid_dim=(num_workers, 1, 1).
//
// BN/NS dispatch (caller picks at register time):
//   K > 4096 && MPE <= 8 → BN=64, NS=8  (gate_up small M)
//   else                 → BN=128, NS=6 (everything else)

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace kernel {
namespace fp8_group_gemm_common {

__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px, %1;\n\t@%%px "
               "mov.s32 %0, 1;\n\t}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}
__device__ __forceinline__ void mb_init(int a, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(c));
}
__device__ __forceinline__ void mb_wait(int a, int p) {
  asm volatile(
      "{\n\t.reg .pred "
      "P1;\n\tLW:\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, "
      "[%0], %1, %2;\n\t@P1 bra.uni DN;\n\tbra.uni LW;\n\tDN:\n\t}" ::"r"(a),
      "r"(p),
      "r"(0x989680));
}
__device__ __forceinline__ void mb_arrive(int a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a)
               : "memory");
}
__device__ __forceinline__ void mb_arrive_tx(int a, int s) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(a),
      "r"(s)
      : "memory");
}
__device__ __forceinline__ void
    tma_ld(int d, void const *t, int x, int y, int m) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
               "tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(d),
               "l"(t),
               "r"(x),
               "r"(y),
               "r"(m)
               : "memory");
}
__device__ __forceinline__ constexpr uint64_t denc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}
__device__ __forceinline__ uint64_t mkdesc(int a) {
  return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}
__device__ __forceinline__ uint64_t mkdesc_sf(int a) {
  return denc(a) | (denc(128) << 32ULL) | (1ULL << 46ULL);
}
__device__ __forceinline__ uint32_t ld_shared_u32_addr(uint32_t addr) {
  uint32_t r;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
  return r;
}
__device__ __forceinline__ void st_shared_u32_addr(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v));
}

template <int BN, int NS>
__device__ __noinline__ void task_impl_tpl(
    // All shapes below are PyTorch convention (innermost dim is last).
    CUtensorMap const *ta_ptr, // A:   [M_total, K]         FP8, K innermost
    CUtensorMap const *tb_ptr, // B:   [E*N, K]             FP8, K innermost
    CUtensorMap const
        *tsfa_ptr, // SFA: [num_sf_k, M_total]  uint32, M innermost
    CUtensorMap const
        *tsfb_ptr, // SFB: [num_sf_k, E*N]      uint32, E*N innermost
    CUtensorMap const *td_ptr, // D:   [M_total, N]         BF16, N innermost
    int const *__restrict__ m_indices,
    int const *__restrict__ active_expert_mask, // [E] int32, 0 = expert
                                                // received no routings this
                                                // iter → skip its tile rows.
                                                // nullptr = legacy / always
                                                // process every tile.
    int const M_total,
    int const N,
    int const K,
    int const E,
    int const worker_idx,
    int const num_workers) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  constexpr int BM = 128, BK = 128, UK = 32;
  constexpr int SF_PER_LOAD = 4; // 4 UE8M0 packed per uint32
  constexpr int NE = 2;
  int const tid = threadIdx.x, wid = tid / 32;
  uint32_t const lid = tid % 32;
  int const nk = (K + BK - 1) / BK;
  int const nn = (N + BN - 1) / BN;
  int const nm = (M_total + BM - 1) / BM;
  int const total = nm * nn;

  // 1024-byte aligned dynamic SMEM (MPK static prefix may leave us
  // 128-aligned).
  extern __shared__ __align__(1024) uint8_t sm_raw_fp8group[];
  int sb_base = __cvta_generic_to_shared(sm_raw_fp8group);
  int sb_aligned = (sb_base + 1023) & ~1023;

  constexpr int STORE_BN = 64;
  constexpr int NUM_TMA_ST = 1;
  constexpr int SCD_STAGE = BM * STORE_BN * 2;
  constexpr int SCD_TOT = SCD_STAGE * NUM_TMA_ST;
  constexpr int SA = BM * BK, SB = BN * BK;
  constexpr int SFA_SIZE = 128 * 4;
  constexpr int SFB_SIZE = BN * 4;

  // Shared addresses (int) for use with PTX instructions.
  auto sCD_addr = [&](int s) { return sb_aligned + s * SCD_STAGE; };
  auto sA_addr = [&](int s) { return sb_aligned + SCD_TOT + s * SA; };
  auto sB_addr = [&](int s) { return sb_aligned + SCD_TOT + NS * SA + s * SB; };
  auto sSFA_addr = [&](int s) {
    return sb_aligned + SCD_TOT + NS * (SA + SB) + s * SFA_SIZE;
  };
  auto sSFB_addr = [&](int s) {
    return sb_aligned + SCD_TOT + NS * (SA + SB) + NS * SFA_SIZE + s * SFB_SIZE;
  };
  int bar_base = sb_aligned + SCD_TOT + NS * (SA + SB + SFA_SIZE + SFB_SIZE);
  bar_base = (bar_base + 7) & ~7;
  int bf = bar_base;
  int be = bf + NS * 8;
  int bsf = be + NS * 8;
  int btf = bsf + NS * 8;
  int bte = btf + NE * 8;
  int tp_addr = bte + NE * 8;

  constexpr int SF_BLOCK_M = 128;
  constexpr int SF_BLOCK_N = ((BN + 127) / 128) * 128;
  constexpr int TMEM_SFA_COLS = SF_BLOCK_M / 32;
  constexpr int TMEM_SFB_COLS = SF_BLOCK_N / 32;
  constexpr int TMEM_SFA = NE * BN;
  constexpr int TMEM_SFB = TMEM_SFA + TMEM_SFA_COLS;
  constexpr int TMEM_TOTAL = NE * BN + TMEM_SFA_COLS + TMEM_SFB_COLS;
  constexpr int TCA = TMEM_TOTAL <= 32    ? 32
                      : TMEM_TOTAL <= 64  ? 64
                      : TMEM_TOTAL <= 128 ? 128
                      : TMEM_TOTAL <= 256 ? 256
                                          : 512;

  if (wid == 0 && elect_one_sync()) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(ta_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tb_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tsfa_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tsfb_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(td_ptr));
  }
  if (wid == 1 && elect_one_sync()) {
    for (int i = 0; i < NS; i++) {
      mb_init(bf + i * 8, 1);
      mb_init(be + i * 8, 1);
      mb_init(bsf + i * 8, 32);
    }
    for (int i = 0; i < NE; i++) {
      mb_init(btf + i * 8, 1);
      mb_init(bte + i * 8, 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  // TMEM allocation is issued by warp 0, unchained from the mb_init if/else.
  // CUTLASS's TMEM allocator requires that "for repeated allocations, the same
  // warp must be used to issue all allocations"
  // (deps/cutlass/include/cute/arch/tmem_allocator_sm100.hpp:59-73). MPK never
  // issues tcgen05.relinquish_alloc_permit, so that requirement spans every
  // task a persistent worker CTA executes, not just one kernel. Every task
  // that is LIVE in the DSv3 build (linear_sm100_mpk and the MLA MTP decode
  // family) allocates from warp 0, so this file allocating from warp 2 was an
  // inconsistency across task boundaries. NOTE the tree is not uniform:
  // mla_decode_sm100.cuh:108 and simple_linear_sm100.cuh:148 still allocate
  // from warp 1 (both are dead code -- neither appears in any generated
  // megakernel), and fp8_group_gemm_sm100_common.cuh still DEALLOCATES from
  // warp 5. This change is an invariant/hygiene fix: it was measured NOT to
  // resolve the observed prefill fault, so it is not a proven root cause.
  // Seed the allocation slot with a sentinel and separate the mbarrier-init
  // writes from the allocation with a CTA barrier. Without this barrier the
  // warp-1 mbarrier inits and the warp-0 tcgen05.alloc both write this shared
  // region with NO ordering between them, and the allocation result at
  // tp_addr (which sits immediately after the mbarrier array) could be
  // overwritten by an mbarrier-init word. The consuming warps then read an
  // mbarrier state value as a TMEM address and tcgen05.mma faults with
  // "Warp Out of range Address". The sentinel also removes the ambiguity that
  // a never-written slot reading 0x00000000 is indistinguishable from a legal
  // allocation base of (lane 0, column 0).
  if (wid == 0 && elect_one_sync()) {
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(tp_addr), "r"(0xDEADBEEFu)
                 : "memory");
  }
  __syncthreads();
  if (wid == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(tp_addr),
        "r"(TCA));
  }
  __syncthreads();
  uint32_t taddr;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(taddr) : "r"(tp_addr));

  if (wid < 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
  }

  constexpr uint32_t base_idesc =
      ((uint32_t)(BN / 8) << 17) | (1u << 23) | ((uint32_t)(BM / 128) << 27);

  // ====== WARP 0: TMA LOAD ======
  if (wid == 0 && elect_one_sync()) {
    int gki = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      int bm = bidx / nn, bn = bidx % nn;
      int m_start = bm * BM;
      int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
      // Contract: all BM rows in this tile share the same expert_id
      // (moe_permute guarantees tile-aligned routing).
      // Skip this whole tile when the expert received no routings this
      // iter (mask written by moe_permute). All warps make the same
      // decision deterministically from m_indices + active_expert_mask
      // so the mbarrier handshakes stay consistent. nullptr mask = always
      // process every tile (legacy / non-MoE callers).
      if (active_expert_mask != nullptr &&
          !__ldg(active_expert_mask + expert_id)) {
        continue;
      }
      int om = m_start;
      int on = expert_id * N + bn * BN;

      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int ph = (gki / NS) & 1;
        mb_wait(be + s * 8, ph ^ 1);
        int mb = bf + s * 8;
        int as_ = sA_addr(s);
        int bs_ = sB_addr(s);
        tma_ld(as_, ta_ptr, ki * BK, om, mb);
        tma_ld(bs_, tb_ptr, ki * BK, on, mb);
        int tx = SA + SB;
        if (ki % SF_PER_LOAD == 0) {
          int sfas_ = sSFA_addr(s);
          int sfbs_ = sSFB_addr(s);
          int sf_k = ki / SF_PER_LOAD;
          tma_ld(sfas_, tsfa_ptr, om, sf_k, mb);
          tma_ld(sfbs_, tsfb_ptr, on, sf_k, mb);
          tx += SFA_SIZE + SFB_SIZE;
        }
        mb_arrive_tx(mb, tx);
      }
    }
  }
  // ====== WARP 2: UTCCP TRANSPOSE ======
  else if (wid == 2) {
    auto utccp_transpose = [&](int ptr_addr) {
      uint32_t v[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v[i] = ld_shared_u32_addr(ptr_addr + ((i ^ (lid >> 3)) * 32 + lid) * 4);
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; i++) {
        st_shared_u32_addr(ptr_addr + (lid * 4 + (i ^ (lid >> 3))) * 4, v[i]);
      }
    };

    int gki = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      // Same skip check as warp 0 — all warps must short-circuit in
      // lockstep, otherwise the mbarrier handshake for this iter's stage
      // is unsynced.
      if (active_expert_mask != nullptr) {
        int bm_skip = bidx / nn;
        int m_start_skip = bm_skip * BM;
        int expert_skip =
            (m_start_skip < M_total) ? __ldg(m_indices + m_start_skip) : 0;
        if (!__ldg(active_expert_mask + expert_skip)) {
          continue;
        }
      }
      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int ph = (gki / NS) & 1;
        mb_wait(bf + s * 8, ph);
        if (ki % SF_PER_LOAD == 0) {
          utccp_transpose(sSFA_addr(s));
          for (int b = 0; b < SF_BLOCK_N; b += 128) {
            utccp_transpose(sSFB_addr(s) + b * 4);
          }
          asm volatile("fence.proxy.async.shared::cta;");
        }
        mb_arrive(bsf + s * 8);
      }
    }
  }
  // ====== WARP 1: MMA ISSUE ======
  else if (wid == 1 && elect_one_sync()) {
    int gki = 0;
    // Accumulator ring position must count PROCESSED tiles only (like gki),
    // NOT raw `iter`: btf/bte arrivals happen only for processed tiles, so
    // phasing the ring on `iter` desyncs wait-parity from actual completions
    // whenever active_expert_mask skips tiles (mixed skip = mb_wait spins
    // forever; all-skip / nullptr-mask paths never exposed it). pacc == iter
    // exactly when nothing is skipped, so the legacy path is unchanged.
    int pacc = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      if (active_expert_mask != nullptr) {
        int bm_skip = bidx / nn;
        int m_start_skip = bm_skip * BM;
        int expert_skip =
            (m_start_skip < M_total) ? __ldg(m_indices + m_start_skip) : 0;
        if (!__ldg(active_expert_mask + expert_skip)) {
          continue;
        }
      }
      int accum_idx = pacc % NE;
      int accum_ph = (pacc / NE) & 1;
      pacc++;
      mb_wait(bte + accum_idx * 8, accum_ph ^ 1);

      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int ph = (gki / NS) & 1;
        mb_wait(bsf + s * 8, ph);
        asm volatile("tcgen05.fence::after_thread_sync;");

        if (ki % SF_PER_LOAD == 0) {
          int sfas_ = sSFA_addr(s);
          uint64_t sfa_desc = mkdesc_sf(sfas_);
          asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                           TMEM_SFA),
                       "l"(sfa_desc));
          for (int b = 0; b < SF_BLOCK_N / 128; b++) {
            int sfbs_ = sSFB_addr(s) + b * 128 * 4;
            uint64_t sfb_desc = mkdesc_sf(sfbs_);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    TMEM_SFB + b * 4),
                "l"(sfb_desc));
          }
        }

        uint32_t sf_id = ki % SF_PER_LOAD;
        uint32_t idesc = base_idesc | (sf_id << 4) | (sf_id << 29);
        int as_ = sA_addr(s);
        int bs_ = sB_addr(s);
        uint32_t tc = taddr + accum_idx * BN;

        for (int k = 0; k < BK / UK; k++) {
          uint64_t ad = mkdesc(as_ + k * UK);
          uint64_t bd = mkdesc(bs_ + k * UK);
          uint32_t en = (ki > 0 || k > 0) ? 1u : 0u;
          asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                       "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
                       "[%0], %1, %2, %3, [%5], [%6], p;\n\t}\n" ::"r"(tc),
                       "l"(ad),
                       "l"(bd),
                       "r"(idesc),
                       "r"(en),
                       "r"(TMEM_SFA),
                       "r"(TMEM_SFB));
        }

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared:"
                     ":cluster.b64 [%0];" ::"r"(be + s * 8)
                     : "memory");
        if (ki == nk - 1) {
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one."
                       "shared::cluster.b64 [%0];" ::"r"(btf + accum_idx * 8)
                       : "memory");
        }
      }
    }
  }
  // ====== WARPS 4-7: EPILOGUE (TMEM → SMEM → TMA STORE) ======
  else if (wid >= 4) {
    int const ew = wid - 4;
    uint32_t tma_st = 0;
    // Processed-tile counter for the accumulator ring — see warp 1: phasing
    // btf/bte on raw `iter` deadlocks under a mixed active_expert_mask.
    int pacc = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      int bm = bidx / nn, bn = bidx % nn;
      int m_start = bm * BM;
      int expert_id = (m_start < M_total) ? __ldg(m_indices + m_start) : 0;
      if (active_expert_mask != nullptr &&
          !__ldg(active_expert_mask + expert_id)) {
        continue;
      }
      int om = m_start;
      int on = bn * BN;
      int accum_idx = pacc % NE;
      int accum_ph = (pacc / NE) & 1;
      pacc++;
      mb_wait(btf + accum_idx * 8, accum_ph);
      asm volatile("tcgen05.fence::after_thread_sync;");
      constexpr int NUM_N_ST = BN / STORE_BN;
#pragma unroll
      for (int si = 0; si < NUM_N_ST; si++) {
        if (ew == 0) {
          asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(NUM_TMA_ST - 1)
                       : "memory");
        }
        // Named barrier 6 (free in MPK convention; bar.sync 0 is the
        // implicit __syncthreads() with count=256 used elsewhere).
        asm volatile("bar.sync 6, 128;");
#pragma unroll
        for (int i = 0; i < 8; i++) {
          uint32_t row = lid, col = i ^ (row & 7u);
          uint32_t tc = accum_idx * BN + si * STORE_BN + i * 8;
          uint32_t so = ew * 32 * 128 + row * 128 + col * 16;
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
          asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
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
          uint32_t sa = sCD_addr(tma_st) + so;
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(sa),
                       "r"(b0),
                       "r"(b1),
                       "r"(b2),
                       "r"(b3)
                       : "memory");
        }
        if (si == NUM_N_ST - 1) {
          asm volatile("tcgen05.fence::before_thread_sync;");
          mb_arrive(bte + accum_idx * 8);
        }
        asm volatile("fence.proxy.async.shared::cta;");
        // Named barrier 6 (free in MPK convention; bar.sync 0 is the
        // implicit __syncthreads() with count=256 used elsewhere).
        asm volatile("bar.sync 6, 128;");
        if (ew == 0 && elect_one_sync()) {
          uint64_t dd = (uint64_t)td_ptr;
          uint32_t sp = sCD_addr(tma_st);
          asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                       "[%0, {%2, %3}], [%1];" ::"l"(dd),
                       "r"(sp),
                       "r"(on + si * STORE_BN),
                       "r"(om)
                       : "memory");
          asm volatile("cp.async.bulk.commit_group;");
        }
        tma_st = (tma_st + 1) % NUM_TMA_ST;
      }
    }
    if (ew == 0) {
      asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(0) : "memory");
    }
    // C3: the tcgen05.dealloc used to sit HERE, under `if (ew == 1)` -- issued
    // by warp 5 only, while the matching tcgen05.alloc is issued by warp 0
    // (:250). That is (a) a convergence violation -- warps 0-3 have already
    // fallen out of this if-else chain, so a `.sync.aligned` dealloc here has
    // no guarantee the CTA is converged -- and (b) a violation of the TMEM
    // allocator contract, which requires alloc and dealloc to be issued by the
    // SAME warp (deps/cutlass/include/cute/arch/tmem_allocator_sm100.hpp:59-73:
    // "for repeated allocations, the same warp must be used to issue all
    // allocations"). MPK never issues tcgen05.relinquish_alloc_permit, so that
    // requirement spans task boundaries for the entire life of the persistent
    // CTA. The identical defect was already found and fixed in the sibling
    // fp8_group_gemm_largem_compact_sm100.cuh:616-631 ("C3"); this file was
    // simply never updated, and it is the only remaining live alloc/dealloc
    // warp split in the megakernel. Moved below.
  }
  // C3 FIX (mirrors fp8_group_gemm_largem_compact_sm100.cuh:621-631): every
  // warp falls through the if-else chain to this point and this task body
  // contains no early `return`, so a full-CTA barrier here is safe. Sync all
  // 256 threads (all 8 warp-specialized loops have exited, so no warp can
  // still be reading TMEM), then deallocate from warp 0 -- the same warp that
  // allocated.
  asm volatile("bar.sync 10, 256;");
  if (wid == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(TCA));
  }
#endif
}

// SMEM size for given BN/NS. Caller (wrapper or MPK runtime) must ensure
// the megakernel's MAX_DYNAMIC_SHARED_MEMORY_SIZE >= this value.
template <int BN, int NS>
__host__ __device__ inline constexpr int smem_size_tpl() {
  constexpr int BM = 128, BK = 128, STORE_BN = 64, NE = 2;
  constexpr int SCD_TOT = BM * STORE_BN * 2 * 1; // NUM_TMA_ST=1
  constexpr int SA = BM * BK, SB = BN * BK;
  constexpr int SFA = 128 * 4, SFB = BN * 4;
  int sz = SCD_TOT + NS * (SA + SB + SFA + SFB) + (NS * 3 + NE * 2) * 8 + 8;
  return ((sz + 1023) & ~1023) + 1024; // +1024 for runtime alignment slack
}

} // namespace fp8_group_gemm_common
} // namespace kernel
