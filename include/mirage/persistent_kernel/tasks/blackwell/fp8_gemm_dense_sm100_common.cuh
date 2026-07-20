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

// Shared body for the smallm / mediumm dense FP8 block-scaled GEMM tasks
// (Blackwell SM100, tcgen05.mma kind::f8f6f4). The two variants differ only
// in NE (TMEM circular buffer stages); both call task_impl_tpl<BN, NS, NE>.
//
// Layout:
//   A: FP8 e4m3 [M, K]            row-major,  1×128 group activation scale
//   B: FP8 e4m3 [N, K]            row-major,  128×128 block weight scale
//   C: BF16     [M, N]            row-major
//   sa: float32 [M, K/128]        row-major
//   sb: float32 [N/128, K/128]    row-major
//
// 256 threads/CTA, 128B-swizzle TMA for A/B. Roles:
//   warp 0     = tensormap prefetch + TMA loader + tcgen05.alloc/dealloc
//   warp 1     = MMA issue (also runs mbarrier init)
//   warps 4..7 = epilogue (TMEM read + dequant + write C)
//
// MPK adaptation: each task instance handles tiles striding by `num_workers`
// from `worker_idx`. Register the layer with grid_dim=(num_workers, 1, 1).

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace kernel {
namespace fp8_gemm_dense_common {

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

// C_row_stride: row stride (elements) of the OUTPUT buffer. Defaults to N
// (dense row-major). Pass the parent row width when C is a narrow column
// view of a wider buffer (e.g. the TP2 gate/up halves of mlp_mid) — at M=1
// the stride never matters, but multi-row writes corrupt the parent buffer
// if rows advance by N instead of the view stride.
template <int BN, int NS, int NE>
__device__ __forceinline__ void task_impl_tpl(CUtensorMap const *ta_ptr,
                                              CUtensorMap const *tb_ptr,
                                              float const *__restrict__ sa,
                                              float const *__restrict__ sb,
                                              __nv_bfloat16 *__restrict__ C,
                                              int const M,
                                              int const N,
                                              int const K,
                                              int const worker_idx,
                                              int const num_workers,
                                              int const C_row_stride = -1) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  constexpr int BM = 128, BK = 128, UK = 32;
  int const tid = threadIdx.x, wid = tid / 32;
  int const nn = (N + BN - 1) / BN, nk = (K + BK - 1) / BK;
  int const total = ((M + BM - 1) / BM) * nn;

  extern __shared__ __align__(1024) uint8_t sm_raw_fp8gemm[];
  int sb_base = __cvta_generic_to_shared(sm_raw_fp8gemm);
  int sb_aligned = (sb_base + 1023) & ~1023;

  constexpr int SA = BM * BK, SB = BN * BK;
  auto sA = [&](int s) { return sb_aligned + s * SA; };
  auto sBl = [&](int s) { return sb_aligned + NS * SA + s * SB; };

  // ------------------------------------------------------------------------
  // ASYNC-AGENT SAFETY (2026-07-20): the mbarrier array and the TMEM
  // allocation slot live in STATIC __shared__, deliberately NOT in the
  // `extern __shared__` arena.
  //
  // Every task body's `extern __shared__` declaration aliases ONE arena at ONE
  // base, and a persistent worker CTA runs heterogeneous tasks back-to-back
  // separated only by __syncthreads(). __syncthreads() orders THREADS; it
  // drains no ASYNCHRONOUS agent. A `tcgen05.commit ... mbarrier::arrive` (and
  // a TMA expect_tx completion) keeps writing an mbarrier's state word after
  // the issuing task has nominally ended, so an arena-resident barrier lets a
  // late arrival land in memory the NEXT task has already reused. Whether that
  // faults or corrupts silently depends only on what occupies that byte.
  //
  // This kernel has a provably dangling agent. Warp 1 commits to `be[s]` at the
  // bottom of its MMA loop; the ONLY waiter is warp 0's `mb_wait(be + s*8)` at
  // the TOP of an iteration that never runs once warp 0's loop has exited.
  // Per stage the wait at ring index n unblocks after exactly n arrivals, and
  // both warps perform the same number of iterations, so exactly ONE arrival
  // per stage — NS in total — outlives the task body.
  //
  // Static __shared__ is not part of the arena. nvcc SUMS per-branch statics
  // rather than overlaying them, and places them BELOW the dynamic arena base
  // (measured on sm_100a: six distinct <BN,NS,NE> instantiations received six
  // distinct, non-overlapping addresses, every one of them below the arena
  // base). These bytes therefore belong to this template instantiation alone:
  // a late arrival lands on storage no other task ever reads or writes, which
  // closes the hazard with NO phase arithmetic and NO added wait. The MLA
  // decode family already used this idiom (mla_decode_sm100.cuh's
  // `__shared__ uint64_t mbar_buf[10]`), which is where it was copied from.
  //
  // Do NOT assert a global "every kernel is safe" invariant from a comment --
  // an earlier round of this same investigation shipped three such comments
  // that turned out to be wrong. The invariant is machine-checked instead:
  // scripts/check_async_barrier_placement.py fails the build if any mbarrier
  // operand resolves back to an `extern __shared__` symbol, and carries an
  // allowlist of the remaining exceptions with written reasons.
  //
  // The arena's trailing barrier bytes are now unused; smem_size_tpl() still
  // reserves them, which is deliberate slack (see the note there).
  __shared__ __align__(16) uint64_t sm_bars_fp8gemm[NS * 2 + NE * 2 + 1];
  int bars_addr = static_cast<int>(__cvta_generic_to_shared(sm_bars_fp8gemm));
  int bf = bars_addr;
  int be = bf + NS * 8;
  int btf = be + NS * 8;
  int bte = btf + NE * 8;
  int tp_addr = bars_addr + (NS * 2 + NE * 2) * 8;
#ifdef MPK_DSV3_ASYNC_BAR_ARENA_UNSAFE
  // FAULT INJECTION (default OFF, never ship enabled). Puts this family's
  // barrier block back in the arena, restoring the defect. The sibling
  // fp8_gemm_dense_qout body does the same under this macro, and both share the
  // SAME `sm_raw_fp8gemm` extern symbol and layout formula, so enabling it
  // restores the exact observed collision: byte 64 of the block is
  // simultaneously NE=1's tp_addr, NE=2's bte[0] and NE=4's btf[2]. With this
  // ON the canary and Class-2 probes must FIRE; with it OFF they must be
  // silent. Without the control, "the canary was silent" is unfalsifiable.
  bars_addr = sb_aligned + NS * (SA + SB);
  bf = bars_addr;
  be = bf + NS * 8;
  btf = be + NS * 8;
  bte = btf + NE * 8;
  tp_addr = bars_addr + (NS * 2 + NE * 2) * 8;
#endif
  constexpr int TC = NE * BN;
  constexpr int TCA = TC <= 32    ? 32
                      : TC <= 64  ? 64
                      : TC <= 128 ? 128
                      : TC <= 256 ? 256
                                  : 512;

  if (wid == 0 && elect_one_sync()) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(ta_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tb_ptr));
  }
  if (wid == 1 && elect_one_sync()) {
    for (int i = 0; i < NS; i++) {
      mb_init(bf + i * 8, 1);
      mb_init(be + i * 8, 1);
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
#ifdef MPK_DSV3_TMEM_GUARD_ALL
  // DIAGNOSTIC (compile-time, default OFF). The sibling fp8_gemm_dense_qout
  // kernel validates this base and traps; this one does not, so a base that
  // tcgen05.alloc never published (0xDEADBEEF) or that was clobbered goes
  // straight into tcgen05.mma and yields SILENT garbage rather than a fault.
  // Enabling this turns that silent corruption into a named failure, which is
  // how we test whether the degenerate-output class shares the cause of the
  // seq>2048 fault (where the qout guard is the believed trap source).
  if (!((taddr >> 16) < 128u && ((taddr & 0xFFFFu) + (uint32_t)TCA) <= 512u)) {
    if (wid == 0 && elect_one_sync()) {
      printf("[MPK FATAL] fp8_gemm_dense: invalid TMEM base 0x%08x (requested "
             "%d cols) on block %d. 0xdeadbeef means tcgen05.alloc never wrote "
             "the slot; any other value means it was overwritten.\n",
             taddr,
             (int)TCA,
             (int)blockIdx.x);
    }
#ifdef MPK_DSV3_TMEM_GUARD_NOTRAP
    // Diagnostic mode: do NOT trap. A trapping megakernel never exits, and the
    // printf FIFO only flushes when it FILLS or when the CTA EXITS -- the
    // driver floors the FIFO at 524288 B, which a handful of guard lines never
    // fills, so trapping makes the guard structurally unable to report. Force a
    // legal TMEM base instead: the math is wrong, but the CTA exits, the FIFO
    // flushes, and we learn whether the guard fired at all and with what value.
    taddr = 0u;
#else
    __trap();
#endif
  }
#endif
  constexpr uint32_t idesc =
      (1u << 4) | ((uint32_t)(BN / 8) << 17) | (8u << 24);

  // Stage index + parity MUST come from a CONTINUOUS K-block counter (gk /
  // gki), exactly like the accumulator ring below already does with
  // gki % NE: the old `s = ki % NS` reset the stage cursor every TILE
  // iteration while the mbarriers' real parity stream is continuous, so any
  // multi-tile-iter task (total tiles > num_workers) with nk % NS != 0
  // desynced at the tile boundary — synccheck "Barrier error. Missing wait"
  // at mb_arrive_tx, then cudaErrorLaunchFailure (2026-06-13; this is the
  // long-parked B36 multi-tile-iter bug — split-K merely forced
  // multi-tile-iter early). For single-tile-iter or nk % NS == 0 the
  // continuous form is bit-identical to the old sequence.
  if (wid == 0 && elect_one_sync()) {
    int gk = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      int bm = bidx / nn, bn = bidx % nn;
      int om = bm * BM, on = bn * BN;
      for (int ki = 0; ki < nk; ki++, gk++) {
        int s = gk % NS;
        int p = (gk / NS) & 1;
        mb_wait(be + s * 8, p ^ 1);
        int as_ = sA(s);
        int bs_ = sBl(s);
        int mb = bf + s * 8;
        tma_ld(as_, ta_ptr, ki * BK, om, mb);
        tma_ld(bs_, tb_ptr, ki * BK, on, mb);
        mb_arrive_tx(mb, SA + SB);
      }
    }
  } else if (wid == 1 && elect_one_sync()) {
    int gki = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      for (int ki = 0; ki < nk; ki++, gki++) {
        int s = gki % NS;
        int p = (gki / NS) & 1;
        mb_wait(bf + s * 8, p);
        int ai = gki % NE;
        int ap = (gki / NE) & 1;
        mb_wait(bte + ai * 8, ap ^ 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        int as_ = sA(s);
        int bs_ = sBl(s);
        uint32_t tc = taddr + ai * BN;
        for (int k = 0; k < BK / UK; k++) {
          uint64_t ad = mkdesc(as_ + k * UK), bd = mkdesc(bs_ + k * UK);
          uint32_t en = (k > 0) ? 1u : 0u;
          asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, "
                       "0;\n\ttcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, "
                       "%2, %3, {%5, %6, %7, %8}, p;\n\t}\n" ::"r"(tc),
                       "l"(ad),
                       "l"(bd),
                       "r"(idesc),
                       "r"(en),
                       "r"(0u),
                       "r"(0u),
                       "r"(0u),
                       "r"(0u));
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared:"
                     ":cluster.b64 [%0];" ::"r"(be + s * 8)
                     : "memory");
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared:"
                     ":cluster.b64 [%0];" ::"r"(btf + ai * 8)
                     : "memory");
      }
    }
  } else if (wid >= 4) {
    int const et = tid - 128, ew = wid - 4;
    int gki = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      int bm = bidx / nn, bn = bidx % nn;
      int om = bm * BM, on = bn * BN;
      int mi = om + et;

      float acc[BN];
#pragma unroll
      for (int i = 0; i < BN; i++) {
        acc[i] = 0.0f;
      }

      for (int ki = 0; ki < nk; ki++, gki++) {
        float sfa = (mi < M) ? __ldg(sa + mi * nk + ki) : 0.0f;
        float sfb0 = __ldg(sb + (on / 128) * nk + ki);
        float sfb1 = 0.0f;
        if (BN > 128) {
          sfb1 = __ldg(sb + ((on + 128) / 128) * nk + ki);
        }

        int ai = gki % NE;
        int ap = (gki / NE) & 1;
        mb_wait(btf + ai * 8, ap);
        asm volatile("tcgen05.fence::after_thread_sync;");

        float sf0 = sfa * sfb0, sf1 = sfa * sfb1;
#pragma unroll
        for (int i = 0; i < BN / 16; i++) {
          uint32_t ta_ = taddr + ((ew * 32) << 16) + ai * BN + i * 16;
          float v[16];
          asm volatile(
              "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
              "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
              : "=f"(v[0]),
                "=f"(v[1]),
                "=f"(v[2]),
                "=f"(v[3]),
                "=f"(v[4]),
                "=f"(v[5]),
                "=f"(v[6]),
                "=f"(v[7]),
                "=f"(v[8]),
                "=f"(v[9]),
                "=f"(v[10]),
                "=f"(v[11]),
                "=f"(v[12]),
                "=f"(v[13]),
                "=f"(v[14]),
                "=f"(v[15])
              : "r"(ta_));
          asm volatile("tcgen05.wait::ld.sync.aligned;");
          float sf = (BN <= 128 || i * 16 < 128) ? sf0 : sf1;
#pragma unroll
          for (int j = 0; j < 16; j++) {
            acc[i * 16 + j] += v[j] * sf;
          }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        mb_arrive(bte + ai * 8);
      }

      if (mi < M) {
        long long const c_stride = (C_row_stride > 0) ? C_row_stride : N;
        __nv_bfloat16 *row = C + (long long)mi * c_stride + on;
#pragma unroll
        for (int n = 0; n < BN; n += 16) {
          if (on + n + 15 < N) {
            nv_bfloat162 b0 = __floats2bfloat162_rn(acc[n + 0], acc[n + 1]);
            nv_bfloat162 b1 = __floats2bfloat162_rn(acc[n + 2], acc[n + 3]);
            nv_bfloat162 b2 = __floats2bfloat162_rn(acc[n + 4], acc[n + 5]);
            nv_bfloat162 b3 = __floats2bfloat162_rn(acc[n + 6], acc[n + 7]);
            nv_bfloat162 b4 = __floats2bfloat162_rn(acc[n + 8], acc[n + 9]);
            nv_bfloat162 b5 = __floats2bfloat162_rn(acc[n + 10], acc[n + 11]);
            nv_bfloat162 b6 = __floats2bfloat162_rn(acc[n + 12], acc[n + 13]);
            nv_bfloat162 b7 = __floats2bfloat162_rn(acc[n + 14], acc[n + 15]);
            uint32_t r0 = *reinterpret_cast<uint32_t *>(&b0);
            uint32_t r1 = *reinterpret_cast<uint32_t *>(&b1);
            uint32_t r2 = *reinterpret_cast<uint32_t *>(&b2);
            uint32_t r3 = *reinterpret_cast<uint32_t *>(&b3);
            uint32_t r4 = *reinterpret_cast<uint32_t *>(&b4);
            uint32_t r5 = *reinterpret_cast<uint32_t *>(&b5);
            uint32_t r6 = *reinterpret_cast<uint32_t *>(&b6);
            uint32_t r7 = *reinterpret_cast<uint32_t *>(&b7);
            asm volatile("st.relaxed.cta.global.L1::no_allocate.v4.b32 [%0], "
                         "{%1,%2,%3,%4};" ::"l"(row + n),
                         "r"(r0),
                         "r"(r1),
                         "r"(r2),
                         "r"(r3)
                         : "memory");
            asm volatile("st.relaxed.cta.global.L1::no_allocate.v4.b32 [%0], "
                         "{%1,%2,%3,%4};" ::"l"(row + n + 8),
                         "r"(r4),
                         "r"(r5),
                         "r"(r6),
                         "r"(r7)
                         : "memory");
          } else {
            for (int j = 0; j < 16 && on + n + j < N; j++) {
              row[n + j] = __float2bfloat16(acc[n + j]);
            }
          }
        }
      }
    }
  }

  __syncthreads();
  // The consumer-warp output writes use st.relaxed.cta.global
  // which has CTA-scope semantics — fine when this kernel is launched
  // standalone (cudaLaunch acts as implicit fence between successive
  // launches), but in the MPK persistent megakernel the next task on a
  // different CTA may see stale L2 lines. Add a global-scope memory
  // fence so downstream tasks (rmsnorm-q/kv, rope-k, MLA-KV-gather, etc.)
  // reading qkv_a_out see the GEMM's writes. The fence is cheap (single
  // membar instruction per CTA exit); standalone tests still pass since
  // it's a no-op when no other CTAs are reading.
  asm volatile("membar.gl;" ::: "memory");
  if (wid == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(TCA));
  }
#endif
}

template <int BN, int NS, int NE>
__host__ __device__ inline constexpr int smem_size_tpl() {
  constexpr int BM = 128, BK = 128;
  return NS * (BM * BK + BN * BK) + (NS * 2 + NE * 2) * 8 + 8 + 1024;
}

} // namespace fp8_gemm_dense_common
} // namespace kernel
