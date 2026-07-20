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

// Dense FP8 block-scaled GEMM, decode-only SplitK variant. Adapted from
// fp8_gemm_dense_sm100_common.cuh::task_impl_tpl. Decomposes the K axis
// across SPLIT_K CTAs (per output tile) and uses
// `red.global.add.noftz.bf16x2` PTX (SM100) to atomically accumulate
// partial results into a pre-zeroed BF16 output.
//
// Caller MUST zero-initialize the output tensor before launch — there is
// no kernel-side guard. Use the Python wrapper which prepends a
// tensor_init task on `output`.
//
// Designed for DSv3 decode O_proj: M=128 (mbt; active_rows=1), K=16384
// (= num_local_q_heads * v_head_dim_absorbed for TP=4), N=7168 (hidden
// shard). Stock mediumm kernel = 56 tiles in 1 underutilized wave. With
// SPLIT_K=4 we get 224 tiles in 3 well-utilized waves, each tile doing
// K/4 work (32 K-iters vs 128). Realistic wallclock 64μs -> ~30-40μs.
//
// NOTE on M_TILE: tcgen05.mma.kind::f8f6f4 has MMA_M=128 hardware tile.
// We keep BM=128 (matches the MMA HW shape). True M=16 would require
// swapAB-style transpose of the operands, which is a larger redesign;
// see linear_fp8_swapAB_sm100.cuh for the pattern. The active-row gate
// (runtime_m_mode=3) still avoids stale writes to rows 1..127 by
// skipping global stores when mi >= runtime_m_.

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace kernel {
namespace fp8_gemm_dense_decode_splitk {

__device__ __forceinline__ uint32_t elect_one_sync_splitk() {
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px, %1;\n\t@%%px "
               "mov.s32 %0, 1;\n\t}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mb_init_sk(int a, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(c));
}
__device__ __forceinline__ void mb_wait_sk(int a, int p) {
  asm volatile(
      "{\n\t.reg .pred "
      "P1;\n\tLW:\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, "
      "[%0], %1, %2;\n\t@P1 bra.uni DN;\n\tbra.uni LW;\n\tDN:\n\t}" ::"r"(a),
      "r"(p),
      "r"(0x989680));
}
__device__ __forceinline__ void mb_arrive_sk(int a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a)
               : "memory");
}
__device__ __forceinline__ void mb_arrive_tx_sk(int a, int s) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(a),
      "r"(s)
      : "memory");
}
__device__ __forceinline__ void
    tma_ld_sk(int d, void const *t, int x, int y, int m) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
               "tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(d),
               "l"(t),
               "r"(x),
               "r"(y),
               "r"(m)
               : "memory");
}

__device__ __forceinline__ constexpr uint64_t denc_sk(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}
__device__ __forceinline__ uint64_t mkdesc_sk(int a) {
  return denc_sk(a) | (denc_sk(1024) << 32ULL) | (1ULL << 46ULL) |
         (2ULL << 61ULL);
}

// Atomic bf16x2 add: adds a packed pair of bf16 values to *ptr.
// SM100 PTX requires an explicit scope qualifier on
// `red.global` for inter-CTA atomic correctness. Without it the default
// is `.cta`, which is undefined behavior across SMs and surfaces as
// `cudaErrorLaunchFailure` for SplitK accumulating into the same output
// tile from different SMs. Use `.relaxed.gpu` so the atomic is visible
// across the whole device with relaxed ordering (downstream consumers
// are gated by mbarrier / membar.gl in the megakernel anyway).
__device__ __forceinline__ void red_add_bf16x2(__nv_bfloat16 *ptr,
                                               uint32_t val) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;" ::"l"(ptr),
               "r"(val)
               : "memory");
#else
  // Fallback (slow): scalar atomic on each half via 32-bit CAS.
  (void)ptr;
  (void)val;
#endif
}

template <int BN, int NS, int NE, int SPLIT_K>
__device__ __forceinline__ void
    task_impl_splitk_tpl(CUtensorMap const *ta_ptr,
                         CUtensorMap const *tb_ptr,
                         float const *__restrict__ sa,
                         float const *__restrict__ sb,
                         __nv_bfloat16 *__restrict__ C,
                         int const M,
                         int const N,
                         int const K,
                         int const worker_idx,
                         int const num_workers) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  constexpr int BM = 128, BK = 128, UK = 32;
  static_assert(SPLIT_K >= 1, "SPLIT_K must be >= 1");
  int const tid = threadIdx.x, wid = tid / 32;
  int const nn = (N + BN - 1) / BN, nk = (K + BK - 1) / BK;
  // nk MUST be divisible by SPLIT_K (caller asserts at registration).
  int const nk_slice = nk / SPLIT_K;
  int const mm = (M + BM - 1) / BM;
  int const total = mm * nn * SPLIT_K;

  extern __shared__ __align__(1024) uint8_t sm_raw_fp8gemm_sk[];
  int sb_base = __cvta_generic_to_shared(sm_raw_fp8gemm_sk);
  int sb_aligned = (sb_base + 1023) & ~1023;

  constexpr int SA = BM * BK, SB = BN * BK;
  auto sA = [&](int s) { return sb_aligned + s * SA; };
  auto sBl = [&](int s) { return sb_aligned + NS * SA + s * SB; };

  int bars_addr = sb_aligned + NS * (SA + SB);
  int bf = bars_addr;
  int be = bf + NS * 8;
  int btf = be + NS * 8;
  int bte = btf + NE * 8;
  int tp_addr = bars_addr + (NS * 2 + NE * 2) * 8;
  constexpr int TC = NE * BN;
  constexpr int TCA = TC <= 32    ? 32
                      : TC <= 64  ? 64
                      : TC <= 128 ? 128
                      : TC <= 256 ? 256
                                  : 512;

  if (wid == 0 && elect_one_sync_splitk()) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(ta_ptr));
    asm volatile("prefetch.tensormap [%0];" ::"l"(tb_ptr));
  }
  if (wid == 1 && elect_one_sync_splitk()) {
    for (int i = 0; i < NS; i++) {
      mb_init_sk(bf + i * 8, 1);
      mb_init_sk(be + i * 8, 1);
    }
    for (int i = 0; i < NE; i++) {
      mb_init_sk(btf + i * 8, 1);
      mb_init_sk(bte + i * 8, 128);
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
  if (wid == 0 && elect_one_sync_splitk()) {
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
  constexpr uint32_t idesc =
      (1u << 4) | ((uint32_t)(BN / 8) << 17) | (8u << 24);

  // Stage index + parity from a CONTINUOUS K-block counter — the per-tile
  // `ki % NS` reset desyncs the mbarrier parity at tile boundaries whenever
  // nk_slice % NS != 0 on a multi-tile-iter task (the B36 bug; see
  // fp8_gemm_dense_sm100_common.cuh for the full note). Split-K ALWAYS
  // multi-tile-iters, which is why this variant crashed since 2026-05-15.
  if (wid == 0 && elect_one_sync_splitk()) {
    int gk = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      // Tile decomposition: bidx -> (k_slice, bm, bn).
      // k_slice is the outermost so adjacent CTAs in a wave share (m,n)
      // tiles and accumulate via reduce-add into the same output region.
      int ks = bidx / (mm * nn);
      int rem = bidx % (mm * nn);
      int bm = rem / nn, bn = rem % nn;
      int om = bm * BM, on = bn * BN;
      int ki_base = ks * nk_slice; // global K-tile offset for this slice
      for (int ki = 0; ki < nk_slice; ki++, gk++) {
        int s = gk % NS;
        int p = (gk / NS) & 1;
        mb_wait_sk(be + s * 8, p ^ 1);
        int as_ = sA(s);
        int bs_ = sBl(s);
        int mb = bf + s * 8;
        tma_ld_sk(as_, ta_ptr, (ki_base + ki) * BK, om, mb);
        tma_ld_sk(bs_, tb_ptr, (ki_base + ki) * BK, on, mb);
        mb_arrive_tx_sk(mb, SA + SB);
      }
    }
  } else if (wid == 1 && elect_one_sync_splitk()) {
    int gki = 0;
    for (int iter = 0;; iter++) {
      int bidx = iter * num_workers + worker_idx;
      if (bidx >= total) {
        break;
      }
      for (int ki = 0; ki < nk_slice; ki++, gki++) {
        int s = gki % NS;
        int p = (gki / NS) & 1;
        mb_wait_sk(bf + s * 8, p);
        int ai = gki % NE;
        int ap = (gki / NE) & 1;
        mb_wait_sk(bte + ai * 8, ap ^ 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        int as_ = sA(s);
        int bs_ = sBl(s);
        uint32_t tc = taddr + ai * BN;
        for (int k = 0; k < BK / UK; k++) {
          uint64_t ad = mkdesc_sk(as_ + k * UK), bd = mkdesc_sk(bs_ + k * UK);
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
      int ks = bidx / (mm * nn);
      int rem = bidx % (mm * nn);
      int bm = rem / nn, bn = rem % nn;
      int om = bm * BM, on = bn * BN;
      int mi = om + et;
      int ki_base = ks * nk_slice;

      float acc[BN];
#pragma unroll
      for (int i = 0; i < BN; i++) {
        acc[i] = 0.0f;
      }

      for (int ki = 0; ki < nk_slice; ki++, gki++) {
        int kg = ki_base + ki; // global K-tile index for scale arrays
        float sfa = (mi < M) ? __ldg(sa + mi * nk + kg) : 0.0f;
        float sfb0 = __ldg(sb + (on / 128) * nk + kg);
        float sfb1 = 0.0f;
        if (BN > 128) {
          sfb1 = __ldg(sb + ((on + 128) / 128) * nk + kg);
        }

        int ai = gki % NE;
        int ap = (gki / NE) & 1;
        mb_wait_sk(btf + ai * 8, ap);
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
        mb_arrive_sk(bte + ai * 8);
      }

      if (mi < M) {
        __nv_bfloat16 *row = C + (long long)mi * N + on;
#pragma unroll
        for (int n = 0; n < BN; n += 16) {
          if (on + n + 15 < N) {
            // Pack acc[n..n+15] into 8 bf16x2 values and atomically add
            // them to global memory. SPLIT_K CTAs concurrently red.add
            // into the same address per (m,n) tile — pre-zeroed output
            // makes the final sum correct.
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
            if constexpr (SPLIT_K > 1) {
              red_add_bf16x2(row + n + 0, r0);
              red_add_bf16x2(row + n + 2, r1);
              red_add_bf16x2(row + n + 4, r2);
              red_add_bf16x2(row + n + 6, r3);
              red_add_bf16x2(row + n + 8, r4);
              red_add_bf16x2(row + n + 10, r5);
              red_add_bf16x2(row + n + 12, r6);
              red_add_bf16x2(row + n + 14, r7);
            } else {
              // SPLIT_K==1: equivalent to the non-split kernel — direct
              // store is faster than an atomic add into a pre-zeroed
              // buffer. Kept for testing / parameter-sweep symmetry.
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
            }
          } else {
            // Tail (partial 16-col group at output edge): per-element
            // scalar reduce-add to handle non-aligned bounds.
            for (int j = 0; j < 16 && on + n + j < N; j += 2) {
              if (on + n + j + 1 < N) {
                nv_bfloat162 b =
                    __floats2bfloat162_rn(acc[n + j], acc[n + j + 1]);
                uint32_t r = *reinterpret_cast<uint32_t *>(&b);
                if constexpr (SPLIT_K > 1) {
                  red_add_bf16x2(row + n + j, r);
                } else {
                  *reinterpret_cast<uint32_t *>(row + n + j) = r;
                }
              } else {
                // Odd tail: scalar bf16 write. Reduce-add path here is
                // intentionally unsupported (output dims are 128-aligned
                // in practice; this branch only protects against ragged
                // edges). SPLIT_K>1 + odd tail is a static violation
                // caught by the registration assert.
                row[n + j] = __float2bfloat16(acc[n + j]);
              }
            }
          }
        }
      }
    }
  }

  __syncthreads();
  // Same global-scope membar as the non-splitk kernel: ensure downstream
  // tasks (allreduce, elementwise_add) on other CTAs see the GEMM's
  // atomic accumulations.
  asm volatile("membar.gl;" ::: "memory");
  if (wid == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(TCA));
  }
#endif
}

template <int BN, int NS, int NE, int SPLIT_K>
__device__ __noinline__ void
    fp8_gemm_dense_decode_splitk_sm100_task_impl(CUtensorMap const *ta_ptr,
                                                 CUtensorMap const *tb_ptr,
                                                 float const *__restrict__ sa,
                                                 float const *__restrict__ sb,
                                                 __nv_bfloat16 *__restrict__ C,
                                                 int const M,
                                                 int const N,
                                                 int const K,
                                                 int const worker_idx,
                                                 int const num_workers) {
  task_impl_splitk_tpl<BN, NS, NE, SPLIT_K>(
      ta_ptr, tb_ptr, sa, sb, C, M, N, K, worker_idx, num_workers);
}

template <int BN, int NS, int NE, int SPLIT_K>
__host__ __device__ inline constexpr int
    fp8_gemm_dense_decode_splitk_smem_size() {
  constexpr int BM = 128, BK = 128;
  // smem layout identical to non-splitk kernel: NS pipelined A+B stages
  // (each SA + SB bytes) plus barriers + 1024 alignment slack. SPLIT_K
  // affects only the K iteration count, not the per-stage tile size.
  return NS * (BM * BK + BN * BK) + (NS * 2 + NE * 2) * 8 + 8 + 1024;
}

} // namespace fp8_gemm_dense_decode_splitk
} // namespace kernel
