// External-reduce companion for fp8_gemm_dense_qkva_splitk_sm100.cuh
// (race-free split-K, 2026-06-02).
//
// The split-K GEMM (EXT_REDUCE path) writes SPLIT_K exclusive FP32 partials per
// output (m_tile, n_tile) into C_partial and does NOT reduce in-kernel. This
// task — gated by MPK's event system to run only AFTER every split-K GEMM CTA
// of a tile has completed (correct ordering via the task DAG, not atomics) —
// sums the SPLIT_K partials in FP32 and casts once to BF16 into the final C.
//
// This replaces the in-kernel last-arriver (gen-tagged atomicAdd +
// fence.sc.gpu) that raced at ~1024 decode iters. With the reduce factored into
// its own event-gated task there are NO inter-CTA atomics and NO data race
// possible.
//
// C_partial layout (written by task_impl_splitk_tpl, COLUMN-MAJOR within tile):
//   tile_sz  = BM * BN              (128 * BN floats)
//   slice_sz = mm * nn * tile_sz    (one full output grid per K-slice)
//   tile_id  = bm * nn + bn
//   element (row et in [0,BM), col n in [0,BN)) of tile (bm,bn) in slice ks:
//     C_partial[ks*slice_sz + tile_id*tile_sz + n*BM + et]
// Output C is row-major [M, N]; element (mi = bm*BM + et, nj = bn*BN + n).
//
// Parallelization: persistent worker loop over output tiles. Each CTA owns one
// (bm,bn) tile per stride; 128 threads, thread et handles output row
// mi = bm*BM + et and loops over the BN columns of the tile. For each column it
// reads SPLIT_K FP32 partials (one per K-slice, strided by slice_sz) — these
// are coalesced across the 128 threads (column-major: consecutive et are
// consecutive addresses) — sums in FP32, and stores BF16. Vectorized 16B
// (st.v4.b32 = 8 bf16) stores when the 16-col group is fully in-bounds.

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {
namespace fp8_gemm_dense_splitk_reduce {

// NCOL: split each (m_tile, n_tile)'s BN columns into NCOL contiguous column
// sub-blocks of width CW=BN/NCOL, so the work-unit count (and thus the active
// CTA count) is mm*nn*NCOL instead of mm*nn. At decode M=1 the un-split reduce
// has only mm*nn (e.g. 17) active CTAs on a 148-SM B200 — badly under-occupied
// AND each CTA's per-element split-K reads stride by slice_sz (~1MB apart, poor
// locality). NCOL>1 fans the columns across more SMs (NCOL=4 -> 68 CTAs at
// qkv_a) and shrinks each CTA's strided-read working set, cutting the reduce
// wall. CW must be a multiple of 16 (the bf16 vector-store group). BN%NCOL==0
// and (BN/NCOL)%16==0 are asserted at registration.
template <int BN, int SPLIT_K, int NCOL = 1>
__device__ __noinline__ void fp8_gemm_dense_splitk_reduce_sm100_task_impl(
    float const *__restrict__ C_partial,
    __nv_bfloat16 *__restrict__ C,
    int const M,
    int const N,
    int const worker_idx,
    int const num_workers) {
  constexpr int BM = 128;
  constexpr int CW = BN / NCOL; // columns per work unit
  static_assert(BN % NCOL == 0, "BN must be divisible by NCOL");
  static_assert(CW % 16 == 0, "BN/NCOL must be a multiple of 16 (bf16 vec)");
  int const tid = threadIdx.x;
  if (tid >= BM) {
    return;
  }
  int const et = tid; // output row within the tile
  int const mm = (M + BM - 1) / BM;
  int const nn = (N + BN - 1) / BN;
  int const total = mm * nn * NCOL;
  size_t const tile_sz = (size_t)BM * BN;
  size_t const slice_sz = (size_t)mm * nn * tile_sz;

  for (int wu = worker_idx; wu < total; wu += num_workers) {
    int const tile_id = wu / NCOL;
    int const cb = wu % NCOL; // column sub-block within the tile
    int const bm = tile_id / nn;
    int const bn = tile_id % nn;
    int const mi = bm * BM + et;
    int const col0 = cb * CW; // first column (within tile) this WU owns
    int const on = bn * BN;
    if (mi >= M) {
      continue;
    }
    // Base of this (tile, row) in slice 0; element n is at +n*BM, slice ks adds
    // ks*slice_sz. col0 offsets to this work unit's column sub-block.
    float const *pbase =
        C_partial + (size_t)tile_id * tile_sz + (size_t)col0 * BM + (size_t)et;
    __nv_bfloat16 *row = C + (long long)mi * N + on + col0;

#pragma unroll
    for (int n = 0; n < CW; n += 16) {
      float a[16];
#pragma unroll
      for (int j = 0; j < 16; j++) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < SPLIT_K; s++) {
          sum += pbase[(size_t)s * slice_sz + (size_t)(n + j) * BM];
        }
        a[j] = sum;
      }
      if (on + col0 + n + 15 < N) {
        nv_bfloat162 b0 = __floats2bfloat162_rn(a[0], a[1]);
        nv_bfloat162 b1 = __floats2bfloat162_rn(a[2], a[3]);
        nv_bfloat162 b2 = __floats2bfloat162_rn(a[4], a[5]);
        nv_bfloat162 b3 = __floats2bfloat162_rn(a[6], a[7]);
        nv_bfloat162 b4 = __floats2bfloat162_rn(a[8], a[9]);
        nv_bfloat162 b5 = __floats2bfloat162_rn(a[10], a[11]);
        nv_bfloat162 b6 = __floats2bfloat162_rn(a[12], a[13]);
        nv_bfloat162 b7 = __floats2bfloat162_rn(a[14], a[15]);
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
        for (int j = 0; j < 16 && on + col0 + n + j < N; j++) {
          row[n + j] = __float2bfloat16(a[j]);
        }
      }
    }
  }
}

} // namespace fp8_gemm_dense_splitk_reduce
} // namespace kernel
