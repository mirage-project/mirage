// scale_promotion.cuh -- Apply fp32 scales to the FP8 MMA partial sums.
// scaled_moe_experts and scaled_gemm_tile use these functions. Terms:
//   piece    a packed 128 x 128 weight tile (e4m3). BlockScales uses one
//            fp32 scale. GateUpScales uses separate gate and up scales.
//   K tile   128 columns of the activation z_q (e4m3). Each token has one fp32
//            scale for each K tile (group_quant).
//   job      the pieces of one task (NATOMS pieces; atom: a piece index)
// This MMA instruction accepts UE8M0 scales, which represent powers of two.
// The checkpoint scales can have other values. Keep those values as follows:
//   1. The MMA uses scale 1 for both operands (a constant scale chunk).
//   2. Each piece of a job has its own TMEM accumulator.
//   3. The epilogue multiplies the partial sum of each piece by the weight
//      scale for each row and the activation scale for each token.
//      Then it adds the scaled partial sums in registers.
// A round is two ring stages (one for each issuer, two pieces each). A round
// uses one accumulator tick (acc_full / acc_empty, as a whole-K job does). A
// job uses PromoRounds<NATOMS>::ROUNDS ticks. The columns of tick a (0..3):
// piece 0 of issuer i at 8 a + 32 i (the columns of runtime.cuh issue_job),
// piece 1 at PROMO_COL1 + 8 a + 32 i.
#pragma once
#include "core.cuh"

namespace static_mk {

constexpr int PROMO_COL1 = 448; // TMEM columns 448..511: piece 1 of each
                                // (tick, issuer)

// the ring stages and the accumulator ticks of a job of NATOMS pieces (a W13
// item of scaled_moe_experts, a scaled_gemm_tile tile)
template <int NATOMS>
struct PromoRounds {
  static constexpr int NST = (NATOMS + 1) / 2, ROUNDS = (NST + 1) / 2;
};

// Read a value from another SM with a GPU-scope load that bypasses L1.
__device__ __forceinline__ float ld_relaxed_f32(float const *p) {
  float v;
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];"
               : "=f"(v)
               : "l"(p)
               : "memory");
  return v;
}

// Fill the scale chunk at shared-memory address dst with UE8M0 1.0.
// Each byte is 127 (exponent zero). Each of the 128 threads writes one word.
// The fence orders each thread's stores before the later tcgen05.cp.
__device__ __forceinline__ void const_sf_fill(uint32_t dst, int i) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(dst + 4 * i), "r"(0x7F7F7F7Fu)
               : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// issuer iss of a job (a W13 item, a scaled_gemm_tile tile). The whole warp
// calls the function; one elected lane issues. issue_promoted is
// runtime.cuh issue_job_warp with one accumulator for each piece:
//   - Round r is stage 2 r (issuer 0) and stage 2 r + 1 (issuer 1). Round r
//     uses accumulator tick wtt + r.
//   - Piece `at` of a stage goes into its own columns. Its first MMA does not
//     accumulate.
//   - Both issuers wait on the `full` of each stage in order (parity rule).
//   - Each issuer arrives once on the acc_full of each round: a commit, or a
//     plain arrive when the round has no stage of this issuer.
// sfc: the constant scale chunk. It gives scale 1 to both operands.
template <uint32_t IDESC, int NATOMS>
__device__ __forceinline__ void issue_promoted(uint32_t base,
                                               uint32_t tb,
                                               int iss,
                                               int &g,
                                               int wtt,
                                               uint32_t sfc STAGE_STAMP_PARAM) {
  constexpr int NST = PromoRounds<NATOMS>::NST;
  for (int s = 0; s < NST; s++, g++) {
    int const tick = wtt + (s >> 1), a = tick & 3;
    if ((s & 1) == 0 && tick >= 4) { // the epilogue drained the columns
      mbar_wait(rt.acc_empty0 + 8 * a, ((tick >> 2) - 1) & 1);
    }
    int const st = g % SMAX;
    mbar_wait(rt.full0 + 8 * st, (g / SMAX) & 1);
    fence_after();
    STAGE_STAMP(s == 0)
    bool const mine = (s & 1) == iss;
    bool const one = issuer_lane<true>();
    if (!mine) {
      if (one) {
        mbar_arrive(rt.empty0 + 8 * st);
      }
    } else {
      int const na = (NATOMS - 2 * s < 2) ? NATOMS - 2 * s : 2;
      for (int at = 0; at < na; at++) {
        uint32_t const acc =
            tb + (at == 0 ? 0u : (uint32_t)PROMO_COL1) + 8 * a + 32 * iss;
        uint64_t const dw = mkdesc(base + st * FSTAGE + at * 16384);
        uint64_t const db = mkdesc(base + st * FSTAGE + W_STAGE + at * 1024);
        if (one) {
          mma_mx(acc, dw, db, idesc_mx(IDESC, 0), 0u, sfc, sfc);
          mma_mx(acc, dw + 2, db + 2, idesc_mx(IDESC, 1), 1u, sfc, sfc);
          mma_mx(acc, dw + 4, db + 4, idesc_mx(IDESC, 2), 1u, sfc, sfc);
          mma_mx(acc, dw + 6, db + 6, idesc_mx(IDESC, 3), 1u, sfc, sfc);
        }
      }
      if (one) {
        tc_commit(rt.empty0 + 8 * st);
      }
    }
    if ((s & 1) == 1 || s == NST - 1) { // the last stage of the round
      bool const had = (2 * (s >> 1) + iss) < NST; // the round has my stage
      if (one) {
        if (had) {
          tc_commit(rt.acc_full0 + 8 * a);
        } else {
          mbar_arrive(rt.acc_full0 + 8 * a);
        }
      }
    }
  }
}

// the layout of the weight scales (a template argument of drain_promoted and of
// the tasks): where the scales of piece row m (128 weight rows) and K tile k0
// start in the scale tensor, how many scales each piece has, and which one row
// `row` of a piece uses. The checkpoint's scales, in their own layout:
//   BlockScales   one scale for each 128 x 128 block, [N / 128][KT] (a plain
//                 weight: shared down, W2; pack_w2)
//   GateUpScales  piece row m = the 64 gate rows, then the 64 up rows of
//                 features [64 m, 64 m + 64); block row m / 2 has the gate and
//                 the up scale, [N / 256][KT][2] (W13, shared gate_up;
//                 pack_w13)
struct BlockScales {
  static constexpr int PER_PIECE = 1;
  static __device__ __forceinline__ size_t first(int m, int KT, int k0) {
    return (size_t)m * KT + k0;
  }
  static __device__ __forceinline__ float
      of(float const *bs, int atom, int row) {
    return bs[atom];
  }
};
struct GateUpScales {
  static constexpr int PER_PIECE = 2;
  static __device__ __forceinline__ size_t first(int m, int KT, int k0) {
    return ((size_t)(m >> 1) * KT + k0) * 2;
  }
  static __device__ __forceinline__ float
      of(float const *bs, int atom, int row) {
    return bs[2 * atom + (row >= 64 ? 1 : 0)];
  }
};

// the 128 epilogue threads: drain the PromoRounds ticks of a job, from tick
// wtt. Each thread has one row. v[t] (t < T = 8) = the sum over the pieces of
// sw(piece, row) sx(t, piece) partial(piece). The function returns the row.
//   bs  the weight scales of the job in shared memory, [NATOMS][WS::PER_PIECE]
//       (WS: their layout, BlockScales or GateUpScales)
//   xs  the activation scales in shared memory, [NATOMS][T]
template <int NATOMS, class WS>
__device__ __forceinline__ int drain_promoted(
    uint32_t tb, int wtt, float const *bs, float const *xs, float *v) {
  constexpr int NST = PromoRounds<NATOMS>::NST,
                ROUNDS = PromoRounds<NATOMS>::ROUNDS;
  int const lg = (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31,
            row = lg * 32 + lane;
  uint32_t const lanes = (uint32_t)(lg * 32) << 16;
#pragma unroll
  for (int t = 0; t < 8; t++) {
    v[t] = 0.0f;
  }
  for (int r = 0; r < ROUNDS; r++) {
    int const tick = wtt + r, a = tick & 3;
    mbar_wait(rt.acc_full0 + 8 * a, (tick >> 2) & 1);
    fence_after();
#pragma unroll
    for (int iss = 0; iss < 2; iss++) {
      int const s = 2 * r + iss;
#pragma unroll
      for (int at = 0; at < 2; at++) {
        int const atom = 2 * s + at;
        if (s < NST && atom < NATOMS) {
          float p[8];
          tmem_ld8(tb + (at == 0 ? 0u : (uint32_t)PROMO_COL1) + 8 * a +
                       32 * iss + lanes,
                   p);
          float const sw = WS::of(bs, atom, row);
#pragma unroll
          for (int t = 0; t < 8; t++) {
            v[t] = fmaf(sw * xs[atom * T + t], p[t], v[t]);
          }
        }
      }
    }
    fence_before();
    mbar_arrive(rt.acc_empty0 + 8 * a); // 128 arrivals
  }
  return row;
}

} // namespace static_mk
