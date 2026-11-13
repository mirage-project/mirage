// tasks/sm90_frag.cuh -- register-fragment helpers for the Hopper static task
// types that run mma.sync (decode: the tokens are the MMA's N = 8 columns).
#pragma once
#include <cstdint>

namespace static_mk {

// four 8 x 8 b16 matrices from shared memory: lanes 8 j .. 8 j + 7 give the
// row addresses (16 B each) of matrix j; register j holds matrix j, lane l its
// row l / 4, elements 2 (l % 4), + 1 (mma.sync's A / B fragment layout)
__device__ __forceinline__ void sm90_ldsm_x4(uint32_t (&r)[4],
                                             uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
               "[%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr));
}

// The B fragments of m16n8k16 for one K tile of 128 values, 8 tokens (N), from
// a TMA box [T tokens][128 K] f16 with the 128-B swizzle (two halves of 64 K,
// T 128 B each; in a half, token n's 16-B chunk c at n 128 + ((c ^ n) << 4);
// box: the half's token 8 g for token group g, 1024 g B in):
// bf[kb][0] = K 16 kb + 2 (l % 4) (+ 1), bf[kb][1] the same + 8, of token l / 4
// -- what a lane would read with 4-byte loads, in 4 ldmatrix.
__device__ __forceinline__ void sm90_b_frags_k128_sw128(char const *box,
                                                        uint32_t (&bf)[8][2]) {
  int const lane = threadIdx.x & 31, mi = lane >> 3, n = lane & 7;
  uint32_t const base = (uint32_t)__cvta_generic_to_shared(box);
#pragma unroll
  for (int kb = 0; kb < 8; kb += 2) {
    int const k = kb + (mi >> 1);          // this lane's matrix: k16 block k,
    int const ch = 2 * (k & 3) + (mi & 1); // its chunk (low / high 8 K)
    uint32_t r[4];
    sm90_ldsm_x4(r, base + (k >> 2) * T * 128 + n * 128 + ((ch ^ n) << 4));
    bf[kb][0] = r[0], bf[kb][1] = r[1], bf[kb + 1][0] = r[2],
    bf[kb + 1][1] = r[3];
  }
}

// The same from any 8 of the box's T tokens: tok, this lane's token for row
// lane % 8 of the group (box: the box itself).
__device__ __forceinline__ void sm90_b_frags_k128_sw128_rows(
    char const *box, int tok, uint32_t (&bf)[8][2]) {
  int const lane = threadIdx.x & 31, mi = lane >> 3, n = tok & 7;
  uint32_t const base = (uint32_t)__cvta_generic_to_shared(box) + tok * 128;
#pragma unroll
  for (int kb = 0; kb < 8; kb += 2) {
    int const k = kb + (mi >> 1);
    int const ch = 2 * (k & 3) + (mi & 1);
    uint32_t r[4];
    sm90_ldsm_x4(r, base + (k >> 2) * T * 128 + ((ch ^ n) << 4));
    bf[kb][0] = r[0], bf[kb][1] = r[1], bf[kb + 1][0] = r[2],
    bf[kb + 1][1] = r[3];
  }
}

} // namespace static_mk
