// tasks/mla_absorb.cuh -- Kimi K3 MLA: the absorbed query of one head (SGLang:
// q_nope . w_kc by torch.bmm, then concat_mla_absorb_q). Task h of grid (heads,
// 1, 1) = head h of this GPU:
//   q      = the q_b projection's output (mla_qb: [T][heads (NOPE + ROPE)]
//   bf16, 0xFF before each launch, polled per 16 B),
//            head h's columns
//   q_abs  = bf16(q[:, :NOPE] . w_kc[h])    w_kc[h] [NOPE][KV], given
//   transposed: wkcT [heads][KV][NOPE] bf16 out[t][h] = [q_abs (KV) | q[:,
//   NOPE:] (ROPE)] bf16 -> the node's output [T][heads][KV + ROPE] (0xFF before
//   each launch:
//            mla_attention polls it per 16 B)
// The head's weight (KV x NOPE bf16, 128 KB) is copied into shared memory as
// the task starts (it needs no input); the MMAs (mma.sync m16n8k16: M = 16 of
// the KV output rows, N = the 8 tokens, K = NOPE) start once q landed. The
// weight rows sit at a 256-B stride with their 16-B chunks swizzled by the row
// (chunk c at c ^ (row & 7)): ldmatrix without bank conflicts. Uses the whole
// dynamic shared memory: a GEMM after it on the SM starts from a fresh ring
// (L.ring_relaid).
#pragma once
#include "../core.cuh"
#include "gemm_tile.cuh" // ldsm_x4, ldsm_x2, mma_16816, valid16h
#include "mla_common.cuh"

namespace static_mk {

template <int HEADS, int NOPE, int ROPE, int KV>
__device__ __forceinline__ void mla_absorb_task(KernelLocals &L,
                                                __nv_bfloat16 const *qin,
                                                __nv_bfloat16 const *wkcT,
                                                __nv_bfloat16 *out,
                                                int h) {
  static_assert(NOPE == 128 && KV == 512 && ROPE % 8 == 0,
                "mla_absorb: the K3 shapes (NOPE 128, KV 512)");
  constexpr int QN = HEADS * (NOPE + ROPE); // q_b's row
  constexpr int WROW = NOPE * 2;            // 256 B per weight row
  static_assert(T == 8, "8 tokens: the MMAs' N, one softmax warp per token");
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  char *const sm = L.sm;
  uint32_t const wbase = su32(sm); // [KV][NOPE] bf16, swizzled
  __nv_bfloat16 *const qn = reinterpret_cast<__nv_bfloat16 *>(
      sm + KV * WROW); // [T][NOPE] bf16, swizzled (256-B rows)
  __nv_bfloat16 *const st = qn + T * NOPE; // [T][KV] bf16 the output, staged
  __syncthreads(); // the SM's previous task is done with the shared memory
  { // the head's weight: KV rows x NOPE, 16-B chunks by cp.async (no registers
    // held)
    __nv_bfloat16 const *src = wkcT + (size_t)h * KV * NOPE;
    for (int i = tid; i < KV * (WROW / 16); i += 256) {
      int const r = i / (WROW / 16), c = i % (WROW / 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       wbase + r * WROW + ((c ^ (r & 7)) << 4)),
                   "l"(src + (size_t)r * NOPE + c * 8)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  // q of head h: T x (NOPE + ROPE) bf16, 16-B vectors polled
  constexpr int VPT = (NOPE + ROPE) / 8; // vectors per token
  for (int v = tid; v < T * VPT; v += 256) {
    int const t = v / VPT, c = (v % VPT) * 8;
    unsigned char const *p = reinterpret_cast<unsigned char const *>(
        qin + (size_t)t * QN + h * (NOPE + ROPE) + c);
    uint4 x;
    do {
      x = ld16_poll(p);
    } while (!valid16h(x));
    if (c < NOPE) { // swizzled: 16-B chunk c / 8 of row t at (c / 8) ^ t
      *reinterpret_cast<uint4 *>(qn + t * NOPE + (((c >> 3) ^ (t & 7)) << 3)) =
          x;
    } else { // q_pe straight into the output row
      *reinterpret_cast<uint4 *>(out + ((size_t)t * HEADS + h) * (KV + ROPE) +
                                 KV + (c - NOPE)) = x;
    }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && L.stage_stamps) {
    L.stage_stamps[0] = gtime();
  }
#endif
  // q_abs^T [KV][T] = wkcT [KV][NOPE] . q_nope^T: warp w rows [64 w, 64 w + 64)
  // (4 blocks of 16), NOPE / 16 k-steps
  float acc[4][4] = {};
  uint32_t const qbase = su32(qn);
#pragma unroll
  for (int ks = 0; ks < NOPE / 16; ks++) {
    uint32_t b0, b1;
    {
      int const t = lane & 7, ch = 2 * ks + ((lane >> 3) & 1);
      ldsm_x2(qbase + t * WROW + ((ch ^ t) << 4), b0, b1);
    }
#pragma unroll
    for (int mb = 0; mb < 4; mb++) {
      int const r = warp * 64 + mb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                ch = 2 * ks + (lane >> 4);
      uint32_t a[4];
      ldsm_x4(wbase + r * WROW + ((ch ^ (r & 7)) << 4), a);
      mma_16816(acc[mb], a, b0, b1);
    }
  }
  { // acc[mb]: rows warp * 64 + mb * 16 + g (+ 8), tokens n, n + 1 -> st
    // [T][KV] bf16
    int const g = lane >> 2, n = 2 * (lane & 3);
#pragma unroll
    for (int mb = 0; mb < 4; mb++) {
      int const r = warp * 64 + mb * 16 + g;
      st[n * KV + r] = __float2bfloat16(acc[mb][0]);
      st[(n + 1) * KV + r] = __float2bfloat16(acc[mb][1]);
      st[n * KV + r + 8] = __float2bfloat16(acc[mb][2]);
      st[(n + 1) * KV + r + 8] = __float2bfloat16(acc[mb][3]);
    }
  }
  __syncthreads();
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && L.stage_stamps) {
    L.stage_stamps[1] = gtime();
  }
#endif
  for (int v = tid; v < T * KV / 8;
       v += 256) { // whole 16-B vectors: the reader polls per 16 B
    int const t = v / (KV / 8), c = (v % (KV / 8)) * 8;
    *reinterpret_cast<uint4 *>(out + ((size_t)t * HEADS + h) * (KV + ROPE) +
                               c) =
        *reinterpret_cast<uint4 const *>(st + t * KV + c);
  }
  __syncthreads();
  L.ring_relaid = true; // the ring's region was used: a GEMM after this task on
                        // the SM starts from a fresh ring
}

// task h: QB = mla_qb (its output [T][heads (NOPE + ROPE)] bf16); params =
// [heads, NOPE, ROPE, KV]; buffer slots = wkcT [heads][KV][NOPE] bf16
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class QB,
          class... IN>
__device__ __forceinline__ void run_mla_absorb(Maps const &,
                                               G const &g,
                                               KernelLocals &L,
                                               StaticTask const &tk) {
  constexpr int HEADS = PARAMS::v[0], NOPE = PARAMS::v[1], ROPE = PARAMS::v[2],
                KV = PARAMS::v[3];
  static_assert(QB::buf >= 0 && QB::v[1] == HEADS * (NOPE + ROPE),
                "mla_absorb: mla_qb's output, N = heads x (NOPE + ROPE)");
  static_assert(SELF::x == HEADS && SELF::buf >= 0,
                "mla_absorb: one task per head, its output");
  mla_absorb_task<HEADS, NOPE, ROPE, KV>(
      L,
      buf_at<__nv_bfloat16 const>(g, QB::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_absorb() {
  return PARAMS::v[3] * PARAMS::v[1] * 2 + T * PARAMS::v[1] * 2 +
         T * PARAMS::v[3] * 2;
}

} // namespace static_mk
