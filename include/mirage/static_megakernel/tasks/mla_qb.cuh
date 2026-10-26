// tasks/mla_qb.cuh -- Kimi K3 MLA: the q_b projection as warp-MMA tiles with
// the weight resident (the pattern of gemm_tile's warp-MMA tile, with a bf16
// store epilogue instead of the push). Task x of grid (N / 64, 1, 1) = weight
// rows [64 x, 64 x + 64):
//   q[t][64 x + r] = bf16(sum_k w[64 x + r][k] q_a[t][k])   -> the node's
//   output [T][N] bf16 (0xFF before each launch: mla_absorb polls it per 16 B)
// The tile's weight (64 x K bf16; K = 1536: 192 KB) is copied into shared
// memory as the task starts (it needs no input), so the copy runs while the
// fused projection and mla_norm run; each of the 8 warps then polls its K slice
// of q_a (mla_norm's 0xFF-prefilled output [T][K], no counter) into shared
// memory and multiplies it (mma.sync m16n8k16: M = 16 weight rows, N = the 8
// tokens); the warps' sums are added through shared memory (the weight's
// region, after every warp's MMAs). Weight rows at a K * 2 B stride, activation
// rows too; 16-B chunk c of row r at (c & ~7) | ((c ^ r) & 7).
#pragma once
#include "../core.cuh"
#include "gemm_tile.cuh" // ldsm_x4, ldsm_x2, mma_16816, valid16h
#include "mla_common.cuh"

namespace static_mk {

template <int K, int N>
__device__ __forceinline__ void mla_qb_task(KernelLocals &L,
                                            __nv_bfloat16 const *w,
                                            __nv_bfloat16 const *act,
                                            __nv_bfloat16 *out,
                                            int x) {
  static_assert(T == 8 && K % (8 * 16) == 0 && N % 64 == 0,
                "mla_qb: 8 tokens, 8 warps x 16-multiples of K, 64-row tiles");
  constexpr int ROWS = 64, RB = K * 2, KW = K / 8; // K per warp
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  char *const sm = L.sm;
  uint32_t const wb = su32(sm);
  char *const as = sm + ROWS * RB;
  uint32_t const ab = su32(as);
  auto sw = [](int r, int c) { return (c & ~7) | ((c ^ r) & 7); };
  __syncthreads(); // the SM's previous task is done with the shared memory
  {
    __nv_bfloat16 const *src = w + (size_t)x * ROWS * K;
    for (int i = tid; i < ROWS * (RB / 16); i += 256) {
      int const r = i / (RB / 16), c = i % (RB / 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       wb + r * RB + (sw(r, c) << 4)),
                   "l"(src + (size_t)r * K + c * 8)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  // this warp's K slice [KW warp, + KW) of the 8 tokens' activation: T x KW / 8
  // vectors, PER per lane, polled together (one round trip per round)
  {
    constexpr int PER = (T * (KW / 8) + 31) / 32;
    uint4 a[PER];
    bool got[PER] = {};
    bool all;
    do {
      all = true;
#pragma unroll
      for (int i = 0; i < PER; i++) {
        int const v = lane + 32 * i;
        if (v < T * (KW / 8) && !got[i]) {
          a[i] = ld16_poll(reinterpret_cast<unsigned char const *>(
              act + (size_t)(v / (KW / 8)) * K +
              (warp * (KW / 8) + v % (KW / 8)) * 8));
        }
      }
#pragma unroll
      for (int i = 0; i < PER; i++) {
        int const v = lane + 32 * i;
        if (v < T * (KW / 8) && !got[i]) {
          got[i] = valid16h(a[i]);
          all = all && got[i];
        }
      }
    } while (!all);
#pragma unroll
    for (int i = 0; i < PER; i++) {
      int const v = lane + 32 * i;
      if (v < T * (KW / 8)) {
        int const t = v / (KW / 8), c = warp * (KW / 8) + v % (KW / 8);
        *reinterpret_cast<uint4 *>(as + t * RB + (sw(t, c) << 4)) = a[i];
      }
    }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads(); // the weight landed (every thread's copies), every warp's
                   // activation slice written
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && L.stage_stamps) {
    L.stage_stamps[0] = gtime();
  }
#endif
  float acc[4][4] = {};
#pragma unroll 4
  for (int q = 0; q < KW / 16; q++) {
    int const ks = warp * (KW / 16) + q;
    uint32_t b0, b1;
    {
      int const t = lane & 7, c = 2 * ks + ((lane >> 3) & 1);
      ldsm_x2(ab + t * RB + (sw(t, c) << 4), b0, b1);
    }
#pragma unroll
    for (int mb = 0; mb < 4; mb++) {
      int const r = mb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                c = 2 * ks + (lane >> 4);
      uint32_t a[4];
      ldsm_x4(wb + r * RB + (sw(r, c) << 4), a);
      mma_16816(acc[mb], a, b0, b1);
    }
  }
  __syncthreads(); // every warp done with the weight: its region holds the
                   // warps' sums [warp][T][ROWS] fp32
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && L.stage_stamps) {
    L.stage_stamps[1] = gtime();
  }
#endif
  float *red = reinterpret_cast<float *>(sm);
  {
    int const g = lane >> 2, n = 2 * (lane & 3);
#pragma unroll
    for (int mb = 0; mb < 4; mb++) {
      float *p = red + warp * T * ROWS + mb * 16 + g;
      p[n * ROWS] = acc[mb][0];
      p[(n + 1) * ROWS] = acc[mb][1];
      p[n * ROWS + 8] = acc[mb][2];
      p[(n + 1) * ROWS + 8] = acc[mb][3];
    }
  }
  __syncthreads();
  if (tid < T * ROWS / 8) { // token t, rows [8 c, 8 c + 8): the 8 warps' sums
                            // in warp order, bf16, one 16-B store
    int const t = tid / (ROWS / 8), c = tid % (ROWS / 8);
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float v = 0.f;
#pragma unroll
      for (int wi = 0; wi < 8; wi++) {
        v += red[wi * T * ROWS + t * ROWS + 8 * c + j];
      }
      o[j] = v;
    }
    *reinterpret_cast<uint4 *>(out + (size_t)t * N + x * ROWS + 8 * c) =
        pack_bf16x8(make_float4(o[0], o[1], o[2], o[3]),
                    make_float4(o[4], o[5], o[6], o[7]));
  }
  __syncthreads();
  L.ring_relaid = true;
}

// task x: A = mla_norm (its output q_a [T][K] bf16, 0xFF before each launch);
// params = [K, N]; buffer slots = the weight [N][K] bf16
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class A,
          class... IN>
__device__ __forceinline__ void run_mla_qb(Maps const &,
                                           G const &g,
                                           KernelLocals &L,
                                           StaticTask const &tk) {
  constexpr int K = PARAMS::v[0], N = PARAMS::v[1];
  static_assert(A::buf >= 0 && SELF::buf >= 0 && SELF::x * 64 == N,
                "mla_qb: mla_norm's q_a, 64-row tiles, its output");
  mla_qb_task<K, N>(L,
                    buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                    buf_at<__nv_bfloat16 const>(g, A::buf),
                    buf_at<__nv_bfloat16>(g, SELF::buf),
                    tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_qb() {
  return 64 * PARAMS::v[0] * 2 + T * PARAMS::v[0] * 2;
}

} // namespace static_mk
