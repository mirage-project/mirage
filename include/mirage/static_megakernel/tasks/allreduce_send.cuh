// tasks/allreduce_send.cuh -- the sending half of an all-reduce of a [T][W]
// partial sum over the GPUs: task x of grid (NSLICE, 1, 1) sends slice x of
// this GPU's partial sum (after every task of the node that writes it: its
// counter) as bf16, one multicast store per 16 B, into slot `rank` of the
// node's output on every GPU (an exchange buffer, [GPUS][T][W] bf16). The
// adding half is a reader of that buffer (sum_gpus, sum_rmsnorm: core.cuh
// allreduce_land_bf16): bf16 partials, one multicast store each, the sums in a
// fixed order (the same result every launch). The exchange region has two sets,
// launch n uses set n % 2 (rg_set), so no GPU writes into a set another GPU may
// still read.
#pragma once
#include "../core.cuh"

namespace static_mk {

// src = the producer's output, fp32 [T x GROUP][W], complete once its NDONE
// tasks are counted in done_cnt: task x of NSLICE takes slice x of the T x W /
// 8 bf16 vectors (token t: its GROUP rows added in order: moe_experts' rows,
// one per (token, k), GROUP K; the shared down sums, GROUP 1) and multicasts
// each into slot g.rank of the output (OFF: its offset in the exchange region,
// [GPUS][T][W] bf16) of every GPU. RS >= 0: then re-arm slice x of buffer RS
// (an exchange buffer no SM reads any more once the producer is done: z_q after
// moe_experts).
template <int NSLICE, int NDONE, int GROUP, int W, size_t OFF, int RS>
__device__ __forceinline__ void allreduce_send_task(G const &g,
                                                    float const *src,
                                                    uint32_t *done_cnt,
                                                    int x STAGE_STAMP_PARAM) {
  if (threadIdx.x == 0) {
    cnt_wait(done_cnt, NDONE);
  }
  if (threadIdx.x == 0) {
    STAGE_STAMP(true)
  } // timing build: the count was seen complete
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(1)
  }
  constexpr int C8 = W / 8; // 16-B bf16 vectors per token
  int const nb = T * C8, blo = (int)((long long)nb * x / NSLICE),
            bhi = (int)((long long)nb * (x + 1) / NSLICE);
  for (int i = blo + threadIdx.x; i < bhi; i += 256) {
    int const t = i / C8, c = i - t * C8;
    size_t const dst =
        OFF + (size_t)g.rank * ((size_t)T * W * 2) + ((size_t)t * C8 + c) * 16;
    if constexpr (GROUP > 1) {
      float4 p0 = make_float4(0.f, 0.f, 0.f, 0.f), p1 = p0;
      float4 const *rows = reinterpret_cast<float4 const *>(src) +
                           ((size_t)t * GROUP * W + 8 * c) / 4;
#pragma unroll
      for (int k = 0; k < GROUP; k++) {
        float4 const a = rows[(size_t)k * (W / 4)],
                     b = rows[(size_t)k * (W / 4) + 1];
        p0.x += a.x;
        p0.y += a.y;
        p0.z += a.z;
        p0.w += a.w;
        p1.x += b.x;
        p1.y += b.y;
        p1.z += b.z;
        p1.w += b.w;
      }
      push16(g, dst, pack_bf16x8(p0, p1));
    } else {
      float4 const *row =
          reinterpret_cast<float4 const *>(src) + ((size_t)t * W + 8 * c) / 4;
      push16(g, dst, pack_bf16x8(row[0], row[1]));
    }
  }
  if constexpr (REARM && RS >= 0) { // the producer is done: no SM reads buffer
                                    // RS any more; slice x of it
    constexpr int NZ = (int)(exchange_bytes[RS] / 16);
    constexpr size_t ZOFF = exchange_offset[RS];
    for (int i = NZ * x / NSLICE + threadIdx.x; i < NZ * (x + 1) / NSLICE;
         i += 256) {
      rearm16(g.rv + ZOFF + (size_t)i * 16 + rg_set());
    }
  }
  __syncthreads();
}

// allreduce_send task x: slice x of SELF::x of its input (the output of node
// SRC, complete once all SRC's tasks are counted) to every GPU, into the node's
// output (an exchange buffer, [GPUS][T][W] bf16). PARAMS {W (the row width),
// rows per token in the input (GROUP: added in order)}; BUF_SLOTS {an
// exchange buffer to re-arm once SRC is done, or -1}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SRC,
          class... REST>
__device__ __forceinline__ void run_allreduce_send(Maps const &,
                                                   G const &g,
                                                   KernelLocals &L,
                                                   StaticTask const &tk) {
  constexpr int W = PARAMS::v[0], GROUP = PARAMS::v[1];
  static_assert(W % 8 == 0 && GROUP >= 1,
                "allreduce_send: params {row width, rows per token}");
  static_assert(SRC::buf >= 0 && SRC::counter >= 0,
                "allreduce_send: its input is a node's output buffer, the node "
                "counts its tasks");
  static_assert(
      SELF::buf >= 0 && exchange_bytes[SELF::buf] == (size_t)GPUS * T * W * 2,
      "allreduce_send: its output, [GPUS][T][W] bf16 in the exchange region");
  allreduce_send_task<SELF::x,
                      SRC::x * SRC::y * SRC::z,
                      GROUP,
                      W,
                      exchange_offset[SELF::buf],
                      slot_at<BUF_SLOTS, 0>()>(
      g,
      buf_at<float const>(g, SRC::buf),
      g.cnt + SRC::counter,
      tk.x STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_allreduce_send() {
  return 0;
}

} // namespace static_mk
