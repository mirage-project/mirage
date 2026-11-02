// tasks/sum_gpus.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// sum_gpus.cuh: the adding half of an all-reduce (allreduce_send.cuh): task
// (x, y) of grid (Q, T, 1) = token y, H / Q-column part x of the row: add the
// GPUs' partial sums (fp32, GPU order) -> the node's output (bf16 [T][H]).
#pragma once
#include "../core.cuh"

namespace static_mk {

template <size_t SOFF, int H, int Q> // SOFF: the sent sums' offset in the
                                     // exchange region
__device__ __forceinline__ void
    sum_gpus_task(G const &g, __hip_bfloat16 *ssum, int q, int t) {
  constexpr size_t S_RANK = (size_t)T * H * 2;
  constexpr int COLS = H / Q;
  static_assert(COLS * Q == H && COLS % 8 == 0 && COLS / 8 <= 256,
                "sum_gpus: whole 8-column vectors, one per thread");
  int const v = STATIC_TIDX;
  unsigned char const *row =
      g.rv + SOFF + rg_set() + (size_t)t * H * 2 + (size_t)q * COLS * 2;
  if (v < COLS / 8) {
    float s[8];
    allreduce_land_bf16<S_RANK>(g, row, v, s);
    // read by residual_add on any CU of this GPU (device-scope polls): stored
    // past the L2
    st16_dev(ssum + (size_t)t * H + q * COLS + 8 * v,
             pack_bf16x8(make_float4(s[0], s[1], s[2], s[3]),
                         make_float4(s[4], s[5], s[6], s[7])));
  }
  __syncthreads();
}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class... IN>
__device__ __forceinline__ void run_sum_gpus(Maps const &,
                                             G const &g,
                                             KernelLocals &,
                                             StaticTask const &tk) {
  static_assert(SEND::buf >= 0,
                "sum_gpus: rows sent by an allreduce_send node");
  static_assert(SELF::y == T && SELF::buf >= 0,
                "sum_gpus: grid (parts, T tokens), its output Ssum");
  sum_gpus_task<exchange_offset[SEND::buf], SEND::v[0], SELF::x>(
      g, buf_at<__hip_bfloat16>(g, SELF::buf), tk.x, tk.y);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sum_gpus() {
  return 0;
}

} // namespace static_mk
