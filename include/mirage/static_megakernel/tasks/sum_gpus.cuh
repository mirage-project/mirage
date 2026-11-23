// tasks/sum_gpus.cuh -- the adding half of an all-reduce (allreduce_send.cuh):
// task (x, y) of grid (Q, T, 1) = token y, H / Q-column part x of the row: add
// the GPUs' partial sums (fp32, GPU order) -> the node's output (bf16 [T][H]).
// H = the row's width (the allreduce_send node's params[0]).
#pragma once
#include "../core.cuh"

namespace static_mk {

// sum_gpus task (q, t) of Q per token: part q of S for token t, thread v < COLS
// / 8 columns 8 v .. 8 v + 7: add S over the GPUs -> ssum (bf16 [T][H], the
// node's buffer)
template <size_t SOFF, int H, int Q> // SOFF: the sent sums' offset in the
                                     // exchange region
__device__ __forceinline__ void
    sum_gpus_task(G const &g, __nv_bfloat16 *ssum, int q, int t) {
  constexpr size_t S_RANK =
      (size_t)T * H * 2;      // one GPU's partial sums in the sent buffer
  constexpr int COLS = H / Q; // a task's columns
  static_assert(COLS * Q == H && COLS % 8 == 0 && COLS / 8 <= 256,
                "sum_gpus: whole 8-column vectors, one per thread");
  int const v = threadIdx.x;
#ifdef STATIC_AR_TWOSHOT
  // (STATIC_AR_TWOSHOT, core.cuh) the owner of token t (its task (q, t)) adds
  // the landing slots in GPU order and multicasts bf16(sum) into the gathered
  // rows; every GPU's task (q, t) polls its vector there (one slot): the same
  // bf16 value as the one-shot's. Task (q, t) waits only for the owner's task
  // (q, t). The steps of ar_twoshot_get, the vector stored before it is
  // re-armed
  if (v < COLS / 8) {
    constexpr size_t AR = ar_rank<H>(), GOFF = ar_g_off<H>();
    int const c = q * (COLS / 8) + v; // the vector in the row
    unsigned char const *const sl = g.rv + SOFF + rg_set();
    size_t const gv = GOFF + ((size_t)t * (H / 8) + c) * 16;
    if (t / AR_TOWN == g.rank) {
      int const lv = (t % AR_TOWN) * (H / 8) + c;
      float s[8];
      allreduce_land_bf16<AR, true>(g, sl, lv, s);
      push16(g,
             SOFF + gv,
             pack_bf16x8(make_float4(s[0], s[1], s[2], s[3]),
                         make_float4(s[4], s[5], s[6], s[7])));
      if constexpr (REARM) {
        allreduce_rearm<AR>(g, sl, lv);
      }
    }
    uint4 u = ld16_poll(sl + gv);
    while (!valid16(u)) {
      u = ld16_poll(sl + gv);
    }
    *reinterpret_cast<uint4 *>(ssum + (size_t)t * H + q * COLS + 8 * v) = u;
    if constexpr (REARM) { // the one reader of this gathered vector
      rearm16(sl + gv);
    }
  }
  __syncthreads();
  return;
#endif
  unsigned char const *row =
      g.rv + SOFF + rg_set() + (size_t)t * H * 2 + (size_t)q * COLS * 2;
  if (v < COLS / 8) {
    float s[8];
    allreduce_land_bf16<S_RANK>(g, row, v, s);
    *reinterpret_cast<uint4 *>(ssum + (size_t)t * H + q * COLS + 8 * v) =
        pack_bf16x8(make_float4(s[0], s[1], s[2], s[3]),
                    make_float4(s[4], s[5], s[6], s[7]));
    if constexpr (REARM) {
      allreduce_rearm<S_RANK>(g, row, v);
    }
  }
  __syncthreads();
}
// sum_gpus task (part x, token y): SEND = the allreduce_send node whose output
// it adds (rows of H = its params[0])
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
      g, buf_at<__nv_bfloat16>(g, SELF::buf), tk.x, tk.y);
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
