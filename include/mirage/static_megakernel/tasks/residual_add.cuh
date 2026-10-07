// tasks/residual_add.cuh -- y = bf16(bf16(o + S) + prefix), task x of grid
// (NSLICE, 1, 1) = slice x of y ([T][H] bf16; H = the node's params[0]): o from
// the GPU that owns the columns (sum_send), S from sum_gpus's buffer, prefix
// (the residual) a graph input
#pragma once
#include "../core.cuh"
#include "sum_send.cuh" // the layout of its input o

namespace static_mk {

// residual_add task x of NSLICE (ssum: sum_gpus's buffer)
template <int NSLICE, size_t OOFF, int H> // OOFF: o's offset in the exchange
                                          // region
__device__ __forceinline__ void residual_add_task(G const &g,
                                                  __nv_bfloat16 *ssum,
                                                  __nv_bfloat16 const *prefix,
                                                  __nv_bfloat16 *y,
                                                  int x) {
  constexpr size_t O_RANK = o_rank_bytes(H);
  int const nv = T * H / 8, lo = (int)((long long)nv * x / NSLICE),
            hi = (int)((long long)nv * (x + 1) / NSLICE);
  for (int i = lo + threadIdx.x; i < hi; i += 256) {
    int const t = i / (H / 8), c = (i - t * (H / 8)) * 8, r = c / (H / GPUS),
              cc = c - r * (H / GPUS);
    // prefix (a graph input) and S (landed long before o) are loaded before o
    // is polled: the three round trips overlap
    uint4 const pu =
        *reinterpret_cast<uint4 const *>(prefix + (size_t)t * H + c);
    unsigned char const *sp =
        reinterpret_cast<unsigned char const *>(ssum + (size_t)t * H + c);
    uint4 su = ld16_relaxed(sp);
    uint4 const ou = poll16(g.rv + OOFF + (size_t)r * O_RANK +
                            (size_t)t * (H / GPUS) * 2 + cc * 2 + rg_set());
    while (!valid16w(su)) {
      __nanosleep(64);
      su = ld16_relaxed(sp);
    }
    __nv_bfloat16 const *ob = reinterpret_cast<__nv_bfloat16 const *>(&ou),
                        *sb = reinterpret_cast<__nv_bfloat16 const *>(&su);
    __nv_bfloat16 const *pb = reinterpret_cast<__nv_bfloat16 const *>(&pu);
    uint4 yu;
    __nv_bfloat16 *yb = reinterpret_cast<__nv_bfloat16 *>(&yu);
    for (int k = 0; k < 8; k++) {
      float const a = __bfloat162float(
          __float2bfloat16(__bfloat162float(ob[k]) + __bfloat162float(sb[k])));
      yb[k] = __float2bfloat16(a + __bfloat162float(pb[k]));
    }
    *reinterpret_cast<uint4 *>(y + (size_t)t * H + c) = yu;
    if constexpr (REARM) { // the one reader of these 16 B of S and of o; after
                           // y is written
      rearm16(sp);
      rearm16(g.rv + OOFF + (size_t)r * O_RANK + (size_t)t * (H / GPUS) * 2 +
              cc * 2 + rg_set());
    }
  }
  __syncthreads();
}

// residual_add task x: slice x of SELF::x of y; SSUM: the addend's node
// (sum_gpus), O: the input's node (sum_send, its output in the exchange
// region); PARAMS {H}; BUF_SLOTS {the residual, y (graph tensors, [T][H] bf16)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SSUM,
          class O,
          class... IN>
__device__ __forceinline__ void run_residual_add(Maps const &,
                                                 G const &g,
                                                 KernelLocals &,
                                                 StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  static_assert(H % (8 * GPUS) == 0,
                "residual_add: whole 8-column vectors of every GPU's part");
  static_assert(SSUM::buf >= 0 && O::buf >= 0 &&
                    exchange_bytes[O::buf] == (size_t)GPUS * o_rank_bytes(H),
                "residual_add: S from sum_gpus's buffer, o from sum_send's");
#ifdef STATIC_RESET_IN_KERNEL
  if constexpr (PDL_TRIGGER == 2) {
    asm volatile("griddepcontrol.launch_dependents;" ::
                     : "memory"); // PDL: near the end (the last task)
  }
#endif
  residual_add_task<SELF::x, exchange_offset[O::buf], H>(
      g,
      buf_at<__nv_bfloat16>(g, SSUM::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_residual_add() {
  return 0;
}

} // namespace static_mk
