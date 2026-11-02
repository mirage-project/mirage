// tasks/residual_add.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// residual_add.cuh: y = bf16(bf16(o + S) + prefix), task x of grid (NSLICE, 1,
// 1) = slice x of y ([T][H] bf16): o from the GPU that owns the columns
// (sum_send, in this GPU's exchange region: system-scope polls), S from
// sum_gpus's buffer (device-scope polls), prefix (the residual) a graph input.
#pragma once
#include "../core.cuh"
#include "sum_send.cuh" // the layout of its input o

namespace static_mk {

template <int NSLICE, size_t OOFF, int H>
__device__ __forceinline__ void residual_add_task(G const &g,
                                                  __hip_bfloat16 *ssum,
                                                  __hip_bfloat16 const *prefix,
                                                  __hip_bfloat16 *y,
                                                  int x) {
  constexpr size_t O_RANK = o_rank_bytes(H);
  int const nv = T * H / 8, lo = (int)((long long)nv * x / NSLICE),
            hi = (int)((long long)nv * (x + 1) / NSLICE);
  for (int i = lo + STATIC_TIDX; i < hi; i += 256) {
    int const t = i / (H / 8), c = (i - t * (H / 8)) * 8, r = c / (H / GPUS),
              cc = c - r * (H / GPUS);
    uint4 const pu =
        ld16_at<AUX_DEV>(prefix, (uint32_t)(((size_t)t * H + c) * 2)); // (device scope: in a fused layer kernel another CU wrote it)
    void const *sp = ssum + (size_t)t * H + c;
    uint32_t const ooff = (uint32_t)(OOFF + (size_t)r * O_RANK +
                                     (size_t)t * (H / GPUS) * 2 + cc * 2 +
                                     rg_set());
    unsigned char const *op = g.rv + ooff;
    uint4 ou = make_uint4(0, 0, 0, 0), su; // both loads in flight together, re-loaded until landed
    bool ok_o = SOLO && r != g.rank, ok_s = false; // SOLO: another GPU's o: zeros
    do {
      if (!ok_o) {
        ou = ld16_at<AUX_SYS>(g.rv, ooff);
      }
      if (!ok_s) {
        su = ld16_at<AUX_DEV>(ssum, (uint32_t)(((size_t)t * H + c) * 2));
      }
      ok_o = ok_o || valid16w(ou);
      ok_s = ok_s || valid16w(su);
      if (!(ok_o && ok_s)) {
        tail_pause();
      }
    } while (!(ok_o && ok_s));
    uint32_t const ow[4] = {ou.x, ou.y, ou.z, ou.w},
                   sw[4] = {su.x, su.y, su.z, su.w},
                   pw[4] = {pu.x, pu.y, pu.z, pu.w};
    uint32_t yw[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float const a0 = round_bf16(bf16_lo(ow[k]) + bf16_lo(sw[k])),
                  a1 = round_bf16(bf16_hi(ow[k]) + bf16_hi(sw[k]));
      yw[k] = pack_bf16x2(a0 + bf16_lo(pw[k]), a1 + bf16_hi(pw[k]));
    }
    *reinterpret_cast<uint4 *>(y + (size_t)t * H + c) =
        make_uint4(yw[0], yw[1], yw[2], yw[3]);
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
  residual_add_task<SELF::x, exchange_offset[O::buf], H>(
      g,
      buf_at<__hip_bfloat16>(g, SSUM::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
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
