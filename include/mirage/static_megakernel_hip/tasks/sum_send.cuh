// tasks/sum_send.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// sum_send.cuh: adds a GEMM's K parts (its output buffer in slots, COMBINE_SLOTS;
// its rows split over the GPUs) and sends the sum (bf16) to every GPU (the
// node's output, an exchange buffer [GPUS][T][H / GPUS] bf16): task (x, y) of
// grid (H / 128, Q, 1). On gfx950 the K parts are polled with device-scope loads
// (the GEMM wrote them with st16_dev) and the sum goes to each GPU by a
// system-scope store (push16).
#pragma once
#include "../core.cuh"

namespace static_mk {

// one GPU's part of the output for rows of H: [T][H / GPUS] bf16
__host__ __device__ constexpr size_t o_rank_bytes(int H) {
  return (size_t)T * (H / GPUS) * 2;
}

constexpr int MAX_SUM_PARTS = 14; // K parts: at most this many

// sum_send task (x, jq) of Q: x = a row block of H (this GPU owns blocks [B
// rank, B rank + B), B = H / GPUS / 128): adds the P K parts (opart: fp32 [P][T]
// [H / GPUS], this GPU's rows) of vectors [jq 128 / Q, (jq + 1) 128 / Q) of
// block x (vector i = token i / 16, rows 8 (i % 16) .. + 8), sends them (bf16)
// to every GPU (OOFF: the output's offset; this GPU's slot: its own rows)
template <int P, int Q, size_t OOFF, int H>
__device__ __forceinline__ void
    sum_send_task(G const &g, float *opart, int x, int jq) {
  static_assert(P <= MAX_SUM_PARTS && Q <= 128, "sum_send");
  static_assert(H % (128 * GPUS) == 0,
                "sum_send: whole 128-row blocks per GPU");
  constexpr size_t O_RANK = o_rank_bytes(H);
  int const i = STATIC_TIDX, t = i >> 4, seg = i & 15, v0 = jq * 128 / Q,
            v1 = (jq + 1) * 128 / Q;
  if (i >= v0 && i < v1) {
    float o8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float4 x0[P], x1[P];
    bool ok[P];
    bool all;
#pragma unroll
    for (int p_ = 0; p_ < P; p_++) {
      ok[p_] = false;
    }
    do {
      all = true;
#pragma unroll
      for (int p_ = 0; p_ < P; p_++) {
        if (!ok[p_]) {
          uint32_t const off = (uint32_t)((((size_t)p_ * T + t) * (H / GPUS) +
                                           x * 128 - g.rank * (H / GPUS) +
                                           seg * 8) * 4);
          x0[p_] = as_float4(ld16_at<AUX_DEV>(opart, off));
          x1[p_] = as_float4(ld16_at<AUX_DEV>(opart, off + 16));
          ok[p_] = validf4(x0[p_]) && validf4(x1[p_]);
        }
        all = all && ok[p_];
      }
      if (!all) {
        poll_pause();
      }
    } while (!all);
#pragma unroll
    for (int p_ = 0; p_ < P; p_++) { // fixed order
      o8[0] += x0[p_].x;
      o8[1] += x0[p_].y;
      o8[2] += x0[p_].z;
      o8[3] += x0[p_].w;
      o8[4] += x1[p_].x;
      o8[5] += x1[p_].y;
      o8[6] += x1[p_].z;
      o8[7] += x1[p_].w;
    }
    push16(g,
           OOFF + (size_t)g.rank * O_RANK + (size_t)t * (H / GPUS) * 2 +
               (size_t)(x * 128 - g.rank * (H / GPUS) + seg * 8) * 2,
           make_uint4(pack_bf16x2(o8[0], o8[1]),
                      pack_bf16x2(o8[2], o8[3]),
                      pack_bf16x2(o8[4], o8[5]),
                      pack_bf16x2(o8[6], o8[7])));
  }
  __syncthreads();
}

// sum_send task (row block x of this GPU's, slice y of SELF::y): adds UP's
// UP::y K parts (UP: a GEMM in slots, its rows split over the GPUs; UP::v[2] =
// its rows N)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class UP,
          class... REST>
__device__ __forceinline__ void run_sum_send(Maps const &,
                                             G const &g,
                                             KernelLocals &,
                                             StaticTask const &tk) {
  constexpr int H = UP::v[2];
  static_assert(UP::buf >= 0 && UP::v[0] == COMBINE_SLOTS,
                "sum_send: a GEMM's K parts in slots");
  static_assert(
      SELF::buf >= 0 &&
          exchange_bytes[SELF::buf] == (size_t)GPUS * o_rank_bytes(H),
      "sum_send: its output, [GPUS][T][H / GPUS] bf16 in the exchange region");
  sum_send_task<UP::y, SELF::y, exchange_offset[SELF::buf], H>(
      g, buf_at<float>(g, UP::buf), tk.x, tk.y);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sum_send() {
  return 0;
}
} // namespace static_mk
