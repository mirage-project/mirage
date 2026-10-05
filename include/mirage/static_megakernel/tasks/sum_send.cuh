// tasks/sum_send.cuh -- adds a GEMM's K parts (its output buffer in slots,
// gemm_tile.cuh COMBINE_SLOTS; its rows split over the GPUs) and sends the sum
// (bf16) to every GPU (the node's output, an exchange buffer [GPUS][T][H /
// GPUS] bf16: each GPU its own rows): task (x, y) of grid (H / 128, Q, 1), this
// GPU's positions of x = its row blocks (H / GPUS / 128 of them), y = 1/Q of
// the block's 128 16-B vectors. H = the GEMM's rows (its params[2]).
#pragma once
#include "../core.cuh"

namespace static_mk {

// one GPU's part of the output for rows of H: [T][H / GPUS] bf16
__host__ __device__ constexpr size_t o_rank_bytes(int H) {
  return (size_t)T * (H / GPUS) * 2;
}

constexpr int MAX_SUM_PARTS = 14; // K parts: at most this many (sum_send keeps
                                  // 2 of its loads per part in registers)

// sum_send task (x, jq) of Q: x = a row block of H (the GEMM's rows; this GPU
// owns blocks [B rank, B rank + B), B = H / GPUS / 128): adds the P K parts
// (opart: the GEMM's buffer, this GPU's rows only: fp32 [P][T][H / GPUS]) of
// vectors [jq 128 / Q, (jq + 1) 128 / Q) of block x (vector i = token i / 16,
// rows 8 (i % 16) .. + 8), sends them (bf16) to every GPU (OOFF: the output's
// offset; this GPU's slot: its own H / GPUS rows)
template <int P, int Q, size_t OOFF, int H>
__device__ __forceinline__ void
    sum_send_task(G const &g, float *opart, int x, int jq) {
  static_assert(P <= MAX_SUM_PARTS && Q <= 128, "sum_send");
  static_assert(H % (128 * GPUS) == 0,
                "sum_send: whole 128-row blocks per GPU");
  constexpr size_t O_RANK = o_rank_bytes(H);
  int const i = threadIdx.x, t = i >> 4, seg = i & 15, v0 = jq * 128 / Q,
            v1 = (jq + 1) * 128 / Q;
  if (i >= v0 && i < v1) {
    float o8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float4 x0[P], x1[P];
    bool ok[P];
    bool all; // all 2 P loads in flight at once
#pragma unroll
    for (int p_ = 0; p_ < P; p_++) {
      ok[p_] = false;
    }
    do {
      all = true;
#pragma unroll
      for (int p_ = 0; p_ < P; p_++) {
        if (!ok[p_]) {
          float const *src = opart + ((size_t)p_ * T + t) * (H / GPUS) +
                             x * 128 - g.rank * (H / GPUS) + seg * 8;
          x0[p_] = ldf4_poll(src);
          x1[p_] = ldf4_poll(src + 4);
        }
      }
#pragma unroll
      for (int p_ = 0; p_ < P; p_++) {
        if (!ok[p_]) {
          ok[p_] = validf4(x0[p_]) && validf4(x1[p_]);
        }
        all = all && ok[p_];
      }
      if (!all) {
        __nanosleep(64);
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
    __nv_bfloat162 const b0 = __floats2bfloat162_rn(o8[0], o8[1]),
                         b1 = __floats2bfloat162_rn(o8[2], o8[3]);
    __nv_bfloat162 const b2 = __floats2bfloat162_rn(o8[4], o8[5]),
                         b3 = __floats2bfloat162_rn(o8[6], o8[7]);
    push16(g,
           OOFF + (size_t)g.rank * O_RANK + (size_t)t * (H / GPUS) * 2 +
               (size_t)(x * 128 - g.rank * (H / GPUS) + seg * 8) * 2,
           make_uint4(*reinterpret_cast<uint32_t const *>(&b0),
                      *reinterpret_cast<uint32_t const *>(&b1),
                      *reinterpret_cast<uint32_t const *>(&b2),
                      *reinterpret_cast<uint32_t const *>(&b3)));
    if constexpr (REARM) { // the one reader of these parts; after o is sent
                           // (off the path to residual_add)
#pragma unroll
      for (int p_ = 0; p_ < P; p_++) {
        float const *src = opart + ((size_t)p_ * T + t) * (H / GPUS) + x * 128 -
                           g.rank * (H / GPUS) + seg * 8;
        rearm16(src);
        rearm16(src + 4);
      }
    }
  }
  __syncthreads();
}

// sum_send task (row block x of this GPU's, slice y of SELF::y): adds UP's
// UP::y K parts (UP: a GEMM in slots, its rows split over the GPUs; UP::v[2] =
// its rows N)
template <class SELF, class PARAMS, class SLOTS, class UP, class... REST>
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

template <class SELF, class PARAMS, class SLOTS, class... IN>
constexpr int smem_sum_send() {
  return 0;
}
} // namespace static_mk
