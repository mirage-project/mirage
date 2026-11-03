// tasks/sum_store.cuh -- the end of DeepSeek-V4.1's attention (model.py L790,
// x = wo_b(o)): wo_b's P K parts (fp32, its buffer in slots [P][T][N], 0xFF
// before the launch: polled) added in part order -> bf16 (fp8_gemm's bf16
// output) -> the layer's output y [T][N] (a graph tensor, written by pointer;
// every value each launch). Task x of grid (N / NC, 1, 1): columns [x NC,
// (x + 1) NC), NC whole 128-column blocks; per block warp t takes token t,
// lane l columns 4 l .. 4 l + 3. Then the node's counter += 1. warm = false
// (this node's first task on the SM): first a dry pass, the same
// code with one round of loads and no stores, while wo_b's parts are on their
// way; one copy of the code (unroll 1).
#pragma once
#include "../core.cuh"

namespace static_mk {

template <int N, int NC, int P>
__device__ __forceinline__ void sum_store_task(
    float const *part, __nv_bfloat16 *y, uint32_t *done_cnt, int x, bool warm) {
  static_assert(T == 8 && NC % 128 == 0 && N % NC == 0,
                "sum_store: a warp per token, whole 128-column blocks");
  int const t = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    for (int c = x * NC + 4 * l; c < (x + 1) * NC; c += 128) {
      float4 v[P];
      bool ok[P];
#pragma unroll
      for (int p = 0; p < P; p++) {
        ok[p] = false;
      }
      bool all; // all P loads in flight at once
      do {
        all = true;
#pragma unroll
        for (int p = 0; p < P; p++) {
          if (!ok[p]) {
            v[p] = ldf4_poll(part + ((size_t)p * T + t) * N + c);
          }
        }
#pragma unroll
        for (int p = 0; p < P; p++) {
          if (!ok[p]) {
            ok[p] = validf4(v[p]);
          }
          all = all && ok[p];
        }
        if (!all && real) {
          poll_pause(STATIC_POLL_NS);
        }
      } while (!all && real);
      float4 s = v[0];
#pragma unroll
      for (int p = 1; p < P; p++) { // in part order
        s.x += v[p].x;
        s.y += v[p].y;
        s.z += v[p].z;
        s.w += v[p].w;
      }
      __nv_bfloat162 const h0 = __floats2bfloat162_rn(s.x, s.y),
                           h1 = __floats2bfloat162_rn(s.z, s.w);
      if (real) {
        *reinterpret_cast<uint2 *>(y + (size_t)t * N + c) =
            make_uint2(*reinterpret_cast<uint32_t const *>(&h0),
                       *reinterpret_cast<uint32_t const *>(&h1));
        if constexpr (REARM) { // the one reader of these parts (config.cuh)
#pragma unroll
          for (int p = 0; p < P; p++) {
            rearm16(part + ((size_t)p * T + t) * N + c);
          }
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
}

// sum_store task x, grid (N / NC, 1, 1); OB: wo_b's node (a gemm_fp8_tile in
// slots: OB::y K parts, OB::v[2] = N rows). PARAMS {N}; BUF_SLOTS {y (a graph
// tensor, [T][N] bf16)}; the node's counter: one, of its tasks; warm: above
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class OB,
          class... REST>
__device__ __forceinline__ void run_sum_store(Maps const &,
                                              G const &g,
                                              KernelLocals &,
                                              StaticTask const &tk,
                                              bool warm = true) {
  constexpr int N = PARAMS::v[0];
  static_assert(SELF::y == 1 && SELF::z == 1 && N % SELF::x == 0,
                "sum_store: grid (column blocks)");
  static_assert(SELF::buf < 0 && SELF::counter >= 0,
                "sum_store: its output a graph tensor, a counter of its tasks");
  static_assert(OB::buf >= 0 && OB::v[0] == COMBINE_SLOTS && OB::v[2] == N &&
                    OB::y >= 1 && OB::y <= 16,
                "sum_store: wo_b's K parts in slots, at most 16 (in "
                "registers)");
  sum_store_task<N, N / SELF::x, OB::y>(
      buf_at<float const>(g, OB::buf),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 0>()),
      g.cnt + SELF::counter,
      tk.x,
      warm);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sum_store() {
  return 0;
}

} // namespace static_mk
