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
// GPUS > 1 (attention TP: each GPU's y is a partial sum): the all-reduce. The
// node's buffer is in the exchange region, [GPUS][T][N] bf16. A task first
// sends the bf16 sums of its columns (lane pairs: one 16-B multicast store per
// 8 columns) into slot `rank` of every GPU, then adds the GPUs' sums of its
// columns in GPU order (fp32) -> bf16 -> y, and re-arms them (0xFF, for launch
// n + 2: the region's two sets). These are the rounding points of the official
// RowParallelLinear (fp8_gemm's bf16 output, the GPUs' parts added in fp32).
// Every GPU runs the same plan, and a task sends before it waits: the task on
// each other GPU that sends these columns waits for nothing this one holds.
// RK > 0 (DeepSeek-V4.1's MoE, model.py MoE.forward): a second input, the
// routed rows [T x RK][N] fp32 of an expert node (row t RK + k: token t's k-th
// expert in ascending id; plain stores, complete when the node's counter
// counts all its tasks): added first, row t RK .. t RK + RK - 1 in order (the
// official loop over the experts), then the K parts (the shared expert's) in
// part order.
#pragma once
#include "../core.cuh"

namespace static_mk {

template <int N, int NC, int P, size_t XOFF, int RK = 0>
__device__ __forceinline__ void sum_store_task(G const &g,
                                               float const *part,
                                               float const *rows,
                                               uint32_t const *rows_cnt,
                                               uint32_t rows_target,
                                               __nv_bfloat16 *y,
                                               uint32_t *done_cnt,
                                               int x,
                                               bool warm) {
  static_assert(T == 8 && NC % 128 == 0 && N % NC == 0,
                "sum_store: a warp per token, whole 128-column blocks");
  int const t = threadIdx.x >> 5, l = threadIdx.x & 31;
  constexpr size_t XR = (size_t)T * N * 2; // one GPU's slot (bf16)
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    if constexpr (RK > 0) {
      if (real) { // the routed rows: all the expert node's tasks (acquire)
        if (threadIdx.x == 0) {
          cnt_wait(rows_cnt, rows_target);
        }
        __syncthreads();
      }
    }
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
      float4 s;
      if constexpr (RK > 0) { // the routed rows first, then every K part
        s = *reinterpret_cast<float4 const *>(rows + (size_t)t * RK * N + c);
#pragma unroll
        for (int k = 1; k < RK; k++) {
          float4 const r = *reinterpret_cast<float4 const *>(
              rows + ((size_t)t * RK + k) * N + c);
          s.x += r.x;
          s.y += r.y;
          s.z += r.z;
          s.w += r.w;
        }
#pragma unroll
        for (int p = 0; p < P; p++) {
          s.x += v[p].x;
          s.y += v[p].y;
          s.z += v[p].z;
          s.w += v[p].w;
        }
      } else {
        s = v[0];
#pragma unroll
        for (int p = 1; p < P; p++) { // in part order
          s.x += v[p].x;
          s.y += v[p].y;
          s.z += v[p].z;
          s.w += v[p].w;
        }
      }
      __nv_bfloat162 const h0 = __floats2bfloat162_rn(s.x, s.y),
                           h1 = __floats2bfloat162_rn(s.z, s.w);
      if (real) {
        uint32_t const w0 = *reinterpret_cast<uint32_t const *>(&h0),
                       w1 = *reinterpret_cast<uint32_t const *>(&h1);
        if constexpr (GPUS == 1) {
          *reinterpret_cast<uint2 *>(y + (size_t)t * N + c) =
              make_uint2(w0, w1);
        } else { // the even lane of a pair: its 4 columns and the next 4
          uint32_t const o0 = __shfl_down_sync(~0u, w0, 1),
                         o1 = __shfl_down_sync(~0u, w1, 1);
          if ((l & 1) == 0) {
            push16(g,
                   XOFF + g.rank * XR + ((size_t)t * N + c) * 2,
                   make_uint4(w0, w1, o0, o1));
          }
        }
        if constexpr (REARM) { // the one reader of these parts (config.cuh)
#pragma unroll
          for (int p = 0; p < P; p++) {
            rearm16(part + ((size_t)p * T + t) * N + c);
          }
        }
      }
    }
  }
  if constexpr (GPUS > 1) { // every GPU's sums of these columns, in GPU order
    for (int c = x * NC + 4 * l; c < (x + 1) * NC; c += 128) {
      // the pair's 16 B (both lanes load them; each takes its half)
      unsigned char const *row =
          g.rv + XOFF + rg_set() + ((size_t)t * N + (c & ~7)) * 2;
      uint4 u[GPUS];
      bool ok[GPUS] = {};
      bool all;
      do {
        all = true;
#pragma unroll
        for (int r = 0; r < GPUS; r++) {
          if (!ok[r]) {
            u[r] = ld16_poll(row + (size_t)r * XR);
          }
        }
#pragma unroll
        for (int r = 0; r < GPUS; r++) {
          if (!ok[r]) {
            ok[r] = valid16(u[r]);
          }
          all = all && ok[r];
        }
        if (!all) {
          poll_pause(STATIC_POLL_NS);
        }
      } while (!all);
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int r = 0; r < GPUS; r++) {
        uint32_t const a = (l & 1) ? u[r].z : u[r].x,
                       b = (l & 1) ? u[r].w : u[r].y;
        float2 const fa = __bfloat1622float2(
                         *reinterpret_cast<__nv_bfloat162 const *>(&a)),
                     fb = __bfloat1622float2(
                         *reinterpret_cast<__nv_bfloat162 const *>(&b));
        s.x += fa.x;
        s.y += fa.y;
        s.z += fb.x;
        s.w += fb.y;
      }
      __nv_bfloat162 const h0 = __floats2bfloat162_rn(s.x, s.y),
                           h1 = __floats2bfloat162_rn(s.z, s.w);
      *reinterpret_cast<uint2 *>(y + (size_t)t * N + c) =
          make_uint2(*reinterpret_cast<uint32_t const *>(&h0),
                     *reinterpret_cast<uint32_t const *>(&h1));
      if constexpr (REARM) { // the one reader of these slots, on this GPU
        __syncwarp();        // both lanes of a pair have loaded them
        if ((l & 1) == 0) {
#pragma unroll
          for (int r = 0; r < GPUS; r++) {
            rearm16(row + (size_t)r * XR);
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

// the first node of a pack (sum_store's second input: the routed rows' node)
template <class... R>
struct sum_store_rows {
  using type = void;
};
template <class A, class... R>
struct sum_store_rows<A, R...> {
  using type = A;
};

// sum_store task x, grid (N / NC, 1, 1); OB: wo_b's node (a gemm_fp8_tile in
// slots: OB::y K parts, OB::v[2] = N rows). PARAMS {N}, or {N, RK}: then
// REST's first is the routed rows' node (its buffer [T x RK][N] fp32, its
// counter of its tasks); BUF_SLOTS {y (a graph tensor, [T][N] bf16)}; GPUS >
// 1: the node's buffer, [GPUS][T][N] bf16 in the exchange region; the node's
// counter: one, of its tasks; warm: above
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
  static_assert(SELF::counter >= 0, "sum_store: a counter of its tasks");
  static_assert(GPUS == 1 ? SELF::buf < 0
                          : SELF::buf >= 0 &&
                                exchange_bytes[SELF::buf < 0 ? 0 : SELF::buf] ==
                                    (size_t)GPUS * T * N * 2,
                "sum_store: one GPU: no buffer (y a graph tensor); GPUS > 1: "
                "its buffer [GPUS][T][N] bf16 in the exchange region");
  static_assert(OB::buf >= 0 && OB::v[0] == COMBINE_SLOTS && OB::v[2] == N &&
                    OB::y >= 1 && OB::y <= 16,
                "sum_store: wo_b's K parts in slots, at most 16 (in "
                "registers)");
  constexpr int RK = param_or0<PARAMS, 1>(); // routed rows per token; 0: none
  float const *rows = nullptr;
  uint32_t const *rows_cnt = nullptr;
  uint32_t rows_target = 0;
  if constexpr (RK > 0) {
    using RR = typename sum_store_rows<REST...>::type;
    static_assert(RR::buf >= 0 && RR::counter >= 0,
                  "sum_store: the routed rows, an expert node's buffer and "
                  "the counter of its tasks");
    rows = buf_at<float const>(g, RR::buf);
    rows_cnt = g.cnt + RR::counter;
    rows_target = RR::x * RR::y * RR::z;
  }
  sum_store_task<N,
                 N / SELF::x,
                 OB::y,
                 (GPUS > 1 ? exchange_offset[SELF::buf < 0 ? 0 : SELF::buf]
                           : 0),
                 RK>(
      g,
      buf_at<float const>(g, OB::buf),
      rows,
      rows_cnt,
      rows_target,
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
