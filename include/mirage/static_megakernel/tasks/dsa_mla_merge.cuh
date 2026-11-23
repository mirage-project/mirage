// tasks/dsa_mla_merge.cuh -- Merge the splits of sparse MLA decode. Task (h, y)
// is head h of the G tokens [G y, G y + G) (G = T / the grid's y: 1, or 8 at
// T = 64, where a task per token is 1024 tasks of ~1.2 us, 7 deep on an SM:
// the last ran ~8.5 us after its splits were written). For each token t:
//   o[t][h] = bf16( sum_s e^(lse_s - m) o_s / sum_s e^(lse_s - m) ),
//   m = max_s lse_s, over the S splits of sparse_mla.
// Each o_s is already the softmax average of its split. Without seq_lens
// (KPOOL > 1), a split with no valid slot has lse = -inf. With seq_lens
// (KPOOL 1), the task reads only the splits that hold a valid slot: split s
// holds slots [s TOPK / S, (s + 1) TOPK / S) (T = 64, ATTN64_MLALIVE:
// dsa_common.cuh mla_split_tiles), and the first min(seq_lens[t], TOPK) slots
// are valid. sparse_mla writes nothing for the other splits (T = 64,
// ATTN64_MERGENS: nor do they add to its counter; the task waits for
// the splits that hold a valid slot). The output is o_lat [T][NH][512] bf16,
// the attention output in the latent space. Then the task adds 1 to counter +
// h (dsa_o_absorb waits for the tasks of each head).
#pragma once
#include "../dsa_common.cuh"
#include <cmath>

namespace static_mk {

// Task (h, y). PARAMS {NH}. BUF_SLOTS {seq_lens (only with KPOOL 1)}. SM: the
// sparse_mla node (grid (S, T); its counter + t counts the splits of token t).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SM,
          class... IN>
__device__ __forceinline__ void run_dsa_mla_merge(Maps const &,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int NH = PARAMS::v[0], S = SM::x, LAT = dsa::MLA_LATENT,
                G = T / SELF::y;
  static_assert(SM::y == T && SM::buf >= 0 && SM::counter >= 0 &&
                    SM::v[0] == NH,
                "dsa_mla_merge: a sparse_mla node's splits of NH heads");
  static_assert(
      SELF::x == NH && SELF::y * G == T && SELF::buf >= 0 && SELF::counter >= 0,
      "dsa_mla_merge: grid (heads, token tiles), o_lat and a counter");
  int const h = tk.x, t0 = tk.y * G;
  // ns(t): the number of splits of token t that hold a valid slot (of S). TOPK
  // and KPOOL are params of SM.
  constexpr int SEQ = slot_or<BUF_SLOTS, 0, -1>(), TOPK = SM::v[1];
  static_assert(
      SEQ < 0 || SM::v[7] == 1,
      "dsa_mla_merge: seq_lens only for a sparse_mla node of KPOOL 1");
  // T = 64 (dsa_common.cuh ATTN64_MERGENS, ATTN64_MLALIVE): ns
  // from mla_live_splits, sparse_mla's own test of an empty split;
  // ATTN64_MERGENS: the wait for the ns splits that hold a valid slot
  // (sparse_mla's empty splits add nothing)
  constexpr bool LIVE_SPLITS =
      SEQ >= 0 && (dsa::ATTN64_MERGENS || dsa::ATTN64_MLALIVE);
  constexpr bool WAIT_LIVE = SEQ >= 0 && dsa::ATTN64_MERGENS;
  auto ns_of = [&](int t) {
    if constexpr (SEQ >= 0) {
      int const live = min(max(buf_at<int const>(g, SEQ)[t], 0), TOPK);
      if constexpr (LIVE_SPLITS) {
        return dsa::mla_live_splits<TOPK, S>(live);
      } else {
        return (live + TOPK / S - 1) / (TOPK / S);
      }
    } else {
      return S;
    }
  };
  // The task uses no shared memory, so it starts with no barrier.
  if (threadIdx.x == 0) {
    if constexpr (WAIT_LIVE) { // the G loads of seq_lens first, then the waits
      int ns[G];
#pragma unroll
      for (int i = 0; i < G; i++) {
        ns[i] = ns_of(t0 + i);
      }
#pragma unroll
      for (int i = 0; i < G; i++) {
        cnt_wait(cnt_now(g) + SM::counter + t0 + i, ns[i]);
      }
    } else {
      for (int i = 0; i < G; i++) {
        cnt_wait(cnt_now(g) + SM::counter + t0 + i, S);
      }
    }
  }
  __syncthreads();
  dsa::stamp_ready(L);
  float const *o = buf_at<float const>(g, SM::buf);
  float const *lse = o + (size_t)T * S * NH * LAT;
  if constexpr (G == 1) {
    int const t = t0, c = 2 * threadIdx.x, ns = ns_of(t);
    // Issue all loads first (the lse values and the column pairs of the thread
    // in the ns splits). Thus they go to L2 at the same time. A split s >= ns
    // gets lse -inf, so its weight is 0.
    float l[S];
    float2 v[S];
#pragma unroll
    for (int s = 0; s < S; s++) {
      l[s] = -INFINITY;
      v[s] = make_float2(0.f, 0.f);
      if (s >= ns) {
        continue;
      }
      asm volatile("ld.global.cg.f32 %0, [%1];"
                   : "=f"(l[s])
                   : "l"(lse + ((size_t)t * S + s) * NH + h)
                   : "memory");
      asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
                   : "=f"(v[s].x), "=f"(v[s].y)
                   : "l"(o + (((size_t)t * S + s) * NH + h) * LAT + c)
                   : "memory");
    }
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < S; s++) {
      m = fmaxf(m, l[s]);
    }
    float wsum = 0.f;
#pragma unroll
    for (int s = 0; s < S; s++) {
      l[s] = m == -INFINITY ? 0.f : __expf(l[s] - m);
      wsum += l[s];
    }
    float const inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
    for (int s = 0; s < S; s++) {
      acc.x += l[s] * v[s].x;
      acc.y += l[s] * v[s].y;
    }
    *reinterpret_cast<__nv_bfloat162 *>(buf_at<__nv_bfloat16>(g, SELF::buf) +
                                        ((size_t)t * NH + h) * LAT + c) =
        __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
  } else {
    // G tokens: thread j has the columns [4 q, 4 q + 4), q = j % 128, of the
    // G / 2 tokens of parity j / 128, in 16-B loads; the G S lse values are
    // loaded once a warp (lane i S + s) and shuffled: a thread per column pair
    // and token with its own lse loads is 32 L2 round trips a thread (the
    // task: 4.5 us)
    static_assert(G % 2 == 0 && G * S <= 32 && LAT == 512,
                  "dsa_mla_merge: token tiles of an even size, a lane per lse");
    constexpr int GH = G / 2;
    int const lane = threadIdx.x & 31, q = threadIdx.x & 127,
              par = threadIdx.x >> 7;
    float lv = -INFINITY;
    if (lane < G * S && lane % S < ns_of(t0 + lane / S)) {
      asm volatile(
          "ld.global.cg.f32 %0, [%1];"
          : "=f"(lv)
          : "l"(lse + ((size_t)(t0 + lane / S) * S + lane % S) * NH + h)
          : "memory");
    }
    float4 v[GH][S];
#pragma unroll
    for (int k = 0; k < GH; k++) {
      int const ns = ns_of(t0 + 2 * k + par);
#pragma unroll
      for (int s = 0; s < S; s++) {
        v[k][s] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (s >= ns) {
          continue;
        }
        asm volatile(
            "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(v[k][s].x), "=f"(v[k][s].y), "=f"(v[k][s].z), "=f"(v[k][s].w)
            : "l"(o + (((size_t)(t0 + 2 * k + par) * S + s) * NH + h) * LAT +
                  4 * q)
            : "memory");
      }
    }
#pragma unroll
    for (int k = 0; k < GH; k++) {
      float l[S], m = -INFINITY;
#pragma unroll
      for (int s = 0; s < S; s++) {
        l[s] = __shfl_sync(0xffffffffu, lv, (2 * k + par) * S + s);
        m = fmaxf(m, l[s]);
      }
      float wsum = 0.f;
#pragma unroll
      for (int s = 0; s < S; s++) {
        l[s] = m == -INFINITY ? 0.f : __expf(l[s] - m);
        wsum += l[s];
      }
      float const inv = wsum > 0.f ? 1.0f / wsum : 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < S; s++) {
        acc.x += l[s] * v[k][s].x;
        acc.y += l[s] * v[k][s].y;
        acc.z += l[s] * v[k][s].z;
        acc.w += l[s] * v[k][s].w;
      }
      __nv_bfloat162 *const d = reinterpret_cast<__nv_bfloat162 *>(
          buf_at<__nv_bfloat16>(g, SELF::buf) +
          ((size_t)(t0 + 2 * k + par) * NH + h) * LAT + 4 * q);
      d[0] = __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
      d[1] = __floats2bfloat162_rn(acc.z * inv, acc.w * inv);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter + h, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_mla_merge() {
  return 0;
}

} // namespace static_mk
