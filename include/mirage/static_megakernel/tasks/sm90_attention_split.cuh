/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// tasks/sm90_attention_split.cuh -- the sm90_attention_split task type: full
// (no window) paged decode attention with the KV split FlashDecoding-style, as
// a pool: one task per SM; the SMs claim the step's units from a queue, a unit
// = request r, split s -- a contiguous range of the request's cached positions
// (sm90_decode_split: about UNITS_PER_SM units an SM over the requests, in
// proportion to their lengths; a few large splits rather than many small
// chunks), all its KV heads: warp h takes KV head h, so a unit reads whole
// cache rows (every head's 128 B of a position). Each request has one token a
// step (decode). A warp streams its head's 32-position tiles of K and V
// through a private 3-stage cp.async ring in shared memory (swizzled 128-B
// rows) and runs mma.sync on them: S = q K^T (ldmatrix), O += P V
// (ldmatrix.trans), an online softmax over the unit's positions. Its 8 query
// heads' output (normalized) and log2-sum-exps go to parts in fp32 for
// sm90_attention_merge (with the sinks). The last split of a request also
// takes the new token: its K (RoPE) and V appended to the cache, its score
// folded in exactly.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// Request r's cached positions (the new token not counted), from the page
// tables
template <int PAGE>
__device__ __forceinline__ int
    sm90_decode_cached(int const *kv_indptr, int const *last_page_len, int r) {
  return max(
      (kv_indptr[r + 1] - kv_indptr[r] - 1) * PAGE + last_page_len[r] - 1, 0);
}
// The splits of request r's cached positions: `tiles` 32-position tiles a
// split, `n` splits. UNITS_PER_SM x sms splits over all requests in proportion
// to their tiles (total: all requests' tiles; at least one a request, so at
// most UNITS_PER_SM x sms + R in all); every task computes the same, and
// writes the requests' first splits (split_base) for the merge.
// WINDOW > 0 (a sliding window): the new token at position `cached` attends to
// keys in (cached - WINDOW, cached], so the cached positions from lo = cached -
// WINDOW + 1 on, tiles from lo / 32 (t0) on; positions below lo in tile t0 are
// masked
struct Sm90DecodeSplit {
  int cached, lo, t0, tiles, n;
};
template <int PAGE, int WINDOW>
__device__ __forceinline__ int
    sm90_decode_tiles(int const *kv_indptr, int const *last_page_len, int r) {
  int const cached = sm90_decode_cached<PAGE>(kv_indptr, last_page_len, r);
  int const lo = WINDOW > 0 ? max(cached - WINDOW + 1, 0) : 0;
  return (cached + 31) / 32 - lo / 32;
}
template <int PAGE, int UNITS_PER_SM, int WINDOW = 0, bool ONE = false>
__device__ __forceinline__ Sm90DecodeSplit sm90_decode_split(
    int const *kv_indptr, int const *last_page_len, int r, int sms, int total) {
  Sm90DecodeSplit s;
  s.cached = sm90_decode_cached<PAGE>(kv_indptr, last_page_len, r);
  s.lo = WINDOW > 0 ? max(s.cached - WINDOW + 1, 0) : 0;
  s.t0 = s.lo / 32;
  int const tiles = (s.cached + 31) / 32 - s.t0;
  int n = total > 0 ? (int)((long long)UNITS_PER_SM * sms * tiles / total) : 1;
  n = ONE ? 1 : max(n, 1);
  s.tiles = max((tiles + n - 1) / n, 1);
  s.n = max((tiles + s.tiles - 1) / s.tiles, 1);
  return s;
}

namespace sm90_dec {

__device__ __forceinline__ void mma_bf16(
    float (&d)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  // rows 8..15 of A are zero (one token's 8 query heads of a KV head)
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// 16 B global -> shared, asynchronously (bytes < 16: the rest zeros)
__device__ __forceinline__ void
    cp_async16(void *dst, void const *src, int bytes) {
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;" ::"r"(
                   (uint32_t)__cvta_generic_to_shared(dst)),
               "l"(src),
               "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

template <bool TRANS>
__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], void const *p) {
  uint32_t const a = (uint32_t)__cvta_generic_to_shared(p);
  if constexpr (TRANS) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, "
                 "%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
                 "[%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
  }
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

} // namespace sm90_dec

// PARAMS {NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, PAGE_SIZE,
// UNITS_PER_SM, P (the splits of all requests at most), QKV_BIAS, R
// (requests)};
// BUF_SLOTS {qkv, k cache, v cache, cos, sin, qo_indptr, kv_indptr,
// kv_indices, last_page_len (sm90_attention's), the qkv bias (QKV_BIAS; qkv is
// then the GEMM's fp32, else bf16) and an unused slot (-1 or sm90_attention's
// staging rows), parts: the splits' outputs [P, Q x D] fp32 (request r's
// split s at split_base[r] + s), then their log2-sum-exps [P, Q] fp32, then
// split_base [R + 1] (int; for the merge)}.
// Counters: the node's, then (a line on) the queue.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention_split(Maps const &,
                                                         G const &g,
                                                         KernelLocals &L,
                                                         StaticTask const &) {
  using namespace sm90_dec;
  using bf16 = __nv_bfloat16;
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2],
                MAX_SEQ = PARAMS::v[3], PAGE = PARAMS::v[4], UPS = PARAMS::v[5],
                P = PARAMS::v[6], R = PARAMS::v[8];
  constexpr bool QKV_BIAS = PARAMS::v[7] == 1;
  constexpr int WINDOW = param_or<PARAMS, 9, 0>(); // 0: full attention
  // DIRECT: one split a request (a sliding window's few tiles), which writes
  // the final output itself with the sinks (sm90_attention_merge's formula
  // for one split; no merge node): BUF_SLOTS 12 the sinks [Q] bf16, 13 the
  // output [T, Q x D] bf16 (last: the node's output)
  constexpr bool DIRECT = param_or<PARAMS, 10, 0>() != 0;
  constexpr int QPK = QH / KVH, QUEUE = SELF::counter + 32, W = (QPK + 2) * D,
                STRIDE = KVH * W, MAX_PAGES = MAX_SEQ / PAGE;
  // qkv: the QKV GEMM's K parts [KP][T][STRIDE] fp32 ("store", store_parts:
  // its grid y), added here
  constexpr int KP = Sm90First<IN...>::type::y;
  static_assert(KP == 1 || QKV_BIAS, "qkv in K parts: the fp32 GEMM output");
  static_assert(D == 64 && QPK == 8 && PAGE % 32 == 0 && KVH == SM90_WARPS,
                "sm90_attention_split: 8 KV heads of 8 query heads of 64");
  constexpr int STAGES = 3, TILE_BYTES = 2 * 32 * D * 2; // K + V, 32 rows
  constexpr float SCALE = 0.125f * 1.4426950408889634f;  // 1/sqrt(D) log2 e
  int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
  int const *kv_indptr = buf_at<int const>(g, slot_at<BUF_SLOTS, 6>());
  int const *kv_indices = buf_at<int const>(g, slot_at<BUF_SLOTS, 7>());
  int const *last_page_len = buf_at<int const>(g, slot_at<BUF_SLOTS, 8>());
  bf16 *k_cache = buf_at<bf16>(g, slot_at<BUF_SLOTS, 1>());
  bf16 *v_cache = buf_at<bf16>(g, slot_at<BUF_SLOTS, 2>());
  float *parts = buf_at<float>(g, slot_at<BUF_SLOTS, 11>());
  float *lse_out = parts + (size_t)P * QH * D;

  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2,
            qd = lane & 3;
  // the unit's page indices; each head's new K and V (the last split)
  __shared__ int pages[MAX_PAGES];
  __shared__ int claim[2];
  extern __shared__ __align__(16) char dyn_smem[];
  __shared__ __align__(16) bf16 new_k[KVH][D], new_v[KVH][D];

  __syncthreads(); // this SM's earlier tasks are done with shared memory
  if (threadIdx.x == 0) {
    claim[0] = atomicAdd(g.cnt + QUEUE, 1u);
  }
  // the units: request-major, then split (split_base[r]: the splits before
  // request r's; unit u's partial is parts' u-th)
  constexpr int RPL = (R + 31) / 32; // requests a lane: RPL lane + i
  static_assert(R <= 64, "sm90_attention_split: two requests a lane");
  __shared__ int split_base[R + 1], total_tiles, dir_hg;
  if (warp == 0) {
    // DIRECT: HG units a request -- head groups of KVH / HG heads, each head's
    // tiles over HG warps (combined in shared memory) -- the most of 8, 4, 2,
    // 1 with R HG units on the SMs: a window's few tiles from many SMs
    int hg = 1;
    if constexpr (DIRECT) {
      hg = KVH;
      while (hg > 1 && R * hg > (int)gridDim.x) {
        hg >>= 1;
      }
    }
    int tiles = 0, n[RPL];
#pragma unroll
    for (int i = 0; i < RPL; i++) {
      int const r = RPL * lane + i;
      tiles +=
          r < R ? sm90_decode_tiles<PAGE, WINDOW>(kv_indptr, last_page_len, r)
                : 0;
    }
    int const total = __reduce_add_sync(~0u, tiles);
    int sum = 0;
#pragma unroll
    for (int i = 0; i < RPL; i++) {
      int const r = RPL * lane + i;
      n[i] = r < R && qo[r + 1] > qo[r]
                 ? (DIRECT ? hg
                           : sm90_decode_split<PAGE, UPS, WINDOW, DIRECT>(
                                 kv_indptr, last_page_len, r, gridDim.x, total)
                                 .n)
                 : 0;
      sum += n[i];
    }
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { // inclusive prefix sum
      int const v = __shfl_up_sync(~0u, incl, o);
      incl += lane >= o ? v : 0;
    }
    int *const base_out = reinterpret_cast<int *>(lse_out + (size_t)P * QH);
    int acc = incl - sum;
#pragma unroll
    for (int i = 0; i < RPL; i++) {
      int const r = RPL * lane + i;
      acc += n[i];
      if (r < R) {
        split_base[r + 1] = acc;
        base_out[r + 1] = acc; // (every task the same values)
      }
    }
    if (lane == 0) {
      split_base[0] = 0, total_tiles = total, dir_hg = hg;
      base_out[0] = 0;
    }
  }
  __syncthreads();
  int const units = split_base[R];
  bool waited = false;
  for (int k = 0;; k++) {
    int const u = claim[k & 1];
    if (u >= units) {
      break;
    }
    int request = 0;
    while (u >= split_base[request + 1]) {
      request++;
    }
    int const split = u - split_base[request];
    Sm90DecodeSplit const sp = sm90_decode_split<PAGE, UPS, WINDOW, DIRECT>(
        kv_indptr, last_page_len, request, gridDim.x, total_tiles);
    // DIRECT: split = the head group; warp w its head hpu split + w % hpu,
    // the head's tiles slice, slice + hg, .. (slice w / hpu); every unit has
    // the new token (slice 0 appends it and folds it in)
    int const hg = DIRECT ? dir_hg : 1, hpu = KVH / hg;
    int const head = DIRECT ? split * hpu + warp % hpu : warp,
              slice = DIRECT ? warp / hpu : 0;
    bool const last = DIRECT || split == sp.n - 1;
    int const tile0 = sp.t0 + (DIRECT ? 0 : split) * sp.tiles,
              tile1 = min(tile0 + sp.tiles, (sp.cached + 31) / 32);
    int const page0 = tile0 * 32 / PAGE,
              page1 = (max(tile1, tile0) * 32 + PAGE - 1) / PAGE;
    int const first_page = kv_indptr[request];
    for (int i = threadIdx.x; i < page1 - page0; i += SM90_THREADS) {
      pages[i] = kv_indices[first_page + page0 + i];
    }
    __syncthreads();

    // the unit's tiles tile0 + i (i < nt), head `head`'s rows, through the
    // warp's own ring of STAGES 32-position tiles in shared memory (K then V,
    // 4 KB each; 16-B chunk c of row r at chunk c ^ (r & 7)), by cp.async
    int const nt = max(tile1 - tile0, 0),
              ntw = nt > slice ? (nt - slice + hg - 1) / hg : 0; // the warp's
    char *const ring = dyn_smem + warp * STAGES * TILE_BYTES;
    auto issue = [&](int i) { // the warp's tile i into stage i % STAGES
      if (i < ntw) {
        int const base = (tile0 + slice + i * hg) * 32;
        int const page = pages[base / PAGE - page0];
        char *const st = ring + (i % STAGES) * TILE_BYTES;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          int const row = 4 * j + (lane >> 3), c = lane & 7, pos = base + row;
          size_t const off =
              (((size_t)page * PAGE + pos % PAGE) * KVH + head) * D + 8 * c;
          bool const in = pos < sp.cached;
          int const sw = (row * 8 + (c ^ (row & 7))) * 16;
          cp_async16(st + sw, in ? k_cache + off : k_cache, in ? 16 : 0);
          cp_async16(st + TILE_BYTES / 2 + sw,
                     in ? v_cache + off : v_cache,
                     in ? 16 : 0);
        }
      }
      cp_async_commit();
    };
#pragma unroll
    for (int i = 0; i < STAGES; i++) {
      issue(i); // the cache needs no producer: before the wait
    }
    // what q and the new K / V need with no producer, before the wait: the
    // RoPE tables at the new token's position, the bias, the token's page
    // (QKV_BIAS: lane (gq, qd)'s q dims 8 j + 2 qd, + 1 (j < 8: each one's
    // RoPE partner, j +- 4, among them); the new K / V's dims lane, lane + 32)
    int const tok = qo[request], pos_new = sp.cached;
    bool const app = last && slice == 0;
    bf16 const *cosp =
        buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)pos_new * D;
    bf16 const *sinp =
        buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 4>()) + (size_t)pos_new * D;
    __nv_bfloat162 qcs[8], qsn[8], qbs[8];
    float kcs[2], ksn[2], kbs[2], vbs[2];
    int new_page = 0;
    if constexpr (QKV_BIAS) {
      bf16 const *bias =
          buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 9>()) + head * W;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int const d = 8 * j + 2 * qd;
        qcs[j] = *reinterpret_cast<__nv_bfloat162 const *>(cosp + d);
        qsn[j] = *reinterpret_cast<__nv_bfloat162 const *>(sinp + d);
        qbs[j] = *reinterpret_cast<__nv_bfloat162 const *>(bias + gq * D + d);
      }
      if (app) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          int const d = lane + 32 * j;
          kcs[j] = __bfloat162float(cosp[d]);
          ksn[j] = __bfloat162float(sinp[d]);
          kbs[j] = __bfloat162float(bias[QPK * D + d]);
          vbs[j] = __bfloat162float(bias[(QPK + 1) * D + d]);
        }
        new_page = kv_indices[first_page + pos_new / PAGE];
      }
    }
    if (!waited) {
      if (threadIdx.x == 0) {
        sm90_wait_inputs<~0u, IN...>(g);
      }
      __syncthreads();
#ifdef STATIC_TIMING_BUILD
      if (threadIdx.x == 0 && L.stage_stamps) {
        L.stage_stamps[0] = gtime(); // the QKV GEMM's rows ready
      }
#endif
      waited = true;
    }
    if (threadIdx.x == 0) {
      claim[(k + 1) & 1] = atomicAdd(g.cnt + QUEUE, 1u); // the next unit
    }
    // q: query head gq of the KV head, dims [16 qd, 16 qd + 16) (+ the bias,
    // rounded to bf16 as the cache's K; RoPE at the new token's position in
    // fp32, rounded again)
    auto qkv_val = [&](int col) { // column col of the KV head's group, bf16
      size_t const i = (size_t)tok * STRIDE + head * W + col;
      if constexpr (QKV_BIAS) {
        float const *q = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + i;
        float x = q[0];
#pragma unroll
        for (int p = 1; p < KP; p++) {
          x += q[(size_t)p * T * STRIDE];
        }
        float const b = __bfloat162float(
            buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 9>())[head * W + col]);
        return __bfloat162float(__float2bfloat16_rn(x + b));
      } else {
        return __bfloat162float(
            buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 0>())[i]);
      }
    };
    auto rope = [&](int row, int d) { // row: q head (< QPK) or QPK (k)
      float const x = qkv_val(row * D + d);
      float const y = qkv_val(row * D + (d < D / 2 ? d + D / 2 : d - D / 2));
      float const c = __bfloat162float(cosp[d]), s = __bfloat162float(sinp[d]);
      return __bfloat162float(
          __float2bfloat16_rn(d < D / 2 ? x * c - y * s : x * c + y * s));
    };
    uint32_t
        qa[8]; // k16 step j: dims 16 j + 2 qd, + 1 and 16 j + 8 + 2 qd, + 1
    if constexpr (QKV_BIAS) {
      // every load of the lane's q (and new K / V) values in flight at once,
      // the same sums and roundings as qkv_val / rope
      float const *qkv = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) +
                         (size_t)tok * STRIDE + head * W;
      float2 qp[KP][8];
      float kp[KP][2], vp[KP][2];
#pragma unroll
      for (int p = 0; p < KP; p++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          qp[p][j] = *reinterpret_cast<float2 const *>(
              qkv + (size_t)p * T * STRIDE + gq * D + 8 * j + 2 * qd);
        }
        if (app) {
#pragma unroll
          for (int j = 0; j < 2; j++) {
            kp[p][j] = qkv[(size_t)p * T * STRIDE + QPK * D + lane + 32 * j];
            vp[p][j] =
                qkv[(size_t)p * T * STRIDE + (QPK + 1) * D + lane + 32 * j];
          }
        }
      }
      float qx[8][2];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float x = qp[0][j].x, y = qp[0][j].y;
#pragma unroll
        for (int p = 1; p < KP; p++) {
          x += qp[p][j].x, y += qp[p][j].y;
        }
        float2 const b = __bfloat1622float2(qbs[j]);
        qx[j][0] = __bfloat162float(__float2bfloat16_rn(x + b.x));
        qx[j][1] = __bfloat162float(__float2bfloat16_rn(y + b.y));
      }
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float2 const c = __bfloat1622float2(qcs[j]),
                     sn = __bfloat1622float2(qsn[j]);
        int const jp = (j + 4) & 7; // the partner dims (d +- D / 2)
        float r[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float const x = qx[j][e], y = qx[jp][e];
          float const ce = e ? c.y : c.x, se = e ? sn.y : sn.x;
          r[e] = __bfloat162float(
              __float2bfloat16_rn(j < 4 ? x * ce - y * se : x * ce + y * se));
        }
        qa[j] = pack_bf16(r[0], r[1]);
      }
      if (app) {
        float kx[2], vx[2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
          float x = kp[0][j], v = vp[0][j];
#pragma unroll
          for (int p = 1; p < KP; p++) {
            x += kp[p][j], v += vp[p][j];
          }
          kx[j] = __bfloat162float(__float2bfloat16_rn(x + kbs[j]));
          vx[j] = __bfloat162float(__float2bfloat16_rn(v + vbs[j]));
        }
        size_t const off =
            (((size_t)new_page * PAGE + pos_new % PAGE) * KVH + head) * D;
#pragma unroll
        for (int j = 0; j < 2; j++) {
          int const d = lane + 32 * j;
          float const y = kx[1 - j];
          bf16 const kd = __float2bfloat16_rn(__bfloat162float(
              __float2bfloat16_rn(j == 0 ? kx[0] * kcs[0] - y * ksn[0]
                                         : kx[1] * kcs[1] + y * ksn[1])));
          bf16 const vd = __float2bfloat16_rn(vx[j]);
          k_cache[off + d] = kd, v_cache[off + d] = vd;
          new_k[head][d] = kd, new_v[head][d] = vd;
        }
        __syncwarp();
      }
    } else {
#pragma unroll
      for (int p = 0; p < 8; p++) {
        int const d = 8 * p + 2 * qd;
        qa[p] = pack_bf16(rope(gq, d), rope(gq, d + 1));
      }
    }
    if (!QKV_BIAS && last && slice == 0) {
      // the new token: its K and V appended to the cache (no other unit reads
      // that position) and kept for the warp's softmax
      int const page = kv_indices[first_page + pos_new / PAGE];
      size_t const off =
          (((size_t)page * PAGE + pos_new % PAGE) * KVH + head) * D;
#pragma unroll
      for (int j = 0; j < 2; j++) {
        int const d = lane + 32 * j;
        bf16 const kd = __float2bfloat16_rn(rope(QPK, d));
        bf16 const vd = __float2bfloat16_rn(qkv_val((QPK + 1) * D + d));
        k_cache[off + d] = kd, v_cache[off + d] = vd;
        new_k[head][d] = kd, new_v[head][d] = vd;
      }
      __syncwarp();
    }
#ifdef STATIC_TIMING_BUILD
    if (threadIdx.x == 0 && L.stage_stamps) {
      L.stage_stamps[1] = gtime(); // warp 0's q (and new K / V) ready
    }
#endif

    // online softmax over the unit's positions (rows gq: lane (gq, qd) holds
    // dims 8 p + 2 qd, + 1 of n tile p)
    float o[8][4];
#pragma unroll
    for (int p = 0; p < 8; p++) {
      o[p][0] = o[p][1] = o[p][2] = o[p][3] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    auto compute = [&](int tt, char const *st) {
      float s[4][4];
#pragma unroll
      for (int i = 0; i < 4; i++) { // n tile i: rows 8 i ..
        s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int h = 0; h < 2;
             h++) { // chunks 4 h .. 4 h + 3: k16 steps 2 h, + 1
          int const row = 8 * i + (lane & 7), c = 4 * h + (lane >> 3);
          uint32_t kw[4];
          ldmatrix4<false>(kw, st + (row * 8 + (c ^ (row & 7))) * 16);
          mma_bf16(s[i], qa[4 * h], qa[4 * h + 1], kw[0], kw[1]);
          mma_bf16(s[i], qa[4 * h + 2], qa[4 * h + 3], kw[2], kw[3]);
        }
      }
      float mx = m;
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int c = 0; c < 2; c++) {
          int const pos = tt * 32 + 8 * i + 2 * qd + c;
          s[i][c] =
              pos < sp.cached && pos >= sp.lo ? s[i][c] * SCALE : -INFINITY;
          mx = fmaxf(mx, s[i][c]);
        }
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      float const alpha = mx == -INFINITY ? 1.f : ex2(m - mx);
      m = mx;
      l *= alpha;
#pragma unroll
      for (int p = 0; p < 8; p++) {
        o[p][0] *= alpha, o[p][1] *= alpha;
      }
      uint32_t pa[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float const p0 = mx == -INFINITY ? 0.f : ex2(s[i][0] - mx),
                    p1 = mx == -INFINITY ? 0.f : ex2(s[i][1] - mx);
        l += p0 + p1;
        pa[i] = pack_bf16(p0, p1);
      }
      // O += P V: k16 step j over rows 16 j .., n tile p = dims 8 p ..
      // (V's 8 x 8 blocks transposed by ldmatrix)
      char const *vs = st + TILE_BYTES / 2;
#pragma unroll
      for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int p = 0; p < 8; p += 2) {
          int const row = 16 * j + (lane & 15), c = p + (lane >> 4);
          uint32_t vw[4];
          ldmatrix4<true>(vw, vs + (row * 8 + (c ^ (row & 7))) * 16);
          mma_bf16(o[p], pa[2 * j], pa[2 * j + 1], vw[0], vw[1]);
          mma_bf16(o[p + 1], pa[2 * j], pa[2 * j + 1], vw[2], vw[3]);
        }
      }
    };
    for (int i = 0; i < ntw; i++) {
      cp_async_wait<STAGES - 1>();
      __syncwarp();
      compute(tile0 + slice + i * hg, ring + (i % STAGES) * TILE_BYTES);
      __syncwarp(); // the stage read: refill it
      issue(i + STAGES);
    }
    cp_async_wait<0>();
    __syncwarp();
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (last && slice == 0) { // the new token, query head gq's score over the
                              // lane's dims
      float sc = 0.f;
#pragma unroll
      for (int p = 0; p < 8; p++) {
        float2 const q = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162 const *>(&qa[p]));
        float2 const k =
            __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(
                &new_k[head][8 * p + 2 * qd]));
        sc += q.x * k.x + q.y * k.y;
      }
      sc += __shfl_xor_sync(0xffffffffu, sc, 1);
      sc += __shfl_xor_sync(0xffffffffu, sc, 2);
      sc *= SCALE;
      float const mx = fmaxf(m, sc), a = m == -INFINITY ? 0.f : ex2(m - mx),
                  pn = ex2(sc - mx);
      l = l * a + pn;
#pragma unroll
      for (int p = 0; p < 8; p++) {
        float2 const v =
            __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(
                &new_v[head][8 * p + 2 * qd]));
        o[p][0] = o[p][0] * a + pn * v.x, o[p][1] = o[p][1] * a + pn * v.y;
      }
      m = mx;
    }
    int const qh = head * QPK + gq;
    if constexpr (DIRECT) {
      if (hg > 1) {
        // the head's slices through the warps' rings (their loads are done):
        // lane (gq, qd)'s m, l and o, merged into slice 0's (log2 units)
        float *red = reinterpret_cast<float *>(ring) + lane * 18;
        red[0] = m, red[1] = l;
#pragma unroll
        for (int p = 0; p < 8; p++) {
          red[2 + 2 * p] = o[p][0], red[3 + 2 * p] = o[p][1];
        }
        __syncthreads();
        if (slice == 0) {
          float M = m;
          for (int sl = 1; sl < hg; sl++) {
            M = fmaxf(M,
                      reinterpret_cast<float const *>(
                          dyn_smem +
                          (warp + sl * hpu) * STAGES * TILE_BYTES)[lane * 18]);
          }
          float const a0 = m == -INFINITY ? 0.f : ex2(m - M);
          l *= a0;
#pragma unroll
          for (int p = 0; p < 8; p++) {
            o[p][0] *= a0, o[p][1] *= a0;
          }
          for (int sl = 1; sl < hg; sl++) {
            float const *r2 =
                reinterpret_cast<float const *>(
                    dyn_smem + (warp + sl * hpu) * STAGES * TILE_BYTES) +
                lane * 18;
            float const a = r2[0] == -INFINITY ? 0.f : ex2(r2[0] - M);
            l += a * r2[1];
#pragma unroll
            for (int p = 0; p < 8; p++) {
              o[p][0] += a * r2[2 + 2 * p], o[p][1] += a * r2[3 + 2 * p];
            }
          }
          m = M;
        }
      }
    }
    if constexpr (DIRECT) {
      if (slice == 0) { // the one split: out = 2^(L - M) O / (2^(L - M) +
                        // 2^(sink log2 e - M)), M = max(L, sink log2 e)
        float const L = m + __log2f(l);
        float const sink = __bfloat162float(buf_at<bf16 const>(
                               g, slot_at<BUF_SLOTS, 12>())[qh]) *
                           1.4426950408889634f;
        float const M = fmaxf(L, sink), w = exp2f(L - M),
                    sc = w / (w + exp2f(sink - M)) / l;
        __nv_bfloat162 *out =
            buf_at<__nv_bfloat162>(g, slot_at<BUF_SLOTS, 13>()) +
            ((size_t)tok * QH + qh) * (D / 2);
#pragma unroll
        for (int p = 0; p < 8; p++) {
          out[4 * p + qd] = __floats2bfloat162_rn(o[p][0] * sc, o[p][1] * sc);
        }
      }
    } else {
#pragma unroll
      for (int p = 0; p < 8; p++) {
        *reinterpret_cast<float2 *>(parts + ((size_t)u * QH + qh) * D + 8 * p +
                                    2 * qd) =
            make_float2(o[p][0] / l, o[p][1] / l);
      }
      if (qd == 0) {
        lse_out[(size_t)u * QH + qh] = m + __log2f(l);
      }
    }
    __syncthreads(); // shared memory free; claim[] written
  }
  if (!waited) { // no unit: the node still ends after its inputs
    if (threadIdx.x == 0) {
      sm90_wait_inputs<~0u, IN...>(g);
    }
    __syncthreads();
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_attention_split,
               SM90_WARPS * 3 * 2 * 32 * PARAMS::v[2] * 2) // the warps' rings

} // namespace static_mk
