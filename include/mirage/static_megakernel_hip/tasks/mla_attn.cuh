// tasks/mla_attn.cuh (HIP, gfx950) -- K3 MLA's absorbed attention for one head
// over the framework's paged KV cache (math: mpk-kernels tests/mla_reference.py
// core(): SGLang's absorb path). Task (x, y) of grid (heads, NS, 1) = head x, KV
// split y:
//   query  = mla_absorb's output of this head ([T][heads][KV + RP] bf16: q_abs,
//            then q_pe; 0xFF-prefilled, polled)
//   rows j of the request, j < seq_k = seq_lens[0] + T (SGLang: the rows before
//   the step, int64): j < seq_lens[0] at table[req x info[3] + j / PS] x PS + j %
//   PS of the cache [rows][KV + RP] bf16 (table: SGLang's req_to_token, int32,
//   page size 1; req = req_pool_indices[0], int64), this step's rows at
//   out_loc[j - seq_lens[0]] (int64; written by mla_norm: read after its count);
//   token t sees j <= seq_k - T + t;
//   s = scale (query . K[j]); online softmax over chunks of 64 rows (e = exp(s -
//   max) rounded to bf16 for the product, the sum of the unrounded e); attn =
//   bf16(sum e K[:KV] / sum e)
//   -> attn [T][heads x KV] bf16 (0xFF-prefilled: mla_vgate polls it)
// The first chunk's rows of earlier steps go into LDS as the task starts (no
// input needed); then the query, then (a chunk with this step's rows) the wait
// for mla_norm's count.
// KV splits (NS > 1): task (h, s) takes split s of the request's chunks (act =
// min(NS, chunks) splits; s >= act: nothing); s > 0 writes its partial (the
// unnormalized sum e K[:KV] [T][KV], then m, l [T]; fp32, device scope) and
// counts on head h's counter; s = 0 runs its share, waits for act - 1 counts,
// adds the partials rescaled to the largest m and writes attn. (The splits of a
// head run at the same time: the node's concurrent axis.)
#pragma once
#include "../core.cuh"
#include "mla_norm.cuh" // mla_st4_dev
#include "mla_absorb.cuh"
#include "mla_attn_grp.cuh"

namespace static_mk {

namespace mla_a {
constexpr int CH = 64; // KV rows per chunk
constexpr int DPAD = 8; // the chunk's LDS row stride: D + 8 bf16 (rows 0 .. 31 of a half-wave in 16 banks, not 1)
}
__device__ __forceinline__ void mla_st8_dev(void *p, float a, float b) {
  v2u const w = {__float_as_uint(a), __float_as_uint(b)};
  asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ float2 mla_ld8_dev(void const *p) {
  v2u w;
  asm volatile("global_load_dwordx2 %0, %1, off sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return make_float2(__uint_as_float(w.x), __uint_as_float(w.y));
}

template <int HEADS, int KV, int RP, int PS, int NS>
__device__ __forceinline__ void mla_attn_task(KernelLocals &L,
                                              __hip_bfloat16 const *query,   // [T][HEADS][KV + RP]
                                              __hip_bfloat16 const *cache,   // [rows][KV + RP]
                                              int const *table,
                                              int const *info,
                                              long long const *seq_lens,
                                              long long const *req_idx,
                                              long long const *out_loc,
                                              uint32_t const *norm_cnt,
                                              uint32_t norm_need,  // mla_norm's tasks
                                              float scale,
                                              __hip_bfloat16 *attn,          // [T][HEADS x KV]
                                              float *part,                   // NS > 1: [HEADS][NS][T KV + 2 T]
                                              uint32_t *scnt,                // NS > 1: [HEADS] split counts
                                              int h,
                                              int s STAGE_STAMP_PARAM) {
  using namespace mla_a;
  constexpr int D = KV + RP;
  constexpr int DL = D + mla_a::DPAD; // the chunk's LDS row stride
  static_assert(KV == 512 && RP == 64 && T == 8, "mla_attn: K3's shapes (thread maps below)");
  int const tid = task_tid_all();
  int const Lp = (int)seq_lens[0], seq_k = Lp + T;
  int const *const rt = table + (size_t)req_idx[0] * (size_t)info[3];
  int const nch = (seq_k + mla_a::CH - 1) / mla_a::CH, act = NS < nch ? NS : nch;
  if (s >= act) {
    return; // no rows for this split
  }
  int const c_lo = s * nch / act, c_hi = (s + 1) * nch / act;
  // LDS: the K chunk [CH][DL] bf16, its row indices [CH], the query [T][D] bf16,
  // scores / e [T][CH], per token m, l, the chunk's correction
  char *const sm = L.sm;
  __hip_bfloat16 *const kc = reinterpret_cast<__hip_bfloat16 *>(sm);
  long long *const rowi = reinterpret_cast<long long *>(sm + mla_a::CH * DL * 2); // [CH] cache rows
  __hip_bfloat16 *const qb16 = reinterpret_cast<__hip_bfloat16 *>(sm + mla_a::CH * DL * 2 + mla_a::CH * 8); // [T][D]
  __hip_bfloat16 *const kT = qb16 + T * D;     // [KV][CH]: the chunk's latent columns transposed (P.V's A operand)
  __hip_bfloat16 *const pb16 = kT + KV * CH;   // [T][CH]: e in bf16 (P.V's B operand)
  float *const sc = reinterpret_cast<float *>(pb16 + T * CH);
  float *const mm = sc + T * CH, *const ll = mm + T, *const cr = ll + T;
  int const lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  __syncthreads(); // the LDS may still be read by the previous task
  if (tid < T) {
    mm[tid] = -3.0e38f;
    ll[tid] = 0.f;
  }
  // the chunk at j0: its row indices, then its rows of earlier steps (plain
  // loads, all in flight; written by earlier steps)
  auto load_old = [&](int j0, int nr) {
    if (tid < CH) {
      int const j = j0 + tid;
      rowi[tid] = j < Lp ? (long long)rt[j / PS] * PS + j % PS : (j < seq_k ? out_loc[j - Lp] : 0);
    }
    __syncthreads();
    int const nold = max(0, min(nr, Lp - j0));
#pragma unroll 4
    for (int v = tid; v < nold * (D / 8); v += 256) {
      int const r = v / (D / 8), c = v % (D / 8);
      *reinterpret_cast<uint4 *>(kc + r * DL + 8 * c) =
          *reinterpret_cast<uint4 const *>(cache + (size_t)rowi[r] * D + 8 * c);
    }
    return nold;
  };
  // its rows of this step (mla_norm's device-scope stores on other CUs; after
  // its count): relaxed agent-scope atomic loads, all in flight
  auto load_new = [&](int nold, int nr) {
    constexpr int NNEW = (T * (D / 8) + 255) / 256;
    int const nnew = (nr - nold) * (D / 8);
    unsigned long long w[NNEW][2] = {};
#pragma unroll
    for (int i = 0; i < NNEW; i++) {
      int const v = tid + 256 * i;
      if (v < nnew) {
        unsigned long long const *const src = reinterpret_cast<unsigned long long const *>(
            cache + (size_t)rowi[nold + v / (D / 8)] * D + 8 * (v % (D / 8)));
        w[i][0] = __hip_atomic_load(src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        w[i][1] = __hip_atomic_load(src + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
#pragma unroll
    for (int i = 0; i < NNEW; i++) {
      int const v = tid + 256 * i;
      if (v < nnew) {
        unsigned long long *const dst =
            reinterpret_cast<unsigned long long *>(kc + (nold + v / (D / 8)) * DL + 8 * (v % (D / 8)));
        dst[0] = w[i][0];
        dst[1] = w[i][1];
      }
    }
  };
  // 1. the first chunk's rows of earlier steps (no input needed)
  int const first = c_lo * CH;
  int nold_first = load_old(first, min(CH, seq_k - first));
  // 2. the query of this head (polled: T x D / 8 = 576 vectors, 3 rounds of 256
  // in flight together)
  {
    constexpr int NQV = T * D / 8, NQR = (NQV + 255) / 256;
    uint4 qv[NQR];
    for (;;) {
      bool all = true;
#pragma unroll
      for (int i = 0; i < NQR; i++) {
        int const v = tid + 256 * i;
        if (v < NQV) {
          int const t = v / (D / 8), c = 8 * (v % (D / 8));
          qv[i] = ld16_at<AUX_DEV>(query, (uint32_t)((((size_t)t * HEADS + h) * D + c) * 2));
          all = all && valid16w(qv[i]);
        }
      }
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int i = 0; i < NQR; i++) {
      int const v = tid + 256 * i;
      if (v < NQV) {
        int const t = v / (D / 8), c = 8 * (v % (D / 8));
        *reinterpret_cast<uint4 *>(qb16 + t * D + c) = qv[i];
      }
    }
  }
  if (tid == 0) { // (timing build: stage stamp 0 = the query landed)
    STAGE_STAMP_AT(0)
  }
  // attn^T by MFMA: wave w holds latent rows l = 128 w + 16 i + 4 (l / 16) + r
  // (register r of tile i) of token l % 16 (tokens >= T: zeros)
  v4f acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
  }
  for (int j0 = first; j0 < c_hi * CH && j0 < seq_k; j0 += CH) {
    int const nr = min(CH, seq_k - j0);
    int const nold = j0 == first ? nold_first : load_old(j0, nr);
    if (nold < nr) { // this step's rows: after mla_norm counted T
      if (tid == 0) {
        cnt_wait_in(norm_cnt, norm_need);
      }
      __syncthreads();
      load_new(nold, nr);
    }
    __syncthreads();
    if (tid == 0 && j0 == first) { // (stage stamp 1: the first chunk in LDS)
      STAGE_STAMP_AT(1)
    }
    // the chunk's latent columns transposed into kT (item = 8 columns x 8 rows;
    // a lane's 8 rows are consecutive in a kT row: conflict-free 16-B writes);
    // rows >= nr as zeros (their e is 0, their stale values must not be NaN)
#pragma unroll
    for (int it = tid; it < (KV / 8) * (CH / 8); it += 256) {
      int const jg = it / (KV / 8), cg = it % (KV / 8);
      uint4 rv[8];
#pragma unroll
      for (int jj = 0; jj < 8; jj++) {
        int const j = 8 * jg + jj;
        rv[jj] = j < nr ? *reinterpret_cast<uint4 const *>(kc + j * DL + 8 * cg) : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        uint32_t w4[4];
#pragma unroll
        for (int jj = 0; jj < 8; jj += 2) {
          uint32_t const a = (&rv[jj].x)[e / 2], b = (&rv[jj + 1].x)[e / 2];
          uint32_t const lo = (e & 1) ? (a >> 16) : (a & 0xFFFFu), hi = (e & 1) ? (b >> 16) : (b & 0xFFFFu);
          w4[jj / 2] = lo | (hi << 16);
        }
        int const d = 8 * cg + e;
        *reinterpret_cast<uint4 *>(kT + d * CH + 8 * mla_kt_chunk(d, jg)) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
      }
    }
    // b. scores S[t][j] = scale (query[t] . row j) by MFMA (wave w: chunk rows
    // 16 w .. + 15; A: the chunk's rows, K = the D columns; B: the tokens' query)
    {
      v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 6
      for (int st = 0; st < D / 32; st++) {
        v4f b = {0.f, 0.f, 0.f, 0.f};
        if (tok < T) {
          b = *reinterpret_cast<v4f const *>(qb16 + tok * D + 32 * st + 8 * grp);
        }
        v4f const a = *reinterpret_cast<v4f const *>(kc + (16 * wave + tok) * DL + 32 * st + 8 * grp);
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b), acc, 0,
                                                      0, 0);
      }
      if (tok < T) {
        int const lim = Lp + tok; // the last row token tok sees
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const j = 16 * wave + 4 * grp + r;
          sc[tok * CH + j] = (j < nr && j0 + j <= lim) ? acc[r] * scale : -3.0e38f;
        }
      }
    }
    __syncthreads();
    // c. per token (wave w: tokens 2 w, 2 w + 1; lane = row): the online softmax
    {
      int const w = tid >> 6, lane = tid & 63;
#pragma unroll
      for (int k = 0; k < 2; k++) {
        int const t = 2 * w + k;
        float const s = sc[t * CH + lane];
        float mx = s;
        mx = wave_max64(mx);
        float const mo = mm[t], mn = fmaxf(mo, mx);
        float const e = s > -1.0e38f ? __expf(s - mn) : 0.f;
        float se = e;
        se = wave_sum64(se);
        pb16[t * CH + lane] = __float2bfloat16(e); // (bf16 for the product)
        if (lane == 0) {
          float const corr = mo > -1.0e38f ? __expf(mo - mn) : 0.f;
          cr[t] = corr;
          ll[t] = ll[t] * corr + se;
          mm[t] = mn;
        }
      }
    }
    __syncthreads();
    // d. attn^T += K[:, :KV]^T . e^T by MFMA (A: kT rows l, K = the chunk's rows j;
    // B: the tokens' e), after the old sums rescaled by the chunk's correction
    {
      float const c = tok < T ? cr[tok] : 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] *= c;
      }
#pragma unroll
      for (int st = 0; st < CH / 32; st++) {
        v4f b = {0.f, 0.f, 0.f, 0.f};
        if (tok < T) {
          b = *reinterpret_cast<v4f const *>(pb16 + tok * CH + 32 * st + 8 * grp);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int const d = 128 * wave + 16 * i + tok;
          v4f const a = *reinterpret_cast<v4f const *>(kT + d * CH + 8 * mla_kt_chunk(d, 4 * st + grp));
          acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b),
                                                           acc[i], 0, 0, 0);
        }
      }
    }
    __syncthreads(); // the chunk's LDS reused by the next one
  }
  int const lrow = 128 * wave + 4 * grp; // + 16 i: this lane's first latent row of tile i
  if constexpr (NS > 1) {
    constexpr int PW = T * KV + 2 * T;
    if (s > 0) { // this split's partial, then one count on head h's counter
      float *const pp = part + ((size_t)h * NS + s) * PW;
      if (tok < T) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          st16_dev_acc(pp + tok * KV + lrow + 16 * i,
                   make_uint4(__float_as_uint(acc[i][0]), __float_as_uint(acc[i][1]), __float_as_uint(acc[i][2]),
                              __float_as_uint(acc[i][3])));
        }
      }
      if (tid < T) {
        mla_st8_dev(pp + T * KV + 2 * tid, mm[tid], ll[tid]);
      }
      task_post(scnt + h, 1);
      return;
    }
    if (tid == 0) {
      cnt_wait_in(scnt + h, act - 1);
    }
    __syncthreads();
    // the partials rescaled to the largest m of each token: M, the new sum and
    // split 0's own factor per token (threads t < T), then every lane its rows
    if (tid < T) {
      float M = mm[tid];
      for (int q = 1; q < act; q++) {
        M = fmaxf(M, mla_ld8_dev(part + ((size_t)h * NS + q) * PW + T * KV + 2 * tid).x);
      }
      float const f0 = mm[tid] > -1.0e38f ? __expf(mm[tid] - M) : 0.f;
      float l = ll[tid] * f0;
      for (int q = 1; q < act; q++) {
        float2 const ml = mla_ld8_dev(part + ((size_t)h * NS + q) * PW + T * KV + 2 * tid);
        l += ml.x > -1.0e38f ? ml.y * __expf(ml.x - M) : 0.f;
      }
      cr[tid] = f0;
      mm[tid] = M;
      ll[tid] = l;
    }
    __syncthreads();
    if (tok < T) {
      float const c0 = cr[tok];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] *= c0;
      }
      for (int q = 1; q < act; q++) {
        float const *const pp = part + ((size_t)h * NS + q) * PW;
        float const mq = mla_ld8_dev(pp + T * KV + 2 * tok).x;
        float const f = mq > -1.0e38f ? __expf(mq - mm[tok]) : 0.f;
        uint4 vq[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          vq[i] = ld16_at<AUX_DEV>(pp, (uint32_t)((tok * KV + lrow + 16 * i) * 4));
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
          acc[i][0] = fmaf(__uint_as_float(vq[i].x), f, acc[i][0]);
          acc[i][1] = fmaf(__uint_as_float(vq[i].y), f, acc[i][1]);
          acc[i][2] = fmaf(__uint_as_float(vq[i].z), f, acc[i][2]);
          acc[i][3] = fmaf(__uint_as_float(vq[i].w), f, acc[i][3]);
        }
      }
    }
  }
  __syncthreads(); // the chunk buffer (kc) free: the head's output staged there, [T][KV] bf16
  if (tok < T) {
    float const inv = 1.f / ll[tok];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      *reinterpret_cast<uint2 *>(kc + tok * KV + lrow + 16 * i) =
          make_uint2(pack_bf16x2(acc[i][0] * inv, acc[i][1] * inv), pack_bf16x2(acc[i][2] * inv, acc[i][3] * inv));
    }
  }
  __syncthreads();
  for (int v = tid; v < T * KV / 8; v += 256) { // each token's KV x 2 B of the head in 16-B stores
    int const t = v / (KV / 8), c = 8 * (v % (KV / 8));
    st16_dev(attn + (size_t)t * HEADS * KV + h * KV + c, *reinterpret_cast<uint4 const *>(kc + t * KV + c));
  }
  __syncthreads();
}

// task (x, y) = head x, KV split y; Q = mla_absorb (its buffer: the query), N =
// mla_norm (its counter); params = [HEADS, KV, RP, PS, scale bits]; buffer slots
// = the KV cache, the token table, info, seq_lens, req_idx, out_loc (KV splits:
// the partials' scratch); counters (KV splits): one per head
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Q,
          class N,
          class... IN>
__device__ __forceinline__ void run_mla_attn(Maps const &,
                                             G const &g,
                                             KernelLocals &L,
                                             StaticTask const &tk) {
  static_assert(Q::buf >= 0, "mla_attn: mla_absorb's output (the query)");
  static_assert(N::counter >= 0, "mla_attn: mla_norm's counter");
  constexpr int HG = PARAMS::v[5]; // heads per task (1: mla_attn_task, mla_g::HG: mla_attn_grp_task)
  static_assert(HG == 1 || HG == mla_g::HG, "mla_attn: 1 head per task or mla_g::HG");
  static_assert(SELF::buf >= 0 && SELF::x * HG == PARAMS::v[0], "mla_attn: one task per head group, its output");
  static_assert(SELF::y == 1 || SELF::counter >= 0, "mla_attn: KV splits count on the node's counters");
  float *part = nullptr;
  uint32_t *scnt = nullptr;
  if constexpr (SELF::y > 1) {
    part = buf_at<float>(g, slot_at<BUF_SLOTS, 6>());
    scnt = cnt_of(g) + SELF::counter;
  }
  if constexpr (HG > 1) {
    mla_attn_grp_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], SELF::y>(
        L,
        buf_at<__hip_bfloat16 const>(g, Q::buf),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
        buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
        buf_at<int const>(g, slot_at<BUF_SLOTS, 2>()),
        buf_at<long long const>(g, slot_at<BUF_SLOTS, 3>()),
        buf_at<long long const>(g, slot_at<BUF_SLOTS, 4>()),
        buf_at<long long const>(g, slot_at<BUF_SLOTS, 5>()),
        cnt_of(g) + N::counter,
        (uint32_t)(N::x * N::y * N::z),
        __int_as_float(PARAMS::v[4]),
        buf_at<__hip_bfloat16>(g, SELF::buf),
        part,
        scnt,
        tk.x,
        tk.y STAGE_STAMP_ARG(L.stage_stamps));
    return;
  }
  mla_attn_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], SELF::y>(
      L,
      buf_at<__hip_bfloat16 const>(g, Q::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 5>()),
      cnt_of(g) + N::counter,
      (uint32_t)(N::x * N::y * N::z),
      __int_as_float(PARAMS::v[4]),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      part,
      scnt,
      tk.x,
      tk.y STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_attn() {
  if constexpr (PARAMS::v[5] > 1) {
    return smem_mla_attn_grp<PARAMS::v[1], PARAMS::v[2], SELF::y>();
  }
  // the K chunk and its row indices, the query, kT, e (bf16); scores, m / l / corr
  return mla_a::CH * (PARAMS::v[1] + PARAMS::v[2] + mla_a::DPAD) * 2 + mla_a::CH * 8 +
         T * (PARAMS::v[1] + PARAMS::v[2]) * 2 + PARAMS::v[1] * mla_a::CH * 2 + T * mla_a::CH * 2 +
         (T * mla_a::CH + 3 * T) * 4;
}

} // namespace static_mk
