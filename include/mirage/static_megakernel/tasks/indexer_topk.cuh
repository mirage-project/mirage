// tasks/indexer_topk.cuh -- DeepSeek-V4.1's indexer top-k (model.py L585-597,
// Indexer.forward, and select_candidate_blocks L600-626): task t = token t,
// grid (T, 1, 1). Of token t's n = (p + 1) / R scores (indexer_score's, bf16),
// the K = min(KTOP, n) largest, in position order -> compress_idx[t][0 .. K)
// (positions, without the window's offset: sparse_attn's compress_idx), -1
// after. The official's topk(sorted=False) then sort: the same set where
// scores do not tie at the K-th; among scores equal to the K-th, this takes the
// lowest positions (torch's choice is unspecified). CSRC (L20, the candidate
// source): also the candidate blocks from the same scores: block b's score =
// the max over positions 8 b .. 8 b + 7 (< n), the block holding position n -
// 1 forced to +inf, the min(TOPB, blocks) largest kept (ties: the lowest
// blocks) -> a byte per block [T][CAP / 8] (1 = kept), read by the Reindex
// layers' indexer_score (later launches: computed after the release). The
// release, after compress_idx: the node's counter t += 1.
// The selection: bf16 scores as 16-bit keys ordered as the values (sign bit
// flipped, negatives inverted), a radix select of 2 passes of 8 bits (shared
// histograms), then a position-ordered compaction (each thread a run of
// positions, a block scan of its counts). warm = false (this node's first task
// on the SM): first a dry pass, the select and the compaction on stand-in
// keys, no global loads or stores, while the scores are on their way; one copy
// of the code for both passes.
#pragma once
#include "../core.cuh"

namespace static_mk {

// timing build: the task's 2 stage stamps at step boundaries IT_STAMP0 and
// IT_STAMP1 (thread 0, after the barrier ending the step): 1 the scores ready
// (the wait), 2 the keys in shared memory, 3 the select's first pass, 4 its
// second, 7 the compaction's counts and their sums, 5 the compaction (the
// list in shared memory)
#ifndef IT_STAMP0
#define IT_STAMP0 1
#endif
#ifndef IT_STAMP1
#define IT_STAMP1 5
#endif
#define IT_MARK(p)                                                             \
  if (threadIdx.x == 0) {                                                      \
    if ((p) == IT_STAMP0) {                                                    \
      STAGE_STAMP_AT(0)                                                        \
    }                                                                          \
    if ((p) == IT_STAMP1) {                                                    \
      STAGE_STAMP_AT(1)                                                        \
    }                                                                          \
  }

constexpr int IT_KEY_PINF = 0xFFFF; // above every bf16 key (a forced block)
// shared memory: keys [CAP] and block keys [CAP / 8] (16-bit), then (16-B
// aligned) the histogram (256 + 2 ints), the scan's 16 ints and the list
// (KTOP ints)
__host__ __device__ constexpr int it_hist_off(int cap) {
  return ((cap + cap / 8) * 2 + 15) / 16 * 16;
}
__host__ __device__ constexpr int it_smem(int cap, int ktop) {
  return it_hist_off(cap) + (260 + 16 + ktop) * 4;
}
__device__ __forceinline__ uint32_t it_key(uint32_t b) { // bf16 bits -> key
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}
constexpr uint32_t IT_KEY_NINF = 0x007Fu; // it_key(-inf = 0xFF80)

// exclusive prefix sums over the CTA's 256 threads of (a, b); red: 16 ints
// of shared memory. Every thread calls it (barriers)
__device__ __forceinline__ void
    it_scan2(int a, int b, int &pa, int &pb, int *red) {
  int const w = threadIdx.x >> 5, l = threadIdx.x & 31;
  int ia = a, ib = b;
  for (int d = 1; d < 32; d <<= 1) {
    int const xa = __shfl_up_sync(0xffffffffu, ia, d),
              xb = __shfl_up_sync(0xffffffffu, ib, d);
    if (l >= d) {
      ia += xa;
      ib += xb;
    }
  }
  if (l == 31) {
    red[w] = ia;
    red[8 + w] = ib;
  }
  __syncthreads();
  int oa = 0, ob = 0;
  for (int i = 0; i < w; i++) {
    oa += red[i];
    ob += red[8 + i];
  }
  pa = oa + ia - a;
  pb = ob + ib - b;
  __syncthreads();
}

// the k-th largest (1 <= k <= n) of the n keys in shared memory: tk the
// threshold key, gt the keys above it (k - gt of the keys equal to tk make k).
// hist: 256 + 2 ints of shared memory. Every thread calls it (barriers)
__device__ __forceinline__ void it_select(uint16_t const *keys,
                                          int n,
                                          int k,
                                          int *hist,
                                          uint32_t &tk,
                                          int &gt STAGE_STAMP_PARAM) {
  int const tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  uint32_t prefix = 0;
  int need = k;
  gt = 0;
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    hist[tid] = 0; // 256 threads, 256 bins
    __syncthreads();
    for (int i = tid; i < n; i += 256) {
      uint32_t const key = keys[i];
      if (pass == 0) {
        atomicAdd(&hist[key >> 8], 1);
      } else if ((key >> 8) == prefix) {
        atomicAdd(&hist[key & 0xFFu], 1);
      }
    }
    __syncthreads();
    if (w == 0) { // lane l: bins 255 - 8 l .. 248 - 8 l, from the top
      int cnt[8], s = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        cnt[j] = hist[255 - 8 * l - j];
        s += cnt[j];
      }
      int incl = s;
      for (int d = 1; d < 32; d <<= 1) {
        int const x = __shfl_up_sync(0xffffffffu, incl, d);
        if (l >= d) {
          incl += x;
        }
      }
      int above = incl - s;
      if (above < need && need <= incl) { // one lane
#pragma unroll
        for (int j = 0; j < 8; j++) {
          if (above + cnt[j] >= need) {
            hist[256] = 255 - 8 * l - j;
            hist[257] = above;
            break;
          }
          above += cnt[j];
        }
      }
    }
    __syncthreads();
    uint32_t const bin = hist[256];
    int const a = hist[257];
    __syncthreads();
    gt += a;
    need -= a;
    prefix = pass == 0 ? bin : (prefix << 8) | bin;
    if (pass == 0) {
      IT_MARK(3)
    }
  }
  tk = prefix;
}

// position-ordered compaction of the selection: the keys above tk, and the
// first `eq` (by position) of those equal to tk. Thread: a run of positions;
// emit(i, j): position i is the j-th selected. Every thread calls it.
template <class EMIT>
__device__ __forceinline__ void it_compact(uint16_t const *keys,
                                           int n,
                                           uint32_t tk,
                                           int eq,
                                           int *red,
                                           EMIT emit STAGE_STAMP_PARAM) {
  int const per = (n + 255) / 256, lo = min(n, (int)threadIdx.x * per),
            hi = min(n, lo + per);
  int ngt = 0, neq = 0;
  for (int i = lo; i < hi; i++) {
    uint32_t const key = keys[i];
    ngt += key > tk;
    neq += key == tk;
  }
  int pg, pe;
  it_scan2(ngt, neq, pg, pe, red);
  IT_MARK(7)
  int o = pg + min(pe, eq), es = pe;
  for (int i = lo; i < hi; i++) {
    uint32_t const key = keys[i];
    if (key > tk || (key == tk && es++ < eq)) {
      emit(i, o++);
    }
  }
}

template <int CAP, int KTOP, int R, bool CSRC, int TOPB>
__device__ __forceinline__ void indexer_topk_task(__nv_bfloat16 const *scores,
                                                  int const *pos,
                                                  int *idx,
                                                  unsigned char *cand,
                                                  uint32_t const *s_cnt,
                                                  uint32_t s_target,
                                                  uint32_t *done_cnt,
                                                  int t,
                                                  char *sm,
                                                  bool warm STAGE_STAMP_PARAM) {
  static_assert(CAP % 8 == 0 && KTOP <= CAP, "indexer_topk: whole blocks of 8");
  int const tid = threadIdx.x, p = pos[t], n = min((p + 1) / R, CAP),
            k = min(KTOP, n);
  uint16_t *const keys = reinterpret_cast<uint16_t *>(sm);
  uint16_t *const bkeys = keys + CAP; // CAP / 8 blocks
  int *const hist = reinterpret_cast<int *>(sm + it_hist_off(CAP)); // 258
  int *const red = hist + 260;                                      // 16
  int *const sel = red + 16;                                        // KTOP
  __syncthreads(); // the shared memory may be the ring's of a GEMM before
  int *const out = idx + (size_t)t * KTOP;
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    if (real) {
      if (tid == 0) {
        cnt_wait(s_cnt + t, s_target); // indexer_score's chunk tasks of token t
      }
      __syncthreads();
      IT_MARK(1)
    }
    { // the scores as keys (relaxed: written this launch by other SMs); the
      // dry pass: stand-ins, a key per position
      unsigned char const *sb =
          reinterpret_cast<unsigned char const *>(scores + (size_t)t * CAP);
      for (int i = 8 * tid; i < n; i += 8 * 256) {
        uint4 const v = real ? ld16_relaxed(sb + 2 * i)
                             : make_uint4(i * 0x9E3779B9u,
                                          i * 0x85EBCA6Bu,
                                          i * 0xC2B2AE35u,
                                          i * 0x27D4EB2Fu);
        uint32_t const wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 8; j++) {
          keys[i + j] =
              (uint16_t)it_key((wv[j / 2] >> (16 * (j & 1))) & 0xFFFFu);
        }
      }
    }
    __syncthreads();
    if (real) {
      IT_MARK(2)
    }
    if (k > 0) {
      uint32_t tk;
      int gt;
      it_select(
          keys, n, k, hist, tk, gt STAGE_STAMP_ARG(real ? stamp : nullptr));
      if (real) {
        IT_MARK(4)
      }
      it_compact(keys, n, tk, k - gt, red, [&](int i, int j) {
        if (real) {
          sel[j] = i;
        }
      } STAGE_STAMP_ARG(real ? stamp : nullptr));
    }
    __syncthreads(); // the keys free for the real pass's
  }
#ifdef STATIC_TIMING_BUILD
  IT_MARK(5)
#endif
  for (int j = tid; j < KTOP; j += 256) { // the list, coalesced; -1 after
    out[j] = j < k ? sel[j] : -1;
  }
  __syncthreads();
  if (tid == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
  if constexpr (CSRC) { // after the release: only later launches read them
    int const nb = (n + 7) / 8, kb = min(TOPB, nb);
    for (int b = tid; b < nb; b += 256) {
      uint32_t m = 0;
      for (int i = 8 * b; i < min(8 * b + 8, n); i++) {
        m = max(m, (uint32_t)keys[i]);
      }
      bkeys[b] = (uint16_t)(b == (n - 1) / 8 ? IT_KEY_PINF : m);
    }
    __syncthreads();
    unsigned char *const cb = cand + (size_t)t * (CAP / 8);
    for (int b = tid; b < CAP / 8; b += 256) {
      cb[b] = 0;
    }
    __syncthreads();
    if (kb > 0) {
      uint32_t tk;
      int gt;
      it_select(bkeys, nb, kb, hist, tk, gt);
      it_compact(bkeys, nb, tk, kb - gt, red, [&](int b, int) {
        cb[b] = bkeys[b] != IT_KEY_NINF; // a block of -inf: not reachable
      });
    }
    __syncthreads();
  }
}

// indexer_topk task t = token t, grid (T, 1, 1); SC: indexer_score's node (its
// buffer scores [T][CAP] bf16, a counter per token of its SC::y chunk tasks).
// PARAMS {CAP, KTOP, R, CSRC, TOPB}; BUF_SLOTS {positions ([T] int32),
// compress_idx ([T][KTOP] int32: the node's output, a graph tensor), with CSRC
// the candidate blocks ([T][CAP / 8] uint8, a graph tensor)}; the node's
// counters: one per token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SC,
          class... REST>
__device__ __forceinline__ void run_indexer_topk(Maps const &,
                                                 G const &g,
                                                 KernelLocals &L,
                                                 StaticTask const &tk,
                                                 bool warm) {
  constexpr int CAP = PARAMS::v[0], KTOP = PARAMS::v[1], R = PARAMS::v[2],
                TOPB = PARAMS::v[4];
  constexpr bool CSRC = PARAMS::v[3] != 0;
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "indexer_topk: one task per token");
  static_assert(SELF::buf < 0 && SELF::counter >= 0,
                "indexer_topk: its output a graph tensor (compress_idx), a "
                "counter per token");
  static_assert(SC::buf >= 0 && SC::counter >= 0 && SC::x == T &&
                    SC::v[2] == CAP,
                "indexer_topk: the scores are indexer_score's (a counter per "
                "token)");
  unsigned char *cand = nullptr;
  if constexpr (CSRC) {
    cand = buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 2>());
  }
  indexer_topk_task<CAP, KTOP, R, CSRC, TOPB>(
      buf_at<__nv_bfloat16 const>(g, SC::buf),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int>(g, slot_at<BUF_SLOTS, 1>()),
      cand,
      g.cnt + SC::counter,
      SC::y,
      g.cnt + SELF::counter,
      tk.x,
      L.sm,
      warm STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_indexer_topk() {
  return it_smem(PARAMS::v[0], PARAMS::v[1]);
}

} // namespace static_mk
