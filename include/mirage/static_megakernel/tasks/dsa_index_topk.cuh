// tasks/dsa_index_topk.cuh -- Select the indexer top-k of DeepSeek Sparse
// Attention (GLM-5.3 / 5.2; GLM-5.3-Flash with UNIT = 4). Two nodes of this
// task type do the selection. Token t has n = min(seq_lens[t] / UNIT, MAXCTX)
// scored entries (UNIT 1: the positions; UNIT 4: the complete pools of 4
// positions, dsa_index_score FLASH). Below, a position is an entry. Each
// position
// has an index score (the dsa_index_score output, fp32 [T][MAXCTX], each a
// bf16 value). The output idx[t][0, TOPK) holds the positions of the TOPK
// largest scores in ascending order. When n <= TOPK, the output holds all n
// positions, then -1. For equal scores at the threshold, the task selects the
// lowest positions. Thus each launch gives the same selection.
//   keys     A bf16 score as a 16-bit key with the same order
//            (dsa::score_key16). The score tasks count the high 12 bits of
//            the keys of token t in hist[t] (dsa::TOPK_BINS bins).
//   chunks   Both nodes have grid (CHK, T). Task (c, t) takes positions
//            [1024 c, 1024 c + 1024) of token t (CHK = MAXCTX / 1024). Each
//            lane takes 4 positions, each warp 128.
//   count    (MODE 0) Wait for the score tasks of token t. Find the bin b1 of
//            hist[t] that holds the TOPK-th largest key. k1 is the number of
//            keys to take from bin b1. Count the keys of the chunk above bin
//            b1. Count the keys in bin b1 by their low 4 bits. Write the
//            record of the chunk.
//   write    (MODE 1) Wait for the count tasks of token t. Read the CHK
//            records. Find the threshold key (b1, b2) and the number of keys
//            to take at it. Find the first output slot of each chunk and the
//            number of threshold keys that it takes. Lower chunks take
//            threshold keys first. Then each warp counts its keys, and ballots
//            give each lane its output slots.
// Each node adds 1 to its counter + t for each task. sparse_mla waits for the
// CHK write tasks of token t.
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {
namespace dsa {

// A chunk record of the count node holds TOPK_REC ints: b1, k1, the number of
// keys above bin b1, then the 16 counts of the keys in bin b1 by their low 4
// bits. static_megakernel.py DSA_TOPK_REC and DSA_TOPK_CH have the same
// values.
constexpr int TOPK_REC = 19, TOPK_CH = 1024;
static_assert(TOPK_CH == 4 * 256,
              "dsa_index_topk: 4 positions for each thread");
static_assert(TOPK_BINS == 16 * 256, "dsa_index_topk: 16 bins for each thread");

// Return the exclusive scan of v over the 256 threads (in thread order).
// *total gets the sum.
__device__ __forceinline__ int topk_block_scan(int v, int *total) {
  __shared__ int warp_sum[8];
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int incl = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int const o = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d) {
      incl += o;
    }
  }
  if (lane == 31) {
    warp_sum[warp] = incl;
  }
  __syncthreads();
  int before = 0, all = 0;
#pragma unroll
  for (int w = 0; w < 8; w++) {
    before += (w < warp) ? warp_sum[w] : 0;
    all += warp_sum[w];
  }
  __syncthreads(); // the next scan writes warp_sum again
  *total = all;
  return before + incl - v;
}

// Return the keys of positions i .. i + 3 of the chunk (i % 4 == 0). Load the
// scores from L2 because other SMs wrote them. valid[j] is true if i + j < c1.
__device__ __forceinline__ void topk_keys(
    float const *scores, int i, int c1, uint32_t (&k)[4], bool (&valid)[4]) {
  float4 const v = i < c1 ? __ldcg(reinterpret_cast<float4 const *>(scores + i))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
  k[0] = score_key16(v.x);
  k[1] = score_key16(v.y);
  k[2] = score_key16(v.z);
  k[3] = score_key16(v.w);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    valid[j] = i + j < c1;
  }
}

// The count task (c, t): write the record of chunk c to rec. hist: hist[t].
// If no bin holds the TOPK-th key, hist[t] does not match the scores. Then
// the record has b1 = 0 and k1 = 0, and the write node writes at most TOPK
// positions.
template <int TOPK>
__device__ __forceinline__ void
    topk_count(float const *scores, int const *hist, int n, int c, int *rec) {
  __shared__ int low[16], wabove[8];
  __shared__ int sel_bin, sel_k;
  int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // Find the bin of the TOPK-th largest key. Thread r reads the 16 bins
  // TOPK_BINS - 1 - 16 r, TOPK_BINS - 2 - 16 r, ..., in this order. Load the
  // counts from L2 because the score tasks added them.
  {
    int hb[16], sum = 0;
    int4 const *h4 =
        reinterpret_cast<int4 const *>(hist + TOPK_BINS - 16 - 16 * tid);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int4 const v = __ldcg(h4 + 3 - q);
      hb[4 * q] = v.w;
      hb[4 * q + 1] = v.z;
      hb[4 * q + 2] = v.y;
      hb[4 * q + 3] = v.x;
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
      sum += hb[j];
    }
    if (tid < 16) {
      low[tid] = 0;
    }
    if (tid == 0) { // the values if no bin holds the TOPK-th key
      sel_bin = 0;
      sel_k = 0;
    }
    int total;
    int above = topk_block_scan(sum, &total);
    if (above < TOPK && above + sum >= TOPK) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        if (above < TOPK && above + hb[j] >= TOPK) {
          sel_bin = TOPK_BINS - 1 - 16 * tid - j;
          sel_k = TOPK - above;
        }
        above += hb[j];
      }
    }
  }
  __syncthreads();
  uint32_t const b1 = sel_bin;
  int const c1 = min(n, (c + 1) * TOPK_CH);
  uint32_t k[4];
  bool valid[4];
  topk_keys(scores, c * TOPK_CH + 4 * tid, c1, k, valid);
  int ab = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    if (valid[j]) {
      ab += (k[j] >> 4) > b1;
      if ((k[j] >> 4) == b1) {
        atomicAdd(&low[k[j] & 15], 1);
      }
    }
  }
  ab = __reduce_add_sync(0xffffffffu, ab);
  if (lane == 0) {
    wabove[warp] = ab;
  }
  __syncthreads();
  if (tid < 16) {
    rec[3 + tid] = low[tid];
  } else if (tid == 16) {
    int s = 0;
    for (int w = 0; w < 8; w++) {
      s += wabove[w];
    }
    rec[0] = (int)b1;
    rec[1] = sel_k;
    rec[2] = s;
  }
}

// The write task (c, t): write the selected positions of chunk c to out
// (idx[t], at most TOPK). recs: the CHK records of token t.
template <int TOPK, int CHK>
__device__ __forceinline__ void
    topk_write(float const *scores, int const *recs, int n, int c, int *out) {
  __shared__ int wcnt[8][2], wbase[8][2];
  __shared__ int thr_s, cbase, ctake;
  int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  static_assert(CHK <= 32, "dsa_index_topk: one lane for each chunk record");
  // Warp 0 does this step in registers. Lane i reads the record of chunk i
  // from L2 (other SMs wrote it). Lane b < 16 also reads the count of the keys
  // in bin b1 with low 4 bits b from each record.
  if (warp == 0) {
    int r[TOPK_REC], hcol[CHK];
#pragma unroll
    for (int q = 0; q < TOPK_REC; q++) {
      r[q] = lane < CHK ? __ldcg(recs + lane * TOPK_REC + q) : 0;
    }
#pragma unroll
    for (int i = 0; i < CHK; i++) {
      hcol[i] = lane < 16 ? __ldcg(recs + i * TOPK_REC + 3 + lane) : 0;
    }
    int const b1 = __shfl_sync(0xffffffffu, r[0], 0),
              k1 = __shfl_sync(0xffffffffu, r[1], 0);
    // Find the low 4 bits b2 of the threshold and the number keq of keys to
    // take at it. h_b is the count of bin b over all chunks. sb is the suffix
    // sum h_b + ... + h_15. b2 is the bin with sb(b2 + 1) < k1 <= sb(b2).
    int hb = 0;
#pragma unroll
    for (int i = 0; i < CHK; i++) {
      hb += hcol[i];
    }
    int sb = hb;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
      int const o = __shfl_down_sync(0xffffffffu, sb, d);
      sb += lane + d < 16 ? o : 0;
    }
    unsigned const hit =
        __ballot_sync(0xffffffffu, lane < 16 && sb - hb < k1 && sb >= k1);
    int const b2 = hit != 0u ? __ffs(hit) - 1 : 0;
    int const s_above = __shfl_sync(0xffffffffu, sb - hb, b2);
    int const keq = hit != 0u ? k1 - s_above : 0;
    int gt = r[2], eq = 0;
#pragma unroll
    for (int b = 0; b < 16; b++) {
      gt += b > b2 ? r[3 + b] : 0;
      eq += b == b2 ? r[3 + b] : 0;
    }
    // Lower chunks take the threshold keys first. Find the equal keys before
    // chunk i, then the keys that chunk i takes and its first output slot.
    int ein = eq;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int const o = __shfl_up_sync(0xffffffffu, ein, d);
      ein += lane >= d ? o : 0;
    }
    int const take = max(0, min(eq, keq - (ein - eq)));
    int sin = gt + take;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int const o = __shfl_up_sync(0xffffffffu, sin, d);
      sin += lane >= d ? o : 0;
    }
    if (lane == c) {
      cbase = sin - gt - take;
      ctake = take;
      thr_s = (b1 << 4) | b2;
    }
  }
  __syncthreads();
  uint32_t const thr = thr_s;
  int const c1 = min(n, (c + 1) * TOPK_CH);
  uint32_t k[4];
  bool valid[4];
  topk_keys(scores, c * TOPK_CH + 4 * tid, c1, k, valid);
  bool gt[4], eq[4];
  unsigned gm[4], em[4];
  int g_all = 0, e_all = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    gt[j] = valid[j] && k[j] > thr;
    eq[j] = valid[j] && k[j] == thr;
    gm[j] = __ballot_sync(0xffffffffu, gt[j]);
    em[j] = __ballot_sync(0xffffffffu, eq[j]);
    g_all += __popc(gm[j]);
    e_all += __popc(em[j]);
  }
  if (lane == 0) {
    wcnt[warp][0] = g_all;
    wcnt[warp][1] = e_all;
  }
  __syncthreads();
  if (tid == 0) { // the first slot of each warp and its keys at the threshold
    int slot = cbase, left = ctake;
    for (int w = 0; w < 8; w++) {
      int const take = min(wcnt[w][1], left);
      wbase[w][0] = slot;
      wbase[w][1] = take;
      slot += wcnt[w][0] + take;
      left -= take;
    }
  }
  __syncthreads();
  unsigned const below = (1u << lane) - 1u;
  int er = 0; // the equal keys of the lower lanes of the warp
#pragma unroll
  for (int j = 0; j < 4; j++) {
    er += __popc(em[j] & below);
  }
  int const eq_take = wbase[warp][1];
  bool take[4];
  int t_below = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    take[j] = gt[j] || (eq[j] && er < eq_take);
    er += eq[j];
    t_below += __popc(__ballot_sync(0xffffffffu, take[j]) & below);
  }
  int pos = wbase[warp][0] + t_below;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    if (take[j] && pos < TOPK) {
      out[pos] = c * TOPK_CH + 4 * tid + j;
    }
    pos += take[j];
  }
}

} // namespace dsa

// Task (c, t). PARAMS {TOPK, MAXCTX, MODE, UNIT (positions per entry)}.
//   MODE 0 (count): BUF_SLOTS {seq_lens (int32 [T]), hist (the token
//   histograms of the score node, int [T][TOPK_BINS])}. P: the
//   dsa_index_score node (grid (chunks, T); its counter + t counts the chunks
//   of token t). The output buffer holds the records, int [T][CHK][TOPK_REC].
//   MODE 1 (write): BUF_SLOTS {seq_lens, idx (the output, int32 [T][TOPK]),
//   the scores (the buffer of the score node, fp32 [T][MAXCTX])}. P: the count
//   node.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class P,
          class... IN>
__device__ __forceinline__ void run_dsa_index_topk(Maps const &,
                                                   G const &g,
                                                   KernelLocals &L,
                                                   StaticTask const &tk) {
  using dsa::TOPK_CH;
  using dsa::TOPK_REC;
  constexpr int TOPK = PARAMS::v[0], MAXCTX = PARAMS::v[1], MODE = PARAMS::v[2],
                UNIT = PARAMS::v[3], CHK = MAXCTX / TOPK_CH;
  constexpr int SEQ = slot_at<BUF_SLOTS, 0>();
  static_assert(MAXCTX % TOPK_CH == 0 && SELF::x == CHK && SELF::y == T &&
                    SELF::counter >= 0,
                "dsa_index_topk: grid (MAXCTX / 1024, T), a counter per token");
  static_assert(TOPK > 0 && MAXCTX >= TOPK && UNIT >= 1,
                "dsa_index_topk: sizes");
  int const c = tk.x, t = tk.y;
  int const n = min(MAXCTX, buf_at<int const>(g, SEQ)[t] / UNIT);
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  if constexpr (MODE == 0) {
    static_assert(P::buf >= 0 && P::counter >= 0 && P::y == T &&
                      P::v[1] == MAXCTX && SELF::buf >= 0,
                  "dsa_index_topk count: the scores of a dsa_index_score "
                  "node over MAXCTX positions, the records");
    if (threadIdx.x == 0) {
      cnt_wait(cnt_now(g) + P::counter + t, P::x);
    }
    __syncthreads();
    dsa::stamp_ready(L);
    if (n > TOPK) {
      dsa::topk_count<TOPK>(buf_at<float const>(g, P::buf) + (size_t)t * MAXCTX,
                            buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()) +
                                (size_t)t * dsa::TOPK_BINS,
                            n,
                            c,
                            buf_at<int>(g, SELF::buf) +
                                ((size_t)t * CHK + c) * TOPK_REC);
    }
  } else {
    static_assert(P::v[2] == 0 && P::x == CHK && P::buf >= 0 && P::counter >= 0,
                  "dsa_index_topk write: the records of a count node");
    if (threadIdx.x == 0) {
      cnt_wait(cnt_now(g) + P::counter + t, CHK);
    }
    __syncthreads();
    dsa::stamp_ready(L);
    int *const out = buf_at<int>(g, slot_at<BUF_SLOTS, 1>()) + (size_t)t * TOPK;
    if (n <= TOPK) { // all positions, then -1: slots [c PER, c PER + PER)
      constexpr int PER = (TOPK + CHK - 1) / CHK;
      for (int i = c * PER + threadIdx.x; i < min(TOPK, c * PER + PER);
           i += 256) {
        out[i] = i < n ? i : -1;
      }
    } else if (c * TOPK_CH < n) {
      dsa::topk_write<TOPK, CHK>(
          buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)t * MAXCTX,
          buf_at<int const>(g, P::buf) + (size_t)t * CHK * TOPK_REC,
          n,
          c,
          out);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter + t, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_index_topk() {
  return 0;
}

} // namespace static_mk
