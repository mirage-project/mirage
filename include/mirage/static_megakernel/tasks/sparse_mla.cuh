// tasks/sparse_mla.cuh -- Sparse MLA decode for DeepSeek Sparse Attention
// (GLM-5.3 / 5.2, and GLM-5.3-Flash with ROPE = 0). The task reads a paged
// bf16 latent cache and uses the absorbed query. Task (s, t) is split s of
// token (request) t, for the NH heads on this GPU. The attention body does
// these steps:
//   1. Gather the selected cache rows into shared memory with a
//      double-buffered cp.async pipeline (64 rows for each tile).
//   2. Compute QK and PV with mma.sync m16n8k16. A tile holds 16 heads. With
//      NH = 8, rows 8..15 are empty.
//   3. Compute the online softmax in fp32.
// The terms:
//   q       [T][NH][D_QK] bf16 (D_QK = 512 latent + ROPE), the dsa_q_absorb
//           output (q_nope W_UK, then the rotated q_pe). Its counter counts its
//           tasks.
//   cache   [pages x PAGE][D_QK] bf16. Row (page, o) holds the normalized
//           latent, then the rotated k_pe, of one position. dsa_kv_write
//           writes the row of this step.
//   idx     [T][TOPK] int32 positions (-1: no position). A dsa_index_topk node
//           (its counter + t) or a graph input supplies idx. A "shared" layer
//           uses the indices of the last "full" layer (GLM IndexShare). The
//           task uses these properties of idx (dsa_index_topk gives them):
//           the first min(seq_lens[t], TOPK) slots hold positions in
//           ascending order and the other slots hold -1. If
//           seq_lens[t] <= TOPK, slot e holds position e. Each token t is a
//           different request, and this step writes only the cache row of
//           position seq_lens[t] - 1. If seq_lens[t] <= TOPK (KPOOL 1), the
//           task does not read idx.
//   pages   block_table [T][MAXP] int32. Position p of request t is in page
//           block_table[t][p / PAGE], row p % PAGE. seq_lens [T] int32: the
//           query is at position seq_lens[t] - 1. The task masks the
//           positions after it.
//   output  For each (t, s), o [NH][512] fp32 and lse [NH]. o is the softmax
//           average over the positions of the split. lse = max + log(sum).
//           With KPOOL > 1, lse is -inf if the split has no valid slot.
//           The node buffer holds o [T][S][NH][512], then lse [T][S][NH].
//           Then the task adds 1 to counter + t. dsa_mla_merge merges the S
//           splits. With KPOOL 1, a split with no valid slot writes nothing
//           and does not wait for its inputs. dsa_mla_merge does not read it.
#pragma once
#include "../dsa_common.cuh"
#include <cmath>

namespace static_mk {
namespace smla {

using bf16 = __nv_bfloat16;
constexpr int LATENT = dsa::MLA_LATENT;
constexpr int HEAD_TILE = 16, KV_TILE = 64, NSTAGE = 2;

// QK splits K (D_QK) into KQ parts. Warp w computes part w % KQ of the scores
// of ROWS = KV_TILE KQ / 8 rows of a tile, from row (w / KQ) ROWS.
constexpr int KQ = 4;

// D_QK: the row width of q and of the cache (512 latent + ROPE, ROPE = 64 or
// 0). q holds the query until each warp has its q fragments in registers.
// After that, the space of q holds the KQ partial scores
// [KQ][HEAD_TILE][KV_TILE] fp32 of a tile (partial_scores).
template <int D_QK>
struct alignas(128) Smem {
  bf16 q[HEAD_TILE * D_QK];
  bf16 kv[NSTAGE][KV_TILE * D_QK];
  bf16 probabilities[HEAD_TILE * KV_TILE];
  float row_max[HEAD_TILE]; // log2 units (sparse_mla_split)
  float row_sum[HEAD_TILE];
  float alpha[HEAD_TILE];
  int64_t cache_rows[NSTAGE][KV_TILE];
  unsigned valid_masks[NSTAGE][KV_TILE / 32];
};

using namespace dsa; // the shared-memory and MMA helpers (dsa_common.cuh)

// Return the position of index slot e = tile * KV_TILE + tid. tile is the
// tile index in the token, not in the split. Return -1 if tid >= KV_TILE or
// tile >= last_tile. KPOOL = 1: idx[e], or e if idx is nullptr (e < seq_len;
// else -1). KPOOL > 1 (GLM-5.3-Flash: idx holds NIDX pool ids, dsa_index_topk
// UNIT = KPOOL): slot e < KPOOL NIDX is position
// idx[e / KPOOL] KPOOL + e % KPOOL; the next KPOOL - 1 slots are the tail, the
// positions after the complete pools: (seq_len / KPOOL) KPOOL + i (the mask
// drops the ones at seq_len or later); the rest are -1.
template <int KPOOL, int NIDX>
__device__ __forceinline__ int
    load_index(int const *idx, int tile, int last_tile, int seq_len) {
  int const tid = threadIdx.x, e = tile * KV_TILE + tid;
  if (tid >= KV_TILE || tile >= last_tile) {
    return -1;
  }
  if constexpr (KPOOL == 1) {
    return idx == nullptr ? (e < seq_len ? e : -1) : __ldcg(idx + e);
  } else {
    if (e < KPOOL * NIDX) {
      int const p = __ldcg(idx + e / KPOOL);
      return p >= 0 ? p * KPOOL + e % KPOOL : -1;
    }
    int const i = e - KPOOL * NIDX;
    return i < KPOOL - 1 ? seq_len / KPOOL * KPOOL + i : -1;
  }
}

// Return the partial scores [KQ][HEAD_TILE][KV_TILE] fp32 of a tile. They use
// the space of sm.q. Call it only after every warp holds its q fragments.
template <int D_QK>
__device__ __forceinline__ float *partial_scores(Smem<D_QK> &sm) {
  static_assert(KQ * HEAD_TILE * KV_TILE * sizeof(float) <= sizeof(sm.q),
                "sparse_mla: the partial scores fit in q's space");
  return reinterpret_cast<float *>(sm.q);
}

// 2^x (ex2.approx, subnormal results flushed to 0)
__device__ __forceinline__ float exp2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <int D_QK>
__device__ __forceinline__ int tile_valid_count(Smem<D_QK> const &sm,
                                                int stage) {
  return __popc(sm.valid_masks[stage][0]) + __popc(sm.valid_masks[stage][1]);
}

// Return the cache page of position pos (block_table[pos / PAGE]), or -1 if
// pos is not in [0, seq_len). sparse_mla_split reads the pages of a tile one
// tile before it starts the copies of that tile (prefetch_tile).
template <int PAGE>
__device__ __forceinline__ int page_of(int const *pages, int pos, int seq_len) {
  return pos >= 0 && pos < seq_len ? __ldg(pages + pos / PAGE) : -1;
}

// Translate the positions of one tile to cache rows. Thread tid < KV_TILE
// gives the position pos of slot tid and its cache page (page_of; -1: no
// position). A page outside [0, num_pages) gives no row. Move the valid rows
// to the front of the list. Then start the cp.async copies of these rows into
// stage `stage`. An empty tile starts no copy. All 256 threads must call this
// function.
template <int PAGE, int D_QK>
__device__ __forceinline__ void prefetch_tile(Smem<D_QK> &sm,
                                              bf16 const *cache,
                                              int num_pages,
                                              int stage,
                                              int pos,
                                              int page) {
  int const tid = threadIdx.x, lane = tid % 32;
  int64_t cache_row = -1;
  unsigned valid_mask = 0;
  if (tid < KV_TILE) {
    if (page >= 0 && page < num_pages) {
      cache_row = int64_t(page) * PAGE + pos % PAGE;
    }
    sm.cache_rows[stage][tid] = cache_row;
    valid_mask = __ballot_sync(0xffffffffu, cache_row >= 0);
    if (lane == 0) {
      sm.valid_masks[stage][tid / 32] = valid_mask;
    }
  }
  __syncthreads();
  int const live_rows = tile_valid_count(sm, stage);
  if (live_rows == 0) {
    return; // all threads return here: no copy and no async group
  }
  if (live_rows < KV_TILE) {
    // Each lane keeps its row in a register from before the barrier. Thus a
    // write to the same array cannot overwrite the source row of another lane.
    if (tid < KV_TILE && cache_row >= 0) {
      unsigned const lower = (1u << lane) - 1u;
      int const prefix = __popc(valid_mask & lower) +
                         (tid >= 32 ? __popc(sm.valid_masks[stage][0]) : 0);
      sm.cache_rows[stage][prefix] = cache_row;
    }
    __syncthreads();
  }
  // round up to 16 rows, because each PV step reads 16 rows
  int const rows_to_load = (live_rows + 15) / 16 * 16;
  for (int i = tid; i < rows_to_load * D_QK / 8; i += 256) {
    int const row = i / (D_QK / 8), col = i % (D_QK / 8) * 8;
    int64_t const r = row < live_rows ? sm.cache_rows[stage][row] : -1;
    copy_async(sm.kv[stage] + swizzle<D_QK>(row, col),
               cache + (r >= 0 ? r * D_QK : 0) + col,
               r >= 0);
  }
  copy_commit();
}

// Compute split `split` of S over the TOPK index slots of one token. Write
// o_part [NH][512] fp32 and lse [NH]. Before the call, the caller starts the
// copies of the first tile of the split into stage 0 (prefetch_tile). idx can
// be nullptr (load_index).
template <int NH, int TOPK, int S, int PAGE, int D_QK, int KPOOL, int NIDX>
__device__ __forceinline__ void sparse_mla_split(Smem<D_QK> &sm,
                                                 bf16 const *q,
                                                 bf16 const *cache,
                                                 int const *idx,
                                                 int const *pages,
                                                 int num_pages,
                                                 int seq_len,
                                                 float softmax_scale,
                                                 float *o_part,
                                                 float *lse,
                                                 int split) {
  static_assert(NH <= HEAD_TILE, "at most 16 heads on a GPU (one head tile)");
  static_assert(TOPK % (KV_TILE * S) == 0,
                "sparse_mla: each split has the same number of whole tiles");
  constexpr int PER = TOPK / KV_TILE / S;
  // the k-steps (16 columns each) of a K part, and the rows of a warp
  constexpr int KSTEPS = D_QK / 16 / KQ, ROWS = KV_TILE * KQ / 8;
  static_assert(D_QK % (16 * KQ) == 0 && ROWS % 16 == 0,
                "QK: whole k-steps in each K part, 16-row steps");
  int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int const kq = warp % KQ, row0 = warp / KQ * ROWS;
  int const first_tile = split * PER, last_tile = first_tile + PER;
  for (int i = tid; i < HEAD_TILE * D_QK / 8; i += 256) {
    int const row = i / (D_QK / 8), col = i % (D_QK / 8) * 8;
    bool const valid = row < NH;
    // Other SMs wrote q. cp.async.cg reads it from L2.
    copy_async(sm.q + swizzle<D_QK>(row, col),
               q + (valid ? row * D_QK : 0) + col,
               valid);
  }
  copy_commit();
  if (tid < HEAD_TILE) {
    sm.row_max[tid] = -INFINITY;
    sm.row_sum[tid] = 0.0f;
  }
  // Each warp computes 64 latent columns. Each lane keeps 32 fp32 outputs.
  float acc[4][8] = {};
  // The online softmax state of head tid / 16: the running max of the scores
  // in log2 units (score x softmax_scale x log2(e)) and the running sum. The
  // 16 lanes of the head hold the same values. Lane 0 also writes them to sm.
  float run_max = -INFINITY, run_sum = 0.0f;
  float const scale_log2 = softmax_scale * 1.4426950408889634f;
  int next_pos =
      load_index<KPOOL, NIDX>(idx, first_tile + 1, last_tile, seq_len);
  int next_page = page_of<PAGE>(pages, next_pos, seq_len);
  copy_wait();
  __syncthreads();
  // q fragments of K part kq: the same for every tile of the split
  uint32_t qa[KSTEPS][4];
#pragma unroll
  for (int j = 0; j < KSTEPS; j++) {
    load_a(qa[j],
           sm.q + swizzle<D_QK>(lane % 16,
                                (kq * KSTEPS + j) * 16 + lane / 16 * 8));
  }
  // Wait until all warps hold their q fragments. From here, the space of q
  // holds the partial scores.
  __syncthreads();
  float *const part = partial_scores(sm);

  for (int tile = first_tile; tile < last_tile; tile++) {
    int const stage = (tile - first_tile) & 1;
    if (tile + 1 < last_tile) {
      int const following =
          load_index<KPOOL, NIDX>(idx, tile + 2, last_tile, seq_len);
      prefetch_tile<PAGE, D_QK>(
          sm, cache, num_pages, stage ^ 1, next_pos, next_page);
      next_pos = following;
      next_page = page_of<PAGE>(pages, following, seq_len);
    }
    int const live_rows = tile_valid_count(sm, stage);
    if (live_rows == 0) {
      copy_wait();
      __syncthreads();
      continue;
    }
    bf16 const *kv = sm.kv[stage];
    // QK: warp (row0, kq) computes the partial scores of K part kq for 16
    // heads x ROWS selected rows. Each ldmatrix x4 gives the B fragments of
    // 16 rows (two n8 tiles).
    if (row0 < live_rows) {
      float sc[ROWS / 8][4] = {};
#pragma unroll
      for (int j = 0; j < KSTEPS; j++) {
        int const k = (kq * KSTEPS + j) * 16;
#pragma unroll
        for (int n = 0; n < ROWS / 16; n++) {
          uint32_t b[4];
          load_a(b,
                 kv + swizzle<D_QK>(row0 + n * 16 + lane / 16 * 8 + lane % 8,
                                    k + (lane / 8 % 2) * 8));
          mma16816(qa[j], b, sc[2 * n]);
          mma16816(qa[j], b + 2, sc[2 * n + 1]);
        }
      }
      float *const pk = part + kq * HEAD_TILE * KV_TILE;
#pragma unroll
      for (int n = 0; n < ROWS / 8; n++) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
          int const row = lane / 4 + r * 8, col = row0 + n * 8 + (lane % 4) * 2;
          *reinterpret_cast<float2 *>(pk + row * KV_TILE + col) =
              make_float2(sc[n][r * 2], sc[n][r * 2 + 1]);
        }
      }
    }
    __syncthreads();
    // online softmax: 16 lanes for each head, 4 scores for each lane. Each
    // lane adds the KQ partial scores of its columns. The select then drops
    // the columns at or after live_rows (no cache row) and the rows at or
    // after NH (no head). Their partial scores are not valid.
    int const row = tid / 16, row_lane = tid % 16;
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const col = row_lane * 2 + (i / 2) * 32 + i % 2;
      float s = 0.0f;
#pragma unroll
      for (int kp = 0; kp < KQ; kp++) {
        s += part[(kp * HEAD_TILE + row) * KV_TILE + col];
      }
      p[i] = col < live_rows && row < NH ? s * scale_log2 : -INFINITY;
    }
    float tile_max = fmaxf(fmaxf(p[0], p[1]), fmaxf(p[2], p[3]));
#pragma unroll
    for (int m = 8; m > 0; m /= 2) {
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, m, 16));
    }
    float const new_max = fmaxf(run_max, tile_max);
    float const alpha =
        run_max == -INFINITY ? 0.0f : exp2_ftz(run_max - new_max);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      p[i] = p[i] == -INFINITY ? 0.0f : exp2_ftz(p[i] - new_max);
      sum += p[i];
    }
#pragma unroll
    for (int m = 8; m > 0; m /= 2) {
      sum += __shfl_xor_sync(0xffffffffu, sum, m, 16);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const col = row_lane * 2 + i * 32;
      *reinterpret_cast<__nv_bfloat162 *>(sm.probabilities +
                                          swizzle<KV_TILE>(row, col)) =
          __float22bfloat162_rn(make_float2(p[i * 2], p[i * 2 + 1]));
    }
    run_sum = alpha * run_sum + sum;
    run_max = new_max;
    if (row_lane == 0) {
      sm.alpha[row] = alpha;
      sm.row_sum[row] = run_sum;
      sm.row_max[row] = run_max;
    }
    __syncthreads();
    // PV. In the m16n8k16 accumulator, registers 0, 1, 4, 5 are row lane / 4
    // and registers 2, 3, 6, 7 are row lane / 4 + 8.
    float const alpha0 = sm.alpha[lane / 4], alpha1 = sm.alpha[lane / 4 + 8];
#pragma unroll
    for (int c = 0; c < 4; c++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[c][i] *= (i & 2) ? alpha1 : alpha0;
      }
    }
#pragma unroll
    for (int k = 0; k < KV_TILE; k += 16) {
      if (k < live_rows) {
        uint32_t a[4];
        load_a(a,
               sm.probabilities +
                   swizzle<KV_TILE>(lane % 16, k + lane / 16 * 8));
#pragma unroll
        for (int c = 0; c < 4; c++) {
          uint32_t b[4];
          load_v(b,
                 kv + swizzle<D_QK>(k + lane % 16,
                                    warp * 64 + c * 16 + lane / 16 * 8));
          mma16816(a, b, acc[c]);
          mma16816(a, b + 2, acc[c] + 4);
        }
      }
    }
    // Wait for the gather of the next tile. The barrier keeps this tile in
    // shared memory until all warps complete PV.
    copy_wait();
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 2; r++) {
    int const h = lane / 4 + r * 8;
    float const sum = sm.row_sum[h];
    float const inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    if (h < NH) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
          int const col = warp * 64 + c * 16 + half * 8 + (lane % 4) * 2;
          int const reg = half * 4 + r * 2;
          *reinterpret_cast<float2 *>(o_part + h * LATENT + col) =
              make_float2(acc[c][reg] * inv, acc[c][reg + 1] * inv);
        }
      }
      if (warp == 0 && lane % 4 == 0) {
        // row_max is in log2 units
        lse[h] = sum > 0.0f ? sm.row_max[h] * 0.6931471805599453f + logf(sum)
                            : -INFINITY;
      }
    }
  }
}

} // namespace smla

// Task (s, t) is split s of token t. PARAMS {NH, TOPK (index slots), PAGE,
// MAXP, num_pages, softmax scale (fp32 bits), ROPE (64 or 0), KPOOL (1: idx
// holds positions; 4: pool ids, load_index), NIDX (idx columns)}. BUF_SLOTS
// {cache, block_table, seq_lens, idx}. The node buffer holds o
// [T][S][NH][512] fp32, then lse [T][S][NH] fp32. Q: the dsa_q_absorb node.
// KV: the dsa_kv_write node. The task waits for all KV tasks before it reads
// the cache row of position seq_lens[t] - 1. IX: the dsa_index_topk node, or
// none. With none, idx is a graph input: the indices of a "shared" layer.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Q,
          class KV,
          class IX,
          class... IN>
__device__ __forceinline__ void run_sparse_mla(Maps const &,
                                               G const &g,
                                               KernelLocals &L,
                                               StaticTask const &tk) {
  constexpr int NH = PARAMS::v[0], TOPK = PARAMS::v[1], PAGE = PARAMS::v[2],
                MAXP = PARAMS::v[3], NUM_PAGES = PARAMS::v[4];
  constexpr int ROPE = PARAMS::v[6], D_QK = smla::LATENT + ROPE,
                KPOOL = PARAMS::v[7], NIDX = PARAMS::v[8];
  constexpr int S = SELF::x;
  using Smem = smla::Smem<D_QK>;
  static_assert(ROPE == 64 || ROPE == 0, "sparse_mla: k_pe of 64 or none");
  static_assert(KPOOL == 1 ? NIDX == TOPK : KPOOL * NIDX + KPOOL - 1 <= TOPK,
                "sparse_mla: the index slots hold the positions (KPOOL 1) or "
                "the pools and the tail");
  constexpr int CACHE = slot_at<BUF_SLOTS, 0>(), BT = slot_at<BUF_SLOTS, 1>(),
                SEQ = slot_at<BUF_SLOTS, 2>(), IDX = slot_at<BUF_SLOTS, 3>();
  static_assert(SELF::y == T && SELF::buf >= 0 && SELF::counter >= 0,
                "sparse_mla: grid (splits, tokens), its output and counters");
  static_assert(Q::buf >= 0 && Q::counter >= 0 && KV::counter >= 0,
                "sparse_mla: q from dsa_q_absorb, the cache from dsa_kv_write");
  static_assert(sizeof(Smem) <= RING_BYTES,
                "sparse_mla: its shared memory is the ring's");
  static_assert(IX::x == 0 || IX::counter >= 0,
                "sparse_mla: the indices' producer counts per token");
  constexpr int PER = TOPK / smla::KV_TILE / S; // the tiles of a split
  int const t = tk.y, s = tk.x;
  int const sl = buf_at<int const>(g, SEQ)[t];
  int const live = min(max(sl, 0), TOPK); // the valid slots (KPOOL 1)
  float *const o = buf_at<float>(g, SELF::buf);
  float *const o_part = o + ((size_t)t * S + s) * NH * smla::LATENT;
  float *const lse =
      o + (size_t)T * S * NH * smla::LATENT + ((size_t)t * S + s) * NH;
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  if (KPOOL == 1 && s * PER * smla::KV_TILE >= live) {
    // The split has no valid slot. The task does not wait for its inputs and
    // writes nothing. dsa_mla_merge finds from seq_lens that the split is
    // empty and does not read it. The task only adds 1 to counter + t.
    dsa::stamp_ready(L);
    if (threadIdx.x == 0) { // nothing written: no release
      cnt_add_relaxed(cnt_now(g) + SELF::counter + t, 1);
    }
    return;
  }
  // If seq_lens[t] <= TOPK (KPOOL 1), slot e holds position e. Then the task
  // does not read idx (idx_t nullptr, load_index) and does not wait for IX.
  int const *const idx_t = KPOOL == 1 && sl <= TOPK
                               ? nullptr
                               : buf_at<int const>(g, IDX) + (size_t)t * NIDX;
  int const *const pages = buf_at<int const>(g, BT) + (size_t)t * MAXP;
  auto const first_tile = [&]() { // start the copies of tile 0 of the split
    int const pos0 =
        smla::load_index<KPOOL, NIDX>(idx_t, s * PER, s * PER + PER, sl);
    smla::prefetch_tile<PAGE, D_QK>(*reinterpret_cast<Smem *>(L.sm),
                                    buf_at<__nv_bfloat16 const>(g, CACHE),
                                    NUM_PAGES,
                                    0,
                                    pos0,
                                    smla::page_of<PAGE>(pages, pos0, sl));
  };
  // A shared layer (IX::x == 0) gets idx as a graph input. In this step,
  // dsa_kv_write changes only the cache row of position seq_lens[t] - 1. The
  // slots are in ascending order, so only the last valid slot can hold this
  // position. If tile 0 of the split ends before that slot, the task starts
  // the copies of tile 0 before its waits.
  bool const early =
      KPOOL == 1 && IX::x == 0 && (s * PER + 1) * smla::KV_TILE < live;
  if (early) {
    first_tile();
  }
  if (threadIdx.x == 0) {
    cnt_wait(cnt_now(g) + KV::counter, KV::x * KV::y * KV::z);
    // wait for all q tasks
    cnt_wait(cnt_now(g) + Q::counter, Q::x * Q::y * Q::z);
    if constexpr (IX::x > 0) {
      if (idx_t != nullptr) { // the IX::x write tasks of token t
        cnt_wait(cnt_now(g) + IX::counter + t, IX::x);
      }
    }
  }
  __syncthreads();
  dsa::stamp_ready(L);
  if (!early) {
    first_tile();
  }
  smla::sparse_mla_split<NH, TOPK, S, PAGE, D_QK, KPOOL, NIDX>(
      *reinterpret_cast<Smem *>(L.sm),
      buf_at<__nv_bfloat16 const>(g, Q::buf) + (size_t)t * NH * D_QK,
      buf_at<__nv_bfloat16 const>(g, CACHE),
      idx_t,
      pages,
      NUM_PAGES,
      sl,
      __int_as_float(PARAMS::v[5]),
      o_part,
      lse,
      s);
  dsa::ring_release();
  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter + t, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sparse_mla() {
  return sizeof(smla::Smem<smla::LATENT + PARAMS::v[6]>);
}

} // namespace static_mk
