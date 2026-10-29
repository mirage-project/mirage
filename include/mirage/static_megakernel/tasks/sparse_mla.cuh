// tasks/sparse_mla.cuh -- Sparse MLA decode for DeepSeek Sparse Attention
// (GLM-5.3 / 5.2). The task reads a paged bf16 latent cache and uses the
// absorbed query. Task (s, t) is split s of token (request) t, for the NH heads
// on this GPU. The attention body does these steps:
//   1. Gather the selected cache rows into shared memory with a double-buffered
//      cp.async pipeline (64 rows for each tile).
//   2. Compute QK and PV with mma.sync m16n8k16. A tile holds 16 heads. With
//      NH = 8, rows 8..15 are empty.
//   3. Compute the online softmax in fp32.
// The terms:
//   q       [T][NH][D_QK] bf16 (D_QK = 512 latent + 64 rope), the dsa_q_absorb
//           output (q_nope W_UK, then the rotated q_pe). Its counter counts its
//           tasks.
//   cache   [pages x PAGE][D_QK] bf16. Row (page, o) holds the normalized
//           latent, then the rotated k_pe, of one position. dsa_kv_write
//           writes the row of this step.
//   idx     [T][TOPK] int32 positions (-1: no position). A dsa_index_topk node
//           (its counter + t) or a graph input supplies idx. A "shared" layer
//           uses the indices of the last "full" layer (GLM IndexShare).
//   pages   block_table [T][MAXP] int32. Position p of request t is in page
//           block_table[t][p / PAGE], row p % PAGE. seq_lens [T] int32: the
//           query is at position seq_lens[t] - 1. The task masks the
//           positions after it.
//   output  For each (t, s), o [NH][512] fp32 and lse [NH]. o is the softmax
//           average over the positions of the split. lse = max + log(sum).
//           lse is -inf if the split has no valid position. The node buffer
//           holds o [T][S][NH][512], then lse [T][S][NH]. Then the task adds
//           1 to counter + t. dsa_mla_merge merges the S splits.
#pragma once
#include "../dsa_common.cuh"
#include <cmath>

namespace static_mk {
namespace smla {

using bf16 = __nv_bfloat16;
constexpr int LATENT = dsa::MLA_LATENT, ROPE = 64, D_QK = LATENT + ROPE;
constexpr int HEAD_TILE = 16, KV_TILE = 64, NSTAGE = 2;

struct alignas(128) Smem {
  bf16 q[HEAD_TILE * D_QK];
  bf16 kv[NSTAGE][KV_TILE * D_QK];
  float scores[HEAD_TILE * KV_TILE];
  bf16 probabilities[HEAD_TILE * KV_TILE];
  float row_max[HEAD_TILE];
  float row_sum[HEAD_TILE];
  float alpha[HEAD_TILE];
  int64_t cache_rows[NSTAGE][KV_TILE];
  unsigned valid_masks[NSTAGE][KV_TILE / 32];
};

using namespace dsa; // the shared-memory and MMA helpers (dsa_common.cuh)

// Return idx[tile * KV_TILE + tid]. tile is the tile index in the token, not
// in the split. Return -1 if tid >= KV_TILE or tile >= last_tile.
__device__ __forceinline__ int
    load_index(int const *idx, int tile, int last_tile) {
  int const tid = threadIdx.x;
  return tid < KV_TILE && tile < last_tile ? __ldcg(idx + tile * KV_TILE + tid)
                                           : -1;
}

__device__ __forceinline__ int tile_valid_count(Smem const &sm, int stage) {
  return __popc(sm.valid_masks[stage][0]) + __popc(sm.valid_masks[stage][1]);
}

// Translate the positions of one tile to cache rows. Each warp reads the page
// table one time for each different page. Move the valid rows to the front of
// the list. Then start the cp.async copies of these rows into stage `stage`.
// An empty tile starts no copy.
template <int PAGE>
__device__ __forceinline__ void prefetch_tile(Smem &sm,
                                              bf16 const *cache,
                                              int const *pages,
                                              int num_pages,
                                              int seq_len,
                                              int stage,
                                              int pos) {
  int const tid = threadIdx.x, lane = tid % 32;
  int64_t cache_row = -1;
  unsigned valid_mask = 0;
  if (tid < KV_TILE) {
    bool const valid = pos >= 0 && pos < seq_len;
    int const logical_page = valid ? pos / PAGE : -1;
    unsigned const peers = __match_any_sync(0xffffffffu, logical_page);
    int const leader = __ffs(peers) - 1;
    int page = -1;
    if (valid && lane == leader) {
      page = __ldg(pages + logical_page);
    }
    page = __shfl_sync(0xffffffffu, page, leader);
    if (valid && page >= 0 && page < num_pages) {
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
// o_part [NH][512] fp32 and lse [NH].
template <int NH, int TOPK, int S, int PAGE>
__device__ __forceinline__ void sparse_mla_split(Smem &sm,
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
  int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
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
  int next_pos = load_index(idx, first_tile + 1, last_tile);
  prefetch_tile<PAGE>(sm,
                      cache,
                      pages,
                      num_pages,
                      seq_len,
                      0,
                      load_index(idx, first_tile, last_tile));
  copy_wait();
  __syncthreads();

  for (int tile = first_tile; tile < last_tile; tile++) {
    int const stage = (tile - first_tile) & 1;
    if (tile + 1 < last_tile) {
      int const following = load_index(idx, tile + 2, last_tile);
      prefetch_tile<PAGE>(
          sm, cache, pages, num_pages, seq_len, stage ^ 1, next_pos);
      next_pos = following;
    }
    int const live_rows = tile_valid_count(sm, stage);
    if (live_rows == 0) {
      copy_wait();
      __syncthreads();
      continue;
    }
    bf16 const *kv = sm.kv[stage];
    // QK: each of the 8 warps computes 16 heads x 8 selected rows
    if (warp * 8 < live_rows) {
      float sc[4] = {};
#pragma unroll
      for (int k = 0; k < D_QK; k += 16) {
        uint32_t a[4], b[2];
        load_a(a, sm.q + swizzle<D_QK>(lane % 16, k + lane / 16 * 8));
        load_k(b,
               kv + swizzle<D_QK>(warp * 8 + lane % 8, k + (lane / 8 % 2) * 8));
        mma16816(a, b, sc);
      }
#pragma unroll
      for (int r = 0; r < 2; r++) {
        int const row = lane / 4 + r * 8, col = warp * 8 + (lane % 4) * 2;
        *reinterpret_cast<float2 *>(sm.scores + row * KV_TILE + col) =
            make_float2(sc[r * 2], sc[r * 2 + 1]);
      }
    }
    __syncthreads();
    // online softmax: 16 lanes for each head, 4 scores for each lane
    int const row = tid / 16, row_lane = tid % 16;
    float p[4];
    float tile_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const col = row_lane * 2 + (i / 2) * 32 + i % 2;
      p[i] = col < live_rows && row < NH
                 ? sm.scores[row * KV_TILE + col] * softmax_scale
                 : -INFINITY;
      tile_max = fmaxf(tile_max, p[i]);
    }
#pragma unroll
    for (int m = 8; m > 0; m /= 2) {
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, m, 16));
    }
    float const old_max = sm.row_max[row];
    float const new_max = fmaxf(old_max, tile_max);
    float const alpha = old_max == -INFINITY ? 0.0f : __expf(old_max - new_max);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      p[i] = p[i] == -INFINITY ? 0.0f : __expf(p[i] - new_max);
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
    __syncwarp(); // all lanes read the old row_max before the leader writes it
    if (row_lane == 0) {
      sm.alpha[row] = alpha;
      sm.row_sum[row] = alpha * sm.row_sum[row] + sum;
      sm.row_max[row] = new_max;
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
        lse[h] = sum > 0.0f ? sm.row_max[h] + logf(sum) : -INFINITY;
      }
    }
  }
}

} // namespace smla

// Task (s, t) is split s of token t. PARAMS {NH, TOPK, PAGE, MAXP, num_pages,
// softmax scale (fp32 bits)}. BUF_SLOTS {cache, block_table, seq_lens, idx}.
// The node buffer holds o [T][S][NH][512] fp32, then lse [T][S][NH] fp32. Q:
// the dsa_q_absorb node. KV: the dsa_kv_write node. The task waits for all KV
// tasks, so the cache rows of this step are present. IX: the dsa_index_topk
// node, or none. With none, idx is a graph input: the indices of a "shared"
// layer.
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
  constexpr int S = SELF::x;
  constexpr int CACHE = slot_at<BUF_SLOTS, 0>(), BT = slot_at<BUF_SLOTS, 1>(),
                SEQ = slot_at<BUF_SLOTS, 2>(), IDX = slot_at<BUF_SLOTS, 3>();
  static_assert(SELF::y == T && SELF::buf >= 0 && SELF::counter >= 0,
                "sparse_mla: grid (splits, tokens), its output and counters");
  static_assert(Q::buf >= 0 && Q::counter >= 0 && KV::counter >= 0,
                "sparse_mla: q from dsa_q_absorb, the cache from dsa_kv_write");
  static_assert(sizeof(smla::Smem) <= RING_BYTES,
                "sparse_mla: its shared memory is the ring's");
  static_assert(IX::x == 0 || IX::counter >= 0,
                "sparse_mla: the indices' producer counts per token");
  int const t = tk.y, s = tk.x;
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  if (threadIdx.x == 0) {
    cnt_wait(g.cnt + KV::counter, KV::x * KV::y * KV::z);
    cnt_wait(g.cnt + Q::counter, Q::x * Q::y * Q::z); // wait for all q tasks
    if constexpr (IX::x > 0) {
      // wait for the IX::x write tasks of token t
      cnt_wait(g.cnt + IX::counter + t, IX::x);
    }
  }
  __syncthreads();
  dsa::stamp_ready(L);
  float *const o = buf_at<float>(g, SELF::buf);
  smla::sparse_mla_split<NH, TOPK, S, PAGE>(
      *reinterpret_cast<smla::Smem *>(L.sm),
      buf_at<__nv_bfloat16 const>(g, Q::buf) + (size_t)t * NH * smla::D_QK,
      buf_at<__nv_bfloat16 const>(g, CACHE),
      buf_at<int const>(g, IDX) + (size_t)t * TOPK,
      buf_at<int const>(g, BT) + (size_t)t * MAXP,
      NUM_PAGES,
      buf_at<int const>(g, SEQ)[t],
      __int_as_float(PARAMS::v[5]),
      o + ((size_t)t * S + s) * NH * smla::LATENT,
      o + (size_t)T * S * NH * smla::LATENT + ((size_t)t * S + s) * NH,
      s);
  dsa::ring_release();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter + t, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sparse_mla() {
  return sizeof(smla::Smem);
}

} // namespace static_mk
