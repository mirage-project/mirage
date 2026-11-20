// tasks/engram_rows.cuh -- DeepSeek-V4.1's Engram table lookup in the kernel
// (11-20, DSV41_EG_ROWS=1): task t of grid (T, 1, 1) = token t's C rows of the
// layer's hash table (SGLang's EngramEmbedding: [rows, DIM] fp8 e4m3 with one
// e8m0 scale a BS-column block; GPU r holds rows [starts[r], starts[r + 1])).
// Every row of token t this GPU holds is dequantized as SGLang's engram_gather
// (fp32(v) 2^(e - 127), e = 0: 2^-127; -0 -> +0, as the sharded lookup's sum
// leaves it; one rounding to bf16) and stored into every GPU's copy of the
// symmetric buffer emb [T][C DIM] bf16 (multicast stores, 16 B each; 0xFF
// before: its one reader, act_quant POLL, re-arms it). Exactly one GPU holds
// each row: the GPUs' tasks of token t together write its row once.
// PEER (SGLang's PEER_READ lookup, its default on one NVLink domain): every
// GPU reads every row from the shard that holds it, through SGLang's own
// mappings of the peers' shards (engram_gather_peer: starts [P] padded with
// int64 max, w_bias / s_bias [P] each shard's base address minus its first
// row's offset; the owner = the last shard whose start <= id), and stores the
// whole row into its own buffer (plain stores).
#pragma once
#include "../core.cuh"
#include <cuda_fp8.h>

namespace static_mk {

template <int C, int DIM, int BS, bool PEER = false, int P = 1>
__device__ __forceinline__ void
    engram_rows_task(long long const *__restrict__ ids,
                     int ids_stride,
                     unsigned char const *__restrict__ w,
                     unsigned char const *__restrict__ s,
                     long long row_lo,
                     long long rows,
                     unsigned char *mc,
                     unsigned char *rv,
                     int t,
                     long long const *wb = nullptr,
                     long long const *sb = nullptr,
                     long long const *st = nullptr) {
  static_assert(!PEER || (P & (P - 1)) == 0, "engram_rows PEER: P a power of two");
  constexpr int VR = DIM / 8; // 16-B output vectors a row (8 bf16)
  static_assert(DIM % BS == 0 && BS % 8 == 0,
                "engram_rows: whole 8-column vectors in a scale block");
#pragma unroll 1
  for (int i = threadIdx.x; i < C * VR; i += 256) {
    int const j = i / VR, c8 = i - j * VR;
    long long const id = ids[(size_t)t * ids_stride + j];
    unsigned char const *wr, *sr;
    if constexpr (PEER) { // the owner's shard, in place
      int owner = 0;
#pragma unroll
      for (int step = P / 2; step > 0; step >>= 1) {
        if (st[owner + step] <= id) {
          owner += step;
        }
      }
      wr = reinterpret_cast<unsigned char const *>(wb[owner] + id * DIM);
      sr = reinterpret_cast<unsigned char const *>(sb[owner] + id * (DIM / BS));
    } else {
      long long const local = id - row_lo;
      if (local < 0 || local >= rows) {
        continue; // (another GPU's row)
      }
      wr = w + local * DIM;
      sr = s + local * (DIM / BS);
    }
    uint2 const q = *reinterpret_cast<uint2 const *>(wr + 8 * c8);
    unsigned const e = sr[(8 * c8) / BS];
    float const sc = e ? __uint_as_float(e << 23) : 5.877471754111438e-39f;
    unsigned char const *qb = reinterpret_cast<unsigned char const *>(&q);
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      __half_raw const h = __nv_cvt_fp8_to_halfraw(
          static_cast<__nv_fp8_storage_t>(qb[k]), __NV_E4M3);
      float const v = __half2float(__half(h)) * sc;
      o[k] = v == 0.f ? 0.f : v;
    }
    size_t const off = ((size_t)t * C * DIM + (size_t)j * DIM + 8 * c8) * 2;
    uint4 const packed = pack_bf16x8(make_float4(o[0], o[1], o[2], o[3]),
                                     make_float4(o[4], o[5], o[6], o[7]));
    if constexpr (PEER) {
      *reinterpret_cast<uint4 *>(rv + off) = packed;
    } else {
      push16_mc(mc, rv, off, packed);
    }
  }
  __syncthreads();
}

// engram_rows task (token x): PARAMS {C, DIM, BS, the ids' row stride (in
// ids)[, 1: PEER, P]} (PEER: the table's slots hold w_bias and s_bias [P]
// int64, starts [P]; emb this GPU's buffer only); BUF_SLOTS {ids [T][C] int64 (rows ids_stride apart), the table's fp8
// rows, its e8m0 scales, emb (this GPU's copy), emb's multicast address,
// starts [GPUS + 1] int64 (each GPU's first row, then the table's size: data,
// not params, so that layers with other tables share the code)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_engram_rows(Maps const &,
                                                G const &g,
                                                KernelLocals &,
                                                StaticTask const &tk) {
  constexpr int C = PARAMS::v[0], DIM = PARAMS::v[1], BS = PARAMS::v[2],
                STRIDE = PARAMS::v[3];
  constexpr bool PEER = param_or<PARAMS, 4, 0>() == 1;
  constexpr int P = param_or<PARAMS, 5, 1>();
  static_assert(SELF::x == T && SELF::y == 1,
                "engram_rows: grid (T tokens, 1, 1)");
  long long const *starts = buf_at<long long const>(g, slot_at<BUF_SLOTS, 5>());
  long long const lo = PEER ? 0 : starts[g.rank], hi = PEER ? 0 : starts[g.rank + 1];
  engram_rows_task<C, DIM, BS, PEER, P>(
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 0>()),
      STRIDE,
      buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 2>()),
      lo,
      hi - lo,
      buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 3>()),
      tk.x,
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 2>()),
      starts);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_engram_rows() {
  return 0;
}

} // namespace static_mk
