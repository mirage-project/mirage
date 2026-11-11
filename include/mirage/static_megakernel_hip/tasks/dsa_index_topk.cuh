// tasks/dsa_index_topk.cuh (HIP, gfx950) -- GLM-5.3-Flash's top-K pooled positions per token (SGLang ROCm:
// kpool_topk_transform.cuh fast_kpool_topk_transform_fused, the pool part; amd/notes/sglang_v0522_glm_dsa_contract.md 1.4
// step 10). Task t = token t, n = seq_lens[t] / 4 complete pools:
//   n <= K   every pool, in order (SGLang takes them all)
//   n > K    the K pools of the largest scores (dsa_index_score's fp32 scores [T][MAXP]): a radix select of the K-th largest
//            key (4 passes of 8 bits over the order-preserving u32 of the score, histograms in LDS), then the pools above it
//            and the ties at it with the lowest pool ids
//   output   pool ids int32 [T][K] in ascending order, -1 past min(n, K) (the node's buffer; read after its count); sparse attention expands a
//            pool g to positions 4 g .. 4 g + 3 and adds the tail positions 4 min(n, K) .. (as SGLang: the pools then
//            seq_lens % 4 positions after the complete pools when n <= K)
// One count per task.
#pragma once
#include "../dsa_util.cuh"

namespace static_mk {
namespace dsa {

__device__ __forceinline__ uint32_t score_key(float s) {
  uint32_t const b = __float_as_uint(s + 0.0f); // (-0 -> +0)
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

} // namespace dsa

// PARAMS {K (512), MAXP}; BUF_SLOTS {seq_lens (int32)}; its buffer: the pool ids (int32 [T][K]); SC: dsa_index_score (its scores [T][MAXP], a count per token of
// its chunks)
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class SC, class... IN>
__device__ __forceinline__ void run_dsa_index_topk(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int K = PARAMS::v[0], MAXP = PARAMS::v[1], PER = (MAXP + 255) / 256;
  static_assert(SC::buf >= 0 && SC::counter >= 0 && SELF::buf >= 0 && SELF::counter >= 0 && SELF::x == T && K % 256 == 0,
                "dsa_index_topk: the scores and their counts per token, a task per token");
  int const t = tk.x, tid = STATIC_TIDX;
  int const n = min(buf_at<int const>(g, slot_at<BUF_SLOTS, 0>())[t] / 4, MAXP);
  int *const out = buf_at<int>(g, SELF::buf) + (size_t)t * K; // (the node's buffer; the harness's graph tensor gets its copy)
  uint32_t *const hist = reinterpret_cast<uint32_t *>(L.sm);  // [256]
  int *const pre = reinterpret_cast<int *>(L.sm + 256 * 4);    // [256 + 2] per-thread counts, scanned
  int *const sel = pre + 258;                                   // [2]: the K-th key's prefix and the count above it
  __syncthreads(); // the LDS may still be read by the previous task
  if (n <= K) {
    for (int i = tid; i < K; i += 256) {
      st4_out_i(out + i, i < n ? i : -1);
    }
  } else {
    if (tid == 0) {
      cnt_wait_in(cnt_of(g) + SC::counter + t, (uint32_t)SC::x);
    }
    __syncthreads();
    float const *sc = buf_at<float const>(g, SC::buf) + (size_t)t * MAXP;
    // this thread's pools: [PER tid, PER tid + PER) (contiguous: the output in ascending order by a scan of counts)
    uint32_t key[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
      int const j = PER * tid + i;
      key[i] = j < n ? dsa::score_key(__uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(sc), (uint32_t)(4 * j), 0,
                                                                                           AUX_DEV)))
                     : 0u; // (key 0: below every score)
    }
    // radix select: the K-th largest key, 8 bits a pass from the top
    uint32_t prefix = 0, pmask = 0;
    int need = K; // how many of the keys matching prefix (in the bits fixed so far) are still to be taken from the top
#pragma unroll 1
    for (int pass = 0; pass < 4; pass++) {
      int const sh = 24 - 8 * pass;
      hist[tid] = 0;
      __syncthreads();
#pragma unroll
      for (int i = 0; i < PER; i++) {
        if (PER * tid + i < n && (key[i] & pmask) == prefix) {
          atomicAdd(&hist[(key[i] >> sh) & 0xFF], 1u);
        }
      }
      __syncthreads();
      if (tid == 0) { // from the top bin down: the bin holding the need-th key
        int acc = 0, b = 255;
        for (; b > 0; b--) {
          if (acc + (int)hist[b] >= need) {
            break;
          }
          acc += hist[b];
        }
        sel[0] = b;
        sel[1] = acc;
      }
      __syncthreads();
      prefix |= (uint32_t)sel[0] << sh;
      pmask |= 0xFFu << sh;
      need -= sel[1];
      __syncthreads();
    }
    uint32_t const thr = prefix; // the K-th largest key; take all keys > thr and the first `need` keys == thr
    int gt = 0, eq = 0;
#pragma unroll
    for (int i = 0; i < PER; i++) {
      if (PER * tid + i < n) {
        gt += key[i] > thr;
        eq += key[i] == thr;
      }
    }
    pre[tid] = eq;
    __syncthreads();
    if (tid == 0) { // exclusive scan of the ties per thread (serial: 256)
      int a = 0;
      for (int i = 0; i < 256; i++) {
        int const v = pre[i];
        pre[i] = a;
        a += v;
      }
    }
    __syncthreads();
    int const eq0 = pre[tid]; // ties before this thread
    int take_eq = min(max(need - eq0, 0), eq);
    int cnt = gt + take_eq;
    __syncthreads();
    pre[tid] = cnt;
    __syncthreads();
    if (tid == 0) {
      int a = 0;
      for (int i = 0; i < 256; i++) {
        int const v = pre[i];
        pre[i] = a;
        a += v;
      }
    }
    __syncthreads();
    int o = pre[tid];
#pragma unroll
    for (int i = 0; i < PER; i++) {
      int const j = PER * tid + i;
      if (j < n) {
        bool const take = key[i] > thr || (key[i] == thr && take_eq > 0);
        if (key[i] == thr && take_eq > 0) {
          take_eq--;
        }
        if (take) {
          st4_out_i(out + o, j);
          o++;
        }
      }
    }
  }
  task_post(cnt_of(g) + SELF::counter, 1);
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_dsa_index_topk() {
  return 256 * 4 + 260 * 4;
}

} // namespace static_mk
