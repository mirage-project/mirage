// tasks/sm90_next_token.cuh -- the sm90_next_token task type: greedy decoding's
// step from the head to the next step's layers, for one request (as MPK's
// argmax_partial / argmax_reduce and embedding tasks), so that no launch sits
// between the steps. With p the position of the step that made the logits
// (from the page table), the token at p + 1 is the prompt's while p + 1 < P,
// else the argmax of the logits (torch's: the first index of the maximum),
// written to tokens[p + 1 - P]; its embedding row goes to x_next and the page
// table advances to p + 1 (pages in order, as kv_indices holds them). Task x
// of grid (ceil(V / BLOCK), 1, 1), BLOCK = 256 PER: the argmax of logits
// [x BLOCK, x BLOCK + BLOCK) into the node's 64-bit key (value, then the lower
// index); the last task to count takes the rest.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// an order-preserving unsigned key of the value (-0 as +0: torch compares the
// values; any value above 0, the key's reset), then the lower index on a tie
__device__ __forceinline__ unsigned long long sm90_argmax_key(float v, int i) {
  uint32_t const b = __float_as_uint(v + 0.0f);
  uint32_t const u = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((unsigned long long)u << 32) | (0xFFFFFFFFu - (uint32_t)i);
}

// PARAMS {V, H, P (the prompt's tokens), G (tokens' slots), page, PER (logits
// a thread)}; BUF_SLOTS
// {logits [T, V] fp32 (row 0), embedding [V, H] bf16, prompt [P] i64, x_next
// [T, H] bf16 (row 0), tokens [G] i64, kv_indptr [2] i32, last_page_len [1]
// i32}; counters {the tasks (for the consumers), the tasks done with their
// argmax, the key (u64, 2)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_next_token(Maps const &,
                                                    G const &g,
                                                    KernelLocals &,
                                                    StaticTask const &tk) {
  constexpr int V = PARAMS::v[0], H = PARAMS::v[1], P = PARAMS::v[2],
                NG = PARAMS::v[3], PAGE = PARAMS::v[4], PER = PARAMS::v[5];
  constexpr int BLOCK = SM90_THREADS * PER, NB = (V + BLOCK - 1) / BLOCK;
  constexpr int CH = 16; // loads in flight a thread
  constexpr int DONE = SELF::counter + 1, KEY = SELF::counter + 2;
  static_assert(KEY % 2 == 0, "sm90_next_token: the key 8-B aligned");
  using Row = Sm90Row<H>;
  __shared__ unsigned long long red[SM90_WARPS];
  __shared__ int last, tok_s;
  unsigned long long *key = (unsigned long long *)(g.cnt + KEY);
  int *kv_indptr = buf_at<int>(g, slot_at<BUF_SLOTS, 5>());
  int *last_page_len = buf_at<int>(g, slot_at<BUF_SLOTS, 6>());
  // the step's position (every task reads it before it counts below; the last
  // task writes it after)
  int const pos = (kv_indptr[1] - kv_indptr[0] - 1) * PAGE + *last_page_len - 1;
  bool const sample = pos + 1 >= P;
  sm90_task_begin<IN...>(g);
  if (sample) {
    float const *logits = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
    int const base = tk.x * BLOCK + threadIdx.x;
    unsigned long long k = 0;
    for (int c = 0; c < PER; c += CH) {
      float v[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int const i = base + (c + j) * SM90_THREADS;
        v[j] = c + j < PER && i < V ? logits[i] : 0.f;
      }
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int const i = base + (c + j) * SM90_THREADS;
        if (c + j < PER && i < V) {
          k = max(k, sm90_argmax_key(v[j], i));
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      k = max(k, __shfl_xor_sync(0xffffffffu, k, o));
    }
    if ((threadIdx.x & 31) == 0) {
      red[threadIdx.x >> 5] = k;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (int w = 1; w < SM90_WARPS; w++) {
        k = max(k, red[w]);
      }
      atomicMax(key, k);
    }
  }
  if (threadIdx.x == 0) {
    // release: this task's key update before its count; acquire: the last
    // task sees every task's
    uint32_t done;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(done)
                 : "l"(g.cnt + DONE)
                 : "memory");
    last = done == NB - 1;
  }
  __syncthreads();
  if (last) {
    if (threadIdx.x == 0) {
      int const q = pos + 1;
      long long tok;
      if (sample) {
        unsigned long long const k = atomicAdd(key, 0ull);
        tok = (long long)(0xFFFFFFFFu - (uint32_t)k);
        if (q - P < NG) {
          buf_at<long long>(g, slot_at<BUF_SLOTS, 4>())[q - P] = tok;
        }
      } else {
        tok = buf_at<long long const>(g, slot_at<BUF_SLOTS, 2>())[q];
      }
      tok_s = (int)tok;
      int const n = q / PAGE + 1;
      kv_indptr[1] = kv_indptr[0] + n;
      *last_page_len = q + 1 - (n - 1) * PAGE;
    }
    __syncthreads();
    uint4 const *row = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>()) +
                       (size_t)tok_s * Row::VEC;
    uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>());
    uint4 rv[Row::PER];
#pragma unroll
    for (int j = 0; j < Row::PER; j++) {
      if (Row::vec(j) < Row::VEC) {
        rv[j] = row[Row::vec(j)];
      }
    }
#pragma unroll
    for (int j = 0; j < Row::PER; j++) {
      if (Row::vec(j) < Row::VEC) {
        out[Row::vec(j)] = rv[j];
      }
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_next_token, 0)

} // namespace static_mk
