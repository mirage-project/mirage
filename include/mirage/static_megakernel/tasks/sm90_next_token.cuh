// tasks/sm90_next_token.cuh -- the sm90_next_token task type: greedy decoding's
// step from the head to the next step's layers (as MPK's
// argmax_partial / argmax_reduce and embedding tasks), for R requests, so
// that no launch sits between the steps. With p the position of request r's
// step that made its logits (row r; from the page table), its token at p + 1
// is its prompt's while p + 1 < P, else the argmax of the row (torch's: the
// first index of the maximum), written to tokens[r][p + 1 - P]. A negative
// prompt token is the argmax too, written over it (prompts of different
// lengths: each one's tail past its length -1, so that request generates from
// its own length; its tokens are prompt[r][len ..] then tokens[r]); its
// embedding row goes to x_next row r and the page table advances to p + 1. Task
// x of grid (ceil(V / BLOCK), 1, 1), BLOCK = 256 PER: the argmax of each row's
// logits [x BLOCK, x BLOCK + BLOCK) into the request's 64-bit key (value, then
// the lower index); the last task to count takes the rest.
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

// PARAMS {V, H, P (the prompt's tokens), G (tokens' slots a request), page, PER
// (logits a thread), R (requests), MP (pages a request: request r's pages are
// r MP ..)}; BUF_SLOTS
// {logits [T, V] fp32 (rows 0 .. R), embedding [V, H] bf16, prompt [R, P] i64,
// x_next [T, H] bf16 (rows 0 .. R), tokens [R, G] i64, kv_indptr [R + 1] i32,
// last_page_len [R] i32, kv_indices [R MP] i32}; counters {the tasks (for the
// consumers), the tasks done with their argmax, a key (u64, 2) a request}
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
                NG = PARAMS::v[3], PAGE = PARAMS::v[4], PER = PARAMS::v[5],
                R = PARAMS::v[6], MP = PARAMS::v[7];
  constexpr int BLOCK = SM90_THREADS * PER, NB = (V + BLOCK - 1) / BLOCK;
  constexpr int CH = 16; // loads in flight a thread
  constexpr int DONE = SELF::counter + 1, KEY = SELF::counter + 2;
  static_assert(KEY % 2 == 0, "sm90_next_token: the keys 8-B aligned");
  static_assert(R <= SM90_THREADS, "sm90_next_token: a thread a request");
  using Row = Sm90Row<H>;
  __shared__ unsigned long long red[SM90_WARPS];
  __shared__ int last, tok_s[R], n_s[R];
  unsigned long long *key = (unsigned long long *)(g.cnt + KEY);
  int *kv_indptr = buf_at<int>(g, slot_at<BUF_SLOTS, 5>());
  int *last_page_len = buf_at<int>(g, slot_at<BUF_SLOTS, 6>());
  float const *logits = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
  long long *prompt = buf_at<long long>(g, slot_at<BUF_SLOTS, 2>());
  // each request's step position (every task reads them before it counts
  // below; the last task writes them after)
  int pos[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    pos[r] =
        (kv_indptr[r + 1] - kv_indptr[r] - 1) * PAGE + last_page_len[r] - 1;
  }
  sm90_task_begin<IN...>(g);
  int const base = tk.x * BLOCK + threadIdx.x;
#pragma unroll 1
  for (int r = 0; r < R; r++) {
    if (pos[r] + 1 < P && prompt[(size_t)r * P + pos[r] + 1] >= 0) {
      continue; // the prompt's next token (the same for every thread)
    }
    float const *lr = logits + (size_t)r * V;
    unsigned long long k = 0;
    for (int c = 0; c < PER; c += CH) {
      float v[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int const i = base + (c + j) * SM90_THREADS;
        v[j] = c + j < PER && i < V ? lr[i] : 0.f;
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
      atomicMax(key + r, k);
    }
    __syncthreads(); // red's next use
  }
  if (threadIdx.x == 0) {
    // release: this task's key updates before its count; acquire: the last
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
    if (threadIdx.x < R) {
      int const r = threadIdx.x, q = pos[r] + 1;
      long long tok = q < P ? prompt[(size_t)r * P + q] : -1;
      if (tok < 0) {
        unsigned long long const k = atomicAdd(key + r, 0ull);
        tok = (long long)(0xFFFFFFFFu - (uint32_t)k);
        if (q < P) {
          prompt[(size_t)r * P + q] = tok;
        } else if (q - P < NG) {
          buf_at<long long>(
              g, slot_at<BUF_SLOTS, 4>())[(size_t)r * NG + q - P] = tok;
        }
      }
      tok_s[r] = (int)tok;
      int const n = q / PAGE + 1;
      n_s[r] = n;
      last_page_len[r] = q + 1 - (n - 1) * PAGE;
    }
    __syncthreads();
    // the page table: request r's n pages r MP .. r MP + n - 1, packed
    int *kv_indices = buf_at<int>(g, slot_at<BUF_SLOTS, 7>());
    int off = 0;
#pragma unroll 1
    for (int r = 0; r < R; r++) {
      for (int j = threadIdx.x; j < n_s[r]; j += SM90_THREADS) {
        kv_indices[off + j] = r * MP + j;
      }
      off += n_s[r];
      if (threadIdx.x == 0) {
        kv_indptr[r + 1] = off;
      }
    }
    uint4 const *emb = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
    uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>());
#pragma unroll 1
    for (int r = 0; r < R; r++) {
      uint4 const *row = emb + (size_t)tok_s[r] * Row::VEC;
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
          out[(size_t)r * Row::VEC + Row::vec(j)] = rv[j];
        }
      }
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_next_token, 0)

} // namespace static_mk
