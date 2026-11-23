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
// the lower index), a warp a row (rows w, w + 8, ..: no block barrier a row);
// the last task to count takes the rest.
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
// r MP ..), RP (0, else a second page table, kv_indices_ring: request r's page
// j at r RP + j % RP, a ring for sliding-window layers' caches)}; BUF_SLOTS
// {logits [T, V] fp32 (rows 0 .. R), embedding [V, H] bf16, prompt [R, P] i64,
// x_next [T, H] bf16 (rows 0 .. R), tokens [R, G] i64, kv_indptr [R + 1] i32,
// last_page_len [R] i32, kv_indices [R MP] i32, kv_indices_ring [R MP] i32
// (RP > 0)}; counters {the tasks (for the
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
                R = PARAMS::v[6], MP = PARAMS::v[7], RP = PARAMS::v[8];
  constexpr int BLOCK = SM90_THREADS * PER, NB = (V + BLOCK - 1) / BLOCK;
  constexpr int CH = 16; // loads in flight a thread
  constexpr int DONE = SELF::counter + 1, KEY = SELF::counter + 2;
  static_assert(KEY % 2 == 0, "sm90_next_token: the keys 8-B aligned");
  static_assert(R <= SM90_THREADS, "sm90_next_token: a thread a request");
  using Row = Sm90Row<H>;
  __shared__ int last, tok_s[R], n_s[R], pos[R];
  unsigned long long *key = (unsigned long long *)(g.cnt + KEY);
  int *kv_indptr = buf_at<int>(g, slot_at<BUF_SLOTS, 5>());
  int *last_page_len = buf_at<int>(g, slot_at<BUF_SLOTS, 6>());
  float const *logits = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
  long long *prompt = buf_at<long long>(g, slot_at<BUF_SLOTS, 2>());
  // each request's step position (every task reads them before it counts
  // below; the last task writes them after)
  if (threadIdx.x < R) {
    int const r = threadIdx.x;
    pos[r] =
        (kv_indptr[r + 1] - kv_indptr[r] - 1) * PAGE + last_page_len[r] - 1;
  }
  sm90_task_begin<IN...>(g);
  __syncthreads(); // pos
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const col0 = tk.x * BLOCK;
#pragma unroll 1
  for (int r = warp; r < R; r += SM90_WARPS) {
    if (pos[r] + 1 < P && prompt[(size_t)r * P + pos[r] + 1] >= 0) {
      continue; // the prompt's next token
    }
    float const *lr = logits + (size_t)r * V;
    unsigned long long k = 0;
    if constexpr (V % 4 == 0 && BLOCK % 128 == 0) {
      // lane l: the float4s l, l + 32, .. of the block (16-B aligned rows)
      constexpr int NV = BLOCK / 128;
#pragma unroll
      for (int c = 0; c < NV; c += CH / 2) {
        float4 v[CH / 2];
#pragma unroll
        for (int j = 0; j < CH / 2; j++) {
          int const i = col0 + 4 * (lane + 32 * (c + j));
          v[j] = c + j < NV && i < V ? *reinterpret_cast<float4 const *>(lr + i)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int j = 0; j < CH / 2; j++) {
          int const i = col0 + 4 * (lane + 32 * (c + j));
          if (c + j < NV && i < V) {
            k = max(k, sm90_argmax_key(v[j].x, i));
            k = max(k, sm90_argmax_key(v[j].y, i + 1));
            k = max(k, sm90_argmax_key(v[j].z, i + 2));
            k = max(k, sm90_argmax_key(v[j].w, i + 3));
          }
        }
      }
    } else {
      constexpr int NS = BLOCK / 32;
      for (int c = 0; c < NS; c += CH) {
        float v[CH];
#pragma unroll
        for (int j = 0; j < CH; j++) {
          int const i = col0 + lane + 32 * (c + j);
          v[j] = c + j < NS && i < V ? lr[i] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < CH; j++) {
          int const i = col0 + lane + 32 * (c + j);
          if (c + j < NS && i < V) {
            k = max(k, sm90_argmax_key(v[j], i));
          }
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      k = max(k, __shfl_xor_sync(0xffffffffu, k, o));
    }
    if (lane == 0) {
      atomicMax(key + r, k);
    }
  }
  __threadfence();
  __syncthreads(); // every warp's key updates before the count
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
    // the page table: request r's n pages r MP .. r MP + n - 1, packed (its
    // offset, the pages before it, by thread r); a warp a request
    int *kv_indices = buf_at<int>(g, slot_at<BUF_SLOTS, 7>());
    int *ring = RP > 0 ? buf_at<int>(g, slot_at<BUF_SLOTS, 8>()) : nullptr;
    if (threadIdx.x < R) {
      int off = 0;
      for (int i = 0; i < (int)threadIdx.x; i++) {
        off += n_s[i];
      }
      pos[threadIdx.x] = off; // (pos read: before the barrier above)
      kv_indptr[threadIdx.x + 1] = off + n_s[threadIdx.x];
    }
    __syncthreads();
    int const warp_l = threadIdx.x >> 5, lane_l = threadIdx.x & 31;
#pragma unroll 1
    for (int r = warp_l; r < R; r += SM90_WARPS) {
      for (int j = lane_l; j < n_s[r]; j += 32) {
        kv_indices[pos[r] + j] = r * MP + j;
        if constexpr (RP > 0) {
          ring[pos[r] + j] = r * RP + j % RP;
        }
      }
    }
    // the embedding rows: a warp a row, all of its loads in flight
    uint4 const *emb = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
    uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>());
    constexpr int NL = (Row::VEC + 31) / 32;
#pragma unroll 1
    for (int r = warp_l; r < R; r += SM90_WARPS) {
      uint4 const *row = emb + (size_t)tok_s[r] * Row::VEC;
      uint4 rv[NL];
#pragma unroll
      for (int j = 0; j < NL; j++) {
        if (lane_l + 32 * j < Row::VEC) {
          rv[j] = row[lane_l + 32 * j];
        }
      }
#pragma unroll
      for (int j = 0; j < NL; j++) {
        if (lane_l + 32 * j < Row::VEC) {
          out[(size_t)r * Row::VEC + lane_l + 32 * j] = rv[j];
        }
      }
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_next_token, 0)

} // namespace static_mk
