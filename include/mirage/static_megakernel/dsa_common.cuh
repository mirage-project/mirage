// dsa_common.cuh -- Shared helpers for the DeepSeek Sparse Attention
// (DSA) tasks of GLM-5.3 / 5.2: tasks/dsa_*.cuh and tasks/sparse_mla.cuh. The
// tasks use the GLM-5 shapes (dsa_q_absorb.cuh, dsa_o_absorb.cuh,
// dsa_index_score.cuh). This file is not a task.
// The numerics follow the vLLM decode path for GLM-5 (the model package
// vllm/models/deepseek_v32 and the file
// vllm/model_executor/layers/sparse_attn_indexer.py). There are two
// differences:
//   1. The latent cache is bf16. vLLM on SM100 uses an fp8_e4m3 cache for
//      GLM-5.
//   2. dsa_index_score adds the heads in fp32 and rounds the score to bf16 one
//      time. DeepGEMM on SM100 rounds each head term to bf16 and adds in bf16.
// The numerics:
//   GEMM outputs   The GEMM nodes add their K parts in int64 fixed point
//                  (gemm_tile.cuh COMBINE_ADD). Each reader rounds the sum to
//                  bf16 before more math, as a bf16 linear layer does.
//   RoPE           Rotate each interleaved pair (x[2i], x[2i+1]) by angle i,
//                  with fp32 math. cos_sin is a graph tensor [positions][64]
//                  bf16. Row p holds cos(p f_i) for i < 32, then sin(p f_i).
//   FP8 (indexer)  e4m3 with one power-of-two (UE8M0) scale for each 128
//                  values: s = 2^ceil(log2(max(amax, 1e-4) / 448)),
//                  q = e4m3(x / s).
#pragma once
#include "core.cuh"

namespace static_mk {
namespace dsa {

// Round v to bf16 and return it as fp32.
__device__ __forceinline__ float bf16r(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// Return the number of atomic adds to each 128-column block of the output of
// P. P is a COMBINE_ADD GEMM node (gemm_tile or scaled_gemm_tile, params
// {mode, K, N, ...}). A 128-column block of the output is 128 weight rows.
// Each task that writes the block adds one time. Thus the number is
// (K parts) x (tasks for each block).
template <class P>
__host__ __device__ constexpr int fx_block_adds() {
  return P::y * (P::x * 128 / P::v[2]);
}
// All 32 lanes of one warp call this. Wait until columns [c0, c1) of the
// output of P are complete. Lane l waits for the 128-column blocks l, l + 32,
// ... of the range, so the waits for the blocks run at the same time.
template <class P>
__device__ __forceinline__ void fx_wait_cols(G const &g, int c0, int c1) {
  static_assert(P::v[0] == COMBINE_ADD && P::counter >= 0,
                "a COMBINE_ADD GEMM node (its counter per 128-column block)");
  int const lane = threadIdx.x & 31;
  for (int b = c0 / 128 + lane; b <= (c1 - 1) / 128; b += 32) {
    cnt_wait(cnt_now(g) + P::counter + b, fx_block_adds<P>());
  }
  __syncwarp();
}
// Return column c of token t of the output of P (int64 fixed point [T][N]) as
// a bf16 value. Other SMs added the sums, so the function loads from L2.
__device__ __forceinline__ float
    fx_bf16(unsigned long long const *out, int N, int t, int c) {
  return bf16r(fx_to_float(__ldcg(out + (size_t)t * N + c)));
}

// In the timing build, thread 0 records stage stamp 0 after the barrier that
// follows the waits. This stamp shows when the inputs of the task are ready.
// In the normal build, this function does nothing.
__device__ __forceinline__ void stamp_ready(KernelLocals &L) {
#ifdef STATIC_TIMING_BUILD
  if (threadIdx.x == 0) {
    L.stage_stamps[0] = gtime();
  }
#endif
}

// Return the 16-bit key of a fp32 score that holds a bf16 value. A larger
// score gives a larger key. The 32-bit key flips all bits of a negative score
// and sets the sign bit of a positive score. The low 16 bits of the score are
// zero, so the high 16 bits of the 32-bit key keep the order.
__device__ __forceinline__ uint32_t score_key16(float s) {
  uint32_t const b = __float_as_uint(s + 0.0f); // -0 -> +0: equal scores tie
  return ((b & 0x80000000u) ? ~b : (b | 0x80000000u)) >> 16;
}
// The number of bins of a token histogram of the index scores. A bin is the
// high 12 bits of score_key16. dsa_index_score adds to the histograms, and
// dsa_index_topk reads them. static_megakernel.py DSA_TOPK_BINS has the same
// value.
constexpr int TOPK_BINS = 4096;
// Return the bytes of one index-cache page of PAGE keys. Each key has 128
// e4m3 values and 4 scale bytes (dsa_index_k_write.cuh). static_megakernel.py
// DSA_INDEX_KEY_BYTES is 128 + 4.
__host__ __device__ constexpr size_t index_page_bytes(int page) {
  return (size_t)page * (128 + 4);
}
// The latent width of MLA (kv_lora_rank): sparse_mla.cuh, dsa_mla_merge.cuh.
constexpr int MLA_LATENT = 512;

// Return the cos and sin of pair i (< 32) at position pos.
__device__ __forceinline__ void
    rope_cs(__nv_bfloat16 const *cos_sin, int pos, int i, float &c, float &s) {
  c = __bfloat162float(cos_sin[(size_t)pos * 64 + i]);
  s = __bfloat162float(cos_sin[(size_t)pos * 64 + 32 + i]);
}
// Rotate the interleaved pair (x, y) = (v[2i], v[2i+1]).
__device__ __forceinline__ void
    rope_pair(float &x, float &y, float c, float s) {
  float const x0 = x, y0 = y;
  x = x0 * c - y0 * s;
  y = y0 * c + x0 * s;
}

// Return the UE8M0 exponent byte e for a group with absolute maximum amax.
// The scale is 2^(e - 127).
__device__ __forceinline__ int ue8m0_exp(float amax) {
  float const r = __fdiv_rn(fmaxf(amax, 1e-4f), 448.0f);
  return ceil_log2_bits(r) + 127;
}
// Return 2^(e - 127) as fp32 (e in 1..254).
__device__ __forceinline__ float exp2_int(int e) {
  return __uint_as_float((uint32_t)e << 23);
}

// One warp quantizes N indexer vectors of 128 values to e4m3 with a UE8M0
// scale each. Lane l holds dims 4 l .. 4 l + 3 of vector j in v[j]. All lanes
// find the UE8M0 exponent e[j] of the 128 values (the scale is 2^(e - 127)).
// Each lane packs its 4 e4m3 bytes into the u32 q[j]. The steps of the N
// vectors run in parallel.
template <int N>
__device__ __forceinline__ void
    quant128(float const (&v)[N][4], uint32_t (&q)[N], int (&e)[N]) {
  float amax[N];
#pragma unroll
  for (int j = 0; j < N; j++) {
    amax[j] = fmaxf(fmaxf(fabsf(v[j][0]), fabsf(v[j][1])),
                    fmaxf(fabsf(v[j][2]), fabsf(v[j][3])));
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      amax[j] = fmaxf(amax[j], __shfl_xor_sync(0xffffffffu, amax[j], d));
    }
  }
#pragma unroll
  for (int j = 0; j < N; j++) {
    e[j] = ue8m0_exp(amax[j]);
    float const inv = exp2_int(254 - e[j]); // 1 / s, exact
    q[j] = (uint32_t)cvt_e4m3_nv(v[j][0] * inv) |
           ((uint32_t)cvt_e4m3_nv(v[j][1] * inv) << 8) |
           ((uint32_t)cvt_e4m3_nv(v[j][2] * inv) << 16) |
           ((uint32_t)cvt_e4m3_nv(v[j][3] * inv) << 24);
  }
}
// The same after RoPE (GLM-5.3 / 5.2): lanes l < 16 first rotate the RoPE
// pairs 2 l and 2 l + 1 (dims [0, 64)) with (c0, s0) and (c1, s1).
template <int N>
__device__ __forceinline__ void rope_quant128(float (&v)[N][4],
                                              int lane,
                                              float c0,
                                              float s0,
                                              float c1,
                                              float s1,
                                              uint32_t (&q)[N],
                                              int (&e)[N]) {
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      rope_pair(v[j][0], v[j][1], c0, s0);
      rope_pair(v[j][2], v[j][3], c1, s1);
    }
  }
  quant128<N>(v, q, e);
}
// The normalized Hadamard transform of N vectors of 128 values in one warp
// (GLM-5.3-Flash: the indexer query and the pooled key). Lane l holds dims
// 4 l .. 4 l + 3 of vector j in v[j]. The butterfly stages run in SGLang's
// order (kpool_fp8_index.py _hadamard128): strides 1 and 2 in the lane, then
// strides 4 .. 64 across lanes (lane xor 1 .. 16), then the scale
// 128^-1/2 (fp32). Each stage maps the pair (a, b) = (x[i], x[i + stride])
// to (a + b, a - b), so the fp32 sums are the same as SGLang's.
template <int N>
__device__ __forceinline__ void hadamard128(float (&v)[N][4], int lane) {
#pragma unroll
  for (int j = 0; j < N; j++) {
    float a = v[j][0], b = v[j][1];
    v[j][0] = a + b;
    v[j][1] = a - b;
    a = v[j][2], b = v[j][3];
    v[j][2] = a + b;
    v[j][3] = a - b;
    a = v[j][0], b = v[j][2];
    v[j][0] = a + b;
    v[j][2] = a - b;
    a = v[j][1], b = v[j][3];
    v[j][1] = a + b;
    v[j][3] = a - b;
  }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    bool const hi = (lane & m) != 0; // this lane holds b of each pair
#pragma unroll
    for (int j = 0; j < N; j++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float const o = __shfl_xor_sync(0xffffffffu, v[j][i], m);
        v[j][i] = hi ? o - v[j][i] : v[j][i] + o;
      }
    }
  }
#pragma unroll
  for (int j = 0; j < N; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      v[j][i] *= 0.08838834764831845f;
    }
  }
}
// Lanes l < 16 load the cos and sin of the RoPE pairs 2 l and 2 l + 1 at
// position pos (rope_quant128 uses them).
__device__ __forceinline__ void rope_cs2(__nv_bfloat16 const *cos_sin,
                                         int pos,
                                         int lane,
                                         float &c0,
                                         float &s0,
                                         float &c1,
                                         float &s1) {
  if (lane < 16) {
    rope_cs(cos_sin, pos, 2 * lane, c0, s0);
    rope_cs(cos_sin, pos, 2 * lane + 1, c1, s1);
  }
}
// Rotate RoPE pair i at position pos. The pair is columns c, c + 1 of token t
// of a COMBINE_ADD GEMM output (fx [T][N]), rounded to bf16. Write the result
// as bf16 to dst[0], dst[1] (dsa_kv_write: k_pe, dsa_q_absorb: q_pe).
__device__ __forceinline__ void fx_rope_store(unsigned long long const *fx,
                                              int N,
                                              int t,
                                              int c,
                                              __nv_bfloat16 const *cos_sin,
                                              int pos,
                                              int i,
                                              __nv_bfloat16 *dst) {
  float x = fx_bf16(fx, N, t, c), y = fx_bf16(fx, N, t, c + 1), cs, sn;
  rope_cs(cos_sin, pos, i, cs, sn);
  rope_pair(x, y, cs, sn);
  *reinterpret_cast<__nv_bfloat162 *>(dst) = __floats2bfloat162_rn(x, y);
}

// ---- shared-memory copies and MMA (sparse_mla.cuh, the absorb tasks) ----
// The ldmatrix variants:
//   load_a  x4: an m16n8k16 A fragment from row-major rows
//   load_k  x2: a B fragment from [n][k] rows
//   load_v  x4.trans: an A or B fragment from [k][m] rows
// The swizzle XORs the 16-byte chunk index with row % 8 in each 128-byte
// segment. STRIDE must be a multiple of 64 bf16.
template <int STRIDE>
__device__ __forceinline__ int swizzle(int row, int col) {
  static_assert(STRIDE % 64 == 0, "swizzle requires 128-byte row alignment");
  return row * STRIDE + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ void
    copy_async(__nv_bfloat16 *dst, void const *src, bool valid) {
  // The caller gives an in-bounds src for a masked row too. With src-size 0,
  // cp.async writes 16 zero bytes.
  unsigned const address = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address),
               "l"(src),
               "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void copy_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void copy_wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void load_a(uint32_t (&r)[4],
                                       __nv_bfloat16 const *ptr) {
  unsigned const address = __cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address)
               : "memory");
}
__device__ __forceinline__ void load_k(uint32_t (&r)[2],
                                       __nv_bfloat16 const *ptr) {
  unsigned const address = __cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(address)
               : "memory");
}
__device__ __forceinline__ void load_v(uint32_t (&r)[4],
                                       __nv_bfloat16 const *ptr) {
  unsigned const address = __cvta_generic_to_shared(ptr);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address)
      : "memory");
}
__device__ __forceinline__ void
    mma16816(uint32_t const *a, uint32_t const *b, float *c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Call this at the end of a task that writes the ring shared memory with plain
// stores. A later GEMM writes the same memory with TMA (the async proxy).
__device__ __forceinline__ void ring_release() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
}

// Return the sum of v over the 256 threads. Each thread gets the sum.
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float part[8];
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, d);
  }
  if (lane == 0) {
    part[warp] = v;
  }
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < 8; w++) {
    s += part[w];
  }
  __syncthreads(); // the next call writes part again
  return s;
}

// The 256 threads compute v[c] = x[c] * rrms * gamma[c] for c < LEN, with
// rrms = rsqrt(mean(x^2) + eps). x[c] is column c0 + c of token t of a
// COMBINE_ADD GEMM output (fx [T][N]), rounded to bf16. The math is fp32.
// Thread i takes the PER columns PER i .. PER i + PER - 1 and returns them in
// v[0 .. PER). PER is 2, 4 or 8 (default LEN / 256). With LEN / PER < 256
// (GLM-5.3-Flash q_c: LEN 1536, PER 8), threads LEN / PER .. 255 take no
// column and return zeros. The thread loads its columns with one or two
// vector loads. Thus its loads go to L2 at the same time. N and c0 must be
// multiples of PER (aligned vector loads).
template <int LEN, int PER = LEN / 256>
__device__ __forceinline__ void fx_rmsnorm_vals(unsigned long long const *fx,
                                                int N,
                                                int t,
                                                int c0,
                                                float const *gamma,
                                                float eps,
                                                float (&v)[PER]) {
  static_assert((PER == 2 || PER == 4 || PER == 8) && LEN % PER == 0 &&
                    LEN / PER <= 256,
                "fx_rmsnorm: 2, 4 or 8 columns for each thread, at most 256 "
                "threads");
  bool const active = LEN / PER == 256 || threadIdx.x < LEN / PER;
  int const cb = PER * threadIdx.x; // the first column of the thread
  unsigned long long const *src = fx + (size_t)t * N + c0 + cb;
  unsigned long long raw[PER] = {};
  if (!active) {
    // no column: zeros (raw = 0 is fixed point 0)
  } else if constexpr (PER == 2) {
    asm volatile("ld.global.cg.v2.u64 {%0, %1}, [%2];"
                 : "=l"(raw[0]), "=l"(raw[1])
                 : "l"(src)
                 : "memory");
  } else {
#pragma unroll
    for (int i = 0; i < PER; i += 4) {
      asm volatile(
          "ld.global.cg.v4.u64 {%0, %1, %2, %3}, [%4];"
          : "=l"(raw[i]), "=l"(raw[i + 1]), "=l"(raw[i + 2]), "=l"(raw[i + 3])
          : "l"(src + i)
          : "memory");
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < PER; i++) {
    v[i] = bf16r(fx_to_float(raw[i]));
    ss += v[i] * v[i];
  }
  float const rrms = rsqrtf(block_sum(ss) / LEN + eps);
  if (!active) {
    return;
  }
#pragma unroll
  for (int i = 0; i < PER; i += 2) {
    float2 const gm = *reinterpret_cast<float2 const *>(gamma + cb + i);
    v[i] *= rrms * gm.x;
    v[i + 1] *= rrms * gm.y;
  }
}
// The same, with the result as bf16 in out[0, LEN) (dsa_kv_write).
template <int LEN>
__device__ __forceinline__ void fx_rmsnorm(unsigned long long const *fx,
                                           int N,
                                           int t,
                                           int c0,
                                           float const *gamma,
                                           float eps,
                                           __nv_bfloat16 *out) {
  constexpr int PER = LEN / 256;
  float v[PER];
  fx_rmsnorm_vals<LEN>(fx, N, t, c0, gamma, eps, v);
  int const cb = PER * threadIdx.x;
#pragma unroll
  for (int i = 0; i < PER; i += 2) {
    *reinterpret_cast<__nv_bfloat162 *>(out + cb + i) =
        __floats2bfloat162_rn(v[i], v[i + 1]);
  }
}

// Compute one head of an absorb GEMM for all T = 8 tokens on tensor cores
// (dsa_q_absorb, dsa_o_absorb):
//   out[t][c] = bf16( sum_k x[t][k] w[k][c] ),  c < M, t < T, k < K
// The MMA is mma.sync m16n8k16. A is w^T (16 columns x 16 k). B is x^T (16 k x
// 8 tokens): T / 8 token tiles, each warp does all of them. The accumulators
// are fp32.
// Shared memory:
//   ws   w [K][M] bf16 (swizzle<M>)
//   xs   x [T][K] bf16 (swizzle<K>)
//   red  the partial sums of the K parts (absorb_red_bytes)
// Warp w computes column tile w % (M / 16) and K part w / (M / 16). There are
// 8 / (M / 16) K parts. Warp kp = 0 adds the parts from red. Row t of out is
// at out + t * ld_out. All 256 threads must call this function. Before the
// call, make ws and xs ready (cp.async wait, then __syncthreads).
template <int M>
__host__ __device__ constexpr int absorb_red_bytes() {
  return (8 / (M / 16) - 1) * (M / 16) * (T / 8) * 32 * 4 * 4;
}
template <int K, int M>
__device__ __forceinline__ void absorb_mma(__nv_bfloat16 const *ws,
                                           __nv_bfloat16 const *xs,
                                           float *red,
                                           __nv_bfloat16 *out,
                                           size_t ld_out) {
  constexpr int MT = M / 16, KP = 8 / MT, KK = K / KP, NTT = T / 8;
  static_assert(T % 8 == 0 && M % 16 == 0 && 8 % MT == 0 && KK % 16 == 0,
                "absorb_mma: whole 16 x 16 tiles, 8-token tiles");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const mt = warp % MT, kp = warp / MT;
  float d[NTT][4];
#pragma unroll
  for (int nt = 0; nt < NTT; nt++) {
    d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;
  }
#pragma unroll 4
  for (int k = kp * KK; k < kp * KK + KK; k += 16) {
    uint32_t a[4];
    load_v(a,
           ws + swizzle<M>(k + (lane & 7) + (lane >> 4) * 8,
                           mt * 16 + ((lane >> 3) & 1) * 8));
#pragma unroll
    for (int nt = 0; nt < NTT; nt++) {
      uint32_t b[2];
      load_k(b,
             xs + swizzle<K>(nt * 8 + (lane & 7), k + ((lane >> 3) & 1) * 8));
      mma16816(a, b, d[nt]);
    }
  }
  if constexpr (KP > 1) {
    if (kp > 0) {
#pragma unroll
      for (int nt = 0; nt < NTT; nt++) {
        *reinterpret_cast<float4 *>(
            red + ((((kp - 1) * MT + mt) * NTT + nt) * 32 + lane) * 4) =
            make_float4(d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
      }
    }
    __syncthreads();
    if (kp == 0) {
#pragma unroll
      for (int p = 1; p < KP; p++) {
#pragma unroll
        for (int nt = 0; nt < NTT; nt++) {
          float4 const r = *reinterpret_cast<float4 const *>(
              red + ((((p - 1) * MT + mt) * NTT + nt) * 32 + lane) * 4);
          d[nt][0] += r.x;
          d[nt][1] += r.y;
          d[nt][2] += r.z;
          d[nt][3] += r.w;
        }
      }
    }
  }
  if (kp == 0) {
    // of token tile nt: d[0] and d[1] hold column mt 16 + lane / 4 of the
    // tokens 2 (lane % 4) and 2 (lane % 4) + 1. d[2] and d[3] hold the column
    // + 8 of the same tokens.
    int const c = mt * 16 + (lane >> 2);
#pragma unroll
    for (int nt = 0; nt < NTT; nt++) {
      int const t = nt * 8 + 2 * (lane & 3);
      out[(size_t)t * ld_out + c] = __float2bfloat16(d[nt][0]);
      out[(size_t)(t + 1) * ld_out + c] = __float2bfloat16(d[nt][1]);
      out[(size_t)t * ld_out + c + 8] = __float2bfloat16(d[nt][2]);
      out[(size_t)(t + 1) * ld_out + c + 8] = __float2bfloat16(d[nt][3]);
    }
  }
}
// Copy the weight block w[k][c] (k < K, c < M, row stride ldw) to ws
// (swizzle<M>) with cp.async. Each thread commits one cp.async group.
template <int K, int M>
__device__ __forceinline__ void
    absorb_load_w(__nv_bfloat16 *ws, __nv_bfloat16 const *w, int ldw) {
  constexpr int CH = M / 8; // 16-byte chunks in a row
  for (int i = threadIdx.x; i < K * CH; i += 256) {
    int const k = i / CH, c = (i % CH) * 8;
    copy_async(ws + swizzle<M>(k, c), w + (size_t)k * ldw + c, true);
  }
  copy_commit();
}

} // namespace dsa
} // namespace static_mk
