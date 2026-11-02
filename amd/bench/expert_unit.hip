// Unit kernels for moe_experts on gfx950 (amd/notes/moe_experts_gfx950.md) at K3's sizes: W13 + SiTU + MXFP8 quantization,
// then W2; their read speed and their values against a CPU reference.
//
// E experts (default 80, about the number 8 tokens touch per step), MXFP4 weights in the gfx950 tile layout
// (amd/tools/shuffle_weights.py: [row tile][K step][64 lanes][16 B]; scales W13 [row tile][K steps / 4][64 lanes][4 B],
// W2 [row tile / 4][64 lanes][4 row tiles x K steps] bytes):
//   W13  768 rows (item order: item i = gate rows 32i..32i+31, then up rows 32i..32i+31) x K = 3584
//   W2   3584 rows x K = 384
// Weights are operand A (FP4, 16 rows), tokens operand B (FP8 e4m3, 16 columns of which 8 are tokens) of
// v_mfma_scale_f32_16x16x128_f8f6f4; the D layout: lane -> token l % 16, register r -> row 4 (l / 16) + r.
//   w13_kernel  one wave = one item (4 row tiles x 28 K steps), weight fragments STAGES K steps ahead in registers, the tokens
//               z_q in LDS; then h = SiTU(gate, up) for its 32 features x 8 tokens -> h (fp32, for the check) and
//               h_q = MXFP8(h) (one scale per token per 32 features: the item), written in the fragment order W2 reads,
//               compact as the task (moe_experts.cuh): hq [expert][3 K steps][4 groups x 8 tokens][32 B],
//               hsc [expert][3 K steps][64 lanes]
//   w2_kernel   one wave = items of W2_TILES row tiles x 3 K steps, the next item's fragments loaded during the current one;
//               y [expert][8 tokens][3584] fp32 (routing weight 1)
// Layout choices (compile-time, from amd/bench/mfma_layout_check.hip):
//   FP8_LIN  0: the fp8 operand's 32 bytes per lane = K 16g..16g+15, 64+16g..64+16g+15 (ISA figure); 1: K 32g..32g+31
//   FP4_HI   0: the lower K in the low nibble; 1: in the high nibble
// Variants: STAGES (W13 K steps in flight per wave: 1, 2, 4, 7, 14), HINT (weight loads: 0 plain, 1 nt),
//           W2_TILES (row tiles per W2 item: 1, 2, 4, 7, 8, 14, 16, 28, 32, 56).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 [-DSTAGES=4 -DHINT=0 -DW2_TILES=8 -DFP8_LIN=0 -DFP4_HI=0] expert_unit.hip
// Run:   ./expert_unit [experts=80] [reps=20]     -> times, read speeds, errors; JSON summary on the last line
#include <cmath>
#include <cstring>
#include <random>

#include "common.h"

// W13_ILV: a W13 item's 4 row tiles interleaved per K step ([item][K step][4 tiles][64][16 B]: a wave's 4 loads of one step =
// one 4 KB block) instead of tile after tile ([item][4 tiles][K step][64][16 B]: 4 blocks 28 KB apart); the same bytes
// SCHED: 1 = a scheduling barrier after each weight load, so the compiler keeps the loads in source order (the prologue's
// order = the loop's: vmcnt counts in issue order, and with the prologue reordered the loop's waits drained to ~9 loads in
// flight at the start of every round)
#ifndef SCHED
#define SCHED 0
#endif
#define SCHED_FENCE() do { if (SCHED) __builtin_amdgcn_sched_barrier(0); } while (0)
#ifndef W13_ILV
#define W13_ILV 0
#endif
constexpr int w13_at(int t, int s, int ks1) { return W13_ILV ? s * 4 + t : t * ks1 + s; }
#ifndef STAGES
#define STAGES 4
#endif
#ifndef HINT
#define HINT 0
#endif
// W2_NO_SMALL: an experiment, W2 without its h_q, h_q scale and weight scale loads (constants): only the weight stream
#ifndef W2_NO_SMALL
#define W2_NO_SMALL 0
#endif
// the layer's W2 differences, one switch each (10-11): W2_HQ_SC1 h_q and its scales loaded at device scope (sc1, as the
// layer's 0xFF poll); W2_Y_SC1 the results stored at device scope (sc1, as the layer's R); W2_QUEUE items taken from a
// per-die queue with an atomic add (the next one's ticket taken before the current item's loads), not dealt by index
typedef unsigned v4u_t __attribute__((ext_vector_type(4)));
#ifndef W2_HQ_SC1
#define W2_HQ_SC1 0
#endif
#ifndef W2_Y_SC1
#define W2_Y_SC1 0
#endif
#ifndef W2_QUEUE
#define W2_QUEUE 0
#endif
#ifndef W2_TILES
#define W2_TILES 8
#endif
#ifndef FP8_LIN
#define FP8_LIN 0
#endif
#ifndef FP4_HI
#define FP4_HI 0
#endif

constexpr int T = 8, LAT = 3584, IR = 384;
// W13: rows 2 IR, K LAT
constexpr int K1 = LAT, KS1 = K1 / 128, KG1 = KS1 / 4, RT1 = 2 * IR / 16, ITEMS1 = IR / 32;
// W2: rows LAT, K IR
constexpr int KS2 = IR / 128, RT2 = LAT / 16, ITEMS2 = RT2 / W2_TILES;
static_assert(KS1 % 4 == 0 && KS1 % STAGES == 0 && KS1 > STAGES, "W13: whole scale groups, whole rounds of STAGES K steps");
static_assert(RT2 % W2_TILES == 0 && W2_TILES % 4 == 0, "W2: whole items of whole 4-tile scale groups");

typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));

// HINT 0: plain loads, 1: nt. (A third, sc1 nt by inline asm, is gone: an asm load without a wait inside is not tracked by the
// compiler, which reuses its destination registers before the data lands. read_bw measures sc1 nt with buffer loads.)
__device__ inline v4i load_frag(const v4i *p) {
#if HINT == 0
  return *p;
#elif HINT == 1
  return __builtin_nontemporal_load(p);
#else
#error "HINT is 0 or 1"
#endif
}

// where feature k (0..127) of a K step sits in the fp8 operand: lane group g (lanes 16g..16g+15) and byte b of the lane's 32
__host__ __device__ inline void fp8_slot(int k, int &g, int &b) {
#if FP8_LIN
  g = k / 32;
  b = k % 32;
#else
  if (k < 64) {
    g = k / 16;
    b = k % 16;
  } else {
    g = (k - 64) / 16;
    b = 16 + (k - 64) % 16;
  }
#endif
}

__device__ inline int ceil_log2_bits(float x) {
  uint32_t const b = __float_as_uint(x);
  return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
}
__device__ inline uint8_t cvt_e4m3(float v) {
  float const c = __builtin_amdgcn_fmed3f(v, -448.0f, 448.0f);
  return (uint8_t)(__builtin_amdgcn_cvt_pk_fp8_f32(c, 0.0f, 0, false) & 0xFF);
}

// ---- W13 ----
// LDS: the tokens' fragments [KS1][64 lanes][32 B] and scales [KS1][64 lanes], then per wave its item's weight scales
// [4 tiles][KG1][64 lanes] dwords (run-time indexed: in LDS, not in registers)
// WAVES: W13 waves per workgroup (one workgroup per CU): 4 (one wave per SIMD) or 8 (two per SIMD: then each wave gets at most
// 256 registers in all)
#ifndef WAVES
#define WAVES 4
#endif
constexpr int LDS_Z = KS1 * 64 * 32, LDS_ZS = KS1 * 64, LDS_WS = 4 * KG1 * 64 * 4, LDS1 = LDS_Z + LDS_ZS + WAVES * LDS_WS;

// one K step of a W13 item: 4 MFMAs (gate 2, up 2 row tiles) on this step's fragments and the tokens' fragment in LDS
__device__ inline void w13_step(v4f *acc, const v4i *frag, const v8i *z_l, const unsigned char *zs_l, const unsigned *ws_l,
                                int ks, int lane) {
  v8i const b = z_l[ks * 64 + lane];
  int const bs = zs_l[ks * 64 + lane];
#pragma unroll
  for (int t = 0; t < 4; t++) {
    v8i a = {frag[t][0], frag[t][1], frag[t][2], frag[t][3], 0, 0, 0, 0};
    int const as = (ws_l[(t * KG1 + ks / 4) * 64 + lane] >> (8 * (ks % 4))) & 0xFF;
    acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc[t], 4 /*A fp4*/, 0 /*B fp8*/, 0, as, 0, bs);
  }
}

__global__ void __launch_bounds__(64 * WAVES, 1)
    w13_kernel(const v4i *w, const unsigned *wsc, const v8i *zfrag, const unsigned char *zsc, float *h, unsigned char *hq,
               unsigned char *hsc, int items) {
  extern __shared__ __attribute__((aligned(16))) char lds[];
  v8i *z_l = reinterpret_cast<v8i *>(lds);
  unsigned char *zs_l = reinterpret_cast<unsigned char *>(lds + LDS_Z);
  for (int i = threadIdx.x; i < LDS_Z / 32; i += blockDim.x) z_l[i] = zfrag[i];
  for (int i = threadIdx.x; i < LDS_ZS; i += blockDim.x) zs_l[i] = zsc[i];
  __syncthreads();
  int const lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  unsigned *ws_l = reinterpret_cast<unsigned *>(lds + LDS_Z + LDS_ZS + wave * LDS_WS);  // [4][KG1][64]
  for (int item = blockIdx.x * WAVES + wave; item < items; item += gridDim.x * WAVES) {
    int const e = item / ITEMS1, it = item % ITEMS1;
    const v4i *wt = w + ((size_t)e * RT1 + 4 * it) * KS1 * 64 + lane;          // + (tile * KS1 + ks) * 64
    const unsigned *st = wsc + ((size_t)e * RT1 + 4 * it) * KG1 * 64 + lane;   // + (tile * KG1 + kg) * 64
    v4f acc[4] = {};
    // the item's weight scales first: the K loop then waits only on weight loads (vmcnt counts in order). All their loads,
    // then the first STAGES steps' weight loads, then the scales' LDS stores (as the layer: tasks/moe_experts.cuh)
    unsigned sreg[4 * KG1];
#pragma unroll
    for (int i = 0; i < 4 * KG1; i++) sreg[i] = __builtin_nontemporal_load(st + (size_t)i * 64);
    v4i frag[STAGES][4];
#pragma unroll
    for (int s = 0; s < STAGES; s++)
#pragma unroll
      for (int t = 0; t < 4; t++) {
        frag[s][t] = load_frag(wt + (size_t)w13_at(t, s, KS1) * 64);
        SCHED_FENCE();
      }
    asm volatile("" ::: "memory");
#pragma unroll
    for (int i = 0; i < 4 * KG1; i++) ws_l[i * 64 + lane] = sreg[i];
    // K steps in rounds of STAGES: the inner loop unrolled (frag[s] by constant index), the outer one not. The last round is
    // peeled off, so the rounds before it prefetch without a branch: a branch around the prefetch makes the compiler wait for
    // every outstanding load at the top of each round (vmcnt(0)), and the stream stops once per round.
#pragma unroll 1
    for (int k0 = 0; k0 < KS1 - STAGES; k0 += STAGES) {
#pragma unroll
      for (int s = 0; s < STAGES; s++) {
        w13_step(acc, frag[s], z_l, zs_l, ws_l, k0 + s, lane);
#pragma unroll
        for (int t = 0; t < 4; t++) {
          frag[s][t] = load_frag(wt + (size_t)w13_at(t, k0 + s + STAGES, KS1) * 64);
          SCHED_FENCE();
        }
      }
    }
#pragma unroll
    for (int s = 0; s < STAGES; s++) w13_step(acc, frag[s], z_l, zs_l, ws_l, KS1 - STAGES + s, lane);
    // h: tiles 0, 1 gate, 2, 3 up; this lane holds token l % 16, features 16 half + 4 (l / 16) + r
    int const tok = lane % 16;
    float hv[2][4];
    float amax = 0.f;
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        float const g = acc[half][r], u = acc[2 + half][r];
        hv[half][r] = 4.0f * tanhf(g * 0.25f) * (1.0f / (1.0f + __expf(-g))) * (25.0f * tanhf(u * 0.04f));
        amax = fmaxf(amax, fabsf(hv[half][r]));
      }
    // the token's 32 features are in lanes tok, tok + 16, tok + 32, tok + 48
    amax = fmaxf(amax, __shfl_xor(amax, 16));
    amax = fmaxf(amax, __shfl_xor(amax, 32));
    int ex = ceil_log2_bits(fmaxf(amax, 1.0e-30f) * (1.0f / 448.0f));
    ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
    float const inv = __uint_as_float((uint32_t)(127 - ex) << 23);
    if (tok < T) {
      int const ks2 = it / 4, q = it % 4;  // W2's K step and 32-feature block of this item
      unsigned char *hq_e = hq + ((size_t)e * KS2) * 4 * T * 32, *hsc_e = hsc + (size_t)e * KS2 * 64;
#pragma unroll
      for (int half = 0; half < 2; half++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const f = 16 * half + 4 * (lane / 16) + r;
          h[((size_t)item * T + tok) * 32 + f] = hv[half][r];
          int g, b;
          fp8_slot(32 * q + f, g, b);
          hq_e[((size_t)ks2 * 4 * T + T * g + tok) * 32 + b] = cvt_e4m3(hv[half][r] * inv);
        }
      if (lane < 16) hsc_e[ks2 * 64 + 16 * q + tok] = (unsigned char)(ex + 127);
    }
  }
}

// W13_CU 1: a workgroup (CU) takes one item at a time, dealt by index (item = block, block + grid, ...: ~4.2 per CU at 90
// experts); wave w computes the item's row tile w over all K steps, every K step's load issued at once, and the next item's
// loads are issued before the current one is computed (two items in flight per CU); the 4 tiles meet in LDS, wave 0 makes h
#ifndef W13_CU
#define W13_CU 0
#endif
// W13_SM 1 (with W13_CU): each expert's W13 fragments step-major, [e][KS1][RT1][64 lanes][16 B] (tile-major: [e][RT1][KS1]..):
// the 12 CUs on an expert's 12 items read one 48 KB window per K step, not 48 separate 28 KB streams
#ifndef W13_SM
#define W13_SM 0
#endif
__global__ void __launch_bounds__(256, 1)
    w13cu_kernel(const v4i *w, const unsigned *wsc, const v8i *zfrag, const unsigned char *zsc, float *h, unsigned char *hq,
                 unsigned char *hsc, int items) {
  extern __shared__ __attribute__((aligned(16))) char lds[];
  v8i *z_l = reinterpret_cast<v8i *>(lds);
  unsigned char *zs_l = reinterpret_cast<unsigned char *>(lds + LDS_Z);
  v4f *xch = reinterpret_cast<v4f *>(lds + LDS_Z + LDS_ZS);   // [4][64]
  for (int i = threadIdx.x; i < LDS_Z / 32; i += blockDim.x) z_l[i] = zfrag[i];
  for (int i = threadIdx.x; i < LDS_ZS; i += blockDim.x) zs_l[i] = zsc[i];
  __syncthreads();
  int const lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  struct Tile { v4i f[KS1]; unsigned sc[KG1]; };
  auto issue = [&](Tile &F, int item) {
    int const e = item / ITEMS1, it = item % ITEMS1;
    size_t const rt = (size_t)e * RT1 + 4 * it + wave;
    const unsigned *st = wsc + rt * KG1 * 64 + lane;
#pragma unroll
    for (int g = 0; g < KG1; g++) F.sc[g] = __builtin_nontemporal_load(st + (size_t)g * 64);
    if (W13_SM) {
      const v4i *we = w + (size_t)e * KS1 * RT1 * 64 + (size_t)(4 * it + wave) * 64 + lane;
#pragma unroll
      for (int k = 0; k < KS1; k++) F.f[k] = load_frag(we + (size_t)k * RT1 * 64);
    } else {
      const v4i *wt = w + rt * KS1 * 64 + lane;
#pragma unroll
      for (int k = 0; k < KS1; k++) F.f[k] = load_frag(wt + (size_t)k * 64);
    }
  };
  auto compute = [&](Tile const &F) {
    v4f acc = {0, 0, 0, 0};
#pragma unroll
    for (int ks = 0; ks < KS1; ks++) {
      v8i const b = z_l[ks * 64 + lane];
      int const bs = zs_l[ks * 64 + lane];
      v8i a = {F.f[ks][0], F.f[ks][1], F.f[ks][2], F.f[ks][3], 0, 0, 0, 0};
      int const as = (F.sc[ks / 4] >> (8 * (ks % 4))) & 0xFF;
      acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, 4, 0, 0, as, 0, bs);
    }
    return acc;
  };
  auto finish = [&](int item) {   // wave 0: h of the item from the 4 tiles in xch
    int const e = item / ITEMS1, it = item % ITEMS1, tok = lane % 16;
    float hv[2][4], amax = 0.f;
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        float const g = xch[half * 64 + lane][r], u = xch[(2 + half) * 64 + lane][r];
        hv[half][r] = 4.0f * tanhf(g * 0.25f) * (1.0f / (1.0f + __expf(-g))) * (25.0f * tanhf(u * 0.04f));
        amax = fmaxf(amax, fabsf(hv[half][r]));
      }
    amax = fmaxf(amax, __shfl_xor(amax, 16));
    amax = fmaxf(amax, __shfl_xor(amax, 32));
    int ex = ceil_log2_bits(fmaxf(amax, 1.0e-30f) * (1.0f / 448.0f));
    ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
    float const inv = __uint_as_float((uint32_t)(127 - ex) << 23);
    if (tok < T) {
      int const ks2 = it / 4, q = it % 4;
      unsigned char *hq_e = hq + ((size_t)e * KS2) * 4 * T * 32, *hsc_e = hsc + (size_t)e * KS2 * 64;
#pragma unroll
      for (int half = 0; half < 2; half++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const f = 16 * half + 4 * (lane / 16) + r;
          h[((size_t)item * T + tok) * 32 + f] = hv[half][r];
          int g, b;
          fp8_slot(32 * q + f, g, b);
          hq_e[((size_t)ks2 * 4 * T + T * g + tok) * 32 + b] = cvt_e4m3(hv[half][r] * inv);
        }
      if (lane < 16) hsc_e[ks2 * 64 + 16 * q + tok] = (unsigned char)(ex + 127);
    }
  };
  Tile A, B;
  int ia = blockIdx.x, stride = gridDim.x;
  if (ia >= items) return;
  issue(A, ia);
  auto step = [&](Tile &cur, Tile &nxt, int ic) {
    if (ic + stride < items) issue(nxt, ic + stride);
    v4f const acc = compute(cur);
    __syncthreads();   // the last item's xch read by wave 0
    xch[wave * 64 + lane] = acc;
    __syncthreads();
    if (wave == 0) finish(ic);
  };
#pragma unroll 1
  for (;;) {
    step(A, B, ia);
    ia += stride;
    if (ia >= items) break;
    step(B, A, ia);
    ia += stride;
    if (ia >= items) break;
  }
}

// ---- W2 ----
// one item: the expert's h_q fragments and scales (operand B), then W2_TILES row tiles x KS2 K steps of weight fragments and
// their scale bytes. All of an item's loads are issued together by w2_load, h_q first: vmcnt counts in order, so a load the
// compute issued itself would make it wait for the next item's prefetch too.
struct W2Frags {
  v8i b[KS2];
  int bs[KS2];
  v4i f[W2_TILES][KS2];
  unsigned sc[W2_TILES / 4][KS2];  // byte KS2 (tile % 4) + k
};

__device__ inline void w2_load(W2Frags &F, const v4i *w, const unsigned char *wsc, const v8i *hq, const unsigned char *hsc, int item,
                               int lane) {
  int const e = item / ITEMS2, t0 = (item % ITEMS2) * W2_TILES;
  const v4i *wt = w + ((size_t)e * RT2 + t0) * KS2 * 64 + lane;
  const unsigned *st = reinterpret_cast<const unsigned *>(wsc) + (((size_t)e * RT2 + t0) / 4 * 64 + lane) * KS2;
  int const col = lane % 16;
#pragma unroll
  for (int k = 0; k < KS2; k++) {
    F.b[k] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    if (W2_NO_SMALL) {   // experiment: no h_q / scale loads (constants), only the weight loads
      F.b[k] = v8i{0x38383838, 0x38383838, 0, 0, 0, 0, 0, 0};
      F.bs[k] = 127;
      continue;
    }
    if (W2_HQ_SC1) {
      __amdgpu_buffer_rsrc_t const rh = __builtin_amdgcn_make_buffer_rsrc(const_cast<v8i *>(hq), 0, 0x7FFFFFFF, 0x00020000),
                                   rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<unsigned char *>(hsc), 0, 0x7FFFFFFF, 0x00020000);
      if (col < T) {
        unsigned const off = (unsigned)((((size_t)e * KS2 + k) * 4 * T + T * (lane / 16) + col) * 32);
        auto const lo = __builtin_amdgcn_raw_buffer_load_b128(rh, off, 0, 16), hi = __builtin_amdgcn_raw_buffer_load_b128(rh, off + 16, 0, 16);
        F.b[k] = v8i{(int)lo[0], (int)lo[1], (int)lo[2], (int)lo[3], (int)hi[0], (int)hi[1], (int)hi[2], (int)hi[3]};
      }
      F.bs[k] = __builtin_amdgcn_raw_buffer_load_b8(rs, (unsigned)(((size_t)e * KS2 + k) * 64 + lane), 0, 16);
      continue;
    }
    if (col < T) F.b[k] = hq[((size_t)e * KS2 + k) * 4 * T + T * (lane / 16) + col];
    F.bs[k] = hsc[((size_t)e * KS2 + k) * 64 + lane];
  }
#pragma unroll
  for (int q = 0; q < W2_TILES / 4; q++)
#pragma unroll
    for (int d = 0; d < KS2; d++) F.sc[q][d] = W2_NO_SMALL ? 0x7f7f7f7fu : st[(size_t)q * 64 * KS2 + d];
#pragma unroll
  for (int t = 0; t < W2_TILES; t++)
#pragma unroll
    for (int k = 0; k < KS2; k++) F.f[t][k] = load_frag(wt + ((size_t)t * KS2 + k) * 64);
}

__device__ inline void w2_compute(W2Frags const &F, float *y, int item, int lane) {
  int const e = item / ITEMS2, t0 = (item % ITEMS2) * W2_TILES;
  int const tok = lane % 16;
#pragma unroll
  for (int t = 0; t < W2_TILES; t++) {
    v4f acc = {0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < KS2; k++) {
      v8i a = {F.f[t][k][0], F.f[t][k][1], F.f[t][k][2], F.f[t][k][3], 0, 0, 0, 0};
      int const sb = KS2 * (t % 4) + k;
      acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, F.b[k], acc, 4, 0, 0, (F.sc[t / 4][sb / 4] >> (8 * (sb % 4))) & 0xFF,
                                                             0, F.bs[k]);
    }
    if (tok < T) {
      float *const yp = &y[((size_t)e * T + tok) * LAT + (t0 + t) * 16 + 4 * (lane / 16)];
      if (W2_Y_SC1) {
        v4u_t const w = {__float_as_uint(acc[0]), __float_as_uint(acc[1]), __float_as_uint(acc[2]), __float_as_uint(acc[3])};
        asm volatile("global_store_dwordx4 %0, %1, off sc1" ::"v"(yp), "v"(w) : "memory");
      } else {
#pragma unroll
        for (int r = 0; r < 4; r++) yp[r] = acc[r];
      }
    }
  }
}

__global__ void __launch_bounds__(256, 1)
    w2_kernel(const v4i *w, const unsigned char *wsc, const v8i *hq, const unsigned char *hsc, float *y, int items, unsigned *qc) {
  int const lane = threadIdx.x & 63, stride = gridDim.x * 4;
  if (W2_QUEUE) {   // per-die queue: die x's items [items x / 8, items (x + 1) / 8), the next ticket taken ahead
    int const die = (int)xcc_id(), lo = items * die / 8, hi = items * (die + 1) / 8;
    auto issue = [&]() {
      int v = 0;
      if (lane == 0) v = (int)atomicAdd(qc + 32 * die, 1u);
      return v;
    };
    auto finish = [&](int v) {
      v = __builtin_amdgcn_readfirstlane(v);
      return lo + v < hi ? lo + v : -1;
    };
    int it = finish(issue());
    if (it < 0) return;
    W2Frags A, B;
    int tk = issue();
    w2_load(A, w, wsc, hq, hsc, it, lane);
#pragma unroll 1
    for (;;) {
      int const nb = finish(tk);
      if (nb < 0) {
        w2_compute(A, y, it, lane);
        return;
      }
      tk = issue();
      w2_load(B, w, wsc, hq, hsc, nb, lane);
      w2_compute(A, y, it, lane);
      it = nb;
      int const na = finish(tk);
      if (na < 0) {
        w2_compute(B, y, it, lane);
        return;
      }
      tk = issue();
      w2_load(A, w, wsc, hq, hsc, na, lane);
      w2_compute(B, y, it, lane);
      it = na;
    }
  }
  int item = blockIdx.x * 4 + (threadIdx.x >> 6);
  // two items in flight: the next one's fragments load while the current one computes (A and B alternate, unrolled by hand so
  // both are indexed by constants)
  if (item >= items) return;
  W2Frags A, B;
  w2_load(A, w, wsc, hq, hsc, item, lane);
#pragma unroll 1
  for (;;) {
    if (item + stride >= items) break;
    w2_load(B, w, wsc, hq, hsc, item + stride, lane);
    w2_compute(A, y, item, lane);
    item += stride;
    if (item + stride >= items) {
      w2_compute(B, y, item, lane);
      return;
    }
    w2_load(A, w, wsc, hq, hsc, item + stride, lane);
    w2_compute(B, y, item, lane);
    item += stride;
  }
  w2_compute(A, y, item, lane);
}

// ---- host ----
static double e4m3(uint8_t v) {
  int s = v >> 7, ex = (v >> 3) & 15, m = v & 7;
  double x = ex ? std::ldexp(1.0 + m / 8.0, ex - 7) : std::ldexp(m / 8.0, -6);
  return s ? -x : x;
}
static double e2m1(uint8_t v) {
  static const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (v & 8) ? -t[v & 7] : t[v & 7];
}
// the round-to-nearest-even e4m3 of x (|x| <= 448), as a byte, for the h_q check
static uint8_t to_e4m3(double x) {
  uint8_t best = 0;
  double bd = 1e300;
  for (int v = 0; v < 256; v++) {
    if ((v & 0x7F) == 0x7F) continue;
    double d = std::fabs(e4m3(v) - x);
    if (d < bd || (d == bd && (v & 1) == 0 && (best & 1))) bd = d, best = v;
  }
  if (x == 0) best = std::signbit(x) ? 0x80 : 0;
  return best;
}

// logical [rows][K] nibbles + [rows][K/32] scales -> the tile layout ([rt][ks][64][16 B]; scales: ks % 4 == 0 [rt][ks/4][64]
// dwords, else [rt/4][64][4 ks] bytes), as amd/tools/shuffle_weights.py
static void shuffle(const uint8_t *q, const uint8_t *s, int rows, int k, uint8_t *tiles, void *scv) {
  int rt = rows / 16, ks = k / 128, kg = ks / 4;
  uint32_t *sc = (uint32_t *)scv;
  uint8_t *sb = (uint8_t *)scv;
  if (ks % 4 == 0)
    for (int r = 0; r < rt; r++)
      for (int g = 0; g < kg; g++)
        for (int l = 0; l < 64; l++) sc[((size_t)r * kg + g) * 64 + l] = 0;
  for (int r = 0; r < rt; r++)
    for (int st = 0; st < ks; st++)
      for (int l = 0; l < 64; l++) {
        size_t row = (size_t)r * 16 + l % 16;
        uint8_t *dst = &tiles[(((size_t)r * ks + st) * 64 + l) * 16];
        for (int b = 0; b < 16; b++) {
          uint8_t lo = q[row * k + st * 128 + 32 * (l / 16) + 2 * b], hi = q[row * k + st * 128 + 32 * (l / 16) + 2 * b + 1];
          dst[b] = FP4_HI ? (hi | lo << 4) : (lo | hi << 4);
        }
        uint8_t const sv = s[row * (k / 32) + 4 * st + l / 16];
        if (ks % 4 == 0) {
          sc[((size_t)r * kg + st / 4) * 64 + l] |= (uint32_t)sv << (8 * (st % 4));
        } else {
          sb[(((size_t)(r / 4)) * 64 + l) * (4 * ks) + ks * (r % 4) + st] = sv;
        }
      }
}

int main(int argc, char **argv) {
  int experts = argc > 1 ? atoi(argv[1]) : 80, reps = argc > 2 ? atoi(argv[2]) : 20;
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  printf("W13_SM %d ", W13_SM);
  printf("W13_CU %d W2_HQ_SC1 %d W2_Y_SC1 %d W2_QUEUE %d; ", W13_CU, W2_HQ_SC1, W2_Y_SC1, W2_QUEUE);
  printf("STAGES %d HINT %d W2_TILES %d FP8_LIN %d FP4_HI %d WAVES %d, %d experts\n", STAGES, HINT, W2_TILES, FP8_LIN, FP4_HI,
         WAVES, experts);
  std::mt19937 rng(3);
  // logical data: W13 q1 [e][768][3584] nibbles, s1 scales; W2 q2 [e][3584][384], s2; tokens z [8][3584] e4m3, zs
  size_t n1 = (size_t)experts * 2 * IR * K1, n2 = (size_t)experts * LAT * IR;
  std::vector<uint8_t> q1(n1), s1(n1 / 32), q2(n2), s2(n2 / 32), z(T * K1), zs(T * (K1 / 32));
  for (auto &v : q1) v = rng() & 0xF;
  for (auto &v : s1) v = 118 + rng() % 6;
  for (auto &v : q2) v = rng() & 0xF;
  for (auto &v : s2) v = 118 + rng() % 6;
  for (auto &v : z) {
    do v = rng() & 0xFF;
    while ((v & 0x7F) == 0x7F);
  }
  for (auto &v : zs) v = 118 + rng() % 6;
  std::vector<uint8_t> w1((size_t)experts * RT1 * KS1 * 64 * 16), w2((size_t)experts * RT2 * KS2 * 64 * 16);
  std::vector<uint32_t> sc1((size_t)experts * RT1 * KG1 * 64);
  std::vector<uint8_t> sc2((size_t)experts * RT2 * KS2 * 64);
  for (int e = 0; e < experts; e++) {
    shuffle(&q1[(size_t)e * 2 * IR * K1], &s1[(size_t)e * 2 * IR * K1 / 32], 2 * IR, K1, &w1[(size_t)e * RT1 * KS1 * 1024],
            &sc1[(size_t)e * RT1 * KG1 * 64]);
    shuffle(&q2[(size_t)e * LAT * IR], &s2[(size_t)e * LAT * IR / 32], LAT, IR, &w2[(size_t)e * RT2 * KS2 * 1024],
            &sc2[(size_t)e * RT2 * KS2 * 64]);
  }
  if (W13_SM) {   // each expert's [RT1][KS1] 1-KB fragments -> [KS1][RT1]
    std::vector<uint8_t> tmp((size_t)RT1 * KS1 * 1024);
    for (int e = 0; e < experts; e++) {
      uint8_t *b = &w1[(size_t)e * RT1 * KS1 * 1024];
      for (int t = 0; t < RT1; t++)
        for (int ks = 0; ks < KS1; ks++) memcpy(&tmp[((size_t)ks * RT1 + t) * 1024], &b[((size_t)t * KS1 + ks) * 1024], 1024);
      memcpy(b, tmp.data(), tmp.size());
    }
  }
  if (W13_ILV) {   // each item's 4 tiles x KS1 fragments of 1 KB: tile-major -> step-major
    std::vector<uint8_t> tmp((size_t)4 * KS1 * 1024);
    for (size_t item = 0; item < (size_t)experts * RT1 / 4; item++) {
      uint8_t *b = &w1[item * 4 * KS1 * 1024];
      for (int t = 0; t < 4; t++)
        for (int ks = 0; ks < KS1; ks++) memcpy(&tmp[(size_t)w13_at(t, ks, KS1) * 1024], &b[((size_t)t * KS1 + ks) * 1024], 1024);
      memcpy(b, tmp.data(), tmp.size());
    }
  }
  // the tokens' fragments: lane l = token l % 16 (8..15 zero), group g
  std::vector<uint8_t> zf(LDS_Z, 0), zsf(LDS_ZS, 127);
  for (int ks = 0; ks < KS1; ks++)
    for (int k = 0; k < 128; k++)
      for (int tok = 0; tok < T; tok++) {
        int g, b;
        fp8_slot(k, g, b);
        zf[(ks * 64 + 16 * g + tok) * 32 + b] = z[tok * K1 + ks * 128 + k];
        if (k % 32 == 0) zsf[ks * 64 + 16 * (k / 32) + tok] = zs[tok * (K1 / 32) + 4 * ks + k / 32];
      }
  int items1 = experts * ITEMS1, items2 = experts * ITEMS2;
  v4i *dw1, *dw2;
  unsigned *dsc1;
  unsigned char *dsc2;
  v8i *dz;
  unsigned char *dzs, *dhq, *dhsc;
  float *dh, *dy;
  CHECK(hipMalloc(&dw1, w1.size()));
  CHECK(hipMalloc(&dw2, w2.size()));
  CHECK(hipMalloc(&dsc1, sc1.size() * 4));
  CHECK(hipMalloc(&dsc2, sc2.size()));
  CHECK(hipMalloc(&dz, zf.size()));
  CHECK(hipMalloc(&dzs, zsf.size()));
  CHECK(hipMalloc(&dh, (size_t)items1 * T * 32 * 4));
  CHECK(hipMalloc(&dhq, (size_t)experts * KS2 * 4 * T * 32));
  CHECK(hipMalloc(&dhsc, (size_t)experts * KS2 * 64));
  CHECK(hipMalloc(&dy, (size_t)experts * T * LAT * 4));
  unsigned *dqc;   // W2_QUEUE: 8 counters, 128 B apart
  CHECK(hipMalloc(&dqc, 8 * 128));
  CHECK(hipMemset(dhq, 0, (size_t)experts * KS2 * 4 * T * 32));
  CHECK(hipMemset(dhsc, 127, (size_t)experts * KS2 * 64));
  CHECK(hipMemcpy(dw1, w1.data(), w1.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dw2, w2.data(), w2.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dsc1, sc1.data(), sc1.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dsc2, sc2.data(), sc2.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dz, zf.data(), zf.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dzs, zsf.data(), zsf.size(), hipMemcpyHostToDevice));
  set_max_lds((const void *)w13_kernel, LDS1);
  set_max_lds((const void *)w13cu_kernel, LDS_Z + LDS_ZS + 4 * 64 * 16);
  // writing 1 GB before each timed launch evicts the weights from the 256 MB Infinity Cache and the L2s
  size_t flush_bytes = 1ull << 30;
  char *flush;
  CHECK(hipMalloc(&flush, flush_bytes));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::vector<double> us1, us2;
  for (int r = 0; r < reps + 1; r++) {
    float ms;
    CHECK(hipMemsetAsync(flush, r & 0xff, flush_bytes));
    CHECK(hipEventRecord(e0));
    if (W13_CU) {
      hipLaunchKernelGGL(w13cu_kernel, dim3(prop.multiProcessorCount), dim3(256), LDS_Z + LDS_ZS + 4 * 64 * 16, 0, dw1, dsc1, dz, dzs, dh,
                         dhq, dhsc, items1);
    } else {
      hipLaunchKernelGGL(w13_kernel, dim3(prop.multiProcessorCount), dim3(64 * WAVES), LDS1, 0, dw1, dsc1, dz, dzs, dh, dhq, dhsc, items1);
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    if (r) us1.push_back(ms * 1e3);
    CHECK(hipMemsetAsync(flush, r & 0xff, flush_bytes));
    CHECK(hipMemsetAsync(dqc, 0, 8 * 128));
    CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(w2_kernel, dim3(prop.multiProcessorCount), dim3(256), 0, 0, dw2, dsc2, (const v8i *)dhq, dhsc, dy, items2, dqc);
    CHECK(hipEventRecord(e1));
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    if (r) us2.push_back(ms * 1e3);
  }
  double bytes1 = w1.size() + sc1.size() * 4.0, bytes2 = w2.size() + (double)sc2.size();
  // ---- checks ----
  std::vector<float> hh((size_t)items1 * T * 32), yy((size_t)experts * T * LAT);
  std::vector<uint8_t> hq((size_t)experts * KS2 * 4 * T * 32), hs((size_t)experts * KS2 * 64);
  CHECK(hipMemcpy(hh.data(), dh, hh.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(yy.data(), dy, yy.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(hq.data(), dhq, hq.size(), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(hs.data(), dhsc, hs.size(), hipMemcpyDeviceToHost));
  // 1. h of 48 random items against the CPU
  double herr = 0, href = 0;
  for (int c = 0; c < 48; c++) {
    int item = rng() % items1, e = item / ITEMS1, it = item % ITEMS1;
    for (int tok = 0; tok < T; tok++)
      for (int f = 0; f < 32; f++) {
        double gu[2];
        for (int side = 0; side < 2; side++) {
          size_t row = (size_t)e * 2 * IR + (4 * it + 2 * side + f / 16) * 16 + f % 16;
          double acc = 0;
          for (int k = 0; k < K1; k++)
            acc += e2m1(q1[row * K1 + k]) * std::ldexp(1.0, s1[row * (K1 / 32) + k / 32] - 127) * e4m3(z[tok * K1 + k]) *
                   std::ldexp(1.0, zs[tok * (K1 / 32) + k / 32] - 127);
          gu[side] = acc;
        }
        double ref = 4 * std::tanh(gu[0] / 4) / (1 + std::exp(-gu[0])) * 25 * std::tanh(gu[1] / 25);
        herr = std::max(herr, std::fabs(hh[((size_t)item * T + tok) * 32 + f] - ref));
        href = std::max(href, std::fabs(ref));
      }
  }
  // 2. h_q of every item from the kernel's own h: the scale exactly, the e4m3 values exactly (ties may differ: counted)
  long hq_bad = 0, hs_bad = 0, hq_n = 0;
  for (int item = 0; item < items1; item++) {
    int e = item / ITEMS1, it = item % ITEMS1, ks2 = it / 4, q = it % 4;
    for (int tok = 0; tok < T; tok++) {
      double amax = 0;
      for (int f = 0; f < 32; f++) amax = std::max(amax, (double)std::fabs(hh[((size_t)item * T + tok) * 32 + f]));
      float a = std::max((float)amax, 1.0e-30f) * (1.0f / 448.0f);
      uint32_t bits;
      memcpy(&bits, &a, 4);
      int ex = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0);
      ex = std::min(127, std::max(-127, ex));
      hs_bad += hs[((size_t)e * KS2 + ks2) * 64 + 16 * q + tok] != ex + 127;
      for (int f = 0; f < 32; f++) {
        int g, b;
        fp8_slot(32 * q + f, g, b);
        double v = std::min(448.0, std::max(-448.0, (double)hh[((size_t)item * T + tok) * 32 + f] * std::ldexp(1.0, -ex)));
        // the value, not the byte: gfx950 rounds a tiny negative value to -0 (0x80), to_e4m3 to +0 (probe 10-10)
        hq_bad += e4m3(hq[(((size_t)e * KS2 + ks2) * 4 * T + T * g + tok) * 32 + b]) != e4m3(to_e4m3(v));
        hq_n++;
      }
    }
  }
  // 3. y of 8 random experts x 64 random rows from the kernel's own h_q
  double yerr = 0, yref = 0;
  for (int c = 0; c < 8; c++) {
    int e = rng() % experts;
    for (int rr = 0; rr < 64; rr++) {
      int row = rng() % LAT;
      for (int tok = 0; tok < T; tok++) {
        double acc = 0;
        for (int k = 0; k < IR; k++) {
          int ks2 = k / 128, g, b;
          fp8_slot(k % 128, g, b);
          double hv = e4m3(hq[(((size_t)e * KS2 + ks2) * 4 * T + T * g + tok) * 32 + b]) *
                      std::ldexp(1.0, hs[((size_t)e * KS2 + ks2) * 64 + 16 * ((k % 128) / 32) + tok] - 127);
          size_t wr = (size_t)e * LAT + row;
          acc += e2m1(q2[wr * IR + k]) * std::ldexp(1.0, s2[wr * (IR / 32) + k / 32] - 127) * hv;
        }
        yerr = std::max(yerr, std::fabs(yy[((size_t)e * T + tok) * LAT + row] - acc));
        yref = std::max(yref, std::fabs(acc));
      }
    }
  }
  double m1 = median(us1), m2 = median(us2);
  printf("W13: %.1f MB, median %.2f us (%.2f TB/s); h max error %.2e of max |h| %.2e; h_q scales wrong %ld, bytes wrong %ld of %ld\n",
         bytes1 / 1e6, m1, bytes1 / m1 / 1e6, herr, href, hs_bad, hq_bad, hq_n);
  printf("W2:  %.1f MB, median %.2f us (%.2f TB/s); y max error %.2e of max |y| %.2e\n", bytes2 / 1e6, m2, bytes2 / m2 / 1e6, yerr,
         yref);
  printf("both: %.2f us for %d experts (%.2f TB/s); caches emptied before each launch, %d launches each\n", m1 + m2, experts,
         (bytes1 + bytes2) / (m1 + m2) / 1e6, reps);
  printf("{\"bench\": \"expert_unit\", \"stages\": %d, \"hint\": %d, \"w2_tiles\": %d, \"fp8_lin\": %d, \"fp4_hi\": %d, \"experts\": %d, "
         "\"w13_us\": %.3f, \"w2_us\": %.3f, \"w13_TBps\": %.3f, \"w2_TBps\": %.3f, \"h_err\": %.3e, \"h_ref\": %.3e, "
         "\"hsc_wrong\": %ld, \"hq_wrong\": %ld, \"y_err\": %.3e, \"y_ref\": %.3e}\n",
         STAGES, HINT, W2_TILES, FP8_LIN, FP4_HI, experts, m1, m2, bytes1 / m1 / 1e6, bytes2 / m2 / 1e6, herr, href, hs_bad, hq_bad,
         yerr, yref);
  return 0;
}
