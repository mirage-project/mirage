// GPT-OSS's routed experts as two kernels after the layer's static megakernel
// (which then ends at the gate; python/mirage/mpk/sm90_moe_split.py): W13 with
// the clamped SwiGLU, then W2 with the top-k combine and the residual.
//
// triton_kernels' matmul_ogs scheme (SGLang's GPT-OSS MoE on Hopper): the
// MXFP4 weights are wgmma's A operand, decoded in registers with the E8M0
// scale folded in, register-A async wgmma (two A buffers in turn: one decoded
// while the other's wgmmas run); B = the expert's routed tokens' activations
// (compact, 8-token groups), each row TMA'd into the 128-B swizzled K tile
// next to the weight tile, in a 4-stage ring a producer warp fills. A CTA is
// one item, (64 outputs, expert): several CTAs an SM, the grid's experts with
// no routed token exit at once. Inside the layer kernel ptxas serializes every
// wgmma (one function at the register cap), hence the separate kernels.
//
// Compile: -DSTATIC_TOKENS= (token rows) -DMOE_H= -DMOE_I= -DMOE_E= -DMOE_TOPK=
#define STATIC_GPUS 1
#define STATIC_SMS 132
#define STATIC_SM90 1
#include "runtime.cuh"
#include <cstdio>

namespace static_mk {

constexpr int H = MOE_H, I = MOE_I, E = MOE_E, TOPK = MOE_TOPK;
constexpr int STAGES = 4;
constexpr float SUM_SCALE = 4.0f; // the decode's values are 1/4 of the weights'

__device__ __forceinline__ void wgmma_wait_1() {
  asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
}

// Eight packed values of one 32-bit word (sm90_moe_pack's order, each byte's
// bits permuted by sm90_moe_split_weights: low-slot nibble at bits {5: sign,
// 2, 1: exponent, 0: mantissa}, high-slot at {7, 4, 3, 6}) to four f16 pairs
// v 2^(s - 129): their E4M3 bytes s.0ee.m00 (v 2^-6) by a shift and a mask
// each, the hardware's E4M3 -> f16, times scale i's f16 2^(s - 123) (sc4: the
// row's four scales' (s - 108) << 2 a byte; every scale in [108, 138]).
__device__ __forceinline__ void
    decode_word(uint32_t w, uint32_t sc4, int i, uint32_t (&out)[4]) {
  uint32_t const lo = (w << 2) & 0x9C9C9C9Cu;
  uint32_t const hi = (w & 0x98989898u) | ((w >> 4) & 0x04040404u);
  uint32_t scale2;
  asm("prmt.b32 %0, %1, 0, %2;"
      : "=r"(scale2)
      : "r"(sc4), "r"(0x0404u | (i << 4) | (i << 12)));
  uint32_t f[4];
  asm("{\n.reg .b16 a, b, c, d;\n"
      "mov.b32 {a, b}, %4;\n"
      "mov.b32 {c, d}, %5;\n"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n"
      "cvt.rn.f16x2.e4m3x2 %1, b;\n"
      "cvt.rn.f16x2.e4m3x2 %2, c;\n"
      "cvt.rn.f16x2.e4m3x2 %3, d;\n}"
      : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
      : "r"(lo), "r"(hi));
#pragma unroll
  for (int c = 0; c < 4; c++) {
    __half2 const r = __hmul2(*reinterpret_cast<__half2 const *>(&f[c]),
                              *reinterpret_cast<__half2 const *>(&scale2));
    out[c] = *reinterpret_cast<uint32_t const *>(&r);
  }
}

// d (64 weight rows x N tokens, fp32) (+)= A (registers: one K16 fragment) B
// (N tokens x K16 at descriptor b)
template <int N>
__device__ __forceinline__ void
    wgmma_k16(float *d, uint32_t const (&a)[4], uint64_t b, int acc);
template <>
__device__ __forceinline__ void
    wgmma_k16<8>(float *d, uint32_t const (&a)[4], uint64_t b, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}
template <>
__device__ __forceinline__ void
    wgmma_k16<16>(float *d, uint32_t const (&a)[4], uint64_t b, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
               "{%0, %1, %2, %3, %4, %5, %6, %7}, "
               "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}"
               : "+f"(d[0]),
                 "+f"(d[1]),
                 "+f"(d[2]),
                 "+f"(d[3]),
                 "+f"(d[4]),
                 "+f"(d[5]),
                 "+f"(d[6]),
                 "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}
template <>
__device__ __forceinline__ void
    wgmma_k16<24>(float *d, uint32_t const (&a)[4], uint64_t b, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, "
               "{%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}"
               : "+f"(d[0]),
                 "+f"(d[1]),
                 "+f"(d[2]),
                 "+f"(d[3]),
                 "+f"(d[4]),
                 "+f"(d[5]),
                 "+f"(d[6]),
                 "+f"(d[7]),
                 "+f"(d[8]),
                 "+f"(d[9]),
                 "+f"(d[10]),
                 "+f"(d[11])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}
template <>
__device__ __forceinline__ void
    wgmma_k16<32>(float *d, uint32_t const (&a)[4], uint64_t b, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}"
      : "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11]),
        "+f"(d[12]),
        "+f"(d[13]),
        "+f"(d[14]),
        "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
}

// d (+)= A (64 K as four K16 fragments) B (N tokens x 64 K at descriptor b:
// 128-B swizzle atoms of 8 rows)
template <int N>
__device__ __forceinline__ void
    wgmma_k64(float *d, uint32_t const (&a)[4][4], uint64_t b, int acc) {
#pragma unroll
  for (int k = 0; k < 4; k++) {
    wgmma_k16<N>(d, a[k], b + 2 * k, k > 0 || acc);
  }
}

struct Args {
  uint8_t const *w; // the projection's blocks [E][N / PR][K tiles][PR x 68]
  __nv_bfloat16 const *bias; // [E][N]
  int const *routing;        // [E][T]: token t's top-k slot + 1, 0: not routed
  __half *act;         // [E][T][I]: W13's output (row j: the j-th routed token)
  __nv_bfloat16 *eo;   // [T][TOPK][H]: W2's per-slot outputs
  float const *topk_w; // [T][TOPK]
  __nv_bfloat16 const *x1; // [T][H] the residual
  __nv_bfloat16 *x_out;    // [T][H]
  unsigned *done;          // [T][H / 64] W2 slots done (the last re-zeroes it)
  float limit, alpha;
};

// GLU: W13 (128-row blocks: 64 gate rows, the 64 up rows of the same outputs;
// N = 2 I, K = H); else W2 (64-row blocks; N = H, K = I). A K tile is two
// items, an A buffer each (buffers alternating across tiles, or four items a
// tile, serialize ptxas' wgmma pipeline: C7513)
template <bool GLU>
struct Cfg {
  static constexpr int K = GLU ? H : I, N = GLU ? 2 * I : H;
  static constexpr int PR = GLU ? 128 : 64, MT = PR / 64, BLOCK = PR * 68;
  static constexpr int KT = (K + 127) / 128;
  static constexpr int ACT_OFF = (BLOCK + 1023) / 1024 * 1024,
                       STAGE = ACT_OFF + T * 256;
  static constexpr int SMEM = STAGES * STAGE + 1024 + 16 * STAGES;
};

// block y of expert e times its cnt routed tokens' rows (NG groups of 8)
template <bool GLU, int NG>
__device__ __forceinline__ void item(CUtensorMap const *amap,
                                     Args const &p,
                                     int y,
                                     int e,
                                     int cnt,
                                     int const *tokl,
                                     int const *slotl,
                                     char *sm,
                                     uint32_t full0,
                                     uint32_t empty0) {
  using C = Cfg<GLU>;
  constexpr int KT = C::KT, BLOCK = C::BLOCK, MT = C::MT;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint8_t const *blocks = p.w + ((size_t)e * (C::N / C::PR) + y) * KT * BLOCK;
  if (warp == 4) { // producer: lane 0 the weights, lane j < cnt row j (both K
                   // halves, half h at h * NG * 1024)
    int const src = lane < cnt ? (GLU ? tokl[lane] : e * T + lane) : 0;
    for (int kt = 0; kt < KT; kt++) {
      int const st = kt % STAGES;
      if (kt >= STAGES) {
        mbar_wait(empty0 + 8 * st, ((kt / STAGES) - 1) & 1);
      }
      uint32_t const fb = full0 + 8 * st, dst = su32(sm + st * C::STAGE);
      if (lane == 0) {
        mbar_expect(fb, BLOCK + cnt * 256);
        bulk_g2s(dst, blocks + (size_t)kt * BLOCK, BLOCK, fb);
      }
      __syncwarp();
      if (lane < cnt) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          tma3(amap,
               fb,
               dst + C::ACT_OFF + h * NG * 1024 + lane * 128,
               0,
               src,
               2 * kt + h,
               EVICT_LAST);
        }
      }
    }
    return;
  }
  // consumers: warp w rows rbase (+ 8) of each m64 tile
  int const g8 = lane >> 2, q = lane & 3, rbase = 16 * warp + g8;
  float acc[MT][NG][4];
  // a K tile's two items, each into its own A buffer: W13 its two m64 tiles
  // (both K halves each), W2 its two K halves; [item][K half][fragment]
  uint32_t r[2][2][4][4];
  // m64 tile mt's K half h into ra
  auto decode = [&](char const *blk, int mt, int h, uint32_t(&ra)[4][4]) {
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
      int const row = 64 * mt + rbase + 8 * rr;
      uint2 const v =
          *reinterpret_cast<uint2 const *>(blk + row * 64 + q * 16 + h * 8);
      uint32_t const sw =
          *reinterpret_cast<uint32_t const *>(blk + C::PR * 64 + row * 4);
      uint32_t const words[2] = {v.x, v.y};
#pragma unroll
      for (int i = 0; i < 2; i++) {
        uint32_t o[4];
        decode_word(words[i], (sw - 0x6C6C6C6Cu) << 2, 2 * h + i, o);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          ra[2 * i + (c >> 1)][rr + 2 * (c & 1)] = o[c];
        }
      }
    }
  };
#pragma unroll 1
  for (int kt = 0; kt < KT; kt++) {
    int const st = kt % STAGES;
    mbar_wait(full0 + 8 * st, (kt / STAGES) & 1);
    char const *stage = sm + st * C::STAGE;
    uint32_t const bx = su32(stage + C::ACT_OFF);
#pragma unroll
    for (int it = 0; it < 2; it++) {
      constexpr int NH = GLU ? 2 : 1; // the item's K halves
      int const mt = GLU ? it : 0, h0 = GLU ? 0 : it;
#pragma unroll
      for (int hh = 0; hh < NH; hh++) {
        decode(stage, mt, h0 + hh, r[it][hh]);
      }
      wgmma_fence();
#pragma unroll
      for (int hh = 0; hh < NH; hh++) {
        wgmma_k64<8 * NG>(&acc[mt][0][0],
                          r[it][hh],
                          gmma_desc(bx + (h0 + hh) * NG * 1024),
                          kt > 0 || h0 + hh > 0);
      }
      wgmma_commit();
      wgmma_wait_1(); // all but this item done: the other buffer free,
                      // stage kt - 1 too
      if (it == 0 && kt > 0) {
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(empty0 + 8 * ((kt - 1) % STAGES));
        }
      }
    }
  }
  wgmma_wait_all();
  __nv_bfloat16 const *b = p.bias + (size_t)e * C::N;
#pragma unroll
  for (int n = 0; n < NG; n++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const j = 8 * n + 2 * q + (i & 1),
                row = y * 64 + rbase + 8 * (i >> 1);
      if (j >= cnt) {
        continue;
      }
      if constexpr (GLU) {
        float gate = acc[0][n][i] * SUM_SCALE + __bfloat162float(b[row]);
        float up = acc[1][n][i] * SUM_SCALE + __bfloat162float(b[I + row]);
        gate = fminf(gate, p.limit);
        up = fminf(fmaxf(up, -p.limit), p.limit);
        p.act[((size_t)e * T + j) * I + row] =
            __float2half((up + 1.0f) * gate / (1.0f + __expf(-gate * p.alpha)));
      } else {
        p.eo[((size_t)tokl[j] * TOPK + slotl[j]) * H + row] = __float2bfloat16(
            acc[0][n][i] * SUM_SCALE + __bfloat162float(b[row]));
      }
    }
  }
  if constexpr (!GLU) {
    // per token: the last of its TOPK experts to finish this 64-row block
    // sums them (x1 + w_k eo_k in k order) into x_out
    __shared__ int last[T];
    __threadfence();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (threadIdx.x < cnt) {
      unsigned *const d = p.done + tokl[threadIdx.x] * (H / 64) + y;
      last[threadIdx.x] = atomicAdd(d, 1u) == TOPK - 1;
      if (last[threadIdx.x]) {
        *d = 0;
      }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    __threadfence();
    for (int x = threadIdx.x; x < cnt * 64; x += 128) {
      int const j = x >> 6, row = y * 64 + (x & 63), t = tokl[j];
      if (!last[j]) {
        continue;
      }
      float a = __bfloat162float(p.x1[(size_t)t * H + row]);
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        a += p.topk_w[t * TOPK + k] *
             __bfloat162float(__ldcg(p.eo + ((size_t)t * TOPK + k) * H + row));
      }
      p.x_out[(size_t)t * H + row] = __float2bfloat16(a);
    }
  }
}

template <bool GLU>
__global__ void __launch_bounds__(160, 3)
    moe_kernel(const __grid_constant__ CUtensorMap amap, const Args p) {
  using C = Cfg<GLU>;
  int const y = blockIdx.x, e = blockIdx.y;
  extern __shared__ __align__(1024) char smem_raw[];
  char *sm = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint32_t const full0 = su32(sm + STAGES * C::STAGE),
                 empty0 = full0 + 8 * STAGES;
  __shared__ int tokl[T], slotl[T], s_cnt;
  int const lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // the kernel before (the layer's gate; W13) done and visible. W13 lets W2
  // start launching (W2 reads nothing before its wait); W2 does not (the next
  // layer's kernel resets its counters before its wait): it triggers at exit
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (GLU) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  if (threadIdx.x < 32) { // expert e's routed tokens, ascending
    int const rt = lane < T ? p.routing[e * T + lane] : 0;
    uint32_t const m = __ballot_sync(~0u, rt > 0);
    if (rt > 0) {
      int const j = __popc(m & ((1u << lane) - 1));
      tokl[j] = lane, slotl[j] = rt - 1;
    }
    if (lane == 0) {
      s_cnt = __popc(m);
    }
  }
  __syncthreads();
  int const cnt = s_cnt;
  auto go = [&](auto ng) {
    item<GLU, decltype(ng)::value>(
        &amap, p, y, e, cnt, tokl, slotl, sm, full0, empty0);
  };
  if (cnt == 0) {
    return;
  } else if (cnt <= 8) {
    go(IC<1>{});
  } else if constexpr (T >= 16) {
    if (cnt <= 16) {
      go(IC<2>{});
    } else if constexpr (T >= 24) {
      if (cnt <= 24) {
        go(IC<3>{});
      } else if constexpr (T >= 32) {
        go(IC<4>{});
      }
    }
  }
}

} // namespace static_mk

// ---- host ----
// f16 [rows, K]: boxes of one row's 64-wide K half, 128-B swizzled
static void map_rows(CUtensorMap *m, void *base, uint64_t rows, uint64_t K) {
  cuuint64_t gd[3] = {64, rows, K / 64}, gs[2] = {K * 2, 128};
  cuuint32_t bd[3] = {64, 1, 1}, es[3] = {1, 1, 1};
  if (cuTensorMapEncodeTiled(m,
                             CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                             3,
                             base,
                             gd,
                             gs,
                             bd,
                             es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) !=
      CUDA_SUCCESS) {
    printf("sm90_moe_split: tensor map error\n");
  }
}

template <bool GLU>
static void
    launch(CUtensorMap const &m, static_mk::Args const &a, cudaStream_t s) {
  using C = static_mk::Cfg<GLU>;
  static bool const smem_set =
      cudaFuncSetAttribute(static_mk::moe_kernel<GLU>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           C::SMEM) == cudaSuccess;
  (void)smem_set;
  cudaLaunchAttribute at = {};
  at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C::N / C::PR, static_mk::E);
  cfg.blockDim = dim3(160);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = s;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, static_mk::moe_kernel<GLU>, m, a);
}

struct Ctx {
  CUtensorMap m13, m2;
};

// the tensor maps of the buffers every layer shares: h2 [T][H] f16 (W13's
// input), act [E][T][I] f16 (W2's)
extern "C" void *moe_ctx(void *h2, void *act) {
  Ctx *c = new Ctx;
  map_rows(&c->m13, h2, static_mk::T, static_mk::H);
  map_rows(&c->m2, act, (uint64_t)static_mk::E * static_mk::T, static_mk::I);
  return c;
}

extern "C" int moe_launch(void *ctx,
                          void const *routing,
                          void const *w13,
                          void const *b13,
                          void *act,
                          void const *w2,
                          void const *b2,
                          void *eo,
                          void const *topk_w,
                          void const *x1,
                          void *x_out,
                          void *done,
                          float limit,
                          float alpha,
                          void *stream) {
  Ctx const *c = static_cast<Ctx const *>(ctx);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  static_mk::Args a{(uint8_t const *)w13,
                    (__nv_bfloat16 const *)b13,
                    (int const *)routing,
                    (__half *)act,
                    (__nv_bfloat16 *)eo,
                    (float const *)topk_w,
                    (__nv_bfloat16 const *)x1,
                    (__nv_bfloat16 *)x_out,
                    (unsigned *)done,
                    limit,
                    alpha};
  launch<true>(c->m13, a, s);
  a.w = (uint8_t const *)w2;
  a.bias = (__nv_bfloat16 const *)b2;
  launch<false>(c->m2, a, s);
  cudaError_t const e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("sm90_moe_split: %s\n", cudaGetErrorString(e));
  }
  return e != cudaSuccess;
}
