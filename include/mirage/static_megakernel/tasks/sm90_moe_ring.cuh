// tasks/sm90_moe_ring.cuh -- the sm90_moe_ring task type: one routed expert's
// MXFP4 projection streamed through the ring (runtime.cuh) on Hopper, decoded
// into the A registers of RS WGMMAs. Task (a, y) of grid (A, N / ROWS, 1): the
// expert in slot a of the active-expert list (nothing when fewer are active)
// times weight rows [y ROWS, y ROWS + ROWS). Two kinds:
//   GLU  W13 with GPT-OSS's clamped SwiGLU fused: a task's 128 rows are the 64
//        gate rows and the 64 up rows of the same 64 outputs; output [A][T][I]
//   W2   output [T, TOPK, N], the routed tokens' rows only
//
// The weights come repacked (static_megakernel.py sm90_moe_ring_pack) as
// blocks [E][N / ROWS][K tiles of 128][ROWS x 68 B], contiguous in the order a
// task streams them: a block is its rows' packed values (ROWS x 64 B, the bytes
// of a row permuted so that MMA thread q's 16 (the A fragment's K columns
// 16 kb + 2q (+ 8), kb = 0..7) are contiguous) then their E8M0 scales (ROWS x
// 4 B); GLU's rows ordered as above.
//
// A ring stage holds NB K tiles: their blocks (one bulk copy) and the 8
// activation rows (NB boxes {64, 8, 2}, 128-B swizzled, as gemm_tile's), as
// many tiles as fit in the stage.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// a ring stage of the ROWS-row task: NB K tiles, the most that fit
template <int ROWS>
struct MoeRingStage {
  static constexpr int act_off(int nb) {
    return (nb * ROWS * 68 + 1023) / 1024 * 1024; // after the blocks
  }
  static constexpr bool fits(int nb) {
    return act_off(nb) + nb * A_STAGE <= FSTAGE;
  }
  static constexpr int NB = fits(5) ? 5 : fits(4) ? 4 : fits(3) ? 3 : 2;
  static constexpr int BLOCK = ROWS * 68, ACT_OFF = act_off(NB);
};

#ifdef STATIC_SM90
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}
// The BF16 pair 2^(scale - 127) of an E8M0 scale (0 decodes to 0).
__device__ __forceinline__ uint32_t moe_scale_bits2(uint32_t scale) {
  uint32_t const bits = scale == 0 ? 0u : scale << 7;
  return bits | bits << 16;
}
// Four packed bytes (eight E2M1 values, low nibble first) to four BF16 pairs
// times the scale pair: each nibble's magnitude looked up (prmt) as the low
// and the high byte of its BF16, the bytes interleaved into pairs, the sign
// (the nibble's top bit, replicated by prmt) merged into bit 15 / 31. Exact.
__device__ __forceinline__ void
    moe_decode_word(uint32_t w, uint32_t scale2, uint32_t (&out)[4]) {
  // BF16 of the magnitudes 0, .5, 1, 1.5 | 2, 3, 4, 6: low bytes, high bytes
  constexpr uint32_t LO0 = 0xc0800000u, LO1 = 0xc0804000u;
  constexpr uint32_t HI0 = 0x3f3f3f00u, HI1 = 0x40404040u;
  uint32_t const m = w & 0x77777777u, lo_signs = w << 4;
#pragma unroll
  for (int half = 0; half < 2; half++) { // nibbles 0-3, 4-7
    uint32_t const sel = half ? m >> 16 : m;
    uint32_t const lo = prmt(LO0, LO1, sel), hi = prmt(HI0, HI1, sel);
#pragma unroll
    for (int p = 0; p < 2; p++) {
      int const k = 2 * half + p; // the byte: nibbles 2k, 2k + 1
      uint32_t const v = prmt(lo, hi, p ? 0x7362u : 0x5140u);
      uint32_t const sign = prmt(lo_signs, w, 0xc080u + 0x1010u * k);
      uint32_t const vs = v | (sign & 0x80008000u);
      __nv_bfloat162 const r =
          __hmul2(*reinterpret_cast<__nv_bfloat162 const *>(&vs),
                  *reinterpret_cast<__nv_bfloat162 const *>(&scale2));
      out[k] = *reinterpret_cast<uint32_t const *>(&r);
    }
  }
}
// d (64 weight rows x 8 tokens) += A (registers, 64 x K16) * B (smem, 8 x K16)
__device__ __forceinline__ void
    wgmma_rs_m64n8k16(float *d, uint32_t const (&a)[4], uint64_t b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
__device__ __forceinline__ void wgmma_wait_1() {
  asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
}
// a contiguous copy global -> shared, completing on mbarrier mb
__device__ __forceinline__ void
    bulk_g2s(void const *src, uint32_t mb, uint32_t sm, uint32_t bytes) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];" ::"r"(sm),
               "l"((uint64_t)src),
               "r"(bytes),
               "r"(mb)
               : "memory");
}
#endif // STATIC_SM90

// PARAMS {GLU (1: W13 + SwiGLU, 0: W2), ROWS (64 or 128; GLU 128), N (weight
// rows per expert), K, E, TOPK, HAS_BIAS, limit (fp32 bits), alpha (fp32
// bits)}; BUF_SLOTS {the active-expert list [E + 1] i32 (count last), routing
// indices [E, T] i32 (1 + top-k slot, else 0), the weight blocks, bias [E, N]
// bf16 in the checkpoint's row order (-1: none), output}; MAP_SLOTS {the
// activation (bf16; GLU [T, K], W2 [A T, K] grouped by slot; box T rows)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_moe_ring(Maps const &maps,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
#ifndef STATIC_SM90
  static_assert(sizeof(SELF) == 0, "sm90_moe_ring: Hopper only");
#else
  constexpr bool GLU = PARAMS::v[0] == 1, HAS_BIAS = PARAMS::v[6] == 1;
  constexpr int ROWS = PARAMS::v[1], N = PARAMS::v[2], K = PARAMS::v[3],
                E = PARAMS::v[4], TOPK = PARAMS::v[5];
  using S = MoeRingStage<ROWS>;
  constexpr int MH = ROWS / 64, NB = S::NB; // m64 blocks, K tiles per stage
  constexpr int KT = (K + 127) / 128, NS = (KT + NB - 1) / NB; // tiles, stages
  static_assert((ROWS == 64 || ROWS == 128) && N % ROWS == 0 && K % 64 == 0,
                "sm90_moe_ring: 64- or 128-row tiles, K in 64-value blocks");
  static_assert(!GLU || ROWS == 128,
                "sm90_moe_ring: GLU's tasks are 64 gate rows and the 64 up "
                "rows");
  // the producers of everything but the activation (IN's first): the expert
  // list, the routing. The activation's the loader waits for after it has
  // started the weights, which depend on nothing (as PDL does between
  // kernels): a task's weight stream starts while its producer finishes.
  __syncthreads(); // this SM's earlier tasks are done with the ring
  if (threadIdx.x == 0) {
    sm90_wait_inputs<~1u, IN...>(g);
  }
  __syncthreads();
  int const *active = buf_at<int const>(g, slot_at<BUF_SLOTS, 0>());
  int const a = tk.x, row0 = tk.y * ROWS;
  if (a < active[E]) {
    int const e = active[a];
    int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 0) {
      if (lane == 0) { // the loader: stage s = K tiles [s NB, s NB + NB)
        void const *amap = &maps.m[slot_at<MAP_SLOTS, 0>()];
        uint8_t const *blocks =
            buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 2>()) +
            (size_t)(e * (N / ROWS) + tk.y) * KT * S::BLOCK;
        int const c0 = L.gl;
        // stage s (ring count c0 + s): its weight blocks, then its
        // activation boxes (the full barrier expects both)
        auto weights = [&](int s) {
          int const c = c0 + s, st = c % SMAX;
          if (c >= SMAX) {
            mbar_wait(rt.empty0 + 8 * st, ((c / SMAX) - 1) & 1);
          }
          int const nt = min(NB, KT - s * NB);
          mbar_expect(rt.full0 + 8 * st, nt * (S::BLOCK + A_STAGE));
          bulk_g2s(blocks + (size_t)s * NB * S::BLOCK,
                   rt.full0 + 8 * st,
                   L.base + st * FSTAGE,
                   nt * S::BLOCK);
        };
        auto activation = [&](int s) {
          int const st = (c0 + s) % SMAX;
          for (int b = 0; b < min(NB, KT - s * NB); b++) {
            tma3(amap,
                 rt.full0 + 8 * st,
                 L.base + st * FSTAGE + S::ACT_OFF + b * A_STAGE,
                 0,
                 GLU ? 0 : a * T,
                 (s * NB + b) * 2,
                 EVICT_LAST);
          }
        };
        // the first stage's weights, then more while the activation's
        // producer is still running (each queues ahead of the activation)
        int pre = 0;
        do {
          weights(pre++);
        } while (pre < min(SMAX, NS) && !sm90_inputs_ready<1u, IN...>(g));
        sm90_wait_inputs<1u, IN...>(g);
        asm volatile("fence.proxy.async.global;" ::: "memory"); // TMA reads it
        for (int s = 0; s < NS; s++) {
          if (s >= pre) {
            weights(s);
          }
          activation(s);
        }
        L.gl = c0 + NS;
      }
    } else if (warp >= 4) { // the MMA warpgroup: decode, WGMMA, epilogue
      int const t = threadIdx.x & 127, q = t & 3;
      int const rbase = 16 * (t >> 5) + ((t & 31) >> 2); // + 64 h (+ 8)
      float acc[MH][4] = {};
      // the A fragments of two K tiles: one decoding while the other's
      // WGMMAs run
      uint32_t regs[2][MH][8][4];
      for (int s = 0; s < NS; s++, L.gi0++) {
        int const st = L.gi0 % SMAX;
        mbar_wait(rt.full0 + 8 * st, (L.gi0 / SMAX) & 1);
        char const *stage = L.sm + st * FSTAGE;
#pragma unroll
        for (int b = 0; b < NB; b++) {
          if (s * NB + b < KT) {
            uint32_t(&r)[MH][8][4] = regs[b & 1];
#pragma unroll
            for (int h = 0; h < MH; h++) {
#pragma unroll
              for (int rr = 0; rr < 2; rr++) { // rows rbase + 64 h (+ 8)
                int const row = 64 * h + rbase + 8 * rr;
                uint4 const v = *reinterpret_cast<uint4 const *>(
                    stage + b * S::BLOCK + row * 64 + q * 16);
                uint32_t const sw = *reinterpret_cast<uint32_t const *>(
                    stage + b * S::BLOCK + ROWS * 64 + row * 4);
                uint32_t const words[4] = {v.x, v.y, v.z, v.w};
                // word i: k blocks 2i, 2i + 1 (32 values, one scale); its
                // byte c is register rr + 2 (c & 1) of k block 2i + c / 2
#pragma unroll
                for (int i = 0; i < 4; i++) {
                  uint32_t p[4];
                  moe_decode_word(
                      words[i], moe_scale_bits2((sw >> (8 * i)) & 0xffu), p);
#pragma unroll
                  for (int c = 0; c < 4; c++) {
                    r[h][2 * i + (c >> 1)][rr + 2 * (c & 1)] = p[c];
                  }
                }
              }
            }
            uint32_t const ast =
                L.base + st * FSTAGE + S::ACT_OFF + b * A_STAGE;
            wgmma_fence();
#pragma unroll
            for (int kb = 0; kb < 8; kb++) {
              uint64_t const dx =
                  gmma_desc(ast + (kb >> 2) * 1024) + 2 * (kb & 3);
#pragma unroll
              for (int h = 0; h < MH; h++) {
                wgmma_rs_m64n8k16(acc[h], r[h][kb], dx);
              }
            }
            wgmma_commit();
            wgmma_wait_1(); // the other K tile's registers are free again
          }
        }
        wgmma_wait_all();
        if (t == 0) {
          mbar_arrive(rt.empty0 + 8 * st);
        }
      }
      // epilogue: acc[h][i] = row 64 h + rbase (+ 8 for i >= 2), token
      // 2q + (i & 1)
      __nv_bfloat16 const *bias =
          HAS_BIAS ? buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()) +
                         (size_t)e * N
                   : nullptr;
      __nv_bfloat16 *out = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 4>());
      if constexpr (GLU) { // out[a][tok][o] = SwiGLU(gate o, up o)
        constexpr int I = N / 2;
        float const limit = __int_as_float(PARAMS::v[7]),
                    alpha = __int_as_float(PARAMS::v[8]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int const o = tk.y * 64 + rbase + 8 * (i >> 1), tok = 2 * q + (i & 1);
          float gate = acc[0][i], up = acc[1][i];
          if constexpr (HAS_BIAS) {
            gate += __bfloat162float(bias[o]);
            up += __bfloat162float(bias[I + o]);
          }
          gate = fminf(gate, limit);
          up = fminf(fmaxf(up, -limit), limit);
          out[((size_t)a * T + tok) * I + o] = __float2bfloat16(
              (up + 1.0f) * gate / (1.0f + __expf(-gate * alpha)));
        }
      } else {
        int const *routing = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
#pragma unroll
        for (int h = 0; h < MH; h++) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            int const row = row0 + 64 * h + rbase + 8 * (i >> 1),
                      tok = 2 * q + (i & 1);
            int const slot = routing[e * T + tok] - 1;
            if (slot >= 0) {
              float v = acc[h][i];
              if constexpr (HAS_BIAS) {
                v += __bfloat162float(bias[row]);
              }
              out[((size_t)tok * TOPK + slot) * N + row] = __float2bfloat16(v);
            }
          }
        }
      }
    }
  }
  sm90_task_end<SELF>(g);
#endif
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_moe_ring() {
  return RING_BYTES;
}

} // namespace static_mk
