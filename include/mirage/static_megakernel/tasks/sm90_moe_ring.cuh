// tasks/sm90_moe_ring.cuh -- the sm90_moe_ring task type: one routed expert's
// MXFP4 projection streamed through the ring (runtime.cuh) on Hopper, decoded
// into the A registers of mma.sync. Task (a, y) of grid (A, N / ROWS, 1): the
// expert in slot a of the active-expert list (nothing when fewer are active)
// times weight rows [y ROWS, y ROWS + ROWS). Two kinds:
//   GLU  W13 with GPT-OSS's clamped SwiGLU fused: a task's ROWS rows are the
//        ROWS / 2 gate rows and the up rows of the same outputs; output
//        [A][T][I] f16 (W2's input)
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
//
// All 8 warps decode and multiply, each on its own (mma.sync m16n8k16: the
// decode latency of one warp hides behind another's). A task's rows are NU
// units -- GLU: pairs of the gate m16 tile p (rows 16 p) and the up tile p
// (rows ROWS / 2 + 16 p); W2: m16 tiles p (rows 16 p) -- dealt to RG row
// groups (unit p to group p % RG); warp w takes group w % RG's units (MT m16
// tiles) for the K tiles kt with kt % KG == w / RG (RG x KG = 8 warps). The
// other K groups' sums reach group 0 for the epilogue through the task's ring
// stages (consumed by then).
// The last warp done with a stage frees it and refills it with the stage SMAX
// later; thread 0 loads the first stages (the weights before it waits for the
// activation's producer).
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

template <class A, class...>
struct first_of {
  using type = A;
};

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
// The f16 pair 2^(scale - 121) of an E8M0 scale: the factor that takes an
// E4M3 value v 2^-6 (moe_decode_word) to v 2^(scale - 127). sm90_moe_ring_pack
// checks the scales are in [107, 136], where it is a normal f16 and every
// product is exact (0 decodes to 0).
__device__ __forceinline__ uint32_t moe_scale_f16x2(uint32_t scale) {
  uint32_t const bits = scale == 0 ? 0u : (scale - 106u) << 10;
  return bits | bits << 16;
}
// Eight packed E2M1 values to four f16 pairs times the scale pair. The packing
// (sm90_moe_ring_pack) puts register c's two values in the low nibbles of
// bytes 2c, 2c + 1 (c = 0, 1) or the high nibbles of bytes 2c - 4, 2c - 3
// (c = 2, 3); a nibble s.ee.m as the E4M3 byte s.0ee.m00 is its value times
// 2^-6 (exact, the zero exponent included), so two shift-and-masks give eight
// E4M3 bytes in register order, the hardware converts them to f16 pairs, one
// multiply scales them.
__device__ __forceinline__ void
    moe_decode_word(uint32_t w, uint32_t scale2, uint32_t (&out)[4]) {
  uint32_t const lo = ((w << 2) & 0x1c1c1c1cu) | ((w << 4) & 0x80808080u);
  uint32_t const hi = ((w >> 2) & 0x1c1c1c1cu) | (w & 0x80808080u);
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
// d (16 weight rows x 8 tokens, fp32) += A (16 x K16) * B (K16 x 8 tokens),
// f16
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             uint32_t const (&a)[4],
                                             uint32_t b0,
                                             uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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
// bits), SLOT_DEPS (W2: the input's producer is a GLU node, which counts its
// tasks per slot too: counter 1 + a, after the total in counter 0)}; BUF_SLOTS
// {the active-expert list [E + 1] i32 (count last), routing indices [E, T] i32
// (1 + top-k slot, else 0), the weight blocks, bias [E, N] bf16 in the
// checkpoint's row order (-1: none), output}; MAP_SLOTS {the activation (f16;
// GLU [T, K], W2 [A T, K] grouped by slot; box T rows)}
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
  constexpr bool GLU = PARAMS::v[0] == 1, HAS_BIAS = PARAMS::v[6] == 1,
                 SLOT_DEPS = PARAMS::v[9] == 1;
  constexpr int ROWS = PARAMS::v[1], N = PARAMS::v[2], K = PARAMS::v[3],
                E = PARAMS::v[4], TOPK = PARAMS::v[5];
  using S = MoeRingStage<ROWS>;
  constexpr int NB = S::NB; // K tiles a stage
  // units, row groups, K groups, a warp's units and m16 tiles
  constexpr int NU = GLU ? ROWS / 32 : ROWS / 16;
  constexpr int RG = NU % 8 == 0 ? 8 : NU % 4 == 0 ? 4 : NU % 2 == 0 ? 2 : 1;
  constexpr int KG = 8 / RG, UPG = NU / RG, MT = GLU ? 2 * UPG : UPG;
  constexpr int XCH_BYTES = (KG - 1) * RG * 32 * MT * 16; // groups 1..'s sums
  constexpr int KT = (K + 127) / 128, NS = (KT + NB - 1) / NB; // tiles, stages
  static_assert(ROWS % (GLU ? 32 : 16) == 0 && N % ROWS == 0 && K % 64 == 0,
                "sm90_moe_ring: whole m16 tiles (GLU: tile pairs), K in "
                "64-value blocks");
  static_assert(XCH_BYTES <= SMAX * FSTAGE &&
                    OFF_WSF + 4 * (SMAX + 1) <= RING_BYTES,
                "sm90_moe_ring: the exchange area and the stage counters");
  static_assert(RG * KG == 8, "sm90_moe_ring: 8 warps");
  // the producers of everything but the activation (IN's first): the expert
  // list, the routing. The activation's thread 0 waits for after it has
  // started the weights, which depend on nothing (as PDL does between
  // kernels): a task's weight stream starts while its producer finishes.
  __syncthreads(); // this SM's earlier tasks are done with the ring
  int *const done = reinterpret_cast<int *>(L.sm + OFF_WSF);
  if (threadIdx.x == 0) {
    sm90_wait_inputs<~1u, IN...>(g);
    for (int i = 0; i < SMAX; i++) {
      done[i] = 0;
    }
    done[SMAX] = L.gl; // the stages so far (the loader's count, the same as
                       // the MMAs' between tasks)
  }
  __syncthreads();
  int const *active = buf_at<int const>(g, slot_at<BUF_SLOTS, 0>());
  int const a = tk.x, row0 = tk.y * ROWS;
  int const c0 = done[SMAX];
  if (a < active[E]) {
    int const e = active[a];
    int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    void const *amap = &maps.m[slot_at<MAP_SLOTS, 0>()];
    uint8_t const *blocks = buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 2>()) +
                            (size_t)(e * (N / ROWS) + tk.y) * KT * S::BLOCK;
    // stage s (ring count c0 + s): its weight blocks, its activation boxes
    // (the full barrier expects both)
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
    constexpr int PRE = NS < SMAX ? NS : SMAX;
    // the activation is there: its producer has counted all its tasks (W2
    // after a GLU node: those of slot a)
    auto act_ready = [&] {
      if constexpr (SLOT_DEPS) {
        using ACT = typename first_of<IN...>::type;
        return cnt_ld(g.cnt + ACT::counter + 1 + a) >= (uint32_t)ACT::y;
      } else {
        return sm90_inputs_ready<1u, IN...>(g);
      }
    };
    if (threadIdx.x == 0) {
      // the first stage's weights, then more while the activation's producer
      // is still running (each queues ahead of the activation)
      int pre = 0;
      do {
        weights(pre++);
      } while (pre < PRE && !act_ready());
      while (!act_ready()) {
        __nanosleep(200);
      }
      asm volatile("fence.proxy.async.global;" ::: "memory"); // TMA reads it
      for (int s = 0; s < PRE; s++) {
        if (s >= pre) {
          weights(s);
        }
        activation(s);
      }
    }
    int const g8 = lane >> 2, q = lane & 3, rg = warp % RG, kg = warp / RG;
    // the first row of the warp's m16 tile mt (its unit mt / 2 (GLU) or mt;
    // GLU: the gate tile, then the up tile); a thread's rows: + g8 (+ 8)
    auto tile_row = [&](int mt) {
      int const p = rg + RG * (GLU ? mt / 2 : mt);
      return 16 * p + ((GLU && (mt & 1)) ? ROWS / 2 : 0);
    };
    float acc[MT][4] = {};
    for (int s = 0; s < NS; s++) {
      int const c = c0 + s, st = c % SMAX;
      mbar_wait(rt.full0 + 8 * st, (c / SMAX) & 1);
      char const *stage = L.sm + st * FSTAGE;
#pragma unroll
      for (int b = 0; b < NB; b++) {
        int const kt = s * NB + b;
        if (kt % KG != kg || kt >= KT) {
          continue;
        }
        // B: token g8's K pairs 2q (b0) and 2q + 8 (b1) of each k16 block,
        // from the 128-B swizzled box (16-B chunk index ^ row)
        char const *act = stage + S::ACT_OFF + b * A_STAGE + g8 * 128 + 4 * q;
        uint32_t bf[8][2];
#pragma unroll
        for (int kb = 0; kb < 8; kb++) {
          char const *half = act + (kb >> 2) * 1024;
          int const ch = 2 * (kb & 3);
          bf[kb][0] =
              *reinterpret_cast<uint32_t const *>(half + ((ch ^ g8) << 4));
          bf[kb][1] = *reinterpret_cast<uint32_t const *>(
              half + (((ch + 1) ^ g8) << 4));
        }
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          uint32_t r[8][4];
#pragma unroll
          for (int rr = 0; rr < 2; rr++) { // rows tile_row(mt) + g8 (+ 8)
            int const row = tile_row(mt) + g8 + 8 * rr;
            uint4 const v = *reinterpret_cast<uint4 const *>(
                stage + b * S::BLOCK + row * 64 + q * 16);
            uint32_t const sw = *reinterpret_cast<uint32_t const *>(
                stage + b * S::BLOCK + ROWS * 64 + row * 4);
            uint32_t const words[4] = {v.x, v.y, v.z, v.w};
            // word i: k blocks 2i, 2i + 1 (32 values, one scale); its
            // register cc is register rr + 2 (cc & 1) of k block 2i + cc / 2
#pragma unroll
            for (int i = 0; i < 4; i++) {
              uint32_t p[4];
              moe_decode_word(
                  words[i], moe_scale_f16x2((sw >> (8 * i)) & 0xffu), p);
#pragma unroll
              for (int cc = 0; cc < 4; cc++) {
                r[2 * i + (cc >> 1)][rr + 2 * (cc & 1)] = p[cc];
              }
            }
          }
#pragma unroll
          for (int kb = 0; kb < 8; kb++) {
            mma_m16n8k16(acc[mt], r[kb], bf[kb][0], bf[kb][1]);
          }
        }
      }
      // the last warp done with the stage frees it and refills it
      __syncwarp();
      if (lane == 0) {
        __threadfence_block(); // this warp's reads of the stage, before
        if (atomicAdd(done + st, 1) == 7) {
          done[st] = 0;
          mbar_arrive(rt.empty0 + 8 * st);
          if (s + SMAX < NS) {
            weights(s + SMAX); // its empty wait passes: the arrive above
            activation(s + SMAX);
          }
        }
      }
    }
    L.gl = L.gi0 = c0 + NS;
    // epilogue: acc[mt][i] = row tile_row(mt) + g8 (+ 8 for i >= 2), token
    // 2q + (i & 1); K groups 1.. add theirs to group 0's, through the ring
    // stages (every warp done reading them: the barrier)
    float4 *xacc = reinterpret_cast<float4 *>(L.sm);
    __syncthreads();
    if (kg > 0) {
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        xacc[(((kg - 1) * RG + rg) * 32 + lane) * MT + mt] =
            make_float4(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
      }
    }
    __syncthreads();
    if (kg == 0) {
#pragma unroll
      for (int k = 1; k < KG; k++) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          float4 const o = xacc[(((k - 1) * RG + rg) * 32 + lane) * MT + mt];
          acc[mt][0] += o.x, acc[mt][1] += o.y, acc[mt][2] += o.z,
              acc[mt][3] += o.w;
        }
      }
      __nv_bfloat16 const *bias =
          HAS_BIAS ? buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()) +
                         (size_t)e * N
                   : nullptr;
      void *out = buf_at<void>(g, slot_at<BUF_SLOTS, 4>());
      int const *routing = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int const tok = 2 * q + (i & 1);
        if constexpr (GLU) { // out[a][tok][o] = SwiGLU(gate o, up o)
          constexpr int I = N / 2;
          float const limit = __int_as_float(PARAMS::v[7]),
                      alpha = __int_as_float(PARAMS::v[8]);
#pragma unroll
          for (int u = 0; u < UPG; u++) { // unit u: tiles 2u (gate), 2u + 1
            int const o =
                tk.y * (ROWS / 2) + tile_row(2 * u) + g8 + 8 * (i >> 1);
            float gate = acc[2 * u][i], up = acc[2 * u + 1][i];
            if constexpr (HAS_BIAS) {
              gate += __bfloat162float(bias[o]);
              up += __bfloat162float(bias[I + o]);
            }
            gate = fminf(gate, limit);
            up = fminf(fmaxf(up, -limit), limit);
            static_cast<__half *>(out)[((size_t)a * T + tok) * I + o] =
                __float2half((up + 1.0f) * gate /
                             (1.0f + __expf(-gate * alpha)));
          }
        } else {
          int const slot = routing[e * T + tok] - 1;
          if (slot >= 0) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
              int const row = row0 + tile_row(mt) + g8 + 8 * (i >> 1);
              float v = acc[mt][i];
              if constexpr (HAS_BIAS) {
                v += __bfloat162float(bias[row]);
              }
              static_cast<__nv_bfloat16 *>(
                  out)[((size_t)tok * TOPK + slot) * N + row] =
                  __float2bfloat16(v);
            }
          }
        }
      }
    }
  } else {
    L.gl = L.gi0 = c0;
  }
  sm90_task_end<SELF>(g);
  if constexpr (GLU) { // and per slot (after the barrier in sm90_task_end)
    if (threadIdx.x == 0) {
      cnt_add(g.cnt + SELF::counter + 1 + tk.x, 1);
    }
  }
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
