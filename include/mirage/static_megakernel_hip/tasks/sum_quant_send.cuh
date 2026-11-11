// tasks/sum_quant_send.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// sum_quant_send.cuh: the quantization of latent columns [128 kt, 128 kt + 128):
// add latent_down's NSPLIT partial sums (K parts in slots, COMBINE_SLOTS,
// polled: device-scope loads, a fixed order so every GPU gets the same z), MXFP8
// per (token, 32 columns): z_q e4m3 bytes + one e8m0 scale byte, then send the
// tile's bytes and its scale chunk to every GPU (push16: one system-scope store
// per GPU; the receivers poll the bytes, 0xFF-prefilled). The output's layout
// is CUDA's (z_q [T][L] e4m3, then the scale chunks [L / 128][512]); moe_experts
// lays it out in its LDS in the MFMA's fragment order. The threads work in groups
// of 32 (lane: columns 4 lane .. + 3): T <= 8 4 groups (threads 0..127), else 8;
// group w takes tokens TPG w .. TPG w + TPG - 1, two per round. AMD has no
// 128-thread barrier, so all 256 threads meet at every __syncthreads.
#pragma once
#include "../core.cuh"

namespace static_mk {

// STATIC_SQS_SCALES_LAST 1 (default 0; T > 8, Y 1): the tile's z_q bytes pushed, their stores acknowledged (vmcnt 0, a barrier),
// then its scale chunk: a landed scale row means its block's bytes landed (moe_experts_csr.cuh STATIC_CSR_ZDMA stages z_q by LDS
// DMA after one check of every scale row)
#ifndef STATIC_SQS_SCALES_LAST
#define STATIC_SQS_SCALES_LAST 0
#endif
__host__ __device__ constexpr size_t zq_scales_at(int L) {
  return (size_t)T * L;
}
// one 128-column scale chunk: 16 B per token (bytes 0..3 its 4 scales), at least SF_CHUNK (static_megakernel.zq_chunk)
constexpr int ZQ_CHUNK = SF_CHUNK > 16 * T ? SF_CHUNK : 16 * T;
__host__ __device__ constexpr size_t zq_bytes(int L) {
  return zq_scales_at(L) + (size_t)(L / 128) * ZQ_CHUNK;
}

// The tile (T x 128 z_q bytes, then its 512-B scale chunk) is made in LDS (sm)
// and pushed from there (MI355X 10-11: written to this GPU's exchange region
// with plain stores and loaded back for the push, an L2 round trip on z_q's
// path; push16 writes this GPU's copy too)
// Y > 1 (grid y, plan regrid; T >= 32): task (x, y) the tokens [NT y, NT y + NT) (NT = ceil(T / Y)), its z_q rows and its
// tokens' 16-B scale rows (task y = 0 also the chunk's rows past T); MI355X dec54, T = 64: 3 tasks of ~12.8 us, 4 polling
// rounds of 2 tokens per group each
// (timing build: stage stamp 0 = thread 0's first round of K parts landed, 1 = every group's tokens quantized, before the
// pushes; the task's end after the pushes' barrier)
// STATIC_SQS_ROUND R (default 2): a group's tokens polled R at a time (every load of a round in flight together); the rounds run one
// after the other (T >= 32: TPG 4 / 8 tokens per group, 2 / 4 rounds of a round trip each)
#ifndef STATIC_SQS_ROUND
#define STATIC_SQS_ROUND 2
#endif
constexpr int SQS_RT = STATIC_SQS_ROUND;
// STATIC_SQS_STAMPS 1 (timing build only, a diagnosis): stage stamp 0 = thread 0's LAST round of K parts landed, 1 = thread 0's group
// done (before the barrier that waits for the other groups)
#ifndef STATIC_SQS_STAMPS
#define STATIC_SQS_STAMPS 0
#endif
constexpr int SQS_TD = T < 32 ? T : 32; // tokens per DMA round
template <class Z, class... REST>
constexpr int sqs_nsplit() { return Z::y; } // (latent_down's K parts)
template <int NSPLIT, size_t ZOFF, int LAT, int Y = 1>
__device__ __forceinline__ void
    sum_quant_send_task(G const &g, float *zpart, int kt, char *sm, int ty = 0 STAGE_STAMP_PARAM) {
  constexpr size_t ZQ_SF = zq_scales_at(LAT);
  constexpr int NT = (T + Y - 1) / Y;                           // tokens per task
  constexpr int NG = NT <= 8 ? 4 : 8, TPG = (NT + NG - 1) / NG; // groups, tokens per group
  int const t0 = Y == 1 ? 0 : NT * ty, nt = Y == 1 ? T : (T - t0 < NT ? T - t0 : NT);
  int const grp = STATIC_TIDX >> 5, lane = STATIC_TIDX & 31;
  bool const worker = grp < NG;
  unsigned char *const zt = reinterpret_cast<unsigned char *>(sm),
                       *const chunk = zt + T * 128;
  __syncthreads(); // the LDS may still be read by the previous task
  for (int i = STATIC_TIDX; i < ZQ_CHUNK / 4; i += 256) { // the scale chunk: zero but for the T scale words (below)
    reinterpret_cast<uint32_t *>(chunk)[i] = 0u;
  }
  __syncthreads();
  for (int i0 = 0; worker && i0 < TPG; i0 += SQS_RT) {
    float4 zp[SQS_RT][NSPLIT];
    bool all;
    do { // the NSPLIT partial sums of both tokens; re-load until all landed
#pragma unroll
      for (int i = 0; i < SQS_RT; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          int const t = t0 + TPG * grp + i0 + i;
          zp[i][p_] = i0 + i < TPG && t < t0 + nt ? as_float4(ld16_at<AUX_DEV>(
              zpart, (uint32_t)((((size_t)p_ * T + t) * LAT +
                                 kt * 128 + 4 * lane) * 4))) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
      all = true;
#pragma unroll
      for (int i = 0; i < SQS_RT; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          all = all && validf4(zp[i][p_]);
        }
      if (!all) {
        poll_pause();
      }
    } while (!all);
#ifdef STATIC_TIMING_BUILD
    if (STATIC_TIDX == 0 && (STATIC_SQS_STAMPS ? i0 + SQS_RT >= TPG : i0 == 0)) {
      STAGE_STAMP_AT(0)
    }
#endif
#pragma unroll
    for (int i = 0; i < SQS_RT; i++) {
      int const t = t0 + TPG * grp + i0 + i;
      if (i0 + i >= TPG || t >= t0 + nt) {
        continue;
      }
      float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) {
        z.x += zp[i][p_].x;
        z.y += zp[i][p_].y;
        z.z += zp[i][p_].z;
        z.w += zp[i][p_].w;
      }
      // z rounded to bf16 before the MXFP8
      // quantization, as SGLang ROCm (its front GEMM writes z in bf16,
      // kimi_k3.py _front_fp32 is False on HIP; aiter quantizes those values)
      z = make_float4(round_bf16(z.x), round_bf16(z.y), round_bf16(z.z), round_bf16(z.w));
      float amax =
          fmaxf(fmaxf(fabsf(z.x), fabsf(z.y)), fmaxf(fabsf(z.z), fabsf(z.w)));
      for (int d = 4; d > 0; d >>= 1) { // over the 8 lanes of the 32 columns
        amax = fmaxf(amax, __shfl_xor(amax, d));
      }
      int ex = ceil_log2_bits(fmaxf(amax, 1.0e-30f) * (1.0f / 448.0f));
      ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
      float const sc = __uint_as_float((uint32_t)(127 - ex) << 23); // 2^-ex
      uint32_t const q = (uint32_t)cvt_e4m3(z.x * sc) |
                         ((uint32_t)cvt_e4m3(z.y * sc) << 8) |
                         ((uint32_t)cvt_e4m3(z.z * sc) << 16) |
                         ((uint32_t)cvt_e4m3(z.w * sc) << 24);
      *reinterpret_cast<uint32_t *>(zt + t * 128 + 4 * lane) = q;
      if ((lane & 7) == 0) {
        chunk[t * 16 + (lane >> 3)] = (uint8_t)(ex + 127);
      }
    }
  }
#ifdef STATIC_TIMING_BUILD
  if (STATIC_SQS_STAMPS && STATIC_TIDX == 0) {
    STAGE_STAMP_AT(1)
  }
#endif
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
  if (!STATIC_SQS_STAMPS && STATIC_TIDX == 0) {
    STAGE_STAMP_AT(1)
  }
#endif
  // to every GPU: the tile's z_q bytes (T tokens x 128 B) and its scale chunk (T <= 8: threads 0..63 the bytes, 64..95
  // the chunk, as before; else each piece by a loop over the threads)
  constexpr int NZ = T * 8, NC = ZQ_CHUNK / 16;
  if constexpr (Y > 1) { // this task's tokens' rows: nt x 8 pieces of z_q, nt scale rows (task 0: the rows past T too)
    int const nzv = nt * 8, ncv = nt + (ty == 0 ? NC - T : 0);
    for (int v = STATIC_TIDX; v < nzv + ncv; v += 256) {
      if (v < nzv) {
        int const t = t0 + (v >> 3), sg = v & 7;
        push16(g, ZOFF + (size_t)t * LAT + kt * 128 + sg * 16, *reinterpret_cast<uint4 const *>(zt + t * 128 + sg * 16));
      } else {
        int const r = v - nzv < nt ? t0 + (v - nzv) : T + (v - nzv - nt);
        push16(g, ZOFF + ZQ_SF + (size_t)kt * ZQ_CHUNK + r * 16, *reinterpret_cast<uint4 const *>(chunk + r * 16));
      }
    }
  } else if constexpr (NZ + NC <= 96 && NZ == 64) {
    if (STATIC_TIDX < 64) {
      int const t = STATIC_TIDX >> 3, sg = STATIC_TIDX & 7;
      push16(g,
             ZOFF + (size_t)t * LAT + kt * 128 + sg * 16,
             *reinterpret_cast<uint4 const *>(zt + t * 128 + sg * 16));
    } else if (STATIC_TIDX < 96) {
      push16(g,
             ZOFF + ZQ_SF + (size_t)kt * ZQ_CHUNK + (STATIC_TIDX - 64) * 16,
             *reinterpret_cast<uint4 const *>(chunk + (STATIC_TIDX - 64) * 16));
    }
  } else if constexpr (STATIC_SQS_SCALES_LAST) {
    for (int v = STATIC_TIDX; v < NZ; v += 256) {
      int const t = v >> 3, sg = v & 7;
      push16(g, ZOFF + (size_t)t * LAT + kt * 128 + sg * 16, *reinterpret_cast<uint4 const *>(zt + t * 128 + sg * 16));
    }
    wait_mem();
    __syncthreads();
    for (int v = STATIC_TIDX; v < NC; v += 256) {
      push16(g, ZOFF + ZQ_SF + (size_t)kt * ZQ_CHUNK + v * 16, *reinterpret_cast<uint4 const *>(chunk + v * 16));
    }
  } else {
    for (int v = STATIC_TIDX; v < NZ + NC; v += 256) {
      if (v < NZ) {
        int const t = v >> 3, sg = v & 7;
        push16(g, ZOFF + (size_t)t * LAT + kt * 128 + sg * 16, *reinterpret_cast<uint4 const *>(zt + t * 128 + sg * 16));
      } else {
        push16(g, ZOFF + ZQ_SF + (size_t)kt * ZQ_CHUNK + (v - NZ) * 16,
               *reinterpret_cast<uint4 const *>(chunk + (v - NZ) * 16));
      }
    }
  }
  __syncthreads();
}

template <class PARAMS>
constexpr bool sqs_skip() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > 1) {
    return PARAMS::v[1] == 1;
  } else {
    return false;
  }
}
// sum_quant_send task x = z columns [128 x, 128 x + 128); adds latent_down's
// Z::y partial sums (K parts in slots). PARAMS {L (z's width)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Z,
          class... REST>
__device__ __forceinline__ void run_sum_quant_send(Maps const &,
                                                   G const &g,
                                                   KernelLocals &KL,
                                                   StaticTask const &tk) {
  constexpr int L = PARAMS::v[0];
  if constexpr (sqs_skip<PARAMS>()) { // (params[1] 1: a placeholder - latent_down makes z_q itself, combine finish_quant; the node
    // keeps the graph's node numbers that plans and recipes name)
    return;
  } else {
  static_assert(
      L % 128 == 0 && Z::v[0] == COMBINE_SLOTS && Z::v[2] == L,
      "sum_quant_send: latent_down's L outputs, its K parts in slots");
  static_assert(SELF::buf >= 0 && exchange_bytes[SELF::buf] == zq_bytes(L),
                "sum_quant_send: its output, z_q, in the exchange region");
  sum_quant_send_task<Z::y, exchange_offset[SELF::buf], L, SELF::y>(
      g, buf_at<float>(g, Z::buf), tk.x, KL.sm, tk.y STAGE_STAMP_ARG(KL.stage_stamps));
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sum_quant_send() {
  return T * 128 + ZQ_CHUNK; // the tile, then its scale chunk
}

} // namespace static_mk
