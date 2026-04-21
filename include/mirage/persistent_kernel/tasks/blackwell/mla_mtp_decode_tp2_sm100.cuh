// MLA Multi-Token Decode for DeepSeek V3 on B200 (SM100a) — TP=2 (64 heads)
//
// Adapted from
// ferret/examples/mla-mtp-decode-q1to8-kv4096/v037_tp2_swapab_unrolled_reduce.cu
// Mechanical changes from the standalone:
//   - __global__ -> __device__ __noinline__
//   - blockIdx.{x,y,z} -> function parameters (block_x, block_y)
//   - __grid_constant__ CUtensorMap by-value -> const CUtensorMap* by pointer
//   - cudaTriggerProgrammaticLaunchCompletion() removed (PDL not supported by
//   MPK)
//   - Added thread guard for compatibility with larger MPK launch configs
// Kernel body (SMEM layout, mbarriers, MMA, softmax, PV, epilogue) is
// byte-identical.

#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>

namespace kernel {
namespace mla_mtp_tp2 {

// Named barrier 1 with count TB=128 for intra-kernel sync.
// Using a named barrier instead of __syncthreads() (barrier 0) because this
// kernel is called from MPK's 256-thread worker CTAs — threads 128..255
// return at the top and never reach __syncthreads(). On SM100 with
// Independent Thread Scheduling, the half-CTA that returned early drifts
// ahead in the outer worker loop, so subsequent __syncthreads() (bar 0)
// fire without warps 4..7 and those warps silently skip the next task's
// body, causing hangs (see bugfix.md Bug 14). bar.sync 1, 128 only syncs
// the active half of the CTA.
#define MLA_TP_SYNC_ACTIVE() asm volatile("bar.sync 1, 128;" ::: "memory")

static constexpr int NUM_HEADS = 64;
static constexpr int D_K = 576;
static constexpr int D_V = 512;
static constexpr int TILE_S = 128; // KV tokens per tile
static constexpr int BK = 64;      // 128B swizzle tile width
static constexpr int MMA_K = 16;
static constexpr int K_ITERS = D_K / BK;  // 9 for QK
static constexpr int V_CHUNKS = D_V / BK; // 8
static constexpr int TB = 128;

// Pipeline stages
static constexpr int NUM_QK_STAGES = 5;
static constexpr int NUM_PV_STAGES = 3;
static constexpr int MAX_STAGES = 5;

// SMEM tile: 128 rows × BK cols × 2 bytes = 16384
static constexpr int TILE_BYTES = 128 * BK * 2;

static constexpr int SMEM_SIZE = NUM_QK_STAGES * 2 * TILE_BYTES;

// Reduce kernel
static constexpr int RD_DV = 2;
static constexpr int RD_TB = 256;

// ============ PTX Helpers ============
namespace ptx {

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t p = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\t"
               "elect.sync _|%%px, 0xFFFFFFFF;\n\t"
               "@%%px mov.s32 %0, 1;\n\t}"
               : "+r"(p));
  return p;
}

__device__ __forceinline__ void mbar_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbar_wait(int addr, int phase) {
  asm volatile("{\n\t.reg .pred P;\n\t"
               "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, "
               "[%0], %1, 0x989680;\n\t"
               "@P bra DONE;\n\t"
               "bra WAIT;\n\t"
               "DONE:\n\t}" ::"r"(addr),
               "r"(phase));
}

__device__ __forceinline__ void mbar_tx(int addr, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4;
}

__device__ __forceinline__ uint64_t make_desc(int smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) |
         (2ULL << 61);
}

__device__ __forceinline__ void tcgen05_mma(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(
          taddr),
      "l"(a_desc),
      "l"(b_desc),
      "r"(idesc),
      "r"(acc));
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

} // namespace ptx

// ============ Main MLA Kernel ============
template <bool SINGLE_TILE>
__device__ __noinline__ void mla_mtp_tp2_main(CUtensorMap const *Q_tm_ptr,
                                              CUtensorMap const *KV_tm_ptr,
                                              nv_bfloat16 *__restrict__ Oa,
                                              float *__restrict__ La,
                                              float ss,
                                              int kv_len,
                                              int sk,
                                              int Q_LEN,
                                              int qpg,
                                              int block_x,
                                              int block_y) {
  if (threadIdx.x >= TB) {
    return;
  }
  int const tid = threadIdx.x;
  int const wid = tid / 32;

  int const gi = block_x / sk;
  int const si = block_x % sk;
  int const bi = block_y;

  int const num_groups = (Q_LEN + qpg - 1) / qpg;
  if (gi >= num_groups) {
    return;
  }

  int const kvt = (kv_len + TILE_S - 1) / TILE_S;
  int const tps = (kvt + sk - 1) / sk;
  int const t0 = si * tps;
  int const t1 = min(t0 + tps, kvt);
  if (t0 >= t1) {
    int block_linear = bi * num_groups * sk + gi * sk + si;
    if (tid < 128) {
      La[block_linear * 128 + tid] = -1e30f;
    }
    return;
  }

  int const hpb = NUM_HEADS;
  int const actual_qpg = min(qpg, Q_LEN - gi * qpg);

  extern __shared__ __align__(1024) char smem_buf[];
  int const smem_base = __cvta_generic_to_shared(smem_buf);
  int const work_smem = smem_base;

  __shared__ uint64_t mbar_buf[12];
  __shared__ int tmem_addr_buf[1];
  int const tma_bar = __cvta_generic_to_shared(&mbar_buf[0]);
  int const mma_bar = __cvta_generic_to_shared(&mbar_buf[MAX_STAGES]);
  int const mainloop_bar = __cvta_generic_to_shared(&mbar_buf[2 * MAX_STAGES]);

  // NOTE: CuTe's TMEM allocator requires tcgen05.alloc/dealloc to be issued
  // by the SAME warp across all kernels co-scheduled in the persistent
  // kernel. All FP8 CUTLASS kernels (linear_fp8, moe_linear, fp8_group_gemm)
  // use warp 0 for both. If MLA kernels use a different warp (e.g. warp 1),
  // subsequent FP8 kernels on the same worker hang inside their
  // tmem_allocator.allocate(). So we must also use warp 0 here.
  if (wid == 0) {
    if (ptx::elect_sync()) {
      for (int i = 0; i < MAX_STAGES; i++) {
        ptx::mbar_init(tma_bar + i * 8, 1);
        ptx::mbar_init(mma_bar + i * 8, 1);
      }
      ptx::mbar_init(mainloop_bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // All 32 threads of warp 0 issue tcgen05.alloc (sync.aligned requires
    // the full warp to participate).
    int addr_smem = __cvta_generic_to_shared(tmem_addr_buf);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(addr_smem),
        "r"(D_V));
  }
  MLA_TP_SYNC_ACTIVE();
  int const taddr = tmem_addr_buf[0];

  int const hpb_bytes = hpb * BK * 2;

  constexpr uint32_t idesc_qk = (1U << 4) | (1U << 7) | (1U << 10) |
                                ((uint32_t)(TILE_S >> 3) << 17) |
                                ((uint32_t)(128 >> 4) << 24);

  constexpr uint32_t idesc_pv = (1U << 4) | (1U << 7) | (1U << 10) |
                                (1U << 16) | ((uint32_t)(BK >> 3) << 17) |
                                ((uint32_t)(128 >> 4) << 24);

  int block_linear = bi * num_groups * sk + gi * sk + si;
  nv_bfloat16 *Oout = Oa + block_linear * D_V * 128;
  float row_max = -1e30f;
  float row_sum = 0.0f;

  float o_save[SINGLE_TILE ? 1 : 128];

  for (int tile = t0; tile < t1; tile++) {
    int const kvs = tile * TILE_S;
    int const tlen = min(TILE_S, kv_len - kvs);

    if (!SINGLE_TILE && tile > t0) {
      for (int c = 0; c < TILE_S; c += 16) {
        int addr = taddr + (tid << 16) + c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(o_save[c + 0]),
              "=f"(o_save[c + 1]),
              "=f"(o_save[c + 2]),
              "=f"(o_save[c + 3]),
              "=f"(o_save[c + 4]),
              "=f"(o_save[c + 5]),
              "=f"(o_save[c + 6]),
              "=f"(o_save[c + 7]),
              "=f"(o_save[c + 8]),
              "=f"(o_save[c + 9]),
              "=f"(o_save[c + 10]),
              "=f"(o_save[c + 11]),
              "=f"(o_save[c + 12]),
              "=f"(o_save[c + 13]),
              "=f"(o_save[c + 14]),
              "=f"(o_save[c + 15])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
      }
    }

    MLA_TP_SYNC_ACTIVE();
    if (wid == 0 && ptx::elect_sync()) {
      for (int i = 0; i < NUM_QK_STAGES; i++) {
        ptx::mbar_init(tma_bar + i * 8, 1);
        ptx::mbar_init(mma_bar + i * 8, 1);
      }
      ptx::mbar_init(mainloop_bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
    MLA_TP_SYNC_ACTIVE();

    if (wid == 0 && ptx::elect_sync()) {
      int phase = 0;
      for (int ki = 0; ki < K_ITERS; ki++) {
        int stage = ki % NUM_QK_STAGES;
        ptx::mbar_wait(mma_bar + stage * 8, phase ^ 1);
        if (stage == NUM_QK_STAGES - 1) {
          phase ^= 1;
        }

        int q_stage = work_smem + stage * 2 * TILE_BYTES;
        int k_stage = q_stage + TILE_BYTES;

        ptx::mbar_tx(tma_bar + stage * 8, TILE_BYTES + hpb_bytes * actual_qpg);
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::"
                     "complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(k_stage),
                     "l"(KV_tm_ptr),
                     "r"(0),
                     "r"(bi * kv_len + kvs),
                     "r"(ki),
                     "r"(tma_bar + stage * 8)
                     : "memory");
        for (int q = 0; q < actual_qpg; q++) {
          int actual_q_idx = gi * qpg + q;
          int global_row = bi * Q_LEN * NUM_HEADS + actual_q_idx * NUM_HEADS;
          asm volatile(
              "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_"
              "tx::bytes "
              "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(q_stage + q * hpb_bytes),
              "l"(Q_tm_ptr),
              "r"(0),
              "r"(global_row),
              "r"(ki),
              "r"(tma_bar + stage * 8)
              : "memory");
        }
      }
    } else if (wid == 1 && ptx::elect_sync()) {
      int phase = 0;
      for (int ki = 0; ki < K_ITERS; ki++) {
        int stage = ki % NUM_QK_STAGES;
        ptx::mbar_wait(tma_bar + stage * 8, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (stage == NUM_QK_STAGES - 1) {
          phase ^= 1;
        }

        int q_stage = work_smem + stage * 2 * TILE_BYTES;
        int k_stage = q_stage + TILE_BYTES;

        for (int k2 = 0; k2 < BK / MMA_K; k2++) {
          uint64_t a_desc = ptx::make_desc(q_stage + k2 * 32);
          uint64_t b_desc = ptx::make_desc(k_stage + k2 * 32);
          ptx::tcgen05_mma(
              taddr, a_desc, b_desc, idesc_qk, (ki == 0 && k2 == 0) ? 0 : 1);
        }
        ptx::tcgen05_commit(mma_bar + stage * 8);
      }
      ptx::tcgen05_commit(mainloop_bar);
    }

    MLA_TP_SYNC_ACTIVE();
    ptx::mbar_wait(mainloop_bar, 0);

    asm volatile("tcgen05.fence::after_thread_sync;");

    if (wid == 0 && ptx::elect_sync()) {
      for (int i = 0; i < NUM_PV_STAGES; i++) {
        ptx::mbar_init(tma_bar + i * 8, 1);
        ptx::mbar_init(mma_bar + i * 8, 1);
      }
      ptx::mbar_init(mainloop_bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
      int V_buf_base_pre = work_smem + 2 * TILE_BYTES;
      for (int vc = 0; vc < min(NUM_PV_STAGES, V_CHUNKS); vc++) {
        int v_smem = V_buf_base_pre + vc * TILE_BYTES;
        ptx::mbar_tx(tma_bar + vc * 8, TILE_BYTES);
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::"
                     "complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(v_smem),
                     "l"(KV_tm_ptr),
                     "r"(0),
                     "r"(bi * kv_len + kvs),
                     "r"(vc),
                     "r"(tma_bar + vc * 8)
                     : "memory");
      }
    }

    int q_in_group = tid / hpb;
    int actual_q_tid = gi * qpg + q_in_group;
    int effective_len;
    if (actual_q_tid < Q_LEN) {
      int causal_limit = kv_len;
      if (Q_LEN > 1) {
        causal_limit = kv_len - Q_LEN + actual_q_tid + 1;
      }
      effective_len = min(tlen, causal_limit - kvs);
      if (effective_len < 0) {
        effective_len = 0;
      }
    } else {
      effective_len = 0;
    }

    float const ss_log2e = ss * 1.4426950408889634f;
    float tile_max = -1e30f;
    for (int c = 0; c < TILE_S; c += 16) {
      float t16[16];
      int addr = taddr + (tid << 16) + c;
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
          : "=f"(t16[0]),
            "=f"(t16[1]),
            "=f"(t16[2]),
            "=f"(t16[3]),
            "=f"(t16[4]),
            "=f"(t16[5]),
            "=f"(t16[6]),
            "=f"(t16[7]),
            "=f"(t16[8]),
            "=f"(t16[9]),
            "=f"(t16[10]),
            "=f"(t16[11]),
            "=f"(t16[12]),
            "=f"(t16[13]),
            "=f"(t16[14]),
            "=f"(t16[15])
          : "r"(addr));
      asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
      for (int i = 0; i < 16; i++) {
        float v = (c + i < effective_len) ? t16[i] * ss_log2e : -1e30f;
        tile_max = fmaxf(tile_max, v);
      }
    }

    int P0_smem = work_smem;
    int P1_smem = work_smem + TILE_BYTES;
    float tile_sum = 0.0f;

    for (int half = 0; half < 2; half++) {
      int p_base = (half == 0) ? P0_smem : P1_smem;
      int row_base = p_base + tid * 128;

      for (int c = half * 64; c < half * 64 + 64; c += 16) {
        float t16[16];
        int addr = taddr + (tid << 16) + c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(t16[0]),
              "=f"(t16[1]),
              "=f"(t16[2]),
              "=f"(t16[3]),
              "=f"(t16[4]),
              "=f"(t16[5]),
              "=f"(t16[6]),
              "=f"(t16[7]),
              "=f"(t16[8]),
              "=f"(t16[9]),
              "=f"(t16[10]),
              "=f"(t16[11]),
              "=f"(t16[12]),
              "=f"(t16[13]),
              "=f"(t16[14]),
              "=f"(t16[15])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

#pragma unroll
        for (int i = 0; i < 16; i++) {
          float e = (c + i < effective_len)
                        ? exp2f(t16[i] * ss_log2e - tile_max)
                        : 0.0f;
          t16[i] = e;
          tile_sum += e;
        }

        int g_start = (c % 64);
#pragma unroll
        for (int gg = 0; gg < 16; gg += 8) {
          int g = g_start + gg;
          uint32_t w0, w1, w2, w3;
          {
            nv_bfloat16 b0 = __float2bfloat16(t16[gg + 0]);
            nv_bfloat16 b1 = __float2bfloat16(t16[gg + 1]);
            w0 = (uint32_t)(*(uint16_t *)&b0) |
                 ((uint32_t)(*(uint16_t *)&b1) << 16);
          }
          {
            nv_bfloat16 b0 = __float2bfloat16(t16[gg + 2]);
            nv_bfloat16 b1 = __float2bfloat16(t16[gg + 3]);
            w1 = (uint32_t)(*(uint16_t *)&b0) |
                 ((uint32_t)(*(uint16_t *)&b1) << 16);
          }
          {
            nv_bfloat16 b0 = __float2bfloat16(t16[gg + 4]);
            nv_bfloat16 b1 = __float2bfloat16(t16[gg + 5]);
            w2 = (uint32_t)(*(uint16_t *)&b0) |
                 ((uint32_t)(*(uint16_t *)&b1) << 16);
          }
          {
            nv_bfloat16 b0 = __float2bfloat16(t16[gg + 6]);
            nv_bfloat16 b1 = __float2bfloat16(t16[gg + 7]);
            w3 = (uint32_t)(*(uint16_t *)&b0) |
                 ((uint32_t)(*(uint16_t *)&b1) << 16);
          }
          int byte_off = g * 2;
          int swizzled =
              (byte_off & ~0xF) ^ ((tid & 7) << 4) | (byte_off & 0xF);
          int saddr = row_base + swizzled;
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(saddr),
                       "r"(w0),
                       "r"(w1),
                       "r"(w2),
                       "r"(w3));
        }
      }
    }

    float nm = fmaxf(row_max, tile_max);
    float corr = exp2f(row_max - nm);
    float ts = tile_sum * exp2f(tile_max - nm);

    if (!SINGLE_TILE && tile > t0) {
      MLA_TP_SYNC_ACTIVE();
      for (int c = TILE_S; c < D_V; c += 16) {
        float t16[16];
        int addr = taddr + (tid << 16) + c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(t16[0]),
              "=f"(t16[1]),
              "=f"(t16[2]),
              "=f"(t16[3]),
              "=f"(t16[4]),
              "=f"(t16[5]),
              "=f"(t16[6]),
              "=f"(t16[7]),
              "=f"(t16[8]),
              "=f"(t16[9]),
              "=f"(t16[10]),
              "=f"(t16[11]),
              "=f"(t16[12]),
              "=f"(t16[13]),
              "=f"(t16[14]),
              "=f"(t16[15])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
        for (int i = 0; i < 16; i++) {
          t16[i] *= corr;
        }
        uint32_t *u = (uint32_t *)t16;
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
            "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
                addr),
            "r"(u[0]),
            "r"(u[1]),
            "r"(u[2]),
            "r"(u[3]),
            "r"(u[4]),
            "r"(u[5]),
            "r"(u[6]),
            "r"(u[7]),
            "r"(u[8]),
            "r"(u[9]),
            "r"(u[10]),
            "r"(u[11]),
            "r"(u[12]),
            "r"(u[13]),
            "r"(u[14]),
            "r"(u[15]));
      }
    }

    row_max = nm;
    row_sum = corr * row_sum + ts;

    int V_buf_base = work_smem + 2 * TILE_BYTES;
    int pv_acc_base = (!SINGLE_TILE && tile > t0) ? 1 : 0;

    MLA_TP_SYNC_ACTIVE();

    if (wid == 0 && ptx::elect_sync()) {
      int phase = 0;
      for (int vc = 0; vc < NUM_PV_STAGES; vc++) {
        int stage = vc % NUM_PV_STAGES;
        if (stage == NUM_PV_STAGES - 1) {
          phase ^= 1;
        }
      }
      for (int vc = NUM_PV_STAGES; vc < V_CHUNKS; vc++) {
        int stage = vc % NUM_PV_STAGES;
        ptx::mbar_wait(mma_bar + stage * 8, phase ^ 1);
        if (stage == NUM_PV_STAGES - 1) {
          phase ^= 1;
        }

        int v_smem = V_buf_base + stage * TILE_BYTES;
        ptx::mbar_tx(tma_bar + stage * 8, TILE_BYTES);
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::"
                     "complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(v_smem),
                     "l"(KV_tm_ptr),
                     "r"(0),
                     "r"(bi * kv_len + kvs),
                     "r"(vc),
                     "r"(tma_bar + stage * 8)
                     : "memory");
      }
    } else if (wid == 1 && ptx::elect_sync()) {
      int phase = 0;
      for (int vc = 0; vc < V_CHUNKS; vc++) {
        int stage = vc % NUM_PV_STAGES;
        ptx::mbar_wait(tma_bar + stage * 8, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (stage == NUM_PV_STAGES - 1) {
          phase ^= 1;
        }

        int v_smem = V_buf_base + stage * TILE_BYTES;
        int out_taddr = taddr + vc * BK;

        int vc_acc_base = (vc < 2) ? 0 : pv_acc_base;
        int first_in_vc = 1;
        for (int k1 = 0; k1 < 2; k1++) {
          int p_addr = (k1 == 0) ? P0_smem : P1_smem;
          int v_k1_off = k1 * 64 * 128;
          for (int k2 = 0; k2 < BK / MMA_K; k2++) {
            uint64_t a_desc = ptx::make_desc(p_addr + k2 * 32);
            uint64_t b_desc = ptx::make_desc(v_smem + v_k1_off + k2 * 16 * 128);
            int acc = (first_in_vc && !vc_acc_base) ? 0 : 1;
            ptx::tcgen05_mma(out_taddr, a_desc, b_desc, idesc_pv, acc);
            first_in_vc = 0;
          }
        }
        ptx::tcgen05_commit(mma_bar + stage * 8);
      }
      ptx::tcgen05_commit(mainloop_bar);
    }

    MLA_TP_SYNC_ACTIVE();
    ptx::mbar_wait(mainloop_bar, 0);

    asm volatile("tcgen05.fence::after_thread_sync;");
    if (!SINGLE_TILE && tile > t0) {
      for (int c = 0; c < TILE_S; c += 16) {
        float t16[16];
        int addr = taddr + (tid << 16) + c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(t16[0]),
              "=f"(t16[1]),
              "=f"(t16[2]),
              "=f"(t16[3]),
              "=f"(t16[4]),
              "=f"(t16[5]),
              "=f"(t16[6]),
              "=f"(t16[7]),
              "=f"(t16[8]),
              "=f"(t16[9]),
              "=f"(t16[10]),
              "=f"(t16[11]),
              "=f"(t16[12]),
              "=f"(t16[13]),
              "=f"(t16[14]),
              "=f"(t16[15])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
        for (int i = 0; i < 16; i++) {
          t16[i] += corr * o_save[c + i];
        }
        uint32_t *u = (uint32_t *)t16;
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
            "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
                addr),
            "r"(u[0]),
            "r"(u[1]),
            "r"(u[2]),
            "r"(u[3]),
            "r"(u[4]),
            "r"(u[5]),
            "r"(u[6]),
            "r"(u[7]),
            "r"(u[8]),
            "r"(u[9]),
            "r"(u[10]),
            "r"(u[11]),
            "r"(u[12]),
            "r"(u[13]),
            "r"(u[14]),
            "r"(u[15]));
      }
    }
  }

  asm volatile("tcgen05.fence::after_thread_sync;");
  int const valid_rows = actual_qpg * hpb;
  if (tid < valid_rows) {
    float inv = (row_sum > 0) ? 1.0f / row_sum : 0.0f;
    for (int vc = 0; vc < V_CHUNKS; vc++) {
      int out_taddr_vc = taddr + vc * BK;
      for (int c = 0; c < BK; c += 16) {
        float t16[16];
        int addr = out_taddr_vc + (tid << 16) + c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(t16[0]),
              "=f"(t16[1]),
              "=f"(t16[2]),
              "=f"(t16[3]),
              "=f"(t16[4]),
              "=f"(t16[5]),
              "=f"(t16[6]),
              "=f"(t16[7]),
              "=f"(t16[8]),
              "=f"(t16[9]),
              "=f"(t16[10]),
              "=f"(t16[11]),
              "=f"(t16[12]),
              "=f"(t16[13]),
              "=f"(t16[14]),
              "=f"(t16[15])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        int base_d = (vc * BK + c) * 128 + tid;
#pragma unroll
        for (int i = 0; i < 16; i++) {
          nv_bfloat16 val = __float2bfloat16(t16[i] * inv);
          asm volatile("st.global.cs.b16 [%0], %1;" ::"l"(
                           (nv_bfloat16 *)(Oout + base_d + i * 128)),
                       "h"(*(uint16_t *)&val)
                       : "memory");
        }
      }
    }
    La[block_linear * 128 + tid] = log2f(fmaxf(row_sum, 1e-30f)) + row_max;
  }

  MLA_TP_SYNC_ACTIVE();
  if (wid == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(D_V));
  }
}

// ============ Reduction Kernel ============
__device__ __noinline__ void
    mla_mtp_tp2_reduce(nv_bfloat16 const *__restrict__ Oa,
                       float const *__restrict__ La,
                       nv_bfloat16 *__restrict__ O,
                       int sk,
                       int num_groups,
                       int Q_LEN,
                       int qpg,
                       int block_x,
                       int block_y,
                       int block_z) {
  if (threadIdx.x >= RD_TB) {
    return;
  }
  int const dv_base = block_x * RD_DV;
  int const gi = block_y;
  int const bi = block_z;
  int const tid = threadIdx.x;

  int const row = tid & 127;
  int const lane = tid >> 7;
  int const d = dv_base + lane;

  int q_in_group = row / NUM_HEADS;
  int h = row % NUM_HEADS;
  int actual_q = gi * qpg + q_in_group;

  if (actual_q >= Q_LEN || d >= D_V) {
    return;
  }

  float const *la_ptr = La + (bi * num_groups * sk + gi * sk) * 128 + row;
  nv_bfloat16 const *oa_ptr =
      Oa + (bi * num_groups * sk + gi * sk) * D_V * 128 + d * 128 + row;

  float maxVal = -1e30f, oldMaxVal = -1e30f;
  float sumVal = 0.0f;
  float acc = 0.0f;

  for (int s = 0; s < sk; s++) {
    float localMax = la_ptr[s * 128];
    float oa_val = __bfloat162float(oa_ptr[(size_t)s * D_V * 128]);

    maxVal = fmaxf(maxVal, localMax);
    float corr0 = exp2f(oldMaxVal - maxVal);
    float corr1 = exp2f(localMax - maxVal);
    oldMaxVal = maxVal;

    sumVal = sumVal * corr0 + corr1;
    acc = acc * corr0 + oa_val * corr1;
  }

  float inv_sum = (sumVal > 0.0f) ? __frcp_rn(sumVal) : 0.0f;
  int o_base = (bi * Q_LEN + actual_q) * NUM_HEADS * D_V + h * D_V;
  O[o_base + d] = __float2bfloat16(acc * inv_sum);
}

} // namespace mla_mtp_tp2
} // namespace kernel
