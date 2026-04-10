/* Copyright 2025 CMU
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

// MLA Multi-Token Prediction Decode for DeepSeek V3 on B200 (SM100a)
// Supports Q_LEN=1,2,3,4 with split-K parallelism, BS=1
// M=128 rows in MMA map to Q_LEN queries × (128/Q_LEN) heads per block
// KV shared across all queries (loaded once from SMEM)
//
// Two tasks: mla_mtp_decode (main attention) + mla_mtp_reduce (merge splits)
// Main: 128 threads, tcgen05 MMA, TMA, warp-specialized pipeline
// Reduce: 512 threads, coalesced reads, La cached in SMEM
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <math.h>
#include <stdint.h>

namespace kernel {

// Constants (DeepSeek V3 MLA)
namespace mla_mtp {

static constexpr int NUM_HEADS = 128;
static constexpr int D_K = 576;
static constexpr int D_V = 512;
static constexpr int TILE_S = 128;
static constexpr int BK = 64;
static constexpr int MMA_K = 16;
static constexpr int K_ITERS = D_K / BK;   // 9
static constexpr int V_CHUNKS = D_V / BK;  // 8
static constexpr int TB = 128;

static constexpr int NUM_QK_STAGES = 5;
static constexpr int NUM_PV_STAGES = 3;
static constexpr int MAX_STAGES = 5;

static constexpr int TILE_BYTES = 128 * BK * 2;

// Reduce constants
static constexpr int RD_DV = 4;
static constexpr int RD_TB = 512;
static constexpr int MAX_SK = 32;

// SMEM for main kernel
static constexpr int MTP_SMEM_SIZE = NUM_QK_STAGES * 2 * TILE_BYTES; // 160KB

#include "sm100_ptx.cuh"

} // namespace mla_mtp

// ============ MLA MTP Decode Device Function ============
// blockIdx.x → decomposed into (gi, si) via params
// blockIdx.y → bi via param
template<bool SINGLE_TILE>
__device__ __noinline__ void mla_mtp_decode_sm100_task_impl(
    const CUtensorMap *Q_tm_ptr,
    const CUtensorMap *KV_tm_ptr,
    nv_bfloat16* __restrict__ Oa, float* __restrict__ La,
    float ss, int kv_len, int sk, int num_head_groups, int Q_LEN,
    int gi, int si, int bi  // head_group, split_idx, batch_idx
) {
    using namespace mla_mtp;
    using namespace kernel::sm100_ptx;

    const int tid = threadIdx.x;
    if (tid >= TB) return; // guard for MPK's 256-thread workers
    const int wid = tid / 32;

    if (gi >= num_head_groups) return;

    const int kvt = (kv_len + TILE_S - 1) / TILE_S;
    const int tps = (kvt + sk - 1) / sk;
    const int t0 = si * tps;
    const int t1 = min(t0 + tps, kvt);
    if (t0 >= t1) return;

    const int hpb = NUM_HEADS / num_head_groups;

    extern __shared__ __align__(1024) char smem_buf[];
    const int smem_base = __cvta_generic_to_shared(smem_buf);

    const int work_smem = smem_base;

    __shared__ uint64_t mbar_buf[12];
    __shared__ int tmem_addr_buf[1];
    const int tma_bar = __cvta_generic_to_shared(&mbar_buf[0]);
    const int mma_bar = __cvta_generic_to_shared(&mbar_buf[MAX_STAGES]);
    const int mainloop_bar = __cvta_generic_to_shared(&mbar_buf[2 * MAX_STAGES]);
    const int q_bar = __cvta_generic_to_shared(&mbar_buf[2 * MAX_STAGES + 1]);

    if (wid == 0 && elect_sync()) {
        for (int i = 0; i < MAX_STAGES; i++) {
            mbar_init(tma_bar + i * 8, 1);
            mbar_init(mma_bar + i * 8, 1);
        }
        mbar_init(mainloop_bar, 1);
        mbar_init(q_bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (wid == 1) {
        int addr_smem = __cvta_generic_to_shared(tmem_addr_buf);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr_smem), "r"(D_V));
    }
    __syncthreads();
    const int taddr = tmem_addr_buf[0];

    const int hpb_bytes = hpb * BK * 2;

    constexpr uint32_t idesc_qk = (1U << 4) | (1U << 7) | (1U << 10)
        | ((uint32_t)(TILE_S >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
    constexpr uint32_t idesc_pv = (1U << 4) | (1U << 7) | (1U << 10)
        | (1U << 16)
        | ((uint32_t)(BK >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);

    int block_linear = bi * num_head_groups * sk + gi * sk + si;
    nv_bfloat16* Oout = Oa + block_linear * D_V * 128;
    float row_max = -1e30f;
    float row_sum = 0.0f;

    float o_save[128];

    for (int tile = t0; tile < t1; tile++) {
        const int kvs = tile * TILE_S;
        const int tlen = min(TILE_S, kv_len - kvs);

        if (tile > t0) {
            for (int c = 0; c < TILE_S; c += 16) {
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(o_save[c+0]), "=f"(o_save[c+1]), "=f"(o_save[c+2]), "=f"(o_save[c+3]),
                      "=f"(o_save[c+4]), "=f"(o_save[c+5]), "=f"(o_save[c+6]), "=f"(o_save[c+7]),
                      "=f"(o_save[c+8]), "=f"(o_save[c+9]), "=f"(o_save[c+10]), "=f"(o_save[c+11]),
                      "=f"(o_save[c+12]), "=f"(o_save[c+13]), "=f"(o_save[c+14]), "=f"(o_save[c+15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
            }
        }

        // QK Phase
        __syncthreads();
        if (wid == 0 && elect_sync()) {
            for (int i = 0; i < NUM_QK_STAGES; i++) {
                mbar_init(tma_bar + i * 8, 1);
                mbar_init(mma_bar + i * 8, 1);
            }
            mbar_init(mainloop_bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();

        if (wid == 0 && elect_sync()) {
            int phase = 0;
            for (int ki = 0; ki < K_ITERS; ki++) {
                int stage = ki % NUM_QK_STAGES;
                mbar_wait(mma_bar + stage * 8, phase ^ 1);
                if (stage == NUM_QK_STAGES - 1) phase ^= 1;

                int q_stage = work_smem + stage * 2 * TILE_BYTES;
                int k_stage = q_stage + TILE_BYTES;

                mbar_tx(tma_bar + stage * 8, TILE_BYTES + hpb_bytes * Q_LEN);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(k_stage), "l"(KV_tm_ptr),
                       "r"(0), "r"(bi * kv_len + kvs), "r"(ki), "r"(tma_bar + stage * 8) : "memory");
                for (int q = 0; q < Q_LEN; q++) {
                    int global_row = bi * Q_LEN * NUM_HEADS + q * NUM_HEADS + gi * hpb;
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1, {%2, %3, %4}], [%5];"
                        :: "r"(q_stage + q * hpb_bytes), "l"(Q_tm_ptr),
                           "r"(0), "r"(global_row), "r"(ki), "r"(tma_bar + stage * 8) : "memory");
                }
            }
        }
        else if (wid == 1 && elect_sync()) {
            int phase = 0;
            for (int ki = 0; ki < K_ITERS; ki++) {
                int stage = ki % NUM_QK_STAGES;
                mbar_wait(tma_bar + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (stage == NUM_QK_STAGES - 1) phase ^= 1;

                int q_stage = work_smem + stage * 2 * TILE_BYTES;
                int k_stage = q_stage + TILE_BYTES;

                for (int k2 = 0; k2 < BK / MMA_K; k2++) {
                    uint64_t a_desc = make_desc(q_stage + k2 * 32);
                    uint64_t b_desc = make_desc(k_stage + k2 * 32);
                    tcgen05_mma(taddr, a_desc, b_desc, idesc_qk, (ki == 0 && k2 == 0) ? 0 : 1);
                }
                tcgen05_commit(mma_bar + stage * 8);
            }
            tcgen05_commit(mainloop_bar);
        }

        __syncthreads();
        mbar_wait(mainloop_bar, 0);

        // Softmax Phase
        asm volatile("tcgen05.fence::after_thread_sync;");

        if (wid == 0 && elect_sync()) {
            for (int i = 0; i < NUM_PV_STAGES; i++) {
                mbar_init(tma_bar + i * 8, 1);
                mbar_init(mma_bar + i * 8, 1);
            }
            mbar_init(mainloop_bar, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
            int v_smem0 = work_smem + 2 * TILE_BYTES;
            mbar_tx(tma_bar, TILE_BYTES);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3, %4}], [%5];"
                :: "r"(v_smem0), "l"(KV_tm_ptr),
                   "r"(0), "r"(bi * kv_len + kvs), "r"(0), "r"(tma_bar) : "memory");
        }

        int q_idx = tid / hpb;
        int causal_limit = kv_len;
        if (Q_LEN > 1) {
            causal_limit = kv_len - Q_LEN + q_idx + 1;
        }
        int effective_len = min(tlen, causal_limit - kvs);
        if (effective_len < 0) effective_len = 0;
        if (q_idx >= Q_LEN) effective_len = 0;

        int P0_smem = work_smem;
        int P1_smem = work_smem + TILE_BYTES;

        // Pass 1: Find global max
        float tile_max = -1e30f;
        for (int c = 0; c < TILE_S; c += 16) {
            float t16[16];
            int addr = taddr + (tid << 16) + c;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                  "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                  "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                  "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float v = (c + i < effective_len) ? t16[i] * ss : -1e30f;
                tile_max = fmaxf(tile_max, v);
            }
        }

        // Pass 2: Compute exp, write P, accumulate sum
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
                    : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                      "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                      "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                      "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    float e = (c + i < effective_len) ? __expf(t16[i] * ss - tile_max) : 0.0f;
                    t16[i] = e;
                    tile_sum += e;
                }

                int g_start = (c % 64);
                #pragma unroll
                for (int gg = 0; gg < 16; gg += 8) {
                    int g = g_start + gg;
                    uint32_t w0, w1, w2, w3;
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+0]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+1]);
                        w0 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+2]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+3]);
                        w1 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+4]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+5]);
                        w2 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    {
                        nv_bfloat16 b0 = __float2bfloat16(t16[gg+6]);
                        nv_bfloat16 b1 = __float2bfloat16(t16[gg+7]);
                        w3 = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
                    }
                    int byte_off = g * 2;
                    int swizzled = (byte_off & ~0xF) ^ ((tid & 7) << 4) | (byte_off & 0xF);
                    int saddr = row_base + swizzled;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
                                 :: "r"(saddr), "r"(w0), "r"(w1), "r"(w2), "r"(w3));
                }
            }
        }

        float nm = fmaxf(row_max, tile_max);
        float corr = __expf(row_max - nm);
        float ts = tile_sum * __expf(tile_max - nm);
        __syncthreads();

        // Scale O[128:511] in TMEM
        if (tile > t0) {
            for (int c = TILE_S; c < D_V; c += 16) {
                float t16[16];
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                      "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                      "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                      "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int i = 0; i < 16; i++) t16[i] *= corr;
                uint32_t* u = (uint32_t*)t16;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                    :: "r"(addr),
                       "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                       "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7]),
                       "r"(u[8]), "r"(u[9]), "r"(u[10]), "r"(u[11]),
                       "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]));
            }
        }

        row_max = nm;
        row_sum = corr * row_sum + ts;

        // PV Phase
        int V_buf_base = work_smem + 2 * TILE_BYTES;
        int pv_acc_base = (tile > t0) ? 1 : 0;

        __syncthreads();

        if (wid == 0 && elect_sync()) {
            int phase = 0;
            for (int vc = 1; vc < V_CHUNKS; vc++) {
                int stage = vc % NUM_PV_STAGES;
                mbar_wait(mma_bar + stage * 8, phase ^ 1);
                if (stage == NUM_PV_STAGES - 1) phase ^= 1;
                int v_smem = V_buf_base + stage * TILE_BYTES;
                mbar_tx(tma_bar + stage * 8, TILE_BYTES);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"(v_smem), "l"(KV_tm_ptr),
                       "r"(0), "r"(bi * kv_len + kvs), "r"(vc), "r"(tma_bar + stage * 8) : "memory");
            }
        }
        else if (wid == 1 && elect_sync()) {
            int phase = 0;
            for (int vc = 0; vc < V_CHUNKS; vc++) {
                int stage = vc % NUM_PV_STAGES;
                mbar_wait(tma_bar + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (stage == NUM_PV_STAGES - 1) phase ^= 1;
                int v_smem = V_buf_base + stage * TILE_BYTES;
                int out_taddr = taddr + vc * BK;
                int vc_acc_base = (vc < 2) ? 0 : pv_acc_base;
                int first_in_vc = 1;
                for (int k1 = 0; k1 < 2; k1++) {
                    int p_addr = (k1 == 0) ? P0_smem : P1_smem;
                    int v_k1_off = k1 * 64 * 128;
                    for (int k2 = 0; k2 < BK / MMA_K; k2++) {
                        uint64_t a_desc = make_desc(p_addr + k2 * 32);
                        uint64_t b_desc = make_desc(v_smem + v_k1_off + k2 * 16 * 128);
                        int acc = (first_in_vc && !vc_acc_base) ? 0 : 1;
                        tcgen05_mma(out_taddr, a_desc, b_desc, idesc_pv, acc);
                        first_in_vc = 0;
                    }
                }
                tcgen05_commit(mma_bar + stage * 8);
            }
            tcgen05_commit(mainloop_bar);
        }

        __syncthreads();
        mbar_wait(mainloop_bar, 0);

        // Merge saved O[0:127]
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (tile > t0) {
            for (int c = 0; c < TILE_S; c += 16) {
                float t16[16];
                int addr = taddr + (tid << 16) + c;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                      "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                      "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                      "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    t16[i] += corr * o_save[c + i];
                uint32_t* u = (uint32_t*)t16;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                    :: "r"(addr),
                       "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                       "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7]),
                       "r"(u[8]), "r"(u[9]), "r"(u[10]), "r"(u[11]),
                       "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]));
            }
        }
    }

    // Epilogue: normalize and write to Oa
    asm volatile("tcgen05.fence::after_thread_sync;");
    float inv = (row_sum > 0) ? 1.0f / row_sum : 0.0f;
    for (int vc = 0; vc < V_CHUNKS; vc++) {
        int out_taddr = taddr + vc * BK;
        for (int c = 0; c < BK; c += 16) {
            float t16[16];
            int addr = out_taddr + (tid << 16) + c;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=f"(t16[0]), "=f"(t16[1]), "=f"(t16[2]), "=f"(t16[3]),
                  "=f"(t16[4]), "=f"(t16[5]), "=f"(t16[6]), "=f"(t16[7]),
                  "=f"(t16[8]), "=f"(t16[9]), "=f"(t16[10]), "=f"(t16[11]),
                  "=f"(t16[12]), "=f"(t16[13]), "=f"(t16[14]), "=f"(t16[15])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int base_d = (vc * BK + c) * 128 + tid;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                nv_bfloat16 val = __float2bfloat16(t16[i] * inv);
                Oout[base_d + i * 128] = val;
            }
        }
    }

    La[block_linear * 128 + tid] = logf(fmaxf(row_sum, 1e-30f)) + row_max;

    __syncthreads();
    if (wid == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(D_V));
    }
}

// ============ MLA MTP Reduce Device Function ============
// NUM_THREADS_: 256 for MPK, 512 for standalone.
// row = tid % 128, lane = tid / 128. Lanes = NUM_THREADS_ / 128.
template <int NUM_THREADS_ = 512>
__device__ __noinline__ void mla_mtp_reduce_sm100_task_impl(
    const nv_bfloat16* __restrict__ Oa,
    const float* __restrict__ La,
    nv_bfloat16* __restrict__ O,
    int sk, int num_head_groups, int Q_LEN,
    int dv_base, int gi, int bi
) {
    using namespace mla_mtp;
    using namespace kernel::sm100_ptx;
    const int tid = threadIdx.x;

    const int row = tid % 128;
    const int lane = tid / 128;
    const int d = dv_base + lane;

    int hpb = NUM_HEADS / num_head_groups;
    int q = row / hpb;
    int h_local = row % hpb;
    int h_global = gi * hpb + h_local;

    __shared__ float la_smem[MAX_SK * 128];
    int la_block_base = (bi * num_head_groups * sk + gi * sk) * 128;
    int la_total = sk * 128;
    for (int i = tid; i < la_total; i += NUM_THREADS_)
        la_smem[i] = La[la_block_base + i];
    __syncthreads();

    if (q >= Q_LEN || h_global >= NUM_HEADS) return;

    int o_base = (bi * Q_LEN + q) * NUM_HEADS * D_V + h_global * D_V;

    float lse_max = -1e30f;
    float sum_exp = 0.0f;
    for (int s = 0; s < sk; s++) {
        float la_val = la_smem[s * 128 + row];
        float new_max = fmaxf(lse_max, la_val);
        sum_exp = sum_exp * __expf(lse_max - new_max) + __expf(la_val - new_max);
        lse_max = new_max;
    }
    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

    if (d < D_V) {
        float acc = 0.0f;
        int oa_base_gi = (bi * num_head_groups * sk + gi * sk) * D_V * 128;
        int oa_d = oa_base_gi + d * 128 + row;
        for (int s = 0; s < sk; s++) {
            float scale = __expf(la_smem[s * 128 + row] - lse_max) * inv_sum;
            acc += scale * __bfloat162float(Oa[oa_d + s * D_V * 128]);
        }
        O[o_base + d] = __float2bfloat16(acc);
    }
}

} // namespace kernel
