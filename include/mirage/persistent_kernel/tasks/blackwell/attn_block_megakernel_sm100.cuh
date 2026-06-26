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

// MPK port of the ferret fused DSv3 decode ATTENTION BLOCK megakernel
// (scratch/megakernels/attn_block_ferret_v126_cold87us_grid136.cuh, 87us cold,
// grid=136, correct vs its own gate). The attention analog of the FFN mega-task
// (ffn_full_megakernel_sm100.cuh). One task runs the WHOLE decode attention:
//   input was rmsnorm'd by the prior task -> qkv_a GEMV -> q_a_ln + kv_a_ln ->
//   q_b GEMV -> YaRN rope(q,k) -> kv_append -> MLA decode (flash, KV-split) ->
//   reduce -> W_UV per-head BMM -> o_proj + residual.
//
// THREE structural adaptations vs the standalone ferret kernel:
//   1. The standalone kernel kept inter-stage activations in GLOBAL __device__
//      arrays (`g_hdeq`, `g_hf8`, ...). That is ILLEGAL in MPK (concurrent
//      layers/tasks would clash on the single global instance). They are moved
//      into a per-task SCRATCH buffer (`Scratch`/`make_scratch`, bound via an
//      input_ptr), exactly like the FFN mega-task's `sc.*`.
//   2. The standalone kernel was cooperative-launched (cg::this_grid().sync()).
//      MPK is NOT cooperative. Every `grid.sync()` is replaced by the MPK
//      atomic `grid_barrier(GridBarrier, NUM_WORKERS)` (the SAME helper as the
//      FFN mega-task; counters live at the top of the scratch buffer).
//   3. `step` (decode position) comes from runtime_config.step[0] (sourced in
//      the task_register snippet, the same way mla_mtp_decode does it).
//
// WEIGHT-SCALE FORMAT (reconciled to the MPK production layout): the three
// dense GEMVs (qkv_a, q_b, o_proj) read qkv_a_s / q_b_s / oproj_s as the MPK
// PER-128-ROW-BLOCK, per-128-K-group fp32 weight_scale_inv [N/128, K/128]
// (row-major) that _attach_fp8_weight already produces — read as a PLAIN fp32,
// IDENTICAL to the production fp8_gemm_dense_finen GEMV (`sb[(col>>7)*nk + g]`,
// no UE8M0 decode; the per-block fp32 value already IS the decoded power-of-2).
// This is a layout/encoding change vs the standalone ferret gate (which used a
// per-ROW UE8M0-packed uint32 scale) — NOT a math change: the multiplied VALUE
// is the same power-of-2, just laid out per-128-block instead of per-row, so
// cosine is unaffected. kvbv_s is the per-head fp32 [H,1,4] kv_b_v_bmm_dense
// scale and ALSO matches the kernel's `const float* kvbv_s`.
#pragma once

#include "mirage/persistent_kernel/runtime_header.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#ifdef MPK_ATTN_DBG
#include <cstdio> // device printf for the per-stage debug taps (default-OFF)
#endif

// ---- MPK grid barrier (VERBATIM from ffn_full_megakernel_sm100.cuh) ----------
struct AttnGridBarrier {
  unsigned int *count; // [1] arrivals in the current generation
  unsigned int *gen;   // [1] generation (sense) counter
};

// Block-collective: call with the WHOLE block; only thread 0 touches global
// mem.
__device__ __forceinline__ void attn_grid_barrier(AttnGridBarrier b,
                                                  int num_participants) {
  __syncthreads(); // all threads of THIS worker arrive first
  // Publish THIS worker's global writes to all CTAs BEFORE arriving — a relaxed
  // atomic count-bump does NOT order prior regular global stores.
  __threadfence();
  __syncthreads(); // ensure EVERY thread's __threadfence has retired before
                   // thread 0 bumps the count / flips the gen.
  if (threadIdx.x == 0) {
    unsigned int my_gen = *((unsigned int volatile *)b.gen);
    unsigned int prev = atomicAdd(b.count, 1u);
    if (prev + 1u == (unsigned int)num_participants) {
      *b.count = 0u;        // reset for the next generation
      __threadfence();      // publish the reset before the flip
      atomicAdd(b.gen, 1u); // release: flip the sense
    } else {
      while (*((unsigned int volatile *)b.gen) == my_gen) { /* spin */
      }
    }
  }
  __syncthreads(); // re-converge this worker's threads, acquire the new gen
}

namespace kernel {
namespace attn_block_megakernel_sm100 {

#define K_HIDDEN 7168
#define K_QLORA 1536
#define K_KVLORA 512
#define K_QKROPE 64
#define K_QKHEAD 576
#define K_VHEAD 128
#define K_QKVAN 2176
#define K_HLOCAL 16
#define K_OIN 2048
#define K_GRP 128
#define K_FP8MAX 448.0f
#define K_EPS 1e-6f
#define NTHREAD 256
#define NWARP 8
#define RB 2 // output rows per warp iteration (memory-level parallelism)

// MPK worker count on B200 (148 SM). Builder asserts num_workers==136. This is
// the grid_barrier participant count (the megakernel is NOT
// cooperative-launched so gridDim.x is the worker count, not
// min(numSM*bps,136)).
#define ATTN_NUM_WORKERS 136

// Flash MLA KV-split: up to MLA_SPLITS splits per head.
#define MLA_SPLITS 8

// ---- BARRIER-REMOVAL LEVERS (ported from ferret workspace6 v131, 87->64.5us
// cold; the 5 levers each remove/merge a grid.sync). MPK_DSV3_ATTN_FAST is
// defaulted ON, so the fast path is the default WITHIN the attn-block
// megakernel (which is itself the default decode attention path); set
// -DMPK_DSV3_ATTN_FAST=0 to fall back to the proven-correct 9-barrier baseline
// (commit 52ed6e64) for A/B regression isolation. The five levers (all gated
// together, matching ferret v131 which ships them all-on):
//   1. HIDDEN_BLOCK_LOCAL — quant hidden[7168] block-local into s_act, removes
//      the quant_hidden->qkv_a barrier (a block __syncthreads replaces it).
//   2. QA_BLOCK_LOCAL     — q_a_layernorm+requant block-local into s_qbdeq,
//      removes the q_a_layernorm->q_b barrier.
//   3. OPROJ_BLOCK_QUANT  — g_red UE8M0 quant block-local into s_odeq, removes
//      the quant_gred->o_proj barrier (merges the old quant+o_proj two-barrier
//      pair into one — the W_UV->* publish barrier is KEPT).
//   4. MLA_ATOMIC_MERGE   — per-head atomicAdd completion counter; the LAST
//      split-block of head h runs that head's merge in-place (device-scope
//      __threadfence release/acquire), removes the MLA-partial->merge barrier.
//   5. WUV_HEAD_SPINWAIT  — per-head .release.gpu/.acquire.gpu readiness flag;
//      W_UV row-blocks spin-wait per head, removes the merge->W_UV barrier.
// Levers 4/5 use DEVICE-scope ordering (intra-GPU, one rank's 16 heads) — NOT
// .sys — which is correct and required (the heads are all on this rank).
#ifndef MPK_DSV3_ATTN_FAST
#define MPK_DSV3_ATTN_FAST 1
#endif

// ---- Per-stage debug taps (default-OFF; compile with -DMPK_ATTN_DBG) --------
// Enable: add -DMPK_ATTN_DBG to the megakernel nvcc flags (e.g. via the
// extra-defines env the runtime forwards), then run with the demo. At decode
// step 0 ONLY, worker 0 / thread 0 prints, for EACH stage, the stage label, the
// `out` buffer pointer (disambiguates the per-layer task — e.g. layer 3 is the
// one whose `out` matches layer_3_attnmega_attn_proj_fused), the first 4 fp32
// values, and a sum-of-abs checksum of the stage's scratch output. The device
// printf FIFO must be bumped on the HOST before CUDA init (cuCtxSetLimit
// cudaLimitPrintfFifoSize, e.g. 32MB) or low-volume prints may not flush — note
// for the main agent. Compiles to NOTHING when MPK_ATTN_DBG is undefined, so
// the default build stays byte-identical and perf/the gate watchdog are
// unaffected.
#ifdef MPK_ATTN_DBG
// Fire the taps at steps 0, 1, AND 2 — so tokens 0,1,2's per-step HIDDEN input
// and raw c_latent can be compared directly (the decisive check: tokens 0,1
// produce near-identical c_latent → is the HIDDEN itself near-identical for
// steps 0,1 [input-binding bug] or does the GEMV flatten distinct hiddens?).
// Fire at clean DISTINCT-token prefill positions (2,3 — tokens[2]=128803,
// tokens[3]=45585; NOT 0/1 which are the duplicate token-0) AND the FIRST
// decode step (14, the first GENERATED token after a 14-token prompt). Step 14
// is the SAFE decode comparison vs the chain: both configs still consume the
// IDENTICAL prompt up to position 13, so the layer-0 input at step 14 matches
// (free-running generation diverges by later steps, so a late step is NOT
// chain-comparable — Codex). The nsp>1 MLA-merge path (KV>64) needs a separate
// forced/replayed-token test, not a free-running step. Adjust 14 to
// (prompt_len) if the prompt length differs.
#define ATTN_DBG_STEP(s) ((s) == 2 || (s) == 3 || (s) == 14)
__device__ __forceinline__ void attn_dbg_tap(char const *label,
                                             void const *out_id,
                                             float const *v,
                                             int n,
                                             int step,
                                             int worker_idx) {
  if (!ATTN_DBG_STEP(step) || worker_idx != 0 || threadIdx.x != 0) {
    return;
  }
  float s = 0.f;
  for (int i = 0; i < n; i++) {
    s += fabsf(v[i]);
  }
  printf("[ATTN_DBG out=%p step=%d] %-10s n=%d sum|.|=%.6f  v[0..3]= %.5f "
         "%.5f %.5f %.5f\n",
         out_id,
         step,
         label,
         n,
         s,
         (n > 0 ? v[0] : 0.f),
         (n > 1 ? v[1] : 0.f),
         (n > 2 ? v[2] : 0.f),
         (n > 3 ? v[3] : 0.f));
}
#define ATTN_DBG_TAP(label, out_id, v, n, step, worker_idx)                    \
  attn_dbg_tap(label, out_id, v, n, step, worker_idx)
#else
#define ATTN_DBG_TAP(label, out_id, v, n, step, worker_idx)                    \
  do {                                                                         \
  } while (0)
#endif

// cos/sin are bound as ONE concatenated buffer [cos(64) | sin(64)] per max_seq
// row (stride 128) to stay under MAX_INPUTS_PER_TASK=14. cos for position `pos`
// at cos_sin[pos*K_COSSIN_STRIDE + d]; sin at cos_sin[pos*K_COSSIN_STRIDE + 64
// + d].
#define K_COSSIN_STRIDE 128
#define K_COSSIN_SINOFF 64

// ===========================================================================
//  Per-task SCRATCH (replaces the standalone kernel's global __device__
//  arrays). Laid out after the 8-byte barrier (count, gen). Every section is
//  16-byte aligned. Scalar (float / fp8) access only — no uint4 reads land in
//  scratch (the cp.async weight ring is in dynamic smem; kv_cache is an
//  external input).
// ===========================================================================
struct AttnScratch {
  float *g_hdeq;         // dequant hidden (S2 act)                    [7168]
  __nv_fp8_e4m3 *g_hf8;  // fp8 hidden                                 [7168]
  float *g_hsc;          // per-group act scale                        [56]
  __nv_fp8_e4m3 *g_qbf8; // fp8 q_a_normed (q_b act)                   [1536]
  float *g_qbsc;         // per-group act scale                        [12]
  float *g_qkva;         // qkv_a_out (G1)                             [2176]
  float *g_qbdeq;        // dequant q_a_normed (S4 act)                [1536]
  float *g_qpe;          // q_b out / post-rope q (G4)        [16*576 = 9216]
  float *g_attn;         // attn_out (G6)                     [16*512 = 8192]
  float *g_attn_deq;     // dequant attn (S12 act)            [16*512 = 8192]
  float *g_red;          // W_UV out (G7)                             [2048]
  float *g_odeq;         // dequant g_red (S13 act)                   [2048]
  float *g_mla_acc;      // un-normalized sum(p*V)         [16*8*512 = 65536]
  float *g_mla_m;        // partial row-max                   [16*8    = 128]
  float *g_mla_l;        // partial exp-sum                   [16*8    = 128]
  // MLA_ATOMIC_MERGE (lever 4): per-head finished-split counter. Zeroed BEFORE
  // the q_b->MLA barrier (that barrier's __threadfence publishes the zeros to
  // every CTA); the LAST split-block of head h (atomicAdd return == nsp-1) runs
  // the in-place merge -> drops the partial->merge barrier.            [16]
  int *g_head_done;
  // WUV_HEAD_SPINWAIT (lever 5): per-head "merge done, W_UV may read"
  // readiness flag. Zeroed alongside g_head_done before the SAME barrier; the
  // merge sets it (device release), the W_UV row-blocks spin-wait (device
  // acquire) -> drops the merge->W_UV barrier.                          [16]
  int *g_head_wuv_ready;
};

static constexpr int ATTN_BARRIER_BYTES = 2 * (int)sizeof(uint32_t);

// Align-up a running byte offset to 16 (defensive — keeps every section 16B
// aligned even though scratch is scalar-accessed today).
__device__ __host__ __forceinline__ size_t attn_au16(size_t x) {
  return (x + 15u) & ~((size_t)15u);
}

// Total scratch bytes (barrier + all arrays, each section 16-byte padded). MUST
// match the builder's ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES.
static constexpr int ATTN_SCRATCH_BYTES =
    ATTN_BARRIER_BYTES +
    /*g_hdeq*/ K_HIDDEN * 4 + /*g_hf8*/ ((K_HIDDEN + 15) & ~15) +
    /*g_hsc*/ (K_HIDDEN / K_GRP) * 4 + /*g_qbf8*/ ((K_QLORA + 15) & ~15) +
    /*g_qbsc*/ ((K_QLORA / K_GRP + 3) & ~3) * 4 + /*g_qkva*/ K_QKVAN * 4 +
    /*g_qbdeq*/ K_QLORA * 4 + /*g_qpe*/ K_HLOCAL * K_QKHEAD * 4 +
    /*g_attn*/ K_HLOCAL * K_KVLORA * 4 +
    /*g_attn_deq*/ K_HLOCAL * K_KVLORA * 4 +
    /*g_red*/ K_OIN * 4 + /*g_odeq*/ K_OIN * 4 +
    /*g_mla_acc*/ K_HLOCAL * MLA_SPLITS * K_KVLORA * 4 +
    /*g_mla_m*/ K_HLOCAL * MLA_SPLITS * 4 +
    /*g_mla_l*/ K_HLOCAL * MLA_SPLITS * 4 +
    /*g_head_done (lever 4)*/ K_HLOCAL * 4 +
    /*g_head_wuv_ready (lever 5)*/ K_HLOCAL * 4 +
    /*16B pad slack between 17 sections*/ 17 * 16 +
    /* +8 so the TOTAL is a multiple of 16 → the scratch tensor's element count
       (bytes/2 bf16) is a multiple of 8, required by tensor_init's 16B-vec
       zero-init static_assert. Keep the total a multiple of 16. */
    8;

// The builder allocates ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES (currently 434864).
// Keep this constant a multiple of 16 (the tensor_init 16B-vec zero-init
// static_assert needs it) and in sync with that builder value.
static_assert(ATTN_SCRATCH_BYTES % 16 == 0,
              "ATTN_SCRATCH_BYTES must be a multiple of 16");

__device__ __forceinline__ AttnScratch attn_make_scratch(uint8_t *base) {
  size_t off = ATTN_BARRIER_BYTES;
  AttnScratch sc;
  off = attn_au16(off);
  sc.g_hdeq = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HIDDEN * 4;
  off = attn_au16(off);
  sc.g_hf8 = reinterpret_cast<__nv_fp8_e4m3 *>(base + off);
  off += (size_t)K_HIDDEN;
  off = attn_au16(off);
  sc.g_hsc = reinterpret_cast<float *>(base + off);
  off += (size_t)(K_HIDDEN / K_GRP) * 4;
  off = attn_au16(off);
  sc.g_qbf8 = reinterpret_cast<__nv_fp8_e4m3 *>(base + off);
  off += (size_t)K_QLORA;
  off = attn_au16(off);
  sc.g_qbsc = reinterpret_cast<float *>(base + off);
  off += (size_t)(K_QLORA / K_GRP) * 4;
  off = attn_au16(off);
  sc.g_qkva = reinterpret_cast<float *>(base + off);
  off += (size_t)K_QKVAN * 4;
  off = attn_au16(off);
  sc.g_qbdeq = reinterpret_cast<float *>(base + off);
  off += (size_t)K_QLORA * 4;
  off = attn_au16(off);
  sc.g_qpe = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HLOCAL * K_QKHEAD * 4;
  off = attn_au16(off);
  sc.g_attn = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HLOCAL * K_KVLORA * 4;
  off = attn_au16(off);
  sc.g_attn_deq = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HLOCAL * K_KVLORA * 4;
  off = attn_au16(off);
  sc.g_red = reinterpret_cast<float *>(base + off);
  off += (size_t)K_OIN * 4;
  off = attn_au16(off);
  sc.g_odeq = reinterpret_cast<float *>(base + off);
  off += (size_t)K_OIN * 4;
  off = attn_au16(off);
  sc.g_mla_acc = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HLOCAL * MLA_SPLITS * K_KVLORA * 4;
  off = attn_au16(off);
  sc.g_mla_m = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HLOCAL * MLA_SPLITS * 4;
  off = attn_au16(off);
  sc.g_mla_l = reinterpret_cast<float *>(base + off);
  off += (size_t)K_HLOCAL * MLA_SPLITS * 4;
  off = attn_au16(off);
  sc.g_head_done = reinterpret_cast<int *>(base + off);
  off += (size_t)K_HLOCAL * 4;
  off = attn_au16(off);
  sc.g_head_wuv_ready = reinterpret_cast<int *>(base + off);
  off += (size_t)K_HLOCAL * 4;
  return sc;
}

// ---- math helpers (VERBATIM from the standalone kernel) ---------------------
__device__ __forceinline__ float k_bf16(float f) {
  return __bfloat162float(__float2bfloat16(f));
}
__device__ __forceinline__ __half2 k_fp8x2_to_h2(const __nv_fp8x2_storage_t s) {
  return __half2(__nv_cvt_fp8x2_to_halfraw2(s, __NV_E4M3));
}
__device__ __forceinline__ uint8_t k_enc_ue8m0(float s) {
  uint32_t b = __float_as_uint(s);
  int e = (int)((b >> 23) & 0xff) - 127;
  int m = (int)(b & 0x7fffff);
  int v = (m == 0 ? e : e + 1) + 127;
  if (v < 0) {
    v = 0;
  }
  if (v > 255) {
    v = 255;
  }
  return (uint8_t)v;
}
__device__ __forceinline__ float k_dec_ue8m0(uint8_t e) {
  return __uint_as_float((uint32_t)e << 23);
}

// ---- cp.async helpers (VERBATIM) --------------------------------------------
__device__ __forceinline__ void k_cpa16(uint32_t smem_addr, void const *gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gptr));
}
__device__ __forceinline__ void k_cpa_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void k_cpa_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ uint4 k_lds_u4(uint32_t saddr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(saddr));
  return v;
}
__device__ __forceinline__ uint32_t k_lds_u32(uint32_t saddr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(saddr));
  return v;
}

// MAC one uint4 (16 fp8 weights) against pre-converted activation half2[8].
__device__ __forceinline__ float k_mac_u4(__half2 const *__restrict__ ah,
                                          uint4 raw) {
  __nv_fp8x2_storage_t const *wp =
      reinterpret_cast<__nv_fp8x2_storage_t const *>(&raw);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    __half2 p = __hmul2(ah[j], k_fp8x2_to_h2(wp[j]));
    s += __low2float(p) + __high2float(p);
  }
  return s;
}

// ===========================================================================
//  L5.1 cp.async DEEP-PREFETCH GEMV (VERBATIM). out[n]=sum_k
//  adeq[k]*W8[n,k]*wsc. Each global warp owns RBT consecutive output rows. The
//  cold FP8 weight uint4 tiles stream via cp.async.cg into a per-warp
//  STAGES-deep smem ring (`wbuf`). Wsc is the MPK per-128-ROW-BLOCK, per-128-
//  K-group fp32 weight_scale_inv [N/128, K/128] (row-major), read as a PLAIN
//  fp32 — identical to the production fp8_gemm_dense_finen GEMV
//  (`sb[(col>>7)*nk + g]`, no UE8M0 decode; the value already IS the decoded
//  power-of-2). RBT rows may span two 128-blocks, so each row's scale-row is
//  derived from its OWN row index (n>>7), NOT a single pre-offset block-row.
// ===========================================================================
template <int RBT, int STAGES>
__device__ __forceinline__ void
    gemv_grid_cpa_t(float const *__restrict__ adeq,
                    __nv_fp8_e4m3 const *__restrict__ W,
                    float const *__restrict__ Wsc,
                    float *__restrict__ out,
                    int N,
                    int K,
                    int gwarp,
                    int gwarps,
                    int lane,
                    uint4 *__restrict__ wbuf) {
  int KGg = K / K_GRP; // K-group count = scale-row stride (nk)
  int nU = K / 16;     // uint4 per row
  int SS = nU >> 5;    // super-steps (32 lanes * uint4); K multiple of 512
  int nblk = (N + RBT - 1) / RBT;
  const uint32_t sbase = __cvta_generic_to_shared(wbuf);
  const uint32_t STRIDE = (uint32_t)(RBT * 32 * 16); // bytes per stage buffer
  for (int blk = gwarp; blk < nblk; blk += gwarps) {
    int n0 = blk * RBT;
    int rb = (n0 + RBT <= N) ? RBT : (N - n0);
    uint4 const *Wn[RBT];
    float const *sn[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      int n = (n0 + t < N) ? n0 + t : N - 1;
      Wn[t] = reinterpret_cast<uint4 const *>(W + (size_t)n * K);
      sn[t] = Wsc + (size_t)(n >> 7) * KGg; // row n's 128-block scale-row
    }
    float acc[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < SS) {
        uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][s * 32 + lane]);
        }
      }
      k_cpa_commit();
    }
    for (int ss = 0; ss < SS; ss++) {
      int sp = ss + (STAGES - 1);
      if (sp < SS) {
        uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][sp * 32 + lane]);
        }
      }
      k_cpa_commit();
      k_cpa_wait<STAGES - 1>();
      __syncwarp();
      int u = ss * 32 + lane;
      int g = (u * 16) >> 7;
      float const *a = adeq + u * 16;
      __half2 ah[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ah[j] = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
      }
      uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        uint4 raw = k_lds_u4(cur + (uint32_t)((t * 32 + lane) * 16));
        // MPK per-128-block fp32 weight scale (already the decoded power-of-2),
        // read as a plain fp32 — same as the production finen GEMV. NO ue8m0.
        float wsc = __ldg(&sn[t][g]);
        acc[t] += k_mac_u4(ah, raw) * wsc;
      }
    }
    k_cpa_wait<0>();
    __syncwarp(); // drain pending groups before ring reuse/return
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      float s = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
      }
      if (lane == 0 && t < rb) {
        out[n0 + t] = k_bf16(s);
      }
    }
  }
}

// q_b cp.async GEMV with FUSED YaRN rope on the pe-part (saves one barrier).
// (VERBATIM). pe offset [512:576) within each head (stride 576). RBT8-aligned
// -> each pe pair lies fully inside one warp's block.
template <int RBT, int STAGES>
__device__ __forceinline__ void
    gemv_grid_cpa_qb_rope_t(float const *__restrict__ adeq,
                            __nv_fp8_e4m3 const *__restrict__ W,
                            float const *__restrict__ Wsc,
                            float *__restrict__ out,
                            int N,
                            int K,
                            __nv_bfloat16 const *__restrict__ cos_sin,
                            int pos,
                            int gwarp,
                            int gwarps,
                            int lane,
                            uint4 *__restrict__ wbuf) {
  int KGg = K / K_GRP; // K-group count = scale-row stride (nk)
  int nU = K / 16;
  int SS = nU >> 5;
  int nblk = (N + RBT - 1) / RBT;
  const uint32_t sbase = __cvta_generic_to_shared(wbuf);
  const uint32_t STRIDE = (uint32_t)(RBT * 32 * 16);
  for (int blk = gwarp; blk < nblk; blk += gwarps) {
    int n0 = blk * RBT;
    int rb = (n0 + RBT <= N) ? RBT : (N - n0);
    uint4 const *Wn[RBT];
    float const *sn[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      int n = (n0 + t < N) ? n0 + t : N - 1;
      Wn[t] = reinterpret_cast<uint4 const *>(W + (size_t)n * K);
      sn[t] = Wsc + (size_t)(n >> 7) * KGg; // row n's 128-block scale-row
    }
    float acc[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < SS) {
        uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][s * 32 + lane]);
        }
      }
      k_cpa_commit();
    }
    for (int ss = 0; ss < SS; ss++) {
      int sp = ss + (STAGES - 1);
      if (sp < SS) {
        uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][sp * 32 + lane]);
        }
      }
      k_cpa_commit();
      k_cpa_wait<STAGES - 1>();
      __syncwarp();
      int u = ss * 32 + lane;
      int g = (u * 16) >> 7;
      float const *a = adeq + u * 16;
      __half2 ah[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ah[j] = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
      }
      uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        uint4 raw = k_lds_u4(cur + (uint32_t)((t * 32 + lane) * 16));
        // MPK per-128-block fp32 weight scale (already the decoded power-of-2),
        // read as a plain fp32 — same as the production finen GEMV. NO ue8m0.
        float wsc = __ldg(&sn[t][g]);
        acc[t] += k_mac_u4(ah, raw) * wsc;
      }
    }
    k_cpa_wait<0>();
    __syncwarp();
    float sv[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      float s = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
      }
      sv[t] = s;
    }
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        sv[t] = k_bf16(sv[t]);
      }
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        int n = n0 + t;
        if (n >= N) {
          continue;
        }
        int off = n % K_QKHEAD;
        if (off >= 512) {
          int peo = off - 512; // 0..63
          if ((peo & 1) == 0 && t + 1 < RBT) {
            float c = __bfloat162float(cos_sin[pos * K_COSSIN_STRIDE + peo]);
            float s = __bfloat162float(
                cos_sin[pos * K_COSSIN_STRIDE + K_COSSIN_SINOFF + peo]);
            float q0 = sv[t], q1 = sv[t + 1];
            sv[t] = k_bf16(q0 * c - q1 * s);
            sv[t + 1] = k_bf16(q1 * c + q0 * s);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        if (t < rb) {
          out[n0 + t] = sv[t]; // already k_bf16'd (roped or raw)
        }
      }
    }
  }
}

// o_proj cp.async GEMV (fused residual add). (VERBATIM). bf16 global output.
template <int RBT, int STAGES>
__device__ __forceinline__ void
    gemv_grid_cpa_oproj_t(float const *__restrict__ adeq,
                          __nv_fp8_e4m3 const *__restrict__ W,
                          float const *__restrict__ Wsc,
                          __nv_bfloat16 const *__restrict__ resid,
                          __nv_bfloat16 *__restrict__ out,
                          int N,
                          int K,
                          int gwarp,
                          int gwarps,
                          int lane,
                          uint4 *__restrict__ wbuf) {
  int KGg = K / K_GRP; // K-group count = scale-row stride (nk)
  int nU = K / 16;
  int SS = nU >> 5;
  int nblk = (N + RBT - 1) / RBT;
  const uint32_t sbase = __cvta_generic_to_shared(wbuf);
  const uint32_t STRIDE = (uint32_t)(RBT * 32 * 16);
  for (int blk = gwarp; blk < nblk; blk += gwarps) {
    int n0 = blk * RBT;
    int rb = (n0 + RBT <= N) ? RBT : (N - n0);
    uint4 const *Wn[RBT];
    float const *sn[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      int n = (n0 + t < N) ? n0 + t : N - 1;
      Wn[t] = reinterpret_cast<uint4 const *>(W + (size_t)n * K);
      sn[t] = Wsc + (size_t)(n >> 7) * KGg; // row n's 128-block scale-row
    }
    float acc[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < SS) {
        uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][s * 32 + lane]);
        }
      }
      k_cpa_commit();
    }
    for (int ss = 0; ss < SS; ss++) {
      int sp = ss + (STAGES - 1);
      if (sp < SS) {
        uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][sp * 32 + lane]);
        }
      }
      k_cpa_commit();
      k_cpa_wait<STAGES - 1>();
      __syncwarp();
      int u = ss * 32 + lane;
      int g = (u * 16) >> 7;
      float const *a = adeq + u * 16;
      __half2 ah[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ah[j] = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
      }
      uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        uint4 raw = k_lds_u4(cur + (uint32_t)((t * 32 + lane) * 16));
        // MPK per-128-block fp32 weight scale (already the decoded power-of-2),
        // read as a plain fp32 — same as the production finen GEMV. NO ue8m0.
        float wsc = __ldg(&sn[t][g]);
        acc[t] += k_mac_u4(ah, raw) * wsc;
      }
    }
    k_cpa_wait<0>();
    __syncwarp();
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      float s = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
      }
      if (lane == 0 && t < rb) {
        out[n0 + t] =
            __float2bfloat16(k_bf16(s + __bfloat162float(resid[n0 + t])));
      }
    }
  }
}

// quantize bf16 hidden -> dequantized fp8 (UE8M0), WARP-cooperative per
// 128-group. (VERBATIM, but writes the fp8 path into scratch g_hf8 instead of a
// global.)
__device__ __forceinline__ void
    quant_hidden_grid(__nv_bfloat16 const *__restrict__ hidden,
                      float *__restrict__ deq,
                      __nv_fp8_e4m3 *__restrict__ g_hf8,
                      float *__restrict__ g_hsc,
                      int n,
                      int gwarp,
                      int gwarps,
                      int lane) {
  int ng = n / K_GRP;
  for (int gx = gwarp; gx < ng; gx += gwarps) {
    __nv_bfloat16 const *h = hidden + gx * K_GRP;
    float v[4];
    float mx = 1e-10f;
    __nv_bfloat162 const *h2 =
        reinterpret_cast<__nv_bfloat162 const *>(h) + lane * 2;
    float2 a0 = __bfloat1622float2(h2[0]);
    float2 a1 = __bfloat1622float2(h2[1]);
    v[0] = a0.x;
    v[1] = a0.y;
    v[2] = a1.x;
    v[3] = a1.y;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mx = fmaxf(mx, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
    float *d = deq + gx * K_GRP + lane * 4;
    __nv_fp8_e4m3 *d8 = g_hf8 + gx * K_GRP + lane * 4;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
      __nv_fp8_e4m3 qf = __nv_fp8_e4m3(q);
      d8[t] = qf;
      d[t] = (float)qf * yq;
    }
    if (lane == 0) {
      g_hsc[gx] = yq;
    }
  }
}

// per-128-group UE8M0 quantize -> dequantized values, WARP-cooperative.
// (VERBATIM)
__device__ __forceinline__ void quant_ue8m0_grid(
    float const *src, float *deq, int n, int gwarp, int gwarps, int lane) {
  int ng = n / K_GRP;
  for (int gx = gwarp; gx < ng; gx += gwarps) {
    float const *s = src + gx * K_GRP + lane * 4;
    float v[4];
    float mx = 1e-10f;
    const float4 a = *reinterpret_cast<float4 const *>(s);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mx = fmaxf(mx, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
    float *d = deq + gx * K_GRP + lane * 4;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
      d[t] = (float)__nv_fp8_e4m3(q) * yq;
    }
  }
}

#if MPK_DSV3_ATTN_FAST
// ---- FAST-lever block-local quant + smem-sourced GEMV helpers (ported from
// ferret workspace6 v131; the quant math is BYTE-IDENTICAL to the grid versions
// above — same UE8M0 amax/448 per-128-group; the GEMVs read activation from
// SHARED instead of global; the WEIGHT scale (Wsc) read is adapted to the MPK
// per-128-ROW-BLOCK fp32 weight_scale_inv format used everywhere in this file,
// `__ldg(&sn[t][g])`, NOT ferret's per-row UE8M0 uint32 decode). ---------------

// Lever 1: block-cooperative UE8M0 quant of bf16 hidden[0:n] -> block-local
// SHARED s_deq (dequantized fp32, byte-identical to quant_hidden_grid's `deq`
// output). Every block quantizes the SAME hidden into its own s_deq (redundant
// but no block idles) so the quant_hidden->qkv_a grid barrier is removed: the
// trailing __syncthreads publishes s_deq within the block before qkv_a reads it.
// Does NOT write g_hf8/g_hsc (the qkv_a GEMV reads the fp32 deq path only).
__device__ __forceinline__ void
    quant_hidden_block_smem(__nv_bfloat16 const *__restrict__ hidden,
                            float *__restrict__ s_deq,
                            int n,
                            int warpl,
                            int lane) {
  int ng = n / K_GRP;
  for (int gx = warpl; gx < ng; gx += NWARP) {
    __nv_bfloat16 const *h = hidden + gx * K_GRP;
    float v[4];
    float mx = 1e-10f;
    __nv_bfloat162 const *h2 =
        reinterpret_cast<__nv_bfloat162 const *>(h) + lane * 2;
    float2 a0 = __bfloat1622float2(h2[0]);
    float2 a1 = __bfloat1622float2(h2[1]);
    v[0] = a0.x;
    v[1] = a0.y;
    v[2] = a1.x;
    v[3] = a1.y;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mx = fmaxf(mx, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
    float *d = s_deq + gx * K_GRP + lane * 4;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
      d[t] = (float)__nv_fp8_e4m3(q) * yq;
    }
  }
  __syncthreads();
}

#if MPK_DSV3_ATTN_PHASE0
// Phase-0 DEEP-FUSION helper (default-OFF; compiled only under
// -DMPK_DSV3_ATTN_PHASE0). FOLDS the LIVE front input_layernorm RMSNorm
// (the separate `rmsnorm_layer(self.x -> hidden_bf16)` task) INTO this kernel:
// the attn-mega consumes self.x RAW (the residual-stream input, `self_x`) and
// RMS-norms it block-locally as a Phase-0 before the UE8M0 quant.
//
// It is `quant_hidden_block_smem` with ONE addition: a block-collective
// RMSNorm pre-pass that REPRODUCES the exact reduction tree of the task it
// replaces. The separate front task that this fold removes is
// `rmsnorm_layer(self.x -> hidden_bf16)`, which on B200 (target_cc>=90)
// dispatches to rms_norm_hopper_impl (rmsnorm_hopper.cuh). To keep the fold as
// bit-faithful as feasible (sub-ULP fp32-reduction drift can flip a knife-edge
// bf16 rounding -> token divergence over 61 layers, the same reason R1 mandates
// rsqrtf over 1.0f/sqrtf), the reduction here mirrors rmsnorm_hopper EXACTLY:
//   - fp32 promote of every element;
//   - intra-warp partial via __shfl_XOR_sync (matches rmsnorm_hopper:150-153);
//   - cross-warp combine of the NWARP partials via a warp-0 __shfl_XOR_sync
//     TREE (matches rmsnorm_hopper:158-165), NOT a sequential Σ red8[i];
//   - rms_rcp = rsqrtf(ss/n + K_EPS), K_EPS=1e-6f (matches eps + rsqrt).
// The per-element normed value normed[j]=bf16(float(self_x[j])*rms_rcp*
// float(input_ln_w[j])) is rounded to bf16 ONCE, exactly as rms_norm_hopper_impl
// writes hidden_bf16 (output=(T)val) and the old quant then read float(bf16).
// The UE8M0 amax/448 per-128-group quant body is byte-identical to
// quant_hidden_block_smem above — ONLY the quantized value changes from raw
// input[0] to the freshly-normed value. Every block computes the SAME ss and
// the SAME s_deq redundantly (no block idles), so NO grid barrier is added (the
// trailing __syncthreads publishes s_deq within the block before qkv_a reads
// it, exactly as quant_hidden_block_smem does). `red8` is the caller's per-warp
// reduction scratch (the SAME [NWARP] buffer rms_rcp_block uses).
//
// REDUCTION-GROUPING is bit-identical by construction (NOT a drift source):
// at the production constants (HIDDEN=7168, NTHREAD=256, bf16 -> CHUNK_SIZE=4,
// TILE_SIZE=1024, NUM_TILES=7), rms_norm_hopper_impl's tiled accumulation visits
// element `tid + 256*(4*for_idx + m)` (for_idx in [0,7), m in [0,4)), and
// `4*for_idx+m` enumerates 0..27 monotonically == this helper's flat
// `for(i=tid;i<7168;i+=NTHREAD)` set AND ORDER. So per-thread partials, the
// tree, the product association, eps, rsqrt, and the single bf16 round all match
// rms_norm_hopper_impl exactly (verified from first principles + Codex +
// ablation-logic-reviewer, 2026-06-25).
//
// NOTE: this is still a MATH-CHANGING lever — the fused kernel and the two-task
// chain are NOT provably byte/token identical. The ONLY possible residual is
// FMA-contraction differences from fusing the rmsnorm + quant into ONE
// compilation unit (the compiler may contract the v*v accumulate / x*(rms_rcp*w)
// differently than the standalone rmsnorm_hopper translation unit) — a MAYBE,
// not a guaranteed last-ULP bound. It MUST therefore be gated by a NUMERIC
// (cosine) correctness check on the box, NOT a token-identity A/B (token streams
// are run-to-run nondeterministic on this TP8/EP2 decode path — the FFN
// cross-CTA FP atomicAdd — so token-identity is an unreliable safety gate here).
__device__ __forceinline__ void rmsnorm_quant_hidden_block_smem(
    __nv_bfloat16 const *__restrict__ self_x,
    __nv_bfloat16 const *__restrict__ input_ln_w,
    float *__restrict__ s_deq,
    float *__restrict__ red8,
    int n,
    int warpl,
    int lane) {
  int tid = threadIdx.x;
  // --- Phase-0 RMSNorm reduction: ss = Σ(float(self_x[i]))² over [0:n], fp32,
  // block-collective. Reduction tree matches rms_norm_hopper_impl exactly. ---
  float ps = 0.f;
  for (int i = tid; i < n; i += NTHREAD) {
    float v = __bfloat162float(self_x[i]);
    ps += v * v;
  }
  // intra-warp xor-shuffle (rmsnorm_hopper:150-153).
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ps += __shfl_xor_sync(0xffffffffu, ps, o);
  }
  if ((tid & 31) == 0) {
    red8[tid >> 5] = ps;
  }
  __syncthreads();
  // cross-warp combine of the NWARP partials inside warp 0 via an xor-shuffle
  // TREE, then broadcast through red8[0] (rmsnorm_hopper:158-165). Computing the
  // combine in warp 0 only (not redundantly per-thread) preserves the canonical
  // associativity order.
  float ss = (tid < NWARP) ? red8[tid] : 0.f;
#pragma unroll
  for (int o = NWARP / 2; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if (tid == 0) {
    red8[0] = ss;
  }
  __syncthreads();
  ss = red8[0]; // uniform across the block
  __syncthreads(); // re-converge before red8 is reused by a later phase
  // rsqrtf (NOT 1.0f/sqrtf) + K_EPS=1e-6f + fp32 promote — matches the canonical
  // rms_norm_hopper_impl exactly.
  float rms_rcp = rsqrtf(ss / (float)n + K_EPS);

  // --- UE8M0 per-128-group quant of the NORMED value (body byte-identical to
  // quant_hidden_block_smem; only the source value is normalized). ---
  int ng = n / K_GRP;
  for (int gx = warpl; gx < ng; gx += NWARP) {
    __nv_bfloat16 const *h = self_x + gx * K_GRP;
    __nv_bfloat16 const *w = input_ln_w + gx * K_GRP;
    float v[4];
    float mx = 1e-10f;
    __nv_bfloat162 const *h2 =
        reinterpret_cast<__nv_bfloat162 const *>(h) + lane * 2;
    __nv_bfloat162 const *w2 =
        reinterpret_cast<__nv_bfloat162 const *>(w) + lane * 2;
    float2 a0 = __bfloat1622float2(h2[0]);
    float2 a1 = __bfloat1622float2(h2[1]);
    float2 g0 = __bfloat1622float2(w2[0]);
    float2 g1 = __bfloat1622float2(w2[1]);
    // normed = bf16(x * (rms_rcp * w)) — fp32 math, single bf16 round. The
    // product ASSOCIATION matches rms_norm_hopper_impl's `val *= rms_rcp * w`
    // (i.e. x * (rms_rcp * w), NOT (x*rms_rcp)*w) to minimize fp32 drift vs the
    // task this fold replaces. The quant below then consumes float(this bf16),
    // identical to how quant_hidden_block_smem consumed the bf16 hidden.
    v[0] = __bfloat162float(__float2bfloat16(a0.x * (rms_rcp * g0.x)));
    v[1] = __bfloat162float(__float2bfloat16(a0.y * (rms_rcp * g0.y)));
    v[2] = __bfloat162float(__float2bfloat16(a1.x * (rms_rcp * g1.x)));
    v[3] = __bfloat162float(__float2bfloat16(a1.y * (rms_rcp * g1.y)));
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mx = fmaxf(mx, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
    float *d = s_deq + gx * K_GRP + lane * 4;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
      d[t] = (float)__nv_fp8_e4m3(q) * yq;
    }
  }
  __syncthreads();
}
#endif // MPK_DSV3_ATTN_PHASE0

// Lever 3: block-cooperative UE8M0 quant of src[0:n] -> block-local SHARED
// s_deq (dequantized values, byte-identical to quant_ue8m0_grid). Every block
// quantizes the SAME src so the quant_gred->o_proj grid barrier is removed (the
// trailing __syncthreads publishes s_deq within the block before o_proj reads).
__device__ __forceinline__ void
    quant_ue8m0_block_smem(float const *__restrict__ src,
                           float *__restrict__ s_deq,
                           int n,
                           int warpl,
                           int lane) {
  int ng = n / K_GRP;
  for (int gx = warpl; gx < ng; gx += NWARP) {
    float const *s = src + gx * K_GRP + lane * 4;
    float4 a = *reinterpret_cast<float4 const *>(s);
    float v[4] = {a.x, a.y, a.z, a.w};
    float mx = 1e-10f;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mx = fmaxf(mx, fabsf(v[t]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
    float *d = s_deq + gx * K_GRP + lane * 4;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      float q = fminf(fmaxf(v[t] / yq, -K_FP8MAX), K_FP8MAX);
      d[t] = (float)__nv_fp8_e4m3(q) * yq;
    }
  }
  __syncthreads();
}

// Lever 2 companion: q_b cp.async GEMV + fused YaRN rope that sources its
// ACTIVATION from block-local SHARED s_adeq (already-dequantized q_a_normed).
// Byte-identical to gemv_grid_cpa_qb_rope_t EXCEPT the activation source
// (shared, via ld.shared.v4.f32) — the weight stream, MAC, rope, and Wsc read
// (MPK per-128-block fp32 `__ldg(&sn[t][g])`) are identical. Lets the caller
// drop the q_a_layernorm->q_b grid barrier.
template <int RBT, int STAGES>
__device__ __forceinline__ void gemv_grid_cpa_qb_rope_smem_t(
    float const *__restrict__ s_adeq,
    __nv_fp8_e4m3 const *__restrict__ W,
    float const *__restrict__ Wsc,
    float *__restrict__ out,
    int N,
    int K,
    __nv_bfloat16 const *__restrict__ cos_sin,
    int pos,
    int gwarp,
    int gwarps,
    int lane,
    uint4 *__restrict__ wbuf) {
  int KGg = K / K_GRP;
  int nU = K / 16;
  int SS = nU >> 5;
  int nblk = (N + RBT - 1) / RBT;
  const uint32_t sbase = __cvta_generic_to_shared(wbuf);
  const uint32_t abase = __cvta_generic_to_shared(s_adeq);
  const uint32_t STRIDE = (uint32_t)(RBT * 32 * 16);
  for (int blk = gwarp; blk < nblk; blk += gwarps) {
    int n0 = blk * RBT;
    int rb = (n0 + RBT <= N) ? RBT : (N - n0);
    uint4 const *Wn[RBT];
    float const *sn[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      int n = (n0 + t < N) ? n0 + t : N - 1;
      Wn[t] = reinterpret_cast<uint4 const *>(W + (size_t)n * K);
      sn[t] = Wsc + (size_t)(n >> 7) * KGg; // row n's 128-block scale-row
    }
    float acc[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < SS) {
        uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][s * 32 + lane]);
        }
      }
      k_cpa_commit();
    }
    for (int ss = 0; ss < SS; ss++) {
      int sp = ss + (STAGES - 1);
      if (sp < SS) {
        uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][sp * 32 + lane]);
        }
      }
      k_cpa_commit();
      k_cpa_wait<STAGES - 1>();
      __syncwarp();
      int u = ss * 32 + lane;
      int g = (u * 16) >> 7;
      uint32_t aoff = abase + (uint32_t)(u * 16) * 4u;
      float4 av0, av1, av2, av3;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av0.x), "=f"(av0.y), "=f"(av0.z), "=f"(av0.w)
                   : "r"(aoff));
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av1.x), "=f"(av1.y), "=f"(av1.z), "=f"(av1.w)
                   : "r"(aoff + 16u));
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av2.x), "=f"(av2.y), "=f"(av2.z), "=f"(av2.w)
                   : "r"(aoff + 32u));
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av3.x), "=f"(av3.y), "=f"(av3.z), "=f"(av3.w)
                   : "r"(aoff + 48u));
      __half2 ah[8];
      ah[0] = __floats2half2_rn(av0.x, av0.y);
      ah[1] = __floats2half2_rn(av0.z, av0.w);
      ah[2] = __floats2half2_rn(av1.x, av1.y);
      ah[3] = __floats2half2_rn(av1.z, av1.w);
      ah[4] = __floats2half2_rn(av2.x, av2.y);
      ah[5] = __floats2half2_rn(av2.z, av2.w);
      ah[6] = __floats2half2_rn(av3.x, av3.y);
      ah[7] = __floats2half2_rn(av3.z, av3.w);
      uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        uint4 raw = k_lds_u4(cur + (uint32_t)((t * 32 + lane) * 16));
        float wsc = __ldg(&sn[t][g]);
        acc[t] += k_mac_u4(ah, raw) * wsc;
      }
    }
    k_cpa_wait<0>();
    __syncwarp();
    float sv[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      float s = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
      }
      sv[t] = s;
    }
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        sv[t] = k_bf16(sv[t]);
      }
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        int n = n0 + t;
        if (n >= N) {
          continue;
        }
        int off = n % K_QKHEAD;
        if (off >= 512) {
          int peo = off - 512;
          if ((peo & 1) == 0 && t + 1 < RBT) {
            float c = __bfloat162float(cos_sin[pos * K_COSSIN_STRIDE + peo]);
            float s = __bfloat162float(
                cos_sin[pos * K_COSSIN_STRIDE + K_COSSIN_SINOFF + peo]);
            float q0 = sv[t], q1 = sv[t + 1];
            sv[t] = k_bf16(q0 * c - q1 * s);
            sv[t + 1] = k_bf16(q1 * c + q0 * s);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        if (t < rb) {
          out[n0 + t] = sv[t];
        }
      }
    }
  }
}

// Lever 3 companion: o_proj cp.async GEMV (fused residual add) that sources its
// ACTIVATION from block-local SHARED s_adeq (already-dequantized g_red).
// Byte-identical to gemv_grid_cpa_oproj_t EXCEPT the activation source. Lets the
// caller drop the quant_gred->o_proj grid barrier.
template <int RBT, int STAGES>
__device__ __forceinline__ void gemv_grid_cpa_oproj_smem_t(
    float const *__restrict__ s_adeq,
    __nv_fp8_e4m3 const *__restrict__ W,
    float const *__restrict__ Wsc,
    __nv_bfloat16 const *__restrict__ resid,
    __nv_bfloat16 *__restrict__ out,
    int N,
    int K,
    int gwarp,
    int gwarps,
    int lane,
    uint4 *__restrict__ wbuf) {
  int KGg = K / K_GRP;
  int nU = K / 16;
  int SS = nU >> 5;
  int nblk = (N + RBT - 1) / RBT;
  const uint32_t sbase = __cvta_generic_to_shared(wbuf);
  const uint32_t abase = __cvta_generic_to_shared(s_adeq);
  const uint32_t STRIDE = (uint32_t)(RBT * 32 * 16);
  for (int blk = gwarp; blk < nblk; blk += gwarps) {
    int n0 = blk * RBT;
    int rb = (n0 + RBT <= N) ? RBT : (N - n0);
    uint4 const *Wn[RBT];
    float const *sn[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      int n = (n0 + t < N) ? n0 + t : N - 1;
      Wn[t] = reinterpret_cast<uint4 const *>(W + (size_t)n * K);
      sn[t] = Wsc + (size_t)(n >> 7) * KGg; // row n's 128-block scale-row
    }
    float acc[RBT];
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      acc[t] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < SS) {
        uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][s * 32 + lane]);
        }
      }
      k_cpa_commit();
    }
    for (int ss = 0; ss < SS; ss++) {
      int sp = ss + (STAGES - 1);
      if (sp < SS) {
        uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
        for (int t = 0; t < RBT; t++) {
          k_cpa16(b + (uint32_t)((t * 32 + lane) * 16), &Wn[t][sp * 32 + lane]);
        }
      }
      k_cpa_commit();
      k_cpa_wait<STAGES - 1>();
      __syncwarp();
      int u = ss * 32 + lane;
      int g = (u * 16) >> 7;
      uint32_t aoff = abase + (uint32_t)(u * 16) * 4u;
      float4 av0, av1, av2, av3;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av0.x), "=f"(av0.y), "=f"(av0.z), "=f"(av0.w)
                   : "r"(aoff));
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av1.x), "=f"(av1.y), "=f"(av1.z), "=f"(av1.w)
                   : "r"(aoff + 16u));
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av2.x), "=f"(av2.y), "=f"(av2.z), "=f"(av2.w)
                   : "r"(aoff + 32u));
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                   : "=f"(av3.x), "=f"(av3.y), "=f"(av3.z), "=f"(av3.w)
                   : "r"(aoff + 48u));
      __half2 ah[8];
      ah[0] = __floats2half2_rn(av0.x, av0.y);
      ah[1] = __floats2half2_rn(av0.z, av0.w);
      ah[2] = __floats2half2_rn(av1.x, av1.y);
      ah[3] = __floats2half2_rn(av1.z, av1.w);
      ah[4] = __floats2half2_rn(av2.x, av2.y);
      ah[5] = __floats2half2_rn(av2.z, av2.w);
      ah[6] = __floats2half2_rn(av3.x, av3.y);
      ah[7] = __floats2half2_rn(av3.z, av3.w);
      uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
      for (int t = 0; t < RBT; t++) {
        uint4 raw = k_lds_u4(cur + (uint32_t)((t * 32 + lane) * 16));
        float wsc = __ldg(&sn[t][g]);
        acc[t] += k_mac_u4(ah, raw) * wsc;
      }
    }
    k_cpa_wait<0>();
    __syncwarp();
#pragma unroll
    for (int t = 0; t < RBT; t++) {
      float s = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
      }
      if (lane == 0 && t < rb) {
        out[n0 + t] =
            __float2bfloat16(k_bf16(s + __bfloat162float(resid[n0 + t])));
      }
    }
  }
}
#endif // MPK_DSV3_ATTN_FAST

// ===========================================================================
//  FLASH MLA partial: block (h,split) computes the un-normalized softmax over
//  its KV sub-range [r0,r1). All 256 threads cooperate. (VERBATIM; reads
//  kv_cache + scratch g_qpe, writes scratch g_mla_*.)
// ===========================================================================
__device__ __noinline__ void
    mla_partial(__nv_bfloat16 const *__restrict__ kv_cache,
                float const *__restrict__ g_qpe,
                float *__restrict__ g_mla_acc,
                float *__restrict__ g_mla_m,
                float *__restrict__ g_mla_l,
                float *__restrict__ s_score,
                float *__restrict__ red8,
                int h,
                int sp,
                int r0,
                int r1,
                float sm,
                int dbg_step) {
  (void)dbg_step;
  int tid = threadIdx.x;
  float const *q = &g_qpe[h * K_QKHEAD];
  int nr = r1 - r0;
  int TPR = NTHREAD / (nr > 0 ? nr : 1);
  if (TPR < 1) {
    TPR = 1;
  }
  if (TPR > 8) {
    TPR = 8;
  }
  if (TPR >= 8) {
    TPR = 8;
  } else if (TPR >= 4) {
    TPR = 4;
  } else if (TPR >= 2) {
    TPR = 2;
  } else {
    TPR = 1;
  }
  {
    int sub = tid % TPR;
    int row = tid / TPR;
    int rows_per_step = NTHREAD / TPR;
    int laneInWarp = tid & 31;
    unsigned grpmask =
        ((TPR >= 32) ? 0xffffffffu
                     : (((1u << TPR) - 1u) << ((laneInWarp / TPR) * TPR)));
    for (int rr = row; rr < nr; rr += rows_per_step) {
      int r = r0 + rr;
      uint4 const *kvr =
          reinterpret_cast<uint4 const *>(&kv_cache[(size_t)r * K_QKHEAD]);
      float dot = 0.f;
      for (int c = sub; c < K_QKHEAD / 8; c += TPR) {
        uint4 kw = kvr[c];
        __nv_bfloat162 const *k2 =
            reinterpret_cast<__nv_bfloat162 const *>(&kw);
        float const *qc = &q[c * 8];
#pragma unroll
        for (int p = 0; p < 4; p++) {
          float2 kf = __bfloat1622float2(k2[p]);
          dot += qc[2 * p] * kf.x + qc[2 * p + 1] * kf.y;
        }
      }
#pragma unroll
      for (int o = TPR >> 1; o > 0; o >>= 1) {
        dot += __shfl_down_sync(grpmask, dot, o, TPR);
      }
      if (sub == 0) {
        s_score[rr] = dot * sm;
      }
    }
  }
  __syncthreads();
  float lmax = -1e30f;
  for (int rr = tid; rr < nr; rr += NTHREAD) {
    lmax = fmaxf(lmax, s_score[rr]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
  }
  if ((tid & 31) == 0) {
    red8[tid >> 5] = lmax;
  }
  __syncthreads();
  float gmax = -1e30f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    gmax = fmaxf(gmax, red8[i]);
  }
  __syncthreads();
#ifdef MPK_ATTN_DBG
  // DECISIVE: dump the PRE-softmax scores for the first 3 KV positions (head 0,
  // split 0). If one position's score dominates by a large margin, attention
  // collapses there (degenerate). Captured from s_score BEFORE the in-place
  // exp. dbg_step is the current decode `step` passed by the task entry.
  if (dbg_step == 2 && h == 0 && sp == 0 && tid == 0) {
    float sc0 = (nr > 0) ? s_score[0] : 0.f;
    float sc1 = (nr > 1) ? s_score[1] : 0.f;
    float sc2 = (nr > 2) ? s_score[2] : 0.f;
    printf("[ATTN_DBG] MLA h0 sp0 nr=%d gmax=%.5f  pre-softmax score[0,1,2]= "
           "%.5f %.5f %.5f\n",
           nr,
           gmax,
           sc0,
           sc1,
           sc2);
  }
#endif
  for (int rr = tid; rr < nr; rr += NTHREAD) {
    s_score[rr] = __expf(s_score[rr] - gmax);
  }
  __syncthreads();
  float lsum = 0.f;
  for (int rr = tid; rr < nr; rr += NTHREAD) {
    lsum += s_score[rr];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lsum += __shfl_xor_sync(0xffffffffu, lsum, o);
  }
  if ((tid & 31) == 0) {
    red8[tid >> 5] = lsum;
  }
  __syncthreads();
  float gsum = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    gsum += red8[i];
  }
#ifdef MPK_ATTN_DBG
  // normalized softmax WEIGHT for KV positions 0,1,2 (head 0, split 0). A
  // weight ~1.0 on a single position = collapse (degenerate); ~1/KV each =
  // uniform.
  if (dbg_step == 2 && h == 0 && sp == 0 && tid == 0) {
    float inv = (gsum > 0.f) ? 1.0f / gsum : 0.f;
    float w0 = (nr > 0) ? s_score[0] * inv : 0.f;
    float w1 = (nr > 1) ? s_score[1] * inv : 0.f;
    float w2 = (nr > 2) ? s_score[2] * inv : 0.f;
    printf("[ATTN_DBG] MLA h0 sp0 gsum=%.5f  softmax_weight[0,1,2]= %.5f %.5f "
           "%.5f  (collapse if one ~1.0)\n",
           gsum,
           w0,
           w1,
           w2);
    // Discriminate softmax-collapse vs V-accum-reads-wrong-row (Codex): print
    // the V (c_latent) of rows 0,1,2 at a NON-tiny dim (d=256) + the weighted
    // reconstruction. If V[d] differs across rows AND weights are healthy but
    // attn_out[d] still ~= V0[d], the V-accumulation indexes the wrong row.
    int dd = 256;
    float v0 =
        (nr > 0) ? __bfloat162float(kv_cache[(size_t)(r0 + 0) * K_QKHEAD + dd])
                 : 0.f;
    float v1 =
        (nr > 1) ? __bfloat162float(kv_cache[(size_t)(r0 + 1) * K_QKHEAD + dd])
                 : 0.f;
    float v2 =
        (nr > 2) ? __bfloat162float(kv_cache[(size_t)(r0 + 2) * K_QKHEAD + dd])
                 : 0.f;
    printf("[ATTN_DBG] MLA h0 V[d=256] rows[0,1,2]= %.5f %.5f %.5f  recon "
           "w.V=%.5f\n",
           v0,
           v1,
           v2,
           w0 * v0 + w1 * v1 + w2 * v2);
  }
#endif
  int base = h * MLA_SPLITS + sp;
  if (tid == 0) {
    g_mla_m[base] = (nr > 0) ? gmax : -1e30f;
    g_mla_l[base] = gsum;
  }
  float *accv = &g_mla_acc[(size_t)base * K_KVLORA];
  for (int d = tid; d < K_KVLORA; d += NTHREAD) {
    float acc = 0.f;
    for (int rr = 0; rr < nr; rr++) {
      acc += s_score[rr] *
             __bfloat162float(kv_cache[(size_t)(r0 + rr) * K_QKHEAD + d]);
    }
    accv[d] = acc;
  }
  __syncthreads();
}

// S10+S11 FUSED: FLASH MLA merge that ALSO quantizes its own head's 512
// attn_out INLINE. (VERBATIM; reads scratch g_mla_*, writes g_attn +
// g_attn_deq.) When MPK_DSV3_ATTN_FAST (lever 5, WUV_HEAD_SPINWAIT), it ALSO
// publishes the per-head "g_attn_deq[h] is ready" flag (device release) so the
// W_UV stage can spin-wait per head instead of a merge->W_UV grid barrier;
// g_head_wuv_ready is the [16] flag array (nullptr when the lever is off).
__device__ __noinline__ void
    mla_merge_quant(float *__restrict__ g_attn,
                    float *__restrict__ g_attn_deq,
                    float const *__restrict__ g_mla_acc,
                    float const *__restrict__ g_mla_m,
                    float const *__restrict__ g_mla_l,
                    float *__restrict__ s_attn,
                    float *__restrict__ red8,
                    int h,
                    int nsp,
                    int *__restrict__ g_head_wuv_ready) {
  (void)g_head_wuv_ready;
  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  float const *mrow = &g_mla_m[h * MLA_SPLITS];
  float const *lrow = &g_mla_l[h * MLA_SPLITS];
  float gmax = -1e30f;
#pragma unroll
  for (int s = 0; s < MLA_SPLITS; s++) {
    if (s < nsp) {
      gmax = fmaxf(gmax, mrow[s]);
    }
  }
  float denom = 0.f;
#pragma unroll
  for (int s = 0; s < MLA_SPLITS; s++) {
    if (s < nsp) {
      denom += lrow[s] * __expf(mrow[s] - gmax);
    }
  }
  float inv = (denom > 0.f) ? 1.0f / denom : 0.f;
  for (int d = tid; d < K_KVLORA; d += NTHREAD) {
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < MLA_SPLITS; s++) {
      if (s < nsp) {
        float w = __expf(mrow[s] - gmax);
        acc += g_mla_acc[((size_t)(h * MLA_SPLITS + s)) * K_KVLORA + d] * w;
      }
    }
    float v = k_bf16(acc * inv);
    s_attn[d] = v;
    g_attn[h * K_KVLORA + d] = v;
  }
  __syncthreads();
  int KGv = K_KVLORA / K_GRP; // 4
  if (warpl < KGv) {
    float const *ar = &s_attn[warpl * K_GRP];
    float *dq = &g_attn_deq[h * K_KVLORA + warpl * K_GRP];
    float mx = 1e-10f;
    for (int j = lane; j < K_GRP; j += 32) {
      float a = fabsf(ar[j]);
      mx = fmaxf(mx, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ot = __shfl_xor_sync(0xffffffffu, mx, o);
      mx = fmaxf(mx, ot);
    }
    float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
    for (int j = lane; j < K_GRP; j += 32) {
      float vq = fminf(fmaxf(ar[j] / ys, -K_FP8MAX), K_FP8MAX);
      dq[j] = (float)__nv_fp8_e4m3(vq) * ys;
    }
  }
#if MPK_DSV3_ATTN_FAST
  // Lever 5 (WUV_HEAD_SPINWAIT): publish "head h's g_attn_deq is ready" to the
  // W_UV stage (replaces the merge->W_UV grid barrier).
  // CRITICAL (ferret v131 line 1147; reviewer+Codex-vetted): the dequant loop
  // above is MULTI-WARP — warps 0..3 (threads 0..127) each wrote a 128-wide
  // slice of g_attn_deq[h]. tid0's __threadfence orders only tid0's OWN prior
  // accesses, so WITHOUT this __syncthreads tid0 could flip the flag before
  // warps 1..3 even issue their g_attn_deq stores -> a consumer CTA that
  // acquires the flag reads lanes 0..31 fresh but 32..127 STALE from a previous
  // decode step (scratch persists). This block-wide __syncthreads makes all 4
  // V-group warps' g_attn_deq[h] writes complete (and, via the barrier, visible
  // to tid0) BEFORE tid0 publishes. Then tid0 does a DEVICE-scope release (PTX
  // st.release.gpu + belt-and-suspenders __threadfence, which transitively
  // orders the OTHER warps' now-syncthreads-published stores) so all of head h's
  // deq is visible to other CTAs BEFORE the flag flips to 1. DEVICE scope is
  // correct (all 16 heads live on this rank's GPU; .sys is NOT needed). The
  // "memory" clobber stops the compiler sinking a g_attn_deq store past the flag.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence(); // device release
    asm volatile("st.global.release.gpu.u32 [%0], %1;\n" ::"l"(
                     &g_head_wuv_ready[h]),
                 "r"(1)
                 : "memory");
  }
#endif
}

// ===========================================================================
//  W_UV per-head BMM (S12). (VERBATIM.) out[h,n]=sum_k a_deq[h,k]*W[h,n,k]*wsc.
//  kvbv_s is PER-HEAD per-128-group fp32 [H, KGv=4] — MATCHES MPK's
//  kv_b_v_bmm_dense.weight_scale_inv (16,1,4) f32.
// ===========================================================================
__device__ __noinline__ void
    wuv_bmm_grid(float const *__restrict__ g_attn_deq,
                 __nv_fp8_e4m3 const *__restrict__ kvbv_w,
                 float const *__restrict__ kvbv_s,
                 float *__restrict__ g_red,
                 int gwarp,
                 int gwarps,
                 int lane,
                 int *__restrict__ g_head_wuv_ready) {
  (void)g_head_wuv_ready;
  int nU = K_KVLORA / 16;     // 32 uint4 per W row
  int KGv = K_KVLORA / K_GRP; // 4
  int rows_per_head = K_VHEAD / RB;
  int nblk = K_HLOCAL * rows_per_head;
#if MPK_DSV3_ATTN_FAST
  // Lever 5: remember the head whose readiness we already acquired. The
  // grid-stride visits blocks of one head consecutively (blk/rows_per_head), so
  // each warp spins at most once per head.
  int last_h = -1;
#endif
  for (int blk = gwarp; blk < nblk; blk += gwarps) {
    int h = blk / rows_per_head;
    int nr0 = (blk % rows_per_head) * RB;
#if MPK_DSV3_ATTN_FAST
    // Lever 5 (WUV_HEAD_SPINWAIT): per-head merge->W_UV spin-wait (replaces the
    // merge->W_UV grid barrier). Lane 0 spins on g_head_wuv_ready[h] with a
    // DEVICE-scope ACQUIRE load (PTX ld.acquire.gpu — device-coherent, NOT
    // __ldg/.nc so it never reads a stale cached value across decode steps);
    // once it observes 1 the matching producer release makes g_attn_deq[h]
    // visible. The "memory" clobber stops the compiler hoisting g_attn_deq
    // reads above the wait; __threadfence is belt-and-suspenders; __syncwarp
    // hands the acquire ordering to lanes 1..31 (all 32 lanes share blk, so h
    // is warp-uniform and the branch is non-divergent). No deadlock: the
    // merge-blocks make unconditional progress so every flag is eventually set.
    if (h != last_h) {
      if (lane == 0) {
        int rdy = 0;
        while (rdy == 0) {
          asm volatile("ld.global.acquire.gpu.u32 %0, [%1];\n"
                       : "=r"(rdy)
                       : "l"(&g_head_wuv_ready[h])
                       : "memory");
          if (rdy == 0) {
            __nanosleep(64);
          }
        }
        __threadfence(); // device acquire (belt-and-suspenders)
      }
      __syncwarp(); // broadcast acquire ordering to all lanes
      last_h = h;
    }
#endif
    float const *ar = &g_attn_deq[h * K_KVLORA];
    float const *sc = kvbv_s + (size_t)h * KGv;
    uint4 const *Wh = reinterpret_cast<uint4 const *>(
        kvbv_w + (size_t)h * K_VHEAD * K_KVLORA);
    float acc[RB];
#pragma unroll
    for (int t = 0; t < RB; t++) {
      acc[t] = 0.f;
    }
    uint4 const *Wr[RB];
#pragma unroll
    for (int t = 0; t < RB; t++) {
      Wr[t] = Wh + (size_t)(nr0 + t) * nU;
    }
    for (int u = lane; u < nU; u += 32) {
      int k0 = u * 16;
      int g = k0 >> 7;
      float wsc = sc[g];
      float const *a = &ar[k0];
      __half2 ah[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ah[j] = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
      }
#pragma unroll
      for (int t = 0; t < RB; t++) {
        uint4 raw = Wr[t][u];
        acc[t] += k_mac_u4(ah, raw) * wsc;
      }
    }
#pragma unroll
    for (int t = 0; t < RB; t++) {
      float s = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, o);
      }
      if (lane == 0) {
        g_red[h * K_VHEAD + nr0 + t] = k_bf16(s);
      }
    }
  }
}

// fp32 block reduction of sum-of-squares over src[0:n] -> 1/sqrt(mean+eps),
// uniform across the block. (VERBATIM.) Redundant per-block (no block idles).
__device__ __forceinline__ float rms_rcp_block(float const *__restrict__ src,
                                               int n,
                                               float *__restrict__ red8) {
  int tid = threadIdx.x;
  float ps = 0.f;
  for (int i = tid; i < n; i += NTHREAD) {
    float x = src[i];
    ps += x * x;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ps += __shfl_down_sync(0xffffffffu, ps, o);
  }
  if ((tid & 31) == 0) {
    red8[tid >> 5] = ps;
  }
  __syncthreads();
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; i++) {
    ss += red8[i];
  }
  __syncthreads();
  return 1.0f / sqrtf(ss / n + K_EPS);
}

// ===========================================================================
//  MPK task entry. Mirrors ffn_full_megakernel_sm100_task_impl: TaskDesc +
//  merge_task_offset (the logical CTA id, == blockIdx.x set by the scheduler)
//  + runtime_config (for step[0]).
//
//  input_ptrs ABI — 14 slots, the HARD MAX_INPUTS_PER_TASK cap. To fit 14 (vs
//  the natural 16+) the two bf16 layernorm weights are CONCATENATED into one
//  `ln_weights` buffer ([q_a_ln(1536) | kv_a_ln(512)]) and cos/sin into one
//  `cos_sin` buffer ([cos(64) | sin(64)] per max_seq row); `out` is NOT an
//  input slot (the kernel writes output_ptrs[0] only). See the builder wrapper.
//  Weight scales (qkv_a_s/q_b_s/oproj_s) are the MPK per-128-ROW-BLOCK,
//  per-128-K-group fp32 weight_scale_inv [N/128, K/128] (row-major) that
//  _attach_fp8_weight produces — the SAME format the production
//  fp8_gemm_dense_finen GEMV reads (plain fp32, no UE8M0 decode). kvbv_s is the
//  per-head fp32 [H,1,4] kv_b_v_bmm_dense scale.
//    [0]  hidden     (bf16, the RMSNorm'd layer input)               (1,7168)
//    [1]  qkv_a_w    (fp8)                                        (2176,7168)
//    [2]  qkv_a_s    (fp32 per-128-block [N/128,K/128] = [17,56])
//    [3]  ln_weights (bf16 [q_a_ln(1536) | kv_a_ln(512)])            (2048,)
//         — under -DMPK_DSV3_ATTN_PHASE0 this is PREPENDED with input_ln so it
//           becomes [input_ln(7168) | q_a_ln(1536) | kv_a_ln(512)] = (9216,)
//           and the kernel RMS-norms the RAW self.x (input[0]) in Phase-0.
//    [4]  q_b_w      (fp8 absorbed)                               (9216,1536)
//    [5]  q_b_s      (fp32 per-128-block [72,12])
//    [6]  cos_sin    (bf16 [cos(64) | sin(64)] per row)        (max_seq,128)
//    [7]  kv_cache   (bf16 FLAT contiguous, rows x 576)  (read history + write
//    the new row [step]; same physical buffer as the output binding) [8]
//    kvbv_w     (fp8) (16,128,512) [9]  kvbv_s     (fp32 [H,1,4]) (16,1,4) [10]
//    oproj_w    (fp8) (7168,2048) [11] oproj_s    (fp32 per-128-block [56,16])
//    [12] residual   (bf16, the layer input self.x) (1,7168) [13] scratch
//    (uint8 AttnScratch base, barrier + activations)
//  + out bound as output_ptrs[0] (the tracked bf16 attn_proj_out write).
// ===========================================================================
__device__ __noinline__ void attn_block_megakernel_sm100_task_impl(
    mirage::runtime::TaskDesc const *task_desc,
    int merge_task_offset,
    mirage::runtime::RuntimeConfig const &runtime_config) {
  __nv_bfloat16 const *hidden =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  __nv_fp8_e4m3 const *qkv_a_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[1]);
  float const *qkv_a_s = static_cast<float const *>(task_desc->input_ptrs[2]);
  // ln_weights layout depends on the Phase-0 deep-fusion flag.
  __nv_bfloat16 const *ln_weights =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[3]);
#if MPK_DSV3_ATTN_PHASE0
  // Phase-0 DEEP-FUSION (default-OFF): the builder PREPENDS input_layernorm so
  // ln_weights = [input_ln(7168) | q_a_ln(1536) | kv_a_ln(512)] = 9216-d. The
  // Phase-0 RMSNorm reads input_ln_w at [0:7168); q_a/kv_a shift past it.
  //   input_ln_w = ln_weights        (offset 0)
  //   q_a_ln_w   = ln_weights + 7168 (offset K_HIDDEN)
  //   kv_a_ln_w  = ln_weights + 8704 (offset K_HIDDEN + K_QLORA)
  // R3 (codex's top non-sync footgun): assert the concatenation offsets at
  // compile time. The kernel only sees a raw pointer (no runtime length), so a
  // static_assert on the layout constants is the strongest available guard —
  // the builder MUST cat exactly [input_ln(K_HIDDEN) | q_a_ln(K_QLORA) |
  // kv_a_ln(K_KVLORA)] in this order for these offsets to be valid.
  static_assert(K_HIDDEN == 7168 && K_QLORA == 1536 && K_KVLORA == 512,
                "Phase-0 ln_weights layout [input_ln(7168)|q_a_ln(1536)|"
                "kv_a_ln(512)]=9216 — these constants pin the concat offsets; "
                "the builder ln_weights_pt MUST match this exact order/size.");
  static_assert((K_HIDDEN + K_QLORA + K_KVLORA) == 9216,
                "Phase-0 ln_weights total length must be 9216-d.");
  __nv_bfloat16 const *input_ln_w = ln_weights;                  // offset 0
  __nv_bfloat16 const *q_a_ln_w = ln_weights + K_HIDDEN;         // offset 7168
  __nv_bfloat16 const *kv_a_ln_w =
      ln_weights + K_HIDDEN + K_QLORA;                           // offset 8704
  (void)input_ln_w; // consumed by rmsnorm_quant_hidden_block_smem in S2
#else
  // Default (Phase-0 OFF): ln_weights = [q_a_ln(1536) | kv_a_ln(512)] (no
  // input_ln — the separate rmsnorm_layer task pre-normalizes hidden).
  static_assert((K_QLORA + K_KVLORA) == 2048,
                "Default ln_weights total length must be 2048-d "
                "([q_a_ln(1536)|kv_a_ln(512)]).");
  __nv_bfloat16 const *q_a_ln_w = ln_weights;            // offset 0
  __nv_bfloat16 const *kv_a_ln_w = ln_weights + K_QLORA; // offset 1536
#endif
  __nv_fp8_e4m3 const *q_b_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[4]);
  float const *q_b_s = static_cast<float const *>(task_desc->input_ptrs[5]);
  // cos_sin = [cos(64) | sin(64)] per max_seq row (concatenated). cosE/sinE are
  // strided views: cosE[r*64+i] -> cos_sin[r*128 + i]; sinE ->
  // cos_sin[r*128+64+i]. The kernel only ever reads cosE[pos*64 +
  // d]/sinE[pos*64 + d], so re-deriving the per-row base below keeps the
  // existing GEMV/rope call sites unchanged.
  __nv_bfloat16 const *cos_sin =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[6]);
  // kv_cache: the flat per-layer persistent KV buffer. input_ptrs[7] resolves
  // (via the generated loader, all_tensors.at(name)+offset, keyed by tensor
  // NAME) to the SAME physical address as the buffer's output binding — for a
  // root cuda_tensor there is no input-snapshot vs output-live distinction, so
  // a write through input_ptrs[7] DOES reach the real buffer and persists
  // across decode steps. The kernel reads history rows [0,step) and writes the
  // new row [step] through this single pointer. (Verified by reviewer+Codex:
  // the FFN mega-task's "input slot is a stale alias" note is about two
  // TensorDesc structs of ONE storage, NOT two buffers.)
  __nv_bfloat16 *kv_cache =
      static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[7]);
  __nv_fp8_e4m3 const *kvbv_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[8]);
  float const *kvbv_s = static_cast<float const *>(task_desc->input_ptrs[9]);
  __nv_fp8_e4m3 const *oproj_w =
      static_cast<__nv_fp8_e4m3 const *>(task_desc->input_ptrs[10]);
  float const *oproj_s = static_cast<float const *>(task_desc->input_ptrs[11]);
  __nv_bfloat16 const *residual =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[12]);
  // The kernel writes the OUTPUT slot only (no `out` input alias — that would
  // push the input count to 15 > MAX_INPUTS_PER_TASK=14, silently overflowing
  // into outputs[]). output_ptrs[0] is the buffer MPK tracks for the dep edge.
  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);
  uint8_t *scratch_base = static_cast<uint8_t *>(task_desc->input_ptrs[13]);

  int const step = runtime_config.step[0];

  AttnGridBarrier barrier;
  barrier.count = reinterpret_cast<unsigned int *>(scratch_base);
  barrier.gen =
      reinterpret_cast<unsigned int *>(scratch_base + sizeof(uint32_t));
  AttnScratch sc = attn_make_scratch(scratch_base);

  // Per-worker dynamic-smem: red8[NWARP] + s_score[512] + a per-warp cp.async
  // weight-prefetch ring. extern __shared__ MUST be __align__(1024) (megakernel
  // convention — a smaller alignment can lower the shared base and misalign
  // other tasks' 1024-aligned TMA/AR accesses in the shared test.cu).
  extern __shared__ __align__(1024) uint8_t s_smem[];
  // CPA_RING_U4 = uint4 per warp; worst-case GEMV is q_b/o_proj RBT8*STAGES4 =
  // 1024 uint4 (16KB/warp). s_wbuf at offset 0 = 1024-aligned (its uint4 reads
  // need 16-byte alignment; offset 0 of a 1024-aligned base satisfies it).
  constexpr int CPA_RING_U4 = 1024;
  uint4 *s_wbuf = reinterpret_cast<uint4 *>(s_smem); // [NWARP*CPA_RING_U4]
  // red8 / s_score placed AFTER the ring, each 16-aligned.
  size_t soff =
      (size_t)NWARP * CPA_RING_U4 * sizeof(uint4); // already 16-aligned
  float *red8 = reinterpret_cast<float *>(s_smem + soff);
  soff += attn_au16((size_t)NWARP * sizeof(float));
  float *s_score = reinterpret_cast<float *>(s_smem + soff);
  // (s_score holds 512 floats = the per-head scores; also reused as s_attn in
  // mla_merge_quant — both are <=512 floats, block-local.)
  soff += attn_au16((size_t)512 * sizeof(float));
#if MPK_DSV3_ATTN_FAST
  // Levers 1-3 block-local activation buffer (28KB f32). ALIASED across three
  // NON-OVERLAPPING phases (exactly as ferret v131): qkv_a reads s_act[0:7168]
  // (dequant hidden), q_b reads s_act[0:1536] (q_a_layernorm deq), o_proj reads
  // s_act[0:2048] (g_red deq). One buffer subsumes the prior per-phase buffers.
  // 16-aligned (the smem-sourced GEMVs read it via ld.shared.v4.f32). Total
  // dynamic smem now ~158KB << the B200 ~216KB pool (bps stays 1). The whole
  // extern array is __align__(1024) (megakernel convention), s_act lands at a
  // 16-aligned offset which the v4.f32 / float4 reads require.
  float *s_act = reinterpret_cast<float *>(s_smem + soff);
  float *s_qbdeq = s_act; // q_b phase reuses the front of s_act
  float *s_odeq = s_act;  // o_proj phase reuses the front of s_act
  soff += attn_au16((size_t)K_HIDDEN * sizeof(float));
  // Lever 4: "this block is head h's last split" flag (block-local, set by tid0
  // after the per-head atomicAdd, broadcast to the block via __syncthreads).
  __shared__ int s_mla_last;
#endif
  (void)soff;

  int tid = threadIdx.x, lane = tid & 31, warpl = tid >> 5;
  uint4 *my_wbuf = s_wbuf + (size_t)warpl * CPA_RING_U4;
  int worker_idx = merge_task_offset; // logical CTA id (== blockIdx.x)
  int gtid = worker_idx * NTHREAD + tid;
  int gthreads = ATTN_NUM_WORKERS * NTHREAD;
  int gwarp = gtid >> 5;
  int gwarps = gthreads >> 5;
  int KV = step + 1, pos = step;

  // EVERY-step header (H4 check: does `step` actually advance across decode
  // steps?). Prints once per invocation from worker0/thread0 regardless of
  // step, so the coordinator can confirm step = 0,1,2,... (a frozen/repeated
  // step -> self-attend -> repetition). Disambiguate per-layer by the out
  // pointer.
#ifdef MPK_ATTN_DBG
  // RANK = runtime_config.my_gpu_id (== nvshmem_my_pe()). DECISIVE for reading
  // the multi-line output: at TP=N (mpirun -np N) each of the N lines at a
  // fixed step is a SEPARATE PROCESS (TP rank), and the SAME layer's `out`
  // buffer has a DIFFERENT virtual address per process. So 4 distinct `out`
  // pointers at one step are the 4 TP RANKS, NOT 4 layers — and "RESID
  // identical across ranks" is EXPECTED (TP replicates the residual) while
  // "attn_proj differs per rank" is EXPECTED (each rank computes a different
  // head subset, pre-AllReduce). Print the rank so ranks (same step, ranks
  // 0..N-1) vs layers (same rank, different out) are unambiguous.
  if (worker_idx == 0 && tid == 0) {
    printf("[ATTN_DBG rank=%d out=%p] INVOKE step=%d KV=%d pos=%d\n",
           runtime_config.my_gpu_id,
           (void *)out,
           step,
           KV,
           pos);
  }
  // DECISIVE fork-resolver: print the actual TOKEN IDs + step[0]. The embed
  // kernel for this build uses input_source=1 -> reads input_tokens[0] (a FIXED
  // index 0, NOT [step]); the runtime is expected to write the current token
  // there each step. So print BOTH: tokens[step[0]] / tokens[step[0]+1] (the
  // serving ring) AND input_tokens[0] (what the embed ACTUALLY consumed).
  //   - input_tokens[0] (or tokens[step]) SAME for steps 0,1  => same token =>
  //     identical hidden is EXPECTED (red herring; the chain has it too).
  //   - DIFFERENT token id for steps 0,1 but HIDDEN identical => the mega reads
  //     a stale hidden (real input-binding/indexing bug).
  if (ATTN_DBG_STEP(step) && worker_idx == 0 && tid == 0) {
    long long s0 = runtime_config.step[0];
    long long tk_s =
        (runtime_config.tokens != nullptr) ? runtime_config.tokens[s0] : -999;
    long long tk_s1 = (runtime_config.tokens != nullptr)
                          ? runtime_config.tokens[s0 + 1]
                          : -999;
    long long itk0 = (runtime_config.input_tokens != nullptr)
                         ? runtime_config.input_tokens[0]
                         : -999;
    printf("[ATTN_DBG out=%p step=%d] TOKENID step0=%lld  tokens[step0]=%lld "
           "tokens[step0+1]=%lld  input_tokens[0]=%lld\n",
           (void *)out,
           step,
           s0,
           tk_s,
           tk_s1,
           itk0);
  }
  // DECISIVE input-binding check: the HIDDEN input (= RMSNorm(self.x) of the
  // CURRENT token). Print first-8 + sum|.| over 7168 at steps 0,1,2. If hidden
  // is near-IDENTICAL for steps 0,1 → the input/residual binding feeds a stale/
  // fixed token (the bug). If hidden is DISTINCT but the raw c_latent below is
  // near-identical for 0,1 → the qkv_a GEMV flattens distinct hiddens.
  if (ATTN_DBG_STEP(step) && worker_idx == 0 && tid == 0) {
    float hs = 0.f;
    for (int i = 0; i < K_HIDDEN; i++) {
      hs += fabsf(__bfloat162float(hidden[i]));
    }
    printf("[ATTN_DBG out=%p step=%d] HIDDEN sum|.|=%.6f  h[0..7]= %.5f %.5f "
           "%.5f %.5f %.5f %.5f %.5f %.5f\n",
           (void *)out,
           step,
           hs,
           __bfloat162float(hidden[0]),
           __bfloat162float(hidden[1]),
           __bfloat162float(hidden[2]),
           __bfloat162float(hidden[3]),
           __bfloat162float(hidden[4]),
           __bfloat162float(hidden[5]),
           __bfloat162float(hidden[6]),
           __bfloat162float(hidden[7]));
    // SHARPER (Codex): `residual` == self.x == the embed output == the mega's
    // OWN rmsnorm INPUT. Tap it to separate the two staleness sources:
    //   - residual DIFFERS for steps 0,1 but HIDDEN (=rmsnorm output) IDENTICAL
    //     => the mega's hidden_bf16 / rmsnorm task is STALE (not re-derived per
    //     decode step) — a mega-specific dependency bug (Codex #1).
    //   - residual IDENTICAL for steps 0,1 => self.x / the shared embed is
    //   stale
    //     (but the working chain reads the same self.x, so that'd be
    //     surprising).
    float rs = 0.f;
    for (int i = 0; i < K_HIDDEN; i++) {
      rs += fabsf(__bfloat162float(residual[i]));
    }
    printf("[ATTN_DBG out=%p step=%d] RESID(self.x) sum|.|=%.6f  r[0..7]= %.5f "
           "%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
           (void *)out,
           step,
           rs,
           __bfloat162float(residual[0]),
           __bfloat162float(residual[1]),
           __bfloat162float(residual[2]),
           __bfloat162float(residual[3]),
           __bfloat162float(residual[4]),
           __bfloat162float(residual[5]),
           __bfloat162float(residual[6]),
           __bfloat162float(residual[7]));
  }
#endif

  // ===================== S2: quantize hidden + qkv_a GEMM
  // =====================
#if MPK_DSV3_ATTN_FAST
  // Lever 1 (HIDDEN_BLOCK_LOCAL): every block quantizes hidden[7168]
  // block-cooperatively ONCE into block-local SHARED s_act (the trailing
  // __syncthreads inside replaces the quant_hidden->qkv_a grid barrier), then
  // qkv_a reads s_act. Byte-identical UE8M0 dequant to quant_hidden_grid's deq.
#if MPK_DSV3_ATTN_PHASE0
  // Phase-0 DEEP-FUSION (default-OFF): `hidden` (input[0]) is the RAW
  // residual-stream self.x; RMS-norm it block-locally with input_ln_w FIRST,
  // then quant into s_act. NO grid barrier added (block-local, same as below).
  rmsnorm_quant_hidden_block_smem(
      hidden /*=raw self.x*/, input_ln_w, s_act, red8, K_HIDDEN, warpl, lane);
#else
  quant_hidden_block_smem(hidden, s_act, K_HIDDEN, warpl, lane);
#endif
  gemv_grid_cpa_t<2, 6>(s_act, // qkv_a reads BLOCK-LOCAL s_act (no grid barrier)
                        qkv_a_w,
                        qkv_a_s,
                        sc.g_qkva,
                        K_QKVAN,
                        K_HIDDEN,
                        gwarp,
                        gwarps,
                        lane,
                        my_wbuf);
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS); // qkv_a -> layernorm (KEPT)
#else
#if MPK_DSV3_ATTN_PHASE0
#error                                                                          \
    "MPK_DSV3_ATTN_PHASE0 (Phase-0 RMSNorm fold) is only implemented on the FAST path (MPK_DSV3_ATTN_FAST=1, the default). With FAST=0 the builder feeds RAW self.x but this baseline path does not normalize it -> wrong. Do not combine MPK_DSV3_ATTN_PHASE0_FUSION=1 with MPK_DSV3_ATTN_FAST=0."
#endif
  quant_hidden_grid(
      hidden, sc.g_hdeq, sc.g_hf8, sc.g_hsc, K_HIDDEN, gwarp, gwarps, lane);
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
  gemv_grid_cpa_t<2, 6>(sc.g_hdeq,
                        qkv_a_w,
                        qkv_a_s,
                        sc.g_qkva,
                        K_QKVAN,
                        K_HIDDEN,
                        gwarp,
                        gwarps,
                        lane,
                        my_wbuf);
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
#endif
  // tap S2: qkv_a_out [2176] = [q_a(1536) | c_latent(512) | k_pe(64) | pad(64)]
  ATTN_DBG_TAP("qkv_a_out", out, sc.g_qkva, K_QKVAN, step, worker_idx);
  // DECISIVE: tap the RAW per-slice GEMV outputs BEFORE any norm — q_a slice
  // [0:1536], c_latent slice [1536:2048], k_pe slice [2048:2112]. If the raw
  // c_latent is token-IDENTICAL across steps (or near-zero) while raw q_a is
  // token-distinct, the bug is in the qkv_a GEMV / fused weight / per-block
  // scale for the c_latent rows (1536-2047, 128-blocks 12-15) — NOT the
  // kv_a_layernorm (which only EXPOSES a tiny/constant c_latent via
  // eps-dominated normalization). Same per-token g_hdeq feeds all three.
  ATTN_DBG_TAP("raw_q_a", out, sc.g_qkva, K_QLORA, step, worker_idx);
  ATTN_DBG_TAP(
      "raw_clatent", out, sc.g_qkva + K_QLORA, K_KVLORA, step, worker_idx);
  ATTN_DBG_TAP("raw_kpe",
               out,
               sc.g_qkva + K_QLORA + K_KVLORA,
               K_QKROPE,
               step,
               worker_idx);

  // ============ S3 q_a_layernorm + S5 kv_a_layernorm + rope_k + append =======
  {
    float q_rcp = rms_rcp_block(sc.g_qkva, K_QLORA, red8);
    float kv_rcp = rms_rcp_block(sc.g_qkva + K_QLORA, K_KVLORA, red8);
    // Confirm eps-domination directly (Codex): if a degenerate-c_latent token
    // has kv_rcp ~ 1/sqrt(eps) ~ 1000, the raw c_latent is eps-dominated (tiny)
    // -> the kv_a GEMV/scale is the bug, not the norm. Also print q_rcp + the
    // per-128-block raw c_latent mean-sq (blocks 12-15 = the 4 c_latent
    // groups).
#ifdef MPK_ATTN_DBG
    if (ATTN_DBG_STEP(step) && worker_idx == 0 && tid == 0) {
      float ms[4]; // mean-sq of each 128-wide c_latent group
#pragma unroll
      for (int b = 0; b < 4; b++) {
        float acc = 0.f;
        for (int i = 0; i < K_GRP; i++) {
          float x = sc.g_qkva[K_QLORA + b * K_GRP + i];
          acc += x * x;
        }
        ms[b] = acc / K_GRP;
      }
      printf("[ATTN_DBG out=%p step=%d] RCP q_rcp=%.4f kv_rcp=%.4f (eps-dom if "
             "~1000)  c_latent_blk_meansq[12..15]= %.3e %.3e %.3e %.3e\n",
             (void *)out,
             step,
             q_rcp,
             kv_rcp,
             ms[0],
             ms[1],
             ms[2],
             ms[3]);
    }
#endif
    int ngq = K_QLORA / K_GRP; // 12
#if MPK_DSV3_ATTN_FAST
    // Lever 2 (QA_BLOCK_LOCAL): every block computes ALL 12 groups (warp w owns
    // groups {w, w+NWARP}) of q_a_layernorm+UE8M0-requant into block-local
    // SHARED s_qbdeq (dequantized values, byte-identical to the grid-strided
    // g_qbdeq below). q_b then reads s_qbdeq from shared -> drops the
    // layernorm->q_b grid barrier (a block __syncthreads, issued before q_b
    // below, replaces it). The global g_qbdeq/g_qbf8/g_qbsc are written ONLY
    // under MPK_ATTN_DBG so the q_a_normed tap still works; the default-fast
    // build skips them (q_b sources s_qbdeq).
    for (int g = warpl; g < ngq; g += NWARP) {
      float const *src = sc.g_qkva + g * K_GRP;
      __nv_bfloat16 const *w = q_a_ln_w + g * K_GRP;
      float nv[4];
      float mx = 1e-10f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float v = k_bf16(src[j] * q_rcp * __bfloat162float(w[j]));
        nv[t] = v;
        mx = fmaxf(mx, fabsf(v));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ot = __shfl_xor_sync(0xffffffffu, mx, o);
        mx = fmaxf(mx, ot);
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
      float *d = s_qbdeq + g * K_GRP; // BLOCK-LOCAL shared (q_b reads this)
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float qv = fminf(fmaxf(nv[t] / yq, -K_FP8MAX), K_FP8MAX);
        d[j] = (float)__nv_fp8_e4m3(qv) * yq;
      }
#ifdef MPK_ATTN_DBG
      // Mirror to the global g_qbdeq (and g_qbf8/g_qbsc) for the q_a_normed tap
      // only — DEBUG-gated so the default-fast build stays clean. Block 0 only
      // (identical across blocks) to avoid 136x redundant global writes.
      if (worker_idx == 0) {
        float *dg = sc.g_qbdeq + g * K_GRP;
        __nv_fp8_e4m3 *d8 = sc.g_qbf8 + g * K_GRP;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          int j = lane + t * 32;
          float qv = fminf(fmaxf(nv[t] / yq, -K_FP8MAX), K_FP8MAX);
          __nv_fp8_e4m3 qf = __nv_fp8_e4m3(qv);
          d8[j] = qf;
          dg[j] = (float)qf * yq;
        }
        if (lane == 0) {
          sc.g_qbsc[g] = yq;
        }
      }
#endif
    }
#else
    for (int g = gwarp; g < ngq; g += gwarps) {
      float const *src = sc.g_qkva + g * K_GRP;
      __nv_bfloat16 const *w = q_a_ln_w + g * K_GRP;
      float nv[4];
      float mx = 1e-10f;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float v = k_bf16(src[j] * q_rcp * __bfloat162float(w[j]));
        nv[t] = v;
        mx = fmaxf(mx, fabsf(v));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ot = __shfl_xor_sync(0xffffffffu, mx, o);
        mx = fmaxf(mx, ot);
      }
      float ys = fmaxf(mx / K_FP8MAX, 1e-10f);
      float yq = k_dec_ue8m0(k_enc_ue8m0(ys));
      float *d = sc.g_qbdeq + g * K_GRP;
      __nv_fp8_e4m3 *d8 = sc.g_qbf8 + g * K_GRP;
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int j = lane + t * 32;
        float qv = fminf(fmaxf(nv[t] / yq, -K_FP8MAX), K_FP8MAX);
        __nv_fp8_e4m3 qf = __nv_fp8_e4m3(qv);
        d8[j] = qf;
        d[j] = (float)qf * yq;
      }
      if (lane == 0) {
        sc.g_qbsc[g] = yq;
      }
    }
#endif
    // kv_a_layernorm -> kv_cache row [0:512). grid-strided over 512 elements.
    for (int i = gtid; i < K_KVLORA; i += gthreads) {
      float v = k_bf16(sc.g_qkva[K_QLORA + i] * kv_rcp *
                       __bfloat162float(kv_a_ln_w[i]));
      kv_cache[(size_t)step * K_QKHEAD + i] = __float2bfloat16(v);
    }
    // rope(k_pe) on g_qkva[2048:2112) + append to kv_cache[step][512:576).
    if (worker_idx == 0 && tid == 0) {
      float kpe[K_QKROPE];
#pragma unroll
      for (int i = 0; i < K_QKROPE; i++) {
        kpe[i] = sc.g_qkva[2048 + i];
      }
#pragma unroll
      for (int pr = 0; pr < K_QKROPE / 2; pr++) {
        int d0 = pr * 2, d1 = d0 + 1;
        float c = __bfloat162float(cos_sin[pos * K_COSSIN_STRIDE + d0]);
        float s = __bfloat162float(
            cos_sin[pos * K_COSSIN_STRIDE + K_COSSIN_SINOFF + d0]);
        float k0 = kpe[d0], k1 = kpe[d1];
        kpe[d0] = k_bf16(k0 * c - k1 * s);
        kpe[d1] = k_bf16(k1 * c + k0 * s);
      }
      for (int i = 0; i < K_QKROPE; i++) {
        kv_cache[(size_t)step * K_QKHEAD + 512 + i] = __float2bfloat16(kpe[i]);
      }
    }
  }
#if MPK_DSV3_ATTN_FAST
  // Lever 2 (QA_BLOCK_LOCAL): the layernorm->q_b grid barrier is DROPPED — only
  // a block __syncthreads is needed before q_b reads block-local s_qbdeq (the
  // q_a_layernorm+requant above was block-cooperative into s_qbdeq). The
  // kv_a_layernorm/rope writes to kv_cache feed MLA, which is published
  // cross-block by the q_b->MLA grid barrier below. Under MPK_ATTN_DBG we KEEP
  // the grid barrier so the kv_cache HISTORY taps below see published cross-CTA
  // writes (debug build only — the default-fast/perf build uses __syncthreads).
#ifdef MPK_ATTN_DBG
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
#else
  __syncthreads();
#endif
#else
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
#endif
  // tap S3/S5: q_a_normed-dequant (q_b input) [1536]; the appended kv_cache row
  // [c_latent(512) | k_pe_rot(64)] is tapped from the live buffer for this
  // step.
  ATTN_DBG_TAP("q_a_normed", out, sc.g_qbdeq, K_QLORA, step, worker_idx);
#ifdef MPK_ATTN_DBG
  if (ATTN_DBG_STEP(step) && worker_idx == 0 && tid == 0) {
    // The JUST-written current row (read back through the kv_cache pointer).
    float kvrow[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      kvrow[i] = __bfloat162float(kv_cache[(size_t)step * K_QKHEAD + i]);
    }
    float s = 0.f;
    for (int i = 0; i < K_QKHEAD; i++) {
      s += fabsf(__bfloat162float(kv_cache[(size_t)step * K_QKHEAD + i]));
    }
    printf("[ATTN_DBG out=%p step=%d] %-10s n=%d sum|.|=%.6f  v[0..3]= %.5f "
           "%.5f %.5f %.5f\n",
           (void *)out,
           step,
           "kv_row[step]",
           K_QKHEAD,
           s,
           kvrow[0],
           kvrow[1],
           kvrow[2],
           kvrow[3]);
    // DECISIVE cross-step check: at step 2 (KV=3), read back the HISTORY rows 0
    // and 1 that steps 0/1 wrote. If these checksums/values DON'T match what
    // step 0/1 printed as their own kv_row[step], the KV write did not persist
    // across invocations (the stale-alias bug) -> degenerate attention. If they
    // match, history is intact and the bug is elsewhere.
    if (step == 2) {
      for (int r = 0; r < 2; r++) {
        float hv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          hv[i] = __bfloat162float(kv_cache[(size_t)r * K_QKHEAD + i]);
        }
        float hs = 0.f;
        for (int i = 0; i < K_QKHEAD; i++) {
          hs += fabsf(__bfloat162float(kv_cache[(size_t)r * K_QKHEAD + i]));
        }
        printf("[ATTN_DBG out=%p step=2] HISTORY row=%d sum|.|=%.6f  v[0..3]= "
               "%.5f %.5f %.5f %.5f\n",
               (void *)out,
               r,
               hs,
               hv[0],
               hv[1],
               hv[2],
               hv[3]);
      }
      // DECISIVE (Codex): pairwise VECTOR diff of the c_latent (V part [0:512])
      // between KV rows (0,1) and (1,2). sum|.| being similar (105.14 vs
      // 105.10) does NOT prove the vectors are near-identical — RMSNorm
      // naturally equalizes magnitudes. Compute sum_abs_diff / max_abs_diff /
      // cosine to settle whether tokens 0,1 actually produce the SAME c_latent
      // (diff ~0, cos ~1 => the KV genuinely can't distinguish them => the real
      // bug) or just similar-magnitude-but-distinct (diff large, cos < 1 => NOT
      // the bug; look elsewhere).
      float sad01 = 0.f, mad01 = 0.f, dot01 = 0.f, n0 = 0.f, n1 = 0.f;
      float sad12 = 0.f, mad12 = 0.f, dot12 = 0.f, n2 = 0.f;
      for (int i = 0; i < K_KVLORA; i++) {
        float a = __bfloat162float(kv_cache[(size_t)0 * K_QKHEAD + i]);
        float b = __bfloat162float(kv_cache[(size_t)1 * K_QKHEAD + i]);
        float c = __bfloat162float(kv_cache[(size_t)2 * K_QKHEAD + i]);
        float d01 = fabsf(a - b), d12 = fabsf(b - c);
        sad01 += d01;
        mad01 = fmaxf(mad01, d01);
        dot01 += a * b;
        sad12 += d12;
        mad12 = fmaxf(mad12, d12);
        dot12 += b * c;
        n0 += a * a;
        n1 += b * b;
        n2 += c * c;
      }
      float cos01 = (n0 > 0.f && n1 > 0.f) ? dot01 / sqrtf(n0 * n1) : 0.f;
      float cos12 = (n1 > 0.f && n2 > 0.f) ? dot12 / sqrtf(n1 * n2) : 0.f;
      printf("[ATTN_DBG out=%p step=2] CLATENT-DIFF rows(0,1): sad=%.6f "
             "max=%.6f cos=%.6f | rows(1,2): sad=%.6f max=%.6f cos=%.6f  "
             "(diff~0 & cos~1 => tokens produce SAME c_latent = the bug)\n",
             (void *)out,
             sad01,
             mad01,
             cos01,
             sad12,
             mad12,
             cos12);
    }
  }
#endif

  // ===================== S4+S6 FUSED: q_b GEMM + rope -> g_qpe
  // ================
#if MPK_DSV3_ATTN_FAST
  // Lever 2: q_b reads its activation from block-local SHARED s_qbdeq (no
  // layernorm->q_b grid barrier). Byte-identical GEMV+rope math.
  gemv_grid_cpa_qb_rope_smem_t<8, 4>(s_qbdeq,
                                     q_b_w,
                                     q_b_s,
                                     sc.g_qpe,
                                     K_HLOCAL * K_QKHEAD,
                                     K_QLORA,
                                     cos_sin,
                                     pos,
                                     gwarp,
                                     gwarps,
                                     lane,
                                     my_wbuf);
  // === HAZARD #1: ZERO-BEFORE-BARRIER ===================================
  // Levers 4 & 5: zero the per-head completion counters AND readiness flags
  // BEFORE the q_b->MLA grid barrier below. That barrier's __threadfence (inside
  // attn_grid_barrier) is what publishes these zero stores to EVERY CTA, so the
  // MLA partials that follow see a FRESHLY-ZEROED count for THIS decode step.
  // These arrays live in the per-task SCRATCH (persist across steps and are NOT
  // re-zeroed by anything else), so without this explicit pre-barrier zero a
  // stale count from the previous step would make the "last split"
  // (atomicAdd == nsp-1) fire early or never -> wrong merge / hang. The zero
  // MUST stay on the producer side of the barrier; do NOT move it past it.
  if (gtid < K_HLOCAL) {
    sc.g_head_done[gtid] = 0;
    sc.g_head_wuv_ready[gtid] = 0;
  }
  attn_grid_barrier(barrier,
                    ATTN_NUM_WORKERS); // q_b->MLA: publishes g_qpe, kv_cache,
                                       // AND the zeroed flags (the barrier's
                                       // __threadfence does the cross-CTA pub)
#else
  gemv_grid_cpa_qb_rope_t<8, 4>(sc.g_qbdeq,
                                q_b_w,
                                q_b_s,
                                sc.g_qpe,
                                K_HLOCAL * K_QKHEAD,
                                K_QLORA,
                                cos_sin,
                                pos,
                                gwarp,
                                gwarps,
                                lane,
                                my_wbuf);
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
#endif
  // tap S4/S6: q_nope_pe post-rope [16*576] (the MLA query). Print head-0's
  // nope-start (first 4) + a checksum over all 16 heads.
  ATTN_DBG_TAP(
      "q_nope_pe", out, sc.g_qpe, K_HLOCAL * K_QKHEAD, step, worker_idx);
  // head-0 q_PE part (g_qpe[512:516], post-rope) + the rope position used. The
  // q_pe drives the relative-position score term; if it's stale/un-roped
  // (==raw) or roped by the wrong pos, the per-position scores go garbage ->
  // collapse.
  ATTN_DBG_TAP("q_pe_h0", out, sc.g_qpe + K_KVLORA, K_QKROPE, step, worker_idx);
#ifdef MPK_ATTN_DBG
  if (ATTN_DBG_STEP(step) && worker_idx == 0 && tid == 0) {
    printf(
        "[ATTN_DBG out=%p step=%d] q_pe rope pos=%d  (q_pe is g_qpe[512:576] "
        "of head 0)\n",
        (void *)out,
        step,
        pos);
  }
#endif

  // ===================== S9/S10: FLASH MLA decode (KV-split)
  // ==================
  double mscale = 0.1 * log(40.0) + 1.0;
  float sm = (float)((1.0 / sqrt(192.0)) * mscale * mscale);
  int nsp = (KV + 63) / 64;
  if (nsp < 1) {
    nsp = 1;
  }
  if (nsp > MLA_SPLITS) {
    nsp = MLA_SPLITS;
  }
  int tile = (KV + nsp - 1) / nsp;
  int ntask = K_HLOCAL * nsp;
#if MPK_DSV3_ATTN_FAST
  // === HAZARD #2: ntask = 16*nsp <= 16*MLA_SPLITS = 128 < ATTN_NUM_WORKERS(136)
  // So every worker runs AT MOST ONE partial task (idx=blockIdx.x guarded by
  // idx<ntask), the per-head atomic counter is incremented exactly nsp times
  // (once per split-block of head h), and ALL of a head's partials run before
  // any W_UV spinner can need that head (all <=128 partial-blocks are among the
  // 136 workers; the producers make unconditional progress). Holds for nsp<=8
  // i.e. KV up to MLA_SPLITS*64 per the tile math.
  // Lever 4 (MLA_ATOMIC_MERGE): one partial per block; the LAST split-block of
  // head h (atomicAdd return == nsp-1) runs that head's merge IN-PLACE -> drops
  // the partial->merge grid barrier. Memory ordering (device-scope release/
  // acquire message-passing): producer writers store g_mla_acc -> mla_partial's
  // trailing __syncthreads (block barrier folds all 256 threads' stores into
  // tid0) -> tid0 __threadfence (device release) -> atomicAdd. Consumer (last
  // block): __threadfence (device acquire, after observing the final count) ->
  // __syncthreads (hand the acquired ordering to the other 255 threads) -> merge
  // reads g_mla_acc from the OTHER split-blocks. DEVICE scope (NOT .sys): all
  // heads/splits are on this rank's GPU.
  {
    int idx = worker_idx;
    if (idx < ntask) {
      int h = idx / nsp, sp = idx % nsp;
      int r0 = sp * tile, r1 = r0 + tile;
      if (r1 > KV) {
        r1 = KV;
      }
      mla_partial(kv_cache,
                  sc.g_qpe,
                  sc.g_mla_acc,
                  sc.g_mla_m,
                  sc.g_mla_l,
                  s_score,
                  red8,
                  h,
                  sp,
                  r0,
                  r1,
                  sm,
                  step); // ends with __syncthreads (publishes acc into tid0)
      if (threadIdx.x == 0) {
        __threadfence(); // device release (publish g_mla_acc)
        int old = atomicAdd(&sc.g_head_done[h], 1);
        s_mla_last = (old == nsp - 1) ? 1 : 0;
        if (s_mla_last) {
          __threadfence(); // device acquire (this block sees all splits' acc)
        }
      }
      __syncthreads(); // broadcast s_mla_last + hand the acquire to all threads
      if (s_mla_last) {
        // S10+S11 FUSED: this last-split block merges head h's nsp partials AND
        // quantizes its 512 attn elems in-block; mla_merge_quant ALSO sets
        // g_head_wuv_ready[h]=1 (device release) at its end (lever 5).
        mla_merge_quant(sc.g_attn,
                        sc.g_attn_deq,
                        sc.g_mla_acc,
                        sc.g_mla_m,
                        sc.g_mla_l,
                        s_score,
                        red8,
                        h,
                        nsp,
                        sc.g_head_wuv_ready);
      }
    }
  }
  // Lever 5 (WUV_HEAD_SPINWAIT): NO grid barrier here — wuv_bmm_grid spin-waits
  // per head on g_head_wuv_ready[h]. The merge-blocks make unconditional
  // progress so every flag is eventually set (no deadlock). The attn_out tap
  // (reads g_attn cross-block) is RELOCATED to after the W_UV->o_proj barrier.
#else
  for (int idx = worker_idx; idx < ntask; idx += ATTN_NUM_WORKERS) {
    int h = idx / nsp, sp = idx % nsp;
    int r0 = sp * tile, r1 = r0 + tile;
    if (r1 > KV) {
      r1 = KV;
    }
    mla_partial(kv_cache,
                sc.g_qpe,
                sc.g_mla_acc,
                sc.g_mla_m,
                sc.g_mla_l,
                s_score,
                red8,
                h,
                sp,
                r0,
                r1,
                sm,
                step);
  }
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
  // S10+S11 FUSED merge+quant (1 block/head; 16 of 136 blocks active).
  for (int h = worker_idx; h < K_HLOCAL; h += ATTN_NUM_WORKERS) {
    mla_merge_quant(sc.g_attn,
                    sc.g_attn_deq,
                    sc.g_mla_acc,
                    sc.g_mla_m,
                    sc.g_mla_l,
                    s_score,
                    red8,
                    h,
                    nsp,
                    /*g_head_wuv_ready=*/nullptr);
  }
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
  // tap S9/S10/S11: MLA attn_out [16*512] (head-0 first 4 + 16-head checksum).
  ATTN_DBG_TAP(
      "attn_out", out, sc.g_attn, K_HLOCAL * K_KVLORA, step, worker_idx);
#endif

  // ===================== S12 W_UV BMM -> g_red ===============================
  wuv_bmm_grid(sc.g_attn_deq,
               kvbv_w,
               kvbv_s,
               sc.g_red,
               gwarp,
               gwarps,
               lane,
#if MPK_DSV3_ATTN_FAST
               sc.g_head_wuv_ready
#else
               nullptr
#endif
  );
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS); // W_UV -> * (KEPT: publishes
                                                // g_red from all warps)
#if MPK_DSV3_ATTN_FAST
  // tap S9/S10/S11 RELOCATED here: the W_UV->* barrier above guarantees every
  // head's merge completed (W_UV consumed g_attn_deq), so g_attn is fully
  // visible cross-CTA now.
  ATTN_DBG_TAP(
      "attn_out", out, sc.g_attn, K_HLOCAL * K_KVLORA, step, worker_idx);
#endif
  // tap S12: W_UV BMM out attn_out_reduced [2048].
  ATTN_DBG_TAP("wuv_reduced", out, sc.g_red, K_OIN, step, worker_idx);

  // ===================== S13: quantize g_red + o_proj GEMM + residual ========
#if MPK_DSV3_ATTN_FAST
  // Lever 3 (OPROJ_BLOCK_QUANT): g_red (visible cross-CTA after the W_UV->*
  // barrier above) is quantized block-cooperatively ONCE into block-local
  // SHARED s_odeq (the trailing __syncthreads inside replaces the
  // quant_gred->o_proj grid barrier), then o_proj reads s_odeq. Byte-identical
  // UE8M0 quant to quant_ue8m0_grid. This merges the old two-barrier pair
  // (W_UV->quant + quant->o_proj) down to the single W_UV->* barrier above.
  quant_ue8m0_block_smem(sc.g_red, s_odeq, K_OIN, warpl, lane);
  gemv_grid_cpa_oproj_smem_t<8, 4>(s_odeq,
                                   oproj_w,
                                   oproj_s,
                                   residual,
                                   out,
                                   K_HIDDEN,
                                   K_OIN,
                                   gwarp,
                                   gwarps,
                                   lane,
                                   my_wbuf);
#else
  quant_ue8m0_grid(sc.g_red, sc.g_odeq, K_OIN, gwarp, gwarps, lane);
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
  gemv_grid_cpa_oproj_t<8, 4>(sc.g_odeq,
                              oproj_w,
                              oproj_s,
                              residual,
                              out,
                              K_HIDDEN,
                              K_OIN,
                              gwarp,
                              gwarps,
                              lane,
                              my_wbuf);
#endif
  // tap S13: final attn_proj_out [7168] (o_proj + residual, pre-AR) — the FULL
  // vector (the prior version was a worker-subset bug that summed only out[0]).
#ifdef MPK_ATTN_DBG
  // FULL-vector attn_proj (o_proj + residual = the attention's residual-stream
  // contribution — the FINAL ground-truth to diff against the chain's
  // attn_proj_out). The o_proj GEMV is grid-strided across all 136 workers, so
  // a grid_barrier is needed BEFORE worker0/thread0 can read the whole 7168 (a
  // bare __syncthreads only makes worker0's OWN rows visible — that's why the
  // old "w0 rows" sum was ~0.005, just |out[0]|). This barrier is DEBUG-ONLY
  // (the default build keeps the FFN-style post-task fence) so it cannot affect
  // perf or the byte-identical default.
  attn_grid_barrier(barrier, ATTN_NUM_WORKERS);
  if (ATTN_DBG_STEP(step) && worker_idx == 0 && tid == 0) {
    float s = 0.f;
    for (int i = 0; i < K_HIDDEN; i++) {
      s += fabsf(__bfloat162float(out[i]));
    }
    printf(
        "[ATTN_DBG out=%p step=%d] attn_proj  n=%d FULLsum|.|=%.6f  v[0..7]= "
        "%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
        (void *)out,
        step,
        K_HIDDEN,
        s,
        __bfloat162float(out[0]),
        __bfloat162float(out[1]),
        __bfloat162float(out[2]),
        __bfloat162float(out[3]),
        __bfloat162float(out[4]),
        __bfloat162float(out[5]),
        __bfloat162float(out[6]),
        __bfloat162float(out[7]));
  }
#endif
  // Publish the output stores globally before MPK signals task completion (the
  // post-task block-sync alone does NOT order other threads' global writes —
  // same class as the FFN mega-task's final __threadfence).
  __threadfence();
  __syncthreads();
}

} // namespace attn_block_megakernel_sm100
} // namespace kernel
