// moe_types.cuh -- types shared by the host code and the task bodies of the MoE layer: the tensor maps (Maps), the layer state
// (G: every buffer the tasks hand results through, the exchange region, the time stamps) and the expert-queue entry.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "config.cuh"
#include "runtime.cuh"

namespace static_mk {

// TMA descriptors, built by moe_host_init. bf16 [rows, K] maps have box {64, rows or 128, 2}; MXFP4 weight maps have one
// 128 x 128 piece per box (x2: two pieces); scale maps one 512-B chunk per box (x2: two).
struct Maps {
  CUtensorMap zq2;                     // z_q, two K tiles per box (W13's activation)
  CUtensorMap wg, wdown, wsgu, wsd;    // router, latent_down, shared gate_up, shared down weights (bf16)
  CUtensorMap x, hs;                   // the layer input x [T, H], h_s [T, SHR] (bf16)
  CUtensorMap zq;                      // z_q, one K tile per box
  CUtensorMap w13, w13sf, w2, w2sf;    // expert weights (MXFP4) and scales
  CUtensorMap w13x2, w13sfx2, w2x2, w2sfx2;
  CUtensorMap hq3, hsf3;               // one expert slot's h_q (3 K tiles) / its 3 scale chunks, one box
  CUtensorMap wup;                     // latent_up weight (bf16)
  CUtensorMap xsf28;                   // all 28 z_q scale chunks, one box
};

// one entry of the expert queue (run_expert_dynamic); stored as 5 ints in the shared-memory copies w2q / w2q7
enum QueueKind { QE_END = -1, QE_SHARED_DOWN = 0, QE_W13 = 1, QE_W2 = 2 };
struct QueueEntry {
  int kind;       // QueueKind
  int a, b;       // QE_W13: expert slot, 64-feature tile m (0..MT13-1)
                  // QE_W2: first item (slot * OT2 + output tile), number of items (<= W2_CHUNK, never crossing a slot)
                  // QE_SHARED_DOWN: 128-row tile of S, K half (0 or 1)
  int hq_buf;     // QE_W2: the h_q segment holding the slot's h_q (0..MAXSEG-1)
  int hq_use;     // QE_W2: how many times that segment was loaded before (its mbarrier parity)
};

struct G {
  // results handed from task to task (T = 8 tokens)
  float *lpart;                        // router partial sums [router K parts][T][NE], 0xFF-prefilled (route adds the parts)
  float *zpart;                        // latent_down partial sums [latent K parts][T][LAT], 0xFF-prefilled (quant adds them)
  float *spart;                        // shared gate_up sums [T][2 SHR], the K parts red.add into it (sact waits for C_SGU)
  __nv_bfloat16 *hs;                   // h_s [T][SHR] (shared down's input)
  float *Sout;                         // shared-expert output [T][H] (the graph's shared_down_sum)
  float *Racc;                         // this GPU's routed-expert sum [T][LAT] (the graph's routed_sum)
  uint8_t *zq, *xsf;                   // z_q [T][LAT] e4m3 and its scale chunks [KT_LAT][512], inside the exchange region
  uint8_t *hq, *hsf;                   // h_q [NSLOT][T][IR] e4m3 and scales [NSLOT][KT2][512] (W13 -> W2)
  uint32_t *cnt;                       // counters C_* (config.cuh)
  float const *gate_bias;              // score correction bias [NE]
  unsigned long long *pairs64;         // routing pairs [T][16]: expert << 32 | weight bits, 0xFF-prefilled (route -> expert queue)
  unsigned char const *w_up;           // this GPU's latent_up weight (the expert queue prefetches its rows into L2)
  // between the GPUs
  int rank, tp;                        // this GPU, number of GPUs
  unsigned gen;                        // launch number + 1; the start barrier waits until every GPU's hello slot holds it
  unsigned char *mc, *rv;              // multicast address of the exchange region (tp > 1), this GPU's copy (RG_* offsets)
  unsigned char *rs_all[TPMAX];        // every GPU's [R|S] receive buffer [TPMAX][RS_RANK] (peer-mapped)
  // tail
  __nv_bfloat16 *Rn, *Ssum, *y;        // normalised routed sum [T][LAT], summed shared output [T][H], layer output [T][H]
  __nv_bfloat16 const *prefix, *gamma; // residual [T][H], RMSNorm weight [LAT]
  float eps;
  float *ss_part;                      // sum-of-squares partials [T][4], 0xFF-prefilled
  float *opart;                        // latent_up partial sums [14][T][H / TPMAX], 0xFF-prefilled
  // time stamps (globaltimer ns) read by the host
  long long *stamps;                   // [NSM][NSTAMP] (config.cuh STAMP_*)
  long long *start_barrier;            // [1] written by every SM once the start barrier is passed: the last SM's time remains
};

__device__ __forceinline__ void push16(G const &g, size_t off, uint4 v) { push16_mc(g.mc, g.rv, g.tp, off, v); }

}  // namespace static_mk
