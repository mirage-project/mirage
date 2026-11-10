// core.cuh (HIP, gfx950) -- what every task type of the static megakernel
// shares on AMD GPUs, and the kernel's start and end; the same structure as
// include/mirage/static_megakernel/core.cuh (the generated layer defines the
// exchange layout, then includes this file and the task types' files). The
// differences:
//   - no PDL: HIP starts a kernel after the previous one ended, so kernel_begin
//     has no griddepcontrol
//   - no multicast: push16 stores into every GPU's exchange region (G::peer)
//   - no TMA / TMEM: Maps holds plain descriptors (MapDesc), KernelLocals no TMEM
//   - stamps: gtime() (s_memrealtime x NS_PER_TICK)
#pragma once
#include "config.cuh"
#include "runtime.cuh"
#include <cstdint>
#include <hip/hip_runtime.h>

#ifndef STATIC_EXCHANGE_SET
#error                                                                         \
    "core.cuh: the generated layer defines STATIC_EXCHANGE_SET, STATIC_EXCHANGE_HELLO, STATIC_EXCHANGE_OFFSETS, STATIC_EXCHANGE_BYTES"
#endif

namespace static_mk {

// Sums / maxima over the 64 lanes of a wave (every lane gets the result; the
// whole wave active): DPP moves within each row of 16 lanes (quad_perm xor 1,
// xor 2, row_half_mirror, row_mirror), then the 4 rows' values by readlane (VALU
// only; MI355X 10-13 kda16: the xor butterfly through ds_bpermute was an LDS
// round trip per step)
template <bool MAX>
__device__ __forceinline__ float wave_red64(float v) {
  auto op = [](float a, float b) { return MAX ? fmaxf(a, b) : a + b; };
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false)));  // quad_perm [1,0,3,2]
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false)));  // quad_perm [2,3,0,1]
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false))); // row_half_mirror
  v = op(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x140, 0xF, 0xF, false))); // row_mirror
  float const r0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0)),
              r1 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16)),
              r2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32)),
              r3 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
  return op(op(r0, r1), op(r2, r3));
}
__device__ __forceinline__ float wave_sum64(float v) { return wave_red64<false>(v); }
__device__ __forceinline__ float wave_max64(float v) { return wave_red64<true>(v); }

// ---- the exchange region's layout (from the layers) ----
constexpr size_t EXCHANGE_SET = STATIC_EXCHANGE_SET,
                 EXCHANGE_HELLO = STATIC_EXCHANGE_HELLO;
constexpr size_t exchange_offset[MAX_BUFS] = STATIC_EXCHANGE_OFFSETS;
constexpr size_t exchange_bytes[MAX_BUFS] = STATIC_EXCHANGE_BYTES;

// ---- the declared tensor views (the layers' "map.<slot>"): on gfx950 a plain
// descriptor, the address and the declared numbers (host.cuh build_maps) ----
enum MapKind {
  MAP_BF16 = 0,   // dims: rows, K, box rows
  MAP_WBLK = 1,   // dims: pieces, box pieces
  MAP_SF = 2,     // dims: chunks, box chunks
  MAP_ACT8 = 3,   // dims: K bytes, rows
  MAP_ACT8KT = 4, // dims: K bytes, rows, K tiles
};
struct MapDesc {
  void *p;
  int kind;
  unsigned long long dims[3];
};
struct Maps {
  MapDesc m[MAX_MAPS];
};

// (STATIC_MAX_RESETS: a kernel with more reset buffers, e.g. MLA + MoE + aggregation 1 in one kernel: 25; the list is in
// the kernel argument G, so the default stays 24)
#ifndef STATIC_MAX_RESETS
#define STATIC_MAX_RESETS 24
#endif
constexpr int MAX_RESETS = STATIC_MAX_RESETS;
struct ResetList {
  unsigned char *p[MAX_RESETS];
  unsigned long long n[MAX_RESETS];
  unsigned v[MAX_RESETS];
  int cnt;
};

// ---- the layer state (the kernel's parameter g) ----
struct G {
  void *buf[MAX_BUFS]; // by slot: node outputs (an exchange buffer: this GPU's
                       // copy, set 0), scratch buffers, graph tensors
  uint32_t *cnt;       // counters (NCNT)
  int rank;            // this GPU (of GPUS)
  unsigned gen;        // launch number + 1 (the start barrier's value)
  unsigned *gen_ptr;
  unsigned *arrive;
  long long par_off;            // the arena's copy 1 at + par_off (host.cuh build_shell_buffers)
  unsigned long long par_mask;  // the buffer slots in the arena (the counters always are)
#if STATIC_MAX_BUFS > 64
  unsigned long long par_mask_hi; // (slots 64 .. 127: e.g. two layer pairs in one kernel)
#endif
  ResetList resets;
  unsigned char *rv;         // this GPU's exchange region (2 sets)
  unsigned char *peer[GPUS]; // every GPU's exchange region (peer[rank] = rv)
  long long *stamps;         // [NSM][NSTAMP] ns
  long long *start_barrier;  // [1] ns: the last CU past the start barrier
};

// the exchange region's set of this launch (its launch number's parity). Read
// as wave-uniform: from LDS the compiler takes it as per lane, and every
// exchange-region buffer load then runs in a loop over the lanes' descriptors
// (MI355X 10-11: 24 such loops in the layer kernel, none with this)
__shared__ unsigned dbuf_par;
__device__ __forceinline__ size_t rg_set() {
  return (size_t)__builtin_amdgcn_readfirstlane(dbuf_par) * EXCHANGE_SET;
}
// a buffer slot's memory: an arena slot in this launch's copy (its parity; the
// buffers the kernel resets have two copies, host.cuh build_shell_buffers)
template <class P>
__device__ __forceinline__ P *buf_at(G const &g, int slot) {
  unsigned const par = __builtin_amdgcn_readfirstlane(dbuf_par);
  return reinterpret_cast<P *>(reinterpret_cast<unsigned char *>(g.buf[slot]) +
#if STATIC_MAX_BUFS > 64
                               ((par & (unsigned)(slot < 64 ? g.par_mask >> slot : g.par_mask_hi >> (slot - 64))) & 1u
                                    ? g.par_off : 0));
#else
                               ((par & (unsigned)(g.par_mask >> slot)) & 1u ? g.par_off : 0));
#endif
}
// the counters of this launch (its copy)
__device__ __forceinline__ uint32_t *cnt_of(G const &g) {
  return g.cnt + (__builtin_amdgcn_readfirstlane(dbuf_par) ? g.par_off / 4 : 0);
}
// STATIC_SOLO (host.cuh): built for GPUS GPUs, run on one; a wait for another
// GPU's data is skipped (its part counts as zeros)
#ifdef STATIC_SOLO
constexpr bool SOLO = true;
#else
constexpr bool SOLO = false;
#endif

// 16 bytes at offset `off` of this launch's set of EVERY GPU's exchange region:
// one system-scope store per GPU (CUDA: one multicast store; SOLO: this GPU's)
__device__ __forceinline__ void push16(G const &g, size_t off, uint4 v) {
  if constexpr (SOLO) {
    st16_sys(g.rv + off + rg_set(), v);
    return;
  }
#pragma unroll
  for (int r = 0; r < GPUS; r++) {
    st16_sys(g.peer[r] + off + rg_set(), v);
  }
}

// 4 bytes to every GPU (system scope)
__device__ __forceinline__ void push4(G const &g, size_t off, uint32_t v) {
#pragma unroll
  for (int r = 0; r < GPUS; r++) {
    asm volatile("global_store_dword %0, %1, off sc0 sc1" ::"v"(SOLO ? g.rv + off + rg_set() : g.peer[r] + off + rg_set()),
                 "v"(v)
                 : "memory");
    if constexpr (SOLO) {
      break;
    }
  }
}
// push16 for 8 bytes
__device__ __forceinline__ void push8(G const &g, size_t off, uint2 v) {
  if constexpr (SOLO) {
    st8_sys(g.rv + off + rg_set(), v);
    return;
  }
#pragma unroll
  for (int r = 0; r < GPUS; r++) {
    st8_sys(g.peer[r] + off + rg_set(), v);
  }
}

template <class PARAMS, int I>
__host__ __device__ constexpr int param_at() {
  static_assert(sizeof(PARAMS::v) / sizeof(int) > I + 1,
                "the node has fewer params");
  return PARAMS::v[I];
}
template <class SLOTS, int I>
__host__ __device__ constexpr int slot_at() {
  static_assert(sizeof(SLOTS::v) / sizeof(int) > I + 1,
                "the node has fewer slots");
  return SLOTS::v[I];
}

// how a gemm_tile node's K parts are combined (as on CUDA)
// COMBINE_PUSH (HIP): one K part, the result as bf16 straight into every GPU's
// copy of the node's exchange buffer [GPUS][T][N / GPUS] (this GPU's slot)
// R in blocks: moe_experts' W2 units count their R tiles per 128-column block,
// and allreduce_send's slices wait for their blocks only (not for every
// moe_experts task): in moe_experts' misc scratch after the trace: word 0 the
// number of experts used (n_live, stored by CU 0), word RBLK_STRIDE (1 + b)
// block b's tiles done (complete at n_live x 8)
// (MI355X 10-11, TP8 with the RMSNorm folded: e2e 113.6-115.5 -> 105.8-108.4 us)
constexpr size_t MOE_MISC_RBLK = 16 * 128 + 8192 + (size_t)NSM * 4 * 32 * 16;
// the counters' spacing (u32): 4 KB apart, on different memory channels
constexpr int RBLK_STRIDE = 1024;

enum Combine {
  COMBINE_SLOTS = 0,
  COMBINE_ADD = 1,
  COMBINE_STORE = 2,
  COMBINE_PUSH = 3,
  COMBINE_FINISH = 4,  // HIP: K parts in a scratch; a row block's last part adds them, pushes bf16 (gemm_tile.cuh)
  COMBINE_FINISH_Q = 5, // HIP: as FINISH, the last part quantizes the 128-row block to MXFP8 z_q and pushes it
                       // (sum_quant_send_task: latent_down without a sum_quant_send node)
  COMBINE_PUSH_ALL = 6, // HIP: as PUSH, but every GPU computes all rows (a partial sum: KDA's o_proj over this GPU's heads)
                        // and sends it to every GPU: [GPUS][T][N] bf16, the reader adds the GPUs' partials
  COMBINE_LOCAL = 7,    // HIP: one K part, a polled input, the result as bf16 [T][N] into this GPU's 0xFF-prefilled buffer
                        // (device-scope stores; its reader polls it: MLA's q_b)
  COMBINE_PUSH_RS = 8   // HIP: as PUSH_ALL (every GPU all rows, a partial sum), but each row block only to the GPU that owns
                        // its rows (GPU j: rows [j N / GPUS, (j + 1) N / GPUS)): [GPUS][N / GPUS / ROWS][T][ROWS] bf16 on GPU j,
                        // the reduce-scatter half (attn_residual's RS form adds the GPUs' partials of its columns)
  , COMBINE_FINISH_RS = 9 // HIP: K parts in a scratch (as FINISH, every GPU all rows); a row block's last part adds them and
                        // sends the bf16 block to the GPU owning its rows only, in PUSH_RS's layout
  , COMBINE_FINISH_ALL = 10 // HIP: K parts in a scratch (every GPU all rows, a polled input); a row block's last part adds them and
                        // pushes the bf16 block to every GPU, in PUSH_ALL's layout with one part ([GPUS][T][N], tile-major)
};

// 8 fp32 values -> one 16-B bf16 vector (round to nearest even)
__device__ __forceinline__ uint4 pack_bf16x8(float4 const &p0,
                                             float4 const &p1) {
  return make_uint4(pack_bf16x2(p0.x, p0.y),
                    pack_bf16x2(p0.z, p0.w),
                    pack_bf16x2(p1.x, p1.y),
                    pack_bf16x2(p1.z, p1.w));
}

// the adding side of an all-reduce (allreduce_send): thread v polls 16-B bf16
// vector v of `row` from every GPU (slot r at row + r * RANK, 0xFF-prefilled,
// written by GPU r: system-scope loads, all GPUS in flight at once; `row` is
// the same for the whole wave) and adds them in fp32 in GPU order -> the 8 sums.
// Every round loads every slot again (no per-slot "landed" flags): MI355X 10-11,
// the flagged loop (load slot r only while it is not landed) built with -O3
// gave wrong words in lanes 12-15 of every 16 (solo, sum_rmsnorm 1 x 8, 2
// vectors per thread); this loop, that loop unrolled once, or -O1 did not.
// SOLO: this GPU's slot only (the others are zeros; adding zeros changes no
// bit: the sum starts at +0).
// All-reduce landing (T >= STATIC_ARS_LAND_MIN_T; the host gives the send node its counter, static_megakernel.py): the
// senders land and add their slices, hc_post_mix reads the sums locally (tasks/allreduce_send.cuh). MI355X 11-08 glmmoe79:
// MoE T 64 277.1 / 276.7 vs 291.6 us, T 32 205.2 vs 217.0; T 8 127.6 vs 121.3 (worse: T < 32 keeps hc_post_mix's polls)
// With the landing, the sender keeps its own bf16 partial in registers (no store into its
// own slot), adds it in GPU order with the landed ones, and writes the sum there - the own slot stays 0xFF until the sum, so
// hc_post_mix polls its vectors instead of waiting for every send's count (one round trip less; same buffers, no host flag).
// MI355X 11-08 glmmoe112 (MoE, A B A B): T 16 156.1 / 157.8 vs 152.8 / 150.8 us, T 32 194.2 / 193.4 vs 184.6 / 184.3, T 64
// 261.7 / 266.0 vs 257.7 / 256.2
// The LDS hand-off (T >= STATIC_ARS_LDSHAND_MIN_T, with the landing): when glm_moe_experts' send
// node has H / 32 slices (column-major, 32 columns each: the MoE harness's default at T >= 16, MPK_ARS_HAND) and the plan puts
// hc_post_mix task x right after send task x on the same CU (the MoE harness's plan, same knob), the send task leaves its landed
// bf16 sums [T][32 columns] in LDS at ARS_HAND_OFF and hc_post_mix reads them there (no own-slot store, no poll of it). The
// kernel cannot see the placement: a plan with H / 32 MoE send slices must pair the CUs. MI355X 11-08 glmmoe117 (x 2, checks
// pass): T 64 252.4 / 250.5 vs 260.5 / 260.2 us, T 32 183.0 / 185.2 vs 187.0 / 186.1, T 16 148.0 / 149.4 vs 149.5 / 151.2
#ifndef STATIC_ARS_LDSHAND_MIN_T // (the host's MPK_ARS_HAND_MIN_T must match; 4: MI355X 11-09 glmmoe135, with LAND from 4: MoE T 8
// 100.3 / 101.4 vs 107.6 / 107.6 us, T 4 79.9 / 80.8 vs 82.8 / 82.3, KDA + MoE pair T 8 160.0 / 159.8 vs 164.2 / 167.5)
#define STATIC_ARS_LDSHAND_MIN_T 4
#endif
constexpr int ARS_HAND_OFF = 96 * 1024;
// (the hand-off applies: P2 the send node's params[2] (2: glm_moe_experts' blocks), NS its slices, W the row width)
#define ARS_HAND_ON(P2, NS, W)                                                                                                 \
  ((P2) == 2 && (NS) * 32 == (W) && T >= STATIC_ARS_LDSHAND_MIN_T && T >= STATIC_ARS_LAND_MIN_T && !SOLO)
template <class PARAMS, int I>
constexpr int param_or0() { // (PARAMS::v[I], or 0 if the node has fewer params)
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return 0;
  }
}
#ifndef STATIC_ARS_LAND_MIN_T // (the smallest T it applies at; the host's MPK_ARS_LAND_MIN_T must match)
#define STATIC_ARS_LAND_MIN_T 4 // (11-09 glmmoe135: from 4 with the LDS hand-off) // (MI355X 11-08 glmmoe95: T 16 164.6 / 165.4 vs 168.0 / 169.1 us; T 8 121.9 / 121.0 vs 117.8 / 116.9)
#endif
template <size_t RANK>
__device__ __forceinline__ void
    allreduce_land_bf16(G const &g, unsigned char const *row, int v, float *s) {
  constexpr int NR = SOLO ? 1 : GPUS;
  uint4 u[NR];
  for (;;) {
    bool all = true;
#pragma unroll
    for (int r = 0; r < NR; r++) {
      int const slot = SOLO ? g.rank : r;
      u[r] = ld16_at<AUX_SYS>(row, (uint32_t)(slot * RANK + (size_t)v * 16));
      all = all && valid16w(u[r]);
    }
    // (MI355X 11-07 glmmoe77: all loads first then a bitwise check: the timing build's landing 11 us earlier, the normal build's
    // e2e not (T 64 301.1 / 290.6 vs 291.4 us, KDA T 64 140.3 vs 133.9) - kept as it was)
    if (all) {
      break;
    }
    __builtin_amdgcn_s_sleep(1); // (tail_pause)
    asm volatile("" ::: "memory");
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
    s[k] = 0.f;
  }
#pragma unroll
  for (int r = 0; r < NR; r++) {
    uint32_t const wd[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      s[2 * k] += bf16_lo(wd[k]);
      s[2 * k + 1] += bf16_hi(wd[k]);
    }
  }
}
// allreduce_land_bf16 over NR slots (GPUS x the push_all gemm's K parts, STATIC_GEMM_PA_KSPLIT): slot q at q RANK, added in slot
// order (GPU-major, then K part: the same sums every launch)
template <size_t RANK, int NR>
__device__ __forceinline__ void allreduce_land_bf16_nr(unsigned char const *row, float *s) {
  uint4 u[NR];
  for (;;) {
    bool all = true;
#pragma unroll
    for (int r = 0; r < NR; r++) {
      u[r] = ld16_at<AUX_SYS>(row, (uint32_t)(r * RANK));
      all = all && valid16w(u[r]);
    }
    if (all) {
      break;
    }
    __builtin_amdgcn_s_sleep(1);
    asm volatile("" ::: "memory");
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
    s[k] = 0.f;
  }
#pragma unroll
  for (int r = 0; r < NR; r++) {
    uint32_t const wd[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      s[2 * k] += bf16_lo(wd[k]);
      s[2 * k + 1] += bf16_hi(wd[k]);
    }
  }
}
#if STATIC_SRN_POLL2
// allreduce_land_bf16 for several vectors of a thread at once (STATIC_SRN_POLL2): every vector's NR loads issued in one burst per
// pass, checked after; the same sums in the same order (MI355X dec265: loads waited for one vector after the other cost a round
// trip each)
template <size_t RANK, int N>
__device__ __forceinline__ void allreduce_land_bf16_n(G const &g, unsigned char const *row, int const (&vec)[N],
                                                      bool const (&on)[N], float (&s)[N][8]) {
  constexpr int NR = SOLO ? 1 : GPUS;
  uint4 u[N][NR];
  for (;;) {
#pragma unroll
    for (int j = 0; j < N; j++)
#pragma unroll
      for (int r = 0; r < NR; r++) {
        int const slot = SOLO ? g.rank : r;
        u[j][r] = on[j] ? ld16_at<AUX_SYS>(row, (uint32_t)(slot * RANK + (size_t)vec[j] * 16)) : make_uint4(0u, 0u, 0u, 0u);
      }
    bool all = true;
#pragma unroll
    for (int j = 0; j < N; j++)
#pragma unroll
      for (int r = 0; r < NR; r++) {
        all = all && valid16w(u[j][r]);
      }
    if (all) {
      break;
    }
    tail_pause();
  }
#pragma unroll
  for (int j = 0; j < N; j++) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
      s[j][k] = 0.f;
    }
#pragma unroll
    for (int r = 0; r < NR; r++) {
      uint32_t const wd[4] = {u[j][r].x, u[j][r].y, u[j][r].z, u[j][r].w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        s[j][2 * k] += bf16_lo(wd[k]);
        s[j][2 * k + 1] += bf16_hi(wd[k]);
      }
    }
  }
}
#endif
// the kernel's local variables (every function is inlined, so they stay in
// registers)
#ifdef STATIC_NODE_END_STAMPS
// STATIC_NODE_END_STAMPS=n0,n1,..: a normal build's per-CU stamps of when it last ended a task of node n_i (slot 8 + i;
// at most 8 nodes): the nodes' end times without the timing build (amd/tools/trace/nodeend_an.py)
constexpr int NODE_END_LIST[] = {STATIC_NODE_END_STAMPS};
constexpr int NODE_END_N = sizeof(NODE_END_LIST) / sizeof(int);
static_assert(NODE_END_N <= 8, "STATIC_NODE_END_STAMPS: at most 8 nodes");
#endif
#ifdef STATIC_RESET_FIRST_NODE
#include <initializer_list>
// the CUs whose first list entry is node n (the generated code's check: static_schedule.task_tables)
constexpr int count_of(int n, std::initializer_list<int> v) {
  int c = 0;
  for (int x : v) {
    c += x == n;
  }
  return c;
}
#endif
// STATIC_EARLY_L2 (early_l2): a task's weight lines, nseg segments of seg_lines 128-B lines, stride bytes apart (p null: none)
struct L2Span {
  char const *p;
  int nseg, seg_lines;
  unsigned stride;
};
struct KernelLocals {
  char *sm; // the 1024-aligned dynamic LDS start
  int sm_id;
  int end_code = -1; // the list's end entry (static_schedule.task_tables): -2 - (rank + 1024 E)
#ifdef STATIC_RESET_FIRST_NODE
  int first_node = -1;          // this CU's first list entry's node (the generated kernel, before kernel_begin)
  int const *tasks = nullptr;   // the GPU's task table ({node, x, y, z} per entry, MAX_TASKS_PER_SM per CU)
#endif
#ifdef STATIC_TIMING_BUILD
  unsigned long long *stage_stamps = nullptr;
#endif
#ifdef STATIC_GEMM_L2WARM
  int prev_node = -1; // the node of the task this CU ran last (tasks/gemm_tile.cuh STATIC_GEMM_L2WARM)
#endif
#if defined(STATIC_GEMM_L2WARM) || defined(STATIC_DRD_L2NEXT) || defined(STATIC_DRD_HOLD)
  int const *nxt = nullptr; // the CU's next list entry {node, x, y, z} (delta_rule_decode.cuh STATIC_DRD_L2NEXT too)
#endif
#ifdef STATIC_DRD_HOLD
  float *hold_dst = nullptr; // (delta_rule_decode.cuh STATIC_DRD_HOLD: this thread's block of the last pair's state, waiting in LDS)
#endif
#ifdef STATIC_GEMM_ACT_KEEP
  int keep_node = -1, keep_y = -1; // the node and grid y of the task this CU ran last (tasks/gemm_tile.cuh STATIC_GEMM_ACT_KEEP)
#endif
};

// STATIC_DRD_HOLD=N (default off; N = the o_proj node): a FLAT recurrence task followed on its CU by node N leaves its last
// pair's state (64 KB) in LDS at STATIC_DRD_HOLD_OFF instead of storing it; node N's task stores it after its own work, so its
// loads are not queued behind those stores (vmcnt counts stores in order: MI355X srvd139, T 64, the state stores ~4 us per kernel)
#ifdef STATIC_DRD_HOLD
#ifndef STATIC_DRD_HOLD_OFF
#define STATIC_DRD_HOLD_OFF (92 * 1024)
#endif
__device__ __forceinline__ int drd_hold_slot(int j, int c, int tid) { return ((j * 4 + c) * 256 + tid) * 16; }
__device__ __forceinline__ void drd_hold_flush(KernelLocals &L) {
  if (L.hold_dst == nullptr) {
    return;
  }
  int const tid = threadIdx.x;
  constexpr int CS = STATIC_DRD_COAL ? 32 : 4; // (delta_rule_decode.cuh DRD_CS)
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int c = 0; c < 4; c++) {
      float4 const v = *reinterpret_cast<float4 const *>(L.sm + STATIC_DRD_HOLD_OFF + drd_hold_slot(j, c, tid));
      *reinterpret_cast<float4 *>(L.hold_dst + (size_t)j * 128 + CS * c) = v;
    }
  L.hold_dst = nullptr;
}
#endif

// before the task loop: LDS, the launch number, the resets of the exchange set the next launch uses

// kernel_begin's touch of the kernel arguments: s_load_dword of line I .. N - 1
template <int I, int N, class P>
__device__ __forceinline__ void karg_touch(P ka) { // (P: the kernarg pointer's type, address space 4)
  if constexpr (I < N) {
    asm volatile("s_load_dword s70, %0, %1" : : "s"(ka), "i"(I * 64) : "s70");
    karg_touch<I + 1, N>(ka);
  } else {
    asm volatile("s_waitcnt lgkmcnt(0)" : : : "s70"); // the clobbered register is free after this
  }
}
__device__ __forceinline__ void kernel_begin(G const &g, KernelLocals &L) {
  extern __shared__ __attribute__((aligned(1024))) char smem_raw[];
  uintptr_t const a = ((uintptr_t)smem_raw + 1023u) & ~(uintptr_t)1023u;
  L.sm = smem_raw + (a - (uintptr_t)smem_raw);
  // every 64-B line of the kernel's arguments (Maps, G, the task list pointer)
  // into the scalar cache at once: the compiler loads them at the start one
  // after the other (each waited for, then spilled to a VGPR lane: 210 SGPR
  // spills), so each missed line cost a memory round trip before the first
  // task. All lines, however large G is (MI355X 10-13: the fused KDA + MoE
  // kernel's G with 64 buffer slots, past the 1.5 KB the first version touched:
  // its first task 8.0 us after the barrier vs 1.1 - 2.3 us for the layers alone);
  // the loads all write s70 (their values unused), one wait for all
  karg_touch<0, (int)((sizeof(Maps) + sizeof(G) + 64 + 63) / 64)>(__builtin_amdgcn_kernarg_segment_ptr());
  L.sm_id = blockIdx.x;
  int const sm_id = L.sm_id;
  // No barrier over the GPUs: the exchange region has two sets.
  __shared__ unsigned gen_s;
  __shared__ long long t_start;
#if defined(STATIC_TIMING_BUILD) || defined(STATIC_PRE_STAMPS)
  __shared__ long long t_entry; // (the entry before the launch count's read)
#endif
  if (threadIdx.x == 0) {
#if defined(STATIC_TIMING_BUILD) || defined(STATIC_PRE_STAMPS)
    t_entry = gtime();
#endif
#ifdef STATIC_EARLY_PREWAIT
    // STATIC_EARLY_PREWAIT (with STATIC_EARLY_RES): as B300's PDL path, the
    // resets and the barrier after them run before the wait for the previous
    // kernel (here the line-up's go): they touch only this kernel's own
    // buffers. CU 0 posts "resident" on entry; the line-up gives the go
    // STATIC_EARLY_PREV_US after every GPU's post (B300's harness: a 5 us
    // stand-in previous kernel, test_plugin --prev-us 5); resets that take
    // longer delay this launch past its start event, and count
    gen_s = g.gen;
    dbuf_par = gen_s & 1u;
    if (sm_id == 0) {
      push16(g,
             EXCHANGE_HELLO + 1024 + (size_t)g.rank * 16,
             make_uint4(gen_s, gen_s, gen_s, gen_s));
    }
#else
    // (no line-up, e.g. the server's per-rank kernels): the launch number from
    // this CU's own count (g.arrive lines 65 .. 72, word sm_id; only this CU
    // touches it, once per launch), not from gen_ptr, which CU 0 advances: a CU
    // starting late would read the advanced value
    {
      static_assert(NSM <= 8 * 32, "the launch counts: one per CU in 8 lines");
      unsigned *const cg = g.arrive + 32 * 65 + sm_id;
      gen_s = __hip_atomic_load(cg, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1;
      __hip_atomic_store(cg, gen_s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    dbuf_par = gen_s & 1u;
#endif
    t_start = gtime();
  }
  __syncthreads();
  unsigned const gen = gen_s;
  // device-scope stores (st16_dev, written through to memory), each thread
  // waiting for its own before the barrier: thread 0's release below covers
  // only its wave's stores, and a reset left dirty in this die's L2 (write
  // back, not coherent with the other dies) would later overwrite what
  // another die wrote there (MI355X 10-11: plain stores here -> Rn back to
  // 0xFF after sum_rmsnorm wrote it, latent_up polled forever)
  // Every buffer the host would reset is reset in the kernel, no re-arm by
  // readers: the local ones in the arena's other copy at the end (kernel_end),
  // the exchange ones here, in the set the NEXT launch uses (other GPUs may
  // already write this launch's set; the next one is idle: every GPU ended the
  // previous launch before this one started)
#if defined(STATIC_EARLY_PREWAIT) && !defined(STATIC_EARLY_RES)
#error "STATIC_EARLY_PREWAIT: with STATIC_EARLY_RES"
#endif
#ifndef STATIC_RESET_FIRST_NODE
  for (int i = 0; i < g.resets.cnt; i++) {
    unsigned const xs = (g.resets.v[i] >> 8) & 3u;
    if (xs != 0 && xs - 1 != ((gen + 1) & 1u)) {
      continue; // an exchange buffer's set this launch uses
    }
    if (xs == 0) {
      continue; // a local buffer: the arena's other copy, at the end (kernel_end)
    }
    unsigned const w = (g.resets.v[i] & 0xFFu) * 0x01010101u;
    uint4 const val = make_uint4(w, w, w, w);
    uint4 *const q = reinterpret_cast<uint4 *>(g.resets.p[i]);
    size_t const n16 = g.resets.n[i] / 16;
    for (size_t k = (size_t)sm_id * blockDim.x + threadIdx.x; k < n16;
         k += (size_t)gridDim.x * blockDim.x) {
      st16_dev(q + k, val);
    }
  }
#endif
  if (threadIdx.x < NSTAMP) {
    g.stamps[sm_id * NSTAMP + threadIdx.x] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_START] = t_start;
#if defined(STATIC_TIMING_BUILD) || defined(STATIC_PRE_STAMPS)
    g.stamps[sm_id * NSTAMP + STAMP_ENTRY] = t_entry;
    g.stamps[sm_id * NSTAMP + STAMP_STORED] = gtime();
#endif
    // the resets are write-through stores, every thread waited for its own
    // (wait_mem above), and the kernel reads what other CUs write with loads
    // that bypass the L2: a relaxed add and wait (the release / acquire pair
    // wrote back and invalidated every die's L2 per CU: MI355X 10-13, TP8,
    // the last CU in -> past this barrier 9.7-9.9 us)
#ifdef STATIC_EARLY_PREWAIT
#if defined(STATIC_TIMING_BUILD) || defined(STATIC_PRE_STAMPS)
    g.stamps[sm_id * NSTAMP + STAMP_BAR] = gtime();
#endif
    while (ld16_sys(g.rv + EXCHANGE_HELLO + 2048 + rg_set()).x < gen) {
      __builtin_amdgcn_s_sleep(1);
    }
    g.stamps[sm_id * NSTAMP + STAMP_START] = gtime(); // (the start: the go)
#endif
    if (sm_id == 0) {
      __hip_atomic_store(
          g.gen_ptr, gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    *g.start_barrier = gtime();
  }
  __syncthreads();
#ifdef STATIC_RESET_FIRST_NODE
  // (STATIC_RESET_FIRST_NODE N: the next launch's exchange resets only on the CUs whose list starts with node N, e.g. moe_experts,
  // which waits for z_q and the routing first: there, after the barrier above and not waited for (their stores complete while
  // the task waits); every other CU starts its first task without them (the default: every CU resets its slice, then the
  // barrier waits for the stores: ~2 us before the first task, MI355X dec170). The set is the next launch's: no other GPU writes
  // it before this GPU has sent its last data of this launch)
  if (L.first_node == STATIC_RESET_FIRST_NODE) {
    __shared__ int rs_n[4], rs_lo[4];
    int const t = (int)threadIdx.x;
    bool const p = t < (int)gridDim.x && L.tasks[(size_t)t * MAX_TASKS_PER_SM * 4] == STATIC_RESET_FIRST_NODE;
    unsigned long long const bn = __ballot(p), bl = __ballot(p && t < sm_id);
    if ((t & 63) == 0) {
      rs_n[t >> 6] = __popcll(bn);
      rs_lo[t >> 6] = __popcll(bl);
    }
    __syncthreads();
    int const ne = rs_n[0] + rs_n[1] + rs_n[2] + rs_n[3], rk = rs_lo[0] + rs_lo[1] + rs_lo[2] + rs_lo[3];
    for (int i = 0; i < g.resets.cnt; i++) {
      unsigned const xs = (g.resets.v[i] >> 8) & 3u;
      if (xs == 0 || xs - 1 != ((gen + 1) & 1u)) {
        continue; // a local buffer (kernel_end) or the set this launch uses
      }
      unsigned const w = (g.resets.v[i] & 0xFFu) * 0x01010101u;
      uint4 const val = make_uint4(w, w, w, w);
      uint4 *const q = reinterpret_cast<uint4 *>(g.resets.p[i]);
      size_t const n16 = g.resets.n[i] / 16;
      for (size_t k = (size_t)rk * blockDim.x + threadIdx.x; k < n16; k += (size_t)ne * blockDim.x) {
        st16_dev(q + k, val);
      }
    }
  }
#endif
}

#ifdef STATIC_EARLY_L2
// STATIC_EARLY_L2=KB with STATIC_EARLY_L2_NODES=n0,n1,.. (needs STATIC_RESET_FIRST_NODE): the CUs whose list starts with that
// node (moe_experts: they wait ~20 us for z_q and the routing while HBM idles) read the weight lines of the listed nodes' tasks
// of their XCD's CUs (CU j on XCD j % 8) into that XCD's L2 before their first task, at most KB each: one 4-B load per 128-B
// line, the results unused. Those tasks run after the expert phase; its nt loads leave the lines in the L2 (MI355X 10-24
// amd/bench/l2_warm.hip dec316: 64 KB per CU re-read 0.84 us after 384 MB of nt reads vs 3.04 cold, 128 KB 1.72 vs 5.04).
// A listed node's task type gives the lines (SPAN: node, x, y -> L2Span; gemm_tile: gemm_tile_l2span); task (j, ti) goes to
// the XCD's reader (j / 8 + ti) % readers.
#ifndef STATIC_RESET_FIRST_NODE
#error "STATIC_EARLY_L2: with STATIC_RESET_FIRST_NODE (the readers)"
#endif
#ifndef STATIC_EARLY_L2_NODES
#error "STATIC_EARLY_L2 needs STATIC_EARLY_L2_NODES"
#endif
constexpr int EARLY_L2_LIST[] = {STATIC_EARLY_L2_NODES};
// STATIC_EARLY_L2_SECOND 1 (default 0): only a listed node's tasks that follow a task of the same node on their CU (a GEMM's
// second round: its first-round tasks issue their first K steps before the activation lands, the second round starts cold;
// MI355X dec320 T 64: the projection's 94 second-round tasks ~10 us after the first round)
#ifndef STATIC_EARLY_L2_SECOND
#define STATIC_EARLY_L2_SECOND 0
#endif
__device__ __forceinline__ bool early_l2_listed(int node) {
  bool listed = false;
#pragma unroll
  for (int i = 0; i < (int)(sizeof(EARLY_L2_LIST) / sizeof(int)); i++) {
    listed = listed || node == EARLY_L2_LIST[i];
  }
  return listed;
}
template <class SPAN>
__device__ __forceinline__ void early_l2(KernelLocals const &L, SPAN span) {
  int const t = (int)threadIdx.x, xcd = L.sm_id & 7, jl = t & 31, c = t >> 5; // (thread t: list j = xcd + 8 jl, entries 8 c ..)
  int *const s_end = reinterpret_cast<int *>(L.sm);   // [32] each list's end entry (the dynamic LDS: no task runs yet)
  int *const s_bal = s_end + 32;                      // [8] readers per wave, readers below this CU per wave
  int *const s_cnt = s_end + 40;                      // [1] this CU's tasks
  int4 *const s_my = reinterpret_cast<int4 *>(L.sm + 256); // [256] this CU's tasks {node, x, y, z}
  bool const p = t < (int)gridDim.x && (t & 7) == xcd &&
                 L.tasks[(size_t)t * MAX_TASKS_PER_SM * 4] == STATIC_RESET_FIRST_NODE;
  unsigned long long const bn = __ballot(p), bl = __ballot(p && t < L.sm_id);
  if ((t & 63) == 0) {
    s_bal[t >> 6] = __popcll(bn);
    s_bal[4 + (t >> 6)] = __popcll(bl);
  }
  if (t < 32) {
    s_end[t] = MAX_TASKS_PER_SM;
  }
  if (t == 0) {
    s_cnt[0] = 0;
  }
  int const j = xcd + 8 * jl;
  bool const jok = j < (int)gridDim.x;
  int4 e[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    e[i] = jok ? reinterpret_cast<int4 const *>(L.tasks)[(size_t)j * MAX_TASKS_PER_SM + c * 8 + i] : make_int4(-1, 0, 0, 0);
  }
#if STATIC_EARLY_L2_SECOND
  int const eprev = jok && c > 0 ? L.tasks[((size_t)j * MAX_TASKS_PER_SM + c * 8 - 1) * 4] : -1; // (the entry before the chunk)
#endif
  __syncthreads();
  int const ne = s_bal[0] + s_bal[1] + s_bal[2] + s_bal[3], rk = s_bal[4] + s_bal[5] + s_bal[6] + s_bal[7];
  int fe = MAX_TASKS_PER_SM;
#pragma unroll
  for (int i = 7; i >= 0; i--) {
    fe = e[i].x < 0 ? c * 8 + i : fe;
  }
  if (jok && fe < MAX_TASKS_PER_SM) {
    atomicMin(&s_end[jl], fe);
  }
  __syncthreads();
  if (jok && ne > 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      int const ti = c * 8 + i;
#if STATIC_EARLY_L2_SECOND
      int const pn = i == 0 ? eprev : e[i > 0 ? i - 1 : 0].x;
      bool const sel = pn == e[i].x;
#else
      bool const sel = true;
#endif
      if (ti < s_end[jl] && sel && early_l2_listed(e[i].x) && (jl + ti) % ne == rk) {
        int const q = atomicAdd(s_cnt, 1);
        if (q < 256) {
          s_my[q] = e[i];
        }
      }
    }
  }
  __syncthreads();
  int const n = s_cnt[0] < 256 ? s_cnt[0] : 256;
  constexpr int BUDGET = STATIC_EARLY_L2 * 1024 / 128, U = 16; // (lines; loads per thread in flight)
  unsigned acc = 0;
  int used = 0;
  for (int q = 0; q < n; q++) {
    int4 const m = s_my[q];
    L2Span const sp = span(m.x, m.y, m.z);
    int const nl = sp.nseg * sp.seg_lines;
    if (sp.p == nullptr || used + nl > BUDGET) {
      continue;
    }
    used += nl;
    for (int l0 = 0; l0 < nl; l0 += 256 * U) {
      unsigned v[U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        int const l = l0 + t + 256 * u;
        v[u] = l < nl ? *reinterpret_cast<unsigned const *>(sp.p + (size_t)(l / sp.seg_lines) * sp.stride +
                                                            (size_t)(l % sp.seg_lines) * 128)
                      : 0u;
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        acc += v[u];
      }
    }
  }
  asm volatile("" ::"v"(acc)); // (the loads' results: kept, so the loads stay)
  __syncthreads();                // (the LDS free again for the tasks)
}
#endif

// after the task loop
__device__ __forceinline__ void kernel_end(G const &g, KernelLocals &L) {
  __syncthreads();
  if (threadIdx.x == 0) {
    g.stamps[L.sm_id * NSTAMP + STAMP_END] = gtime();
  }
  // the arena's other copy (the next launch's) to its reset values, this CU's
  // slice, after its last task (nothing of this launch touches that copy; the
  // next launch starts after this one ends: stream order)
  {
    long long const off = __builtin_amdgcn_readfirstlane(dbuf_par) ? 0 : g.par_off;
    // only the SMs without a task of the graph's last node take slices (rank / E from the list's
    // end entry): the last tasks' SMs end without the stores (MI355X 10-15: KDA's residual CUs end last; their slices and
    // then the stores' completion came after the residual's own output stores)
    int const code = -2 - L.end_code, rk = code & 1023, ne = code >> 10;
    if (L.end_code > -2 || rk == 1023 || ne == 0) {
      return;
    }
    for (int i = 0; i < g.resets.cnt; i++) {
      if (!((g.resets.v[i] >> 11) & 1u)) {
        continue;
      }
      unsigned const w = (g.resets.v[i] & 0xFFu) * 0x01010101u;
      uint4 const val = make_uint4(w, w, w, w);
      uint4 *const q = reinterpret_cast<uint4 *>(g.resets.p[i] + off);
      size_t const n16 = g.resets.n[i] / 16;
      for (size_t k = (size_t)rk * blockDim.x + threadIdx.x; k < n16; k += (size_t)ne * blockDim.x) {
        st16_dev(q + k, val);
      }
    }
  }
}

} // namespace static_mk
