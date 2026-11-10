// tasks/gemm_tile.cuh (HIP, gfx950) -- one bf16 GEMM tile, the same task as
// include/mirage/static_megakernel/tasks/gemm_tile.cuh: out[t][row0 + r] = sum
// over K tiles [k0, k0 + nk) of weight[row0 + r][k] * act[t][k], for r < ROWS
// (128, 64, 32 or 16) and the T tokens; `mode` (Combine, core.cuh) says how the K parts
// are combined in the node's output buffer. On gfx950 (no TMA, no tcgen05):
//   - the task's activation slice [T][nk 128] goes into LDS first: after its
//     producer counted all its tasks (cnt_wait_in, ld16_in), or polled
//     (`polled`: its producer's 0xFF-prefilled buffer, each 16 B as it lands)
//   - the 4 wavefronts split the row tiles of 16 (ROWS 128: 2 each; 64: 1 each;
//     32: 2 waves per tile, each half the task's K; 16: 4 waves, a quarter
//     each; the K parts of a tile added through LDS in wave order);
//     each streams its weight fragments straight from the row-major bf16 weight
//     into registers, GEMM_STAGES K steps of 32 ahead (one 16-B load per lane:
//     row l % 16, K 8 (l / 16) .. + 7 of the step), into
//     v_mfma_f32_16x16x32_bf16 with the tokens' fragment from LDS
//   - the result (D: lane -> token l % 16, register r -> row 4 (l / 16) + r)
//     goes out per mode: COMBINE_SLOTS 16-B device-scope stores (its readers
//     poll them), COMBINE_ADD fixed-point atomics, COMBINE_STORE st16_out; then
//     the counters (task_post) as on CUDA
// The BF16 operand layout (lane l: K 8 (l / 16) .. + 7 of the 32) is the ISA's
// figure, checked on day one by amd/bench/mfma_layout_check.hip.
#pragma once
#include "../core.cuh"
#include "sum_quant_send.cuh" // COMBINE_FINISH_Q: the z_q tile and its push

namespace static_mk {

#ifndef STATIC_GEMM_STAGES
#define STATIC_GEMM_STAGES 8
#endif
// K steps of 32 in flight per wave (a variant): 8 steps x 2 row tiles = 16 KB
// of weights per wave, 64 KB per CU (Little's law at 8 TB/s and ~1 us: ~32 KB
// per CU of 256)
constexpr int GEMM_STAGES = STATIC_GEMM_STAGES;
// the stages of a task of S K steps: the largest divisor of S up to GEMM_STAGES
// (whole rounds; e.g. shared down's 12 steps: 6)
constexpr int gemm_stages(int S) {
  for (int d = GEMM_STAGES < S ? GEMM_STAGES : S; d > 1; d--) {
    if (S % d == 0) {
      return d;
    }
  }
  return 1;
}

// the activation slice of a task in LDS: [T][KP] bf16, rows gemm_act_pitch apart
// (STATIC_GEMM_ACT_PAD 1: rows KP + 8 elements apart: with KP * 2 a multiple of
// 256 B the 8 tokens' fragments of a K step sit on the same LDS banks; MI355X
// 10-16 amd/bench/lds_mfma.hip: a 24-step loop from LDS 3638 -> 2038 clocks)
#ifndef STATIC_GEMM_ACT_PAD
#define STATIC_GEMM_ACT_PAD 0
#endif
// STATIC_GEMM_ACT_DMA 1 (T >= 16 only; off until measured): a task's counted or graph-input activation slice by LDS DMA (fill_act)
#ifndef STATIC_GEMM_ACT_DMA
#define STATIC_GEMM_ACT_DMA 0
#endif
// STATIC_GEMM_ACT_DMA_PAD 1 (default 0; with STATIC_GEMM_ACT_DMA and STATIC_GEMM_ACT_PAD): padded rows filled by LDS DMA too
// when a row is whole 1-KB wave pieces (the chunk's KPC a multiple of 512): each wave's DMA instruction lands inside one row,
// its LDS base at the row's padded start (the default with padded rows: register rounds of 8 pieces per thread, one round trip
// after the other - 4 at T 64 x 1024 columns; MI355X dec399, T 64 projection second round: x landed ~5 us after the task start)
#ifndef STATIC_GEMM_ACT_DMA_PAD
#define STATIC_GEMM_ACT_DMA_PAD 0
#endif
constexpr int GEMM_ACT_AUX = AUX_DEV;
// the RMSNorm folded into a GEMM (gemm_norm_fill): tokens per wave per polled round (dec47, T = 64: 4 297.6 / 297.9 vs 2
// 293.8 / 292.3 us; T = 32 the same)
#ifndef STATIC_GEMM_NORM_QR
#define STATIC_GEMM_NORM_QR 2
#endif
#ifdef STATIC_GEMM_DBG_ACT
template <class OFF, class LDSAT>
__device__ __forceinline__ void gemm_dbg_act(G const &g, void const *src, int nv, int at, OFF &&off_of, LDSAT &&lds_at,
                                             int per_token) {
  for (int v = STATIC_TIDX; v < nv; v += 256) {
    uint4 const a = ld16_at<AUX_DEV>(src, off_of(v));
    uint4 const b = *reinterpret_cast<uint4 const *>(lds_at(v));
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
      atomicAdd(reinterpret_cast<unsigned long long *>(g.stamps + 8 + 2 * at), 1ull);
      atomicMax(reinterpret_cast<unsigned long long *>(g.stamps + 9 + 2 * at), (unsigned long long)(v / per_token + 1));
    }
  }
}
#endif
template <class P>
__device__ __forceinline__ v4f gemm_lds_b(P const *p) {
  return *reinterpret_cast<v4f const *>(p);
}
template <int KP>
constexpr int gemm_act_pitch() {
  return KP + (STATIC_GEMM_ACT_PAD ? 8 : 0);
}
template <int KP>
constexpr int gemm_act_bytes() {
  return T * gemm_act_pitch<KP>() * 2;
}
// the tokens as the MFMA's B columns: GEMM_NC tiles of 16 tokens (T <= 16: one tile, tokens >= T padded)
constexpr int GEMM_NC = (T + 15) / 16;
// after it (16-B aligned): the partial sums of the waves that take K parts 1 ..
// of a row tile (ROWS 32 / 16): at most 3 waves x 64 lanes x 16 B per column tile
constexpr int GEMM_RED_BYTES = 3 * 64 * 16 * GEMM_NC;
// the activation chunk of a task (KPC columns): a task's KP columns are filled into LDS and multiplied in KP / KPC chunks,
// one after another, when [T][KP] does not fit beside the K parts' sums (T > 16: 64 tokens x 7168 x 2 B); one chunk
// (KPC = KP) whenever it fits
template <int ROWS, int KP>
constexpr int gemm_act_kpc() {
  constexpr int TT = ROWS / 16, KSPLIT = TT >= 4 ? 1 : 4 / TT;
  for (int d = 1; d <= KP / 32; d++) {
    if (KP % d == 0 && (KP / d) % (32 * KSPLIT) == 0 &&
        T * (KP / d + (STATIC_GEMM_ACT_PAD ? 8 : 0)) * 2 + 16 + GEMM_RED_BYTES <= 160 * 1024 - 1024) {
      return KP / d;
    }
  }
  return 32 * KSPLIT;
}

// the weight's layout (params[4], HIP): 0 the model's [N][K] (a wave's load: 16
// rows x 64 B, rows K * 2 bytes apart); 1 shuffled [N / 16][K / 32][64 lanes][16
// B]: lane l's fragment of row tile r, K step k (row 16 r + l % 16, K 32 k + 8
// (l / 16) .. + 7) at ((r K / 32 + k) 64 + l) 16 bytes, a wave's load 1 KB
// contiguous (static_megakernel.shuffle_gemm_weight; MI355X 10-11
// amd/bench/gemm_load.hip: 10-25% faster, with nt loads)
enum GemmLayout { GEMM_ROWMAJOR = 0, GEMM_SHUFFLED = 1 };
// the shuffled weight loads' cache bits: 2 (nt)
constexpr int GEMM_SHUF_AUX = 2;
// STATIC_GEMM_LANE_ASM (default 0, A/B kda51): the activation fragment's lane
// offset made in the loop (volatile v_mbcnt) instead of one hoisted VGPR: in the
// fused KDA + MoE build that VGPR was spilled and its reload before the loop
// made every round wait vmcnt(0)
#ifndef STATIC_GEMM_LANE_ASM
#define STATIC_GEMM_LANE_ASM 0
#endif
// STATIC_GEMM_LDSW 3: a task with a polled activation whose whole weight slice fits in LDS (after its activation slice
// and the K parts' sums, within GEMM_LDS_MAX) copies it there first, every load issued at once
// (global_load_lds_dwordx4: no registers held, so ~128 KB per CU in flight instead of GEMM_STAGES steps per wave), then
// fills the activation (waiting for its producer: KDA's o_proj waits for the recurrence) and runs the MFMAs from LDS
#ifndef STATIC_GEMM_LDSW
#define STATIC_GEMM_LDSW 0
#endif
static_assert(STATIC_GEMM_LDSW == 0 || STATIC_GEMM_LDSW == 3, "STATIC_GEMM_LDSW: 0 (off) or 3");
constexpr int GEMM_LDS_MAX = 156 * 1024;
// the weight slice in LDS: whether a task (ROWS x KP) copies it there, and
// where it starts (after the activation slice and the K parts' sums)
template <int ROWS, int KP>
constexpr int gemm_w_lds_at() {
  return ((gemm_act_bytes<KP>() + 15) & ~15) + GEMM_RED_BYTES;
}
// STATIC_GEMM_L2WARM=KB with STATIC_GEMM_L2WARM_NODES=n0,n1,..: a listed node's first task on a CU (it waits for its
// activation, e.g. a MoE + attention kernel's projection during the MoE's tail), once its own weight loads are issued,
// reads the first KB of the CU's next task's weight slice (its next list entry, the same node) into the XCD's L2: one
// 4-B load per 128-B line, the results unused; that task then streams them from the L2 (MI355X 10-16
// amd/bench/l2_warm.hip, one kernel: 64 KB per CU pre-read 0.64 vs 3.04 us cold, 128 KB 3.08 vs 5.16; nt pre-reads are
// not kept). Shuffled weights only
#ifdef STATIC_GEMM_L2WARM
#ifndef STATIC_GEMM_L2WARM_NODES
#error "STATIC_GEMM_L2WARM needs STATIC_GEMM_L2WARM_NODES"
#endif
constexpr int GEMM_L2WARM_LIST[] = {STATIC_GEMM_L2WARM_NODES};
// the next task's grid position (x, y) if this task is the CU's first of a listed node and the next is of the same
// node; else x = -1
__device__ __forceinline__ int2 gemm_l2warm_next(KernelLocals const &L, int node) {
  bool listed = false;
#pragma unroll
  for (int i = 0; i < (int)(sizeof(GEMM_L2WARM_LIST) / sizeof(int)); i++) {
    listed = listed || node == GEMM_L2WARM_LIST[i];
  }
  if (!listed || L.prev_node == node || L.nxt == nullptr) {
    return make_int2(-1, 0);
  }
  return make_int2(L.nxt[0] == node ? L.nxt[1] : -1, L.nxt[2]);
}
#endif
#ifndef STATIC_GEMM_HYB4_REGS
#define STATIC_GEMM_HYB4_REGS 4
#endif
// HYB: the K steps of a wave's first row tile copied into LDS (from w_at, 4 waves x 1 KB per step); the rest held in registers.
// All S when they fit in the launch's LDS (160 KB per CU less the kernel's 1 KB of static LDS; the launch adds 1 KB:
// config.cuh smem_launch_bytes): T <= 8; at T = 16 the activation slice (32 KB) leaves room for S - 2
template <int S>
constexpr int gemm_hyb_lds_steps(int w_at) {
  return (158 * 1024 - w_at) / 4096 < S ? (158 * 1024 - w_at) / 4096 : S;
}
// GEMM_RPER: the activation's 16-B pieces per thread per round (registers) - polled: a round trip per round (KDA o_proj at
// T 64: 32 pieces per thread, 4 rounds)
constexpr int GEMM_RPER = 8;
template <int ROWS, int KP, bool POLLED = false>
constexpr bool gemm_ldsw() {
  return STATIC_GEMM_LDSW == 3 && POLLED && gemm_w_lds_at<ROWS, KP>() + ROWS * KP * 2 <= GEMM_LDS_MAX;
}
// 8 bytes at device scope (COMBINE_LOCAL: another CU of this GPU polls them)
__device__ __forceinline__ void st8x_dev_gemm(void *p, uint2 v) {
  v2u const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(p), "v"(w) : "memory");
}
// one 16-B piece per lane from global memory into LDS at lds + 16 lane (lds:
// the same in every lane)
__device__ __forceinline__ void lds_dma16(void const *src, void *lds) {
  __builtin_amdgcn_global_load_lds(
      (__attribute__((address_space(1))) void *)(const_cast<void *>(src)),
      (__attribute__((address_space(3))) void *)(lds),
      16,
      0,
      GEMM_SHUF_AUX);
}

// the same with a given cache policy (an activation another CU wrote: AUX_DEV)
template <int AUX>
__device__ __forceinline__ void lds_dma16_aux(void const *src, void *lds) {
  __builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void *)(const_cast<void *>(src)),
                                   (__attribute__((address_space(3))) void *)(lds), 16, 0, AUX);
}

// one wave's row tiles (NT of 16 rows from row tile tile0 of the weight) over KS
// of the task's K columns (S = KS / 32 steps, from K step ks0 of the weight);
// act = the LDS slice at the wave's first column (rows KA apart); acc[NT x GEMM_NC]: row tile i, token tile c at
// acc[i * GEMM_NC + c]. fill_act():
// called by every wave after its first STG steps' weight loads are issued
// (they need nothing from another task), before the first step reads the LDS
// slice: it fills the slice (and waits for the producer); so the task's weight
// round trip and its activation round trip overlap (MI355X 10-11: a 16 / 32-row
// task took 4-5 us whatever its size: activation, then weights, then stores, one
// memory round trip after the other)
template <int NT, int KS, int KA, int KW, int LAYOUT, bool WLDS, bool HYB = false, bool HYB4 = false, class FILL>
__device__ __forceinline__ void gemm_wave(__hip_bfloat16 const *weight,
                                          int tile0,
                                          int ks0,
                                          char const *act,
                                          char *wlds,
                                          v4f *acc,
                                          FILL &&fill_act) {
  constexpr int S = KS / 32, STG = gemm_stages(S);
  int const lane = STATIC_TIDX & 63, tok = lane % 16, g = lane / 16;
  // weight fragment of row tile i, K step s: row 16 (tile0 + i) + l % 16,
  // columns 32 (ks0 + s) + 8 g .. + 7
  __hip_bfloat16 const *wl =
      weight + (size_t)(16 * tile0 + lane % 16) * KW + 32 * ks0 + 8 * g;
  __amdgpu_buffer_rsrc_t const rs = buf_rsrc(weight);
  auto wfrag = [&](int i, int s) {
    if constexpr (LAYOUT == GEMM_SHUFFLED) {
      return __builtin_bit_cast(
          v4f,
          __builtin_amdgcn_raw_buffer_load_b128(
              rs,
              (uint32_t)((((size_t)(tile0 + i) * (KW / 32) + ks0 + s) * 64 +
                          lane) *
                         16),
              0,
              GEMM_SHUF_AUX));
    } else {
      return *reinterpret_cast<v4f const *>(wl + (size_t)16 * i * KW +
                                            32 * s);
    }
  };
  // the activation's fragment: lanes of tokens >= T load token tok % T's (the
  // MFMA's output column n depends only on B's column n: those columns are never
  // stored). No branch: in a large kernel (the fused KDA + MoE) the compiler made
  // the guarded load a branch per step and waited at each join for nearly every
  // weight load in flight (MI355X 10-13: vmcnt(1 .. 6) before each MFMA)
  // token of lane token-column x in tile c: 16 c + x, clamped to T - 1 (T <= 16 a power of 2: x & (T - 1))
  auto tok_of = [](int x, int c) {
    if constexpr (T <= 16 && (T & (T - 1)) == 0) {
      return x & (T - 1);
    } else {
      return 16 * c + x < T ? 16 * c + x : T - 1;
    }
  };
  auto bfrag = [&](int s, int c) {
    if constexpr (STATIC_GEMM_LANE_ASM) { // the lane from v_mbcnt each step: no loop-invariant address to keep (or spill)
      int ln;
      asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(ln));
      return gemm_lds_b(act + ((size_t)tok_of(ln % 16, c) * KA + 32 * s + 8 * (ln / 16)) * 2);
    }
    return gemm_lds_b(act + ((size_t)tok_of(tok, c) * KA + 32 * s + 8 * g) * 2);
  };
  auto step = [&](int s, v4f const *f) {
#pragma unroll
    for (int c = 0; c < GEMM_NC; c++) {
      v8bf const b = __builtin_bit_cast(v8bf, bfrag(s, c));
#pragma unroll
      for (int i = 0; i < NT; i++) {
        acc[i * GEMM_NC + c] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            __builtin_bit_cast(v8bf, f[i]), b, acc[i * GEMM_NC + c], 0, 0, 0);
      }
    }
  };
  if constexpr (HYB4) { // (gemm_tile_layer hybrid=4) row tile 0's steps into LDS as HYB, tiles 1 .. 3 in registers: a 256-row
    // task's whole weight slice on chip before the wait (MI355X 10-23 dec149, T 8: the 350 128-row tasks' second round
    // ended ~6 us after the first; 175 of these hold the KDA projection in one round)
    static_assert(NT == 4 && LAYOUT == GEMM_SHUFFLED, "gemm_wave HYB4: four row tiles per wave, shuffled weights");
    constexpr int HL = gemm_hyb_lds_steps<S>(((T * KA * 2 + 15) & ~15) + GEMM_RED_BYTES);
    // (STATIC_GEMM_HYB4_REGS: more of tile 0's steps in registers when T's activation slice leaves less LDS, T = 32: 10)
    static_assert(HL > 0 && S - HL <= STATIC_GEMM_HYB4_REGS, "gemm_wave HYB4: at most STATIC_GEMM_HYB4_REGS of tile 0's steps in registers");
#pragma unroll
    for (int s = 0; s < HL; s++) {
      lds_dma16(reinterpret_cast<char const *>(weight) + (((size_t)tile0 * (KW / 32) + ks0 + s) * 64 + lane) * 16,
                wlds + s * 1024);
    }
    v4f fr[3][S];
#pragma unroll
    for (int t = 0; t < 3; t++)
#pragma unroll
      for (int s = 0; s < S; s++) {
        fr[t][s] = wfrag(1 + t, s);
      }
    v4f fr0[S - HL > 0 ? S - HL : 1];
#pragma unroll
    for (int s = HL; s < S; s++) {
      fr0[s - HL] = wfrag(0, s);
    }
    fill_act();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
    for (int s = 0; s < S; s++) {
      v4f f[4];
      f[0] = s < HL ? *reinterpret_cast<v4f const *>(wlds + s * 1024 + lane * 16) : fr0[s < HL ? 0 : s - HL];
      f[1] = fr[0][s];
      f[2] = fr[1][s];
      f[3] = fr[2][s];
      step(s, f);
    }
    return;
  }
  if constexpr (HYB) { // (gemm_hyb) row tile 0's S steps into LDS (wlds: [S][64 lanes][16 B]), row tile 1's in registers
    static_assert(NT == 2 && LAYOUT == GEMM_SHUFFLED, "gemm_wave HYB: two row tiles per wave, shuffled weights");
    // tile 0's steps HL .. S - 1 in registers too when LDS has no room for them (T = 16)
    constexpr int HL = gemm_hyb_lds_steps<S>(((T * KA * 2 + 15) & ~15) + GEMM_RED_BYTES);
    static_assert(HL > 0 && S - HL <= 4, "gemm_wave HYB: at most 4 of tile 0's steps in registers");
#pragma unroll
    for (int s = 0; s < HL; s++) {
      lds_dma16(reinterpret_cast<char const *>(weight) + (((size_t)tile0 * (KW / 32) + ks0 + s) * 64 + lane) * 16,
                wlds + s * 1024);
    }
    v4f fr1[S];
#pragma unroll
    for (int s = 0; s < S; s++) {
      fr1[s] = wfrag(1, s);
    }
    if constexpr (HL < S) {
      v4f fr0[S - HL];
#pragma unroll
      for (int s = HL; s < S; s++) {
        fr0[s - HL] = wfrag(0, s);
      }
      fill_act();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
      for (int s = 0; s < S; s++) {
        v4f f[2];
        f[0] = s < HL ? *reinterpret_cast<v4f const *>(wlds + s * 1024 + lane * 16) : fr0[s < HL ? 0 : s - HL];
        f[1] = fr1[s];
        step(s, f);
      }
      return;
    }
    fill_act();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // (the copies and tile 1's loads, issued before the activation's)
#pragma unroll
    for (int s = 0; s < S; s++) {
      v4f f[2];
      f[0] = *reinterpret_cast<v4f const *>(wlds + s * 1024 + lane * 16);
      f[1] = fr1[s];
      step(s, f);
    }
    return;
  }
  if constexpr (WLDS) { // wlds: this wave's [S][NT][64 lanes][16 B]
#pragma unroll
    for (int s = 0; s < S; s++)
#pragma unroll
      for (int i = 0; i < NT; i++) {
        void const *src;
        if constexpr (LAYOUT == GEMM_SHUFFLED) {
          src = reinterpret_cast<char const *>(weight) +
                (((size_t)(tile0 + i) * (KW / 32) + ks0 + s) * 64 + lane) * 16;
        } else {
          src = wl + (size_t)16 * i * KW + 32 * s;
        }
        lds_dma16(src, wlds + (s * NT + i) * 1024);
      }
    fill_act();
    // the copies count in vmcnt, issued before the activation's loads: all
    // landed once those did; the wait the compiler sees for the LDS reads
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll 4
    for (int s = 0; s < S; s++) {
      v4f f[NT];
#pragma unroll
      for (int i = 0; i < NT; i++) {
        f[i] = *reinterpret_cast<v4f const *>(wlds + (s * NT + i) * 1024 +
                                              lane * 16);
      }
      step(s, f);
    }
    return;
  }
  v4f frag[STG][NT];
#pragma unroll
  for (int s = 0; s < STG; s++)
#pragma unroll
    for (int i = 0; i < NT; i++) {
      frag[s][i] = wfrag(i, s);
    }
  fill_act();
  // rounds of STG steps; the last round peeled, so the others prefetch without
  // a branch (a branch around the prefetch makes the compiler wait for every
  // outstanding load at the top of each round)
#pragma unroll 1
  for (int k0 = 0; k0 < S - STG; k0 += STG) {
#pragma unroll
    for (int s = 0; s < STG; s++) {
      step(k0 + s, frag[s]);
#pragma unroll
      for (int i = 0; i < NT; i++) {
        frag[s][i] = wfrag(i, k0 + s + STG);
      }
      // a scheduling barrier after each K step's refill loads: step s + 1 waits for its own loads only (MI355X 10-14:
      // in the fused KDA + MoE kernel the compiler moved the refills so each round waited vmcnt(0) twice; kda50: MLA
      // 46.8 / 45.9 / 46.3 vs 48.5 / 47.6 / 47.9 us, KDA 47.1-47.4 vs 47.6-47.8, MoE the same)
      __builtin_amdgcn_sched_barrier(0);
    }
  }
#pragma unroll
  for (int s = 0; s < STG; s++) {
    step(S - STG + s, frag[s]);
  }
}

// The RMSNorm folded into latent_up (NORM): the activation slice [T][KP] is
// made from the all-reduce's rows instead of RMSNorm's output: per 16-B piece
// the GPUS bf16 slots of R (allreduce_send's exchange buffer at ROFF, [GPUS][T]
// [KW] bf16, polled at system scope) added in GPU order (fp32, as sum_rmsnorm),
// times gamma, rounded to bf16 (Rn = that x rstd: rstd, a per-token scalar, is
// applied to the GEMM's sums by gemm_finish). Each wave holds whole tokens (KP
// 512: wave w, piece q: token w + 4 q, lane l: columns 8 l .. + 7), adds the
// token's squares over its KP columns (a fixed butterfly) and lane 0 stores the
// sum (16 B at ssq + token * 16, device scope) before the task counts itself.
#ifndef STATIC_GEMM_QUANT_COOP
#define STATIC_GEMM_QUANT_COOP 0
#endif
#ifndef STATIC_GEMM_FINISH_FLAG
#define STATIC_GEMM_FINISH_FLAG 0
#endif
#ifndef STATIC_GEMM_NORM_POLL2
#define STATIC_GEMM_NORM_POLL2 0
#endif
constexpr int GEMM_NORM_AUX = AUX_SYS;
template <int KP, int KW, int NPART, size_t ROFF>
__device__ __forceinline__ void gemm_norm_fill(G const &g,
                                               __hip_bfloat16 const *gamma,
                                               char *sm,
                                               int k0c,
                                               unsigned char *ssq
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_NORM_RTT)
                                               , unsigned long long *rtt_stamp = nullptr
#endif
                                               ) {
  static_assert(KP == 512 || KP == 256, "norm: a wave per token, 64 lanes x 8 (or 4) columns");
  // wave w: tokens w + 4 q, QR per round (q0 = 0, QR, ..); a token >= T is skipped
  constexpr int QW = (T + 3) / 4, QR = STATIC_GEMM_NORM_QR;
  if constexpr (KP == 512) {
  constexpr int NR = SOLO ? 1 : GPUS;
  constexpr size_t RANK = (size_t)T * KW * 2;
  int const lane = STATIC_TIDX & 63, wave = STATIC_TIDX >> 6;
  unsigned char const *const rbase = g.rv + ROFF + rg_set();
  // gamma of the lane's 8 columns first: in flight while R is polled
  uint4 const gw = *reinterpret_cast<uint4 const *>(gamma + k0c + 8 * lane);
  for (int q0 = 0; q0 < QW; q0 += QR) {
  uint4 u[QR][NR];
  bool ok[QR][NR];
#pragma unroll
  for (int q = 0; q < QR; q++)
#pragma unroll
    for (int r = 0; r < NR; r++) {
      ok[q][r] = wave + 4 * (q0 + q) >= T;
    }
  bool all;
  do {
    all = true;
#if STATIC_GEMM_NORM_POLL2
    // (STATIC_GEMM_NORM_POLL2: every not-yet-valid row of the round reloaded in one burst - the loads of rows of tokens < T issued
    // back to back, then checked - instead of each load checked inside its own branch: the compiler waited for each before the
    // next (MI355X dec261 / dec264: the first pass of 16 loads per lane took ~5.5 us vs ~1.2 us for 64 KB read at once))
#pragma unroll
    for (int q = 0; q < QR; q++)
#pragma unroll
      for (int r = 0; r < NR; r++) {
        // (STATIC_GEMM_NORM_POLL2 2: only rows not yet valid - a pass re-read all 131 KB of the CU's slice at ~37 GB/s, ~3.5 us, so
        // the slots landing during a pass cost up to two more full passes: MI355X 10-25 dec375, T 16, fill complete 7-10 us after
        // the last R ack; the loads still in one burst, the checks in the loop below)
        if (wave + 4 * (q0 + q) < T && (STATIC_GEMM_NORM_POLL2 < 2 || !ok[q][r])) {
          int const slot = SOLO ? g.rank : r;
          u[q][r] = ld16_at<GEMM_NORM_AUX>(
              rbase, (uint32_t)(slot * RANK +
                                ((size_t)(wave + 4 * (q0 + q)) * KW + k0c + 8 * lane) * 2));
        }
      }
#pragma unroll
    for (int q = 0; q < QR; q++)
#pragma unroll
      for (int r = 0; r < NR; r++) {
        ok[q][r] = ok[q][r] || valid16w(u[q][r]);
        all = all && ok[q][r];
      }
#else
#pragma unroll
    for (int q = 0; q < QR; q++)
#pragma unroll
      for (int r = 0; r < NR; r++) {
        if (!ok[q][r]) {
          int const slot = SOLO ? g.rank : r;
          u[q][r] = ld16_at<GEMM_NORM_AUX>(
              rbase, (uint32_t)(slot * RANK +
                                ((size_t)(wave + 4 * (q0 + q)) * KW + k0c + 8 * lane) * 2));
          ok[q][r] = valid16w(u[q][r]);
        }
        all = all && ok[q][r];
      }
#endif
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_NORM_RTT)
    // (STATIC_GEMM_NORM_RTT, the timing build: stage stamp 1 = the first poll pass's loads returned, thread 0)
    if (q0 == 0 && rtt_stamp != nullptr && STATIC_TIDX == 0) {
      *rtt_stamp = gtime();
      rtt_stamp = nullptr;
    }
#endif
    if (!all) {
      tail_pause();
    }
  } while (!all);
  uint32_t const g4[4] = {gw.x, gw.y, gw.z, gw.w};
#pragma unroll
  for (int q = 0; q < QR; q++) {
    int const t = wave + 4 * (q0 + q);
    if (t >= T) {
      continue;
    }
    float sv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < NR; r++) { // GPU order
      uint32_t const wd[4] = {u[q][r].x, u[q][r].y, u[q][r].z, u[q][r].w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        sv[2 * k] += bf16_lo(wd[k]);
        sv[2 * k + 1] += bf16_hi(wd[k]);
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      ss += sv[k] * sv[k];
    }
#pragma unroll
    for (int d = 32; d > 0; d >>= 1) {
      ss += __shfl_xor(ss, d);
    }
    uint32_t a[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      a[k] = pack_bf16x2(sv[2 * k] * bf16_lo(g4[k]),
                         sv[2 * k + 1] * bf16_hi(g4[k]));
    }
  *reinterpret_cast<uint4 *>(sm + ((size_t)t * gemm_act_pitch<KP>() + 8 * lane) * 2) =
      make_uint4(a[0], a[1], a[2], a[3]);
  
    if (lane == 0) {
      st16_dev(ssq + (size_t)t * 16,
               make_uint4(__float_as_uint(ss), 0u, 0u, 0u));
    }
  }
  }
  } else { // KP 256 (latent_up with 14 K parts): 4 columns per lane, 8-B loads (half the bytes per poll round)
    constexpr int NR = SOLO ? 1 : GPUS;
    constexpr size_t RANK = (size_t)T * KW * 2;
    int const lane = STATIC_TIDX & 63, wave = STATIC_TIDX >> 6;
    unsigned char const *const rbase = g.rv + ROFF + rg_set();
    uint2 const gw = *reinterpret_cast<uint2 const *>(gamma + k0c + 4 * lane);
    typedef unsigned gn_v2u __attribute__((ext_vector_type(2)));
    for (int q0 = 0; q0 < QW; q0 += QR) {
    uint2 u[QR][NR];
    bool ok[QR][NR];
#pragma unroll
    for (int q = 0; q < QR; q++)
#pragma unroll
      for (int r = 0; r < NR; r++) {
        ok[q][r] = wave + 4 * (q0 + q) >= T;
      }
    bool all;
    do {
      all = true;
#pragma unroll
      for (int q = 0; q < QR; q++)
#pragma unroll
        for (int r = 0; r < NR; r++) {
          if (!ok[q][r]) {
            int const slot = SOLO ? g.rank : r;
            gn_v2u const w = __builtin_amdgcn_raw_buffer_load_b64(
                buf_rsrc(rbase),
                (uint32_t)(slot * RANK + ((size_t)(wave + 4 * (q0 + q)) * KW + k0c + 4 * lane) * 2), 0, GEMM_NORM_AUX);
            u[q][r] = make_uint2(w.x, w.y);
            ok[q][r] = u[q][r].x != 0xFFFFFFFFu && u[q][r].y != 0xFFFFFFFFu;
          }
          all = all && ok[q][r];
        }
      if (!all) {
        tail_pause();
      }
    } while (!all);
    uint32_t const g2[2] = {gw.x, gw.y};
#pragma unroll
    for (int q = 0; q < QR; q++) {
      int const t = wave + 4 * (q0 + q);
      if (t >= T) {
        continue;
      }
      float sv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < NR; r++) { // GPU order
        uint32_t const wd[2] = {u[q][r].x, u[q][r].y};
#pragma unroll
        for (int k = 0; k < 2; k++) {
          sv[2 * k] += bf16_lo(wd[k]);
          sv[2 * k + 1] += bf16_hi(wd[k]);
        }
      }
      float ss = 0.f;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        ss += sv[k] * sv[k];
      }
#pragma unroll
      for (int d = 32; d > 0; d >>= 1) {
        ss += __shfl_xor(ss, d);
      }
      uint32_t a2[2];
#pragma unroll
      for (int k = 0; k < 2; k++) {
        a2[k] = pack_bf16x2(sv[2 * k] * bf16_lo(g2[k]), sv[2 * k + 1] * bf16_hi(g2[k]));
      }
    *reinterpret_cast<uint2 *>(sm + ((size_t)t * gemm_act_pitch<KP>() + 4 * lane) * 2) = make_uint2(a2[0], a2[1]);
    
      if (lane == 0) {
        st16_dev(ssq + (size_t)t * 16, make_uint4(__float_as_uint(ss), 0u, 0u, 0u));
      }
    }
    }
  }
  __syncthreads();
}

// STATIC_GEMM_COOP_FINISH 1 (default 0; FINISH without quantization, FINISH_RS): every K part's task of a row block waits
// until the block's NPART parts are counted, then adds and sends its 1 / NPART of the block's items, instead of the last one
// doing all of them (MI355X dec191, fused KDA T 64: latent_up's 7 finishers each read 7 x 32 KB of parts and pushed 128 KB to
// every GPU, ending ~16 us after their sums while the other parts' tasks were done; moe_agg1 waited for them). The waits are
// safe: a K part's task waits only for the other parts' tasks of its node, which depend on nothing after them
#ifndef STATIC_GEMM_COOP_FINISH
#define STATIC_GEMM_COOP_FINISH 0
#endif
template <int ROWS, int NPART, size_t OOFF>
__device__ __forceinline__ void gemm_finish_rs(G const &g, float const *parts, int N, uint32_t *bcnt, int orow0,
                                               int split = 0) {
  static_assert(NPART <= 14, "finish_rs: at most 14 K parts");
  __shared__ int fin_last;
  wait_mem();
  __syncthreads();
  if (STATIC_TIDX == 0) {
    uint32_t const old = __hip_atomic_fetch_add(bcnt, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    fin_last = (old + 1) % NPART == 0;
    if constexpr (STATIC_GEMM_COOP_FINISH) { // (every part: until the block's NPART are counted)
      cnt_wait_in(bcnt, (old / NPART + 1) * NPART);
      fin_last = 1;
    }
  }
  __syncthreads();
  asm volatile("" ::: "memory"); // the parts' loads after the count
  int const NS = N / GPUS, j = orow0 / NS;
  if (!fin_last || (SOLO && j != g.rank)) {
    return;
  }
  constexpr int NV = T * ROWS / 8;
  int const vlo = STATIC_GEMM_COOP_FINISH ? NV * split / NPART : 0, vhi = STATIC_GEMM_COOP_FINISH ? NV * (split + 1) / NPART : NV;
  unsigned char *pj = g.rv;
  if constexpr (!SOLO) { // (a select chain, as COMBINE_PUSH_RS)
    pj = g.peer[0];
#pragma unroll
    for (int i = 1; i < GPUS; i++) {
      pj = j == i ? g.peer[i] : pj;
    }
  }
  pj += rg_set();
  for (int v = vlo + STATIC_TIDX; v < vhi; v += 256) { // item v: token v / (ROWS / 8), rows orow0 + 8 (v % (ROWS / 8)) .. + 7
    int const t = v / (ROWS / 8), r = orow0 + 8 * (v % (ROWS / 8));
    float4 x0[NPART], x1[NPART];
#pragma unroll
    for (int p_ = 0; p_ < NPART; p_++) {
      uint32_t const off = (uint32_t)((((size_t)p_ * T + t) * N + r) * 4);
      x0[p_] = as_float4(ld16_at<AUX_DEV>(parts, off));
      x1[p_] = as_float4(ld16_at<AUX_DEV>(parts, off + 16));
    }
    float o8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p_ = 0; p_ < NPART; p_++) { // fixed order
      o8[0] += x0[p_].x;
      o8[1] += x0[p_].y;
      o8[2] += x0[p_].z;
      o8[3] += x0[p_].w;
      o8[4] += x1[p_].x;
      o8[5] += x1[p_].y;
      o8[6] += x1[p_].z;
      o8[7] += x1[p_].w;
    }
    st16_sys(pj + OOFF + (size_t)g.rank * T * NS * 2 + (size_t)(orow0 - j * NS) * T * 2 + (size_t)v * 16,
             make_uint4(pack_bf16x2(o8[0], o8[1]), pack_bf16x2(o8[2], o8[3]), pack_bf16x2(o8[4], o8[5]),
                        pack_bf16x2(o8[6], o8[7])));
  }
  __syncthreads();
}

// COMBINE_FINISH_ALL: as gemm_finish_rs, but the row block's last K part pushes the bf16 sums to every GPU (PUSH_ALL's layout,
// one part: OOFF + (rank T N + orow0 T) 2 + 16 v): the K parts keep each task's activation slice small (DSA o_proj T 64: 128 KB
// instead of 512 KB) and the all-reduce starts per row block, without an allreduce_send step
template <int ROWS, int NPART, size_t OOFF>
__device__ __forceinline__ void gemm_finish_all(G const &g, float const *parts, int N, uint32_t *bcnt, int orow0
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
                                                , unsigned long long *fstamp = nullptr
#endif
                                                ) {
  static_assert(NPART <= 14, "finish_all: at most 14 K parts");
  __shared__ int fin_last;
  wait_mem();
  __syncthreads();
  if (STATIC_TIDX == 0) {
    uint32_t const old = __hip_atomic_fetch_add(bcnt, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    fin_last = (old + 1) % NPART == 0;
  }
  __syncthreads();
  asm volatile("" ::: "memory"); // the parts' loads after the count
  if (!fin_last) {
    return;
  }
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
  // (STATIC_GEMM_FIN_STAMP, the timing build, probe 11-09: stage stamp 0 = this task is the row block's finisher, 1 = its pushes
  // complete - every thread waited for its stores (vmcnt 0), thread 0 stamps after the barrier)
  if (fstamp && STATIC_TIDX == 0) {
    fstamp[0] = gtime();
  }
#endif
  constexpr int NV = T * ROWS / 8;
  for (int v = STATIC_TIDX; v < NV; v += 256) { // item v: token v / (ROWS / 8), rows orow0 + 8 (v % (ROWS / 8)) .. + 7
    int const t = v / (ROWS / 8), r = orow0 + 8 * (v % (ROWS / 8));
    float4 x0[NPART], x1[NPART];
#pragma unroll
    for (int p_ = 0; p_ < NPART; p_++) {
      uint32_t const off = (uint32_t)((((size_t)p_ * T + t) * N + r) * 4);
      x0[p_] = as_float4(ld16_at<AUX_DEV>(parts, off));
      x1[p_] = as_float4(ld16_at<AUX_DEV>(parts, off + 16));
    }
    float o8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p_ = 0; p_ < NPART; p_++) { // fixed order
      o8[0] += x0[p_].x;
      o8[1] += x0[p_].y;
      o8[2] += x0[p_].z;
      o8[3] += x0[p_].w;
      o8[4] += x1[p_].x;
      o8[5] += x1[p_].y;
      o8[6] += x1[p_].z;
      o8[7] += x1[p_].w;
    }
    uint4 const sv = make_uint4(pack_bf16x2(o8[0], o8[1]), pack_bf16x2(o8[2], o8[3]), pack_bf16x2(o8[4], o8[5]),
                                pack_bf16x2(o8[6], o8[7]));
    push16(g, OOFF + (size_t)g.rank * T * N * 2 + (size_t)orow0 * T * 2 + (size_t)v * 16, sv);
  }
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
  wait_mem();
  __syncthreads();
  if (fstamp && STATIC_TIDX == 0) {
    fstamp[1] = gtime();
  }
#endif
  __syncthreads();
}

// COMBINE_FINISH, after a task stored its K part (st16_dev) of rows orow0 ..
// + ROWS of parts [NPART][T][N] fp32: the task counts itself on its row
// block's counter (bypass protocol: its stores complete first), and the one
// that makes the count a multiple of NPART adds the NPART parts in order (the
// same fp32 adds as sum_send) and pushes the bf16 rows to every GPU's slot
// g.rank of the output (OOFF: [GPUS][T][N] bf16 in the exchange region, as
// COMBINE_PUSH and sum_send). No task waits: the last to count finishes.
// NORM_K > 0 (the RMSNorm folded in, gemm_norm_fill): each task also stored
// its K part's sums of squares, [row blocks][NPART][T] 16-B entries after the
// parts; the finisher adds them in order and scales its sums by rstd =
// 1 / sqrt(sum / NORM_K + eps) before the bf16 push.
#if STATIC_GEMM_QUANT_COOP == 2
template <int NPART, size_t OOFF, int QN>
__device__ __noinline__ void gemm_quant_coop(G const &g, float *parts, int kt, char *sm, int split) {
  sum_quant_send_task<NPART, OOFF, QN, NPART>(g, parts, kt, sm, split);
}
#endif
template <int ROWS, int NPART, size_t OOFF, int NORM_K = 0, int EPS_BITS = 0,
          int QUANT_N = 0>
__device__ __forceinline__ void gemm_finish(G const &g,
                                            float const *parts,
                                            int N,
                                            uint32_t *bcnt,
                                            int orow0,
                                            char *sm = nullptr,
                                            int split = 0
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
                                            , unsigned long long *fstamp = nullptr
#endif
                                            ) {
  static_assert(NPART <= 14, "finish: at most 14 K parts");
  __shared__ int fin_last;
#if STATIC_GEMM_FINISH_FLAG
  // (STATIC_GEMM_FINISH_FLAG, with STATIC_GEMM_COOP_FINISH: no count - the K parts (and the folded norm's sums of squares) are
  // their own flags, 0xFF before each launch (MPK_FINISH_FLAG=1: the scratch reset), every part polls its share of them: no
  // wait for its own stores, no counter add and poll; MI355X dec261-265: each was a ~1-2 us round trip in the tail)
  static_assert(QUANT_N > 0 || STATIC_GEMM_COOP_FINISH, "STATIC_GEMM_FINISH_FLAG: with STATIC_GEMM_COOP_FINISH");
  if constexpr (QUANT_N == 0) {
    if (STATIC_TIDX == 0) {
      fin_last = 1;
    }
    __syncthreads();
  } else {
#endif
  wait_mem();
  __syncthreads();
  if (STATIC_TIDX == 0) {
    uint32_t const old = __hip_atomic_fetch_add(
        bcnt, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    fin_last = (old + 1) % NPART == 0;
#if STATIC_GEMM_QUANT_COOP
    if constexpr (QUANT_N > 0) { // (STATIC_GEMM_QUANT_COOP: every part waits for the tile's NPART counts, then makes its tokens)
      cnt_wait_in(bcnt, (old / NPART + 1) * NPART);
      fin_last = 1;
    }
#endif
    if constexpr (STATIC_GEMM_COOP_FINISH && QUANT_N == 0) { // (every part: until the block's NPART are counted)
      cnt_wait_in(bcnt, (old / NPART + 1) * NPART);
      fin_last = 1;
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
      // (STATIC_GEMM_FIN_STAMP, the timing build: stage stamp 1 = every K part of the row block counted, thread 0)
      if (fstamp) {
        fstamp[1] = gtime();
      }
#endif
    }
  }
  __syncthreads();
#if STATIC_GEMM_FINISH_FLAG
  }
#endif
  asm volatile("" ::: "memory"); // the parts' loads after the count
  if constexpr (QUANT_N > 0) { // COMBINE_FINISH_Q: z_q of z columns [orow0, + 128)
    static_assert(ROWS == 128, "finish_quant: 128-row tasks (one z_q tile)");
#if STATIC_GEMM_QUANT_COOP
    // (STATIC_GEMM_QUANT_COOP, T >= 32: the tile's NPART K-part tasks each add the parts of tokens [split T / NPART, ..), quantize and
    // push them - sum_quant_send's token groups (Y = NPART) - instead of the last part doing the whole tile: each reads 1 / NPART of the
    // parts (MI355X 10-26: sum_quant_send's 3 tasks per GPU read 262 KB each at T 64, z_q ~16-20 us after latent_down)
    // (STATIC_GEMM_QUANT_COOP 2: through a function of its own - inlined into the fused T 64 kernel it took 507 -> 512 VGPRs + 70 spills)
    if constexpr (STATIC_GEMM_QUANT_COOP == 2) {
      gemm_quant_coop<NPART, OOFF, QUANT_N>(g, const_cast<float *>(parts), orow0 / 128, sm, split);
    } else {
      sum_quant_send_task<NPART, OOFF, QUANT_N, NPART>(g, const_cast<float *>(parts), orow0 / 128, sm, split);
    }
#else
    if (fin_last) { // (uniform: an LDS word) every thread: the tile in LDS
      sum_quant_send_task<NPART, OOFF, QUANT_N>(
          g, const_cast<float *>(parts), orow0 / 128, sm);
    }
#endif
    __syncthreads();
    return;
  }
#if STATIC_GEMM_COOP_FINISH
  for (int v = T * ROWS / 8 * split / NPART + STATIC_TIDX; fin_last && v < T * ROWS / 8 * (split + 1) / NPART; v += 256) {
#else
  for (int v = STATIC_TIDX; fin_last && v < T * ROWS / 8; v += 256) {
#endif
    // item v: token v / (ROWS / 8), rows orow0 + 8 (v % (ROWS / 8)) .. + 7
    int const t = v / (ROWS / 8),
              r = orow0 + 8 * (v % (ROWS / 8));
    float4 x0[NPART], x1[NPART];
#if STATIC_GEMM_FINISH_FLAG
    float sqf[NPART];
    for (;;) { // every part's two vectors (and its sum of squares) in one burst, then checked
#pragma unroll
      for (int p_ = 0; p_ < NPART; p_++) {
        uint32_t const off = (uint32_t)((((size_t)p_ * T + t) * N + r) * 4);
        x0[p_] = as_float4(ld16_at<AUX_DEV>(parts, off));
        x1[p_] = as_float4(ld16_at<AUX_DEV>(parts, off + 16));
      }
      if constexpr (NORM_K > 0) {
        unsigned char const *ssa = reinterpret_cast<unsigned char const *>(parts + (size_t)NPART * T * N);
#pragma unroll
        for (int p_ = 0; p_ < NPART; p_++) {
          sqf[p_] = __uint_as_float(
              ld16_at<AUX_DEV>(ssa, (uint32_t)((((size_t)(orow0 / ROWS) * NPART + p_) * T + t) * 16)).x);
        }
      }
      bool all = true;
#pragma unroll
      for (int p_ = 0; p_ < NPART; p_++) {
        all = all && validf4(x0[p_]) && validf4(x1[p_]);
        if constexpr (NORM_K > 0) {
          all = all && __float_as_uint(sqf[p_]) != 0xFFFFFFFFu;
        }
      }
      if (all) {
        break;
      }
      tail_pause();
    }
#else
#pragma unroll
    for (int p_ = 0; p_ < NPART; p_++) {
      uint32_t const off = (uint32_t)((((size_t)p_ * T + t) * N + r) * 4);
      x0[p_] = as_float4(ld16_at<AUX_DEV>(parts, off));
      x1[p_] = as_float4(ld16_at<AUX_DEV>(parts, off + 16));
    }
#endif
    float rstd = 1.f;
    if constexpr (NORM_K > 0) { // the K parts' sums of squares of token t, in order
#if STATIC_GEMM_FINISH_FLAG
      float const *const sq = sqf;
#else
      unsigned char const *ssa = reinterpret_cast<unsigned char const *>(
          parts + (size_t)NPART * T * N);
      float sq[NPART];
#pragma unroll
      for (int p_ = 0; p_ < NPART; p_++) {
        sq[p_] = __uint_as_float(
            ld16_at<AUX_DEV>(ssa, (uint32_t)((((size_t)(orow0 / ROWS) * NPART +
                                                p_) * T + t) * 16)).x);
      }
#endif
      float tot = 0.f;
#pragma unroll
      for (int p_ = 0; p_ < NPART; p_++) {
        tot += sq[p_];
      }
      rstd = rsqrtf(tot / NORM_K + __int_as_float(EPS_BITS));
    }
    float o8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p_ = 0; p_ < NPART; p_++) { // fixed order
      o8[0] += x0[p_].x;
      o8[1] += x0[p_].y;
      o8[2] += x0[p_].z;
      o8[3] += x0[p_].w;
      o8[4] += x1[p_].x;
      o8[5] += x1[p_].y;
      o8[6] += x1[p_].z;
      o8[7] += x1[p_].w;
    }
    if constexpr (NORM_K > 0) {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        o8[k] *= rstd;
      }
    }
    push16(g,
           OOFF + (size_t)g.rank * T * N * 2 + ((size_t)t * N + r) * 2,
           make_uint4(pack_bf16x2(o8[0], o8[1]),
                      pack_bf16x2(o8[2], o8[3]),
                      pack_bf16x2(o8[4], o8[5]),
                      pack_bf16x2(o8[6], o8[7])));
  }
  __syncthreads();
}

// COMBINE_PUSH_ALL's output tile-major (MI355X 10-13: push_pattern one hop 4.2 vs 5.6 us; KDA 54.0 / 53.7 vs 56.8 /
// 57.2; all four of 10-13 together 46.3 / 46.6), [GPUS][N / ROWS][T]
// [ROWS] bf16: a task's T x ROWS block is contiguous, staged in LDS and pushed with
// 16-B stores (one 2 T ROWS-B segment per GPU instead of 8-B stores at T
// segments); attn_residual reads it so (its PR template argument)

// the whole task: ROWS rows from row0 (of the weight), K part `split` (KP
// columns from k0c); out / N / cnt / mode as CUDA's gemm_tile_task
template <int ROWS, int KP, int KW, bool POLLED, int LAYOUT, size_t OOFF = 0,
          int NPART = 1, bool NORM = false, size_t ROFF = 0, int EPS_BITS = 0,
          int QUANT_N = 0, bool WAITED = false, bool READY = false, bool HYB = false, bool HYB4 = false>
__device__ __forceinline__ void gemm_tile_task(G const &g,
                                               __hip_bfloat16 const *weight,
                                               __hip_bfloat16 const *act_src,
                                               uint32_t const *act_cnt,
                                               uint32_t act_need,
                                               int mode,
                                               void *out,
                                               int N,
                                               uint32_t *cnt,
                                               int wrow0,
                                               int orow0,
                                               int split,
                                               int k0c,
                                               __hip_bfloat16 const *gamma,
                                               char *sm STAGE_STAMP_PARAM,
                                               int2 warm = make_int2(-1, 0)
#ifdef STATIC_GEMM_ACT_KEEP
                                               , bool keep = false
#endif
                                               ) {
  static_assert(ROWS == 128 || ROWS == 64 || ROWS == 32 || ROWS == 16 || (HYB4 && ROWS == 256),
                "128, 64, 32 or 16 weight rows per task (HYB4: 256)");
  // TT row tiles of 16; NT per wave; KSPLIT waves per tile, each KS columns
  constexpr int TT = ROWS / 16, NT = TT >= 4 ? TT / 4 : 1,
                KSPLIT = TT >= 4 ? 1 : 4 / TT, KPC = gemm_act_kpc<ROWS, KP>(), ACH = KP / KPC, KS = KPC / KSPLIT;
  static_assert(KS % 32 == 0, "a wave's K columns: whole steps of 32");
  __syncthreads(); // the LDS below may still be read by the previous task
  // 1. the activation slice -> LDS ([T][KP] bf16; chunk cc of ACH: [T][KPC]): in gemm_wave, once every wave's first
  // weight loads are in flight (fill_act)
  [[maybe_unused]] int cc = 0;
  auto fill_act = [&]() {
    if constexpr (ACH > 1) {
      if (cc > 0) {
        __syncthreads(); // every wave's MFMAs of the previous chunk (reading its activation) done
      }
    }
#ifdef STATIC_GEMM_ACT_KEEP
    // STATIC_GEMM_ACT_KEEP (default off): this CU's previous task was this node's task of the same K part (grid y: the same
    // activation slice, already waited for), and nothing ran between them: its slice is still in LDS (the plan pairs them:
    // plan_one_gpu.kpair_plan, MPK_KPAIR)
    if constexpr (!POLLED && !NORM && ACH == 1 && !HYB && !HYB4) {
      if (keep) {
        if (STATIC_TIDX == 0) {
          STAGE_STAMP(true)
        }
        return;
      }
    }
#endif
    if constexpr (NORM) {
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_NORM_WSTAMP)
      // (STATIC_GEMM_NORM_WSTAMP, the timing build: stage stamp 1 = every load issued before the fill landed, e.g. the task's
      // weights - a diagnosis: R's polls wait in issue order behind them)
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
      if (STATIC_TIDX == 0) { // the timing build: stage stamp 1 = the R poll starts
        STAGE_STAMP_AT(1)
      }
      gemm_norm_fill<KP, KW, NPART, ROFF>(
          g, gamma, sm, k0c,
          reinterpret_cast<unsigned char *>(reinterpret_cast<float *>(out) +
                                            (size_t)NPART * T * N) +
              (((size_t)(orow0 / ROWS) * NPART + split) * T) * 16
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_NORM_RTT)
          , stamp ? stamp + 1 : nullptr
#endif
          );
      if (STATIC_TIDX == 0) {
        STAGE_STAMP(true)
      }
      return;
    }
#ifdef STATIC_GEMM_L2WARM
    // (STATIC_GEMM_L2WARM: the next task's first KB, WB bytes of each of its RT row tiles, before the wait)
    constexpr int RT = ROWS / 16,
                  WB = STATIC_GEMM_L2WARM * 1024 / RT < KP * 32 ? STATIC_GEMM_L2WARM * 1024 / RT : KP * 32,
                  WNL = RT * (WB / 128), WPER = (WNL + 255) / 256;
    unsigned wv[WPER];
#pragma unroll
    for (int j = 0; j < WPER; j++) {
      wv[j] = 0;
    }
    if constexpr (!POLLED && WAITED && LAYOUT == GEMM_SHUFFLED) {
      if (warm.x >= 0) {
        __amdgpu_buffer_rsrc_t const rsw = buf_rsrc(weight);
#pragma unroll
        for (int j = 0; j < WPER; j++) {
          int const l = STATIC_TIDX + 256 * j;
          if (l < WNL) {
            wv[j] = __builtin_amdgcn_raw_buffer_load_b32(
                rsw,
                (uint32_t)((((size_t)warm.x * RT + l / (WB / 128)) * (KW / 32) + (size_t)warm.y * (KP / 32)) * 1024 +
                           (l % (WB / 128)) * 128),
                0, 0);
          }
        }
      }
    }
#endif
    if constexpr (!POLLED) {
      if (act_cnt != nullptr) {
        if (STATIC_TIDX == 0) {
          cnt_wait_in(act_cnt, act_need);
        }
        __syncthreads();
      }
    }
    // this thread's 16-B pieces (v = STATIC_TIDX + 256 q), in rounds of at most RPER pieces (registers), all loads of
    // a round in flight together (tracked); polled: re-loaded until all landed
    constexpr int NV = T * KPC / 8, PER = (NV + 255) / 256, RPER = PER < GEMM_RPER ? PER : GEMM_RPER;
    int kcc = k0c; // this chunk's first K column
    if constexpr (ACH > 1) {
      kcc += cc * KPC;
    }
    auto off_of = [&](int v) {
      return (uint32_t)((((size_t)(v / (KPC / 8))) * KW + kcc +
                         (v % (KPC / 8)) * 8) * 2);
    };
    // T > 16, not polled, rows not padded: by LDS DMA (global_load_lds; piece v lands at 16 v: a wave's 64 pieces are 1 KB
    // in a row), every piece of the slice in flight at once and no registers held (the register rounds below are RPER
    // pieces per thread per round trip: 4 round trips for 64 tokens x 1024 columns)
    if constexpr (STATIC_GEMM_ACT_DMA && !POLLED &&
                  (STATIC_GEMM_ACT_PAD == 0 || (STATIC_GEMM_ACT_DMA_PAD && (KPC / 8) % 64 == 0)) && T >= 16 &&
                  NV % 256 == 0) {
      int const wvi = __builtin_amdgcn_readfirstlane((int)STATIC_TIDX >> 6);
#pragma unroll 4
      for (int q = 0; q < NV / 256; q++) {
        int v = (int)STATIC_TIDX + 256 * q;
        char const *const src = reinterpret_cast<char const *>(act_src) + off_of(v);
#if STATIC_GEMM_ACT_PAD
        // (STATIC_GEMM_ACT_DMA_PAD: the wave's 64 pieces, from piece v0, inside row v0 / (KPC / 8) of the padded slice)
        int const v0 = 256 * q + 64 * wvi;
        char *const dst = sm + ((size_t)(v0 / (KPC / 8)) * gemm_act_pitch<KPC>() + (v0 % (KPC / 8)) * 8) * 2;
#else
        char *const dst = sm + ((size_t)256 * q + 64 * wvi) * 16;
#endif
        if (act_cnt != nullptr || READY) {
          lds_dma16_aux<GEMM_ACT_AUX>(src, dst);
        } else {
          lds_dma16_aux<0>(src, dst);
        }
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // (the DMA's loads count in vmcnt; the weight loads issued before too)
      __syncthreads();
      if (STATIC_TIDX == 0) {
        STAGE_STAMP(true)
      }
      return;
    }
#pragma unroll 1
    for (int q0 = 0; q0 < PER; q0 += RPER) {
    uint4 x[RPER];
    bool ok[RPER];
#pragma unroll
    for (int q = 0; q < RPER; q++) {
      ok[q] = STATIC_TIDX + 256 * (q0 + q) >= NV;
    }
    bool all;
    do {
#pragma unroll
      for (int q = 0; q < RPER; q++) {
        if (!ok[q]) {
          uint32_t const off = off_of(STATIC_TIDX + 256 * (q0 + q));
          x[q] = POLLED ? ld16_at<AUX_DEV>(act_src, off)
                        : (act_cnt != nullptr || READY ? ld16_at<GEMM_ACT_AUX>(act_src, off) : ld16_at<0>(act_src, off));
        }
      }
      all = true;
#pragma unroll
      for (int q = 0; q < RPER; q++) {
        ok[q] = ok[q] || !POLLED || valid16w(x[q]);
        all = all && ok[q];
      }
      if (!all) {
        tail_pause();
      }
    } while (!all);
#pragma unroll
    for (int q = 0; q < RPER; q++) {
      int const v = STATIC_TIDX + 256 * (q0 + q);
      if (v < NV) {
      *reinterpret_cast<uint4 *>(
          sm + ((size_t)(v / (KPC / 8)) * gemm_act_pitch<KPC>() + (v % (KPC / 8)) * 8) * 2) = x[q];
      
      }
    }
    }
    __syncthreads();
#ifdef STATIC_GEMM_DBG_ACT
    if constexpr (!POLLED) {
      gemm_dbg_act(g, act_src, NV, 0, off_of, [&](int v) {
        return sm + ((size_t)(v / (KPC / 8)) * gemm_act_pitch<KPC>() + (v % (KPC / 8)) * 8) * 2; }, KPC / 8);
    }
#endif
    if (STATIC_TIDX == 0) {
      STAGE_STAMP(true)
    }
#ifdef STATIC_GEMM_L2WARM
#pragma unroll
    for (int j = 0; j < WPER; j++) {
      asm volatile("" ::"v"(wv[j])); // (the warm loads' results: used here, once the activation's loads waited anyway)
    }
#endif
  };
  // 2. this wave's row tiles (its first: tile0) over its K part kp
  // the wave number as a wave-uniform value (the LDS copies' base is a scalar)
  int const wave = __builtin_amdgcn_readfirstlane(STATIC_TIDX >> 6),
            lane = STATIC_TIDX & 63;
  int const tile0 = KSPLIT == 1 ? NT * wave : wave % TT,
            kp = KSPLIT == 1 ? 0 : wave / TT;
  v4f acc[NT * GEMM_NC];
#pragma unroll
  for (int i = 0; i < NT * GEMM_NC; i++) {
    acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
  }
  constexpr bool WLDS = !HYB && !HYB4 && gemm_ldsw<ROWS, KP, POLLED>();
  static_assert(!HYB4 || (!POLLED && !NORM && NT == 4 && KSPLIT == 1 && !HYB),
                "gemm_tile HYB4: a 256-row task (not polled), one K part per wave");
  static_assert(!HYB || (!POLLED && !NORM && NT == 2 && KSPLIT == 1),
                "gemm_tile HYB: a 128-row task (not polled), one K part per wave");
  static_assert(ACH == 1 || (!HYB && !NORM && !WLDS), "gemm_tile: activation chunks without HYB, NORM, weights in LDS");
  if constexpr (ACH == 1) {
    gemm_wave<NT, KS, gemm_act_pitch<KP>(), KW, LAYOUT, WLDS, HYB, HYB4>(
        weight,
        wrow0 / 16 + tile0,
        (k0c + kp * KS) / 32,
        sm + (size_t)kp * KS * 2,
        sm + gemm_w_lds_at<ROWS, KP>() + (size_t)wave * ((HYB || HYB4) ? gemm_hyb_lds_steps<KS / 32>(gemm_w_lds_at<ROWS, KP>()) : NT * (KS / 32)) * 1024,
        acc,
        fill_act);
#ifdef STATIC_GEMM_DBG_ACT
    if constexpr (!POLLED && !NORM) {
      __syncthreads();
      constexpr int NVD = T * KP / 8;
      gemm_dbg_act(g, act_src, NVD, 1, [&](int v) {
        return (uint32_t)((((size_t)(v / (KP / 8))) * KW + k0c + (v % (KP / 8)) * 8) * 2); }, [&](int v) {
        return sm + ((size_t)(v / (KP / 8)) * gemm_act_pitch<KP>() + (v % (KP / 8)) * 8) * 2; }, KP / 8);
    }
#endif
  } else {
    for (cc = 0; cc < ACH; cc++) {
      gemm_wave<NT, KS, gemm_act_pitch<KPC>(), KW, LAYOUT, WLDS, HYB>(
          weight,
          wrow0 / 16 + tile0,
          (k0c + cc * KPC + kp * KS) / 32,
          sm + (size_t)kp * KS * 2,
          sm + gemm_w_lds_at<ROWS, KP>() + (size_t)wave * ((HYB || HYB4) ? gemm_hyb_lds_steps<KS / 32>(gemm_w_lds_at<ROWS, KP>()) : NT * (KS / 32)) * 1024,
          acc,
          fill_act);
    }
  }
  if constexpr (KSPLIT > 1) { // K parts 1 .. -> LDS; part 0 adds them in order
    v4f *const red = reinterpret_cast<v4f *>(
        sm + ((gemm_act_bytes<KPC>() + 15) & ~15));
    if (kp > 0) {
#pragma unroll
      for (int c = 0; c < GEMM_NC; c++) {
        red[(((kp - 1) * TT + tile0) * GEMM_NC + c) * 64 + lane] = acc[c];
      }
    }
    __syncthreads();
    if (kp == 0) {
#pragma unroll
      for (int j = 1; j < KSPLIT; j++) {
#pragma unroll
        for (int c = 0; c < GEMM_NC; c++) {
          v4f const o = red[(((j - 1) * TT + tile0) * GEMM_NC + c) * 64 + lane];
          acc[c][0] += o[0];
          acc[c][1] += o[1];
          acc[c][2] += o[2];
          acc[c][3] += o[3];
        }
      }
    }
  }
  if constexpr (!NORM) { // (timing build: stage stamp 1 = this wave's MFMAs done (thread 0); NORM tasks: the R poll)
    if (STATIC_TIDX == 0) {
      STAGE_STAMP_AT(1)
    }
  }
  // 3. the result (the waves of K part 0): lane -> token l % 16, registers ->
  // rows 4 (l / 16) .. + 3
  static_assert(gemm_act_bytes<KPC>() >= T * ROWS * 2, "push tile: the block in the activation's LDS");
  __hip_bfloat16 *const ptile = reinterpret_cast<__hip_bfloat16 *>(sm); // [T][ROWS] (the activation is read by now)
  if constexpr (KSPLIT == 1) {
    if (mode == COMBINE_PUSH_ALL || mode == COMBINE_PUSH_RS) {
      __syncthreads(); // every wave's MFMAs (reading the activation) done
    }
  }
#pragma unroll
  for (int c = 0; c < GEMM_NC; c++) {
  int const tok = 16 * c + lane % 16;
  if (tok < T && kp == 0) {
#pragma unroll
    for (int i = 0; i < NT; i++) {
      v4f const &ac = acc[i * GEMM_NC + c];
      int const row = orow0 + 16 * (tile0 + i) + 4 * (lane / 16);
      uint4 const v4 = make_uint4(__float_as_uint(ac[0]),
                                  __float_as_uint(ac[1]),
                                  __float_as_uint(ac[2]),
                                  __float_as_uint(ac[3]));
      if (mode == COMBINE_SLOTS || mode == COMBINE_FINISH ||
          mode == COMBINE_FINISH_Q || mode == COMBINE_FINISH_RS || mode == COMBINE_FINISH_ALL) {
        st16_dev_acc(reinterpret_cast<float *>(out) +
                     ((size_t)split * T + tok) * N + row,
                 v4);
      } else if (mode == COMBINE_LOCAL) { // bf16 into this GPU's buffer
        st8x_dev_gemm(reinterpret_cast<__hip_bfloat16 *>(out) + (size_t)tok * N + row,
                      make_uint2(pack_bf16x2(ac[0], ac[1]),
                                 pack_bf16x2(ac[2], ac[3])));
      } else if (mode == COMBINE_PUSH_ALL || mode == COMBINE_PUSH_RS) { // the block in LDS, pushed below
        *reinterpret_cast<uint2 *>(ptile + tok * ROWS + (row - orow0)) =
            make_uint2(pack_bf16x2(ac[0], ac[1]), pack_bf16x2(ac[2], ac[3]));
      } else if (mode == COMBINE_PUSH) { // N: this GPU's rows
        push8(g,
              OOFF + (size_t)g.rank * T * N * 2 + ((size_t)tok * N + row) * 2,
              make_uint2(pack_bf16x2(ac[0], ac[1]),
                         pack_bf16x2(ac[2], ac[3])));
      } else if (mode == COMBINE_ADD) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
          red_add_fx(reinterpret_cast<unsigned long long *>(out) +
                         (size_t)tok * N + row + r,
                     ac[r]);
        }
      } else { // (store: K part `split` at [split][T][N], one part: [T][N])
        st16_dev_acc(reinterpret_cast<float *>(out) + ((size_t)split * T + tok) * N + row, v4); // (st16_out: MFMA results)
      }
    }
  }
  }
  if (mode == COMBINE_PUSH_RS) { // the T x ROWS block to the GPU that owns its rows only, its tile-major slot g.rank
    __syncthreads();
    int const NS = N / GPUS, j = orow0 / NS;
    if (!SOLO || j == g.rank) {
      unsigned char *pj = g.rv;
      if constexpr (!SOLO) { // (a select chain: g.peer indexed by a run-time value would put G in memory)
        pj = g.peer[0];
#pragma unroll
        for (int i = 1; i < GPUS; i++) {
          pj = j == i ? g.peer[i] : pj;
        }
      }
      pj += rg_set();
      for (int v = STATIC_TIDX; v < T * ROWS / 8; v += 256) {
        st16_sys(pj + OOFF + (size_t)g.rank * T * NS * 2 + (size_t)(orow0 - j * NS) * T * 2 + (size_t)v * 16,
                 *reinterpret_cast<uint4 const *>(ptile + 8 * v));
      }
    }
  }
  if (mode == COMBINE_PUSH_ALL) { // the T x ROWS block: 16 B per thread to every GPU
    __syncthreads();
    for (int v = STATIC_TIDX; v < T * ROWS / 8; v += 256) {
      push16(g, OOFF + ((size_t)g.rank * NPART + split) * T * N * 2 + (size_t)orow0 * T * 2 + (size_t)v * 16,
             *reinterpret_cast<uint4 const *>(ptile + 8 * v));
    }
  }
  if (mode == COMBINE_ADD || mode == COMBINE_STORE) { // per 128-row block / per task
    task_post(cnt + (mode == COMBINE_ADD ? orow0 / 128 : 0), 1);
  } else if (mode == COMBINE_FINISH) {
    gemm_finish<ROWS, NPART, OOFF, NORM ? KW : 0, EPS_BITS>(
        g, reinterpret_cast<float const *>(out), N, cnt + orow0 / ROWS, orow0, nullptr, split
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
        , stamp
#endif
        );
  } else if (mode == COMBINE_FINISH_Q) {
    gemm_finish<ROWS, NPART, OOFF, 0, 0, QUANT_N>(
        g, reinterpret_cast<float const *>(out), N, cnt + orow0 / ROWS, orow0, sm, split);
  } else if (mode == COMBINE_FINISH_RS) {
    gemm_finish_rs<ROWS, NPART, OOFF>(g, reinterpret_cast<float const *>(out), N, cnt + orow0 / ROWS, orow0, split);
  } else if (mode == COMBINE_FINISH_ALL) {
    gemm_finish_all<ROWS, NPART, OOFF>(g, reinterpret_cast<float const *>(out), N, cnt + orow0 / ROWS, orow0
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_GEMM_FIN_STAMP)
        , stamp
#endif
        );
  } else {
    __syncthreads();
  }
}

// COMBINE_FINISH: the buffer slot of the K parts (BUF_SLOTS[0]); other modes -1
template <int MODE, class SLOTS>
constexpr int gemm_parts_slot() {
  if constexpr (MODE == COMBINE_FINISH || MODE == COMBINE_FINISH_Q || MODE == COMBINE_FINISH_RS || MODE == COMBINE_FINISH_ALL) {
    return slot_at<SLOTS, 0>();
  } else {
    return -1;
  }
}

// a non-polled activation: a producer's output buffer in this launch's arena copy (a reset buffer has two; the tensor's map
// is copy 0: MI355X dec32 / dec33, KDA decode T = 32: o_proj read y_rec's other copy every other launch - 0xFF at launch 0,
// the last launch's y after - the server's bs 32 / 64 garbage), else (a graph input) the tensor's map
template <class ACT, int AMAP>
__device__ __forceinline__ __hip_bfloat16 const *gemm_counted_act(G const &g, Maps const &maps) {
  if constexpr (ACT::x > 0 && ACT::buf >= 0) {
    return buf_at<__hip_bfloat16 const>(g, ACT::buf);
  } else {
    return reinterpret_cast<__hip_bfloat16 const *>(maps.m[AMAP].p);
  }
}

// an optional param (0 when the node has fewer)
template <class PARAMS, int I>
constexpr int gemm_param_or0() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return 0;
  }
}
// the norm's gamma (BUF_SLOTS[1]) when the RMSNorm is folded in, else -1
template <bool NORM, class SLOTS>
constexpr int gemm_gamma_slot() {
  if constexpr (NORM) {
    return slot_at<SLOTS, 1>();
  } else {
    return -1;
  }
}

// STATIC_EARLY_L2 (core.cuh early_l2): task (x, y)'s weight lines - its RT row tiles' K part y (shuffled layout: per 16-row
// tile K / 32 pieces of 1 KB); none for a row-major weight
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class ACT, class... IN>
__device__ __forceinline__ L2Span gemm_tile_l2span(Maps const &maps, int x, int y) {
  constexpr int K = PARAMS::v[1], N = PARAMS::v[2], WMAP = slot_at<MAP_SLOTS, 0>();
  constexpr int KP = K / SELF::y, RT = N / SELF::x / 16;
  if constexpr (param_at<PARAMS, 4>() != GEMM_SHUFFLED) {
    return L2Span{nullptr, 0, 0, 0u};
  } else {
    char const *const w = reinterpret_cast<char const *>(maps.m[WMAP].p);
    return L2Span{w + ((size_t)x * RT * (K / 32) + (size_t)y * (KP / 32)) * 1024, RT, KP / 32 * 8, (unsigned)(K / 32 * 1024)};
  }
}

// GEMM tile (x, y): weight rows [ROWS x, ROWS x + ROWS) times K part y of
// SELF::y -> the node's output buffer. PARAMS: {mode (Combine), K, N (the
// weight's rows), polled, the weight's layout (GemmLayout)}. MAP_SLOTS: {the weight's map, the activation's map
// (-1: polled)}: on gfx950 a map is the tensor's address (host.cuh). ACT: the
// node that writes the activation (none: a graph input).
#ifdef STATIC_DRD_HOLD
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_gemm_tile(); // (below; run_gemm_tile's STATIC_DRD_HOLD check)
#endif
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ACT,
          class... IN>
__device__ __forceinline__ void run_gemm_tile(Maps const &maps,
                                              G const &g,
                                              KernelLocals &L,
                                              StaticTask const &tk) {
  constexpr int mode = PARAMS::v[0], K = PARAMS::v[1], N = PARAMS::v[2],
                WMAP = slot_at<MAP_SLOTS, 0>(), AMAP = slot_at<MAP_SLOTS, 1>();
  constexpr bool POLLED = param_at<PARAMS, 3>() == 1;
  constexpr int LAYOUT = param_at<PARAMS, 4>();
  // params[5] 1: the RMSNorm folded in (gemm_norm_fill; ACT: the allreduce_send
  // of R, BUF_SLOTS[1] gamma), params[6] its eps (fp32 bits)
  constexpr bool NORM = gemm_param_or0<PARAMS, 5>() == 1;
  constexpr int EPS_BITS = gemm_param_or0<PARAMS, 6>();
  static_assert(!NORM || (mode == COMBINE_FINISH && POLLED &&
                          gemm_gamma_slot<NORM, BUF_SLOTS>() >= 0 &&
                          exchange_bytes[ACT::buf] == (size_t)GPUS * T * K * 2),
                "norm: finish, polled from the [GPUS][T][K] bf16 rows of an "
                "allreduce_send, gamma in buffer slot 1, no in-kernel re-arm");
  static_assert(LAYOUT == GEMM_ROWMAJOR || LAYOUT == GEMM_SHUFFLED,
                "params[4]: the weight's layout (GemmLayout)");
  static_assert(K > 0 && K % (128 * SELF::y) == 0,
                "params = {mode, K, N, polled}; each K part whole 128-column "
                "tiles");
  constexpr int KP = K / SELF::y; // the task's K columns
  constexpr int ROWS = N / SELF::x;
  static_assert(ROWS * SELF::x == N &&
                    (ROWS == 128 || ROWS == 64 || ROWS == 32 || ROWS == 16 || (gemm_param_or0<PARAMS, 7>() == 4 && ROWS == 256)),
                "128, 64, 32 or 16 weight rows per task (hybrid 4: 256)");
  static_assert(mode == COMBINE_SLOTS || mode == COMBINE_ADD ||
                    mode == COMBINE_STORE || mode == COMBINE_PUSH ||
                    mode == COMBINE_FINISH || mode == COMBINE_FINISH_Q ||
                    mode == COMBINE_PUSH_ALL || mode == COMBINE_LOCAL || mode == COMBINE_PUSH_RS ||
                    mode == COMBINE_FINISH_RS || mode == COMBINE_FINISH_ALL,
                "mode: Combine");
  static_assert(mode != COMBINE_FINISH_RS ||
                    (!POLLED && gemm_parts_slot<mode, BUF_SLOTS>() >= 0 && SELF::counter >= 0 && ROWS >= 64 &&
                     exchange_bytes[SELF::buf] == (size_t)GPUS * T * (N / GPUS) * 2 && (N / GPUS) % ROWS == 0),
                "finish_rs: a counted input, the K parts in buffer slot 0, a counter per row block (64 rows or more), the "
                "output [GPUS][N / GPUS / ROWS][T][ROWS] bf16 in the exchange region"); 
  static_assert(mode != COMBINE_FINISH_Q ||
                    (!POLLED && gemm_parts_slot<mode, BUF_SLOTS>() >= 0 &&
                     SELF::counter >= 0 &&
                     exchange_bytes[SELF::buf] == zq_bytes(N)),
                "finish_quant: K parts in buffer slot 0, a counter per 128-row "
                "block, z_q (and its scale chunks) in the exchange region");
  static_assert((mode != COMBINE_PUSH && mode != COMBINE_PUSH_ALL &&
                 mode != COMBINE_LOCAL && mode != COMBINE_PUSH_RS) ||
                    SELF::y == 1 || mode == COMBINE_PUSH_ALL,
                "push: one K part (push_all: K parts each pushed to its own slot, the readers add them; store: K parts at "
                "[part][T][N], allreduce_send params[2] 3 adds them)");
  static_assert(mode != COMBINE_PUSH_RS || (exchange_bytes[SELF::buf] == (size_t)GPUS * T * (N / GPUS) * 2 &&
                                            (N / GPUS) % ROWS == 0),
                "push_rs: the output [GPUS][N / GPUS / ROWS][T][ROWS] bf16 in the exchange region, whole row tiles per GPU");
  static_assert(mode != COMBINE_FINISH_ALL || (POLLED && gemm_parts_slot<mode, BUF_SLOTS>() >= 0 && SELF::counter >= 0 &&
                                               exchange_bytes[SELF::buf] == (size_t)GPUS * T * N * 2),
                "finish_all: a polled input, the K parts in buffer slot 0, a counter per row block, the output [GPUS][T][N] bf16 "
                "in the exchange region");
  static_assert(mode != COMBINE_PUSH_ALL || exchange_bytes[SELF::buf] == (size_t)GPUS * SELF::y * T * N * 2,
                "push_all: the output [GPUS][K parts][T][N] bf16 in the exchange region (its input polled or counted)");
  static_assert(mode != COMBINE_PUSH ||
                    (POLLED && exchange_bytes[SELF::buf] ==
                                   (size_t)GPUS * T * (N / GPUS) * 2),
                "push: a polled input, the output [GPUS][T][N / GPUS] bf16 in "
                "the exchange region");
  static_assert(mode != COMBINE_FINISH ||
                    (POLLED && gemm_parts_slot<mode, BUF_SLOTS>() >= 0 &&
                     SELF::counter >= 0 &&
                     exchange_bytes[SELF::buf] ==
                         (size_t)GPUS * T * (N / GPUS) * 2),
                "finish: a polled input, the K parts in buffer slot 0, a "
                "counter per row block, the output [GPUS][T][N / GPUS] bf16 in "
                "the exchange region");
  static_assert(SELF::buf >= 0 && (mode == COMBINE_SLOTS || mode == COMBINE_PUSH ||
                                   mode == COMBINE_PUSH_ALL || mode == COMBINE_LOCAL || mode == COMBINE_PUSH_RS ||
                                   SELF::counter >= 0),
                "the node's output buffer and counters");
  static_assert(gemm_act_bytes<gemm_act_kpc<ROWS, KP>()>() + 16 + GEMM_RED_BYTES <= 160 * 1024 - 1024,
                "the activation slice (chunk) and the K parts' sums in LDS");
  __hip_bfloat16 const *weight =
      reinterpret_cast<__hip_bfloat16 const *>(maps.m[WMAP].p);
  uint32_t *const cnt = cnt_of(g) + (SELF::counter < 0 ? 0 : SELF::counter);
  if constexpr (POLLED && gemm_param_or0<PARAMS, 8>() > 0) { // (gemm_tile_layer skip_seq: every token <= K pools - no task works)
    constexpr int SK = gemm_param_or0<PARAMS, 8>();
    static_assert(mode == COMBINE_LOCAL, "skip_seq: a local gemm");
    int const *const sl = buf_at<int const>(g, slot_at<BUF_SLOTS, 0>());
    bool const need = (int)STATIC_TIDX < T && sl[STATIC_TIDX] / 4 > SK;
    if (!__syncthreads_or(need)) {
      return;
    }
  }
  if constexpr (POLLED) {
    static_assert(AMAP < 0 && ACT::buf >= 0,
                  "polled: no activation map, ACT's buffer");
    // the output: this GPU's rows only, [K parts][T][N / GPUS] (PUSH_ALL: all
    // rows, [GPUS][T][N])
    constexpr bool ALL = mode == COMBINE_PUSH_ALL || mode == COMBINE_LOCAL || mode == COMBINE_PUSH_RS;
    constexpr bool ALLR = ALL || mode == COMBINE_STORE || mode == COMBINE_FINISH_ALL; // (all rows: a polled store gemm writes [T][N] fp32, its OOFF stays 0)
    gemm_tile_task<ROWS, KP, K, true, LAYOUT,
                   mode == COMBINE_PUSH || mode == COMBINE_FINISH || mode == COMBINE_FINISH_ALL || ALL
                       ? exchange_offset[SELF::buf]
                       : 0,
                   SELF::y, NORM, NORM ? exchange_offset[ACT::buf] : 0,
                   EPS_BITS>(g,
                                      weight,
                                      buf_at<__hip_bfloat16 const>(g, ACT::buf),
                                      nullptr,
                                      0,
                                      mode,
                                      buf_at<void>(g, mode == COMBINE_FINISH || mode == COMBINE_FINISH_ALL
                                                          ? gemm_parts_slot<mode, BUF_SLOTS>()
                                                          : SELF::buf),
                                      ALLR ? N : N / GPUS,
                                      cnt,
                                      tk.x * ROWS,
                                      ALLR ? tk.x * ROWS : tk.x * ROWS - g.rank * (N / GPUS),
                                      tk.y,
                                      tk.y * KP,
                                      NORM ? buf_at<__hip_bfloat16 const>(
                                                 g, gemm_gamma_slot<NORM, BUF_SLOTS>())
                                           : nullptr,
                                      L.sm STAGE_STAMP_ARG(L.stage_stamps));
  } else {
    uint32_t const *act_cnt = nullptr;
    uint32_t need = 0;
    if constexpr (ACT::x > 0) {
      static_assert(ACT::counter >= 0,
                    "the activation's producer counts its tasks");
      act_cnt = cnt_of(g) + ACT::counter;
      need = ACT::x * ACT::y * ACT::z;
    }
    gemm_tile_task<ROWS, KP, K, false, LAYOUT,
                   mode == COMBINE_FINISH_Q || mode == COMBINE_PUSH_ALL || mode == COMBINE_PUSH_RS || mode == COMBINE_FINISH_RS
                       ? exchange_offset[SELF::buf]
                       : 0,
                   SELF::y, false, 0, 0, mode == COMBINE_FINISH_Q ? N : 0, (ACT::x > 0),
                   false,
                   gemm_param_or0<PARAMS, 7>() == 1, // (gemm_tile_layer hybrid: gemm_wave HYB)
                   gemm_param_or0<PARAMS, 7>() == 4 // (hybrid=4: gemm_wave HYB4)
                   >(
        g,
        weight,
        gemm_counted_act<ACT, AMAP>(g, maps),
        act_cnt,
        need,
        mode,
        buf_at<void>(g, mode == COMBINE_FINISH_Q || mode == COMBINE_FINISH_RS ? gemm_parts_slot<mode, BUF_SLOTS>()
                                                                              : SELF::buf),
        N,
        cnt,
        tk.x * ROWS,
        tk.x * ROWS,
        tk.y,
        tk.y * KP,
        nullptr,
        L.sm STAGE_STAMP_ARG(L.stage_stamps)
#ifdef STATIC_GEMM_L2WARM
        , gemm_l2warm_next(L, tk.node)
#elif defined(STATIC_GEMM_ACT_KEEP)
        , make_int2(-1, 0)
#endif
#ifdef STATIC_GEMM_ACT_KEEP
        , L.keep_node == tk.node && L.keep_y == tk.y
#endif
        );
  }
#ifdef STATIC_DRD_HOLD
  // (a held recurrence state: stored after this task's own loads, pushes and counts; only a node whose LDS stays below the held
  // state can be the hold node)
  if constexpr (smem_gemm_tile<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS>() <= STATIC_DRD_HOLD_OFF) {
    if (tk.node == STATIC_DRD_HOLD) {
      drd_hold_flush(L);
    }
  } else {
    if (tk.node == STATIC_DRD_HOLD) {
      __builtin_trap(); // (STATIC_DRD_HOLD names a GEMM whose LDS overlaps the held state)
    }
  }
#endif
}


// dynamic LDS: the activation slice, then the K parts' sums (ROWS 32 / 16)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gemm_tile() {
  constexpr int KP = PARAMS::v[1] / SELF::y, ROWS = PARAMS::v[2] / SELF::x;
  if constexpr (gemm_param_or0<PARAMS, 7>() == 1 || gemm_param_or0<PARAMS, 7>() == 4) { // (HYB: half the rows' weights in
    // LDS, at T = 16 all but 1-2 K steps; HYB4: a quarter of a 256-row task's)
    static_assert(ROWS == (gemm_param_or0<PARAMS, 7>() == 4 ? 256 : 128), "HYB: 128-row tasks (HYB4 256), a row tile per wave in LDS");
    return gemm_w_lds_at<ROWS, KP>() + 4 * gemm_hyb_lds_steps<KP / 32>(gemm_w_lds_at<ROWS, KP>()) * 1024;
  }
  return gemm_ldsw<ROWS, KP, param_at<PARAMS, 3>() == 1>()
             ? gemm_w_lds_at<ROWS, KP>() + ROWS * KP * 2
             : ((gemm_act_bytes<gemm_act_kpc<ROWS, KP>()>() + 15) & ~15) + GEMM_RED_BYTES;
}

} // namespace static_mk
