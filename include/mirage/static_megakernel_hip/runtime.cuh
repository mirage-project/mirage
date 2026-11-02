// runtime.cuh (HIP, gfx950) -- what every task body of the static megakernel
// uses on AMD GPUs; the same names as include/mirage/static_megakernel/
// runtime.cuh where the meaning is the same:
//   1. counters and polls in global memory, the exchange store, the clock
//   2. math: SiTU, the e4m3 conversion, fixed-point sums
// Not here yet: the GEMM ring (CUDA: TMA + tcgen05 + mbarriers); on gfx950 a
// task's GEMM streams weights through LDS (global_load_lds) into scaled MFMAs,
// written with moe_experts.
//
// Memory model (amd/notes/cdna4_isa_facts.md, CDNA4 ISA guide p.84-86): each of
// the 8 XCDs (dies) has its own L2. A load or store with sc1 (device scope)
// bypasses the L2; plain ones stay in the die's L2, which other dies do not see
// until the writer's die writes it back (buffer_wbl2 sc1, the whole L2's dirty
// lines) and the reader's die drops its stale lines (buffer_inv sc1). Two
// hand-off protocols, checked by amd/bench/handoff_stress.hip:
//   release / acquire   plain stores, cnt_add (store-release: buffer_wbl2) ->
//                       cnt_wait (sc1 polls, buffer_inv), plain loads
//   bypass              st16_dev stores (sc1), cnt_add_bypass (wait for the
//                       stores, then the add; no write-back) -> cnt_wait_bypass
//                       (sc1 polls, no invalidate), ld16_dev loads (sc1)
#pragma once
#include "config.cuh"
#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace static_mk {

typedef unsigned v4u __attribute__((ext_vector_type(4))); // asm operands
typedef unsigned v2u __attribute__((ext_vector_type(2)));
// MFMA operand / accumulator types
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

// ---------------- 1. global memory ----------------
// 16-B and 4-B loads / stores at device scope (sc1: past the die's L2)
// A store of more than 8 bytes: its data VGPRs must not be written by a VALU
// right after it (gfx950; the failing case below had 1 wait state; s_nop 1 gives 2;
// the count is inferred, not from the ISA text; the compiler inserts them for its own
// stores, not after inline asm): the s_nop 1 below. MI355X 10-14: without it,
// attn_residual's u_out store (16 B) was followed by a v_pk_fma writing its
// data VGPRs one SALU later -> u_out held the next values (non-finite u).
__device__ __forceinline__ void st16_dev(void *p, uint4 v) {
  v4u const w = {v.x, v.y, v.z, v.w};
  asm volatile("global_store_dwordx4 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w)
               : "memory");
}
// st16_dev of registers an MFMA wrote (its accumulators, no VALU op in between): the compiler puts no wait states before
// an inline asm that reads an MFMA's result, so the store could read it before the MFMA wrote it (MI355X dec26 - dec28:
// gemm_tile's K parts / S rows of 4 - 8 tokens of a 16-token tile off, only in some builds and launches, the activation
// in LDS right; the store 1 wait state after the MFMA). 24 wait states first (a 16-pass MFMA needs ~19)
__device__ __forceinline__ void st16_dev_acc(void *p, uint4 v) {
  v4u const w = {v.x, v.y, v.z, v.w};
  asm volatile("s_nop 7\n s_nop 7\n s_nop 7\n global_store_dwordx4 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w)
               : "memory");
}
__device__ __forceinline__ uint4 ld16_dev(void const *p) {
  v4u w;
  asm volatile("global_load_dwordx4 %0, %1, off sc1\n s_waitcnt vmcnt(0)"
               : "=v"(w)
               : "v"(p)
               : "memory");
  return make_uint4(w.x, w.y, w.z, w.w);
}
// at system scope (sc0 sc1): another GPU's memory (the exchange region).
__device__ __forceinline__ void st16_sys(void *p, uint4 v) {
  v4u const w = {v.x, v.y, v.z, v.w};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n s_nop 1" ::"v"(p), // (s_nop: st16_dev)
               "v"(w)
               : "memory");
}
__device__ __forceinline__ void st8_sys(void *p, uint2 v) {
  v2u const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1" ::"v"(p),
               "v"(w)
               : "memory");
}
__device__ __forceinline__ uint4 ld16_sys(void const *p) {
  v4u w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(w)
               : "v"(p)
               : "memory");
  return make_uint4(w.x, w.y, w.z, w.w);
}
// 1 byte at device scope
__device__ __forceinline__ void st8_dev(void *p, uint8_t v) {
  uint32_t const w = v;
  asm volatile("global_store_byte %0, %1, off sc1" ::"v"(p), "v"(w) : "memory");
}
// Loads the compiler tracks (it waits only as far as a value's use needs, so
// many can be in flight): a buffer load from a WAVE-UNIFORM base plus a
// per-lane byte offset (a per-lane base makes the compiler loop over the
// lanes), with the cache-policy bits AUX: 0 plain, 16 sc1 (device scope), 17
// sc0 sc1 (system scope). The asm loads above wait inside the asm (the
// compiler cannot track them): fine for a poll of one address, not for many
// loads at once. 0x00020000: the descriptor's last dword for gfx9 (as AMD's
// composable_kernel); 0x7FFFFFFF bytes from the base are in range.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc(void const *base) {
  return __builtin_amdgcn_make_buffer_rsrc(
      const_cast<void *>(base), 0, 0x7FFFFFFF, 0x00020000);
}
template <int AUX>
__device__ __forceinline__ uint4 ld16_at(void const *base, uint32_t off) {
  v4u const w =
      __builtin_amdgcn_raw_buffer_load_b128(buf_rsrc(base), off, 0, AUX);
  return make_uint4(w.x, w.y, w.z, w.w);
}
template <int AUX>
__device__ __forceinline__ uint8_t ld8_at(void const *base, uint32_t off) {
  return __builtin_amdgcn_raw_buffer_load_b8(buf_rsrc(base), off, 0, AUX);
}
constexpr int AUX_DEV = 16, AUX_SYS = 17;
__device__ __forceinline__ float4 as_float4(uint4 const &u) {
  return make_float4(__uint_as_float(u.x),
                     __uint_as_float(u.y),
                     __uint_as_float(u.z),
                     __uint_as_float(u.w));
}

// wait until this thread's stores (and loads) have completed
__device__ __forceinline__ void wait_mem() {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// counters (u32). The add of the bypass protocol: the caller's data went out with st16_dev
// (sc1); wait for those stores, then a relaxed add (no write-back)
__device__ __forceinline__ void cnt_add_bypass(uint32_t *c, uint32_t v) {
  wait_mem();
  __hip_atomic_fetch_add(c, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint32_t cnt_ld(uint32_t const *c) {
  return __hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
// the wait of the bypass protocol: the caller reads the data with ld16_dev /
// ld16_in_at. The relaxed loads of the count order nothing: the memory clobber
// keeps the compiler from issuing the data loads before the wait (MI355X 10-10:
// moe_experts' W2 read h_q before its W13 items wrote it)
__device__ __forceinline__ void cnt_wait_bypass(uint32_t const *c,
                                                uint32_t target) {
  while (cnt_ld(c) < target) {
    __builtin_amdgcn_s_sleep(1);
  }
  asm volatile("" ::: "memory");
}

// sums several CUs add into: int64 fixed point, scale 2^40 (the same sum in any
// order, so the result is the same every launch)
__device__ __forceinline__ void red_add_fx(unsigned long long *p, float v) {
  long long const q = __float2ll_rn(v * 0x1p40f);
  __hip_atomic_fetch_add(
      p, (unsigned long long)q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ float fx_to_float(unsigned long long q) {
  return __ll2float_rn((long long)q) * 0x1p-40f;
}

// the die (XCD, 0..7) this wave runs on
__device__ __forceinline__ int xcc_id() {
  unsigned v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(v));
  return (int)(v & 7u);
}

// the clock in ns: s_memrealtime (100 MHz on gfx950, host.cuh checks the rate)
__device__ __forceinline__ long long gtime() {
  return (long long)(wall_clock64() * NS_PER_TICK);
}

// Between two rounds of a poll of tracked loads (ld16_at): a pause the compiler
// can neither remove nor move a load across. A loop whose only effect is plain
// loads of the same addresses is one the compiler may assume ends (C++ forward
// progress): it reads once and drops the check. MI355X 10-10: topk_route summed
// K parts that were still 0xFF (its loop had no sleep).
__device__ __forceinline__ void poll_pause() {
  __builtin_amdgcn_s_sleep(1);
  asm volatile("" ::: "memory");
}
// the pause of the tasks after moe_experts that poll every GPU's slots (R, S,
// o): 64 cycles (as poll_pause). In TP8 they start on the
// CUs done with moe_experts and poll 8 slots per vector while the other CUs
// still read W2 (MI355X 10-12: W2 22.5 us in TP8, 16.9 solo)
__device__ __forceinline__ void tail_pause() {
  __builtin_amdgcn_s_sleep(1);
  asm volatile("" ::: "memory");
}

// Hand-off by the data itself: a buffer is filled with 0xFF bytes before the
// launch; 0xFFFFFFFF never occurs in the data (e4m3 values come from
// cvt_e4m3's clamp, never 0xFF; UE8M0 scales are <= 254). A 16-B store is not
// known to be seen whole by another die, so a reader checks every 4-B word
// (valid16w) and re-loads until none is all-ones.
__device__ __forceinline__ bool valid16w(uint4 const &v) {
  return v.x != 0xFFFFFFFFu && v.y != 0xFFFFFFFFu && v.z != 0xFFFFFFFFu &&
         v.w != 0xFFFFFFFFu;
}
__device__ __forceinline__ uint4 poll16(void const *p) {
  for (;;) {
    uint4 const v = ld16_dev(p);
    if (valid16w(v)) {
      return v;
    }
    __builtin_amdgcn_s_sleep(1);
  }
}
// the same poll for another GPU's stores (the exchange region): system scope
__device__ __forceinline__ uint4 poll16_sys(void const *p) {
  for (;;) {
    uint4 const v = ld16_sys(p);
    if (valid16w(v)) {
      return v;
    }
    __builtin_amdgcn_s_sleep(1);
  }
}
__device__ __forceinline__ bool validf(float v) {
  return __float_as_uint(v) != 0xFFFFFFFFu;
}
__device__ __forceinline__ bool validf4(float4 const &v) {
  return validf(v.x) && validf(v.y) && validf(v.z) && validf(v.w);
}
// 4 fp32 at device scope (a K part another CU of this GPU wrote with st16_dev)
__device__ __forceinline__ float4 ldf4_dev(float const *p) {
  uint4 const u = ld16_dev(p);
  return make_float4(__uint_as_float(u.x),
                     __uint_as_float(u.y),
                     __uint_as_float(u.z),
                     __uint_as_float(u.w));
}

// The protocol every producer / consumer pair of the layer uses (the bypass protocol, decided by
// amd/bench/handoff_stress.hip): a task's outputs other CUs read go out with st16_out (st16_dev), its counter with
// cnt_post (cnt_add_bypass); a reader waits with cnt_wait_in (cnt_wait_bypass) and loads with ld16_in (sc1).
__device__ __forceinline__ void st16_out(void *p, uint4 v) {
  st16_dev(p, v);
}
__device__ __forceinline__ void st8_out(void *p, uint8_t v) {
  st8_dev(p, v);
}
// the protocol's tracked loads: base wave-uniform, off per lane (bytes)
__device__ __forceinline__ uint4 ld16_in_at(void const *base, uint32_t off) {
  return ld16_at<AUX_DEV>(base, off);
}
__device__ __forceinline__ uint8_t ld8_in_at(void const *base, uint32_t off) {
  return ld8_at<AUX_DEV>(base, off);
}
__device__ __forceinline__ void cnt_post(uint32_t *c, uint32_t v) {
  cnt_add_bypass(c, v);
}
__device__ __forceinline__ void cnt_wait_in(uint32_t const *c,
                                            uint32_t target) {
  cnt_wait_bypass(c, target);
}

// a task's end: every thread's outputs (st16_out) are complete, then thread 0
// adds v to the task's counter. Call from all threads. (With the bypass
// protocol a wave's s_waitcnt covers only its own stores, so each thread waits
// for its own before the barrier.)
// a wavefront's end of a piece of work (no block barrier): its outputs are
// complete, then lane 0 adds v to c. Call from all lanes of the wave.
__device__ __forceinline__ void wave_post(uint32_t *c, uint32_t v) {
  wait_mem();
  if ((threadIdx.x & 63) == 0) {
    cnt_post(c, v);
  }
}
__device__ __forceinline__ void task_post(uint32_t *c, uint32_t v) {
  wait_mem();
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_post(c, v);
  }
}
// 8 bytes a CU of this GPU added with atomics (red_add_fx), at device scope
__device__ __forceinline__ unsigned long long ld64_dev(void const *p) {
  return __hip_atomic_load(reinterpret_cast<unsigned long long const *>(p),
                           __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
}

// the timing build's stage stamps (as on CUDA): a task function takes
// STAGE_STAMP_PARAM, writes STAGE_STAMP / STAGE_STAMP_AT, its caller passes
// STAGE_STAMP_ARG(L.stage_stamps)
// STATIC_STAMP_SET: which places a task's 2 stage stamps mark, where a task
// has several sets (the KDA recurrence, attn_residual)
#ifndef STATIC_STAMP_SET
#define STATIC_STAMP_SET 0
#endif
#ifdef STATIC_TIMING_BUILD
#define STAGE_STAMP_PARAM , unsigned long long *stamp = nullptr
#define STAGE_STAMP(cond)                                                      \
  if (stamp && (cond))                                                         \
    *stamp = gtime();
#define STAGE_STAMP_AT(i)                                                      \
  if (stamp)                                                                   \
    stamp[i] = gtime();
#define STAGE_STAMP_ARG(p) , (p)
#else
#define STAGE_STAMP_PARAM
#define STAGE_STAMP(cond)
#define STAGE_STAMP_AT(i)
#define STAGE_STAMP_ARG(p)
#endif

// STATIC_TID_ASM 1 (default 0): a task's thread index through a volatile v_mov, so the compiler cannot hoist the per-lane values
// made from it (LDS addresses, lane predicates) out of the kernel's task loop and keep them live across every task (MI355X
// 10-24, fused KDA T 16: 7 such VGPRs spilled to scratch, 32 B per lane, reloaded in attn_residual and delta_rule_decode; a
// launch with scratch costs ~1.4-1.8 us more in a HIP graph, amd/bench/launch_cost.hip)
#ifndef STATIC_TID_ASM
#define STATIC_TID_ASM 0
#endif
// STATIC_TID_ASM 3: also every other read of the thread index in the task files (STATIC_TIDX, unsigned as threadIdx.x)
#if STATIC_TID_ASM >= 3
#define STATIC_TIDX ((unsigned)task_tid_all())
#else
#define STATIC_TIDX threadIdx.x
#endif
// STATIC_TID_ASM 2: also every other task type's thread index (task_tid_all)
__device__ __forceinline__ int task_tid_all() {
#if STATIC_TID_ASM >= 2
  int t;
  asm volatile("v_mov_b32 %0, %1" : "=v"(t) : "v"((int)threadIdx.x));
  return t;
#else
  return (int)threadIdx.x;
#endif
}
__device__ __forceinline__ int task_tid() {
#if STATIC_TID_ASM
  int t;
  asm volatile("v_mov_b32 %0, %1" : "=v"(t) : "v"((int)threadIdx.x));
  return t;
#else
  return (int)threadIdx.x;
#endif
}

// ---------------- 2. math ----------------
// 2 fp32 -> 2 bf16 (round to nearest even) in one word, a in the low half
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  return (uint32_t)__builtin_bit_cast(uint16_t, __float2bfloat16(a)) |
         ((uint32_t)__builtin_bit_cast(uint16_t, __float2bfloat16(b)) << 16);
}
__device__ __forceinline__ float bf16_lo(uint32_t w) {
  return __uint_as_float(w << 16);
}
__device__ __forceinline__ float bf16_hi(uint32_t w) {
  return __uint_as_float(w & 0xFFFF0000u);
}
// fp32 -> bf16 -> fp32 (the rounding a bf16 store applies)
__device__ __forceinline__ float round_bf16(float a) {
  return bf16_lo(pack_bf16x2(a, 0.0f));
}
// SiTU of gate g and up u: 4 tanh(g / 4) sigmoid(g) x 25 tanh(u / 25)
__device__ __forceinline__ float situ(float g, float u) {
  return 4.0f * tanhf(g * 0.25f) * (1.0f / (1.0f + __expf(-g))) *
         (25.0f * tanhf(u * 0.04f));
}
// ceil(log2(x)) of a positive normal x from its bits
__device__ __forceinline__ int ceil_log2_bits(float x) {
  uint32_t const b = __float_as_uint(x);
  return (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
}
// fp32 -> OCP e4m3 (gfx950's v_cvt_pk_fp8_f32 is OCP e4m3; gfx942's was the
// FNUZ variant). The instruction takes no clamp, so |v| is clamped to 448 (the
// largest e4m3) first: the same as CUDA's satfinite for finite v.
__device__ __forceinline__ uint8_t cvt_e4m3(float v) {
  float const c = __builtin_amdgcn_fmed3f(v, -448.0f, 448.0f);
  return (uint8_t)(__builtin_amdgcn_cvt_pk_fp8_f32(c, 0.0f, 0, false) & 0xFF);
}

} // namespace static_mk
