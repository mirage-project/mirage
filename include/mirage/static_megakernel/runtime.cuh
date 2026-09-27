// runtime.cuh -- what every task body of the static megakernel uses:
//   1. PTX helpers: mbarrier, TMA, tcgen05 (MMA, TMEM), polls and counters in global memory, the multicast store
//   2. the per-SM state: barrier addresses, shared tables, the CTA prologue and TMEM allocation
//   3. the ring that streams one GEMM tile through shared memory: load_job (loader warp) -> issue_job (two issuer warps) ->
//      drain_acc (four epilogue warps)
// One CTA of 256 threads per SM. Warp roles: 0 loader, 1 and 6 issuers, 2..5 epilogue, 7 second loader (expert queue).
//
// Parity rule (proven by deadlocks and illegal-instruction faults, 2026-09-10): a thread may wait on an mbarrier by parity only if
// the barrier cannot be two phases ahead of the phase it waits for. So a loader waits on `empty` only for its OWN stages, and each
// issuer waits on `full` for EVERY stage in order.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "config.cuh"

namespace static_mk {

// ---------------- 1. PTX helpers ----------------
__device__ __forceinline__ uint32_t su32(void const *p) { return (uint32_t)__cvta_generic_to_shared(p); }

// mbarrier
__device__ __forceinline__ void mbar_init(uint32_t a, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int ph) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@P bra D;\nbra W;\nD:\n}"
               ::"r"(a), "r"(ph) : "memory");
}

// TMA: a 3-D / 2-D box from global memory into shared memory, completing on mbarrier mb
__device__ __forceinline__ void tma3(void const *d, uint32_t mb, uint32_t sm, int c0, int c1, int c2, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               ::"r"(sm), "l"((uint64_t)d), "r"(mb), "r"(c0), "r"(c1), "r"(c2), "l"(hint) : "memory");
}
__device__ __forceinline__ void tma2(void const *d, uint32_t mb, uint32_t sm, int c0, int c1, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               ::"r"(sm), "l"((uint64_t)d), "r"(mb), "r"(c0), "r"(c1), "l"(hint) : "memory");
}

// tcgen05: shared-memory matrix descriptors, MMA, commit, fences, TMEM load
__device__ __forceinline__ uint64_t denc(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }
__device__ __forceinline__ uint64_t mkdesc(uint32_t a) {      // 128-B swizzled K-major tile, 1024-B stride
  return denc(a) | (denc(1024) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}
__device__ __forceinline__ uint64_t mkdesc_sf(uint32_t a) { return denc(a) | (denc(128) << 32ULL) | (1ULL << 46ULL); }   // scale chunk
__device__ __forceinline__ uint32_t idesc_mx(uint32_t sf) { return IDESC_MX | (sf << 4) | (sf << 29); }   // scale id sf of 0..3
__device__ __forceinline__ void cp_sf(uint32_t tcol, uint32_t smem) {   // one 512-B scale chunk shared memory -> TMEM
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tcol), "l"(mkdesc_sf(smem)) : "memory");
}
__device__ __forceinline__ void mma_bf16(uint32_t d, uint64_t a, uint64_t b, uint32_t en) {   // en = 0: overwrite d, 1: add
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(a), "l"(b), "r"(IDESC_BF16), "r"(en));
}
__device__ __forceinline__ void mma_mx(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t en, uint32_t sfa, uint32_t sfb) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(en), "r"(sfa), "r"(sfb));
}
__device__ __forceinline__ void tc_commit(uint32_t mb) {   // mb gets one arrival when this thread's MMAs so far are done
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mb) : "memory");
}
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_ld8(uint32_t addr, float *v) {   // 8 fp32 columns of this thread's TMEM lane
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// math
__device__ __forceinline__ float situ(float g, float u) {   // K3's shared / expert activation of gate g and up u
  return 4.0f * tanhf(g * 0.25f) * (1.0f / (1.0f + __expf(-g))) * (25.0f * tanhf(u * 0.04f));
}
__device__ __forceinline__ uint8_t cvt_e4m3(float v) {
  uint16_t p;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p) : "f"(0.0f), "f"(v));
  return (uint8_t)(p & 0xFF);
}

// counters in global memory
__device__ __forceinline__ void cnt_add(uint32_t *c, uint32_t v) {   // release: this thread's writes before are visible to a cnt_wait
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(c), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t cnt_ld(uint32_t const *c) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c) : "memory");
  return v;
}
__device__ __forceinline__ void cnt_wait(uint32_t const *c, uint32_t target) { while (cnt_ld(c) < target) { __nanosleep(200); } }
__device__ __forceinline__ void red_add(float *p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// Hand-off by the data itself: a buffer is filled with 0xFF bytes before the launch; the value 0xFFFFFFFF never occurs in the data
// (e4m3 satfinite never gives 0xFF, UE8M0 scales are <= 254, arithmetic NaNs are 0x7F..), so a consumer re-loads until its words are
// not all-ones -- no flag, no fence.
__device__ __forceinline__ float4 ldf4_relaxed(float const *p) {
  float4 v;
  asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld16_relaxed(unsigned char const *p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
// the same loads without the memory clobber: still re-executed every iteration, but several can be in flight at once
// (with the clobber, 28 loads went one round trip at a time: 3.2 us)
__device__ __forceinline__ float4 ldf4_poll(float const *p) {
  float4 v;
  asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ld16_poll(unsigned char const *p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 poll16(unsigned char const *p) {   // re-load until the 16 bytes are not all-ones
  for (;;) {
    uint4 const v = ld16_relaxed(p);
    if ((v.x & v.y & v.z & v.w) != 0xFFFFFFFFu) return v;
    __nanosleep(64);
  }
}
__device__ __forceinline__ bool validf(float v) { return __float_as_uint(v) != 0xFFFFFFFFu; }   // fp32 written by one 4-B store
__device__ __forceinline__ bool validf4(float4 const &v) { return validf(v.x) && validf(v.y) && validf(v.z) && validf(v.w); }
__device__ __forceinline__ bool valid16(uint4 const &v) { return (v.x & v.y & v.z & v.w) != 0xFFFFFFFFu; }   // written by one 16-B store
__device__ __forceinline__ bool valid16w(uint4 const &v) {   // every 4-B word written
  return v.x != 0xFFFFFFFFu && v.y != 0xFFFFFFFFu && v.z != 0xFFFFFFFFu && v.w != 0xFFFFFFFFu;
}
__device__ __forceinline__ uint32_t has_ff(uint32_t w) { uint32_t const x = ~w; return (x - 0x01010101u) & ~x & 0x80808080u; }   // some byte is 0xFF

// store 16 bytes at offset `off` of the exchange region of EVERY GPU (multicast address mc), or of the local copy rv with one GPU
__device__ __forceinline__ void push16_mc(unsigned char *mc, unsigned char *rv, int tp, size_t off, uint4 v) {
  if (tp > 1)   // multimem.st exists only for .f32 vectors
    asm volatile("multimem.st.weak.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(mc + off), "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)), "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w))
                 : "memory");
  else
    *reinterpret_cast<uint4 *>(rv + off) = v;
}

// ---------------- 2. per-SM state ----------------
// shared-memory addresses of the mbarriers (each 8 B, in `bars`)
struct Rt {
  char *smp;             // generic pointer to the 1024-aligned dynamic shared memory
  uint32_t full0;        // [SMAX] ring stage loaded (count 1 + TMA bytes)
  uint32_t empty0;       // [SMAX] ring stage free again (count 2: both issuers)
  uint32_t acc_full0;    // [4] TMEM accumulator stage complete (count 2: both issuers commit)
  uint32_t acc_empty0;   // [4] accumulator stage drained (count 128: the epilogue threads)
  uint32_t misc0;        // [4] phase switch: z scale chunks loaded ([0]) and copied to TMEM ([2])
  uint32_t hq0;          // [MAXSEG] h_q segment: the loader's first load landed
  uint32_t sfb0;         // [MAXSEG] h_q segment scales copied to TMEM
  uint32_t qfull0;       // [W2QD] expert-queue entry published (issuers / epilogue copy)
  uint32_t qempty0;      // [W2QD] entry retired by the epilogue (count 128)
  uint32_t qfull7_0;     // [W2QD] entry published (warp 7's copy)
  uint32_t qempty7_0;    // [W2QD] warp 7 has read the entry
  uint32_t hqv0;         // [MAXSEG] h_q segment checked by warp 7
  uint32_t hqr0;         // [MAXSEG] warp 7's reload of the segment landed
};
__shared__ Rt rt;
__shared__ __align__(8) uint64_t bars[2 * SMAX + 8 + 4 + 4 * MAXSEG + 4 * W2QD];
__shared__ int w2q[W2QD][5];    // expert-queue entries (QueueEntry fields) for the issuers and the epilogue
__shared__ int w2q7[W2QD][5];   // the same entries, warp 7's copy
__shared__ uint32_t tmem_slot;
__shared__ int s_ints[64];      // [1] the number of live experts; [16..23] per-warp counts of build_table
// the routing table of this step, built on every SM by build_table
__shared__ int cntE_s[NE];                                   // tokens per expert
__shared__ int sel_ss[T * 16]; __shared__ float wsel_ss[T * 16];   // routing pairs (token t, k) -> expert, weight
__shared__ int eid_s[NSLOT], ntok_s[NSLOT], tok_s[NSLOT * T];      // per slot: expert id, token count, tokens
__shared__ float w_s[NSLOT * T];                                   // per slot and token: routing weight
__device__ __forceinline__ char *rt_sm() { return rt.smp; }

// barrier addresses and init. Call from all 256 threads.
__device__ __forceinline__ void cta_prologue(char *sm) {
  if (threadIdx.x == 0) {
    int b = 0;
    auto take = [&](int n) { uint32_t const a = su32(&bars[b]); b += n; return a; };
    rt.smp = sm;
    rt.full0 = take(SMAX); rt.empty0 = take(SMAX); rt.acc_full0 = take(4); rt.acc_empty0 = take(4); rt.misc0 = take(4);
    rt.hq0 = take(MAXSEG); rt.sfb0 = take(MAXSEG);
    rt.qfull0 = take(W2QD); rt.qempty0 = take(W2QD); rt.qfull7_0 = take(W2QD); rt.qempty7_0 = take(W2QD);
    rt.hqv0 = take(MAXSEG); rt.hqr0 = take(MAXSEG);
    for (int s = 0; s < SMAX; s++) { mbar_init(rt.full0 + 8 * s, 1); mbar_init(rt.empty0 + 8 * s, 2); }
    for (int s = 0; s < 4; s++) { mbar_init(rt.acc_full0 + 8 * s, 2); mbar_init(rt.acc_empty0 + 8 * s, 128); }
    for (int s = 0; s < 4; s++) mbar_init(rt.misc0 + 8 * s, 1);
    for (int s = 0; s < MAXSEG; s++) {
      mbar_init(rt.hq0 + 8 * s, 1); mbar_init(rt.sfb0 + 8 * s, 1); mbar_init(rt.hqv0 + 8 * s, 1); mbar_init(rt.hqr0 + 8 * s, 1);
    }
    for (int s = 0; s < W2QD; s++) {
      mbar_init(rt.qfull0 + 8 * s, 1); mbar_init(rt.qempty0 + 8 * s, 128); mbar_init(rt.qfull7_0 + 8 * s, 1); mbar_init(rt.qempty7_0 + 8 * s, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
}
__device__ __forceinline__ uint32_t tmem_alloc_512() {   // warp 1 allocates 512 columns; every thread reads the base after the __syncthreads
  if ((threadIdx.x >> 5) == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(&tmem_slot)), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  return tmem_slot;
}
__device__ __forceinline__ void tmem_dealloc_512(uint32_t tb) {
  if ((threadIdx.x >> 5) == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tb), "r"(512));
}
// fresh ring (and accumulator) barriers between parts whose stage counts differ (the phase switch, shared down, the tail);
// the caller resets its stage counters
__device__ __forceinline__ void ring_reinit(bool acc_too) {
  for (int st = 0; st < SMAX; st++) { mbar_init(rt.full0 + 8 * st, 1); mbar_init(rt.empty0 + 8 * st, 2); }
  if (acc_too)
    for (int s_ = 0; s_ < 4; s_++) { mbar_init(rt.acc_full0 + 8 * s_, 2); mbar_init(rt.acc_empty0 + 8 * s_, 128); }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// ---------------- 3. the ring ----------------
// A job = one output tile of 128 rows x 8 tokens, streamed through the ring: `nst` stages, each holding weights (+ activations),
// consumed by the two issuer warps into TMEM accumulator stage `a`, drained by the 128 epilogue threads (one row each).
//   kind 0  bf16 tile: stage = K 128 of the weight (box {64, 128 rows, 2}) + of the activation (box {64, 8, 2}); the issuers take
//           whole stages in turn
//   kind 1  W13 item: stage = two MXFP4 K 128 pieces + their scale chunks + two e4m3 z_q K tiles; each issuer takes one piece
//   kind 2  W2 chunk: stage = two MXFP4 K 128 pieces + scale chunks (h_q is resident in shared memory); issued by the expert queue
// Stage counters `g` count stages since the last ring_reinit: stage slot g % SMAX, phase g / SMAX.
struct TileJob {
  int kind;
  int nst;                    // ring stages of this job
  CUtensorMap const *wmap;    // kind 0: the weight; kind 1, 2: the MXFP4 weight, one 128 x 128 piece per box
  int row0, k0;               // kind 0: first weight row, first K 128 tile; kind 1: first z_q K tile
  CUtensorMap const *amap;    // kind 0: the activation (nullptr: the activation is already in the stage); kind 1: z_q, one K tile
  CUtensorMap const *amap2;   // kind 1: z_q, two K tiles per box
  CUtensorMap const *sfmap;   // kind 1, 2: the weight scales, one chunk per box
  int tile0, chunk0;          // kind 1, 2: index of the first weight piece / scale chunk
  int natoms;                 // kind 1, 2: weight pieces of the job
  CUtensorMap const *wmap2;   // kind 1, 2: the weight, two pieces per box
  CUtensorMap const *sfmap2;  // kind 1, 2: the scales, two chunks per box
};

// loader: arm stages [s0, min(nst, s1)) of the job. par >= 0: two loaders split the stages, this one arms those with s % 2 == par.
// The loop body is one wait + one expect + 2..3 TMA issues per stage (each extra instruction here costs ~40 ns per stage).
__device__ __forceinline__ void load_job(TileJob const &j, uint32_t base, int &g, int par = -1, int s0 = 0, int s1 = 1 << 20) {
  int st = g % SMAX; uint32_t ph = ((g / SMAX) - 1) & 1;   // stage slot and its `empty` parity, advanced as g advances
  int const send = j.nst < s1 ? j.nst : s1;
  for (int s = s0; s < send; s++, g++) {
    bool const mine = par < 0 || (s & 1) == par;
    if (!mine) { if (++st == SMAX) { st = 0; ph ^= 1u; } continue; }   // wait on `empty` only for own stages (parity rule)
    if (g >= SMAX) mbar_wait(rt.empty0 + 8 * st, ph);
    uint32_t const full = rt.full0 + 8 * st, dst = base + st * FSTAGE;
    if (j.kind == 0) {
      if (j.amap) {
        mbar_expect(full, W_STAGE + A_STAGE);
        tma3(j.wmap, full, dst, 0, j.row0, (j.k0 + s) * 2, EVICT_FIRST);
        tma3(j.amap, full, dst + W_STAGE, 0, 0, (j.k0 + s) * 2, EVICT_LAST);
      } else {   // the activation tile is written into the stage by this SM (the tail's latent_up)
        mbar_expect(full, W_STAGE);
        tma3(j.wmap, full, dst, 0, j.row0, (j.k0 + s) * 2, EVICT_FIRST);
      }
    } else {
      int const na = (j.natoms - 2 * s < 2) ? j.natoms - 2 * s : 2;   // weight pieces in this stage
      mbar_expect(full, na * (8192 + SF_CHUNK + (j.kind == 1 ? T * 128 : 0)));
      uint32_t const sf_dst = base + OFF_WSF + st * 2 * SF_CHUNK;
      if (na == 2) {
        tma3(j.wmap2, full, dst, 0, 0, j.tile0 + 2 * s, EVICT_FIRST);
        tma3(j.sfmap2, full, sf_dst, 0, 0, j.chunk0 + 2 * s, EVICT_FIRST);
      } else {
        tma3(j.wmap, full, dst, 0, 0, j.tile0 + 2 * s, EVICT_FIRST);
        tma3(j.sfmap, full, sf_dst, 0, 0, j.chunk0 + 2 * s, EVICT_FIRST);
      }
      if (j.kind == 1) {
        if (na == 2) tma3(j.amap2, full, dst + W_STAGE, 0, 0, j.k0 + 2 * s, EVICT_LAST);
        else tma2(j.amap, full, dst + W_STAGE, (j.k0 + 2 * s) * 128, 0, EVICT_LAST);
      }
    }
    if (++st == SMAX) { st = 0; ph ^= 1u; }
  }
}

// issuer iss (0 or 1): consume the job's stages into accumulator stage a (kind 0: the stages with s % 2 == iss; kind 1: piece iss of
// every stage). Both issuers wait on EVERY stage in order (parity rule). TMEM columns: acc = tb + 8 a + 32 iss (8 fp32 each);
// weight scales at tb + 64 + 4 iss; kind 1's z_q scales from sfb_base.
__device__ __forceinline__ void issue_job(TileJob const &j, uint32_t base, uint32_t tb, int iss, int &g, int a, uint32_t sfb_base) {
  uint32_t const acc = tb + 8 * a + 32 * iss, sfa = tb + 64 + 4 * iss;
  bool first = true;
  for (int s = 0; s < j.nst; s++, g++) {
    bool const mine = (s & 1) == iss;
    int const st = g % SMAX;
    mbar_wait(rt.full0 + 8 * st, (g / SMAX) & 1); fence_after();
    // the issuer that does not own this stage arrives on `empty` once it has seen the fill: the loader re-arms the slot only after
    // BOTH issuers passed it (otherwise an issuer 5+ stages behind waits for a `full` phase that already completed twice: deadlock)
    if (!mine) { mbar_arrive(rt.empty0 + 8 * st); continue; }
    if (j.kind == 0) {
      uint32_t const wst = base + st * FSTAGE, ast = base + st * FSTAGE + W_STAGE;
      for (int at = 0; at < 2; at++) {   // two 64-column halves of K 128, 4 MMAs of K 16 each
        uint64_t const dw = mkdesc(wst + at * 16384), dx = mkdesc(ast + at * 1024);
        for (int k = 0; k < 4; k++) mma_bf16(acc, dw + 2 * k, dx + 2 * k, (first && k == 0) ? 0u : 1u);
        first = false;
      }
    } else {   // kind 1 (W13)
      int const na = (j.natoms - 2 * s < 2) ? j.natoms - 2 * s : 2;
      for (int at = 0; at < na; at++) {
        int const atom = 2 * s + at;
        cp_sf(sfa, base + OFF_WSF + (st * 2 + at) * SF_CHUNK);
        uint64_t const dw = mkdesc(base + st * FSTAGE + at * 16384);
        uint64_t const db = mkdesc(base + st * FSTAGE + W_STAGE + at * 1024);
        uint32_t const sfb = sfb_base + 4 * (j.k0 + atom);
        mma_mx(acc, dw, db, idesc_mx(0), first ? 0u : 1u, sfa, sfb);
        mma_mx(acc, dw + 2, db + 2, idesc_mx(1), 1u, sfa, sfb);
        mma_mx(acc, dw + 4, db + 4, idesc_mx(2), 1u, sfa, sfb);
        mma_mx(acc, dw + 6, db + 6, idesc_mx(3), 1u, sfa, sfb);
        first = false;
      }
    }
    tc_commit(rt.empty0 + 8 * st);   // `empty` count 2: the owning issuer's MMAs done + the other issuer's arrive
  }
  tc_commit(rt.acc_full0 + 8 * a);   // `acc_full` count 2: every job gives both issuers at least one stage / piece
}

// epilogue (warps 2..5, 128 threads): wait for accumulator stage a, add the two issuers' accumulators -> v[8] (one per token) of
// this thread's row; returns the row (0..127)
__device__ int drain_acc(uint32_t tb, int a, int parity, float *v) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, lgrp = warp & 3, row = lgrp * 32 + lane;
  mbar_wait(rt.acc_full0 + 8 * a, parity); fence_after();
  float v2[8];
  tmem_ld8(tb + 8 * a + ((uint32_t)(lgrp * 32) << 16), v);
  tmem_ld8(tb + 8 * a + 32 + ((uint32_t)(lgrp * 32) << 16), v2);
  for (int t = 0; t < 8; t++) v[t] += v2[t];
  fence_before(); mbar_arrive(rt.acc_empty0 + 8 * a);   // 128 arrivals
  return row;
}

}  // namespace static_mk
