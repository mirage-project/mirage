// core.cuh -- what every task type of the static megakernel shares, and the
// kernel's start and end. The generated layer (static_schedule.py) defines
// first the kernel-wide sizes (config.cuh: STATIC_TOKENS, STATIC_GPUS,
// STATIC_SMS) and
//   STATIC_EXCHANGE_SET       bytes of one set of the cross-GPU exchange region
//   (two sets: launch n uses set n % 2) STATIC_EXCHANGE_HELLO     the start
//   barrier's slots in a set (one 16-B slot per GPU) STATIC_EXCHANGE_OFFSETS
//   per buffer slot: its offset in a set (an exchange buffer; else 0)
//   STATIC_EXCHANGE_BYTES     per buffer slot: its bytes in a set (an exchange
//   buffer; else 0)
// then includes this file and the task types' files (tasks/<name>.cuh), each
// with its body and its task function run_<name>. A task function's template
// arguments: its node (StaticNode: grid, first counter, output buffer slot),
// its params (StaticParams), its own buffer slots (StaticBufSlots: indices
// into G::buf of the scratch buffers, graph tensors and other buffers its layer
// declared) and map slots (StaticMapSlots: indices into Maps::m), each in the
// layer's order, then per input the node that writes it (StaticNode with that
// node's params as v[]; StaticNode<0, 0, 0>: a graph input).
#pragma once
#include "config.cuh"
#include "runtime.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#ifndef STATIC_EXCHANGE_SET
#error                                                                         \
    "core.cuh: the generated layer defines STATIC_EXCHANGE_SET, STATIC_EXCHANGE_HELLO, STATIC_EXCHANGE_OFFSETS, STATIC_EXCHANGE_BYTES"
#endif

namespace static_mk {

// ---- the exchange region's layout (from the layers) ----
constexpr size_t EXCHANGE_SET = STATIC_EXCHANGE_SET,
                 EXCHANGE_HELLO = STATIC_EXCHANGE_HELLO;
constexpr size_t exchange_offset[MAX_BUFS] = STATIC_EXCHANGE_OFFSETS;
constexpr size_t exchange_bytes[MAX_BUFS] = STATIC_EXCHANGE_BYTES;

// ---- the tensor maps: TMA descriptors by slot, built by the host code from
// the layers' declarations ----
struct Maps {
  CUtensorMap m[MAX_MAPS];
};

#ifdef STATIC_RESET_IN_KERNEL
// buffers reset at the start of each launch (STATIC_RESET_IN_KERNEL): byte
// value v[i] over n[i] bytes at p[i] (n a multiple of 16).
// STATIC_NO_START_BARRIER: each has two sets stride[i] bytes apart, and a
// launch resets the set it does not use, at its end (kernel_end)
constexpr int MAX_RESETS = 24;
struct ResetList {
  unsigned char *p[MAX_RESETS];
  unsigned long long n[MAX_RESETS];
  unsigned long long stride[MAX_RESETS];
  unsigned v[MAX_RESETS];
  int cnt;
};
#ifdef STATIC_L2_PREFETCH
// the layer's weights kernel_begin prefetches into L2 (StaticMegakernel
// l2_prefetch_weights, the host's "pf." list): n[i] bytes at p[i] (multiples of
// 16), in the order the layer reads them, as nb batches of PF_BATCH bytes of
// the ranges laid end to end; ctr[2]: the batches claimed so far by the
// launches of each parity
constexpr int MAX_PF = 8;
constexpr unsigned long long PF_BATCH = 1ull << 20, PF_CHUNK = 64ull << 10;
struct PrefetchList {
  unsigned char const *p[MAX_PF];
  unsigned long long n[MAX_PF];
  int cnt;
  unsigned nb;
  unsigned long long *ctr;
};
#endif
#endif
#ifdef STATIC_NO_START_BARRIER
#ifndef STATIC_RESET_IN_KERNEL
#error                                                                         \
    "STATIC_NO_START_BARRIER is a serving mode: it needs STATIC_RESET_IN_KERNEL"
#endif
static_assert(GPUS == 1, "STATIC_NO_START_BARRIER: one GPU");
#endif

// ---- the layer state (the kernel's parameter g) ----
struct G {
  void *buf[MAX_BUFS]; // by slot: the nodes' outputs (an exchange buffer: this
                       // GPU's copy, set 0), scratch buffers, graph tensors
                       // read by pointer
  uint32_t *cnt;       // counters (NCNT): the nodes' lines
  int rank;            // this GPU (of GPUS)
  unsigned gen; // launch number + 1; the start barrier waits until every GPU's
                // hello slot holds it
#ifdef STATIC_RESET_IN_KERNEL
  unsigned *gen_ptr; // the launch number in device memory (gen unused): every
                     // SM reads it + 1 at the start, SM 0 stores it back once
                     // all SMs have (a CUDA graph replay gets a new one)
  unsigned *arrive;  // SMs done with the resets, over all launches (only grows)
  ResetList resets;  // the buffers kernel_begin resets (the host's list)
#ifdef STATIC_L2_PREFETCH
  PrefetchList pf; // the weights kernel_begin prefetches into L2
#endif
#endif
#ifdef STATIC_NO_START_BARRIER
  unsigned *gen_sm; // [NSM] the launch number per SM: each SM reads its own + 1
                    // and stores it back (no SM waits for another)
  unsigned long long par_bytes[MAX_BUFS]; // a buffer slot with two sets: the
                                          // second's offset (else 0); the
                                          // counters have two sets of NCNT
#endif
  unsigned char *mc, *rv; // the exchange region: multicast address (GPUS > 1),
                          // this GPU's copy (2 sets)
  long long *stamps; // [NSM][NSTAMP] time stamps (globaltimer ns) read by the
                     // host (config.cuh STAMP_*)
  long long *start_barrier; // [1] written by every SM once the start barrier is
                            // passed: the last SM's time remains
};

// the exchange region's set of this launch: its launch number's parity, set at
// the kernel's start
__shared__ unsigned dbuf_par;
__device__ __forceinline__ size_t rg_set() {
  return (size_t)dbuf_par * EXCHANGE_SET;
}
#ifdef STATIC_NO_START_BARRIER
// this launch's set: per buffer slot its offset, and the counters it uses (set
// at the kernel's start from the launch number's parity)
__shared__ unsigned long long par_off[MAX_BUFS];
__shared__ uint32_t *cnt_cur;
#ifndef STATIC_RESET_SMS_N
#define STATIC_RESET_SMS_N 0
#define STATIC_RESET_SMS_MASK0 0ull
#define STATIC_RESET_SMS_MASK1 0ull
#define STATIC_RESET_SMS_MASK2 0ull
#define STATIC_RESET_SMS_MASK3 0ull
#endif
// the SMs that reset the next launch's set right after the start (their first
// task waits for other tasks anyway: static_schedule.reset_sms); none: every SM
// at its end
constexpr int RESET_SMS_N = STATIC_RESET_SMS_N;
// sm's index among the reset SMs, or -1 (the mask's 64-bit words as
// immediates: no memory access)
__device__ __forceinline__ int reset_sm_rank(int sm) {
  constexpr unsigned long long M0 = STATIC_RESET_SMS_MASK0,
                               M1 = STATIC_RESET_SMS_MASK1,
                               M2 = STATIC_RESET_SMS_MASK2,
                               M3 = STATIC_RESET_SMS_MASK3;
  int const w = sm >> 6, b = sm & 63;
  unsigned long long const word = w == 0 ? M0 : w == 1 ? M1 : w == 2 ? M2 : M3;
  if (!((word >> b) & 1ull)) {
    return -1;
  }
  int r = __popcll(word & ((1ull << b) - 1ull));
  r += w > 0 ? __popcll(M0) : 0;
  r += w > 1 ? __popcll(M1) : 0;
  r += w > 2 ? __popcll(M2) : 0;
  return r;
}
#endif
// buffer `slot` (G::buf) as a P pointer (STATIC_NO_START_BARRIER: in this
// launch's set)
template <class P>
__device__ __forceinline__ P *buf_at(G const &g, int slot) {
#ifdef STATIC_NO_START_BARRIER
  return reinterpret_cast<P *>(reinterpret_cast<char *>(g.buf[slot]) +
                               par_off[slot]);
#else
  return reinterpret_cast<P *>(g.buf[slot]);
#endif
}
// the nodes' counters (STATIC_NO_START_BARRIER: this launch's set)
__device__ __forceinline__ uint32_t *cnt_now(G const &g) {
#ifdef STATIC_NO_START_BARRIER
  return cnt_cur;
#else
  return g.cnt;
#endif
}
// a multicast store into this launch's set of every GPU's exchange region (off:
// an offset in the set)
__device__ __forceinline__ void push16(G const &g, size_t off, uint4 v) {
  push16_mc(g.mc, g.rv, off + rg_set(), v);
}

// params[I] of a node (StaticParams<P...>::v holds P..., 0)
template <class PARAMS, int I>
__host__ __device__ constexpr int param_at() {
  static_assert(sizeof(PARAMS::v) / sizeof(int) > I + 1,
                "the node has fewer params");
  return PARAMS::v[I];
}
// params[I] of a node, or DEF when the node has fewer params
template <class PARAMS, int I, int DEF>
__host__ __device__ constexpr int param_or() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return DEF;
  }
}

// entry I of a node's buffer slots or map slots (StaticBufSlots<S...>::v,
// StaticMapSlots<S...>::v hold S..., -1)
template <class SLOTS, int I>
__host__ __device__ constexpr int slot_at() {
  static_assert(sizeof(SLOTS::v) / sizeof(int) > I + 1,
                "the node has fewer slots");
  return SLOTS::v[I];
}
// entry I of a node's buffer slots or map slots, or DEF when it has fewer
template <class SLOTS, int I, int DEF>
__host__ __device__ constexpr int slot_or() {
  if constexpr (sizeof(SLOTS::v) / sizeof(int) > I + 1) {
    return SLOTS::v[I];
  } else {
    return DEF;
  }
}

// how a gemm_tile node's K parts are combined in its output buffer out (N = the
// weight's rows; task K part `split`, thread row `row`, v[t] = token t). The
// node's consumer reads the buffer in that layout (its producer's params[0]).
enum Combine {
  COMBINE_SLOTS = 0, // fp32 out[split][t][N]: out[split][t][row0 + row] = v[t];
                     // one 0xFF-prefilled slot per K part, the reader polls
                     // them and adds the parts in a fixed order
  COMBINE_ADD =
      1, // int64 out[t][N]: += v[t] in fixed point (runtime.cuh red_add_fx: the
         // same sum in any order), then the node's counter [row0 / 128] += 1
         // per task; the reader waits for all K parts of its 128-row blocks
  COMBINE_STORE =
      2, // fp32 out[t][N] = v[t], plain stores (one K part), then the node's
         // counter [0] += 1 per task; the reader waits for all tasks
  COMBINE_STORE_BF16 = 3, // bf16 out[t][N] = bf16(v[t]), else as COMBINE_STORE
                          // (a layer's final bf16 output, read after the
                          // kernel)
};

// 8 fp32 values -> one 16-B bf16 vector
__device__ __forceinline__ uint4 pack_bf16x8(float4 const &p0,
                                             float4 const &p1) {
  __nv_bfloat162 const h0 = __floats2bfloat162_rn(p0.x, p0.y),
                       h1 = __floats2bfloat162_rn(p0.z, p0.w);
  __nv_bfloat162 const h2 = __floats2bfloat162_rn(p1.x, p1.y),
                       h3 = __floats2bfloat162_rn(p1.z, p1.w);
  return make_uint4(*reinterpret_cast<uint32_t const *>(&h0),
                    *reinterpret_cast<uint32_t const *>(&h1),
                    *reinterpret_cast<uint32_t const *>(&h2),
                    *reinterpret_cast<uint32_t const *>(&h3));
}

// the adding side of an all-reduce (allreduce_send): thread v polls 16-B bf16
// vector v of `row` from every GPU (slot r at row + r * RANK, 0xFF-prefilled)
// and adds them in fp32 in GPU order -> the 8 sums. REARM: re-armed by
// allreduce_rearm after the caller used them.
template <size_t RANK>
__device__ __forceinline__ void
    allreduce_land_bf16(G const &g, unsigned char const *row, int v, float *s) {
  uint4 u[GPUS];
  bool ok[GPUS] = {};
  bool all;
  do {
    all = true;
#pragma unroll
    for (int r = 0; r < GPUS; r++) {
      if (!ok[r]) {
        u[r] = ld16_poll(row + (size_t)r * RANK + (size_t)v * 16);
      }
    }
#pragma unroll
    for (int r = 0; r < GPUS; r++) {
      if (!ok[r]) {
        ok[r] = valid16(u[r]);
      }
      all = all && ok[r];
    }
  } while (!all);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    s[k] = 0.f;
  }
#pragma unroll
  for (int r = 0; r < GPUS;
       r++) { // a fixed bound (unrolled), so u stays in registers
    uint32_t const wd[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 const f =
          __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&wd[k]));
      s[2 * k] += f.x;
      s[2 * k + 1] += f.y;
    }
  }
}
template <size_t RANK>
__device__ __forceinline__ void
    allreduce_rearm(G const &g, unsigned char const *row, int v) {
  for (int r = 0; r < GPUS; r++) {
    rearm16(row + (size_t)r * RANK + (size_t)v * 16);
  }
}

// the kernel's local variables (every function is inlined, so they stay in
// registers)
struct KernelLocals {
  uint32_t
      base; // 1024-aligned dynamic shared memory start (shared-window address)
  char *sm; // the same, as a pointer
  int sm_id;
  uint32_t tb;      // TMEM base
  int gl, gi0, gi1; // ring stage counters: loader, issuer 0, issuer 1 (they
                    // advance by the same stages)
  int wt;           // accumulator tiles so far (all roles)
  bool ring_relaid; // a task re-laid the ring's shared memory (moe_experts'
                    // phase switch): a GEMM after it starts with fresh ring
                    // barriers
#ifdef STATIC_TIMING_BUILD
  unsigned long long *stage_stamps =
      nullptr; // this list entry's 2 GEMM stage stamps (set by the generated
               // loop)
#endif
};

#if defined(STATIC_RESET_IN_KERNEL) && defined(STATIC_L2_PREFETCH)
// batch b of the prefetch list into L2 (cp.async.bulk.prefetch.L2: fire and
// forget, the lines land after the CTA moved on), PF_CHUNK bytes an
// instruction; false when b is past the list
__device__ __forceinline__ bool pf_batch(PrefetchList const &pl,
                                         unsigned long long b) {
  if (b >= pl.nb) {
    return false;
  }
  unsigned long long o = b * PF_BATCH, left = PF_BATCH;
  for (int i = 0; i < pl.cnt && left > 0; i++) {
    if (o >= pl.n[i]) {
      o -= pl.n[i];
      continue;
    }
    for (; o < pl.n[i] && left > 0;) {
      unsigned long long const c = min(min(PF_CHUNK, pl.n[i] - o), left);
      asm volatile(
          "cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pl.p[i] + o),
          "r"((uint32_t)c)
          : "memory");
      o += c;
      left -= c;
    }
    o = 0;
  }
  return true;
}
#endif

#ifdef STATIC_NO_START_BARRIER
// share idx of n of the resets of the set the next launch uses: no task of this
// launch touches it, the previous launch (which used it) is over
// (griddepcontrol.wait), and the next launch's tasks start after this grid's
// end. 16-B stores, not waited for
__device__ __forceinline__ void reset_next_set(G const &g, int idx, int n) {
  unsigned const other = dbuf_par ^ 1u;
  for (int i = 0; i < g.resets.cnt; i++) {
    unsigned const w = g.resets.v[i] * 0x01010101u;
    uint4 const val = make_uint4(w, w, w, w);
    uint4 *const q =
        reinterpret_cast<uint4 *>(g.resets.p[i] + other * g.resets.stride[i]);
    size_t const n16 = g.resets.n[i] / 16;
    for (size_t k = (size_t)idx * blockDim.x + threadIdx.x; k < n16;
         k += (size_t)n * blockDim.x) {
      q[k] = val;
    }
  }
}
#endif

// before the task loop: shared memory, barriers, the resets
// (STATIC_RESET_IN_KERNEL) or the start barrier over the GPUs, TMEM
__device__ __forceinline__ void kernel_begin(G const &g, KernelLocals &L) {
  extern __shared__ __align__(1024) char smem_raw[];
  L.base = (su32(smem_raw) + 1023u) & ~1023u;
  L.sm = smem_raw + (L.base - su32(smem_raw));
  L.sm_id = blockIdx.x;
  int const sm_id = L.sm_id;
  cta_prologue(L.sm);
#ifdef STATIC_NO_START_BARRIER
  // serving with no start barrier (one GPU): every buffer the start would reset
  // has two sets (and so do the counters); launch n uses set n % 2, the other
  // was reset at the end of launch n - 1 (kernel_end), which this launch's
  // tasks follow (griddepcontrol.wait below), and no task of launch n touches
  // it. The launch number: this SM's own word. PDL as below
  __shared__ unsigned gen_s;
  long long t_start = 0;
  if (threadIdx.x == 0) {
    if constexpr (PDL_TRIGGER == 1) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    t_start = gtime();
    unsigned volatile *const gw = g.gen_sm + sm_id;
    unsigned const gen = *gw + 1;
    *gw = gen;
    gen_s = gen;
    dbuf_par = gen & 1u;
    cnt_cur = g.cnt + (size_t)(gen & 1u) * NCNT;
  }
  if (threadIdx.x < NSTAMP) {
    g.stamps[sm_id * NSTAMP + threadIdx.x] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < MAX_BUFS; i += blockDim.x) {
    par_off[i] = (gen_s & 1u) ? g.par_bytes[i] : 0ull; // read by the tasks
                                                       // after tmem_alloc_512's
                                                       // __syncthreads
  }
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_START] = t_start;
    *g.start_barrier = gtime();
  }
#elif defined(STATIC_RESET_IN_KERNEL)
  // the resets left to the start (the host's list: buffers only this GPU
  // writes; the others are re-armed by their readers, REARM), 16-B stores
  // spread over all SMs, each SM its own stamp row. Every SM waits until every
  // SM of this GPU has done its resets (release / acquire on the arrival count,
  // gpu scope: no other GPU writes into these buffers). No barrier over the
  // GPUs: the exchange region has two sets (EXCHANGE_SET), and another GPU's
  // launch n writes only into set n % 2, which this GPU last read in launch n -
  // 2
  // PDL (programmatic dependent launch): the next kernel may launch now (it
  // waits for this grid's end before reading y), and everything up to the TMEM
  // allocation below runs before the wait for the previous kernel: it touches
  // only this kernel's own buffers, never the previous kernel's output (x,
  // prefix). Both instructions are no-ops without a programmatic launch.
  // Only thread 0 uses the launch number (the arrival count, dbuf_par): its
  // load is in flight while the threads do the resets, which do not need it
  long long t_start = 0;
  unsigned gen = 0;
  if (threadIdx.x == 0) {
    if constexpr (PDL_TRIGGER == 1) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    t_start = gtime();
    gen = *(unsigned const volatile *)g.gen_ptr + 1;
  }
  for (int i = 0; i < g.resets.cnt; i++) {
    unsigned const w = g.resets.v[i] * 0x01010101u;
    uint4 const val = make_uint4(w, w, w, w);
    uint4 *const q = reinterpret_cast<uint4 *>(g.resets.p[i]);
    size_t const n16 = g.resets.n[i] / 16;
    for (size_t k = (size_t)sm_id * blockDim.x + threadIdx.x; k < n16;
         k += (size_t)gridDim.x * blockDim.x) {
      q[k] = val;
    }
  }
  if (threadIdx.x < NSTAMP) {
    g.stamps[sm_id * NSTAMP + threadIdx.x] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    dbuf_par = gen & 1u; // the other threads read it after tmem_alloc_512's
                         // __syncthreads
    g.stamps[sm_id * NSTAMP + STAMP_START] = t_start;
    cnt_add(g.arrive, 1u); // release: this SM's resets (all its threads': the
                           // barrier above) before its arrival
#ifdef STATIC_L2_PREFETCH
    // while it waits: the batches of the layer's weights not claimed yet. The
    // CTAs that start first (on the SMs the previous kernel leaves first: PDL)
    // claim them; one that arrives last finds the barrier passed and claims
    // none. Launch gen claims on ctr[gen % 2]; SM 0 zeroes the other one below
    // (launch gen - 1, which used it, has ended: every SM is in this launch)
    // The next claim is in flight with each check of the barrier, so a CTA
    // leaves as soon as the check says so (it does not wait for a claim; a
    // claim left unused is a batch not prefetched)
    unsigned long long *const ctr = g.pf.ctr + (gen & 1u);
    unsigned long long b = atomicAdd(ctr, 1ull);
    bool more = true;
    while ((int)(cnt_ld(g.arrive) - gen * gridDim.x) < 0) {
      if (more) {
        more = pf_batch(g.pf, b);
        if (more) {
          b = atomicAdd(ctr, 1ull);
        }
      }
    } // acquire: every SM of this launch has arrived (wrap-safe)
    if (sm_id == 0) {
      g.pf.ctr[(gen & 1u) ^ 1u] = 0ull;
    }
#else
    while ((int)(cnt_ld(g.arrive) - gen * gridDim.x) < 0) {
    } // acquire: every SM of this launch has arrived (wrap-safe)
#endif
    if (sm_id == 0) {
      *g.gen_ptr = gen; // read by the next launch (after this one ends)
    }
    *g.start_barrier = gtime();
  }
#else
  // without STATIC_RESET_IN_KERNEL: the host resets the buffers and starts all
  // GPUs together; the start barrier lines the GPUs up (the layer's measured
  // span starts there): SM 0 of each GPU writes the launch number into its
  // hello slot on every GPU (this launch's set), every SM waits for all slots
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.wait;" ::
                     : "memory"); // PDL: from here on global memory is read
                                  // (no-op without a programmatic launch)
    g.stamps[sm_id * NSTAMP + STAMP_START] = gtime();
    unsigned const gen = g.gen;
    dbuf_par =
        gen &
        1u; // the other threads read it after tmem_alloc_512's __syncthreads
    if (sm_id == 0) {
      push16(g,
             EXCHANGE_HELLO + (size_t)g.rank * 16,
             make_uint4(gen, gen, gen, gen));
    }
    for (int r = 0; r < GPUS; r++) {
      unsigned const volatile *hp =
          (unsigned const volatile *)(g.rv + EXCHANGE_HELLO + (size_t)r * 16 +
                                      rg_set());
      while (*hp < gen) {
      }
    }
    *g.start_barrier = gtime();
  }
#endif
  L.tb = tmem_alloc_512(); // its __syncthreads also publishes rt
#ifdef STATIC_RESET_IN_KERNEL
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.wait;" ::
                     : "memory"); // PDL: from here on the previous kernel's
                                  // output is read
  }
  __syncthreads();
#endif
#ifdef STATIC_NO_START_BARRIER
  if constexpr (RESET_SMS_N > 0) { // the next launch's resets on the SMs whose
                                   // first task waits anyway
    int const rank = reset_sm_rank(sm_id);
    if (rank >= 0) {
      reset_next_set(g, rank, RESET_SMS_N);
    }
  }
#endif
  L.gl = 0;
  L.gi0 = 0;
  L.gi1 = 0;
  L.wt = 0;
  L.ring_relaid = false;
}

// after the task loop (STATIC_NO_START_BARRIER with no reset SMs: first this
// SM's share of the next launch's resets, reset_next_set)
__device__ __forceinline__ void kernel_end(G const &g, KernelLocals &L) {
  __syncthreads();
#ifdef STATIC_NO_START_BARRIER
  if constexpr (RESET_SMS_N == 0) {
    reset_next_set(g, L.sm_id, gridDim.x);
    __syncthreads();
  }
#endif
  if (threadIdx.x == 0) {
    g.stamps[L.sm_id * NSTAMP + STAMP_END] = gtime();
  }
  tmem_dealloc_512(L.tb);
}

} // namespace static_mk
