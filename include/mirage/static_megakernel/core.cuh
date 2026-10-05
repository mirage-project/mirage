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
// its params (StaticParams), its own slots (StaticSlots: the buffers, graph
// tensors and tensor maps its layer declared, in the layer's order), then per
// input the node that writes it (StaticNode with that node's params as v[];
// StaticNode<0, 0, 0>: a graph input).
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
// value v[i] over n[i] bytes at p[i] (n a multiple of 16)
constexpr int MAX_RESETS = 24;
struct ResetList {
  unsigned char *p[MAX_RESETS];
  unsigned long long n[MAX_RESETS];
  unsigned v[MAX_RESETS];
  int cnt;
};
#endif

// ---- the layer state (the kernel's parameter g) ----
struct G {
  void *buf[MAX_BUFS]; // by slot: the nodes' outputs (an exchange buffer: this
                       // GPU's copy, set 0), scratch buffers, graph tensors
                       // read by pointer
  uint32_t *cnt; // counters (NCNT): the nodes' lines
  int rank, tp;  // this GPU, number of GPUs
  unsigned gen;  // launch number + 1; the start barrier waits until every GPU's
                 // hello slot holds it
#ifdef STATIC_RESET_IN_KERNEL
  unsigned *gen_ptr; // the launch number in device memory (gen unused): every
                     // SM reads it + 1 at the start, SM 0 stores it back once
                     // all SMs have (a CUDA graph replay gets a new one)
  unsigned *arrive; // SMs done with the resets, over all launches (only grows)
  ResetList resets; // the buffers kernel_begin resets (the host's list)
  int pdl_trigger;  // when this kernel lets the next one launch
                    // (griddepcontrol.launch_dependents): 0 at its end
                    // (implicit), 1 at its start, 2 when a CTA starts a
                   // residual_add task (CTAs without one: at their end)
#endif
  unsigned char *mc, *rv; // the exchange region: multicast address (tp > 1),
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
// buffer `slot` (G::buf) as a P pointer
template <class P>
__device__ __forceinline__ P *buf_at(G const &g, int slot) {
  return reinterpret_cast<P *>(g.buf[slot]);
}
// a multicast store into this launch's set of every GPU's exchange region (off:
// an offset in the set)
__device__ __forceinline__ void push16(G const &g, size_t off, uint4 v) {
  push16_mc(g.mc, g.rv, g.tp, off + rg_set(), v);
}

// params[I] of a node, 0 when the node has fewer params (StaticParams<P...>::v
// holds P..., 0)
template <class PARAMS, int I>
__host__ __device__ constexpr int param_at() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return 0;
  }
}

// slot I of a node's own slots (StaticSlots<S...>::v holds S..., -1)
template <class SLOTS, int I>
__host__ __device__ constexpr int slot_at() {
  static_assert(sizeof(SLOTS::v) / sizeof(int) > I + 1,
                "the node has fewer slots");
  return SLOTS::v[I];
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
  bool ok[GPUS];
  bool all;
#pragma unroll
  for (int r = 0; r < GPUS; r++) {
    ok[r] = r >= g.tp;
  }
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
    if (r < g.tp) {
      uint32_t const wd[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 const f = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162 const *>(&wd[k]));
        s[2 * k] += f.x;
        s[2 * k + 1] += f.y;
      }
    }
  }
}
template <size_t RANK>
__device__ __forceinline__ void
    allreduce_rearm(G const &g, unsigned char const *row, int v) {
  for (int r = 0; r < g.tp; r++) {
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

// before the task loop: shared memory, barriers, the resets
// (STATIC_RESET_IN_KERNEL) or the start barrier over the GPUs, TMEM
__device__ __forceinline__ void
    kernel_begin(Maps const &maps, G const &g, KernelLocals &L) {
  (void)maps;
  extern __shared__ __align__(1024) char smem_raw[];
  L.base = (su32(smem_raw) + 1023u) & ~1023u;
  L.sm = smem_raw + (L.base - su32(smem_raw));
  L.sm_id = blockIdx.x;
  int const sm_id = L.sm_id;
  cta_prologue(L.sm);
#ifdef STATIC_RESET_IN_KERNEL
  // the resets left to the start (the host's list: buffers only this GPU
  // writes; the others are re-armed by their readers, REARM), 16-B stores
  // spread over all SMs, each SM its own stamp row. Every SM waits until every
  // SM of this GPU has done its resets (release / acquire on the arrival count,
  // gpu scope: no other GPU writes into these buffers). No barrier over the
  // GPUs: the exchange region has two sets (EXCHANGE_SET), and another GPU's
  // launch n writes only into set n % 2, which this GPU last read in launch n -
  // 2
  __shared__ unsigned gen_s;
  __shared__ long long t_start;
  // PDL (programmatic dependent launch): the next kernel may launch now (it
  // waits for this grid's end before reading y), and everything up to the TMEM
  // allocation below runs before the wait for the previous kernel: it touches
  // only this kernel's own buffers, never the previous kernel's output (x,
  // prefix). Both instructions are no-ops without a programmatic launch.
  if (threadIdx.x == 0) {
    if (g.pdl_trigger == 1) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    t_start = gtime();
    gen_s = *(unsigned const volatile *)g.gen_ptr + 1;
    dbuf_par = gen_s & 1u;
  }
  __syncthreads();
  unsigned const gen = gen_s;
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
    g.stamps[sm_id * NSTAMP + STAMP_START] = t_start;
    cnt_add(g.arrive, 1u); // release: this SM's resets (all its threads': the
                           // barrier above) before its arrival
    while ((int)(cnt_ld(g.arrive) - gen * gridDim.x) < 0) {
    } // acquire: every SM of this launch has arrived (wrap-safe)
    if (sm_id == 0) {
      *g.gen_ptr = gen; // read by the next launch (after this one ends)
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
    for (int r = 0; r < g.tp; r++) {
      unsigned const volatile *hp =
          (unsigned const volatile *)(g.rv + EXCHANGE_HELLO + (size_t)r * 16 +
                                      rg_set());
      while (*hp < gen) {
      }
    }
#endif
    *g.start_barrier = gtime();
  }
  L.tb = tmem_alloc_512(); // its __syncthreads also publishes rt
#ifdef STATIC_RESET_IN_KERNEL
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.wait;" ::
                     : "memory"); // PDL: from here on the previous kernel's
                                  // output is read
  }
  __syncthreads();
#endif
  L.gl = 0;
  L.gi0 = 0;
  L.gi1 = 0;
  L.wt = 0;
  L.ring_relaid = false;
}

// after the task loop
__device__ __forceinline__ void kernel_end(G const &g, KernelLocals &L) {
  __syncthreads();
  if (threadIdx.x == 0) {
    g.stamps[L.sm_id * NSTAMP + STAMP_END] = gtime();
  }
  tmem_dealloc_512(L.tb);
}

} // namespace static_mk
