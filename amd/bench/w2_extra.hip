// w2_extra.hip (copy of w2_infl.hip, 10-26): the 24 KB items plus the kernel's per-item extras, only on the lanes of real pairs
// (lane % 16 < W2NC, the kernel's p < nc; default 16 = every lane), 8 = the item's weight scales, 8 dwords per lane, plain loads. -DW2X bits: 1 = the expert's h_q by
// device-scope (sc1) loads, 6 x 16 B per lane (as expert_w2c_hq, issued after the weights), 2 = 8 x 16 B sc1 stores per lane (the R
// rows), 4 = h_q first (before the weights). Built once per W2X.
// w2_infl.hip (copy of w2_size.hip, 10-26): W2 read rate by items in flight per wave (1 / 2) and item size (12 / 24 / 48 KB) -
// pcw1: with all 256 CUs reading, more bytes in flight per CU read slower (64 KB 26 GB/s per CU, 256 KB 21). Original header:
// Microbenchmark: does the CSR W2 phase read faster with larger items? (10-25: in fused KDA T 64 the W2 items, 24 KB each, read
// ~4.4 GB/s per wave vs ~6.3 for the 118 KB W13 units; w2_pattern.hip: the 24 KB pattern alone 5.07 TB/s with per-die tickets,
// 5.82 static. dec404 / dec414: the ticket's vmcnt(0) and the inline-asm stores' waits are not what limits it.)
//
// 320 experts (K3 at bs 64) out of 896, each 672 KB of W2 weights, contiguous, at their ids' places in a 616 MB array. 256
// workgroups x 4 waves; a wave takes items of KB KB (24 / 48: 28 / 14 items per expert) from its die's share, by
// a per-die ticket (agent-scope atomic through an opaque address: no wait right after it - moe_experts_csr.cuh STATIC_TAKE_ASYNC)
// or statically (wave i of the die: items i, i + 128, ..). Two items in flight per wave (the next item's loads issued before the
// current one is used), or one (48 KB: two would not fit the registers). Weights only (nt loads of 1 KB per wave). Rate = 215 MB / kernel time (events, 1 GB read before each).
//
// w2_tw_kernel: a ticket wave per workgroup feeds three loader waves through an LDS ring (see the kernel).
// w2_sat_kernel: per-wave tickets by scalar atomics (s_atomic_add glc, waited with lgkmcnt), the die = the hardware XCC_ID.
//
// NC counters per die (w2_kernel NC): the die's items split over NC counter lines, a wave moving on when its counter's share is
// out.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 w2_size.hip -o w2_size
// Run:   ./w2_size
#include <algorithm>
#include <random>
#include <string>

#include "common.h"

typedef int v4i __attribute__((ext_vector_type(4)));
typedef unsigned v4u_x __attribute__((ext_vector_type(4)));
#ifndef W2_NUSED
#define W2_NUSED 370
#endif
constexpr int NEXP = 896, NUSED = W2_NUSED, EXP_KB = 672;   // (-DW2_NUSED=896: every expert, 3x the items)
constexpr size_t EXP_BYTES = (size_t)EXP_KB * 1024;

#ifndef W2X
#define W2X 0
#endif
#ifndef W2NC
#define W2NC 16
#endif
__device__ char *g_hq; // (W2X: the h_q buffer [NEXP][6 KB], the R scratch)
__device__ char *g_rs;
template <int KB>
struct Item {
  v4i f[KB];
  v4i h[6];
  unsigned sc[8];
  int e, it;
};
__device__ __forceinline__ void load_hq(v4i (&h)[6], int e) {
  int const lane = threadIdx.x & 63;
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(g_hq + (size_t)e * 6144, 0, 0x7FFFFFFF, 0x00020000);
#pragma unroll
  for (int i = 0; i < 6; i++) {
    h[i] = (lane % 16) < W2NC ? __builtin_bit_cast(v4i, __builtin_amdgcn_raw_buffer_load_b128(rs, (unsigned)((i * 64 + lane) * 16), 0, 16))
                              : v4i{0, 0, 0, 0};
  }
}
template <int KB>
__device__ __forceinline__ void load_item(Item<KB> &F, char const *w, int e, int it) {
  int const lane = threadIdx.x & 63;
  F.e = e;
  F.it = it;
  if constexpr ((W2X & 1) && (W2X & 4)) {
    load_hq(F.h, e);
  }
  v4i const *wt = reinterpret_cast<v4i const *>(w + (size_t)e * EXP_BYTES + (size_t)it * KB * 1024) + lane;
#pragma unroll
  for (int i = 0; i < KB; i++) {
    F.f[i] = __builtin_nontemporal_load(wt + (size_t)i * 64);
  }
  if constexpr ((W2X & 1) && !(W2X & 4)) {
    load_hq(F.h, e);
  }
  if constexpr (W2X & 8) { // (the kernel's F.sc: 8 dwords per lane, plain loads, after the weights)
    unsigned const *st = reinterpret_cast<unsigned const *>(g_hq + (size_t)NEXP * 6144) + ((size_t)e * 28 + it) * 512 + lane;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      F.sc[i] = st[i * 64];
    }
  }
}
template <int KB>
__device__ __forceinline__ unsigned use_item(Item<KB> const &F) {
  unsigned a = 0;
#pragma unroll
  for (int i = 0; i < KB; i++) {
    a ^= (unsigned)(F.f[i].x + F.f[i].y + F.f[i].z + F.f[i].w);
  }
  if constexpr (W2X & 1) {
#pragma unroll
    for (int i = 0; i < 6; i++) {
      a ^= (unsigned)F.h[i].y;
    }
  }
  if constexpr (W2X & 8) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      a ^= F.sc[i];
    }
  }
  if constexpr (W2X & 2) {
    int const lane = threadIdx.x & 63;
    __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(g_rs, 0, 0x7FFFFFFF, 0x00020000);
    v4u_x const v = {a, a + 1, a + 2, a + 3};
    if constexpr (W2X & 16) { // (packed: the item's W2NC x 512 B of rows by full-lane 1 KB stores, ceil(W2NC / 2) instructions)
#pragma unroll
      for (int i = 0; i < (W2NC + 1) / 2; i++) {
        __builtin_amdgcn_raw_buffer_store_b128(v, rs, (unsigned)(((((size_t)(F.e * 28 + F.it) * 8 + i) * 64 + lane) * 16) % (64u << 20)), 0, 16);
      }
    } else if ((lane % 16) < W2NC) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        __builtin_amdgcn_raw_buffer_store_b128(v, rs, (unsigned)(((((size_t)(F.e * 28 + F.it) * 8 + i) * 64 + lane) * 16) % (64u << 20)), 0, 16);
      }
    }
  }
  return a;
}

// NC counters per die (NC > 1, tickets): counter j of the die hands out its share [lo_j, hi_j) of the die's items; a wave starts at
// counter gw % NC and moves to the next one when its share is out (one more ticket each), done after all NC
// DUMMY (STATIC only): each item also issues a per-die ticket atomic whose value is not used (the atomic's cost apart from the order)
// PAIR (tickets): a ticket names two adjacent items (one atomic per two items), the second taken without an atomic
template <int KB, bool STATIC, int INFL = 2, int NC = 1, bool DUMMY = false, bool PAIR = false>
__global__ void __launch_bounds__(256, 1) w2_kernel(char const *w, int const *used, unsigned *ctr, unsigned *out) {
  constexpr int IPE = EXP_KB / KB;
  int const die = blockIdx.x & 7, wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int const n = NUSED * IPE, lo = n * die / 8, hi = n * (die + 1) / 8;
  int const gw = (blockIdx.x >> 3) * 4 + wave, nw = 32 * 4;
  int k = 0, cj = gw % NC, tries = 0, pend = -1;
  auto lo_of = [&](int j) { return lo + (hi - lo) * j / NC; };
  auto issue = [&]() -> unsigned {
    if constexpr (PAIR) {
      if (pend >= 0) {
        return 0xFFFFFFFFu; // (the pair's second item: no atomic)
      }
    }
    if constexpr (STATIC && !DUMMY) {
      return 0u;
    } else {
      unsigned v = 0;
      if (lane == 0) {
        unsigned z;
        asm volatile("v_mov_b32 %0, 0" : "=v"(z));
        v = __hip_atomic_fetch_add(ctr + (die * 8 + cj) * 32 + z, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      return v;
    }
  };
  auto finish = [&](unsigned v) -> int {
    if constexpr (STATIC) {
      if constexpr (DUMMY) {
        asm volatile("" ::"v"(v)); // (the ticket's value waited for, then dropped)
      }
      return lo + gw + (k++) * nw;
    } else if constexpr (PAIR) {
      if (v == 0xFFFFFFFFu && pend >= 0) {
        int const it = pend;
        pend = -1;
        return it;
      }
      int const it = lo + 2 * (int)__builtin_amdgcn_readfirstlane(v);
      if (it >= hi) {
        return hi;
      }
      pend = it + 1 < hi ? it + 1 : -1;
      return it;
    } else {
      for (;;) {
        int const it = lo_of(cj) + (int)__builtin_amdgcn_readfirstlane(v);
        if (it < lo_of(cj + 1)) {
          return it;
        }
        if (++tries >= NC) {
          return hi;
        }
        cj = (cj + 1) % NC;
        v = issue();
      }
    }
  };
  unsigned acc = 0;
  if constexpr (INFL == 1) { // (one item in flight: its loads, the next ticket, then use it)
    Item<KB> A;
    int ia = finish(issue());
    while (ia < hi) {
      load_item<KB>(A, w, used[ia / IPE], ia % IPE);
      unsigned const tk = issue();
      acc ^= use_item<KB>(A);
      ia = finish(tk);
    }
    if (acc == 0x12345678u) {
      out[blockIdx.x] = acc;
    }
    return;
  }
  if constexpr (INFL == 3) { // (two in flight, each ticket issued one item ahead - as moe_experts_csr.cuh's W2 loop: the ticket
    // for the item after next goes out right after the next one's ticket is read, before its loads)
    Item<KB> A, B;
    int ia = finish(issue());
    unsigned tk = issue();
    if (ia < hi) {
      load_item<KB>(A, w, used[ia / IPE], ia % IPE);
    }
    int ib = ia < hi ? finish(tk) : hi;
    tk = issue();
    if (ib < hi) {
      load_item<KB>(B, w, used[ib / IPE], ib % IPE);
    }
    while (ia < hi) {
      acc ^= use_item<KB>(A);
      if (ib >= hi) {
        break;
      }
      int const ic = finish(tk);
      tk = issue();
      if (ic < hi) {
        load_item<KB>(A, w, used[ic / IPE], ic % IPE);
      }
      acc ^= use_item<KB>(B);
      if (ic >= hi) {
        break;
      }
      ia = ic;
      ib = finish(tk);
      tk = issue();
      if (ib < hi) {
        load_item<KB>(B, w, used[ib / IPE], ib % IPE);
      }
    }
    if (acc == 0x12345678u) {
      out[blockIdx.x] = acc;
    }
    return;
  }
  Item<KB> A, B;
  int ia = finish(issue());
  if (ia < hi) {
    load_item<KB>(A, w, used[ia / IPE], ia % IPE);
  }
  int ib = ia < hi ? finish(issue()) : hi;
  if (ib < hi) {
    load_item<KB>(B, w, used[ib / IPE], ib % IPE);
  }
  while (ia < hi) {
    unsigned const tk = ib < hi ? issue() : 0u;
    acc ^= use_item<KB>(A);
    if (ib >= hi) {
      break;
    }
    int const ic = finish(tk);
    if (ic < hi) {
      load_item<KB>(A, w, used[ic / IPE], ic % IPE);
    }
    unsigned const tk2 = ic < hi ? issue() : 0u;
    acc ^= use_item<KB>(B);
    if (ic >= hi) {
      break;
    }
    ia = ic;
    ib = finish(tk2);
    if (ib < hi) {
      load_item<KB>(B, w, used[ib / IPE], ib % IPE);
    }
  }
  if (acc == 0x12345678u) {
    out[blockIdx.x] = acc;
  }
}

// Ticket wave (TW): wave 3 of each workgroup only takes the die's tickets (B items per atomic, up to 2 atomics in flight) and
// writes the items into an LDS ring of R entries ((sequence << 32) | item, one 64-bit store); waves 0..2 claim ring entries by an
// LDS atomic and spin on the entry's sequence. The loader waves' vmcnt then holds only their weight loads (a global ticket's
// return is waited for with vmcnt, which also waits for every load issued before it: dec419's dummy atomic cost ~15 %).
template <int KB, int B, bool CNT = false, int R = 16>
__global__ void __launch_bounds__(256, 1) w2_tw_kernel(char const *w, int const *used, unsigned *ctr, unsigned *out) {
  constexpr int IPE = EXP_KB / KB;
  __shared__ unsigned long long ring[R];
  __shared__ int head, tail_seen;
  int const die = blockIdx.x & 7, wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int const n = NUSED * IPE, lo = n * die / 8, hi = n * (die + 1) / 8;
  if (threadIdx.x == 0) {
    head = 0;
    tail_seen = 0;
  }
  for (int i = threadIdx.x; i < R; i += 256) {
    ring[i] = ~0ull;
  }
  __syncthreads();
  if (wave == 3) { // the ticket wave
    if (lane == 0) {
      int j = 0, left = 3 * 2 + 2; // (sentinels once the range is out: every consumer wave sees one)
      bool out_of_range = false;
      while (left > 0) {
        unsigned v = 0;
        if (!out_of_range) {
          unsigned z;
          asm volatile("v_mov_b32 %0, 0" : "=v"(z));
          v = __hip_atomic_fetch_add(ctr + die * 32 + z, (unsigned)B, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        for (int b = 0; b < B && left > 0; b++) {
          int it = out_of_range ? hi : lo + (int)v + b;
          if (it >= hi) {
            it = hi;
            out_of_range = true;
            left--;
          }
          while (j - __hip_atomic_load(&head, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) >= R) {
            __builtin_amdgcn_s_sleep(1);
          }
          __hip_atomic_store(&ring[j % R], ((unsigned long long)j << 32) | (unsigned)it, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_WORKGROUP);
          j++;
        }
      }
    }
    return;
  }
  auto claim = [&]() -> int { // the next ring entry's item (LDS only; CNT: each item counted at ctr[2048 + item], a check run)
    int k = 0;
    if (lane == 0) {
      k = __hip_atomic_fetch_add(&head, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    k = __builtin_amdgcn_readfirstlane(k);
    unsigned long long e;
    for (;;) {
      e = __hip_atomic_load(&ring[k % R], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      if ((int)(e >> 32) == k && e != ~0ull) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    if constexpr (CNT) {
      if (lane == 0 && (int)(unsigned)e < hi) {
        atomicAdd(ctr + 2048 + (int)(unsigned)e, 1u);
      }
    }
    return (int)(unsigned)e;
  };
  unsigned acc = 0;
  Item<KB> A, Bi;
  int ia = claim();
  if (ia < hi) {
    load_item<KB>(A, w, used[ia / IPE], ia % IPE);
  }
  int ib = ia < hi ? claim() : hi;
  if (ib < hi) {
    load_item<KB>(Bi, w, used[ib / IPE], ib % IPE);
  }
  while (ia < hi) {
    acc ^= use_item<KB>(A);
    if (ib >= hi) {
      break;
    }
    int const ic = claim();
    if (ic < hi) {
      load_item<KB>(A, w, used[ic / IPE], ic % IPE);
    }
    acc ^= use_item<KB>(Bi);
    if (ic >= hi) {
      break;
    }
    ia = ic;
    ib = claim();
    if (ib < hi) {
      load_item<KB>(Bi, w, used[ib / IPE], ib % IPE);
    }
  }
  if (acc == 0x12345678u) {
    out[blockIdx.x] = acc;
  }
}

// Scalar-atomic tickets (SAT): each wave takes its die's tickets with s_atomic_add glc (the die = the hardware XCC_ID, so a counter
// is only used by its own XCD's CUs: a scalar atomic is done in that XCD's L2) and waits for them with lgkmcnt, so the wait does
// not cover the wave's weight loads in flight (a vector atomic's return is counted by vmcnt, in order with the loads). The next
// ticket is issued before the current item's loads (AHEAD).
template <int KB, bool ULDS = false>
__global__ void __launch_bounds__(256, 1) w2_sat_kernel(char const *w, int const *used_g, unsigned *ctr, unsigned *out) {
  constexpr int IPE = EXP_KB / KB;
  // ULDS: the expert ids by scalar loads (a per-item vector load of the id is waited for with vmcnt(0): every weight load in
  // flight drained once per item)
  unsigned xv;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xv));
  int const die = (int)(xv & 7u);
  int const n = NUSED * IPE, lo = n * die / 8, hi = n * (die + 1) / 8;
  unsigned *const c = ctr + die * 32;
  auto issue = [&]() -> unsigned {
    unsigned v = 1;
    asm volatile("s_atomic_add %0, %1, 0x0 glc" : "+s"(v) : "s"(c) : "memory");
    return v;
  };
  auto finish = [&](unsigned v) -> int {
    asm volatile("s_waitcnt lgkmcnt(0)" : "+s"(v));
    int const it = lo + (int)v;
    return it < hi ? it : hi;
  };
  auto eid = [&](int i) -> int {
    if constexpr (ULDS) { // (a scalar load: waited for with lgkmcnt, not vmcnt)
      int e;
      asm volatile("s_load_dword %0, %1, %2\n s_waitcnt lgkmcnt(0)" : "=s"(e) : "s"(used_g), "s"(i * 4) : "memory");
      return e;
    } else {
      return used_g[i];
    }
  };
  unsigned acc = 0;
  Item<KB> A, B;
  int ia = finish(issue());
  unsigned tk = issue();
  if (ia < hi) {
    load_item<KB>(A, w, eid(ia / IPE), ia % IPE);
  }
  int ib = ia < hi ? finish(tk) : hi;
  tk = issue();
  if (ib < hi) {
    load_item<KB>(B, w, eid(ib / IPE), ib % IPE);
  }
  while (ia < hi) {
    acc ^= use_item<KB>(A);
    if (ib >= hi) {
      break;
    }
    int const ic = finish(tk);
    tk = issue();
    if (ic < hi) {
      load_item<KB>(A, w, eid(ic / IPE), ic % IPE);
    }
    acc ^= use_item<KB>(B);
    if (ic >= hi) {
      break;
    }
    ia = ic;
    ib = finish(tk);
    tk = issue();
    if (ib < hi) {
      load_item<KB>(B, w, eid(ib / IPE), ib % IPE);
    }
  }
  if (acc == 0x12345678u) {
    out[blockIdx.x] = acc;
  }
}

// SAT with a count per item (the coverage check)
template <int KB>
__global__ void __launch_bounds__(256, 1) w2_sat_count(unsigned *ctr) {
  constexpr int IPE = EXP_KB / KB;
  unsigned xv;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xv));
  int const die = (int)(xv & 7u), lane = threadIdx.x & 63;
  int const n = NUSED * IPE, lo = n * die / 8, hi = n * (die + 1) / 8;
  unsigned *const c = ctr + die * 32;
  for (;;) {
    unsigned v = 1;
    asm volatile("s_atomic_add %0, %1, 0x0 glc\n s_waitcnt lgkmcnt(0)" : "+s"(v) : "s"(c) : "memory");
    int const it = lo + (int)v;
    if (it >= hi) {
      break;
    }
    if (lane == 0) {
      atomicAdd(ctr + 8192 + it, 1u);
    }
  }
}

__global__ void flush_k(int const *p, size_t n, int *sink) {
  int acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc += p[i];
  if (acc == 0x7FFFFFFF) *sink = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  char *w;
  unsigned *ctr, *out;
  int *used, *flush;
  size_t const wbytes = (size_t)NEXP * EXP_BYTES, flush_n = (size_t)256 << 20;
  CHECK(hipMalloc(&w, wbytes));
  CHECK(hipMemset(w, 1, wbytes));
  CHECK(hipMalloc(&ctr, (2048 + NUSED * EXP_KB) * 4));
  CHECK(hipMalloc(&out, 256 * 4));
  CHECK(hipMalloc(&flush, flush_n * 4));
  CHECK(hipMemset(flush, 1, flush_n * 4));
  std::vector<int> ids(NEXP);
  for (int i = 0; i < NEXP; i++) ids[i] = i;
  std::mt19937 rng(1);
  std::shuffle(ids.begin(), ids.end(), rng);
  ids.resize(NUSED);
  std::sort(ids.begin(), ids.end());
  CHECK(hipMalloc(&used, NUSED * 4));
  CHECK(hipMemcpy(used, ids.data(), NUSED * 4, hipMemcpyHostToDevice));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  struct V {
    std::string name;
    int kb;
    void (*k)(char const *, int const *, unsigned *, unsigned *);
  };
  std::vector<V> vs = {
      {"24 KB tickets 2 in flight", 24, w2_kernel<24, false>}, {"24 KB static  2 in flight", 24, w2_kernel<24, true>},
      {"24 KB tickets 1 in flight", 24, w2_kernel<24, false, 1>},
  };
  {
    char *hq, *rs;
    CHECK(hipMalloc(&hq, (size_t)NEXP * 6144 + (size_t)NEXP * 28 * 2048));
    CHECK(hipMemset(hq, 3, (size_t)NEXP * 6144 + (size_t)NEXP * 28 * 2048));
    CHECK(hipMalloc(&rs, 64u << 20));
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hq), &hq, sizeof(hq)));
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_rs), &rs, sizeof(rs)));
    printf("W2X %d W2NC %d (1: h_q sc1 loads, 2: sc1 row stores, 4: h_q first, 8: scales)\n", W2X, W2NC);
  }
  double const bytes = (double)NUSED * EXP_BYTES;
  for (int round = 0; round < 2; round++) {
    for (auto &v : vs) {
      std::vector<double> ts;
      for (int rep = 0; rep < 9; rep++) {
        hipLaunchKernelGGL(flush_k, dim3(1024), dim3(256), 0, 0, flush, flush_n, flush + 1);
        CHECK(hipMemset(ctr, 0, 64 * 32 * 4));
        CHECK(hipEventRecord(e0, 0));
        hipLaunchKernelGGL(v.k, dim3(256), dim3(256), 0, 0, (char const *)w, (int const *)used, ctr, out);
        CHECK(hipEventRecord(e1, 0));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        if (rep) ts.push_back(ms * 1e3);
      }
      std::sort(ts.begin(), ts.end());
      double const t = ts[ts.size() / 2];
      printf("round %d %s: %d items, %.1f us, %.2f TB/s, %.2f GB/s per wave\n", round, v.name.c_str(), NUSED * (EXP_KB / v.kb),
             t, bytes / t / 1e6, bytes / t / 1e3 / 1024);
    }
  }
  return 0;
}
