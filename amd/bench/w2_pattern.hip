// Microbenchmark: the read rate of moe_experts_csr's W2 phase pattern on MI355X (gfx950). Question (10-24, dec320 T 64): the
// W2 items read ~3 GB/s per wave (8988 items of 24 KB in ~66 us, ~3.4 TB/s) while the W13 units read ~6.3 GB/s per wave;
// two items in flight per wave (STATIC_CSR_W2_AHEAD, dec322) and plain loads (STATIC_CSR_W2_PLAIN, dec325) did not help.
// Which part of an item's accesses sets the rate?
//
// 320 experts (K3 at bs 64) out of 896, each 28 items of 24 KB (8 row tiles x 3 K steps x 1 KB: the W2 weights of one expert,
// 672 KB, contiguous; experts at their ids' places in a 616 MB array). 256 workgroups x 4 waves; a wave takes items from
// its die's counter (one 128-B line per die, an agent-scope atomic per item, the ticket for the next item issued before the
// current item's loads, as the task), the die's share of the items expert-major. Per item:
//   W   24 nt loads of 1 KB per wave (64 lanes x 16 B)
//   S   + 8 4-B scale loads per lane (AITER's W2 scales: 2 dwords per 4-tile group x 2 K pairs, 256 B apart)
//   H   + 6 device-scope (sc1) 16-B loads per lane from a 1 MB h region (the item's pairs' h_q, as expert_w2c_hq)
// in flight: 1 (load, wait, use, next) or 2 (the next item's loads issued before the current item's wait).
// Also: STATIC = the items assigned statically (no counter). Rate = bytes of weights / kernel time (events, 1 GB read before).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 w2_pattern.hip -o w2_pattern
// Run:   ./w2_pattern
#include <algorithm>
#include <random>
#include <string>

#include "common.h"

typedef int v4i __attribute__((ext_vector_type(4)));
constexpr int NEXP = 896, NUSED = 320, IPE = 28, ITEM_KB = 24;
constexpr size_t EXP_BYTES = (size_t)IPE * ITEM_KB * 1024;

struct Item {
  v4i f[24];
  unsigned sc[8];
  uint4 h[6];
};

template <bool S, bool H>
__device__ __forceinline__ void load_item(Item &F, char const *w, unsigned const *sc, char const *h, int e, int it) {
  int const lane = threadIdx.x & 63;
  v4i const *wt = reinterpret_cast<v4i const *>(w + (size_t)e * EXP_BYTES + (size_t)it * ITEM_KB * 1024) + lane;
#pragma unroll
  for (int i = 0; i < 24; i++) {
    F.f[i] = __builtin_nontemporal_load(wt + (size_t)i * 64);
  }
  if constexpr (S) {
    unsigned const *st = sc + ((size_t)e * IPE + it) * 8 * 64 + lane;
#pragma unroll
    for (int d = 0; d < 8; d++) {
      F.sc[d] = st[d * 64];
    }
  }
  if constexpr (H) {
    __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(h), 0, 0x7FFFFFFF, 0x00020000);
    unsigned const base = (unsigned)(((e * 7 + it) % 1024) * 1024);
#pragma unroll
    for (int k = 0; k < 6; k++) {
      v4i const v = __builtin_bit_cast(v4i, __builtin_amdgcn_raw_buffer_load_b128(rs, base + (unsigned)((k * 64 + lane) * 16) % 1048576, 0, 16));
      F.h[k] = make_uint4(v.x, v.y, v.z, v.w);
    }
  }
}
template <bool S, bool H>
__device__ __forceinline__ unsigned use_item(Item const &F) {
  unsigned a = 0;
#pragma unroll
  for (int i = 0; i < 24; i++) {
    a ^= (unsigned)(F.f[i].x + F.f[i].y + F.f[i].z + F.f[i].w);
  }
  if constexpr (S) {
#pragma unroll
    for (int d = 0; d < 8; d++) {
      a ^= F.sc[d];
    }
  }
  if constexpr (H) {
#pragma unroll
    for (int k = 0; k < 6; k++) {
      a ^= F.h[k].x + F.h[k].w;
    }
  }
  return a;
}

// items of die x: [lo(x), lo(x + 1)) of the expert-major list (item i: expert used[i / IPE], group i % IPE)
template <bool S, bool H, int INFL, bool STATIC>
__global__ void __launch_bounds__(256, 1) w2_kernel(char const *w, unsigned const *sc, char const *h, int const *used,
                                                     unsigned *ctr, unsigned *out) {
  int const die = blockIdx.x & 7, wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int const n = NUSED * IPE, lo = n * die / 8, hi = n * (die + 1) / 8;
  unsigned *const c = ctr + die * 32;
  int const gw = (blockIdx.x >> 3) * 4 + wave, nw = 32 * 4; // (STATIC: the wave's index on its die, waves per die)
  auto take = [&](int k) -> int {
    if constexpr (STATIC) {
      return lo + gw + k * nw;
    } else {
      unsigned v = 0;
      if (lane == 0) {
        v = __hip_atomic_fetch_add(c, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      return lo + (int)__builtin_amdgcn_readfirstlane(v);
    }
  };
  unsigned acc = 0;
  Item A, B;
  int k = 0;
  int ia = take(k++);
  if (ia < hi) {
    load_item<S, H>(A, w, sc, h, used[ia / IPE], ia % IPE);
  }
  int ib = ia < hi ? take(k++) : hi;
  if (INFL == 2 && ib < hi) {
    load_item<S, H>(B, w, sc, h, used[ib / IPE], ib % IPE);
  }
  while (ia < hi) {
    if (INFL == 1 && ib < hi) { // (1 in flight: the next item's loads after this one is used)
      acc ^= use_item<S, H>(A);
      load_item<S, H>(A, w, sc, h, used[ib / IPE], ib % IPE);
      ia = ib;
      ib = take(k++);
      continue;
    }
    acc ^= use_item<S, H>(A);
    if (ib >= hi) {
      break;
    }
    int const ic = take(k++);
    if (ic < hi) {
      load_item<S, H>(A, w, sc, h, used[ic / IPE], ic % IPE);
    }
    acc ^= use_item<S, H>(B);
    if (ic >= hi) {
      break;
    }
    ia = ic;
    ib = take(k++);
    if (ib < hi) {
      load_item<S, H>(B, w, sc, h, used[ib / IPE], ib % IPE);
    }
  }
  if (acc == 0x12345678u) {
    out[blockIdx.x] = acc;
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
  print_device(0);
  char *w, *h;
  unsigned *sc, *ctr, *out;
  int *used, *flush;
  size_t const wbytes = (size_t)NEXP * EXP_BYTES, flush_n = (size_t)256 << 20;
  CHECK(hipMalloc(&w, wbytes));
  CHECK(hipMemset(w, 1, wbytes));
  CHECK(hipMalloc(&sc, (size_t)NEXP * IPE * 8 * 64 * 4));
  CHECK(hipMemset(sc, 2, (size_t)NEXP * IPE * 8 * 64 * 4));
  CHECK(hipMalloc(&h, 1 << 20));
  CHECK(hipMemset(h, 3, 1 << 20));
  CHECK(hipMalloc(&ctr, 8 * 32 * 4));
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
    void (*k)(char const *, unsigned const *, char const *, int const *, unsigned *, unsigned *);
  };
  std::vector<V> vs = {
      {"W     1 in flight", w2_kernel<false, false, 1, false>}, {"W     2 in flight", w2_kernel<false, false, 2, false>},
      {"W+S   1 in flight", w2_kernel<true, false, 1, false>},  {"W+S+H 1 in flight", w2_kernel<true, true, 1, false>},
      {"W+S+H 2 in flight", w2_kernel<true, true, 2, false>},   {"W     1 static   ", w2_kernel<false, false, 1, true>},
      {"W     2 static   ", w2_kernel<false, false, 2, true>},  {"W+S+H 2 static   ", w2_kernel<true, true, 2, true>},
  };
  double const bytes = (double)NUSED * EXP_BYTES;
  for (auto &v : vs) {
    std::vector<double> ts;
    for (int rep = 0; rep < 8; rep++) {
      hipLaunchKernelGGL(flush_k, dim3(1024), dim3(256), 0, 0, flush, flush_n, flush + 1);
      CHECK(hipMemset(ctr, 0, 8 * 32 * 4));
      CHECK(hipEventRecord(e0, 0));
      hipLaunchKernelGGL(v.k, dim3(256), dim3(256), 0, 0, (char const *)w, (unsigned const *)sc, (char const *)h,
                         (int const *)used, ctr, out);
      CHECK(hipEventRecord(e1, 0));
      CHECK(hipEventSynchronize(e1));
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      if (rep) ts.push_back(ms * 1e3);
    }
    double const t = median(ts);
    printf("%s: %d items of %d KB, %.1f us, %.2f TB/s, %.2f GB/s per wave\n", v.name.c_str(), NUSED * IPE, ITEM_KB, t,
           bytes / t / 1e6, bytes / t / 1e3 / 1024);
  }
  return 0;
}
