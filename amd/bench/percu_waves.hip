// Microbenchmark: per-CU read rate when few CUs read, by waves per CU. The question: our megakernel runs one 256-thread
// workgroup per CU (one wave per SIMD); a tail task (moe_agg1, latent_up, attn_residual at T 64: 49-64 CUs busy) reads
// ~100-200 KB on its CU at ~25-36 GB/s (dec237: one WG per CU, 8..192 CUs, 36 GB/s per CU). Does the rate grow with more
// waves per SIMD (2, 4 or 8 workgroups per CU, held there by their LDS size) or with more loads in flight per wave?
// Case: NCU CUs (8, 64) each read their own 2 MB (HBM, after a 1 GB read elsewhere) or the same 2 MB twice (second pass:
// L2 hits); WPC workgroups per CU x 256 threads x DEPTH 16-B loads per round. Rate = bytes / (max end - min start) of the
// timed pass (s_memrealtime), GB/s per CU. The workgroups of one CU share its chunk (interleaved by 16 B per thread). Grid
// 256 x WPC (every CU holds WPC workgroups); per CU the first arrival takes an ordinal, CUs with ordinal >= NCU idle.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 percu_waves.hip -o percu_waves
// Run:   ./percu_waves
#include "common.h"

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

template <int DEPTH>
__global__ void __launch_bounds__(256) rd(const v4u_t *buf, size_t chunk16, int wpc, int ncu, int passes, unsigned *slot,
                                          unsigned long long *t0, unsigned long long *t1, unsigned *sink) {
  extern __shared__ char lds[];
  // every CU holds exactly wpc workgroups (grid 256 x wpc, LDS allows wpc per CU); per CU (key from HW_ID + XCC_ID) the first
  // arrival takes a CU ordinal, the others wait for it; only CUs with ordinal < ncu read (sub = the workgroup's index on its CU)
  __shared__ int s_ord, s_sub;
  if (threadIdx.x == 0) {
    unsigned const h = hw_id(), key = (xcc_id() << 8) | (((h >> 13) & 7) << 5) | (((h >> 12) & 1) << 4) | ((h >> 8) & 15);
    unsigned const sub = atomicAdd(&slot[4096 + key], 1u);
    if (sub == 0) {
      unsigned const o = atomicAdd(&slot[8192], 1u);
      __hip_atomic_store(&slot[key], o + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    unsigned o;
    while ((o = __hip_atomic_load(&slot[key], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) == 0) __builtin_amdgcn_s_sleep(1);
    s_ord = (int)o - 1;
    s_sub = (int)sub;
  }
  __syncthreads();
  int const cu = s_ord, sub = s_sub;
  unsigned acc = 0;
  if (cu < ncu && sub < wpc) {
    const v4u_t *p = buf + (size_t)cu * chunk16;
    for (int pass = 0; pass < passes; pass++) {
      __syncthreads();
      unsigned long long s = realtime();
      size_t const stride = (size_t)256 * wpc;
      for (size_t i = (size_t)sub * 256 + threadIdx.x; i < chunk16; i += stride * DEPTH) {
        v4u_t v[DEPTH];
#pragma unroll
        for (int d = 0; d < DEPTH; d++) {
          size_t const k = i + (size_t)d * stride;
          v[d] = k < chunk16 ? __builtin_nontemporal_load(p + k) : v4u_t{0, 0, 0, 0};
        }
#pragma unroll
        for (int d = 0; d < DEPTH; d++) acc ^= v[d][d & 3];
      }
      __syncthreads();
      unsigned long long e = realtime();
      if (threadIdx.x == 0 && pass == passes - 1) {
        t0[cu * wpc + sub] = s;
        t1[cu * wpc + sub] = e;
      }
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc + lds[0];
}

__global__ void flush(const v4u_t *big, size_t n16, unsigned *sink) {
  unsigned acc = 0;
  for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n16; i += (size_t)gridDim.x * 256) acc ^= __builtin_nontemporal_load(big + i)[0];
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

template <int DEPTH>
double run(const v4u_t *buf, const v4u_t *big, size_t big16, int ncu, int wpc, bool l2, unsigned long long *t0,
           unsigned long long *t1, unsigned *sink, unsigned *slot, double tpn) {
  size_t const chunk = 2ull << 20, chunk16 = chunk / 16;
  int const lds = (160 * 1024) / wpc - 1024; // wpc workgroups fit one CU, wpc + 1 do not
  set_max_lds((const void *)rd<DEPTH>, lds);
  std::vector<double> r;
  for (int rep = 0; rep < 5; rep++) {
    flush<<<1024, 256>>>(big, big16, sink);
    CHECK(hipMemset(slot, 0, 8200 * 4));
    rd<DEPTH><<<256 * wpc, 256, lds>>>(buf, chunk16, wpc, ncu, l2 ? 2 : 1, slot, t0, t1, sink);
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long long> a(ncu * wpc), b(ncu * wpc);
    CHECK(hipMemcpy(a.data(), t0, a.size() * 8, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(b.data(), t1, b.size() * 8, hipMemcpyDeviceToHost));
    unsigned long long lo = ~0ull, hi = 0;
    for (size_t i = 0; i < a.size(); i++) lo = std::min(lo, a[i]), hi = std::max(hi, b[i]);
    double const us = (hi - lo) / tpn / 1e3;
    r.push_back(chunk / us / 1e3); // GB/s per CU (each CU read its 2 MB)
  }
  return median(r);
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  CHECK(hipSetDevice(0));
  size_t const BIG = 1ull << 30;
  v4u_t *buf, *big;
  unsigned long long *t0, *t1;
  unsigned *sink, *slot;
  CHECK(hipMalloc(&slot, 8200 * 4));
  CHECK(hipMalloc(&buf, 256ull * (2ull << 20)));
  CHECK(hipMalloc(&big, BIG));
  CHECK(hipMemset(buf, 1, 256ull * (2ull << 20)));
  CHECK(hipMemset(big, 2, BIG));
  CHECK(hipMalloc(&t0, 4096 * 8));
  CHECK(hipMalloc(&t1, 4096 * 8));
  CHECK(hipMalloc(&sink, 16));
  double const tpn = wall_ticks_per_ns(0);
  for (int l2 : {0, 1})
    for (int ncu : {8, 64, 256})
      for (int wpc : {1, 2, 4, 8}) {
        double const d4 = run<4>(buf, big, BIG / 16, ncu, wpc, l2, t0, t1, sink, slot, tpn);
        double const d16 = run<16>(buf, big, BIG / 16, ncu, wpc, l2, t0, t1, sink, slot, tpn);
        printf("%s %3d CUs, %d WG (%2d waves) per CU: depth 4 %6.1f GB/s per CU, depth 16 %6.1f GB/s per CU\n",
               l2 ? "l2 " : "hbm", ncu, wpc, 4 * wpc, d4, d16);
      }
  return 0;
}
