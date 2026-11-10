// Microbenchmark: system-scope reads of one small region by many CUs at once (the T 16 MoE tail: 49 latent_up tasks each read 128 KB
// of R's 917 KB exchange slots, every K part's 128 KB by 7 tasks) vs the same bytes from separate regions. Question (MI355X 10-24,
// dec261 / dec263): latent_up's first poll pass (64 KB per CU) took ~5.5 us whatever the rest of the GPU did; read_sys.hip (each CU
// its own 512 KB) reads ~50 GB/s per CU with few CUs.
//
// C workgroups (one per CU) read R bytes each, 16 loads (16 B, sc0 sc1) in flight per thread, then one wait, per round; layouts:
//   shared  workgroup w reads K part (w % P) of a region of P x R bytes (P = 7: each part read by C / 7 CUs), as [8 slots][16 rows]
//           of 1 KB pieces 7 KB apart (R's [slot][token][3584 bf16] rows, a 512-column K part)
//   private workgroup w reads its own region, the same piece layout
//   private contiguous: workgroup w reads its own R bytes contiguously
// Timed per workgroup with s_memrealtime (first load issued -> all landed), median over workgroups, median of 5 runs; cold (1 GB read
// first) and warm (the region read once just before).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 read_hot.hip -o read_hot
// Run:   ./read_hot
#include "common.h"

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v4u_t ld_sys(uint4 const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1" : "=v"(w) : "v"(p) : "memory");
  return w;
}

// LAYOUT 0 shared K parts, 1 private K-part layout, 2 private contiguous. bytes per WG = 8 slots x 16 rows x 1 KB = 128 KB.
template <int LAYOUT>
__global__ void __launch_bounds__(256, 1) hot_kernel(uint4 const *buf, int parts, unsigned long long *t, unsigned *sink) {
  int const w = blockIdx.x, tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  constexpr size_t ROW16 = 7168 / 16 * 2 / 2;          // a 3584-bf16 row = 7 KB = 448 vectors
  constexpr size_t SLOT16 = 16 * ROW16;                // 16 rows per slot
  size_t base16;
  int kp;
  if constexpr (LAYOUT == 0) {
    base16 = 0;
    kp = w % parts;
  } else {
    base16 = (size_t)w * 8 * SLOT16;
    kp = w % parts;
  }
  __syncthreads();
  unsigned long long const t0 = realtime();
  unsigned acc = 0;
  // 128 pieces of 1 KB (64 lanes x 16 B): piece = (slot r, row q); wave w takes rows wave + 4 j; 16 loads per thread per round
#pragma unroll 1
  for (int round = 0; round < 2; round++) {
    v4u_t v[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
      int const r = j & 7, q = wave + 4 * ((j >> 3) + 2 * round);   // slot r, row q (16 rows per slot)
      size_t off;
      if constexpr (LAYOUT == 2) {
        off = base16 + ((size_t)(r * 16 + q)) * 64 + lane;
      } else {
        off = base16 + (size_t)r * SLOT16 + (size_t)q * ROW16 + (size_t)kp * 64 + lane;
      }
      v[j] = ld_sys(buf + off);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
    for (int j = 0; j < 16; j++) acc ^= v[j].x ^ v[j].w;
  }
  unsigned long long const t1 = realtime();
  if (tid == 0) t[w] = t1 - t0;
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void flush_kernel(uint4 const *bg, size_t n16, unsigned *sink) {
  unsigned acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) acc ^= bg[i].x;
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  size_t const REGION = 8ull * 16 * 7168, BG = 1ull << 30;   // one [8][16][7 KB] region = 917504 B
  int const MAXC = 256;
  uint4 *buf, *bg;
  unsigned long long *t;
  unsigned *sink;
  CHECK(hipMalloc(&buf, REGION * MAXC));
  CHECK(hipMalloc(&bg, BG));
  CHECK(hipMalloc(&t, MAXC * 8));
  CHECK(hipMalloc(&sink, 16));
  CHECK(hipMemset(buf, 1, REGION * MAXC));
  CHECK(hipMemset(bg, 2, BG));
  double const tpn = wall_ticks_per_ns(0);
  char const *names[3] = {"shared region, K parts", "private regions, K parts", "private, contiguous"};
  void (*ks[3])(uint4 const *, int, unsigned long long *, unsigned *) = {hot_kernel<0>, hot_kernel<1>, hot_kernel<2>};
  for (int cus : {49, 98, 196}) {
    for (int warm : {0, 1}) {
      for (int k = 0; k < 3; k++) {
        std::vector<double> runs;
        for (int r = 0; r < 5; r++) {
          hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, 0, bg, BG / 16, sink);
          if (warm) hipLaunchKernelGGL(ks[k], dim3(cus), dim3(256), 0, 0, buf, 7, t, sink);
          hipLaunchKernelGGL(ks[k], dim3(cus), dim3(256), 0, 0, buf, 7, t, sink);
          CHECK(hipDeviceSynchronize());
          std::vector<unsigned long long> h(cus);
          CHECK(hipMemcpy(h.data(), t, cus * 8, hipMemcpyDeviceToHost));
          std::vector<double> us;
          for (int c = 0; c < cus; c++) us.push_back(h[c] / tpn / 1e3);
          runs.push_back(median(us));
        }
        double const u = median(runs);
        printf("%3d CUs, %s, %-26s: 128 KB per CU (2 rounds of 64 KB) in %5.2f us = %5.1f GB/s per CU\n", cus, warm ? "warm" : "cold",
               names[k], u, 131072.0 / u / 1e3);
      }
    }
  }
  return 0;
}
