// Microbenchmark: how much one memory-bound pass varies from GPU to GPU and launch to launch. In the server the KDA T 64
// kernel's expert phase ends on the slowest rank a median 8.5 us (p90 19) after the median rank (srvd153 stamps, 67 layers),
// with the same work on every rank (TP8) and a different slowest rank from layer to layer; lined up in the harness (dec455) the
// 8 GPUs' expert ends spread 11 us at T 64 and 24 us at T 32 with starts within 2 us. Is a plain read just as uneven?
// Case: every GPU reads its own BYTES (default 800 MB, ~the T 64 expert stream) with a grid-stride nt read (1024 x 256 threads,
// 6.2 TB/s on GPU 0 in read_bw), after a 1 GB read of other data; REPS rounds. Modes: "all" = the 8 GPUs launched together each
// round (one host thread, back to back), "solo" = one GPU at a time. Time per GPU from hipEvents around its read kernel.
// Prints per GPU the median / min / max time, and per round the spread (max - min over GPUs) in "all" mode.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 gpu_var.hip -o gpu_var
// Run:   ./gpu_var [MB] [REPS]
#include "common.h"

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

__global__ void __launch_bounds__(256) rd(const v4u_t *buf, size_t n16, unsigned *sink) {
  unsigned acc = 0;
  for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n16; i += (size_t)gridDim.x * 256) acc ^= __builtin_nontemporal_load(buf + i)[0];
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  size_t const MB = argc > 1 ? atoi(argv[1]) : 800, BIG = 1ull << 30, n16 = (MB << 20) / 16;
  int const REPS = argc > 2 ? atoi(argv[2]) : 30;
  int ng = 0;
  CHECK(hipGetDeviceCount(&ng));
  std::vector<v4u_t *> buf(ng), big(ng);
  std::vector<unsigned *> sink(ng);
  std::vector<hipEvent_t> e0(ng), e1(ng);
  for (int g = 0; g < ng; g++) {
    CHECK(hipSetDevice(g));
    CHECK(hipMalloc(&buf[g], MB << 20));
    CHECK(hipMalloc(&big[g], BIG));
    CHECK(hipMalloc(&sink[g], 16));
    CHECK(hipMemset(buf[g], 1, MB << 20));
    CHECK(hipMemset(big[g], 2, BIG));
    CHECK(hipEventCreate(&e0[g]));
    CHECK(hipEventCreate(&e1[g]));
    CHECK(hipDeviceSynchronize());
  }
  for (int solo = 0; solo < 2; solo++) {
    std::vector<std::vector<double>> t(ng);
    std::vector<double> spread;
    for (int r = 0; r < REPS + 2; r++) {
      for (int g0 = 0; g0 < (solo ? ng : 1); g0++) {
        int const lo = solo ? g0 : 0, hi = solo ? g0 + 1 : ng;
        for (int g = lo; g < hi; g++) { // the flush: 1 GB of other data on each GPU
          CHECK(hipSetDevice(g));
          rd<<<1024, 256>>>(big[g], BIG / 16, sink[g]);
        }
        for (int g = lo; g < hi; g++) {
          CHECK(hipSetDevice(g));
          CHECK(hipDeviceSynchronize());
        }
        for (int g = lo; g < hi; g++) {
          CHECK(hipSetDevice(g));
          CHECK(hipEventRecord(e0[g], 0));
          rd<<<1024, 256>>>(buf[g], n16, sink[g]);
          CHECK(hipEventRecord(e1[g], 0));
        }
        for (int g = lo; g < hi; g++) {
          CHECK(hipSetDevice(g));
          CHECK(hipEventSynchronize(e1[g]));
        }
      }
      if (r < 2) continue; // (warm-up rounds)
      double mn = 1e30, mx = 0;
      for (int g = 0; g < ng; g++) {
        float ms = 0;
        CHECK(hipSetDevice(g));
        CHECK(hipEventElapsedTime(&ms, e0[g], e1[g]));
        t[g].push_back(ms * 1e3);
        mn = std::min(mn, (double)ms * 1e3);
        mx = std::max(mx, (double)ms * 1e3);
      }
      spread.push_back(mx - mn);
    }
    printf("== %s: %zu MB per GPU, %d rounds\n", solo ? "solo (one GPU at a time)" : "all 8 GPUs together", MB, REPS);
    for (int g = 0; g < ng; g++) {
      auto v = t[g];
      std::sort(v.begin(), v.end());
      printf("GPU %d: median %7.1f us (%.2f TB/s) min %7.1f max %7.1f p90 %7.1f\n", g, median(v), (MB << 20) / median(v) / 1e6,
             v.front(), v.back(), v[(v.size() * 9) / 10]);
    }
    if (!solo) {
      std::sort(spread.begin(), spread.end());
      printf("per round max - min over GPUs: median %.1f us, p90 %.1f, max %.1f\n", median(spread), spread[(spread.size() * 9) / 10],
             spread.back());
    }
  }
  return 0;
}
