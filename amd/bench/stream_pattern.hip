// Microbenchmark: HBM read rate of the GEMM / expert weight-stream pattern on MI355X (gfx950): every wave reads 1 KB per
// instruction (64 lanes x 16 B, global_load_lds_dwordx4 into a per-wave LDS ring: no registers, so the depth in flight is
// exactly the ring's, kept by explicit s_waitcnt vmcnt). One workgroup of 4 waves per CU, 256 workgroups, 1 GB read before
// each launch (L2 / MALL empty). Layouts of a workgroup's bytes (CHUNK per workgroup, contiguous per workgroup):
//   0 wave:   wave w streams its own quarter of the chunk (4 streams per CU, as gemm_tile's row tiles per wave)
//   1 inter:  the 4 waves take 1 KB pieces in turn (piece 4 k + w: one stream per CU, as read_bw's chunked)
// Depth D: K steps (1 KB copies) in flight per wave. Rate = bytes / (median kernel time, events).
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 stream_pattern.hip -o stream_pattern
// Run:   ./stream_pattern [chunk KB per workgroup = 384]   -> one line per case, then a JSON summary
#include <string>

#include "common.h"

template <int D, int LAYOUT, int AUX = 2>
__global__ void stream_kernel(char const *src, size_t chunk, unsigned *sink) {
  extern __shared__ char lds[];
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  char const *const base = src + (size_t)blockIdx.x * chunk;
  int const steps = (int)(chunk / 1024 / 4); // 1 KB pieces per wave
  char *const ring = lds + wave * D * 1024;
  auto piece = [&](int k) -> char const * {
    size_t const p = LAYOUT == 0 ? (size_t)wave * steps + k : (size_t)4 * k + wave;
    return base + p * 1024 + lane * 16;
  };
  auto dma = [&](int k) {
    __builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void *)(const_cast<char *>(piece(k))),
                                     (__attribute__((address_space(3))) void *)(ring + (k % D) * 1024), 16, 0, AUX);
  };
  for (int k = 0; k < D && k < steps; k++) dma(k);
  unsigned acc = 0;
  for (int k = 0; k + D < steps; k++) {
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(D - 1) : "memory");
    acc ^= *reinterpret_cast<unsigned const *>(ring + (k % D) * 1024 + lane * 16);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dma(k + D);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  for (int k = steps > D ? steps - D : 0; k < steps; k++) acc ^= *reinterpret_cast<unsigned const *>(ring + (k % D) * 1024 + lane * 16);
  if (acc == 0x12345678u) sink[0] = acc; // (keeps the reads)
}

// warm: after the flush, one untimed pass (default cache policy) over the same bytes, then the timed pass: whether a
// kernel's reads hit what the previous kernel loaded into the L2s (the same workgroup -> the same die)
template <int D, int LAYOUT, int AUX = 2, bool WARM = false>
double run(char const *src, size_t chunk, unsigned *sink, char *flush, size_t flush_bytes) {
  set_max_lds((const void *)stream_kernel<D, LAYOUT, AUX>, 4 * D * 1024);
  hipLaunchKernelGGL((stream_kernel<D, LAYOUT, AUX>), dim3(256), dim3(256), 4 * D * 1024, 0, src, chunk, sink);
  CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  std::vector<double> us;
  for (int i = 0; i < 20; i++) {
    CHECK(hipMemsetAsync(flush, i & 0x7F, flush_bytes, 0));
    if (WARM) {
      hipLaunchKernelGGL((stream_kernel<D, LAYOUT, 0>), dim3(256), dim3(256), 4 * D * 1024, 0, src, chunk, sink);
    }
    CHECK(hipEventRecord(a, 0));
    hipLaunchKernelGGL((stream_kernel<D, LAYOUT, AUX>), dim3(256), dim3(256), 4 * D * 1024, 0, src, chunk, sink);
    CHECK(hipEventRecord(b, 0));
    CHECK(hipEventSynchronize(b));
    float ms;
    CHECK(hipEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1e3);
  }
  return median(us);
}

int main(int argc, char **argv) {
  size_t const chunk = (size_t)(argc > 1 ? atoi(argv[1]) : 384) * 1024;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  size_t const bytes = chunk * 256, flush_bytes = (size_t)1 << 30;
  char *src, *flush;
  unsigned *sink;
  CHECK(hipMalloc(&src, bytes));
  CHECK(hipMalloc(&flush, flush_bytes));
  CHECK(hipMalloc(&sink, 16));
  CHECK(hipMemset(src, 1, bytes));
  std::string json;
  auto line = [&](char const *lay, int d, double us) {
    double const tbs = bytes / (us * 1e-6) / 1e12;
    printf("layout %-5s depth %2d (%3d KB per CU in flight): %7.2f us for %zu KB -> %.2f TB/s\n", lay, d, 4 * d, us,
           bytes >> 10, tbs);
    char row[160];
    snprintf(row, sizeof(row), "%s{\"layout\": \"%s\", \"depth\": %d, \"us\": %.3f, \"tbs\": %.3f}", json.empty() ? "" : ", ", lay,
             d, us, tbs);
    json += row;
  };
  line("wave", 4, run<4, 0>(src, chunk, sink, flush, flush_bytes));
  line("wave", 8, run<8, 0>(src, chunk, sink, flush, flush_bytes));
  line("wave", 16, run<16, 0>(src, chunk, sink, flush, flush_bytes));
  line("wave", 32, run<32, 0>(src, chunk, sink, flush, flush_bytes));
  line("inter", 4, run<4, 1>(src, chunk, sink, flush, flush_bytes));
  line("inter", 8, run<8, 1>(src, chunk, sink, flush, flush_bytes));
  line("inter", 16, run<16, 1>(src, chunk, sink, flush, flush_bytes));
  line("inter", 32, run<32, 1>(src, chunk, sink, flush, flush_bytes));
  line("cold0", 16, run<16, 0, 0>(src, chunk, sink, flush, flush_bytes));
  line("warm0", 16, run<16, 0, 0, true>(src, chunk, sink, flush, flush_bytes));
  line("warmN", 16, run<16, 0, 2, true>(src, chunk, sink, flush, flush_bytes));
  printf("{\"bench\": \"stream_pattern\", \"chunk_kb\": %zu, \"rows\": [%s]}\n", chunk >> 10, json.c_str());
  return 0;
}
