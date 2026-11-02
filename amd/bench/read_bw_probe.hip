// Microbenchmark: read speed on MI355X (gfx950) by number of CUs reading, cache hint, and where the data sits.
//
// A decode step at small batch is mostly weight reads, so the per-layer floor is bytes / read speed. Three places:
//   hbm      1 GB, read after reading 1 GB of other data (the 256 MB Infinity Cache and the 4 MB L2s hold none of it)
//   mall     128 MB, read twice in one launch: the second pass is timed (fits in the 256 MB Infinity Cache, not in L2)
//   l2       each die reads its own 2 MB, twice: the second pass is timed (fits in the die's 4 MB L2)
// Cache hints on the 16-byte loads (gfx950 bits, CDNA4 ISA guide):
//   default  no bits
//   nt       non-temporal (Fleet: activations stored this way so they do not push weights out of L2)
//   sc1 nt   "streaming" (Fleet's weight loads: allocate in L2, evict first)
// Workgroups: 256 threads, one per CU (96 KB LDS); `cus` of them read, the others exit. Time = the launch (hipEvent) for hbm;
// for mall / l2 the readers stamp the second pass with s_memrealtime (min start, max end).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 read_bw.hip -o read_bw
// Run:   ./read_bw            -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

enum Hint { H_DEFAULT = 0, H_NT = 1, H_SC1_NT = 2 };

template <int HINT>
__device__ inline uint4 load16(const uint4 *p) {
  uint4 v;
  if constexpr (HINT == H_DEFAULT) {
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(v) : "v"(p) : "memory");
  } else if constexpr (HINT == H_NT) {
    asm volatile("global_load_dwordx4 %0, %1, off nt" : "=v"(v) : "v"(p) : "memory");
  } else {
    asm volatile("global_load_dwordx4 %0, %1, off sc1 nt" : "=v"(v) : "v"(p) : "memory");
  }
  return v;
}

// passes over [base, base + n16) by `readers` workgroups; the last pass is stamped. per_die: each die's readers read their own
// slice (die * n16 .. ), else all readers split the whole range.
template <int HINT>
__global__ void read_kernel(const uint4 *buf, size_t n16, int readers, int passes, bool per_die, unsigned *slots,
                            unsigned long long *t, unsigned *sink) {
  extern __shared__ char lds[];
  __shared__ int my_slot;
  __shared__ unsigned my_die;
  if (threadIdx.x == 0) {
    lds[0] = 0;
    my_die = xcc_id();
    my_slot = (int)atomicAdd(&slots[per_die ? my_die : 0], 1u);
  }
  __syncthreads();
  int slot = my_slot;
  if (slot >= readers) return;
  const uint4 *base = per_die ? buf + my_die * n16 : buf;
  size_t chunk = (n16 + readers - 1) / readers, lo = slot * chunk, hi = lo + chunk < n16 ? lo + chunk : n16;
  unsigned acc = 0;
  unsigned long long nload = 0;
  for (int p = 0; p < passes; ++p) {
    __syncthreads();
    if (p == passes - 1 && threadIdx.x == 0) atomicMin(&t[0], realtime());
    // 4 loads in flight per thread
    for (size_t i = lo + threadIdx.x; i < hi; i += 4 * blockDim.x) {
      uint4 v0 = load16<HINT>(base + i);
      uint4 v1 = i + blockDim.x < hi ? load16<HINT>(base + i + blockDim.x) : make_uint4(0, 0, 0, 0);
      uint4 v2 = i + 2 * blockDim.x < hi ? load16<HINT>(base + i + 2 * blockDim.x) : make_uint4(0, 0, 0, 0);
      uint4 v3 = i + 3 * blockDim.x < hi ? load16<HINT>(base + i + 3 * blockDim.x) : make_uint4(0, 0, 0, 0);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      acc ^= v0.x ^ v1.y ^ v2.z ^ v3.w;
      nload += 1 + (i + blockDim.x < hi) + (i + 2 * blockDim.x < hi) + (i + 3 * blockDim.x < hi);
    }
    __syncthreads();
    if (p == passes - 1 && threadIdx.x == 0) atomicMax(&t[1], realtime());
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads
  atomicAdd((unsigned long long *)(sink + 2), nload);
}

using Kernel = void (*)(const uint4 *, size_t, int, int, bool, unsigned *, unsigned long long *, unsigned *);

int main() {
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int ncu = prop.multiProcessorCount;
  double tpn = wall_ticks_per_ns(0);
  const struct {
    const char *name;
    Kernel k;
  } hints[] = {{"default", read_kernel<H_DEFAULT>}, {"nt", read_kernel<H_NT>}, {"sc1 nt", read_kernel<H_SC1_NT>}};
  for (auto &h : hints) set_max_lds((const void *)h.k, ONE_PER_CU_LDS);
  const size_t GB = 1ull << 30;
  uint4 *buf, *other;
  unsigned *slots, *sink;
  unsigned long long *t;
  CHECK(hipMalloc(&buf, GB));
  CHECK(hipMalloc(&other, GB));
  CHECK(hipMemset(buf, 1, GB));
  CHECK(hipMemset(other, 2, GB));
  CHECK(hipMalloc(&slots, 64 * sizeof(unsigned)));
  CHECK(hipMalloc(&sink, 16));
  CHECK(hipMalloc(&t, 2 * sizeof(unsigned long long)));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::string json;
  auto run = [&](Kernel k, const uint4 *b, size_t n16, int readers, int passes, bool per_die) {
    CHECK(hipMemset(slots, 0, 64 * sizeof(unsigned)));
    unsigned long long init[2] = {~0ull, 0};
    CHECK(hipMemcpy(t, init, sizeof(init), hipMemcpyHostToDevice));
    CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(k, dim3(ncu), dim3(256), ONE_PER_CU_LDS, 0, b, n16, readers, passes, per_die, slots, t, sink);
    CHECK(hipGetLastError());
    CHECK(hipEventRecord(e1));
    CHECK(hipDeviceSynchronize());
    float ms;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    unsigned long long tt[2];
    CHECK(hipMemcpy(tt, t, sizeof(tt), hipMemcpyDeviceToHost));
    unsigned long long nl = 0;
    CHECK(hipMemcpy(&nl, sink + 2, 8, hipMemcpyDeviceToHost));
    CHECK(hipMemset(sink, 0, 16));
    fprintf(stderr, "loaded bytes %.3f GB of %.3f GB requested (x passes %d)\n", nl * 16.0 / 1e9, n16 * 16.0 * passes / 1e9, passes);
    if (getenv("READ_BW_DEBUG"))
      fprintf(stderr, "debug: readers %d passes %d per_die %d n16 %zu: launch %.0f ns, stamps %llu %llu\n", readers, passes,
              (int)per_die, n16, ms * 1e6, tt[0], tt[1]);
    return std::make_pair(ms * 1e6, (tt[1] - tt[0]) / tpn);  // (launch ns, stamped last pass ns)
  };
  for (auto &h : hints) {
    for (int cus : {8, 16, 32, 64, 128, 192, ncu}) {
      // hbm: 1 GB after 1 GB of other data, launch time
      run(h.k, other, GB / 16, ncu, 1, false);
      double ns = run(h.k, buf, GB / 16, cus, 1, false).first;
      // mall: 128 MB, second pass stamped
      double mall_ns = run(h.k, buf, (128ull << 20) / 16, cus, 2, false).second;
      double hbm = GB / ns, mall = (128ull << 20) / mall_ns;
      printf("%-8s %3d CUs: hbm %.2f TB/s  infinity cache %.2f TB/s\n", h.name, cus, hbm / 1e3, mall / 1e3);
      char row[200];
      snprintf(row, sizeof(row), "%s{\"hint\": \"%s\", \"cus\": %d, \"hbm_GBps\": %.0f, \"mall_GBps\": %.0f}", json.empty() ? "" : ", ",
               h.name, cus, hbm, mall);
      json += row;
    }
    // l2: each die's readers (4 per die .. 32 per die) read the die's own 2 MB twice
    for (int per_die : {4, 8, 16, 32}) {
      double l2_ns = run(h.k, buf, (2u << 20) / 16, per_die, 2, true).second;
      double l2 = 8.0 * (2u << 20) / l2_ns;
      printf("%-8s %2d CUs per die: l2 (8 dies x 2 MB) %.2f TB/s\n", h.name, per_die, l2 / 1e3);
      char row[200];
      snprintf(row, sizeof(row), ", {\"hint\": \"%s\", \"cus_per_die\": %d, \"l2_GBps\": %.0f}", h.name, per_die, l2);
      json += row;
    }
  }
  printf("{\"bench\": \"read_bw\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
