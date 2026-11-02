// Microbenchmark: read speed on MI355X (gfx950) by bytes in flight per CU, number of CUs reading, cache hint, and where the
// data sits.
//
// A decode step at small batch is mostly weight reads, so the per-layer floor is bytes / read speed. Three places:
//   hbm      1 GB, read after reading 1 GB of other data (the 256 MB Infinity Cache and the 4 MB L2s hold none of it)
//   mall     128 MB, read twice in one launch: the second pass is timed (fits in the 256 MB Infinity Cache, not in L2)
//   l2       each die reads its own 2 MB, twice: the second pass is timed (fits in the die's 4 MB L2)
// Cache hints on the 16-byte loads (gfx950 bits, CDNA4 ISA guide):
//   default  no bits
//   nt       non-temporal (Fleet: activations stored this way so they do not push weights out of L2)
//   sc1 nt   "streaming" (Fleet's weight loads: allocate in L2, evict first)
// Bytes in flight per CU = workgroups per CU (WPC: 1, 2 or 4, held there by their LDS size) x 256 threads x DEPTH loads
// (4, 8 or 16, issued together, then one wait) x 16 B: 16 KB .. 256 KB. Run 3 (10-10, depth 4, one workgroup per CU, 16 KB):
// 2.5 TB/s from HBM on 256 CUs, while a plain HIP read (1024 x 256 threads) reached 6.2 TB/s: the reads are limited by
// what is in flight, and this sweep finds how much is needed.
// Two layouts of the reads: chunked (each workgroup one contiguous chunk of n / readers) and interleaved (vector i read by
// workgroup (i / 256) % readers, as a grid-stride loop: at any moment the workgroups read neighbouring addresses). Run 4
// (10-10, chunked): 2.1 / 3.6 / 4.7 TB/s at 1 / 2 / 4 workgroups per CU whatever the depth; a plain grid-stride HIP read 6.2.
// Time = the launch (hipEvent) for hbm; for mall / l2 the readers stamp the second pass with s_memrealtime (min start, max end).
// Every case checks the number of 16-byte loads done against the bytes asked for.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 read_bw.hip -o read_bw
// Run:   ./read_bw            -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

enum Hint { H_DEFAULT = 0, H_NT = 1, H_SC1_NT = 2 };
const char *HINT_NAME[3] = {"default", "nt", "sc1 nt"};

// Buffer loads from the workgroup's base (one descriptor, the same in every lane) at a per-lane byte offset: the compiler tracks
// them (an inline-asm load without a wait inside is not tracked: the compiler reuses its destination registers before the data
// lands; the first version of this benchmark lost its loop counter that way and read 4 MB of 1 GB). Cache bits (gfx950): sc0 1,
// nt 2, sc1 16.
typedef unsigned v4u_t __attribute__((ext_vector_type(4)));
constexpr int AUX_OF[3] = {0, 2, 16 | 2};

template <int HINT>
__device__ inline v4u_t load16(__amdgpu_buffer_rsrc_t rsrc, size_t i) {
  return __builtin_amdgcn_raw_buffer_load_b128(rsrc, (unsigned)(i * 16), 0, AUX_OF[HINT]);
}

// passes over [base, base + n16) by `readers` workgroups; the last pass is stamped. per_die: each die's readers read their own
// slice (die * n16 .. ), else all readers split the whole range.
template <int HINT, int DEPTH>
__global__ void read_kernel(const uint4 *buf, size_t n16, int readers, int passes, bool per_die, bool inter, unsigned *slots,
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
  // 0x00020000: gfx9's last descriptor dword (as composable_kernel). The range is exactly the data: a load past it returns 0
  // without a memory access, so the loads below need no bounds branch (a branch around each load made the compiler wait for
  // it: run 4 measured one load in flight per wave whatever DEPTH)
  __amdgpu_buffer_rsrc_t rsrc =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<uint4 *>(base), 0, (int)(n16 * 16), 0x00020000);
  size_t chunk = (n16 + readers - 1) / readers, lo = slot * chunk, hi = lo + chunk < n16 ? lo + chunk : n16;
  unsigned acc = 0;
  unsigned long long nload = 0;  // 16-byte loads done: checked against the bytes asked for
  for (int p = 0; p < passes; ++p) {
    __syncthreads();
    if (p == passes - 1 && threadIdx.x == 0) atomicMin(&t[0], realtime());
    // chunked: [lo, hi) in steps of the workgroup; interleaved: from slot x 256 in steps of all readers' threads
    size_t const first = inter ? (size_t)slot * blockDim.x + threadIdx.x : lo + threadIdx.x,
                 end = inter ? n16 : hi, step = inter ? (size_t)readers * blockDim.x : blockDim.x;
    for (size_t i = first; i < end; i += DEPTH * step) {
      v4u_t v[DEPTH];
#pragma unroll
      for (int d = 0; d < DEPTH; ++d) v[d] = load16<HINT>(rsrc, i + d * step < end ? i + d * step : n16);
#pragma unroll
      for (int d = 0; d < DEPTH; ++d) {
        acc ^= v[d].x ^ v[d].w;
        nload += i + d * step < end;
      }
    }
    __syncthreads();
    if (p == passes - 1 && threadIdx.x == 0) atomicMax(&t[1], realtime());
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads
  atomicAdd((unsigned long long *)(sink + 2), nload);
}

using Kernel = void (*)(const uint4 *, size_t, int, int, bool, bool, unsigned *, unsigned long long *, unsigned *);

// kernels[hint][depth index]: depth 4, 8, 16
const int DEPTHS[3] = {4, 8, 16};
Kernel kernels[3][3] = {
    {read_kernel<0, 4>, read_kernel<0, 8>, read_kernel<0, 16>},
    {read_kernel<1, 4>, read_kernel<1, 8>, read_kernel<1, 16>},
    {read_kernel<2, 4>, read_kernel<2, 8>, read_kernel<2, 16>},
};
// LDS per workgroup that holds WPC workgroups per CU (160 KB per CU): 1 -> 96 KB, 2 -> 64 KB, 4 -> 36 KB
int lds_for(int wpc) { return wpc == 1 ? 96 * 1024 : wpc == 2 ? 64 * 1024 : 36 * 1024; }

int main() {
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int ncu = prop.multiProcessorCount;
  double tpn = wall_ticks_per_ns(0);
  for (auto &row : kernels)
    for (auto k : row) set_max_lds((const void *)k, 96 * 1024);
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
  auto add = [&](const char *row) {
    json += json.empty() ? "" : ", ";
    json += row;
  };
  // one launch: `grid` workgroups (LDS for wpc per CU), `readers` of them read
  auto run = [&](Kernel k, int grid, int wpc, const uint4 *b, size_t n16, int readers, int passes, bool per_die,
                 bool inter = false) {
    CHECK(hipMemset(slots, 0, 64 * sizeof(unsigned)));
    CHECK(hipMemset(sink, 0, 16));
    unsigned long long init[2] = {~0ull, 0};
    CHECK(hipMemcpy(t, init, sizeof(init), hipMemcpyHostToDevice));
    CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(k, dim3(grid), dim3(256), lds_for(wpc), 0, b, n16, readers, passes, per_die, inter, slots, t, sink);
    CHECK(hipGetLastError());
    CHECK(hipEventRecord(e1));
    CHECK(hipDeviceSynchronize());
    float ms;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    unsigned long long tt[2], loads;
    CHECK(hipMemcpy(tt, t, sizeof(tt), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&loads, sink + 2, 8, hipMemcpyDeviceToHost));
    size_t want = n16 * passes * (per_die ? 8 : 1);
    if (loads != want) {
      printf("WRONG: %llu 16-byte loads done, %zu asked for (readers %d, passes %d)\n", loads, want, readers, passes);
      exit(1);
    }
    return std::make_pair(ms * 1e6, (tt[1] - tt[0]) / tpn);  // (launch ns, stamped last pass ns)
  };
  // hbm: 1 GB after 1 GB of other data, launch time; median of 5
  auto hbm = [&](Kernel k, int wpc, int readers, bool inter = false) {
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
      run(k, ncu * wpc, wpc, other, GB / 16, ncu * wpc, 1, false);
      v.push_back(GB / run(k, ncu * wpc, wpc, buf, GB / 16, readers, 1, false, inter).first);
    }
    return median(v);  // bytes per ns = GB/s
  };
  // 1. all CUs read from HBM: bytes in flight per CU (workgroups per CU x depth) for each hint, chunked and interleaved
  double best = 0;
  int best_h = 0, best_d = 0, best_w = 1, best_i = 0;
  for (int inter = 0; inter < 2; ++inter)
    for (int h = 0; h < 3; ++h)
      for (int wpc : {1, 2, 4})
        for (int d = 0; d < 3; ++d) {
          double bw = hbm(kernels[h][d], wpc, ncu * wpc, inter);
          int kb = wpc * 256 * DEPTHS[d] * 16 / 1024;
          printf("%-11s %-8s all %d CUs, %d workgroups per CU x depth %2d = %3d KB in flight per CU: hbm %.2f TB/s\n",
                 inter ? "interleaved" : "chunked", HINT_NAME[h], ncu, wpc, DEPTHS[d], kb, bw / 1e3);
          char row[240];
          snprintf(row, sizeof(row), "{\"layout\": \"%s\", \"hint\": \"%s\", \"cus\": %d, \"wpc\": %d, \"depth\": %d, "
                   "\"kb_in_flight_per_cu\": %d, \"hbm_GBps\": %.0f}", inter ? "interleaved" : "chunked", HINT_NAME[h], ncu, wpc,
                   DEPTHS[d], kb, bw);
          add(row);
          if (bw > best) best = bw, best_h = h, best_d = d, best_w = wpc, best_i = inter;
        }
  printf("best: %s, %s, %d workgroups per CU x depth %d: %.2f TB/s\n", best_i ? "interleaved" : "chunked", HINT_NAME[best_h],
         best_w, DEPTHS[best_d], best / 1e3);
  // 2. fewer CUs reading (depth 16, one workgroup per CU = 64 KB in flight per CU, interleaved): hbm and the Infinity Cache
  for (int h = 0; h < 3; ++h) {
    Kernel k = kernels[h][2];
    for (int cus : {8, 16, 32, 64, 128, 192, ncu}) {
      double bw = hbm(k, 1, cus, true);
      double mall_ns = run(k, ncu, 1, buf, (128ull << 20) / 16, cus, 2, false, true).second;
      double mall = (128ull << 20) / mall_ns;
      printf("%-8s %3d CUs, depth 16, interleaved: hbm %.2f TB/s  infinity cache %.2f TB/s\n", HINT_NAME[h], cus, bw / 1e3, mall / 1e3);
      char row[200];
      snprintf(row, sizeof(row), "{\"hint\": \"%s\", \"cus\": %d, \"wpc\": 1, \"depth\": 16, \"hbm_GBps\": %.0f, \"mall_GBps\": %.0f}",
               HINT_NAME[h], cus, bw, mall);
      add(row);
    }
    // l2: each die's readers (4 per die .. 32 per die) read the die's own 2 MB twice
    for (int per_die : {4, 8, 16, 32}) {
      double l2_ns = run(k, ncu, 1, buf, (2u << 20) / 16, per_die, 2, true, true).second;
      double l2 = 8.0 * (2u << 20) / l2_ns;
      printf("%-8s %2d CUs per die, depth 16, interleaved: l2 (8 dies x 2 MB) %.2f TB/s\n", HINT_NAME[h], per_die, l2 / 1e3);
      char row[200];
      snprintf(row, sizeof(row), "{\"hint\": \"%s\", \"cus_per_die\": %d, \"depth\": 16, \"l2_GBps\": %.0f}", HINT_NAME[h], per_die, l2);
      add(row);
    }
  }
  printf("{\"bench\": \"read_bw\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
