// Microbenchmark: does traffic on a few hot lines (device-scope atomics, or polling loads of counters) slow the bulk weight
// reads of the other waves on MI355X (gfx950)?
//
// moe_experts (MI355X 10-12, TP8 trace): the W13 units that started at 40-44 us took 15.9 us (8 at other times) while ~380
// waves that had no W13 unit left went through the queue counters (8 relaxed agent-scope loads, then an atomic add, again
// until a queue had items), and the weight reads ran at 1.9 TB/s.
//
// Every CU runs one workgroup of 4 waves. Waves 0 .. 4 - H - 1 read: together they stream 1 GB once with non-temporal
// 16-byte loads (moe_experts' wload), 8 loads per lane in flight. The other H waves per CU do, until the readers are done:
//   idle     exit at once
//   sleep    s_sleep in a loop (no memory traffic)
//   add L    lane 0: atomic add (relaxed, agent scope, value returned) on one of L counters (128 B apart), back to back
//   load8    lane 0: 8 relaxed agent-scope loads of 8 counters (128 B apart), back to back (take_finish's queue scan)
// The readers' time is from the first reader's start to the last reader's end (s_memrealtime). Before each case 1 GB of
// other data is read, so the read comes from HBM.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 hot_line.hip -o hot_line
// Run:   ./hot_line   -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

enum Kind { K_IDLE = 0, K_SLEEP = 1, K_ADD = 2, K_LOAD8 = 3 };

typedef int v4i_t __attribute__((ext_vector_type(4)));

__global__ void hot_kernel(const v4i_t *buf, size_t n16, int hw, int kind, int lines, unsigned *ctr, unsigned *done,
                           unsigned long long *t, unsigned long long *ops, unsigned *sink) {
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int const rpc = 4 - hw;                       // reading waves per CU
  int const nreaders = gridDim.x * rpc;
  if (wave < rpc) {
    int const r = blockIdx.x * rpc + wave;
    size_t const chunk = (n16 + nreaders - 1) / nreaders;
    size_t const lo = r * chunk, hi = lo + chunk < n16 ? lo + chunk : n16;
    if (lane == 0) atomicMin(&t[0], realtime());
    unsigned acc = 0;
    for (size_t i = lo + lane; i < hi; i += 8 * 64) {
      v4i_t v[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        v[j] = i + j * 64 < hi ? __builtin_nontemporal_load(buf + i + j * 64) : v4i_t{0, 0, 0, 0};
      }
#pragma unroll
      for (int j = 0; j < 8; j++) acc ^= (unsigned)v[j][j & 3];
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads
    if (lane == 0) {
      atomicMax(&t[1], realtime());
      __hip_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return;
  }
  if (kind == K_IDLE) return;
  unsigned long long n = 0;
  unsigned x = 0;
  int const hwave = blockIdx.x * hw + (wave - rpc);
  for (int it = 0;; it++) {
    if ((it & 15) == 0 &&
        __hip_atomic_load(done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= (unsigned)nreaders) {
      break;
    }
    if (kind == K_SLEEP) {
      __builtin_amdgcn_s_sleep(8);
    } else if (kind == K_ADD) {
      if (lane == 0) {
        x += __hip_atomic_fetch_add(ctr + 32 * ((hwave + it) % lines), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      x = __builtin_amdgcn_readfirstlane(x);
    } else {
      if (lane == 0) {
        unsigned c[8];
#pragma unroll
        for (int j = 0; j < 8; j++) c[j] = __hip_atomic_load(ctr + 32 * j, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#pragma unroll
        for (int j = 0; j < 8; j++) x += c[j];
      }
      x = __builtin_amdgcn_readfirstlane(x);
    }
    n++;
  }
  if (x == 0x9e3779b9u) sink[1] = x;
  if (lane == 0) atomicAdd(ops, n);
}

int main() {
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int const ncu = prop.multiProcessorCount;
  double const tpn = wall_ticks_per_ns(0);
  set_max_lds((const void *)hot_kernel, ONE_PER_CU_LDS);
  const size_t GB = 1ull << 30;
  v4i_t *buf, *other;
  unsigned *ctr, *done, *sink;
  unsigned long long *t, *ops;
  CHECK(hipMalloc(&buf, GB));
  CHECK(hipMalloc(&other, GB));
  CHECK(hipMemset(buf, 1, GB));
  CHECK(hipMemset(other, 2, GB));
  CHECK(hipMalloc(&ctr, 64 * 128));
  CHECK(hipMalloc(&done, 4));
  CHECK(hipMalloc(&sink, 16));
  CHECK(hipMalloc(&t, 16));
  CHECK(hipMalloc(&ops, 8));
  auto run = [&](const v4i_t *b, int hw, int kind, int lines, double *ops_per_us) {
    CHECK(hipMemset(ctr, 0, 64 * 128));
    CHECK(hipMemset(done, 0, 4));
    CHECK(hipMemset(ops, 0, 8));
    unsigned long long init[2] = {~0ull, 0};
    CHECK(hipMemcpy(t, init, sizeof(init), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(hot_kernel, dim3(ncu), dim3(256), ONE_PER_CU_LDS, 0, b, GB / 16, hw, kind, lines, ctr, done, t, ops,
                       sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    unsigned long long tt[2], no;
    CHECK(hipMemcpy(tt, t, sizeof(tt), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&no, ops, 8, hipMemcpyDeviceToHost));
    double const ns = (tt[1] - tt[0]) / tpn;
    if (ops_per_us) *ops_per_us = no / (ns / 1e3);
    return ns;
  };
  struct Case {
    const char *name;
    int hw, kind, lines;
  } cases[] = {{"idle", 1, K_IDLE, 1},    {"sleep", 1, K_SLEEP, 1},  {"add 1", 1, K_ADD, 1},  {"add 8", 1, K_ADD, 8},
               {"add 64", 1, K_ADD, 64},  {"load8", 1, K_LOAD8, 8},  {"idle", 2, K_IDLE, 1},  {"add 8", 2, K_ADD, 8},
               {"load8", 2, K_LOAD8, 8}};
  std::string json;
  for (int rep = 0; rep < 3; rep++) {
    for (auto &c : cases) {
      run(other, 0, K_IDLE, 1, nullptr);  // 1 GB of other data first: the read below comes from HBM
      double opus = 0;
      double const ns = run(buf, c.hw, c.kind, c.lines, &opus);
      double const tbs = GB / ns / 1e3;
      int const hwaves = c.hw * ncu;
      printf("rep %d %-7s %d hammer waves per CU (%d): readers %.2f TB/s (%.1f us); hammer ops %.0f per us (%.3f per wave per us)\n",
             rep, c.name, c.hw, hwaves, tbs, ns / 1e3, opus, opus / hwaves);
      char row[240];
      snprintf(row, sizeof(row), "%s{\"rep\": %d, \"case\": \"%s\", \"hw\": %d, \"read_TBps\": %.3f, \"ops_per_us\": %.0f}",
               json.empty() ? "" : ", ", rep, c.name, c.hw, tbs, opus);
      json += row;
    }
  }
  printf("{\"bench\": \"hot_line\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
