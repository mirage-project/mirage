// Microbenchmark: what the L2 write-back (release) and the L2 invalidate (acquire) cost on MI355X (gfx950), as a function of
// how much data sits in the die's L2.
//
// A hand-off between dies needs the producer's die to write its dirty L2 lines back (buffer_wbl2) and the consumer's die to
// drop its stale lines (buffer_inv). Fleet (arXiv 2604.15379) says the write-back's cost grows with the number of dirty lines
// and stalls later memory accesses. This measures it inside one launch (a kernel's end writes the L2 back by itself):
//   1. every workgroup reads its die (XCC id) and takes a slot on that die (an atomic per die)
//   2. on the target die, the workers (slots 1..writers) each store `bytes / writers` (default stores: dirty lines in L2)
//      or, for the invalidate case, load them (clean valid lines in L2); then add 1 to a die-local counter
//   3. slot 0 waits for the counter, then times one agent-scope release fence (write-back) or acquire fence (invalidate),
//      with both clocks, and then times one 64 B load right after it (the stall Fleet describes)
// For bytes in 0 .. 8 MB (the L2 is 4 MB) and reps launches each; medians are printed.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 l2_writeback.hip -o l2_writeback
// Run:   ./l2_writeback [reps=20]        -> one line per size and kind, then a JSON summary on the last line
#include <string>

#include "common.h"

enum Kind { WRITEBACK = 0, INVALIDATE = 1 };

// out[0..3] = {fence ticks (realtime), fence cycles (shader clock), next-load cycles, writers that took part}
template <int KIND>
__global__ void fence_cost_kernel(uint4 *buf, size_t bytes, unsigned target_die, int writers, unsigned *slots, unsigned *done,
                                  unsigned long long *out) {
  extern __shared__ char lds[];
  __shared__ int my_slot;
  if (threadIdx.x == 0) {
    lds[0] = 0;
    unsigned die = xcc_id();
    my_slot = die == target_die ? (int)atomicAdd(&slots[die], 1u) : -1;
  }
  __syncthreads();
  int slot = my_slot;
  if (slot < 0 || slot > writers) return;
  if (slot >= 1) {
    size_t per = bytes / writers / sizeof(uint4), base = (slot - 1) * per;
    if constexpr (KIND == WRITEBACK) {
      for (size_t i = threadIdx.x; i < per; i += blockDim.x) buf[base + i] = make_uint4(slot, i, 1, 2);
    } else {
      uint4 acc = make_uint4(0, 0, 0, 0);
      for (size_t i = threadIdx.x; i < per; i += blockDim.x) {
        uint4 v = buf[base + i];
        acc.x ^= v.x;
      }
      if (acc.x == 0x12345678u) buf[0].y = 1;  // keep the loads
    }
    __syncthreads();
    // die-local: same L2, no fence needed (Fleet's intra-XCD counters)
    if (threadIdx.x == 0) __hip_atomic_fetch_add(done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    return;
  }
  if (threadIdx.x != 0) return;
  while (__hip_atomic_load(done, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)writers) __builtin_amdgcn_s_sleep(1);
  // the fence, then a wait for it to finish (the compiler only waits before a later store, so the wait is explicit here)
  unsigned long long r0 = realtime(), c0 = shader_clock();
  if constexpr (KIND == WRITEBACK) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  } else {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  unsigned long long c1 = shader_clock(), r1 = realtime();
  unsigned v = __hip_atomic_load((unsigned *)(buf + (bytes / sizeof(uint4)) + 64), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  unsigned long long c2 = shader_clock();
  out[0] = r1 - r0;
  out[1] = c1 - c0;
  out[2] = c2 - c1 + (v == 0xdeadbeefu);
  out[3] = writers;
}

int main(int argc, char **argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 20;
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int nwg = prop.multiProcessorCount, writers = 24;  // 32 CUs per die: slot 0 times, 24 workers fill, the rest idle
  double tpn = wall_ticks_per_ns(0);
  set_max_lds((const void *)fence_cost_kernel<WRITEBACK>, ONE_PER_CU_LDS);
  set_max_lds((const void *)fence_cost_kernel<INVALIDATE>, ONE_PER_CU_LDS);
  const size_t sizes[] = {0, 64 << 10, 256 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20};
  uint4 *buf;
  unsigned *slots, *done;
  unsigned long long *out;
  CHECK(hipMalloc(&buf, (16 << 20) + 4096));
  CHECK(hipMemset(buf, 0, (16 << 20) + 4096));
  CHECK(hipMalloc(&slots, 64 * sizeof(unsigned)));
  CHECK(hipMalloc(&done, sizeof(unsigned)));
  CHECK(hipMalloc(&out, 4 * sizeof(unsigned long long)));
  std::string json;
  for (int kind = 0; kind < 2; ++kind) {
    for (size_t bytes : sizes) {
      std::vector<double> ns, cyc, next;
      for (int r = 0; r < reps; ++r) {
        CHECK(hipMemset(slots, 0, 64 * sizeof(unsigned)));
        CHECK(hipMemset(done, 0, sizeof(unsigned)));
        CHECK(hipMemset(out, 0, 4 * sizeof(unsigned long long)));
        auto k = kind == WRITEBACK ? fence_cost_kernel<WRITEBACK> : fence_cost_kernel<INVALIDATE>;
        hipLaunchKernelGGL(k, dim3(nwg), dim3(256), ONE_PER_CU_LDS, 0, buf, bytes, 0u, bytes ? writers : 0, slots, done, out);
        CHECK(hipDeviceSynchronize());
        unsigned long long o[4];
        CHECK(hipMemcpy(o, out, sizeof(o), hipMemcpyDeviceToHost));
        ns.push_back(o[0] / tpn);
        cyc.push_back((double)o[1]);
        next.push_back((double)o[2]);
      }
      printf("%-10s %7zu KB in die 0's L2: fence %.0f ns (%.0f shader cycles), next load %.0f cycles (medians of %d)\n",
              kind == WRITEBACK ? "write-back" : "invalidate", bytes >> 10, median(ns), median(cyc), median(next), reps);
      char row[256];
      snprintf(row, sizeof(row), "%s{\"kind\": \"%s\", \"bytes\": %zu, \"fence_ns\": %.1f, \"fence_cycles\": %.0f, "
               "\"next_load_cycles\": %.0f}", json.empty() ? "" : ", ", kind == WRITEBACK ? "writeback" : "invalidate", bytes,
               median(ns), median(cyc), median(next));
      json += row;
    }
  }
  printf("{\"bench\": \"l2_writeback\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
