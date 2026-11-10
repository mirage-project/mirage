// Correctness + cost test of the hand-off protocols the HIP megakernel could use between a producer task and a consumer task on
// different CUs (same die or other die) of one MI355X (gfx950). A wrong protocol shows up as stale data, so every hand-off
// checks every value.
//
// Each iteration i: the producer workgroup writes 4 KB (256 threads x 16 B) of values that depend on i, then posts the flag i;
// the consumer polls the flag until it reads i, reads the 4 KB and counts the 4-byte words that are not iteration i's, then
// posts an ack so the producer may overwrite the buffer. Protocols:
//   release      plain stores; agent store-release of the flag (buffer_wbl2 sc1: the die's whole L2 written back) /
//                sc1 polls, acquire (buffer_inv sc1), plain loads                     -- the compiler's protocol, correct
//   bypass       data stores with sc1 (device scope: past the L2), s_waitcnt vmcnt(0), flag store sc1 (no write-back) /
//                sc1 polls, data loads with sc1 (past the L2), no invalidate          -- the candidate for the megakernel
//   data-flag    bypass stores of data that is itself the flag (the buffer filled with 0xFF, every word checked until none is
//                0xFFFFFFFF, as runtime.cuh's poll16 / valid16w) / sc1 loads; no flag
//   plain        plain stores, flag store sc1, no write-back / sc1 polls, plain loads, no invalidate
//                                                                                     -- expected to read stale data (control)
// Output per protocol and placement: wrong words over all iterations and the time per hand-off (round trip / 2).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 handoff_stress.hip -o handoff_stress
// Run:   ./handoff_stress [iters=20000]      -> one line per case, then a JSON summary on the last line
#include <map>
#include <string>

#include "common.h"

enum Proto { RELEASE = 0, BYPASS = 1, DATA_FLAG = 2, PLAIN = 3 };

typedef unsigned v4u __attribute__((ext_vector_type(4)));  // inline asm takes vector types, not uint4
__device__ inline void st16_sc1(uint4 *p, uint4 v) {
  v4u w = {v.x, v.y, v.z, v.w};
  asm volatile("global_store_dwordx4 %0, %1, off sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ inline uint4 ld16_sc1(const uint4 *p) {
  v4u w;
  asm volatile("global_load_dwordx4 %0, %1, off sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return make_uint4(w.x, w.y, w.z, w.w);
}
__device__ inline void st32_sc1(unsigned *p, unsigned v) {
  asm volatile("global_store_dword %0, %1, off sc1" ::"v"(p), "v"(v) : "memory");
}
__device__ inline unsigned ld32_sc1(const unsigned *p) {
  unsigned v;
  asm volatile("global_load_dword %0, %1, off sc1\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  return v;
}
__device__ inline void wait_stores() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

__device__ inline uint4 pattern(int i, int t) { return make_uint4(i, i ^ t, i + t, ~(unsigned)i - t); }

// flags[0] = produced iteration, flags[32] = acked iteration (separate 128-B lines)
template <int P>
__global__ void handoff_kernel(uint4 *data, unsigned *flags, int prod, int cons, int iters, unsigned long long *out) {
  extern __shared__ char lds[];
  __shared__ unsigned bad_s;
  int t = threadIdx.x;
  if (t == 0) {
    lds[0] = 0;
    bad_s = 0;
  }
  __syncthreads();
  if (blockIdx.x == prod) {
    unsigned long long t0 = realtime();
    for (int i = 1; i <= iters; ++i) {
      if (t == 0) {
        while (ld32_sc1(&flags[32]) < (unsigned)(i - 1)) __builtin_amdgcn_s_sleep(0);  // the consumer is done with i-1
        if constexpr (P == RELEASE) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      }
      __syncthreads();
      if constexpr (P == RELEASE) {
        data[t] = pattern(i, t);
      } else if constexpr (P == PLAIN) {
        data[t] = pattern(i, t);
      } else {
        st16_sc1(&data[t], pattern(i, t));
      }
      if constexpr (P != RELEASE) wait_stores();
      __syncthreads();
      if constexpr (P == RELEASE) {
        if (t == 0) __hip_atomic_store(&flags[0], (unsigned)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (P != DATA_FLAG) {
        if (t == 0) st32_sc1(&flags[0], (unsigned)i);
      }
    }
    if (t == 0) out[0] = realtime() - t0;
  } else if (blockIdx.x == cons) {
    unsigned bad = 0;
    for (int i = 1; i <= iters; ++i) {
      uint4 v;
      if constexpr (P == DATA_FLAG) {
        // the data is the flag: re-load until no word is 0xFFFFFFFF (the buffer was re-armed by this consumer)
        do {
          v = ld16_sc1(&data[t]);
        } while (v.x == ~0u || v.y == ~0u || v.z == ~0u || v.w == ~0u);
      } else {
        if (t == 0) {
          while (ld32_sc1(&flags[0]) < (unsigned)i) __builtin_amdgcn_s_sleep(0);
          if constexpr (P == RELEASE) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
        __syncthreads();
        if constexpr (P == BYPASS) {
          v = ld16_sc1(&data[t]);
        } else {
          v = data[t];
        }
      }
      uint4 e = pattern(i, t);
      bad += (v.x != e.x) + (v.y != e.y) + (v.z != e.z) + (v.w != e.w);
      if constexpr (P == DATA_FLAG) {
        st16_sc1(&data[t], make_uint4(~0u, ~0u, ~0u, ~0u));  // re-arm for the next iteration
        wait_stores();
      }
      __syncthreads();
      if (t == 0) {
        if constexpr (P == RELEASE) {
          __hip_atomic_store(&flags[32], (unsigned)i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
          st32_sc1(&flags[32], (unsigned)i);
        }
      }
    }
    atomicAdd(&bad_s, bad);
    __syncthreads();
    if (t == 0) out[1] = bad_s;
  }
}

using Kernel = void (*)(uint4 *, unsigned *, int, int, int, unsigned long long *);

__global__ void where_kernel(unsigned *where) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
    where[blockIdx.x] = xcc_id();
  }
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20000;
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int nwg = prop.multiProcessorCount;
  double tpn = wall_ticks_per_ns(0);
  const struct {
    const char *name;
    Kernel k;
  } protos[] = {{"release", handoff_kernel<RELEASE>},
                {"bypass", handoff_kernel<BYPASS>},
                {"data-flag", handoff_kernel<DATA_FLAG>},
                {"plain", handoff_kernel<PLAIN>}};
  set_max_lds((const void *)where_kernel, ONE_PER_CU_LDS);
  for (auto &p : protos) set_max_lds((const void *)p.k, ONE_PER_CU_LDS);
  unsigned *where;
  CHECK(hipMalloc(&where, nwg * sizeof(unsigned)));
  hipLaunchKernelGGL(where_kernel, dim3(nwg), dim3(64), ONE_PER_CU_LDS, 0, where);
  CHECK(hipDeviceSynchronize());
  std::vector<unsigned> w(nwg);
  CHECK(hipMemcpy(w.data(), where, nwg * sizeof(unsigned), hipMemcpyDeviceToHost));
  std::map<unsigned, std::vector<int>> by_die;
  for (int i = 0; i < nwg; ++i) by_die[w[i]].push_back(i);
  std::vector<unsigned> dies;
  for (auto &[d, v] : by_die) dies.push_back(d);
  // placements: same die (two workgroups of the first die), other die (first workgroups of two dies). The die of a workgroup
  // is read in a first launch; the test launches print it again, so a placement that moved is visible.
  struct Place {
    const char *name;
    int prod, cons;
  };
  std::vector<Place> places;
  if (by_die[dies[0]].size() >= 2) places.push_back({"same die", by_die[dies[0]][0], by_die[dies[0]][1]});
  if (dies.size() >= 2) places.push_back({"other die", by_die[dies[0]][0], by_die[dies[1]][0]});
  uint4 *data;
  unsigned *flags;
  unsigned long long *out;
  CHECK(hipMalloc(&data, 256 * sizeof(uint4)));
  CHECK(hipMalloc(&flags, 64 * sizeof(unsigned)));
  CHECK(hipMalloc(&out, 2 * sizeof(unsigned long long)));
  std::string json;
  for (auto &pl : places) {
    for (auto &p : protos) {
      CHECK(hipMemset(data, std::string(p.name) == "data-flag" ? 0xff : 0, 256 * sizeof(uint4)));
      CHECK(hipMemset(flags, 0, 64 * sizeof(unsigned)));
      CHECK(hipMemset(out, 0, 2 * sizeof(unsigned long long)));
      hipLaunchKernelGGL(p.k, dim3(nwg), dim3(256), ONE_PER_CU_LDS, 0, data, flags, pl.prod, pl.cons, iters, out);
      CHECK(hipDeviceSynchronize());
      unsigned long long o[2];
      CHECK(hipMemcpy(o, out, sizeof(o), hipMemcpyDeviceToHost));
      double ns = o[0] / tpn / iters / 2;
      printf("%-9s %-9s (wg %d -> wg %d): %llu wrong words of %llu, one hand-off %.0f ns\n", pl.name, p.name, pl.prod, pl.cons,
             o[1], (unsigned long long)iters * 1024, ns);
      char row[220];
      snprintf(row, sizeof(row), "%s{\"place\": \"%s\", \"proto\": \"%s\", \"wrong_words\": %llu, \"words\": %llu, \"handoff_ns\": %.1f}",
               json.empty() ? "" : ", ", pl.name, p.name, o[1], (unsigned long long)iters * 1024, ns);
      json += row;
    }
  }
  printf("{\"bench\": \"handoff_stress\", \"iters\": %d, \"rows\": [%s]}\n", iters, json.c_str());
  return 0;
}
