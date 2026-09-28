/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared by the primitive-cost microbenchmarks: error checking, clock
// calibration, per-SM measurement, statistics, a shared-GPU guard and JSON
// output.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvml.h>
#include <unistd.h>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__);  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// A bad launch configuration is reported by cudaGetLastError, not by the next
// synchronize; without this a failed launch would silently re-report the
// previous pass's output buffers.
#define CHECK_LAUNCH()                                                         \
  do {                                                                         \
    CHECK(cudaGetLastError());                                                 \
    CHECK(cudaDeviceSynchronize());                                            \
  } while (0)

// The runtime's synchronization primitives, copied verbatim from
// include/mirage/persistent_kernel/mpk_atoms.cuh so the benchmarks time
// exactly the instructions the megakernel issues.
__device__ __forceinline__ unsigned long long int
    atom_add_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int val) {
  unsigned long long int old_val;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
               : "=l"(old_val)
               : "l"(addr), "l"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_gpu_u64(unsigned long long int *addr) {
  unsigned long long int val;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long int *addr,
                                                   unsigned long long int val) {
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
}

__device__ __forceinline__ unsigned long long gtimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int sm_id() {
  int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// Timing uses clock64(), which has 2-cycle granularity; %globaltimer advances
// only every 32 ns on SM100 and cannot resolve a single operation. This
// converts cycles to time using the SM clock measured now, not a nominal one.
__global__ void calibrate_clock_kernel(double *ghz_out) {
  if (threadIdx.x || blockIdx.x) {
    return;
  }
  unsigned long long g0 = gtimer_ns();
  long long c0 = clock64();
  while (gtimer_ns() - g0 < 2000000ull) { // 2 ms
  }
  long long c1 = clock64();
  unsigned long long g1 = gtimer_ns();
  *ghz_out = (double)(c1 - c0) / (double)(g1 - g0);
}

inline double measure_sm_clock_ghz() {
  double *d_ghz, ghz = 0.0;
  CHECK(cudaMalloc(&d_ghz, sizeof(double)));
  calibrate_clock_kernel<<<1, 1>>>(d_ghz);
  CHECK_LAUNCH();
  CHECK(cudaMemcpy(&ghz, d_ghz, sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_ghz));
  return ghz;
}

// ---------------------------------------------------------------------------
// Per-SM measurement.
//
// Several of these costs differ ~2x from one SM to another, so a number taken
// from a single block is one arbitrary sample. Instead every SM measures in
// turn. The host forces exactly one block per SM by requesting more than half
// an SM's shared memory, and blocks take turns on a ticket so each measures
// with the rest of the machine idle. Waiting blocks back off with __nanosleep:
// without it, their ticket polling loads the memory system enough to inflate
// the SM being measured several-fold. Quadrupling the backoff changes no
// median by more than 0.3%, so this one is long enough.
// ---------------------------------------------------------------------------
constexpr unsigned TURN_BACKOFF_NS = 2000;

__device__ __forceinline__ void wait_turn(unsigned long long *ticket) {
  while (ld_acquire_gpu_u64(ticket) != (unsigned long long)blockIdx.x) {
    __nanosleep(TURN_BACKOFF_NS);
  }
}

__device__ __forceinline__ void pass_turn(unsigned long long *ticket) {
  atom_add_release_gpu_u64(ticket, 1ull);
}

// Dynamic shared memory that allows only one resident block per SM.
inline int one_block_per_sm_smem() {
  int per_sm = 0, per_block = 0;
  CHECK(cudaDeviceGetAttribute(
      &per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  CHECK(cudaDeviceGetAttribute(
      &per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  int want = per_sm / 2 + 1024;
  if (want > per_block) {
    printf("cannot force one block per SM: need %d B, block limit %d B\n",
           want,
           per_block);
    exit(1);
  }
  return want;
}

template <typename Kernel>
inline void allow_dynamic_smem(Kernel kernel, int bytes) {
  CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// Per-block output buffers are sized for this many SMs.
constexpr int MAX_SMS = 1024;

inline void require_sm_count_fits(int nsm) {
  if (nsm > MAX_SMS) {
    printf("FAIL: %d SMs, buffers are sized for %d\n", nsm, MAX_SMS);
    exit(1);
  }
}

// Theoretical HBM bandwidth from the memory clock and bus width (double data
// rate), in GB/s. 0 if the device does not report them.
inline double hbm_peak_gbps() {
  int khz = 0, bits = 0;
  if (cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&bits, cudaDevAttrGlobalMemoryBusWidth, 0) !=
          cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return 2.0 * khz * 1e3 * (bits / 8.0) / 1e9;
}

// The SM counts a scaling sweep visits, limited to what the device has.
inline std::vector<int> sm_counts(int nsm) {
  std::vector<int> out;
  for (int n : {1, 2, 8, 32, 74, nsm}) {
    if (n <= nsm && (out.empty() || n > out.back())) {
      out.push_back(n);
    }
  }
  return out;
}

// A per-SM pass is only meaningful if every SM was measured exactly once.
inline void require_every_sm_once(std::vector<int> const &smid, int nsm) {
  std::vector<int> seen(4096, 0);
  for (int s : smid) {
    if (s < 0 || s >= 4096 || seen[s]++) {
      printf("FAIL: SM %d measured more than once -- one-block-per-SM "
             "placement did not hold\n",
             s);
      exit(1);
    }
  }
  if ((int)smid.size() != nsm) {
    printf("FAIL: measured %zu SMs, expected %d\n", smid.size(), nsm);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// Statistics.
// ---------------------------------------------------------------------------

// Repetitions of an on-SM measurement that one SM stands for: median, with
// min/max as evidence the number is stable.
constexpr int REPS = 7;

struct Stat {
  double med, lo, hi;
};

inline Stat summarize(double *v, int n) {
  std::sort(v, v + n);
  return Stat{v[n / 2], v[0], v[n - 1]};
}

inline void report(char const *label, Stat s, char const *unit) {
  printf("%-44s %8.2f %s  [%.2f .. %.2f]\n", label, s.med, unit, s.lo, s.hi);
}

// Samples across SMs (and addresses): the distribution across the machine.
struct Dist {
  double med, p10, p90, lo, hi;
};

inline Dist distribution(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) {
    return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
  };
  return Dist{at(0.5), at(0.1), at(0.9), v.front(), v.back()};
}

// Device-scope costs can fall into two separated groups: an SM reaching an
// address homed near it, or far from it. The samples split at their widest
// relative gap if it is at least SPLIT_RATIO and leaves at least 5% of the
// samples on each side; otherwise they are one group.
constexpr double SPLIT_RATIO = 1.25;

inline bool find_split(std::vector<double> v, double *cut) {
  if (v.size() < 40) {
    return false;
  }
  std::sort(v.begin(), v.end());
  size_t const edge = (v.size() + 19) / 20;
  double best = 1.0;
  size_t at = 0;
  for (size_t i = edge; i <= v.size() - edge; i++) {
    if (v[i - 1] > 0 && v[i] / v[i - 1] > best) {
      best = v[i] / v[i - 1];
      at = i;
    }
  }
  if (best < SPLIT_RATIO) {
    return false;
  }
  *cut = 0.5 * (v[at - 1] + v[at]);
  return true;
}

// ---------------------------------------------------------------------------
// Shared-GPU guard.
//
// Another process on the same GPU is time-sliced with this one, and clock64()
// keeps counting while this kernel is switched out, so every number inflates
// -- including purely on-SM ones -- with nothing in the output to show it.
// Two independent checks: NVML lists other compute processes on the device,
// and an ALU sentinel (a fixed dependent instruction chain, which takes the
// same number of cycles on every idle SM) shows any SM that was interrupted.
// ---------------------------------------------------------------------------
struct ShareState {
  int other_procs = -1; // -1: could not be determined
};

inline ShareState probe_gpu_sharing() {
  ShareState s;
  char bus[32];
  if (cudaDeviceGetPCIBusId(bus, sizeof(bus), 0) != cudaSuccess ||
      nvmlInit_v2() != NVML_SUCCESS) {
    return s;
  }
  nvmlDevice_t h;
  if (nvmlDeviceGetHandleByPciBusId_v2(bus, &h) == NVML_SUCCESS) {
    unsigned n = 64;
    nvmlProcessInfo_t info[64];
    nvmlReturn_t r = nvmlDeviceGetComputeRunningProcesses(h, &n, info);
    if (r == NVML_SUCCESS) {
      int others = 0;
      for (unsigned i = 0; i < n; i++) {
        if (info[i].pid != (unsigned)getpid()) {
          others++;
        }
      }
      s.other_procs = others;
    } else if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
      s.other_procs = (int)n; // more processes than fit: certainly shared
    }
  }
  nvmlShutdown();
  return s;
}

__global__ void alu_sentinel_kernel(unsigned long long *ticket,
                                    double *cycles_per_op,
                                    int *smid_out,
                                    int iters) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  unsigned x = blockIdx.x + 1;
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(x));
  }
  long long t1 = clock64();
  cycles_per_op[blockIdx.x] = (double)(t1 - t0) / iters + (x == 7u ? 1e-9 : 0);
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// Returns max/min of the per-SM sentinel. 1.00 on an idle GPU.
inline double alu_sentinel_spread(int nsm, int smem) {
  unsigned long long *ticket;
  double *d_c;
  int *d_s;
  CHECK(cudaMalloc(&ticket, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_c, nsm * sizeof(double)));
  CHECK(cudaMalloc(&d_s, nsm * sizeof(int)));
  allow_dynamic_smem(alu_sentinel_kernel, smem);
  CHECK(cudaMemset(ticket, 0, sizeof(unsigned long long)));
  alu_sentinel_kernel<<<nsm, 32, smem>>>(ticket, d_c, d_s, 200000);
  CHECK_LAUNCH();
  std::vector<double> c(nsm);
  std::vector<int> s(nsm);
  CHECK(
      cudaMemcpy(c.data(), d_c, nsm * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(s.data(), d_s, nsm * sizeof(int), cudaMemcpyDeviceToHost));
  require_every_sm_once(s, nsm);
  CHECK(cudaFree(ticket));
  CHECK(cudaFree(d_c));
  CHECK(cudaFree(d_s));
  auto mm = std::minmax_element(c.begin(), c.end());
  return *mm.second / *mm.first;
}

// Tolerated sentinel spread. An idle GPU measures well under this; a
// time-sliced SM overshoots it by far.
constexpr double SENTINEL_TOLERANCE = 1.02;

// Cycles are converted with the clock measured at the start; if it drifted by
// the end, the conversion is off by as much.
constexpr double CLOCK_TOLERANCE = 0.01;

struct Guard {
  ShareState start, end;
  double spread_start = 0, spread_end = 0;
  double ghz_start = 0, ghz_end = 0;
  bool clock_drifted() const {
    return ghz_start > 0 && ghz_end > 0 &&
           std::abs(ghz_end / ghz_start - 1.0) > CLOCK_TOLERANCE;
  }
  bool provisional() const {
    return start.other_procs > 0 || end.other_procs > 0 ||
           spread_start > SENTINEL_TOLERANCE ||
           spread_end > SENTINEL_TOLERANCE || clock_drifted();
  }
  void print() const {
    printf("shared-GPU guard: other processes %d -> %d, ALU sentinel spread "
           "%.4f -> %.4f\n",
           start.other_procs,
           end.other_procs,
           spread_start,
           spread_end);
    printf("SM clock %.3f -> %.3f GHz%s\n",
           ghz_start,
           ghz_end,
           clock_drifted() ? " (drifted)" : "");
    if (start.other_procs < 0 || end.other_procs < 0) {
      printf("(NVML could not list processes; only the sentinel was "
             "checked)\n");
    }
    if (provisional()) {
      printf("WARNING: the GPU was shared or interrupted during this run. "
             "Numbers are PROVISIONAL and should not be compared with a clean "
             "run.\n");
    }
  }
};

struct Json {
  FILE *f = nullptr;
  bool first = true;
  void open(char const *path) {
    f = path ? fopen(path, "w") : nullptr;
    if (path && !f) {
      printf("cannot write %s\n", path);
      exit(1);
    }
    if (f) {
      fprintf(f, "{\n");
    }
  }
  void kv(char const *k, double v) {
    if (!f) {
      return;
    }
    fprintf(f, "%s  \"%s\": %.4f", first ? "" : ",\n", k, v);
    first = false;
  }
  void kvs(char const *k, char const *v) {
    if (!f) {
      return;
    }
    fprintf(f, "%s  \"%s\": \"%s\"", first ? "" : ",\n", k, v);
    first = false;
  }
  // Records a per-SM distribution as <key>_med, _p10, _p90, _min, _max.
  void dist(char const *k, Dist d) {
    char buf[128];
    char const *suffix[5] = {"med", "p10", "p90", "min", "max"};
    double vals[5] = {d.med, d.p10, d.p90, d.lo, d.hi};
    for (int i = 0; i < 5; i++) {
      snprintf(buf, sizeof(buf), "%s_%s", k, suffix[i]);
      kv(buf, vals[i]);
    }
  }
  void close() {
    if (f) {
      fprintf(f, "\n}\n");
      fclose(f);
    }
  }
};
