/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Cost of the on-device profiler stamp the megakernel emits around every task
// and (with stage profiling on) every stage of every task.
//
// The runtime's profiler (include/mirage/persistent_kernel/profiler.h) records
// one 64-bit entry per event: a tag (event id, block/group, begin/end) and a
// timestamp read from %globaltimer_lo. PROFILER_EVENT_START is
//   if (write_thread) { entry.tag = ...; entry.delta = get_timestamp();
//                      *write_ptr = entry.raw; write_ptr += stride; }
//   __threadfence_block();
// and PROFILER_EVENT_END is the same with the fence before the body. Every
// task pays a START+END pair; with stage profiling on, every stage of every
// task pays another pair. The stamp cost is indistinguishable from kernel
// work in the trace unless it is measured and reported -- which this does.
//
// The PTX below is copied verbatim from profiler.h so the numbers are the
// instructions the megakernel issues. The store hits the same global-memory
// buffer the runtime uses (the profiler ring is in gmem), so the
// memory-system cost is included. Each primitive is measured on every SM.

#include "microbench_common.cuh"

#include <map>

constexpr int R_SM = 3;

__device__ __forceinline__ uint32_t get_timestamp() {
  uint32_t volatile ret;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ uint32_t encode_tag(uint32_t block_group_idx,
                                               uint32_t event_idx,
                                               uint32_t event_type) {
  return (block_group_idx << 11) | (event_idx << 2) | event_type;
}

union ProfilerEntry {
  struct {
    uint32_t tag;
    uint32_t delta_time;
  };
  uint64_t raw;
};

// Full PROFILER_EVENT_START, per SM, one write thread.
__global__ void stamp_start_per_sm(uint64_t *ring,
                                   unsigned long long *ticket,
                                   int iters,
                                   double *out,
                                   int *smid_out,
                                   uint64_t *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  uint32_t const tag_base = encode_tag((uint32_t)blockIdx.x, 0, 0);
  uint64_t *write_ptr = ring + (uint64_t)blockIdx.x;
  uint64_t const stride = (uint64_t)gridDim.x;
  uint64_t acc = 0;
  for (int r = 0; r < R_SM; r++) {
    ProfilerEntry e;
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      e.tag = tag_base | ((uint32_t)i << 19) | 0x0;
      e.delta_time = get_timestamp();
      *write_ptr = e.raw;
      write_ptr += stride;
      __threadfence_block();
      acc += e.raw;
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// Full PROFILER_EVENT_END, per SM.
__global__ void stamp_end_per_sm(uint64_t *ring,
                                 unsigned long long *ticket,
                                 int iters,
                                 double *out,
                                 int *smid_out,
                                 uint64_t *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  uint32_t const tag_base = encode_tag((uint32_t)blockIdx.x, 0, 0);
  uint64_t *write_ptr = ring + (uint64_t)blockIdx.x;
  uint64_t const stride = (uint64_t)gridDim.x;
  uint64_t acc = 0;
  for (int r = 0; r < R_SM; r++) {
    ProfilerEntry e;
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      __threadfence_block();
      e.tag = tag_base | ((uint32_t)i << 19) | 0x1;
      e.delta_time = get_timestamp();
      *write_ptr = e.raw;
      write_ptr += stride;
      acc += e.raw;
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// One START+END pair: the cost a single task (or stage) event adds.
__global__ void stamp_pair_per_sm(uint64_t *ring,
                                   unsigned long long *ticket,
                                   int iters,
                                   double *out,
                                   int *smid_out,
                                   uint64_t *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  uint32_t const tag_base = encode_tag((uint32_t)blockIdx.x, 0, 0);
  uint64_t *write_ptr = ring + (uint64_t)blockIdx.x;
  uint64_t const stride = (uint64_t)gridDim.x;
  uint64_t acc = 0;
  for (int r = 0; r < R_SM; r++) {
    ProfilerEntry e;
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      e.tag = tag_base | ((uint32_t)i << 19) | 0x0;
      e.delta_time = get_timestamp();
      *write_ptr = e.raw;
      write_ptr += stride;
      __threadfence_block();
      __threadfence_block();
      e.tag = tag_base | ((uint32_t)i << 19) | 0x1;
      e.delta_time = get_timestamp();
      *write_ptr = e.raw;
      write_ptr += stride;
      acc += e.raw;
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// The timestamp read alone: mov.u32 %globaltimer_lo.
__global__ void globaltimer_per_sm(unsigned long long *ticket,
                                   int iters,
                                   double *out,
                                   int *smid_out,
                                   uint64_t *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  uint64_t acc = 0;
  for (int r = 0; r < R_SM; r++) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      acc += get_timestamp();
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// 64-bit store to the ring alone (relaxed), and store + __threadfence_block,
// so the fence's marginal cost is the difference.
__global__ void store_fence_per_sm(uint64_t *ring,
                                   unsigned long long *ticket,
                                   int iters,
                                   double *out_store,
                                   double *out_fence,
                                   int *smid_out,
                                   uint64_t *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  uint64_t *write_ptr = ring + (uint64_t)blockIdx.x;
  uint64_t const stride = (uint64_t)gridDim.x;
  uint64_t acc = 0;
  for (int r = 0; r < R_SM; r++) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      *write_ptr = (uint64_t)i;
      write_ptr += stride;
      acc += *write_ptr;
    }
    long long t1 = clock64();
    double store = (double)(t1 - t0) / iters;
    t0 = clock64();
    for (int i = 0; i < iters; i++) {
      *write_ptr = (uint64_t)i;
      write_ptr += stride;
      __threadfence_block();
      acc += *write_ptr;
    }
    t1 = clock64();
    out_store[blockIdx.x * R_SM + r] = store;
    out_fence[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters - store;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

using PerSm = std::map<int, double>;

static PerSm per_sm_medians(double const *d_out,
                            int const *d_smid,
                            int nsm,
                            double ghz) {
  std::vector<double> raw(nsm * R_SM);
  std::vector<int> sm(nsm);
  CHECK(cudaMemcpy(
      raw.data(), d_out, raw.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(
      cudaMemcpy(sm.data(), d_smid, nsm * sizeof(int), cudaMemcpyDeviceToHost));
  require_every_sm_once(sm, nsm);
  PerSm med;
  for (int b = 0; b < nsm; b++) {
    double *v = raw.data() + b * R_SM;
    std::sort(v, v + R_SM);
    med[sm[b]] = v[R_SM / 2] / ghz;
  }
  return med;
}

static std::vector<double> pool(PerSm const &m) {
  std::vector<double> v;
  for (auto const &kv : m) {
    v.push_back(kv.second);
  }
  return v;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  char const *json_path = (argc > 1) ? argv[1] : nullptr;
  int const ITERS = 4000;

  Guard guard;
  guard.start = probe_gpu_sharing();

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0, runtime = 0;
  CHECK(cudaDriverGetVersion(&driver));
  CHECK(cudaRuntimeGetVersion(&runtime));
  int const nsm = prop.multiProcessorCount;
  int const smem = one_block_per_sm_smem();

  require_sm_count_fits(nsm);
  double ghz = measure_sm_clock_ghz();
  guard.ghz_start = ghz;
  guard.spread_start = alu_sentinel_spread(nsm, smem);

  printf("device       : %s (SM %d.%d, %d SMs)\n",
         prop.name,
         prop.major,
         prop.minor,
         nsm);
  printf("driver/runtime: %d / %d\n", driver, runtime);
  printf("SM clock     : %.3f GHz (measured)\n\n", ghz);

  size_t const ring_elems =
      (size_t)nsm * (size_t)(ITERS + 1) * 2 + (size_t)nsm + 16;
  uint64_t *d_ring;
  unsigned long long *d_ticket;
  double *d_out, *d_out2;
  int *d_smid;
  uint64_t *d_sink;
  CHECK(cudaMalloc(&d_ring, ring_elems * sizeof(uint64_t)));
  CHECK(cudaMalloc(&d_ticket, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_out, MAX_SMS * R_SM * sizeof(double)));
  CHECK(cudaMalloc(&d_out2, MAX_SMS * R_SM * sizeof(double)));
  CHECK(cudaMalloc(&d_smid, MAX_SMS * sizeof(int)));
  CHECK(cudaMalloc(&d_sink, MAX_SMS * sizeof(uint64_t)));

  allow_dynamic_smem(stamp_start_per_sm, smem);
  allow_dynamic_smem(stamp_end_per_sm, smem);
  allow_dynamic_smem(stamp_pair_per_sm, smem);
  allow_dynamic_smem(globaltimer_per_sm, smem);
  allow_dynamic_smem(store_fence_per_sm, smem);

  Json js;
  js.open(json_path);
  js.kvs("device", prop.name);
  js.kv("sm_count", nsm);
  js.kv("driver_version", driver);
  js.kv("sm_clock_ghz", ghz);

  printf("each row pools %d SMs.\n", nsm);
  printf("%-44s %8s %8s %8s   %-18s %s\n",
         "",
         "median",
         "p10",
         "p90",
         "[min .. max]",
         "samples");
  auto ticketed = [&]() {
    CHECK(cudaMemset(d_ticket, 0, sizeof(unsigned long long)));
  };
  auto row = [&](char const *label, char const *key, std::vector<double> v) {
    Dist d = distribution(v);
    printf("%-44s %8.2f %8.2f %8.2f   [%5.2f .. %5.2f]   %zu\n",
           label,
           d.med,
           d.p10,
           d.p90,
           d.lo,
           d.hi,
           v.size());
    js.dist(key, d);
  };

  // --- full START ---
  ticketed();
  stamp_start_per_sm<<<nsm, 32, smem>>>(
      d_ring, d_ticket, ITERS, d_out, d_smid, d_sink);
  CHECK_LAUNCH();
  row("PROFILER_EVENT_START (full)",
      "profiler_event_start_ns",
      pool(per_sm_medians(d_out, d_smid, nsm, ghz)));

  // --- full END ---
  ticketed();
  stamp_end_per_sm<<<nsm, 32, smem>>>(
      d_ring, d_ticket, ITERS, d_out, d_smid, d_sink);
  CHECK_LAUNCH();
  row("PROFILER_EVENT_END (full)",
      "profiler_event_end_ns",
      pool(per_sm_medians(d_out, d_smid, nsm, ghz)));

  // --- one START+END pair (one task event) ---
  ticketed();
  stamp_pair_per_sm<<<nsm, 32, smem>>>(
      d_ring, d_ticket, ITERS, d_out, d_smid, d_sink);
  CHECK_LAUNCH();
  row("START+END pair (one task event)",
      "profiler_event_pair_ns",
      pool(per_sm_medians(d_out, d_smid, nsm, ghz)));

  // --- globaltimer read alone ---
  ticketed();
  globaltimer_per_sm<<<nsm, 32, smem>>>(
      d_ticket, ITERS, d_out, d_smid, d_sink);
  CHECK_LAUNCH();
  row("mov.u32 %globaltimer_lo (read only)",
      "globaltimer_read_ns",
      pool(per_sm_medians(d_out, d_smid, nsm, ghz)));

  // --- store alone, and store + fence (fence marginal) ---
  ticketed();
  store_fence_per_sm<<<nsm, 32, smem>>>(
      d_ring, d_ticket, ITERS, d_out, d_out2, d_smid, d_sink);
  CHECK_LAUNCH();
  row("st.b64 to profiler ring (relaxed)",
      "store_relaxed_ns",
      pool(per_sm_medians(d_out, d_smid, nsm, ghz)));
  row("__threadfence_block() marginal",
      "threadfence_block_marginal_ns",
      pool(per_sm_medians(d_out2, d_smid, nsm, ghz)));

  guard.end = probe_gpu_sharing();
  guard.spread_end = alu_sentinel_spread(nsm, smem);
  guard.ghz_end = measure_sm_clock_ghz();
  printf("\n");
  guard.print();
  js.kv("other_gpu_processes",
        std::max(guard.start.other_procs, guard.end.other_procs));
  js.kv("alu_sentinel_spread", std::max(guard.spread_start, guard.spread_end));
  js.kv("provisional", guard.provisional() ? 1 : 0);

  js.close();
  if (json_path) {
    printf("wrote %s\n", json_path);
  }
  return 0;
}
