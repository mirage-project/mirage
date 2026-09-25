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

// Cost of the synchronization primitives the megakernel runtime is built from.
//
// Every dependency edge in the task graph is one of these: a worker finishing
// a task bumps an event counter with atom.add.release.gpu, the scheduler bumps
// each worker's ready counter the same way, both wait by polling with
// ld.acquire.gpu, and stages hand off through fences and barriers. Choosing
// between an aggregate counter and per-producer counters, or between a counter
// and polling the data itself, is a comparison of these numbers -- so they
// should be measured rather than assumed.
//
// Costs that involve the memory system depend on which SM issues them and on
// where the address lives, differing up to ~2x from one pair to another. So
// each is measured on every SM (see microbench_common.cuh) for NADDR addresses
// spread across 1 GiB. Where the samples separate into a near and a far group
// they are reported apart, with a check that the groups are a fixed partition
// of the SMs.

#include "microbench_common.cuh"

#include <map>
#include <set>

// Background-traffic buffer, 16x L2, so L2 cannot absorb the writes and the
// rate the generator sustains is the rate reaching HBM (at twice L2's size it
// measured above the HBM peak). Power of two: the walk masks, not divides.
#define SCRATCH_ELEMS (256ull * 1024ull * 1024ull) // 2 GiB of u64
#define SCRATCH_CHUNKS (SCRATCH_ELEMS * 8 / 512)   // 512 B, one warp's store
constexpr int BG_WARPS = 4; // writer warps in each background block

// Repetitions per SM; each SM reports the median of these.
constexpr int R_SM = 3;

// Addresses each memory-sensitive primitive is measured against, spread
// across ADDR_SPAN so they land in different L2 slices and partitions.
constexpr int NADDR = 8;
constexpr size_t ADDR_SPAN = 1ull << 30;

__host__ __device__ inline unsigned long long *
    nth_address(unsigned long long *buf, int a) {
  // odd element offset keeps the addresses from sharing an alignment pattern
  return buf + (size_t)a * (ADDR_SPAN / sizeof(unsigned long long) / NADDR) +
         (size_t)a * 37;
}

// ---------------------------------------------------------------------------
// Atomic add on one address, one thread, per SM.
//
//   CONSUME=1  the returned old value is used, so the thread waits for the
//              atomic's response before going on.
//   CONSUME=0  the result is discarded.
//
// Either way .release puts a full fence (MEMBAR.GPU) before every atomic, so
// consecutive atomics never overlap; CONSUME=1 additionally waits for the
// returned value. No instruction dependency needs to be manufactured, and any
// attempt to do so through the address is folded away by the compiler. In the
// runtime the per-task event trigger and the schedule-queue enqueue use the
// result, so CONSUME=1 is the cost on every task completion; the scheduler's
// ready-counter bumps discard it.
// ---------------------------------------------------------------------------
template <int CONSUME>
__global__ void atomic_per_sm(unsigned long long *x,
                              unsigned long long *ticket,
                              int iters,
                              double *out,
                              int *smid_out,
                              unsigned long long *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  unsigned long long acc = 0;
  for (int r = 0; r < R_SM; r++) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      unsigned long long old = atom_add_release_gpu_u64(x, 1ull);
      if (CONSUME) {
        acc += old;
      }
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// ---------------------------------------------------------------------------
// Raw acquire load of one hot address, per SM. Acquire ordering serializes
// consecutive loads (the same loop with ld.relaxed runs ~12x faster), so each
// iteration pays a full load round trip.
// ---------------------------------------------------------------------------
__global__ void acquire_per_sm(unsigned long long *x,
                               unsigned long long *ticket,
                               int iters,
                               double *out,
                               int *smid_out,
                               unsigned long long *sink) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  unsigned long long acc = 0;
  for (int r = 0; r < R_SM; r++) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      acc += ld_acquire_gpu_u64(x);
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  sink[blockIdx.x] = acc;
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// ---------------------------------------------------------------------------
// One iteration of the runtime's poll, per SM. This is the worker loop in
// persistent_kernel.cuh (the scheduler's has the same shape): acquire-load the
// current queue's ready counter, compare against the next position, move to
// the next queue, back off with __nanosleep(10). A worker polls one queue on a
// single GPU and rotates between a local and a remote one when num_gpus > 1,
// so both are measured. Its period bounds how soon a waiting worker notices
// new work. The counters here never advance, so the loop runs until the
// iteration bound.
// ---------------------------------------------------------------------------
__global__ void runtime_poll_per_sm(unsigned long long *ready,
                                    int queue_stride,
                                    int num_worker_queues,
                                    unsigned long long *ticket,
                                    int iters,
                                    double *out,
                                    int *smid_out) {
  if (threadIdx.x) {
    return;
  }
  // Shared, as the worker keeps its queue positions and queue ids.
  __shared__ unsigned long long next_task_pos[2], last_task_pos[2];
  __shared__ int worker_queue_ids[2];
  wait_turn(ticket);
  worker_queue_ids[0] = 0;
  worker_queue_ids[1] = queue_stride;
  for (int r = 0; r < R_SM; r++) {
    for (int q = 0; q < 2; q++) {
      next_task_pos[q] = 0;
      last_task_pos[q] = 0;
    }
    int queue_idx = 0, i = 0;
    long long t0 = clock64();
    while (next_task_pos[queue_idx] == last_task_pos[queue_idx] && i < iters) {
      last_task_pos[queue_idx] =
          ld_acquire_gpu_u64(ready + worker_queue_ids[queue_idx]);
      if (next_task_pos[queue_idx] < last_task_pos[queue_idx]) {
        break;
      } else {
        queue_idx = (queue_idx == num_worker_queues - 1) ? 0 : queue_idx + 1;
      }
      __nanosleep(10);
      i++;
    }
    long long t1 = clock64();
    out[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters;
  }
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// ---------------------------------------------------------------------------
// Device-scope fence, per SM: a relaxed store followed by __threadfence(),
// minus the same loop without the fence.
// ---------------------------------------------------------------------------
__global__ void fence_per_sm(unsigned long long *x,
                             unsigned long long *ticket,
                             int iters,
                             double *out_fence,
                             double *out_store,
                             int *smid_out) {
  if (threadIdx.x) {
    return;
  }
  wait_turn(ticket);
  for (int r = 0; r < R_SM; r++) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      st_relaxed_gpu_u64(x, (unsigned long long)i);
    }
    long long t1 = clock64();
    double store = (double)(t1 - t0) / iters;
    t0 = clock64();
    for (int i = 0; i < iters; i++) {
      st_relaxed_gpu_u64(x, (unsigned long long)i);
      __threadfence();
    }
    t1 = clock64();
    out_store[blockIdx.x * R_SM + r] = store;
    out_fence[blockIdx.x * R_SM + r] = (double)(t1 - t0) / iters - store;
  }
  smid_out[blockIdx.x] = sm_id();
  pass_turn(ticket);
}

// ---------------------------------------------------------------------------
// Acquire load while every other SM streams writes through a buffer larger
// than L2. The scheduler polls while the machine is busy, so the idle number
// alone could mislead. One launch per poller; the host rotates the poller over
// all blocks to cover every SM. Each background block runs BG_WARPS writer
// warps; every lane stores 16 bytes, so a warp writes 512 contiguous bytes per
// step, and each warp records how much it wrote and when, so the host reports
// the bandwidth the background actually reached instead of assuming it.
// ---------------------------------------------------------------------------
__global__ void acquire_under_load(unsigned long long *x,
                                   uint4 *scratch,
                                   unsigned long long *arrive,
                                   unsigned long long *stop,
                                   int iters,
                                   int poller,
                                   double *out,
                                   int *smid_out,
                                   unsigned long long *sink,
                                   unsigned long long *bg_bytes,
                                   unsigned long long *bg_window) {
  if ((int)blockIdx.x != poller) {
    if (threadIdx.x == 0) {
      atom_add_release_gpu_u64(arrive, 1ull);
    }
    int const w = threadIdx.x / 32, lane = threadIdx.x % 32;
    // spreads the warps apart
    unsigned long long chunk = (blockIdx.x * BG_WARPS + w) * 4099ull;
    unsigned long long steps = 0;
    uint4 const v = make_uint4(threadIdx.x, blockIdx.x, 0, 0);
    unsigned long long g0 = gtimer_ns();
    bool done = false;
    while (!done) {
      for (int i = 0; i < 16; i++) {
        chunk = (chunk + 7919ull) & (SCRATCH_CHUNKS - 1);
        scratch[chunk * 32 + lane] = v;
      }
      steps += 16;
      unsigned long long s = lane == 0 ? ld_acquire_gpu_u64(stop) : 0;
      done = __shfl_sync(0xffffffffu, s, 0) != 0ull;
    }
    unsigned long long g1 = gtimer_ns();
    if (lane == 0) {
      int const slot = blockIdx.x * BG_WARPS + w;
      bg_bytes[slot] = steps * 512;
      bg_window[2 * slot] = g0;
      bg_window[2 * slot + 1] = g1;
    }
    return;
  }
  if (threadIdx.x) {
    return;
  }
  atom_add_release_gpu_u64(arrive, 1ull);
  // Start timing only once every background block is running.
  while (ld_acquire_gpu_u64(arrive) < (unsigned long long)gridDim.x) {
  }
  unsigned long long acc = 0;
  for (int r = 0; r < R_SM; r++) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
      acc += ld_acquire_gpu_u64(x);
    }
    long long t1 = clock64();
    out[r] = (double)(t1 - t0) / iters;
  }
  sink[0] = acc;
  smid_out[0] = sm_id();
  st_relaxed_gpu_u64(stop, 1ull);
}

// ---------------------------------------------------------------------------
// Atomic add, contended: one thread in each of N blocks, one block per SM,
// hammering ONE address. This is the shape of a phase barrier -- every SM
// incrementing one counter. Blocks rendezvous on-device before timing so the
// window excludes launch skew; a host-side gate cannot be used, since it would
// be queued behind this kernel on the stream and deadlock.
// ---------------------------------------------------------------------------
__global__ void atomic_contended(unsigned long long *ctr,
                                 unsigned long long *arrive,
                                 unsigned long long *cycles_out,
                                 int *smid_out,
                                 int iters) {
  if (threadIdx.x) {
    return;
  }
  atom_add_release_gpu_u64(arrive, 1ull);
  while (ld_acquire_gpu_u64(arrive) < (unsigned long long)gridDim.x) {
  }
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    atom_add_release_gpu_u64(ctr, 1ull);
  }
  long long t1 = clock64();
  cycles_out[blockIdx.x] = (unsigned long long)(t1 - t0);
  smid_out[blockIdx.x] = sm_id();
}

// ---------------------------------------------------------------------------
// __syncthreads() across a whole CTA, swept over block size: the intra-task
// barrier the warp-specialized kernels take between stages. It never leaves
// the SM, so one SM's number stands for all of them.
// ---------------------------------------------------------------------------
__global__ void barrier_cost(unsigned long long *sink, int iters) {
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    __syncthreads();
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) {
    sink[0] = (unsigned long long)(t1 - t0);
  }
}

// ---------------------------------------------------------------------------

// Per-SM medians (ns) from a kernel that wrote R_SM repetitions per block,
// keyed by SM id so passes over different addresses can be compared SM by SM.
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

// One full per-SM pass per address.
using PerAddr = std::vector<PerSm>;

template <typename Pass>
static PerAddr over_addresses(unsigned long long *buf, Pass pass) {
  PerAddr pa;
  for (int a = 0; a < NADDR; a++) {
    pa.push_back(pass(nth_address(buf, a)));
  }
  return pa;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0); // so a hang still shows progress
  char const *json_path = (argc > 1) ? argv[1] : nullptr;
  int const ITERS = 2000;

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
  double const peak = hbm_peak_gbps();
  guard.ghz_start = ghz;
  guard.spread_start = alu_sentinel_spread(nsm, smem);

  printf("device       : %s (SM %d.%d, %d SMs)\n",
         prop.name,
         prop.major,
         prop.minor,
         nsm);
  printf("driver/runtime: %d / %d\n", driver, runtime);
  printf("SM clock     : %.3f GHz (measured)\n\n", ghz);

  unsigned long long *d_x, *d_ready, *d_ticket, *d_arrive, *d_stop, *d_scratch,
      *d_sink, *d_cycles, *d_contended;
  double *d_out, *d_out2;
  int *d_smid;
  CHECK(cudaMalloc(&d_x, ADDR_SPAN + 4096));
  CHECK(cudaMalloc(&d_ready, ADDR_SPAN + 4096));
  CHECK(cudaMalloc(&d_ticket, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_contended, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_arrive, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_stop, sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_scratch, SCRATCH_ELEMS * sizeof(unsigned long long)));
  unsigned long long *d_bg_bytes, *d_bg_window;
  CHECK(
      cudaMalloc(&d_bg_bytes, MAX_SMS * BG_WARPS * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_bg_window,
                   2 * MAX_SMS * BG_WARPS * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_sink, MAX_SMS * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_cycles, MAX_SMS * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&d_out, MAX_SMS * R_SM * sizeof(double)));
  CHECK(cudaMalloc(&d_out2, MAX_SMS * R_SM * sizeof(double)));
  CHECK(cudaMalloc(&d_smid, MAX_SMS * sizeof(int)));
  CHECK(cudaMemset(d_x, 0, ADDR_SPAN + 4096));
  CHECK(cudaMemset(d_ready, 0, ADDR_SPAN + 4096));

  allow_dynamic_smem(atomic_per_sm<0>, smem);
  allow_dynamic_smem(atomic_per_sm<1>, smem);
  allow_dynamic_smem(acquire_per_sm, smem);
  allow_dynamic_smem(runtime_poll_per_sm, smem);
  allow_dynamic_smem(fence_per_sm, smem);
  allow_dynamic_smem(acquire_under_load, smem);
  allow_dynamic_smem(atomic_contended, smem);

  Json js;
  js.open(json_path);
  js.kvs("device", prop.name);
  js.kv("sm_count", nsm);
  js.kv("driver_version", driver);
  js.kv("sm_clock_ghz", ghz);

  printf("each row pools %d SMs x %d addresses. Where the samples form two "
         "separated groups,\nnear and far are reported apart, with a check "
         "that the SMs split into two fixed groups.\n",
         nsm,
         NADDR);
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
    printf("%-44s %8.1f %8.1f %8.1f   [%5.1f .. %5.1f]   %zu\n",
           label,
           d.med,
           d.p10,
           d.p90,
           d.lo,
           d.hi,
           v.size());
    js.dist(key, d);
  };
  auto finish = [&](char const *label, char const *key, PerAddr const &pa) {
    std::vector<double> all;
    for (PerSm const &m : pa) {
      for (auto const &kv : m) {
        all.push_back(kv.second);
      }
    }
    double cut = 0;
    if (!find_split(all, &cut)) {
      row(label, key, all);
      return;
    }
    std::vector<double> near, far;
    for (double x : all) {
      (x < cut ? near : far).push_back(x);
    }
    char lbl[96], k[96];
    snprintf(lbl, sizeof(lbl), "%s, near", label);
    snprintf(k, sizeof(k), "%s_near", key);
    row(lbl, k, near);
    snprintf(lbl, sizeof(lbl), "%s, far", label);
    snprintf(k, sizeof(k), "%s_far", key);
    row(lbl, k, far);
    // Near/far is a fixed partition of the SMs if every address's near set
    // is either the first address's near set or its exact complement.
    std::set<int> first;
    for (auto const &kv : pa[0]) {
      if (kv.second < cut) {
        first.insert(kv.first);
      }
    }
    int same = 0, flipped = 0, neither = 0;
    for (PerSm const &m : pa) {
      bool is_same = true, is_flipped = true;
      for (auto const &kv : m) {
        bool const is_near = kv.second < cut;
        bool const in_first = first.count(kv.first) > 0;
        is_same &= (is_near == in_first);
        is_flipped &= (is_near != in_first);
      }
      same += is_same;
      flipped += is_flipped && !is_same;
      neither += !is_same && !is_flipped;
    }
    printf("  SM groups of %zu and %zu: %d addresses near the first, %d near "
           "the second, %d fit neither\n",
           first.size(),
           pa[0].size() - first.size(),
           same,
           flipped,
           neither);
    snprintf(k, sizeof(k), "%s_addresses_fitting_two_groups", key);
    js.kv(k, same + flipped);
  };

  // --- atomic add, one address, result discarded vs consumed ---
  finish("atom.add.gpu.u64 discarded",
         "atomic_add_discarded_ns",
         over_addresses(d_x, [&](unsigned long long *x) {
           ticketed();
           atomic_per_sm<0>
               <<<nsm, 32, smem>>>(x, d_ticket, ITERS, d_out, d_smid, d_sink);
           CHECK_LAUNCH();
           return per_sm_medians(d_out, d_smid, nsm, ghz);
         }));

  finish("atom.add.gpu.u64 consumed",
         "atomic_add_consumed_ns",
         over_addresses(d_x, [&](unsigned long long *x) {
           ticketed();
           atomic_per_sm<1>
               <<<nsm, 32, smem>>>(x, d_ticket, ITERS, d_out, d_smid, d_sink);
           CHECK_LAUNCH();
           return per_sm_medians(d_out, d_smid, nsm, ghz);
         }));

  // --- acquire load, idle machine ---
  finish("ld.acquire.gpu.u64 (idle)",
         "acquire_idle_ns",
         over_addresses(d_x, [&](unsigned long long *x) {
           ticketed();
           acquire_per_sm<<<nsm, 32, smem>>>(
               x, d_ticket, ITERS, d_out, d_smid, d_sink);
           CHECK_LAUNCH();
           return per_sm_medians(d_out, d_smid, nsm, ghz);
         }));

  // --- acquire load, every other SM streaming writes ---
  {
    std::vector<double> bg_gbps;
    char lbl[64];
    snprintf(lbl, sizeof(lbl), "ld.acquire.gpu.u64 (%d SMs writing)", nsm - 1);
    finish(lbl,
           "acquire_loaded_ns",
           over_addresses(d_x, [&](unsigned long long *x) {
             PerSm v;
             std::vector<int> sm(nsm);
             int const slots = nsm * BG_WARPS;
             std::vector<unsigned long long> bytes(slots), win(2 * slots);
             for (int p = 0; p < nsm; p++) {
               CHECK(cudaMemset(d_arrive, 0, sizeof(unsigned long long)));
               CHECK(cudaMemset(d_stop, 0, sizeof(unsigned long long)));
               acquire_under_load<<<nsm, 32 * BG_WARPS, smem>>>(
                   x,
                   (uint4 *)d_scratch,
                   d_arrive,
                   d_stop,
                   ITERS,
                   p,
                   d_out,
                   d_smid,
                   d_sink,
                   d_bg_bytes,
                   d_bg_window);
               CHECK_LAUNCH();
               CHECK(cudaMemcpy(bytes.data(),
                                d_bg_bytes,
                                slots * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
               CHECK(cudaMemcpy(win.data(),
                                d_bg_window,
                                2 * slots * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
               unsigned long long total = 0, start = ~0ull, end = 0;
               for (int i = 0; i < slots; i++) {
                 if (i / BG_WARPS != p) {
                   total += bytes[i];
                   start = std::min(start, win[2 * i]);
                   end = std::max(end, win[2 * i + 1]);
                 }
               }
               bg_gbps.push_back((double)total / (double)(end - start));
               double r[R_SM];
               CHECK(cudaMemcpy(r, d_out, sizeof(r), cudaMemcpyDeviceToHost));
               CHECK(cudaMemcpy(
                   &sm[p], d_smid, sizeof(int), cudaMemcpyDeviceToHost));
               std::sort(r, r + R_SM);
               v[sm[p]] = r[R_SM / 2] / ghz;
             }
             require_every_sm_once(sm, nsm);
             return v;
           }));
    Dist bg = distribution(bg_gbps);
    printf("  background writes: median %.0f GB/s (%.0f%% of HBM peak), "
           "lowest %.0f GB/s\n",
           bg.med,
           peak > 0 ? 100.0 * bg.med / peak : 0.0,
           bg.lo);
    if (peak > 0 && bg.med > peak * 1.02) {
      printf("  (above the HBM peak: L2 is absorbing the writes, so this is "
             "not HBM pressure)\n");
    }
    js.dist("acquire_loaded_background_gbps", bg);
  }

  // --- the runtime's poll loop, one iteration ---
  for (int queues : {1, 2}) {
    char lbl[64], key[64];
    snprintf(lbl,
             sizeof(lbl),
             "runtime poll iteration, %d queue%s",
             queues,
             queues > 1 ? "s" : "");
    snprintf(key, sizeof(key), "runtime_poll_iteration_%d_queues_ns", queues);
    finish(lbl, key, over_addresses(d_ready, [&](unsigned long long *ready) {
             ticketed();
             runtime_poll_per_sm<<<nsm, 32, smem>>>(
                 ready, nsm, queues, d_ticket, ITERS, d_out, d_smid);
             CHECK_LAUNCH();
             return per_sm_medians(d_out, d_smid, nsm, ghz);
           }));
  }

  // --- fence, marginal over a store-only baseline ---
  {
    PerAddr store;
    PerAddr fence = over_addresses(d_x, [&](unsigned long long *x) {
      ticketed();
      fence_per_sm<<<nsm, 32, smem>>>(
          x, d_ticket, ITERS, d_out, d_out2, d_smid);
      CHECK_LAUNCH();
      store.push_back(per_sm_medians(d_out2, d_smid, nsm, ghz));
      return per_sm_medians(d_out, d_smid, nsm, ghz);
    });
    finish("__threadfence() marginal", "threadfence_marginal_ns", fence);
    finish("st.relaxed.gpu.u64 (fence baseline)", "relaxed_store_ns", store);
  }

  // --- __syncthreads across block sizes (on-SM, one SM suffices) ---
  printf("\n%-44s %8s      %s\n",
         "(one SM, 7 repetitions)",
         "median",
         "[min .. max]");
  for (int bs : {128, 256, 512}) {
    double v[REPS];
    for (int r = 0; r < REPS; r++) {
      barrier_cost<<<1, bs>>>(d_sink, ITERS);
      CHECK_LAUNCH();
      unsigned long long c;
      CHECK(cudaMemcpy(&c, d_sink, sizeof(c), cudaMemcpyDeviceToHost));
      v[r] = (double)c / ITERS / ghz;
    }
    Stat st = summarize(v, REPS);
    char lbl[64], key[64];
    snprintf(lbl, sizeof(lbl), "__syncthreads() (%d threads)", bs);
    report(lbl, st, "ns");
    snprintf(key, sizeof(key), "syncthreads_%d_threads_ns", bs);
    js.kv(key, st.med);
  }

  // --- contended atomic: the phase-barrier shape ---
  printf("\natom.add.release.gpu.u64 on ONE address, N SMs contending\n");
  printf("  %-8s %16s %18s\n", "SMs", "slowest ns/op", "aggregate ops/us");
  for (int nb : sm_counts(nsm)) {
    CHECK(cudaMemset(d_contended, 0, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_arrive, 0, sizeof(unsigned long long)));
    atomic_contended<<<nb, 32, smem>>>(
        d_contended, d_arrive, d_cycles, d_smid, ITERS * 5);
    CHECK_LAUNCH();
    std::vector<unsigned long long> hc(nb);
    std::vector<int> sm(nb);
    CHECK(cudaMemcpy(hc.data(),
                     d_cycles,
                     nb * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(
        sm.data(), d_smid, nb * sizeof(int), cudaMemcpyDeviceToHost));
    require_every_sm_once(sm, nb);
    // Verify the work actually happened: every block's increments must be
    // present in the counter, or the timing means nothing.
    unsigned long long final_ctr = 0;
    CHECK(cudaMemcpy(
        &final_ctr, d_contended, sizeof(final_ctr), cudaMemcpyDeviceToHost));
    unsigned long long expect = (unsigned long long)nb * ITERS * 5;
    if (final_ctr != expect) {
      printf("  FAIL: counter %llu != expected %llu\n", final_ctr, expect);
      exit(1);
    }
    // The slowest SM bounds the barrier: that is what the runtime waits on.
    unsigned long long mx = *std::max_element(hc.begin(), hc.end());
    double per_op = (double)mx / (ITERS * 5) / ghz;
    double agg = (double)nb * ITERS * 5 / ((double)mx / ghz / 1000.0);
    printf("  %-8d %16.1f %18.1f\n", nb, per_op, agg);
    char key[64];
    snprintf(key, sizeof(key), "atomic_contended_%d_sms_ns", nb);
    js.kv(key, per_op);
  }

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
