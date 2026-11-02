// rank_host.cuh (HIP) -- the per-process host code of a generated layer kernel
// (one tensor-parallel rank per process, as in a serving framework), for any
// layer; the AMD counterpart of B300's mpk_layer_host.cuh, with the same API.
// The generator (amd/tools/gen_rank_lib.py) takes the compiler's layer.cu,
// keeps its kernel and task tables, drops its host part (which drives all GPUs
// from one process) and includes this file after defining
//   MPK_NUM_GPUS, MPK_NLISTS          GPUs and CU lists of the schedules
//   mpk_tasks_of[g], mpk_begin_of[g]  the generated task tables of GPU g
//   MPK_TABLES                        the tables file (included below):
//     mpk_slot_args[][2]              the slot args, as static_host_init gets them
//     mpk_weight_names[MPK_NW]        the graph tensors a layer binds once
//                                     (weights), in mpk_layer_create's order
//     mpk_run_names[MPK_NR]           the graph tensors bound per run (inputs,
//                                     outputs, states), in mpk_run's order
//     MPK_H                           the hidden width
// The buffers between the tasks, their resets and the maps come from host.cuh
// (build_bufs, build_shell_buffers, build_maps), as static_host_init builds
// them. The kernel resets its buffers and keeps its launch number in device
// memory (per CU), so one run is one launch, which a HIP graph can capture.
//
// The exchange region: one per rank (both sets), allocated here
// (mpk_region_alloc) and shared with the other ranks' processes by HIP IPC
// handles (mpk_region_handle / mpk_region_open); the caller exchanges the
// handles over its process group. The kernel's push16 stores into every rank's
// region (G::peer), as in the single-process harness (host.cuh, peer access).
//
// API (extern "C"), on the calling process's current HIP device:
//   mpk_sizes(T, H, rg_bytes)           the exchange region's bytes (both sets)
//   mpk_region_alloc(bytes, &p)         this rank's region (zero-filled)
//   mpk_region_handle(p, h[64])         its IPC handle
//   mpk_region_open(h[64], &p)          a peer's region, mapped in this process
//   mpk_init(rank, tp, rv_all)          once per process: the buffers between
//                                       the tasks (shared by all layers), the
//                                       task table; rv_all: every rank's region
//   mpk_num_weights() / mpk_weight_name(i), mpk_num_run() / mpk_run_name(i)
//   mpk_layer_create(w[MPK_NW])         per model layer: this rank's weights
//                                       (device pointers) -> the layer handle
//   mpk_run(layer, stream, io[MPK_NR])  one run on `stream`: one launch
//   mpk_span(layer, bar, end)           the last run on this GPU (synchronous):
//                                       last CU past its start, latest CU end (ns)
//   mpk_stamps(layer, out, n)           the last run's per-CU stamps [NSM][NSTAMP]
//   mpk_tstamp(stream, out)             one wall-clock stamp (ns) into *out
//   mpk_test_align(stream)              tests: the ranks line up on the GPU
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include "mirage/static_megakernel_hip/host.cuh"

#include MPK_TABLES

namespace mpk_host {
using namespace static_mk;
using namespace static_host;

struct Layer {
  Maps maps;
  G g;
  std::map<std::string, void *> weights;
  std::map<std::string, void *> mapped; // the run tensors the maps were built with
  bool maps_built = false;
};
static struct {
  bool inited = false;
  G scratch;
  StaticTask *tasks = nullptr;
  GpuState owned;
  unsigned *align_slots[MPK_NUM_GPUS] = {}; // mpk_test_align: every rank's slots
  unsigned *align_round = nullptr;
  int rank = 0, tp = 0;
  std::set<std::string> map_sources; // the graph tensors read through maps
} P;
static std::map<std::string, std::string> slot_arg_map() {
  std::map<std::string, std::string> a;
  for (auto const &kv : mpk_slot_args) {
    a[kv[0]] = kv[1];
  }
  return a;
}
} // namespace mpk_host

extern "C" int mpk_sizes(int *T_, int *H_, unsigned long long *rg_bytes) {
  *T_ = static_mk::T;
  *H_ = MPK_H;
  *rg_bytes = 2 * static_mk::EXCHANGE_SET;
  return 0;
}
extern "C" int mpk_num_weights() { return MPK_NW; }
extern "C" char const *mpk_weight_name(int i) {
  return (i >= 0 && i < MPK_NW) ? mpk_weight_names[i] : nullptr;
}
extern "C" int mpk_num_run() { return MPK_NR; }
extern "C" char const *mpk_run_name(int i) {
  return (i >= 0 && i < MPK_NR) ? mpk_run_names[i] : nullptr;
}

// this rank's exchange region (hipMalloc, as the harness's)
extern "C" int mpk_region_alloc(unsigned long long bytes, void **p) {
  hipError_t e = hipMalloc(p, bytes);
  if (e != hipSuccess) {
    return (int)e;
  }
  e = hipMemset(*p, 0, bytes);
  if (e == hipSuccess) {
    e = hipDeviceSynchronize();
  }
  return (int)e;
}
extern "C" int mpk_region_handle(void *p, unsigned char *h) {
  static_assert(sizeof(hipIpcMemHandle_t) <= 64, "the handle in 64 bytes");
  hipIpcMemHandle_t ih;
  hipError_t const e = hipIpcGetMemHandle(&ih, p);
  if (e != hipSuccess) {
    return (int)e;
  }
  memset(h, 0, 64);
  memcpy(h, &ih, sizeof(ih));
  return 0;
}
extern "C" int mpk_region_open(unsigned char const *h, void **p) {
  hipIpcMemHandle_t ih;
  memcpy(&ih, h, sizeof(ih));
  return (int)hipIpcOpenMemHandle(p, ih, hipIpcMemLazyEnablePeerAccess);
}

extern "C" int mpk_init(int rank, int tp, void **rv_all) {
  using namespace mpk_host;
  if (P.inited) {
    return 0;
  }
  if (tp != MPK_NUM_GPUS || rank < 0 || rank >= tp) {
    fprintf(stderr, "mpk_init: rank %d of %d; the layer is built for %d GPUs\n", rank, tp, MPK_NUM_GPUS);
    return 1;
  }
  if (MPK_NLISTS != NSM) {
    fprintf(stderr, "mpk_init: %d task lists, the kernel is built for %d CUs\n", MPK_NLISTS, NSM);
    return 1;
  }
  int dev = 0, ncu = 0, khz = 0;
  MKS_CK(hipGetDevice(&dev));
  MKS_CK(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev));
  if (ncu < NSM) {
    fprintf(stderr, "mpk_init: the GPU has %d CUs, the kernel needs %d\n", ncu, NSM);
    return 1;
  }
  MKS_CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev));
  if ((long long)khz * NS_PER_TICK != 1000000) {
    fprintf(stderr, "mpk_init: wall clock %d kHz; the stamps assume %lld ns per tick\n", khz, NS_PER_TICK);
    return 1;
  }
  GpuState &s = P.owned;
  G &g = s.g;
  g = G{};
  g.rank = rank;
  g.rv = (unsigned char *)rv_all[rank];
  for (int q = 0; q < GPUS; q++) {
    g.peer[q] = (unsigned char *)rv_all[q];
  }
  build_bufs(s, {}, slot_arg_map());
  build_shell_buffers(s);
  // a graph tensor some task reads by pointer that is another node's output
  // buffer: the same memory (the harness binds both to one tensor by name)
  {
    auto const bufs = slot_args(slot_arg_map(), "buf.", MAX_BUFS);
    for (int i = 0; i < MAX_BUFS; i++) {
      if (!bufs[i].empty() && bufs[i][0] == "tensor" && s.bufs.count(bufs[i][1])) {
        g.buf[i] = s.bufs[bufs[i][1]];
        for (ParPending const &q : s.par_pending) { // an arena buffer: the slot follows the launch's copy
          if (q.name == bufs[i][1]) {
#if STATIC_MAX_BUFS > 64
            if (i >= 64) {
              g.par_mask_hi |= 1ull << (i - 64);
            } else
#endif
            g.par_mask |= 1ull << i;
          }
        }
      }
    }
  }
#if defined(STATIC_EARLY_PREWAIT) || defined(STATIC_EARLY_RES)
#error "rank_host.cuh: build without the harness's line-up flags"
#endif
  // the resets (host.cuh static_host_reset, launch 0): every buffer filled here
  // once; the kernel then refills each before its use (the exchange buffers:
  // the set the next launch uses)
  ResetList &rl = g.resets;
  rl.cnt = 0;
  for (BufReset const &b : s.resets) {
    MKS_CK(hipMemset(b.p, b.value, b.bytes));
    if (b.arena) { // both copies
      MKS_CK(hipMemset(static_cast<unsigned char *>(b.p) + g.par_off, b.value, b.bytes));
    }
    if (rl.cnt == MAX_RESETS || b.value < 0 || b.bytes % 16 || (uintptr_t)b.p % 16) {
      fprintf(stderr, "mpk_init: reset %d of at most %d (%zu bytes, value %d): not 16-B\n", rl.cnt, MAX_RESETS, b.bytes,
              b.value);
      return 1;
    }
    rl.p[rl.cnt] = (unsigned char *)b.p;
    rl.n[rl.cnt] = b.bytes;
    rl.v[rl.cnt] = ((unsigned)b.value & 0xFFu) | ((unsigned)(b.xset + 1) << 8) | ((b.arena ? 1u : 0u) << 11);
    rl.cnt++;
  }
  MKS_CK(hipMemset(g.stamps, 0, NSM * NSTAMP * 8));
  MKS_CK(hipDeviceSynchronize());
  MKS_CK(hipFuncSetAttribute((void const *)layer_kernel, hipFuncAttributeMaxDynamicSharedMemorySize,
                             (int)STATIC_SMEM_BYTES));
  std::vector<StaticTask> table((size_t)MPK_NLISTS * MAX_TASKS_PER_SM, StaticTask{-1, 0, 0, 0});
  for (int cu = 0; cu < MPK_NLISTS; cu++) {
    int const b = mpk_begin_of[rank][cu], n = mpk_begin_of[rank][cu + 1] - b;
    if (n > MAX_TASKS_PER_SM - 1) {
      fprintf(stderr, "mpk_init: CU %d: %d tasks, at most %d\n", cu, n, MAX_TASKS_PER_SM - 1);
      return 1;
    }
    for (int i = 0; i < n; i++) {
      table[(size_t)cu * MAX_TASKS_PER_SM + i] = mpk_tasks_of[rank][b + i];
    }
  }
  P.tasks = dmalloc<StaticTask>(s, table.size() * sizeof(StaticTask));
  MKS_CK(hipMemcpy(P.tasks, table.data(), table.size() * sizeof(StaticTask), hipMemcpyHostToDevice));
  for (auto const &m : slot_args(slot_arg_map(), "map.", MAX_MAPS)) {
    if (m.size() > 1 && m[1].compare(0, 2, "t:") == 0) {
      P.map_sources.insert(m[1].substr(2));
    }
  }
  P.scratch = g;
  // mpk_test_align's slots: bytes [3072, 3072 + 4 tp) of set 0's hello area
  // (the harness's line-up uses [0, 128), [1024, 1152), [2048, 2064); the
  // kernel has no start barrier over the GPUs, so nothing else uses it), on
  // every rank
  static_assert(3072 + 4 * MPK_NUM_GPUS <= 4096, "mpk_test_align: the slots in the hello area");
  for (int q = 0; q < tp; q++) {
    P.align_slots[q] = (unsigned *)((unsigned char *)rv_all[q] + EXCHANGE_HELLO + 3072);
  }
  MKS_CK(hipMemset(P.align_slots[rank], 0, 4 * tp));
  P.align_round = dmalloc<unsigned>(s, 4);
  MKS_CK(hipMemset(P.align_round, 0, 4));
  P.rank = rank;
  P.tp = tp;
  MKS_CK(hipDeviceSynchronize());
  P.inited = true;
  return 0;
}

extern "C" void *mpk_layer_create(void **w) {
  using namespace mpk_host;
  if (!P.inited) {
    fprintf(stderr, "mpk_layer_create before mpk_init\n");
    return nullptr;
  }
  Layer *L = new Layer();
  L->g = P.scratch;
  if (getenv("MPK_LAYER_STAMPS")) { // (diagnosis) the layer's own stamps: mpk_stamps(layer) = this layer's last run
    MKS_CK(hipMalloc(&L->g.stamps, NSM * NSTAMP * 8));
    MKS_CK(hipMemset(L->g.stamps, 0, NSM * NSTAMP * 8));
  }
  for (int i = 0; i < MPK_NW; i++) {
    L->weights[mpk_weight_names[i]] = w[i];
  }
  auto const bufs = slot_args(slot_arg_map(), "buf.", MAX_BUFS);
  for (int i = 0; i < MAX_BUFS; i++) {
    if (!bufs[i].empty() && bufs[i][0] == "tensor" && L->weights.count(bufs[i][1])) {
      L->g.buf[i] = L->weights[bufs[i][1]];
    }
  }
  return L;
}

extern "C" int mpk_run(void *layer, hipStream_t st, void **io) {
  using namespace mpk_host;
  Layer *L = (Layer *)layer;
  G &g = L->g;
  std::map<std::string, void *> run;
  for (int i = 0; i < MPK_NR; i++) {
    run[mpk_run_names[i]] = io[i];
  }
  bool remap = !L->maps_built;
  for (auto const &kv : run) {
    if (P.map_sources.count(kv.first) && L->mapped[kv.first] != kv.second) {
      remap = true;
    }
  }
  if (remap) {
    std::map<std::string, void *> tensors = L->weights;
    for (auto const &kv : run) {
      tensors[kv.first] = kv.second;
    }
    build_maps(L->maps, P.owned, tensors, slot_arg_map());
    L->mapped = run;
    L->maps_built = true;
  }
  // the run tensors read or written by pointer: graph tensors and node output
  // buffers (e.g. the layer's output)
  auto const bufs = slot_args(slot_arg_map(), "buf.", MAX_BUFS);
  for (int i = 0; i < MAX_BUFS; i++) {
    if (!bufs[i].empty() && (bufs[i][0] == "tensor" || bufs[i][0] == "out") && run.count(bufs[i][1])) {
      g.buf[i] = run[bufs[i][1]];
#if STATIC_MAX_BUFS > 64
      if (i >= 64) {
        g.par_mask_hi &= ~(1ull << (i - 64));
      } else
#endif
      g.par_mask &= ~(1ull << i); // (the caller's tensor: one copy)
    }
  }
  hipLaunchKernelGGL(layer_kernel, dim3(MPK_NLISTS), dim3(256), (size_t)STATIC_SMEM_BYTES, st, L->maps, g,
                     (StaticTask const *)P.tasks);
  return (int)hipGetLastError();
}

// `stream` waits until *p >= value (hipStreamWaitValue32; capturable), then sets *p to 0 (reset: 1): the next kernel's
// stream waits until the kernel before it has all its workgroups in
extern "C" int mpk_stream_wait_ge(void *stream, unsigned *p, unsigned value, int reset) {
  hipError_t e = hipStreamWaitValue32((hipStream_t)stream, p, value, hipStreamWaitValueGte, 0xFFFFFFFFu);
  if (e == hipSuccess && reset) {
    e = hipMemsetD32Async((hipDeviceptr_t)p, 0, 1, (hipStream_t)stream);
  }
  return (int)e;
}

// one wall-clock stamp (ns) into *out on `stream`
__global__ void mpk_tstamp_kernel(long long *out) {
  if (threadIdx.x == 0) {
    *out = (long long)static_mk::gtime();
  }
}
extern "C" int mpk_tstamp(void *stream, long long *out) {
  hipLaunchKernelGGL(mpk_tstamp_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream, out);
  return (int)hipGetLastError();
}

// tests only: the ranks line up on the GPU (round n = this rank's counter + 1;
// lane q < tp writes n into rank q's slot[rank] and polls slot q of its own)
struct MpkAlignSlots {
  unsigned *p[MPK_NUM_GPUS];
};
__global__ void mpk_test_align_kernel(MpkAlignSlots s, int rank, int tp, unsigned *round) {
  unsigned const n = *round + 1;
  int const q = threadIdx.x;
  if (q < tp) {
    __hip_atomic_store(s.p[q] + rank, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    while (__hip_atomic_load(s.p[rank] + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != n) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  if (q == 0) {
    *round = n;
  }
}
extern "C" int mpk_test_align(void *stream) {
  using namespace mpk_host;
  if (!P.inited) {
    return 1;
  }
  MpkAlignSlots s{};
  for (int q = 0; q < P.tp; q++) {
    s.p[q] = P.align_slots[q];
  }
  hipLaunchKernelGGL(mpk_test_align_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream, s, P.rank, P.tp, P.align_round);
  return (int)hipGetLastError();
}

// the last run's per-CU stamps (ns, [NSM][NSTAMP]) into out; returns NSTAMP
extern "C" int mpk_stamps(void *layer, long long *out, int n) {
  using namespace mpk_host;
  G const &g = ((Layer *)layer)->g;
  if (n < NSM * NSTAMP) {
    return -1;
  }
  MKS_CK(hipMemcpy(out, g.stamps, NSM * NSTAMP * 8, hipMemcpyDeviceToHost));
  return NSTAMP;
}

// the last run on this GPU: the last CU past its start (bar), the latest CU end
extern "C" int mpk_span(void *layer, long long *bar, long long *end) {
  using namespace mpk_host;
  G const &g = ((Layer *)layer)->g;
  std::vector<long long> stm(NSM * NSTAMP);
  MKS_CK(hipMemcpy(stm.data(), g.stamps, NSM * NSTAMP * 8, hipMemcpyDeviceToHost));
  MKS_CK(hipMemcpy(bar, g.start_barrier, 8, hipMemcpyDeviceToHost));
  *end = 0;
  for (int i = 0; i < NSM; i++) {
    *end = std::max(*end, stm[i * NSTAMP + STAMP_END]);
  }
  return 0;
}
