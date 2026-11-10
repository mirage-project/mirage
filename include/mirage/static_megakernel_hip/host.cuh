// host.cuh (HIP) -- the host code of the generated layer on AMD GPUs: the same
// declarations and the same functions as include/mirage/static_megakernel/
// host.cuh (static_schedule.py calls them by name), with three differences:
//   1. no TMA: a "map.<slot>" declaration becomes a plain descriptor (Maps::m[slot]
//      = {pointer, kind, dims}); the task templates compute addresses from it.
//   2. no multicast: the exchange region is one allocation per GPU (both sets);
//      every GPU gets every GPU's address (peer access), G::peer[r], and a
//      push16 stores into each of them (core.cuh).
//   3. the clock: the stamps are s_memrealtime ticks x NS_PER_TICK (ns, as on
//      CUDA); init checks the device's wall clock rate.
//   "buf.<slot>" -> "<kind> <name> <bytes> <reset> <by reader>": G::buf[slot]
//     (kind: out, scratch, tensor, exchange; reset: byte value before each
//     launch, -1 none; by reader 1: re-armed by its reader in the kernel).
//   "map.<slot>" -> "<kind> <source> <dims...>": Maps::m[slot]; source
//     t:<tensor name> or b:<buffer slot>+<byte offset>; kind and dims: bf16 <rows>
//     <K> <box rows> | wblk <pieces> <box pieces> | sf <chunks> <box chunks> |
//     act8 <K bytes> <rows> | act8kt <K bytes> <rows> <K tiles>
// Buffers are allocated in the same order as on CUDA (node outputs, scratch,
// counters, stamps).
// OPEN (needs the CDNA4 memory-model facts, amd/notes/cdna4_isa_facts.md): which
// memory type the exchange region and the counters need (coarse-grained
// hipMalloc vs fine-grained / uncached) for cross-GPU and cross-die polling.
#pragma once
#include "core.cuh"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <map>
#include <string>
#include <vector>

namespace static_host {
using namespace static_mk;

#define MKS_CK(x)                                                              \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr,                                                          \
              "static_host: %s @%d: %s\n",                                     \
              #x,                                                              \
              __LINE__,                                                        \
              hipGetErrorString(e_));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct BufReset {
  void *p;
  size_t bytes;
  int value;
  bool by_reader;
  int xset = -1; // an exchange buffer's set (0, 1); -1: a local buffer
  bool arena = false; // in the two-copy arena: copy 1 at + G::par_off
};
// Every local buffer the host would reset (node outputs, scratch, the counters)
// lives in one arena allocated twice (copy 1 at + par_off); a launch of parity p
// uses copy p (core.cuh buf_at, cnt_of) and resets copy 1 - p at its end
// (kernel_end); the exchange buffers keep their two sets (the next one reset at
// the start, as before). No barrier at the start, nothing for the host to do per
// launch (a HIP graph replays it)
struct ParPending {
  int slot;
  std::string name;
  size_t bytes;
  int value;
  bool by_reader;
  bool trim; // moe_experts.misc: its head and tail only
  void *mirror; // a buffer the caller gave (a harness reads its outputs there): the launch's copy copied to it after the launch
};
struct GpuState {
  G g;
  Maps maps;
  std::vector<BufReset> resets;
  std::map<std::string, void *> bufs;
  std::map<std::string, size_t> buf_bytes; // allocated here: readable by name
  std::vector<void *> owned;
  std::vector<ParPending> par_pending; // build_bufs -> build_shell_buffers
  std::vector<std::pair<std::string, void *>> par_mirror; // caller buffers: static_host_par_mirror
};
static std::vector<GpuState> g_gpus;

template <typename P>
static P *dmalloc(GpuState &s, size_t bytes) {
  void *p;
  MKS_CK(hipMalloc(&p, bytes));
  s.owned.push_back(p);
  return static_cast<P *>(p);
}

// the args "<prefix><slot>" (slot < n), split at spaces; a missing slot is an
// empty entry
static std::vector<std::vector<std::string>>
    slot_args(std::map<std::string, std::string> const &a,
              std::string const &prefix,
              int n) {
  std::vector<std::vector<std::string>> out(n);
  for (auto const &kv : a) {
    if (kv.first.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    int const slot = std::stoi(kv.first.substr(prefix.size()));
    if (slot < 0 || slot >= n) {
      fprintf(stderr,
              "static_host: %s: slot out of range (at most %d)\n",
              kv.first.c_str(),
              n);
      exit(1);
    }
    size_t i = 0;
    while (i < kv.second.size()) {
      size_t const j = kv.second.find(' ', i);
      out[slot].push_back(kv.second.substr(
          i, j == std::string::npos ? std::string::npos : j - i));
      if (j == std::string::npos) {
        break;
      }
      i = j + 1;
    }
  }
  return out;
}

// the buffers (G::buf) from the slot args, as on CUDA; exchange buffers point
// into this GPU's own exchange region (g.rv)
static void build_bufs(GpuState &s,
                       std::map<std::string, void *> const &given,
                       std::map<std::string, std::string> const &args) {
  auto const bufs = slot_args(args, "buf.", MAX_BUFS);
  for (int i = 0; i < MAX_BUFS; i++) {
    if (!bufs[i].empty() && bufs[i].size() != 5) {
      fprintf(stderr,
              "static_host: buf.%d: expected <kind> <name> <bytes> <reset> <by "
              "reader>\n",
              i);
      exit(1);
    }
  }
  for (char const *kind : {"out", "scratch", "tensor", "exchange"}) {
    for (int i = 0; i < MAX_BUFS; i++) {
      if (bufs[i].empty() || bufs[i][0] != kind) {
        continue;
      }
      std::string const &name = bufs[i][1];
      size_t const bytes = std::stoull(bufs[i][2]);
      int const reset = std::stoi(bufs[i][3]);
      bool const by_reader = bufs[i][4] == "1";
      auto it = given.find(name);
      void *p = (it != given.end() && it->second) ? it->second : nullptr;
      if (bufs[i][0] == "exchange") {
        p = s.g.rv + exchange_offset[i];
        if (exchange_bytes[i] != bytes) {
          fprintf(stderr,
                  "static_host: buf.%d (%s): %zu bytes, the layout says %zu\n",
                  i,
                  name.c_str(),
                  bytes,
                  exchange_bytes[i]);
          exit(1);
        }
        if (reset >= 0) {
          for (int set = 0; set < 2; set++) {
            s.resets.push_back(BufReset{(unsigned char *)p + set * EXCHANGE_SET, bytes, reset, by_reader, set});
          }
        }
        s.g.buf[i] = p;
        s.bufs[name] = p;
        s.buf_bytes[name] = EXCHANGE_SET + bytes; // readable by name: both sets (set 1 at + EXCHANGE_SET)
        continue;
      }
      if (bufs[i][0] == "tensor") {
        if (!p) { // (a graph tensor slot no given tensor binds: the kernel would read address 0 + offsets)
          fprintf(stderr, "static_host: buf.%d (%s): no tensor given\n", i, name.c_str());
        }
        s.g.buf[i] = p;
        continue;
      }
      if (bytes == 0) {
        continue;
      }
      if (reset >= 0) { // in the two-copy arena (build_shell_buffers)
        // a reset buffer the caller gives (a harness's output tensor): the kernel uses the arena's two copies; after a
        // launch static_host_par_mirror copies that launch's copy to the caller's (the per-rank library gives none)
        // moe_experts' scratch without its item trace (written by the timing build only, 512 KB): the queues before
        // it, the R blocks' counters after it (core.cuh MOE_MISC_RBLK)
#ifndef STATIC_TIMING_BUILD
        bool const trim = name == "moe_experts.misc";
#else
        bool const trim = false;
#endif
        s.par_pending.push_back(ParPending{i, name, bytes, reset, by_reader, trim, p});
        continue;
      }
      if (!p) {
        p = dmalloc<unsigned char>(s, bytes);
        MKS_CK(hipMemset(p, 0, bytes));
      }
      s.g.buf[i] = p;
      s.bufs[name] = p;
      s.buf_bytes[name] = bytes;
    }
  }
}

static void build_shell_buffers(GpuState &s) {
  G &g = s.g;
  { // the arena: the counters, then the deferred reset buffers (256-B aligned); allocated twice
    size_t at = ((size_t)NCNT * 4 + 255) & ~(size_t)255;
    std::vector<size_t> offs;
    for (ParPending const &q : s.par_pending) {
      offs.push_back(at);
      at += (q.bytes + 255) & ~(size_t)255;
    }
    unsigned char *const arena = dmalloc<unsigned char>(s, 2 * at);
    MKS_CK(hipMemset(arena, 0, 2 * at));
    g.par_off = (long long)at;
    g.par_mask = 0;
#if STATIC_MAX_BUFS > 64
    g.par_mask_hi = 0;
#endif
    g.cnt = reinterpret_cast<uint32_t *>(arena);
    s.resets.push_back(BufReset{g.cnt, NCNT * 4, 0, false, -1, true});
    for (size_t k = 0; k < s.par_pending.size(); k++) {
      ParPending const &q = s.par_pending[k];
      unsigned char *const p = arena + offs[k];
      g.buf[q.slot] = p;
#if STATIC_MAX_BUFS > 64
      if (q.slot >= 64) {
        g.par_mask_hi |= 1ull << (q.slot - 64);
      } else
#endif
      g.par_mask |= 1ull << q.slot;
      s.bufs[q.name] = p;
      s.buf_bytes[q.name] = q.bytes;
      if (q.mirror) {
        s.par_mirror.push_back({q.name, q.mirror});
      }
      if (q.trim) {
        size_t const trace = (size_t)NSM * 4 * 32 * 16, head = MOE_MISC_RBLK - trace;
        s.resets.push_back(BufReset{p, head, q.value, q.by_reader, -1, true});
        s.resets.push_back(BufReset{p + MOE_MISC_RBLK, q.bytes - MOE_MISC_RBLK, q.value, q.by_reader, -1, true});
      } else {
        s.resets.push_back(BufReset{p, q.bytes, q.value, q.by_reader, -1, true});
      }
    }
  }
  g.stamps = dmalloc<long long>(s, NSM * NSTAMP * 8);
  MKS_CK(hipMemset(g.stamps, 0, NSM * NSTAMP * 8));
  g.start_barrier = dmalloc<long long>(s, 8);
  MKS_CK(hipMemset(g.start_barrier, 0, 8));
  g.gen_ptr = dmalloc<unsigned>(s, 128);
  MKS_CK(hipMemset(g.gen_ptr, 0, 128));
  // (lines 65 .. 72: the per-CU launch counts, core.cuh kernel_begin; lines 0 .. 64 not used now)
  g.arrive = dmalloc<unsigned>(s, 73 * 128);
  MKS_CK(hipMemset(g.arrive, 0, 73 * 128));
}

// the descriptors (Maps::m) from the slot args: the source's address, the kind
// and its numbers as declared (core.cuh MapKind)
static void build_maps(Maps &maps,
                       GpuState const &s,
                       std::map<std::string, void *> const &tensors,
                       std::map<std::string, std::string> const &args) {
  auto const ms = slot_args(args, "map.", MAX_MAPS);
  for (int i = 0; i < MAX_MAPS; i++) {
    if (ms[i].empty()) {
      continue;
    }
    std::vector<std::string> const &m = ms[i];
    std::string const &src = m[1];
    unsigned char *p = nullptr;
    if (src.compare(0, 2, "t:") == 0) {
      auto it = tensors.find(src.substr(2));
      if (it != tensors.end()) {
        p = (unsigned char *)it->second;
      }
      if (!p) {
        auto b = s.bufs.find(src.substr(2));
        p = (b != s.bufs.end()) ? (unsigned char *)b->second : nullptr;
      }
    } else if (src.compare(0, 2, "b:") == 0) {
      size_t const plus = src.find('+');
      p = (unsigned char *)s.g.buf[std::stoi(src.substr(2, plus - 2))] +
          (plus == std::string::npos ? 0 : std::stoull(src.substr(plus + 1)));
    }
    if (!p) {
      fprintf(stderr, "static_host: map.%d: no tensor %s\n", i, src.c_str());
      exit(1);
    }
    std::string const &kind = m[0];
    static std::map<std::string, std::pair<MapKind, size_t>> const kinds = {
        {"bf16", {MAP_BF16, 5}},
        {"wblk", {MAP_WBLK, 4}},
        {"sf", {MAP_SF, 4}},
        {"act8", {MAP_ACT8, 4}},
        {"act8kt", {MAP_ACT8KT, 5}}};
    auto k = kinds.find(kind);
    if (k == kinds.end() || m.size() != k->second.second) {
      fprintf(stderr,
              "static_host: map.%d: bad declaration (%s, %zu fields)\n",
              i,
              kind.c_str(),
              m.size());
      exit(1);
    }
    MapDesc &d = maps.m[i];
    d.p = p;
    d.kind = k->second.first;
    for (size_t j = 2; j < m.size(); j++) {
      d.dims[j - 2] = std::stoull(m[j]);
    }
  }
}

} // namespace static_host

// STATIC_SOLO: the layer built for GPUS GPUs, run on ONE (GPU 0's task lists,
// rank 0): one exchange region, every peer pointer at it; the kernel skips the
// waits for other GPUs' data (core.cuh). For timing a GPU's share of the layer
// on one GPU (amd/tests/test_layer_one_gpu.py --solo); its numbers are not the
// layer's (the other GPUs' contributions are zeros).
#ifdef STATIC_SOLO
constexpr int RUN_GPUS = 1;
#else
constexpr int RUN_GPUS = static_host::GPUS;
#endif

// ---- what the generated layer calls ----
// once: peer access, the exchange regions (one per GPU, both sets,
// zero-filled), the wall clock check, per GPU the layer state G, its buffers
// and its descriptors
static void static_host_init(StaticContext &ctx,
                             std::map<std::string, std::string> const &args) {
  using namespace static_host;
  if (ctx.num_gpus != RUN_GPUS) {
    fprintf(stderr,
            "static_host: %d GPUs, the layer runs on %d\n",
            ctx.num_gpus,
            RUN_GPUS);
    exit(1);
  }
  if (ctx.num_lists != NSM) {
    fprintf(stderr,
            "static_host: %d task lists per GPU, the bodies are built for %d "
            "CUs\n",
            ctx.num_lists,
            NSM);
    exit(1);
  }
  for (int r = 0; r < RUN_GPUS; r++) {
    int khz = 0;
    MKS_CK(hipDeviceGetAttribute(
        &khz, hipDeviceAttributeWallClockRate, ctx.gpus[r].gpu));
    if ((long long)khz * NS_PER_TICK != 1000000) {
      fprintf(stderr,
              "static_host: GPU %d's wall clock runs at %d kHz; the stamps "
              "assume %lld ns per tick\n",
              r,
              khz,
              NS_PER_TICK);
      exit(1);
    }
  }
  if (RUN_GPUS > 1) {
    for (int r = 0; r < RUN_GPUS; r++) {
      MKS_CK(hipSetDevice(ctx.gpus[r].gpu));
      for (int p = 0; p < RUN_GPUS; p++) {
        if (p == r) {
          continue;
        }
        int can = 0;
        MKS_CK(hipDeviceCanAccessPeer(&can, ctx.gpus[r].gpu, ctx.gpus[p].gpu));
        if (!can) {
          fprintf(stderr, "static_host: no peer access %d->%d\n", r, p);
          exit(1);
        }
        hipError_t e = hipDeviceEnablePeerAccess(ctx.gpus[p].gpu, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
          fprintf(stderr,
                  "static_host: peer enable %d->%d: %s\n",
                  r,
                  p,
                  hipGetErrorString(e));
          exit(1);
        }
        (void)hipGetLastError();
      }
    }
  }
  g_gpus.assign(RUN_GPUS, GpuState());
  unsigned char *regions[GPUS] = {};
  for (int r = 0; r < RUN_GPUS; r++) {
    MKS_CK(hipSetDevice(ctx.gpus[r].gpu));
    regions[r] = dmalloc<unsigned char>(g_gpus[r], 2 * EXCHANGE_SET);
    MKS_CK(hipMemset(regions[r], 0, 2 * EXCHANGE_SET));
  }
  for (int r = 0; r < RUN_GPUS; r++) {
    StaticGpuView const &v = ctx.gpus[r];
    GpuState &s = g_gpus[r];
    MKS_CK(hipSetDevice(v.gpu));
    G &g = s.g;
    g = G{};
    g.rank = r;
    g.rv = regions[r];
    for (int p = 0; p < GPUS; p++) {
      g.peer[p] = regions[RUN_GPUS == 1 ? r : p];
    }
    build_bufs(s, v.tensors, args);
    build_shell_buffers(s);
    build_maps(s.maps, s, v.tensors, args);
  }
}

// before each launch, on the GPU's stream: every buffer of s.resets, the
// stamps; then the launch number g.gen
// after launch `launch_number` finished: its arena copy of every reset buffer the caller gave, to the caller's
static void static_host_par_mirror(int r, unsigned long long launch_number) {
  using namespace static_host;
  GpuState &s = g_gpus[r];
  (void)launch_number;
  unsigned gen = 0; // the launch's generation as the kernel counted it (core.cuh kernel_begin: CU 0's count)
  MKS_CK(hipMemcpy(&gen, s.g.arrive + 32 * 65, 4, hipMemcpyDeviceToHost));
  size_t const off = (gen & 1u) ? (size_t)s.g.par_off : 0;
  for (auto const &m : s.par_mirror) {
    MKS_CK(hipMemcpy(m.second, static_cast<unsigned char *>(s.bufs[m.first]) + off, s.buf_bytes[m.first], hipMemcpyDeviceToDevice));
  }
}
static void static_host_reset(int r,
                              hipStream_t st,
                              unsigned long long launch_number) {
  using namespace static_host;
  GpuState &s = g_gpus[r];
  G &g = s.g;
  // every buffer once, before the first launch; from then on the kernel resets
  // them (core.cuh kernel_begin, kernel_end). The stamps: kernel_begin.
  if (launch_number == 0) {
    g.resets.cnt = 0;
    for (BufReset const &b : s.resets) {
      MKS_CK(hipMemsetAsync(b.p, b.value, b.bytes, st));
      if (b.arena) { // both copies
        MKS_CK(hipMemsetAsync(static_cast<unsigned char *>(b.p) + g.par_off, b.value, b.bytes, st));
      }
      if (b.value > 0xFF) { // (a harness bound, MPK_BOUND_FRONT_READY: the low byte before the first launch only, no in-kernel reset)
        continue;
      }
      if (g.resets.cnt == MAX_RESETS || b.value < 0 || b.bytes % 16 != 0) {
        fprintf(stderr,
                "static_host: in-kernel resets: %d of at most %d, value %d, "
                "%zu bytes (a multiple of 16)\n",
                g.resets.cnt + 1,
                MAX_RESETS,
                b.value,
                b.bytes);
        exit(1);
      }
      g.resets.p[g.resets.cnt] = static_cast<unsigned char *>(b.p);
      g.resets.n[g.resets.cnt] = b.bytes;
      // bits 8..9: an exchange buffer's set + 1 (the kernel resets the set the next launch uses); bit 11: in the arena
      // (core.cuh kernel_end)
      g.resets.v[g.resets.cnt] =
          ((unsigned)b.value & 0xFFu) | ((unsigned)(b.xset + 1) << 8);
      g.resets.v[g.resets.cnt] |= (b.arena ? 1u : 0u) << 11;
      g.resets.cnt++;
    }
    MKS_CK(hipMemsetAsync(g.stamps, 0, NSM * NSTAMP * 8, st));
  }
  g.gen = (unsigned)(launch_number + 1);
}

// GPU r's instrumentation buffer `name` (p = nullptr: no such buffer):
// "counters" [NCNT] uint32, "stamps" [NSM][NSTAMP] ns, "start_barrier" [1] ns
static void static_host_buffer(int r,
                               std::string const &name,
                               void const *&p,
                               size_t &bytes) {
  using namespace static_host;
  G const &g = g_gpus[r].g;
  p = nullptr;
  bytes = 0;
  if (name == "counters") {
    p = g.cnt;
    bytes = NCNT * 4;
  } else if (name == "stamps") {
    p = g.stamps;
    bytes = NSM * NSTAMP * 8;
  } else if (name == "start_barrier") {
    p = g.start_barrier;
    bytes = 8;
  } else if (g_gpus[r].buf_bytes.count(name)) { // a buffer allocated here
    p = g_gpus[r].bufs.at(name);
    bytes = g_gpus[r].buf_bytes.at(name);
  }
}

static void static_host_finalize(StaticContext &ctx) {
  using namespace static_host;
  for (size_t r = 0; r < g_gpus.size(); r++) {
    (void)hipSetDevice(ctx.gpus[r].gpu);
    for (void *p : g_gpus[r].owned) {
      (void)hipFree(p);
    }
  }
  g_gpus.clear();
}
