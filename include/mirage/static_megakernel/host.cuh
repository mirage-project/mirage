// host.cuh -- the host code of the generated layer (static_schedule.py includes
// it in the layer's host part and calls the functions at the end of this file).
// It builds what the layers declared (static_megakernel.py host_slot_args), the
// same way for every task type:
//   "buf.<slot>" -> "<kind> <name> <bytes> <reset> <by reader>": G::buf[slot].
//     kind: out (a node's output: the graph tensor of that name if the caller
//       gave one, else allocated here), scratch (allocated here; 0 bytes:
//       none), tensor (the graph tensor of that name, read by pointer) or
//       exchange (in the exchange region at core.cuh exchange_offset[slot];
//       both sets).
//     reset: the byte value it is filled with before each launch (-1: none).
//     by reader 1: its one reader re-arms it in the kernel (REARM), so a
//       STATIC_RESET_IN_KERNEL host fills it only once.
//   "map.<slot>" -> "<kind> <source> <dims...>": Maps::m[slot].
//     source: t:<tensor name> (a graph tensor or a node's output buffer by its
//       tensor name) or b:<buffer slot>+<byte offset>.
//     kind and dims: bf16 <rows> <K> <box rows> | wblk <pieces> <box pieces> |
//       sf <chunks> <box chunks> | act8 <K bytes> <rows> |
//       act8kt <K bytes> <rows> <K tiles>
// The node output buffers are allocated first, then the scratch buffers, then
// the counters and the time stamps. Keep this order: the buffers' addresses
// decide where they fall in L2, and a different order makes the layer slower.
#pragma once
#include "core.cuh"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>

namespace static_host {
using namespace static_mk;

#define MKS_CU(x)                                                              \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char *s_;                                                          \
      cuGetErrorString(r_, &s_);                                               \
      fprintf(stderr, "static_host: %s @%d: %s\n", #x, __LINE__, s_);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define MKS_CK(x)                                                              \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "static_host: %s @%d: %s\n",                                     \
              #x,                                                              \
              __LINE__,                                                        \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// a buffer filled with byte `value` before each launch; by_reader: its reader
// re-arms it in the kernel, so a STATIC_RESET_IN_KERNEL host fills it only once
// (the other buffers: kernel_begin fills them at each launch's start)
struct BufReset {
  void *p;
  size_t bytes;
  int value;
  bool by_reader;
};
struct GpuState {
  G g;       // the kernel's parameter g (by value)
  Maps maps; // the kernel's parameter maps (__grid_constant__)
  std::vector<BufReset>
      resets; // every buffer filled before each launch (static_host_reset)
  std::map<std::string, void *> bufs; // the node output buffers by tensor name
  std::vector<void *> owned;          // cudaMalloc'd here
};
static std::vector<GpuState> g_gpus;

// ---- tensor maps ----
// bf16 [rows, K] -> dims {64, rows, K / 64}, box {64, box_rows, 2}, 128-B
// swizzle
static void map_bf16(
    CUtensorMap *m, void *base, uint64_t rows, uint32_t box_rows, uint64_t K) {
  cuuint64_t gd[3] = {64, rows, K / 64};
  cuuint64_t gs[2] = {K * 2, 128};
  cuuint32_t bd[3] = {64, box_rows, 2}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m,
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                3,
                                base,
                                gd,
                                gs,
                                bd,
                                es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// MXFP4 weight stored piece by piece: [piece][128 rows][64 B]; box = bt pieces
static void
    map_wblk(CUtensorMap *m, void *base, uint64_t ntiles, uint32_t bt = 1) {
  cuuint64_t gd[3] = {128, 128, ntiles};
  cuuint64_t gs[2] = {64, 8192};
  cuuint32_t bd[3] = {128, 128, bt}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m,
                                CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                                3,
                                base,
                                gd,
                                gs,
                                bd,
                                es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// scale chunks of 512 B; box = bc chunks
static void
    map_sf(CUtensorMap *m, void *base, uint64_t nchunks, uint32_t bc = 1) {
  cuuint64_t gd[3] = {128, 4, nchunks};
  cuuint64_t gs[2] = {128, 512};
  cuuint32_t bd[3] = {128, 4, bc}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                3,
                                base,
                                gd,
                                gs,
                                bd,
                                es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// e4m3 [rows, Kb] as {128, rows, Kb / 128}; box {128, 8, nkt}: nkt K tiles of 8
// rows
static void map_act8_kt(
    CUtensorMap *m, void *base, uint64_t Kb, uint64_t rows, uint32_t nkt) {
  cuuint64_t gd[3] = {128, rows, Kb / 128};
  cuuint64_t gs[2] = {Kb, 128};
  cuuint32_t bd[3] = {128, 8, nkt}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                3,
                                base,
                                gd,
                                gs,
                                bd,
                                es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// e4m3 [rows, Kb]; box {128, 8}
static void map_act8(CUtensorMap *m, void *base, uint64_t Kb, uint64_t rows) {
  cuuint64_t gd[2] = {Kb, rows};
  cuuint64_t gs[1] = {Kb};
  cuuint32_t bd[2] = {128, 8}, es[2] = {1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                2,
                                base,
                                gd,
                                gs,
                                bd,
                                es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ---- exchange region: one allocation per GPU bound to one multicast object; a
// multimem.st through the multicast address mcVA
//      lands in every GPU's copy. recv[r] = GPU r's copy.
static void mc_alloc(size_t need, unsigned char *&mcVA, unsigned char *recv[]) {
  for (int r = 0; r < GPUS; r++) {
    MKS_CK(cudaSetDevice(r));
    MKS_CK(cudaFree(0));
  }
  CUmulticastObjectProp mp = {};
  mp.numDevices = GPUS;
  mp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t mgran = 0;
  MKS_CU(cuMulticastGetGranularity(
      &mgran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  size_t const mcSize = (need + mgran - 1) / mgran * mgran;
  mp.size = mcSize;
  CUmemGenericAllocationHandle mcH;
  MKS_CU(cuMulticastCreate(&mcH, &mp));
  for (int r = 0; r < GPUS; r++) {
    CUdevice dev;
    MKS_CU(cuDeviceGet(&dev, r));
    MKS_CU(cuMulticastAddDevice(mcH, dev));
  }
  CUmemAccessDesc ad[GPUS];
  for (int r = 0; r < GPUS; r++) {
    ad[r].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad[r].location.id = r;
    ad[r].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  for (int r = 0; r < GPUS; r++) {
    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = r;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t agran = 0;
    MKS_CU(cuMemGetAllocationGranularity(
        &agran, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t const asz = (mcSize + agran - 1) / agran * agran;
    CUmemGenericAllocationHandle ph;
    MKS_CU(cuMemCreate(&ph, asz, &ap, 0));
    MKS_CU(cuMulticastBindMem(mcH, 0, ph, 0, mcSize, 0));
    CUdeviceptr va;
    MKS_CU(cuMemAddressReserve(&va, asz, 0, 0, 0));
    MKS_CU(cuMemMap(va, asz, 0, ph, 0));
    MKS_CU(cuMemSetAccess(va, asz, ad, GPUS));
    recv[r] = reinterpret_cast<unsigned char *>(va);
    MKS_CK(cudaSetDevice(r));
    MKS_CK(cudaMemset(recv[r], 0, mcSize));
  }
  CUdeviceptr va;
  MKS_CU(cuMemAddressReserve(&va, mcSize, 0, 0, 0));
  MKS_CU(cuMemMap(va, mcSize, 0, mcH, 0));
  MKS_CU(cuMemSetAccess(va, mcSize, ad, GPUS));
  mcVA = reinterpret_cast<unsigned char *>(va);
}

template <typename P>
static P *dmalloc(GpuState &s, size_t bytes) {
  void *p;
  MKS_CK(cudaMalloc(&p, bytes));
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

// ---- the buffers and maps of one GPU (init; a host that runs one GPU per
// process can call the same functions); g.rv must be set ---- the buffers
// (G::buf) from the slot args: node outputs, then scratch buffers (allocated,
// zero-filled, unless `given` has the tensor), graph tensors (from `given`; a
// host that sets them later may leave them out: nullptr), exchange buffers;
// their resets into s.resets
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
            s.resets.push_back(BufReset{(unsigned char *)p + set * EXCHANGE_SET,
                                        bytes,
                                        reset,
                                        by_reader});
          }
        }
        s.g.buf[i] = p;
        s.bufs[name] = p;
        continue;
      }
      if (bufs[i][0] == "tensor") {
        s.g.buf[i] = p;
        continue;
      }
      if (bytes == 0) { // a scratch buffer this build's node does not read
        continue;
      }
      if (!p) {
        p = dmalloc<unsigned char>(s, bytes);
        MKS_CK(cudaMemset(p, 0, bytes));
      }
      s.g.buf[i] = p;
      s.bufs[name] = p;
      if (reset >= 0) {
        s.resets.push_back(BufReset{p, bytes, reset, by_reader});
      }
    }
  }
}
// the kernel's own buffers: the counters, the time stamps
static void build_shell_buffers(GpuState &s) {
  G &g = s.g;
  g.cnt = dmalloc<uint32_t>(s, NCNT * 4);
  g.stamps = dmalloc<long long>(s, NSM * NSTAMP * 8);
  MKS_CK(cudaMemset(g.stamps, 0, NSM * NSTAMP * 8));
  g.start_barrier = dmalloc<long long>(s, 8);
  MKS_CK(cudaMemset(g.start_barrier, 0, 8));
  s.resets.push_back(BufReset{g.cnt, NCNT * 4, 0, false});
}
// the tensor maps (Maps::m) from the slot args; a t: source by name from
// `tensors`, else a node output buffer of s
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
    auto u = [&](int k) { return std::stoull(m[k]); };
    std::string const &kind = m[0];
    if (kind == "bf16" && m.size() == 5) {
      map_bf16(&maps.m[i], p, u(2), (uint32_t)u(4), u(3));
    } else if (kind == "wblk" && m.size() == 4) {
      map_wblk(&maps.m[i], p, u(2), (uint32_t)u(3));
    } else if (kind == "sf" && m.size() == 4) {
      map_sf(&maps.m[i], p, u(2), (uint32_t)u(3));
    } else if (kind == "act8" && m.size() == 4) {
      map_act8(&maps.m[i], p, u(2), u(3));
    } else if (kind == "act8kt" && m.size() == 5) {
      map_act8_kt(&maps.m[i], p, u(2), u(3), (uint32_t)u(4));
    } else {
      fprintf(stderr,
              "static_host: map.%d: bad declaration (%s, %zu fields)\n",
              i,
              kind.c_str(),
              m.size());
      exit(1);
    }
  }
}

} // namespace static_host

// ---- what the generated layer calls ----
// once: peer access, the multicast exchange region (both sets, zero-filled),
// per GPU the layer state G, its buffers and its maps
static void static_host_init(StaticContext &ctx,
                             std::map<std::string, std::string> const &args) {
  using namespace static_host;
  if (ctx.num_gpus != GPUS) {
    fprintf(stderr,
            "static_host: %d GPUs, the layer is built for %d\n",
            ctx.num_gpus,
            GPUS);
    exit(1);
  }
  if (ctx.num_lists != NSM) {
    fprintf(
        stderr,
        "static_host: %d task lists per GPU, the bodies are built for %d SMs\n",
        ctx.num_lists,
        NSM);
    exit(1);
  }
  MKS_CU(cuInit(0));
  // peer access between every pair of GPUs
  if (GPUS > 1) {
    for (int r = 0; r < GPUS; r++) {
      MKS_CK(cudaSetDevice(r));
      for (int p = 0; p < GPUS; p++) {
        if (p != r) {
          int can = 0;
          MKS_CK(cudaDeviceCanAccessPeer(&can, r, p));
          if (!can) {
            fprintf(stderr, "static_host: no peer access %d->%d\n", r, p);
            exit(1);
          }
          cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
          if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr,
                    "static_host: peer enable %d->%d: %s\n",
                    r,
                    p,
                    cudaGetErrorString(e));
            exit(1);
          }
          cudaGetLastError();
        }
      }
    }
  }
  unsigned char *mcVA = nullptr, *recv[GPUS] = {};
  if (GPUS > 1) {
    mc_alloc(2 * EXCHANGE_SET,
             mcVA,
             recv); // the exchange region's two sets (core.cuh EXCHANGE_SET),
                    // zero-filled
  }
  g_gpus.assign(GPUS, GpuState());
  for (int r = 0; r < GPUS; r++) {
    StaticGpuView const &v = ctx.gpus[r];
    GpuState &s = g_gpus[r];
    MKS_CK(cudaSetDevice(v.gpu));
    G &g = s.g;
    g = G{};
    g.rank = r;
    if (GPUS > 1) {
      g.mc = mcVA;
      g.rv = recv[r];
    } else {
      g.rv = dmalloc<unsigned char>(s, 2 * EXCHANGE_SET);
      MKS_CK(cudaMemset(g.rv, 0, 2 * EXCHANGE_SET));
      g.mc = nullptr;
    }
    build_bufs(s, v.tensors, args);
    build_shell_buffers(s);
    build_maps(s.maps, s, v.tensors, args);
  }
}

// before each launch, on the GPU's stream: every buffer of s.resets, the
// stamps; then the launch number g.gen
static void static_host_reset(int r,
                              cudaStream_t st,
                              unsigned long long launch_number) {
  using namespace static_host;
  GpuState &s = g_gpus[r];
  G &g = s.g;
  for (BufReset const &b : s.resets) {
    MKS_CK(cudaMemsetAsync(b.p, b.value, b.bytes, st));
  }
  MKS_CK(cudaMemsetAsync(g.stamps, 0, NSM * NSTAMP * 8, st));
  g.gen = (unsigned)(launch_number + 1);
}

// GPU r's instrumentation buffer `name`, for a test harness to read (p =
// nullptr: no such buffer): "counters" [NCNT] uint32 (the nodes' lines,
// NODE_COUNTER_LINES of 32), "stamps" [NSM][NSTAMP] globaltimer ns (config.cuh
// STAMP_*), "start_barrier" [1] globaltimer ns (the last SM past the start
// barrier)
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
  }
}

static void static_host_finalize(StaticContext &ctx) {
  using namespace static_host;
  for (size_t r = 0; r < g_gpus.size(); r++) {
    cudaSetDevice(ctx.gpus[r].gpu);
    for (void *p : g_gpus[r].owned) {
      cudaFree(p);
    }
  }
  g_gpus.clear();
}
