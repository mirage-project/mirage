// moe_host.cuh -- host code of the K3 MoE task family (written from the hand-written reference kernel's host code). The generated
// layer (static_schedule.py) includes this file and calls:
//   moe_host_init(ctx, args)                     once: peer access, the multicast exchange region, per GPU the layer state G, its
//                                                buffers and the tensor maps (the per-SM task tables are the generator's)
//   moe_host_reset(ctx, gpu, stream, launch)     before each launch, after the generator's 512 MB L2 write: counters and the
//                                                0xFF / 0 fills of the buffers
//   the kernel, with (moe_host::g_gpus[gpu].maps, moe_host::g_gpus[gpu].g) and the generator's task table as parameters
//   moe_host_timing(ctx, gpu, bar, end)          the span: last SM past the start barrier, latest SM end (globaltimer ns)
//   moe_host_phase_stamps(ctx, gpu, out)         per SM: expert queue start / end
//   moe_host_report(ctx), moe_host_finalize(ctx)
// The weights, x, the residual, the bias, gamma and the outputs y, R, S are the graph's tensors (args: role -> tensor name);
// the router and latent_down K parts are args too (the compiler's choice).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "moe_kernel.cuh"

namespace moe_host {
using namespace static_mk;

#define MKS_CU(x)                                                                                            \
  do {                                                                                                       \
    CUresult r_ = (x);                                                                                       \
    if (r_ != CUDA_SUCCESS) { const char *s_; cuGetErrorString(r_, &s_); fprintf(stderr, "moe_host: %s @%d: %s\n", #x, __LINE__, s_); exit(1); } \
  } while (0)
#define MKS_CK(x)                                                                                            \
  do {                                                                                                       \
    cudaError_t e_ = (x);                                                                                    \
    if (e_ != cudaSuccess) { fprintf(stderr, "moe_host: %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } \
  } while (0)

struct GpuState {
  G g;                       // the kernel's parameter g (by value)
  Maps maps;                 // the kernel's parameter maps (__grid_constant__)
  int nsplit_router = 0, nsplit_latent = 0;
  std::vector<void *> owned; // cudaMalloc'd here
};
static std::vector<GpuState> g_gpus;

// ---- tensor maps ----
// bf16 [rows, K] -> dims {64, rows, K / 64}, box {64, box_rows, 2}, 128-B swizzle
static void map_bf16(CUtensorMap *m, void *base, uint64_t rows, uint32_t box_rows, uint64_t K) {
  cuuint64_t gd[3] = {64, rows, K / 64}; cuuint64_t gs[2] = {K * 2, 128}; cuuint32_t bd[3] = {64, box_rows, 2}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// MXFP4 weight stored piece by piece: [piece][128 rows][64 B]; box = bt pieces
static void map_wblk(CUtensorMap *m, void *base, uint64_t ntiles, uint32_t bt = 1) {
  cuuint64_t gd[3] = {128, 128, ntiles}; cuuint64_t gs[2] = {64, 8192}; cuuint32_t bd[3] = {128, 128, bt}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, base, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// scale chunks of 512 B; box = bc chunks
static void map_sf(CUtensorMap *m, void *base, uint64_t nchunks, uint32_t bc = 1) {
  cuuint64_t gd[3] = {128, 4, nchunks}; cuuint64_t gs[2] = {128, 512}; cuuint32_t bd[3] = {128, 4, bc}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, base, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// e4m3 [rows, Kb] as {128, rows, Kb / 128}; box {128, 8, nkt}: nkt K tiles of 8 rows
static void map_act8_kt(CUtensorMap *m, void *base, uint64_t Kb, uint64_t rows, uint32_t nkt) {
  cuuint64_t gd[3] = {128, rows, Kb / 128}; cuuint64_t gs[2] = {Kb, 128}; cuuint32_t bd[3] = {128, 8, nkt}, es[3] = {1, 1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, base, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
// e4m3 [rows, Kb]; box {128, 8}
static void map_act8(CUtensorMap *m, void *base, uint64_t Kb, uint64_t rows) {
  cuuint64_t gd[2] = {Kb, rows}; cuuint64_t gs[1] = {Kb}; cuuint32_t bd[2] = {128, 8}, es[2] = {1, 1};
  MKS_CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ---- exchange region: one allocation per GPU bound to one multicast object; a multimem.st through the multicast address mcVA
//      lands in every GPU's copy. recv[r] = GPU r's copy.
static void mc_alloc(int tp, size_t need, unsigned char *&mcVA, unsigned char *recv[]) {
  for (int r = 0; r < tp; r++) { MKS_CK(cudaSetDevice(r)); MKS_CK(cudaFree(0)); }
  CUmulticastObjectProp mp = {}; mp.numDevices = tp; mp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t mgran = 0; MKS_CU(cuMulticastGetGranularity(&mgran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  size_t const mcSize = (need + mgran - 1) / mgran * mgran; mp.size = mcSize;
  CUmemGenericAllocationHandle mcH; MKS_CU(cuMulticastCreate(&mcH, &mp));
  for (int r = 0; r < tp; r++) { CUdevice dev; MKS_CU(cuDeviceGet(&dev, r)); MKS_CU(cuMulticastAddDevice(mcH, dev)); }
  CUmemAccessDesc ad[TPMAX];
  for (int r = 0; r < tp; r++) { ad[r].location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad[r].location.id = r; ad[r].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
  for (int r = 0; r < tp; r++) {
    CUmemAllocationProp ap = {}; ap.type = CU_MEM_ALLOCATION_TYPE_PINNED; ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ap.location.id = r;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t agran = 0; MKS_CU(cuMemGetAllocationGranularity(&agran, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t const asz = (mcSize + agran - 1) / agran * agran;
    CUmemGenericAllocationHandle ph; MKS_CU(cuMemCreate(&ph, asz, &ap, 0));
    MKS_CU(cuMulticastBindMem(mcH, 0, ph, 0, mcSize, 0));
    CUdeviceptr va; MKS_CU(cuMemAddressReserve(&va, asz, 0, 0, 0)); MKS_CU(cuMemMap(va, asz, 0, ph, 0)); MKS_CU(cuMemSetAccess(va, asz, ad, tp));
    recv[r] = reinterpret_cast<unsigned char *>(va);
    MKS_CK(cudaSetDevice(r)); MKS_CK(cudaMemset(recv[r], 0, mcSize));
  }
  CUdeviceptr va; MKS_CU(cuMemAddressReserve(&va, mcSize, 0, 0, 0)); MKS_CU(cuMemMap(va, mcSize, 0, mcH, 0)); MKS_CU(cuMemSetAccess(va, mcSize, ad, tp));
  mcVA = reinterpret_cast<unsigned char *>(va);
}

template <typename P>
static P *dmalloc(GpuState &s, size_t bytes) { void *p; MKS_CK(cudaMalloc(&p, bytes)); s.owned.push_back(p); return static_cast<P *>(p); }

static std::string arg(std::map<std::string, std::string> const &a, char const *k) {
  auto it = a.find(k); if (it == a.end()) { fprintf(stderr, "moe_host: missing argument %s\n", k); exit(1); } return it->second;
}
static void *tensor(StaticGpuView const &v, std::map<std::string, std::string> const &a, char const *role) {
  std::string const name = arg(a, role);
  auto it = v.tensors.find(name);
  if (it == v.tensors.end() || it->second == nullptr) { fprintf(stderr, "moe_host: GPU %d has no tensor %s (%s)\n", v.gpu, name.c_str(), role); exit(1); }
  return it->second;
}

}  // namespace moe_host

static void moe_host_init(StaticContext &ctx, std::map<std::string, std::string> const &args) {
  using namespace moe_host;
  int const tp = ctx.num_gpus;
  if (tp > TPMAX) { fprintf(stderr, "moe_host: %d GPUs, the bodies support at most %d\n", tp, TPMAX); exit(1); }
  if (ctx.num_lists != NSM) { fprintf(stderr, "moe_host: %d task lists per GPU, the bodies are built for %d SMs\n", ctx.num_lists, NSM); exit(1); }
  int const ks_r = std::stoi(arg(args, "router_ksplit")), ks_l = std::stoi(arg(args, "latent_ksplit"));
  MKS_CU(cuInit(0));
  // peer access between every pair of GPUs (the tail reads the other GPUs' [R|S] buffers)
  if (tp > 1)
    for (int r = 0; r < tp; r++) {
      MKS_CK(cudaSetDevice(r));
      for (int p = 0; p < tp; p++) if (p != r) {
        int can = 0; MKS_CK(cudaDeviceCanAccessPeer(&can, r, p)); if (!can) { fprintf(stderr, "moe_host: no peer access %d->%d\n", r, p); exit(1); }
        cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) { fprintf(stderr, "moe_host: peer enable %d->%d: %s\n", r, p, cudaGetErrorString(e)); exit(1); }
        cudaGetLastError();
      }
    }
  unsigned char *mcVA = nullptr, *recv[TPMAX] = {};
  if (tp > 1) mc_alloc(tp, RG_END, mcVA, recv);
  g_gpus.assign(tp, GpuState());
  for (int r = 0; r < tp; r++) {
    StaticGpuView const &v = ctx.gpus[r];
    GpuState &s = g_gpus[r];
    MKS_CK(cudaSetDevice(v.gpu));
    void *x = tensor(v, args, "x"), *w_router = tensor(v, args, "router_weight"), *w_down = tensor(v, args, "latent_down_weight");
    void *w_sgu = tensor(v, args, "shared_gate_up_weight"), *w_sd = tensor(v, args, "shared_down_weight"), *w_up = tensor(v, args, "latent_up_weight");
    void *bias = tensor(v, args, "score_correction_bias"), *gamma = tensor(v, args, "gamma"), *prefix = tensor(v, args, "prefix");
    void *w13 = tensor(v, args, "w13_blocks"), *w13sf = tensor(v, args, "w13_scales"), *w2 = tensor(v, args, "w2_blocks"), *w2sf = tensor(v, args, "w2_scales");
    s.nsplit_router = ks_r; s.nsplit_latent = ks_l;
    G &g = s.g; g = G{};
    g.rank = r; g.tp = tp; g.eps = 1e-5f;
    g.gamma = (__nv_bfloat16 const *)gamma; g.prefix = (__nv_bfloat16 const *)prefix; g.gate_bias = (float *)bias;
    g.w_up = (unsigned char const *)w_up;
    if (tp > 1) { g.mc = mcVA; g.rv = recv[r]; }
    else { g.rv = dmalloc<unsigned char>(s, RG_END); MKS_CK(cudaMemset(g.rv, 0, RG_END)); g.mc = nullptr; }
    g.rs_all[r] = dmalloc<unsigned char>(s, TPMAX * RS_RANK);   // every GPU's pointer is filled in below
    g.zq = g.rv + RG_ZQ; g.xsf = g.rv + RG_ZSF;
    MKS_CK(cudaMemset(g.rv + RG_HELLO, 0, TPMAX * 16));         // hello slots start at 0; g.gen counts from 1
    // the graph's output tensors (y, and this GPU's routed / shared sums, which the checks read)
    g.y = (__nv_bfloat16 *)tensor(v, args, "y"); g.Racc = (float *)tensor(v, args, "routed_sum"); g.Sout = (float *)tensor(v, args, "shared_down_sum");
    // the buffers between the tasks (also for the graph's intermediate tensors h_s, routing pairs, shared gate_up)
    g.hs = dmalloc<__nv_bfloat16>(s, T * SHR * 2);
    g.pairs64 = dmalloc<unsigned long long>(s, T * 16 * 8);
    g.spart = dmalloc<float>(s, (size_t)T * 2 * SHR * 4);
    g.lpart = dmalloc<float>(s, (size_t)ks_r * T * NE * 4);
    g.zpart = dmalloc<float>(s, (size_t)ks_l * T * LAT * 4);
    g.hq = dmalloc<uint8_t>(s, (size_t)NSLOT * T * IR);
    g.hsf = dmalloc<uint8_t>(s, (size_t)NSLOT * KT2 * SF_CHUNK);
    g.Rn = dmalloc<__nv_bfloat16>(s, T * LAT * 2);
    g.ss_part = dmalloc<float>(s, T * 4 * 4);
    g.opart = dmalloc<float>(s, (size_t)UP_KPARTS * T * (H / TPMAX) * 4);
    g.Ssum = dmalloc<__nv_bfloat16>(s, T * H * 2);
    g.cnt = dmalloc<uint32_t>(s, NCNT * 4);
    g.stamps = dmalloc<long long>(s, NSM * NSTAMP * 8); MKS_CK(cudaMemset(g.stamps, 0, NSM * NSTAMP * 8));
    g.start_barrier = dmalloc<long long>(s, 8); MKS_CK(cudaMemset(g.start_barrier, 0, 8));
    Maps &maps = s.maps;
    map_bf16(&maps.wg, w_router, NE, 128, H); map_bf16(&maps.wdown, w_down, LAT, 128, H); map_bf16(&maps.wsgu, w_sgu, 2 * SHR, 128, H);
    map_bf16(&maps.wsd, w_sd, H, 128, SHR); map_bf16(&maps.wup, w_up, H, 128, LAT);
    map_bf16(&maps.x, x, T, T, H); map_bf16(&maps.hs, g.hs, T, T, SHR);
    map_act8(&maps.zq, g.zq, LAT, T); map_act8_kt(&maps.zq2, g.zq, LAT, T, 2); map_sf(&maps.xsf28, g.xsf, KT_LAT, KT_LAT);
    map_wblk(&maps.w13, w13, (uint64_t)NE * MT13 * KT_LAT); map_sf(&maps.w13sf, w13sf, (uint64_t)NE * MT13 * KT_LAT);
    map_wblk(&maps.w13x2, w13, (uint64_t)NE * MT13 * KT_LAT, 2); map_sf(&maps.w13sfx2, w13sf, (uint64_t)NE * MT13 * KT_LAT, 2);
    map_wblk(&maps.w2, w2, (uint64_t)NE * OT2 * KT2); map_sf(&maps.w2sf, w2sf, (uint64_t)NE * OT2 * KT2);
    map_wblk(&maps.w2x2, w2, (uint64_t)NE * OT2 * KT2, 2); map_sf(&maps.w2sfx2, w2sf, (uint64_t)NE * OT2 * KT2, 2);
    map_act8_kt(&maps.hq3, g.hq, IR, (uint64_t)NSLOT * T, KT2); map_sf(&maps.hsf3, g.hsf, (uint64_t)NSLOT * KT2, KT2);
  }
  for (int r = 0; r < tp; r++) for (int p = 0; p < tp; p++) g_gpus[r].g.rs_all[p] = g_gpus[p].g.rs_all[p];   // every GPU knows every GPU's [R|S] buffer
}

// before each launch, on the GPU's stream: zero the counters and the red.add targets; 0xFF-fill every buffer whose reader polls
// for the data; then the launch number g.gen
static void moe_host_reset(StaticContext &ctx, int r, cudaStream_t st, unsigned long long launch_number) {
  using namespace moe_host;
  GpuState &s = g_gpus[r]; G &g = s.g;
  MKS_CK(cudaMemsetAsync(g.cnt, 0, NCNT * 4, st));
  MKS_CK(cudaMemsetAsync(g.Racc, 0, T * LAT * 4, st));                                        // red.add targets
  MKS_CK(cudaMemsetAsync(g.Sout, 0, T * H * 4, st));
  MKS_CK(cudaMemsetAsync(g.spart, 0, T * 2 * SHR * 4, st));
  MKS_CK(cudaMemsetAsync(g.lpart, 0xFF, (size_t)s.nsplit_router * T * NE * 4, st));           // polled by their readers
  MKS_CK(cudaMemsetAsync(g.zpart, 0xFF, (size_t)s.nsplit_latent * T * LAT * 4, st));
  MKS_CK(cudaMemsetAsync(g.ss_part, 0xFF, T * 4 * 4, st));
  MKS_CK(cudaMemsetAsync(g.Rn, 0xFF, T * LAT * 2, st));
  MKS_CK(cudaMemsetAsync(g.opart, 0xFF, (size_t)UP_KPARTS * T * (H / TPMAX) * 4, st));
  MKS_CK(cudaMemsetAsync(g.Ssum, 0xFF, T * H * 2, st));
  MKS_CK(cudaMemsetAsync(g.pairs64, 0xFF, T * 16 * 8, st));
  MKS_CK(cudaMemsetAsync(g.rv + RG_ZQ, 0xFF, RG_ZSF + KT_LAT * SF_CHUNK, st));
  MKS_CK(cudaMemsetAsync(g.rv + RG_O, 0xFF, TPMAX * O_RANK, st));
  MKS_CK(cudaMemsetAsync(g.rs_all[r], 0xFF, TPMAX * RS_RANK, st));
  // h_q and its scales: zero. warp 7 reloads an h_q segment after counter C_HQ says it is written, and checks that no scale byte is 0
  MKS_CK(cudaMemsetAsync(g.hsf, 0, (size_t)NSLOT * KT2 * SF_CHUNK, st));
  MKS_CK(cudaMemsetAsync(g.hq, 0, (size_t)NSLOT * T * IR, st));
  MKS_CK(cudaMemsetAsync(g.stamps, 0, NSM * NSTAMP * 8, st));
  g.gen = (unsigned)(launch_number + 1);
  (void)ctx;
}

// the span of the last launch on GPU r: bar = the last SM past the start barrier, end = the latest SM end; globaltimer ns
static void moe_host_timing(StaticContext &ctx, int r, long long &bar, long long &end) {
  using namespace moe_host;
  G const &g = g_gpus[r].g;
  MKS_CK(cudaSetDevice(ctx.gpus[r].gpu));
  std::vector<long long> st(NSM * NSTAMP);
  MKS_CK(cudaMemcpy(st.data(), g.stamps, NSM * NSTAMP * 8, cudaMemcpyDeviceToHost));
  MKS_CK(cudaMemcpy(&bar, g.start_barrier, 8, cudaMemcpyDeviceToHost));
  end = 0; for (int i = 0; i < NSM; i++) end = std::max(end, st[i * NSTAMP + STAMP_END]);
}

// per SM of GPU r, in the order of the family's phase_names: queue_start (z_q landed, before the phase switch), queue_end (the
// expert queue done, before the tail); out[sm * 2 + k]; globaltimer ns
static void moe_host_phase_stamps(StaticContext &ctx, int r, long long *out) {
  using namespace moe_host;
  G const &g = g_gpus[r].g;
  MKS_CK(cudaSetDevice(ctx.gpus[r].gpu));
  std::vector<long long> st(NSM * NSTAMP);
  MKS_CK(cudaMemcpy(st.data(), g.stamps, NSM * NSTAMP * 8, cudaMemcpyDeviceToHost));
  for (int i = 0; i < NSM; i++) { out[i * 2] = st[i * NSTAMP + STAMP_QUEUE_START]; out[i * 2 + 1] = st[i * NSTAMP + STAMP_QUEUE_END]; }
}

// per GPU: the counters and how many SMs passed each stamp (readable while a launch has not finished: a non-blocking stream)
static void moe_host_report(StaticContext &ctx) {
  using namespace moe_host;
  for (size_t r = 0; r < g_gpus.size(); r++) {
    G const &g = g_gpus[r].g;
    cudaSetDevice(ctx.gpus[r].gpu);
    cudaStream_t nb; if (cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking) != cudaSuccess) return;
    std::vector<uint32_t> cnt(NCNT); std::vector<long long> stm(NSM * NSTAMP);
    cudaMemcpyAsync(cnt.data(), g.cnt, NCNT * 4, cudaMemcpyDeviceToHost, nb);
    cudaMemcpyAsync(stm.data(), g.stamps, NSM * NSTAMP * 8, cudaMemcpyDeviceToHost, nb);
    if (cudaStreamSynchronize(nb) != cudaSuccess) { printf("moe GPU %zu: diagnostics not readable\n", r); cudaStreamDestroy(nb); continue; }
    int n_past[NSTAMP] = {0}, hq_done = 0, sgu_done = 0;
    for (int i = 0; i < NSM; i++) for (int k = 0; k < NSTAMP; k++) n_past[k] += stm[i * NSTAMP + k] != 0;
    for (int s_ = 0; s_ < NSLOT; s_++) hq_done += cnt[C_HQ + s_];
    for (int m = 0; m < N_SGU_TILES; m++) sgu_done += cnt[C_SGU + m];
    printf("moe GPU %zu: counters sact %u / %d, shared gate_up K parts %d, W13 items %d, queue entries taken %u, SMs done with the queue %u / %d"
           " | SMs past: start %d, queue start %d, queue end %d, end %d\n",
           r, cnt[C_HS], N_SACT, sgu_done, hq_done, cnt[C_W2NEXT], cnt[C_EXPDONE], NSM,
           n_past[STAMP_START], n_past[STAMP_QUEUE_START], n_past[STAMP_QUEUE_END], n_past[STAMP_END]);
    cudaStreamDestroy(nb);
  }
  fflush(stdout);
}

static void moe_host_finalize(StaticContext &ctx) {
  using namespace moe_host;
  for (size_t r = 0; r < g_gpus.size(); r++) {
    cudaSetDevice(ctx.gpus[r].gpu);
    for (void *p : g_gpus[r].owned) cudaFree(p);
  }
  g_gpus.clear();
}
