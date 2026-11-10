// hop_big.hip: hop_size.hip with T rows and NMAX columns set at build time (-DT_ROWS=64 -DNMAX_COLS=4096: the GLM MoE sums at
// T 64, 512 KB per peer from 256 senders). MI355X 11-07 glmmoe69: in the kernel, the other GPUs' slots landed ~24 us after
// their pushes (this GPU's own +4 us) at T 64, ~4 us at T 8.
// Microbenchmark: one xGMI all-to-all hop on MI355X by payload size (push_pattern.hip's tile16 pattern, payload n columns).
// Every GPU sends a [T][n] bf16 block into its slot of every GPU's receive buffer (T n 2 B per peer); S sender workgroups (one
// per CU), workgroup w owning columns [R w, R w + R) (R = n / S), its T x R block contiguous, 16-B system-scope stores from
// one wave; then each workgroup polls its share of the slots of its own buffer until every word holds the iteration (the data
// is its own flag, system-scope loads), then the next iteration. An iteration = one hop of the slowest sender. n = 7168: the
// o_proj all-to-all now (115 KB per peer); n = 896: a reduce-scatter's share (one eighth of the row, 14 KB per peer).
// Polls (PAR): 0 = each of the 8 slots of a vector polled one after the other (8 round trips in a row: the first version,
// whose ~4 us per hop measured that, not xGMI: amd/bench/ping_pong.hip, a one-way signal between two GPUs is ~0.6 us);
// 1 = the 8 slots' loads in flight together, re-issued only for the slots not landed (as the kernels' landing loops).
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 hop_size.hip -o hop_size
// Run:   ./hop_size [gpus=8] [iters=200]     -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

#ifndef T_ROWS
#define T_ROWS 8
#endif
#ifndef NMAX_COLS
#define NMAX_COLS 7168
#endif
constexpr int MAX_GPUS = 8, T = T_ROWS, NMAX = NMAX_COLS;

struct Peers {
  unsigned char *recv[MAX_GPUS];  // recv[p] = GPU p's receive buffer: [G][T NMAX] bf16 (slot g = from GPU g)
};

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void st16_sys(void *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ v4u_t load16_sys(void const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return w;
}

// POLL_SC (11-09): the poll loads' cache bits - 3 sc0 sc1 (system scope, the kernels' landing loads), 2 sc1 (device scope),
// 1 sc0, 0 none. MI355X 11-09 glmprof23: in the kernel, 128 CUs x 256 threads x 4 system-scope 16-B loads of landed data took ~14 us
#ifndef POLL_SC
#define POLL_SC 3
#endif
__device__ __forceinline__ v4u_t load16_sys_nw(void const *p) { // (no wait: the caller waits)
  v4u_t w;
  if constexpr (POLL_SC == 3) {
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1" : "=v"(w) : "v"(p) : "memory");
  } else if constexpr (POLL_SC == 2) {
    asm volatile("global_load_dwordx4 %0, %1, off sc1" : "=v"(w) : "v"(p) : "memory");
  } else if constexpr (POLL_SC == 1) {
    asm volatile("global_load_dwordx4 %0, %1, off sc0" : "=v"(w) : "v"(p) : "memory");
  } else {
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(w) : "v"(p) : "memory");
  }
  return w;
}

// PRE_KB > 0: before each hop every workgroup reads PRE_KB KB of a local buffer (16-B loads, summed into a dummy store): the
// memory system's state after a stream, as the kernel's W2 phase leaves it before its all-reduce
#ifndef PRE_KB
#define PRE_KB 0
#endif
#ifndef NO_HOP
#define NO_HOP 0
#endif
// RESET_FF 1 (11-09): as the kernel's exchange buffers: two sets, 0xFF-prefilled; iteration it pushes into set it & 1 and
// polls for no word 0xFFFFFFFF; first every workgroup resets its share of the other set (the one the previous iteration
// landed) to 0xFF with device-scope stores (st16_dev: sc1) and waits for them. Workgroup w's poll share = what the other
// GPUs' workgroup w push, so no barrier is needed: a GPU pushes w's share of set s only after it landed w's share of the
// previous iteration, which this GPU's workgroup w pushed after its reset of that set.
// RESET_FF 2: as 1, and after landing every workgroup also reads its share of the set with plain loads (lines allocated in this
// XCD's L2, as a consumer with plain loads would); RESET_FF 3: as 1 with the reset by plain stores (dirty lines in L2)
#ifndef RESET_FF
#define RESET_FF 0
#endif
// PUSH_ILV 1 (11-09): the kernel's finish_all push pattern - all 256 threads push, each 16-B vector to every GPU in turn (r inner),
// instead of one wave pushing all of one GPU's block, then the next GPU's
#ifndef PUSH_ILV
#define PUSH_ILV 0
#endif
// SENDER_XCD 1 (11-09): every sender on one XCD (blocks 8 w; workgroups go to the XCDs round-robin) - MI355X glmprof20: the attention
// o_proj finish_all's latest tasks (the finishers that push) all on CUs of one residue mod 8
#ifndef SENDER_XCD
#define SENDER_XCD 0
#endif
__device__ __forceinline__ void st16_dev(void *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
}
template <int PAR>
__global__ void hop_kernel(Peers peers, int me, int gpus, int n, int senders, int iters, unsigned long long *t, unsigned *fail,
                           v4u_t const *pre, unsigned *sink) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  if (SENDER_XCD && blockIdx.x % 8 != 0) return; // (SENDER_XCD: only blocks 8 w - one XCD - send and poll)
  int const wg = SENDER_XCD ? blockIdx.x / 8 : blockIdx.x, lane = threadIdx.x & 63, R = n / senders;
  size_t const SLOT = (size_t)T * NMAX * 2, n16 = (size_t)T * n * 2 / 16, lo = n16 * wg / senders, hi = n16 * (wg + 1) / senders;
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (__hip_atomic_load(fail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) break;
    if constexpr (PRE_KB > 0) {
      constexpr size_t PV = (size_t)PRE_KB * 1024 / 16; // 16-B vectors per workgroup
      v4u_t const *pw = pre + ((size_t)wg * PV + (size_t)(it & 1) * gridDim.x * PV); // (two halves: no L2 reuse between hops)
      unsigned acc = 0;
      for (size_t v = threadIdx.x; v < PV; v += blockDim.x) {
        v4u_t const q = __builtin_nontemporal_load(pw + v);
        acc ^= q.x ^ q.w;
      }
      if (acc == 0x12345678u) atomicAdd(sink, 1u);
      __syncthreads();
    }
    if constexpr (NO_HOP) { // (the read alone, to subtract)
      continue;
    }
    size_t const SET = RESET_FF ? (size_t)(it & 1) * gpus * SLOT : 0;
    if constexpr (RESET_FF) {
      size_t const other = (size_t)((it + 1) & 1) * gpus * SLOT;
      for (int r = 0; r < gpus; r++)
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
          if (RESET_FF == 3) {
            asm volatile("global_store_dwordx4 %0, %1, off" ::"v"(peers.recv[me] + other + (size_t)r * SLOT + i * 16),
                         "v"(v4u_t{~0u, ~0u, ~0u, ~0u}) : "memory");
          } else {
            st16_dev(peers.recv[me] + other + (size_t)r * SLOT + i * 16, 0xFFFFFFFFu);
          }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
    }
    if constexpr (PUSH_ILV) { // (as gemm_finish_all's push16: every thread, each vector to every GPU in turn)
      for (int v = threadIdx.x; v < T * R / 8; v += blockDim.x)
        for (int r = 0; r < gpus; r++)
          st16_sys(peers.recv[r] + SET + (size_t)me * SLOT + (size_t)wg * T * R * 2 + (size_t)v * 16, (unsigned)it);
    } else if (threadIdx.x < 64) {
      for (int r = 0; r < gpus; r++) {
        unsigned char *const base = peers.recv[r] + SET + (size_t)me * SLOT;
        for (int v = lane; v < T * R / 8; v += 64) st16_sys(base + (size_t)wg * T * R * 2 + (size_t)v * 16, (unsigned)it);
      }
    }
    if constexpr (PAR == 1) {
      for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        unsigned done = 0, all = (1u << gpus) - 1;
        long long spins = 0;
        while (done != all) {
          v4u_t w[MAX_GPUS];
#pragma unroll
          for (int r = 0; r < MAX_GPUS; r++) {
            if (r < gpus && !(done & (1u << r))) w[r] = load16_sys_nw(peers.recv[me] + SET + (size_t)r * SLOT + i * 16);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
          for (int r = 0; r < MAX_GPUS; r++) {
            if (r < gpus && !(done & (1u << r)) &&
                (RESET_FF ? (w[r].x != ~0u && w[r].y != ~0u && w[r].z != ~0u && w[r].w != ~0u)
                          : (w[r].x >= (unsigned)it && w[r].y >= (unsigned)it && w[r].z >= (unsigned)it && w[r].w >= (unsigned)it)))
              done |= 1u << r;
          }
          if (done == all) break;
          if (++spins > (1ll << 16)) {
            atomicAdd(fail, 1u);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
      if constexpr (RESET_FF == 2) {
        unsigned acc = 0;
        for (int r = 0; r < gpus; r++)
          for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            v4u_t q; // (plain: no sc bits; a C++ volatile load would get them)
            asm volatile("global_load_dwordx4 %0, %1, off\n s_waitcnt vmcnt(0)" : "=v"(q) : "v"(peers.recv[me] + SET + (size_t)r * SLOT + i * 16) : "memory");
            acc ^= q.x ^ q.w;
          }
        if (acc == 0x12345678u) atomicAdd(sink, 1u);
      }
    } else
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      for (int r = 0; r < gpus; r++) {
        void const *p = peers.recv[me] + (size_t)r * SLOT + i * 16;
        long long spins = 0;
        for (;;) {
          v4u_t const w = load16_sys(p);
          if (w.x >= (unsigned)it && w.y >= (unsigned)it && w.z >= (unsigned)it && w.w >= (unsigned)it) break;
          if (++spins > (1ll << 16)) {
            atomicAdd(fail, 1u);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax(&t[0], realtime() - t0);
}

int main(int argc, char **argv) {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  int const gpus = argc > 1 ? std::min(atoi(argv[1]), ndev) : std::min(ndev, MAX_GPUS);
  int const iters = argc > 2 ? atoi(argv[2]) : 200;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  size_t const slot = (size_t)T * NMAX * 2;
  Peers peers;
  std::vector<unsigned long long *> t(gpus);
  std::vector<unsigned *> fail(gpus), sink(gpus, nullptr);
  std::vector<v4u_t *> pre(gpus, nullptr);
  std::vector<hipStream_t> streams(gpus);
  for (int g = 0; g < gpus; ++g) {
    CHECK(hipSetDevice(g));
    for (int p = 0; p < gpus; ++p) {
      if (p == g) continue;
      int can;
      CHECK(hipDeviceCanAccessPeer(&can, g, p));
      if (!can) {
        printf("{\"bench\": \"hop_big\", \"error\": \"GPU %d cannot access GPU %d\"}\n", g, p);
        return 0;
      }
      CHECK(hipDeviceEnablePeerAccess(p, 0));
    }
    CHECK(hipMalloc(&peers.recv[g], (RESET_FF ? 2 : 1) * gpus * slot));
    CHECK(hipMalloc(&t[g], 16));
    CHECK(hipMalloc(&fail[g], 4));
    if (PRE_KB > 0 || RESET_FF == 2) {
      CHECK(hipMalloc(&pre[g], (size_t)2 * 256 * PRE_KB * 1024));
      CHECK(hipMemset(pre[g], 1, (size_t)2 * 256 * PRE_KB * 1024));
      CHECK(hipMalloc(&sink[g], 4));
    }
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    set_max_lds((const void *)hop_kernel<0>, ONE_PER_CU_LDS);
    set_max_lds((const void *)hop_kernel<1>, ONE_PER_CU_LDS);
  }
  double const tpn = wall_ticks_per_ns(0);
  std::string json;
  // (11-09: also 128 / 64 senders - the MoE send's 128 slices, the attention o_proj finish_all's 64 row blocks at T 32 / 64)
  int const cases[][2] = {{4096, 32}, {4096, 16}, {2048, 32}, {1024, 32}};
  for (int par : {1})
  for (auto const &c : cases) {
    int const n = c[0], senders = c[1];
    for (int g = 0; g < gpus; ++g) {
      CHECK(hipSetDevice(g));
      CHECK(hipMemset(peers.recv[g], RESET_FF ? 0xFF : 0, (RESET_FF ? 2 : 1) * gpus * slot));
      CHECK(hipMemset(t[g], 0, 16));
      CHECK(hipMemset(fail[g], 0, 4));
      CHECK(hipDeviceSynchronize());
    }
    for (int g = 0; g < gpus; ++g) {
      CHECK(hipSetDevice(g));
      if (par)
        hipLaunchKernelGGL(hop_kernel<1>, dim3(SENDER_XCD ? 8 * senders : senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n, senders, iters,
                           t[g], fail[g], pre[g], sink[g]);
      else
        hipLaunchKernelGGL(hop_kernel<0>, dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n, senders, iters,
                           t[g], fail[g], pre[g], sink[g]);
      CHECK(hipGetLastError());
    }
    std::vector<double> per;
    unsigned fails = 0;
    for (int g = 0; g < gpus; ++g) {
      CHECK(hipSetDevice(g));
      CHECK(hipStreamSynchronize(streams[g]));
      unsigned long long ticks;
      unsigned f;
      CHECK(hipMemcpy(&ticks, t[g], 8, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(&f, fail[g], 4, hipMemcpyDeviceToHost));
      per.push_back(ticks / tpn / iters);
      fails += f;
    }
    double const worst = *std::max_element(per.begin(), per.end());
    printf("%d GPUs, %6d B per peer (n %4d), %3d senders (%3d columns each), polls %s: one hop %.2f us (slowest GPU; median "
           "%.2f)%s\n", gpus, T * n * 2, n, senders, n / senders, par ? "parallel" : "serial  ", worst / 1e3, median(per) / 1e3,
           fails ? " POLLS GAVE UP" : "");
    char row[200];
    snprintf(row, sizeof(row), "%s{\"n\": %d, \"senders\": %d, \"par\": %d, \"slowest_us\": %.3f, \"median_us\": %.3f, \"fails\": %u}",
             json.empty() ? "" : ", ", n, senders, par, worst / 1e3, median(per) / 1e3, fails);
    json += row;
  }
  printf("{\"bench\": \"hop_big\", \"gpus\": %d, \"iters\": %d, \"rows\": [%s]}\n", gpus, iters, json.c_str());
  return 0;
}
