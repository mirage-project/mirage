// Microbenchmark: the latency of one cross-GPU signal on MI355X (the floor of an xGMI hop, which the layers' exchanges
// measure at ~4 us: push_latency.hip, hop_size.hip). Two GPUs, one thread each: GPU A stores the iteration number into a
// word of GPU B's memory, B polls its own word until it holds it, then stores it into A's word, A polls; per round trip
// two one-way signals. Also the same between two CUs of one GPU (two workgroups), for comparison.
// Variants: the store's cache bits (sys: sc0 sc1, nt: nt sc0 sc1), the poll's (sys: sc0 sc1, dev: sc1), a pause between
// polls (none / s_sleep 1), the pollers per side (1 thread, or 4 waves polling with staggered starts: the first to see it
// stores the answer).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 ping_pong.hip -o ping_pong
// Run:   ./ping_pong [iters=2000]       -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

template <int SK>
__device__ __forceinline__ void st_flag(unsigned *p, unsigned v) {
  if constexpr (SK == 0) {
    asm volatile("global_store_dword %0, %1, off sc0 sc1" ::"v"(p), "v"(v) : "memory");
  } else {
    asm volatile("global_store_dword %0, %1, off nt sc0 sc1" ::"v"(p), "v"(v) : "memory");
  }
}
template <int PK>
__device__ __forceinline__ unsigned ld_flag(unsigned const *p) {
  unsigned v;
  if constexpr (PK == 0) {
    asm volatile("global_load_dword %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  } else {
    asm volatile("global_load_dword %0, %1, off sc1\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  }
  return v;
}

// role 0 starts each round trip, role 1 answers. mine: the word this side polls; peer: the word it stores into.
// waves: 1 (thread 0 of wave 0 polls) or 4 (thread 0 of each wave polls, wave w starting w x ~quarter of a poll later;
// whichever sees the value first answers; the others see the round's flag in LDS)
template <int SK, int PK, int SLEEP, int WAVES>
__global__ void pp_kernel(unsigned *mine, unsigned *peer, int role, int iters, unsigned long long *t, unsigned *fail) {
  __shared__ unsigned seen;
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  if (lane != 0 || wave >= WAVES) return;
  if (wave == 0) seen = 0;
  __syncthreads();
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (role == 0 && wave == 0) st_flag<SK>(peer, (unsigned)it);
    if (WAVES > 1 && wave > 0) {
      for (int k = 0; k < 40 * wave; k++) asm volatile("s_nop 7");  // stagger
    }
    long long spins = 0;
    for (;;) {
      if (WAVES > 1 && __hip_atomic_load(&seen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) >= (unsigned)it) break;
      unsigned const v = ld_flag<PK>(mine);
      if (v >= (unsigned)it) {
        if (WAVES > 1) {
          unsigned const old = atomicMax(&seen, (unsigned)it);
          if (old < (unsigned)it && role == 1) st_flag<SK>(peer, (unsigned)it);
        } else if (role == 1) {
          st_flag<SK>(peer, (unsigned)it);
        }
        break;
      }
      if (++spins > (1ll << 22)) {
        atomicAdd(fail, 1u);
        it = iters + 1;
        break;
      }
      if (SLEEP) __builtin_amdgcn_s_sleep(1);
    }
  }
  if (wave == 0) t[0] = realtime() - t0;
}

// fan-out: GPU 0 (thread 0) stores the iteration into a word of each of the 7 other GPUs, each of them (thread 0) polls its
// word and stores the iteration into its own slot of GPU 0's 7 words, GPU 0 polls those 7 (loads in flight together);
// VEC > 0: every message is VEC 16-B vectors (lanes 0 .. VEC - 1 of wave 0 store / poll one each)
template <int VEC>
__global__ void fan_kernel(unsigned *const *bufs, int me, int ng, int iters, unsigned long long *t, unsigned *fail) {
  int const lane = threadIdx.x;
  if (lane >= (VEC > 0 ? VEC : 1)) return;
  unsigned *const mine = bufs[me];
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (me == 0) {
      for (int p = 1; p < ng; p++) {
        unsigned *const dst = bufs[p] + 4 * lane;
        if (VEC > 0) {
          v4u_t const w = {(unsigned)it, (unsigned)it, (unsigned)it, (unsigned)it};
          asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(dst), "v"(w) : "memory");
        } else {
          st_flag<0>(dst, (unsigned)it);
        }
      }
      unsigned done = 0, all = ((1u << ng) - 1) & ~1u;
      long long spins = 0;
      while (done != all) {
        unsigned v[8];
#pragma unroll
        for (int p = 1; p < 8; p++) {
          if (p < ng && !(done & (1u << p)))
            asm volatile("global_load_dword %0, %1, off sc0 sc1" : "=v"(v[p]) : "v"(mine + 256 * p + 4 * lane) : "memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
        for (int p = 1; p < 8; p++) {
          if (p < ng && !(done & (1u << p)) && v[p] >= (unsigned)it) done |= 1u << p;
        }
        if (++spins > (1ll << 22)) { atomicAdd(fail, 1u); it = iters + 1; break; }
      }
    } else {
      long long spins = 0;
      while (ld_flag<0>(mine + 4 * lane) < (unsigned)it) {
        if (++spins > (1ll << 22)) { atomicAdd(fail, 1u); it = iters + 1; break; }
      }
      unsigned *const dst = bufs[0] + 256 * me + 4 * lane;
      if (VEC > 0) {
        v4u_t const w = {(unsigned)it, (unsigned)it, (unsigned)it, (unsigned)it};
        asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(dst), "v"(w) : "memory");
      } else {
        st_flag<0>(dst, (unsigned)it);
      }
    }
  }
  if (lane == 0) t[0] = realtime() - t0;
}

// fan-out with the 7 stores and the 7 polls spread: MODE 1 = lane p - 1 of wave 0 stores to / polls GPU p (one store
// instruction for all 7), MODE 2 = wave p - 1 (of 7) stores to / polls GPU p
template <int MODE>
__global__ void fan2_kernel(unsigned *const *bufs, int me, int ng, int iters, unsigned long long *t, unsigned *fail) {
  int const lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  int const p = MODE == 1 ? lane + 1 : wave + 1; // this thread's peer (GPU 0's side)
  bool const act = MODE == 1 ? (wave == 0 && p < ng) : (lane == 0 && p < ng);
  unsigned *const mine = bufs[me];
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (me == 0) {
      if (act) {
        st_flag<0>(bufs[p], (unsigned)it);
        long long spins = 0;
        while (ld_flag<0>(mine + 256 * p) < (unsigned)it) {
          if (++spins > (1ll << 22)) { atomicAdd(fail, 1u); break; }
        }
      }
      __syncthreads();
    } else if (threadIdx.x == 0) {
      long long spins = 0;
      while (ld_flag<0>(mine) < (unsigned)it) {
        if (++spins > (1ll << 22)) { atomicAdd(fail, 1u); it = iters + 1; break; }
      }
      st_flag<0>(bufs[0] + 256 * me, (unsigned)it);
    }
  }
  if (threadIdx.x == 0) t[0] = realtime() - t0;
}

// fan-out from one wave, store variants: K 0 = 7 instructions (lane 0, one per peer) sc0 sc1; 1 = the same with sc1 only;
// 2 = sc0 only; 3 = 7 instructions, each with lanes 0 .. 6 storing to peers 1 .. 7 (49 stores, sc0 sc1; the peer reads
// the last); polls by lanes 0 .. 6 (one per peer) in one instruction
template <int K>
__global__ void fan3_kernel(unsigned *const *bufs, int me, int ng, int iters, unsigned long long *t, unsigned *fail) {
  int const lane = threadIdx.x & 63;
  unsigned *const mine = bufs[me];
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (me == 0) {
      if (K == 3) {
        if (lane < ng - 1) {
          for (int r = 0; r < 7; r++) {
            unsigned *const dst = bufs[lane + 1] + 4 * r;
            asm volatile("global_store_dword %0, %1, off sc0 sc1" ::"v"(dst), "v"((unsigned)it) : "memory");
          }
        }
      } else if (lane == 0) {
        for (int p = 1; p < ng; p++) {
          unsigned *const dst = bufs[p] + 24;
          if (K == 0) asm volatile("global_store_dword %0, %1, off sc0 sc1" ::"v"(dst), "v"((unsigned)it) : "memory");
          if (K == 1) asm volatile("global_store_dword %0, %1, off sc1" ::"v"(dst), "v"((unsigned)it) : "memory");
          if (K == 2) asm volatile("global_store_dword %0, %1, off sc0" ::"v"(dst), "v"((unsigned)it) : "memory");
        }
      }
      if (lane < ng - 1) {
        long long spins = 0;
        while (ld_flag<0>(mine + 256 * (lane + 1)) < (unsigned)it) {
          if (++spins > (1ll << 22)) { atomicAdd(fail, 1u); break; }
        }
      }
    } else if (lane == 0) {
      long long spins = 0;
      while (ld_flag<0>(mine + 24) < (unsigned)it) {
        if (++spins > (1ll << 20)) { atomicAdd(fail, 1u); it = iters + 1; break; }
      }
      st_flag<0>(bufs[0] + 256 * me, (unsigned)it);
    }
  }
  if (lane == 0) t[0] = realtime() - t0;
}

template <int SK, int PK, int SLEEP, int WAVES>
double run_case(unsigned *buf[2], int dev[2], int iters, unsigned long long *t[2], unsigned *fail[2], hipStream_t st[2],
                bool same_gpu, unsigned *f_out) {
  for (int s = 0; s < 2; s++) {
    CHECK(hipSetDevice(dev[s]));
    CHECK(hipMemset(buf[s], 0, 256));
    CHECK(hipMemset(t[s], 0, 16));
    CHECK(hipMemset(fail[s], 0, 4));
    CHECK(hipDeviceSynchronize());
  }
  for (int s = 0; s < 2; s++) {
    CHECK(hipSetDevice(dev[s]));
    hipLaunchKernelGGL((pp_kernel<SK, PK, SLEEP, WAVES>), dim3(1), dim3(256), 0, st[s], buf[s], buf[1 - s], s, iters, t[s],
                       fail[s]);
    CHECK(hipGetLastError());
  }
  unsigned long long ticks = 0;
  unsigned f = 0;
  for (int s = 0; s < 2; s++) {
    CHECK(hipSetDevice(dev[s]));
    CHECK(hipStreamSynchronize(st[s]));
    unsigned long long tk;
    unsigned fs;
    CHECK(hipMemcpy(&tk, t[s], 8, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&fs, fail[s], 4, hipMemcpyDeviceToHost));
    if (s == 0) ticks = tk;
    f += fs;
  }
  *f_out = f;
  (void)same_gpu;
  return (double)ticks / wall_ticks_per_ns(dev[0]) / iters / 2;  // one-way ns
}

int main(int argc, char **argv) {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  int const iters = argc > 1 ? atoi(argv[1]) : 2000;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  std::string json;
  // pairs: GPU 0 with GPU 1, 4, 7; then GPU 0 with itself (two workgroups, two streams)
  for (int other : {1, 4, 7, 0}) {
    if (other >= ndev) continue;
    int dev[2] = {0, other};
    unsigned *buf[2];
    unsigned long long *t[2];
    unsigned *fail[2];
    hipStream_t st[2];
    for (int s = 0; s < 2; s++) {
      CHECK(hipSetDevice(dev[s]));
      if (dev[0] != dev[1]) {
        int can;
        CHECK(hipDeviceCanAccessPeer(&can, dev[s], dev[1 - s]));
        if (!can) {
          printf("GPU %d cannot access GPU %d\n", dev[s], dev[1 - s]);
          return 0;
        }
        hipError_t e = hipDeviceEnablePeerAccess(dev[1 - s], 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CHECK(e);
        (void)hipGetLastError();
      }
      CHECK(hipMalloc(&buf[s], 256));
      CHECK(hipMalloc(&t[s], 16));
      CHECK(hipMalloc(&fail[s], 4));
      CHECK(hipStreamCreateWithFlags(&st[s], hipStreamNonBlocking));
    }
    struct Row { const char *name; double ns; unsigned f; };
    std::vector<Row> rows;
    unsigned f;
    double v;
    v = run_case<0, 0, 0, 1>(buf, dev, iters, t, fail, st, other == 0, &f); rows.push_back({"store sys, poll sys, no pause, 1 poller", v, f});
    v = run_case<0, 0, 1, 1>(buf, dev, iters, t, fail, st, other == 0, &f); rows.push_back({"store sys, poll sys, s_sleep 1, 1 poller", v, f});
    v = run_case<1, 0, 0, 1>(buf, dev, iters, t, fail, st, other == 0, &f); rows.push_back({"store nt sys, poll sys, no pause, 1 poller", v, f});
    v = run_case<0, 1, 0, 1>(buf, dev, iters, t, fail, st, other == 0, &f); rows.push_back({"store sys, poll dev, no pause, 1 poller", v, f});
    v = run_case<0, 0, 0, 4>(buf, dev, iters, t, fail, st, other == 0, &f); rows.push_back({"store sys, poll sys, no pause, 4 staggered pollers", v, f});
    for (auto &r : rows) {
      printf("GPU 0 <-> GPU %d%s: %-52s one way %.2f us%s\n", other, other == 0 ? " (two CUs)" : "", r.name, r.ns / 1e3,
             r.f ? " POLLS GAVE UP" : "");
      char row[256];
      snprintf(row, sizeof(row), "%s{\"peer\": %d, \"case\": \"%s\", \"one_way_us\": %.3f, \"fails\": %u}", json.empty() ? "" : ", ",
               other, r.name, r.ns / 1e3, r.f);
      json += row;
    }
    for (int s = 0; s < 2; s++) {
      CHECK(hipSetDevice(dev[s]));
      CHECK(hipFree(buf[s]));
      CHECK(hipFree(t[s]));
      CHECK(hipFree(fail[s]));
      CHECK(hipStreamDestroy(st[s]));
    }
  }
  // fan-out from GPU 0 to the other 7 and back
  if (ndev >= 8) {
    int const ng = 8;
    std::vector<unsigned *> b(ng);
    std::vector<unsigned long long *> tt(ng);
    std::vector<unsigned *> ff(ng);
    std::vector<hipStream_t> ss(ng);
    unsigned **dbufs[8];
    for (int g = 0; g < ng; g++) {
      CHECK(hipSetDevice(g));
      for (int p = 0; p < ng; p++) {
        if (p == g) continue;
        hipError_t e = hipDeviceEnablePeerAccess(p, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CHECK(e);
        (void)hipGetLastError();
      }
      CHECK(hipMalloc(&b[g], 8 * 256 * 4));
      CHECK(hipMalloc(&tt[g], 16));
      CHECK(hipMalloc(&ff[g], 4));
      CHECK(hipStreamCreateWithFlags(&ss[g], hipStreamNonBlocking));
    }
    for (int g = 0; g < ng; g++) {
      CHECK(hipSetDevice(g));
      CHECK(hipMalloc(&dbufs[g], ng * sizeof(unsigned *)));
      CHECK(hipMemcpy(dbufs[g], b.data(), ng * sizeof(unsigned *), hipMemcpyHostToDevice));
    }
    for (int vec : {0, 1001, 1002, 2000, 2001, 2002, 2003}) {
      for (int g = 0; g < ng; g++) {
        CHECK(hipSetDevice(g));
        CHECK(hipMemset(b[g], 0, 8 * 256 * 4));
        CHECK(hipMemset(ff[g], 0, 4));
        CHECK(hipDeviceSynchronize());
      }
      for (int g = 0; g < ng; g++) {
        CHECK(hipSetDevice(g));
        if (vec == 0) hipLaunchKernelGGL(fan_kernel<0>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 1) hipLaunchKernelGGL(fan_kernel<1>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 16) hipLaunchKernelGGL(fan_kernel<16>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 64) hipLaunchKernelGGL(fan_kernel<64>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 1001) hipLaunchKernelGGL(fan2_kernel<1>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 1002) hipLaunchKernelGGL(fan2_kernel<2>, dim3(1), dim3(448), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 2000) hipLaunchKernelGGL(fan3_kernel<0>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 2001) hipLaunchKernelGGL(fan3_kernel<1>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 2002) hipLaunchKernelGGL(fan3_kernel<2>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        if (vec == 2003) hipLaunchKernelGGL(fan3_kernel<3>, dim3(1), dim3(64), 0, ss[g], dbufs[g], g, ng, iters, tt[g], ff[g]);
        CHECK(hipGetLastError());
      }
      unsigned long long tk = 0;
      unsigned fs = 0;
      for (int g = 0; g < ng; g++) {
        CHECK(hipSetDevice(g));
        CHECK(hipStreamSynchronize(ss[g]));
        unsigned f;
        if (g == 0) CHECK(hipMemcpy(&tk, tt[g], 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&f, ff[g], 4, hipMemcpyDeviceToHost));
        fs += f;
      }
      double const rt = (double)tk / wall_ticks_per_ns(0) / iters;
      printf("fan-out GPU 0 -> 7 GPUs -> GPU 0, %s: round trip %.2f us (half %.2f)%s\n",
             vec == 0 ? "4-B flags" : vec == 1 ? "one 16-B vector each" : vec == 16 ? "16 x 16-B vectors each" :
             vec == 64 ? "64 x 16-B vectors each" : vec == 1001 ? "4-B flags, lane per peer" : vec == 1002 ? "4-B flags, wave per peer" :
             vec == 2000 ? "7 store instr sc0 sc1, lane polls" : vec == 2001 ? "7 store instr sc1 only, lane polls" :
             vec == 2002 ? "7 store instr sc0 only, lane polls" : "7 store instr x 7 lanes (peer per lane), lane polls",
             rt / 1e3, rt / 2e3, fs ? " POLLS GAVE UP" : "");
    }
  }
  printf("{\"bench\": \"ping_pong\", \"iters\": %d, \"rows\": [%s]}\n", iters, json.c_str());
  return 0;
}
