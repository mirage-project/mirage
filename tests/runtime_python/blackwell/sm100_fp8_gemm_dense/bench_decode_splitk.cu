// =============================================================================
// SplitK handoff bench B: fp8_gemm_dense DECODE SPLITK at DSv3 O_proj shape
// =============================================================================
//
// PURPOSE
//   Standalone microbenchmark for the broken SplitK kernel
//     include/mirage/persistent_kernel/tasks/blackwell/
//       fp8_gemm_dense_decode_splitk_sm100.cuh
//   To validate any rewrite of that kernel against the baseline produced by
//   bench_decode_no_splitk.cu (companion file).
//
// CURRENT STATUS (2026-05-16)
//   Kernel CRASHES with `cudaErrorLaunchFailure` ("unspecified launch
//   failure") in EVERY tested configuration:
//     SK=4 + num_workers=80   → crash (multi-tile-iter mbarrier stale state)
//     SK=4 + num_workers=128  → crash
//     SK=2 + num_workers=128  → crash (total=112 ≤ 128, single tile/CTA,
//                                       so the multi-tile-iter bug should
//                                       NOT trigger — but kernel still
//                                       crashes)
//     SK=1 + num_workers=128  → DEADLOCK (1h+ spin)
//     SK=4 + num_workers=224 (= total)  → crash (perfect 1-tile-per-CTA,
//                                                still crashes)
//   So the bug is NOT just multi-tile-iter — the kernel body has a
//   fundamental issue even in the 1-tile-per-CTA configuration. **Standalone
//   bench reproduces the crash without any megakernel scheduling involved.**
//
//   Bench output today:
//     ABORT: warmup CUDA error 719 (unspecified launch failure)
//
//   Once you rewrite the kernel, this bench should report timings in the
//   same format as bench_decode_no_splitk.cu and meet the target below.
//
// TARGET PERFORMANCE (relative to bench_decode_no_splitk.cu)
//   - Baseline (no-SplitK): steady-state p50 ≈ 58 μs per-CTA per-iter
//   - Roofline (HBM-bound): 14.7 μs
//   - SplitK=4 target: steady-state p50 ≤ 30 μs (≥ 2× speedup over
//     no-SplitK; brings us within ~2× of roofline)
//   - Acceptance: A/B p50 ratio ≥ 1.7×, where A = no-SplitK p50,
//     B = SplitK p50. Below 1.5× means SplitK isn't worth shipping.
//
// HOW TO BUILD + RUN
//   cd /home/muhengl/mirage
//   /usr/local/cuda/bin/nvcc -O3 -gencode=arch=compute_100a,code=sm_100a \
//       -I include \
//       -I include/mirage/persistent_kernel \
//       --expt-relaxed-constexpr -std=c++20 \
//       tests/runtime_python/blackwell/sm100_fp8_gemm_dense/bench_decode_splitk.cu
//       \ -o /tmp/bench_decode_splitk -lcuda
//   CUDA_VISIBLE_DEVICES=1 /tmp/bench_decode_splitk 0
//
// WHAT THE KERNEL OWNER NEEDS TO DO
//   1. Fix or rewrite:
//        include/mirage/persistent_kernel/tasks/blackwell/
//          fp8_gemm_dense_decode_splitk_sm100.cuh
//      so this bench runs to completion at SK ∈ {2, 4, 8} and produces
//      timing output matching the format of bench_decode_no_splitk.cu.
//   2. Verify A/B ratio ≥ 1.7× (this bench p50 ≤ 30 μs, when baseline is
//      ~58 μs). If your kernel is even faster than that, great.
//   3. End-to-end integration is gated by `MPK_DSV3_DECODE_OPROJ_SPLITK=1`
//      env var in `python/mirage/mpk/models/deepseek_v3/builder.py:661`.
//      Once your kernel passes the bench at the target speedup, we can
//      flip that env-ON by default in the builder and verify the e2e win
//      via demo/deepseek_v3/demo.py (TP=4 EP=2 19-layer trace).
//   4. The TASK_FP8_GEMM_DENSE_DECODE_SPLITK_SM100 task type wiring (task
//      tuple, runtime metadata, codegen) in
//        src/kernel/task_register.cc:6044
//        (register_fp8_gemm_dense_decode_splitk_sm100_task)
//        src/kernel/runtime.cc:355         (kv_idx / request_id metadata)
//        src/kernel/graph.cc:871           (task tuple registration)
//        include/mirage/persistent_kernel/tma.cuh:1607, :2270  (TMA
//        descriptors)
//      is already in place — just the kernel body needs to be correct.
//   5. The Python wrapper that prepends `tensor_init_layer` to pre-zero the
//      output (required by SplitK's red.add accumulation) is in
//        python/mirage/mpk/persistent_kernel.py:2978
//          fp8_gemm_dense_decode_splitk_layer
//
// BUG NOTES (from debugging history — may help isolate root cause)
//   - The original SplitK kernel uses `red.relaxed.gpu.global.add.noftz.bf16x2`
//     for inter-CTA accumulation. We bisected: replacing it with a plain
//     `st.relaxed.cta.global.b32` (giving wrong results but eliminating
//     the atomic) STILL crashes. So the atomic instruction is not the bug.
//   - Per-warp roles match the working mediumm/smallm kernels (warp 0 TMA,
//     warp 1 MMA, warp 2 tcgen05.alloc, warps 4-7 epilogue). mbarrier init
//     counts are identical (bf/be = count 1, btf/bte = count 1/128).
//   - Smem layout is the same (input + weight pipelined NS stages,
//     barrier + tcgen05 alloc state). Smem size is identical to mediumm.
//   - With num_workers=224 (≥ total), per-CTA loop runs exactly 1 tile
//     iter then `break`. No multi-tile-iter state carry-over possible.
//     Still crashes — so the bug is in the single-tile path.
//   - The kernel WORKS as a trampoline: if `task_impl_splitk_tpl` body is
//     replaced with `kernel::fp8_gemm_dense_common::task_impl_tpl<BN,NS,NE>(
//     args...)` directly, it runs (no crash, no SplitK speedup obviously).
//     So the wiring/dispatch is correct — only the SplitK body is broken.
//
// CROSS-REF
//   - Working baseline: bench_decode_no_splitk.cu
//   - Builder env gate: python/mirage/mpk/models/deepseek_v3/builder.py
//     :661 (`MPK_DSV3_DECODE_OPROJ_SPLITK`)
//   - Shape rationale: see header of bench_decode_no_splitk.cu
//
// =============================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <vector>

#include "tasks/blackwell/fp8_gemm_dense_decode_splitk_sm100.cuh"

using bf16 = __nv_bfloat16;

constexpr int M = 128, N = 7168, K = 16384;
constexpr int BN = 128, NS = 3, NE = 2, SPLIT_K = 4;
constexpr int N_ITER_MAX = 256;

__device__ uint32_t per_iter_ns[256 * N_ITER_MAX];

__device__ __forceinline__ uint32_t get_globaltimer() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ret));
  return ret;
}

__global__
    __launch_bounds__(256,
                      1) void persistent_splitk_bench(CUtensorMap const *ta_ptr,
                                                      CUtensorMap const *tb_ptr,
                                                      float const *sa,
                                                      float const *sb,
                                                      bf16 *C,
                                                      int n_iter) {
  for (int it = 0; it < n_iter; it++) {
    __syncthreads();
    uint32_t t0 = 0;
    if (threadIdx.x == 0) {
      t0 = get_globaltimer();
    }
    kernel::fp8_gemm_dense_decode_splitk::
        fp8_gemm_dense_decode_splitk_sm100_task_impl<BN, NS, NE, SPLIT_K>(
            ta_ptr,
            tb_ptr,
            sa,
            sb,
            C,
            M,
            N,
            K,
            /*worker_idx=*/blockIdx.x,
            /*num_workers=*/gridDim.x);
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t t1 = get_globaltimer();
      per_iter_ns[blockIdx.x * N_ITER_MAX + it] = t1 - t0;
    }
  }
}

void make_tma_desc(CUtensorMap &desc, void *base, uint64_t outer, uint64_t k) {
  uint64_t gd[2] = {(uint64_t)k, (uint64_t)outer};
  uint64_t gs[1] = {(uint64_t)k};
  uint32_t bd[2] = {128, 128};
  uint32_t es[2] = {1, 1};
  CUresult err = cuTensorMapEncodeTiled(&desc,
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
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (err != CUDA_SUCCESS) {
    char const *errstr;
    cuGetErrorString(err, &errstr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errstr);
    exit(1);
  }
}

void run_bench(int num_workers,
               int n_iter,
               CUtensorMap *d_ta,
               CUtensorMap *d_tb,
               float *d_sa,
               float *d_sb,
               bf16 *d_c,
               size_t smem) {
  // Pre-zero C for SplitK red.add accumulation.
  cudaMemset(d_c, 0, M * N * sizeof(bf16));

  for (int i = 0; i < 3; i++) {
    persistent_splitk_bench<<<num_workers, 256, smem>>>(
        d_ta, d_tb, d_sa, d_sb, d_c, n_iter);
    cudaMemset(d_c, 0, M * N * sizeof(bf16));
  }
  cudaDeviceSynchronize();
  auto rc = cudaGetLastError();
  if (rc != cudaSuccess) {
    printf("  ABORT (num_workers=%d): warmup CUDA error %d (%s)\n",
           num_workers,
           rc,
           cudaGetErrorString(rc));
    return;
  }

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  persistent_splitk_bench<<<num_workers, 256, smem>>>(
      d_ta, d_tb, d_sa, d_sb, d_c, n_iter);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  rc = cudaGetLastError();
  if (rc != cudaSuccess) {
    printf("  ABORT (num_workers=%d): timed-run CUDA error %d (%s)\n",
           num_workers,
           rc,
           cudaGetErrorString(rc));
    return;
  }

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, s, e);

  std::vector<uint32_t> h_iter(256 * N_ITER_MAX);
  cudaMemcpyFromSymbol(
      h_iter.data(), per_iter_ns, sizeof(uint32_t) * 256 * N_ITER_MAX);

  std::vector<uint32_t> max_per_iter(n_iter, 0);
  for (int it = 0; it < n_iter; it++) {
    for (int wi = 0; wi < num_workers; wi++) {
      uint32_t v = h_iter[wi * N_ITER_MAX + it];
      if (v > max_per_iter[it]) {
        max_per_iter[it] = v;
      }
    }
  }

  std::vector<uint32_t> steady;
  steady.reserve((size_t)num_workers * (n_iter - 1));
  for (int wi = 0; wi < num_workers; wi++) {
    for (int it = 1; it < n_iter; it++) {
      steady.push_back(h_iter[wi * N_ITER_MAX + it]);
    }
  }
  std::sort(steady.begin(), steady.end());
  uint32_t ss_min = steady.front();
  uint32_t ss_p50 = steady[steady.size() / 2];
  uint32_t ss_max = steady.back();

  printf(
      "== num_workers=%d N_ITER=%d (total tiles = %d, K-slice depth = %d) ==\n",
      num_workers,
      n_iter,
      (N / BN) * SPLIT_K,
      K / 128 / SPLIT_K);
  printf("  cudaEvent total: %.2f μs (=%.3f μs/iter)\n",
         total_ms * 1000.0,
         total_ms * 1000.0 / n_iter);
  printf("  iter[0] cold: %u ns (%.2f μs)\n",
         max_per_iter[0],
         max_per_iter[0] / 1000.0);
  printf("  iter[1]: %u ns (%.2f μs)\n",
         max_per_iter[1],
         max_per_iter[1] / 1000.0);
  if (n_iter > 4) {
    printf("  iter[%d]: %u ns (%.2f μs)\n",
           n_iter - 1,
           max_per_iter[n_iter - 1],
           max_per_iter[n_iter - 1] / 1000.0);
  }
  printf("  steady-state (iters 1..%d × %d CTAs):\n", n_iter - 1, num_workers);
  printf("    min=%u  p50=%u  max=%u ns  (p50=%.2f μs, max=%.2f μs)\n",
         ss_min,
         ss_p50,
         ss_max,
         ss_p50 / 1000.0,
         ss_max / 1000.0);
}

int main(int argc, char **argv) {
  int dev = (argc > 1) ? atoi(argv[1]) : 0;
  cudaSetDevice(dev);
  printf("=== fp8_gemm_dense SplitK standalone bench ===\n");
  printf("Shape: M=%d K=%d N=%d, BN=%d NS=%d NE=%d SK=%d\n",
         M,
         K,
         N,
         BN,
         NS,
         NE,
         SPLIT_K);
  printf("Tiles total = mm*nn*SK = 1*%d*%d = %d\n",
         N / BN,
         SPLIT_K,
         (N / BN) * SPLIT_K);
  printf("CURRENT STATUS: kernel crashes with unspecified launch failure in\n");
  printf("ALL tested num_workers (including 224 = perfect 1-tile-per-CTA).\n");
  printf("Target after rewrite: p50 ≤ 30 μs (cf bench_decode_no_splitk ~58 "
         "μs).\n");

  size_t a_bytes = (size_t)M * K;
  size_t b_bytes = (size_t)N * K;
  size_t sa_bytes = (size_t)M * (K / 128) * sizeof(float);
  size_t sb_bytes = (size_t)(N / 128) * (K / 128) * sizeof(float);
  size_t c_bytes = (size_t)M * N * sizeof(bf16);

  void *d_a, *d_b, *d_c;
  float *d_sa, *d_sb;
  cudaMalloc(&d_a, a_bytes);
  cudaMalloc(&d_b, b_bytes);
  cudaMalloc(&d_sa, sa_bytes);
  cudaMalloc(&d_sb, sb_bytes);
  cudaMalloc(&d_c, c_bytes);
  cudaMemset(d_a, 1, a_bytes);
  cudaMemset(d_b, 1, b_bytes);
  cudaMemset(d_sa, 0x3f, sa_bytes);
  cudaMemset(d_sb, 0x3f, sb_bytes);

  CUtensorMap ta, tb;
  make_tma_desc(ta, d_a, M, K);
  make_tma_desc(tb, d_b, N, K);

  CUtensorMap *d_ta, *d_tb;
  cudaMalloc(&d_ta, sizeof(CUtensorMap));
  cudaMalloc(&d_tb, sizeof(CUtensorMap));
  cudaMemcpy(d_ta, &ta, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tb, &tb, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  size_t smem = kernel::fp8_gemm_dense_decode_splitk::
      fp8_gemm_dense_decode_splitk_smem_size<BN, NS, NE, SPLIT_K>();
  printf("smem size: %zu bytes\n", smem);
  cudaFuncSetAttribute(persistent_splitk_bench,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem);

  // Sweep num_workers: 80 = builder default (creates multi-tile-iter);
  // 128 = megakernel full worker pool; 224 = match total tile count
  // (perfect 1-tile-per-CTA). All three currently crash.
  for (int nw : {80, 128, 224}) {
    printf("\n");
    run_bench(nw, /*n_iter=*/16, d_ta, d_tb, d_sa, d_sb, (bf16 *)d_c, smem);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_sa);
  cudaFree(d_sb);
  cudaFree(d_ta);
  cudaFree(d_tb);
  return 0;
}
