// =============================================================================
// SplitK handoff bench A: fp8_gemm_dense MEDIUMM at DSv3 O_proj decode shape
// =============================================================================
//
// PURPOSE
//   Standalone microbenchmark reproducing the FP8 dense GEMM call that
//   dominates DSv3 decode O_proj wallclock. Used together with bench_decode_
//   splitk.cu (companion file) to discriminate "kernel itself is slow" vs
//   "megakernel framing is slow".
//
// CURRENT STATUS (2026-05-16, measured)
//   - Kernel under test: fp8_gemm_dense_mediumm_sm100_task_impl<128, 3>
//     in `include/mirage/persistent_kernel/tasks/blackwell/
//     fp8_gemm_dense_mediumm_sm100.cuh` (thin trampoline to
//     fp8_gemm_dense_common::task_impl_tpl<128, 3, NE=4> in
//     `fp8_gemm_dense_sm100_common.cuh`).
//   - Shape: M=128 (mbt), K=16384 (= 32 q-heads * 512 v-dim-absorbed), N=7168
//     (per TP=4 rank, RowParallel — N is full hidden, K is sharded across
//     ranks). Active rows in decode iter = 1; kernel still runs full M=128
//     MMA but `mi < runtime_m` gates the global write.
//   - Roofline (memory bound): B fp8 weight = N*K = 7168*16384 = 117 MB,
//     B200 HBM 8 TB/s → 14.7 μs min wallclock for one full GEMM.
//   - Stock tile count: mm=1, nn=N/BN=56, total=56. Only ~38% of B200's 148
//     SMs busy without K-split.
//   - Measured (steady-state per-CTA wallclock, GPU 1): ~58 μs.
//     **4× over roofline.** That's the gap a working SplitK should close.
//   - Megakernel max (perfetto, same shape, same kernel): 72.5 μs.
//   - Megakernel framing overhead = 72.5 - 58 = 14 μs (added on top of the
//     standalone kernel time when run inside the persistent megakernel).
//
// HOW TO BUILD + RUN
//   cd /home/muhengl/mirage
//   /usr/local/cuda/bin/nvcc -O3 -gencode=arch=compute_100a,code=sm_100a \
//       -I include \
//       -I include/mirage/persistent_kernel \
//       --expt-relaxed-constexpr -std=c++20 \
//       tests/runtime_python/blackwell/sm100_fp8_gemm_dense/bench_decode_no_splitk.cu \
//       -o /tmp/bench_decode_no_splitk -lcuda
//   # Run on any clean B200 (pick a free GPU index; default is 0):
//   CUDA_VISIBLE_DEVICES=1 /tmp/bench_decode_no_splitk 0
//
// WHAT THE OUTPUT LOOKS LIKE (CURRENT)
//   == N_ITER=16 ==
//     cudaEvent total: 957.76 μs (=59.860 μs/iter)
//     iter[0] max-across-CTAs: 67584 ns (67.58 μs) [COLD]
//     steady-state (iters 1..15 × 80 CTAs):
//       min=640  p50=57920  p99=59648  max=60480 ns
//       p50=57.92 μs  max=60.48 μs
//
//   The number to optimize is **steady-state p50** (currently 57.92 μs).
//   Roofline lower bound is ~14.7 μs. Target: ≤30 μs (≈2× speedup from
//   SplitK + saturated HBM utilization).
//
// FOR THE KERNEL OWNER
//   1. This bench MUST keep working when you optimize the kernel.
//   2. The kernel under test is in:
//        include/mirage/persistent_kernel/tasks/blackwell/
//          fp8_gemm_dense_sm100_common.cuh::task_impl_tpl
//      The smallm/mediumm wrappers are thin trampolines around it.
//   3. You should also use the companion bench `bench_decode_splitk.cu` to
//      validate your new SplitK kernel against this baseline. Speedup
//      target: ≥2× over this bench's p50 (so ≤30 μs).
//   4. The full megakernel test path is `demo/deepseek_v3/demo.py` with
//      `MPK_DSV3_NEW_MOE=1 --layers 0-19 --max-num-batched-tokens 128
//      --ep-size 2` on 4× B200. End-to-end win is bounded by per-MoE-layer
//      wallclock currently at 441 μs/layer/token (vs vLLM reference 143 μs).
//
// SPLITK NECESSITY (roofline argument)
//   - 56 active CTAs / 148 SM = 38% utilization. Even at HBM peak per active
//     SM, total achievable BW is bounded by aggregate HBM (8 TB/s sustained
//     ≈ 5-6 TB/s typical). 56 SMs cannot saturate HBM.
//   - With SplitK=4: total = 56*4 = 224 tiles. With num_workers=148 we'd use
//     all SMs in 2 waves. Per-CTA work shrinks to K/4 = 32 K-iters.
//   - Theoretical 1.5-2× speedup; brings us close to roofline.
//
// CROSS-REF
//   - Companion bench: bench_decode_splitk.cu (broken SplitK kernel, target
//     for rewrite).
//   - Builder integration: python/mirage/mpk/models/deepseek_v3/builder.py:
//     633-637 (selects mediumm/smallm based on max_seq_length) and 661-737
//     (env-gated MPK_DSV3_DECODE_OPROJ_SPLITK path — currently default off
//     because the splitk kernel crashes).
//
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "tasks/blackwell/fp8_gemm_dense_mediumm_sm100.cuh"

using bf16 = __nv_bfloat16;

constexpr int M = 128, N = 7168, K = 16384;
constexpr int BN = 128, NS = 3;          // mediumm uses NE=4 internally
constexpr int NUM_WORKERS = 80;          // _fp8_dense_num_workers() default
constexpr int N_ITER_MAX = 256;

__device__ uint32_t per_iter_ns[NUM_WORKERS * N_ITER_MAX];

__device__ __forceinline__ uint32_t get_globaltimer() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ret));
  return ret;
}

__global__ __launch_bounds__(256, 1) void
    persistent_gemm_bench(CUtensorMap const *ta_ptr,
                          CUtensorMap const *tb_ptr,
                          float const *sa, float const *sb,
                          bf16 *C, int n_iter) {
  for (int it = 0; it < n_iter; it++) {
    __syncthreads();
    uint32_t t0 = 0;
    if (threadIdx.x == 0) t0 = get_globaltimer();
    // __noinline__ wrapper that mirrors megakernel dispatch.
    kernel::fp8_gemm_dense_mediumm::fp8_gemm_dense_mediumm_sm100_task_impl<BN, NS>(
        ta_ptr, tb_ptr, sa, sb, C, M, N, K,
        /*worker_idx=*/blockIdx.x,
        /*num_workers=*/NUM_WORKERS);
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
  CUresult err = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      base, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (err != CUDA_SUCCESS) {
    char const *errstr = nullptr;
    cuGetErrorString(err, &errstr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errstr);
    exit(1);
  }
}

void run_bench(int n_iter, CUtensorMap *d_ta, CUtensorMap *d_tb,
               float *d_sa, float *d_sb, bf16 *d_c, size_t smem) {
  for (int i = 0; i < 3; i++) {
    persistent_gemm_bench<<<NUM_WORKERS, 256, smem>>>(d_ta, d_tb, d_sa, d_sb, d_c, n_iter);
  }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  persistent_gemm_bench<<<NUM_WORKERS, 256, smem>>>(d_ta, d_tb, d_sa, d_sb, d_c, n_iter);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, s, e);

  std::vector<uint32_t> h_iter(NUM_WORKERS * N_ITER_MAX);
  cudaMemcpyFromSymbol(h_iter.data(), per_iter_ns,
                       sizeof(uint32_t) * NUM_WORKERS * N_ITER_MAX);

  std::vector<uint32_t> max_per_iter(n_iter, 0);
  for (int it = 0; it < n_iter; it++) {
    for (int wi = 0; wi < NUM_WORKERS; wi++) {
      uint32_t v = h_iter[wi * N_ITER_MAX + it];
      if (v > max_per_iter[it]) max_per_iter[it] = v;
    }
  }

  std::vector<uint32_t> steady(NUM_WORKERS * (n_iter - 1));
  for (int wi = 0; wi < NUM_WORKERS; wi++) {
    for (int it = 1; it < n_iter; it++) {
      steady[wi * (n_iter - 1) + (it - 1)] = h_iter[wi * N_ITER_MAX + it];
    }
  }
  std::sort(steady.begin(), steady.end());
  uint32_t ss_min = steady.front();
  uint32_t ss_p50 = steady[steady.size() / 2];
  uint32_t ss_p99 = steady[(int)(steady.size() * 0.99)];
  uint32_t ss_max = steady.back();

  printf("\n== N_ITER=%d ==\n", n_iter);
  printf("  cudaEvent total: %.2f μs (=%.3f μs/iter)\n",
         total_ms * 1000.0, total_ms * 1000.0 / n_iter);
  printf("  iter[0] max-across-CTAs: %u ns (%.2f μs) [COLD]\n",
         max_per_iter[0], max_per_iter[0] / 1000.0);
  printf("  iter[1] max-across-CTAs: %u ns (%.2f μs)\n",
         max_per_iter[1], max_per_iter[1] / 1000.0);
  if (n_iter > 4) {
    printf("  iter[%d] max-across-CTAs: %u ns (%.2f μs)\n",
           n_iter / 2, max_per_iter[n_iter / 2],
           max_per_iter[n_iter / 2] / 1000.0);
    printf("  iter[%d] max-across-CTAs: %u ns (%.2f μs)\n",
           n_iter - 1, max_per_iter[n_iter - 1],
           max_per_iter[n_iter - 1] / 1000.0);
  }
  printf("  steady-state (iters 1..%d × %d CTAs):\n", n_iter - 1, NUM_WORKERS);
  printf("    min=%u  p50=%u  p99=%u  max=%u ns\n",
         ss_min, ss_p50, ss_p99, ss_max);
  printf("    p50=%.2f μs  max=%.2f μs\n",
         ss_p50 / 1000.0, ss_max / 1000.0);
}

int main(int argc, char **argv) {
  int dev = (argc > 1) ? atoi(argv[1]) : 0;
  cudaSetDevice(dev);
  printf("=== fp8_gemm_dense MEDIUMM standalone bench (baseline, NO SplitK) ===\n");
  printf("Shape: M=%d K=%d N=%d, BN=%d NS=%d NE=4 (mediumm), num_workers=%d\n",
         M, K, N, BN, NS, NUM_WORKERS);
  printf("Roofline: B weight = N*K = %.1f MB, at 8 TB/s ≈ %.2f μs (full GPU)\n",
         (double)N * K / 1e6, (double)N * K / 8e12 * 1e6);
  printf("Current measured: steady-state p50 ≈ 58 μs (4× over roofline).\n");
  printf("Target after SplitK rewrite: ≤30 μs (≥2× speedup over baseline).\n");

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
  cudaMemset(d_c, 0, c_bytes);

  CUtensorMap ta, tb;
  make_tma_desc(ta, d_a, M, K);
  make_tma_desc(tb, d_b, N, K);

  CUtensorMap *d_ta, *d_tb;
  cudaMalloc(&d_ta, sizeof(CUtensorMap));
  cudaMalloc(&d_tb, sizeof(CUtensorMap));
  cudaMemcpy(d_ta, &ta, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tb, &tb, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  size_t smem = kernel::fp8_gemm_dense_common::smem_size_tpl<BN, NS, 4>();
  printf("smem size: %zu bytes\n", smem);
  cudaFuncSetAttribute(persistent_gemm_bench,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

  // N_ITER sweep: cold-cache first iter, steady-state later iters, and watch
  // for DVFS / preemption outliers at large N_ITER.
  for (int n : {16, 64, 256}) {
    run_bench(n, d_ta, d_tb, d_sa, d_sb, (bf16*)d_c, smem);
  }

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  cudaFree(d_sa); cudaFree(d_sb);
  cudaFree(d_ta); cudaFree(d_tb);
  return 0;
}
