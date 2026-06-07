// =============================================================================
// Gate-1: fine-N (BN) × pipeline-depth (NS) sweep at active_rows=1 (decode M=1)
// =============================================================================
//
// PURPOSE (reviewer + Codex-mandated discriminator, 2026-06-05)
//   The L6 critical-path ledger found FP8_GEMM_DENSE_SMALLM is SINGLE-WAVE,
//   latency-bound (wall/1-CTA-latency ≈ 1.0; single-CTA latency 18-29μs =
//   5-15× the weights-only HBM floor). Ferret's M=128 sweep found BN=32/NS=8
//   gives 1.4× on qkv_a, attributing it to NS pipeline depth (not occupancy).
//   BUT NS=8 fits SMEM only at BN≤64, so the headline is a BN+NS BUNDLE.
//
//   This bench ISOLATES the two axes AT active_rows=1 (decode), which is the
//   real shape (ferret measured M=128). Decisive controls:
//     BN128/NS3  baseline (in-tree config)
//     BN128/NS6  pure-NS isolation (fits SMEM)   <- if this ≈ BN32/NS8, NS is the lever
//     BN32 /NS3  pure-BN isolation               <- if this already ≈20.5μs, BN is the lever
//     BN32 /NS8  combined qkv_a/gate_up candidate
//     BN64 /NS8  o_proj candidate
//   M=1 vs M=128 also confirms the M-invariance identity (expect equal walls).
//
//   The metric that matters (Codex): steady-state MAX-across-CTAs (layer wall is
//   set by the slowest active CTA), plus cudaEvent/iter. NOT per-CTA p50.
//
//   B TMA descriptor box_outer = BN (the sole fine-N delta vs in-tree mediumm).
//
// BUILD (no GPU needed to compile):
//   /usr/local/cuda/bin/nvcc -O3 -gencode=arch=compute_100a,code=sm_100a \
//     -I include -I include/mirage/persistent_kernel \
//     --expt-relaxed-constexpr -std=c++20 \
//     tests/runtime_python/blackwell/sm100_fp8_gemm_dense/bench_finen_grid.cu \
//     -o /tmp/bench_finen_grid -lcuda
//   CUDA_VISIBLE_DEVICES=<clean> /tmp/bench_finen_grid
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

#include "tasks/blackwell/fp8_gemm_dense_mediumm_sm100.cuh"

using bf16 = __nv_bfloat16;

constexpr int N_ITER = 64;
constexpr int NW_MAX = 256;   // max CTAs we ever launch (o_proj BN32 = 224)

__device__ uint32_t per_iter_ns[NW_MAX * N_ITER];

__device__ __forceinline__ uint32_t get_globaltimer() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ret));
  return ret;
}

template <int BN, int NS>
__global__ __launch_bounds__(256, 1) void bench_kernel(CUtensorMap const *ta_ptr,
                                                       CUtensorMap const *tb_ptr,
                                                       float const *sa,
                                                       float const *sb,
                                                       bf16 *C,
                                                       int M,
                                                       int N,
                                                       int K,
                                                       int num_workers,
                                                       int n_iter) {
  for (int it = 0; it < n_iter; it++) {
    __syncthreads();
    uint32_t t0 = 0;
    if (threadIdx.x == 0)
      t0 = get_globaltimer();
    kernel::fp8_gemm_dense_mediumm::fp8_gemm_dense_mediumm_sm100_task_impl<BN, NS>(
        ta_ptr, tb_ptr, sa, sb, C, M, N, K, blockIdx.x, num_workers);
    __syncthreads();
    if (threadIdx.x == 0)
      per_iter_ns[blockIdx.x * N_ITER + it] = get_globaltimer() - t0;
  }
}

// box_outer parameterized: A uses BM=128, B uses BN.
static void make_tma_desc(CUtensorMap &desc, void *base, uint64_t outer,
                          uint64_t k, uint32_t box_outer) {
  uint64_t gd[2] = {(uint64_t)k, (uint64_t)outer};
  uint64_t gs[1] = {(uint64_t)k};
  uint32_t bd[2] = {128, box_outer};
  uint32_t es[2] = {1, 1};
  CUresult err = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, gd, gs, bd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (err != CUDA_SUCCESS) {
    char const *e = nullptr;
    cuGetErrorString(err, &e);
    fprintf(stderr, "cuTensorMapEncodeTiled failed (box_outer=%u): %s\n",
            box_outer, e);
    exit(1);
  }
}

// Global buffers sized for the largest shape we test (N<=7168, K<=7168, M<=128).
static void *g_a, *g_b, *g_c;
static float *g_sa, *g_sb;

template <int BN, int NS>
static void run_cfg(char const *tag, int M, int N, int K) {
  int num_workers = (N + BN - 1) / BN;          // one CTA per N-tile (single wave if <=148 SMs)
  if (num_workers > NW_MAX) num_workers = NW_MAX;

  // (Re)build descriptors for this shape + BN box.
  CUtensorMap ta, tb;
  make_tma_desc(ta, g_a, /*outer=*/128, /*k=*/K, /*box_outer=*/128);
  make_tma_desc(tb, g_b, /*outer=*/N,   /*k=*/K, /*box_outer=*/BN);
  CUtensorMap *d_ta, *d_tb;
  cudaMalloc(&d_ta, sizeof(CUtensorMap));
  cudaMalloc(&d_tb, sizeof(CUtensorMap));
  cudaMemcpy(d_ta, &ta, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tb, &tb, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  size_t smem = kernel::fp8_gemm_dense_common::smem_size_tpl<BN, NS, 4>();
  cudaError_t fe = cudaFuncSetAttribute(
      bench_kernel<BN, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (fe != cudaSuccess) {
    printf("  %-26s SMEM=%6zu B  SKIP (FuncSetAttribute: %s)\n", tag, smem,
           cudaGetErrorString(fe));
    cudaFree(d_ta); cudaFree(d_tb);
    return;
  }

  // warmup
  for (int i = 0; i < 3; i++)
    bench_kernel<BN, NS><<<num_workers, 256, smem>>>(
        d_ta, d_tb, g_sa, g_sb, (bf16 *)g_c, M, N, K, num_workers, N_ITER);
  cudaError_t le = cudaDeviceSynchronize();
  if (le != cudaSuccess) {
    printf("  %-26s SMEM=%6zu B  LAUNCH-FAIL: %s\n", tag, smem,
           cudaGetErrorString(le));
    cudaFree(d_ta); cudaFree(d_tb);
    return;
  }

  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  bench_kernel<BN, NS><<<num_workers, 256, smem>>>(
      d_ta, d_tb, g_sa, g_sb, (bf16 *)g_c, M, N, K, num_workers, N_ITER);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float tot_ms = 0; cudaEventElapsedTime(&tot_ms, s, e);

  std::vector<uint32_t> h(num_workers * N_ITER);
  cudaMemcpyFromSymbol(h.data(), per_iter_ns, sizeof(uint32_t) * num_workers * N_ITER);

  // steady-state (iters 1..N_ITER-1): max-across-CTAs per iter, then median of those maxima
  std::vector<uint32_t> maxima;
  for (int it = 1; it < N_ITER; it++) {
    uint32_t mx = 0;
    for (int wi = 0; wi < num_workers; wi++)
      mx = std::max(mx, h[wi * N_ITER + it]);
    maxima.push_back(mx);
  }
  std::sort(maxima.begin(), maxima.end());
  uint32_t max_med = maxima[maxima.size() / 2];
  uint32_t max_max = maxima.back();

  printf("  %-26s CTAs=%3d SMEM=%6zuB  evt/iter=%7.2fus  ss-max(med)=%6.2fus  ss-max(max)=%6.2fus\n",
         tag, num_workers, smem, tot_ms * 1000.0 / N_ITER,
         max_med / 1000.0, max_max / 1000.0);
  cudaFree(d_ta); cudaFree(d_tb);
}

int main(int argc, char **argv) {
  int dev = (argc > 1) ? atoi(argv[1]) : 0;
  cudaSetDevice(dev);

  // Largest shape footprint: N<=7168, K<=7168, M<=128.
  const int MAXM = 128, MAXN = 7168, MAXK = 7168;
  size_t a_bytes = (size_t)MAXM * MAXK;
  size_t b_bytes = (size_t)MAXN * MAXK;
  size_t sa_bytes = (size_t)MAXM * (MAXK / 128) * sizeof(float);
  size_t sb_bytes = (size_t)(MAXN / 128) * (MAXK / 128) * sizeof(float);
  size_t c_bytes = (size_t)MAXM * MAXN * sizeof(bf16);
  cudaMalloc(&g_a, a_bytes);  cudaMalloc(&g_b, b_bytes);
  cudaMalloc(&g_sa, sa_bytes); cudaMalloc(&g_sb, sb_bytes);
  cudaMalloc(&g_c, c_bytes);
  cudaMemset(g_a, 1, a_bytes); cudaMemset(g_b, 1, b_bytes);
  cudaMemset(g_sa, 0x3f, sa_bytes); cudaMemset(g_sb, 0x3f, sb_bytes);
  cudaMemset(g_c, 0, c_bytes);

  printf("=== Gate-1: fine-N(BN) × pipeline-depth(NS) @ active_rows, NE=4 ===\n");
  printf("metric: ss-max(med) = median over iters of (max-across-CTAs) = layer-wall proxy\n\n");

  printf("--- qkv_a shape (K=7168, N=2176), M=1 (decode active_rows=1) ---\n");
  run_cfg<128, 3>("BN128/NS3 base", 1, 2176, 7168);
  run_cfg<128, 6>("BN128/NS6 pure-NS", 1, 2176, 7168);
  run_cfg<32, 3>("BN32 /NS3 pure-BN", 1, 2176, 7168);
  run_cfg<32, 8>("BN32 /NS8 candidate", 1, 2176, 7168);
  run_cfg<16, 8>("BN16 /NS8 (A-reload?)", 1, 2176, 7168);
  run_cfg<64, 8>("BN64 /NS8", 1, 2176, 7168);

  printf("\n--- qkv_a shape, M=128 (confirm M-invariance identity vs M=1) ---\n");
  run_cfg<128, 3>("BN128/NS3 base  M128", 128, 2176, 7168);
  run_cfg<32, 8>("BN32 /NS8 cand  M128", 128, 2176, 7168);

  printf("\n--- o_proj shape (K=1792, N=7168), M=1 ---\n");
  run_cfg<128, 3>("BN128/NS3 base", 1, 7168, 1792);
  run_cfg<64, 4>("BN64 /NS4 candidate", 1, 7168, 1792);
  run_cfg<64, 8>("BN64 /NS8", 1, 7168, 1792);
  run_cfg<32, 8>("BN32 /NS8 (2-wave)", 1, 7168, 1792);

  printf("\n--- gate_up shape (K=7168, N=1024), M=1 ---\n");
  run_cfg<128, 3>("BN128/NS3 base", 1, 1024, 7168);
  run_cfg<32, 8>("BN32 /NS8 candidate", 1, 1024, 7168);

  cudaFree(g_a); cudaFree(g_b); cudaFree(g_c); cudaFree(g_sa); cudaFree(g_sb);
  return 0;
}
