// Standalone test for the device-function port of MLA MTP TP=8 (no PDL).
// Mirrors v037's host code; only difference is calling the __device__
// function via a thin __global__ shim that forwards blockIdx.
//
// Goal: verify the port (mla_mtp_tp8_main / mla_mtp_tp8_reduce) is
// correctness-equivalent and perf-equivalent (within noise) to the
// original standalone v037_nopdl, so we know the mechanical port is clean
// before doing MPK plumbing.

#include "mirage/persistent_kernel/tasks/blackwell/mla_mtp_decode_tp8_sm100.cuh"

#include <algorithm>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace kernel::mla_mtp_tp8;

// ===== Thin __global__ shims forwarding blockIdx to __device__ functions =====
template <bool SINGLE_TILE>
__global__ __launch_bounds__(TB) void shim_main(
    const __grid_constant__ CUtensorMap Q_tm,
    const __grid_constant__ CUtensorMap KV_tm,
    nv_bfloat16 *Oa,
    float *La,
    float ss,
    int kv_len,
    int sk,
    int Q_LEN,
    int qpg,
    int Q_LEN_real) {
  mla_mtp_tp8_main<SINGLE_TILE>(&Q_tm,
                                &KV_tm,
                                Oa,
                                La,
                                ss,
                                kv_len,
                                sk,
                                Q_LEN,
                                qpg,
                                Q_LEN_real,
                                blockIdx.x,
                                blockIdx.y);
}

__global__ __launch_bounds__(RD_TB, 4) void shim_reduce(nv_bfloat16 const *Oa,
                                                        float const *La,
                                                        nv_bfloat16 *O,
                                                        int sk,
                                                        int num_groups,
                                                        int Q_LEN,
                                                        int qpg) {
  mla_mtp_tp8_reduce(Oa,
                     La,
                     O,
                     sk,
                     num_groups,
                     Q_LEN,
                     qpg,
                     blockIdx.x,
                     blockIdx.y,
                     blockIdx.z);
}

// ===== Reference kernel (copied verbatim from v037) =====
__global__ void ref_k(nv_bfloat16 const *Q,
                      nv_bfloat16 const *KV,
                      nv_bfloat16 *O,
                      float ss,
                      int kl,
                      int Q_LEN) {
  int h = blockIdx.x, b = blockIdx.y, q = blockIdx.z;
  int t = threadIdx.x;
  nv_bfloat16 const *qr = Q + (q * NUM_HEADS + h) * D_K;
  nv_bfloat16 const *kv = KV + b * kl * D_K;
  nv_bfloat16 *o = O + (q * NUM_HEADS + h) * D_V;

  int causal_lim = (Q_LEN > 1) ? (kl - Q_LEN + q + 1) : kl;

  extern __shared__ float sc[];
  for (int i = t; i < kl; i += blockDim.x) {
    if (i < causal_lim) {
      float s = 0;
      for (int d = 0; d < D_K; d++) {
        s += __bfloat162float(qr[d]) * __bfloat162float(kv[i * D_K + d]);
      }
      sc[i] = s * ss;
    } else {
      sc[i] = -1e30f;
    }
  }
  __syncthreads();
  __shared__ float m2[2];
  if (t == 0) {
    float mx = -1e30f;
    for (int i = 0; i < causal_lim; i++) {
      mx = fmaxf(mx, sc[i]);
    }
    m2[0] = mx;
  }
  __syncthreads();
  for (int i = t; i < kl; i += blockDim.x) {
    sc[i] = (i < causal_lim) ? expf(sc[i] - m2[0]) : 0.0f;
  }
  __syncthreads();
  if (t == 0) {
    float s = 0;
    for (int i = 0; i < causal_lim; i++) {
      s += sc[i];
    }
    m2[1] = 1.0f / s;
  }
  __syncthreads();
  for (int i = t; i < kl; i += blockDim.x) {
    sc[i] *= m2[1];
  }
  __syncthreads();
  for (int d = t; d < D_V; d += blockDim.x) {
    float v = 0;
    for (int i = 0; i < causal_lim; i++) {
      v += sc[i] * __bfloat162float(kv[i * D_K + d]);
    }
    o[d] = __float2bfloat16(v);
  }
}

// ===== Host helpers =====
static void fill(nv_bfloat16 *d, int n) {
  for (int i = 0; i < n; i++) {
    d[i] = __float2bfloat16(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
  }
}
static void ck(CUresult e) {
  if (e != CUDA_SUCCESS) {
    char const *s;
    cuGetErrorString(e, &s);
    fprintf(stderr, "CU: %s\n", s);
    exit(1);
  }
}

int main(int argc, char **argv) {
  cuInit(0);
  int B = 1, KL = 4096;
  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "--b=", 4)) {
      B = atoi(argv[i] + 4);
    }
    if (!strncmp(argv[i], "--k=", 4)) {
      KL = atoi(argv[i] + 4);
    }
  }
  float ss = 1.0f / sqrtf((float)D_K);

  fprintf(stderr,
          "=== MLA MTP TP=8 (Mirage device-function port, no PDL) ===\n");
  fprintf(stderr,
          "H=%d, D_K=%d, D_V=%d, B=%d, KV_LEN=%d\n\n",
          NUM_HEADS,
          D_K,
          D_V,
          B,
          KL);

  size_t KVs = (size_t)B * KL * D_K;
  nv_bfloat16 *hKV = new nv_bfloat16[KVs];
  srand(42);
  fill(hKV, KVs);
  nv_bfloat16 *dKV;
  cudaMalloc(&dKV, KVs * 2);
  cudaMemcpy(dKV, hKV, KVs * 2, cudaMemcpyHostToDevice);

  CUtensorMap KVtm;
  {
    uint64_t gd[3] = {64, (uint64_t)B * KL, (uint64_t)K_ITERS};
    uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
    uint32_t bd[3] = {64, (uint32_t)TILE_S, 1};
    uint32_t es[3] = {1, 1, 1};
    ck(cuTensorMapEncodeTiled(&KVtm,
                              CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                              3,
                              (void *)dKV,
                              gd,
                              gs,
                              bd,
                              es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_NONE,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  for (int Q_LEN_real = 1; Q_LEN_real <= 4; Q_LEN_real++) {
    int Q_LEN = (Q_LEN_real + 1) & ~1; // pad to even
    int qpg = 2;
    int num_groups = Q_LEN / qpg;
    int hpb = NUM_HEADS;

    int max_sk = (KL + TILE_S - 1) / TILE_S;
    int sk = max_sk;

    int kvt = (KL + TILE_S - 1) / TILE_S;
    int tps = (kvt + sk - 1) / sk;
    bool single_tile = (tps == 1);

    size_t Qs_r = (size_t)Q_LEN_real * NUM_HEADS * D_K;
    size_t Qs = (size_t)Q_LEN * NUM_HEADS * D_K; // padded
    size_t Os_r = (size_t)Q_LEN_real * NUM_HEADS * D_V;
    size_t Os = (size_t)Q_LEN * NUM_HEADS * D_V;

    nv_bfloat16 *hQ = new nv_bfloat16[Qs_r], *hO = new nv_bfloat16[Os],
                *hOr = new nv_bfloat16[Os_r];
    fill(hQ, Qs_r);

    nv_bfloat16 *dQ, *dO, *dOr, *dOa;
    float *dLa;
    cudaMalloc(&dQ, Qs * 2);
    cudaMemset(dQ, 0, Qs * 2); // pad to 0
    cudaMalloc(&dO, Os * 2);
    cudaMalloc(&dOr, Os * 2);

    int total_blocks = B * num_groups * sk;
    cudaMalloc(&dOa, (size_t)total_blocks * D_V * 128 * 2);
    cudaMalloc(&dLa, (size_t)total_blocks * 128 * 4);

    cudaMemcpy(dQ, hQ, Qs_r * 2, cudaMemcpyHostToDevice);

    CUtensorMap Qtm;
    {
      uint64_t gd[3] = {64, (uint64_t)B * Q_LEN * NUM_HEADS, (uint64_t)K_ITERS};
      uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
      uint32_t bd[3] = {64, (uint32_t)hpb, 1};
      uint32_t es[3] = {1, 1, 1};
      ck(cuTensorMapEncodeTiled(&Qtm,
                                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                3,
                                (void *)dQ,
                                gd,
                                gs,
                                bd,
                                es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    // Reference (uses Q_LEN_real)
    {
      dim3 g(NUM_HEADS, B, Q_LEN_real);
      int sm = (KL + 1) * 4;
      if (sm > 48000) {
        cudaFuncSetAttribute(
            ref_k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
      }
      ref_k<<<g, 256, sm>>>(dQ, dKV, dOr, ss, KL, Q_LEN_real);
      cudaDeviceSynchronize();
    }

    cudaFuncSetAttribute(shim_main<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_SIZE);
    cudaFuncSetAttribute(shim_main<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_SIZE);

    auto run_main = [&]() {
      dim3 g(num_groups * sk, B);
      if (single_tile) {
        shim_main<true><<<g, TB, SMEM_SIZE>>>(
            Qtm, KVtm, dOa, dLa, ss, KL, sk, Q_LEN, qpg, Q_LEN_real);
      } else {
        shim_main<false><<<g, TB, SMEM_SIZE>>>(
            Qtm, KVtm, dOa, dLa, ss, KL, sk, Q_LEN, qpg, Q_LEN_real);
      }
    };
    auto run_reduce = [&]() {
      dim3 rg((D_V + RD_DV - 1) / RD_DV, num_groups, B);
      shim_reduce<<<rg, RD_TB>>>(dOa, dLa, dO, sk, num_groups, Q_LEN, qpg);
    };
    auto run = [&]() {
      run_main();
      run_reduce();
    };

    run();
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "Q%d: ERR: %s\n", Q_LEN_real, cudaGetErrorString(err));
      cudaFree(dQ);
      cudaFree(dO);
      cudaFree(dOr);
      cudaFree(dOa);
      cudaFree(dLa);
      delete[] hQ;
      delete[] hO;
      delete[] hOr;
      continue;
    }

    cudaMemcpy(hO, dO, Os_r * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOr, dOr, Os_r * 2, cudaMemcpyDeviceToHost);
    float mx = 0;
    for (size_t i = 0; i < Os_r; i++) {
      float r = __bfloat162float(hOr[i]), g = __bfloat162float(hO[i]);
      mx = fmaxf(mx, fabsf(r - g) / fmaxf(fabsf(r), 1e-3f));
    }
    fprintf(stderr,
            "Q%d: err=%.6f sk=%d groups=%d qpg=%d\n",
            Q_LEN_real,
            mx,
            sk,
            num_groups,
            qpg);

    // Warmup
    for (int i = 0; i < 50; i++) {
      run();
    }
    cudaDeviceSynchronize();

    // Best-of-5 timing, 200 iters per trial
    int const N = 200;
    int const TRIALS = 5;
    float best_ms = 1e30f;
    for (int t = 0; t < TRIALS; t++) {
      cudaEvent_t st, en;
      cudaEventCreate(&st);
      cudaEventCreate(&en);
      cudaDeviceSynchronize();
      cudaEventRecord(st);
      for (int i = 0; i < N; i++) {
        run();
      }
      cudaEventRecord(en);
      cudaEventSynchronize(en);
      float total_ms;
      cudaEventElapsedTime(&total_ms, st, en);
      float avg_ms = total_ms / N;
      if (avg_ms < best_ms) {
        best_ms = avg_ms;
      }
      cudaEventDestroy(st);
      cudaEventDestroy(en);
    }
    double fl = (double)B * NUM_HEADS * Q_LEN_real * KL * (D_K + D_V);
    double tflops = fl / (best_ms / 1000.0) / 1e12;
    fprintf(stderr,
            "Q%d: %.2f TFLOPS, %.1f us\n",
            Q_LEN_real,
            tflops,
            best_ms * 1000);

    cudaFree(dQ);
    cudaFree(dO);
    cudaFree(dOr);
    cudaFree(dOa);
    cudaFree(dLa);
    delete[] hQ;
    delete[] hO;
    delete[] hOr;
  }
  return 0;
}
