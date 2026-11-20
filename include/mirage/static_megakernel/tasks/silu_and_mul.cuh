// tasks/silu_and_mul.cuh -- the shared-expert activation of h_s features
// [F x, F x + F), F = 32, 64, 96 or 128:
//   hs[t][f] = bf16(SiLU(gate) up), SiLU(g) = g / (1 + exp(-g))
// (GLM-5.3, DeepSeek-V3 and other SwiGLU experts); with a swiglu limit
// (GLM-5.3-Flash), gate clamped to at most the limit and up to [-limit, limit]
// first. The layout and the counters
// are the same as in situ_and_mul.cuh (K3's SiTU).
//   spart  shared gate_up's output, [T][2 SHR], int64 fixed point (COMBINE_ADD
//          of gemm_tile.cuh or scaled_gemm_tile.cuh). It is 2 SHR / 128 blocks
//          of 128 columns: block b = 64 gate values (features 64 b .. 64 b +
//          63), then their 64 up values. Feature f is in block f / 64, at
//          position f % 64.
//   SHR    the features: half of shared gate_up's rows
// Steps of a task (F threads work, one feature each):
//   1. Wait until the NSPLIT K parts of each block that the task reads are
//      added (shared gate_up's counters sgu_cnt[block]: 1 block, or 2 when the
//      features cross a block boundary).
//   2. Write hs[t][f].
//   3. Add 1 to this node's counter done_cnt.
#pragma once
#include "../core.cuh"

namespace static_mk {

// 64 features per task (the default grid): exactly one block, two warps.
// LIMIT: the swiglu limit's fp32 bits (0: none)
template <int NSPLIT, int SHR, int LIMIT>
__device__ __forceinline__ void
    silu_and_mul_task_64(unsigned long long const *spart,
                         uint32_t *sgu_cnt,
                         uint32_t *done_cnt,
                         __nv_bfloat16 *hs,
                         int tile) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    cnt_wait(sgu_cnt + tile, NSPLIT);
  }
  __syncthreads();
  if (warp < 2) {
    int const f = warp * 32 + lane;
    float gv[T], uv[T];
#pragma unroll
    for (int t = 0; t < T; t++) {
      gv[t] = fx_to_float(
          spart[t * (2 * SHR) + tile * 128 + f]); // fixed point (COMBINE_ADD)
      uv[t] = fx_to_float(spart[t * (2 * SHR) + tile * 128 + f + 64]);
    }
#pragma unroll
    for (int t = 0; t < T; t++) {
      hs[t * SHR + tile * 64 + f] =
          __float2bfloat16(silu2_limit<LIMIT>(gv[t], uv[t]));
    }
    __syncwarp();
    asm volatile("bar.sync 2, 64;" ::: "memory");
    if (threadIdx.x == 0) {
      cnt_add(done_cnt, 1);
    }
  }
  __syncthreads();
}

template <int NSPLIT, int F, int SHR, int LIMIT>
__device__ __forceinline__ void
    silu_and_mul_task(unsigned long long const *spart,
                      uint32_t *sgu_cnt,
                      uint32_t *done_cnt,
                      __nv_bfloat16 *hs,
                      int x) {
  if constexpr (F == 64) {
    silu_and_mul_task_64<NSPLIT, SHR, LIMIT>(spart, sgu_cnt, done_cnt, hs, x);
    return;
  }
  static_assert(
      F % 32 == 0 && F <= 128,
      "32, 64, 96 or 128 features per task (whole warps; at most 2 blocks)");
  int const f0 = F * x; // the task's first feature
  if (threadIdx.x == 0) {
    for (int b = f0 / 64; b <= (f0 + F - 1) / 64; b++) {
      cnt_wait(sgu_cnt + b, NSPLIT);
    }
  }
  __syncthreads();
  if (threadIdx.x < F) {
    int const f = f0 + threadIdx.x, b = f / 64,
              j = f % 64; // feature f: block b, position j
    float gv[T], uv[T];
#pragma unroll
    for (int t = 0; t < T; t++) {
      gv[t] = fx_to_float(spart[t * (2 * SHR) + b * 128 + j]);
      uv[t] = fx_to_float(spart[t * (2 * SHR) + b * 128 + j + 64]);
    }
#pragma unroll
    for (int t = 0; t < T; t++) {
      hs[t * SHR + f] = __float2bfloat16(silu2_limit<LIMIT>(gv[t], uv[t]));
    }
    __syncwarp();
    asm volatile("bar.sync 2, %0;" ::"n"(F) : "memory");
    if (threadIdx.x == 0) {
      cnt_add(done_cnt, 1);
    }
  }
  __syncthreads();
}

// silu_and_mul task x = h_s features [F x, F x + F), F = SHR / SELF::x.
//   SGU  shared gate_up, a gemm_tile or scaled_gemm_tile node (params {mode, K,
//        N, ...}). The task adds its SGU::y partial sums.
// Each task of shared gate_up adds 1 to the counter of its 128-row block. Thus
// a block gets SGU::y adds with 128-row tasks, and 2 SGU::y adds with 64-row
// tasks (SGU::x = 2 SHR / 64). PARAMS {} or {the swiglu limit as fp32 bits}.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SGU,
          class... REST>
__device__ __forceinline__ void run_silu_and_mul(Maps const &,
                                                 G const &g,
                                                 KernelLocals &,
                                                 StaticTask const &tk) {
  constexpr int SHR =
      SGU::v[2] / 2; // the features: half of shared gate_up's rows
  static_assert(SGU::v[0] == COMBINE_ADD && SHR % 64 == 0,
                "silu_and_mul: shared gate_up's 2 SHR sums, added in fixed "
                "point, blocks of 64 features");
  static_assert(SGU::x * 128 % (2 * SHR) == 0,
                "shared gate_up tasks of 128 or 64 rows");
  static_assert(SHR % SELF::x == 0, "silu_and_mul tasks of equal size");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "silu_and_mul: its output h_s and its counter");
  silu_and_mul_task<SGU::y *(SGU::x * 128 / (2 * SHR)),
                    SHR / SELF::x,
                    SHR,
                    param_or<PARAMS, 0, 0>()>(
      buf_at<unsigned long long const>(g, SGU::buf),
      cnt_now(g) + SGU::counter,
      cnt_now(g) + SELF::counter,
      buf_at<__nv_bfloat16>(g, SELF::buf),
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_silu_and_mul() {
  return 0;
}

} // namespace static_mk
