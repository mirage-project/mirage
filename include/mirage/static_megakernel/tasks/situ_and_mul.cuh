// tasks/situ_and_mul.cuh -- the shared-expert activation of h_s features [F x,
// F x + F), F = 32, 64, 96 or 128. spart (shared gate_up's output, [T][2 SHR],
// int64 fixed point: gemm_tile.cuh COMBINE_ADD; SHR = the features, half of
// shared gate_up's rows) is 2 SHR / 128 blocks of 128 columns: block b = 64
// gate values (features 64 b .. 64 b + 63), then their 64 up values. Feature f
// is in block f / 64 at position f % 64. Wait until the NSPLIT K parts of every
// block the task reads are added (shared gate_up's counters sgu_cnt[block]: 1
// block, or 2 when the features cross a block boundary), then hs[t][f] =
// bf16(SiTU(gate, up)), then this node's counter done_cnt += 1. F threads work,
// one feature each.
#pragma once
#include "../core.cuh"

namespace static_mk {

// 64 features per task (the default grid): exactly one block, two warps
template <int NSPLIT, int SHR>
__device__ __forceinline__ void
    situ_and_mul_task_64(unsigned long long const *spart,
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
      gv[t] = fx_to_float(spart[t * (2 * SHR) + tile * 128 +
                                f]); // fixed point (gemm_tile.cuh COMBINE_ADD)
      uv[t] = fx_to_float(spart[t * (2 * SHR) + tile * 128 + f + 64]);
    }
#pragma unroll
    for (int t = 0; t < T; t++) {
      hs[t * SHR + tile * 64 + f] = __float2bfloat16(situ(gv[t], uv[t]));
    }
    __syncwarp();
    asm volatile("bar.sync 2, 64;" ::: "memory");
    if (threadIdx.x == 0) {
      cnt_add(done_cnt, 1);
    }
  }
  __syncthreads();
}

template <int NSPLIT, int F, int SHR>
__device__ __forceinline__ void
    situ_and_mul_task(unsigned long long const *spart,
                      uint32_t *sgu_cnt,
                      uint32_t *done_cnt,
                      __nv_bfloat16 *hs,
                      int x) {
  if constexpr (F == 64) {
    situ_and_mul_task_64<NSPLIT, SHR>(spart, sgu_cnt, done_cnt, hs, x);
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
      hs[t * SHR + f] = __float2bfloat16(situ(gv[t], uv[t]));
    }
    __syncwarp();
    asm volatile("bar.sync 2, %0;" ::"n"(F) : "memory");
    if (threadIdx.x == 0) {
      cnt_add(done_cnt, 1);
    }
  }
  __syncthreads();
}

// situ_and_mul task x = h_s features [F x, F x + F), F = SHR / SELF::x; adds
// shared gate_up's SGU::y partial sums. Each 128-row block of shared gate_up
// gets one counter add per task: SGU::y with 128-row tasks, 2 SGU::y with
// 64-row tasks (SGU::x = 2 SHR / 64)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SGU,
          class... REST>
__device__ __forceinline__ void run_situ_and_mul(Maps const &,
                                                 G const &g,
                                                 KernelLocals &,
                                                 StaticTask const &tk) {
  constexpr int SHR =
      SGU::v[2] / 2; // the features: half of shared gate_up's rows
  static_assert(SGU::v[0] == COMBINE_ADD && SHR % 64 == 0,
                "situ_and_mul: shared gate_up's 2 SHR sums, added in fixed "
                "point, blocks of 64 features");
  static_assert(SGU::x * 128 % (2 * SHR) == 0,
                "shared gate_up tasks of 128 or 64 rows");
  static_assert(SHR % SELF::x == 0, "situ_and_mul tasks of equal size");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "situ_and_mul: its output h_s and its counter");
  situ_and_mul_task<SGU::y *(SGU::x * 128 / (2 * SHR)), SHR / SELF::x, SHR>(
      buf_at<unsigned long long const>(g, SGU::buf),
      g.cnt + SGU::counter,
      g.cnt + SELF::counter,
      buf_at<__nv_bfloat16>(g, SELF::buf),
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_situ_and_mul() {
  return 0;
}

} // namespace static_mk
