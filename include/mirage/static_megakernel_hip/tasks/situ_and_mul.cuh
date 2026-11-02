// tasks/situ_and_mul.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// situ_and_mul.cuh: the shared-expert activation of h_s features [F x, F x + F),
// F = 32, 64, 96 or 128. spart (shared gate_up's output, [T][2 SHR], int64 fixed
// point: COMBINE_ADD) is 2 SHR / 128 blocks of 128 columns: block b = 64 gate
// values (features 64 b .. + 63), then their 64 up values. Wait until the NSPLIT
// K parts of every block the task reads are added (sgu_cnt[block]), then hs[t]
// [f] = bf16(SiTU(gate, up)), then this node's counter += 1 (task_post). On
// gfx950 a thread makes 16-B vectors: token t, 8 consecutive features (F / 8
// x T items over the 256 threads); the sums are read at device scope (atomics wrote them).
#pragma once
#include "../core.cuh"

namespace static_mk {

template <int NSPLIT, int F, int SHR>
__device__ __forceinline__ void
    situ_and_mul_task(unsigned long long const *spart,
                      uint32_t *sgu_cnt,
                      uint32_t *done_cnt,
                      __hip_bfloat16 *hs,
                      int x) {
  static_assert(F % 32 == 0 && F <= 128,
                "32, 64, 96 or 128 features per task (at most 2 blocks)");
  int const f0 = F * x; // the task's first feature
  if (STATIC_TIDX == 0) {
    for (int b = f0 / 64; b <= (f0 + F - 1) / 64; b++) {
      cnt_wait_in(sgu_cnt + b, NSPLIT);
    }
  }
  __syncthreads();
  for (int v = STATIC_TIDX; v < F / 8 * T; v += 256) { // item v: token v / (F / 8), 8 features
    int const t = v / (F / 8), f = f0 + 8 * (v % (F / 8)),
              b = f / 64, j = f % 64; // features f .. f + 7: block b, from j
    float h[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      size_t const at = (size_t)t * (2 * SHR) + b * 128 + j + i;
      h[i] = situ(fx_to_float(ld64_dev(spart + at)),
                  fx_to_float(ld64_dev(spart + at + 64)));
    }
    st16_out(hs + (size_t)t * SHR + f,
             make_uint4(pack_bf16x2(h[0], h[1]),
                        pack_bf16x2(h[2], h[3]),
                        pack_bf16x2(h[4], h[5]),
                        pack_bf16x2(h[6], h[7])));
  }
  task_post(done_cnt, 1);
}

// situ_and_mul task x = h_s features [F x, F x + F), F = SHR / SELF::x; adds
// shared gate_up's SGU::y partial sums (one counter add per task per 128-row
// block: SGU::y with 128-row tasks, 2 SGU::y with 64-row tasks)
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
  constexpr int SHR = SGU::v[2] / 2;
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
      cnt_of(g) + SGU::counter,
      cnt_of(g) + SELF::counter,
      buf_at<__hip_bfloat16>(g, SELF::buf),
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
