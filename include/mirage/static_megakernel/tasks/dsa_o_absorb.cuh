// tasks/dsa_o_absorb.cuh -- Compute the MLA value up-projection (GLM-5.3 /
// 5.2). Task x is (head h, value column block cb) of the NH heads on this GPU.
// Each task does all T tokens:
//   v[t][h 256 + cb 64 + c] = bf16(sum_k o_lat[t][h][k] W_UV[h][k][cb 64 + c])
// W_UV [NH][512][256] bf16 is the dequantized V part of kv_b_proj. o_lat is the
// dsa_mla_merge output. o_proj reads v [T][NH 256] bf16 through group_quant.
// The steps:
//   1. Start the copy of the W_UV block (512 x 64) to shared memory. The
//      weights do not depend on this step, so the copy runs during the wait.
//   2. Wait for the T merge tasks of head h (the merge counts per head). Copy
//      o_lat of head h to shared memory.
//   3. Compute the block on tensor cores (dsa::absorb_mma).
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {

constexpr int OAB_LAT = 512, OAB_COLS = 64;
// shared memory: the W_UV block [512][64] bf16, o_lat [T][512] bf16, then the
// K-split partial sums of absorb_mma
constexpr int OAB_OFF_X = OAB_LAT * OAB_COLS * 2,
              OAB_OFF_RED = OAB_OFF_X + T * OAB_LAT * 2;

// Task x is h * 4 + cb. PARAMS {NH}. BUF_SLOTS {W_UV (bf16 [NH][512][256])}.
// OL: the dsa_mla_merge node (grid (NH, T); its counter + h counts the tasks
// of head h).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class OL,
          class... IN>
__device__ __forceinline__ void run_dsa_o_absorb(Maps const &,
                                                 G const &g,
                                                 KernelLocals &L,
                                                 StaticTask const &tk) {
  constexpr int NH = PARAMS::v[0], LAT = OAB_LAT, VD = 256;
  constexpr int WUV = slot_at<BUF_SLOTS, 0>();
  static_assert(SELF::x == NH * 4 && SELF::buf >= 0 && SELF::counter >= 0,
                "dsa_o_absorb: 4 tasks per head, its output v and counter");
  static_assert(OL::buf >= 0 && OL::counter >= 0 && OL::x == NH && OL::y == T,
                "dsa_o_absorb: o_lat from dsa_mla_merge, a counter per head");
  int const h = tk.x / 4, cb = tk.x % 4, tid = threadIdx.x;
  auto *const ws = reinterpret_cast<__nv_bfloat16 *>(L.sm);
  auto *const xs = reinterpret_cast<__nv_bfloat16 *>(L.sm + OAB_OFF_X);
  float *const red = reinterpret_cast<float *>(L.sm + OAB_OFF_RED);
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  dsa::absorb_load_w<LAT, OAB_COLS>(ws,
                                    buf_at<__nv_bfloat16 const>(g, WUV) +
                                        (size_t)h * LAT * VD + cb * OAB_COLS,
                                    VD);
  if (tid == 0) {
    cnt_wait(cnt_now(g) + OL::counter + h, T);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  // o_lat of head h: other SMs wrote it, cp.async.cg reads it from L2
  __nv_bfloat16 const *o = buf_at<__nv_bfloat16 const>(g, OL::buf);
  for (int i = tid; i < T * LAT / 8; i += 256) {
    int const t = i / (LAT / 8), k = (i % (LAT / 8)) * 8;
    dsa::copy_async(
        xs + dsa::swizzle<LAT>(t, k), o + ((size_t)t * NH + h) * LAT + k, true);
  }
  dsa::copy_commit();
  dsa::copy_wait();
  __syncthreads();
  dsa::absorb_mma<LAT, OAB_COLS>(ws,
                                 xs,
                                 red,
                                 buf_at<__nv_bfloat16>(g, SELF::buf) + h * VD +
                                     cb * OAB_COLS,
                                 (size_t)NH * VD);
  dsa::ring_release();
  if (tid == 0) {
    cnt_add(cnt_now(g) + SELF::counter, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_o_absorb() {
  return OAB_OFF_RED + dsa::absorb_red_bytes<OAB_COLS>();
}

} // namespace static_mk
