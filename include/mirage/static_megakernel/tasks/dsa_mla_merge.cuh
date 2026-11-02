// tasks/dsa_mla_merge.cuh -- Merge the splits of sparse MLA decode. Task (h, t)
// is head h of token t:
//   o[t][h] = bf16( sum_s e^(lse_s - m) o_s / sum_s e^(lse_s - m) ),
//   m = max_s lse_s, over the S splits of sparse_mla.
// Each o_s is already the softmax average of its split. A split with no valid
// position has lse = -inf. The output is o_lat [T][NH][512] bf16, the attention
// output in the latent space. Then the task adds 1 to counter + h (dsa_o_absorb
// waits for the T tasks of each head).
#pragma once
#include "../dsa_common.cuh"
#include <cmath>

namespace static_mk {

// Task (h, t). PARAMS {NH}. SM: the sparse_mla node (grid (S, T); its
// counter + t counts the splits of token t).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SM,
          class... IN>
__device__ __forceinline__ void run_dsa_mla_merge(Maps const &,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int NH = PARAMS::v[0], S = SM::x, LAT = dsa::MLA_LATENT;
  static_assert(SM::y == T && SM::buf >= 0 && SM::counter >= 0 &&
                    SM::v[0] == NH,
                "dsa_mla_merge: a sparse_mla node's splits of NH heads");
  static_assert(SELF::x == NH && SELF::y == T && SELF::buf >= 0 &&
                    SELF::counter >= 0,
                "dsa_mla_merge: grid (heads, tokens), o_lat and a counter");
  int const h = tk.x, t = tk.y;
  // The task uses no shared memory, so it starts with no barrier.
  if (threadIdx.x == 0) {
    cnt_wait(g.cnt + SM::counter + t, S);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  float const *o = buf_at<float const>(g, SM::buf);
  float const *lse = o + (size_t)T * S * NH * LAT;
  int const c = 2 * threadIdx.x;
  // Issue all loads first (the S lse values and the S column pairs of the
  // thread). Thus they go to L2 at the same time.
  float l[S];
  float2 v[S];
#pragma unroll
  for (int s = 0; s < S; s++) {
    asm volatile("ld.global.cg.f32 %0, [%1];"
                 : "=f"(l[s])
                 : "l"(lse + ((size_t)t * S + s) * NH + h)
                 : "memory");
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
                 : "=f"(v[s].x), "=f"(v[s].y)
                 : "l"(o + (((size_t)t * S + s) * NH + h) * LAT + c)
                 : "memory");
  }
  float m = -INFINITY;
#pragma unroll
  for (int s = 0; s < S; s++) {
    m = fmaxf(m, l[s]);
  }
  float wsum = 0.f;
#pragma unroll
  for (int s = 0; s < S; s++) {
    l[s] = m == -INFINITY ? 0.f : __expf(l[s] - m);
    wsum += l[s];
  }
  float const inv = wsum > 0.f ? 1.0f / wsum : 0.f;
  float2 acc = make_float2(0.f, 0.f);
#pragma unroll
  for (int s = 0; s < S; s++) {
    acc.x += l[s] * v[s].x;
    acc.y += l[s] * v[s].y;
  }
  *reinterpret_cast<__nv_bfloat162 *>(buf_at<__nv_bfloat16>(g, SELF::buf) +
                                      ((size_t)t * NH + h) * LAT + c) =
      __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter + h, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_mla_merge() {
  return 0;
}

} // namespace static_mk
