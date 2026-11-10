// tasks/row_rmsnorm.cuh -- RMSNorm of the rows a residual_add node wrote: task
// (q, t) of grid (Q, T, 1) = token t, part q of Q of the row's H columns:
//   out[t][c] = bf16(r[t][c] rstd_t gamma[c]), rstd_t = rsqrt(mean_c r^2 + eps)
// r is the residual_add node's sum before its bf16 rounding (its h32 scratch,
// fp32 [T][H]); the task waits until every task of that node is counted
// (residual_add_layer counted), then reads the whole row itself (each of a
// token's Q tasks adds the same squares in the same order: no swap between
// them) and writes its own columns. This is the norm of SGLang's fused add +
// RMSNorm after a sublayer (FlashInfer: fp32 h = x + residual, its squares and
// h rstd w; only the residual it stores is rounded), with the add done by
// residual_add: the next sublayer's input from the same kernel.
#pragma once
#include "../core.cuh"

namespace static_mk {

// row_rmsnorm task (q, t): H / 8 vectors of 8 columns over the 256 threads
template <int Q, int H, int EPS_BITS>
__device__ __forceinline__ void row_rmsnorm_task(float const *r,
                                                 __nv_bfloat16 const *gamma,
                                                 __nv_bfloat16 *out,
                                                 int q,
                                                 int t) {
  constexpr int NV = H / 8, VPT = (NV + 255) / 256, COLS = H / Q;
  static_assert(H % (8 * Q) == 0,
                "row_rmsnorm: whole 8-column vectors per part");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float const *row = r + (size_t)t * H;
  float s[VPT][8];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * 256 + threadIdx.x;
    if (vec < NV) {
      float4 const u0 = *reinterpret_cast<float4 const *>(row + 8 * vec),
                   u1 = *reinterpret_cast<float4 const *>(row + 8 * vec + 4);
      s[j][0] = u0.x, s[j][1] = u0.y, s[j][2] = u0.z, s[j][3] = u0.w;
      s[j][4] = u1.x, s[j][5] = u1.y, s[j][6] = u1.z, s[j][7] = u1.w;
      ss += s[j][0] * s[j][0] + s[j][1] * s[j][1] + s[j][2] * s[j][2] +
            s[j][3] * s[j][3];
      ss += s[j][4] * s[j][4] + s[j][5] * s[j][5] + s[j][6] * s[j][6] +
            s[j][7] * s[j][7];
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, d);
  }
  __shared__ float red[8];
  if (lane == 0) {
    red[warp] = ss;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = 0.f;
    for (int w = 0; w < 8; w++) {
      tot += red[w];
    }
    red[0] = rsqrtf(tot / H + __int_as_float(EPS_BITS));
  }
  __syncthreads();
  float const rstd = red[0];
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * 256 + threadIdx.x, c = 8 * vec;
    if (vec < NV && c >= q * COLS && c < (q + 1) * COLS) {
      uint4 const gu = *reinterpret_cast<uint4 const *>(gamma + c);
      __nv_bfloat16 const *gb = reinterpret_cast<__nv_bfloat16 const *>(&gu);
      float4 n0, n1;
      n0.x = s[j][0] * rstd * __bfloat162float(gb[0]);
      n0.y = s[j][1] * rstd * __bfloat162float(gb[1]);
      n0.z = s[j][2] * rstd * __bfloat162float(gb[2]);
      n0.w = s[j][3] * rstd * __bfloat162float(gb[3]);
      n1.x = s[j][4] * rstd * __bfloat162float(gb[4]);
      n1.y = s[j][5] * rstd * __bfloat162float(gb[5]);
      n1.z = s[j][6] * rstd * __bfloat162float(gb[6]);
      n1.w = s[j][7] * rstd * __bfloat162float(gb[7]);
      *reinterpret_cast<uint4 *>(out + (size_t)t * H + c) = pack_bf16x8(n0, n1);
    }
  }
  __syncthreads();
}

// row_rmsnorm task (x, y) = (part x of SELF::x, token y). RA: the residual_add
// node (counted: its counter gets one add per task, after the task's stores).
// PARAMS {H, eps (fp32 bits)}; BUF_SLOTS {the residual_add node's h32 scratch
// (fp32 [T][H]), gamma ([H] bf16), out ([T][H] bf16)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class RA,
          class... IN>
__device__ __forceinline__ void run_row_rmsnorm(Maps const &,
                                                G const &g,
                                                KernelLocals &,
                                                StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  static_assert(RA::counter >= 0,
                "row_rmsnorm: its input from a counted residual_add node");
  static_assert(SELF::y == T, "row_rmsnorm: a task row per token");
  if (threadIdx.x == 0) {
    cnt_wait(cnt_now(g) + RA::counter, RA::x * RA::y * RA::z);
  }
  __syncthreads();
  row_rmsnorm_task<SELF::x, H, PARAMS::v[1]>(
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 2>()),
      tk.x,
      tk.y);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_row_rmsnorm() {
  return 0;
}

} // namespace static_mk
