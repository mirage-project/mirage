// tasks/dsa_q_absorb.cuh -- Compute the absorbed MLA query (GLM-5.3 / 5.2:
// NOPE 192, ROPE 64; GLM-5.3-Flash: NOPE 256, ROPE 0). Task x is (head h,
// latent column block cb) of the NH heads on this GPU. Each task does all T
// tokens:
//   q[t][h][cb 128 + c] = bf16( sum_k q_nope[t][h][k] W_UK[h][k][cb 128 + c] )
// W_UK [NH][NOPE][512] bf16 is the dequantized K part of kv_b_proj. With
// ROPE = 64, the task with cb = 0 also writes the rotated q_pe:
//   q[t][h][512 + d] = bf16(RoPE(q_pe[t][h]))  (interleaved, at seq_lens[t]-1)
// With QHD = NOPE + ROPE, q_nope is columns [h QHD, h QHD + NOPE) of q_b, and
// q_pe is columns [h QHD + NOPE, h QHD + QHD). A COMBINE_ADD GEMM node makes
// q_b (the demo uses a scaled_gemm_tile). The task rounds q_b to bf16.
// sparse_mla reads the output q [T][NH][512 + ROPE] bf16. The steps:
//   1. Start the copy of the W_UK block (NOPE x 128) to shared memory. The
//      weights do not depend on this step, so the copy runs during the wait.
//   2. Wait for the q_b columns of head h. Write q_nope to shared memory.
//   3. Compute the block on tensor cores (dsa::absorb_mma).
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {

constexpr int QAB_COLS = 128;
// shared memory: the W_UK block [NOPE][128] bf16, then q_nope [T][NOPE] bf16
__host__ __device__ constexpr int qab_off_x(int nope) {
  return nope * QAB_COLS * 2;
}

// Task x is h * 4 + cb. PARAMS {NH, NOPE, ROPE}. BUF_SLOTS {W_UK (bf16
// [NH][NOPE][512]), cos_sin (not read with ROPE = 0), seq_lens}. QB: q_b
// (COMBINE_ADD, NH x (NOPE + ROPE) columns).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class QB,
          class... IN>
__device__ __forceinline__ void run_dsa_q_absorb(Maps const &,
                                                 G const &g,
                                                 KernelLocals &L,
                                                 StaticTask const &tk) {
  constexpr int NH = PARAMS::v[0], NOPE = PARAMS::v[1], ROPE = PARAMS::v[2],
                QHD = NOPE + ROPE, LAT = 512, D = LAT + ROPE;
  constexpr int N = QB::v[2];
  constexpr int WUK = slot_at<BUF_SLOTS, 0>(), CS = slot_at<BUF_SLOTS, 1>(),
                SEQ = slot_at<BUF_SLOTS, 2>();
  static_assert(N == NH * QHD && (ROPE == 64 || ROPE == 0) && NOPE % 64 == 0,
                "dsa_q_absorb: q_b has NOPE + ROPE columns per head");
  static_assert(SELF::x == NH * 4 && SELF::buf >= 0 && SELF::counter >= 0,
                "dsa_q_absorb: 4 tasks per head, its output q and counter");
  static_assert(ROPE == 0 || T * 32 == 256,
                "dsa_q_absorb: the q_pe step uses token tid / 32");
  auto *const ws = reinterpret_cast<__nv_bfloat16 *>(L.sm);
  auto *const xs = reinterpret_cast<__nv_bfloat16 *>(L.sm + qab_off_x(NOPE));
  int const h = tk.x / 4, cb = tk.x % 4, tid = threadIdx.x;
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  dsa::absorb_load_w<NOPE, QAB_COLS>(ws,
                                     buf_at<__nv_bfloat16 const>(g, WUK) +
                                         (size_t)h * NOPE * LAT + cb * QAB_COLS,
                                     LAT);
  if (tid < 32) {
    dsa::fx_wait_cols<QB>(g, h * QHD, h * QHD + QHD);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  auto const *qb = buf_at<unsigned long long const>(g, QB::buf);
  for (int i = tid; i < T * NOPE / 2; i += 256) {
    int const t = i / (NOPE / 2), k = 2 * (i % (NOPE / 2));
    *reinterpret_cast<__nv_bfloat162 *>(xs + dsa::swizzle<NOPE>(t, k)) =
        __floats2bfloat162_rn(dsa::fx_bf16(qb, N, t, h * QHD + k),
                              dsa::fx_bf16(qb, N, t, h * QHD + k + 1));
  }
  dsa::copy_wait();
  __syncthreads();
  __nv_bfloat16 *q = buf_at<__nv_bfloat16>(g, SELF::buf);
  dsa::absorb_mma<NOPE, QAB_COLS>(
      ws, xs, nullptr, q + (size_t)h * D + cb * QAB_COLS, (size_t)NH * D);
  if (ROPE > 0 && cb == 0) {
    // q_pe: thread tid takes token tid / 32 and pair tid % 32
    int const t = tid / 32, i = tid % 32;
    // an idle slot (seq_lens[t] <= 0) uses position 0
    int const pos = max(buf_at<int const>(g, SEQ)[t] - 1, 0);
    dsa::fx_rope_store(qb,
                       N,
                       t,
                       h * QHD + NOPE + 2 * i,
                       buf_at<__nv_bfloat16 const>(g, CS),
                       pos,
                       i,
                       q + ((size_t)t * NH + h) * D + LAT + 2 * i);
  }
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
constexpr int smem_dsa_q_absorb() {
  return qab_off_x(PARAMS::v[1]) + T * PARAMS::v[1] * 2;
}

} // namespace static_mk
