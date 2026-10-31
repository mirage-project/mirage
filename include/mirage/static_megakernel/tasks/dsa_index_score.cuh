// tasks/dsa_index_score.cuh -- Compute the index scores of the DSA lightning
// indexer (GLM-5.3 / 5.2 "full" layers; GLM-5.3-Flash with FLASH = 1, see
// below). Task (c, t) is chunk c of the context of token t: positions
// [c CH, c CH + CH) below n = min(seq_lens[t], MAXCTX).
// The caller must keep seq_lens[t] <= MAXCTX = MAXP x PAGE. Each slot t must be
// a different request (see step 1).
//   score[t][j] = bf16( sum_h w_h relu(q8_h . k8_j) s_j )
//   q_h   Columns [128 h, 128 h + 128) of wq_b(q_c), rounded to bf16. A
//         COMBINE_ADD scaled_gemm_tile node makes wq_b(q_c). Then RoPE on
//         dims [0, 64) at position seq_lens[t] - 1 (interleaved), then e4m3
//         with the UE8M0 scale qs_h.
//   w_h   bf16(weights_proj(x)[h] qs_h / 64). 64 = sqrt(128) sqrt(32): the
//         softmax scale 1/sqrt(128) times the head-count scale 1/sqrt(32).
//         weights_proj(x)[h] is column 128 + h of the indexer projection (a
//         COMBINE_ADD GEMM node; the demo uses a bf16 gemm_tile).
//   k8_j  The key of position j in the index cache (dsa_index_k_write page
//         layout). s_j is its UE8M0 scale.
// s_j and qs_h are powers of two. Thus a multiplication by them is exact, and
// the order of these multiplications does not change the fp32 result.
// FLASH = 1 (GLM-5.3-Flash, SGLang's IndexerKPool): j counts the pooled keys
// (dsa_kpool_update), n = min(seq_lens[t] / 4, MAXCTX) complete pools. q_h is
// Hadamard-128 of the bf16 wq_b output (a bf16 gemm_tile), rounded to bf16,
// before the e4m3 quantization (no RoPE). w_h = ((wp_h 32^-1/2) qs_h)
// 128^-1/2 in fp32, wp_h = column 256 + h of the indexer projection
// [wk | gate | weights_proj], not rounded. s_j is the fp32 scale of the pooled
// key (a power of two).
// The steps:
//   1. Before the wait, start the cp.async copies of the pages of the chunk to
//      shared memory. Each page has 8 KB of keys and 256 B of scales. Warp w
//      copies the pages w and w + 8 of the chunk. A row is 128 B. kswz XORs
//      the 16-B chunks of a row with the row index (the 128-B swizzle of the
//      tensor core). In this step, only the key of position n - 1 changes.
//      Another slot of the same request would change more keys, so each slot
//      must be a different request.
//   2. After the wait, compute q8 and w of the 32 heads in shared memory. Each
//      task of token t computes the same values. Copy the row of position
//      n - 1 again.
//   3. Warp 0 issues tcgen05 MMAs (kind::f8f6f4, e4m3). For each group of 128
//      positions, D [128 positions][32 heads] fp32 goes to TMEM columns
//      [32 group, 32 group + 32). The tensor core reads the keys and q8 in
//      shared memory. The task does not allocate TMEM. A GEMM task drains
//      its accumulators before it returns. Thus TMEM is free at a task
//      boundary.
//   4. Each thread reads the 32 heads of one position from TMEM. It applies
//      relu and w_h and adds the heads. Then it stores the score and adds the
//      key to the task histogram. At the end, the task adds the task
//      histogram to the token histogram.
//   output  scores fp32 [T][MAXCTX] (bf16 values) in the node buffer. Then
//           the task adds 1 to counter + t. dsa_index_topk waits for all
//           chunks.
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {
namespace dsa {

constexpr int ISC_HEADS = 32, ISC_D = 128, ISC_MAX_PAGES = 16,
              ISC_MAX_POS = ISC_MAX_PAGES * 64;
// Shared memory (offsets from the 1024-aligned base):
//   ISC_K      the keys of the pages of the chunk, [16][64][128] e4m3 (kswz)
//   ISC_Q      q8, [32][128] e4m3 (kswz)
//   ISC_SC     the 64 scales of each page (4 bytes each, as in the cache)
//   ISC_W      w, [32] fp32
//   ISC_HIST   the histogram of the score keys of the task (TOPK_BINS ints).
//              The task adds it to the token histogram at the end.
//   ISC_LIST   the bins that are not zero (at most one for each position)
//   ISC_NLIST  the number of those bins
//   ISC_BAR    the MMA mbarriers, one for each group of 128 positions. The
//              task invalidates them at the end, because a later GEMM
//              writes this memory with TMA.
constexpr int ISC_K = 0, ISC_Q = ISC_K + ISC_MAX_POS * 128,
              ISC_SC = ISC_Q + ISC_HEADS * ISC_D,
              ISC_W = ISC_SC + ISC_MAX_POS * 4,
              ISC_HIST = ISC_W + ISC_HEADS * 4,
              ISC_LIST = ISC_HIST + TOPK_BINS * 4,
              ISC_NLIST = ISC_LIST + ISC_MAX_POS * 4, ISC_BAR = ISC_NLIST + 16,
              ISC_END = ISC_BAR + 8 * 8;
// tcgen05 kind::f8f6f4 instruction descriptor: e4m3 x e4m3 -> fp32, both
// K-major, M = 128 (positions), N = 32 (heads)
constexpr uint32_t ISC_IDESC =
    (1u << 4) | ((ISC_HEADS / 8u) << 17) | ((128u / 16u) << 24);

// Return the offset of byte b of row r in a page buffer. kswz XORs the 16-B
// chunk index b / 16 with r % 8.
__device__ __forceinline__ int kswz(int r, int b) {
  return r * 128 + ((((b >> 4) ^ (r & 7))) << 4) + (b & 15);
}
// D (TMEM) = A B^T (en = 0) or D + A B^T (en = 1). One K step of 32 e4m3.
__device__ __forceinline__ void
    mma_f8(uint32_t d, uint64_t a, uint64_t b, uint32_t en) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a),
      "l"(b),
      "r"(ISC_IDESC),
      "r"(en));
}
// Read 32 fp32 columns of the TMEM lane of this thread.
__device__ __forceinline__ void tmem_ld32(uint32_t addr, float *v) {
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
      "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%"
      "27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]),
        "=r"(r[1]),
        "=r"(r[2]),
        "=r"(r[3]),
        "=r"(r[4]),
        "=r"(r[5]),
        "=r"(r[6]),
        "=r"(r[7]),
        "=r"(r[8]),
        "=r"(r[9]),
        "=r"(r[10]),
        "=r"(r[11]),
        "=r"(r[12]),
        "=r"(r[13]),
        "=r"(r[14]),
        "=r"(r[15]),
        "=r"(r[16]),
        "=r"(r[17]),
        "=r"(r[18]),
        "=r"(r[19]),
        "=r"(r[20]),
        "=r"(r[21]),
        "=r"(r[22]),
        "=r"(r[23]),
        "=r"(r[24]),
        "=r"(r[25]),
        "=r"(r[26]),
        "=r"(r[27]),
        "=r"(r[28]),
        "=r"(r[29]),
        "=r"(r[30]),
        "=r"(r[31])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 32; i++) {
    v[i] = __uint_as_float(r[i]);
  }
}

// Step 1. Start the cp.async copies of the pages w + 8 j of the chunk (warp w)
// to ISC_K and ISC_SC. bt: the block table row of the token.
template <int PPW, int PAGE>
__device__ __forceinline__ void isc_copy_pages(unsigned char *sm,
                                               unsigned char const *icache,
                                               int const *bt,
                                               int c0,
                                               int c1,
                                               int warp,
                                               int lane) {
#pragma unroll
  for (int j = 0; j < PPW; j++) {
    int const cp = warp + 8 * j, lp = c0 / PAGE + cp;
    if (lp * PAGE < c1) {
      unsigned char const *pg =
          icache + (size_t)__ldg(bt + lp) * index_page_bytes(PAGE);
      unsigned char *const kb = sm + ISC_K + cp * 64 * 128;
#pragma unroll
      for (int i = 0; i < 16; i++) { // 16-B chunk q = lane + 32 i: row q / 8
        int const q = lane + 32 * i;
        copy_async(
            reinterpret_cast<__nv_bfloat16 *>(kb + kswz(q / 8, (q % 8) * 16)),
            pg + 16 * q,
            true);
      }
      if (lane < 16) { // the 256 scale bytes
        copy_async(reinterpret_cast<__nv_bfloat16 *>(sm + ISC_SC + cp * 256 +
                                                     16 * lane),
                   pg + 64 * 128 + 16 * lane,
                   true);
      }
    }
  }
  copy_commit();
}

// Step 2. Write q8 and w of the heads 4 warp .. 4 warp + 3 of token t to
// ISC_Q and ISC_W. Lane l takes dims 4 l .. 4 l + 3. iq: the wq_b output
// [T][NQ]; kw: the indexer projection [T][NKW]. (c0, s0, c1, s1): the RoPE
// values of the lane (rope_cs2; FLASH: not used).
template <int NQ, int NKW, bool FLASH>
__device__ __forceinline__ void isc_q8_w(unsigned char *sm,
                                         unsigned long long const *iq,
                                         unsigned long long const *kw,
                                         int t,
                                         int warp,
                                         int lane,
                                         float c0,
                                         float s0,
                                         float c1,
                                         float s1) {
  float *const wsm = reinterpret_cast<float *>(sm + ISC_W);
  // Issue all loads of the lane first (lane 0 also loads the 4 values of w).
  // Thus the loads need one L2 round trip. Each head uses one 32-B load, so the
  // SM reads each 32-B sector of q one time.
  float q[4][4], wv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    unsigned long long x[4];
    asm volatile("ld.global.cg.v4.u64 {%0, %1, %2, %3}, [%4];"
                 : "=l"(x[0]), "=l"(x[1]), "=l"(x[2]), "=l"(x[3])
                 : "l"(iq + (size_t)t * NQ + (4 * warp + j) * ISC_D + 4 * lane)
                 : "memory");
#pragma unroll
    for (int i = 0; i < 4; i++) {
      q[j][i] = bf16r(fx_to_float(x[i]));
    }
    if (lane == 0) {
      wv[j] =
          FLASH ? fx_to_float(__ldcg(kw + (size_t)t * NKW + 256 + 4 * warp + j))
                : fx_bf16(kw, NKW, t, 128 + 4 * warp + j);
    }
  }
  uint32_t q8[4];
  int e[4];
  if constexpr (FLASH) {
    hadamard128<4>(q, lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        q[j][i] = bf16r(q[j][i]);
      }
    }
    quant128<4>(q, q8, e);
  } else {
    rope_quant128<4>(q, lane, c0, s0, c1, s1, q8, e);
  }
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int const h = 4 * warp + j;
    // q8 rows use the page swizzle (kswz), as the MMA B operand needs
    *reinterpret_cast<uint32_t *>(sm + ISC_Q + kswz(h, 4 * lane)) = q8[j];
    if (lane == 0) {
      wsm[h] = FLASH ? wv[j] * 0.17677669529663687f * exp2_int(e[j]) *
                           0.08838834764831845f
                     : bf16r(wv[j] * exp2_int(e[j]) * (1.0f / 64.0f));
    }
  }
}

// Step 3. All lanes of one warp call this. The lanes run the issue loop, so
// the operands stay in uniform registers. One elected lane issues the 4 MMAs
// and the commit of each of the ng groups. bar: the address of ISC_BAR.
__device__ __forceinline__ void
    isc_issue(uint32_t base, uint32_t tb, uint32_t bar, int ng) {
  fence_after();
  base = warp_same<true>(base);
  tb = warp_same<true>(tb);
  uint64_t const dq = mkdesc(base + ISC_Q);
  for (int gq = 0; gq < ng; gq++) {
    uint64_t const dk = mkdesc(base + ISC_K + gq * 128 * 128);
    if (issuer_lane<true>()) {
#pragma unroll
      for (int k = 0; k < 4; k++) { // + 32 B = + 2 in the descriptor
        mma_f8(tb + 32 * gq, dk + 2 * k, dq + 2 * k, k == 0 ? 0u : 1u);
      }
      tc_commit(bar + 8 * gq);
    }
  }
  __syncwarp();
}

// Step 4. Warp w reads TMEM lanes [32 (w % 4), 32 (w % 4) + 32). It reads the
// groups w / 4, w / 4 + 2, and so on. Warp 0 issues the MMAs. Thus, for TMEM
// lanes 0..31, warp 4 reads the groups 0..5, and warp 0 reads the groups 6
// and 7 after the issue. out: the scores of token t; positions [c0, c1).
// FLASH: the key scales are fp32 (else UE8M0 bytes).
template <bool FLASH>
__device__ __forceinline__ void isc_epilogue(unsigned char *sm,
                                             uint32_t tb,
                                             uint32_t bar,
                                             int ng,
                                             int c0,
                                             int c1,
                                             float *out,
                                             int warp,
                                             int lane) {
  float const *wsm = reinterpret_cast<float const *>(sm + ISC_W);
  int *const shist = reinterpret_cast<int *>(sm + ISC_HIST);
  int *const blist = reinterpret_cast<int *>(sm + ISC_LIST);
  int *const nlist = reinterpret_cast<int *>(sm + ISC_NLIST);
  int const qd = warp & 3;
  int const g0 = qd != 0 ? warp >> 2 : (warp == 0 ? 6 : 0),
            g1 = qd == 0 && warp == 4 ? min(ng, 6) : ng, step = qd == 0 ? 1 : 2;
  float wr[ISC_HEADS];
#pragma unroll
  for (int h = 0; h < ISC_HEADS; h++) {
    wr[h] = wsm[h];
  }
  for (int gq = g0; gq < g1; gq += step) {
    mbar_wait(bar + 8 * gq, 0);
    fence_after();
    float v[32];
    tmem_ld32(tb + 32 * gq + ((uint32_t)(32 * qd) << 16), v);
    int const o = 128 * gq + 32 * qd + lane; // position c0 + o
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < ISC_HEADS; h++) {
      s += wr[h] * fmaxf(v[h], 0.f);
    }
    if (c0 + o < c1) {
      float const ks =
          FLASH ? *reinterpret_cast<float const *>(sm + ISC_SC + 4 * o)
                : exp2_int(sm[ISC_SC + 4 * o]);
      float const sc = bf16r(s * ks);
      out[c0 + o] = sc;
      int const bin = (int)(score_key16(sc) >> 4);
      if (atomicAdd(shist + bin, 1) == 0) { // the first key of the bin
        blist[atomicAdd(nlist, 1)] = bin;
      }
    }
  }
  fence_before(); // the TMEM reads, before the MMAs of a later task
}

// Add the task histogram to the token histogram ghist. Only the bins in the
// list are not zero. All 256 threads call this after a __syncthreads.
__device__ __forceinline__ void isc_flush_hist(unsigned char const *sm,
                                               int *ghist) {
  int const *shist = reinterpret_cast<int const *>(sm + ISC_HIST);
  int const *blist = reinterpret_cast<int const *>(sm + ISC_LIST);
  int const nl = *reinterpret_cast<int const *>(sm + ISC_NLIST);
  for (int i = threadIdx.x; i < nl; i += 256) {
    int const b = blist[i];
    atomicAdd(ghist + b, shist[b]);
  }
}

} // namespace dsa

// Task (c, t). PARAMS {32 (heads), MAXCTX, PAGE (64), MAXP, FLASH (0 / 1)}.
// BUF_SLOTS {cos_sin (FLASH: not read), seq_lens, block_table (FLASH: of the
// pooled cache), index cache (FLASH: the pooled cache), hist (int
// [T][TOPK_BINS], zero before the launch: the counts of the high 12 bits of
// the score keys)}. IQ: wq_b (COMBINE_ADD, 4096 columns). KW: the indexer
// projection (COMBINE_ADD; column 128 + h is w_h, FLASH 256 + h). IKW: the
// dsa_index_k_write node (FLASH: dsa_kpool_update). The task waits for all IKW
// tasks, so the keys of this step are in the cache.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class IQ,
          class KW,
          class IKW,
          class... IN>
__device__ __forceinline__ void run_dsa_index_score(Maps const &,
                                                    G const &g,
                                                    KernelLocals &L,
                                                    StaticTask const &tk) {
  using namespace dsa;
  constexpr int NIH = PARAMS::v[0], MAXCTX = PARAMS::v[1], PAGE = PARAMS::v[2],
                MAXP = PARAMS::v[3];
  constexpr bool FLASH = PARAMS::v[4] != 0;
  constexpr int W0 = FLASH ? 256 : 128; // the column of w_0 in KW
  constexpr int NQ = IQ::v[2], NKW = KW::v[2], CH = MAXCTX / SELF::x;
  constexpr int CS = slot_at<BUF_SLOTS, 0>(), SEQ = slot_at<BUF_SLOTS, 1>(),
                BT = slot_at<BUF_SLOTS, 2>(), ICACHE = slot_at<BUF_SLOTS, 3>(),
                HIST = slot_at<BUF_SLOTS, 4>();
  static_assert(NIH == ISC_HEADS && NQ == NIH * ISC_D && NKW >= W0 + NIH,
                "dsa_index_score: 32 heads of 128 dims");
  static_assert(PAGE == 64 && CH * SELF::x == MAXCTX && CH % PAGE == 0,
                "dsa_index_score: chunks of whole 64-position pages");
  constexpr int PPW = (CH / PAGE + 7) / 8; // pages of each warp
  static_assert(ISC_END <= RING_BYTES && CH <= ISC_MAX_POS,
                "dsa_index_score: its shared memory is the ring's (at most "
                "1024 positions in a chunk)");
  static_assert(
      MAXP * PAGE >= MAXCTX,
      "dsa_index_score: the block table covers MAXCTX positions (MAXP "
      "is its row stride: SGLang's may be wider)");
  // the keys of all T tokens: dsa_index_k_write's T tasks, or
  // dsa_kpool_update's T tasks, or its T / NT tasks of NT tokens of a request
  // each (its params[4], a speculative verify step)
  constexpr int IKW_NT = (PARAMS::v[4] == 1 && IKW::v[4] > 0) ? IKW::v[4] : 1;
  static_assert(
      IKW::counter >= 0 && IKW::x * IKW::y * IKW::z * IKW_NT == T,
      "dsa_index_score: the keys of all T tokens (dsa_index_k_write, or "
      "dsa_kpool_update per token or per request)");
  static_assert(SELF::y == T && SELF::buf >= 0 && SELF::counter >= 0,
                "dsa_index_score: grid (chunks, tokens), scores, a counter per "
                "token");
  int const t = tk.y, cx = tk.x, tid = threadIdx.x, warp = tid >> 5,
            lane = tid & 31;
  unsigned char *const sm = reinterpret_cast<unsigned char *>(L.sm);
  int const seq = buf_at<int const>(g, SEQ)[t];
  int const n = min(MAXCTX, FLASH ? seq / 4 : seq); // FLASH: complete pools
  int const c0 = cx * CH, c1 = min(c0 + CH, n);
  int const *bt = buf_at<int const>(g, BT) + (size_t)t * MAXP;
  unsigned char const *icache = buf_at<unsigned char const>(g, ICACHE);
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  // Step 1. The keys before position n - 1 do not change in this step. Thus
  // the copies can run during the wait. After the wait, the task copies the
  // row of position n - 1 again.
  isc_copy_pages<PPW, PAGE>(sm, icache, bt, c0, c1, warp, lane);
  for (int b = tid; b < TOPK_BINS; b += 256) {
    reinterpret_cast<int *>(sm + ISC_HIST)[b] = 0;
  }
  if (tid == 0) {
    *reinterpret_cast<int *>(sm + ISC_NLIST) = 0;
  }
  // Also before the wait, load the RoPE values of position n - 1 (an idle
  // slot, seq_lens[t] <= 0, uses position 0). Find the page of the key of
  // position n - 1 if this warp copied it.
  float rc0 = 1.f, rs0 = 0.f, rc1 = 1.f, rs1 = 0.f;
  if constexpr (!FLASH) {
    rope_cs2(buf_at<__nv_bfloat16 const>(g, CS),
             max(n - 1, 0),
             lane,
             rc0,
             rs0,
             rc1,
             rs1);
  }
  int const lpn = (n - 1) / PAGE, on = (n - 1) % PAGE, cpn = lpn - c0 / PAGE;
  bool const own_n = n >= 1 && c0 <= n - 1 && n - 1 < c1 && cpn % 8 == warp;
  unsigned char const *const pgn =
      own_n ? icache + (size_t)__ldg(bt + lpn) * index_page_bytes(PAGE)
            : nullptr;
  if (tid == 0) {
    for (int gq = 0; gq < 8; gq++) { // the commits of the MMAs
      mbar_init(L.base + ISC_BAR + 8 * gq, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    fx_wait_cols<IQ>(g, 0, NQ);
    fx_wait_cols<KW>(g, W0, W0 + NIH);
    if (lane == 0) {
      cnt_wait(cnt_now(g) + IKW::counter, IKW::x * IKW::y * IKW::z);
    }
  }
  __syncthreads();
  stamp_ready(L);
  // A page that was copied before the wait can hold the key of position n - 1.
  // dsa_index_k_write writes this key in this step. Load its row and scale
  // again now, together with the loads of q. Store them after the copies.
  uint4 rown = make_uint4(0u, 0u, 0u, 0u);
  uint32_t scn = 0u;
  if (own_n) {
    if (lane < 8) {
      rown = __ldcg(reinterpret_cast<uint4 const *>(pgn + on * 128) + lane);
    } else if (lane == 8) {
      scn = __ldcg(reinterpret_cast<uint32_t const *>(pgn + 64 * 128) + on);
    }
  }
  // Step 2.
  isc_q8_w<NQ, NKW, FLASH>(sm,
                           buf_at<unsigned long long const>(g, IQ::buf),
                           buf_at<unsigned long long const>(g, KW::buf),
                           t,
                           warp,
                           lane,
                           rc0,
                           rs0,
                           rc1,
                           rs1);
  copy_wait();
  __syncwarp(); // the copies of all lanes of the warp are complete
  if (own_n) {
    if (lane < 8) {
      *reinterpret_cast<uint4 *>(sm + ISC_K + cpn * 64 * 128 +
                                 kswz(on, 16 * lane)) = rown;
    } else if (lane == 8) {
      *reinterpret_cast<uint32_t *>(sm + ISC_SC + cpn * 256 + 4 * on) = scn;
    }
  }
  // the tensor core (the async proxy) reads q8 and the pages
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); // q8, w and the pages are in shared memory
  // Steps 3 and 4. ng is the number of groups of 128 positions (0 if the
  // chunk is empty).
  int const ng = (max(c1 - c0, 0) + 127) / 128;
  if (ng > 0) {
    if (warp == 0) {
      isc_issue(L.base, L.tb, L.base + ISC_BAR, ng);
    }
    isc_epilogue<FLASH>(sm,
                        L.tb,
                        L.base + ISC_BAR,
                        ng,
                        c0,
                        c1,
                        buf_at<float>(g, SELF::buf) + (size_t)t * MAXCTX,
                        warp,
                        lane);
  }
  __syncthreads();
  if (tid < 8) { // all waits are done: invalidate the mbarriers
    asm volatile(
        "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(L.base + ISC_BAR + 8 * tid)
        : "memory");
  }
  isc_flush_hist(sm, buf_at<int>(g, HIST) + (size_t)t * TOPK_BINS);
  ring_release();
  if (tid == 0) {
    cnt_add(cnt_now(g) + SELF::counter + t, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_index_score() {
  return dsa::ISC_END;
}

} // namespace static_mk
