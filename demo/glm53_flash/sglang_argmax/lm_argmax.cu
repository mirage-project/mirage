// lm_argmax: greedy next-token ids from the TP-sharded lm_head output without gathering the logits (GLM-5.3-Flash copy of
// Kimi K3's mpk_kernels csrc/k3_lm_argmax/layer_plugin.cu; same kernel, same key, same exchange protocol).
//
// SGLang's decode tail (logits_processor.py _get_logits + sampler.py greedy branch) per step, GLM-5.3-Flash bs 64, TP 4,
// vocab 154880 (38720 columns per rank):
//   all_gather (multimem)                          [64][38720] bf16 per rank -> [64][154880] bf16 on every rank (19.8 MB)
//   copy into the fp32 logits buffer               -> [64][154880] fp32
//   torch.argmax(logits, -1)                       -> int64 [64]   (eager, after the graph replay)
// This kernel gives the same int64 ids: every rank reduces its own [rows][W] shard to one 64-bit key per (row, part), stores
// the key into every rank's exchange slot (peer stores over NVLink), and block (row, 0) of every rank reduces the
// world x NP keys of its row.
//
// Key (unsigned 64 bit, larger = wins):
//   bits 63..32  ordered(value): a monotone map of the fp32 value; every NaN maps to 0xFFFFFFFF (above +inf), -0 to +0
//   bits 31..18  tag: the low 14 bits of this row's launch count (0 inside the block reduction)
//   bits 17..0   IDX_MASK - global column
// torch's ArgMaxOps (GreaterOrNan): a NaN beats every number, NaN vs NaN and equal values -> the lower index. The key order
// is the same total order, so the maximum key is torch.argmax's answer.
//
// Value of column j: float(bf16(float(x[j]) * scale)) when use_scale (logits.mul_ on bf16: opmath float, rounded to bf16), else
// float(x[j]). Columns with global index >= V (vocab padding) are skipped (SGLang slices them off after the gather).
//
// Cross-launch safety (graph replays and eager calls on one stream, no PDL; every rank launches the same sequence):
//   g = cnt[row] + 1 (read by every part's thread 0 before its store; block (row, 0) writes cnt[row] = g after it saw all
//   world x NP keys, so a part's read always precedes the write). Slots are double-buffered by g & 1: rank A's store of launch
//   g + 2 into parity g & 1 needs A's launch g + 1 to have ended, i.e. A saw B's key of g + 1, which B stores only after B's
//   launch g (the one reading parity g & 1) ended. A polled slot was last written by launch g - 2 of that row (tag g - 2), or
//   is still zero from the allocation (tag 0) while g < 3; the row's tag is 0 again only at g = 16384, long after every slot
//   of the row was written.
//   NP is fixed (not chosen per row count) so every launch that has row r writes all of row r's polled slots.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int MAXR = 64;        // rows (decode batch) per launch
constexpr int MAXW = 8;         // ranks
constexpr int NP = 2;           // parts per row (grid.y): 64 rows -> 128 CTAs on 148 SMs
constexpr int THREADS = 512;
constexpr int UNR = 4;          // 16-byte loads in flight per thread
constexpr int IDX_BITS = 18;    // global column < 2^18 = 262144 (vocab 154880)
constexpr unsigned TAG_BITS = 14;
constexpr unsigned long long IDX_MASK = (1ull << IDX_BITS) - 1;
constexpr unsigned TAG_MASK = (1u << TAG_BITS) - 1;
constexpr unsigned long long TIMEOUT_NS = 30ull * 1000 * 1000 * 1000;   // a lost peer: trap instead of a silent hang
constexpr size_t SLOT_WORDS = 2ull * MAXR * MAXW * NP;                   // [parity][row][src rank][part]

struct Peers {
  unsigned long long* p[MAXW];
};

__host__ __device__ inline uint32_t ordered_bits(uint32_t f) {
  if ((f & 0x7FFFFFFFu) > 0x7F800000u) return 0xFFFFFFFFu;   // NaN (any sign / payload)
  if (f == 0x80000000u) f = 0u;                              // -0 == +0
  return (f & 0x80000000u) ? ~f : (f | 0x80000000u);
}

__host__ __device__ inline unsigned long long make_key(uint32_t fbits, uint32_t gidx) {
  return ((unsigned long long)ordered_bits(fbits) << 32) | (IDX_MASK - gidx);
}

__device__ __forceinline__ unsigned long long umax64(unsigned long long a, unsigned long long b) { return a > b ? a : b; }

__device__ __forceinline__ void st_relaxed_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned long long ld_relaxed_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint4 ld_stream16(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned long long warp_max(unsigned long long v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = umax64(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}

__global__ void __launch_bounds__(THREADS) lm_argmax_kernel(const __nv_bfloat16* __restrict__ x, int ldx, int W, int col0,
                                                             int V, float scale, int use_scale,
                                                             unsigned long long* __restrict__ slots, Peers peers, int rank,
                                                             int world, int* __restrict__ cnt, long long* __restrict__ ids) {
  const int r = blockIdx.x, p = blockIdx.y, t = threadIdx.x;
  __shared__ unsigned long long red[THREADS / 32];
  __shared__ unsigned long long key_s;
  __shared__ unsigned g_s;
  if (t == 0) g_s = (unsigned)(*(volatile int*)&cnt[r]) + 1u;

  const int wp = W / NP;   // host: W % (8 * NP) == 0
  const int c0 = col0 + p * wp;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)r * ldx + (size_t)p * wp);
  unsigned long long best = 0;   // below every real key (a real key's high word is >= 0x007FFFFF)
  const int nv = wp / 8;   // 16-byte vectors of this part (bs 64, TP 4, W 38720: 2420 -> <= 5 per thread, 2 rounds of UNR)
  for (int base = t; base < nv; base += THREADS * UNR) {
    uint4 q[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int v = base + u * THREADS;
      q[u] = v < nv ? ld_stream16(src + v) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int v = base + u * THREADS;
      if (v >= nv) continue;
      const uint32_t w4[4] = {q[u].x, q[u].y, q[u].z, q[u].w};
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        uint32_t fb = (k & 1) ? (w4[k >> 1] & 0xFFFF0000u) : (w4[k >> 1] << 16);
        if (use_scale) fb = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(__uint_as_float(fb) * scale)) << 16;
        const uint32_t gidx = (uint32_t)(c0 + v * 8 + k);
        if (gidx < (uint32_t)V) best = umax64(best, make_key(fb, gidx));
      }
    }
  }
  best = warp_max(best);
  if ((t & 31) == 0) red[t >> 5] = best;
  __syncthreads();
  if (t < 32) {
    best = warp_max(t < THREADS / 32 ? red[t] : 0ull);
    if (t == 0) key_s = best | ((unsigned long long)(g_s & TAG_MASK) << IDX_BITS);
  }
  __syncthreads();
  const unsigned g = g_s, tag = g & TAG_MASK, par = g & 1u;
  const size_t row_base = ((size_t)par * MAXR + r) * MAXW * NP;   // this row's [src rank][part] words
  if (t < world) {
    unsigned long long* dst = peers.p[0];   // select, not a dynamic index (that copies the parameter struct to local memory)
#pragma unroll
    for (int i = 1; i < MAXW; ++i) dst = (t == i) ? peers.p[i] : dst;
    st_relaxed_sys(dst + row_base + (size_t)rank * NP + p, key_s);
  }
  if (p != 0) return;

  // block (row, 0): every rank's every part of this row
  best = 0;
  if (t < world * NP) {
    const unsigned long long* s = slots + row_base + t;
    const unsigned long long t0 = globaltimer();
    unsigned long long v;
    for (;;) {
      v = ld_relaxed_sys(s);
      if ((unsigned)((v >> IDX_BITS) & TAG_MASK) == tag) break;
      if (globaltimer() - t0 > TIMEOUT_NS) __trap();
    }
    best = v;
  }
  if (t < 32) {
    best = warp_max(best);
    if (t == 0) {
      ids[r] = (long long)(IDX_MASK - (best & IDX_MASK));
      *(volatile int*)&cnt[r] = (int)g;
    }
  }
}

}  // namespace

extern "C" {

// bytes of the symmetric exchange region (every rank the same)
unsigned long long flash_lm_argmax_slot_bytes() { return SLOT_WORDS * sizeof(unsigned long long); }

int flash_lm_argmax_max_rows() { return MAXR; }

int flash_lm_argmax_max_ranks() { return MAXW; }

// x: this rank's [rows][W] bf16 shard (row stride ldx elements), global columns col0 .. col0 + W - 1; V: vocab (columns >= V
// skipped). slots: this rank's copy of the exchange region; peer_ptrs: every rank's copy mapped here (world entries).
// cnt: int32 [MAXR] launch counts (zero at start, private to this rank); ids: int64 [rows]. Returns a cudaError_t.
int flash_lm_argmax(void* stream, const void* x, int rows, int ldx, int W, int col0, int V, float scale, int use_scale,
                    void* slots, const unsigned long long* peer_ptrs, int rank, int world, void* cnt, void* ids) {
  if (rows < 1 || rows > MAXR || world < 1 || world > MAXW || rank < 0 || rank >= world) return (int)cudaErrorInvalidValue;
  if (W <= 0 || W % (8 * NP) != 0 || ldx < W || ldx % 8 != 0 || V <= 0 || (unsigned long long)V > IDX_MASK) {
    return (int)cudaErrorInvalidValue;
  }
  if (((uintptr_t)x & 15) != 0 || ((uintptr_t)slots & 7) != 0) return (int)cudaErrorMisalignedAddress;
  Peers pp{};
  for (int i = 0; i < world; ++i) pp.p[i] = reinterpret_cast<unsigned long long*>(peer_ptrs[i]);
  lm_argmax_kernel<<<dim3(rows, NP), THREADS, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)x, ldx, W, col0, V, scale, use_scale, (unsigned long long*)slots, pp, rank, world, (int*)cnt,
      (long long*)ids);
  return (int)cudaGetLastError();
}

}
