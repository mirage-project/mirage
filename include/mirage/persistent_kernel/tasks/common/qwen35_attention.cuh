/* Copyright 2026 The Mirage Team. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/common/qwen35_sdpa.cuh"

namespace kernel {
// One synchronized token of the dense text decoder. The caller supplies BF16
// RoPE tables and owns the position; all three text MRoPE axes are identical.
template <typename T, int Q_HEADS=24, int KV_HEADS=4, int ROWS=1>
__device__ __forceinline__ void qwen35_qk_rope_task(
    void const *qg_ptr, void const *k_ptr, void const *qw_ptr,
    void const *kw_ptr, void const *rope_ptr, void *q_out, void *k_out,
    void *gate_out) {
  auto qg = static_cast<T const *>(qg_ptr);
  auto k = static_cast<T const *>(k_ptr);
  auto qw = static_cast<float const *>(qw_ptr);
  auto kw = static_cast<float const *>(kw_ptr);
  auto rope = static_cast<T const *>(rope_ptr); // [cos(64), sin(64)]
  auto qo = static_cast<T *>(q_out);
  auto ko = static_cast<T *>(k_out);
  auto gate = static_cast<T *>(gate_out);
  // The wider model stages normalized values in its output buffers to stay
  // within the persistent worker's static shared-memory budget.
  __shared__ T normalized[Q_HEADS == 8 ? (Q_HEADS + KV_HEADS) * 256 : 1];
  __shared__ float inv[Q_HEADS + KV_HEADS];
  __shared__ float sums[256];
  // Native mean shares a 512-thread block among output heads. At 16 or
  // more batch*head rows it uses 32 lanes per head, otherwise 64. Four
  // heads share this task block; every physical lane reaches each barrier.
  for (int group = 0; group < (Q_HEADS + KV_HEADS + 3) / 4; ++group) {
    int h = group * 4 + threadIdx.x / 64, lane = threadIdx.x % 64;
    int lanes = h < Q_HEADS ? (ROWS * Q_HEADS >= 16 ? 32 : 64)
                           : (ROWS * KV_HEADS >= 16 ? 32 : 64);
    float ss = 0.f;
    if (h < Q_HEADS + KV_HEADS && lane < lanes) {
      auto x = h < Q_HEADS ? qg + h * 512 : k + (h - Q_HEADS) * 256;
      float parts[4] = {};
      for (int at = lane * 4; at < 256; at += lanes * 4)
        for (int j = 0; j < 4; ++j) {
          float v = float(x[at+j]);
          parts[j] = __fadd_rn(parts[j], __fmul_rn(v, v));
        }
      ss = __fadd_rn(__fadd_rn(__fadd_rn(parts[0], parts[1]), parts[2]), parts[3]);
    }
    sums[threadIdx.x] = ss;
    __syncthreads();
    if (lane < 32) {
      if (lanes == 64) ss = __fadd_rn(ss, sums[threadIdx.x + 32]);
      for (int offset = 1; offset < 32; offset <<= 1)
        ss = __fadd_rn(ss, __shfl_down_sync(0xffffffff, ss, offset));
      if (lane == 0 && h < Q_HEADS + KV_HEADS) inv[h] = rsqrtf(ss / 256.f + 1.e-6f);
    }
    __syncthreads();
  }
  __syncthreads();
  for (int i = threadIdx.x; i < (Q_HEADS + KV_HEADS) * 256; i += blockDim.x) {
    int h = i / 256, d = i % 256;
    auto x = h < Q_HEADS ? qg + h * 512 : k + (h - Q_HEADS) * 256;
    auto w = h < Q_HEADS ? qw : kw;
    T value = T(__fmul_rn(__fmul_rn(float(x[d]), inv[h]), 1.f + w[d]));
    if constexpr (Q_HEADS == 8) normalized[i] = value;
    else (h < Q_HEADS ? qo + h*256 : ko + (h-Q_HEADS)*256)[d] = value;
  }
  __syncthreads();
  if constexpr (Q_HEADS != 8) {
    for (int h = 0; h < Q_HEADS + KV_HEADS; ++h) {
      int d = threadIdx.x;
      auto destination = h < Q_HEADS ? qo + h*256 : ko + (h-Q_HEADS)*256;
      T value = destination[d];
      int other = d < 32 ? d+32 : d-32;
      float rotated = d < 64 ? float(destination[other]) * (d < 32 ? -1.f : 1.f) : 0.f;
      // Every thread reads its rotary partner before any thread writes.
      __syncthreads();
      if (d < 64) {
        T left = T(__fmul_rn(float(value), float(rope[d])));
        T right = T(__fmul_rn(rotated, float(rope[64+d])));
        value = T(__fadd_rn(float(left), float(right)));
      }
      destination[d] = value;
      if (h < Q_HEADS) gate[h*256+d] = qg[h*512+256+d];
      __syncthreads();
    }
    return;
  }
  for (int i = threadIdx.x; i < (Q_HEADS + KV_HEADS) * 256; i += blockDim.x) {
    int h = i / 256, d = i % 256;
    T value = normalized[i];
    if (d < 64) {
      int other = d < 32 ? d + 32 : d - 32;
      float rotated = float(normalized[h * 256 + other]) * (d < 32 ? -1.f : 1.f);
      // HF rounds each product before adding; do not contract into an FMA.
      T left = T(__fmul_rn(float(value), float(rope[d])));
      T right = T(__fmul_rn(rotated, float(rope[64 + d])));
      value = T(__fadd_rn(float(left), float(right)));
    }
    if (h < Q_HEADS) {
      qo[i] = value;
      gate[i] = qg[h * 512 + 256 + d];
    } else {
      ko[(h - Q_HEADS) * 256 + d] = value;
    }
  }
  __syncthreads();
}

template <typename T, int CAPACITY, int KV_HEADS=4>
__device__ __forceinline__ void qwen35_kv_store_task(
    void const *key, void const *value, void const *position,
    void *key_cache, void *value_cache) {
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= CAPACITY) return;
  auto k = static_cast<T const *>(key);
  auto v = static_cast<T const *>(value);
  auto kc = static_cast<T *>(key_cache);
  auto vc = static_cast<T *>(value_cache);
  for (int i = threadIdx.x; i < KV_HEADS * 256; i += blockDim.x) {
    kc[pos * KV_HEADS * 256 + i] = k[i];
    vc[pos * KV_HEADS * 256 + i] = v[i];
  }
  __syncthreads();
}

// Isolated value product, also used by the focused GPU numerical-path test.
template <typename T, int CAPACITY, bool FORCE_TENSOR=false, int MMA_K=16, int RESIDUE_TILE=0, int KV_HEADS=4, int Q_PER_KV=6>
__device__ __forceinline__ void qwen35_value_task(
    void const *probabilities, void const *gate, void const *position,
    void const *value_cache, void const *head_id, void *gated_output,
    int full_length=0, int rows=1) {
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= CAPACITY) return;
  int kv_head = int(*static_cast<int64_t const *>(head_id)) / Q_PER_KV;
  auto p = static_cast<T const *>(probabilities);
  auto g = static_cast<T const *>(gate);
  auto v = static_cast<T const *>(value_cache);
  auto y = static_cast<T *>(gated_output);
  bool blackwell_wmma = false;
  int blackwell_parts = 0, blackwell_chunk = 0;
  bool blackwell_sequential_reduce = false;
  bool blackwell_two_level = false;
  bool blackwell_wave64 = false;
#if __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  if constexpr (!FORCE_TENSOR && KV_HEADS == 4 && Q_PER_KV == 6) {
    // Pinned cuBLAS 12.8 selects its align2 WMMA kernel at these lengths
    // for [B,24,1,K] @ [B,24,K,256]. Profiled over every K=1..1024 at
    // B=1,4,32; its 64-element K tile consumes the partial tile first.
    // Other lengths retain their existing path. This is backend dispatch,
    // independent of input values; see profile_blackwell_values_gpu.py.
    int length = int(pos) + 1;
    blackwell_wmma = length == 60 || length == 62 || length == 66 ||
                     length == 126 || length == 130 || length == 132 ||
                     length == 134;
    // Native short GEMVX changes at batch 16. The interleaved two-way
    // schedule was checked at every B=1..32; only B<16 uses it at K=11..38.
    // The later two-way ranges are common to the tested batch sizes.
    bool two_way = (rows < 16 && length >= 11 && length <= 38 && length % 8) ||
                   (length >= 48 && length <= 80 && length % 8 == 0) ||
                   (length >= 72 && length <= 124 && length % 2 == 0 && length % 8);
    if (rows == 8 && (length == 6 || length == 7 ||
                      length == 9 || length == 10)) {
      // Native short B8 GEMV2N accumulates these four lengths serially.
      // The previous NSP fallback missed BF16 ties in 100 input sets.
      blackwell_parts = 1; blackwell_chunk = 1;
    } else if (length <= 40 && length % 8 == 0) {
      // Short aligned inputs use GEMV too: NSP's 32 four-element lanes
      // at B=1/2, two contiguous GEMV2N halves at B=3..7, and a serial
      // GEMV2N accumulator at B>=8. All five lengths were profiled and
      // compared bitwise at every B=1..32, plus saved real layer inputs.
      blackwell_parts = rows < 3 ? 32 : rows < 8 ? 2 : 1;
      blackwell_chunk = rows < 3 ? 4 : (length + blackwell_parts - 1) / blackwell_parts;
    } else if (rows == 8 && length >= 39 && length <= 58 &&
               length % 8 != 0) {
      // Pinned B8 GEMVX uses four contiguous chunks for odd K and two
      // contiguous chunks for even K here. Each length matched 100 native
      // BF16 inputs; K40/48/56 keep their independently measured paths.
      blackwell_parts = length & 1 ? 4 : 2;
      blackwell_chunk = (length + blackwell_parts - 1) / blackwell_parts;
    } else if (rows == 8 &&
               ((length >= 69 && length <= 119 && (length & 1)) ||
                (length >= 88 && length <= 128 && length % 8 == 0))) {
      // Native B8 GEMVX uses four contiguous FP32 chunks at these odd
      // lengths and aligned K=88..128. The previous path missed BF16
      // values at every listed aligned length and most odd lengths;
      // each chosen length matched 100 native BF16 input sets.
      blackwell_parts = 4;
      blackwell_chunk = (length + 3) / 4;
    } else if (two_way) {
      blackwell_parts = 2; blackwell_chunk = 1;
    } else if (((length & 1) && length >= 59 && length <= 67) ||
               length == 68 || length == 70) {
      blackwell_parts = 4; blackwell_chunk = (length + 3) / 4;
    } else if ((length & 1) && length >= 121 && length <= 135) {
      blackwell_parts = 8; blackwell_chunk = 1;
    } else if (__CUDA_ARCH__ == 1000 && (rows == 4 || rows == 8) && length == 136) {
      // Native B4/B8 BF16 P×V dispatches to GEMVX at K=136. Four contiguous
      // FP32 chunks matched all 16 saved Qwen3.8 attention layers bitwise at
      // both batch sizes; the aligned WMMA path diverged at this decode step.
      blackwell_parts = 4;
      blackwell_chunk = (length + 3) / 4;
    } else if (__CUDA_ARCH__ == 1000 && (rows == 4 || rows == 8) && length == 137) {
      // Native B4/B8 GEMVX interleaves four FP32 accumulators at K=137.
      // Replaying each batch's first divergent token on its 16 real attention
      // layers matched the native P×V output bitwise with this schedule.
      blackwell_parts = 4;
      blackwell_chunk = 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 8 &&
               length >= 138 && length <= 191) {
      // Native B8 GEMVX P×V at K=138..191 was profiled and matched bitwise
      // across varied BF16 operands. Aligned multiples of eight use four
      // contiguous FP32 chunks; K=186,188,190 use eight interleaved lanes;
      // the remaining lengths use four interleaved lanes. K=140 and K=141
      // also matched all 16 saved Qwen3.8 attention layers on real inputs.
      if (length % 8 == 0) {
        blackwell_parts = 4;
        blackwell_chunk = (length + 3) / 4;
      } else {
        blackwell_parts = length >= 186 && length % 2 == 0 ? 8 : 4;
        blackwell_chunk = 1;
      }
    } else if (__CUDA_ARCH__ == 1000 && rows == 8 &&
               length >= 193 && length <= 271 && length % 8 != 0) {
      // B8 Qwen3.8 native P×V switches to four interleaved lanes for odd
      // K, eight interleaved lanes for even K=194..216, then eight
      // contiguous ceil(K/8) chunks with serial final addition for even
      // K=218..270. The aligned lengths retain the tensor-core path.
      blackwell_parts = length % 2 == 0 && length >= 194 ? 8 : 4;
      blackwell_sequential_reduce = length >= 218 && length % 2 == 0;
      blackwell_chunk = blackwell_sequential_reduce ? (length + 7) / 8 : 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 8 &&
               length >= 1020 && length <= 1023) {
      // B8 native GEMV2N uses two 64-token subchunks within each of
      // eight 128-token chunks at even K. Odd K uses eight interleaved
      // FP32 lanes. This is the same boundary observed for MoE B16.
      blackwell_parts = 8;
      blackwell_chunk = length % 2 == 0 ? 128 : 1;
      blackwell_two_level = length % 2 == 0;
    } else if (__CUDA_ARCH__ == 1000 && rows == 8 &&
               length >= 274 && length <= 310 &&
               length % 8 != 0 && length % 2 == 0) {
      // Native B8 GEMV2N keeps eight contiguous ceil(K/8) chunks with
      // serial final addition through this even, nonaligned interval.
      // K274, K276, and K310 matched fifty dense-probability inputs each.
      blackwell_parts = 8;
      blackwell_chunk = (length + 7) / 8;
      blackwell_sequential_reduce = true;
    } else if (__CUDA_ARCH__ == 1000 && rows == 8 &&
               length >= 273 && length <= 311 && (length & 1)) {
      // B8 GEMVX changes from four to eight interleaved accumulators
      // after K=288. Dense-probability BF16 inputs exposed ties that
      // the earlier sixteen-lane schedule missed at odd K289..309.
      blackwell_parts = length <= 287 ? 4 : 8;
      blackwell_chunk = 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 8 &&
               length >= 313 && length <= 1019 && length % 8 != 0) {
      // Dense B8 GEMVX uses eight interleaved lanes for odd K. Even K
      // uses GEMV2N's eight contiguous ceil(K/8) chunks, with 64-term
      // tiles once the chunks exceed that width. Dense-probability inputs
      // exposed BF16 ties missed by the earlier sixteen-lane path.
      blackwell_parts = 8;
      blackwell_chunk = (length & 1) ? 1 : (length + 7) / 8;
      blackwell_wave64 = (length & 1) == 0;
    } else if (__CUDA_ARCH__ == 1000 && rows == 16 &&
               length >= 65 && length <= 160 &&
               (((length & 1) && length <= 119 && length >= 69) ||
                (length >= 88 && length % 8 == 0) ||
                (length >= 137 && length <= 159 && length % 8 != 0))) {
      // Native B16 P×V uses four contiguous chunks through K=136 at
      // these lengths, then four interleaved lanes at K=137..159.
      // Compared at every K=1..160 with twenty varied BF16 input sets.
      blackwell_parts = 4;
      blackwell_chunk = length >= 137 && length % 8 != 0 ? 1 : (length + 3) / 4;
    } else if (__CUDA_ARCH__ == 1000 && rows == 16 &&
               length >= 161 && length <= 185) {
      // B16 uses four interleaved lanes at nonaligned K and four
      // contiguous chunks at K=168/176/184. Each length matched fifty
      // native BF16 input sets, including the former K=168 token failure.
      blackwell_parts = 4;
      blackwell_chunk = length % 8 == 0 ? (length + 3) / 4 : 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 16 &&
               length >= 186 && length <= 191) {
      // B16 GEMV2N at K=186/188/190 is equivalent to eight interleaved
      // FP32 lanes; adjacent odd lengths retain four. All six lengths
      // matched twenty native BF16 input sets.
      blackwell_parts = length % 2 == 0 ? 8 : 4;
      blackwell_chunk = 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 16 &&
               length >= 193 && length <= 1019 && length % 8 != 0) {
      // Native dense B16 GEMVX uses four interleaved FP32 lanes for odd
      // K193..287, eight for later odd K. Even nonaligned K194..214 uses
      // eight lanes, K218..310 two lanes, then eight from K314 onward.
      // All 724 nonaligned lengths matched native BF16 P×V in the dense
      // probability sweep (five independent inputs at each length).
      blackwell_parts = (length & 1) ? (length <= 287 ? 4 : 8) :
                        (length <= 214 || length >= 314 ? 8 : 2);
      blackwell_chunk = 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 16 &&
               length >= 1020 && length <= 1023) {
      // Near capacity, B16 native P×V reduces eight interleaved lanes at
      // each K=1020..1023. All four matched twenty BF16 input sets.
      blackwell_parts = 8;
      blackwell_chunk = 1;
    } else if (__CUDA_ARCH__ == 1000 && rows == 32 &&
               length >= 160 && length <= 184 && length % 8 == 0) {
      // Pinned B32 cuBLAS uses GEMVX (not Tensor Cores) at K=160,168,176,184.
      // Its four contiguous FP32 chunks match all 16 saved attention layers
      // at K=160 and varied B32 operands at each of the four lengths.
      blackwell_parts = 4;
      blackwell_chunk = (length + 3) / 4;
    } else if (__CUDA_ARCH__ == 1000 && rows == 32 &&
               length >= 136 && (length % 8) &&
               !((length & 1) == 0 && length >= 218 && length <= 310)) {
      // The B32 GEMVX kernel interleaves its FP32 accumulators across K.
      // The tested decode ranges use four lanes for odd K and for short
      // even K, then eight lanes for nonaligned even K from K=184 onward.
      // K=218..310 even uses a different native schedule; keep its prior
      // fallback until that schedule has been isolated and validated.
      // Keep the BF16 product and gate rounding below unchanged.
      blackwell_parts = ((length & 1) || length < 184) ? 4 : 8;
      blackwell_chunk = 1;
    }
  } else if constexpr (!FORCE_TENSOR && KV_HEADS == 2 && Q_PER_KV == 8) {
    int length = int(pos) + 1;
    // The pinned B8 cuBLAS switches to its align2 tensor-core reduction at
    // these seven unaligned lengths. No tested FP32 GEMV reduction order
    // reproduces its BF16 results; use the same residue-first WMMA schedule
    // as the Qwen3.8 attention path.
    blackwell_wmma = ((rows == 8 &&
                       (length == 60 || length == 62 || length == 66 ||
                        length == 126 || length == 130 || length == 132 ||
                        length == 134)) ||
                      (rows == 16 &&
                       (length == 60 || length == 62 || length == 66 ||
                        length == 126 || length == 130 || length == 132 ||
                        length == 134)));
    // B1 GEMVX interleaves its FP32 accumulators at short unaligned lengths:
    // sixteen lanes for K=17..31 and thirty-two for K=33..63. The K=22
    // layer-35 HF operands expose a BF16 tie with the old 32-by-4 schedule.
    // The unaligned intervals matched native cuBLAS over 100 BF16 inputs
    // per length, including K=49..63 where the previous fallback diverged.
    if (rows == 1 && length >= 17 && length <= 63 && length % 8 != 0) {
      blackwell_parts = length < 32 ? 16 : 32;
      blackwell_chunk = 1;
    }
    // Native B1 GEMVX retains 32 interleaved accumulators at unaligned
    // K=65..143. All such lengths matched 100 pinned cuBLAS BF16 inputs;
    // the previous 32-by-4 fallback differs already at K=65.
    if (rows == 1 && length >= 65 && length <= 143 && length % 8 != 0) {
      blackwell_parts = 32;
      blackwell_chunk = 1;
    }
    // Pinned B1 cuBLAS uses gemvNSP at aligned K=8..40. Its 32 lanes
    // consume four contiguous BF16 products each; real layer-7 K=16 and
    // 20 independent BF16 inputs matched this schedule bitwise.
    if (rows == 1 && length <= 40 && length % 8 == 0) {
      blackwell_parts = 32;
      blackwell_chunk = 4;
    }
    // Pinned Qwen3.6 B1 cuBLAS selects GEMVX for aligned K=48..80.
    // Its two interleaved accumulators matched real K=64/72/80 operands
    // and 20 BF16 random seeds at each length; WMMA misses BF16 ties.
    if (rows == 1 && length >= 48 && length <= 80 && length % 8 == 0) {
      blackwell_parts = 2;
      blackwell_chunk = 1;
    }
    // At aligned K=88..144, native cuBLAS changes GEMVX dispatch. Four
    // contiguous FP32 chunks matched 100 BF16 inputs at each length;
    // the previous WMMA path differed at every tested aligned length.
    if (rows == 1 && length >= 88 && length <= 144 && length % 8 == 0) {
      blackwell_parts = 4;
      blackwell_chunk = (length + 3) / 4;
    }
    // Qwen3.6 B8 changes native GEMV dispatch through K=31. Serial
    // GEMV2N at K=1..7/9..10, 32 four-element NSP lanes at K=8,
    // two interleaved GEMVX lanes at K=11..15/17..23/25..31, and two
    // contiguous GEMV2N halves at K=16/24 each matched 100 pinned
    // cuBLAS BF16 inputs per length.
    if (rows == 8 && length <= 31) {
      if (length == 8) {
        blackwell_parts = 32; blackwell_chunk = 4;
      } else if (length == 16 || length == 24) {
        blackwell_parts = 2; blackwell_chunk = length / 2;
      } else if (length >= 11) {
        blackwell_parts = 2; blackwell_chunk = 1;
      } else {
        blackwell_parts = 1; blackwell_chunk = 1;
      }
    }
    // B8 [16,1,K] x [16,K,256] changes its cuBLAS GEMV dispatch at these
    // cache lengths. The selected schedules matched 100 varied BF16 inputs
    // per length against the pinned native matmul. The seven WMMA lengths
    // above were excluded from the FP32 GEMV sweep.
    if (rows == 8 && length >= 32 && length <= 143 && !blackwell_wmma) {
      if (length == 32 || length == 40 ||
          (length >= 40 && length <= 58 &&
                           (length & 1) == 0 && length % 8 != 0)) {
        blackwell_parts = 2;
        blackwell_chunk = (length + 1) / 2;
      } else if ((length >= 33 && length <= 38) ||
                 length == 48 || length == 56 || length == 64 ||
                 length == 72 || length == 80 ||
                 (length >= 72 && length <= 124 &&
                  (length & 1) == 0 && length % 8 != 0)) {
        blackwell_parts = 2;
        blackwell_chunk = 1;
      } else if (length >= 121 && length <= 135 && (length & 1)) {
        blackwell_parts = 8;
        blackwell_chunk = 1;
      } else if (length >= 137) {
        blackwell_parts = 4;
        blackwell_chunk = 1;
      } else {
        blackwell_parts = 4;
        blackwell_chunk = (length + 3) / 4;
      }
    }
    // Pinned B200 cuBLAS changes the B8 Qwen3.6 P×V GEMV reduction beyond
    // K=143. Compare the production task against the native BF16 matmul, not
    // an algebraically equivalent FP32 sum: midpoint rounding is observable
    // in the following GDN state. Aligned K=192,200,...,264 retains WMMA.
    if (rows == 8 && length >= 144 && length <= 271 && length % 8 != 0) {
      blackwell_parts = length >= 186 && length % 2 == 0 ? 8 : 4;
      blackwell_sequential_reduce = length >= 218 && length % 2 == 0;
      // K=218..270 even uses eight contiguous ceil(K/8) chunks before the
      // serial final sum; its diagnostic mode is 4204, not interleaved 8.
      blackwell_chunk = blackwell_sequential_reduce ? (length + 7) / 8 : 1;
    } else if (rows == 8 && length >= 144 && length <= 184 && length % 8 == 0) {
      blackwell_parts = 4;
      blackwell_chunk = (length + 3) / 4;
    }
    // B16 selects different native GEMV2N/GEMVX reductions at K<=80.
    // Each branch matched the pinned BF16 matmul over ten independent
    // random input sets per length; K=60/62/66 use the align2 WMMA path.
    if (rows == 16 && length <= 80 && !blackwell_wmma) {
      if (length <= 10 || length == 16 || length == 24 ||
          length == 32 || length == 40) {
        blackwell_parts = 1;
        blackwell_chunk = 1;
      } else if (length <= 38 || length == 48 || length == 56 ||
                 length == 64 || length == 72 || length == 74 ||
                 length == 76 || length == 78 || length == 80) {
        blackwell_parts = 2;
        blackwell_chunk = 1;
      } else if (length == 42 || length == 44 || length == 46 ||
                 length == 50 || length == 52 || length == 54 ||
                 length == 58) {
        blackwell_parts = 2;
        blackwell_chunk = (length + 1) / 2;
      } else {
        blackwell_parts = 4;
        blackwell_chunk = (length + 3) / 4;
      }
    }
    // B16 GEMVX continues to change its FP32 reduction with cache length.
    // Each selected schedule matched the pinned BF16 matmul over ten varied
    // input sets per length. K=126/130/132/134 use align2 WMMA above.
    if (rows == 16 && length >= 81 && length <= 160 && !blackwell_wmma) {
      if (length <= 120) {
        if ((length & 1) || length % 8 == 0) {
          blackwell_parts = 4;
          blackwell_chunk = (length + 3) / 4;
        } else {
          blackwell_parts = 2;
          blackwell_chunk = 1;
        }
      } else if (length <= 135) {
        if (length % 8 == 0) {
          blackwell_parts = 4;
          blackwell_chunk = (length + 3) / 4;
        } else if (length & 1) {
          blackwell_parts = 8;
          blackwell_chunk = 1;
        } else {
          blackwell_parts = 2;
          blackwell_chunk = 1;
        }
      } else if (length % 8 == 0) {
        blackwell_parts = 4;
        blackwell_chunk = (length + 3) / 4;
      } else {
        blackwell_parts = 4;
        blackwell_chunk = 1;
      }
    }
    if (rows == 16 && length >= 161 && length <= 191) {
      // Native B16 MoE P×V uses the same contiguous/interleaved transition
      // as the dense 24-head path at K161..191. Every length matched
      // thirty BF16 input sets; even K186/188/190 uses eight lanes.
      blackwell_parts = length >= 186 && length % 2 == 0 ? 8 : 4;
      blackwell_chunk = length < 186 && length % 8 == 0 ?
                            (length + 3) / 4 : 1;
    }
    if (rows == 16 && length >= 193 && length <= 1019 &&
        length % 8 != 0 && ((length & 1) || length <= 214)) {
      // Native B16 GEMVX uses four interleaved lanes for odd K193..287,
      // eight for odd K289..1019 and even K194..214. Each length was
      // compared with native BF16 P×V. The remaining even nonaligned
      // lengths use a separate GEMV2N layout.
      blackwell_parts = (length & 1) && length <= 287 ? 4 : 8;
      blackwell_chunk = 1;
    }
    if (((rows == 16 && length >= 218 && length <= 1022) ||
         (rows == 8 && length >= 274 && length <= 310)) &&
        length % 8 != 0 && length % 2 == 0) {
      // cuBLAS 12.8's B200 GEMV2N forms eight contiguous ceil(K/8)
      // chunks. Each chunk accumulates at most 64 products before adding
      // the next subchunk; the eight chunk sums are then added in order.
      // This matched native BF16 P×V at every tested B16 even nonaligned
      // K=218..1022, including dense-probability operands.
      blackwell_parts = 8;
      blackwell_chunk = (length + 7) / 8;
      blackwell_wave64 = true;
    }
    if (rows == 8 && length >= 273 && length <= 311 &&
        (length & 1)) {
      // Native B8 GEMVX changes from four to sixteen interleaved lanes
      // after K287; these odd lengths matched dense-probability inputs.
      blackwell_parts = length <= 287 ? 4 : 16;
      blackwell_chunk = 1;
    }
    if (rows == 8 && length >= 313 && length <= 1019 &&
        length % 8 != 0) {
      // The MoE [16,1,K] native GEMVX dispatch uses sixteen interleaved
      // FP32 accumulators here. Its head layout differs from dense B8.
      blackwell_parts = 16;
      blackwell_chunk = 1;
    }
    // At the 1020-token boundary, the pinned native GEMVX reductions use
    // interleaved lanes. B16 even K uses the GEMV2N schedule above.
    if (rows == 8 && length >= 1020 && length <= 1023) {
      blackwell_parts = 16;
      blackwell_chunk = 1;
    } else if (rows == 16 && (length == 1021 || length == 1023)) {
      blackwell_parts = 8;
      blackwell_chunk = 1;
    }
  }
#endif
  // The pinned H100 cuBLAS selects GEMV for an unaligned reduction dimension
  // and Tensor Cores for aligned BF16 inputs. Preserve the BF16 product boundary
  // before multiplying by the independently rounded BF16 sigmoid gate.
  if (!FORCE_TENSOR && !blackwell_wmma && ((pos + 1) % 8 != 0 || blackwell_parts)) {
    int d = threadIdx.x;
    float partial[32] = {};
    // Pinned cuBLAS 12.8 / H100 PCIe: gemvNSP uses 32 lanes of four
    // consecutive K elements through length 425; gemvx then uses 16
    // contiguous ceil(length/16) chunks. Both reduce with descending offsets.
    // These are measured backend choices for [8,1,K] @ [8,K,256], not a
    // portable guarantee about another cuBLAS version or GPU's dispatch.
    int parts = pos < 425 ? 32 : 16;
    int chunk = pos < 425 ? 4 : (int(pos) + 16) / 16;
    if (blackwell_parts) { parts = blackwell_parts; chunk = blackwell_chunk; }
    if (blackwell_wave64) {
      int length = int(pos) + 1;
      for (int lane = 0; lane < 8; ++lane) {
        int end = min((lane + 1) * chunk, length);
        for (int begin = lane * chunk; begin < end; begin += 64) {
          float tile_sum = 0.f;
          for (int t = begin; t < min(begin + 64, end); ++t)
            tile_sum = __fmaf_rn(float(p[t]),
                float(v[t * (KV_HEADS * 256) + kv_head * 256 + d]), tile_sum);
          partial[lane] = __fadd_rn(partial[lane], tile_sum);
        }
      }
      for (int lane = 1; lane < 8; ++lane)
        partial[0] = __fadd_rn(partial[0], partial[lane]);
    } else if (blackwell_two_level) {
      // Eight 128-token chunks, each split into two contiguous 64-token
      // accumulators. Combine each pair, then add the eight chunks in
      // order. This reproduces B16 native GEMV2N at K1020 and K1022.
      for (int t = 0; t <= pos; ++t) {
        int lane = t / 128, half = (t % 128) / 64;
        int index = 2 * lane + half;
        partial[index] = __fmaf_rn(float(p[t]),
            float(v[t * (KV_HEADS * 256) + kv_head * 256 + d]), partial[index]);
      }
      for (int lane = 0; lane < 8; ++lane)
        partial[lane] = __fadd_rn(partial[2 * lane], partial[2 * lane + 1]);
      for (int lane = 1; lane < 8; ++lane)
        partial[0] = __fadd_rn(partial[0], partial[lane]);
    } else {
      for (int t = 0; t <= pos; ++t) {
        int lane = (t / chunk) % parts;
        partial[lane] = __fmaf_rn(float(p[t]),
            float(v[t * (KV_HEADS * 256) + kv_head * 256 + d]), partial[lane]);
      }
      if (blackwell_sequential_reduce) {
        for (int i = 1; i < parts; ++i)
          partial[0] = __fadd_rn(partial[0], partial[i]);
      } else {
        #pragma unroll
        for (int offset = parts / 2; offset; offset >>= 1)
          for (int i = 0; i < offset; ++i)
            partial[i] = __fadd_rn(partial[i], partial[i + offset]);
      }
    }
    float sum = partial[0];
    T sigmoid = T(1.f / (1.f + expf(-float(g[d]))));
    y[d] = T(__fmul_rn(float(T(sum)), float(sigmoid)));
  } else {
    // Register fragments avoid adding 12.5 KiB of static shared memory to
    // every persistent worker. Only output row zero is live. Each warp owns
    // 32 channels, in four m16n8k16 tiles; never read unused cache slots.
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int group = lane / 4, pair = (lane % 4) * 2;
    for (int part = 0; part < 4; ++part) {
      int channel = warp * 32 + part * 8;
      float c[4] = {};
      // Unaligned WMMA paths consume the partial K tile first. Prefill
      // supplies its tile width; Blackwell decode uses the 64-element tile
      // above. Zero keeps the ordinary contiguous decode schedule.
      int tile = blackwell_wmma ? 64 : RESIDUE_TILE;
      int length = blackwell_wmma ? int(pos) + 1 : full_length;
      int residue=tile ? length%tile : 0;
      int padded=(residue+MMA_K-1)/MMA_K*MMA_K;
      int span=tile ? padded+length-residue : int(pos)+1;
      for (int step = 0; step < span; step += MMA_K) {
        int base=step<padded ? step : step-padded+residue;
        int limit=step<padded ? residue : span;
        uint32_t a[4] = {}, b[2] = {};
        #pragma unroll
        for (int half = 0; half < MMA_K/8; ++half) {
          int t = base + pair + half * 8;
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            if (t + j <= pos && t+j<limit) {
              uint32_t pv = __bfloat16_as_ushort(__float2bfloat16(float(p[t+j])));
              if (group == 0) a[half * 2] |= pv << (16 * j);
              uint32_t vv = __bfloat16_as_ushort(__float2bfloat16(
                  float(v[(t+j) * (KV_HEADS * 256) + kv_head * 256 + channel + group])));
              b[half] |= vv << (16 * j);
            }
          }
        }
        if constexpr(MMA_K==8) {
          asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
              : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
              : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        } else asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
      if (group == 0) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          int d = channel + pair + j;
          T sigmoid = T(1.f / (1.f + expf(-float(g[d]))));
          y[d] = T(__fmul_rn(float(T(c[j])), float(sigmoid)));
        }
      }
    }
  }
  __syncthreads();
}

template <typename T, int CAPACITY, bool FORCE_TENSOR=false, int MMA_K=16, int KV_HEADS=4, int Q_PER_KV=6>
__device__ __forceinline__ void qwen35_scores_task(
    void const *query, void const *position, void const *key_cache,
    void const *head_id, float *scores, int rows=1) {
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= CAPACITY) return;
  int kv_head = int(*static_cast<int64_t const *>(head_id)) / Q_PER_KV;
  auto q = static_cast<T const *>(query);
  auto k = static_cast<T const *>(key_cache);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
#if __CUDA_ARCH__ == 1000
  if (FORCE_TENSOR || (KV_HEADS == 4 && Q_PER_KV == 6 &&
                       (pos + 1) % 8 == 0 &&
                       !(rows == 8 && pos + 1 <= 64)) ||
      (KV_HEADS == 2 && Q_PER_KV == 8 &&
       ((rows == 8 && (pos + 1) >= 104) ||
        (rows == 16 && (pos + 1) >= 40)) &&
       (pos + 1) % 8 == 0)) {
#else
  if (FORCE_TENSOR || (pos + 1) % 8 == 0) {
#endif
    // B8 Qwen3.6 aligned K>=104 selects MMA16 in the pinned cuBLAS path.
    // The GEMV fallback misses BF16 ties; isolated tests cover the shorter
    // range and sampled aligned lengths through K=1016.
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int group = lane / 4, pair = (lane % 4) * 2;
    for (int token = warp * 8; token <= pos; token += 64) {
      float c[4] = {};
      for (int base = 0; base < 256; base += MMA_K) {
        uint32_t a[4] = {}, b[2] = {};
        #pragma unroll
        for (int half = 0; half < MMA_K/8; ++half) {
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            int d = base + pair + half * 8 + j;
            if (group == 0) a[half*2] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(q[d])))) << (16*j);
            if (token + group <= pos) b[half] |= uint32_t(__bfloat16_as_ushort(
                __float2bfloat16(float(k[(token+group)*(KV_HEADS*256)+kv_head*256+d])))) << (16*j);
          }
        }
        if constexpr(MMA_K==8) {
          asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
              : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
              : "r"(a[0]), "r"(a[1]), "r"(b[0]));
        } else asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      }
      if (group == 0) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)
          if (token + pair + j <= pos)
            scores[token+pair+j] = float(T(__fmul_rn(float(T(c[j])), .0625f)));
      }
    }
    __syncthreads();
    return;
  }
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    // Measured Torch 2.7/cuBLAS 12.8 H100 PCIe dispatch for [8,1,256]
    // @ [8,256,L]. These choices depend on shape, library and GPU; they are
    // not a generic cuBLAS contract. GEMVX reduces a tree, while GEMV2T
    // adds 16 strided partials serially (confirmed in the native instructions).
    int length = int(pos) + 1;
    bool gemv2t = (length >= 388 && length <= 416) || length >= 444;
    bool strided = length >= 388;
    int parts = length >= 57 && length < 140 ? 32 : 16;
    float partial[32] = {};
    for (int d = 0; d < 256; ++d) {
      int lane = strided ? d % 16 : d / (256 / parts);
      partial[lane] = __fmaf_rn(float(q[d]),
          float(k[t * (KV_HEADS * 256) + kv_head * 256 + d]), partial[lane]);
    }
    if (gemv2t) {
      for (int i = 1; i < 16; ++i) partial[0] = __fadd_rn(partial[0], partial[i]);
    } else {
      for (int offset = parts / 2; offset; offset >>= 1)
        for (int i = 0; i < offset; ++i)
          partial[i] = __fadd_rn(partial[i], partial[i + offset]);
    }
    scores[t] = float(T(__fmul_rn(float(T(partial[0])), 0.0625f)));
  }
#else
  // The pinned SM100 GEMVX paths use partial FP32 accumulators. A serial
  // 256-element dot crosses BF16 rounding boundaries on real Qwen3.6 K=65
  // operands even when its FP32 error is tiny.
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float dot = 0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    if constexpr ((KV_HEADS == 4 && Q_PER_KV == 6) ||
                  (KV_HEADS == 2 && Q_PER_KV == 8)) {
      // Small batches instead use contiguous reductions for short caches.
      // Measured across B=1..32 and every K=1..80: eight 32-element
      // chunks for K<=3; sixteen 16-element chunks through K=59 (B<4)
      // or K=31 (B=4..7). Qwen3.6 B1 changes reduction layout again at
      // K=60, 64, 65, 97, and the GEMV2T aligned range K=112..144.
      int length = int(pos) + 1;
      bool qwen38_b8 = KV_HEADS == 4 && rows == 8;
      bool qwen36_b1 = KV_HEADS == 2 && rows == 1;
      bool qwen36_b8 = KV_HEADS == 2 && rows == 8;
      bool qwen36_b16 = KV_HEADS == 2 && rows == 16;
      bool contiguous = (KV_HEADS == 4 && rows < 8 &&
                         length <= (rows < 4 ? 59 : 31)) ||
                        (qwen38_b8 && length <= 64 && length % 8 == 0) ||
                        (qwen36_b1 && (length <= 63 ||
                                       (length >= 65 && length <= 96) ||
                                       length == 104)) ||
                        (qwen36_b8 &&
                         (length <= 32 ||
                          (length <= 96 && length % 8 == 0))) ||
                        (qwen36_b16 && length <= 32 && length % 8 == 0);
      bool serial = qwen36_b1 && length >= 112 && length <= 144 &&
                    length % 8 == 0;
      // K=60..63 and K=65..96 use eight contiguous parts. K=64 and
      // unaligned K=97..143 use 32 interleaved parts (except K=104).
      // GEMV2T uses 16 interleaved parts with serial final addition.
      // These choices matched 100 BF16 inputs per length on the pinned B200.
      int parts = contiguous && length > 3 ? 16 : 8;
      if (qwen38_b8 && length <= 64) {
        // Pinned B8 cuBLAS uses GEMVX through aligned K=64, switching to
        // GEMM at aligned K=72. Eight interleaved parts matched every
        // unaligned K=1..31 over 100 CUDA BF16 inputs per length, including
        // the real K=9 layer-43 tie. Aligned K<=64 uses eight contiguous
        // parts; a real K=64 layer-3 score exposes the old WMMA BF16 tie.
        parts = 8;
      }
      if constexpr (KV_HEADS == 2) {
        if (qwen36_b8) {
          // Native B8 cuBLAS uses contiguous 16-way reductions through
          // K=31, except K=3 and aligned K=8..32, which use eight parts.
          // K=33..103 and unaligned K=105..143 use eight interleaved
          // parts; aligned K=40..96 uses eight contiguous parts. Aligned
          // K=104..136 takes the tensor-core path above. Each selected
          // mode matched 100 pinned CUDA BF16 inputs per cache length.
          parts = length == 3 || length % 8 == 0 ? 8 :
                  length <= 31 ? 16 : 8;
        } else if (qwen36_b1) {
          parts = serial ? 16 : contiguous ? (length >= 60 ? 8 : 16) :
                  (length == 64 || (length >= 97 && length <= 143) ? 32 : 16);
        } else if (qwen36_b16) {
          // Native B16 cuBLAS GEMVX uses eight FP32 lanes at K<=80:
          // interleaved for unaligned K, contiguous for K=8/16/24/32.
          // The aligned K>=40 tensor-core path is selected above.
          parts = 8;
        } else {
          parts = 16;
        }
      }
      float partial[32] = {};
      for (int d = 0; d < 256; ++d) {
        int lane = contiguous ? d / (256 / parts) : d % parts;
        partial[lane] = __fmaf_rn(float(q[d]),
            float(k[t*(KV_HEADS*256)+kv_head*256+d]), partial[lane]);
      }
      if (serial) {
        for (int lane = 1; lane < parts; ++lane)
          partial[0] = __fadd_rn(partial[0], partial[lane]);
      } else {
        for (int offset = parts / 2; offset; offset >>= 1)
          for (int lane = 0; lane < offset; ++lane)
            partial[lane] = __fadd_rn(partial[lane], partial[lane+offset]);
      }
      dot = partial[0];
    } else
#endif
    for (int d = 0; d < 256; ++d)
      dot = __fadd_rn(dot, __fmul_rn(float(q[d]), float(k[t*(KV_HEADS*256)+kv_head*256+d])));
    scores[t] = float(T(__fmul_rn(float(T(dot)), .0625f)));
  }
#endif
  __syncthreads();
}

template <typename T, int CAPACITY, bool FORCE_TENSOR=false, int KV_HEADS=4, int Q_PER_KV=6>
__device__ __forceinline__ void qwen35_attention_task(
    void const *query, void const *gate, void const *position,
    void const *key_cache, void const *value_cache, void const *head_id,
    void *gated_output, void *probabilities, int rows=1) {
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= CAPACITY) return;
  auto p = static_cast<T *>(probabilities);
  __shared__ float scores[CAPACITY], max_score, denominator;
  qwen35_scores_task<T, CAPACITY, FORCE_TENSOR, 16, KV_HEADS, Q_PER_KV>(query, position, key_cache, head_id, scores, rows);
  if (threadIdx.x == 0) {
    float peak = scores[0];
    for (int t = 1; t <= pos; ++t) peak = fmaxf(peak, scores[t]);
    max_score = peak;
  }
  __syncthreads();
  for (int t = threadIdx.x; t <= pos; t += blockDim.x)
    scores[t] = expf(scores[t] - max_score);
  __syncthreads();
  if (threadIdx.x < 32) {
    // Match PyTorch PersistentSoftmax: strided per-lane sums, then descending
    // XOR shuffles. A serial denominator differs at long-context BF16 ties.
    float sum = 0.f;
    for (int t = threadIdx.x; t <= pos; t += 32) sum = __fadd_rn(sum, scores[t]);
    for (int offset = 16; offset; offset >>= 1)
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    if (threadIdx.x == 0) denominator = sum;
  }
  __syncthreads();
  for (int t = threadIdx.x; t < CAPACITY; t += blockDim.x) {
    T probability = t <= pos ? T(scores[t] / denominator) : T(0.f);
    p[t] = probability;
    if (t <= pos) scores[t] = float(probability);
  }
  __syncthreads();
  qwen35_value_task<T, CAPACITY, FORCE_TENSOR, 16, 0, KV_HEADS, Q_PER_KV>(probabilities, gate, position,
                                   value_cache, head_id, gated_output, 0, rows);
  __syncthreads();
}
} // namespace kernel
