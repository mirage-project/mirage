// SPDX-License-Identifier: Apache-2.0
// Uniform/Gumbel transform adapted from vLLM PR #51367 at
// 6af68f63030a7b27f8c07fad5b6237dcb3964e5c.
// Copyright contributors to the vLLM project (uniform/Gumbel transform).
#pragma once
#include "mirage/persistent_kernel/serving_config.h"
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdint.h>

namespace mirage {
namespace serving {
__device__ inline uint32_t score_bits(float score) {
  uint32_t bits = __float_as_uint(score == 0.f ? 0.f : score);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
// Reverse token IDs make argmax prefer the lowest ID when scores tie.
__device__ inline uint64_t score_key(float score, uint32_t token) {
  return (uint64_t(score_bits(score)) << 32) | (0xffffffffu - token);
}
struct Sum {
  __device__ float operator()(float a, float b) const { return a + b; }
};

struct Maximum {
  __device__ uint64_t operator()(uint64_t a, uint64_t b) const { return a > b ? a : b; }
};

struct Draw {
  float score = -INFINITY, noise = 0;
  int token = 0;
};

// Compare score + temperature * noise without rounding a large common logit
// offset into the noise. This is equivalent to centered Gumbel-max, but needs
// neither a preliminary maximum nor a second vocabulary pass. Near
// cancellation, use FP64 to resolve the sign instead of depending on the
// reduction order.
struct ChooseDraw {
  float temperature;
  __device__ Draw operator()(Draw a, Draw b) const {
    if (!isfinite(a.score) || !isfinite(b.score)) {
      return score_key(a.score, a.token) >= score_key(b.score, b.token) ? a : b;
    }
    if (a.score == b.score) {
      if (a.noise != b.noise) {
        return a.noise > b.noise ? a : b;
      }
      return a.token < b.token ? a : b;
    }
    float delta = a.score - b.score;
    float shift = temperature * (b.noise - a.noise);
    float margin = delta - shift;
    // Eight FP32 epsilons conservatively cover the subtraction/product errors.
    if (fabsf(margin) > 0x1p-20f * (fabsf(delta) + fabsf(shift))) {
      return margin > 0 ? a : b;
    }
    double exact = (double(a.score) - b.score) + double(temperature) * (double(a.noise) - b.noise);
    if (exact != 0) {
      return exact > 0 ? a : b;
    }
    return a.token < b.token ? a : b;
  }
};

// Reserve the final 2 KiB of the existing 16 KiB shared slab for CUB.
constexpr int CUB_STORAGE_OFFSET_BYTES = 14 * 1024;
// Candidates use an int ID and a float score, plus an optional weight.
static_assert(sizeof(int) == sizeof(float));
constexpr int COMPACT_WORDS_PER_TOKEN = 2;
constexpr int COMPACT_WORDS_WITH_WEIGHT = 3;

template <int Threads, typename T, typename Op>
__device__ inline T cub_block_reduce(T value, Op op) {
  using Reduce = cub::BlockReduce<T, Threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
  struct Storage {
    typename Reduce::TempStorage reduce;
    T aggregate;
  };
  static_assert(CUB_STORAGE_OFFSET_BYTES + sizeof(Storage) <= SAMPLING_SHARED_BYTES);
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage = *reinterpret_cast<Storage *>(sampling_shared + CUB_STORAGE_OFFSET_BYTES);
  T result = Reduce(storage.reduce).Reduce(value, op);
  if (threadIdx.x == 0) {
    storage.aggregate = result;
  }
  __syncthreads();
  result = storage.aggregate;
  __syncthreads();
  return result;
}

template <typename T, typename Op>
__device__ inline T block_reduce(T value, float *, Op op) {
  return blockDim.x == 128 ? cub_block_reduce<128>(value, op) : cub_block_reduce<256>(value, op);
}

// Reconstruct (bits + 0.5) / 2^32 from two exact 16-bit conversions.
// Keeping all source bits is important at u -> 0, the winning Gumbel tail.
__device__ inline float sampling_uniform(uint32_t bits) {
  return float(bits >> 16) * 0x1p-16f + (float(bits & 0xffffu) + 0.5f) * 0x1p-32f;
}

__device__ inline float gumbel_from_uniform(float u) {
  // Degree-8 expansion of -log1p(-u). Unlike log(1-u), it preserves tiny u.
  // This is the polynomial and endpoint handling used by vLLM's FP32 path.
  float polynomial = 1.f / 8.f;
#pragma unroll
  for (int order = 7; order >= 1; --order) {
    polynomial = 1.f / order + u * polynomial;
  }
  float exponential = u < 0.25f ? u * polynomial : -logf(fmaxf(1.f - u, 0x1p-24f));
  return -logf(exponential);
}

// Philox subsequence = generation position, offset = token ID. A thread caches
// the most recent group of four: dense vector loads reuse all curand4 outputs;
// sparse candidate lists remain independent of traversal order and block size.
// The stream depends only on seed, position, and token, never the batch slot.
struct SamplingRng {
  uint64_t seed;
  uint32_t position;
  mutable uint32_t group = 0xffffffffu;
  mutable uint4 values{};
  __device__ SamplingRng(uint64_t seed, uint32_t position) : seed(seed), position(position) {}
  __device__ uint32_t bits(uint32_t token) const {
    if (group != token / 4) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, position, token & ~uint32_t(3), &state);
      values = curand4(&state);
      group = token / 4;
    }
    return (token & 3) == 0 ? values.x : (token & 3) == 1 ? values.y : (token & 3) == 2 ? values.z : values.w;
  }
  __device__ float gumbel(uint32_t token) const { return gumbel_from_uniform(sampling_uniform(bits(token))); }
};

// A thread loads adjacent logits as a vector; the tail or an unaligned row
// falls back to scalar loads. Warp coalescing and per-thread width are
// separate.
template <typename T, int Width>
struct alignas(Width * sizeof(T)) SampleVector {
  T values[Width];
};

template <typename T, int Width>
__device__ inline SampleVector<T, Width> load_vector(T const *values, int start, int size) {
  if (start + Width <= size && uintptr_t(values + start) % alignof(SampleVector<T, Width>) == 0) {
    return *reinterpret_cast<SampleVector<T, Width> const *>(values + start);
  }
  SampleVector<T, Width> result{};
#pragma unroll
  for (int i = 0; i < Width; ++i) {
    if (start + i < size) {
      result.values[i] = values[start + i];
    }
  }
  return result;
}

struct SamplingScratch {
  int *frequency_counts;
  uint32_t *seen_total;
  uint32_t *seen_generated;
  float *workspace;
  int *history_cursor;
  int bitset_words;

  __device__ SamplingScratch(float *scratch, int padded_vocab)
      : frequency_counts(reinterpret_cast<int *>(scratch)), bitset_words(sampling_bitset_words(padded_vocab)) {
    seen_total = reinterpret_cast<uint32_t *>(frequency_counts + sampling_frequency_words(padded_vocab));
    seen_generated = seen_total + bitset_words;
    workspace = reinterpret_cast<float *>(seen_generated + bitset_words);
    history_cursor = reinterpret_cast<int *>(workspace + SCRATCH_WORKSPACE);
  }
};

__device__ inline void prepare_history(SamplingScratch const &scratch, ServingConfig const *cfg,
    long long const *history, int history_len, int prompt_len, int generation_position, int vocab, float frequency,
    bool reuse_history) {
  // Reuse only private row state initialized by this request's first sample.
  // Disabling caching rebuilds the histograms for each generated token.
  bool cached = reuse_history && cfg->cache_history && generation_position > 0;
  int begin = cached ? *scratch.history_cursor : 0;
  if (!cached) {
    if (frequency != 0.f) {
      for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        scratch.frequency_counts[v] = 0;
      }
    }
    for (int v = threadIdx.x; v < scratch.bitset_words; v += blockDim.x) {
      scratch.seen_total[v] = 0;
      scratch.seen_generated[v] = 0;
    }
    __syncthreads();
  }
  for (int i = begin + threadIdx.x; i < history_len; i += blockDim.x) {
    long long token = history[i];
    if (token >= 0 && token < vocab) {
      atomicOr(scratch.seen_total + (token >> 5), uint32_t(1) << (token & 31));
      if (i >= prompt_len) {
        atomicOr(scratch.seen_generated + (token >> 5), uint32_t(1) << (token & 31));
        if (frequency != 0.f) {
          atomicAdd(scratch.frequency_counts + token, 1);
        }
      }
    }
  }
  // All threads must consume the old cursor before it is updated.
  __syncthreads();
  if (threadIdx.x == 0) {
    *scratch.history_cursor = history_len;
  }
}

template <typename T, bool Plain = false>
struct SamplingScoreReader {
  using Element = T;
  static constexpr bool IsPlain = Plain;
  T const *logits;
  ServingConfig const *cfg;
  int const *frequency_counts;
  uint32_t const *seen_total;
  uint32_t const *seen_generated;
  float repetition, frequency, presence;
  float center, temperature;
  int bias_count;
  uint64_t minimum_score_key = 0;

  __device__ float apply_bias(int v, float score) const {
    // Bias IDs are sorted by the host packer. Keep the small bias table in
    // ServingConfig instead of materializing a modified vocabulary row.
    if (bias_count) {
      int lo = 0, hi = bias_count;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cfg->biases[mid].token < v) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (lo < bias_count && cfg->biases[lo].token == v) {
        score += float(cfg->biases[lo].value);
      }
    }
    return score;
  }

  __device__ float apply_generated_penalties(int v, float score) const {
    // Most vocabulary IDs never occur in the generated history. Avoid a
    // dense count-array read unless its bit is set.
    if ((frequency != 0.f || presence != 0.f) && (seen_generated[v >> 5] & (uint32_t(1) << (v & 31)))) {
      if (frequency != 0.f) {
        score -= frequency * frequency_counts[v];
      }
      if (presence != 0.f) {
        score -= presence;
      }
    }
    return score;
  }

  __device__ float raw(int v, float score) const {
    if constexpr (Plain) {
      return isnan(score) ? -INFINITY : score;
    }
    score = apply_bias(v, score);
    if (repetition != 1.f && (seen_total[v >> 5] & (uint32_t(1) << (v & 31)))) {
      score = score > 0 ? score / repetition : score * repetition;
    }
    score = apply_generated_penalties(v, score);
    return isnan(score) ? -INFINITY : score;
  }
  __device__ float raw(int v) const { return raw(v, static_cast<float>(logits[v])); }
  __device__ float unmasked_scaled(int v, float logit) const { return (raw(v, logit) - center) / temperature; }
  __device__ float scaled(int v, float logit) const {
    float score = unmasked_scaled(v, logit);
    return score_key(score, v) >= minimum_score_key ? score : -INFINITY;
  }
  __device__ float operator()(int v) const { return scaled(v, static_cast<float>(logits[v])); }
};

// Both sampling paths use the same vocabulary traversal. The vector load is
// most valuable for plain unfiltered sampling, and also avoids a second logit
// load when penalties or biases are applied.
template <int Width = 16, typename T, typename Visit>
__device__ inline void scan_logits(T const *logits, int vocab, Visit visit) {
  for (int base = Width * threadIdx.x; base < vocab; base += Width * blockDim.x) {
    auto values = load_vector<T, Width>(logits, base, vocab);
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      int v = base + i;
      if (v < vocab) {
        visit(v, static_cast<float>(values.values[i]));
      }
    }
  }
}

// Adapted from FlashInfer DeviceSamplingFromProb and
// TopK/TopPSamplingFromProbKernel: d84335d8c54e0c8f9f7e22faa15c7e34a58abf94,
// Copyright (c) 2024 FlashInfer team. Apache-2.0:
// https://www.apache.org/licenses/LICENSE-2.0
__device__ inline float sampling_midpoint(float a, float b) {
  return __fmul_rn(__fadd_rn(a, b), .5f);
}

template <bool FindMaximum>
struct IndexReduce {
  __device__ int operator()(int a, int b) const { return FindMaximum ? max(a, b) : min(a, b); }
};

struct PivotStats {
  float mass0, mass1;
  int count0, count1;
  __device__ PivotStats operator+(PivotStats b) const {
    return {mass0 + b.mass0, mass1 + b.mass1, count0 + b.count0, count1 + b.count1};
  }
};

template <int Threads, typename Reader>
__device__ inline int sample_from_weights(Reader const &reader, int vocab, float lower_weight, float threshold,
    float *workspace, float const *cached_weights) {
  constexpr int Width = 16;
  using Scan = cub::BlockScan<float, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  static_assert(CUB_STORAGE_OFFSET_BYTES + sizeof(typename Scan::TempStorage) <= SAMPLING_SHARED_BYTES);
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(sampling_shared + CUB_STORAGE_OFFSET_BYTES);
  float scanned_mass = 0;
  int last_valid_token = -1;
  for (int base = 0; base < vocab; base += Threads * Width) {
    float weights[Width], cdf[Width];
    int start = base + threadIdx.x * Width;
    SampleVector<typename Reader::Element, Width> logits{};
    SampleVector<float, Width> cached{};
    if (cached_weights) {
      cached = load_vector<float, Width>(cached_weights, start, vocab);
    }
    if (!cached_weights) {
      logits = load_vector<typename Reader::Element, Width>(reader.logits, start, vocab);
    }
    float thread_mass = 0;
#pragma unroll
    for (int j = 0; j < Width; ++j) {
      int v = start + j;
      float score = v < vocab && !cached_weights ? reader.scaled(v, float(logits.values[j])) : -INFINITY;
      float weight = cached_weights ? cached.values[j] : expf(score);
      bool above = weight > lower_weight;
      weights[j] = v < vocab && above ? weight : 0.f;
      thread_mass += weights[j];
      if (weights[j] > 0) {
        last_valid_token = v;
      }
    }
    float mass = block_reduce(thread_mass, workspace, Sum{});
    if (scanned_mass + mass > threshold) {
      Scan(storage).InclusiveSum(weights, cdf);
      __syncthreads();
      int first_selected_token = vocab;
#pragma unroll
      for (int j = 0; j < Width; ++j) {
        if (weights[j] > 0 && scanned_mass + cdf[j] > threshold) {
          first_selected_token = min(first_selected_token, base + int(threadIdx.x) * Width + j);
        }
      }
      int token = block_reduce(first_selected_token, workspace, IndexReduce<false>{});
      if (token < vocab) {
        return token;
      }
      // Different reduction/scan summation orders can disagree at the endpoint.
    }
    scanned_mass += mass;
  }
  return block_reduce(last_valid_token, workspace, IndexReduce<true>{});
}

template <bool TopK, int Threads, typename Reader>
__device__ __noinline__ int dual_pivot_sample(Reader const &reader, int vocab, float target, int best_token,
    uint64_t seed, uint32_t position, float *workspace, float *cached_weights) {
  float remaining_mass = 0;
  scan_logits<4>(reader.logits, vocab, [&](int v, float value) {
    float weight = expf(reader.scaled(v, value));
    if (cached_weights) {
      cached_weights[v] = weight;
    }
    remaining_mass += weight;
  });
  remaining_mass = block_reduce(remaining_mass, workspace, Sum{});
  if constexpr (!TopK) {
    target *= remaining_mass;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(seed, position, 0, &state);
  float lower_weight = 0.f, upper_weight = 1.f;
  auto accepted = [&](float mass, int count) {
    if constexpr (TopK) {
      return count < int(target);
    } else {
      return mass < target;
    }
  };
  do {
    int token = sample_from_weights<Threads>(
        reader, vocab, lower_weight, curand_uniform(&state) * remaining_mass, workspace, cached_weights);
    if (token < 0) {
      return best_token;
    }
    float sampled_weight = cached_weights ? cached_weights[token] : expf(reader(token));
    float midpoint_weight = sampling_midpoint(sampled_weight, upper_weight);
    PivotStats stats{0, 0, 0, 0};
    auto accumulate = [&](int v, float value) {
      float weight = cached_weights ? value : expf(reader.scaled(v, value));
      if (weight > sampled_weight) {
        stats.mass0 += weight;
        ++stats.count0;
      }
      if (weight > midpoint_weight) {
        stats.mass1 += weight;
        ++stats.count1;
      }
    };
    if (cached_weights) {
      scan_logits(cached_weights, vocab, accumulate);
    } else {
      scan_logits(reader.logits, vocab, accumulate);
    }
    stats = block_reduce(stats, workspace, cuda::std::plus<PivotStats>{});
    if (accepted(stats.mass0, stats.count0)) {
      return token;
    }
    if (accepted(stats.mass1, stats.count1)) {
      // Sampled pivot rejected; midpoint accepted.
      lower_weight = sampled_weight;
      upper_weight = midpoint_weight;
      remaining_mass = stats.mass0;
    } else {
      // Both pivots rejected.
      lower_weight = midpoint_weight;
      remaining_mass = stats.mass1;
    }
  } while (lower_weight < upper_weight);
  return best_token;
}

struct CutoffResult {
  // Ordered bits of the minimum retained scaled score.
  uint32_t score_threshold;
  // Includes every token tied at the boundary.
  int retained_count;
};

// FlashInfer radix-select approach (topk.cuh, d84335d8): choose one byte
// per round. CUB builds histograms and scans; no token-ID tie rank or k cap.
// Replicate CUB histogram bins per warp; extra bins absorb masked/tail items.
template <int Threads, typename Reader>
__device__ __noinline__ CutoffResult radix_cutoff(Reader const &reader, int vocab, int k, float *) {
  constexpr int Width = 4;
  constexpr int RadixBits = 8;
  constexpr int RadixBins = 1 << RadixBits;
  constexpr int WarpBins = RadixBins + WARP_SIZE;
  constexpr int Warps = Threads / WARP_SIZE;
  using Histogram = cub::BlockHistogram<uint32_t, Threads, Width, WarpBins * Warps, cub::BLOCK_HISTO_ATOMIC>;
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto *histogram = reinterpret_cast<unsigned int *>(sampling_shared);
  auto &storage = *reinterpret_cast<typename Histogram::TempStorage *>(sampling_shared + CUB_STORAGE_OFFSET_BYTES);
  auto &scan = *reinterpret_cast<typename Scan::TempStorage *>(sampling_shared + CUB_STORAGE_OFFSET_BYTES);
  static_assert(WarpBins * Warps * sizeof(unsigned int) + 3 * sizeof(int) <= CUB_STORAGE_OFFSET_BYTES);
  static_assert(CUB_STORAGE_OFFSET_BYTES + sizeof(typename Histogram::TempStorage) <= SAMPLING_SHARED_BYTES);
  static_assert(CUB_STORAGE_OFFSET_BYTES + sizeof(typename Scan::TempStorage) <= SAMPLING_SHARED_BYTES);
  struct BucketSelection {
    int bin, higher_count, bucket_count;
  };
  auto &selected_bucket = *reinterpret_cast<BucketSelection *>(histogram + WarpBins * Warps);
  uint32_t score_prefix = 0, prefix_mask = 0;
  int remaining_rank = k, higher_count = 0;
  for (int shift = 24; shift >= 0; shift -= RadixBits) {
    Histogram(storage).InitHistogram(histogram);
    __syncthreads();
    for (int base = 0; base < vocab; base += Threads * Width) {
      int start = base + threadIdx.x * Width;
      auto values = load_vector<typename Reader::Element, Width>(reader.logits, start, vocab);
      // Masked items use per-lane bins outside the score bins.
      uint32_t thread_bins[Width];
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        int v = start + i;
        uint32_t key = v < vocab ? score_bits(reader.unmasked_scaled(v, float(values.values[i]))) : 0;
        uint32_t bin = v < vocab && (key & prefix_mask) == score_prefix ? (key >> shift) & (RadixBins - 1)
                                                                        : RadixBins + (threadIdx.x & (WARP_SIZE - 1));
        thread_bins[i] = (threadIdx.x / WARP_SIZE) * WarpBins + bin;
      }
      Histogram(storage).Composite(thread_bins, histogram);
    }
    __syncthreads();
    int bin_counts[RadixBins / Threads], cumulative_counts[RadixBins / Threads];
#pragma unroll
    for (int i = 0; i < RadixBins / Threads; ++i) {
      int bin = (RadixBins - 1) - int(threadIdx.x) * (RadixBins / Threads) - i;
      int bin_count = 0;
#pragma unroll
      for (int warp = 0; warp < Warps; ++warp) {
        bin_count += histogram[warp * WarpBins + bin];
      }
      bin_counts[i] = bin_count;
    }
    Scan(scan).InclusiveSum(bin_counts, cumulative_counts);
#pragma unroll
    for (int i = 0; i < RadixBins / Threads; ++i) {
      if (cumulative_counts[i] >= remaining_rank && cumulative_counts[i] - bin_counts[i] < remaining_rank) {
        selected_bucket.bin = (RadixBins - 1) - int(threadIdx.x) * (RadixBins / Threads) - i;
        selected_bucket.higher_count = cumulative_counts[i] - bin_counts[i];
        selected_bucket.bucket_count = bin_counts[i];
      }
    }
    __syncthreads();
    score_prefix |= uint32_t(selected_bucket.bin) << shift;
    prefix_mask |= uint32_t(RadixBins - 1) << shift;
    higher_count += selected_bucket.higher_count;
    remaining_rank -= selected_bucket.higher_count;
    int retained_count = higher_count + selected_bucket.bucket_count;
    __syncthreads();
    if (selected_bucket.bucket_count == 1 || shift == 0) {
      return {score_prefix, retained_count};
    }
  }
  return {score_prefix, k};
}
// Stable compaction preserves token-ID tie order while the sampler uses dense
// indices.
struct CompactSamplingReader {
  using Element = float;
  float const *logits;
  __device__ float scaled(int, float value) const { return value; }
  __device__ float operator()(int index) const { return logits[index]; }
};

template <int Threads, typename Reader>
__device__ __noinline__ int sample_compact_top_p(Reader reader, int vocab, int retained_count, float p, int best,
    SamplingRng const &rng, float *workspace, float *reusable_storage) {
  constexpr int Width = 16;
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  static_assert(CUB_STORAGE_OFFSET_BYTES + sizeof(typename Scan::TempStorage) <= SAMPLING_SHARED_BYTES);
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(sampling_shared + CUB_STORAGE_OFFSET_BYTES);
  // Layout: [retained_count IDs][retained_count scores][optional weights].
  int *token_ids =
      reusable_storage ? reinterpret_cast<int *>(reusable_storage) : reinterpret_cast<int *>(sampling_shared);
  float *scores = reinterpret_cast<float *>(token_ids + retained_count);
  int offset = 0, best_index = -1;
  for (int base = 0; base < vocab; base += Threads * Width) {
    int start = base + threadIdx.x * Width;
    auto values = load_vector<typename Reader::Element, Width>(reader.logits, start, vocab);
    uint32_t selected = 0;
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      int v = start + i;
      if (v < vocab && score_key(reader.scaled(v, float(values.values[i])), v) >= reader.minimum_score_key) {
        selected |= uint32_t(1) << i;
      }
    }
    int prefix, total;
    Scan(storage).ExclusiveSum(__popc(selected), prefix, total);
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      if (selected & (uint32_t(1) << i)) {
        int index = offset + prefix++;
        token_ids[index] = start + i;
        scores[index] = reader.scaled(start + i, float(values.values[i]));
        if (start + i == best) {
          best_index = index;
        }
      }
    }
    offset += total;
    __syncthreads();
  }
  best_index = block_reduce(best_index, workspace, IndexReduce<true>{});
  // Frequency counts stay intact. Otherwise the entire count buffer is idle.
  int available_words = reusable_storage ? vocab : CUB_STORAGE_OFFSET_BYTES / sizeof(int);
  float *weights = retained_count <= available_words / COMPACT_WORDS_WITH_WEIGHT ? scores + retained_count : nullptr;
  CompactSamplingReader compact{scores};
  int index = dual_pivot_sample<false, Threads>(
      compact, retained_count, p, best_index, rng.seed, rng.position, workspace, weights);
  return token_ids[index];
}

// Read processed FP32 logits from scratch without changing the input tensor.
struct CachedCutoffReader {
  using Element = float;
  float const *logits;
  float center, temperature;
  __device__ float unmasked_scaled(int, float value) const { return (value - center) / temperature; }
};

template <int Threads, typename Reader>
__device__ inline int sample_filtered(Reader reader, int vocab, int k, float p, int best, SamplingRng const &rng,
    float *workspace, float *reusable_storage) {
  if (k > 0 && p < 1.f) {
    CutoffResult cutoff;
    // Reuse idle counts only when processing scores is expensive. Plain
    // logits and active frequency penalties use the direct reader.
    if constexpr (!Reader::IsPlain) {
      if (reusable_storage) {
        cutoff = radix_cutoff<Threads>(
            CachedCutoffReader{reusable_storage, reader.center, reader.temperature}, vocab, k, workspace);
      } else {
        cutoff = radix_cutoff<Threads>(reader, vocab, k, workspace);
      }
    } else {
      cutoff = radix_cutoff<Threads>(reader, vocab, k, workspace);
    }
    // Retain the entire boundary-score group.
    reader.minimum_score_key = uint64_t(cutoff.score_threshold) << 32;
    // Two words per retained token; oversized tie groups use the full-row path.
    int available_words = reusable_storage ? vocab : CUB_STORAGE_OFFSET_BYTES / int(sizeof(int));
    if (cutoff.retained_count <= available_words / COMPACT_WORDS_PER_TOKEN) {
      return sample_compact_top_p<Threads>(
          reader, vocab, cutoff.retained_count, p, best, rng, workspace, reusable_storage);
    }
  }
  if (p < 1.f) {
    return dual_pivot_sample<false, Threads>(
        reader, vocab, p, best, rng.seed, rng.position, workspace, reusable_storage);
  }
  return dual_pivot_sample<true, Threads>(
      reader, vocab, float(k), best, rng.seed, rng.position, workspace, reusable_storage);
}

template <bool Plain, typename T, typename Reader>
__device__ inline int sample_unfiltered(
    T const *logits, int vocab, Reader const &reader, float temperature, SamplingRng const &rng, float *workspace) {
  Draw draw;
  ChooseDraw choose{temperature};
  scan_logits(logits, vocab, [&](int v, float logit) {
    float score;
    if constexpr (Plain) {
      score = isnan(logit) ? -INFINITY : logit;
    } else {
      score = reader.raw(v, logit);
    }
    Draw candidate{score, isfinite(score) ? rng.gumbel(v) : 0.f, v};
    draw = choose(draw, candidate);
  });
  return block_reduce(draw, workspace, choose).token;
}

template <int BlockThreads = 0, typename T, bool Plain>
__device__ inline int sample_token(T const *logits, int vocab, ServingConfig const *cfg,
    SamplingScoreReader<T, Plain> reader, bool penalties, int generation_position, float *workspace,
    float *reusable_storage) {
  // Inspect the original double so a positive temperature that underflows
  // FP32 does not silently become greedy (tied maxima must still be sampled).
  int k = int(cfg->top_k);
  float top_p = float(cfg->top_p);
  bool top_k = k > 0 && k < vocab;
  bool greedy = cfg->temperature == 0.0 || k == 1;
  float temperature = fmaxf(float(cfg->temperature), 0x1.0p-126f);
  bool unfiltered = !greedy && !top_k && top_p == 1.f;
  SamplingRng rng{uint64_t(cfg->seed), uint32_t(generation_position)};

  if (unfiltered) {
    if (!penalties && cfg->bias_count == 0) {
      return sample_unfiltered<true>(logits, vocab, reader, temperature, rng, workspace);
    }
    return sample_unfiltered<false>(logits, vocab, reader, temperature, rng, workspace);
  }

  uint64_t best = 0;
  scan_logits(logits, vocab, [&](int v, float logit) {
    float score = reader.raw(v, logit);
    // Cache processed logits in the idle count row during the existing max
    // pass.
    if constexpr (!Plain) {
      if (reusable_storage && top_k && top_p < 1.f && !greedy) {
        reusable_storage[v] = score;
      }
    }
    uint64_t key = score_key(score, v);
    best = key > best ? key : best;
  });
  best = block_reduce(best, workspace, Maximum{});
  int best_token = 0xffffffffu - uint32_t(best);
  if (greedy) {
    return best_token;
  }

  float max_score = reader.raw(best_token);
  if (!isfinite(max_score)) {
    return best_token;
  }
  reader.center = max_score;
  reader.temperature = temperature;
  if constexpr (BlockThreads == 128 || BlockThreads == 256) {
    return sample_filtered<BlockThreads>(
        reader, vocab, top_k ? k : 0, top_p, best_token, rng, workspace, reusable_storage);
  } else if (blockDim.x == 128) {
    return sample_filtered<128>(reader, vocab, top_k ? k : 0, top_p, best_token, rng, workspace, reusable_storage);
  } else {
    return sample_filtered<256>(reader, vocab, top_k ? k : 0, top_p, best_token, rng, workspace, reusable_storage);
  }
}

template <int BlockThreads = 0, typename T, typename Token>
__device__ inline void sample(T const *logits, float *scratch, Token *output, int padded_vocab,
    ServingConfig const *cfg, long long const *history, int history_len, int prompt_len, int generation_position,
    bool reuse_history = false) {
  int vocab = min(padded_vocab, int(cfg->vocab_size));
  SamplingScratch state(scratch, padded_vocab);
  float repetition = float(cfg->repetition_penalty);
  float frequency = float(cfg->frequency_penalty);
  float presence = float(cfg->presence_penalty);
  bool penalties = frequency != 0 || presence != 0 || repetition != 1;
  if (penalties) {
    prepare_history(state, cfg, history, history_len, prompt_len, generation_position, vocab, frequency, reuse_history);
  }
  SamplingScoreReader<T> reader{logits, cfg, state.frequency_counts, state.seen_total, state.seen_generated, repetition,
      frequency, presence, 0.f, 1.f, int(cfg->bias_count)};
  // Reuse idle frequency storage for weights; active history counts stay
  // intact.
  float *reusable_storage = frequency == 0.f ? reinterpret_cast<float *>(state.frequency_counts) : nullptr;
  int token;
  if (!penalties && cfg->bias_count == 0) {
    SamplingScoreReader<T, true> plain{logits};
    token = sample_token<BlockThreads>(
        logits, vocab, cfg, plain, false, generation_position, state.workspace, reusable_storage);
  } else {
    token = sample_token<BlockThreads>(
        logits, vocab, cfg, reader, penalties, generation_position, state.workspace, reusable_storage);
  }
  if (threadIdx.x == 0) {
    *output = token;
  }
}

#if defined(MODE_ONLINE_PINNED)
template <int BlockThreads, typename T>
__device__ inline void sample_request(T const *logits, float *scratch, long long *output, int padded_vocab, int slot,
    mirage::runtime::RuntimeConfig const &config) {
  int row = config.request_ids[slot];
  if (row < 0) {
    return;
  }
  // Scratch follows the buffer row, not the batch slot (which can compact).
  scratch += row * sampling_scratch_words(padded_vocab);
  int start = config.qo_indptr_buffer[slot], end = config.qo_indptr_buffer[slot + 1];
  int prompt = config.prompt_length[row];
  for (int i = start; i < end; ++i) {
    int position = config.step[row] + i - start;
    if (position + 1 < prompt) {
      continue;
    }
    sample<BlockThreads>(logits + i * padded_vocab, scratch, output + i, padded_vocab, config.generation_config + row,
        config.tokens + row * MPK_MAX_SEQ_LENGTH, position + 1, prompt, position + 1 - prompt, true);
    __syncthreads();
  }
}
#endif
} // namespace serving
} // namespace mirage
