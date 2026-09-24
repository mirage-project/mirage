// SPDX-License-Identifier: Apache-2.0
// Uniform/Gumbel transform adapted from vLLM PR #51367 at
// 6af68f63030a7b27f8c07fad5b6237dcb3964e5c.
// Copyright contributors to the vLLM project (uniform/Gumbel transform).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <cub/block/block_radix_sort.cuh>
#include "mirage/persistent_kernel/serving_config.h"
#include <curand_kernel.h>

namespace mirage { namespace serving {
__device__ inline float option(double value) {
  return static_cast<float>(value);
}
__device__ inline uint64_t score_key(float score, uint32_t token) {
  uint32_t bits = __float_as_uint(score == 0.f ? 0.f : score);
  bits = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (uint64_t(bits) << 32) | (0xffffffffu - token);
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
// neither a preliminary maximum nor a second vocabulary pass. Near cancellation,
// use FP64 to resolve the sign instead of depending on the reduction order.
struct ChooseDraw {
  float temperature;
  __device__ Draw operator()(Draw a, Draw b) const {
    if (!isfinite(a.score) || !isfinite(b.score))
      return score_key(a.score, a.token) >= score_key(b.score, b.token) ? a : b;
    if (a.score == b.score) {
      if (a.noise != b.noise) return a.noise > b.noise ? a : b;
      return a.token < b.token ? a : b;
    }
    float delta = a.score - b.score;
    float shift = temperature * (b.noise - a.noise);
    float margin = delta - shift;
    // Eight FP32 epsilons conservatively cover the subtraction/product errors.
    if (fabsf(margin) > 0x1p-20f * (fabsf(delta) + fabsf(shift)))
      return margin > 0 ? a : b;
    double exact = (double(a.score) - b.score) + double(temperature) * (double(a.noise) - b.noise);
    if (exact != 0) return exact > 0 ? a : b;
    return a.token < b.token ? a : b;
  }
};

template<typename T>
__device__ inline T shuffle_xor(T value, int offset) {
  return __shfl_xor_sync(0xffffffff, value, offset);
}
__device__ inline Draw shuffle_xor(Draw value, int offset) {
  return {shuffle_xor(value.score, offset), shuffle_xor(value.noise, offset),
          shuffle_xor(value.token, offset)};
}
template<typename T, typename Op>
__device__ inline T warp_reduce(T value, Op op) {
  for (int offset = WARP_SIZE / 2; offset; offset /= 2)
    value = op(value, shuffle_xor(value, offset));
  return value;
}

// Only warp aggregates use scratch. The barriers protect publication and reuse.
// T{} is the identity: zero for sums/unsigned maxima, -infinity for draws.
template<typename T, typename Op>
__device__ inline T block_reduce(T value, float *workspace, Op op) {
  int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  auto partial = reinterpret_cast<T *>(workspace);
  value = warp_reduce(value, op);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  value = warp_reduce(lane < blockDim.x / WARP_SIZE ? partial[lane] : T{}, op);
  __syncthreads();
  return value;
}

// Reconstruct (bits + 0.5) / 2^32 from two exact 16-bit conversions.
// Keeping all source bits is important at u -> 0, the winning Gumbel tail.
__device__ inline float sampling_uniform(uint32_t bits) {
  return float(bits >> 16) * 0x1p-16f +
         (float(bits & 0xffffu) + 0.5f) * 0x1p-32f;
}

__device__ inline float gumbel_from_uniform(float u) {
  // Degree-8 expansion of -log1p(-u). Unlike log(1-u), it preserves tiny u.
  // This is the polynomial and endpoint handling used by vLLM's FP32 path.
  float polynomial = 1.f / 8.f;
  #pragma unroll
  for (int order = 7; order >= 1; --order)
    polynomial = 1.f / order + u * polynomial;
  float exponential = u < 0.25f ? u * polynomial
                                : -logf(fmaxf(1.f - u, 0x1p-24f));
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
  __device__ SamplingRng(uint64_t seed, uint32_t position)
      : seed(seed), position(position) {}
  __device__ uint32_t bits(uint32_t token) const {
    if (group != token / 4) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, position, token & ~uint32_t(3), &state);
      values = curand4(&state);
      group = token / 4;
    }
    return (token & 3) == 0 ? values.x : (token & 3) == 1 ? values.y
                    : (token & 3) == 2 ? values.z : values.w;
  }
  __device__ float gumbel(uint32_t token) const {
    return gumbel_from_uniform(sampling_uniform(bits(token)));
  }
};

// Scores are recomputed on each pass. The 64-bit key makes top-k and top-p
// boundaries exact even for ties, without a vocabulary-sized candidate list.
template<bool Weighted, typename Reader>
__device__ inline uint64_t cutoff(Reader const &reader, int vocab,
                                  uint64_t lower, float target, float *workspace,
                                  int const *candidates = nullptr,
                                  int candidate_count = -1) {
  uint64_t prefix = 0, mask = 0;
  int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  for (int shift = 64 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
    uint64_t digit_mask = uint64_t(RADIX_BINS - 1) << shift;
    // Inverted token IDs have fixed leading ones above the vocabulary range.
    if (shift < 32 && (uint32_t(vocab - 1) >> shift) == 0) {
      prefix |= digit_mask;
      mask |= digit_mask;
      continue;
    }
    float bins[RADIX_BINS] = {};
    float counts[RADIX_BINS] = {};
    int size = candidate_count >= 0 ? candidate_count : vocab;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      int v = candidate_count >= 0 ? candidates[i] : i;
      float score = reader(v);
      uint64_t key = score_key(score, v);
      if (key >= lower && (key & mask) == prefix) {
        int digit = (key >> shift) & (RADIX_BINS - 1);
        float weight = Weighted ? expf(score) : 1.f;
        #pragma unroll
        for (int bin = 0; bin < RADIX_BINS; ++bin) {
          bins[bin] += bin == digit ? weight : 0.f;
          if constexpr (Weighted) counts[bin] += bin == digit ? 1.f : 0.f;
        }
      }
    }
    #pragma unroll
    for (int bin = 0; bin < RADIX_BINS; ++bin) {
      float mass = warp_reduce(bins[bin], Sum{});
      if (lane == 0) workspace[warp * RADIX_BINS + bin] = mass;
      if constexpr (Weighted) {
        float count = warp_reduce(counts[bin], Sum{});
        if (lane == 0) workspace[(warps + warp) * RADIX_BINS + bin] = count;
      }
    }
    __syncthreads();
    // Every warp reads one bin per lane, then broadcasts during the scan.
    float mass = 0, count = 0;
    if (lane < RADIX_BINS) {
      for (int w = 0; w < warps; ++w) {
        mass += workspace[w * RADIX_BINS + lane];
        if constexpr (Weighted) count += workspace[(warps + w) * RADIX_BINS + lane];
      }
    }
    if constexpr (!Weighted) count = mass;
    int selected = 0;
    for (int bin = RADIX_BINS - 1; bin >= 0; --bin) {
      float amount = __shfl_sync(0xffffffff, mass, bin);
      if (amount > 0) {
        selected = bin;
        if (amount >= target) break;
        target -= amount;
      }
    }
    float selected_count = __shfl_sync(0xffffffff, count, selected);
    __syncthreads();
    prefix |= uint64_t(selected) << shift;
    mask |= digit_mask;
    if (selected_count == 1.f) {
      uint64_t found = 0;
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        int v = candidate_count >= 0 ? candidates[i] : i;
        uint64_t key = score_key(reader(v), v);
        if (key >= lower && (key & mask) == prefix) found = key;
      }
      return block_reduce(found, workspace, Maximum{});
    }
  }
  return prefix;
}

__device__ inline uint64_t sampled_key(float score, int v, uint64_t lower,
                                       SamplingRng const &rng) {
  if (score_key(score, v) < lower || !isfinite(score)) return 0;
  return score_key(score + rng.gumbel(v), v);
}

template<typename T, int Width = 16>
struct alignas(Width * sizeof(T)) SampleVector {
  T values[Width];
};
template<typename T, int Width = 16>
__device__ inline SampleVector<T, Width> load_vector(T const *values, int start, int size) {
  if (start + Width <= size && uintptr_t(values + start) % alignof(SampleVector<T, Width>) == 0)
    return *reinterpret_cast<SampleVector<T, Width> const *>(values + start);
  SampleVector<T, Width> result{};
  #pragma unroll
  for (int i = 0; i < Width; ++i)
    if (start + i < size) result.values[i] = values[start + i];
  return result;
}

template<typename T>
struct SamplingScoreReader {
  T const *logits;
  ServingConfig const *cfg;
  int const *frequency_counts;
  uint32_t const *seen_total;
  uint32_t const *seen_generated;
  float repetition, frequency, presence;
  float center, temperature;
  int bias_count;

  __device__ float raw(int v) const {
    float score = static_cast<float>(logits[v]);
    // Bias IDs are sorted by the host packer. Keep the small bias table in
    // ServingConfig instead of materializing a modified vocabulary row.
    if (bias_count) {
      int lo = 0, hi = bias_count;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cfg->biases[mid].token < v) lo = mid + 1;
        else hi = mid;
      }
      if (lo < bias_count && cfg->biases[lo].token == v)
        score += option(cfg->biases[lo].value);
    }
    if (repetition != 1.f &&
        (seen_total[v >> 5] & (uint32_t(1) << (v & 31))))
      score = score > 0 ? score / repetition : score * repetition;
    // Most vocabulary IDs never occur in the generated history. Avoid a
    // dense count-array read unless its bit is set.
    if ((frequency != 0.f || presence != 0.f) &&
        (seen_generated[v >> 5] & (uint32_t(1) << (v & 31)))) {
      if (frequency != 0.f) score -= frequency * frequency_counts[v];
      if (presence != 0.f) score -= presence;
    }
    return isnan(score) ? -INFINITY : score;
  }
  __device__ float operator()(int v) const {
    return (raw(v) - center) / temperature;
  }
};

// A 1024-token probe supplies a safe lower bound for top-k. The full row is
// checked before using the bounded shortlist, so skewed rows fall back to
// repeated scans. Both probe storage and shortlist fit in CTA shared memory.
template<typename Reader>
__device__ inline uint64_t sampled_top_k_bound(Reader const &reader, int vocab,
                                               int k, float *workspace) {
  constexpr int ProbeSize = 1024, Threads = 128, Items = ProbeSize / Threads;
  int rank = min(k, max(4, int((int64_t(k) * 4 * ProbeSize + vocab - 1) / vocab)));
  using Sort = cub::BlockRadixSort<uint64_t, Threads, Items>;
  static_assert(sizeof(typename Sort::TempStorage) <= SAMPLING_SHARED_BYTES);
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage = *reinterpret_cast<typename Sort::TempStorage *>(sampling_shared);
  uint64_t keys[Items];
  #pragma unroll
  for (int i = 0; i < Items; ++i) {
    int index = threadIdx.x * Items + i;
    int token = int(uint64_t(index) * vocab / ProbeSize);
    keys[i] = score_key(reader(token), token);
  }
  Sort(storage).SortDescending(keys);
  __syncthreads();
  uint64_t found = 0;
  #pragma unroll
  for (int i = 0; i < Items; ++i)
    if (int(threadIdx.x) * Items + i == rank - 1) found = keys[i];
  return block_reduce(found, workspace, Maximum{});
}

// FlashInfer-style dual-pivot rejection for top-p without top-k.
// https://flashinfer.ai/2025/03/10/sampling.html
// Reusing the token-indexed Philox Gumbel on every round makes the result equal
// to a single Gumbel-max draw over the exact nucleus.
template<typename Reader>
__device__ inline int dual_pivot_top_p(Reader const &reader, int vocab,
                                      float top_p, int best_token,
                                      SamplingRng const &rng, float *workspace) {
  float mass = 0.f;
  uint64_t winner = 0;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    float score = reader(v);
    mass += expf(score);
    uint64_t candidate = sampled_key(score, v, 0, rng);
    winner = max(winner, candidate);
  }
  mass = block_reduce(mass, workspace, Sum{});
  winner = block_reduce(winner, workspace, Maximum{});
  float target = top_p * mass;
  if (target <= 1.f || winner == 0) return best_token;
  uint64_t high = score_key(reader(best_token), best_token);
  uint64_t low = 0;
  for (int round = 0; round < 64 && low < high; ++round) {
    int token = 0xffffffffu - uint32_t(winner);
    uint64_t pivot = score_key(reader(token), token);
    uint64_t midpoint = pivot + (high - pivot) / 2;
    float mass_above_pivot = 0.f, mass_above_midpoint = 0.f;
    uint64_t next_pivot = 0, next_midpoint = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
      float score = reader(v);
      uint64_t key = score_key(score, v);
      bool above_pivot = key > pivot;
      bool above_midpoint = key > midpoint;
      float weight = expf(score);
      if (above_pivot) mass_above_pivot += weight;
      if (above_midpoint) mass_above_midpoint += weight;
      if (above_pivot) {
        uint64_t draw = sampled_key(score, v, 0, rng);
        next_pivot = max(next_pivot, draw);
        if (above_midpoint) next_midpoint = max(next_midpoint, draw);
      }
    }
    mass_above_pivot = block_reduce(mass_above_pivot, workspace, Sum{});
    mass_above_midpoint = block_reduce(mass_above_midpoint, workspace, Sum{});
    next_pivot = block_reduce(next_pivot, workspace, Maximum{});
    next_midpoint = block_reduce(next_midpoint, workspace, Maximum{});
    if (mass_above_pivot < target) return token;
    if (mass_above_midpoint < target) {
      low = pivot;
      high = midpoint;
      winner = next_pivot;
    } else {
      low = midpoint;
      winner = next_midpoint;
    }
    if (winner == 0) break;
  }
  return best_token;
}

template<typename T, typename Token>
__device__ inline void sample(T const *logits, float *scratch, Token *output,
                              int padded_vocab, ServingConfig const *cfg,
                              long long const *history, int history_len,
                              int prompt_len, int generation_position,
                              bool reuse_history = false) {
  int vocab = min(padded_vocab, int(cfg->vocab_size));
  int *frequency_counts = reinterpret_cast<int *>(scratch);
  int bitset_words = (padded_vocab + 31) / 32;
  auto *seen_total = reinterpret_cast<uint32_t *>(frequency_counts + ((padded_vocab + 1) & ~1));
  auto *seen_generated = seen_total + bitset_words;
  float *workspace = reinterpret_cast<float *>(seen_generated + bitset_words);
  int *seen = reinterpret_cast<int *>(workspace + SCRATCH_WORKSPACE);
  float repetition = option(cfg->repetition_penalty);
  float frequency = option(cfg->frequency_penalty);
  float presence = option(cfg->presence_penalty);
  bool penalties = frequency != 0 || presence != 0 || repetition != 1;
  if (penalties) {
    // Reuse only private row state initialized by this request's first sample.
    // Disabling caching rebuilds the histograms for each generated token.
    bool cached = reuse_history && cfg->cache_history && generation_position > 0;
    int begin = cached ? *seen : 0;
    if (!cached) {
      if (frequency != 0.f)
        for (int v = threadIdx.x; v < vocab; v += blockDim.x)
          frequency_counts[v] = 0;
      for (int v = threadIdx.x; v < bitset_words; v += blockDim.x) {
        seen_total[v] = 0;
        seen_generated[v] = 0;
      }
      __syncthreads();
    }
    for (int i = begin + threadIdx.x; i < history_len; i += blockDim.x) {
      long long token = history[i];
      if (token >= 0 && token < vocab) {
        atomicOr(seen_total + (token >> 5), uint32_t(1) << (token & 31));
        if (i >= prompt_len) {
          atomicOr(seen_generated + (token >> 5), uint32_t(1) << (token & 31));
          if (frequency != 0.f) atomicAdd(frequency_counts + token, 1);
        }
      }
    }
    // All threads must consume the old cursor before it is updated.
    __syncthreads();
    if (threadIdx.x == 0) *seen = history_len;
  }
  SamplingScoreReader<T> reader{logits, cfg, frequency_counts,
                                seen_total, seen_generated, repetition,
                                frequency, presence, 0.f, 1.f,
                                int(cfg->bias_count)};
  // Inspect the original double so a positive temperature that underflows
  // FP32 does not silently become greedy (tied maxima must still be sampled).
  int k = int(cfg->top_k);
  float top_p = option(cfg->top_p);
  bool top_k = k > 0 && k < vocab;
  bool greedy = cfg->temperature == 0.0 || k == 1;
  float temperature = fmaxf(option(cfg->temperature), 0x1.0p-126f);
  bool unfiltered = !greedy && !top_k && top_p == 1.f;
  SamplingRng rng{uint64_t(cfg->seed), uint32_t(generation_position)};
  Draw draw;
  ChooseDraw choose{temperature};
  uint64_t best = 0;
  if (unfiltered && !penalties && cfg->bias_count == 0) {
    constexpr int Width = 16;
    for (int base = Width * threadIdx.x; base < vocab; base += Width * blockDim.x) {
      auto values = load_vector<T, Width>(logits, base, vocab);
      #pragma unroll
      for (int i = 0; i < Width; ++i) {
        int v = base + i;
        if (v >= vocab) continue;
        float score = static_cast<float>(values.values[i]);
        if (isnan(score)) score = -INFINITY;
        Draw candidate{score, isfinite(score) ? rng.gumbel(v) : 0.f, v};
        draw = choose(draw, candidate);
      }
    }
    draw = block_reduce(draw, workspace, choose);
    if (threadIdx.x == 0) *output = draw.token;
    return;
  }
  auto consume = [&](int v) {
    float score = reader.raw(v);
    if (unfiltered) {
      Draw candidate{score, isfinite(score) ? rng.gumbel(v) : 0.f, v};
      draw = choose(draw, candidate);
    } else {
      uint64_t key = score_key(score, v);
      best = key > best ? key : best;
    }
  };
  if (vocab < 4096) {
    for (int v = threadIdx.x; v < vocab; v += blockDim.x)
      consume(v);
  } else {
    constexpr int Width = 16;
    for (int base = Width * threadIdx.x; base < vocab; base += Width * blockDim.x) {
      #pragma unroll
      for (int i = 0; i < Width; ++i) {
        int v = base + i;
        if (v < vocab) consume(v);
      }
    }
  }
  if (unfiltered) {
    draw = block_reduce(draw, workspace, choose);
    if (threadIdx.x == 0) *output = draw.token;
    return;
  }
  best = block_reduce(best, workspace, Maximum{});
  int best_token = 0xffffffffu - uint32_t(best);
  float max_score = greedy ? 0.f : reader.raw(best_token);
  if (greedy || !isfinite(max_score)) {
    if (threadIdx.x == 0) *output = best_token;
    return;
  }
  reader.center = max_score;
  reader.temperature = temperature;
  if (!top_k && top_p < 1.f) {
    int token = dual_pivot_top_p(reader, vocab, top_p, best_token, rng, workspace);
    if (threadIdx.x == 0) *output = token;
    return;
  }
  uint64_t lower = 0;
  int const *candidates = nullptr;
  int candidate_count = -1;
  if (top_k && k <= 256 && vocab >= 4096 && blockDim.x == 128) {
    uint64_t probe = sampled_top_k_bound(reader, vocab, k, workspace);
    extern __shared__ __align__(16) unsigned char sampling_shared[];
    auto *shortlist = reinterpret_cast<int *>(sampling_shared);
    __shared__ int shortlist_count;
    if (threadIdx.x == 0) shortlist_count = 0;
    __syncthreads();
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
      if (score_key(reader(v), v) >= probe) {
        int index = atomicAdd(&shortlist_count, 1);
        if (index < 1024) shortlist[index] = v;
      }
    }
    __syncthreads();
    if (shortlist_count >= k && shortlist_count <= 1024) {
      lower = probe;
      candidates = shortlist;
      candidate_count = shortlist_count;
    }
  }
  if (top_k)
    lower = cutoff<false>(reader, vocab, lower, float(k), workspace,
                          candidates, candidate_count);
  if (top_p < 1.f) {
    float mass = 0;
    int size = candidate_count >= 0 ? candidate_count : vocab;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      int v = candidate_count >= 0 ? candidates[i] : i;
      float score = reader(v);
      if (score_key(score, v) >= lower) mass += expf(score);
    }
    mass = block_reduce(mass, workspace, Sum{});
    if (top_p * mass <= 1.f) {
      // The highest logit's unnormalized weight is one.
      if (threadIdx.x == 0) *output = best_token;
      return;
    }
    lower = cutoff<true>(reader, vocab, lower, top_p * mass, workspace,
                         candidates, candidate_count);
  }
  best = 0;
  int size = candidate_count >= 0 ? candidate_count : vocab;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    int v = candidate_count >= 0 ? candidates[i] : i;
    uint64_t key = sampled_key(reader(v), v, lower, rng);
    best = key > best ? key : best;
  }
  best = block_reduce(best, workspace, Maximum{});
  if (threadIdx.x == 0) *output = best ? 0xffffffffu - uint32_t(best) : best_token;
}

#if defined(MODE_ONLINE_PINNED)
template<typename T>
__device__ inline void sample_request(T const *logits, float *scratch, long long *output,
                                     int padded_vocab, int slot,
                                     mirage::runtime::RuntimeConfig const &config) {
  int row = config.request_ids[slot];
  if (row < 0) return;
  // Scratch follows the buffer row, not the batch slot (which can compact).
  scratch += row * sampling_scratch_words(padded_vocab);
  int start = config.qo_indptr_buffer[slot], end = config.qo_indptr_buffer[slot + 1];
  for (int i = start; i < end; ++i) {
    int position = config.step[row] + i - start;
    int prompt = config.prompt_length[row];
    if (position + 1 < prompt) continue;
    sample(logits + i * padded_vocab, scratch, output + i, padded_vocab,
           config.generation_config + row,
           config.tokens + row * MPK_MAX_SEQ_LENGTH, position + 1, prompt,
           position + 1 - prompt, true);
    __syncthreads();
  }
}
#endif
}} // namespace mirage::serving
