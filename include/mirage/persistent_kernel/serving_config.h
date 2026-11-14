#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mirage { namespace serving {

constexpr int MAX_EOS = 16;
constexpr int MAX_BIASES = 256;

struct LogitBias {
  int64_t token;
  double value;
};

// Fixed-size ring record shared by the host packer and persistent CUDA worker.
// All fields are eight bytes so the record also fits the existing int64 buffers.
struct ServingConfig {
  int64_t max_new_tokens;
  int64_t seed;
  int64_t vocab_size;
  double temperature;
  double top_p;
  int64_t top_k;
  double frequency_penalty;
  double presence_penalty;
  double repetition_penalty;
  int64_t eos_count;
  int64_t bias_count;
  int64_t cache_history;
  int64_t eos_ids[MAX_EOS];
  LogitBias biases[MAX_BIASES];
};

constexpr int CONFIG_WORDS = sizeof(ServingConfig) / sizeof(int64_t);
static_assert(std::is_standard_layout<ServingConfig>::value, "ServingConfig must have stable layout");
static_assert(std::is_trivially_copyable<ServingConfig>::value, "ServingConfig must be copyable");
static_assert(alignof(ServingConfig) <= alignof(int64_t),
              "ServingConfig alignment exceeds int64 backing storage");
static_assert(sizeof(ServingConfig) == 540 * sizeof(int64_t), "ServingConfig size changed");
static_assert(sizeof(LogitBias) == 2 * sizeof(int64_t), "bias entry size changed");
static_assert(offsetof(ServingConfig, max_new_tokens) == 0 * sizeof(int64_t), "budget layout changed");
static_assert(offsetof(ServingConfig, seed) == 1 * sizeof(int64_t), "seed layout changed");
static_assert(offsetof(ServingConfig, vocab_size) == 2 * sizeof(int64_t), "vocab layout changed");
static_assert(offsetof(ServingConfig, temperature) == 3 * sizeof(int64_t), "temperature layout changed");
static_assert(offsetof(ServingConfig, top_p) == 4 * sizeof(int64_t), "top-p layout changed");
static_assert(offsetof(ServingConfig, top_k) == 5 * sizeof(int64_t), "top-k layout changed");
static_assert(offsetof(ServingConfig, frequency_penalty) == 6 * sizeof(int64_t), "frequency layout changed");
static_assert(offsetof(ServingConfig, presence_penalty) == 7 * sizeof(int64_t), "presence layout changed");
static_assert(offsetof(ServingConfig, repetition_penalty) == 8 * sizeof(int64_t), "repetition layout changed");
static_assert(offsetof(ServingConfig, eos_count) == 9 * sizeof(int64_t), "EOS count layout changed");
static_assert(offsetof(ServingConfig, bias_count) == 10 * sizeof(int64_t), "bias count layout changed");
static_assert(offsetof(ServingConfig, cache_history) == 11 * sizeof(int64_t), "history layout changed");
static_assert(offsetof(ServingConfig, eos_ids) == 12 * sizeof(int64_t), "EOS layout changed");
static_assert(offsetof(ServingConfig, biases) == 28 * sizeof(int64_t), "bias layout changed");

constexpr int FINISH_NONE = 0;
constexpr int FINISH_STOP = 1;
constexpr int FINISH_LENGTH = 2;

// Sampler implementation and scratch allocation constants.
constexpr int MAX_SAMPLING_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int RADIX_BITS = 4;
constexpr int RADIX_BINS = 16;
constexpr int SCRATCH_WORKSPACE = 2 * RADIX_BINS * MAX_SAMPLING_THREADS / WARP_SIZE;
constexpr int SCRATCH_STATE_WORDS = 2;
constexpr int SAMPLING_SHARED_BYTES = 16 * 1024;

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int sampling_frequency_words(int vocab) {
  return (vocab + 1) & ~1;
}

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int sampling_bitset_words(int vocab) {
  return (vocab + 31) / 32;
}

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int sampling_scratch_words(int vocab) {
  // Generated-token frequency plus two bitsets (all tokens and generated).
  // Round the count array to an even word count for uint64 CTA reductions.
  return sampling_frequency_words(vocab) + 2 * sampling_bitset_words(vocab) +
         SCRATCH_WORKSPACE + SCRATCH_STATE_WORDS;
}

inline void pack_config(ServingConfig *out, int64_t budget, int64_t seed,
                        int64_t vocab, double temperature, double top_p,
                        int64_t top_k, double frequency, double presence,
                        double repetition, bool cache_history,
                        int64_t const *eos, int eos_count,
                        int64_t const *bias_ids, double const *bias_values,
                        int bias_count) {
  *out = {};
  out->max_new_tokens = budget;
  out->seed = seed;
  out->vocab_size = vocab;
  out->temperature = temperature;
  out->top_p = top_p;
  out->top_k = top_k;
  out->frequency_penalty = frequency;
  out->presence_penalty = presence;
  out->repetition_penalty = repetition;
  out->eos_count = eos_count;
  out->bias_count = bias_count;
  out->cache_history = cache_history;
  for (int i = 0; i < eos_count; ++i) out->eos_ids[i] = eos[i];
  for (int i = 0; i < bias_count; ++i)
    out->biases[i] = {bias_ids[i], bias_values[i]};
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline int finish_reason(ServingConfig const *cfg, long long const *history,
                         int history_len, int prompt_len, int max_seq_length,
                         long long fallback_eos = -1) {
  int generated = history_len - prompt_len;
  if (generated > 0) {
    auto token = history[history_len - 1];
    if (cfg->eos_count == 0 && token == fallback_eos) return FINISH_STOP;
    for (int j = 0; j < cfg->eos_count; ++j)
      if (token == cfg->eos_ids[j]) return FINISH_STOP;
  }
  if (history_len >= max_seq_length ||
      (cfg->max_new_tokens > 0 && generated >= cfg->max_new_tokens))
    return FINISH_LENGTH;
  return FINISH_NONE;
}

}} // namespace mirage::serving
