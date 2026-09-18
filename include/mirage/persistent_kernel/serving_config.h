#pragma once
#include <stdint.h>

namespace mirage { namespace serving {
#define SERVING_CONSTANT(name, value) constexpr int name = value;
#include "serving_config.def"
#undef SERVING_CONSTANT

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr int sampling_scratch_words(int vocab) {
  return SCRATCH_VOCAB_ARRAYS * vocab + SCRATCH_WORKSPACE + SCRATCH_STATE_WORDS;
}

// Called by the scheduler after committing an output token. Match only the
// generated suffix: prompt tokens must never complete a user stop sequence.
// Keep this host/device so the same stopping rules can be tested without CUDA.
#ifdef __CUDACC__
__host__ __device__
#endif
inline int finish_reason(int64_t const *cfg, long long const *history,
                         int history_len, int prompt_len, int max_seq_length,
                         bool cancelled, long long fallback_eos = -1) {
  if (cancelled) return FINISH_CANCELLED;
  int generated = history_len - prompt_len;
  if (generated > 0) {
    auto token = history[history_len - 1];
    if (cfg[EOS_COUNT] == 0 && token == fallback_eos) return FINISH_STOP;
    for (int j = 0; j < cfg[EOS_COUNT]; ++j)
      if (token == cfg[EOS_IDS + j]) return FINISH_STOP;
    for (int j = 0; j < cfg[STOP_COUNT]; ++j) {
      auto seq = cfg + STOP_SEQUENCES + j * STOP_STRIDE;
      int length = int(seq[0]);
      if (length > generated) continue;
      bool matches = length > 0;
      for (int k = 0; k < length && matches; ++k)
        matches = history[history_len - length + k] == seq[k + 1];
      if (matches) return FINISH_STOP;
    }
  }
  if (history_len >= max_seq_length ||
      (cfg[MAX_NEW_TOKENS] > 0 && generated >= cfg[MAX_NEW_TOKENS]))
    return FINISH_LENGTH;
  return FINISH_NONE;
}
}} // namespace mirage::serving
