#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <curand_kernel.h>
#include <limits>
#include <type_traits>

namespace mirage {
namespace serving {

constexpr int BITS_PER_WORD = std::numeric_limits<uint32_t>::digits;

// For dividing history_buffer into multiple ranges
class HistoryBuffer {
public:
  __device__ HistoryBuffer(uint32_t* buffer, int padded_vocab) {
    uint32_t words = (padded_vocab + BITS_PER_WORD - 1) / BITS_PER_WORD;
    frequency_counts = buffer;
    seen_total = buffer + padded_vocab;
    seen_generated = buffer + padded_vocab + words;    
  }
  uint32_t *frequency_counts;
  uint32_t *seen_total;
  uint32_t *seen_generated;  
};

// For dividing buffer into multiple regions
// 1. scores = vocab * sizeof(float)
// 2. CTA partials = SAMPLING_PARTIAL_WORDS * sizeof(uint32_t)
// 3. meta data = 4 * sizeof(uint32_t)
class RadixBuffer {
public:
  __device__ RadixBuffer(char *buffer, int padded_vocab, int slot) {
    char *base = buffer + size_t(slot) * sampling_buffer_bytes(padded_vocab);
    scores = reinterpret_cast<float *>(base);
    histogram = reinterpret_cast<uint32_t *>(base + size_t(padded_vocab) * sizeof(float));
    ready = histogram + SAMPLING_PARTIAL_WORDS;
    radix_prefix = ready + 1;
    arrival = ready + 2;
    rank = ready + 3;
  }
  float *scores;
  uint32_t *histogram;
  uint32_t *ready;
  uint32_t *radix_prefix;
  uint32_t *arrival;
  uint32_t *rank;
};



enum class SamplePhase {
  PREPARE = 0,
  SCORE = 1,
  RADIX = 2
};

struct PivotMass {
  float first, second;
  __device__ PivotMass operator+(PivotMass other) const {
    return {first + other.first, second + other.second};
  }
};

struct JointPivotMass {
  float first, second, first_mass, second_mass;
  __device__ JointPivotMass operator+(JointPivotMass other) const {
    return {first + other.first, second + other.second,
            first_mass + other.first_mass, second_mass + other.second_mass};
  }
};

template <int Threads, typename Mass = PivotMass>
struct PivotShared {
  union {
    typename cub::BlockReduce<float, Threads>::TempStorage mass;
    typename cub::BlockReduce<Mass, Threads>::TempStorage pivots;
    typename cub::BlockScan<float, Threads>::TempStorage scan;
  };
  float total, offset, tile;
  int selected;
};

template <int Threads>
struct JointPivotShared : PivotShared<Threads, JointPivotMass> {
  float first_mass, second_mass;
  double target;
};

template <int Threads>
union SamplingShared {
  JointPivotShared<Threads> pivot;
  typename cub::BlockReduce<uint32_t, Threads>::TempStorage maximum;
};

// SamplePhase::PREPARE prepare history data
__device__ inline void prepare_history(
  uint32_t* history_buffer, 
  char* radix_buffer,
  int padded_vocab, 
  int slot,
  mirage::runtime::RuntimeConfig const &config
) {
  
  int row = config.request_ids[slot];
  RadixBuffer radix_buf(radix_buffer, padded_vocab, slot);
  
  uint32_t *ptr = radix_buf.histogram;

  for (int i = threadIdx.x; i < SAMPLING_PARTIAL_WORDS + 4; i += blockDim.x) {
    ptr[i] = 0;
  }

  auto const *cfg = config.generation_config + row;
  if (threadIdx.x == 0)
    *radix_buf.rank = cfg->top_k;

  bool need_frequency = cfg->frequency_penalty != 0.0f;
  bool need_total = cfg->repetition_penalty != 1.f;
  bool need_generated = cfg->presence_penalty != 0.f || cfg->frequency_penalty != 0.f;

  // We do not need to prepare the history data
  if (!need_total && !need_generated) {
    return;
  }

  int start = config.qo_indptr_buffer[slot];
  int end = config.qo_indptr_buffer[slot + 1];
  int prompt_len = config.prompt_length[row];
  int step = config.step[row];
  int current_len = step + end - start;
  int bitset_words = (padded_vocab + BITS_PER_WORD - 1) / BITS_PER_WORD;
  int history_words = padded_vocab + 2 * bitset_words;

  HistoryBuffer buffer(
      history_buffer + row * history_words,
      padded_vocab);

  int generation_position = current_len - prompt_len;
  int vocab = int(cfg->vocab_size);
  long long const *history_tokens =
      config.tokens + row * MPK_MAX_SEQ_LENGTH;
  bool cached = generation_position > 0;
  int begin = cached ? step : 0;

  // clear the array
  if (!cached) {
    if (need_frequency) {
      for (int i = threadIdx.x; i < padded_vocab; i += blockDim.x) {
        buffer.frequency_counts[i] = 0;
      }
    }

    int bitset_words = (padded_vocab + BITS_PER_WORD - 1) / BITS_PER_WORD;
    if (need_total) {
      for (int i = threadIdx.x; i < bitset_words; i += blockDim.x) {
        buffer.seen_total[i] = 0;
      }
    }

    if (need_generated) {
      for (int i = threadIdx.x; i < bitset_words; i += blockDim.x) {
        buffer.seen_generated[i] = 0;
      }
    }
  }

  __syncthreads();

  for (int i = begin + threadIdx.x; i < current_len; i += blockDim.x) {
    long long token = history_tokens[i];
    // assert token is within the range?
    uint32_t index = static_cast<uint32_t>(token) >> 5;
    uint32_t mask = 1u << (token & 31);

    if (need_total) {
        atomicOr(&buffer.seen_total[index], mask);
    }

    if (i >= prompt_len) {
        if (need_generated) {
            atomicOr(&buffer.seen_generated[index], mask);
        }

        if (need_frequency) {
            atomicAdd(&buffer.frequency_counts[token], 1u);
        }
    }
  }
}

__device__ inline float get_logit_bias(
  ServingConfig const *cfg,
  long long token
) {
  int begin = 0, end = cfg->bias_count;
  while (begin < end) {
    int mid = (begin + end) / 2;
    if (cfg->biases[mid].token < token) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin < cfg->bias_count && cfg->biases[begin].token == token
      ? static_cast<float>(cfg->biases[begin].value) : 0.f;
}

struct ScorePenalties {
  bool repetition, frequency, presence;
  float repetition_value, frequency_value, presence_value;
};

template<typename T>
__device__ inline float get_score(
  T const *logits,
  long long token,
  const HistoryBuffer& history_buffer,
  ServingConfig const* cfg,
  ScorePenalties const &penalties
) {
  float score = static_cast<float>(logits[token]);

  // 1. logit bias
  score += get_logit_bias(cfg, token);

  // 2. Repetition penalty
  if (penalties.repetition) {
    uint32_t mask = 1u << (token & 31);

    bool seen = (history_buffer.seen_total[token >> 5] & mask) != 0;

    if (seen) {
      float penalty = penalties.repetition_value;
      score = score > 0.f ? score / penalty : score * penalty;
    }
  }

  // 3. Frequency / Presence Penalty
  if (penalties.frequency || penalties.presence) {
    uint32_t mask = 1u << (token & 31);
    bool generated = (history_buffer.seen_generated[token >> 5] & mask) != 0;

    if (generated) {
      if (penalties.frequency) {
        score -= penalties.frequency_value *
                 static_cast<float>(history_buffer.frequency_counts[token]);
      }

      if (penalties.presence) {
        score -= penalties.presence_value;
      }
    }
  }

  return isnan(score) ? -std::numeric_limits<float>::infinity() : score;
}

template <typename T>
__device__ inline void sample_scores(
  T const* logits,
  uint32_t* history_buffer,
  char *buffer,
  int padded_vocab,
  int slot,
  int part_id,
  int num_cta,
  mirage::runtime::RuntimeConfig const &config
) {
  int row = config.request_ids[slot];
  int start = config.qo_indptr_buffer[slot];
  int end = config.qo_indptr_buffer[slot + 1];
  int prompt_len = config.prompt_length[row];
  int current_len = config.step[row] + end - start;
  int bitset_words = (padded_vocab + BITS_PER_WORD - 1) / BITS_PER_WORD;
  int history_words = padded_vocab + 2 * bitset_words;

  HistoryBuffer history(history_buffer + row * history_words, padded_vocab);

  int logit_row = end - 1;

  T const *row_logits = logits + logit_row * padded_vocab;
  
  float *scores = RadixBuffer(buffer, padded_vocab, slot).scores;

  auto* cfg = config.generation_config + config.request_ids[slot];

  ScorePenalties penalties{
      cfg->repetition_penalty != 1.f,
      cfg->frequency_penalty != 0.f,
      cfg->presence_penalty != 0.f,
      static_cast<float>(cfg->repetition_penalty),
      static_cast<float>(cfg->frequency_penalty),
      static_cast<float>(cfg->presence_penalty)};

  int vocab = cfg->vocab_size;

  long long begin = static_cast<long long>(vocab) * part_id / num_cta;
  long long token_end = static_cast<long long>(vocab) * (part_id + 1) / num_cta;

  for (long long token = begin + threadIdx.x; token < token_end; token += blockDim.x) {
    scores[token] = get_score(row_logits, token, history, cfg, penalties);
  }
}

__device__ inline uint32_t float_bits(float f) {
  uint32_t bits = __float_as_uint(f);  
  bits = (bits == 0x80000000u) ? 0u : bits;
  uint32_t mask = -int32_t(bits >> 31) | 0x80000000u;
  return bits ^ mask;
}

__device__ inline float float_from_bits(uint32_t bits) {
  return __uint_as_float((bits & 0x80000000u) ? (bits ^ 0x80000000u) : ~bits);
}

template <int Threads, typename T>
__device__ inline void sample_scores_with_max(
  T const* logits,
  uint32_t* history_buffer,
  char *buffer,
  int padded_vocab,
  int slot,
  int part_id,
  int num_cta,
  mirage::runtime::RuntimeConfig const &config
) {
  sample_scores(logits, history_buffer, buffer, padded_vocab, slot, part_id, num_cta, config);
  __syncthreads();

  RadixBuffer buf(buffer, padded_vocab, slot);
  int row = config.request_ids[slot];
  int vocab = config.generation_config[row].vocab_size;
  int begin = static_cast<long long>(vocab) * part_id / num_cta;
  int end = static_cast<long long>(vocab) * (part_id + 1) / num_cta;

  uint32_t maximum = 0;
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    maximum = max(maximum, float_bits(buf.scores[i]));
  }

  using BlockReduce = cub::BlockReduce<uint32_t, Threads>;
  extern __shared__ __align__(16) char sampling_shared[];
  auto &storage = *reinterpret_cast<typename BlockReduce::TempStorage *>(sampling_shared);
  maximum = BlockReduce(storage).Reduce(maximum, cub::Max());

  if (threadIdx.x == 0) {
    atomicMax(buf.radix_prefix, maximum);
  }
}

__device__ inline void sampling_barrier(
    RadixBuffer const &buf, int num_cta, uint32_t phase) {
  if (threadIdx.x == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> arrivals(*buf.arrival);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(*buf.ready);
    if (arrivals.fetch_add(1u, cuda::memory_order_acq_rel) == phase * num_cta - 1)
      ready.store(phase, cuda::memory_order_release);
    while (ready.load(cuda::memory_order_acquire) < phase) {}
  }
  __syncthreads();
}

template <int Threads, bool Joint>
__device__ inline void sample_pivot(
    char *buffer, long long *output, int padded_vocab, int slot, int vocab,
    int logit_row, int position, int part_id, int num_cta,
    ServingConfig const *cfg) {
  using Reduce = cub::BlockReduce<float, Threads>;
  using Mass = std::conditional_t<Joint, JointPivotMass, PivotMass>;
  using Shared = std::conditional_t<Joint, JointPivotShared<Threads>, PivotShared<Threads>>;
  using PivotReduce = cub::BlockReduce<Mass, Threads>;
  using Scan = cub::BlockScan<float, Threads>;
  extern __shared__ __align__(16) char sampling_shared[];
  auto &shared = *reinterpret_cast<Shared *>(sampling_shared);
  RadixBuffer buf(buffer, padded_vocab, slot);
  auto *partials = reinterpret_cast<float *>(buf.histogram);
  bool topk = Joint || cfg->top_k > 0;
  int tid = threadIdx.x;
  int begin = int(int64_t(vocab) * part_id / num_cta);
  int end = int(int64_t(vocab) * (part_id + 1) / num_cta);
  float maximum = float_from_bits(*buf.radix_prefix);
  float temperature = static_cast<float>(cfg->temperature);
  if (!topk) {
    for (int i = begin + tid; i < end; i += Threads)
      buf.scores[i] = expf((buf.scores[i] - maximum) / temperature);
  }
  __syncthreads();

  float low = topk ? -INFINITY : 0.f;
  float high = topk ? maximum : 1.f;
  float target = 0.f;
  int low_token = 0;
  uint32_t phase = 0;
  for (int round = 0;; ++round) {
    float mass = 0.f;
    for (int i = begin + tid; i < end; i += Threads) {
      float w = buf.scores[i];
      if (w > low || (w == low && i < low_token))
        mass += topk ? expf((w - maximum) / temperature) : w;
    }
    mass = Reduce(shared.mass).Sum(mass);
    if (tid == 0)
      partials[part_id] = mass;
    __syncthreads();
    sampling_barrier(buf, num_cta, ++phase);

    if (tid == 0) {
      float total = 0.f;
      for (int part = 0; part < num_cta; ++part)
        total += partials[part];
      shared.total = total;
      if constexpr (Joint) {
        if (round == 0)
          shared.target = cfg->top_p * total;
      }
      curandStatePhilox4_32_10_t rng;
      curand_init(cfg->seed, position, round, &rng);
      float u = fminf(curand_uniform(&rng), 0x1.fffffep-1f) * total;
      int selected = 0;
      while (selected + 1 < num_cta && u >= partials[selected])
        u -= partials[selected++];
      shared.selected = selected;
      shared.offset = u;
    }
    __syncthreads();
    if (round == 0) {
      target = topk ? static_cast<float>(cfg->top_k)
                    : static_cast<float>(cfg->top_p) * shared.total;
    }

    if (part_id == shared.selected) {
      float offset = shared.offset;
      for (int base = begin; base < end; base += Threads) {
        int i = base + tid;
        float value = i < end ? buf.scores[i] : 0.f;
        float w = 0.f;
        if (i < end && (value > low || (value == low && i < low_token)))
          w = topk ? expf((value - maximum) / temperature) : value;
        float prefix;
        Scan(shared.scan).InclusiveSum(w, prefix);
        if (tid == Threads - 1)
          shared.tile = prefix;
        __syncthreads();
        if (offset < shared.tile) {
          if (w > 0.f && offset >= prefix - w && offset < prefix)
            *buf.rank = i;
          break;
        }
        offset -= shared.tile;
        __syncthreads();
      }
    }
    __syncthreads();
    sampling_barrier(buf, num_cta, ++phase);
    int sampled = *buf.rank;
    float pivot = buf.scores[sampled];
    float midpoint = (pivot + high) * .5f;
    Mass sums{};
    for (int i = begin + tid; i < end; i += Threads) {
      float w = buf.scores[i];
      float weight = 0.f;
      if constexpr (Joint)
        weight = expf((w - maximum) / temperature);
      if (w > pivot || (w == pivot && i < sampled)) {
        sums.first += topk ? 1.f : w;
        if constexpr (Joint)
          sums.first_mass += weight;
      }
      if (w > midpoint) {
        sums.second += topk ? 1.f : w;
        if constexpr (Joint)
          sums.second_mass += weight;
      }
    }
    sums = PivotReduce(shared.pivots).Sum(sums);
    auto *pivots = reinterpret_cast<Mass *>(buf.histogram);
    if (tid == 0)
      pivots[part_id] = sums;
    __syncthreads();
    sampling_barrier(buf, num_cta, ++phase);
    if (tid == 0) {
      Mass total{};
      for (int part = 0; part < num_cta; ++part)
        total = total + pivots[part];
      shared.total = total.first;
      shared.offset = total.second;
      if constexpr (Joint) {
        shared.first_mass = total.first_mass;
        shared.second_mass = total.second_mass;
      }
    }
    __syncthreads();
    bool accept = shared.total < target;
    bool accept_midpoint = shared.offset < target;
    if constexpr (Joint) {
      accept = accept && shared.first_mass < shared.target;
      accept_midpoint = accept_midpoint && shared.second_mass < shared.target;
    }
    if (accept) {
      if (part_id == 0 && tid == 0)
        output[logit_row] = sampled;
      return;
    }
    if (accept_midpoint) {
      low = pivot;
      low_token = sampled;
      high = midpoint;
    } else {
      low = midpoint;
      low_token = 0;
    }
    sampling_barrier(buf, num_cta, ++phase);
  }
}

template <int NUM_THREADS, SamplePhase PHASE, typename T>
__device__ void sample_request(
  T const* logits, 
  uint32_t *history_buffer, 
  char *buffer,
  long long *output, 
  int padded_vocab, 
  int num_cta,
  int slot,
  int part_id,
  mirage::runtime::RuntimeConfig const &config
  ) {
    // checking
    int row = config.request_ids[slot];

    // The request is empty
    if (row < 0) {
      return;
    }

    int start = config.qo_indptr_buffer[slot];
    int end = config.qo_indptr_buffer[slot + 1];
    int prompt_len = config.prompt_length[row];
    int step = config.step[row];
    int current_len = step + end - start;

    // we are in chunked prefill stage
    if (current_len < prompt_len) {
      return;
    }

    int vocab = config.generation_config[row].vocab_size;

    if constexpr (PHASE == SamplePhase::PREPARE) {
      prepare_history(history_buffer, buffer, padded_vocab, slot, config);
    } else if constexpr (PHASE == SamplePhase::SCORE) {
      sample_scores_with_max<NUM_THREADS>(logits, history_buffer, buffer, padded_vocab, slot, part_id, num_cta, config);
    } else if constexpr (PHASE == SamplePhase::RADIX) {
      auto const *cfg = config.generation_config + row;
      if (cfg->top_k > 0 && cfg->top_p < 1.) {
        sample_pivot<NUM_THREADS, true>(buffer, output, padded_vocab, slot, vocab,
            end - 1, current_len - prompt_len, part_id, num_cta, cfg);
      } else {
        sample_pivot<NUM_THREADS, false>(buffer, output, padded_vocab, slot, vocab,
            end - 1, current_len - prompt_len, part_id, num_cta, cfg);
      }
    }
  }
  
}
}
