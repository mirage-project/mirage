#pragma once

namespace mirage { namespace serving {
constexpr int SCRATCH_VOCAB_ARRAYS = 4;
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
constexpr int sampling_scratch_words(int vocab) {
  return SCRATCH_VOCAB_ARRAYS * vocab + SCRATCH_WORKSPACE + SCRATCH_STATE_WORDS;
}
}} // namespace mirage::serving
