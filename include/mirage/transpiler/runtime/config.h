#pragma once

// There are some macros that the transpiler should define
#ifdef USE_PLACEHOLDER
#define USE_NVSHMEM                                                            \
  0                // Whether to use NVSHMEM (means the kernel is distributed)
#define NUM_GPUS 1 // The number of GPUs
#endif

#ifndef USE_NVSHMEM
static_assert(0);
#endif

#ifndef NUM_GPUS
static_assert(0);
#endif
