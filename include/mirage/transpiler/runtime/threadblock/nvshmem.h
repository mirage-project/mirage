// communication.h - Implementation of thread block level nvshmem functions
// allreduce/allgather/reducescatter
#pragma once

#if USE_NVSHMEM

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#endif

namespace tb {

#define NVSHMEM_CHECK_COLLECTIVE(stmt)                                         \
  do {                                                                         \
    int result = (stmt);                                                       \
    if (result != NVSHMEMX_SUCCESS) {                                          \
      fprintf(stderr,                                                          \
              "[%s:%d] NVSHMEM error %d in %s\n",                              \
              __FILE__,                                                        \
              __LINE__,                                                        \
              result,                                                          \
              #stmt);                                                          \
      assert(false);                                                           \
    }                                                                          \
  } while (0)

static inline void nvshmem_half_sum_reduce(half_t *dest,
                                           half_t const *source,
                                           size_t size) {
#if USE_NVSHMEM
  NVSHMEM_CHECK_COLLECTIVE(
      nvshmem_half_sum_reduce(NVSHMEM_TEAM_WORLD, dest, source, size));
#else
  fprintf(
      stderr,
      "Error: nvshmem_half_sum_reduce called but NVSHMEM is not enabled.\n");
  assert(false);
#endif
}
} // namespace tb