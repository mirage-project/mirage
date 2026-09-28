#include "mirage/persistent_kernel/tasks/blackwell/sparse_mla_sm100.cuh"

struct SparseMlaCompileArgs {
  void const *q;
  void const *cache;
  int const *indices;
  int const *counts;
  int const *qo;
  int const *ki;
  int const *pages;
  int const *last;
  void *output;
  float *partial;
  float *lse;
  int requests, num_pages, capacity;
  float scale;
};

template <int H, int R, int PAGE, int SPLITS>
__global__ __launch_bounds__(256) void
    sparse_mla_compile_compute(SparseMlaCompileArgs a) {
  kernel::sparse_mla_sm100_task_impl<H, R, PAGE, SPLITS>(a.q,
                                                      a.cache,
                                                      a.indices,
                                                      a.counts,
                                                      a.output,
                                                      a.partial,
                                                      a.lse,
                                                      a.qo,
                                                      a.ki,
                                                      a.pages,
                                                      a.last,
                                                      a.requests,
                                                      a.num_pages,
                                                      a.scale,
                                                      a.capacity,
                                                      blockIdx.x,
                                                      blockIdx.y,
                                                      blockIdx.z);
}

template <int H, int SPLITS>
__global__ __launch_bounds__(256) void
    sparse_mla_compile_reduce(SparseMlaCompileArgs a) {
  kernel::sparse_mla_reduce_sm100_task_impl<H, SPLITS>(
      a.partial, a.lse, a.output, blockIdx.x, blockIdx.y);
}

#define INSTANTIATE_COMPUTE(H, R, PAGE, SPLITS)                            \
  template __global__ void sparse_mla_compile_compute<H, R, PAGE, SPLITS>( \
      SparseMlaCompileArgs);

#define INSTANTIATE_SPLITS(H, R, PAGE) \
  INSTANTIATE_COMPUTE(H, R, PAGE, 1)   \
  INSTANTIATE_COMPUTE(H, R, PAGE, 2)   \
  INSTANTIATE_COMPUTE(H, R, PAGE, 4)   \
  INSTANTIATE_COMPUTE(H, R, PAGE, 8)

#define INSTANTIATE_SHAPES(H)     \
  INSTANTIATE_SPLITS(H, 0, 64)    \
  INSTANTIATE_SPLITS(H, 64, 64)   \
  INSTANTIATE_SPLITS(H, 0, 128)   \
  INSTANTIATE_SPLITS(H, 64, 128)

#define INSTANTIATE_REDUCE(H, SPLITS)                            \
  template __global__ void sparse_mla_compile_reduce<H, SPLITS>( \
      SparseMlaCompileArgs);

#define INSTANTIATE_HEADS(H)   \
  INSTANTIATE_SHAPES(H)        \
  INSTANTIATE_REDUCE(H, 2)     \
  INSTANTIATE_REDUCE(H, 4)     \
  INSTANTIATE_REDUCE(H, 8)

INSTANTIATE_HEADS(8)
INSTANTIATE_HEADS(16)
INSTANTIATE_HEADS(32)
INSTANTIATE_HEADS(64)

#undef INSTANTIATE_HEADS
#undef INSTANTIATE_REDUCE
#undef INSTANTIATE_SHAPES
#undef INSTANTIATE_SPLITS
#undef INSTANTIATE_COMPUTE
