#include "mirage/threadblock/graph.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"
#include <curand_kernel.h>
#include <random>

// All functions in this file is for convenience and assumes there is only 1
// threadblock.

namespace mirage {
namespace threadblock {

inline Graph create_single_threadblock_graph(unsigned int num_threads) {
  return Graph({1, 1, 1}, {num_threads, 1, 1}, 1);
}

template <typename Element, typename Layout>
inline STensor
    allocate_stensor(Graph &bgraph,
                     cutlass::HostTensor<Element, Layout> const &host_tensor) {
  static_assert(std::is_same_v<Element, cutlass::half_t>, "Only f16.");
  STensor tensor;
  tensor.num_dims = Layout::kRank;
  tensor.data_type = type::DT_FLOAT16;
  int contig_dim = -1;
  if constexpr (std::is_same_v<Layout, cutlass::layout::RowMajor>) {
    tensor.layout = layout::SmemRowMajor;
    contig_dim = 1;
  } else if constexpr (std::is_same_v<Layout, cutlass::layout::ColumnMajor>) {
    tensor.layout = layout::SmemColumnMajor;
    contig_dim = 0;
  } else {
    static_assert(std::is_same_v<Element, void>, "Unsupported layout.");
  }
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = host_tensor.extent()[i];
    // tensor.stride[i] =
    //     i == contig_dim ? 1 : host_tensor.stride(i - (i > contig_dim));
  }
  tensor.owner_op = nullptr;
  tensor.owner_ts_idx = -1;
  tensor.smem_offset = bgraph.allocate(tensor);
  return tensor;
}

CUTLASS_DEVICE
void copy_global_to_shared(char *smem_buffer,
                           STensor const &tensor,
                           void *g_ptr_) {
  // Only the first thread copies. TODO: make all threads copy.
  if (cutlass::thread0()) {
    char *s_ptr = smem_buffer + tensor.smem_offset;
    char *g_ptr = reinterpret_cast<char *>(g_ptr_);
    size_t size = tensor.size();
    for (size_t i = 0; i < size; i++) {
      s_ptr[i] = g_ptr[i];
    }
  }
  __syncthreads();
}
CUTLASS_DEVICE
void copy_shared_to_global(char *smem_buffer,
                           STensor const &tensor,
                           void *g_ptr_) {
  if (cutlass::thread0()) {
    char *s_ptr = smem_buffer + tensor.smem_offset;
    char *g_ptr = reinterpret_cast<char *>(g_ptr_);
    size_t size = tensor.size();
    for (size_t i = 0; i < size; i++) {
      g_ptr[i] = s_ptr[i];
    }
  }
}

template <typename Element, typename Layout>
void random_fill_tensor(cutlass::HostTensor<Element, Layout> &host_tensor,
                        size_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t size = host_tensor.size();
  Element *ptr = host_tensor.host_data();
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = static_cast<Element>(dist(gen));
  }
  host_tensor.sync_device();
}

template <typename Element, typename Layout>
void zero_fill_tensor(cutlass::HostTensor<Element, Layout> &host_tensor) {
  size_t size = host_tensor.size();
  Element *ptr = host_tensor.host_data();
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = static_cast<Element>(0);
  }
  host_tensor.sync_device();
}

template <typename DT>
__global__ void random_fill_device_tensor(mirage::kernel::DTensor const dtensor,
                                          int num_elements,
                                          unsigned long long seed = 0) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  curandState state;
  if (i < num_elements) {
    curand_init(seed, i, 0, &state);
    ((DT *)dtensor.data_ptr)[i] = curand_uniform(&state);
  }
}

template <typename DT>
__global__ void
    checkTensorsEqual(void *A, void *B, int *not_equals, size_t num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    if (fabs((float)((DT *)A)[i] - (float)((DT *)B)[i]) >= 1e-6) {
      // printf("not equal %d, %f, %f\n", i, (float)((DT *)A)[i], (float)((DT
      // *)B)[i]);
      atomicAdd(not_equals, 1);
    }
    // else{
    //   printf(" equal %d, %f, %f\n", i, (float)((DT *)A)[i], (float)((DT
    //   *)B)[i]);
    //   // atomicAdd(not_equals, 1);
    // }
  }
}

} // namespace threadblock
} // namespace mirage
