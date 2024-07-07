// element_unary.h - Implementation of element wise unary operators
#pragma once
#include <cassert>

enum class ElementUnaryOpType { EXP };

template <typename T, ElementUnaryOpType OP>
__device__ __forceinline__ T perform_element_unary_op(T a) {
  if constexpr (OP == ElementUnaryOpType::EXP) {
    if constexpr (std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, __half>) {
      return (T)expf((float)a);
    } else {
      assert(0);
    }
  } else {
    assert(0);
  }
}

// TODO: Optimize this kernel
template <typename T, ElementUnaryOpType OP>
__global__ void element_unary_kernel_fwd(T *__restrict__ out,
                                         T const *__restrict__ in,
                                         int numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    out[idx] = perform_element_unary_op<T, OP>(in[idx]);
  }
}

template <typename T, ElementUnaryOpType OP>
class ElementUnaryKernel {
public:
  static void run(T *out, T const *in, int numel) {
    constexpr int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;
    element_unary_kernel_fwd<T, OP><<<num_blocks, block_size>>>(out, in, numel);
  }
};

template <typename T>
using ElementExpKernel = ElementUnaryKernel<T, ElementUnaryOpType::EXP>;
