// element_binary.h - Implementation of element wise binary operators
#pragma once
#include <cassert>

enum class ElementBinaryOpType { ADD, MUL, DIV };

template <typename T, ElementBinaryOpType OP>
__device__ __forceinline__ T perform_element_binary_op(T a, T b) {
  if constexpr (OP == ElementBinaryOpType::ADD) {
    return a + b;
  } else if constexpr (OP == ElementBinaryOpType::MUL) {
    return a * b;
  } else if constexpr (OP == ElementBinaryOpType::DIV) {
    return a / b;
  } else {
    assert(0);
  }
}

// TODO: Optimize this kernel
template <typename T, ElementBinaryOpType OP>
__global__ void element_binary_kernel_fwd(T *__restrict__ out,
                                          T const *__restrict__ in1,
                                          T const *__restrict__ in2,
                                          int numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    out[idx] = perform_element_binary_op<T, OP>(in1[idx], in2[idx]);
  }
}

template <typename T, ElementBinaryOpType OP>
class ElementBinaryKernel {
public:
  static void run(T *out, T const *in1, T const *in2, int numel) {
    constexpr int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;
    element_binary_kernel_fwd<T, OP>
        <<<num_blocks, block_size>>>(out, in1, in2, numel);
  }
};

template <typename T>
using ElementAddKernel = ElementBinaryKernel<T, ElementBinaryOpType::ADD>;

template <typename T>
using ElementMulKernel = ElementBinaryKernel<T, ElementBinaryOpType::MUL>;

template <typename T>
using ElementDivKernel = ElementBinaryKernel<T, ElementBinaryOpType::DIV>;
