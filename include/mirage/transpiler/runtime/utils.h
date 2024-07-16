#pragma once
#include <cstdlib>
#include <iostream>

#include <cuda_runtime_api.h>

#define CHECK_CUDA(status)                                                     \
  do {                                                                         \
    if (status != 0) {                                                         \
      std::cerr << "Cuda failure: " << status << cudaGetErrorString(status)    \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CDIV(a, b) (((a) + (b)-1) / (b))

template <class var_t, var_t Start, var_t End, var_t Inc, class func_t>
constexpr void constexpr_for(func_t &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<var_t, Start + Inc, End, Inc>(f);
  }
}

namespace cute {
template <typename T>
void println(T const &t) {
  cute::print(t);
  printf("\n");
}
} // namespace cute
