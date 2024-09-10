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

template <typename T>
inline static constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <int a, int b>
inline static C<(a + b - 1) / b> ceil_div_cute(C<a> const &, C<b> const &) {
  return {};
}

template <typename T>
inline static T round_to_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

template <int a, int b>
inline static C<((a + b - 1) / b) * b> round_to_multiple_cute(C<a> const &,
                                                              C<b> const &) {
  return {};
}

template <class var_t, var_t Start, var_t End, var_t Inc, class func_t>
constexpr void constexpr_for(func_t &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<var_t, Start + Inc, End, Inc>(f);
  }
}

namespace cute {
template <typename T>
__host__ __device__ void println(T const &t) {
  cute::print(t);
  printf("\n");
}
} // namespace cute

// Print a warning message at run time. This message is printed only once
#define PRINT_RUNTIME_WARNING_ONCE(msg)                                        \
  do {                                                                         \
    static bool __warned = false;                                              \
    if (!__warned) {                                                           \
      std::cerr << "Warning: " << msg << std::endl;                            \
      __warned = true;                                                         \
    }                                                                          \
  } while (0)

#define SWAP(a, b)                                                             \
  {                                                                            \
    auto tmp = a;                                                              \
    a = b;                                                                     \
    b = tmp;                                                                   \
  }
