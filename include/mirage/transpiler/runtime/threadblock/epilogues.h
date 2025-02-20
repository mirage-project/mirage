// epilogues.h - Implementation of thread block level epilogues
#pragma once

#include "cute/config.hpp"
#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "element_unary.h"
#include "utils.h"

namespace tb {

// Every epilogue has the same interface: run(data, dst_ptr, dst_phy_pos)
// Every chain of epilogues ends with a EpilogueStore
// Epilogues can be chained like EpilogueExp<T, EpilogueExp<T,
// EpilogueStore<...>>>
template <typename T, bool IS_ACCUM>
class EpilogueStoreMaybeAccum {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    if constexpr (IS_ACCUM) {
      dst_ptr[dst_phy_pos] += data;
    } else {
      dst_ptr[dst_phy_pos] = data;
    }
  }
};

template <typename T>
using EpilogueStore = EpilogueStoreMaybeAccum<T, false>;

template <typename T>
using EpilogueStoreAccum = EpilogueStoreMaybeAccum<T, true>;

template <typename T, class NextEpilogue>
class EpilogueExp {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::EXP>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueSILU {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::SILU>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueGELU {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::GELU>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueRELU {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::RELU>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueClamp {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::CLAMP>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueSquare {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::SQUARE>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueSqrt {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::SQRT>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

template <typename T, class NextEpilogue>
class EpilogueMulScalar {
public:
  CUTE_DEVICE
  static void run(T const &data,
                  T *__restrict__ dst_ptr,
                  int64_t dst_phy_pos,
                  float const *epilogue_scalars) {
    assert(epilogue_scalars);
    T x = perform_element_unary_op<T, ElementUnaryOpType::MULSCALAR>(
        data, epilogue_scalars[0]);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos, ++epilogue_scalars);
  }
};

} // namespace tb
