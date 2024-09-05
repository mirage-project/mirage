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
  static void run(T const &data, T *__restrict__ dst_ptr, int64_t dst_phy_pos) {
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
  static void run(T const &data, T *__restrict__ dst_ptr, int64_t dst_phy_pos) {
    T x = perform_element_unary_op<T, ElementUnaryOpType::EXP>(data);
    NextEpilogue::run(x, dst_ptr, dst_phy_pos);
  }
};

} // namespace tb
