// output.h - Implementation of threadblock level output operators
//
// We provide these implementations:
// - Non-chunked, synchronous copy
// - Chunked, synchronous copy
// - Copy using the Tensor Memory Accelerator (TMA)
//
// For the meaning of "chunked" and "asynchronous", please refer to `input.h`
// in the same directory.

#pragma once

#include "input.h"
#include <cute/layout.hpp>
using namespace cute;

namespace tb {

// Type 1: Non-chunked, synchronous copy
// The same as the input case
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
using OutputNonChunkedSyncCopy =
    InputNonChunkedSyncCopy<T, DstLayout, SrcLayout, NUM_THREADS>;

// Type 2: Chunked, synchronous copy
// The same as the input case
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
using OutputChunkedSyncCopy =
    InputChunkedSyncCopy<T, DstLayout, SrcLayout, NUM_THREADS>;

// Type 3: Copy using the Tensor Memory Accelerator (TMA)

} // namespace tb
