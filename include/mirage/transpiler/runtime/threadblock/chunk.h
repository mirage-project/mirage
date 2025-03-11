#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

template <typename T, 
          typename Dst0Layout,
          typename Dst1Layout,
          typename SrcLayout,
          int CHUNK_SIZE,
          int CHUNK_DIM,
          int NUM_THREADS,
          typename Epilogue>
class ChunkKernel {
public:
    using Numel = decltype(cute::size(DstLayout{}));

    
}
}
